"""Interleaved synthetic services with real declarations/property validators.

All systemd/proc/cgroup ports are replaced. Only temporary JSON is written.
"""
import copy
from contextlib import ExitStack, contextmanager
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

Q = Path(__file__).resolve().parent
sys.path[:0] = [str(Q), str(Q.parent)]
import qwen35_sampler_runtime_adapter_v8 as a
import qwen35_sampler_runtime_service_v8 as s
import qwen35_sampler_runtime_contract_v8 as c
import qwen35_sampler_runtime_io_v8 as io
import qwen35_sampler_runtime_adapter_v4 as old_a
import qwen35_sampler_runtime_service_v4 as old_s
import minicpm5_layer0_execution_protocol_v4 as original_process


class World:
    def __init__(self, directory, adapter=a, service=s):
        self.a, self.s = adapter, service
        self.spec = {'run_name':'synthetic-scope', 'output_root':str(Path(directory)/'output')}
        self.admission = {'path':str(Path(directory)/'admission.json'), 'sha256':'a'*64}
        self.units = c.units(self.spec['run_name'])
        self.props = {}; self.procs = {}; self.services = {}; self.phases = {}
        self.counter = 100; self.list_calls = 0; self.list_hook = lambda n:None
        self.extra = []; self.list_transform = lambda names:names
        self.drains = []; self.blocked_groups = set()
        self.p = SimpleNamespace(command=self.command, terminal=original_process.terminal,
             proc=lambda pid:copy.deepcopy(self.procs.get(pid)), invocation=lambda v:bool(re.fullmatch('[a-f0-9]{32}',v)),
             empty_group=self.empty_group, gone=self.gone, GROUP_PREFIX='/synthetic/')
        self.co = self.a.coordinator_service(self.spec,self.admission)
        self.install(self.co)

    def install(self, obj, pending=False):
        self.counter += 1; pid=self.counter
        group='/synthetic/'+obj.unit+'.service'
        row={k:v for k,v in obj.props.items() if k not in ('RuntimeMaxSec','TimeoutStopSec')}
        row.update({k.replace('Sec','USec'):v+'s' for k,v in obj.props.items() if k in ('RuntimeMaxSec','TimeoutStopSec')})
        command=self.s.command(obj.unit,obj.argv,obj.env,obj.props,obj.log,obj.parent)
        row.update(Id=obj.unit+'.service',LoadState='loaded',ActiveState='activating' if pending else 'active',
             SubState='start' if pending else 'running',MainPID='0' if pending else str(pid),ControlPID='0',Job='',
             InvocationID=f'{pid:032x}',ControlGroup=group,Result='success',ExecMainPID=str(pid),ExecMainCode='0',
             ExecMainStatus='0',ExecStart='{ argv[]='+' '.join(command[command.index('/usr/bin/env'):])+' ; }',
             ExecStopPost='',Environment='',BindsTo='' if obj.parent is None else obj.parent+'.service',
             After='' if obj.parent is None else obj.parent+'.service',StandardOutput='append',StandardError='inherit')
        process={'pid':pid,'start_ticks':str(pid*10),'ppid':1,'argv':obj.argv,'cgroup':'0::'+group}
        owner={'process':copy.deepcopy(process),'service_owner':io.service_owner(obj.unit,process,row)}
        self.props[obj.unit]=row
        if not pending:self.procs[pid]=process
        obj.owner=None if pending else owner;obj.started=True
        obj.installed_invocation=None if pending else row['InvocationID']
        self.services[obj.unit]=obj
        return obj

    def phase(self, name, *, pending=False, ack=False, declared=True, wrong_admission=False, arm='baseline'):
        directory=str(c.control_root(self.spec)/'shared-native_preflight') if (arm,name)==('shared','native_preflight') else self.spec['output_root']+'/'+arm+'/model'
        phase=self.s.expected_phase(self.spec,self.admission,self.co.owner,'driver-original',arm,name,directory)
        if wrong_admission:phase['admission']={**self.admission,'sha256':'b'*64}
        stage=Path(phase['control']);stage.mkdir(parents=True)
        ref=io.once(stage/'phase-spec.json',phase) if declared else {'path':str(stage/'phase-spec.json'),'sha256':'c'*64}
        obj=self.a.phase_service(self.spec,phase,ref)
        self.install(obj,pending=pending)
        if ack:io.once(stage/'supervisor-owner.json',obj.owner)
        self.phases[name]=obj
        return obj

    def stop(self, obj, code=1, status=0, result='success'):
        row=self.props[obj.unit];self.procs.pop(int(row['ExecMainPID']),None)
        row.update(MainPID='0',ActiveState='active',SubState='exited',ExecMainCode=str(code),ExecMainStatus=str(status),Result=result)

    def empty_group(self, group):
        if group in self.blocked_groups:raise ValueError('synthetic removed phase has live descendant')
        if any(p['cgroup']=='0::'+group for p in self.procs.values()):raise ValueError('synthetic group still populated')
        self.drains.append(group)
        return {'group':group,'absent':True}

    def gone(self, owner):
        if owner['pid'] in self.procs:raise ValueError('synthetic captured owner not gone')

    def properties(self, unit):
        if unit in self.props:return copy.deepcopy(self.props[unit])
        return dict(Id=unit+'.service',LoadState='not-found',InvocationID='',ActiveState='inactive',SubState='dead',
                    MainPID='0',ControlPID='0',ExecMainPID='0',Job='',ControlGroup='')

    def command(self, argv, **kwargs):
        if argv!=a.ActiveScopeMonitor.LIST_ARGV:raise AssertionError('unexpected mocked command')
        self.list_calls+=1;self.list_hook(self.list_calls)
        names=[u for u,v in self.props.items() if not self.p.terminal(v)]+self.extra
        names=self.list_transform(names)
        return ''.join(u+'.service loaded active running synthetic\n' for u in names)

    def monitor(self):return a.ActiveScopeMonitor(self.spec,self.admission,self.co,lambda:None)

    @contextmanager
    def patched(self):
        with ExitStack() as stack:
            for module in (self.a,self.s):
                stack.enter_context(patch.object(module,'process_api',return_value=self.p))
                stack.enter_context(patch.object(module,'properties',side_effect=self.properties))
            stack.enter_context(patch.object(original_process,'command',side_effect=self.command))
            stack.enter_context(patch.object(a.time,'sleep'))
            yield self


class ScopeTests(unittest.TestCase):
    def world(self, adapter=a, service=s):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        return World(temp.name,adapter,service)

    def test_01_exact_old_sequence_fails_when_phase_starts_after_enumeration(self):
        w=self.world(old_a,old_s)
        with w.patched():
            discovered=old_a.declared_services(w.spec,w.admission,w.co.owner)
            live=[obj.unit for obj in discovered if obj.started and not w.p.terminal(obj.observe())]
            w.list_hook=lambda n:w.phase('cold') if n==1 else None
            with self.assertRaisesRegex(RuntimeError,'Competing active heavy service'):
                original_process.active_heavy(live)
        self.assertIn('cold',w.phases)
        self.assertNotIn(w.units['baseline:cold'],live)

    def test_02_same_interleaving_reauthenticates_new_unacknowledged_phase(self):
        w=self.world();monitor=w.monitor()
        w.list_hook=lambda n:w.phase('cold') if n==2 else None
        with w.patched():services,value=monitor.observe()
        self.assertEqual(monitor.observation['attempts'],2)
        cold=next(obj for obj in services if obj.unit==w.units['baseline:cold'])
        self.assertEqual(cold.owner,w.phases['cold'].owner)
        self.assertFalse((Path(cold.log).parent/'supervisor-owner.json').exists())
        self.assertEqual(value['MainPID'],str(w.co.owner['process']['pid']))

    def test_03_unknown_service_rejected_and_raw_conflict_saved(self):
        w=self.world();w.extra=['trn1-unrelated-job'];monitor=w.monitor()
        with w.patched(),self.assertRaisesRegex(ValueError,'unknown competing'):monitor.observe()
        self.assertIn('trn1-unrelated-job',monitor.failure_evidence['events'][0]['raw_systemctl_list'])

    def test_04_undeclared_exact_name_is_not_exempted(self):
        w=self.world();w.phase('cold',declared=False);monitor=w.monitor()
        with w.patched(),self.assertRaisesRegex(ValueError,'undeclared phase unit'):monitor.observe()
        self.assertTrue(any('raw_properties' in x for e in monitor.failure_evidence['events'] for x in e.get('services',[])))

    def test_05_same_prefix_unknown_name_rejected(self):
        w=self.world();w.extra=[w.units['baseline:cold']+'-unreviewed']
        with w.patched(),self.assertRaisesRegex(ValueError,'unknown competing'):w.monitor().observe()

    def test_06_invocation_replacement_rejected_between_snapshots(self):
        w=self.world();phase=w.phase('cold');monitor=w.monitor()
        def replace(n):
            if n==2:w.props[phase.unit]['InvocationID']='f'*32
        w.list_hook=replace
        with w.patched(),self.assertRaisesRegex(ValueError,'invocation changed'):monitor.observe()
        self.assertEqual(w.list_calls,2)
        with w.patched(),self.assertRaisesRegex(ValueError,'invocation changed'):
            a.declared_services(w.spec,w.admission,w.co.owner,monitor.retained)

    def test_07_same_pid_new_start_ticks_rejected(self):
        w=self.world();phase=w.phase('cold');monitor=w.monitor()
        def replace(n):
            if n==2:w.procs[phase.owner['process']['pid']]['start_ticks']='999999'
        w.list_hook=replace
        with w.patched(),self.assertRaisesRegex(ValueError,'process identity changed'):monitor.observe()

    def test_08_two_simultaneously_active_phases_rejected(self):
        w=self.world();w.phase('numeric');w.phase('cold')
        with w.patched(),self.assertRaisesRegex(ValueError,'at most one simultaneously'):w.monitor().observe()

    def test_09_persistent_snapshot_churn_is_bounded(self):
        w=self.world();cold=w.phase('cold');monitor=w.monitor()
        w.list_transform=lambda names:[u for u in names if u!=cold.unit] if w.list_calls%2 else names
        with w.patched(),self.assertRaisesRegex(ValueError,'changed continuously'):monitor.observe()
        self.assertEqual(w.list_calls,2*a.ActiveScopeMonitor.ATTEMPTS)
        self.assertFalse(monitor.failure_evidence['observation_successful'])

    def test_10_removed_old_phase_is_actually_drained_before_retry(self):
        w=self.world();old=w.phase('numeric');monitor=w.monitor()
        def transition(n):
            if n==2:w.stop(old);w.phase('cold')
        w.list_hook=transition
        with w.patched():monitor.observe()
        self.assertIn('/synthetic/'+old.unit+'.service',w.drains)
        self.assertTrue(any(e['kind']=='removed_unit_complete_terminal' and e['unit']==old.unit
                            for e in monitor.observation['events']))

    def test_11_removed_phase_with_remaining_descendant_rejected(self):
        w=self.world();old=w.phase('numeric');monitor=w.monitor()
        def transition(n):
            if n==2:
                w.stop(old);w.phase('cold');w.blocked_groups.add('/synthetic/'+old.unit+'.service')
        w.list_hook=transition
        with w.patched(),self.assertRaisesRegex(ValueError,'live descendant'):monitor.observe()
        self.assertEqual(w.list_calls,2)

    def test_12_exact_unit_with_wrong_admission_rejected(self):
        w=self.world();w.phase('cold',wrong_admission=True)
        with w.patched(),self.assertRaisesRegex(ValueError,'exact declared phase'):w.monitor().observe()

    def test_13_complete_resource_contract_remains_required(self):
        w=self.world();cold=w.phase('cold');w.props[cold.unit]['MemoryMax']='999';monitor=w.monitor()
        with w.patched(),self.assertRaisesRegex(ValueError,'resource changed: MemoryMax'):monitor.observe()
        values=[x for e in monitor.failure_evidence['events'] for x in e.get('services',[])]
        self.assertTrue(any(x.get('raw_properties',{}).get('MemoryMax')=='999' for x in values))

    def test_14_legitimate_unacknowledged_pending_install_is_fully_checked(self):
        w=self.world();cold=w.phase('cold',pending=True);monitor=w.monitor()
        with w.patched():services,_=monitor.observe()
        captured=next(obj for obj in services if obj.unit==cold.unit)
        self.assertIsNone(captured.owner)
        self.assertEqual(captured.installed_invocation,w.props[cold.unit]['InvocationID'])
        self.assertEqual(monitor.observation['attempts'],1)

    def test_15_retained_phase_declaration_cannot_be_replaced(self):
        w=self.world();cold=w.phase('cold');monitor=w.monitor()
        with w.patched():
            monitor.observe()
            import json
            path=Path(cold.log).parent/'phase-spec.json';value=json.loads(path.read_text());value['run_id']='replaced'
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'retained phase declaration changed'):monitor.observe()

    def test_16_real_SIGTERM_systemd_success_retained_without_logical_success(self):
        w=self.world();cold=w.phase('cold',ack=True)
        with w.patched():
            w.stop(cold,code=2,status=15,result='success')
            terminal=cold.drained();cold.validate_saved_terminal(terminal)
        self.assertEqual(terminal['systemd_exit_status'],{'ExecMainCode':2,'ExecMainStatus':15,'Result':'success'})
        self.assertNotEqual(terminal['systemd_exit_status'],{'ExecMainCode':1,'ExecMainStatus':0,'Result':'success'})

    def test_17_declaration_arriving_during_absence_group_check_is_authenticated(self):
        w=self.world();monitor=w.monitor();original=w.p.empty_group;published=[]
        def interleave(group):
            if group=='/synthetic/'+w.units['baseline:cold']+'.service' and not published:
                published.append(w.phase('cold'));raise RuntimeError('Live group descendants')
            return original(group)
        w.p.empty_group=interleave
        with w.patched():monitor.observe()
        self.assertEqual(monitor.observation['attempts'],2)
        self.assertEqual(monitor.retained[w.units['baseline:cold']].owner,published[0].owner)

    def test_18_other_cgroup_error_is_not_swallowed_by_new_declaration(self):
        w=self.world();original=w.p.empty_group;published=[]
        def interleave(group):
            if group=='/synthetic/'+w.units['baseline:cold']+'.service' and not published:
                published.append(w.phase('cold'));raise RuntimeError('Group symlink')
            return original(group)
        w.p.empty_group=interleave
        with w.patched(),self.assertRaisesRegex(RuntimeError,'Group symlink'):w.monitor().observe()

    def test_19_exact_phase_with_other_coordinator_owner_rejected(self):
        w=self.world();cold=w.phase('cold')
        import json
        path=Path(cold.log).parent/'phase-spec.json';phase=json.loads(path.read_text())
        phase['coordinator']['process']['start_ticks']='other-owner';path.write_text(json.dumps(phase))
        with w.patched(),self.assertRaisesRegex(ValueError,'exact declared phase'):w.monitor().observe()

    def test_20_unit_command_is_fully_authenticated(self):
        w=self.world();cold=w.phase('cold');w.props[cold.unit]['ExecStart']='{ argv[]=/unreviewed ; }'
        with w.patched(),self.assertRaisesRegex(ValueError,'ExecStart changed'):w.monitor().observe()


if __name__=='__main__':unittest.main(verbosity=2)
