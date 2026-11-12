"""Actual v4 service and declared-scope checks with explicit mocked IO only."""
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from contextlib import ExitStack, contextmanager

Q = Path(__file__).resolve().parent
sys.path[:0] = [str(Q), str(Q.parent)]
import qwen35_sampler_runtime_contract_v4 as contract
import qwen35_sampler_runtime_service_v4 as service
import qwen35_sampler_runtime_service_v3 as old_service
import qwen35_sampler_runtime_adapter_v4 as adapter


def reference(path, value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())


class Fixture:
    def __init__(self, failure=False):
        self.spec = dict(run_name='collected-test', output_root='/mnt/nvme/neuron-work/qwen35-sampler-pair-collected-test')
        self.admission = dict(path='/saved/admission.json', sha256='a'*64)
        self.coordinator = dict(process=dict(pid=100, start_ticks='1000', ppid=1, argv=['co'], cgroup='co'),
                                service_owner=dict(unit=contract.units('collected-test')['coordinator'], pid=100,
                                                   start_ticks=1000, invocation_id='b'*32))
        self.phase = service.expected_phase(self.spec, self.admission, self.coordinator, 'driver', 'baseline', 'numeric',
                                            self.spec['output_root']+'/baseline/model')
        self.phase_ref = reference(Path(self.phase['control'])/'phase-spec.json', self.phase)
        self.group_prefix = '/fake/'
        self.group = self.group_prefix+self.phase['unit']+'.service'
        self.owner = dict(process=dict(pid=101, start_ticks='1001', ppid=1, argv=service.argv('supervisor', self.phase_ref),
                                       cgroup='0::'+self.group),
                          service_owner=dict(unit=self.phase['unit'], pid=101, start_ticks=1001, invocation_id='c'*32))
        self.env = contract.native_environment(dict(artifact='/original-artifact', split_correction_sha256='d'*64))
        cmd = service.command(self.phase['unit'], self.owner['process']['argv'], self.env, self.phase['properties'],
                              self.phase['log'], self.coordinator['service_owner']['unit'])
        props = self.phase['properties']
        self.properties = {k:v for k,v in props.items() if k not in ('RuntimeMaxSec','TimeoutStopSec')}
        self.properties.update(Id=self.phase['unit']+'.service', LoadState='loaded',
            ActiveState='failed' if failure else 'active', SubState='failed' if failure else 'exited',
            MainPID='0', ControlPID='0', Job='', InvocationID='c'*32, ControlGroup='', ExecMainPID='101',
            ExecMainCode='1', ExecMainStatus='1' if failure else '0', Result='exit-code' if failure else 'success',
            ExecStart='{ argv[]='+' '.join(cmd[cmd.index('/usr/bin/env'):])+' ; }', ExecStopPost='', Environment='',
            BindsTo=self.coordinator['service_owner']['unit']+'.service', After=self.coordinator['service_owner']['unit']+'.service',
            RuntimeMaxUSec=props['RuntimeMaxSec']+'s', TimeoutStopUSec=props['TimeoutStopSec']+'s',
            StandardOutput='append', StandardError='inherit')
        self.terminal = dict(server_owner=copy.deepcopy(self.owner['service_owner']), process_owner=copy.deepcopy(self.owner['process']),
            unacknowledged_terminal=None, exit_observed=True,
            systemd_exit_status=dict(ExecMainCode=1, ExecMainStatus=1 if failure else 0, Result='exit-code' if failure else 'success'),
            properties=copy.deepcopy(self.properties), group_observations=[dict(group=self.group, absent=True)]*2,
            remaining_processes=[], cleanup_complete=True, observed_unix_ns=123456789)
        self.missing = dict(Id=self.phase['unit']+'.service', LoadState='not-found', ActiveState='inactive', SubState='dead',
                            MainPID='0', ControlPID='0', ExecMainPID='0', Job='', InvocationID='', ControlGroup='')
        self.current = copy.deepcopy(self.missing)
        self.log = []; self.alive = False; self.survivors = False; self.mutate_identity = False
        self.objects = {}; self.refresh()
        self.fake = NS(GROUP_PREFIX=self.group_prefix, invocation=lambda v: type(v) is str and len(v)==32 and v!='0'*32,
            terminal=lambda v: v['MainPID']==v['ControlPID']=='0' and not v['Job'] and
                (v['ActiveState'] in ('inactive','failed') or (v['ActiveState'],v['SubState'])==('active','exited')),
            gone=self.gone, empty_group=self.empty_group, proc=lambda pid: copy.deepcopy(self.owner['process']) if self.alive else None,
            command=lambda *a,**k: (_ for _ in ()).throw(AssertionError('unexpected signal or real command')))

    def refresh(self):
        root=Path(self.phase['control'])
        self.owner_ref=reference(root/'supervisor-owner.json',self.owner)
        self.terminal_ref=reference(root/'root-terminal.json',self.terminal)
        self.objects={self.phase_ref['path']:copy.deepcopy(self.phase),self.owner_ref['path']:copy.deepcopy(self.owner),
                      self.terminal_ref['path']:copy.deepcopy(self.terminal)}

    def read(self, ref):
        value=copy.deepcopy(self.objects[ref['path']])
        if reference(ref['path'],value)!=ref: raise ValueError('authenticated metadata digest changed')
        return value

    def identity(self,path): return dict(path=str(path), inode=2 if self.mutate_identity else 1)

    def gone(self, owner):
        self.log.append('gone')
        if self.alive: raise ValueError('Owned PID/start still live')
        if owner!=self.owner['process']: raise ValueError('wrong gone owner')

    def empty_group(self,group):
        self.log.append('empty')
        if self.survivors: raise ValueError('Live group descendants')
        return dict(group=group,absent=True)

    def properties_for(self,unit):
        self.log.append('properties')
        if unit==self.phase['unit']: return copy.deepcopy(self.current)
        return dict(self.missing, Id=unit+'.service')

    def make(self, module=service):
        return module.Service(self.phase['unit'],self.owner['process']['argv'],self.env,self.phase['properties'],self.phase['log'],
                              parent=self.coordinator['service_owner']['unit'],owner=self.owner)

    @contextmanager
    def patched(self, module=service):
        with ExitStack() as stack:
            for name,value in [('process_api',lambda:self.fake),('properties',self.properties_for)]:
                stack.enter_context(patch.object(module,name,value))
            if module is service:
                stack.enter_context(patch.object(service,'read',self.read))
                stack.enter_context(patch.object(service,'identity',self.identity))
            yield

    def bind(self,s): s.bind_saved_terminal(self.phase,self.phase_ref,self.owner_ref,self.terminal_ref)


class CollectedTerminalTests(unittest.TestCase):
    def test_01_original_collected_owner_rejection_reproduced(self):
        f=Fixture()
        with f.patched(old_service),self.assertRaisesRegex(ValueError,'cannot disappear before audit'):
            f.make(old_service).cleanup()

    def test_02_successful_saved_exit_reused_without_rewriting_status(self):
        f=Fixture()
        with f.patched():
            s=f.make();f.bind(s);got=s.cleanup()
        self.assertEqual(got,f.terminal);self.assertNotIn('never_loaded',got)
        self.assertTrue(s.historical_terminal_reuse['saved_exit_status_is_historical'])
        self.assertEqual(f.log[-6:],['properties','gone','empty','properties','gone','empty'])

    def test_03_failed_saved_exit_remains_failed(self):
        f=Fixture(True)
        with f.patched():
            s=f.make();f.bind(s);got=s.drained()
        self.assertEqual(got['systemd_exit_status'],dict(ExecMainCode=1,ExecMainStatus=1,Result='exit-code'))

    def test_04_missing_saved_exit_never_becomes_never_loaded(self):
        f=Fixture()
        with f.patched(),self.assertRaises(ValueError):f.make().cleanup()

    def test_05_missing_or_extra_terminal_fields_reject(self):
        for field in list(Fixture().terminal):
            f=Fixture();del f.terminal[field];f.refresh()
            with self.subTest(field=field),f.patched(),self.assertRaises(ValueError):f.bind(f.make())
        f=Fixture();f.terminal['never_loaded']=True;f.refresh()
        with f.patched(),self.assertRaises(ValueError):f.bind(f.make())

    def test_06_false_or_incomplete_exit_drain_facts_reject(self):
        for field,value in [('cleanup_complete',False),('exit_observed',False),('remaining_processes',[101]),
                            ('unacknowledged_terminal',{}),('observed_unix_ns',False)]:
            f=Fixture();f.terminal[field]=value;f.refresh()
            with self.subTest(field=field),f.patched(),self.assertRaises(ValueError):f.bind(f.make())

    def test_07_changed_receipt_owners_reject(self):
        for field,value in [('pid',102),('start_ticks','1002'),('ppid',2),('argv',['other']),('cgroup','other')]:
            f=Fixture();f.terminal['process_owner'][field]=value;f.refresh()
            with self.subTest(field=field),f.patched(),self.assertRaises(ValueError):f.bind(f.make())

    def test_08_saved_invocation_mismatch_rejects(self):
        for location in ('server_owner','properties'):
            f=Fixture();f.terminal[location]['invocation_id' if location=='server_owner' else 'InvocationID']='e'*32;f.refresh()
            with self.subTest(location=location),f.patched(),self.assertRaises(ValueError):f.bind(f.make())

    def test_09_wrong_phase_reference_or_directory_rejects(self):
        f=Fixture()
        with f.patched():
            with self.assertRaises(ValueError):f.make().bind_saved_terminal(f.phase,dict(f.phase_ref,path='/other/phase-spec.json'),f.owner_ref,f.terminal_ref)
            wrong=copy.deepcopy(f.phase);wrong['unit']='trn1-qwen35-sampler-other'
            with self.assertRaises(ValueError):f.make().bind_saved_terminal(wrong,f.phase_ref,f.owner_ref,f.terminal_ref)

    def test_10_saved_exit_status_mismatch_and_false_success_reject(self):
        for field,value in [('ExecMainCode',2),('ExecMainStatus',1),('Result','exit-code')]:
            f=Fixture();f.terminal['systemd_exit_status'][field]=value;f.refresh()
            with self.subTest(field=field),f.patched(),self.assertRaises(ValueError):f.bind(f.make())
        f=Fixture();f.terminal['properties']['ExecMainStatus']='1';f.terminal['systemd_exit_status']['ExecMainStatus']=1;f.refresh()
        with f.patched(),self.assertRaises(ValueError):f.bind(f.make())

    def test_11_incomplete_or_wrong_historical_groups_reject(self):
        faults=[[],[dict(group='/other',absent=True)]*2,[dict(group=Fixture().group,absent=False)]*2,
                [dict(group=Fixture().group,absent=False,cgroups=0)]*2,
                [dict(group=Fixture().group,absent=True),dict(group=Fixture().group,absent=False,cgroups=1)]]
        for groups in faults:
            f=Fixture();f.terminal['group_observations']=groups;f.refresh()
            with self.subTest(groups=groups),f.patched(),self.assertRaises(ValueError):f.bind(f.make())

    def test_12_live_saved_properties_or_modified_resources_reject(self):
        for field,value in [('MainPID','101'),('MemoryMax','1'),('ExecStart','wrong'),('BindsTo','other.service')]:
            f=Fixture();f.terminal['properties'][field]=value;f.refresh()
            with self.subTest(field=field),f.patched(),self.assertRaises(ValueError):f.bind(f.make())

    def test_13_current_live_owner_or_descendant_rejects(self):
        for field in ('alive','survivors'):
            f=Fixture()
            with f.patched():
                s=f.make();f.bind(s);setattr(f,field,True)
                with self.subTest(field=field),self.assertRaises(ValueError):s.cleanup()
                self.assertIsNone(s.historical_terminal_reuse)

    def test_14_current_pending_job_pid_or_invocation_rejects(self):
        for field,value in [('MainPID','101'),('ControlPID','12'),('ExecMainPID','101'),('Job','pending'),
                            ('InvocationID','c'*32),('ControlGroup','/other'),('ActiveState','active'),('Id','other.service')]:
            f=Fixture();f.current[field]=value
            with self.subTest(field=field),f.patched():
                s=f.make();f.bind(s)
                with self.assertRaises(ValueError):s.cleanup()

    def test_15_saved_bytes_or_identity_change_rejects_on_reuse(self):
        for field in ('bytes','identity'):
            f=Fixture()
            with f.patched():
                s=f.make();f.bind(s)
                if field=='bytes':f.objects[f.terminal_ref['path']]['exit_observed']=False
                else:f.mutate_identity=True
                with self.subTest(field=field),self.assertRaises(ValueError):s.cleanup()

    def test_16_second_observation_change_rejects(self):
        f=Fixture();calls=[0]
        def change(unit):
            calls[0]+=1;value=f.properties_for(unit)
            if calls[0]>1:value['Job']='new'
            return value
        with f.patched(),patch.object(service,'properties',change):
            s=f.make();f.bind(s)
            with self.assertRaises(ValueError):s.cleanup()

    def test_17_reappeared_invocation_cannot_reuse_old_terminal(self):
        f=Fixture()
        with f.patched():
            s=f.make();f.bind(s);f.current=dict(f.properties,InvocationID='f'*32)
            with self.assertRaises(ValueError):s.cleanup()

    def test_18_loaded_failure_without_saved_receipt_still_audits(self):
        f=Fixture(True);f.current=copy.deepcopy(f.properties)
        with f.patched():s=f.make();got=s.cleanup()
        self.assertEqual(got['systemd_exit_status']['ExecMainStatus'],1)
        self.assertIsNone(s.historical_terminal_reuse)

    def test_19_missing_unacknowledged_terminal_still_rejects(self):
        f=Fixture()
        with f.patched():
            s=f.make();s.owner=None;s.unacknowledged=dict(unit=s.unit)
            with self.assertRaises(ValueError):f.bind(s)
            with self.assertRaises(ValueError):s.cleanup()

    def test_20_declared_scope_retains_collected_service(self):
        f=Fixture();paths=set(f.objects)
        with f.patched(),ExitStack() as stack:
            stack.enter_context(patch.object(Path,'exists',lambda p:str(p) in paths))
            stack.enter_context(patch.object(adapter,'read',f.read))
            stack.enter_context(patch.object(adapter,'ref',lambda p:reference(p,f.objects[str(p)])))
            stack.enter_context(patch.object(adapter,'read_if_present',lambda p:copy.deepcopy(f.objects.get(str(p)))))
            stack.enter_context(patch.object(adapter,'properties',f.properties_for))
            stack.enter_context(patch.object(adapter,'process_api',lambda:f.fake))
            stack.enter_context(patch.object(adapter,'native_environment',lambda case:f.env))
            found=adapter.declared_services(f.spec,f.admission,f.coordinator)
            self.assertEqual(len(found),2);s=found[1]
            self.assertTrue(s.started);self.assertEqual(s.cleanup(),f.terminal)
            self.assertEqual(s.historical_terminal_reuse['references']['phase_spec'],f.phase_ref)

    def test_21_signal_option_matches_supported_systemctl_syntax(self):
        f=Fixture(True);f.current=dict(f.properties,MainPID='101',ActiveState='active',SubState='running');calls=[]
        def command(argv,**kwargs):
            calls.append(argv);f.current=copy.deepcopy(f.properties)
        f.fake.command=command
        with f.patched():f.make().cleanup()
        self.assertEqual(len(calls),1);self.assertIn('--kill-who=all',calls[0]);self.assertNotIn('--kill-whom=all',calls[0])
        self.assertEqual(calls[0][-1],f.phase['unit']);self.assertNotIn('stop',calls[0])

    def test_22_signal_failure_status_is_preserved(self):
        f=Fixture(True);f.terminal['systemd_exit_status']=dict(ExecMainCode=2,ExecMainStatus=15,Result='signal')
        f.terminal['properties'].update(ExecMainCode='2',ExecMainStatus='15',Result='signal');f.refresh()
        with f.patched():s=f.make();f.bind(s);got=s.cleanup()
        self.assertEqual(got['systemd_exit_status'],f.terminal['systemd_exit_status'])

    def test_23_old_kill_option_rejected_new_option_supported_in_mock(self):
        for module,succeeds in ((old_service,False),(service,True)):
            f=Fixture(True);f.current=dict(f.properties,MainPID='101',ActiveState='active',SubState='running')
            def supported(argv,**kwargs):
                if '--kill-whom=all' in argv:raise RuntimeError('unrecognized option --kill-whom=all')
                if '--kill-who=all' not in argv:raise AssertionError('required exact group option')
                f.current=copy.deepcopy(f.properties)
            f.fake.command=supported
            with self.subTest(version=module.__name__),f.patched(module):
                if succeeds:self.assertTrue(f.make(module).cleanup()['cleanup_complete'])
                else:
                    with self.assertRaisesRegex(RuntimeError,'unrecognized option'):f.make(module).cleanup()

    def test_24_runtime_leaf_overlap_cannot_replace_frozen_v1_pin(self):
        old=str(Q/'qwen35_sampler_native_leaves_v1.py');new=str(Q/'qwen35_sampler_native_leaves_v2.py')
        delivery=dict(new_source_sha256={old:'a'*64},prospective_host_dependency_sha256={'/reviewed/helper.py':'b'*64})
        pins={str(Q/name):'c'*64 for name in contract.RUNTIME_FILES}
        pins.update({path:'d'*64 for path in contract.STARTUP_FILES});pins['/reviewed/helper.py']='b'*64;pins[old]='a'*64
        host=contract.host_sources({'source_sha256':pins},delivery)
        self.assertEqual(host[old],'a'*64);self.assertIn(new,host)
        changed=copy.deepcopy(pins);changed[old]='f'*64
        with self.assertRaisesRegex(ValueError,'cannot replace a frozen'):contract.host_sources({'source_sha256':changed},delivery)
        missing=copy.deepcopy(pins);del missing[new]
        with self.assertRaisesRegex(ValueError,'must be reviewed'):contract.host_sources({'source_sha256':missing},delivery)


if __name__=='__main__':unittest.main(verbosity=2)
