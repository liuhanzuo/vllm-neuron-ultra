"""Pure runtime-source tests. No native, accelerator, real service, or lock IO."""
import ast
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from contextlib import ExitStack

Q = Path(__file__).resolve().parent
ROOT = Q.parent
sys.path[:0] = [str(Q), str(ROOT)]
import qwen35_sampler_runtime_contract_v3 as contract
import qwen35_sampler_runtime_io_v3 as io
import qwen35_sampler_runtime_service_v3 as service
import qwen35_sampler_runtime_adapter_v3 as adapter
import qwen35_sampler_runtime_history_v3 as history


class RuntimeTests(unittest.TestCase):
    def test_imports_and_help_are_stdlib_and_inert(self):
        names = [name.removesuffix('.py') for name in contract.RUNTIME_FILES]
        script = """
import sys, importlib, importlib.abc
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, *args):
  if fullname.split('.')[0] in {'torch','torch_neuronx','neuronx_distributed','transformers','numpy','vllm','vllm_neuron'}:
   raise AssertionError('native import: '+fullname)
sys.meta_path.insert(0,Block())
import subprocess, os
subprocess.Popen=lambda *a,**k: (_ for _ in ()).throw(AssertionError('launch during import'))
os.open=lambda *a,**k: (_ for _ in ()).throw(AssertionError('FD IO during import'))
for name in NAMES: importlib.import_module(name)
assert not {'torch','numpy','transformers'} & set(sys.modules)
import run_qwen35_sampler_runtime_v3 as cli
try: cli.main(['--help'])
except SystemExit as e: assert e.code==0
else: raise AssertionError('help must exit before runtime')
""".replace('NAMES', repr(names))
        result = subprocess.run([sys.executable, '-B', '-c', script], env={**os.environ, 'PYTHONPATH': str(Q)+':'+str(ROOT)},
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        self.assertIn(b'--spec-sha256', result.stdout)

    def test_frozen_science_bytes_unchanged(self):
        path = Q / 'qwen35-sampler-pair-source-delivery-20261111-v1.json'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), '2b50426e39728dc20800aa0030544af9b37307bca01c61e39de5dd4320bef04b')
        delivery = json.loads(path.read_text())
        for name, sha in delivery['new_source_sha256'].items():
            self.assertEqual(hashlib.sha256(Path(name).read_bytes()).hexdigest(), sha)

    def test_storage_budget_uses_actual_serialized_payloads(self):
        value = contract.budget()
        self.assertEqual(sum(value.values()), 2 * 1024**3)
        self.assertLess(287083673, int(contract.MODEL_PROPERTIES['LimitFSIZE']))
        self.assertLess(2 * 402337099, value['model_payload_bytes'])
        self.assertEqual(len(contract.PHASES), 20)
        self.assertEqual(6*contract.LIMITS['server_log_bytes']+14*contract.LIMITS['leaf_log_bytes']+
                         contract.LIMITS['coordinator_log_bytes'], value['all_logs_bytes'])
        self.assertEqual(contract.LOCK_IDENTITY, {'device':66306, 'inode':8294421})

    def test_compact_host_map_preserves_full_historical_map(self):
        delivery = {'new_source_sha256': {'/sources/science.py': 'a'*64}, 'prospective_host_dependency_sha256': {'/sources/native.py': 'b'*64}}
        pins = {f'/historical/source-{n}.py':'f'*64 for n in range(50000)}
        pins.update(delivery['new_source_sha256']); pins.update(delivery['prospective_host_dependency_sha256'])
        pins.update({str(Q/name):'c'*64 for name in contract.RUNTIME_FILES})
        pins.update({name:'d'*64 for name in contract.STARTUP_FILES})
        saved = copy.deepcopy(pins)
        host = contract.host_sources({'source_sha256':pins}, delivery)
        self.assertEqual(pins, saved)
        self.assertLess(len(host), 30)
        self.assertTrue(set(host) < set(pins))
        bad = copy.deepcopy(delivery); bad['new_source_sha256']['/sources/science.py'] = 'e'*64
        with self.assertRaises(ValueError): contract.host_sources({'source_sha256':pins}, bad)
        huge = {'phase':'measured','extra':'x'*65536}
        with self.assertRaises(ValueError): contract.native_environment({'artifact':'/artifact','split_correction_sha256':'a'*64},
                                                                        host_identity=huge, host_receipt='/receipt')

    def test_preflight_and_driver_uuid_are_independent(self):
        checks = []
        obj = adapter.Adapter({'path':'/admission','sha256':'a'*64}, {'attempt_id':'preflight-outer'},
            SimpleNamespace(spec={'current_serial_catalog':'complete'},host={}),
            SimpleNamespace(check=lambda **kw: checks.append(kw)),
            SimpleNamespace(owner={'service_owner':{'unit':'u','pid':1,'start_ticks':2,'invocation_id':'a'*32}}))
        obj.admission['spec'] = 'reviewed'
        obj.guard('before_phase_native_preflight','preflight-outer')
        self.assertIsNone(obj.run_id)
        obj.guard('before_output_creation','fresh-driver')
        obj.guard('baseline:state_copy','fresh-driver')
        obj.guard('preflight-identity-check','preflight-outer')
        self.assertEqual(obj.run_id,'fresh-driver')
        with self.assertRaises(ValueError): obj.guard('second_driver','other-driver')
        self.assertEqual(len(checks),4)

    def test_running_handoff_is_lossless_and_actual_tool_is_required(self):
        previous={'schema':'old','updated_at':'old','serial_execution':True,'live_tool_sessions':[], 'active_heavy_units':[],
                  'families':list(range(15)),'mini_operator_current':{'failed_exit':1},'full_history':{'owners':[{'pid':4}]}}
        admission={'freshness':{'latest_handoff':{'path':'/old','sha256':'a'*64}},'spec':'spec','review':'review',
                   'outer_owner':{'pid':1},'current_serial_catalog':'catalog','units':{'coordinator':'co','baseline:long':'long'}}
        co={'service_owner':{'unit':'co'},'process':{'pid':2}}
        value=history.expected_running(previous,'2026-11-11T20:00:00+00:00',{'kind':'tool_session','session_id':123},
                                       {'path':'/admission','sha256':'a'*64},admission,{'path':'/co','sha256':'b'*64},co)
        self.assertEqual(value['families'], previous['families'])
        self.assertEqual(value['mini_operator_current'], previous['mini_operator_current'])
        self.assertEqual(value['full_history'], previous['full_history'])
        self.assertEqual(previous['active_heavy_units'],[])
        with self.assertRaises(ValueError):
            history.expected_running(previous,'2026-11-11T20:00:00+00:00',{'kind':'tool_session','session_id':True},
                                     {},admission,{},co)

    def test_exact_units_and_original_environment(self):
        names = contract.units('source-test')
        self.assertEqual(len(names),len(set(names.values())))
        env=contract.native_environment({'artifact':'/artifact','split_correction_sha256':'a'*64})
        self.assertEqual(env['NEURON_RT_VISIBLE_CORES'],'0-1')
        self.assertEqual(env['OMP_NUM_THREADS'],env['MKL_NUM_THREADS'])
        self.assertFalse(any(k.startswith('VLLM_DP_') for k in env))
        self.assertFalse({'TRN1_CPU_SAMPLER_FAST','NEURON_CPU_SAMPLER_FAST','TRN1_SAMPLER_DIAGNOSTIC_DIR'} & set(env))
        command=service.command(names['baseline:long'],['/python','-B','/leaf'],env,contract.MODEL_PROPERTIES,'/fresh.log',names['coordinator'])
        self.assertNotIn('--collect',command)
        self.assertIn('RemainAfterExit=yes',command)
        self.assertIn('BindsTo='+names['coordinator']+'.service',command)
        with self.assertRaises(ValueError): service.command(names['coordinator'],['/python'],{'JSON':'has space'},contract.COORDINATOR_PROPERTIES,'/log')

    def test_phase_contract_retains_model_path_and_properties(self):
        spec={'run_name':'test','output_root':'/mnt/nvme/neuron-work/qwen35-sampler-pair-test'}
        co={'process':{'pid':10},'service_owner':{'unit':contract.units('test')['coordinator']}}
        phase=service.expected_phase(spec,{'path':'/admission','sha256':'a'*64},co,'driver','baseline','long',spec['output_root']+'/baseline/model')
        self.assertEqual(phase['properties']['LimitFSIZE'],str(512*2**20))
        self.assertEqual(phase['directory'],spec['output_root']+'/baseline/model')
        self.assertEqual(phase['coordinator'],co)
        self.assertFalse(phase['self_exit_claimed'])
        with self.assertRaises(ValueError): service.expected_phase(spec,{},co,'driver','baseline','shorter_long','/x')

    def fake_terminal_service(self, *, captured=False):
        value={'LoadState':'loaded','MainPID':'0','ControlPID':'0','Job':'','ActiveState':'failed','SubState':'failed',
               'InvocationID':'a'*32,'ExecMainPID':'12','ExecMainCode':'1','ExecMainStatus':'1','Result':'exit-code'}
        proc={'pid':12,'start_ticks':'55','ppid':1,'argv':['/python'],'cgroup':'0::/fake'}
        owner={'process':proc,'service_owner':{'unit':'trn1-qwen35-sampler-test-coordinator','pid':12,'start_ticks':55,'invocation_id':'a'*32}}
        s=service.Service('trn1-qwen35-sampler-test-coordinator',['/python'],{}, {},'/unused.log',owner=owner if captured else None)
        s.started=True
        fake=SimpleNamespace(terminal=lambda v:v['MainPID']=='0',gone=lambda o:None,proc=lambda pid:None,
                             empty_group=lambda group:{'group':group,'absent':True},GROUP_PREFIX='/fake/')
        return s,value,fake

    def test_unacknowledged_quick_failure_preserves_real_failure_and_drains(self):
        s,value,fake=self.fake_terminal_service()
        with patch.object(service,'properties',return_value=value),patch.object(service,'verify_properties',side_effect=lambda v,*a:v),patch.object(service,'process_api',return_value=fake):
            result=s.cleanup()
        self.assertIsNone(result['process_owner']);self.assertIsNone(result['server_owner'])
        self.assertFalse(result['unacknowledged_terminal']['start_ticks_observed'])
        self.assertEqual(result['systemd_exit_status'],{'ExecMainCode':1,'ExecMainStatus':1,'Result':'exit-code'})
        self.assertTrue(result['cleanup_complete'])

    def test_unacknowledged_pid_cannot_be_assumed_gone(self):
        s,value,fake=self.fake_terminal_service(); fake.proc=lambda pid:{'pid':pid,'start_ticks':'unknown'}
        with patch.object(service,'properties',return_value=value),patch.object(service,'verify_properties',side_effect=lambda v,*a:v),patch.object(service,'process_api',return_value=fake):
            with self.assertRaises(ValueError): s.drained()
        self.assertIsNone(s.terminal)

    def test_unacknowledged_terminal_cannot_disappear_into_never_started(self):
        s,value,fake=self.fake_terminal_service()
        with patch.object(service,'properties',return_value=value),patch.object(service,'verify_properties',side_effect=lambda v,*a:v),patch.object(service,'process_api',return_value=fake):
            s.drained()
        with patch.object(service,'properties',return_value={'LoadState':'not-found'}),patch.object(service,'process_api',return_value=fake):
            with self.assertRaises(ValueError):s.cleanup()
        self.assertEqual(s.terminal['systemd_exit_status']['ExecMainStatus'],1)

    def test_captured_invocation_change_rejected_before_stop(self):
        s,value,fake=self.fake_terminal_service(captured=True);value['InvocationID']='b'*32
        calls=[];fake.command=lambda *a,**k:calls.append(a)
        with patch.object(service,'properties',return_value=value),patch.object(service,'verify_properties',side_effect=lambda v,*a:v),patch.object(service,'process_api',return_value=fake):
            with self.assertRaises(ValueError):s.cleanup()
        self.assertEqual(calls,[])

    def test_pending_install_and_exact_env_exec_transition(self):
        s,value,fake=self.fake_terminal_service()
        value.update(MainPID='0',ActiveState='activating',SubState='start')
        wrapper=service.command(s.unit,s.argv,s.env,s.props,s.log,s.parent)
        wrapper=wrapper[wrapper.index('/usr/bin/env'):]
        current={'pid':12,'start_ticks':'55','ppid':1,'argv':wrapper,'cgroup':'0::'+fake.GROUP_PREFIX+s.unit+'.service'}
        fake.proc=lambda pid:current
        with patch.object(service,'properties',return_value=value),patch.object(service,'verify_properties',side_effect=lambda v,*a:v),patch.object(service,'process_api',return_value=fake):
            s.observe();self.assertIsNone(s.owner)
            value.update(MainPID='12',ActiveState='active',SubState='running')
            s.observe();self.assertIsNone(s.owner)
            current={**current,'argv':s.argv}
            s.live();self.assertEqual(s.owner['process'],current)
            self.assertEqual(s.installed_invocation,'a'*32)

    def test_pending_unknown_argv_or_changed_invocation_rejected(self):
        s,value,fake=self.fake_terminal_service();value.update(MainPID='12',ActiveState='active',SubState='running')
        fake.proc=lambda pid:{'pid':12,'start_ticks':'55','ppid':1,'argv':['/unreviewed'],'cgroup':'0::'+s.group}
        with patch.object(service,'properties',return_value=value),patch.object(service,'verify_properties',side_effect=lambda v,*a:v),patch.object(service,'process_api',return_value=fake):
            with self.assertRaises(ValueError):s.observe()
            value['InvocationID']='b'*32
            with self.assertRaises(ValueError):s.observe()

    def test_pending_install_disappearance_cannot_be_never_loaded(self):
        s,value,fake=self.fake_terminal_service();value.update(MainPID='0',ActiveState='activating',SubState='start')
        with patch.object(service,'properties',return_value=value),patch.object(service,'verify_properties',side_effect=lambda v,*a:v),patch.object(service,'process_api',return_value=fake):
            s.observe()
        with patch.object(service,'properties',return_value={'LoadState':'not-found'}),patch.object(service,'process_api',return_value=fake):
            with self.assertRaises(ValueError):s.cleanup()

    def test_failed_systemd_install_has_explicit_nonexit_receipt(self):
        s,value,fake=self.fake_terminal_service()
        with patch.object(service,'properties',return_value={'LoadState':'not-found'}),patch.object(service,'process_api',return_value=fake):
            result=s.cleanup()
        self.assertIs(result,s.terminal);self.assertTrue(result['never_loaded'])
        self.assertFalse(result['exit_observed']);self.assertNotIn('systemd_exit_status',result)

    def test_failure_cleanup_signals_owned_group_without_transient_stop_gc(self):
        s,value,fake=self.fake_terminal_service(captured=True)
        value.update(MainPID='12',ActiveState='active',SubState='running');calls=[]
        fake.proc=lambda pid:s.owner['process'] if value['MainPID']!='0' else None
        def signal_owned(argv,**kw):
            calls.append(argv);value.update(MainPID='0',ActiveState='failed',SubState='failed')
        fake.command=signal_owned
        with patch.object(service,'properties',side_effect=lambda unit:dict(value)),patch.object(service,'verify_properties',side_effect=lambda v,*a:v),patch.object(service,'process_api',return_value=fake):
            result=s.cleanup()
        self.assertTrue(result['cleanup_complete']);self.assertEqual(len(calls),1)
        self.assertIn('--signal=SIGTERM',calls[0]);self.assertNotIn('stop',calls[0])

    def test_live_service_is_not_completed(self):
        s,value,fake=self.fake_terminal_service(captured=True); value.update(MainPID='12',ActiveState='active',SubState='running')
        fake.proc=lambda pid:s.owner['process']
        with patch.object(service,'properties',return_value=value),patch.object(service,'verify_properties',side_effect=lambda v,*a:v),patch.object(service,'process_api',return_value=fake):
            s.live()
            self.assertIsNone(s.terminal)
            with self.assertRaises(ValueError):s.drained()

    def test_lock_release_requires_complete_drain(self):
        lock=io.RetainedLock();lock.fd=5
        with patch.object(io.os,'close') as close,patch.object(io.fcntl,'flock') as flock:
            with self.assertRaises(ValueError):lock.release(False)
        close.assert_not_called();flock.assert_not_called()

    def test_atomic_publish_and_terminal_storage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'out';root.mkdir();spec={'output_root':str(root)}
            value={'a':[1,2,3]};out=io.once(root/'result.json',value)
            self.assertEqual(io.read(out),value)
            self.assertEqual((root/'result.json').stat().st_nlink,1)
            with self.assertRaises(ValueError):io.once(root/'result.json',value)
            result=io.storage(spec,terminal=True)
            self.assertEqual(result['entries'],2)

    def test_known_hardlink_window_retries_but_unknown_links_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'out';root.mkdir();spec={'output_root':str(root)}
            temp=root/'receipt.json.publishing';temp.write_text('{}');os.link(temp,root/'receipt.json')
            calls=[]
            def settle(_):
                calls.append(True)
                if temp.exists():temp.unlink()
            with patch.object(io.time,'sleep',side_effect=settle):result=io.storage(spec)
            self.assertEqual(len(calls),1);self.assertEqual(result['entries'],2)
            os.link(root/'receipt.json',root/'unrelated.json')
            with self.assertRaises(ValueError):io.storage(spec)

    def test_ack_reader_tolerates_only_known_atomic_publication_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);temp=root/'ack.json.publishing';temp.write_text('{"ack":true}')
            os.link(temp,root/'ack.json');calls=[]
            def settle(_):
                calls.append(True)
                if temp.exists():temp.unlink()
            with patch.object(io.time,'sleep',side_effect=settle):result=io.read_if_present(root/'ack.json')
            self.assertEqual(result,{'ack':True});self.assertEqual(len(calls),1)
            os.link(root/'ack.json',root/'unrelated.json')
            with self.assertRaises(ValueError):io.read_if_present(root/'ack.json')

    def test_temporary_unlinked_before_peer_stat_is_known_publication_race(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp=Path(tmp)/'ack.json.publishing';final=Path(tmp)/'ack.json'
            temp.write_text('{}');os.link(temp,final)
            old_lstat=Path.lstat;triggered=[False]
            def racing(path):
                if path==final and not triggered[0]:
                    triggered[0]=True;temp.unlink()
                return old_lstat(path)
            with patch.object(Path,'lstat',racing):
                with self.assertRaises(io.PublicationRace):io.observed_stat(temp)
            self.assertTrue(triggered[0]);self.assertEqual(final.stat().st_nlink,1)

    def test_only_exact_declared_temporary_roots_are_admitted(self):
        roots=(Path('/out'),Path('/out-runtime'))
        for path in ('/out/tmp/a.so','/out-runtime/tmp/a','/out-runtime/baseline-long/tmp/native/a.py',
                     '/out-runtime/shared-native_preflight/tmp/cache'):
            self.assertEqual(io.storage_class(Path(path),roots),'temporary_and_directory_bytes')
        for path in ('/out-runtime/unknown/tmp/a','/out-runtime/baseline-long/other/a.so'):
            with self.assertRaises(ValueError):io.storage_class(Path(path),roots)

    def test_declared_native_temporary_disappearance_is_finitely_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'out';control=Path(str(output)+'-runtime');native=control/'baseline-long'/'tmp'
            native.mkdir(parents=True);(native/'ephemeral').write_text('temp')
            original=io.observed_stat;called=[False]
            def interleaved(path):
                if path.name=='ephemeral' and not called[0]:called[0]=True;path.unlink()
                return original(path)
            with patch.object(io,'observed_stat',side_effect=interleaved),patch.object(io.time,'sleep'):
                result=io.storage({'output_root':str(output)})
            self.assertTrue(called[0]);self.assertEqual(result['entries'],3)

    def test_api_intent_published_before_exec_is_pending_not_ready(self):
        obj=object.__new__(adapter.Phase)
        obj.adapter=SimpleNamespace(observer=SimpleNamespace(check=lambda:None))
        obj.service=SimpleNamespace(live=lambda:None)
        obj.stage=Path('/control');obj.directory=Path('/capture');obj.phase_ref={'path':'/phase','sha256':'a'*64}
        obj.child={'pid':8,'start_ticks':'9','ppid':7,'cgroup':'0::/owned','argv':['/python','bootstrap']}
        intent={'phase_spec':obj.phase_ref,'bootstrap_owner':obj.child,'argv':['/python','original-entry']}
        fake=SimpleNamespace(proc=lambda pid:obj.child)
        with patch.object(adapter,'read_if_present',return_value=intent) as reader,patch.object(adapter,'process_api',return_value=fake),patch.object(adapter.urllib.request,'urlopen') as request:
            self.assertIsNone(obj.api_ready())
        reader.assert_called_once_with(obj.stage/'api-exec-intent.json');request.assert_not_called()

    def test_declared_scope_rejects_owner_receipt_with_vanished_unit(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec={'run_name':'test','output_root':str(Path(tmp)/'out')}
            co={'process':{'pid':1},'service_owner':{'unit':contract.units('test')['coordinator'],'invocation_id':'a'*32}}
            admission={'path':'/admission','sha256':'a'*64}
            phase=service.expected_phase(spec,admission,co,'run','baseline','long',spec['output_root']+'/baseline/model')
            stage=Path(phase['control']);stage.mkdir(parents=True)
            io.once(stage/'phase-spec.json',phase)
            io.once(stage/'supervisor-owner.json',{'process':{'pid':2},'service_owner':{'unit':phase['unit'],'invocation_id':'b'*32}})
            fake=SimpleNamespace(empty_group=lambda group:{},GROUP_PREFIX='/fake/')
            with patch.object(adapter,'properties',return_value={'LoadState':'not-found','InvocationID':''}),patch.object(adapter,'process_api',return_value=fake):
                with self.assertRaisesRegex(ValueError,'previously acknowledged service disappeared'):
                    adapter.declared_services(spec,admission,co)

    def test_outer_uses_actual_drained_coordinator_object_before_lease_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec={'run_name':'test','output_root':str(Path(tmp)/'out'),'running_handoff_path':str(Path(tmp)/'handoff.json'),
                  'source_sha256':{'/p.py':'a'*64},'current_serial_catalog':'catalog','history_overlay':'overlay'}
            control=contract.control_root(spec);events=[]
            lock=SimpleNamespace(acquire=lambda:events.append('acquire'),check=lambda:None,owner={'pid':1},fd=8,initial={},
                                 release=lambda drained:events.append(('release',drained)))
            original=SimpleNamespace(unit=contract.units('test')['coordinator'],owner={'process':{'pid':2}},terminal=None,started=True)
            replacement=SimpleNamespace(unit=original.unit,owner=original.owner,terminal=None,started=True)
            terminal={'systemd_exit_status':{'ExecMainCode':1,'ExecMainStatus':0,'Result':'success'},'cleanup_complete':True}
            def cleanup(): replacement.terminal=terminal;events.append('drained');return terminal
            replacement.cleanup=cleanup
            original.observe=lambda:{'MainPID':'0'}
            replacement.observe=lambda:{'MainPID':'0'}
            bundle=SimpleNamespace(spec=spec,names=contract.units('test'),resolved={'plan_sha256':'plan'},host={'/p.py':'a'*64})
            bundle.resolve=lambda:bundle;bundle.physical_pair=lambda authority:{};bundle.freshness=lambda:{'latest_handoff':'old'}
            def service_factory(value,admission_ref):
                def start(guard):
                    io.once(control/'coordinator-owner.json',original.owner)
                    Path(spec['running_handoff_path']).write_text('{}')
                    io.once(control/'coordinator-result.json',{'admission':admission_ref,'coordinator':original.owner})
                original.start=start;return original
            fakep=SimpleNamespace(active_heavy=lambda *a:[],empty_group=lambda x:{},GROUP_PREFIX='/fake/',terminal=lambda v:v['MainPID']=='0')
            running={'observation':{'path':'actual','identity':{}},'reference':{},'value':{}}
            fake_latest=SimpleNamespace(latest_handoff=lambda:running['observation'])
            with ExitStack() as stack:
                stack.enter_context(patch('builtins.print'))
                for name,value in [('RetainedLock',lambda:lock),('Bundle',lambda *a:bundle),('resources',lambda *a,**k:{}),
                    ('process_api',lambda:fakep),('graph_reuse',lambda *a:{}),('properties',lambda *a:{'LoadState':'not-found','InvocationID':''}),
                    ('coordinator_service',service_factory),('admit_shape',lambda *a:None),('check_running',lambda *a:running),
                    ('declared_services',lambda *a:[replacement])]:stack.enter_context(patch.object(adapter,name,value))
                stack.enter_context(patch.dict(sys.modules,{'research.minicpm5_layer1_operator_sdk_runtime_completion_v4':fake_latest}))
                # The package attribute must agree with the patched module.
                import research
                stack.enter_context(patch.object(research,'minicpm5_layer1_operator_sdk_runtime_completion_v4',fake_latest,create=True))
                result=adapter.outer({'path':'/spec','sha256':'a'*64},{'path':'/review','sha256':'b'*64})
            self.assertEqual(result,0);self.assertIsNone(original.terminal)
            saved=json.loads((control/'outer-terminal.json').read_text())
            self.assertEqual(saved['complete_owned_unit_terminals'],[terminal])
            self.assertLess(events.index('drained'),events.index(('release',True)))

    def test_phase_finish_is_idempotent_after_startup_error(self):
        calls=[];end={'cleanup_complete':True}
        obj=object.__new__(adapter.Phase)
        obj.end=end;obj.service=SimpleNamespace(started=True,drained=lambda:calls.append('drain'))
        obj.adapter=SimpleNamespace(observer=SimpleNamespace(current='old'),active=obj)
        self.assertIs(obj.finish(),end)
        self.assertEqual(calls,['drain'])
        self.assertIsNone(obj.adapter.active)

    def test_known_rename_disappearance_gets_finite_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'out';root.mkdir();spec={'output_root':str(root)}
            (root/'cold.json.tmp').write_text('{}')
            original=io.observed_stat; once=[False]
            def interleaved(path):
                if path.name=='cold.json.tmp' and not once[0]:
                    once[0]=True;path.replace(root/'cold.json')
                return original(path)
            with patch.object(io,'observed_stat',side_effect=interleaved),patch.object(io.time,'sleep'):
                result=io.storage(spec)
            self.assertTrue(once[0]);self.assertEqual(result['entries'],2)
            with patch.object(io,'storage_scan',side_effect=io.PublicationRace('still writing')),patch.object(io.time,'sleep') as sleep:
                with self.assertRaises(ValueError):io.storage(spec)
                self.assertEqual(sleep.call_count,7)

    def test_unknown_disappearance_symlink_and_unbounded_file_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'out';root.mkdir();spec={'output_root':str(root)}
            (root/'unknown').symlink_to('/etc/passwd')
            with self.assertRaises(ValueError):io.storage(spec)
            (root/'unknown').unlink()
            with (root/'payload.pt').open('wb') as f:f.truncate(contract.LIMITS['model_single_file_bytes']+1)
            with self.assertRaises(ValueError):io.storage(spec)
            with self.assertRaises(FileNotFoundError):io.observed_stat(root/'missing.json')

    def test_native_preflight_has_actual_stride0_and_expanded_rejection(self):
        text=(Q/'qwen35_sampler_runtime_preflight_v1.py').read_text(); tree=ast.parse(text)
        self.assertIn('torch.as_strided(base, (1,), (0,))',text)
        self.assertIn('expand(4)',text)
        self.assertIn('nonempty_expanded_stride0_rejected',text)
        self.assertIn('general_overlapping_layout_support_claimed',text)
        self.assertFalse(any(isinstance(n,(ast.Import,ast.ImportFrom)) and
                             ((isinstance(n,ast.ImportFrom) and n.module=='torch') or
                              (isinstance(n,ast.Import) and any(a.name=='torch' for a in n.names))) for n in tree.body))
        from qwen35_sampler_pair_protocol_v1 import SAMPLER_GATES
        self.assertEqual(len(SAMPLER_GATES['controls']),13)

    def test_real_native_result_receipt_cannot_be_systemd_run_returncode(self):
        text=(Q/'qwen35_sampler_runtime_service_v3.py').read_text()
        self.assertIn('process.wait(timeout=15)',text)
        self.assertIn('"child_returncode": code',text)
        self.assertIn('"actual_popen_wait_observed": True',text)
        self.assertIn('close_fds=True',text)
        self.assertIn('start_new_session=True',text)
        adapter_text=(Q/'qwen35_sampler_runtime_adapter_v3.py').read_text()
        self.assertIn('child_returncode"] == 0',adapter_text)
        self.assertIn('"ExecMainCode": 1, "ExecMainStatus": 0, "Result": "success"',adapter_text)
        self.assertNotIn('orchestrate.run_case',adapter_text)


if __name__=='__main__':unittest.main(verbosity=2)
