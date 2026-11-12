"""Stdlib source/mock tests. No actual lock, systemd, native import or compiler."""
import ast
import contextlib
import copy
import hashlib
import importlib.abc
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
from typing import Dict
import unittest
from unittest.mock import Mock, patch

Q=Path(__file__).resolve().parent
sys.path[:0]=[str(Q),str(Q.parent)]
class RejectNative(importlib.abc.MetaPathFinder):
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in {'torch','torch_xla','torch_neuronx','neuronxcc','numpy','ml_dtypes','transformers','vllm','libneuronxla','ctypes','_ctypes'}:
            raise AssertionError('Native import forbidden: '+name)
sys.meta_path.insert(0,RejectNative())
import qwen35_integer_reference_serial_contract_v1 as c
import qwen35_integer_reference_serial_v1 as s
import qwen35_sampler_runtime_service_v12 as original
import qwen35_predicated_slot_sdk_environment_v1 as sdk_env
from qwen35_package_preflight_history_v2 import History


def fixture_spec():
    spec=dict(schema='qwen35_integer_reference_serial_spec_v1',run_name='mock',
        output_root='/mnt/nvme/neuron-work/qwen35-integer-reference-slot-mock',
        history_overlay=dict(path='/overlay.json',sha256='1'*64),current_serial_catalog=dict(path='/catalog.json',sha256='2'*64),
        source_sha256={str(Q/'qwen35_integer_reference_serial_v1.py'):'3'*64},input_extension={row['path']:dict(sha256=row['sha256'],identity={}) for row in c.protocol.REFERENCE_EVIDENCE},
        running_handoff_path=str(Q/'continuation-handoff-mock.json'),lfm_r2_scope=dict(path='/lfm.json',sha256='4'*64),
        source_delivery=dict(path='/source-delivery.json',sha256='5'*64),primitive_manifest=copy.deepcopy(c.MANIFEST),
        workload=c.protocol.workload(),limits=copy.deepcopy(c.LIMITS),lock_identity=copy.deepcopy(c.LOCK_IDENTITY),
        native_python=c.PYTHON,automatic_successor=False)
    ref=dict(path='/future-spec.json',sha256='6'*64)
    review=dict(schema='qwen35_sampler_runtime_source_review_v1',reviewer='/root',passed=True,blockers=[],
        spec=ref,source_sha256=spec['source_sha256'],launch_authorized=True)
    return spec,ref,review


def fixture_report(name='conv'):
    case=c.protocol.layout(name); artifacts=[]; graphs=[]
    for mode in c.protocol.MODES:
        for variant in c.protocol.VARIANTS:
            artifact=dict(path='/mock/'+name+'/'+mode+'-'+variant+'/model.pt',size_bytes=4,sha256='a'*64,layout=name,mode=mode,variant=variant)
            artifacts.append(artifact)
            graphs.append(dict(mode=mode,variant=variant,artifact=artifact,compile_seconds=1.,
                input_output_alias_indices=[1] if mode=='persistent_alias' else [],state_name='cache' if mode=='persistent_alias' else None))
    checks=[]
    for descriptor in c.expected_calls():
        keys=['updated_output','updates_unchanged','slots_unchanged']+(['cache_input_unchanged'] if descriptor['mode']=='standalone' else ['old_read','persistent_state'])
        checks.append(dict(descriptor,passed=True,elapsed_ns=10 if descriptor['phase']=='measured' else None,
            checks={key:dict(passed=True,shape_equal=True,dtype_equal=True,bitwise_equal=True,
                bytes_compared=32 if key=='slots_unchanged' else case['tensor_bytes'],actual_sha256='a'*64,expected_sha256='a'*64) for key in keys}))
    timing={}
    for mode in c.protocol.MODES:
        rows=[dict(round=i,order=list(c.protocol.paired_order(i)),elapsed_ns={variant:[10]*16 for variant in c.protocol.VARIANTS}) for i in range(10)]
        timing[mode]=dict(paired_rows=rows,complete=True,**c.protocol.summarize_pairs(rows))
    words=((0,0x8000),(0x7fc1,0x7fc2)) if name!='recurrent' else ((0,0x80000000),(0x7fc00001,0x7fc00002))
    report=dict(schema='qwen35_integer_reference_primitive_result_v1',case=case,reference_implementation=dict(c.protocol.REFERENCE_IMPLEMENTATION),source_sha256=dict(c.protocol.FROZEN_SOURCES),
        complete=True,passed=True,graphs=graphs,checks=checks,timing=timing,controls=[dict(left_word=a,right_word=b,passed=True) for a,b in words],
        runtime_counts={key:c.protocol.counts()['per_layout'][key+'_calls'] for key in ('correctness','warmup','measured','reset')},
        timing_scope=c.protocol.TIMING_SCOPE,full_model_qualified=False,model_speedup_claim=False,lfm105_tasks_replaced=False,
        required_resources=dict(c.protocol.RESOURCES))
    return report,artifacts


class ContractTests(unittest.TestCase):
    def test_exact_workload_and_original_reuse(self):
        spec,ref,review=fixture_spec(); self.assertEqual(len(c.validate(spec,review,ref)),1)
        self.assertIs(s.History,History); self.assertIs(s.Service,original.Service)
        self.assertEqual(c.protocol.counts()['total']['total_runtime_calls'],46368)
        self.assertEqual(c.PROPERTIES['MemoryMax'],str(6*c.GIB)); self.assertEqual(c.PROPERTIES['MemoryHigh'],str(5*c.GIB))
        self.assertEqual(c.PROPERTIES['MemorySwapMax'],'0'); self.assertEqual(c.PROPERTIES['RuntimeMaxSec'],'7200')
        self.assertEqual(c.LIMITS['lfm_separate_required_free_bytes_unchanged'],54*c.GIB)
        self.assertEqual(len(c.LAYOUTS),8); self.assertEqual(c.LIMITS['cpus'],[6,7])

    def test_no_contract_shrink_or_relaxation(self):
        for key,value in [('timeout_seconds_per_layout',901),('memory_max_bytes',12*c.GIB),('output_cap_bytes_total',9*c.GIB),('fresh_processes',7)]:
            spec,ref,review=fixture_spec(); spec['limits'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError): c.validate(spec,review,ref)
        spec,ref,review=fixture_spec(); spec['workload']['correctness_mappings'].pop()
        with self.assertRaises(ValueError): c.validate(spec,review,ref)

    def test_env_and_argv_nc0_offline_per_layout_cache(self):
        work=Path('/mock/kv4096'); env=c.environment(work)
        self.assertEqual(env['NEURON_RT_VISIBLE_CORES'],'0'); self.assertEqual(env['NEURON_COMPILE_CACHE_URL'],'/mock/kv4096/cache/neuron')
        self.assertEqual(env['TMPDIR'],'/mock/kv4096/tmp'); self.assertNotIn('NEURON_CC_FLAGS',env)
        command=s.entry_argv('layout',dict(path='/binding.json',sha256='a'*64),work)
        self.assertEqual(command[:5],[c.PYTHON,'-I','-B','-X','pycache_prefix=/mock/kv4096/bytecode-never-created'])

    def test_all_eight_complete_reports(self):
        count=0
        for name in c.LAYOUTS:
            report,artifacts=fixture_report(name); c.validate_report(report,name,artifacts)
            self.assertLess(len(json.dumps(report).encode()),16*c.MIB)
            count+=len(report['checks'])
        self.assertEqual(count,46368)

    def test_report_rejects_missing_old_state_short_bytes_and_forged_timing(self):
        mutations=[lambda r:r['checks'].pop(),lambda r:r['checks'][0]['slots'].__setitem__(0,3),
            lambda r:r.__setitem__('schema','qwen35_predicated_slot_primitive_result_v1'),
            lambda r:r['reference_implementation'].__setitem__('original_model_baseline_speedup_claim',True),
            lambda r:r['reference_implementation'].__setitem__('integer_select_types',['s16','s16']),
            lambda r:r['timing']['standalone'].pop('reference_implementation'),
            lambda r:r['graphs'][2].__setitem__('input_output_alias_indices',[]),
            lambda r:r['controls'][1].__setitem__('passed',False),
            lambda r:r['runtime_counts'].__setitem__('reset',0),
            lambda r:r['checks'][0]['checks']['updated_output'].__setitem__('bytes_compared',4),
            lambda r:r['timing']['standalone']['paired_rows'][0]['elapsed_ns']['candidate'].__setitem__(0,11),
            lambda r:next(x for x in r['checks'] if x['mode']=='persistent_alias')['checks'].pop('old_read'),
            lambda r:next(x for x in r['checks'] if x['mode']=='persistent_alias')['checks']['persistent_state'].__setitem__('passed',False)]
        for index,mutation in enumerate(mutations):
            report,artifacts=fixture_report(); mutation(report)
            with self.subTest(index=index),self.assertRaises((ValueError,KeyError)): c.validate_report(report,'conv',artifacts)

    def test_full_history_transport_not_projection(self):
        from qwen35_package_preflight_serial_v2 import history_parts,join_history
        value=dict(catalog={'runs':list(range(145))},children=list(range(874)),original_model_exit_code=1)
        manifest,parts=history_parts(value); self.assertEqual(join_history(manifest,parts),value)
        with self.assertRaises(ValueError): join_history(manifest,parts[:-1])

    def test_original_read_reference_shape_for_frozen_manifest_and_history(self):
        # These are actual accepted metadata/source reads, never native payloads.
        from prepare_qwen35_integer_reference_serial_v1 import static_local_sources,reference
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); refs=s.save_history(root,dict(original_model_exit_code=1,all_children=list(range(874))))
            manifest=s.read_history_reference(refs[0]); parts=[s.read_history_reference(row) for row in refs[1:]]
            self.assertEqual(s.join_history(manifest,parts)['all_children'],list(range(874)))
            with self.assertRaises(ValueError):s.read_history_reference(dict(refs[0],bytes=0))
            # Real manifest, real complete inventories, fake source-only delivery path.
            local=static_local_sources(); sdk={str(path):reference(path)['sha256'] for name in c.SDK_ROOTS for path in (c.SITE/name).rglob('*.py')}
            delivery=dict(schema='qwen35_integer_reference_serial_source_delivery_v1',primitive_manifest=c.MANIFEST,source_only=True,
                workload=c.protocol.workload(),limits=c.LIMITS,static_local_source_sha256=local,sdk_source_sha256=sdk)
            path=root/'delivery.json'; path.write_text(json.dumps(delivery)); spec=dict(source_delivery=s.ref(path),source_sha256=dict(local,**sdk),input_extension={row['path']:dict(sha256=row['sha256'],identity={}) for row in c.protocol.REFERENCE_EVIDENCE})
            self.assertEqual(c.verify_delivery(spec)['primitive_manifest'],c.MANIFEST)
            for evidence in c.protocol.REFERENCE_EVIDENCE:
                missing=copy.deepcopy(spec); missing['input_extension'].pop(evidence['path'])
                forged=copy.deepcopy(spec); forged['input_extension'][evidence['path']]['sha256']='0'*64
                for invalid in (missing,forged):
                    with self.subTest(evidence=evidence['path']),self.assertRaisesRegex(ValueError,'actual select evidence'):
                        c.verify_delivery(invalid)


class StorageTests(unittest.TestCase):
    def test_atomic_complete_report_update_and_oversize_preserves_previous(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'report.json'; c.update_json(path,dict(complete=False,checks=[1]),256)
            c.update_json(path,dict(complete=True,checks=list(range(10))),256)
            before=path.read_bytes(); self.assertTrue(json.loads(before)['complete'])
            with self.assertRaises(ValueError): c.update_json(path,dict(checks=list(range(1000))),256)
            self.assertEqual(path.read_bytes(),before); self.assertFalse(path.with_name('report.json.tmp').exists())

    def test_sparse_file_caps_and_cache_counted(self):
        for name,size in [('model.pt',128*c.MIB+1),('report.json',16*c.MIB+1),('native.log',8*c.MIB+1)]:
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory); work=root/'conv'; work.mkdir(); path=work/name
                with path.open('wb') as stream: stream.truncate(size)
                with self.subTest(name=name),self.assertRaises(ValueError): c.scan_output(root)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); work=root/'conv'/'cache'; work.mkdir(parents=True)
            for name in ('a','b'):
                with (work/name).open('wb') as stream: stream.truncate(600*c.MIB)
            with self.assertRaisesRegex(ValueError,'Each layout'): c.scan_output(root)

    def test_total_includes_common_metadata_and_no_external_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in c.LAYOUTS:
                (root/name).mkdir()
                with (root/name/'cache-data').open('wb') as stream: stream.truncate(c.GIB)
            (root/'overhead.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'Total output'): c.scan_output(root)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'alias').symlink_to('/tmp')
            with self.assertRaises(ValueError): c.scan_output(root)

    def test_900_second_clock_cannot_be_reset_or_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            c.update_json(root/'current-layout.json',dict(name='conv',started=100.,active=True),4096)
            with patch.object(c.time,'monotonic',return_value=999.): self.assertEqual(c.current_layout(root)['name'],'conv')
            with patch.object(c.time,'monotonic',return_value=1000.),self.assertRaisesRegex(ValueError,'900 second'): c.current_layout(root)

    def test_initial_40_gib_and_remaining_reservation_formula(self):
        # Exercise the real resources function with only OS observations replaced.
        spec,_,_=fixture_spec()
        fake_stat=lambda path: SimpleNamespace(st_dev=1 if path==c.ROOT else 2)
        def trial(free,allocated=0):
            with patch.object(c,'scan_output',return_value={'allocated_bytes':allocated}),patch.object(c,'current_layout'),\
                    patch.object(Path,'is_mount',return_value=True),patch.object(Path,'stat',autospec=True,side_effect=fake_stat),\
                    patch.object(c.os,'statvfs',side_effect=lambda path:SimpleNamespace(f_bavail=28*c.GIB if path==c.ROOT else free,f_frsize=1)),\
                    patch.object(Path,'read_text',return_value='MemAvailable: 4194304 kB\n'):
                return c.resources(spec)
        trial(40*c.GIB)
        with self.assertRaises(ValueError): trial(40*c.GIB-1)
        trial(37*c.GIB,3*c.GIB)
        with self.assertRaises(ValueError): trial(37*c.GIB-1,3*c.GIB)


class ScheduleTests(unittest.TestCase):
    def exercise(self,root,fail=None):
        spec,_,_=fixture_spec(); spec['output_root']=str(root); root.mkdir()
        owner=dict(process=dict(pid=1,start_ticks='2',cgroup='0::/owned'),service_owner={}); seen=[]; drained=[]
        def capture(command,env,work,binding_ref,own,guard,started):
            name=work.name; seen.append(name)
            self.assertFalse((work/'artifacts').exists())
            if name==fail: raise ValueError('injected layout failure')
            s.once(work/'leaf-result.json',dict(layout=name)); return {'layout':name}
        with patch.object(s,'only_supervisor',side_effect=lambda *args:drained.append(len(seen))),\
                patch.object(s,'capture_layout',side_effect=capture),patch.object(s,'validate_layout',side_effect=lambda root,name,*args:dict(layout=name)):
            if fail:
                with self.assertRaisesRegex(ValueError,'injected'): s.run_layouts(spec,dict(path='/a',sha256='a'*64),owner,lambda:None)
            else: self.assertEqual([x['layout'] for x in s.run_layouts(spec,dict(path='/a',sha256='a'*64),owner,lambda:None)],list(c.LAYOUTS))
        return seen,drained

    def test_exact_eight_fresh_sequential_layouts(self):
        with tempfile.TemporaryDirectory() as directory:
            seen,drained=self.exercise(Path(directory)/'run')
            self.assertEqual(seen,list(c.LAYOUTS)); self.assertEqual(drained,list(range(8)))

    def test_failure_has_no_retry_or_successor(self):
        with tempfile.TemporaryDirectory() as directory:
            seen,drained=self.exercise(Path(directory)/'run',fail='kv256')
            self.assertEqual(seen,list(c.LAYOUTS[:4])); self.assertEqual(drained,list(range(4)))

    def test_descendant_blocks_next_layout(self):
        own=dict(pid=1,start_ticks='2'); api=SimpleNamespace(proc=lambda pid:own)
        with patch.object(s,'process_api',return_value=api),patch.object(s,'group_members',return_value={1,2}),self.assertRaises(ValueError): s.only_supervisor('/owned',own)


class CaptureTests(unittest.TestCase):
    def exercise(self,root,code=0,descendant=False,timeout=False):
        command=[s.PYTHON,'-I','-B','/mock-leaf.py']; stdout=SimpleNamespace(fileno=lambda:99,close=Mock())
        child=SimpleNamespace(pid=333,stdout=stdout,poll=lambda:code,wait=Mock(return_value=code),returncode=code)
        active=[True]; selector=SimpleNamespace(register=lambda *a:None,get_map=lambda:{99:True} if active[0] else {},
            select=lambda *a:[(SimpleNamespace(fileobj=stdout),None)],unregister=lambda *a:active.__setitem__(0,False),close=Mock())
        own=dict(pid=222,cgroup='0::/owned'); native=dict(pid=333,ppid=222,start_ticks='99',argv=command,cgroup='0::/owned')
        api=SimpleNamespace(proc=lambda pid:native if pid==333 else own); original_read=os.read; blocks=iter([b'log',b''])
        read_pipe=lambda fd,size:next(blocks) if fd==99 else original_read(fd,size)
        with patch.object(s.subprocess,'Popen',return_value=child) as popen,patch.object(s.selectors,'DefaultSelector',return_value=selector),\
                patch.object(s,'process_api',return_value=api),patch.object(s.os,'read',side_effect=read_pipe),\
                patch.object(s,'only_supervisor',side_effect=ValueError('undrained descendant') if descendant else None):
            start=time.monotonic()-901 if timeout else time.monotonic()
            if code or descendant or timeout:
                with self.assertRaises(ValueError): s.capture_layout(command,{},root,dict(path='/b',sha256='b'*64),own,lambda:None,start)
            else: self.assertEqual(s.capture_layout(command,{},root,dict(path='/b',sha256='b'*64),own,lambda:None,start)['actual_returncode'],0)
        self.assertEqual(popen.call_count,1); self.assertTrue(popen.call_args.kwargs['close_fds'])
        child.wait.assert_called_once()
        receipt=json.loads((root/'native-process-exit.json').read_text()); self.assertEqual(receipt['actual_returncode'],code)
        self.assertFalse(receipt['self_exit_claimed']); self.assertTrue(receipt['popen_wait_observed'])

    def test_actual_popen_wait_exit_and_descendant_gate(self):
        for code,descendant,timeout in [(0,False,False),(7,False,False),(0,True,False),(0,False,True)]:
            with tempfile.TemporaryDirectory() as directory,self.subTest(code=code,descendant=descendant,timeout=timeout):
                self.exercise(Path(directory),code,descendant,timeout)


class LeafCallbackTests(unittest.TestCase):
    def test_actual_layout_adapter_supplies_guard_complete_reports_and_artifact_records(self):
        import qwen35_integer_reference_device_leaf_v1 as leaf
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); work=root/'conv'; work.mkdir(); (work/'tmp').mkdir(); (work/'cache').mkdir()
            spec,_,_=fixture_spec(); spec['output_root']=str(root)
            spec['source_sha256'].update(sdk_env.SOURCE_SHA256)
            sdk_environ=c.environment(work); sdk_contexts=actual_contexts(sdk_environ)
            original_environment=sdk_contexts.environment
            supervisor=dict(process=dict(pid=222,start_ticks='22',cgroup='0::/owned'),service_owner={})
            admission=dict(spec={'path':'/mock-spec','sha256':'a'*64},review={'path':'/mock-review','sha256':'b'*64},outer_owner={},retained_fd=9)
            admission_ref=s.once(root/'admission.json',admission); s.once(root/'supervisor-owner.json',supervisor)
            binding=dict(schema='qwen35_integer_reference_layout_binding_v1',admission=admission_ref,layout='conv',supervisor=supervisor,started=time.monotonic())
            binding_ref=s.once(work/'layout-binding.json',binding)
            own=dict(pid=333,start_ticks='33',ppid=222,cgroup='0::/owned',argv=s.entry_argv('layout',binding_ref,work))
            s.once(work/'layout-release.json',dict(binding=binding_ref,owner=own)); events=[]
            def fake_leaf(name,output_dir,*,guard,save_report,record_artifact):
                self.assertEqual(name,'conv'); self.assertEqual(output_dir,work/'artifacts'); self.assertFalse(output_dir.exists())
                guard('before_native_imports'); events.append('guard'); output_dir.mkdir()
                report,artifacts=fixture_report(name); save_report(dict(report,complete=False,passed=False)); events.append('initial-save')
                sdk_environ.update(sdk_env.INITIALIZED)
                guard('before_compile:standalone:reference')
                with sdk_contexts.mock_neuron_cores(),sdk_contexts.set_pjrt_device(True):
                    self.assertEqual(sdk_environ['PJRT_DEVICE'],'CPU')
                guard('after_compile:standalone:reference')
                self.assertEqual(sdk_environ['PJRT_DEVICE'],'NEURON');self.assertNotIn('key',sdk_environ)
                for graph,info in zip(report['graphs'],artifacts):
                    graph_dir=output_dir/(info['mode']+'-'+info['variant']); graph_dir.mkdir(); path=graph_dir/'model.pt'; path.write_bytes(b'mock')
                    info.update(path=str(path),size_bytes=4,sha256=hashlib.sha256(b'mock').hexdigest()); record_artifact(info)
                save_report(report); events.append('final-save'); return report
            service=SimpleNamespace(live=Mock())
            with patch.object(s.os,'environ',sdk_environ),patch.dict(sys.modules,{'torch_neuronx.contexts':sdk_contexts}),patch.object(s,'check_process_settings'),\
                    patch.object(s,'load_spec',return_value=spec),patch.object(s,'check_environment'),patch.object(s,'process_api',return_value=SimpleNamespace(proc=lambda pid:own)),\
                    patch.object(s,'Service',return_value=service),patch.object(s,'supervisor_argv',return_value=['mock']),\
                    patch.object(s,'descriptor'),patch.object(s,'check_group'),patch.object(s,'check_ack'),patch.object(s,'loaded_sources'),\
                    patch.object(s.sys,'executable',c.PYTHON),patch.object(s.sys,'version_info',(3,10)),patch.object(leaf,'run_case',side_effect=fake_leaf) as call:
                self.assertEqual(s.layout(binding_ref),0); call.assert_called_once()
            receipt=s.read(s.ref(work/'leaf-result.json')); self.assertEqual(receipt['owner'],own); self.assertFalse(receipt['self_exit_claimed'])
            self.assertEqual(events,['guard','initial-save','final-save']); service.live.assert_called()
            self.assertEqual(len(list(work.glob('artifact-*.json'))),4)
            self.assertIs(sdk_contexts.environment,original_environment)
            self.assertNotIn('torch',sys.modules)


class OriginalServiceTests(unittest.TestCase):
    def service(self,root):
        return s.Service('trn1-qwen35-sampler-mock-predicated-slot',['/pinned/python','/pinned/slot.py'],{'PATH':'/usr/bin'},s.PROPERTIES,root/'service.log')

    def test_no_install_failure_uses_original_double_absence_drain(self):
        with tempfile.TemporaryDirectory() as directory:
            service=self.service(Path(directory)); empty={'LoadState':'not-found','InvocationID':''}
            api=SimpleNamespace(GROUP_PREFIX='/mock/',empty_group=Mock(return_value={'absent':True}),command=Mock(side_effect=ValueError('no install')))
            with patch.object(original,'properties',return_value=empty),patch.object(original,'process_api',return_value=api):
                with self.assertRaisesRegex(ValueError,'no install'):service.start(lambda:None)
                wait=Mock(side_effect=AssertionError('must not hold'))
                proof=s.retain_until_drained(service.cleanup,lambda:dict(physically_drained=True,terminal=s.owned_terminal(service)),Mock(),wait)
                self.assertTrue(proof['terminal']['never_loaded']); self.assertEqual(api.empty_group.call_count,3); wait.assert_not_called()

    def test_installed_failed_service_keeps_actual_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            service=self.service(Path(directory)); installed=[False]
            command=original.command(service.unit,service.argv,service.env,service.props,service.log)
            value=dict(service.props,Id=service.unit+'.service',LoadState='loaded',InvocationID='a'*32,RuntimeMaxUSec='2h',TimeoutStopUSec='15s',
                ExecStopPost='',Environment='',ExecStart='{ argv[]='+' '.join(command[command.index('/usr/bin/env'):])+' ; }',
                ControlGroup='/mock/'+service.unit+'.service',StandardOutput='append',StandardError='inherit',MainPID='0',ControlPID='0',Job='',
                ActiveState='failed',SubState='failed',ExecMainPID='987',ExecMainCode='1',ExecMainStatus='7',Result='exit-code')
            api=SimpleNamespace(GROUP_PREFIX='/mock/',command=lambda *a,**k:installed.__setitem__(0,True),empty_group=Mock(return_value={'absent':True}),
                proc=lambda pid:None,invocation=lambda x:x=='a'*32,terminal=lambda current:current['MainPID']=='0')
            with patch.object(original,'properties',side_effect=lambda unit:value if installed[0] else {'LoadState':'not-found','InvocationID':''}),\
                    patch.object(original,'process_api',return_value=api):
                with self.assertRaisesRegex(ValueError,'before actual owner'):service.start(lambda:None)
                service.cleanup(); terminal=s.owned_terminal(service)
                self.assertEqual(terminal['systemd_exit_status'],dict(ExecMainCode=1,ExecMainStatus=7,Result='exit-code'))
                self.assertTrue(terminal['exit_observed'])

    def test_external_terminal_must_match_running_session(self):
        terminal=dict(schema='qwen35_integer_reference_outer_terminal_v1',self_exit_claimed=False,actual_external_tool_exit_pending=True,
            whole_owned_scope_drained=True,admission={'path':'/a','sha256':'a'*64},outer_owner={'pid':1},failed=False,result_present=True)
        running=dict(qwen35_sampler_current_execution=dict(admission=terminal['admission'],outer_owner=terminal['outer_owner'],tool=dict(kind='tool_session',session_id=99)))
        tool=dict(kind='tool_session',session_id=99,exit_code=0,terminal_chunk='abc123')
        self.assertTrue(s.reconcile(terminal,running,tool,actual_owner_gone=True,actual_service_drained=True)['passed'])
        for key,value in [('session_id',100),('exit_code',False),('terminal_chunk','')]:
            with self.subTest(key=key),self.assertRaises(ValueError):s.reconcile(terminal,running,dict(tool,**{key:value}),actual_owner_gone=True,actual_service_drained=True)
        with self.assertRaises(ValueError):s.reconcile(terminal,running,tool,actual_owner_gone=False,actual_service_drained=True)


class OuterLifecycleTests(unittest.TestCase):
    def exercise(self,root,launch_failure=False,delayed_drain=False,result_failed=False):
        spec,spec_ref,_=fixture_spec(); spec['output_root']=str(root); spec['running_handoff_path']=str(root.parent/'running.json')
        events=[]; owner=dict(pid=42,start_ticks='77',ppid=1,argv=['outer'],cgroup='0::/outer')
        lock=SimpleNamespace(owner=owner,fd=9,initial={'descriptor':True},acquire=lambda:events.append('acquire'),
            check=lambda:None,release=lambda value:events.append(('release',value)))
        history=SimpleNamespace(resolve=lambda:history,physical_pair=lambda authority:{'both':'complete'},
            freshness=lambda:{'latest_handoff':{'path':'/previous.json','sha256':'0'*64}},unchanged=lambda:None,
            resolved={'plan_sha256':'a'*64},meta=SimpleNamespace(identities={}))
        terminal=dict(systemd_exit_status=dict(ExecMainCode=1,ExecMainStatus=0,Result='success'),cleanup_complete=True)
        service=SimpleNamespace(started=False,terminal=None,owner={'process':{'pid':43},'service_owner':{'unit':s.names(spec)['coordinator']}})
        attempts=[0]
        def start(guard):
            events.append('start'); service.started=True
            if launch_failure:raise ValueError('ambiguous launch')
            s.once(root/'supervisor-owner.json',service.owner); Path(spec['running_handoff_path']).write_text('{}')
            s.once(root/'supervisor-result.json',dict(complete=not result_failed,passed=not result_failed,
                admission=s.ref(root/'admission.json'),owner=service.owner))
        def drained():
            attempts[0]+=1; events.append('drain')
            if delayed_drain and attempts[0]==1:raise RuntimeError('resource tracker still live')
            return terminal
        service.start=start; service.observe=lambda:{}; service.cleanup=lambda:terminal; service.drained=drained
        api=SimpleNamespace(GROUP_PREFIX='/mock/',active_heavy=lambda *a:[],empty_group=lambda *a:{},terminal=lambda value:True)
        with patch.object(s,'load_spec',return_value=spec),patch.object(s,'RetainedLock',return_value=lock),\
                patch.object(s,'resources',return_value={}),patch.object(s,'History',return_value=history),\
                patch.object(s,'Service',return_value=service),patch.object(s,'process_api',return_value=api),\
                patch.object(s,'properties',return_value={'LoadState':'not-found','InvocationID':''}),\
                patch.object(s,'check_running',return_value={'observation':{},'value':{}}),patch.object(s,'audit_results'),\
                patch.object(s,'supervisor_argv',return_value=['/mock/supervisor']),\
                patch.object(s.time,'sleep',side_effect=lambda delay:events.append('retry')),patch('sys.stdout',new=io.StringIO()):
            import research
            latest=SimpleNamespace(latest_handoff=lambda:{})
            with patch.dict(sys.modules,{'research.minicpm5_layer1_operator_sdk_runtime_completion_v4':latest}),\
                    patch.object(research,'minicpm5_layer1_operator_sdk_runtime_completion_v4',latest,create=True):
                if launch_failure or result_failed:
                    with self.assertRaises((ValueError,RuntimeError)):s.outer(spec_ref,{'path':'/review.json','sha256':'b'*64})
                else:self.assertEqual(s.outer(spec_ref,{'path':'/review.json','sha256':'b'*64}),0)
        return events,json.loads((root/'outer-terminal.json').read_text())

    def test_success_root_ack_drain_then_unlock_and_full_history_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'out'; events,terminal=self.exercise(root)
            self.assertEqual(events[0],'acquire'); self.assertEqual(events[-1],('release',True))
            self.assertFalse(terminal['failed']); self.assertTrue(terminal['actual_external_tool_exit_pending']); self.assertFalse(terminal['self_exit_claimed'])
            admission=json.loads((root/'admission.json').read_text())
            self.assertGreaterEqual(len(admission['worker_history']),2); self.assertEqual(admission['workload'],c.protocol.workload())

    def test_partial_install_keeps_lease_until_eventual_drain(self):
        with tempfile.TemporaryDirectory() as directory:
            events,terminal=self.exercise(Path(directory)/'out',launch_failure=True,delayed_drain=True)
            self.assertTrue(terminal['failed']); self.assertEqual(events.count('start'),1); self.assertEqual(events.count('drain'),2)
            self.assertIn('retry',events); self.assertEqual(events[-1],('release',True))

    def test_zero_service_exit_does_not_promote_incomplete_result(self):
        with tempfile.TemporaryDirectory() as directory:
            events,terminal=self.exercise(Path(directory)/'out',result_failed=True)
            self.assertTrue(terminal['failed']); self.assertEqual(events[-1],('release',True))


def source_functions(relative, names, namespace):
    path = c.SITE / relative
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
             and node.name in names]
    assert {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def actual_contexts(environ):
    namespace = source_functions('torch_neuronx/contexts.py',
        {'environment', 'set_pjrt_device', 'mock_neuron_cores'},
        {'os': SimpleNamespace(environ=environ), 'Dict': Dict, 'contextmanager': contextlib.contextmanager})
    module = SimpleNamespace(__file__=sdk_env.CONTEXTS,
        **{name: namespace[name] for name in ('environment', 'set_pjrt_device', 'mock_neuron_cores')})
    # Original functions resolve the replacement through their real globals.
    class Module:
        __file__ = sdk_env.CONTEXTS
        def __getattr__(self, name): return namespace[name]
        def __setattr__(self, name, value): namespace[name] = value
    return Module()


class SDKEnvironmentTests(unittest.TestCase):
    def owner(self, optional=False):
        clean = c.environment(Path('/mock/slot-only'))
        env = dict(clean)
        owner = sdk_env.SDKEnvironment(clean, env, sdk_env.SOURCE_SHA256)
        owner.check('before_leaf'); owner.check('before_native_imports')
        env.update(sdk_env.INITIALIZED)
        if optional: env.update(sdk_env.OPTIONAL_INITIALIZED)
        owner.check('before_compile:standalone:reference')
        return owner, env

    def test_actual_sdk_default_functions_reproduce_frozen_v2_failure(self):
        import qwen35_predicated_slot_serial_contract_v2 as frozen
        work = Path('/mock/slot-environment-only'); env = frozen.environment(work)
        baseline = dict(env)
        fakeos = SimpleNamespace(environ=env, sched_getaffinity=lambda pid: {6,7},
            getpriority=lambda which,pid:19, PRIO_PROCESS=0)
        fake_sys = SimpleNamespace(dont_write_bytecode=True, flags=SimpleNamespace(isolated=True),
            pycache_prefix=str(frozen.bytecode_prefix(work)))
        with patch.object(frozen, 'os', fakeos), patch.object(frozen, 'sys', fake_sys):
            frozen.check_environment(work)
            namespace = source_functions('torch_xla/__init__.py',
                {'_set_missing_flags','_setup_xla_flags','_setup_default_env'},
                {'os': fakeos, 're': __import__('re'), 'tpu': SimpleNamespace(num_available_chips=lambda:0)})
            namespace['_setup_default_env'](); namespace['_setup_xla_flags']()
            self.assertEqual({key:value for key,value in env.items() if baseline.get(key)!=value},
                {key:sdk_env.INITIALIZED[key] for key in ('TF_CPP_MIN_LOG_LEVEL','GRPC_VERBOSITY','XLA_FLAGS')})
            with self.assertRaisesRegex(ValueError, 'Exact clean slot environment'): frozen.check_environment(work)

    def test_actual_pinned_initialization_functions_produce_exact_ten_defaults(self):
        env=c.environment(Path('/mock/slot-only'));clean=dict(env)
        fakeos=SimpleNamespace(environ=env)
        xla=source_functions('torch_xla/__init__.py',
            {'_set_missing_flags','_setup_xla_flags','_setup_default_env'},
            {'os':fakeos,'re':__import__('re'),'tpu':SimpleNamespace(num_available_chips=lambda:0)})
        xla['_setup_default_env']();xla['_setup_xla_flags']()
        lib=source_functions('libneuronxla/__init__.py',
            {'configure_environment','_maybe_setenv','_default_xla_ir_shape_cache_size','_default_rt_stochastic_rounding_enable'},
            {'_os':fakeos,'_libnrt':SimpleNamespace(check_compatible_libnrt=Mock())})
        xla_neuron=source_functions('torch_neuronx/xla.py',
            {'set_rt_root_comm_id','set_allreduce_bucket_size'}, {'os':fakeos})
        utilities=source_functions('torch_neuronx/utils/utils.py',{'_maybe_setenv'},{'os':fakeos})
        helper=Mock(return_value=SimpleNamespace(stdout=(sdk_env.INITIALIZED['NEURON_LIBRARY_PATH']+'\n').encode()))
        initializer=source_functions('torch_neuronx/initialization/initializer.py',
            {'Initializer','_disable_functionalization','_patch_zero1_redundancy_optimizer','_remove_unused_gpu_flags'},
            {'os':fakeos,'_torch_version_tuple':(2,8),'_maybe_setenv':utilities['_maybe_setenv'],
             'libneuronxla':SimpleNamespace(configure_environment=lib['configure_environment']),
             'subprocess':SimpleNamespace(run=helper,PIPE=-1),'logger':SimpleNamespace(info=Mock()),
             'set_rt_root_comm_id':xla_neuron['set_rt_root_comm_id'],'is_torchelastic_launched':lambda:False})
        initializer['Initializer']();xla_neuron['set_allreduce_bucket_size']()
        path=c.SITE/'torch_neuronx/__init__.py';tree=ast.parse(path.read_text())
        branches=[node for node in tree.body if isinstance(node,ast.If)
                  and 'JAX_LOGGING_LEVEL' in ast.unparse(node)]
        self.assertEqual(len(branches),1)
        exec(compile(ast.Module(body=branches,type_ignores=[]),str(path),'exec'),
             {'os':fakeos,'xla':SimpleNamespace(_torch_version_tuple=(2,8))})
        self.assertEqual({key:value for key,value in env.items() if key not in clean},sdk_env.INITIALIZED)
        helper.assert_called_once_with(['libneuronpjrt-path'],stdout=-1)
        self.assertEqual({key:env[key] for key in clean},clean)

    def test_actual_sdk_context_bug_normal_and_exception_paths(self):
        for initialized in (False,True):
            for exception in (False,True):
                env = {'PJRT_DEVICE':'NEURON'} if initialized else {}
                module = actual_contexts(env)
                try:
                    with module.mock_neuron_cores(), module.set_pjrt_device(True):
                        self.assertEqual(env, dict(sdk_env.MOCK_TRACE, **sdk_env.CPU_TRACE))
                        if exception: raise RuntimeError('actual mock trace failure')
                except RuntimeError: pass
                self.assertEqual(env, {'PJRT_DEVICE':'CPU','key':'NEURON'} if initialized else {})

    def test_exact_initialization_latch_and_all_fixed_inputs_preserved(self):
        for optional in (False,True):
            owner, env = self.owner(optional)
            for key,value in owner.clean.items(): self.assertEqual(env[key],value)
            self.assertEqual(owner.phase,'initialized'); owner.check('before_call:standalone:reference:measured')
            owner.check('after_leaf'); owner.close()

    def test_unexpected_inherited_partial_or_mutated_initialization_rejected(self):
        clean = c.environment(Path('/mock/slot-only'))
        with self.assertRaises(ValueError): sdk_env.SDKEnvironment(clean,dict(clean,PJRT_DEVICE='NEURON'),sdk_env.SOURCE_SHA256)
        for mutation in ('missing','unknown','cpu','fixed','optional','preinit'):
            env=dict(clean); owner=sdk_env.SDKEnvironment(clean,env,sdk_env.SOURCE_SHA256)
            if mutation=='preinit':
                env.update(sdk_env.INITIALIZED)
                with self.assertRaises(ValueError):owner.check('before_native_imports')
                continue
            owner.check('before_native_imports');env.update(sdk_env.INITIALIZED)
            if mutation=='missing':env.pop('XLA_IR_SHAPE_CACHE_SIZE')
            if mutation=='unknown':env['NEURON_FAKE_SWITCH']='1'
            if mutation=='cpu':env['PJRT_DEVICE']='CPU'
            if mutation=='fixed':env['NEURON_RT_VISIBLE_CORES']='1'
            if mutation=='optional':env['PJRT_DYNAMIC_PLUGINS']='0'
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):owner.check('before_compile:standalone:reference')

    def test_post_latch_missing_extra_cpu_and_branch_drift_rejected(self):
        for mutation in ('missing','extra','cpu','key','cache','latebranch','removebranch'):
            owner,env=self.owner(optional=mutation=='removebranch')
            if mutation=='missing':env.pop('GRPC_VERBOSITY')
            if mutation=='extra':env['NEURON_RT_STOCHASTIC_ROUNDING_EN']='1'
            if mutation=='cpu':env['PJRT_DEVICE']='CPU'
            if mutation=='key':env['key']='NEURON'
            if mutation=='cache':env['NEURON_COMPILE_CACHE_URL']='/other'
            if mutation=='latebranch':env['PJRT_DYNAMIC_PLUGINS']='1'
            if mutation=='removebranch':env.pop('PJRT_DYNAMIC_PLUGINS')
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):owner.check('before_call:standalone:reference:measured')

    def test_real_sdk_contexts_fixed_normal_exception_and_repeated_modes(self):
        for exception in (False,True):
            owner,env=self.owner(); initial=dict(env);module=actual_contexts(env);original=module.environment
            owner.install_context_fix(module)
            for index in range(4):
                error=RuntimeError('original trace error')
                try:
                    with module.mock_neuron_cores(), module.set_pjrt_device(True):
                        self.assertEqual(env,dict(initial,**sdk_env.MOCK_TRACE,**sdk_env.CPU_TRACE))
                        if exception:raise error
                except RuntimeError as caught:self.assertIs(caught,error)
                self.assertEqual(env,initial);self.assertNotIn('key',env)
                owner.check('after_compile:standalone:reference')
            owner.close();self.assertIs(module.environment,original)

    def test_context_fix_rejects_unknown_callers_values_and_drift(self):
        owner,env=self.owner();module=actual_contexts(env);owner.install_context_fix(module)
        with self.assertRaises(ValueError):module.environment(sdk_env.MOCK_TRACE)
        with self.assertRaises(ValueError):
            with module.set_pjrt_device(True):pass
        with self.assertRaises(ValueError):
            with module.mock_neuron_cores(),module.set_pjrt_device(True):env['NEURON_RT_VISIBLE_CORES']='2'
        self.assertEqual(env['PJRT_DEVICE'],'NEURON');self.assertNotIn('key',env)
        self.assertEqual(env['NEURON_RT_VISIBLE_CORES'],'2')  # Do not hide unauthorized mutations.
        with self.assertRaises(ValueError):owner.check('after_compile:standalone:reference')
        with self.assertRaises(ValueError):owner.close()

    def test_original_trace_error_preserved_when_environment_also_drifts(self):
        owner,env=self.owner();module=actual_contexts(env);original=module.environment;owner.install_context_fix(module)
        error=RuntimeError('actual trace failure')
        with self.assertRaises(RuntimeError) as caught:
            with module.mock_neuron_cores(),module.set_pjrt_device(True):
                env['unknown']='1';raise error
        self.assertIs(caught.exception,error);self.assertEqual(env['PJRT_DEVICE'],'NEURON')
        with self.assertRaises(ValueError):owner.close()
        self.assertIs(module.environment,original)

    def test_complete_source_pins_and_frozen_leaf_protocol_unchanged(self):
        sdk_env.verify_sources(sdk_env.SOURCE_SHA256)
        pins=dict(sdk_env.SOURCE_SHA256);pins.pop(next(iter(pins)))
        with self.assertRaises(ValueError):sdk_env.verify_sources(pins)
        import qwen35_predicated_slot_serial_contract_v2 as frozen
        self.assertEqual(c.LIMITS,frozen.LIMITS);self.assertEqual(c.PROPERTIES,frozen.PROPERTIES)
        self.assertEqual({k:v for k,v in c.protocol.verify_sources().items() if k in frozen.protocol.FROZEN_SOURCES},frozen.protocol.FROZEN_SOURCES)
        self.assertEqual(set(c.SDK_ROOTS)-set(frozen.SDK_ROOTS),{'libneuronxla'})


if __name__=='__main__':unittest.main(verbosity=2)
