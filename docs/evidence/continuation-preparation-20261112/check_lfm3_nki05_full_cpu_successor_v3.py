"""Pure metadata/source/mock integration checks; no real runtime or process."""
import ast
from contextlib import nullcontext
import copy
import hashlib
import importlib.abc
import json
from pathlib import Path
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class DenyNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','numpy','ml_dtypes','nki','neuronxcc','torch_neuronx','transformers','vllm'}:
            raise AssertionError('Native imports forbidden in pure v3 checks: ' + fullname)


sys.meta_path.insert(0, DenyNative())
import lfm3_nki05_full_cpu_leaf_v3 as science
import run_lfm3_nki05_full_cpu_serial_v3 as runner
import lfm3_full_cpu_known_warnings_v1 as known_warnings
import check_lfm3_compact_writeback_and_fixture_leaf_v1 as leaf_checks

ROOT=Path(__file__).resolve().parent
ORIGINAL=ROOT/'run_lfm3_nki05_transport_serial_v4.py'


def definitions(path):
    return {node.name:node for node in ast.parse(path.read_text()).body
            if isinstance(node,(ast.FunctionDef,ast.ClassDef))}


class Host:
    """Temporary synthetic JSON only, never real specs/catalog/owners/processes."""
    SELF='/source-only-mock.py'
    FILE_CAP=256*2**10
    environment=staticmethod(runner.environment)
    def __init__(self,root):
        self.root=root
        self.spec={'output_root':str(root)}
        self.admission={'spec':{'mock':True},'review':{'mock':True},'attempt_id':'0'*32}
        self.history=SimpleNamespace(serial=SimpleNamespace(sources=SimpleNamespace(load=lambda ref:self.admission)))
        self.p=SimpleNamespace(read=self.read,sha=self.sha)

    def bytecode_guard(self, root):
        assert Path(root) == self.root

    def warning_receipt(self, mode):
        return known_warnings.KnownWarningLog(mode, lambda *args, **kwargs:None).snapshot()

    def read_bundle(self,*args): return self.spec

    def sha(self,path):
        if Path(path)==science.legacy.p.RUNTIME_PLAN: return science.legacy.p.RUNTIME_PLAN_SHA
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def read(self,path,wanted=None):
        if Path(path)==science.legacy.p.RUNTIME_PLAN: return {'runtime':{}}
        raw=Path(path).read_bytes()
        if wanted is not None: assert hashlib.sha256(raw).hexdigest()==wanted
        return json.loads(raw)

    def once(self,path,value):
        raw=runner.encode(value)
        assert len(raw)<=self.FILE_CAP
        with Path(path).open('xb') as out: out.write(raw)

    def reference(self,path): return {'path':str(path),'sha256':self.sha(path)}

    def report_failure(self,path,error,**values): self.once(path,dict(error=str(error),**values))


def fake_outcome(host,stage,phase,begin,end,failed=False):
    return dict(stage=stage,phase=phase,begin=begin,end=end,pid=123,start_ticks='pure-mock',
                returncode=1 if failed else 0,passed=not failed,seconds=0.0,report=None,
                error='pure mock failure' if failed else None,secondary_errors=[])


def mock_coordinator(root,fail=None):
    host=Host(root);ref={'path':str(root/'admission.json'),'sha256':'0'*64};calls=[]
    def one(host,spec,admission_ref,admission,owner,root,stage,phase,begin,end,admit,timeout):
        assert 0<timeout<=14400
        calls.append((stage,phase,begin,end))
        failed=fail is not None and fail(stage,phase,begin,end)
        return fake_outcome(host,stage,phase,begin,end,failed), RuntimeError('pure mock failure') if failed else None
    runtime=SimpleNamespace(WORKER_PYCACHE=Path('/unused'),verify_runtime_distributions=lambda plan:[dict(files=1,bytes=1)]*7)
    with patch.dict(sys.modules,{'lfm3_movement_device_protocol_v3':runtime}),patch.object(science,'run_one_batch',one):
        try:
            science.execute_full_cpu(host,ref,{'pure_mock_owner':True},root,lambda:True)
        except RuntimeError:
            assert fail is not None
    return host,ref,calls


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.values=science.matrices()

    def test_01_imports_are_inert_and_old_new_source_syntax_compatible(self):
        self.assertFalse(any(name.split('.')[0] in {'numpy','ml_dtypes','torch','nki','neuronxcc'} for name in sys.modules))
        for file in (Path(science.__file__),Path(runner.__file__)):
            ast.parse(file.read_text(),feature_version=(3,10))
        self.assertEqual(hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(),'82344500580eaf04cae647e0d351ff1a46df9d44dca12769441f9c31ee185b88')
        self.assertEqual(hashlib.sha256(Path(science.legacy.__file__).read_bytes()).hexdigest(),'f34462be2e8286db80e2dadf5915debfb14bf0af5dd285a4457e66241c03176a')

    def test_02_retained_lock_history_proof_service_drain_functions_exact(self):
        before,after=definitions(ORIGINAL),definitions(Path(runner.__file__))
        unchanged=('encode','digest','reference','packed','unpacked','proof_parts','join_proof','save_proof',
                   'once','report_failure','Metadata','descriptor_proof','acquire','execution_source_paths',
                   'check_latest','worker_argv','service_command','service_identity',
                   'OwnedService','drain_and_release')
        for name in unchanged:
            self.assertEqual(ast.dump(before[name],include_attributes=False),ast.dump(after[name],include_attributes=False),name)
        self.assertEqual(ast.dump(ast.Module(body=before['outer'].body[-1].finalbody,type_ignores=[]),include_attributes=False),
                         ast.dump(ast.Module(body=after['outer'].body[-1].finalbody,type_ignores=[]),include_attributes=False))
        before_guard=next(node for node in ast.walk(before['outer']) if isinstance(node,ast.FunctionDef) and node.name=='guard')
        after_guard=next(node for node in ast.walk(after['outer']) if isinstance(node,ast.FunctionDef) and node.name=='guard')
        self.assertEqual(ast.dump(before_guard,include_attributes=False),ast.dump(after_guard,include_attributes=False))

    def test_03_only_explicit_full_pilot_limits_and_nvme_placement_changed(self):
        self.assertEqual((science.BATCH_SECONDS,science.PHASE_SECONDS,science.WORKER_SECONDS),(1800,14400,30000))
        self.assertEqual(runner.PROPS,dict(MemoryMax=str(10*2**30),MemoryHigh=str(8*2**30),MemorySwapMax='0',TasksMax='32',
            AllowedCPUs='6-7',CPUAffinity='6-7',Nice='19',KillMode='control-group',RemainAfterExit='yes',SendSIGKILL='yes',
            WorkingDirectory=str(ROOT),LimitFSIZE=str(256*2**10),LimitAS=str(12*2**30),LimitCPU='30000',
            RuntimeMaxSec='30000',TimeoutStopSec='15'))
        self.assertEqual((runner.OUTPUT_CAP,runner.MAX_FILES,runner.NVME_FLOOR,runner.FLOOR),(2**30,2048,30*2**30,28*2**30))
        self.assertEqual(runner.LOCK,dict(device=66306,inode=8294421))
        self.assertLess(874*2+7+4+64,runner.MAX_FILES)

    def test_04_all_original_cases_batches_in_three_complete_ordered_passes(self):
        expected=['original_fixture','original_fixture','new_fixture','new_fixture','controls','candidate','candidate']
        planned=science.schedule()
        self.assertEqual([row['stage'] for row in planned],expected)
        for stage in ('original_fixture','new_fixture','candidate'):
            entries=[row for row in planned if row['stage']==stage]
            self.assertEqual([row['phase'] for row in entries],list(science.PHASES))
            self.assertEqual(sum(len(row['bounds']) for row in entries),291)
            self.assertEqual(sum(end-begin for row in entries for begin,end in row['bounds']),3716)
            for row in entries: self.assertEqual(row['bounds'],science.legacy.p.batches(row['phase']))
        self.assertEqual(next(row for row in planned if row['stage']=='controls'),dict(stage='controls',phase='extended_cpu',bounds=[[0,1]]))
        self.assertEqual(science.protocol_manifest()['total_children'],874)

    def test_05_child_interpreters_and_arguments_are_fixed(self):
        ref={'path':'/pure/admission.json','sha256':'a'*64}
        for stage in science.STAGES:
            command=science.command(runner,ref,stage,'extended_cpu',0,16)
            self.assertEqual(command[0],science.OLD_PYTHON if stage=='original_fixture' else science.NEW_PYTHON)
            self.assertEqual(command[1:5],['-B',runner.SELF,'--mode','batch'])
            env=science.child_environment(runner,Path('/pure'),'0'*32,stage)
            self.assertEqual(env['TMPDIR'],str(Path('/pure')/stage/'tmp'))
            self.assertEqual(env['NEURON_RT_VISIBLE_CORES'],'')
        with self.assertRaises(RuntimeError): science.command(runner,ref,'device','extended_cpu',0,16)

    def test_06_concrete_spec_rejects_shortened_cases_limits_and_old_namespace(self):
        spec=dict(schema='lfm3_nki05_full_cpu_spec_v1',root_finalized=True,runner=runner.SELF,
            diagnostic=str(Path(science.__file__).resolve()),source_sha256={runner.SELF:'0'*64},
            serial_catalog={'path':str(ROOT/'research/source-only-catalog.json'),'sha256':'1'*64},
            terminal_handoff={'path':str(ROOT/'research/source-only-terminal.json'),'sha256':'2'*64},
            output_root='/mnt/nvme/lfm3-nki05-full-cpu-source-test',unit='trn1-lfm3-nki05-full-cpu-source-test',
            resources=copy.deepcopy(runner.PROPS),lock_identity=runner.LOCK,
            canonical_sha256={str(ROOT/'continuation-state.json'):'3'*64},root_floor_bytes=runner.FLOOR,
            output_cap_bytes=runner.OUTPUT_CAP,nvme_floor_bytes=runner.NVME_FLOOR,max_files=runner.MAX_FILES,
            cpu_protocol=science.protocol_manifest(),model_qualification_claim=False,automatic_successor_launch=False)
        self.assertEqual(runner.validate_spec(spec),spec)
        for mutate in (lambda value:value['cpu_protocol'].update(full_cases=32),
                       lambda value:value['resources'].update(RuntimeMaxSec='60000'),
                       lambda value:value.update(output_root=str(ROOT/'lfm3-nki05-transport-retry')),
                       lambda value:value.update(nvme_floor_bytes=5*2**30),
                       lambda value:value.update(automatic_successor_launch=True)):
            value=copy.deepcopy(spec);mutate(value)
            with self.assertRaises(RuntimeError): runner.validate_spec(value)

    def test_07_original_child_identity_stop_and_io_audit_are_reused(self):
        source=Path(science.__file__).read_text()
        for call in ('legacy.child_identity(child,argv)','legacy.stop_child(child,child_owner)',
                     'legacy.p.audit_child(phase,(begin,end))','runtime.verify_runtime_distributions(old_plan[\'runtime\'])'):
            self.assertIn(call,source)
        for forbidden in ('os.kill(','.kill(','.terminate(','flock(',"systemctl",'torch_neuronx.trace','nki.jit('):
            self.assertNotIn(forbidden,source)
        self.assertEqual(source.count('subprocess.Popen('),1)

    def test_08_mock_whole_fixed_schedule_finishes_without_scientific_or_process_execution(self):
        with tempfile.TemporaryDirectory(prefix='lfm-pure-check-') as folder:
            root=Path(folder);host,ref,calls=mock_coordinator(root)
            self.assertEqual(len(calls),874)
            result=host.read(root/'diagnostic-result.json')
            science.require_full_success(result)
            self.assertEqual([row['path'] for row in result['stages']],[str(root/(stage+'-complete.json')) for stage in science.STAGES])
            self.assertFalse(result['full_model_qualified'])
            # Mock scheduler receipts cannot pass the independent outer audit:
            # real batch reports, ACKs and parent identity were never generated.
            with self.assertRaises(FileNotFoundError): science.audit_full_success(host,result)

    def test_09_failed_old_or_new_pass_prevents_controls_and_candidate(self):
        for fail_stage in ('original_fixture','new_fixture'):
            with tempfile.TemporaryDirectory(prefix='lfm-pure-check-') as folder:
                root=Path(folder)
                host,ref,calls=mock_coordinator(root,lambda stage,phase,begin,end:stage==fail_stage and begin==0)
                self.assertEqual(calls[-1][0],fail_stage)
                self.assertFalse(any(row[0] in ('controls','candidate') for row in calls))
                result=host.read(root/'diagnostic-result.json')
                self.assertFalse(result['cpu_full_gate_passed'])
                self.assertTrue((root/'worker-failure.json').is_file())
                phase=host.read(root/fail_stage/'regression_cpu-outcomes.json')
                self.assertFalse(phase['complete'])
                self.assertEqual(phase['outcomes'][-1]['returncode'],1)

    def test_10_failed_controls_prevent_candidate_and_failed_candidate_stops_first(self):
        for fail_stage in ('controls','candidate'):
            with tempfile.TemporaryDirectory(prefix='lfm-pure-check-') as folder:
                root=Path(folder)
                host,ref,calls=mock_coordinator(root,lambda stage,phase,begin,end:stage==fail_stage and begin==0)
                self.assertEqual(calls[-1][0],fail_stage)
                if fail_stage=='controls': self.assertFalse(any(row[0]=='candidate' for row in calls))
                else: self.assertEqual(sum(row[0]=='candidate' for row in calls),1)
                self.assertFalse(host.read(root/'diagnostic-result.json')['passed'])

    def test_11_real_byte_comparison_and_input_mutation_gate_with_small_pure_arrays(self):
        case=dict(id=0,seed=1,kind='pure_mock',dtype='bfloat16',capacity=4,width=128,slots=[0,0])
        operands=leaf_checks.fake_operands()
        fixture,expected=science.fixtures.fixture_record(case,'regression_cpu',operands)
        row=science.scientific_record(case,fixture,operands,expected,expected.copy())
        self.assertTrue(row['passed'])
        altered=expected.copy();altered.raw[-1]^=1
        row=science.scientific_record(case,fixture,operands,expected,altered)
        self.assertFalse(row['passed']);self.assertFalse(row['bitwise_equal'])
        self.assertEqual(len(row['first16_mismatches']),1)
        operands[0].raw[0]^=1
        row=science.scientific_record(case,fixture,operands,expected,expected.copy())
        self.assertFalse(row['passed']);self.assertFalse(row['inputs_unchanged'])

    def test_12_original_control_receipt_gate_reused_and_not_counted_as_full_case(self):
        arms=leaf_checks.synthetic_control_receipts(self.values['extended_cpu'][0])
        report=dict(complete=True,passed=True,admission={'mock':True},stage='controls',phase='extended_cpu',
                    begin=0,end=1,full_model_qualified=False,new_speedup_claim=False,arms=arms,
                    controls_validation=science.fixtures.validate_small_controls(arms),
                    warning_log=known_warnings.KnownWarningLog('batch', lambda *args, **kwargs:None).snapshot())
        science.validate_report(report,'controls','extended_cpu',0,1,self.values['extended_cpu'][:1],{'mock':True})
        report['arms'][0]['inputs_unchanged']=True
        with self.assertRaises(RuntimeError): science.validate_report(report,'controls','extended_cpu',0,1,self.values['extended_cpu'][:1],{'mock':True})

    def test_13_complete_claim_without_all_original_counts_or_with_promotion_rejected(self):
        with tempfile.TemporaryDirectory(prefix='lfm-pure-check-') as folder:
            root=Path(folder);host,ref,calls=mock_coordinator(root)
            original=host.read(root/'diagnostic-result.json')
            for mutate in (lambda result:result['stage_counts']['candidate'].update(cases=32),
                           lambda result:result.update(device_behavior_proven=True),
                           lambda result:result.update(original_simulator_failure_preserved=False),
                           lambda result:result.update(new_speedup_claim=True)):
                result=copy.deepcopy(original);mutate(result)
                with self.assertRaises(RuntimeError): science.require_full_success(result)

    def test_14_no_native_modules_loaded_by_pure_integration_suite(self):
        self.assertFalse(any(name.split('.')[0] in {'numpy','ml_dtypes','torch','nki','neuronxcc'} for name in sys.modules))

    def test_15_candidate_gate_requires_each_original_byte_and_extraction_result(self):
        cases=self.values['regression_cpu'][:16];checks=[];scientific=[];events=[]
        probe,module=leaf_checks.mock_shim()
        for case in cases:
            operands=leaf_checks.fake_operands(case['dtype'],case['capacity'],case['width'],case['slots'])
            fixture,expected=science.fixtures.fixture_record(case,'regression_cpu',operands)
            probe.begin_case('regression_cpu',case['id'],operands)
            actual=module.simulate_kernel(lambda *xs:leaf_checks.Tensor(expected),operands,{})
            events.append(probe.end_case());checks.append(fixture)
            scientific.append(science.scientific_record(case,fixture,operands,expected,actual))
        probe.restore()
        report=dict(complete=True,passed=True,admission={'mock':True},stage='candidate',phase='regression_cpu',
                    begin=0,end=16,full_model_qualified=False,new_speedup_claim=False,checks=checks,
                    original_fixture_match=True,scientific=scientific,extractions=events,probe=probe.report(),
                    warning_log=known_warnings.KnownWarningLog('batch', lambda *args, **kwargs:None).snapshot())
        science.validate_report(report,'candidate','regression_cpu',0,16,cases,{'mock':True})
        mutations=(lambda value:value['checks'][0].pop('expected'),
                   lambda value:value['checks'][0]['inputs'][0].update(sha256='not-a-full-hash'),
                   lambda value:value['scientific'][0].update(bitwise_equal=False),
                   lambda value:value['scientific'][0].update(inputs_unchanged=False),
                   lambda value:value['extractions'][0].update(substitutions=3),
                   lambda value:value['probe'].update(restored=False),
                   lambda value:value.update(original_fixture_match=False))
        for mutate in mutations:
            changed=copy.deepcopy(report);mutate(changed)
            with self.assertRaises(RuntimeError): science.validate_report(changed,'candidate','regression_cpu',0,16,cases,{'mock':True})
        self.assertLess(len(runner.encode(report)),runner.FILE_CAP)

    def test_16_nonzero_child_exit_preserved_and_existing_stop_called(self):
        with tempfile.TemporaryDirectory(prefix='lfm-pure-check-') as folder:
            root=Path(folder);host=Host(root)
            target=root/'original_fixture'/'regression_cpu';target.mkdir(parents=True)
            child=SimpleNamespace(pid=123,returncode=1,wait=lambda **kw:1)
            owner=dict(pid=123,start_ticks='pure-mock')
            ref={'path':str(root/'admission.json'),'sha256':'0'*64}
            with patch.object(science,'legacy_child_binding',lambda *xs:nullcontext()),\
                 patch.object(science.subprocess,'Popen',return_value=child),\
                 patch.object(science.legacy,'child_identity',return_value=owner),\
                 patch.object(science.legacy,'stop_child') as stop:
                outcome,error=science.run_one_batch(host,{},ref,host.admission,{},root,
                    'original_fixture','regression_cpu',0,16,lambda:True,1800)
            self.assertIsInstance(error,RuntimeError)
            self.assertEqual(outcome['returncode'],1)
            self.assertFalse(outcome['passed'])
            stop.assert_called_once_with(child,owner)

    def test_17_missing_child_identity_cannot_be_called_drained_or_successful(self):
        with tempfile.TemporaryDirectory(prefix='lfm-pure-check-') as folder:
            root=Path(folder);host=Host(root)
            (root/'original_fixture'/'regression_cpu').mkdir(parents=True)
            child=SimpleNamespace(pid=123,returncode=None,wait=lambda **kw:0)
            ref={'path':str(root/'admission.json'),'sha256':'0'*64}
            with patch.object(science,'legacy_child_binding',lambda *xs:nullcontext()),\
                 patch.object(science.subprocess,'Popen',return_value=child),\
                 patch.object(science.legacy,'child_identity',side_effect=RuntimeError('pure identity failure')),\
                 patch.object(science.legacy,'stop_child') as stop:
                outcome,error=science.run_one_batch(host,{},ref,host.admission,{},root,
                    'original_fixture','regression_cpu',0,16,lambda:True,1800)
            self.assertFalse(outcome['passed']);self.assertIsNone(outcome['returncode'])
            self.assertTrue(outcome['secondary_errors']);stop.assert_not_called()
            # The unchanged outer still owns cleanup/drain before descriptor release.

    def test_18_final_pure_test_boundary_still_no_native_modules(self):
        self.assertFalse(any(name.split('.')[0] in {'numpy','ml_dtypes','torch','nki','neuronxcc'} for name in sys.modules))

    def test_19_atomic_temporary_unlink_does_not_abort_valid_completed_report(self):
        class Entry:
            def __init__(self,name,states): self.name=name;self.states=list(states)
            def lstat(self):
                value=self.states.pop(0) if len(self.states)>1 else self.states[0]
                if isinstance(value,Exception): raise value
                return value
            def with_name(self,name): return Entry(name,[FileNotFoundError(name)])
        def info(links,inode=7,mode=stat.S_IFREG):
            return SimpleNamespace(st_mode=mode,st_size=100,st_blocks=8,st_nlink=links,st_dev=2,st_ino=inode)
        final=Entry('a.json',[info(2),info(1)])
        temporary=Entry('.a.json.tmp',[FileNotFoundError()])
        owned=SimpleNamespace(rglob=lambda pattern:[final,temporary])
        nvme=SimpleNamespace(is_mount=lambda:True,stat=lambda:SimpleNamespace(st_dev=2))
        source=SimpleNamespace(stat=lambda:SimpleNamespace(st_dev=1))
        with patch.object(runner,'Path',lambda value:owned if value=='pure-owned' else nvme),\
             patch.object(runner,'ROOT',source),\
             patch.object(runner.os,'statvfs',return_value=SimpleNamespace(f_bavail=64*2**30,f_frsize=1)):
            self.assertEqual(runner.usage('pure-owned'),4096)
            for entry in (Entry('a.json',[FileNotFoundError()]),Entry('a.json',[info(2),info(1,8)]),
                          Entry('.a.json.tmp',[info(1,mode=stat.S_IFLNK)])):
                owned.rglob=lambda pattern:[entry]
                with self.assertRaises((RuntimeError,FileNotFoundError)): runner.usage('pure-owned')

    def test_20_only_warning_receipts_and_bootstrap_differ_from_frozen_v2(self):
        frozen = {
            'run_lfm3_nki05_full_cpu_serial_v2.py':'01a39c067f41b2a32e70f23e78019c9524892267fcd6f7a7d9c992279551d0aa',
            'lfm3_nki05_full_cpu_leaf_v2.py':'cee465642e4cc7101b2e216c5cd71f727e5820c3485977c1dbed897e3fe82b16',
        }
        for name,sha in frozen.items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),sha)
        old=ast.parse((ROOT/'run_lfm3_nki05_full_cpu_serial_v2.py').read_text())
        new=ast.parse(Path(runner.__file__).read_text())
        new.body=[node for node in new.body
                  if not (isinstance(node,ast.FunctionDef) and node.name=='warning_receipt')
                  and not (isinstance(node,ast.Assign) and ast.unparse(node.targets[0])=='WARNING_LOG')
                  and not (isinstance(node,ast.Import) and len(node.names)==1 and node.names[0].name=='lfm3_full_cpu_known_warnings_v1')]
        for node in new.body:
            if isinstance(node,ast.Import):
                for name in node.names:
                    if name.name=='lfm3_nki05_full_cpu_leaf_v3': name.name='lfm3_nki05_full_cpu_leaf_v2'
        self.assertEqual(ast.dump(old,include_attributes=False),ast.dump(new,include_attributes=False))
        old=ast.parse((ROOT/'lfm3_nki05_full_cpu_leaf_v2.py').read_text())
        new=ast.parse(Path(science.__file__).read_text())
        class RemoveWarningReceipts(ast.NodeTransformer):
            def visit_Import(self,node):
                return None if node.names[0].name=='lfm3_full_cpu_known_warnings_v1' else node
            def visit_Expr(self,node):
                if isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='known_warnings.validate_receipt':
                    return None
                return self.generic_visit(node)
            def visit_Assign(self,node):
                if ast.unparse(node.targets[0])=="result['warning_log']": return None
                return self.generic_visit(node)
            def visit_Call(self,node):
                node.keywords=[k for k in node.keywords if k.arg!='warning_log']
                return self.generic_visit(node)
        new=RemoveWarningReceipts().visit(new)
        self.assertEqual(ast.dump(old,include_attributes=False),ast.dump(new,include_attributes=False))

    def test_21_prefix_is_transmitted_before_each_interpreter_start(self):
        with tempfile.TemporaryDirectory(prefix='lfm-pure-v2-') as folder:
            root=Path(folder);prefix=root/'tmp'/'empty-pycache'
            ref={'path':str(root/'admission.json'),'sha256':'0'*64}
            spec={'output_root':str(root),'unit':'source-only-never-started'}
            command=runner.service_command(spec,ref,'0'*32)
            boundary=command.index('/usr/bin/env')
            self.assertEqual(command[boundary+1],'-i')
            self.assertIn('PYTHONPYCACHEPREFIX='+str(prefix),command[boundary+2:command.index(runner.PYTHON)])
            self.assertEqual(command[command.index(runner.PYTHON)+1],'-B')
            for stage in science.STAGES:
                env=science.child_environment(runner,root,'0'*32,stage)
                self.assertEqual(env['PYTHONPYCACHEPREFIX'],str(prefix))
                self.assertEqual(env['PYTHONDONTWRITEBYTECODE'],'1')
                cmd=science.command(runner,ref,stage,'extended_cpu',0,16)
                self.assertEqual(cmd[0],science.OLD_PYTHON if stage=='original_fixture' else science.NEW_PYTHON)
                self.assertEqual(cmd[1],'-B')
            self.assertFalse(prefix.exists());self.assertEqual(list(root.iterdir()),[])

    def test_22_original_isolation_predicate_rejects_partial_and_changed_configuration(self):
        with tempfile.TemporaryDirectory(prefix='lfm-pure-v2-') as folder:
            root=Path(folder);env=runner.environment(root,'0'*32);prefix=env['PYTHONPYCACHEPREFIX']
            original=runner.bytecode.isolated
            def call(values,configured,write=True,flag=1):
                state=SimpleNamespace(pycache_prefix=configured,dont_write_bytecode=write,
                                      flags=SimpleNamespace(dont_write_bytecode=flag))
                with patch.object(runner.bytecode,'os',SimpleNamespace(environ=values)),\
                     patch.object(runner.bytecode,'sys',state):
                    runner.bytecode_guard(root)
            with patch.object(runner.bytecode,'isolated',wraps=original) as predicate:
                call(env,prefix);predicate.assert_called_once_with(Path(prefix))
            missing={k:v for k,v in env.items() if k!='PYTHONPYCACHEPREFIX'}
            for values,configured,write,flag in ((missing,None,True,1),(env,None,True,1),
                    (missing,prefix,True,1),(env,prefix+'-other',True,1),(env,prefix,False,1),
                    (env,prefix,True,0),(dict(env,PYTHONDONTWRITEBYTECODE='0'),prefix,True,1)):
                with self.subTest(configured=configured,write=write,flag=flag),self.assertRaises(AssertionError):
                    call(values,configured,write,flag)
            self.assertFalse(Path(prefix).exists());self.assertEqual(list(root.iterdir()),[])

    def test_23_existing_file_directory_or_dangling_link_prevents_command_construction(self):
        with tempfile.TemporaryDirectory(prefix='lfm-pure-v2-') as folder:
            root=Path(folder);(root/'tmp').mkdir();prefix=root/'tmp'/'empty-pycache'
            spec={'output_root':str(root),'unit':'source-only-never-started'}
            ref={'path':str(root/'admission.json'),'sha256':'0'*64}
            for kind in ('file','directory','dangling_link'):
                if kind=='file': prefix.write_text('synthetic cache blocker')
                elif kind=='directory': prefix.mkdir()
                else: prefix.symlink_to(root/'absent-target')
                with self.subTest(kind=kind),self.assertRaises(RuntimeError):
                    runner.service_command(spec,ref,'0'*32)
                if kind=='directory': prefix.rmdir()
                else: prefix.unlink()
            with self.assertRaises(RuntimeError): runner.environment(Path('relative-root'),'0'*32)

    def test_24_guard_failure_prevents_RECORD_verification_and_batch_native_import(self):
        with tempfile.TemporaryDirectory(prefix='lfm-pure-v2-') as folder:
            root=Path(folder);host=Host(root);ref={'path':str(root/'admission.json'),'sha256':'0'*64}
            runtime=SimpleNamespace(WORKER_PYCACHE=Path('/unused'),verify_runtime_distributions=lambda plan:[])
            with patch.object(host,'bytecode_guard',side_effect=RuntimeError('pure isolation rejection')) as guard,\
                 patch.dict(sys.modules,{'lfm3_movement_device_protocol_v3':runtime}),\
                 patch.object(runtime,'verify_runtime_distributions') as verifier,\
                 patch.object(science,'run_one_batch') as batch:
                with self.assertRaisesRegex(RuntimeError,'pure isolation rejection'):
                    science.execute_full_cpu(host,ref,{'pure_mock_owner':True},root,lambda:True)
                guard.assert_called_once_with(root);verifier.assert_not_called();batch.assert_not_called()
            result=host.read(root/'diagnostic-result.json')
            self.assertFalse(result['complete']);self.assertFalse(result['cpu_full_gate_passed'])
            for stage in science.STAGES:
                with patch.object(host,'bytecode_guard',side_effect=RuntimeError('pure isolation rejection')) as guard:
                    with self.assertRaisesRegex(RuntimeError,'pure isolation rejection'):
                        science.execute_batch_science(host,root,stage,'extended_cpu',0,1,ref,lambda:True,{})
                    guard.assert_called_once_with(root)
            self.assertNotIn('numpy',sys.modules)

    def test_25_existing_prefix_stops_batch_before_Popen(self):
        with tempfile.TemporaryDirectory(prefix='lfm-pure-v2-') as folder:
            root=Path(folder);host=Host(root);(root/'tmp'/'empty-pycache').mkdir(parents=True)
            ref={'path':str(root/'admission.json'),'sha256':'0'*64}
            with patch.object(science,'legacy_child_binding',lambda *args:nullcontext()),\
                 patch.object(science.subprocess,'Popen') as launch,\
                 patch.object(science.legacy,'stop_child') as stop:
                outcome,error=science.run_one_batch(host,{},ref,host.admission,{},root,
                    'original_fixture','regression_cpu',0,16,lambda:True,1800)
                launch.assert_not_called();stop.assert_not_called()
            self.assertIsInstance(error,RuntimeError);self.assertFalse(outcome['passed'])

    def test_26_original_WORKER_PYCACHE_matches_verified_startup_prefix(self):
        with tempfile.TemporaryDirectory(prefix='lfm-pure-v2-') as folder:
            root=Path(folder);host=Host(root);host.bytecode_guard=runner.bytecode_guard
            env=runner.environment(root,'0'*32);prefix=env['PYTHONPYCACHEPREFIX'];observed=[]
            state=SimpleNamespace(pycache_prefix=prefix,dont_write_bytecode=True,flags=SimpleNamespace(dont_write_bytecode=1))
            runtime=SimpleNamespace(WORKER_PYCACHE=Path('/unused'))
            def verify(plan):
                observed.append(str(runtime.WORKER_PYCACHE))
                self.assertEqual(str(runtime.WORKER_PYCACHE),prefix)
                self.assertEqual(state.pycache_prefix,env['PYTHONPYCACHEPREFIX'])
                raise RuntimeError('pure stop before any RECORD IO')
            runtime.verify_runtime_distributions=verify
            with patch.dict(sys.modules,{'lfm3_movement_device_protocol_v3':runtime}),\
                 patch.object(runner.bytecode,'os',SimpleNamespace(environ=env)),\
                 patch.object(runner.bytecode,'sys',state):
                with self.assertRaisesRegex(RuntimeError,'pure stop before any RECORD IO'):
                    science.execute_full_cpu(host,{'path':str(root/'admission.json'),'sha256':'0'*64},
                                             {'pure_mock_owner':True},root,lambda:True)
            self.assertEqual(observed,[prefix]);self.assertEqual(runtime.WORKER_PYCACHE,Path('/unused'))
            self.assertFalse(Path(prefix).exists())


if __name__=='__main__': unittest.main(verbosity=2)
