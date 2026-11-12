"""Standard-library/mock only; all native imports and actual runtime operations blocked."""
import ast
import copy
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import resource
import stat
import sys
import tempfile
import types
import unittest
from unittest import mock

BLOCKED=('torch','torch_neuronx','torch_xla','numpy','ml_dtypes','nki','neuronxcc','ctypes','vllm','vllm_neuron')
class BlockNative(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in BLOCKED:raise AssertionError('Native import prohibited: '+fullname)
sys.meta_path.insert(0,BlockNative())
import run_lfm3_nki05_device_serial_v1 as runner
import lfm3_nki05_device_pipeline_v1 as pipeline
import lfm3_nki05_device_protocol_v1 as protocol
import lfm3_nki05_device_runtime_v1 as runtime
import lfm3_nki05_full_cpu_metadata_v1 as cpu
import lfm3_nki05_device_budget_v1 as budget
import lfm3_nki05_device_sdk_v1 as sdk

ROOT=Path(__file__).resolve().parent
FILES=('run_lfm3_nki05_device_serial_v1.py','lfm3_nki05_device_pipeline_v1.py','lfm3_nki05_device_protocol_v1.py',
       'lfm3_nki05_device_runtime_v1.py','lfm3_nki05_full_cpu_metadata_v1.py','lfm3_nki05_device_budget_v1.py','lfm3_nki05_device_sdk_v1.py')

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def parse(file):return ast.parse((ROOT/file).read_text())
def function(file,name):return next(x for x in parse(file).body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name==name)
def dump(node):return ast.dump(node,include_attributes=False)
def warning(mode):return dict(schema='lfm3_full_cpu_known_warning_receipt_v1',mode=mode,scope='calls_delivered_to_showwarning',known=[dict(x,observed=0,forwarded=0,deduplicated=0) for x in cpu.known_warnings.KNOWN],other_forwarded=0)

class CpuWorld:
    """Lazy synthetic full 874-child future evidence; never current real r2 files."""
    def __init__(self,fault=None):
        self.fault=fault;self.values=cpu.matrices();self.plans=cpu.schedule();self.cache={};self.reads=[]
        self.outer=dict(pid=4100,start_ticks='500',ppid=4000,argv=['synthetic-outer'],cgroup='0::/synthetic-outer')
        self.parent=dict(pid=4200,start_ticks='600',ppid=1583,argv=['synthetic-worker'],cgroup='0::/synthetic-service',unit='trn1-lfm3-nki05-full-cpu-r2-20261112',invocation_id='a'*32)
        self.tasks=[(p['stage'],p['phase'],a,b) for p in self.plans for a,b in p['bounds']]
        self.index={task:i for i,task in enumerate(self.tasks)}
        self.external=ROOT/'research/synthetic-lfm-full-cpu-root-exit.json'
        self.binding=dict(spec=cpu.CPU_SPEC,admission=cpu.CPU_ADMISSION,
            **{key:self.ref(cpu.CPU_ROOT/name) for key,name in dict(result='diagnostic-result.json',worker_result='worker-result.json',service_exit='service-exit.json',outer_result='outer-result.json',cleanup='cleanup-drained.json').items()},actual_exit=self.ref(self.external))
        self.host=types.SimpleNamespace(p=types.SimpleNamespace(read=self.read),reference=self.ref)
    def ref(self,path):
        path=Path(path)
        if str(path)==cpu.CPU_SPEC['path']:return dict(cpu.CPU_SPEC)
        if str(path)==cpu.CPU_ADMISSION['path']:return dict(cpu.CPU_ADMISSION)
        return dict(path=str(path),sha256=digest(self.value(path)))
    def read(self,path,wanted=None):
        self.reads.append(str(path))
        if wanted is not None:assert wanted==self.ref(path)['sha256']
        return copy.deepcopy(self.value(Path(path)))
    def fixture(self,phase,case):
        shapes=([case['capacity'],case['width']],[len(case['slots']),case['width']],[len(case['slots'])],[case['capacity'],case['width']])
        descriptors=[]
        for i,(dtype,shape) in enumerate(zip((case['dtype'],case['dtype'],'int32',case['dtype']),shapes)):
            size=2 if dtype=='bfloat16' else 4
            for n in shape:size*=n
            sha=cpu.OLD_FIRST_HASHES[i] if phase=='extended_cpu' and case['id']==0 else digest([phase,case['id'],i])
            descriptors.append(dict(dtype=dtype,shape=shape,bytes=size,sha256=sha))
        return dict(phase=phase,id=case['id'],case=copy.deepcopy(case),inputs=descriptors[:3],expected=descriptors[3])
    def controls(self):
        fixture=self.fixture('extended_cpu',self.values['extended_cpu'][0]);descriptors=fixture['inputs']+[fixture['expected']];arms=[]
        for index,name in enumerate(('original','candidate','restored_original')):
            mode='precise_bf16_writeback' if index==1 else 'original';events=[]
            for position,(operand,descriptor) in enumerate(zip(('cache','updates','slots','output'),descriptors)):
                changed=index==1 and position<2
                events.append(dict(operand=operand,tensor_dtype=descriptor['dtype'],extracted_dtype='float32' if index!=1 and position<2 else descriptor['dtype'],shape=descriptor['shape'],bytes=descriptor['bytes']*(2 if index!=1 and position<2 else 1),requested_precise=True if position==3 else 'omitted',effective_precise=changed or position==3,substitution=changed))
            probe=dict(mode=mode,installed=True,restored=True,failed=False,completed_cases=1,extraction_calls=4,substitutions=2 if index==1 else 0,installed_sdk_source_modified=False)
            extraction=dict(phase='extended_cpu',id=0,mode=mode,extraction_calls=4,substitutions=2 if index==1 else 0,events=events)
            arms.append(dict(name=name,fixture=fixture,shape_equal=True,dtype_equal=True,bitwise_equal=True,actual_sha256=cpu.OLD_FIRST_HASHES[3],inputs_unchanged=index==1,input_after_sha256=list(cpu.OLD_FIRST_HASHES[:3]) if index==1 else ['f'*64,'e'*64,cpu.OLD_FIRST_HASHES[2]],probe=probe,extraction=extraction))
        return arms
    def batch(self,stage,phase,begin,end):
        index=self.index[(stage,phase,begin,end)]
        value=dict(complete=True,passed=True,admission=cpu.CPU_ADMISSION,stage=stage,phase=phase,begin=begin,end=end,pid=10000+index,start_ticks=str(20000+index),full_model_qualified=False,new_speedup_claim=False,warning_log=warning('batch'))
        if stage=='controls':
            value['arms']=self.controls();value['controls_validation']=cpu.validate_small_controls(value['arms']);return value
        value['checks']=[self.fixture(phase,case) for case in self.values[phase][begin:end]]
        if stage in ('new_fixture','candidate'):value['original_fixture_match']=True
        if stage=='candidate':
            value['scientific']=[];value['extractions']=[]
            for fixture in value['checks']:
                value['scientific'].append(dict(case=fixture['case'],fixture=fixture,passed=True,dtype_equal=True,shape_equal=True,bitwise_equal=True,inputs_unchanged=True,actual_sha256=fixture['expected']['sha256'],expected_sha256=fixture['expected']['sha256'],input_sha256=[x['sha256'] for x in fixture['inputs']],input_after_sha256=[x['sha256'] for x in fixture['inputs']],first16_mismatches=[]))
                value['extractions'].append(dict(phase=phase,id=fixture['id'],mode='precise_bf16_writeback',extraction_calls=4,substitutions=2 if fixture['case']['dtype']=='bfloat16' else 0))
            value['probe']=dict(installed=True,restored=True,failed=False,completed_cases=end-begin,extraction_calls=4*(end-begin),substitutions=sum(x['substitutions'] for x in value['extractions']))
        if self.fault=='fixture_drift' and stage=='new_fixture' and phase=='regression_cpu' and begin==0:value['checks'][0]['expected']['sha256']='f'*64
        if self.fault=='candidate_byte_failure' and stage=='candidate' and phase=='extended_cpu' and begin==0:value['scientific'][0]['bitwise_equal']=False
        return value
    def value(self,path):
        name=str(path)
        if name in self.cache:return self.cache[name]
        if name==cpu.CPU_SPEC['path']:value=dict(output_root=str(cpu.CPU_ROOT),runner=cpu.CPU_SELF,unit=self.parent['unit'],source_sha256={})
        elif name==cpu.CPU_ADMISSION['path']:value=dict(spec=cpu.CPU_SPEC,outer_owner=self.outer)
        elif path==self.external:value=dict(reviewer='/root',actual_tool_exit_observed=True,spec=cpu.CPU_SPEC,actual_tool=dict(kind='tool_session',session_id=cpu.CPU_SESSION,terminal_chunk='synthetic-exit',exit_code=1 if self.fault=='outer_exit_failure' else 0),owners=[self.outer,self.parent])
        else:
            relative=path.relative_to(cpu.CPU_ROOT);parts=relative.parts
            if len(parts)==1:
                if parts[0]=='worker-owner.json':value=self.parent
                elif parts[0]=='diagnostic-result.json':
                    value=dict(schema='lfm3_nki05_full_cpu_result_v1',complete=True,passed=True,warning_log=warning('worker'),admission=cpu.CPU_ADMISSION,stage_counts={stage:dict(cases=1,batches=1) if stage=='controls' else dict(cases=3716,batches=291) for stage in cpu.STAGES},stages=[self.ref(cpu.CPU_ROOT/(stage+'-complete.json')) for stage in cpu.STAGES],cpu_full_gate_passed=True,simulator_writeback_workaround_required=True,original_simulator_failure_preserved=True,device_behavior_proven=False,alias_gate_passed=False,full_model_qualified=False,new_speedup_claim=False,automatic_successor_launch=False)
                    if self.fault=='partial':del value['stage_counts']['candidate']
                elif parts[0].endswith('-complete.json'):
                    stage=parts[0][:-14];plans=[p for p in self.plans if p['stage']==stage]
                    value=dict(complete=True,passed=True,stage=stage,admission=cpu.CPU_ADMISSION,cases=1 if stage=='controls' else 3716,batches=1 if stage=='controls' else 291,phases=[self.ref(cpu.CPU_ROOT/stage/(p['phase']+'-outcomes.json')) for p in plans])
                elif parts[0]=='worker-result.json':value=dict(complete=True,passed=True,owner=self.parent,admission=cpu.CPU_ADMISSION,diagnostic=self.ref(cpu.CPU_ROOT/'diagnostic-result.json'),self_exit_claimed=False,full_model_qualified=False,new_speedup_claim=False)
                elif parts[0]=='service-exit.json':value=dict(owner=self.parent,observed_by_outer=self.outer,successful=True,properties=dict(Result='success',ExecMainCode='1',ExecMainStatus='0'),drain=dict(physically_drained=True))
                elif parts[0]=='outer-result.json':value=dict(complete=True,passed=True,owner=self.outer,service_exit=self.ref(cpu.CPU_ROOT/'service-exit.json'),worker_result=self.ref(cpu.CPU_ROOT/'worker-result.json'),self_exit_claimed=False,full_model_qualified=False,new_speedup_claim=False)
                elif parts[0]=='cleanup-drained.json':value=dict(physically_drained=True,owner=self.parent,self_exit_claimed=False)
                else:raise AssertionError(name)
            elif len(parts)==2:
                stage=parts[0];phase=parts[1].removesuffix('-outcomes.json');plan=next(p for p in self.plans if p['stage']==stage and p['phase']==phase);outcomes=[]
                for begin,end in plan['bounds']:
                    i=self.index[(stage,phase,begin,end)]
                    outcomes.append(dict(stage=stage,phase=phase,begin=begin,end=end,pid=10000+i,start_ticks=str(20000+i),passed=True,returncode=0,error=None,secondary_errors=[],report=self.ref(cpu.result_path(cpu.CPU_ROOT,stage,phase,begin,end))))
                if self.fault=='missing_exit' and stage=='candidate' and phase=='extended_cpu':outcomes.pop()
                value=dict(complete=True,passed=True,stage=stage,phase=phase,admission=cpu.CPU_ADMISSION,outcomes=outcomes)
            else:
                stage,phase,filename=parts;stem=filename.removesuffix('-launch-owner.json').removesuffix('.json');begin,end=map(int,stem.split('-'));i=self.index[(stage,phase,begin,end)]
                if filename.endswith('-launch-owner.json'):
                    value=dict(pid=10000+i,start_ticks=str(20000+i),parent=dict(self.parent,cgroup=self.parent['cgroup'].removeprefix('0::')),cgroup=self.parent['cgroup'].removeprefix('0::'),argv=cpu.command(cpu.CPU_ADMISSION,stage,phase,begin,end))
                    if self.fault=='ack_drift' and i==873:value['start_ticks']='999'
                else:value=self.batch(stage,phase,begin,end)
        self.cache[name]=value;return value


def ref(name):return dict(path=str(ROOT/'research'/name),sha256='a'*64)
def spec():
    binding=dict(spec=cpu.CPU_SPEC,admission=cpu.CPU_ADMISSION,**{key:dict(path=str(cpu.CPU_ROOT/name),sha256='b'*64) for key,name in dict(result='diagnostic-result.json',worker_result='worker-result.json',service_exit='service-exit.json',outer_result='outer-result.json',cleanup='cleanup-drained.json').items()},actual_exit=ref('future-r2-exit.json'))
    return dict(schema='lfm3_nki05_device_spec_v1',root_finalized=True,runner=runner.SELF,diagnostic=pipeline.__file__,source_sha256={'/synthetic.py':'a'*64},serial_catalog=ref('future-catalog.json'),terminal_handoff=ref('future-handoff.json'),output_root='/mnt/nvme/lfm3-nki05-device-synthetic',unit='trn1-lfm3-nki05-device-synthetic',resources=budget.PROPS,lock_identity=runner.LOCK,canonical_sha256={str(ROOT/'continuation-state.json'):'a'*64},root_floor_bytes=budget.ROOT_FLOOR,output_cap_bytes=budget.OUTPUT_BYTES,nvme_floor_bytes=budget.NVME_FLOOR,max_files=budget.MAX_FILES,device_protocol=protocol.manifest(),full_cpu_prerequisite=binding,sdk_manifest=ref('future-sdk.json'),device_budget=budget.manifest(),model_qualification_claim=False,automatic_successor_launch=False)

class CpuGateTests(unittest.TestCase):
    def test_complete_full_synthetic_874_exit_audit(self):
        world=CpuWorld();result=cpu.audit(world.host,world.binding)
        self.assertEqual(len(result['children']),874);self.assertTrue(result['actual_outer_exit_verified'])
        self.assertEqual(result['actual_tool']['exit_code'],0)
    def test_partial_original_success_cannot_admit(self):
        world=CpuWorld('partial')
        with self.assertRaises(RuntimeError):cpu.audit(world.host,world.binding)
    def test_missing_actual_child_exit_cannot_admit(self):
        world=CpuWorld('missing_exit')
        with self.assertRaises(RuntimeError):cpu.audit(world.host,world.binding)
    def test_native_fixture_drift_cannot_admit(self):
        world=CpuWorld('fixture_drift')
        with self.assertRaises(RuntimeError):cpu.audit(world.host,world.binding)
    def test_candidate_byte_failure_cannot_admit(self):
        world=CpuWorld('candidate_byte_failure')
        with self.assertRaises(RuntimeError):cpu.audit(world.host,world.binding)
    def test_ack_pid_start_drift_cannot_admit(self):
        world=CpuWorld('ack_drift')
        with self.assertRaises(RuntimeError):cpu.audit(world.host,world.binding)
    def test_outer_tool_exit_failure_cannot_admit(self):
        world=CpuWorld('outer_exit_failure')
        with self.assertRaises(RuntimeError):cpu.audit(world.host,world.binding)
    def test_fixed_actual_r2_bindings(self):
        value=spec()['full_cpu_prerequisite'];value['spec']=ref('other-cpu.json')
        with self.assertRaises(RuntimeError):cpu.validate_binding(value)
    def test_original_matrices_and_bounds(self):
        values=cpu.matrices();self.assertEqual([len(values[p]) for p in cpu.PHASES],[1648,2068])
        self.assertEqual(sum(len(p['bounds']) for p in cpu.schedule()),874)

class SourceTests(unittest.TestCase):
    def test_import_is_inert_and_native_blocked(self):
        self.assertIsNone(runner.history);self.assertFalse(set(BLOCKED)&{name.split('.')[0] for name in sys.modules})
        self.assertNotIn('lfm3_nki05_compact_writeback_v1',sys.modules)
    def test_retained_lock_descriptor_and_complete_service_drain_exact(self):
        for name in ('descriptor_proof','acquire','drain_and_release'):
            self.assertEqual(dump(function('run_lfm3_nki05_full_cpu_serial_v3.py',name)),dump(function('run_lfm3_nki05_device_serial_v1.py',name)),name)
        old=function('run_lfm3_nki05_full_cpu_serial_v3.py','OwnedService');new=function('run_lfm3_nki05_device_serial_v1.py','OwnedService')
        for name in ('launch','sample','cleanup','drained'):
            self.assertEqual(dump(next(x for x in old.body if isinstance(x,ast.FunctionDef) and x.name==name)),dump(next(x for x in new.body if isinstance(x,ast.FunctionDef) and x.name==name)),name)
    def test_original_cpu_validator_functions_exact(self):
        for name in ('stage_complete','require_full_success'):
            self.assertEqual(dump(function('lfm3_nki05_full_cpu_leaf_v3.py',name)),dump(function('lfm3_nki05_full_cpu_metadata_v1.py',name)))
        for name in ('original_batches','validate_matrices','validate_small_controls'):
            self.assertEqual(dump(function('lfm3_typed_fixture_ledger_leaf_v1.py',name)),dump(function('lfm3_nki05_full_cpu_metadata_v1.py',name)))
    def test_original_scientific_report_validator_exact_except_local_metadata_helper(self):
        old=function('lfm3_nki05_full_cpu_leaf_v3.py','validate_report')
        class Helper(ast.NodeTransformer):
            def visit_Attribute(self,node):
                if isinstance(node.value,ast.Name) and node.value.id=='fixtures' and node.attr=='validate_small_controls':return ast.Name(id='validate_small_controls',ctx=ast.Load())
                return self.generic_visit(node)
        self.assertEqual(dump(Helper().visit(old)),dump(function('lfm3_nki05_full_cpu_metadata_v1.py','validate_report')))
    def test_no_new_runtime_ast_extraction(self):
        for file in FILES:
            node=parse(file)
            self.assertFalse(any(isinstance(n,ast.Import) and any(x.name=='ast' for x in n.names) for n in ast.walk(node)),file)
            self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('exec','eval','compile') for n in ast.walk(node)),file)
    def test_complete_task_matrix_first_raw_graph_and_retained4096(self):
        value=protocol.manifest();self.assertEqual(len(value['tasks']),91);self.assertEqual(value['graphs'],113);self.assertEqual(value['checks'],3491)
        self.assertEqual(value['tasks'][0],dict(phase='raw_device',index=-1));self.assertEqual(value['first_actual_graph'],dict(dtype='bfloat16',shape=[4,7168]))
        self.assertTrue(value['context4096_target_retained']);self.assertFalse(value['context4096_qualified'])
        self.assertEqual(value['original_model_gates']['comparisons']['measured_requests_per_variant'],704)
    def test_concrete_context_implements_every_abstract_method(self):self.assertFalse(runtime.Context.__abstractmethods__)
    def test_invoke_connects_each_actual_scientific_leaf(self):
        cases=[('raw_device',-1,'lfm3_nki05_raw_device_leaf_v1','run'),('regression_device',-1,'lfm3_nki05_regression_device_leaf_v1','run'),('regression_alias',-1,'lfm3_nki05_regression_alias_leaf_v1','run'),('extended_device',0,'lfm3_nki05_extended_device_leaf_v2','device'),('extended_alias',0,'lfm3_nki05_extended_device_leaf_v2','alias'),('fp32_alias',0,'lfm3_nki05_device_supplements_v2','fp32_alias'),('wrapper_controls',0,'lfm3_nki05_device_supplements_v2','wrapper_controls')]
        for phase,index,module,name in cases:
            with mock.patch(module+'.'+name) as call:
                pipeline.invoke('mock-context',dict(phase=phase,index=index));call.assert_called_once_with(*(['mock-context'] if index==-1 else ['mock-context',index]))

class AdmissionTests(unittest.TestCase):
    def test_exact_future_spec_shape(self):self.assertEqual(runner.validate_spec(spec()),spec())
    def test_partial_or_old_catalog_rejected(self):
        value=spec();value['serial_catalog']=pipeline.MINIMUM_CATALOG
        with self.assertRaises(RuntimeError):runner.validate_spec(value)
    def test_modified_budgets_or_protocol_rejected(self):
        for field in ('resources','device_protocol','device_budget'):
            value=copy.deepcopy(spec());value[field]={}
            with self.assertRaises(RuntimeError):runner.validate_spec(value)
    def test_successor_outer_checks_full_cpu_before_lock_or_history(self):
        with mock.patch.object(runner,'read_bundle',return_value=spec()),mock.patch.object(runner.science,'prerequisite',side_effect=RuntimeError('r2 still live')),mock.patch.object(runner,'load_history') as history,mock.patch.object(runner,'acquire') as acquire,mock.patch.object(runner.os,'sched_setaffinity'),mock.patch.object(runner.os,'setpriority'),mock.patch.object(runner.resource,'setrlimit'):
            with self.assertRaisesRegex(RuntimeError,'r2 still live'):runner.outer(ref('spec.json'),ref('review.json'))
            history.assert_not_called();acquire.assert_not_called()
    def test_clean_env_and_bounded_systemd_command(self):
        value=spec();admission=ref('admission.json');env=runner.environment(Path(value['output_root']),'test')
        self.assertEqual(env['NEURON_RT_VISIBLE_CORES'],'0');self.assertEqual(env['NEURON_LOGICAL_NC_CONFIG'],'1');self.assertEqual(env['NKI_SIMULATOR'],'0')
        self.assertNotIn('PYTHONPATH',env);self.assertNotIn('LD_PRELOAD',env)
        self.assertTrue(env['NEURON_COMPILE_CACHE_URL'].startswith(value['output_root']+'/cache/'))
        cmd=runner.service_command(value,admission,'test')
        for k,v in budget.PROPS.items():self.assertIn(k+'='+v,cmd)
        self.assertEqual(cmd[-len(runner.worker_argv(admission)):],runner.worker_argv(admission))
    def test_fixed_child_command_and_isolated_tmp(self):
        task=protocol.tasks()[0];argv=pipeline.child_argv(runner,ref('admission.json'),task)
        self.assertEqual(argv[-4:],['--phase','raw_device','--index','-1'])
        env=pipeline.child_environment(runner,Path('/mnt/nvme/lfm3-nki05-device-test'),'x',task)
        self.assertTrue(env['TMPDIR'].endswith('/tasks/raw_device-all-tmp'))
    def test_foreign_context_owner_rejected_before_any_io(self):
        with self.assertRaises(RuntimeError):runtime.Context(types.SimpleNamespace(__file__='/foreign.py'),None,None,None)
    def test_jit_rejected_before_sdk_authentication(self):
        value=runtime.Context.__new__(runtime.Context);value.sdk_verified=False
        with self.assertRaises(RuntimeError):value.writer('raw_device')
    def test_actual_sdk_versions_and_no_native_binary_override(self):
        rows=[]
        for name,version in sorted(sdk.REQUIRED.items()):
            directory=sdk.SITE/(name.replace('-','_')+'-'+version+'.dist-info')
            rows.append(dict(name=name,version=version,directory=str(directory),files={x:dict(path=str(directory/x),sha256='a'*64) for x in ('METADATA','RECORD')}))
        rows.sort(key=lambda x:x['directory']);value=dict(schema='lfm3_nki05_device_sdk_manifest_v1',prefix=str(sdk.PREFIX),site=str(sdk.SITE),distributions=rows,complete_prefix_census=True,source_overrides={},native_payload_verified=False)
        sdk.validate_manifest(value)
        value['source_overrides'][str(sdk.SITE/'native.so')]=dict(record_sha256='a'*64,actual_sha256='b'*64,review=ref('review.json'))
        with self.assertRaises(RuntimeError):sdk.validate_manifest(value)
    def test_new_sdk_peak_remains_unknown(self):
        value=budget.manifest();self.assertIsNone(value['measured_new_compiler_peak_bytes']);self.assertIsNone(value['measured_new_artifact_bytes'])
        self.assertEqual(value['properties']['MemorySwapMax'],'0');self.assertEqual(value['artifact_bytes'],256*2**20)
        self.assertEqual(value['phase_seconds'],14400);self.assertEqual(value['total_output_bytes'],24*2**30)

def science_value(task):
    """Synthetic JSON only; exercises completeness validators, never device truth."""
    sources={'/synthetic.py':'a'*64};phase=task['phase']
    def artifact(case):return dict(schema=protocol.ARTIFACT_SCHEMA,variant=protocol.VARIANT,software=protocol.DEVICE_SOFTWARE,source_sha256=sources,phase=phase,case=case,sha256='b'*64)
    def checks(slots,fields):return [dict(slots=row,**{key:True for key in fields}) for row in slots]
    value=dict(complete=True,passed=True)
    five=[[0,1,2,3],[3,1,0,2],[3,-1,-1,-1],[-1]*4,[2,2,0,3]]
    if phase=='raw_device':
        value.update(graphs=[],checks=[])
        for dtype in ('bfloat16','float32'):
            for width in (7168,65536):
                value['graphs'].append(dict(dtype=dtype,width=width,artifact_sha256='b'*64,artifact_identity=artifact(dict(dtype=dtype,shape=[4,width],rows=4,variant='candidate'))))
                value['checks'].extend(dict(row,dtype=dtype,width=width) for row in checks(protocol.MAPPINGS,('bitwise_equal','inputs_unchanged')))
    elif phase=='regression_device':
        specs=[(f'kv{n}_b4',[4,2,n,128],4,'torch.bfloat16') for n in (128,256,512,1024,2048)]
        specs += [('conv_b4',[4,1024,7],4,'torch.bfloat16'),('kv2048_b1',[4,2,2048,128],1,'torch.bfloat16'),('rnj_kv4096_b4',[4,4,4096,128],4,'torch.bfloat16'),('nemotron_ssm_b4',[4,64,80,128],4,'torch.float32')]
        value['graphs']=[dict(name=name,shape=shape,dtype=dtype,artifact_sha256='b'*64,artifact_identity=artifact(dict(name=name,shape=shape,rows=rows,dtype=dtype,variant='candidate')),checks=checks(five if rows==4 else [[0],[3],[-1]],('bitwise_equal','cache_input_unchanged','updates_input_unchanged'))) for name,shape,rows,dtype in specs]
    elif phase=='regression_alias':
        value['graphs']=[dict(length=n,kv_heads=heads,variant=label,passed=True,artifact_identity=artifact(dict(variant=label,length=n,kv_heads=heads,shape=[4,heads,n,128],rows=4,dtype='bfloat16')),checks=checks(five,('attention_finite','cache_k_bitwise','cache_v_bitwise','input_k_bitwise','input_v_bitwise'))) for n,heads in [(128,2),(256,2),(512,2),(1024,2),(2048,2),(4096,4)] for label in ('reference','candidate')]
    else:
        s=protocol.graph_spec(task);value['spec']=s
        if phase=='extended_device':
            value.update(artifact_identity=artifact(dict(s,variant='candidate')),artifact_sha256='b'*64,
                checks=[dict(row,actual_sha256='c'*64,expected_sha256='c'*64) for row in checks(protocol.mappings(s['capacity'],s['rows']),('bitwise_equal','cache_input_unchanged','updates_input_unchanged','slots_input_unchanged'))])
        elif phase=='extended_alias':
            b,n=s['capacity'],s['length'];conv=s['kind']=='conv';shapes=[[b,1024,7]] if conv else [[b,2,n,128]]*2
            value.update(artifact_identity=artifact(dict(s,dtype='bfloat16',rows=b,shapes=shapes)),artifact_sha256='b'*64,attention_rounding_is_diagnostic_only=True,initial_sha256=['c'*64]*len(shapes),inputs_sha256=[['c'*64]*(2 if conv else 5) for row in protocol.alias_mappings(b)],checks=[dict(row,step=i,output_sha256='c'*64,state_sha256=['c'*64]*len(shapes)) for i,row in enumerate(checks(protocol.alias_mappings(b),('output_finite','cache_outputs_bitwise','persistent_states_bitwise','bindings_unchanged','inputs_unchanged','output_bitwise')))])
        elif phase=='fp32_alias':
            b=s['capacity'];sequence=protocol.control_mappings(b,b)+[list(range(b)),[-1]*b,list(reversed(range(b)))]
            value.update(artifact_identity=artifact(s),checks=[dict(row,step=i) for i,row in enumerate(checks(sequence,('prior_state_bitwise','cache_output_bitwise','persistent_state_bitwise','bindings_unchanged','inputs_unchanged')))])
        else:
            value.update(artifact_identity=artifact(s),checks=checks(protocol.control_mappings(s['capacity'],s['rows']),('bitwise_equal','dtype_shape_preserved','inputs_unchanged')),trace_dispatch_counts=dict(original=1 if s['rows']==1 else 0,typed=0 if s['rows']==1 else 1))
    return value,sources


class ScienceCompletenessTests(unittest.TestCase):
    def test_every_full_91_task_validator_positive(self):
        for task in protocol.tasks():
            with self.subTest(task=task):
                value,sources=science_value(task);self.assertEqual(protocol.validate_task(task,value,sources),value)
    def test_drop_last_case_rejected_for_each_task(self):
        for task in protocol.tasks():
            with self.subTest(task=task):
                value,sources=science_value(task)
                (value['graphs'][-1]['checks'] if task['phase'] in ('regression_device','regression_alias') else value['checks']).pop()
                with self.assertRaises(RuntimeError):protocol.validate_task(task,value,sources)
    def test_false_final_strict_gate_rejected_in_each_phase(self):
        fields=dict(raw_device='bitwise_equal',regression_device='updates_input_unchanged',regression_alias='input_v_bitwise',extended_device='slots_input_unchanged',extended_alias='persistent_states_bitwise',fp32_alias='prior_state_bitwise',wrapper_controls='dtype_shape_preserved')
        for phase,field in fields.items():
            task=next(x for x in reversed(protocol.tasks()) if x['phase']==phase);value,sources=science_value(task)
            rows=value['graphs'][-1]['checks'] if phase in ('regression_device','regression_alias') else value['checks'];rows[-1][field]=False
            with self.subTest(phase=phase),self.assertRaises(RuntimeError):protocol.validate_task(task,value,sources)
    def test_huge_oob_mapping_order_and_production_dispatch_preserved(self):
        task=next(x for x in protocol.tasks() if x['phase']=='wrapper_controls');value,sources=science_value(task)
        value['trace_dispatch_counts']=dict(original=0,typed=1)
        with self.assertRaises(RuntimeError):protocol.validate_task(task,value,sources)
    def test_attention_rounding_is_diagnostic_but_conv_bytes_are_required(self):
        task=next(x for x in protocol.tasks() if x['phase']=='extended_alias' and protocol.graph_spec(x)['kind']!='conv');value,sources=science_value(task);value['checks'][0]['output_bitwise']=False
        protocol.validate_task(task,value,sources)
        task=next(x for x in protocol.tasks() if x['phase']=='extended_alias' and protocol.graph_spec(x)['kind']=='conv');value,sources=science_value(task);value['checks'][0]['output_bitwise']=False
        with self.assertRaises(RuntimeError):protocol.validate_task(task,value,sources)
    def test_full_hash_and_sequential_reuse_receipts_required(self):
        for phase in ('extended_device','extended_alias','fp32_alias'):
            task=next(x for x in protocol.tasks() if x['phase']==phase);value,sources=science_value(task)
            if phase=='extended_device':value['checks'][-1]['expected_sha256']='d'*64
            elif phase=='extended_alias':value['inputs_sha256'].pop()
            else:value['checks'][-1]['step']=0
            with self.subTest(phase=phase),self.assertRaises(RuntimeError):protocol.validate_task(task,value,sources)


class ServiceTransitionTests(unittest.TestCase):
    def run_ack(self,kind):
        value=spec();ref_value=ref('admission.json');a=dict(attempt_id='test');owned=runner.OwnedService(value,ref_value,a)
        command=runner.service_command(value,ref_value,'test');wrapper=command[command.index('/usr/bin/env'):]
        base=dict(pid=123,start_ticks='456',ppid=1,cgroup='0::'+runner.p.GROUP_PREFIX+value['unit']+'.service')
        process=[dict(base,argv=wrapper),dict(base,argv=runner.worker_argv(ref_value))]
        if kind=='unknown':process[0]['argv']=['foreign']
        if kind=='reused':process[1]['start_ticks']='457'
        if kind=='timeout':process=[dict(base,argv=wrapper)]
        sample=dict(MainPID='123',InvocationID='a'*32)
        with mock.patch.object(owned,'sample',return_value=sample),mock.patch.object(runner.p,'proc',side_effect=process),mock.patch.object(runner.p,'terminal',return_value=False),mock.patch.object(runner,'once') as saved,mock.patch.object(runner.time,'sleep'),mock.patch.object(runner.time,'monotonic',side_effect=[0,61] if kind=='timeout' else [0,1]):
            if kind=='success':
                result=owned.acknowledge(lambda:None);self.assertEqual(result['argv'],runner.worker_argv(ref_value));self.assertEqual(result['start_ticks'],'456');saved.assert_called_once()
            else:
                with self.assertRaises(RuntimeError):owned.acknowledge(lambda:None)
                saved.assert_not_called()
    def test_exact_env_then_python_same_process(self):self.run_ack('success')
    def test_unknown_argv_fails(self):self.run_ack('unknown')
    def test_env_transition_cannot_change_start_ticks(self):self.run_ack('reused')
    def test_env_transition_keeps_original60s_deadline(self):self.run_ack('timeout')


class PhaseReceiptTests(unittest.TestCase):
    def world(self,fault=None):
        root=Path('/mnt/nvme/lfm3-nki05-device-synthetic');task=protocol.tasks()[0];label=protocol.key(task);admission_ref=ref('admission.json');spec_ref=ref('spec.json');sdk_ref=ref('sdk-manifest.json')
        parent=dict(pid=10,start_ticks='100',ppid=1,argv=['synthetic-worker'],cgroup='0::/synthetic',unit='synthetic',invocation_id='a'*32)
        child=dict(pid=11,start_ticks='110',ppid=10,argv=pipeline.child_argv(runner,admission_ref,task),cgroup='0::/synthetic');ack=dict(pid=11,start_ticks='110',argv=child['argv'],cgroup='/synthetic',parent=dict(parent,cgroup='/synthetic'))
        value,sources=science_value(task);data={};refs=[]
        def reference(path):return dict(path=str(path),sha256=digest(data[str(path)]))
        for i,graph in enumerate(value['graphs']):
            path=root/'raw_device'/str(i)/'model.pt';identity=graph['artifact_identity'];identity.update(path=str(path),admission=admission_ref,task=task,child=child)
            target=path.parent/'artifact-identity.json';data[str(target)]=identity;refs.append(reference(target))
        data[str(root/'worker-owner.json')]=parent;data[admission_ref['path']]=dict(spec=spec_ref);data[spec_ref['path']]=dict(sdk_manifest=sdk_ref)
        data[str(root/'tasks'/(label+'-owner.json'))]=ack;data[str(protocol.scientific_path(root,task))]=value
        sdk_path=root/'tasks'/(label+'-sdk.json');data[str(sdk_path)]=dict(complete=True,passed=True,child=child,admission=admission_ref,manifest=ref('wrong-sdk.json') if fault=='sdk_manifest' else sdk_ref)
        report=dict(complete=True,passed=True,task=task,admission=admission_ref,child=child,science=reference(protocol.scientific_path(root,task)),sdk=reference(sdk_path),artifacts=refs,warning_log=warning('batch'),full_model_qualified=False,new_speedup_claim=False,context4096_qualified=False,self_exit_claimed=False)
        if fault=='missing_artifact':report['artifacts']=refs[:-1]
        result_path=root/'tasks'/(label+'-result.json');data[str(result_path)]=report
        row=dict(task=task,returncode=1 if fault=='nonzero_exit' else 0,passed=True,error=None,secondary_errors=[],report=reference(result_path),pid=11,start_ticks='110')
        data[str(root/'raw_device/phase-outcomes.json')]=dict(complete=True,passed=True,admission=admission_ref,phase='raw_device',outcomes=[row])
        def read(path,wanted=None):return data[str(path)]
        host=types.SimpleNamespace(p=types.SimpleNamespace(read=read,gone=mock.Mock()),PYTHON=runner.PYTHON,SELF=runner.SELF,reference=reference,known_warnings=cpu.known_warnings)
        return host,root,sources,admission_ref
    def test_phase_binds_every_actual_receipt(self):
        host,root,sources,admission=self.world();value=pipeline.read_phase(host,root,'raw_device',sources,admission,require_summary=False)
        self.assertEqual(len(value['graphs']),4);self.assertEqual(len(value['checks']),36);host.p.gone.assert_called_once()
    def test_wrong_sdk_manifest_cannot_satisfy_phase(self):
        host,root,sources,admission=self.world('sdk_manifest')
        with self.assertRaises(RuntimeError):pipeline.read_phase(host,root,'raw_device',sources,admission,require_summary=False)
    def test_dropped_artifact_receipt_cannot_satisfy_phase(self):
        host,root,sources,admission=self.world('missing_artifact')
        with self.assertRaises(RuntimeError):pipeline.read_phase(host,root,'raw_device',sources,admission,require_summary=False)
    def test_actual_nonzero_exit_cannot_satisfy_phase(self):
        host,root,sources,admission=self.world('nonzero_exit')
        with self.assertRaises(RuntimeError):pipeline.read_phase(host,root,'raw_device',sources,admission,require_summary=False)


class SdkBytecodeTests(unittest.TestCase):
    def reject_bytecode(self,encoded):
        import smollm3_record_bytecode_v1 as bytecode
        with tempfile.TemporaryDirectory() as tmp:
            prefix=Path(tmp);site=prefix/'site-packages';directory=site/'mock-1.dist-info';directory.mkdir(parents=True)
            metadata=b'Name: mock\nVersion: 1\n';record=('mock/__pycache__/a.cpython-312.pyc,'+encoded+',12\n').encode()
            files={}
            for name,raw in [('METADATA',metadata),('RECORD',record)]:
                path=directory/name;path.write_bytes(raw);files[name]=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest())
            value=dict(distributions=[dict(directory=str(directory),name='mock',version='1',files=files)],source_overrides={})
            context=types.SimpleNamespace(phase='raw_device',run_root=prefix,authorize=mock.Mock(),fast_authorize=mock.Mock())
            with mock.patch.object(sdk,'SITE',site),mock.patch.object(sdk,'PREFIX',prefix),mock.patch.object(sdk,'validate_manifest'),mock.patch.object(sdk.sys,'prefix',str(prefix)),mock.patch.object(sdk.sys,'version_info',(3,12)),mock.patch.object(bytecode,'isolated'):
                with self.assertRaisesRegex(RuntimeError,'no bytecode exemption'):sdk.verify(value,context)
    def test_hashed_pyc_cannot_expand_old_single_numpy_exception(self):self.reject_bytecode('sha256='+'a'*43)
    def test_unhashed_pyc_cannot_expand_old_single_numpy_exception(self):self.reject_bytecode('')


class SerialChildTests(unittest.TestCase):
    def failure_case(self,mode):
        from contextlib import nullcontext
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'tasks').mkdir();task=protocol.tasks()[0];child=types.SimpleNamespace(pid=111,returncode=7,wait=mock.Mock(return_value=7));ack=dict(start_ticks='200')
            host=types.SimpleNamespace(ROOT=ROOT,PYTHON='synthetic-python',SELF='synthetic-runner',once=mock.Mock(),reference=mock.Mock(),environment=lambda *a:{})
            with mock.patch.object(pipeline,'child_binding',return_value=nullcontext()),mock.patch.object(pipeline.subprocess,'Popen',return_value=child),mock.patch.object(pipeline.legacy,'child_identity',side_effect=RuntimeError('ACK failure') if mode=='unacknowledged' else None,return_value=ack),mock.patch.object(pipeline.legacy,'stop_child',side_effect=RuntimeError('drain failure') if mode=='drain_failure' else None) as stopped:
                outcome,error=pipeline.run_one(host,{},ref('admission.json'),dict(attempt_id='test'),{},root,task,lambda:True,30)
                self.assertIsNotNone(error);self.assertFalse(outcome['passed'])
                if mode=='unacknowledged':stopped.assert_not_called();self.assertTrue(outcome['secondary_errors'])
                else:stopped.assert_called_once_with(child,ack);self.assertEqual(outcome['returncode'],7)
                if mode=='drain_failure':self.assertTrue(outcome['secondary_errors'])
    def test_nonzero_actual_exit_is_preserved_and_child_drained(self):self.failure_case('exit_failure')
    def test_unacknowledged_pid_never_signalled_by_child_helper(self):self.failure_case('unacknowledged')
    def test_failed_child_drain_cannot_report_success(self):self.failure_case('drain_failure')


class FileAndDrainTests(unittest.TestCase):
    def test_complete_artifact_hash_with_mock_bytes_only(self):
        path=mock.Mock();path.is_absolute.return_value=True;path.resolve.return_value=path;path.is_symlink.return_value=False;path.parents=[]
        info=types.SimpleNamespace(st_mode=stat.S_IFREG,st_nlink=1,st_dev=1,st_ino=2,st_size=4,st_mtime_ns=3,st_ctime_ns=4)
        with mock.patch.object(runtime,'Path',return_value=path),mock.patch.object(runtime.os,'open',return_value=19),mock.patch.object(runtime.os,'fstat',return_value=info),mock.patch.object(runtime.os,'read',side_effect=[b'abcd',b'']),mock.patch.object(runtime.os,'close'):
            self.assertEqual(runtime.stable_hash('synthetic',4),hashlib.sha256(b'abcd').hexdigest())
    def test_artifact_cap_rejects_before_read(self):
        path=mock.Mock();path.is_absolute.return_value=True;path.resolve.return_value=path;path.is_symlink.return_value=False;path.parents=[]
        info=types.SimpleNamespace(st_mode=stat.S_IFREG,st_nlink=1,st_size=5)
        with mock.patch.object(runtime,'Path',return_value=path),mock.patch.object(runtime.os,'open',return_value=19),mock.patch.object(runtime.os,'fstat',return_value=info),mock.patch.object(runtime.os,'read') as read,mock.patch.object(runtime.os,'close'):
            with self.assertRaises(RuntimeError):runtime.stable_hash('synthetic',4)
            read.assert_not_called()
    def test_scientific_progress_only_owns_its_exact_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'raw_device').mkdir();value=runtime.Context.__new__(runtime.Context)
            value.run_root=root;value.task=dict(phase='raw_device',index=-1);value.phase='raw_device';value.saved={}
            value.authorize=mock.Mock(return_value=True);value.fast_authorize=mock.Mock(return_value=True)
            value.host=types.SimpleNamespace(encode=lambda x:json.dumps(x).encode(),p=types.SimpleNamespace(sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()))
            path=root/'raw_device/device.json';value.save(path,{'n':1},'raw_device');value.save(path,{'n':2},'raw_device')
            self.assertEqual(json.loads(path.read_text()),{'n':2})
            path.write_text('{"foreign":true}')
            with self.assertRaises(RuntimeError):value.save(path,{'n':3},'raw_device')
            with self.assertRaises(RuntimeError):value.save(root/'other.json',{},'raw_device')
    def test_drain_before_unlock_original_mechanism(self):
        order=[];owned=types.SimpleNamespace(cleanup=lambda:order.append('cleanup'),drained=lambda:dict(physically_drained=True))
        with mock.patch.object(runner.fcntl,'flock',side_effect=lambda *a:order.append('unlock')),mock.patch.object(runner.os,'close',side_effect=lambda *a:order.append('close')):
            runner.drain_and_release(owned,19,lambda x:None,lambda:None,lambda p:order.append('record'))
        self.assertEqual(order,['cleanup','record','unlock','close'])
    def test_no_unlock_when_drain_remains_unproved(self):
        owned=types.SimpleNamespace(cleanup=lambda:None,drained=mock.Mock(side_effect=[RuntimeError('live descendant'),dict(physically_drained=True)]))
        with mock.patch.object(runner.fcntl,'flock') as unlock,mock.patch.object(runner.os,'close') as close:
            def while_undrained():unlock.assert_not_called();close.assert_not_called()
            retry=mock.Mock(side_effect=while_undrained)
            runner.drain_and_release(owned,19,lambda x:None,retry,lambda p:None)
            retry.assert_called_once();unlock.assert_called_once();close.assert_called_once()

if __name__=='__main__':
    resource.setrlimit(resource.RLIMIT_AS,(256*2**20,256*2**20));resource.setrlimit(resource.RLIMIT_CPU,(60,60));resource.setrlimit(resource.RLIMIT_FSIZE,(2*2**20,2*2**20))
    os.sched_setaffinity(0,{7});os.nice(19)
    unittest.main(verbosity=2)
