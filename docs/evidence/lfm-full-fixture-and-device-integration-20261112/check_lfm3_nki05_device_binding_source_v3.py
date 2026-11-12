"""Source/mock verification only. Never imports native libraries or runs devices."""
import ast
import copy
import hashlib
import importlib
import importlib.abc
import json
import math
import os
from pathlib import Path
import resource
import sys
import types
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parent
BLOCKED=('torch','torch_neuronx','torch_xla','numpy','ml_dtypes','nki','neuronxcc','ctypes','vllm','vllm_neuron')
class NativeBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in BLOCKED:
            raise AssertionError('Native import forbidden in source/mock checks: '+fullname)
BLOCKER=NativeBlocker()
sys.meta_path.insert(0,BLOCKER)
NAMES=['slot_cache_nki05_typed_jit_v1','slot_cache_nki05_typed_torch_v1',
    'lfm3_nki05_typed_dispatch_v1','lfm3_nki05_device_context_v1','lfm3_nki05_device_oracle_v1',
    'lfm3_nki05_device_matrix_v1','lfm3_nki05_raw_device_leaf_v1',
    'lfm3_nki05_regression_device_leaf_v1','lfm3_nki05_regression_alias_leaf_v1',
    'lfm3_nki05_extended_device_leaf_v2','lfm3_nki05_device_supplements_v2']
MODULES={name:importlib.import_module(name) for name in NAMES}
jit=MODULES[NAMES[0]];wrapper=MODULES[NAMES[1]];dispatch=MODULES[NAMES[2]]
ctx=MODULES[NAMES[3]];matrix=MODULES[NAMES[5]];raw=MODULES[NAMES[6]]
regdev=MODULES[NAMES[7]];regalias=MODULES[NAMES[8]];extended=MODULES[NAMES[9]];supp=MODULES[NAMES[10]]


def tree(name):return ast.parse((ROOT/name).read_text())
def function(name,fn):return next(x for x in tree(name).body if isinstance(x,ast.FunctionDef) and x.name==fn)
def dump(node):return ast.dump(node,include_attributes=False)
def subkey(node,key):return isinstance(node,ast.Subscript) and isinstance(node.slice,ast.Constant) and node.slice.value==key

def cpu_proof():
    return dict(schema='lfm3_nki05_full_cpu_result_v1',complete=True,passed=True,
        stage_counts={**{k:dict(cases=3716,batches=291) for k in ('original_fixture','new_fixture','candidate')},'controls':dict(cases=1,batches=1)},
        cpu_full_gate_passed=True,simulator_writeback_workaround_required=True,original_simulator_failure_preserved=True,
        device_behavior_proven=False,alias_gate_passed=False,full_model_qualified=False,new_speedup_claim=False,automatic_successor_launch=False)

class FakeContext(ctx.DeviceContext):
    def __init__(self,allow=True):self.allow=allow;self.saved=[];self.proof=cpu_proof();self.bad_receipt={}
    def authorize(self,phase):return self.allow
    def prerequisite(self,name):return self.proof
    def paths(self,phase):return ROOT,Path('/source-only-mock-output')/phase
    def sources(self):return {'immutable-mock-source.py':'a'*64}
    def save(self,path,value,phase):self.saved.append((str(path),value,phase))
    def hash_artifact(self,path,phase):return 'b'*64
    def record_artifact(self,path,phase,case):
        return dict(schema=ctx.ARTIFACT_SCHEMA,variant=ctx.VARIANT,software=dict(jit.DEVICE_SOFTWARE),
            source_sha256=self.sources(),phase=phase,path=str(path),case=case,sha256='b'*64,**self.bad_receipt)

class Tensor:
    """Payload values are raw integer words; no floats, arithmetic or dtype views."""
    def __init__(self,data,shape,dtype,device='cpu',contiguous=True,events=None):
        self.data=list(data);self.shape=tuple(shape);self.dtype=dtype;self.device=device;self.ndim=len(self.shape)
        self.contiguous=contiguous;self.events=[] if events is None else events
        assert len(self.data)==math.prod(self.shape)
    def is_contiguous(self):return self.contiguous
    def reshape(self,*shape):
        shape=tuple(shape[0]) if len(shape)==1 and isinstance(shape[0],(tuple,list)) else tuple(shape)
        self.events.append(('reshape',self.dtype,shape))
        return Tensor(self.data,shape,self.dtype,self.device,self.contiguous,self.events)
    def __ge__(self,other):return Tensor([v>=other for v in self.data],self.shape,'bool',events=self.events)
    def __lt__(self,other):return Tensor([v<other for v in self.data],self.shape,'bool',events=self.events)
    def __and__(self,other):return Tensor([a and b for a,b in zip(self.data,other.data)],self.shape,'bool',events=self.events)
    def to(self,dtype):
        assert self.dtype in ('int32','int64') and dtype=='int32','Payload narrowing is forbidden'
        self.events.append(('to',list(self.data),dtype))
        return Tensor([((v+2**31)%2**32)-2**31 for v in self.data],self.shape,dtype,events=self.events)

class FakeTorch:
    bfloat16='bfloat16';float32='float32';int32='int32';int64='int64'
    @staticmethod
    def full_like(tensor,value):return Tensor([value]*len(tensor.data),tensor.shape,tensor.dtype,events=tensor.events)
    @staticmethod
    def where(valid,yes,no):
        result=[a if chosen else b for chosen,a,b in zip(valid.data,yes.data,no.data)]
        yes.events.append(('where',result))
        return Tensor(result,yes.shape,yes.dtype,events=yes.events)

class FakeKernel:
    def __init__(self):self.calls=[]
    def __call__(self,cache,updates,slots):
        self.calls.append((cache,updates,slots));b,width=cache.shape;result=cache.data[:]
        for row,slot in enumerate(slots.data):
            if 0<=slot<b:result[slot*width:(slot+1)*width]=updates.data[row*width:(row+1)*width]
        return Tensor(result,cache.shape,cache.dtype)

def inputs(b=4,rows=4,dtype='bfloat16',slots=None,slot_dtype='int64'):
    if slots is None:slots=list(range(rows))
    # Raw exceptional words plus row marker; per-row payloads differ.
    bits=(0,0x8000,1,0x7fc1,0x7fa1,0x7f80,0xff80,0x8001) if dtype=='bfloat16' else (0,0x80000000,1,0x7fc00001,0x7fa00001,0x7f800000,0xff800000,0x80000001)
    c=Tensor([bits[i%len(bits)]^(row+1) for row in range(b) for i in range(256)],(b,2,128),dtype)
    u=Tensor([bits[::-1][i%len(bits)]^(row+17) for row in range(rows) for i in range(256)],(rows,2,128),dtype)
    s=Tensor(slots,(rows,),slot_dtype)
    return c,u,s

class ScienceNormalizer(ast.NodeTransformer):
    """TEST ONLY: allow path/artifact wiring and selection of the original device branch."""
    def visit_If(self,node):
        if isinstance(node.test,ast.Compare) and dump(node.test.left)==dump(ast.Name(id='mode',ctx=ast.Load())):
            value=ast.literal_eval(node.test.comparators[0])
            chosen=node.body if value=='device' else node.orelse
            output=[]
            for stmt in chosen:
                changed=self.visit(stmt)
                output.extend(changed if isinstance(changed,list) else [changed] if changed else [])
            return output
        return self.generic_visit(node)
    def visit_Assign(self,node):
        if any(isinstance(t,ast.Name) and t.id=='artifact_identity' or subkey(t,'artifact_identity') or subkey(t,'source_sha256') for t in node.targets):return None
        return self.generic_visit(node)
    def visit_Dict(self,node):
        pairs=[]
        for key,value in zip(node.keys,node.values):
            if isinstance(key,ast.Constant) and key.value=='artifact_identity':continue
            if isinstance(key,ast.Constant) and key.value=='artifact_sha256':value=ast.Constant(value='ARTIFACT_SHA_WIRING')
            pairs.append((key,value))
        node.keys=[x for x,y in pairs];node.values=[y for x,y in pairs]
        return self.generic_visit(node)
    def visit_BinOp(self,node):
        if isinstance(node.op,ast.Div) and isinstance(node.right,ast.Constant) and node.right.value in ('extended_device','extended_alias') and dump(node.left)==dump(ast.Attribute(value=ast.Name(id='p',ctx=ast.Load()),attr='DEST',ctx=ast.Load())):
            return node.left
        return self.generic_visit(node)

def science(nodes):
    return dump(ScienceNormalizer().visit(ast.Module(body=copy.deepcopy(nodes),type_ignores=[])))

class FactoryTests(unittest.TestCase):
    def test_real_api_shape(self):
        fn=lambda:None;api=types.SimpleNamespace(jit=mock.Mock(return_value='JITTED'))
        result=jit.wrap_device_kernel(fn,api,dict(NEURON_PLATFORM_TARGET_OVERRIDE='trn1',NEURON_LOGICAL_NC_CONFIG='1'))
        self.assertEqual(result,'JITTED');api.jit.assert_called_once_with(fn)
    def test_environment_rejections(self):
        valid=dict(NEURON_PLATFORM_TARGET_OVERRIDE='trn1',NEURON_LOGICAL_NC_CONFIG='1',NKI_SIMULATOR='0')
        for key,value in [('NEURON_PLATFORM_TARGET_OVERRIDE','trn2'),('NEURON_LOGICAL_NC_CONFIG','2'),('NKI_SIMULATOR','1')]:
            with self.subTest(key=key),self.assertRaises(RuntimeError):jit.wrap_device_kernel(lambda:None,types.SimpleNamespace(jit=lambda f:f),dict(valid,**{key:value}))
    def test_noncallable_rejected(self):
        with self.assertRaises(RuntimeError):jit.wrap_device_kernel(None,None,{})
    def test_admission_before_metadata_or_native(self):
        with mock.patch.object(jit.metadata,'version',side_effect=AssertionError('metadata reached')):
            for admit in (None,lambda:False,lambda:1):
                with self.assertRaises(RuntimeError):jit.make_device_kernel(admit)
    def test_exact_software_rejected_before_native(self):
        with mock.patch.object(jit.metadata,'version',return_value='OLD'),self.assertRaises(RuntimeError):jit.make_device_kernel(lambda:True)
    def test_factory_calls_observer_free_function_and_jit(self):
        plain=lambda:None;kernel=importlib.import_module('slot_cache_nki05_typed_selector_v1')
        api=types.SimpleNamespace(jit=mock.Mock(return_value='compiled-function'))
        admission=mock.Mock(return_value=True)
        with mock.patch.dict(sys.modules,{'nki':api}),mock.patch.object(kernel,'make_kernel',return_value=plain),mock.patch.object(jit.metadata,'version',side_effect=jit.DEVICE_SOFTWARE.__getitem__),mock.patch.dict(os.environ,dict(NEURON_PLATFORM_TARGET_OVERRIDE='trn1',NEURON_LOGICAL_NC_CONFIG='1',NKI_SIMULATOR='0')):
            self.assertEqual(jit.make_device_kernel(admission),'compiled-function')
        api.jit.assert_called_once_with(plain);self.assertEqual(admission.call_count,2)
    def test_factory_rechecks_admission(self):
        kernel=importlib.import_module('slot_cache_nki05_typed_selector_v1')
        with mock.patch.dict(sys.modules,{'nki':types.SimpleNamespace(jit=lambda f:f)}),mock.patch.object(kernel,'make_kernel',return_value=lambda:None),mock.patch.object(jit.metadata,'version',side_effect=jit.DEVICE_SOFTWARE.__getitem__),mock.patch.dict(os.environ,dict(NEURON_PLATFORM_TARGET_OVERRIDE='trn1',NEURON_LOGICAL_NC_CONFIG='1',NKI_SIMULATOR='0')),self.assertRaises(RuntimeError):
            jit.make_device_kernel(mock.Mock(side_effect=[True,False]))

class WrapperTests(unittest.TestCase):
    def test_all_supplemental_selectors_match_unbounded_integer_oracle(self):
        for spec in supp.wrapper_specs():
            b,rows,dtype=spec['capacity'],spec['rows'],spec['dtype']
            for slots in supp.control_mappings(b,rows):
                with self.subTest(b=b,rows=rows,dtype=dtype,slots=slots):
                    c,u,s=inputs(b,rows,dtype,slots);kernel=FakeKernel();before=(c.data[:],u.data[:],s.data[:])
                    actual=wrapper.apply_typed_writer(c,u,s,FakeTorch,kernel)
                    expected=c.data[:]
                    for row,slot in enumerate(slots):
                        if 0<=slot<b:expected[slot*256:(slot+1)*256]=u.data[row*256:(row+1)*256]
                    self.assertEqual(actual.data,expected);self.assertEqual(actual.shape,c.shape);self.assertEqual(actual.dtype,dtype)
                    self.assertEqual((c.data,u.data,s.data),before)
                    self.assertEqual(kernel.calls[0][2].data,[x if 0<=x<b else -1 for x in slots])
                    self.assertTrue(all(x[0]=='reshape' for x in c.events+u.events))
    def test_normalization_occurs_before_narrowing(self):
        c,u,s=inputs(slots=[2**32,2**32+3,-2**32,-2**63]);kernel=FakeKernel()
        wrapper.apply_typed_writer(c,u,s,FakeTorch,kernel)
        self.assertEqual(s.events,[('where',[-1]*4),('to',[-1]*4,'int32')])
    def test_int32_slots(self):
        c,u,s=inputs(slot_dtype='int32');out=wrapper.apply_typed_writer(c,u,s,FakeTorch,FakeKernel());self.assertEqual(out.data,u.data)
    def test_last_writer_duplicates(self):
        c,u,s=inputs(slots=[2,2,-1,2]);out=wrapper.apply_typed_writer(c,u,s,FakeTorch,FakeKernel())
        self.assertEqual(out.data[512:768],u.data[768:1024]);self.assertEqual(out.data[:512],c.data[:512])
    def test_dtype_device_shape_contiguity_rejections(self):
        mutations=[lambda c,u,s:setattr(c,'dtype','float16'),lambda c,u,s:setattr(u,'dtype','float32'),lambda c,u,s:setattr(s,'dtype','float32'),lambda c,u,s:setattr(s,'device','other'),lambda c,u,s:setattr(u,'contiguous',False),lambda c,u,s:setattr(s,'shape',(2,2)),lambda c,u,s:setattr(u,'shape',(4,1,256)),lambda c,u,s:setattr(c,'shape',(3,2,128))]
        for mutation in mutations:
            c,u,s=inputs();mutation(c,u,s)
            with self.assertRaises(AssertionError):wrapper.apply_typed_writer(c,u,s,FakeTorch,FakeKernel())
    def test_width_alignment_rejected(self):
        c=Tensor([0]*4*129,(4,129),'float32');u=Tensor([1]*4*129,(4,129),'float32');s=Tensor(range(4),(4,),'int64')
        with self.assertRaises(AssertionError):wrapper.apply_typed_writer(c,u,s,FakeTorch,FakeKernel())
    def test_wrong_output_dtype_rejected(self):
        c,u,s=inputs()
        with self.assertRaises(AssertionError):wrapper.apply_typed_writer(c,u,s,FakeTorch,lambda a,b,c:Tensor(a.data,a.shape,'float32'))
    def test_wrong_output_shape_rejected(self):
        c,u,s=inputs()
        with self.assertRaises(AssertionError):wrapper.apply_typed_writer(c,u,s,FakeTorch,lambda a,b,c:Tensor(a.data,(2,512),a.dtype))
    def test_lazy_writer_admission_prevents_torch_import(self):
        writer=wrapper.make_typed_writer(FakeKernel(),lambda:False)
        with self.assertRaises(AssertionError):writer(*inputs())
    def test_lazy_writer_rechecks_after(self):
        writer=wrapper.make_typed_writer(FakeKernel(),mock.Mock(side_effect=[True,False]))
        with mock.patch.dict(sys.modules,{'torch':FakeTorch}),self.assertRaises(AssertionError):writer(*inputs())
    def test_production_dispatch_one_row_and_multiple(self):
        for b in (4,8,16):
            for rows in sorted({1,4,b}):
                original=mock.Mock(return_value='original');typed=mock.Mock(return_value='typed')
                write=dispatch.make_writer(b,original,typed)
                self.assertEqual(write(*inputs(b,rows)),'original' if rows==1 else 'typed')
                self.assertEqual(original.call_count,1 if rows==1 else 0);self.assertEqual(typed.call_count,0 if rows==1 else 1)

class ContextTests(unittest.TestCase):
    def test_abstract_context_not_runnable(self):
        with self.assertRaises(TypeError):ctx.DeviceContext()
    def test_every_leaf_unbound_rejects_before_native(self):
        calls=[lambda:raw.run(None),lambda:regdev.run(None),lambda:regalias.run(None),lambda:extended.device(None,0),lambda:extended.alias(None,0),lambda:supp.fp32_alias(None,0),lambda:supp.wrapper_controls(None,0)]
        for call in calls:
            with self.assertRaises(RuntimeError):call()
    def test_explicit_denial_rejects_before_native(self):
        with self.assertRaises(RuntimeError):raw.run(FakeContext(False))
    def test_partial_cpu_proof_rejected_before_native(self):
        value=FakeContext();value.proof['stage_counts']['candidate']['cases']=16
        with self.assertRaises(RuntimeError):raw.run(value)
    def test_complete_cpu_summary_only(self):ctx.require_cpu_summary(cpu_proof())
    def test_each_cpu_pass_and_controls_required(self):
        for stage in ('original_fixture','new_fixture','candidate','controls'):
            value=cpu_proof();del value['stage_counts'][stage]
            with self.assertRaises(RuntimeError):ctx.require_cpu_summary(value)
    def test_cpu_counts_and_flags_cannot_promote(self):
        for key in ('complete','passed','cpu_full_gate_passed','simulator_writeback_workaround_required','original_simulator_failure_preserved','device_behavior_proven','alias_gate_passed','full_model_qualified','new_speedup_claim','automatic_successor_launch'):
            value=cpu_proof();value[key]=not value[key]
            with self.assertRaises(RuntimeError):ctx.require_cpu_summary(value)
    def test_unknown_phase_rejected(self):
        with self.assertRaises(RuntimeError):ctx.require_context(FakeContext(),'serving')
    def test_matrix_io_bound_to_phase(self):
        value=FakeContext();p=value.protocol('extended_device');p.save(p.DEST/'a.json',{'x':1})
        self.assertEqual(value.saved,[('/source-only-mock-output/extended_device/a.json',{'x':1},'extended_device')]);self.assertEqual(p.sha(p.DEST/'fake.pt'),'b'*64)
    def test_artifact_receipt_exact_identity(self):
        value=FakeContext();phase='raw_device';case={'shape':[4,256],'dtype':'float32','variant':'candidate'}
        receipt=value.artifact(value.paths(phase)[1]/'model.pt',phase,case)
        self.assertEqual(receipt['software'],jit.DEVICE_SOFTWARE);self.assertEqual(receipt['case'],case)
    def test_old_artifact_reuse_rejected(self):
        with self.assertRaises(RuntimeError):FakeContext().artifact('/old-package/model.pt','raw_device',{})
    def test_receipt_drift_rejected(self):
        for key in ('schema','variant','software','source_sha256','case','sha256','phase','path'):
            value=FakeContext();original=value.record_artifact
            value.record_artifact=lambda p,ph,c,key=key:dict(original(p,ph,c),**{key:'drift'})
            with self.assertRaises(RuntimeError):value.artifact(value.paths('raw_device')[1]/'model.pt','raw_device',{})
    def test_runtime_budgets_and_4096_still_pending(self):
        self.assertFalse(ctx.RUNTIME_ADMITTED);self.assertIsNone(ctx.COMPILE_PEAK_BYTES);self.assertIsNone(ctx.COMPILE_ARTIFACT_CAP_BYTES)
        self.assertTrue(ctx.CONTEXT4096_REQUIRED)
        for key in ('cold4096_required','long4096_required','slot_reuse4096_required','serving_graph4096_required'):self.assertIs(ctx.REQUIRED_GATES[key],True)
        self.assertEqual(ctx.FULL_MODEL_GATES['comparisons']['measured_requests_per_variant'],704)
        self.assertEqual(ctx.FULL_MODEL_GATES['comparisons']['measured_output_tokens_per_variant'],189952)
        self.assertEqual(ctx.FULL_MODEL_GATES['b16']['long_steps'],1529)

class ScientificSourceTests(unittest.TestCase):
    def test_all_original_pins_unchanged(self):
        manifest=json.loads((ROOT/'research/lfm3-nki05-device-pretest-source-manifest-20261112-v1.json').read_text())
        for name,digest in {**manifest['predecessors'],**manifest['kernel']}.items():self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),digest,name)
    def test_matrix_function_bodies_exact(self):
        for name in ('mappings','cpu_cases','device_specs','alias_specs','alias_mappings'):
            self.assertEqual(dump(function('lfm3_batch_slots_bits_protocol_v4.py',name)),dump(function('lfm3_nki05_device_matrix_v1.py',name)),name)
    def test_full_original_matrix_counts(self):
        self.assertEqual(len(matrix.cpu_cases()),2068);self.assertEqual(len(matrix.device_specs()),42)
        self.assertEqual(sum(len(matrix.mappings(x['capacity'],x['rows'])) for x in matrix.device_specs()),2520)
        self.assertEqual(len(matrix.alias_specs()),24);self.assertEqual(sum(len(matrix.alias_mappings(x['capacity'])) for x in matrix.alias_specs()),192)
    def test_raw_mapping_and_entire_nested_device_loop_exact(self):
        old=function('probe_slot_cache_nki_predicated.py','main');new=function('lfm3_nki05_raw_device_leaf_v1.py','run')
        self.assertEqual(science([next(x for x in old.body if isinstance(x,ast.Try)).body[0]]),science([next(x for x in new.body if isinstance(x,ast.Try)).body[0]]))
        node=next(x for x in tree('probe_slot_cache_nki_predicated.py').body if isinstance(x,ast.Assign) and isinstance(x.targets[0],ast.Name) and x.targets[0].id=='MAPPINGS')
        self.assertEqual(raw.MAPPINGS,ast.literal_eval(node.value));self.assertEqual(len(raw.MAPPINGS)*4,36)
    def test_regression_device_full_try_exact(self):
        old=function('lfm3_package_primitive_device_v2.py','main');new=function('lfm3_nki05_regression_device_leaf_v1.py','run')
        self.assertEqual(science([next(x for x in old.body if isinstance(x,ast.Try))]),science([next(x for x in new.body if isinstance(x,ast.Try))]))
        for name in ('shapes','mappings'):
            def rows(fn):return [x for x in fn.body if isinstance(x,(ast.Assign,ast.AugAssign)) and ((isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in x.targets)) or (isinstance(x,ast.AugAssign) and isinstance(x.target,ast.Name) and x.target.id==name))]
            self.assertEqual(science(rows(old)),science(rows(new)))
    def test_regression_alias_full_try_and_class_exact(self):
        old=function('lfm3_package_primitive_alias_v2.py','main');new=function('lfm3_nki05_regression_alias_leaf_v1.py','run')
        for typ in (ast.Try,ast.ClassDef):self.assertEqual(science([next(x for x in old.body if isinstance(x,typ))]),science([next(x for x in new.body if isinstance(x,typ))]))
    def test_extended_complete_scientific_bodies_exact(self):
        for name in ('device','alias'):
            old=function('lfm3_batch_slots_bits_extended_v4.py',name);new=function('lfm3_nki05_extended_device_leaf_v2.py',name)
            def scientific_tail(fn):
                start=next(i for i,x in enumerate(fn.body) if isinstance(x,ast.Expr) and isinstance(x.value,ast.Call) and isinstance(x.value.func,ast.Attribute) and x.value.func.attr=='set_num_threads')
                return fn.body[start:]
            self.assertEqual(science(scientific_tail(old)),science(scientific_tail(new)),name)
    def test_byte_compare_helpers_exact(self):
        for name in ('torch_same','tensor_sha'):self.assertEqual(dump(function('lfm3_batch_slots_bits_extended_v4.py',name)),dump(function('lfm3_nki05_extended_device_leaf_v2.py',name)))
    def test_sequential_oracle_exact(self):
        old=function('lfm3-package-native-core-v1/work/target/vllm_neuron/models/lfm3/helpers.py','store_slots')
        new=function('lfm3_nki05_device_oracle_v1.py','store_slots')
        self.assertEqual(science([x for x in old.body if not (isinstance(x,ast.Expr) and isinstance(x.value,ast.Constant) and isinstance(x.value.value,str))]),science([x for x in new.body if not isinstance(x,ast.Import)]))
    def test_wrapper_semantics_ast_same_after_explicit_injection(self):
        old=function('slot_cache_typed_selector_torch_v1.py','store_slots_typed_selector');new=function('slot_cache_nki05_typed_torch_v1.py','apply_typed_writer')
        class Names(ast.NodeTransformer):
            def visit_Name(self,node):
                if node.id=='torch':node.id='torch_api'
                if node.id=='slot_cache_typed_selector':node.id='device_kernel'
                return node
        expected=[x for x in old.body if not isinstance(x,(ast.Import,ast.ImportFrom))]
        expected=Names().visit(ast.Module(body=expected,type_ignores=[]))
        self.assertEqual(dump(expected),dump(ast.Module(body=new.body,type_ignores=[])))
    def test_no_runtime_ast_rewrite_simulator_or_shell_launcher(self):
        for name in NAMES:
            source=(ROOT/(name+'.py')).read_text();parsed=ast.parse(source)
            imported=[x.name.split('.')[0] for n in ast.walk(parsed) if isinstance(n,ast.Import) for x in n.names]
            self.assertNotIn('ast',imported);self.assertNotIn('subprocess',imported)
            calls=[n.func.id for n in ast.walk(parsed) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
            self.assertFalse({'exec','eval','compile'}&set(calls))
            self.assertNotIn('simulate_kernel',source);self.assertNotIn('writeback_shim',source)
            self.assertFalse(any(isinstance(n,ast.If) and '__main__' in ast.unparse(n.test) for n in parsed.body))
    def test_extended_alias_receipt_contains_actual_shapes_dtype_and_variant(self):
        parsed=function('lfm3_nki05_extended_device_leaf_v2.py','alias')
        calls=[x for x in ast.walk(parsed) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='artifact']
        self.assertEqual(len(calls),1)
        case=calls[0].args[2]
        self.assertEqual(ast.unparse(case.args[0]),'spec')
        self.assertEqual({x.arg for x in case.keywords},{'dtype','rows','shapes'})
        self.assertIn('initial',ast.unparse(case))

    def test_actual_trace_and_alias_operations_retained(self):
        for name in NAMES[6:]:
            source=(ROOT/(name+'.py')).read_text();self.assertIn('torch_neuronx.trace(',source);self.assertIn('context.artifact(',source)
        for name in (NAMES[8],NAMES[9],NAMES[10]):
            source=(ROOT/(name+'.py')).read_text();self.assertIn('input_output_aliases=',source);self.assertIn('move_trace_to_device(',source);self.assertIn('compiled.states._parameters',source)
    def test_supplemental_matrices_preserve_originals(self):
        self.assertEqual(len(supp.wrapper_specs()),16);self.assertEqual(len(supp.fp32_alias_specs()),6)
        for spec in supp.fp32_alias_specs():self.assertEqual(spec['shape'],[spec['capacity'],64,80,128])
        for b in (4,8,16):
            for rows in sorted({1,4,b}):
                mappings=supp.control_mappings(b,rows)
                self.assertIn([2**32]*rows,mappings);self.assertIn([2**32+b-1]*rows,mappings)
                self.assertIn([-2**63]*rows,mappings);self.assertIn([2**63-1]*rows,mappings)
    def test_supplemental_raw_patterns_distinguish_every_slot(self):
        # Same pure word construction, without allocating a Torch tensor.
        for dtype in ('bfloat16','float32'):
            words=list(supp.bit_patterns(dtype))
            for b in (4,8,16):
                rows=[tuple(words[-i:]+words[:-i]) for i in range(b)]
                self.assertEqual(len(set(rows)),b)
        self.assertIn('raw.roll(row)',(ROOT/'lfm3_nki05_device_supplements_v2.py').read_text())
    def test_no_native_modules_loaded(self):
        self.assertFalse(set(BLOCKED)&{name.split('.')[0] for name in sys.modules})

if __name__=='__main__':
    resource.setrlimit(resource.RLIMIT_AS,(256*2**20,256*2**20))
    resource.setrlimit(resource.RLIMIT_CPU,(60,60))
    resource.setrlimit(resource.RLIMIT_FSIZE,(2*2**20,2*2**20))
    os.sched_setaffinity(0,{7});os.nice(19)
    unittest.main(verbosity=2)
