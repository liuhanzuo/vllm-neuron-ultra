"""Source/metadata/virtual-shape tests only; no actual native/model/raw I/O.

The 507-weight fixture has only dimensions and old metadata digests; it is not
weight data, native accuracy evidence or a resource estimate. Tiny durable JSON
uses the prior fake filesystem and BytesIO exclusively.
"""
import ast
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import dory_stage_a_worker_preparation_v1 as worker
import dory_stage_a_durable_metadata_v1 as durable
import prepare_dory_stage_a_source_inputs_v1 as preparer
import dory_stage_a_artifact_transaction_v2 as transaction
from check_dory_stage_a_artifact_transaction_v2 import FakeFS, FakeFile
from test_dory_stage_a_tensor_adapter_v1 import FakeTorch, patterned, independent_expected


class DurableMetadataTests(unittest.TestCase):
    def setUp(self):
        self.fs=FakeFS()
        self.patches=[patch.object(transaction,'os',self.fs),patch.object(transaction,'Path',self.fs.Path),
                      patch.object(durable,'os',self.fs)]
        for p in self.patches:p.start()
        self.addCleanup(lambda:[p.stop() for p in reversed(self.patches)])

    def write(self,value=None,limit=65536):
        return durable.write_json_exclusive('/admitted','manifest.json',{'x':1} if value is None else value,
                                            max_bytes=limit,guard=lambda:None)

    def test_exact_utf8_and_order_durable_identity(self):
        value={'tensors':{'z':{'x':1},'a':{'x':2}},'text':'原始'}
        result=self.write(value)
        raw=self.fs.entries['manifest.json'].data
        self.assertEqual(json.loads(raw),value)
        self.assertEqual(list(json.loads(raw)['tensors']),['z','a'])
        self.assertEqual(result['sha256'],hashlib.sha256(raw).hexdigest())
        self.assertEqual(result['bytes'],len(raw))
        self.assertFalse(result['actual_process_exit_verified_here'])
        events=self.fs.events
        self.assertLess(events.index(('flush','manifest.json')),events.index(('fsync','manifest.json')))
        self.assertLess(events.index(('fsync','manifest.json')),events.index(('close_file','manifest.json')))
        self.assertLess(events.index(('close_file','manifest.json')),events.index(('fsync','/admitted')))
        self.assertEqual(self.fs.fds,{})

    def test_nonfinite_and_unsupported_rejected_before_create(self):
        for value in ({'x':float('nan')},{'x':float('inf')},{1:'key'},{'x':b'bytes'}):
            with self.subTest(value=repr(value)),self.assertRaises((RuntimeError,ValueError)):self.write(value)
        self.assertEqual(self.fs.events,[])

    def test_byte_bound_before_directory_or_file_open(self):
        with self.assertRaisesRegex(RuntimeError,'byte bound'):self.write({'x':'中文'},limit=10)
        self.assertEqual(self.fs.events,[])

    def test_cyclic_and_depth_bounded_before_open(self):
        cycle=[];cycle.append(cycle)
        deep=[]
        for i in range(40):deep=[deep]
        for value in (cycle,deep):
            with self.assertRaises(RuntimeError):self.write(value)
        self.assertEqual(self.fs.events,[])

    def test_existing_json_not_overwritten(self):
        first=self.write();old=self.fs.entries['manifest.json']
        with self.assertRaises(FileExistsError):self.write({'x':2})
        self.assertIs(self.fs.entries['manifest.json'],old)
        self.assertEqual(hashlib.sha256(old.data).hexdigest(),first['sha256'])

    def test_failure_stages_leave_partial_and_no_receipt(self):
        for operation,name in [('write','manifest.json'),('flush','manifest.json'),('fsync','manifest.json'),
                               ('close_file','manifest.json'),('fsync','/admitted')]:
            with self.subTest(operation=operation,name=name):
                self.fs=FakeFS()
                with patch.object(transaction,'os',self.fs),patch.object(transaction,'Path',self.fs.Path),patch.object(durable,'os',self.fs):
                    self.fs.fail=(operation,name)
                    with self.assertRaises(OSError):self.write()
                    self.assertIn('manifest.json',self.fs.entries)
                    self.fs.fail=None
                    for entry in self.fs.entries.values():
                        if not entry.closed:entry.close()

    def test_change_after_close_rejected(self):
        original=FakeFile.close
        def changed(stream):original(stream);stream.data+=b'x'
        with patch.object(FakeFile,'close',changed):
            with self.assertRaises(RuntimeError):self.write()
        self.assertIn('manifest.json',self.fs.entries)

    def test_guard_expiry_before_open(self):
        def deny():raise ValueError('lease expired')
        with self.assertRaisesRegex(ValueError,'lease expired'):
            durable.write_json_exclusive('/admitted','manifest.json',{},max_bytes=50,guard=deny)
        self.assertEqual(self.fs.events,[])


class SourceInputsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.prepared=preparer.prepare_source_inputs(guard=lambda:None)

    def test_actual_pinned_metadata_under_2MiB_cap(self):
        p=self.prepared
        self.assertEqual(len(p['cases']),6)
        self.assertEqual(len(p['previous_loader_tensors']),507)
        self.assertEqual(sum(r['bytes'] for r in p['previous_loader_tensors']),7950458880)
        self.assertTrue(all(Path(r['path']).stat().st_size <= 2*1024**2 for r in p['original_references']))
        self.assertEqual(p['stage_a_bytes'],1467113472)
        self.assertEqual([x['identity']['event'] for x in p['cases']],[1,4,7,9,11,6179])
        self.assertFalse(p['runtime_bound']);self.assertFalse(p['runtime_admitted'])
        for key in ('checkpoint','output','spec','admission','producer_exit'):self.assertNotIn(key,p)

    def test_digest_is_integrity_check_not_authentication_claim(self):
        p=deepcopy(self.prepared);p['quality_gates']['cosine_min']=.9
        with self.assertRaises(RuntimeError):preparer.validate_source_inputs(p)
        p['input_sha256']=transaction.digest({k:v for k,v in p.items() if k!='input_sha256'})
        with self.assertRaises(RuntimeError):preparer.validate_source_inputs(p)

    def test_all507_exact_BF16_byte_extents_even_if_rehashed(self):
        p=deepcopy(self.prepared)
        p['previous_loader_tensors'][0]['bytes']+=2;p['previous_loader_tensors'][1]['bytes']-=2
        p['input_sha256']=transaction.digest({k:v for k,v in p.items() if k!='input_sha256'})
        with self.assertRaisesRegex(RuntimeError,'BF16 loader tensor'):preparer.validate_source_inputs(p)

    def test_extra_runtime_fields_rejected_even_if_rehashed(self):
        p=deepcopy(self.prepared);p['admission']={'fake':True}
        p['input_sha256']=transaction.digest({k:v for k,v in p.items() if k!='input_sha256'})
        with self.assertRaisesRegex(RuntimeError,'Exact source-input schema'):preparer.validate_source_inputs(p)

    def test_dtype_and_rope_change_rejected(self):
        for change in ('dtype','rope'):
            p=deepcopy(self.prepared)
            if change=='dtype':p['previous_loader_tensors'][0]['dtype']='F32'
            else:p['expected_model']['rotary_tables'][0]['base']=1
            p['input_sha256']=transaction.digest({k:v for k,v in p.items() if k!='input_sha256'})
            with self.subTest(change=change),self.assertRaises(RuntimeError):preparer.validate_source_inputs(p)

    def test_non_json_path_rejected_before_open(self):
        with patch.object(preparer.os,'open') as opened:
            with self.assertRaises(RuntimeError):
                preparer.read_pinned_json({'path':'/synthetic/weights.bin','sha256':'0'*64},guard=lambda:None)
            opened.assert_not_called()

    def test_changed_actual_pin_rejected(self):
        ref=dict(preparer.DESIGN);ref['sha256']='0'*64
        with self.assertRaisesRegex(RuntimeError,'SHA256'):preparer.read_pinned_json(ref,guard=lambda:None)

    def test_duplicate_json_keys_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'duplicate'):preparer.unique([('x',1),('x',2)])


class DType(str):
    def __str__(self):return 'torch.'+super().__str__()


class NativeFixtureTorch(FakeTorch):
    __version__='2.14.0+cpu'
    bfloat16=DType('bfloat16')
    def __init__(self):
        super().__init__();self.default_dtype='original-f32';self.inference=False;self.settings=[]
        self.nn=SimpleNamespace(functional=SimpleNamespace(linear=lambda *a:None))
    def set_num_threads(self,n):self.settings.append(('threads',n))
    def set_num_interop_threads(self,n):self.settings.append(('interop',n))
    def set_float32_matmul_precision(self,p):self.settings.append(('precision',p))
    def get_default_dtype(self):return self.default_dtype
    def set_default_dtype(self,v):self.default_dtype=v
    def is_inference_mode_enabled(self):return self.inference
    @contextmanager
    def inference_mode(self):
        old=self.inference;self.inference=True
        try:yield
        finally:self.inference=old
    def arange(self,end,*,dtype,device):return self.tensor(list(range(end)),dtype=dtype,device=device)


class VirtualWeight:
    def __init__(self,name,shape,api,parts=None,sha=None):
        self.name,self.shape,self.dtype,self.parts,self.sha=name,tuple(shape),api.bfloat16,parts,sha
        self.device=SimpleNamespace(type='cpu');self.api=api
    def numel(self):return math.prod(self.shape)
    def __getitem__(self,selection):
        assert isinstance(selection,slice) and selection.step is None
        row=self.parts[(self.name,selection.start,selection.stop)]
        return VirtualWeight(row['name'],row['shape'],self.api,sha=row['sha256'])


class Patches:
    def __init__(self):self.saved=[]
    def setattr(self,obj,key,value):self.saved.append((obj,key,getattr(obj,key)));setattr(obj,key,value)
    def undo(self):
        for obj,key,value in reversed(self.saved):setattr(obj,key,value)
        self.saved.clear()


class NativeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.prepared=preparer.prepare_source_inputs(guard=lambda:None)

    def setUp(self):
        self.torch=NativeFixtureTorch();self.events=[];self.config_active=False
        rows=self.prepared['previous_loader_tensors'];self.weights={};shapes={};parts={}
        for row in rows:
            self.weights[row['name']]=VirtualWeight(row['name'],row['shape'],self.torch,sha=row['sha256'])
            target,begin,end=worker.mapped_parameter(row['name'],row['shape'])
            shape=[end,*row['shape'][1:]]
            if target in shapes:shape[0]=max(shape[0],shapes[target][0])
            shapes[target]=shape;parts[(target,begin,end)]=row
        self.params={n:VirtualWeight(n,s,self.torch,parts=parts) for n,s in shapes.items()}
        class Cache:pass
        caches=[]
        for key in worker.CACHE_IDS:
            c=Cache();c.prefix=str(key);c.keys=c.values=None;caches.append(c)
        rope1=SimpleNamespace(cos_sin_cache=VirtualWeight('rope',[262144,256],self.torch),base=10000)
        rope2=SimpleNamespace(cos_sin_cache=VirtualWeight('rope',[262144,256],self.torch),base=10000000)
        layers=[SimpleNamespace(mixer=SimpleNamespace(rotary_emb=rope1 if i<56 else rope2)) for i in range(72)]
        backend_modules={n:SimpleNamespace(cpu_linear=SimpleNamespace(__code__=SimpleNamespace(co_names=tuple(names))))
                         for n,names in self.prepared['expected_model']['backends'].items()}
        self.model=SimpleNamespace(backbone=SimpleNamespace(layers=layers,num_loops=2),
            parameters=lambda:list(self.params.values()),named_parameters=lambda:list(self.params.items()),
            load_weights=lambda rows:set(self.params),logits_processor=SimpleNamespace(_gather_logits=lambda x:x),
            named_modules=lambda:list(backend_modules.items()),modules=lambda:caches)
        self.initial_gather=self.model.logits_processor._gather_logits
        def construct(**kwargs):
            self.events.append('construct');self.assertEqual(self.torch.default_dtype,self.torch.bfloat16)
            self.assertTrue(self.torch.inference);self.assertTrue(self.config_active)
            return self.model
        @contextmanager
        def config_context(vc):
            old=self.config_active;self.config_active=True
            try:yield
            finally:self.config_active=old
        self.api=SimpleNamespace(torch=self.torch,monkeypatch_factory=Patches,
            upstream=SimpleNamespace(ReferenceAttention=Cache,_prepare_for_inference=lambda m:self.events.append('prepare')),
            official_source=SimpleNamespace(Attention=object(),DoryForCausalLM=construct),
            ops=SimpleNamespace(onednn_mm=lambda *a:None),parallel_state=SimpleNamespace(_TP=object()),
            VllmConfig=lambda **kw:SimpleNamespace(**kw),DeviceConfig=lambda **kw:SimpleNamespace(**kw),
            CompilationConfig=lambda **kw:SimpleNamespace(**kw),DoryConfig=lambda **kw:SimpleNamespace(**kw),
            set_current_vllm_config=config_context,extract_layer_index=lambda p:int(p))
        self.initial_op=self.api.ops.onednn_mm;self.initial_attention=self.api.official_source.Attention
        self.initial_tp=self.api.parallel_state._TP
        # In-memory synthetic future binding only, never written or used to open a checkpoint.
        self.checkpoint=dict(path='/synthetic/never-opened/model.safetensors',device=1,inode=2,
            size=worker.loader.FILE_BYTES,mtime_ns=3,ctime_ns=4)
        self.proof=dict(full_sha256=worker.loader.FULL_SHA,full_file_hash_verified=True,
            file_bytes_read=worker.loader.FILE_BYTES,weight_bytes=worker.loader.DATA_BYTES,
            tensors=deepcopy(rows),chunk_bytes=worker.loader.CHUNK_BYTES,separately_qualified_model=False,
            checkpoint_identity=self.checkpoint,config_sha256=worker.loader.CONFIG_SHA,
            header_sha256=worker.loader.HEADER_SHA,parameters=3975229440,
            physical_weight_sharing='same 507 tensors for both original loops',loader_source_sha256=preparer.LOADER_SHA,
            model_forward_executed=False,peak_memory_fit_proven=False,new_speedup_claim=False)
        self.hashes=[];self.writes=[]
        def hash_weight(t,adapter):self.hashes.append(t.name);return t.sha
        def capture(torch,official,caches,counts,prepared,root,*,guard):
            self.events.append('capture');self.assertIs(official,self.model)
            self.assertEqual(sorted(caches),list(worker.CACHE_IDS));self.assertTrue(self.torch.inference)
            self.assertTrue(self.config_active)
            self.assertEqual(prepared,self.prepared)
            return {'synthetic_metadata_fixture':True},[]
        def write(root,name,value,*,max_bytes,guard):
            self.assertFalse(self.torch.inference);self.assertFalse(self.config_active)
            self.assertEqual(self.torch.default_dtype,'original-f32')
            self.assertIs(self.api.ops.onednn_mm,self.initial_op)
            self.assertIs(self.model.logits_processor._gather_logits,self.initial_gather)
            self.writes.append((name,deepcopy(value),max_bytes))
            return {'synthetic_metadata_receipt':True,'name':name}
        self.patches=[patch.object(worker.loader,'source_contract',return_value=dict(config=self.prepared['expected_model']['config'],
                      tensor_count=507,parameters=3975229440)),
            patch.object(worker.loader,'load_checkpoint',side_effect=lambda *a,**kw:(dict(self.weights),deepcopy(self.proof))),
            patch.object(worker,'tensor_sha256',side_effect=hash_weight),patch.object(worker,'capture_cases',side_effect=capture),
            patch.object(worker,'write_json_exclusive',side_effect=write)]
        for p in self.patches:p.start()
        self.addCleanup(lambda:[p.stop() for p in reversed(self.patches)])

    def run_worker(self):return worker.run_worker(self.api,self.prepared,self.checkpoint,'/synthetic/output',guard=lambda:None)

    def assert_restored(self):
        self.assertEqual(self.torch.default_dtype,'original-f32');self.assertFalse(self.torch.inference)
        self.assertFalse(self.config_active);self.assertIs(self.api.ops.onednn_mm,self.initial_op)
        self.assertIs(self.api.official_source.Attention,self.initial_attention)
        self.assertIs(self.api.parallel_state._TP,self.initial_tp)

    def test_actual507_mapping_and_official_only_source_wiring(self):
        result=self.run_worker();self.assert_restored()
        self.assertEqual(len(self.params),399)
        self.assertEqual(sum(p.numel() for p in self.params.values()),3975229440)
        self.assertEqual(len(self.hashes),507)
        self.assertEqual(self.events,['construct','prepare','capture'])
        self.assertEqual([x[0] for x in self.writes],['stage-a-loader-proof.json','stage-a-loaded-parameters.json',
            'stage-a-model.json','stage-a-artifacts.json','stage-a-producer.json'])
        final=self.writes[-1][1]
        self.assertFalse(final['producer_process_exit_observed'])
        self.assertFalse(final['independent_saved_byte_review_complete'])
        self.assertTrue(final['requires_actual_exit_and_separate_review'])
        model=self.writes[2][1]
        self.assertFalse(model['native_source_provenance_authenticated_here'])
        self.assertFalse(model['candidate_constructed']);self.assertFalse(model['functional_linear_route_installed'])
        self.assertEqual(result['name'],'stage-a-producer.json')

    def test_real_adapter_weight_digest_tiny_strided_fixture(self):
        # Explicitly undo only the digest stub for this tiny real adapter chain.
        original=patch.object(worker,'tensor_sha256').get_original()[0]
        source=Path(worker.__file__).read_text()
        tree=ast.parse(source);node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='tensor_sha256')
        namespace={'hashlib':hashlib};exec(compile(ast.Module(body=[node],type_ignores=[]),'<digest-source>','exec'),namespace)
        tensor=patterned(self.torch,[3,4],stride=[8,1],offset=1)
        adapter=worker.NativeTensorAdapter(self.torch)
        self.assertEqual(namespace['tensor_sha256'](tensor,adapter),hashlib.sha256(independent_expected(tensor)).hexdigest())

    def test_507_proof_mismatch_stops_before_model(self):
        self.proof['tensors'][0]['sha256']='0'*64
        with self.assertRaises(RuntimeError):self.run_worker()
        self.assertEqual(self.events,[]);self.assertEqual(self.writes,[]);self.assert_restored()

    def test_extra_loader_proof_field_rejected(self):
        self.proof['invented_completed_exit']=True
        with self.assertRaises(RuntimeError):self.run_worker()
        self.assertEqual(self.writes,[]);self.assert_restored()

    def test_wrong_weight_slice_hash_rejected(self):
        with patch.object(worker,'tensor_sha256',return_value='0'*64):
            with self.assertRaisesRegex(RuntimeError,'Exact bytes'):self.run_worker()
        self.assertNotIn('capture',self.events);self.assertEqual(self.writes,[]);self.assert_restored()

    def test_model_construction_exception_restores_default_dtype(self):
        def fail(**kw):raise ValueError('construction failed')
        self.api.official_source.DoryForCausalLM=fail
        with self.assertRaisesRegex(ValueError,'construction failed'):self.run_worker()
        self.assertEqual(self.writes,[]);self.assert_restored()

    def test_missing_physical_parameter_rejected(self):
        self.params.pop(next(iter(self.params)))
        with self.assertRaisesRegex(RuntimeError,'399 physical'):self.run_worker()
        self.assertEqual(self.writes,[]);self.assert_restored()

    def test_backend_count_or_identity_change_rejected(self):
        self.model.named_modules=lambda:[]
        with self.assertRaisesRegex(RuntimeError,'182 oneDNN'):self.run_worker()
        self.assertEqual(self.writes,[]);self.assert_restored()

    def test_missing_cache_or_rope_change_rejected(self):
        self.model.backbone.layers[0].mixer.rotary_emb.base=1
        with self.assertRaisesRegex(RuntimeError,'Original RoPE'):self.run_worker()
        self.assertEqual(self.writes,[]);self.assert_restored()

    def test_capture_failure_restores_context_no_metadata(self):
        with patch.object(worker,'capture_cases',side_effect=ValueError('capture failed')):
            with self.assertRaisesRegex(ValueError,'capture failed'):self.run_worker()
        self.assertEqual(self.writes,[]);self.assert_restored()

    def test_failure_persisting_artifacts_never_writes_final_producer(self):
        original=worker.write_json_exclusive.side_effect
        def fail(root,name,value,**kw):
            if name=='stage-a-artifacts.json':raise OSError('manifest fsync failure')
            return original(root,name,value,**kw)
        with patch.object(worker,'write_json_exclusive',side_effect=fail):
            with self.assertRaisesRegex(OSError,'manifest fsync'):self.run_worker()
        self.assertEqual(len(self.writes),3);self.assert_restored()

    def test_invalid_future_checkpoint_schema_rejected_before_loader(self):
        self.checkpoint['inode']=True
        with self.assertRaisesRegex(RuntimeError,'checkpoint identity'):self.run_worker()
        worker.loader.load_checkpoint.assert_not_called()


class CaptureLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.prepared=preparer.prepare_source_inputs(guard=lambda:None)

    def setUp(self):
        self.torch=NativeFixtureTorch();self.counts=[0];self.events=[];self.fail_at=None
        self.caches={key:SimpleNamespace(keys=None,values=None) for key in worker.CACHE_IDS}
        outer=self
        class Observer:
            def __init__(self,official,caches,adapter,counts,guard):self.handles=[];self.ops=adapter;outer.observer=self
            def install(self):self.handles=[1]*148;outer.events.append('install')
            def close(self):self.handles.clear();outer.events.append('observer-close')
        class Store:
            def __init__(self,root,*,guard):outer.events.append('store-open')
            def abort(self):outer.events.append('store-abort')
        class Batch:
            def __init__(self,cases,observer,adapter,store,*,guard):self.index=0;self.cases=cases
            def collect(self,tokens,positions):
                case=self.cases[self.index]
                outer.assertEqual(tokens.tolist(),case['input_ids'])
                outer.assertEqual(positions.tolist(),list(range(case['identity']['end'])))
                outer.assertTrue(all(c.keys is None and c.values is None and c.start_pos==0 for c in outer.caches.values()))
                for c in outer.caches.values():c.keys=c.values=object()
                if self.index==outer.fail_at:raise ValueError('synthetic third forward failed')
                outer.counts[0]+=281;outer.events.append(case['identity']['event']);self.index+=1
            def finish(self):outer.events.append('batch-finish');return {'synthetic_metadata_fixture':True}
        self.patches=[patch.object(worker,'StageAObserver',Observer),patch.object(worker,'ArchiveFileStore',Store),
                      patch.object(worker,'CaptureArtifactBatch',Batch)]
        for p in self.patches:p.start()
        self.addCleanup(lambda:[p.stop() for p in reversed(self.patches)])

    def capture(self):
        return worker.capture_cases(self.torch,object(),self.caches,self.counts,self.prepared,'/synthetic',guard=lambda:None)

    def test_all_original_tokens_positions_and_cache_isolation(self):
        artifacts,isolation=self.capture()
        self.assertEqual([x for x in self.events if type(x)is int],[1,4,7,9,11,6179])
        self.assertEqual(self.counts[0],1686)
        self.assertEqual([x['rows'] for x in isolation],[2060,2060,23,127,511,31])
        self.assertEqual(self.events[-3:],['batch-finish','observer-close','store-abort'])
        self.assertTrue(all(c.keys is None and c.values is None and c.start_pos==0 for c in self.caches.values()))
        self.assertEqual(self.observer.handles,[])

    def test_failed_forward_clears_all_caches_removes_hooks_no_finish(self):
        self.fail_at=2
        with self.assertRaisesRegex(ValueError,'third forward'):self.capture()
        self.assertNotIn('batch-finish',self.events)
        self.assertEqual(self.events[-2:],['observer-close','store-abort'])
        self.assertEqual(self.observer.handles,[])
        self.assertTrue(all(c.keys is None and c.values is None for c in self.caches.values()))

    def test_existing_cache_history_rejected_before_store(self):
        self.caches[0].keys=object()
        with self.assertRaisesRegex(RuntimeError,'initially empty'):self.capture()
        self.assertEqual(self.events,[])

    def test_incomplete_cache_set_rejected(self):
        self.caches.pop(0)
        with self.assertRaises(RuntimeError):self.capture()
        self.assertEqual(self.events,[])


class StaticBoundaryTests(unittest.TestCase):
    def test_no_native_import_launch_or_reviewer_call(self):
        tree=ast.parse(Path(worker.__file__).read_text())
        imports=[];calls=[]
        for n in ast.walk(tree):
            if isinstance(n,ast.Import):imports += [x.name for x in n.names]
            if isinstance(n,ast.ImportFrom):imports.append(n.module)
            if isinstance(n,ast.Call):calls.append(ast.unparse(n.func))
        self.assertFalse(any(x.split('.')[0] in ('torch','numpy','pytest','vllm','transformers') for x in imports))
        self.assertFalse(any('DoryReferenceCandidate' in x or 'review_saved_batch' in x or 'subprocess' in x for x in calls))
        self.assertNotIn('torch',sys.modules);self.assertNotIn('numpy',sys.modules)
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='setattr'
            and len(n.args)>1 and isinstance(n.args[1],ast.Constant) and n.args[1].value=='linear' for n in ast.walk(tree)))
        mains=[n for n in tree.body if isinstance(n,ast.If)]
        self.assertTrue(any(any(isinstance(x,ast.Raise) for x in n.body) for n in mains))


if __name__=='__main__':unittest.main(verbosity=2)
