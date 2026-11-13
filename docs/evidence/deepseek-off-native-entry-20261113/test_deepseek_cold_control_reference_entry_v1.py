"""In-memory native loader tests executing the frozen complete reference loop."""
import ast
import copy
from contextlib import ExitStack, nullcontext, redirect_stdout
import hashlib
import io
import json
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as N
import unittest
from unittest.mock import Mock, patch

R = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(R), str(R/'research')]
import deepseek_cold_control_reference_entry_v1 as entry
import deepseek_cold_control_entry_v1 as control
import deepseek_cold_kv_contract_v2 as contract
import deepseek_lifecycle_full_protocol_v2 as protocol_module
import validate_deepseek_cold_control_v1 as driver

SOURCE_MARKER = b'Synthetic native-source identity, no native implementation.'
REAL_SHA = hashlib.sha256
REAL_READ_BYTES = Path.read_bytes
REAL_READ_TEXT = Path.read_text


class Tensor:
    def __init__(self, values=None, shape=None):
        self.values = values
        self.shape = tuple(shape) if shape is not None else self.infer(values)
    @staticmethod
    def infer(values):
        return (len(values), *Tensor.infer(values[0])) if isinstance(values, (list, tuple)) and values else ()
    def __getitem__(self, index):
        if index is None:
            return Tensor(shape=(1, *self.shape))
        return Tensor(shape=(151936,))
    def detach(self): return self
    def float(self): return self
    def cpu(self): return self
    def clone(self): return Tensor(self.values, self.shape)


class Fixture:
    def __init__(self):
        self.dest = Path('/synthetic/deepseek-off-native/meta')
        self.ref = self.dest/'cold-reference'
        self.raw = Path('/synthetic/deepseek-off-native/raw')
        self.native_source = Path('/synthetic/native_source.py')
        self.protocol = dict(native_source_sha256=entry.NATIVE_SOURCE_SHA256, revision='frozen-revision',
            criteria=copy.deepcopy(protocol_module.CRITERIA),
            fixtures=[dict(input_ids=[fixture*10000+i for i in range(3072)]) for fixture in range(2)])
        self.config = dict(rope_scaling=dict(rope_type='yarn', factor=4., attn_factor=1.25), marker='original')
        self.files = {str(entry.MODEL_PATH/'config.json'):json.dumps(self.config).encode()}
        self.dirs = {str(self.dest), str(self.raw)}
        self.symlinks = set(); self.extra_existing = set(); self.events = []
        self.count = entry.PARAMETERS; self.tied = False; self.placement = 'cpu'; self.dtype = 'torch.bfloat16'
        self.loading = {}; self.load_error = None; self.loop_error = None; self.capture_error = None
        self.revoked = False; self.revoke_at = None; self.guard_error = RuntimeError('ownership lost')
        self.guard_count = 0; self.fail_guard = None; self.forward_calls = []
        self.native_calls = 0; self.load_calls = []; self.captured_frames = []; self.kv_records = []
        self.finished = False; self.damage_manifest = None; self.damage_report = None
        self.leave_trace = False; self.trace = None; self.fail_assignment = False
        self.protocol_change = False; self.build_count = 0; self.native_hash = entry.NATIVE_SOURCE_SHA256
        self.already_native = None; self.own_forward = False; self.original_override = None
        owner = self
        class Model:
            @classmethod
            def from_pretrained(cls, path, **kwargs):
                owner.events.append('load'); owner.load_calls.append((path, kwargs))
                if owner.load_error: raise owner.load_error
                if owner.revoke_at == 'load': owner.revoked = True
                model = cls(); owner.model = model
                if owner.own_forward:
                    model.forward = model.forward; owner.original_override = model.forward
                return model, owner.loading
            def __setattr__(self, name, value):
                object.__setattr__(self, name, value)
                if name == 'forward' and owner.fail_assignment:
                    owner.fail_assignment = False
                    raise RuntimeError('partially installed forward')
            def __init__(self):
                self.training = True; self.dtype = owner.dtype
                self.lm_head = N(weight=N(data_ptr=lambda:2))
                self.model = N(embed_tokens=N(weight=N(data_ptr=lambda:2 if owner.tied else 1)))
            def parameters(self):
                return [N(numel=lambda:owner.count, device=N(type=owner.placement))]
            def eval(self): self.training = False; owner.events.append('eval'); return self
            def __call__(self, **kwargs): return self.forward(**kwargs)
            def forward(self, **kwargs):
                owner.events.append('forward'); owner.forward_calls.append(kwargs)
                if owner.loop_error: raise owner.loop_error
                if owner.revoke_at == 'forward': owner.revoked = True
                length = kwargs['attention_mask'].shape[1]
                return N(logits=Tensor(shape=(1,1,151936)), past_key_values=N(get_seq_length=lambda:length))
        self.cls = Model; self.model = None
        self.torch = ModuleType('torch'); self.torch.bfloat16 = 'torch.bfloat16'; self.torch.long = 'torch.int64'
        self.torch.set_num_threads = lambda n:self.events.append(('threads',n))
        self.torch.tensor = lambda values:Tensor(values)
        self.torch.ones = lambda shape, **kwargs:Tensor(shape=shape)
        self.torch.arange = lambda n:Tensor(list(range(n)))
        self.torch.stack = lambda values:Tensor(shape=(len(values),151936))
        self.torch.isfinite = lambda tensor:N(all=lambda:True)
        self.torch.inference_mode = nullcontext
        self.torch.save = self.save
        self.hf = N(__file__=str(self.native_source), Qwen3ForCausalLM=Model)
        self.controls = ModuleType('qwen3_controls_reference'); self.controls.hf = self.hf
        # Use the original canonical_config body with a fake config class.
        tree = ast.parse(REAL_READ_TEXT(R/'qwen3_controls_reference.py'))
        fn = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'canonical_config')
        env = dict(copy=copy, math=math, Qwen3Config=N(from_dict=lambda d:N(raw=d)))
        exec(compile(ast.Module(body=[fn],type_ignores=[]), '<frozen-canonical-config>', 'exec'), env)
        def canonical(value):
            self.events.append('canonical-config')
            result = env['canonical_config'](value)
            if self.revoke_at == 'config': self.revoked = True
            return result
        self.controls.canonical_config = canonical
        self.capture_module = ModuleType('deepseek_cold_control_capture_v1')
        class Capture:
            def __init__(self, backend, destination, data_root):
                assert backend == 'native' and destination == owner.dest and data_root == owner.raw
                owner.events.append('capture-init')
                if owner.capture_error: raise owner.capture_error
                owner.dirs.update(map(str,(owner.dest/'cache-captures/native',owner.raw/'native')))
            def native(self, cache, prefix, fixture, step, ids):
                frame = dict(prefix=prefix,fixture=fixture,step=step)
                assert frame == contract.frames('native')[len(owner.captured_frames)]
                assert cache.get_seq_length() == prefix+step
                expected_ids = owner.protocol['fixtures'][fixture]['input_ids']
                assert ids == (expected_ids[:prefix] if step == 0 else [expected_ids[prefix+step-1]])
                owner.captured_frames.append(frame)
                owner.kv_records.extend((frame,stage) for stage in contract.stages())
            def finish(self):
                owner.finished = True
                value = dict(complete=True,backend='native',scope=contract.scope(),frames=[
                    dict(frame=f,key=contract.key(f,'native'),path='/synthetic/frame-'+str(i)+'.json',sha256='c'*64)
                    for i,f in enumerate(owner.captured_frames)],full_frame_count=42,cache_records=3024,
                    compiled_model_unchanged=True,full_model_qualified=False,new_speedup_claim=False)
                if owner.damage_manifest: owner.damage_manifest(value)
                owner.files[str(owner.dest/'cache-captures/native/manifest.json')] = json.dumps(value).encode()
                if owner.revoke_at == 'finish': owner.revoked = True
                if owner.leave_trace: owner.trace = object()
                return value
        self.capture_module.Capture = Capture
        self.modules = {m.__name__:m for m in (self.torch,self.controls,self.capture_module)}
    def native(self):
        self.events.append('native-imports'); self.native_calls += 1
        sys.modules.update(self.modules)
        return self.torch, self.hf, self.controls.canonical_config
    def guard(self):
        self.events.append('guard'); self.guard_count += 1
        if self.revoked or self.guard_count == self.fail_guard: raise self.guard_error
    def build(self):
        self.build_count += 1
        value = copy.deepcopy(self.protocol)
        if self.protocol_change and self.build_count > 1: value['native_source_sha256'] = '0'*64
        return value
    def save(self, rows, path):
        self.saved_rows = rows
        self.files[str(path)] = ('synthetic-full-logits:'+str(len(rows))).encode()
    def read_bytes(self, path):
        if path == self.native_source: return SOURCE_MARKER
        if str(path) in self.files: return self.files[str(path)]
        if path.suffix == '.py' and path.is_relative_to(R): return REAL_READ_BYTES(path)
        raise AssertionError('Unexpected real input read: '+str(path))
    def read_text(self, path, *args, **kwargs): return self.read_bytes(path).decode()
    def write_text(self, path, text, *args, **kwargs):
        if path == self.ref/'cold-reference.json' and self.damage_report:
            value=json.loads(text); self.damage_report(value); text=json.dumps(value)
        self.files[str(path)] = text.encode(); return len(text)
    def mkdir(self, path, *args, **kwargs):
        if str(path) in self.dirs: raise FileExistsError(str(path))
        self.dirs.add(str(path))
    def digest(self, data=b''):
        return N(hexdigest=lambda:self.native_hash) if data == SOURCE_MARKER else REAL_SHA(data)
    def run(self):
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules,{}))
            if self.already_native: sys.modules[self.already_native] = ModuleType(self.already_native)
            for obj,name,value in (
                (entry,'_native_dependencies',self.native),(entry.sys,'gettrace',lambda:self.trace),
                (protocol_module,'build',self.build),(protocol_module,'read',lambda p:json.loads(self.read_text(p))),
                (hashlib,'sha256',self.digest),(Path,'resolve',lambda p:p),
                (Path,'is_dir',lambda p:str(p) in self.dirs),(Path,'is_symlink',lambda p:str(p) in self.symlinks),
                (Path,'exists',lambda p:str(p) in self.dirs or str(p) in self.files or str(p) in self.extra_existing),
                (Path,'read_bytes',lambda p:self.read_bytes(p)),
                (Path,'read_text',lambda p,*a,**kw:self.read_text(p,*a,**kw)),
                (Path,'write_text',lambda p,*a,**kw:self.write_text(p,*a,**kw)),
                (Path,'mkdir',lambda p,*a,**kw:self.mkdir(p,*a,**kw))):
                stack.enter_context(patch.object(obj,name,value))
            self.control_call=stack.enter_context(patch.object(control,'reference',wraps=control.reference))
            self.driver_call=stack.enter_context(patch.object(driver,'reference',wraps=driver.reference))
            with redirect_stdout(io.StringIO()):
                return entry.run(self.dest,self.ref,self.raw,self.guard)
    def restored(self, case):
        if self.model is not None:
            case.assertEqual('forward' in vars(self.model),self.own_forward)
            if self.own_forward: case.assertIs(self.model.forward,self.original_override)


class NativeEntry(unittest.TestCase):
    def test_original_leaf_and_scientific_loop_execute_all42_calls_and3024_records(self):
        f=Fixture();out=f.run()
        f.control_call.assert_called_once();f.driver_call.assert_called_once()
        self.assertEqual(len(f.forward_calls),42);self.assertEqual(f.captured_frames,contract.frames('native'))
        self.assertEqual(len(f.kv_records),3024);self.assertTrue(f.finished)
        self.assertEqual(len(f.saved_rows),14);self.assertEqual(f.build_count,2)
        self.assertEqual(out['native_frames'],42);self.assertEqual(out['full_native_KV_tensors'],3024)
        self.assertTrue(out['original_raw_reproduction_pending']);self.assertTrue(out['forward_attribute_restored'])
        for k in ('full_model_qualified','neutrality_proven','device_execution_performed','new_speedup_claim'):
            self.assertFalse(out[k])
        f.restored(self)

    def test_exact_original_loader_and_yarn_config_transform(self):
        f=Fixture();f.run();self.assertEqual(len(f.load_calls),1)
        path,args=f.load_calls[0];self.assertEqual(path,str(entry.MODEL_PATH))
        cfg=args.pop('config')
        self.assertEqual(args,dict(dtype='torch.bfloat16',attn_implementation='eager',local_files_only=True,output_loading_info=True))
        self.assertEqual(cfg.raw['rope_scaling'],dict(rope_type='yarn',factor=4.,attention_factor=(1.+.1*math.log(4.))*1.25))
        self.assertEqual(f.config['rope_scaling']['attn_factor'],1.25)
        self.assertNotIn('attention_factor',f.config['rope_scaling'])
        self.assertEqual([x for x in f.events if isinstance(x,tuple) and x[0]=='threads'],[('threads',2),('threads',2)])

    def test_guard_before_native_import_and_before_checkpoint_load(self):
        for count in (1,2,3,4):
            f=Fixture();f.fail_guard=count
            with self.subTest(count=count),self.assertRaisesRegex(RuntimeError,'ownership lost'):f.run()
            self.assertEqual(len(f.load_calls),0)
            if count<=2:self.assertEqual(f.native_calls,0)
            self.assertNotIn(str(f.ref),f.dirs)

    def test_preimported_native_packages_and_trace_rejected(self):
        for module in entry.NATIVE_MODULES:
            f=Fixture();f.already_native=module+'.synthetic_child'
            with self.subTest(module=module),self.assertRaisesRegex(ValueError,'Fresh original native process'):f.run()
            self.assertEqual(f.native_calls,0)
        f=Fixture();f.trace=object()
        with self.assertRaisesRegex(ValueError,'Fresh original native process'):f.run()
        self.assertEqual(f.native_calls,0)

    def test_missing_raising_guard_is_rejected_before_import(self):
        with self.assertRaisesRegex(ValueError,'execution guard'):
            entry.run('/synthetic/meta','/synthetic/meta/cold-reference','/synthetic/raw',None)

    def test_original_source_pin_and_protocol_recheck_reject_change(self):
        for kind in ('native','first_protocol','second_protocol'):
            f=Fixture()
            if kind=='native':f.native_hash='0'*64
            elif kind=='first_protocol':f.protocol['native_source_sha256']='0'*64
            else:f.protocol_change=True
            with self.subTest(kind=kind),self.assertRaisesRegex(ValueError,'native source'):f.run()
            self.assertEqual(len(f.forward_calls),0);f.restored(self)

    def test_incomplete_checkpoint_load_and_loader_failure_propagate(self):
        for key in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'):
            f=Fixture();f.loading={key:['bad']}
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'checkpoint load'):f.run()
            self.assertEqual(len(f.forward_calls),0)
        f=Fixture();f.load_error=RuntimeError('loader failure')
        with self.assertRaisesRegex(RuntimeError,'loader failure'):f.run()

    def test_original_parameter_count_untied_cpu_and_bf16_checks(self):
        for attr,value,reason in [('count',1,'parameter count'),('tied',True,'untied'),
                                  ('placement','cuda','CPU parameter'),('dtype','torch.float32','BF16 eval')]:
            f=Fixture();setattr(f,attr,value)
            with self.subTest(attr=attr),self.assertRaisesRegex(ValueError,reason):f.run()
            self.assertEqual(len(f.forward_calls),0)

    def test_guard_revocation_at_config_load_forward_or_finish_propagates(self):
        for point in ('config','load','forward','finish'):
            f=Fixture();f.revoke_at=point
            with self.subTest(point=point),self.assertRaisesRegex(RuntimeError,'ownership lost'):f.run()
            f.restored(self)
            if point=='config':self.assertEqual(len(f.load_calls),0)
            if point in ('config','load'):self.assertEqual(len(f.forward_calls),0)

    def test_matching_device_failure_message_is_not_accepted_on_native_path(self):
        f=Fixture();f.revoke_at='forward';f.guard_error=SystemExit('Strict cold-bucket numerical/state gate failed')
        with self.assertRaises(SystemExit):f.run()
        f.restored(self)

    def test_driver_and_partial_capture_failure_restore_inherited_forward(self):
        for attr in ('loop_error','capture_error'):
            f=Fixture();setattr(f,attr,RuntimeError('original failure'))
            with self.subTest(attr=attr),self.assertRaisesRegex(RuntimeError,'original failure'):f.run()
            f.restored(self)

    def test_own_forward_restored_on_success_and_failure(self):
        for fail in (False,True):
            f=Fixture();f.own_forward=True
            if fail:
                f.loop_error=RuntimeError('original failure')
                with self.assertRaisesRegex(RuntimeError,'original failure'):f.run()
            else:f.run()
            f.restored(self)

    def test_partial_forward_assignment_restored(self):
        f=Fixture();f.fail_assignment=True
        with self.assertRaisesRegex(RuntimeError,'partially installed'):f.run()
        f.restored(self);self.assertEqual(len(f.forward_calls),0)

    def test_full_native_manifest_rejects_missing_reordered_or_wrong_last_frame(self):
        changes=[lambda v:v['frames'].pop(),lambda v:v['frames'].reverse(),
                 lambda v:v['frames'][-1]['frame'].update(step=1),lambda v:v.update(cache_records=3023),
                 lambda v:v.update(full_model_qualified=True)]
        for change in changes:
            f=Fixture();f.damage_manifest=change
            with self.subTest(change=change),self.assertRaises(AssertionError):f.run()
            f.restored(self)

    def test_native_report_requires_complete_finite_source_revision_criteria_and42(self):
        changes=[lambda v:v.update(complete=False),lambda v:v.update(all_finite=False),
                 lambda v:v.update(frames=41),lambda v:v.update(error='partial'),
                 lambda v:v.update(revision='changed'),lambda v:v['criteria'].update(kl_max_nats=1),
                 lambda v:v.update(source_sha256='0'*64),lambda v:v.update(tensor_sha256='not-a-hash'),
                 lambda v:v['prefixes'].pop()]
        for change in changes:
            f=Fixture();f.damage_report=change
            with self.subTest(change=change),self.assertRaises(ValueError):f.run()
            f.restored(self)

    def test_unexpected_residual_trace_rejected_after_restoration(self):
        f=Fixture();f.leave_trace=True
        with self.assertRaisesRegex(ValueError,'residual native observer trace'):f.run()
        f.restored(self)

    def test_existing_native_outputs_rejected_before_native_import(self):
        for tail,root in [('cold-reference','dest'),('cache-captures/native','dest'),('native','raw')]:
            f=Fixture();f.extra_existing.add(str(getattr(f,root)/tail))
            with self.subTest(tail=tail),self.assertRaisesRegex(ValueError,'Fresh complete'):f.run()
            self.assertEqual(f.native_calls,0)

    def test_wrong_reference_overlapping_model_raw_and_old_namespace_rejected(self):
        for kind in ('reference','raw','model','old'):
            f=Fixture()
            if kind=='reference':f.ref=f.dest/'other-reference'
            elif kind=='raw':f.raw=f.dest/'raw';f.dirs.add(str(f.raw))
            elif kind=='model':f.dest=entry.MODEL_PATH/'metadata';f.ref=f.dest/'cold-reference';f.dirs.add(str(f.dest))
            else:f.dest=Path('/synthetic/deepseek-cold-kv-diagnostic-v2/meta');f.ref=f.dest/'cold-reference';f.dirs.add(str(f.dest))
            with self.subTest(kind=kind),self.assertRaises(ValueError):f.run()
            self.assertEqual(f.native_calls,0)

    def test_symlink_outputs_and_ancestors_rejected(self):
        for path in ('ref','dest','parent'):
            f=Fixture();f.symlinks.add(str(f.ref if path=='ref' else f.dest if path=='dest' else f.dest.parent))
            with self.subTest(path=path),self.assertRaisesRegex(ValueError,'Ordinary independent'):f.run()
            self.assertEqual(f.native_calls,0)

    def test_frozen_scientific_sources_and_no_new_scientific_loop(self):
        expected={
            'deepseek_cold_control_entry_v1.py':'73eb3ff32f34abe8e52d074bf4cf6a62fa32510b2de24d4022dc7b712f60a96c',
            'validate_deepseek_cold_control_v1.py':'dae4e897d87d003144f31841231a4cfa57e1917db31ccf53251bf155ff3140f4',
            'deepseek_cold_control_capture_v1.py':'1a7170ca3993b711573f8bd8357cbfa0eeba8ed82edd82ad6f753ae7dc282a92',
            'deepseek_cold_boundary_reference_entry_v1.py':'c7ec5dc4f5637e77f6c0ee6004d2315c6984eb35398905cc79587d8f800b76f8',
            'qwen3_controls_reference.py':'437a54e1035b33e45a87953d47980b33eafe287ff09e5857555cf34b0f666170',
            'deepseek_cold_kv_contract_v2.py':'2bea0aa5f027b52c4ececee90cfee53065ecaf633263262dea1a8df3468d7d07',
            'deepseek_lifecycle_full_protocol_v2.py':'05d63e0ca2d87479ad542602b581a8b0dc9fb65725bf4f9237f51ddc8eb8e269'}
        for name,digest in expected.items():self.assertEqual(REAL_SHA(REAL_READ_BYTES(R/name)).hexdigest(),digest,name)
        tree=ast.parse(REAL_READ_TEXT(R/'deepseek_cold_control_reference_entry_v1.py'))
        run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        self.assertFalse(any(isinstance(n,(ast.For,ast.While))for n in ast.walk(run)))
        for name in ('compile','unlink','replace','rmtree','kill','flock','load_state_dict','reset'):
            self.assertFalse(any(isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr==name for n in ast.walk(tree)),name)


if __name__ == '__main__': unittest.main(verbosity=2)
