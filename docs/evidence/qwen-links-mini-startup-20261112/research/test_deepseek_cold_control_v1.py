"""Source/stdlib/mock controls. Never imports a native library or tensor."""
import ast
import copy
import hashlib
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as N
import unittest
from unittest.mock import Mock, patch

ROOT=Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
sys.path.insert(0,str(ROOT))


class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'torch','numpy','transformers','vllm','torch_neuronx','neuronx_distributed','neuronx_distributed_inference'}:
            raise AssertionError('native import forbidden: '+fullname)


sys.meta_path.insert(0,Blocker())
import deepseek_cold_control_entry_v1 as entry
import verify_deepseek_cold_control_v1 as verifier


def functions(name):
    return {n.name:n for n in ast.parse((ROOT/name).read_text()).body if isinstance(n,ast.FunctionDef)}


class Source(unittest.TestCase):
    def test_exact_mechanical_derivations(self):
        d=json.loads((ROOT/'research/deepseek-cold-control-mechanical-delta-20261112-v1.json').read_text())
        for row in d['files']:
            original=Path(row['source']).read_bytes(); self.assertEqual(hashlib.sha256(original).hexdigest(),row['source_sha256'])
            s=original.decode()
            if row.get('source_prefix_before_function'):
                node=functions(Path(row['source']).name)[row['source_prefix_before_function']]
                s='\n'.join(s.splitlines()[:node.lineno-1])+'\n'
            for before,after in row['replacements']: self.assertIn(before,s); s=s.replace(before,after)
            self.assertEqual(s,Path(row['result']).read_text())

    def test_every_original_native_and_device_loop_unchanged(self):
        a=functions('validate_deepseek_cold_kv_v2.py');z=functions('validate_deepseek_cold_control_v1.py')
        for name in ('main','reference'):
            loops=lambda n:[ast.dump(v,include_attributes=False) for v in ast.walk(n) if isinstance(v,ast.For)]
            self.assertEqual(loops(a[name]),loops(z[name]))

    def test_all_KV_writer_methods_unchanged_except_paths(self):
        methods=[]
        for name in ('deepseek_cold_kv_capture_v2.py','deepseek_cold_control_capture_v1.py'):
            c=next(n for n in ast.parse((ROOT/name).read_text()).body if isinstance(n,ast.ClassDef))
            methods.append({n.name:ast.dump(n,include_attributes=False) for n in c.body if isinstance(n,ast.FunctionDef) and n.name!='__init__'})
        self.assertEqual(methods[0],methods[1]);self.assertEqual(set(methods[0]),{'write','native','device','finish'})

    def test_no_model_loader_compiler_or_launch_added(self):
        t=ast.parse((ROOT/'validate_deepseek_cold_control_v1.py').read_text())
        self.assertNotIn('reference_entry',functions('validate_deepseek_cold_control_v1.py'))
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and
            n.func.attr in ('compile','from_pretrained') for n in ast.walk(t)))
        self.assertNotIn('model.load(', (ROOT/'validate_deepseek_cold_control_v1.py').read_text())

    def test_original_device_metric_validator_retains_all14_failures(self):
        source=functions('deepseek_cold_kv_protocol_v2.py')['check_device_metadata']
        text=ast.unparse(source)
        self.assertIn('[33, 35, 36, 38, 40, 42, 45, 47, 48, 50, 60, 62, 68, 70]',text)
        self.assertIn("now['rows'] == prior['rows']",text)
        self.assertIn('check_device_metadata', (ROOT/'deepseek_cold_control_entry_v1.py').read_text())


class Entry(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);p=Path(self.tmp.name)
        self.dest=p/'meta';self.dest.mkdir();self.raw=p/'raw';self.raw.mkdir();(self.dest/'cold-reference').mkdir()
        (self.dest/'cold-reference/cold-reference.json').write_text('{}')
        self.guard=Mock(); self.case={'original':'case'};self.old={'old':'metadata'};self.model=object()
        self.main=Mock(side_effect=SystemExit('Strict cold-bucket numerical/state gate failed'))
        p=ModuleType('deepseek_cold_kv_protocol_v2');p.expected_case=lambda:self.case
        p.check_device_metadata=Mock(return_value={'passed':True});p.check_copy=Mock();p.check_lifecycle=Mock()
        p.read=lambda _:dict(state_tensors_changed_by_warmup=144)
        driver=ModuleType('validate_deepseek_cold_control_v1');driver.main=self.main
        self.p=p;self.modules={p.__name__:p,driver.__name__:driver}

    def run_device(self):
        with patch.dict(sys.modules,self.modules):
            return entry.device(self.model,self.case,self.dest,self.raw,self.old,self.guard)

    def test_original_expected_failure_followed_by_full_metadata_and_lifecycle_checks(self):
        result=self.run_device();self.main.assert_called_once_with(self.model,self.case,self.dest,self.dest/'cold-reference',self.raw,self.guard)
        self.p.check_device_metadata.assert_called_once();self.assertIs(self.p.check_device_metadata.call_args.args[1],self.old)
        self.p.check_copy.assert_called_once();self.p.check_lifecycle.assert_called_once()
        self.assertTrue(result['original_failed_gate_preserved']);self.assertTrue(result['original_raw_reproduction_pending'])
        self.assertFalse(result['full_model_qualified']);self.assertFalse(result['new_speedup_claim'])

    def test_changed_success_or_other_failure_not_accepted(self):
        for outcome in (None,SystemExit('another error'),RuntimeError('unexpected')):
            self.main.side_effect=outcome
            with self.subTest(outcome=str(outcome)),self.assertRaises((ValueError,RuntimeError)):self.run_device()
        self.p.check_device_metadata.assert_not_called()

    def test_full_metadata_rejection_propagates(self):
        self.p.check_device_metadata.side_effect=ValueError('failed14 changed')
        with self.assertRaisesRegex(ValueError,'failed14 changed'):self.run_device()

    def test_guard_precedes_import_and_output(self):
        guard=Mock(side_effect=RuntimeError('not admitted'))
        with self.assertRaisesRegex(RuntimeError,'not admitted'):
            entry.device(None,None,'/absent','/absent',None,guard)

    def test_old_nested_and_nonfresh_outputs_rejected(self):
        with self.assertRaisesRegex(ValueError,'independent'):
            entry.paths('/tmp/deepseek-cold-kv-diagnostic-v2',self.raw,'device',self.guard)
        with self.assertRaisesRegex(ValueError,'disjoint'):
            entry.paths(self.dest,self.dest,'device',self.guard)
        (self.dest/'cold-buckets.pt').write_text('mock only')
        with self.assertRaisesRegex(ValueError,'fresh'):self.run_device()


class RawComparisons(unittest.TestCase):
    def fixture(self,backend):
        prefixes=(32,128,256,512,1024,2048,2052)
        frame_fn=functions('deepseek_cold_kv_contract_v2.py')['frames'];env={'PREFIXES':prefixes}
        exec(compile(ast.Module(body=[frame_fn],type_ignores=[]),'<original-frames>','exec'),env)
        frames=env['frames'](backend);names=[f'layers.{n}.{s}' for n in range(36) for s in ('key','value')]
        contract=ModuleType('deepseek_cold_kv_contract_v2');contract.frames=env['frames'];contract.stages=lambda:names
        contract.check_manifest=Mock();contract.check_frame=Mock();contract.check_record=Mock()
        codec=ModuleType('deepseek_cold_kv_codec_v2');codec.decode=lambda row,value,previous:value;codec.same=lambda a,b:a==b
        torch=ModuleType('torch');records={};payloads={}; manifests={}
        for root in ('old','new'):
            entries=[]
            for i,frame in enumerate(frames):
                path=f'/{root}/{backend}/{i}.json';payload=f'/{root}/{backend}/{i}.mock'
                rows=[dict(stage=name,full_sha256=f'{i}:{name}') for name in names]
                records[path]=dict(frame=frame,rows=rows,payload=payload,payload_file_sha256='mockhash')
                payloads[payload]={name:f'{i}:{name}:full-history-sentinel' for name in names}
                entries.append(dict(frame=frame,path=path,sha256='mockhash'))
            manifests[f'/{root}/cache-captures/{backend}/manifest.json']=dict(complete=True,frames=entries)
        torch.load=lambda path,**kwargs:payloads[str(path)]
        return contract,codec,torch,records,payloads,manifests

    def compare(self,backend,damage=None):
        c,codec,torch,records,payloads,manifests=self.fixture(backend)
        if damage:damage(records,payloads,manifests)
        with patch.dict(sys.modules,{c.__name__:c,codec.__name__:codec,'torch':torch}),\
             patch.object(verifier,'pinned_read',side_effect=lambda path,pins:manifests[str(path)]),\
             patch.object(verifier,'digest',return_value='mockhash'),\
             patch.object(verifier,'read',side_effect=lambda path:records[str(path)]):
            return verifier._compare_cache_backend(Path('/old'),Path('/new'),backend,{})

    def test_all9072_native_and_device_full_KV_pairs_compared(self):
        self.assertEqual(self.compare('native'),3024);self.assertEqual(self.compare('device'),6048)

    def test_omitted_or_reordered_cache_frames_rejected(self):
        for reverse in (True,False):
            def damage(r,p,m):
                rows=m['/new/cache-captures/device/manifest.json']['frames']
                if reverse:rows.reverse()
                else:rows.pop()
            with self.subTest(reverse=reverse),self.assertRaisesRegex(ValueError,'omitted/reordered'):self.compare('device',damage)

    def test_last_layer_last_frame_history_difference_rejected(self):
        def damage(r,p,m):p['/new/device/83.mock']['layers.35.value']='changed old prefix sentinel'
        with self.assertRaisesRegex(ValueError,'full KV bytes'):self.compare('device',damage)

    def test_missing_raw_admission_rejected_before_any_deserialization(self):
        torch=ModuleType('torch');torch.load=Mock(side_effect=AssertionError('must not load'));torch.float32='mock'
        codec=ModuleType('deepseek_cold_kv_codec_v2');codec.same=Mock()
        protocol=ModuleType('deepseek_cold_kv_protocol_v2')
        for name in ('check_copy','check_lifecycle','check_device_metadata'):setattr(protocol,name,Mock())
        with patch.dict(sys.modules,{'torch':torch,codec.__name__:codec,protocol.__name__:protocol}):
            with self.assertRaisesRegex(ValueError,'missing caller admission'):
                verifier.verify('/old','/new',admitted_sha256={},execution_guard=lambda:None)
        torch.load.assert_not_called()


class OriginalMetadata(unittest.TestCase):
    def setUp(self):
        path=ROOT/'deepseek-cold-kv-diagnostic-v2/cold-buckets.json'
        self.assertLess(path.stat().st_size,2*2**20)
        self.original=json.loads(path.read_text())
        scope={}
        frame=functions('qwen3_family_prior_mask_protocol_v1.py')['frame_pass']
        exec(compile(ast.Module(body=[frame],type_ignores=[]),'<original-frame-pass>','exec'),scope)
        old=N(CRITERIA=self.original['criteria'],BUCKETS=(128,256,512,1024,2048,4096),diagnostic=N(frame_pass=scope['frame_pass']))
        scope=dict(c=N(old=old,PREFIXES=(32,128,256,512,1024,2048,2052)),expected_case=lambda:self.original['case'])
        fn=functions('deepseek_cold_kv_protocol_v2.py')['check_device_metadata']
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<original-full-metadata-check>','exec'),scope)
        self.check=scope['check_device_metadata']

    def test_saved_original84_metadata_satisfies_unmodified_validator(self):
        out=self.check(copy.deepcopy(self.original),self.original)
        self.assertEqual(out['failed_indices'],[33,35,36,38,40,42,45,47,48,50,60,62,68,70])
        self.assertTrue(out['all84_original_rows_and_metrics_exact'])

    def test_failure_row_or_slot_state_omission_rejected(self):
        for damage in ('failure','state','frame'):
            z=copy.deepcopy(self.original)
            if damage=='failure':z['experiments'][2]['rows'][9]['metrics']['cosine']=1.0
            elif damage=='state':z['experiments'][0]['initial_zero_state_checks'].pop()
            else:z['experiments'][0]['rows'].pop()
            with self.subTest(damage=damage),self.assertRaises(AssertionError):self.check(z,self.original)


if __name__=='__main__':unittest.main(verbosity=2)
