"""Synthetic coordination/output checks. Native model and math are mocked."""
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from contextlib import ExitStack
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research import minicpm5_layer1_own_history_cpu_driver_v1 as d


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='mini-own-history-driver-synthetic-')
        self.root=Path(self.tmp.name);self.rank=0;self.guards=0
    def tearDown(self):self.tmp.cleanup()
    def guard(self):self.guards+=1
    def context(self,rank=0):
        return SimpleNamespace(_sources_checked=True,permit=dict(rank=rank),guard=lambda role:self.guard(),
            plan=dict(output_root=str(self.root),source_sha256={str(Path(m.__file__).resolve()):hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                for m in (d,d.stream,d.shards,d.native,d.constructor)}))
    def torch(self,rank=0):
        return SimpleNamespace(is_inference_mode_enabled=lambda:True,get_num_threads=lambda:1,
            distributed=SimpleNamespace(is_initialized=lambda:True,get_world_size=lambda:2,get_rank=lambda:rank,get_backend=lambda:'gloo'))
    def items(self,rank=0):
        items=[]
        for ai,arm in enumerate(d.stream.raw.ARMS):
            for ci in d.stream.boundary.DECODE_CALLS:
                raw=bytes([ci+ai])*16384
                case=dict(rank=rank,arm=arm,call=dict(index=ci),bucket=2048 if ci<7 else 128,
                    original_call=dict(path='/synthetic/original.json',sha256='a'*64))
                items.append(dict(case=case,normalized_raw=bytes(16384),cache_raw={},expected_attention_raw=raw,
                    audited_call=dict(payload_sha256='b'*64)))
        return items
    def native(self,torch,model,case,normalized,caches,expected,hashes,guard):
        guard()
        return dict(complete=True,rank=case['rank'],arm=case['arm'],call=case['call'],
            all_four_rows_exact=True,normalized_original_layout=True,full_input_cache_unchanged=True,
            original_official_weights_unchanged=True,paired_history_intervention_performed=False,
            full_model_qualified=False,new_speedup_claim=False,attention_output_sha256=hashlib.sha256(expected).hexdigest()),expected
    def run_fake(self,rank=0,items=None,native=None,context=None):
        calls=self.items(rank)if items is None else items
        class FakeStream:
            def __init__(self,*args):pass
            def __iter__(self):yield from calls
            def finish(self):return dict(complete=True,synthetic=True)
        with ExitStack()as stack:
            stack.enter_context(patch.object(d.stream,'RankStream',FakeStream))
            stack.enter_context(patch.object(d.constructor,'build',return_value=(object(),dict(synthetic=True))))
            stack.enter_context(patch.object(d.constructor,'check_load_metadata',return_value=True))
            stack.enter_context(patch.object(d.constructor,'select_bucket',side_effect=lambda model,bucket:dict(bucket=bucket)))
            stack.enter_context(patch.object(d.shards,'verify_model',return_value=dict(complete=True,rank=rank,all_four_projection_shards_raw_exact=True,hashes={})))
            stack.enter_context(patch.object(d.native,'replay',side_effect=native or self.native))
            return d.run_rank(context or self.context(rank),self.torch(rank),{},dict(rank=rank),{},lambda *a:None,self.guard)
    def test_all36_calls_both_ranks_save_complete_four_row_outputs(self):
        for rank in (0,1):
            result=self.run_fake(rank)
            self.assertEqual((result['replay_calls'],result['physical_rows'],result['raw_output_bytes']),(18,72,294912))
            self.assertFalse(result['full_model_qualified']);self.assertFalse(result['attention_causality_proven'])
            self.assertEqual([(x['result']['arm'],x['result']['call']['index'])for x in result['records']],
                [(a,c)for a in d.stream.raw.ARMS for c in d.stream.boundary.DECODE_CALLS])
            for row in result['records']:
                raw=Path(row['output']['path']).read_bytes();self.assertEqual(len(raw),16384)
                self.assertEqual(hashlib.sha256(raw).hexdigest(),row['result']['attention_output_sha256'])
        self.assertEqual(len(list(self.root.rglob('attention.bf16'))),36)
    def test_missing_last_decode_cannot_report_completion(self):
        with self.assertRaisesRegex(ValueError,'All18'):self.run_fake(items=self.items()[:-1])
    def test_duplicate_decode_cannot_overwrite_previous_result(self):
        items=self.items();items[1]=copy.deepcopy(items[0])
        with self.assertRaises(FileExistsError):self.run_fake(items=items)
    def test_native_byte_mismatch_cannot_save_success(self):
        def wrong(*args):
            result,raw=self.native(*args);return result,b'!'+raw[1:]
        with self.assertRaisesRegex(ValueError,'Complete four-row'):self.run_fake(native=wrong)
        self.assertEqual(list(self.root.rglob('attention.bf16')),[])
    def test_native_contract_failure_cannot_save_success(self):
        def wrong(*args):
            result,raw=self.native(*args);result['full_input_cache_unchanged']=False;return result,raw
        with self.assertRaisesRegex(ValueError,'preserves original contract'):self.run_fake(native=wrong)
        self.assertEqual(list(self.root.rglob('attention.bf16')),[])
    def test_unpinned_native_source_rejected_before_model(self):
        ctx=self.context();ctx.plan['source_sha256'].pop(str(Path(d.native.__file__).resolve()))
        with patch.object(d.constructor,'build',side_effect=AssertionError('must not construct')):
            with self.assertRaisesRegex(ValueError,'source closure'):
                d.run_rank(ctx,None,None,None,None,lambda *a:None,self.guard)
    def test_guard_denial_before_any_native_access(self):
        def deny():raise RuntimeError('not admitted')
        with self.assertRaisesRegex(RuntimeError,'not admitted'):
            d.run_rank(None,None,None,None,None,lambda *a:None,deny)
    def test_output_writer_rejects_partial_rows_and_prefill(self):
        for arm,ci,raw in ((d.stream.raw.ARMS[0],4,bytes(4096)),(d.stream.raw.ARMS[0],0,bytes(16384)),
            ('wrong',4,bytes(16384)),(d.stream.raw.ARMS[0],True,bytes(16384))):
            with self.subTest(arm=arm,call=ci),self.assertRaisesRegex(ValueError,'complete original'):
                d.save_output(self.context(),arm,ci,raw,self.guard)
    def test_saved_bytes_are_actually_reread_and_checked(self):
        with patch.object(d.stream.payload_io,'read_owned_payload',return_value=b'!'*16384):
            with self.assertRaisesRegex(ValueError,'Independently reread'):
                d.save_output(self.context(),d.stream.raw.ARMS[0],4,bytes(16384),self.guard)
    def test_symlink_output_root_rejected(self):
        linked=self.root/'link';linked.symlink_to(self.root,target_is_directory=True)
        ctx=self.context();ctx.plan['output_root']=str(linked)
        with self.assertRaisesRegex(ValueError,'Canonical'):
            d.save_output(ctx,d.stream.raw.ARMS[0],4,bytes(16384),self.guard)
    def test_output_guard_revocation_before_write_leaves_no_success_file(self):
        count=[0]
        def guard():
            count[0]+=1
            if count[0]==2:raise RuntimeError('revoked')
        with self.assertRaisesRegex(RuntimeError,'revoked'):
            d.save_output(self.context(),d.stream.raw.ARMS[0],4,bytes(16384),guard)
        self.assertEqual(list(self.root.rglob('attention.bf16')),[])


if __name__=='__main__':
    assert not {'torch','numpy','torch_neuronx','transformers'}&set(sys.modules)
    unittest.main(verbosity=2)
