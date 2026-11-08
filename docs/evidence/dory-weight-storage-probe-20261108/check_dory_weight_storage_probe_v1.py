"""Source/stdlib regression checks; actual Torch storage semantics remain untested."""
import copy,gc,hashlib,sys,unittest,weakref
from pathlib import Path
from unittest.mock import patch
Q=Path(__file__).resolve().parent;sys.path.insert(0,str(Q))
import dory_weight_storage_probe_v1 as p


def saved():
    rows=[dict(shape=list(shape),buffer_type=kind)for kind in ('plain','tracked')for shape in p.SHAPES];records={};payloads={}
    for row in rows:
        raw=p.fixture_bytes(row['shape']);label=row['buffer_type']+'_'+str(len(raw)//2)
        for stage in p.STAGES:
            name=label+'_'+stage;actual=(b'\x80\x3f'+raw[2:])if stage=='mutation'else raw
            records[name]=dict(name=name,shape=row['shape'],dtype='bfloat16',bytes=len(actual),sha256=hashlib.sha256(actual).hexdigest());payloads[name]=actual
    return rows,records,payloads


class Source(unittest.TestCase):
    def test_bounded_exact_plan(self):
        v=p.plan();self.assertEqual((v['cases'],v['saved_tensors'],v['saved_raw_bytes']),(6,18,786864));self.assertLess(v['saved_raw_bytes'],p.MAX_LIVE_SYNTHETIC_BYTES)
    def test_all_bf16_bit_patterns_without_arithmetic(self):
        raw=p.fixture_bytes((256,256));words={int.from_bytes(raw[i:i+2],'little')for i in range(0,len(raw),2)}
        self.assertEqual(words,set(range(65536)))
    def test_unsupported_shapes_rejected(self):
        for shape in [(),(0,),(512,512),(4,4),(-1,)]:
            with self.assertRaises(ValueError):p.fixture_bytes(shape)
    def test_original507_tensor_contract_still_matches(self):
        contract=p.source_check();self.assertEqual(contract['tensor_count'],507);self.assertEqual(contract['data_bytes'],7950458880)
    def test_tracked_owner_is_weak_referenceable(self):
        buf=p.TrackedBuffer(b'abcd');weak=weakref.ref(buf);view=memoryview(buf);del buf;gc.collect();self.assertIsNotNone(weak());view.release();del view;gc.collect();self.assertIsNone(weak())
    def test_lease_failure_precedes_any_torch_or_source_use(self):
        with patch.object(p,'source_check')as source:
            with self.assertRaisesRegex(RuntimeError,'expired'):p.run(torch=object(),guard=lambda:(_ for _ in ()).throw(RuntimeError('expired')),emit=lambda *a:None)
            source.assert_not_called()
    def test_missing_sink_rejected_before_work(self):
        with self.assertRaises(ValueError):p.run(torch=object(),guard=lambda:True,emit=None)
    def test_unimported_torch_rejected(self):
        with self.assertRaises(ValueError):p.run(torch=object(),guard=lambda:True,emit=lambda *a:None)
    def test_no_framework_imported_or_future_execution(self):
        self.assertFalse({'torch','torch_neuronx','numpy','transformers'}&set(sys.modules));self.assertFalse(list(Q.glob('dory-weight-storage-r*-request.json')))


class SavedBytes(unittest.TestCase):
    def setUp(self):self.rows,self.records,self.payloads=saved()
    def review(self):return p.validate_saved(self.records,self.rows,lambda record:self.payloads[record['name']])
    def test_complete_raw_words_independently_derived(self):
        result=self.review();self.assertTrue(result['all_saved_words_exact']);self.assertFalse(result['torch_storage_lifetime_reexecuted']);self.assertFalse(result['official_weights_verified'])
    def test_missing_case_rejected(self):
        self.rows.pop()
        with self.assertRaises(ValueError):self.review()
    def test_duplicate_case_rejected(self):
        self.rows[-1]=self.rows[0]
        with self.assertRaises(ValueError):self.review()
    def test_missing_stage_rejected(self):
        self.records.pop(next(iter(self.records)))
        with self.assertRaises(ValueError):self.review()
    def test_extra_stage_rejected(self):
        self.records['extra']=self.records[next(iter(self.records))]
        with self.assertRaises(ValueError):self.review()
    def test_one_raw_byte_changed_rejected_even_with_unmodified_hash(self):
        key='plain_65536_detached';v=self.payloads[key];self.payloads[key]=v[:-1]+bytes([v[-1]^1])
        with self.assertRaises(ValueError):self.review()
    def test_changed_raw_and_matching_declared_hash_still_rejected(self):
        key='tracked_32_initial';v=self.payloads[key];v=bytes([v[0]^1])+v[1:];self.payloads[key]=v;self.records[key]['sha256']=hashlib.sha256(v).hexdigest()
        with self.assertRaises(ValueError):self.review()
    def test_truncation_rejected(self):
        key='tracked_65536_initial';self.payloads[key]=self.payloads[key][:-2]
        with self.assertRaises(ValueError):self.review()
    def test_wrong_stage_words_rejected(self):
        self.payloads['plain_32_mutation']=self.payloads['plain_32_initial']
        with self.assertRaises(ValueError):self.review()
    def test_reported_size_cannot_shrink(self):
        self.records['tracked_4_detached']['bytes']-=2
        with self.assertRaises(ValueError):self.review()
    def test_mislabeled_shape_rejected(self):
        self.records['plain_32_initial']['shape']=[32]
        with self.assertRaises(ValueError):self.review()
    def test_mislabeled_dtype_rejected(self):
        self.records['plain_32_initial']['dtype']='float16'
        with self.assertRaises(ValueError):self.review()
    def test_nonbyte_reader_rejected(self):
        with self.assertRaises(ValueError):p.validate_saved(self.records,self.rows,lambda record:bytearray(self.payloads[record['name']]))

if __name__=='__main__':unittest.main(verbosity=2)
