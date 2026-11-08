"""Regression checks for exact original 4,087,706-byte plan admission only."""
import hashlib,json,tempfile,unittest,sys
from pathlib import Path
Q=Path(__file__).resolve().parent;sys.path[:0]=[str(Q),str(Q.parent)]
import review_minicpm5_frame1_attention_metadata_v2 as old
import review_minicpm5_frame1_attention_metadata_v3 as new
import review_minicpm5_frame1_attention_complete_v3 as complete
class Reader(unittest.TestCase):
 def test_original_failure_reproduced(self):
  with self.assertRaises(RuntimeError):old.load(new.BASE_PLAN)
 def test_exact_original_plan_allowed(self):
  d=new.load(new.BASE_PLAN);self.assertEqual(len(d['source_sha256']),2022);self.assertEqual(new.BASE_PLAN,complete.BASE_PLAN)
 def test_wrong_plan_hash_rejected(self):
  with self.assertRaises(RuntimeError):new.load(dict(new.BASE_PLAN,sha256='0'*64))
 def test_other_large_metadata_stays_rejected(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t)/'large.json';data=b'"'+b'x'*(2*2**20)+b'"';p.write_bytes(data)
   with self.assertRaises(RuntimeError):new.load(dict(path=str(p),sha256=hashlib.sha256(data).hexdigest()))
 def test_small_valid_metadata_unchanged(self):
  p=Q/'minicpm5-attention-r2-native-observation-20261108-v1.json';ref=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest());self.assertEqual(new.load(ref),old.load(ref))
 def test_symlink_cannot_bypass_small_limit(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t)/'plan.json';p.symlink_to(new.BASE_PLAN['path'])
   with self.assertRaises(RuntimeError):new.load(dict(path=str(p),sha256=new.BASE_PLAN['sha256']))
 def test_complete_reader_pins_new_metadata(self):
  self.assertIs(complete.meta,new);complete.authentic_sources();self.assertEqual(complete.SOURCE_PINS[str(Path(new.__file__))],hashlib.sha256(Path(new.__file__).read_bytes()).hexdigest())
 def test_no_native_frameworks_loaded(self):self.assertFalse({'torch','numpy','transformers','torch_neuronx'}&set(sys.modules))
if __name__=='__main__':unittest.main()
