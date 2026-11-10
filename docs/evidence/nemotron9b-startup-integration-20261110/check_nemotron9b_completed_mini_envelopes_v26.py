"""Real completed metadata envelopes; no tensor, SDK or service execution."""
import hashlib,json,sys,unittest
from pathlib import Path
from unittest.mock import Mock
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import nemotron9b_package_metadata_v25 as old
import nemotron9b_package_metadata_v26 as m
import nemotron9b_package_mini_envelopes_v26 as e
class Envelopes(unittest.TestCase):
 def test_all_three_original_full_files_read_under_original32mib_limit(self):
  for path,row in e.ROWS.items():
   with self.subTest(path=path):
    calls=[]
    def read(name,cap):calls.append((str(name),cap));return m.stable_bytes(name,cap)
    raw=e.text(path,row['sha256'],read)
    self.assertEqual(len(raw),row['bytes']);self.assertEqual(hashlib.sha256(raw).hexdigest(),row['sha256'])
    self.assertEqual(calls,[(e.FAILED['path'],8*2**20),(path,32*2**20)])
    self.assertEqual(m.sha(path,row['sha256']),row['sha256'])
 def test_old_ordinary_limit_rejects_all_three(self):
  for path,row in e.ROWS.items():
   with self.subTest(path=path),self.assertRaisesRegex(RuntimeError,'bounded metadata'):old.sha(path,row['sha256'])
 def test_unknown_path_or_wrong_hash_never_opens(self):
  for path,digest in [('/tmp/anything.json','0'*64),*[(x,'0'*64)for x in e.ROWS]]:
   read=Mock()
   with self.subTest(path=path),self.assertRaisesRegex(RuntimeError,'before open'):e.text(path,digest,read)
   read.assert_not_called()
 def test_corrupt_failed_exit_never_opens_large_record(self):
  path=next(iter(e.ROWS));read=Mock(return_value=b'{}')
  with self.assertRaisesRegex(RuntimeError,'failed-r1 receipt'):e.text(path,e.ROWS[path]['sha256'],read)
  self.assertEqual(read.call_count,1)
 def test_truncated_large_bytes_rejected_after_valid_receipt(self):
  proof=m.stable_bytes(e.FAILED['path'],8*2**20);path=next(iter(e.ROWS));read=Mock(side_effect=[proof,b'{}'])
  with self.assertRaisesRegex(RuntimeError,'bounded envelope'):e.text(path,e.ROWS[path]['sha256'],read)
  self.assertEqual(read.call_count,2)
 def test_shapes_cannot_promote_failed_source_record_to_model_success(self):
  path=next(x for x,v in e.ROWS.items()if v['role']=='sources')
  value=json.loads(m.stable_bytes(path,32*2**20))
  for field in ('native_runtime_identity_verified','model_forward_provenance_verified','full_model_qualified','new_speedup_claim'):
   altered=dict(value,**{field:True})
   with self.subTest(field=field),self.assertRaisesRegex(RuntimeError,'qualification'):e.shape('sources',altered)
  self.assertEqual(m.LIMITS,old.LIMITS)
if __name__=='__main__':unittest.main(verbosity=2)
