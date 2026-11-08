"""Bounded predecessor metadata integration; no native imports/model execution."""
import copy,hashlib,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import nemotron9b_package_metadata_v9 as m
import nemotron9b_package_metadata_v8 as old
h=m.completed_smol
CAT=R/'research/root-post-smol-sio12-complete-predecessors-20261108-v1.json'
SHA='958a913c7b32594a1f8231bcbff07cdf194312301f33244a0eca4cc38cf1ab2a'
def fixture():
 return dict(schema='root_post_smollm3_sio13_complete_predecessors_v1',reviewer='/root',root_finalized=True,all_planned_launches_accounted_for=True,rows=[dict(unit='synthetic-only-no-launch')],processes=[dict(pid=1)],source_sha256={'/synthetic.py':'a'*64,'/synthetic.json':'b'*64},complete_count=1,appended_terminal_audit=dict(path='/synthetic.json',sha256='b'*64))
class Completed(unittest.TestCase):
 def test_actual_original_reader_rejection_reproduced(self):
  with self.assertRaisesRegex(RuntimeError,'Ordinary bounded metadata'):old.text(CAT,SHA)
 def test_complete_actual_catalog_load_preserves_original_failure(self):
  raw=m.text(CAT,SHA);self.assertEqual(hashlib.sha256(raw).hexdigest(),SHA)
  v=m.load(dict(path=str(CAT),sha256=SHA,bytes=len(raw)));self.assertEqual(len(v['rows']),760);self.assertEqual(len(v['processes']),146)
  self.assertGreater(len(raw),8*2**20);self.assertTrue(v['root_finalized']);self.assertIn('appended_terminal_audit',v)
 def test_missing_digest_refused_before_open(self):
  with patch.object(m,'stable_bytes',side_effect=AssertionError('must not open')),self.assertRaisesRegex(RuntimeError,'digest required'):m.text(CAT)
 def test_wrong_digest_rejected(self):
  with self.assertRaisesRegex(RuntimeError,'fingerprint changed'):m.text(CAT,'0'*64)
 def test_catalog_reference_wrong_byte_count_rejected(self):
  with self.assertRaisesRegex(RuntimeError,'byte count'):m.load(dict(path=str(CAT),sha256=SHA,bytes=1))
 def test_future_catalog_schema_transport_has_no_completion_claim(self):
  p=CAT.with_name('root-post-smol-sio13-complete-predecessors-20261108-v1.json');raw=json.dumps(fixture()).encode()
  with patch.object(m,'stable_bytes',return_value=raw)as read:self.assertEqual(m.text(p,hashlib.sha256(raw).hexdigest()),raw)
  read.assert_called_once_with(p,16*2**20)
 def test_invalid_finalization_schema_counts_or_history_rejected(self):
  changes=[lambda v:v.update(root_finalized=False),lambda v:v.update(schema='other'),lambda v:v.update(complete_count=True),lambda v:v.update(complete_count=2),lambda v:v.update(all_planned_launches_accounted_for=False),lambda v:v.update(source_sha256={}),lambda v:v.update(source_sha256={'/bad.py':'bad'}),lambda v:v.update(appended_terminal_audit={}),lambda v:v['source_sha256'].pop('/synthetic.json'),lambda v:v.update(rows=[dict(unit='same'),dict(unit='same')],complete_count=2)]
  path=CAT.with_name('root-post-smol-sio13-complete-predecessors-20261108-v1.json')
  for change in changes:
   value=fixture();change(value);raw=json.dumps(value).encode()
   with self.subTest(value=value),patch.object(m,'stable_bytes',return_value=raw),self.assertRaises((RuntimeError,ValueError)):m.text(path,hashlib.sha256(raw).hexdigest())
 def test_unrecognized_large_json_keeps_ordinary_cap(self):
  with tempfile.TemporaryDirectory()as tmp:
   p=Path(tmp)/'unknown.json'
   with p.open('wb')as f:f.truncate(8*2**20+1)
   with self.assertRaisesRegex(RuntimeError,'Ordinary bounded metadata'):m.text(p)
 def test_literal_json_extension_required(self):self.assertFalse(h.accepts(CAT.with_name(CAT.name.replace('.json','Xjson'))))
 def test_wrong_catalog_location_not_recognized(self):self.assertFalse(h.accepts(Path('/tmp')/CAT.name))
 def test_catalog_symlink_and_oversize_refused(self):
  with tempfile.TemporaryDirectory()as tmp,patch.object(h,'ROOT',Path(tmp)):
   directory=Path(tmp)/'research';directory.mkdir();p=directory/CAT.name;p.symlink_to(CAT)
   with self.assertRaisesRegex(RuntimeError,'Canonical ordinary'):m.text(p,SHA)
   p.unlink()
   with p.open('wb')as f:f.truncate(16*2**20+1)
   with self.assertRaisesRegex(RuntimeError,'Ordinary bounded'):m.text(p,SHA)
class Installed(unittest.TestCase):
 def test_actual_two_receipts(self):
  for p,d in h.INSTALLATION_RECEIPTS.items():
   with self.subTest(path=p):self.assertEqual(hashlib.sha256(m.text(p,d)).hexdigest(),d);self.assertEqual(m.sha(p),d)
 def test_receipt_wrong_digest_and_modified_bytes_rejected(self):
  p=next(iter(h.INSTALLATION_RECEIPTS))
  with self.assertRaisesRegex(RuntimeError,'installed receipt hash'):m.text(p,'0'*64)
  with patch.object(m,'stable_bytes',return_value=b'changed'),self.assertRaisesRegex(RuntimeError,'receipt changed'):m.text(p)
 def test_receipt_exact64byte_cap(self):
  p=next(iter(h.INSTALLATION_RECEIPTS))
  with patch.object(m,'stable_bytes',return_value=b'')as read:m.text(p)
  read.assert_called_once_with(Path(p),64)
 def test_receipt_basename_elsewhere_still_rejected(self):
  with tempfile.TemporaryDirectory()as tmp:
   p=Path(tmp)/'REQUESTED';p.write_bytes(b'')
   with self.assertRaisesRegex(RuntimeError,'Reviewed UTF8 text only'):m.text(p)
 def test_existing_limits_and_envelope_schemas_preserved(self):
  self.assertEqual(m.LIMITS,old.LIMITS)
  for k,v in old.ENVELOPE_SCHEMAS.items():self.assertEqual(m.ENVELOPE_SCHEMAS[k],v)
if __name__=='__main__':unittest.main(verbosity=2)
