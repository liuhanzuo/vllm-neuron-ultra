"""Real 44-file installed-wheel closure plus mutation refusal; no native imports."""
import hashlib,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import smollm3_observer_metadata_source_v4 as reader
class Receipts(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.p=self.root/'INSTALLER';self.p.write_bytes(b'pip\n');self.sha=hashlib.sha256(self.p.read_bytes()).hexdigest()
  self.mock=patch.object(reader,'INSTALLATION_RECEIPTS',{str(self.p):self.sha});self.mock.start()
 def tearDown(self):self.mock.stop();self.tmp.cleanup()
 def test_exact_receipt_with_and_without_explicit_hash(self):
  self.assertEqual(reader.source_bytes(self.p),b'pip\n');self.assertEqual(reader.source_bytes(self.p,self.sha),b'pip\n')
 def test_wrong_requested_digest_rejected(self):
  with self.assertRaises(RuntimeError):reader.source_bytes(self.p,'0'*64)
 def test_modified_receipt_rejected(self):
  self.p.write_bytes(b'changed\n')
  with self.assertRaises(RuntimeError):reader.source_bytes(self.p)
 def test_missing_receipt_rejected(self):
  self.p.unlink()
  with self.assertRaises(FileNotFoundError):reader.source_bytes(self.p)
 def test_symlink_receipt_rejected(self):
  target=self.root/'other';self.p.rename(target);self.p.symlink_to(target)
  with self.assertRaises((RuntimeError,OSError)):reader.source_bytes(self.p)
 def test_same_basename_elsewhere_still_rejected(self):
  p=self.root/'sub';p.mkdir();p=p/'INSTALLER';p.write_bytes(b'pip\n')
  with self.assertRaises(RuntimeError):reader.source_bytes(p,self.sha)
 def test_oversized_receipt_rejected(self):
  self.p.write_bytes(b'a'*65)
  with self.assertRaises(RuntimeError):reader.source_bytes(self.p)
 def test_payload_extension_still_rejected(self):
  p=self.root/'model.neff';p.write_bytes(b'not a model')
  with self.assertRaises(RuntimeError):reader.source_bytes(p)
class ActualWheel(unittest.TestCase):
 def test_original_reader_reproduces_first_failure(self):
  import smollm3_observer_metadata_source_v1 as old
  p=Path(next(iter(reader.INSTALLATION_RECEIPTS)))
  with self.assertRaisesRegex(RuntimeError,'Reviewed UTF8 text only'):old.source_bytes(p)
 def test_all44_original_installed_files(self):
  wheel_path=R/'smollm3-cache-ablation-native-v3/wheel.json'
  wheel=json.loads(reader.source_bytes(wheel_path));self.assertEqual(len(wheel['installed_files']),44)
  target=Path('/mnt/nvme/smollm3-cache-ablation-native-v3/target')
  for n,digest in wheel['installed_files'].items():
   with self.subTest(file=n):self.assertEqual(hashlib.sha256(reader.source_bytes(target/n)).hexdigest(),digest)
  self.assertEqual(set(reader.INSTALLATION_RECEIPTS),{str(target/n)for n in wheel['installed_files']if Path(n).name in ('REQUESTED','INSTALLER')})
 def test_predecessor_logic_unchanged_except_exact_receipts(self):
  old=(R/'smollm3_observer_metadata_source_v1.py').read_text().split('    if name in inherited.NON_UTF8:')[1]
  new=(R/'smollm3_observer_metadata_source_v4.py').read_text().split('    if name in inherited.NON_UTF8:')[1]
  self.assertEqual(old,new)
class CompletedCatalog(unittest.TestCase):
 def test_actual_complete_catalog_with_and_without_digest(self):
  p,d=next(iter(reader.COMPLETED_CATALOGS.items()));raw=reader.source_bytes(p,d)
  self.assertGreater(len(raw),8*2**20);self.assertEqual(raw,reader.source_bytes(p));self.assertEqual(len(json.loads(raw)['rows']),760)
 def test_wrong_catalog_digest_rejected(self):
  p=next(iter(reader.COMPLETED_CATALOGS))
  with self.assertRaisesRegex(RuntimeError,'Exact completed catalog pin'):reader.source_bytes(p,'0'*64)
 def test_catalog_bytes_changed_rejected(self):
  p=next(iter(reader.COMPLETED_CATALOGS))
  with patch.object(reader.inherited,'stable_bytes',return_value=b'{}'),self.assertRaisesRegex(RuntimeError,'Completed catalog bytes changed'):reader.source_bytes(p)
 def test_catalog_wrong_schema_rejected(self):
  p=next(iter(reader.COMPLETED_CATALOGS));raw=json.dumps(dict(schema='wrong',reviewer='/root',root_finalized=True,complete_count=760,rows=[{}]*760,processes=[{}]*146,source_sha256={})).encode()
  with patch.object(reader,'COMPLETED_CATALOGS',{p:hashlib.sha256(raw).hexdigest()}),patch.object(reader.inherited,'stable_bytes',return_value=raw),self.assertRaisesRegex(RuntimeError,'Exact completed catalog schema'):reader.source_bytes(p)
 def test_unlisted_large_json_still_rejected(self):
  with tempfile.TemporaryDirectory()as tmp:
   p=Path(tmp)/'other.json'
   with p.open('wb')as f:f.truncate(8*2**20+1)
   with self.assertRaisesRegex(RuntimeError,'Ordinary bounded metadata'):reader.source_bytes(p)


class CurrentHistory(unittest.TestCase):
 def test_actual_sio13_completed_history(self):
  path=R/'research/root-post-smol-sio13-complete-predecessors-20261108-v1.json';digest='b29dca3e250b1d3aafb9a9dbdabad210631966d3f76400e392fbdb97b33e96b6'
  raw=reader.source_bytes(path,digest);value=json.loads(raw);self.assertEqual(len(value['rows']),785);self.assertEqual(len(value['processes']),147)
 def test_new_completed_history_requires_explicit_hash(self):
  path=R/'research/root-post-smol-sio13-complete-predecessors-20261108-v1.json'
  for digest in [None,'0'*64]:
   with self.subTest(digest=digest),self.assertRaises(RuntimeError):reader.source_bytes(path,digest)
 def test_real_v10_large_spec_review_and_primitive(self):
  paths=[R/'research/nemotron9b-package-runtime-v10-r4-20261108-spec-v2.json',R/'research/nemotron9b-package-runtime-v10-r4-20261108-review-v2.json',R/'nemotron9b-package-r4-20261108-control/split-nemotron-conv-b1-t128-original-worker-result.json']
  for path in paths:
   digest=hashlib.sha256(path.read_bytes()).hexdigest()
   with self.subTest(path=str(path)):
    raw=reader.source_bytes(path,digest);self.assertEqual(hashlib.sha256(raw).hexdigest(),digest);self.assertGreater(len(raw),8*2**20)
 def test_changed_envelope_hash_rejected(self):
  path=R/'research/nemotron9b-package-runtime-v10-r4-20261108-spec-v2.json'
  with self.assertRaisesRegex(RuntimeError,'fingerprint changed'):reader.source_bytes(path,'0'*64)
 def test_large_wrong_schema_rejected(self):
  path=R/'research/nemotron9b-package-runtime-v10-r4-20261108-spec-v2.json';raw=json.dumps(dict(schema='wrong',padding='x'*(8*2**20))).encode()
  with patch.object(reader.inherited,'stable_bytes',return_value=raw),self.assertRaisesRegex(RuntimeError,'research envelope schema'):reader.source_bytes(path,hashlib.sha256(raw).hexdigest())
 def test_primitive_phase_mismatch_rejected(self):
  path=R/'nemotron9b-package-r4-20261108-control/split-nemotron-conv-b1-t128-original-worker-result.json';v=json.loads(path.read_text());v['phase']='different';raw=json.dumps(v).encode()
  self.assertGreater(len(raw),8*2**20)
  with patch.object(reader.inherited,'stable_bytes',return_value=raw),self.assertRaisesRegex(RuntimeError,'filename/phase binding'):reader.source_bytes(path,hashlib.sha256(raw).hexdigest())
 def test_unlisted_oversized_and_named_over32mib_rejected(self):
  with tempfile.TemporaryDirectory()as tmp:
   p=Path(tmp)/'other.json'
   with p.open('wb')as f:f.truncate(32*2**20+1)
   with self.assertRaisesRegex(RuntimeError,'Ordinary bounded metadata'):reader.source_bytes(p)
   with patch.object(reader.inherited,'research_envelope',return_value=True),self.assertRaisesRegex(RuntimeError,'Ordinary bounded metadata'):reader.source_bytes(p)
 def test_exact_original_three_encoding_exceptions(self):
  self.assertEqual(len(reader.inherited.NON_UTF8),3)
  for path,digest in reader.inherited.NON_UTF8.items():
   self.assertEqual(hashlib.sha256(reader.source_bytes(path,digest)).hexdigest(),digest)
   with self.assertRaises(RuntimeError):reader.source_bytes(path,'0'*64)
 def test_v3_source_unchanged_except_inherited_version(self):
  self.assertEqual((R/'smollm3_observer_metadata_source_v3.py').read_text().replace('import nemotron9b_package_metadata_v7 as inherited','import nemotron9b_package_metadata_v10 as inherited'),(R/'smollm3_observer_metadata_source_v4.py').read_text())
if __name__=='__main__':unittest.main(verbosity=2)

