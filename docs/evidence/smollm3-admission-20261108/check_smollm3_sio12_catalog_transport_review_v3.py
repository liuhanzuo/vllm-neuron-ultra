"""Root transport closure checks on real temporary evidence files."""
import hashlib,json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
R=Path('/home/ubuntu/experiments/trn1-latest-models-20261031');sys.path.insert(0,str(R))
from research import smollm3_sio12_catalog_transport_review_v2 as review
import smollm3_observer_metadata_store_v1 as store
import smollm3_observer_invariance_lifecycle_v21 as life
class Closure(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);(self.root/'control').mkdir(mode=0o700);self.patch=patch.object(life,'DEST',self.root);self.patch.start();store.provision(self.root)
  life.record(self.root/'control/source-admission.json',{'source_stats':{'/source.py':dict(device=1,inode=1,size=0,mtime_ns=1,ctime_ns=1,realpath='/source.py')}})
  self.catalog=dict(source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in (self.root/'control/source-maps').glob('*.json')},rows=[{'unit':'retained','failure':True}],processes=[{'pid':7}],root_finalized=False,requires_root_terminal_review=True)
  self.path=self.root/'control/successor-catalog.json';life.record(self.path,self.catalog);self.ref={'path':str(self.path),'sha256':hashlib.sha256(self.path.read_bytes()).hexdigest()}
 def tearDown(self):self.patch.stop();self.tmp.cleanup()
 def test_all_physical_dependencies_retained_without_success_inference(self):
  cat,proof=review.collect(self.root,self.ref);self.assertEqual(cat,self.catalog);self.assertTrue(proof['closure_authenticated']);self.assertFalse(proof['extra_evidence_requires_review'])
  self.assertFalse(proof['actual_exit_observed']);self.assertFalse(proof['physical_drain_proven']);self.assertFalse(proof['root_finalization_granted']);self.assertFalse(proof['scientific_success_proven'])
  files=set(str(p)for d in ['source-maps','catalog-maps']for p in (self.root/'control'/d).glob('*.json'))|{str(self.path)};self.assertEqual(set(proof['physical_sha256']),files)
 def test_changed_catalog_reference_rejected(self):
  with self.assertRaises(ValueError):review.collect(self.root,dict(self.ref,sha256='0'*64))
 def test_changed_chunk_rejected(self):
  path=next((self.root/'control/catalog-maps').glob('*.json'));path.write_bytes(path.read_bytes()+b' ')
  with self.assertRaises(ValueError):review.collect(self.root,self.ref)
 def test_missing_chunk_rejected(self):
  next((self.root/'control/catalog-maps').glob('*.json')).unlink()
  with self.assertRaises(FileNotFoundError):review.collect(self.root,self.ref)
 def test_pending_failed_publication_is_retained(self):
  path=self.root/'control/catalog-maps'/('.pending-'+'a'*64+'.json-7-'+'b'*32);life.write_once(path,b'incomplete retained bytes')
  cat,proof=review.collect(self.root,self.ref);self.assertEqual(cat,self.catalog);self.assertTrue(proof['extra_evidence_requires_review']);self.assertEqual(len(proof['retained_pending_files']),1);self.assertIn(str(path),proof['physical_sha256']);self.assertNotIn(str(path),proof['authenticated_source_sha256_additions'])
 def test_extra_catalog_chunk_retained_and_flagged(self):
  raw=b'{}';path=self.root/'control/catalog-maps'/(hashlib.sha256(raw).hexdigest()+'.json');life.write_once(path,raw)
  _,proof=review.collect(self.root,self.ref);self.assertTrue(proof['extra_evidence_requires_review']);self.assertEqual(len(proof['unreferenced_catalog_chunks']),1);self.assertIn(str(path),proof['physical_sha256'])
 def test_ordinary_source_chunk_missing_from_catalog_rejected(self):
  raw=b'{}';life.write_once(self.root/'control/source-maps'/(hashlib.sha256(raw).hexdigest()+'.json'),raw)
  with self.assertRaises(ValueError):review.collect(self.root,self.ref)
 def test_unexpected_directory_or_symlink_rejected(self):
  (self.root/'control/catalog-maps'/'unreviewed').symlink_to('/dev/null')
  with self.assertRaises(ValueError):review.collect(self.root,self.ref)
 def test_non_json_extra_is_retained_but_not_a_source_pin(self):
  for raw in (b'\xff',b'{"x":1,"x":2}',b'{"x":NaN}',b'incomplete'):
   path=self.root/'control/catalog-maps'/(hashlib.sha256(raw).hexdigest()+'.json');life.write_once(path,raw)
   _,proof=review.collect(self.root,self.ref)
   self.assertIn(str(path),proof['physical_sha256']);self.assertNotIn(str(path),proof['authenticated_source_sha256_additions']);self.assertTrue(proof['extra_evidence_requires_review'])
  self.assertEqual(len(proof['non_source_extra_files']),4)
if __name__=='__main__':unittest.main(verbosity=2)
