"""Synthetic finalizer construction and small archive fixtures; no actual exit claims."""
import copy,hashlib,io,json,os,sys,tarfile,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from research import finalize_completed_cache_archival_20261108_v1 as f
from research import check_nemotron9b_post_smol_maintenance_v1 as fixtures

class BuildChecks(unittest.TestCase):
 def setUp(self):
  self.spec,self.catalog,self.docs=fixtures.fixture();e=self.catalog['maintenance_extension'];self.prior=self.docs[e['prior_catalog']['path']];self.ext=e
 def built(self,**changes):
  e=self.ext;args=dict(prior=self.prior,prior_ref=e['prior_catalog'],start=self.docs[e['started']['path']],start_ref=e['started'],result=self.docs[e['result']['path']],result_ref=e['result'],session=999,chunk='abc123',observed='synthetic-fixture-only',drains=[dict(unit='fixture-historical-service',samples=[])],archive_check=dict(files_rehashed=1039,directories_verified=253),source_refs=[dict(path='/synthetic-fixture/finalizer.py',sha256='f'*64)])
  args.update(changes);return f.build(**args)
 def test_builder_passes_independent_contract_without_rewriting_history(self):
  before=copy.deepcopy(self.prior);exited,audit,catalog,spec=self.built();docs=dict(self.docs,**{str(f.EXIT):exited,str(f.AUDIT):audit})
  value=f.bridge.validate(spec,catalog,lambda ref:docs[ref['path']]);self.assertEqual(value['catalog'],before);self.assertEqual(self.prior,before);self.assertFalse(value['fresh_physical_drain_proven']);self.assertFalse(value['new_speedup_claim'])
  for path,value in [(f.EXIT,exited),(f.AUDIT,audit),(f.CATALOG,catalog)]:self.assertEqual(f.value_ref(path,value)['sha256'],hashlib.sha256(f.bytes_for(value)).hexdigest())
 def test_no_process_or_service_drain_omission(self):
  with self.assertRaises(AssertionError):self.built(drains=[])
 def test_wrong_service_drained_rejected(self):
  with self.assertRaises(AssertionError):self.built(drains=[dict(unit='different')])
 def test_incomplete_archive_verification_rejected(self):
  with self.assertRaises(AssertionError):self.built(archive_check=dict(files_rehashed=1038,directories_verified=253))
 def test_nonactual_session_and_chunk_rejected(self):
  for changes in [dict(session=0),dict(session=False),dict(chunk='unknown')]:
   with self.assertRaises(AssertionError):self.built(**changes)
 def test_original_pin_cannot_be_rebound(self):
  with self.assertRaises(AssertionError):self.built(source_refs=[dict(path='/source-only-fixture/unchanged.py',sha256='0'*64)])
 def test_failed_tool_exit_rejected_before_any_read(self):
  with patch.object(f,'ref')as read:
   with self.assertRaisesRegex(AssertionError,'successful original'):f.main(types.SimpleNamespace(session=999,tool_chunk='abc123',exit_code=1))
   read.assert_not_called()
 def test_hash_bound_completed_history_read_retains_expected(self):
  with patch.object(f.p.metadata,'sha',return_value='a'*64)as sha:
   self.assertEqual(f.ref(Path('/synthetic/history.json'),'a'*64),dict(path='/synthetic/history.json',sha256='a'*64));sha.assert_called_once_with(Path('/synthetic/history.json'),'a'*64)
 def test_metadata_output_is_bounded_and_exclusive(self):
  with self.assertRaises(AssertionError):f.bytes_for(dict(text='too large'),1)
  with tempfile.TemporaryDirectory()as tmp:
   path=Path(tmp)/'record.json';value=dict(synthetic=True);self.assertEqual(f.record(path,value),f.value_ref(path,value))
   with self.assertRaises(FileExistsError):f.record(path,value)

class ArchiveChecks(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(prefix='archive-finalizer-test-');self.root=Path(self.tmp.name);self.archive=self.root/'cache.tar.gz';self.cache=self.root/'absent-original-cache';self.data=b'synthetic member bytes';self.member='MODULE_fixture/model.neff'
  identity=dict(mode=0o100600,uid=os.getuid(),gid=os.getgid(),size=len(self.data),sha256=hashlib.sha256(self.data).hexdigest())
  self.manifest=dict(files={self.member:identity},directories={str(self.cache/'MODULE_fixture'):dict(mode=0o40700,uid=os.getuid(),gid=os.getgid())});self.candidate=dict(cache_root=str(self.cache))
  self.make();self.patcher=patch.object(f.bridge,'ARCHIVE_ROOT',self.root);self.patcher.start()
 def tearDown(self):self.patcher.stop();self.tmp.cleanup()
 def make(self,extra=None):
  with tarfile.open(self.archive,'w:gz')as archive:
   directory=tarfile.TarInfo('MODULE_fixture');directory.type=tarfile.DIRTYPE;directory.mode=0o700;directory.uid=os.getuid();directory.gid=os.getgid();archive.addfile(directory)
   member=tarfile.TarInfo(self.member);member.size=len(self.data);member.mode=0o600;member.uid=os.getuid();member.gid=os.getgid();archive.addfile(member,io.BytesIO(self.data))
   if extra=='duplicate':archive.addfile(member,io.BytesIO(self.data))
   if extra=='symlink':
    link=tarfile.TarInfo('MODULE_fixture/link');link.type=tarfile.SYMTYPE;link.linkname='/etc/passwd';archive.addfile(link)
  self.result=dict(archive=dict(path=str(self.archive),sha256=hashlib.sha256(self.archive.read_bytes()).hexdigest()),manifest=dict(path='/synthetic/files.json',sha256='a'*64))
 def verify(self):return f.verify_archive(self.result,self.manifest,self.candidate)
 def test_archive_and_each_member_rehashed(self):
  result=self.verify();self.assertEqual(result['files_rehashed'],1);self.assertEqual(result['directories_verified'],1)
 def test_wrong_archive_hash_rejected(self):
  self.result['archive']['sha256']='0'*64
  with self.assertRaises(AssertionError):self.verify()
 def test_wrong_member_hash_rejected(self):
  self.manifest['files'][self.member]['sha256']='0'*64
  with self.assertRaises(AssertionError):self.verify()
 def test_duplicate_member_rejected(self):
  self.make('duplicate')
  with self.assertRaises(AssertionError):self.verify()
 def test_unlisted_symlink_rejected(self):
  self.make('symlink')
  with self.assertRaises(AssertionError):self.verify()
 def test_unretired_original_directory_rejected(self):
  (self.cache/'MODULE_fixture').mkdir(parents=True)
  with self.assertRaises(AssertionError):self.verify()
 def test_mode_metadata_mismatch_rejected(self):
  self.manifest['files'][self.member]['mode']=0o100644
  with self.assertRaises(AssertionError):self.verify()
if __name__=='__main__':unittest.main(verbosity=2)
