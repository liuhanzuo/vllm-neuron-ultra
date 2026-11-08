"""Differential lexical/identity semantics and actual inherited transport tests."""
import ast,copy,itertools,json,os,random,sys,tempfile,types,unittest
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import smollm3_observer_metadata_codec_v2 as old
import smollm3_observer_metadata_codec_v4 as new
import smollm3_stat_identity_v2 as identity
import smollm3_observer_invariance_lifecycle_v22 as life

class Paths(unittest.TestCase):
 def compare(self,name):
  try:wanted=old.absolute(name).suffix
  except (ValueError,TypeError):
   with self.assertRaises((ValueError,TypeError)):new.absolute_suffix(name)
  else:self.assertEqual(new.absolute_suffix(name),wanted,repr(name))
 def test_posix_roots_and_canonical_parts(self):
  for root in ['','/','//','///','////']:
   for parts in itertools.product(['a','b.py','','.','..','..abc','.hidden','t.','\\','中'],repeat=3):self.compare(root+'/'.join(parts))
 def test_fuzz_lexical_equivalence(self):
  rng=random.Random(84373)
  for _ in range(20000):self.compare(''.join(rng.choice('/.ab\\中\n')for _ in range(rng.randrange(35))))
 def test_exact_size_and_type_domain(self):
  for n in [0,1,4095,4096,4097]:self.compare('/'+'x'*n)
  for x in [None,False,2,b'/abc','/a\x00b']:self.compare(x)
 def test_suffix_corner_cases(self):
  for name in ['.py','..py','...', 'a..b','x.pt','x.pt.','x.so','x.raw','x.PY','é.json','中.文']:
   for prefix in ['/','//','/root/']:self.compare(prefix+name)
 def test_all_real_retained_source_names_and_targets(self):
  expected=life.read(R/'sio13/control/source-admission.json')['source_stats'];self.assertEqual(len(expected),37315)
  for name,row in expected.items():self.compare(name);self.compare(row['realpath'])
 def test_full_map_domain_matches_original(self):
  for name in ['/a.py','//a.py','/a.raw','/a/../x.py','/a//b.py']:
   for kind in ['sha256','stat','fingerprint']:
    val='a'*64 if kind!='stat'else dict(device=1,inode=1,size=0,mtime_ns=1,ctime_ns=1,realpath='/a.py')
    entries={name:val}
    try:expected=old.validate_map(kind,entries)
    except ValueError:
     with self.assertRaises(ValueError):new.validate_map(kind,entries)
    else:self.assertEqual(new.validate_map(kind,entries),expected)

class Stat(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.file=self.root/'file.py';self.file.write_text('old')
 def tearDown(self):self.tmp.cleanup()
 def same(self,path):self.assertEqual(identity.file_stat(path),life.file_stat(path))
 def test_ordinary_and_path_arguments(self):self.same(self.file);self.same(str(self.file))
 def test_symlink_and_parent_symlink(self):
  link=self.root/'link';link.symlink_to(self.file);self.same(link)
  directory=self.root/'dir';directory.mkdir();(directory/'child').write_text('x');parent=self.root/'parent';parent.symlink_to(directory,target_is_directory=True);self.same(parent/'child')
 def test_target_replacement_not_cached(self):
  link=self.root/'link';link.symlink_to(self.file);before=identity.file_stat(link)
  other=self.root/'next';other.write_text('new');other.replace(self.file);self.same(link);self.assertNotEqual(before,identity.file_stat(link))
 def test_link_redirection_not_cached(self):
  link=self.root/'link';link.symlink_to(self.file);before=identity.file_stat(link)
  other=self.root/'second.py';other.write_text('new');link.unlink();link.symlink_to(other);self.same(link);self.assertNotEqual(before,identity.file_stat(link))
 def test_mutation_changes_identity(self):
  before=identity.file_stat(self.file);self.file.write_text('modified');self.same(self.file);self.assertNotEqual(before,identity.file_stat(self.file))
 def test_descriptor_closed_after_readlink_failure(self):
  from unittest.mock import patch
  before=len(os.listdir('/proc/self/fd'))
  with patch.object(identity.os,'readlink',side_effect=RuntimeError('fixture readlink failure')):
   with self.assertRaises(RuntimeError):identity.file_stat(self.file)
  self.assertEqual(len(os.listdir('/proc/self/fd')),before)
 def test_opath_needs_no_file_body_read_permission(self):
  self.file.chmod(0);self.same(self.file)
 def test_opath_flags_and_close_success(self):
  from unittest.mock import patch
  before=len(os.listdir('/proc/self/fd'));original=os.open;flags=[]
  def opened(path,flag,*args,**kwargs):flags.append(flag);return original(path,flag,*args,**kwargs)
  with patch.object(identity.os,'open',opened):self.same(self.file)
  self.assertEqual(flags,[os.O_PATH|os.O_CLOEXEC]);self.assertEqual(len(os.listdir('/proc/self/fd')),before)
 def test_missing_and_loop_rejected(self):
  for fn in [life.file_stat,identity.file_stat]:
   with self.assertRaises((OSError,RuntimeError)):fn(self.root/'absent')
  loop=self.root/'loop';loop.symlink_to(loop)
  for fn in [life.file_stat,identity.file_stat]:
   with self.assertRaises((OSError,RuntimeError)):fn(loop)

class Preservation(unittest.TestCase):
 def test_only_map_path_checks_changed(self):
  a=ast.parse((R/'smollm3_observer_metadata_codec_v2.py').read_text());b=ast.parse((R/'smollm3_observer_metadata_codec_v4.py').read_text())
  def kept(tree):return [ast.dump(n)for n in tree.body if not isinstance(n,ast.FunctionDef)or n.name not in ['validate_map','absolute_suffix']]
  self.assertEqual(kept(a),kept(b))
 def test_store_only_codec_import_changed(self):
  self.assertEqual((R/'smollm3_observer_metadata_store_v1.py').read_text().replace('smollm3_observer_metadata_codec_v2','smollm3_observer_metadata_codec_v4'),(R/'smollm3_observer_metadata_store_v3.py').read_text())
 def test_no_cache_or_budget_changes(self):
  for name in ['RECORD_SCHEMA','CHUNK_SCHEMA','FIELDS','CHUNK_ENTRIES','CHUNK_CAP','RECORD_CAP','EXPANSION_CAP','ENTRY_CAP','DEPTH_CAP','FORBIDDEN_SUFFIXES']:self.assertEqual(getattr(old,name),getattr(new,name))

def inherited(file,replacements):
 module=types.ModuleType(file);module.__file__=str(R/'research'/file);source=Path(module.__file__).read_text()
 for before,after in replacements:source=source.replace(before,after)
 exec(compile(source,module.__file__,'exec'),module.__dict__);return module
if __name__=='__main__':
 codec=inherited('check_smollm3_metadata_codec_v2.py',[('smollm3_observer_metadata_codec_v2','smollm3_observer_metadata_codec_v4')])
 store=inherited('check_smollm3_metadata_store_v1.py',[('smollm3_observer_metadata_store_v1','smollm3_observer_metadata_store_v3')])
 suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(t)for t in [Paths,Stat,Preservation,codec.previous.Codec,codec.Catalog,store.Store])
 result=unittest.TextTestRunner(verbosity=2).run(suite);raise SystemExit(0 if result.wasSuccessful()else 1)
