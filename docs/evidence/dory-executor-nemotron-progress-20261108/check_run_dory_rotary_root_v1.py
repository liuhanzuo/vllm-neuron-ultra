"""Source, synthetic-file and mocked-precondition checks; no native import/run."""
import ast,base64,copy,hashlib,json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_dory_rotary_root_v1 as f

class Request(unittest.TestCase):
 def setUp(self):
  self.patches=[patch.object(f,'source_paths',return_value=['/source.py']),patch.object(f,'ref',return_value=dict(path='/source.py',sha256='a'*64)),patch.object(f.p,'canonical',return_value={'fixed':'digest'}),patch.object(f,'validate_runtime_identity',return_value=[])]
  for item in self.patches:item.start();self.addCleanup(item.stop)
  self.value=dict(schema='dory_rotary_runtime_request_v1',run_id='synthetic',python=str(f.PYTHON),torch_record=dict(path=str(f.RECORD),sha256=f.RECORD_SHA),dependency_records=copy.deepcopy(f.RECORDS),runtime_identity=dict(executable=str(f.EXECUTABLE),python=str(f.PYTHON)),budgets=copy.deepcopy(f.BUDGET),science=dict(unique_conditions=96,rounds=2,executed_comparisons=192,raw_tensors=53,raw_bytes=2230320,full_cache_positions=262144),output_root=str(f.R/'dory-rotary-synthetic'),canonical_sha256={'fixed':'digest'},source_sha256={'/source.py':'a'*64},native_reference_qualified=False,new_speedup_claim=False,automatic_successor_launch=False)
 def reject(self):
  with self.assertRaises(RuntimeError):f.validate_request(self.value)
 def test_original(self):self.assertIs(f.validate_request(self.value),self.value)
 def test_reduced_matrix(self):self.value['science']['unique_conditions']=95;self.reject()
 def test_shortened_cache(self):self.value['science']['full_cache_positions']=2048;self.reject()
 def test_budget_increase(self):self.value['budgets']['address_space_bytes']*=2;self.reject()
 def test_dependency_rebind(self):self.value['dependency_records']['numpy-2.3.5.dist-info/RECORD']='f'*64;self.reject()
 def test_unreviewed_interpreter(self):self.value['runtime_identity']['executable']='/usr/bin/python3';self.reject()
 def test_stale_science_source(self):self.value['source_sha256']['/source.py']='f'*64;self.reject()
 def test_missing_source(self):self.value['source_sha256']={};self.reject()
 def test_canonical_changed(self):self.value['canonical_sha256']={};self.reject()
 def test_false_speedup(self):self.value['new_speedup_claim']=True;self.reject()
 def test_output_escape(self):self.value['output_root']='/tmp/escape';self.reject()

class Terminal(unittest.TestCase):
 def setUp(self):
  self.cat=dict(path=str(f.history.CURRENT),sha256='b'*64);self.exit=dict(path='/synthetic/exit',sha256='c'*64);self.owner=dict(pid=500001,start_ticks='9000')
  self.value=dict(schema='nemotron9b_r6_finalizer_external_exit_v1',catalog=self.cat,owner=self.owner,actual_tool_exit_observed=True,exact_owner_gone=True,catalog_validated=True,actual_exit=dict(session_id=999999,exit_code=0,tool_chunk='abc123'))
  self.patches=[patch.object(f.history,'load',return_value={'nemotron9b_package_r6_append':{'finalizer_owner':self.owner}}),patch.object(f,'load',side_effect=lambda ref:self.value),patch.object(f.p.low,'gone')]
  for item in self.patches:item.start();self.addCleanup(item.stop)
 def test_real_terminal_required(self):self.assertTrue(f.validate_terminal(self.cat,self.exit))
 def test_no_actual_exit(self):
  self.value['actual_tool_exit_observed']=False
  with self.assertRaises(RuntimeError):f.validate_terminal(self.cat,self.exit)
 def test_failed_finalizer(self):
  self.value['actual_exit']['exit_code']=1
  with self.assertRaises(RuntimeError):f.validate_terminal(self.cat,self.exit)
 def test_live_owner(self):
  with patch.object(f.p.low,'gone',side_effect=RuntimeError('live')):
   with self.assertRaises(RuntimeError):f.validate_terminal(self.cat,self.exit)
 def test_stale_prior_catalog(self):
  with self.assertRaises(RuntimeError):f.validate_terminal(dict(self.cat,path='/old/catalog'),self.exit)
 def test_prepare_cannot_acquire_while_run_active(self):
  with patch.object(f,'ensure_observer',side_effect=RuntimeError('active')),patch.object(f,'acquire')as acquire:
   with self.assertRaises(RuntimeError):f.prepare(self.cat,self.exit,'synthetic')
   acquire.assert_not_called()

class Files(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory(prefix='dory-runner-source-');self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve()
 def test_binary_bytes(self):
  path=self.root/'library';path.write_bytes(b'original')
  row=f.verify_binary(path,hashlib.sha256(b'original').hexdigest(),8,lambda:None)
  self.assertEqual(row['identity']['st_size'],8)
  path.write_bytes(b'altered!')
  with self.assertRaises(RuntimeError):f.verify_binary(path,row['sha256'],8,lambda:None)
 def test_binary_size(self):
  path=self.root/'binary';path.write_bytes(b'abc')
  with self.assertRaises(RuntimeError):f.verify_binary(path,'a'*64,4,lambda:None)
 def test_binary_symlink(self):
  path=self.root/'binary';path.write_bytes(b'abc');link=self.root/'link';link.symlink_to(path)
  with self.assertRaises(RuntimeError):f.file_identity(link)
 def test_hash_guard_failure_preserved(self):
  path=self.root/'binary';path.write_bytes(b'abc')
  with self.assertRaisesRegex(RuntimeError,'guard failed'):f.verify_binary(path,hashlib.sha256(b'abc').hexdigest(),3,lambda:(_ for _ in ()).throw(RuntimeError('guard failed')))
 def record(self,lines):
  base=self.root/'lib';base.mkdir(exist_ok=True);info=base/'pkg.dist-info';info.mkdir(exist_ok=True);record=info/'RECORD';record.write_text(lines)
  return base,record,hashlib.sha256(record.read_bytes()).hexdigest()
 def parse(self,lines):
  base,record,digest=self.record(lines)
  with patch.object(f,'BASE',base),patch.object(f,'PYTHON',self.root/'bin/python'):return f.parse_record(record,digest)
 def test_record_valid(self):
  digest=base64.urlsafe_b64encode(hashlib.sha256(b'x').digest()).decode().rstrip('=')
  self.assertEqual(len(self.parse('pkg/file,sha256='+digest+',1\npkg.dist-info/RECORD,,\n')),1)
 def test_record_path_escape(self):
  with self.assertRaises(RuntimeError):self.parse('../../../escape,sha256='+'a'*43+',1\npkg.dist-info/RECORD,,\n')
 def test_record_unhashed_payload(self):
  with self.assertRaises(RuntimeError):self.parse('pkg/payload,,\npkg.dist-info/RECORD,,\n')
 def test_record_duplicates(self):
  with self.assertRaises(RuntimeError):self.parse('pkg/file,sha256='+'a'*43+',1\npkg/file,sha256='+'a'*43+',1\npkg.dist-info/RECORD,,\n')
 def test_record_wrong_pin(self):
  _,record,_=self.record('pkg.dist-info/RECORD,,\n')
  with self.assertRaises(RuntimeError):f.parse_record(record,'0'*64)
 def test_installed_record_metadata_only(self):
  total=0;count=0
  for name,digest in f.RECORDS.items():
   rows=f.parse_record(f.BASE/name,digest,torch=name.startswith('torch-'));total+=sum(row[2]for row in rows);count+=len(rows)
  self.assertGreater(count,11720);self.assertLess(total,800*2**20)
 def test_output_cap(self):
  (self.root/'output').write_bytes(b'123')
  with patch.dict(f.BUDGET,dict(output_bytes=2,root_floor_bytes=0)):
   with self.assertRaises(RuntimeError):f.usage(self.root)
 def test_output_symlink(self):
  (self.root/'link').symlink_to('/etc/passwd')
  with self.assertRaises(RuntimeError):f.usage(self.root)
 def test_output_hardlink(self):
  source=self.root/'source';source.write_bytes(b'x');os.link(source,self.root/'link')
  with self.assertRaises(RuntimeError):f.usage(self.root)
 def test_output_special_file(self):
  os.mkfifo(self.root/'fifo')
  with self.assertRaises(RuntimeError):f.usage(self.root)
 def test_immutable_metadata(self):
  path=self.root/'result.json';f.once(path,dict(passed=False))
  with self.assertRaises(FileExistsError):f.once(path,dict(passed=True))
  self.assertFalse(json.loads(path.read_text())['passed'])

class Lifecycle(unittest.TestCase):
 def test_native_process_creation_rejected(self):
  barrier=f.NoNativeChildren()
  for event in ('subprocess.Popen','os.fork','os.exec','os.posix_spawn'):
   with self.subTest(event=event):
    with self.assertRaises(RuntimeError):barrier.audit(event,())
  barrier.audit('open',());barrier.active=False;barrier.audit('subprocess.Popen',())
 def test_native_children_prevent_clean_release(self):
  with patch.object(f,'own_children',return_value=[500001]):
   with self.assertRaises(RuntimeError):f.NoNativeChildren().verify()
 def test_clean_drain_never_signals(self):
  with patch.object(f,'own_children',return_value=[]),patch.object(f.os,'pidfd_open')as opened:
   f.drain_children();opened.assert_not_called()
 def test_unresolved_child_drain_fails(self):
  with patch.object(f,'own_children',return_value=[500001]),patch.object(f.os,'pidfd_open')as opened:
   with self.assertRaises(RuntimeError):f.drain_children(0)
   opened.assert_not_called()
 def test_owned_child_drain_uses_pidfd(self):
  with patch.object(f,'own_children',side_effect=[[500001],[500001],[]]),patch.object(f.os,'pidfd_open',return_value=91)as opened,patch.object(f.signal,'pidfd_send_signal')as killed,patch.object(f.os,'close')as closed,patch.object(f.os,'waitpid',return_value=(500001,9)),patch.object(f.time,'sleep'):
   f.drain_children();opened.assert_called_once_with(500001);killed.assert_called_once_with(91,f.signal.SIGKILL);closed.assert_called_once_with(91)
 def test_source_closure(self):
  paths=f.source_paths()
  for name in ('nemotron9b_package_controller_v14.py','nemotron9b_package_protocol_v14.py','nemotron9b_package_window_v14.py','minicpm5_layer0_execution_protocol_v4.py'):
   self.assertIn(str(f.R/name),paths)
  self.assertIn(str(Path(f.__file__)),paths)
  self.assertNotIn('torch',sys.modules);self.assertNotIn('numpy',sys.modules)
 def test_no_terminal_outputs_created(self):
  self.assertFalse(f.history.CURRENT.exists())

if __name__=='__main__':unittest.main(verbosity=2)
