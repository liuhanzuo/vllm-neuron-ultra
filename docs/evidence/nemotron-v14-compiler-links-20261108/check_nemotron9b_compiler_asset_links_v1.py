import copy,json,os,socket,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import nemotron9b_compiler_asset_links_v1 as links
import nemotron9b_package_storage_inventory_candidate_v1 as old
import nemotron9b_package_storage_inventory_candidate_v2 as new
class Checks(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
  self.work=Path(self.temp.name)/'work';self.compiler=self.work/'compile-work';self.group=self.compiler/'token_generation_model/_tp0_bk0/neuronxcc-fixture0/sg00';self.group.mkdir(parents=True)
  self.policy=links.load_policy()
  for name,row in self.policy['assets'].items():(self.group/name).symlink_to(row['path'])
 def inventory(self,root=None):return new.inventory(root or self.work,compiler_root=self.compiler,compiler_assets=self.policy)
 def test_actual_installed_ten_assets_and_content(self):
  self.assertEqual(len(self.policy['assets']),10);self.assertEqual(self.policy['target_bytes'],228068)
  self.assertTrue(links.verify_assets(self.policy,hash_content=True))
 def test_old_failure_and_new_complete_work(self):
  with self.assertRaisesRegex(RuntimeError,'Only prepared model view symlinks'):old.inventory(self.work)
  self.assertEqual(self.inventory(),sum(max(p.lstat().st_size,p.lstat().st_blocks*512)for p in self.group.iterdir()))
 def test_compile_category_same_accounting(self):self.assertEqual(self.inventory(),self.inventory(self.compiler))
 def test_record_binds_actual_link_and_pinned_target(self):
  p=self.group/'act_info.json';row=links.link_record(p,self.compiler,self.policy)
  self.assertEqual(row['target'],self.policy['assets'][p.name]);self.assertEqual(row['stat'],links.attributes(p.lstat()));self.assertTrue(row['compiler_asset'])
 def test_undeclared_context_rejected(self):
  with self.assertRaises(RuntimeError):new.inventory(self.work)
 def test_partial_context_rejected(self):
  with self.assertRaises(RuntimeError):new.inventory(self.work,compiler_root=self.compiler)
 def test_extra_asset_policy_rejected(self):
  value=copy.deepcopy(self.policy);value['assets']['extra.json']=value['assets']['act_info.json']
  with self.assertRaises(RuntimeError):links.verify_assets(value)
 def test_target_pin_changed_rejected(self):
  value=copy.deepcopy(self.policy);value['assets']['act_info.json']['sha256']='0'*64
  with self.assertRaises(RuntimeError):links.verify_assets(value)
 def test_wrong_absolute_target_rejected(self):
  p=self.group/'act_info.json';p.unlink();p.symlink_to(self.policy['assets']['silu_and_others.json']['path'])
  with self.assertRaises(RuntimeError):self.inventory()
 def test_relative_target_rejected(self):
  p=self.group/'act_info.json';p.unlink();p.symlink_to(os.path.relpath(self.policy['assets']['act_info.json']['path'],p.parent))
  with self.assertRaises(RuntimeError):self.inventory()
 def test_unreviewed_name_rejected(self):
  (self.group/'unreviewed.bin').symlink_to(self.policy['assets']['act_info.json']['path'])
  with self.assertRaises(RuntimeError):self.inventory()
 def test_link_outside_compile_scope_rejected(self):
  (self.work/'act_info.json').symlink_to(self.policy['assets']['act_info.json']['path'])
  with self.assertRaises(RuntimeError):self.inventory()
 def test_wrong_graph_or_bucket_rejected(self):
  for parent in ['unreviewed_model/_tp0_bk0/neuronxcc-fixture0/sg00','token_generation_model/_tp2_bk0/neuronxcc-fixture0/sg00','token_generation_model/_tp0_bk6/neuronxcc-fixture0/sg00']:
   with self.subTest(parent=parent):
    target=self.compiler/parent;target.mkdir(parents=True);p=target/'act_info.json';p.symlink_to(self.policy['assets'][p.name]['path'])
    with self.assertRaises(RuntimeError):self.inventory()
    p.unlink()
 def test_directory_symlink_rejected(self):
  (self.work/'directory').symlink_to(self.group,target_is_directory=True)
  with self.assertRaises(RuntimeError):self.inventory()
 def test_link_swapped_during_observation_rejected(self):
  p=self.group/'act_info.json';real=os.readlink;calls=[]
  def readlink(path,*args,**kwargs):
   value=real(path,*args,**kwargs)
   if Path(path)==p:
    calls.append(1)
    if len(calls)==1:p.unlink();p.symlink_to(self.policy['assets']['silu_and_others.json']['path'])
   return value
  with patch.object(links.os,'readlink',side_effect=readlink):
   with self.assertRaises(RuntimeError):links.link_record(p,self.compiler,self.policy)
 def test_full_content_verification_not_only_stat(self):
  with patch.object(links.os,'read',side_effect=[b'x',b'']):
   with self.assertRaises(RuntimeError):links.verify_assets(self.policy,hash_content=True)
 def test_regular_file_and_hardlink_rules_retained(self):
  p=self.work/'regular';p.write_bytes(b'x'*8192);baseline=self.inventory();self.assertGreaterEqual(baseline,8192)
  os.link(p,self.work/'hardlink')
  with self.assertRaises(RuntimeError):self.inventory()
 def test_original_temporary_socket_rule_retained(self):
  tmp=self.work/'temporary/tmp';tmp.mkdir(parents=True);sock=socket.socket(socket.AF_UNIX);self.addCleanup(sock.close)
  sock.bind(str(tmp/'12345678-1234-1234-1234-123456789abc'))
  new.inventory(self.work,ipc_root=tmp,compiler_root=self.compiler,compiler_assets=self.policy)
  with self.assertRaises(RuntimeError):self.inventory()
if __name__=='__main__':unittest.main(verbosity=2)
