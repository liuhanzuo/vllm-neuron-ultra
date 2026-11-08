"""Real stdlib socket fixtures plus retained r4 output, no native/model launch."""
from pathlib import Path
import os,socket,sys,tempfile,unittest,uuid
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import nemotron9b_package_storage_inventory_candidate_v1 as c
import nemotron9b_package_protocol_v10 as old
class Inventory(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory(prefix='ni-');self.root=Path(self.temp.name);self.ipc=self.root/'temporary/tmp';self.ipc.mkdir(parents=True);self.sockets=[]
 def tearDown(self):
  for s in self.sockets:s.close()
  self.temp.cleanup()
 def bind(self,path):
  s=socket.socket(socket.AF_UNIX);s.bind(str(path));self.sockets.append(s)
 def test_original_rejects_actual_uuid_socket_candidate_accepts_only_explicitly(self):
  self.bind(self.ipc/str(uuid.uuid4()))
  with self.assertRaisesRegex(RuntimeError,'Ordinary new output file'):old._usage_once(self.root,False)
  with self.assertRaisesRegex(RuntimeError,'declared temporary'):c.inventory(self.root)
  self.assertEqual(c.inventory(self.root,ipc_root=self.ipc),0);self.assertEqual(c.inventory(self.root/'temporary',ipc_root=self.ipc),0)
 def test_identical_actual_current_output_totals(self):
  work=Path('/mnt/nvme/nemotron9b-package-r4-20261108')
  for root in [work,work/'temporary']:
   self.assertEqual(c.inventory(root,ipc_root=work/'temporary/tmp'),old._usage_once(root,False))
 def test_regular_file_accounting_unchanged(self):
  (self.root/'data.json').write_text('metadata')
  self.assertEqual(c.inventory(self.root,ipc_root=self.ipc),old._usage_once(self.root,False))
 def test_socket_wrong_directory_or_name_rejected(self):
  for path in [self.root/str(uuid.uuid4()),self.ipc/'unexpected']:
   self.bind(path)
   with self.assertRaises(RuntimeError):c.inventory(self.root,ipc_root=self.ipc)
   path.unlink()
 def test_fifo_rejected(self):
  os.mkfifo(self.ipc/str(uuid.uuid4()))
  with self.assertRaisesRegex(RuntimeError,'Ordinary new output file'):c.inventory(self.root,ipc_root=self.ipc)
 def test_hardlinked_regular_file_rejected(self):
  p=self.root/'a.json';p.write_text('x');os.link(p,self.root/'b.json')
  with self.assertRaisesRegex(RuntimeError,'Ordinary new output file'):c.inventory(self.root,ipc_root=self.ipc)
 def test_directory_symlink_rejected(self):
  (self.root/'link').symlink_to(self.ipc,target_is_directory=True)
  with self.assertRaisesRegex(RuntimeError,'directory symlinks'):c.inventory(self.root,ipc_root=self.ipc)
 def test_undeclared_file_symlink_rejected(self):
  p=self.root/'a.json';p.write_text('x');(self.root/'other').symlink_to(p)
  with self.assertRaisesRegex(RuntimeError,'model view symlinks'):c.inventory(self.root,ipc_root=self.ipc)
 def test_original_model_view_rule_preserved(self):
  (self.root/'view').mkdir();(self.root/'view/link').symlink_to('/nonexistent-fixture')
  self.assertEqual(c.inventory(self.root,True),old._usage_once(self.root,True))
 def test_original_log_cap_preserved(self):
  with (self.root/'large.log').open('wb')as f:f.truncate(64*2**20+1)
  with self.assertRaisesRegex(RuntimeError,'Per-file'):c.inventory(self.root,ipc_root=self.ipc)
 def test_unrelated_inventory_cannot_enable_socket_exception(self):
  unrelated=self.root/'artifact';unrelated.mkdir()
  with self.assertRaisesRegex(RuntimeError,'complete work or temporary'):c.inventory(unrelated,ipc_root=self.ipc)
if __name__=='__main__':unittest.main(verbosity=2)
