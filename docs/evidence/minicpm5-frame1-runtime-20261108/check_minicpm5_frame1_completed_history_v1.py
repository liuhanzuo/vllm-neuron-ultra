import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
R=Path('/home/ubuntu/experiments/trn1-latest-models-20261031');sys.path.insert(0,str(R))
import minicpm5_frame1_completed_history_v1 as m
class Checks(unittest.TestCase):
 def test_actual_full_catalog(self):
  row=m.load(dict(path=str(m.PATH),sha256=m.SHA));self.assertEqual(len(row['rows']),919);self.assertEqual(len(row['processes']),170)
 def test_wrong_hash(self):
  with self.assertRaises(RuntimeError):m.load(dict(path=str(m.PATH),sha256='0'*64))
 def test_wrong_size(self):
  with self.assertRaises(RuntimeError):m.load(dict(path=str(m.PATH),sha256=m.SHA,bytes=m.SIZE-1))
 def test_extra_field(self):
  with self.assertRaises(RuntimeError):m.load(dict(path=str(m.PATH),sha256=m.SHA,skip=True))
 def test_ordinary_limit_retained(self):
  self.assertEqual(m.original.LIMIT,8*2**20);self.assertEqual(m.original.ENVELOPE_LIMIT,32*2**20)
  with tempfile.TemporaryDirectory()as directory:
   path=Path(directory)/'ordinary.json';path.write_bytes(b' '* (8*2**20+1))
   with self.assertRaisesRegex(RuntimeError,'bounded metadata'):m.text(path)
 def test_ordinary_text_delegation(self):
  path=R/'minicpm5_frame1_mlp_worker_v2.py';raw=path.read_bytes();self.assertEqual(m.text(path,hashlib.sha256(raw).hexdigest()),raw)
 def test_raw_payload_rejected(self):
  with self.assertRaisesRegex(RuntimeError,'Reviewed UTF8 text'):m.text(Path('/tmp/never-open-this.safetensors'))
 def test_only_history_import_changed(self):
  before=(R/'minicpm5_frame1_mlp_runner_v3.py').read_text().replace('import nemotron9b_package_metadata_v13 as history_metadata','import minicpm5_frame1_completed_history_v1 as history_metadata')
  self.assertEqual(before,(R/'minicpm5_frame1_mlp_runner_v4.py').read_text())
 def test_real_fresh_predecessor_then_current_history(self):
  code="""import sys,json
sys.path.insert(0,'/home/ubuntu/experiments/trn1-latest-models-20261031')
import minicpm5_frame1_mlp_runner_v4 as r
assert 'smollm3_observer_invariance_environment_v2'not in sys.modules
b,o,p=r.old.check_prerequisites(r.p.read_ref(r.BASE_SPEC))
r.load_history_helpers()
c=r.history_metadata.load(dict(path=str(r.history_metadata.PATH),sha256=r.history_metadata.SHA))
assert len(c['rows'])==919 and 'torch'not in sys.modules
print(json.dumps(dict(actual_old_prerequisites=True,owners=len(o),current_services=919,native_imported=False)))
"""
  p=subprocess.run([sys.executable,'-B','-c',code],cwd=R,text=True,capture_output=True,timeout=120)
  self.assertEqual(p.returncode,0,p.stdout+p.stderr);print(p.stdout,flush=True)
if __name__=='__main__':unittest.main(verbosity=2)
