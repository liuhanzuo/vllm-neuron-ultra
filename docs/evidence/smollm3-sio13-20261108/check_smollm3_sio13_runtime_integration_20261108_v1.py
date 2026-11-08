"""Check the migrated runtime before importing any native inference library."""
import ast,hashlib,json,re,sys,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
Q=Path(__file__).resolve().parent;R=Q.parent;sys.path[:0]=[str(Q),str(R)]
import smollm3_observer_invariance_protocol_v18 as p
import smollm3_observer_invariance_protocol_v17 as old
import smollm3_observer_invariance_lifecycle_v22 as life
import smollm3_observer_invariance_worker_v18 as worker
class StopBeforeNative(Exception):pass
class Integration(unittest.TestCase):
 def test_original_science_only_namespace_changes(self):
  a=old.template(old.load_base());b=p.template(p.load_base())
  for field in ['work','ipc_layout']:a.pop(field);b.pop(field)
  self.assertEqual(a,b);self.assertEqual(p.matrix(),old.matrix())
  self.assertEqual(p.DEST,R/'sio13');self.assertFalse(p.DEST.exists())
  for row in p.matrix():
   for kind in ['server','client']:
    self.assertNotEqual(p.unit(row['run_id'],kind),old.unit(row['run_id'],kind))
    self.assertEqual(p.resource(kind),old.resource(kind))
 def test_complete_native_setup_source_phase_with44_real_files(self):
  base=p.load_base();wheelpath=R/'smollm3-cache-ablation-native-v3/wheel.json';seen=[]
  def advance(stage):self.assertEqual(stage,'import');raise StopBeforeNative()
  context=SimpleNamespace(row=p.matrix()[0],base=base,plan={'source_sha256':{str(wheelpath):hashlib.sha256(wheelpath.read_bytes()).hexdigest()}},poll=lambda:None,advance_environment=advance)
  original=p.bounded_source
  def read(path,wanted=None):seen.append(str(path));return original(path,wanted)
  with patch.object(p,'bounded_source',side_effect=read),self.assertRaises(StopBeforeNative):worker.native_setup(context)
  wheel=json.loads(wheelpath.read_text());self.assertEqual(seen,[str(wheelpath)]+[str(p.TARGET/n)for n in wheel['installed_files']])
  self.assertEqual(len(wheel['installed_files']),44)
  self.assertFalse(any(n in sys.modules for n in ['torch','numpy','vllm','vllm_neuron','neuronxcc']))
 def test_nine_runtime_modules_only_declared_migration(self):
  m=json.loads((Q/'smollm3-sio13-source-migration-20261108-v1.json').read_text());mapping=m['mapping'];pattern=re.compile('|'.join(re.escape(k)for k in sorted(mapping,key=len,reverse=True)))
  for a,b in m['runtime_pairs']:
   parent=R/(a+'.py');self.assertEqual(hashlib.sha256(parent.read_bytes()).hexdigest(),m['frozen_parent_sha256'][str(parent)])
   expected=pattern.sub(lambda match:mapping[match.group()],parent.read_text()).replace("ROOT/'sio12'","ROOT/'sio13'").replace('Fresh sio12','Fresh sio13').replace('Frozen sio12','Frozen sio13').replace('frozen sio12','frozen sio13').replace('independent sio12','independent sio13').replace('reuse sio12','reuse sio13')
   with self.subTest(source=b):self.assertEqual(expected,(R/(b+'.py')).read_text());ast.parse(expected)
 def test_lifecycle_and_worker_share_same_protocol_and_reader(self):
  self.assertIs(life.p,p);self.assertIs(worker.p,p)
  self.assertEqual(Path(p.source_metadata.__file__).name,'smollm3_observer_metadata_source_v3.py')
  self.assertEqual(life.UNIT,'trn1-latest-verify-smollm3-observer-invariance-v14-supervisor')
if __name__=='__main__':unittest.main(verbosity=2)
