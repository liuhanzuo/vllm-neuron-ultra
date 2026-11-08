"""Reader integration preserves the prior title fix and all experiment checks."""
from pathlib import Path
import ast,hashlib,importlib,json,sys,unittest
R=Path(__file__).resolve().parents[1];Q=R/'research';sys.path.insert(0,str(R))
import smollm3_observer_invariance_protocol_v19 as old
import smollm3_observer_invariance_protocol_v20 as new
import smollm3_observer_invariance_lifecycle_v24 as life
manifest=json.loads((Q/'smollm3-sio14-runtime-source-candidate-20261108-v2.json').read_text())
class Integration(unittest.TestCase):
 def test_all_hashes_and_exact_source_delta(self):
  for row in manifest['files']:
   for ref in [row,row['parent']]:self.assertEqual(hashlib.sha256(Path(ref['path']).read_bytes()).hexdigest(),ref['sha256'])
   text=Path(row['parent']['path']).read_text()
   for a,b in manifest['replacements'].items():text=text.replace(a,b)
   if Path(row['path']).name=='smollm3_observer_invariance_protocol_v20.py':text=text.replace("'nemotron9b_package_metadata_v7.py',","'nemotron9b_package_metadata_v7.py','nemotron9b_package_metadata_v10.py',\n                 'nemotron9b_completed_smol_metadata_v1.py',")
   self.assertEqual(text,Path(row['path']).read_text())
 def test_all_runtime_peers_bind_to_same_generation(self):
  modules={Path(x['path']).stem:importlib.import_module(Path(x['path']).stem)for x in manifest['files']}
  self.assertIs(life.p,new);self.assertIs(new.title_runtime.lifecycle(),life)
  self.assertIs(life.witness.title_runtime,new.title_runtime)
  for m in modules.values():
   if hasattr(m,'p'):self.assertIs(m.p,new)
  self.assertEqual(new.source_metadata.__name__,'smollm3_observer_metadata_source_v4')
  self.assertEqual(new.source_metadata.inherited.__name__,'nemotron9b_package_metadata_v10')
 def test_no_native_or_runtime_admission(self):
  self.assertFalse(new.native_loaded());self.assertFalse(new.DEST.exists());self.assertEqual(new.DEST,old.DEST)
 def test_entire_scientific_template_unchanged(self):
  base=new.load_base();self.assertEqual(new.template(base),old.template(base));self.assertEqual(new.matrix(),old.matrix());self.assertEqual(new.title_runtime.title.DEADLINE_SECONDS,15)
 def test_full_guard_state_machine_and_durable_writer_unchanged(self):
  def node(path):return next(x for x in ast.parse(path.read_text()).body if isinstance(x,ast.ClassDef)and x.name=='RuntimeContext')
  self.assertEqual(ast.dump(node(R/'smollm3_observer_invariance_protocol_v19.py')),ast.dump(node(R/'smollm3_observer_invariance_protocol_v20.py')))
 def test_transport_closure_requires_new_reader_dependencies(self):
  base=new.load_base();value=new.template(base);pins=dict(base['source_sha256']);pins[str(new.BASE)]=new.BASE_SHA
  for path in R.glob('*.py'):pins[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
  value['source_sha256']=pins;new.validate_template(value,base)
  for name in ['smollm3_observer_metadata_source_v4.py','nemotron9b_package_metadata_v10.py','nemotron9b_completed_smol_metadata_v1.py']:
   removed=pins.pop(str(R/name))
   with self.subTest(name=name),self.assertRaises(RuntimeError):new.validate_template(value,base)
   pins[str(R/name)]=removed
if __name__=='__main__':unittest.main(verbosity=2)
