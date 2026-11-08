"""Source integration checks; no service, native framework or inference launch."""
from pathlib import Path
import ast,hashlib,importlib,json,sys,unittest
R=Path(__file__).resolve().parents[1];Q=R/'research';sys.path.insert(0,str(R))
import smollm3_observer_invariance_protocol_v18 as original
import smollm3_observer_invariance_protocol_v19 as candidate
import smollm3_observer_invariance_lifecycle_v23 as life
import smollm3_enginecore_title_runtime_v12 as runtime
import smollm3_title_publication_candidate_v1 as helper
manifest=json.loads((Q/'smollm3-sio14-runtime-source-candidate-20261108-v1.json').read_text());replacements=manifest['replacements']
def renamed(text):
 for a,b in replacements.items():text=text.replace(a,b)
 return text
def node(source,name):return next(n for n in ast.parse(source).body if isinstance(n,(ast.ClassDef,ast.FunctionDef))and n.name==name)
class Integration(unittest.TestCase):
 def test_all_parent_and_candidate_hashes_exact(self):
  for row in manifest['files']:
   for x in [row,row['parent']]:self.assertEqual(hashlib.sha256(Path(x['path']).read_bytes()).hexdigest(),x['sha256'])
 def test_all_nine_modules_import_without_native_side_effects(self):
  for row in manifest['files']:importlib.import_module(Path(row['path']).stem)
  self.assertFalse(candidate.native_loaded());self.assertFalse(candidate.DEST.exists());self.assertEqual(len(manifest['files']),9)
 def test_actual_runtime_callback_bindings(self):
  self.assertIs(life.p,candidate);self.assertIs(life.title_runtime,runtime);self.assertIs(candidate.title_runtime,runtime);self.assertIs(runtime.lifecycle(),life)
  self.assertEqual(life.metadata_store.__name__,'smollm3_observer_metadata_store_v4');self.assertEqual(life.metadata_store.codec.__name__,'smollm3_observer_metadata_codec_v5')
  self.assertEqual(life.file_stat(Path(__file__)),life.stat_identity.file_stat(Path(__file__)))
 def test_science_resources_and_fixed_deadlines_preserved(self):
  self.assertEqual(candidate.matrix(),original.matrix());self.assertEqual(len(candidate.matrix()),12);self.assertEqual(sum(x['requests']for x in candidate.matrix()),264);self.assertEqual(sum(x['output_tokens']for x in candidate.matrix()),33696)
  for role in ['server','client']:self.assertEqual(candidate.resource(role),original.resource(role))
  for k in ['BASE_SHA','BASE_CANONICAL_SHA','OUTPUT_CAP','CONTROL_CAP','ROLES']:self.assertEqual(getattr(candidate,k),getattr(original,k))
  self.assertEqual(runtime.title.DEADLINE_SECONDS,15)
 def test_protocol_ordinary_checks_identical(self):
  a=node(renamed((R/'smollm3_observer_invariance_protocol_v18.py').read_text()),'RuntimeContext');b=node((R/'smollm3_observer_invariance_protocol_v19.py').read_text(),'RuntimeContext')
  self.assertEqual([ast.dump(n)for n in a.body],[ast.dump(n)for n in b.body if not(isinstance(n,ast.FunctionDef)and n.name=='record_engine_title_before')])
 def test_inserted_writer_is_tested_candidate(self):
  a=node((R/'smollm3_title_publication_candidate_v1.py').read_text().replace('runtime.','title_runtime.'),'record_engine_title_before');cls=node((R/'smollm3_observer_invariance_protocol_v19.py').read_text(),'RuntimeContext');b=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='record_engine_title_before');self.assertEqual(ast.dump(a),ast.dump(b))
 def test_other_module_code_only_namespace_and_import_changes(self):
  exempt=['smollm3_enginecore_title_runtime_v12.py','smollm3_observer_invariance_protocol_v19.py','smollm3_observer_invariance_lifecycle_v23.py']
  for row in manifest['files']:
   if Path(row['path']).name not in exempt:self.assertEqual(ast.dump(ast.parse(renamed(Path(row['parent']['path']).read_text()))),ast.dump(ast.parse(Path(row['path']).read_text())))
 def test_lifecycle_complete_checks_except_identity_implementation_identical(self):
  a=ast.parse(renamed((R/'smollm3_observer_invariance_lifecycle_v22.py').read_text()));b=ast.parse((R/'smollm3_observer_invariance_lifecycle_v23.py').read_text())
  def keep(tree):return [ast.dump(n)for n in tree.body if not(isinstance(n,ast.FunctionDef)and n.name=='file_stat')and not(isinstance(n,ast.Import)and any(x.name=='smollm3_stat_identity_v2'for x in n.names))]
  self.assertEqual(keep(a),keep(b))
 def test_title_runtime_is_tested_candidate_with_peer_imports(self):
  self.assertEqual(ast.dump(ast.parse(renamed((R/'smollm3_title_deadline_candidate_v1.py').read_text()))),ast.dump(ast.parse((R/'smollm3_enginecore_title_runtime_v12.py').read_text())))
if __name__=='__main__':unittest.main(verbosity=2)
