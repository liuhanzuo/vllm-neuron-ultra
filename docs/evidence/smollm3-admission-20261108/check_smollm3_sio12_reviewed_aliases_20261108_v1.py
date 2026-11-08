"""Exercise frozen-handoff loading with exact aliases and fail-closed mutations."""
import ast,copy,hashlib,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
Q=Path(__file__).resolve().parent;R=Q.parent;sys.path[:0]=[str(Q),str(R)]
import smollm3_observer_invariance_lifecycle_v21 as life
class Aliases(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
  self.target=self.root/'metadata.json';self.target.write_bytes(b'{"checked":true}')
  self.alias=self.root/'alias.json';self.alias.symlink_to(self.target)
  self.sha=hashlib.sha256(self.target.read_bytes()).hexdigest()
  self.pins={str(self.alias):self.sha};self.aliases={str(self.alias):dict(target=str(self.target),sha256=self.sha)}
 def tearDown(self):self.tmp.cleanup()
 def load(self):
  handoff=dict(source_template={'path':str(life.TEMPLATE),'sha256':'a'*64},source_frozen=True,source_sha256=self.pins,original_metadata_symlink_resolutions=self.aliases)
  spec=dict(source_template=handoff['source_template'],final_predecessor_catalog='catalog',callable_handoff='handoff',callable_root_acceptance='accepted',runtime_source_sha256=dict(self.pins),historical_payload_sha256={})
  accepted=dict(reviewer='/root',passed=True,independent_review=True,source_handoff='handoff',blockers=[])
  def reference(ref):
   if isinstance(ref,dict)and ref.get('path')=='spec':return spec
   if ref==spec['source_template']:return {}
   return {'catalog':{},'handoff':handoff,'accepted':accepted}[ref]
  with patch.object(life,'reference',side_effect=reference),patch.object(life,'template_reference',side_effect=lambda x:x),patch.object(life.p,'load_base',return_value={}),patch.object(life,'validate_spec'),patch.object(life.historical,'partition',return_value=dict(source_sha256={},provenance_sha256={},historical_payload_sha256={})):
   return life.load_spec('spec','a'*64)
 def test_reviewed_alias_passes_real_load_spec_byte_check(self):self.assertEqual(self.load()[3]['source_sha256'],self.pins)
 def test_ordinary_reader_still_rejects_alias(self):
  with self.assertRaises(RuntimeError):life.p.bounded_source(self.alias,self.sha)
 def test_redirected_alias_same_bytes_rejected(self):
  other=self.root/'other.json';other.write_bytes(self.target.read_bytes());self.alias.unlink();self.alias.symlink_to(other)
  with self.assertRaises(RuntimeError):self.load()
 def test_wrong_recorded_hash_rejected(self):
  self.aliases[str(self.alias)]['sha256']='0'*64
  with self.assertRaises(RuntimeError):self.load()
 def test_unreviewed_alias_rejected(self):
  self.aliases={}
  with self.assertRaises(RuntimeError):self.load()
 def test_missing_target_rejected(self):
  self.target.unlink()
  with self.assertRaises((RuntimeError,FileNotFoundError)):self.load()
 def test_changed_target_bytes_rejected(self):
  self.target.write_bytes(b'{"checked":false}')
  with self.assertRaises(RuntimeError):self.load()
 def test_alias_replaced_by_ordinary_file_rejected(self):
  self.alias.unlink();self.alias.write_bytes(self.target.read_bytes())
  with self.assertRaises(RuntimeError):self.load()
 def test_target_alias_rejected(self):
  other=self.root/'other.json';other.write_bytes(self.target.read_bytes());self.target.unlink();self.target.symlink_to(other)
  with self.assertRaises(RuntimeError):self.load()
 def test_ordinary_source_passes(self):
  self.pins={str(self.target):self.sha};self.aliases={};self.assertEqual(self.load()[3]['source_sha256'],self.pins)
 def test_all_four_real_reviewed_aliases(self):
  handoff=json.loads((Q/'smollm3-sio12-metadata-source-handoff-20261108-v1.json').read_text());self.aliases=handoff['original_metadata_symlink_resolutions'];self.assertEqual(len(self.aliases),4)
  self.pins={k:handoff['source_sha256'][k]for k in self.aliases};self.assertEqual(len(self.load()[3]['source_sha256']),4)
class Preservation(unittest.TestCase):
 def test_successors_have_only_reviewed_changes(self):
  migration=json.loads((Q/'smollm3-sio12-reviewed-alias-migration-20261108-v1.json').read_text());mapping=migration['mapping']
  import re
  pattern=re.compile('|'.join(re.escape(k)for k in sorted(mapping,key=len,reverse=True)))
  for old,new in mapping.items():
   if not (R/(old+'.py')).exists()and not(Q/(old+'.py')).exists():continue
   root=R if (R/(old+'.py')).exists()else Q
   expected=pattern.sub(lambda m:mapping[m.group()],(root/(old+'.py')).read_text())
   if old=='smollm3_observer_invariance_lifecycle_v20':expected=expected.replace(migration['old_load_block'],migration['new_load_block'])
   with self.subTest(source=old):self.assertEqual(expected,(root/(new+'.py')).read_text());ast.parse(expected)
 def test_scientific_matrix_and_resource_constants_unchanged(self):
  import smollm3_observer_invariance_protocol_v16 as old
  self.assertEqual(old.template(old.load_base()),life.p.template(life.p.load_base()))
if __name__=='__main__':unittest.main(verbosity=2)
