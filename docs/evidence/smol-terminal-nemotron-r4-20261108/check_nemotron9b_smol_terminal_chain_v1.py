"""Actual completed sio13/sio12 metadata and negative admission controls only."""
import copy, hashlib, json, sys, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import nemotron9b_package_metadata_v9 as metadata
import nemotron9b_smol_terminal_chain_v1 as chain
CATALOG=dict(path=str(ROOT/'research/root-post-smol-sio13-complete-predecessors-20261108-v1.json'),
 sha256='b29dca3e250b1d3aafb9a9dbdabad210631966d3f76400e392fbdb97b33e96b6')
DOCUMENTS={}
def loaded(ref):
 if ref['path'] not in DOCUMENTS:DOCUMENTS[ref['path']]=metadata.load(ref)
 return DOCUMENTS[ref['path']]
CURRENT=loaded(CATALOG);selected=[];current=CURRENT;ref=CATALOG
while chain.catalog_number(ref['path'])is not None:
 audit_ref=current['appended_terminal_audit'];audit=loaded(audit_ref)
 item=dict(catalog=ref,terminal_audit=audit_ref,**{k:audit['references'][k]for k in chain.FIELDS})
 selected.append(item)
 for k in chain.FIELDS:loaded(item[k])
 ref=current['appended_from'];current=loaded(ref)
SPEC=dict(final_predecessor_catalog=CATALOG,source_sha256=dict(CURRENT['source_sha256']),
 required_smol_predecessor_chain=selected,required_predecessor_releases=CURRENT['root_release_refs'])
SPEC['source_sha256'][CATALOG['path']]=CATALOG['sha256']

class Checks(unittest.TestCase):
 def setUp(self):self.spec=copy.deepcopy(SPEC);self.catalog=CURRENT;self.changed={}
 def document(self,key,index=0):
  ref=self.spec['required_smol_predecessor_chain'][index][key]
  if ref['path']not in self.changed:self.changed[ref['path']]=copy.deepcopy(loaded(ref))
  return self.changed[ref['path']]
 def run_chain(self):return chain.validate(self.spec,self.catalog,lambda r:self.changed.get(r['path'],loaded(r)))
 def rejected(self):
  with self.assertRaises((RuntimeError,KeyError,TypeError)):self.run_chain()
 def test_actual_failed13_and12_preserved(self):
  result=self.run_chain();self.assertEqual(len(result['producers']),2)
  self.assertEqual([x['original_exit_code']for x in result['producers']],[1,1])
  self.assertFalse(result['fresh_physical_drain_proven']);self.assertFalse(result['scientific_success_inferred'])
 def test_no_chain(self):self.spec.pop('required_smol_predecessor_chain');self.rejected()
 def test_omitted_latest(self):self.spec['required_smol_predecessor_chain']=selected[1:];self.rejected()
 def test_omitted_prior(self):self.spec['required_smol_predecessor_chain']=selected[:1];self.rejected()
 def test_missing_audit_pin(self):self.spec['source_sha256'].pop(selected[0]['terminal_audit']['path']);self.rejected()
 def test_missing_terminal_document_pin(self):self.spec['source_sha256'].pop(selected[0]['terminal']['path']);self.rejected()
 def test_unfinalized_audit(self):self.document('terminal_audit')['passed']=False;self.rejected()
 def test_original_failure_promoted(self):self.document('terminal_audit')['experiment_qualified']=True;self.rejected()
 def test_original_error_rewritten(self):self.document('controller')['original_error']=None;self.rejected()
 def test_original_exit_promoted(self):self.document('exited')['exit_code']=0;self.rejected()
 def test_fake_tool_completion(self):self.document('exited')['execution_handle']['actual_completion_observed']=False;self.rejected()
 def test_different_tool_session(self):self.document('exited')['execution_handle']['session_id']+=1;self.rejected()
 def test_different_owner(self):self.document('release')['external_owner']['start_ticks']='1';self.rejected()
 def test_release_schema_relabel(self):v=self.document('release');v['all_current_groups_recursively_drained']=v.pop('all_planned_groups_drained');self.rejected()
 def test_release_failed_flag(self):self.document('release')['failed']=False;self.rejected()
 def test_model_speedup_inferred(self):self.document('terminal_audit')['new_speedup_claim']=True;self.rejected()
 def test_old_service_removed(self):self.catalog=copy.deepcopy(CURRENT);self.catalog['rows'].pop(0);self.rejected()
 def test_old_release_omitted(self):self.spec['required_predecessor_releases']=self.spec['required_predecessor_releases'][1:];self.rejected()
 def test_prior_failure_promoted(self):self.document('terminal_audit',1)['experiment_qualified']=True;self.rejected()
 def test_stale_inherited_latest_ignored(self):
  self.catalog=copy.deepcopy(CURRENT);self.catalog['latest_producer_qualified']=True
  result=self.run_chain();self.assertTrue(all(not x['experiment_qualified']for x in result['producers']))
if __name__=='__main__':unittest.main(verbosity=2)
