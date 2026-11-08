import copy,hashlib,sys,unittest,types
from pathlib import Path
from unittest.mock import patch
R=Path(__file__).resolve().parents[1];Q=R/'research';sys.path.insert(0,str(R));sys.path.insert(0,str(Q))
import finalize_nemotron9b_package_r6_root_v1 as f
import nemotron9b_package_r6_completed_history_v1 as h
import nemotron9b_r4_terminal_metadata_v2 as m
import nemotron9b_package_protocol_v14 as p
SPEC,REVIEW,LIVE=f.setup();PRIOR=h.prior.load(h.PRIOR)
def fixture():
 result=copy.deepcopy(PRIOR)
 rows=[dict(unit=SPEC['units'][stage],synthetic=True)for stage in ['controller',*p.STAGES]]
 final=dict(pid=500001,start_ticks='900001',ppid=1,argv=['fixture-finalizer'],cgroup='fixture')
 processes=[dict(LIVE['coordinator_owner'],role='nemotron9b_package_r6_external_launcher'),dict(final,role='nemotron9b_package_r6_finalizer',next_admission_must_prove_gone=True)]
 release=dict(path='/fixture/release',sha256='a'*64);audit=dict(path='/fixture/audit',sha256='b'*64);exit=dict(path='/fixture/exit',sha256='c'*64);out=dict(producer_qualified=False)
 result['rows']+=rows;result['processes']+=processes;result['root_release_refs'].append(release)
 if 'process_predecessors'in result:result['process_predecessors']=copy.deepcopy(result['processes'])
 result.update(schema=h.SCHEMA,complete_count=978,appended_from=h.PRIOR,appended_terminal_audit=audit,latest_producer='nemotron9b_package_r6',latest_producer_qualified=False,latest_producer_exit=exit,latest_producer_terminal_audit=audit,root_finalized=True,all_planned_launches_accounted_for=True,
  nemotron9b_package_r6_append=dict(services=rows,processes=processes,actual_exit=exit,terminal_review=audit,release=release,execution_spec=h.SPEC,source_review=h.REVIEW,launch=h.LAUNCH,all_planned_services=53,finalizer_owner=final))
 return result,release,audit,exit,out,final
class Append(unittest.TestCase):
 def setUp(self):self.value,self.release,self.audit,self.exit,self.out,self.final=fixture()
 def check(self):return h.validate_extension(PRIOR,self.value,SPEC,self.release,self.audit,self.exit,self.out,self.final)
 def reject(self):
  with self.assertRaises(RuntimeError):self.check()
 def test_exact_shape(self):self.assertTrue(self.check())
 def test_prior_row_drop(self):self.value['rows'].pop(0);self.reject()
 def test_new_stage_drop(self):self.value['rows'].pop();self.reject()
 def test_reorder(self):self.value['rows'][-1],self.value['rows'][-2]=self.value['rows'][-2],self.value['rows'][-1];self.reject()
 def test_process_drop(self):self.value['processes'].pop();self.reject()
 def test_release_rewrite(self):self.value['root_release_refs'].pop();self.reject()
 def test_source_rebind(self):self.value['source_sha256'][next(iter(PRIOR['source_sha256']))]='0'*64;self.reject()
 def test_qualification_promotion(self):self.value['latest_producer_qualified']=True;self.reject()
 def test_extra_field(self):self.value['invented']=True;self.reject()
class Builder(unittest.TestCase):
 def test_no_real_exit_no_output(self):
  with self.assertRaises(RuntimeError):f.main(types.SimpleNamespace(actual_exit=None,actual_exit_sha256=None,recovery_receipt=[]))
  self.assertFalse(f.AUDIT.exists());self.assertFalse(f.CATALOG.exists())
 def test_source_hashes_and_original(self):
  self.assertEqual(SPEC['stages'],list(p.STAGES));self.assertEqual(len(PRIOR['rows']),925);self.assertEqual(len(PRIOR['processes']),182)
 def test_current_check_requires_only_real_retained_rows(self):
  value=f.check_current();self.assertEqual(value['execution_spec'],f.SPEC);self.assertEqual(value['source_review'],f.REVIEW);self.assertFalse(value['terminal_finalization'])
 def test_wrong_launch_owner_rejected(self):
  receipt=dict(schema='nemotron9b_package_r6_actual_external_exit_v1',reviewer='/root',actual_tool_exit_observed=True,execution_spec=f.SPEC,source_review=f.REVIEW,owner=dict(LIVE['coordinator_owner'],pid=1),exact_owner_gone=True,actual_exit=dict(session_id=46838,exit_code=1,tool_chunk='abcdef'))
  with patch.object(m,'load',return_value=receipt):
   with self.assertRaises(RuntimeError):f.main(types.SimpleNamespace(actual_exit='/fixture/exit',actual_exit_sha256='d'*64,recovery_receipt=[]))
if __name__=='__main__':unittest.main(verbosity=2)
