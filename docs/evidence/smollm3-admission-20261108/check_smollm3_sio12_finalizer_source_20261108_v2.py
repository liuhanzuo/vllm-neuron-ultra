"""Source/inert filesystem checks. No runtime, service, proc or device probes."""
import ast,copy,hashlib,json,os,socket,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
Q=Path(__file__).resolve().parent;R=Q.parent
sys.path[:0]=[str(Q),str(R)]
import finalize_smollm3_sio12_root_20261108_v2 as f
import check_smollm3_sio10_finalizer_source_20261107_v2 as old
import check_smollm3_sio12_catalog_transport_review_v3 as transport_tests
from smollm3_observer_metadata_store_v1 import provision

class Finalizer(unittest.TestCase):
 def test_outcomes_require_original_exit_and_keep_science_separate(self):
  with patch.object(f,'SESSION',51136):
   for pc,lc,inv in [(True,True,True),(True,True,False),(False,True,False),(True,False,False),(False,False,False)]:
    with self.subTest(pc=pc,lc=lc,inv=inv):self.assertEqual(f.validate_outcome(*old.fixture(pc,lc,inv)),(pc and lc,pc))
 def test_invalid_terminal_evidence_rejected(self):
  mutations=[lambda x:x[4].__setitem__('actual_exit_observed_by_root',False),lambda x:x[4].__setitem__('gone',False),lambda x:x[4].__setitem__('exit_code',False),lambda x:x[4].__setitem__('exit_code',1),lambda x:x[4]['execution_handle'].__setitem__('session_id',123),lambda x:x.__setitem__(5,'123456'),lambda x:x[1].__setitem__('released',False),lambda x:x[0].__setitem__('full_model_qualified',True),lambda x:x[0].__setitem__('new_speedup_claim',True),lambda x:x[0].__setitem__('cleanup_errors',['not drained']),lambda x:x[0].__setitem__('protocol_completed',False)]
  with patch.object(f,'SESSION',51136):
   for index,mutate in enumerate(mutations):
    value=old.fixture();mutate(value)
    with self.subTest(index=index),self.assertRaises(AssertionError):f.validate_outcome(*value)
 def test_preserved_scientific_and_drain_functions(self):
  def functions(path):return {n.name:ast.dump(n,include_attributes=False)for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
  a=functions(Q/'finalize_smollm3_sio10_root_20261107_v2.py');b=functions(Path(f.__file__))
  for name in ('record','validate_identity','physical_drain','validate_matrix_api'):self.assertEqual(a[name],b[name])
 def test_actual_old_full_byte_proof_survives_dynamic_source_count(self):
  spec=json.loads((Q/'smollm3-sio10-execution-spec-r1-20261107-v1.json').read_text())
  plan=json.loads(Path(spec['source_template']['path']).read_text());base=json.loads(Path(plan['base_plan']['path']).read_text());handoff=json.loads(Path(spec['callable_handoff']['path']).read_text())
  control=R/'sio10/control';source=json.loads((control/'source-admission.json').read_text());payload=json.loads((control/'payload-admission.json').read_text());final=json.loads((control/'final-identity-rehash.json').read_text())
  result=f.validate_proof(spec,base,handoff['original_metadata_symlink_resolutions'],source,payload,final);self.assertEqual(len(result),28)
  changed=copy.deepcopy(final);changed['source_stats'].pop(next(iter(changed['source_stats'])))
  with self.assertRaises(AssertionError):f.validate_proof(spec,base,handoff['original_metadata_symlink_resolutions'],source,payload,changed)
 def test_preserves_all_682_completed_services_and106_processes(self):
  prior=json.loads((Q/'nemotron9b-package-r2-20261108-complete-catalog.json').read_text())
  units=[f.protocol.unit(r['run_id'],k)for r in f.protocol.matrix()for k in ('server','client')]+[f.life.UNIT]
  added=[dict(unit=u,recorded_cgroups=['/sys/fs/cgroup'+f.life.GROUP_PREFIX+u+'.service'],owner=None,captured_process_owner=None,never_launched=True,invocation_id=None,installed_argv=None,absent_records=['/synthetic/'+u+'-launch.json','/synthetic/'+u+'-owner.json'],historical_invocation_unknown=False)for u in units]
  owner=dict(pid=999999999,start_ticks='123456',argv=['synthetic-only-no-launch'])
  candidate=copy.deepcopy(prior);candidate['rows']+=added;candidate['processes'].append(dict(owner,role='external_launcher_and_final_auditor',self_exit_not_yet_observed=True,next_admission_must_prove_gone=True))
  spec=dict(planned_units=units,runtime_source_sha256={},historical_payload_sha256=prior['historical_payload_sha256'],final_predecessor_catalog={'path':'/synthetic/prior.json','sha256':'a'*64})
  candidate.update(complete_count=707,previous_catalog=spec['final_predecessor_catalog'],root_finalized=False,all_planned_launches_accounted_for=True,requires_root_terminal_review=True,external_self_exit_claimed=False)
  if 'process_predecessors'in candidate:candidate['process_predecessors']=copy.deepcopy(candidate['processes'])
  with patch.object(f,'PRIOR_ROWS',682),patch.object(f,'PRIOR_PROCESSES',106):
   self.assertEqual(f.validate_extension(prior,candidate,spec,owner),added)
   for change in [lambda c:c['rows'].__setitem__(0,c['rows'][1]),lambda c:c['source_sha256'].pop(next(iter(prior['source_sha256']))),lambda c:c.__setitem__('root_finalization_observed_at','changed'),lambda c:c['processes'].pop(0),lambda c:c['rows'][-1].__setitem__('launch_without_authenticated_owner',True)]:
    c=copy.deepcopy(candidate);change(c)
    with self.assertRaises((AssertionError,RuntimeError,KeyError)):f.validate_extension(prior,c,spec,owner)
 def test_runtime_namespace_stays_absent(self):self.assertFalse(f.WORK.exists())

class Inventory(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);(self.root/'control').mkdir(mode=0o700)
 def tearDown(self):self.tmp.cleanup()
 def write(self,name,data):
  path=self.root/name;path.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
  fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
  with os.fdopen(fd,'wb')as stream:stream.write(data)
  return path
 def test_every_output_and_pending_byte_retained(self):
  a=self.write('r00/output.raw',b'\xff\x00raw');b=self.write('control/.pending-retained',b'incomplete')
  v=f.output_inventory(self.root);self.assertEqual(v['bytes'],15);self.assertEqual({r['path']:r['sha256']for r in v['files']},{str(a):hashlib.sha256(a.read_bytes()).hexdigest(),str(b):hashlib.sha256(b.read_bytes()).hexdigest()});self.assertEqual(len(v['pending_files']),1);self.assertFalse(v['model_payload_reads']);self.assertFalse(v['raw_output_added_to_source_pins'])
 def test_symlink_rejected(self):
  (self.root/'link').symlink_to('/dev/null')
  with self.assertRaises(ValueError):f.output_inventory(self.root)
 def test_shared_hardlink_rejected(self):
  p=self.write('control/a.json',b'{}');os.link(p,self.root/'control/b.json')
  with self.assertRaises(ValueError):f.output_inventory(self.root)
 def test_oversized_file_rejected_before_read(self):
  p=self.write('control/a.raw',b'')
  with p.open('r+b')as stream:stream.truncate(64*2**20+1)
  with self.assertRaises(AssertionError):f.output_inventory(self.root)
 def test_aggregate_budget_enforced(self):
  self.write('control/a.raw',b'123')
  with patch.object(f.protocol,'OUTPUT_CAP',2),self.assertRaises(AssertionError):f.output_inventory(self.root)
 def test_owned_short_ipc_socket_preserved_without_reading(self):
  (self.root/'t00').mkdir(mode=0o700);p=self.root/'t00/s';sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
  try:
   sock.bind(str(p));v=f.output_inventory(self.root);self.assertEqual(len(v['sockets']),1);self.assertEqual(v['files'],[])
  finally:sock.close()
 def test_socket_outside_named_ipc_root_rejected(self):
  sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
  try:
   sock.bind(str(self.root/'control/s'))
   with self.assertRaises(AssertionError):f.output_inventory(self.root)
  finally:sock.close()
 def test_file_replaced_during_inventory_rejected(self):
  p=self.write('control/a.raw',b'123');original=f.transport.store.ordinary;calls=[]
  def changed(path,*args):
   value=original(path,*args);calls.append(str(path))
   if len(calls)==1:p.write_bytes(b'4567')
   return value
  with patch.object(f.transport.store,'ordinary',changed),self.assertRaises(AssertionError):f.output_inventory(self.root)
 def test_complete_extra_evidence_requires_exact_root_disposition(self):
  closure=dict(extra_evidence_requires_review=True,catalog_reference=dict(path='/synthetic/catalog.json',sha256='a'*64,bytes=7));inventory=dict(pending_files=[dict(path='/synthetic/pending')]);exited=dict(path='/synthetic/exit.json',sha256='b'*64)
  args=SimpleNamespace(extra_evidence_review=None,extra_evidence_review_sha256=None)
  with self.assertRaises(AssertionError):f.validate_extra_evidence(args,closure,inventory,exited)
  p=self.write('control/review.json',f.life.json_bytes(dict(reviewer='/root',passed=True,actual_external_exit=exited,catalog_reference={k:closure['catalog_reference'][k]for k in ('path','sha256')},closure_sha256=f.protocol.digest(closure),inventory_sha256=f.protocol.digest(inventory),retain_all_bytes=True,scientific_success_inferred=False,disposition='Retain interrupted bytes; no success inference.')))
  args=SimpleNamespace(extra_evidence_review=str(p),extra_evidence_review_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
  with patch.object(f,'RESEARCH',p.parent):
   self.assertEqual(f.validate_extra_evidence(args,closure,inventory,exited)['sha256'],args.extra_evidence_review_sha256)
   with self.assertRaises(AssertionError):f.validate_extra_evidence(args,closure,dict(inventory,pending_files=[]),exited)

if __name__=='__main__':
 suite=unittest.TestSuite()
 for cls in (Finalizer,Inventory,transport_tests.Closure):suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(cls))
 result=unittest.TextTestRunner(verbosity=2).run(suite)
 if not result.wasSuccessful():sys.exit(1)
 print(json.dumps(dict(passed=True,tests=result.testsRun,source_only=True,terminal_finalization_executed=False,native_or_model_payload_reads=False,new_speedup_claim=False)))
