"""Actual writer controls and structural preservation for deferred intent creation."""
from pathlib import Path
import ast,copy,json,sys,tempfile,threading,types,unittest
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import smollm3_title_publication_candidate_v1 as helper
import smollm3_title_deadline_candidate_v1 as runtime
import smollm3_observer_invariance_lifecycle_v22 as life
import smollm3_observer_invariance_protocol_v18 as protocol
import check_smollm3_enginecore_title_transition_v7 as fixture
class Publication(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.clock=10.;self.events=[]
  self.permit,self.observation,self.controller,self.witness=fixture.sample();self.hook=types.SimpleNamespace(called=True,committed=False)
  self.old=(helper.directory,helper.time,runtime.observe);helper.directory=lambda ignored:self.root;helper.time=types.SimpleNamespace(monotonic=lambda:self.clock);runtime.observe=lambda ignored:copy.deepcopy(self.observation)
  gl=dict(life.write_once.__globals__);gl['DEST']=self.root;self.atomic=types.FunctionType(life.write_once.__code__,gl)
  self.context=types.SimpleNamespace(callback_lock=threading.RLock(),permit=self.permit,row=self.permit['run'],title_witness=self.hook,_title_token=None,poll=self.poll,external_writer=self.writer)
 def tearDown(self):helper.directory,helper.time,runtime.observe=self.old;self.tmp.cleanup()
 def poll(self):
  self.events.append('poll')
  if len(self.events)==1:self.clock+=20
 def writer(self,name,raw):self.events.append('write');return self.atomic(self.root/name,raw)
 def call(self):return helper.record_engine_title_before(self.context,self.hook,self.witness)
 def test_two_polls_real_atomic_writer_and_timestamp_after_precheck(self):
  before=self.call();self.assertEqual(self.events,['poll','write','poll']);self.assertEqual(before['created_monotonic'],30.);self.assertEqual(before['deadline_monotonic'],45.)
  self.assertEqual(json.loads((self.root/'title-before-101.json').read_text()),before);runtime.title.validate_before(self.permit,before)
 def test_precheck_failure_writes_nothing(self):
  def fail():raise RuntimeError('precheck')
  self.context.poll=fail
  with self.assertRaisesRegex(RuntimeError,'precheck'):self.call()
  self.assertEqual(list(self.root.iterdir()),[])
 def test_postcheck_failure_retains_written_intent(self):
  def fail_after():
   self.events.append('poll')
   if len(self.events)>1:raise RuntimeError('postcheck')
  self.context.poll=fail_after
  with self.assertRaisesRegex(RuntimeError,'postcheck'):self.call()
  self.assertTrue((self.root/'title-before-101.json').exists())
 def test_bad_durable_receipt_rejected(self):
  def bad(name,raw):r=self.writer(name,raw);r['durable']=False;return r
  self.context.external_writer=bad
  with self.assertRaises(RuntimeError):self.call()
 def test_existing_intent_not_overwritten(self):
  self.call();p=self.root/'title-before-101.json';before=p.read_bytes()
  with self.assertRaises(FileExistsError):self.call()
  self.assertEqual(p.read_bytes(),before)
 def test_wrong_hook_rejected(self):
  self.context.title_witness=object()
  with self.assertRaises(RuntimeError):self.call()
  self.assertEqual(list(self.root.iterdir()),[])
 def test_original_run_deadline_still_applies(self):
  self.permit['shared_run_deadline_monotonic']=29.
  with self.assertRaises(RuntimeError):self.call()
  self.assertEqual(list(self.root.iterdir()),[])
 def test_original_writer_validation_expressions_preserved(self):
  tree=ast.parse((R/'smollm3_observer_invariance_protocol_v18.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='RuntimeContext');writer=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='write')
  helper_tree=ast.parse((R/'smollm3_title_publication_candidate_v1.py').read_text());candidate=next(n for n in helper_tree.body if isinstance(n,ast.FunctionDef))
  def checks(node):return [ast.dump(n.value)for n in ast.walk(node)if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Name)and n.value.func.id=='require']
  self.assertEqual(checks(writer),checks(candidate)[1:])
  for node in [writer,candidate]:self.assertEqual(sum(isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and isinstance(n.func.value,ast.Name)and n.func.value.id=='self'and n.func.attr=='poll'for n in ast.walk(node)),2)
 def test_other_runtime_functions_identical_and_deadline_unchanged(self):
  a=ast.parse((R/'smollm3_enginecore_title_runtime_v11.py').read_text());b=ast.parse((R/'smollm3_title_deadline_candidate_v1.py').read_text())
  def kept(tree):return [ast.dump(n)for n in tree.body if not(isinstance(n,ast.FunctionDef)and n.name=='begin')]
  self.assertEqual(kept(a),kept(b));self.assertEqual(runtime.title.DEADLINE_SECONDS,15)
if __name__=='__main__':unittest.main(verbosity=2)
