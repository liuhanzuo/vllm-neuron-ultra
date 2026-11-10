"""Synthetic timing/identity regressions, including the complete worker HTTP path."""
import ast,copy,json,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
Q=Path(__file__).resolve().parent;R=Q.parent
sys.path[:0]=[str(Q),str(R)]
import nemotron9b_package_startup_v27 as s
import nemotron9b_package_worker_v27 as w
CHILD=dict(pid=101,start_ticks='200',ppid=100,cgroup='0::/owned',argv=['python','worker'])
TARGET=['python','-B','-m','exact.module']
class Timing(unittest.TestCase):
 def setUp(self):
  self.now=0.;self.logs=[];self.prepared=[];self.actual=copy.deepcopy(CHILD)
  self.intent=False;self.code=None;self.target_seen=[]
 def clock(self):return self.now
 def sleep(self,seconds):self.now+=1.
 def guard(self):pass
 def validate(self):
  if not self.intent:raise RuntimeError('Bad authenticated intent')
 def observe(self):
  self.target_seen.append(True)
  return dict(actual=True)
 def run_case(self,**changes):
  args=dict(child_owner=CHILD,target_argv=TARGET,read_process=lambda:copy.deepcopy(self.actual),
   poll=lambda:self.code,guard=self.guard,intent_ready=lambda:self.intent,validate_intent=self.validate,
   observe_target=self.observe,save_preparation=self.prepared.append,save_diagnostic=self.logs.append,
   clock=self.clock,sleep=self.sleep)
  args.update(changes);return s.await_prepared_exec(**args)
 def test_preparation_over_60_and_separate_exec_budget(self):
  def guard():
   self.intent=self.now>=66.2
   self.actual['argv']=TARGET if self.now>=120 else CHILD['argv']
  self.assertEqual(self.run_case(guard=guard),dict(actual=True))
  self.assertEqual(self.prepared[0]['elapsed_seconds'],67.)
  self.assertEqual(self.now,120.)
 def test_preparation_deadline(self):
  with self.assertRaisesRegex(RuntimeError,'preparation deadline'):self.run_case()
  self.assertEqual(self.now,300.);self.assertFalse(self.target_seen)
 def test_ready_at_preparation_deadline_rejected(self):
  def guard():self.intent=self.now>=300;self.actual['argv']=TARGET
  with self.assertRaisesRegex(RuntimeError,'without authenticated'):
   self.run_case(guard=guard)
  # Original argv remains through the deadline in the separate exact-boundary case.
  self.setUp()
  def late():self.intent=self.now>=300
  with self.assertRaisesRegex(RuntimeError,'preparation deadline'):self.run_case(guard=late)
 def test_preparation_validation_cannot_overrun_budget(self):
  self.intent=True
  def validate():self.now=300
  with self.assertRaisesRegex(RuntimeError,'preparation deadline'):self.run_case(validate_intent=validate)
 def test_exec_timeout_fresh_and_bounded(self):
  def guard():self.intent=self.now>=66
  with self.assertRaisesRegex(RuntimeError,'observation deadline'):self.run_case(guard=guard)
  self.assertEqual(self.now,126.)
 def test_all_kernel_identity_changes_rejected(self):
  for key,value in [('pid',102),('start_ticks','201'),('ppid',99),('cgroup','0::/other')]:
   with self.subTest(key=key):
    self.setUp();self.actual[key]=value
    with self.assertRaisesRegex(RuntimeError,'kernel identity'):self.run_case()
 def test_unexpected_argv_rejected(self):
  self.actual['argv']=['unrelated']
  with self.assertRaisesRegex(RuntimeError,'Unexpected'):self.run_case()
 def test_missing_or_exited_child_rejected(self):
  with self.assertRaisesRegex(RuntimeError,'disappeared'):self.run_case(read_process=lambda:None)
  self.code=0
  with self.assertRaisesRegex(RuntimeError,'exited'):self.run_case()
 def test_transition_requires_intent_even_if_target_already_visible(self):
  for argv in (TARGET,[]):
   with self.subTest(argv=argv):
    self.actual['argv']=argv
    with self.assertRaisesRegex(RuntimeError,'without authenticated'):self.run_case()
 def test_bad_intent_rejected(self):
  with self.assertRaisesRegex(RuntimeError,'Bad authenticated'):self.run_case(intent_ready=lambda:True)
 def test_fast_target_with_valid_intent(self):
  self.intent=True;self.actual['argv']=TARGET
  self.assertEqual(self.run_case(),dict(actual=True))
 def test_existing_empty_grace_retained(self):
  self.intent=True;self.actual['argv']=[]
  with self.assertRaisesRegex(RuntimeError,'exec grace'):self.run_case()
 def test_guard_failure_preserved(self):
  def guard():raise RuntimeError('cancelled')
  with self.assertRaisesRegex(RuntimeError,'cancelled'):self.run_case(guard=guard)
 def test_nonfinite_and_unbounded_budgets_rejected(self):
  for name in ('preparation_seconds','exec_seconds'):
   for val in (True,0,-1,float('inf'),float('nan'),301,'60'):
    with self.subTest(name=name,val=val),self.assertRaises(ValueError):self.run_case(**{name:val})
 def test_argv_must_be_nonempty_and_distinct(self):
  for argv in ([],CHILD['argv']):
   with self.subTest(argv=argv),self.assertRaises(ValueError):self.run_case(target_argv=argv)
 def ack(self,**changes):
  args=dict(ready=lambda:self.now>=170,poll=lambda:self.code,guard=self.guard,
    observe_target=lambda:dict(exact=True),expected=dict(exact=True),
    clock=self.clock,sleep=self.sleep)
  args.update(changes);s.await_post_exec_ack(**args)
 def test_ack_has_fresh_60_seconds(self):
  self.now=120;self.ack();self.assertEqual(self.now,170)
 def test_ack_deadline_not_shared_with_preparation(self):
  self.now=120
  with self.assertRaisesRegex(RuntimeError,'ACK timeout'):self.ack(ready=lambda:False)
  self.assertEqual(self.now,180)
 def test_ready_ack_still_requires_live_identity(self):
  self.code=0
  with self.assertRaisesRegex(RuntimeError,'exited'):self.ack(ready=lambda:True)
  self.code=None
  with self.assertRaisesRegex(RuntimeError,'identity'):self.ack(ready=lambda:True,observe_target=lambda:None)
class EmptyHandoff(unittest.TestCase):
 setUp=Timing.setUp
 clock=Timing.clock
 sleep=Timing.sleep
 guard=Timing.guard
 validate=Timing.validate
 observe=Timing.observe
 run_case=Timing.run_case
 def test_prepare_empty_grace_not_restarted(self):
  self.intent=True;self.actual['argv']=[]
  def validate():
   self.now=.3;self.actual['argv']=TARGET
  with self.assertRaisesRegex(RuntimeError,'exec grace'):self.run_case(validate_intent=validate)
  self.assertFalse(self.target_seen)
 def test_prepare_empty_reversion_rejected(self):
  self.intent=True;self.actual['argv']=[]
  def validate():self.actual['argv']=CHILD['argv']
  with self.assertRaisesRegex(RuntimeError,'reverted'):self.run_case(validate_intent=validate)
class WorkerIntegration(unittest.TestCase):
 def test_all_other_functions_ast_unchanged(self):
  def funcs(path):return {n.name:ast.dump(n,include_attributes=False)for n in ast.parse(path.read_text().replace('_v24','_v27')).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
  before=funcs(R/'nemotron9b_package_worker_v24.py')
  after=funcs(R/'nemotron9b_package_worker_v27.py')
  self.assertEqual(set(before),set(after))
  self.assertEqual([k for k in before if before[k]!=after[k]],['http_stage','serve_child'])
 def exercise(self,fail=False):
  with tempfile.TemporaryDirectory()as tmp:
   c=Path(tmp);rows={};now=[0.];cleanups=[];closed=[];observation=dict(exact=True)
   parent=dict(pid=100);controller=dict(pid=99);spec=dict(control_root=tmp,evidence_root=tmp,work_root=tmp)
   spec_ref=dict(path='spec',sha256='s');review_ref=dict(path='review',sha256='r')
   command=CHILD['argv'][:-1]+['worker','--server-child']
   child_owner=dict(CHILD,argv=command)
   def once(path,value):
    assert path not in rows
    rows[path]=copy.deepcopy(value);path.write_text(json.dumps(value))
   def ref(path):return dict(path=str(path),sha256='test-digest')
   def guard():
    now[0]+=1
    intent=c/'compile-server-exec.json'
    if now[0]>=67 and intent not in rows and not fail:
     once(intent,dict(original_owner=child_owner,argv=TARGET,ack=ref(c/'compile-server-ack.json')))
     s.publish_prepared(c/'compile-server-prepared')
    post=c/'compile-server-post-exec.json'
    if post in rows and now[0]>=170 and c/'compile-server-observed.json' not in rows:
     once(c/'compile-server-observed.json',dict(parent=parent,observation=observation,post_exec=ref(post),observer='serial-controller',controller=controller))
   def proc(pid):return dict(child_owner,argv=TARGET if now[0]>=120 and not fail else command)
   def observe(*args):return observation
   once(c/'admission.json',dict(controller=controller))
   once(c/'workload-inputs.json',{'32':list(range(32)),'128':list(range(128))})
   fakep=types.SimpleNamespace(argv=lambda *a:command[:-1],server_argv=lambda *a:TARGET,
    group=lambda *a:'/owned',low=types.SimpleNamespace(proc=proc),
    once=once,read=lambda path:rows[path],ref=ref,case=lambda *a:dict(served_name='test'))
   lifecycle=types.ModuleType('nemotron9b_package_server_lifecycle_v27')
   lifecycle.observe_expected=observe
   lifecycle.shutdown=lambda *a:cleanups.append(a)
   lifecycle.validate_chain=lambda *a:None
   transport=types.ModuleType('nemotron9b_package_http_transport_v27')
   transport.request=lambda port,name,ids,count,**kw:dict(usage=dict(prompt_tokens=len(ids),completion_tokens=count))
   child=types.SimpleNamespace(pid=101,poll=lambda:None)
   sock=types.SimpleNamespace(connect_ex=lambda *a:1)
   class Context:
    def __init__(self,value):self.value=value
    def __enter__(self):return self.value
    def __exit__(self,*args):pass
   original_start,original_ack=s.await_prepared_exec,s.await_post_exec_ack
   def start(**kw):return original_start(**kw,clock=lambda:now[0],sleep=lambda x:None)
   def ack(**kw):return original_ack(**kw,clock=lambda:now[0],sleep=lambda x:None)
   with patch.object(w,'p',fakep),patch.object(w.socket,'socket',lambda:Context(sock)),patch.object(w.subprocess,'Popen',lambda *a,**kw:child),patch.object(w.os,'pidfd_open',lambda *a:55),patch.object(w.os,'close',closed.append),patch.object(w.urllib.request,'urlopen',lambda *a,**kw:Context(types.SimpleNamespace(status=200))),patch.dict(sys.modules,{lifecycle.__name__:lifecycle,transport.__name__:transport}),patch.object(s,'await_prepared_exec',start),patch.object(s,'await_post_exec_ack',ack):
    if fail:
     with self.assertRaisesRegex(RuntimeError,'preparation deadline'):
      w.http_stage(spec,'compile',parent,guard,spec_ref,review_ref)
    else:
     result=w.http_stage(spec,'compile',parent,guard,spec_ref,review_ref)
     self.assertTrue(result['passed']);self.assertEqual(len(result['smoke']),2)
     self.assertTrue(rows[c/'compile-server-preparation.json']['elapsed_seconds']>=67)
   self.assertEqual(len(cleanups),1);self.assertEqual(closed,[55])
   self.assertEqual(cleanups[0][7],not fail)
   if fail:self.assertIn('preparation deadline',cleanups[0][8])
 def test_actual_worker_flow_accepts_67s_preparation_53s_exec_50s_ack(self):self.exercise()
 def test_actual_worker_timeout_retains_cleanup_and_error(self):self.exercise(fail=True)

class ReadyDirectory(unittest.TestCase):
 def test_absent_marker_does_not_read_partial_exec_file(self):
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp);intent=root/'server-exec.json';intent.write_text('{')
   self.assertFalse(s.prepared_ready(root/'server-prepared'))
   self.assertEqual(intent.read_text(),'{')
 def test_publish_once_and_validate_directory(self):
  with tempfile.TemporaryDirectory()as tmp:
   p=Path(tmp)/'prepared';s.publish_prepared(p);self.assertTrue(s.prepared_ready(p))
   with self.assertRaises(FileExistsError):s.publish_prepared(p)
 def test_file_symlink_and_wrong_permissions_rejected(self):
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp);p=root/'prepared';p.write_text('bad')
   with self.assertRaisesRegex(RuntimeError,'Invalid'):s.prepared_ready(p)
   p.unlink();p.symlink_to(root)
   with self.assertRaisesRegex(RuntimeError,'Invalid'):s.prepared_ready(p)
   p.unlink();s.publish_prepared(p);p.chmod(0o755)
   with self.assertRaisesRegex(RuntimeError,'Invalid'):s.prepared_ready(p)
 def test_partial_intent_waits_until_real_publication(self):
  case=Timing();case.setUp()
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp);p=root/'prepared';intent=root/'exec.json';intent.write_text('{')
   reads=[]
   def guard():
    if case.now>=67 and not p.exists():
     intent.write_text('{"complete":true}');s.publish_prepared(p)
     case.actual['argv']=TARGET
   def validate():
    reads.append(json.loads(intent.read_text()))
    self.assertGreaterEqual(case.now,67.)
   self.assertEqual(case.run_case(guard=guard,intent_ready=lambda:s.prepared_ready(p),validate_intent=validate),dict(actual=True))
   self.assertEqual(reads,[dict(complete=True),dict(complete=True)])
class ChildPublication(unittest.TestCase):
 def exercise(self,write_failure=False,health_failure=False):
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp);events=[];spec={'control_root':tmp};sr={'path':'spec','sha256':'s'};rr={'path':'review','sha256':'r'}
   command=['python','worker'];actual=dict(CHILD,pid=w.os.getpid(),argv=command+['--server-child'])
   owner=dict(pid=actual['ppid']);ack=dict(execution_spec=sr,source_review=rr,child=actual,parent=owner)
   records={root/'compile-server-ack.json':ack,root/'compile-owner.json':owner,root/'admission.json':dict(admitted=True)}
   (root/'compile-server-ack.json').write_text(json.dumps(ack))
   def require(value,msg):
    if not value:raise RuntimeError(msg)
   def bundle(*args):events.append('bundle');return spec,None
   def health(*args):
    events.append('health')
    if health_failure:raise RuntimeError('health failed')
   def once(path,value):
    events.append('intent-write')
    if write_failure:raise RuntimeError('write failed')
    path.write_text(json.dumps(value));records[path]=value
   original=s.publish_prepared
   def publish(path):
    self.assertEqual(events[-1],'intent-write')
    value=json.loads((root/'compile-server-exec.json').read_text())
    self.assertEqual(value['original_owner'],actual)
    events.append('publish');original(path)
   def execute(*args):
    self.assertTrue(s.prepared_ready(root/'compile-server-prepared'));events.append('exec')
   fake=types.SimpleNamespace(require=require,bundle=bundle,read=lambda path:records[path],
    low=types.SimpleNamespace(proc=lambda pid:actual),argv=lambda *args:command,
    service_identity=lambda *args:events.append('service-identity'),healthy=health,
    server_argv=lambda *args:TARGET,ref=lambda path:dict(path=str(path),sha256='test'),
    once=once,PYTHON='python')
   with patch.object(w,'p',fake),patch.object(s,'publish_prepared',publish),patch.object(w.os,'execve',execute):
    if write_failure or health_failure:
     with self.assertRaisesRegex(RuntimeError,'failed'):w.serve_child('compile',sr,rr)
     self.assertFalse((root/'compile-server-prepared').exists());self.assertNotIn('exec',events)
    else:
     w.serve_child('compile',sr,rr)
     self.assertEqual(events,['bundle','service-identity','health','intent-write','publish','exec'])
 def test_actual_child_publishes_after_health_and_complete_intent(self):self.exercise()
 def test_actual_child_write_failure_never_publishes_or_executes(self):self.exercise(write_failure=True)
 def test_actual_child_health_failure_never_publishes_or_executes(self):self.exercise(health_failure=True)

if __name__=='__main__':unittest.main(verbosity=2)
