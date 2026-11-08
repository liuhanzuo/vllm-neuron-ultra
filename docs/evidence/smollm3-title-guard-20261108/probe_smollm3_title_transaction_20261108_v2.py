"""Real title source/writer + retained guard workload; synthetic process/native state.

This does not authenticate a live model worker or emulate a separately scheduled
controller. It tests the full title code path with actual monotonic deadlines,
full-size source bindings, real exclusive durable publication, and measured
source/metadata work inside each worker poll. It never launches inference.
"""
from pathlib import Path
import copy,datetime,hashlib,inspect,json,os,resource,signal,sys,threading,time,types
R=Path(__file__).resolve().parents[1];Q=R/'research';sys.path.insert(0,str(R))
resource.setrlimit(resource.RLIMIT_AS,(512*2**20,512*2**20));resource.setrlimit(resource.RLIMIT_CPU,(280,280));resource.setrlimit(resource.RLIMIT_FSIZE,(16*2**20,16*2**20))
os.nice(19);os.sched_setaffinity(0,{7})
def alarm(sig,frame):raise TimeoutError('300second title source diagnostic')
signal.signal(signal.SIGALRM,alarm);signal.alarm(300)
import smollm3_observer_invariance_lifecycle_v22 as life
import smollm3_observer_invariance_protocol_v18 as protocol
import smollm3_observer_metadata_store_v4 as store
import smollm3_stat_identity_v2 as fast_stat
import smollm3_title_deadline_candidate_v1 as runtime
import smollm3_title_publication_candidate_v1 as title_publication
import argparse
parser=argparse.ArgumentParser();parser.add_argument("--late-controller",action="store_true");args=parser.parse_args();run_label="late"if args.late_controller else "early"
import check_smollm3_enginecore_title_transition_v7 as original
owner=life.process(os.getpid());print(json.dumps(dict(owner=owner)),flush=True)
out=Q/('smollm3-title-transaction-source-probe-20261108-v2-'+run_label);out.mkdir(mode=0o700);root=out/'r00';root.mkdir(mode=0o700);control=out/'control';control.mkdir(mode=0o700)
retained_root=R/'sio13';parent=retained_root/'r00/permit-server-2622787.json';child=retained_root/'r00/permit-server-2626813.json'
child_bundle=store.decode(retained_root,child,life.read_bytes(child));real_permit=child_bundle['permit'];admission=child_bundle['admission']
spec=json.loads((Q/'smollm3-observer-sio13-execution-spec-20261108-v1.json').read_text());template=spec['source_template'];plan=json.loads(protocol.bounded_source(Path(template['path']),template['sha256']));base=protocol.load_base()
assert protocol.digest(plan)==real_permit['source_plan_sha256'] and protocol.digest(base)==protocol.BASE_CANONICAL_SHA
wanted_permit=protocol.digest(real_permit)
checks=[];transactions=[];poll_times=[]
def guard_work():
 start=time.monotonic();trace=sys.gettrace()is not None
 for path in [parent,child,parent]:
  bundle=store.decode(retained_root,path,life.read_bytes(path))
 for name in spec['runtime_source_sha256']:assert fast_stat.file_stat(name)==admission['source_stats'][name],name
 for name,wanted in admission['source_stats'].items():assert fast_stat.file_stat(name)==wanted,name
 for name,wanted in admission['payload_stats'].items():assert fast_stat.file_stat(name)==wanted,name
 a=copy.deepcopy(real_permit);b=copy.deepcopy(a);assert a==b==real_permit
 assert protocol.digest(plan)==real_permit['source_plan_sha256'] and protocol.digest(base)==protocol.BASE_CANONICAL_SHA
 poll_times.append(dict(start=start,end=time.monotonic(),trace_owner_installed=trace));return None

def sampled():
 permit,initial,controller,witness=original.sample()
 # Real source-map cardinality/bytes, but deliberately synthetic process IDs.
 permit['root_runtime_source_sha256']=copy.deepcopy(real_permit['root_runtime_source_sha256'])
 return permit,initial,controller,witness

def fixture_check(name,condition):
 if not condition:raise AssertionError(name)
 checks.append(name)

writer_globals=dict(life.write_once.__globals__);writer_globals['DEST']=out
atomic_writer=types.FunctionType(life.write_once.__code__,writer_globals)
write_globals=dict(protocol.RuntimeContext.write.__globals__);write_globals['directory']=lambda ignored:root
actual_context_write=types.FunctionType(protocol.RuntimeContext.write.__code__,write_globals)
title_globals=dict(title_publication.record_engine_title_before.__globals__);title_globals["directory"]=lambda ignored:root
actual_title_write=types.FunctionType(title_publication.record_engine_title_before.__code__,title_globals)
def external_writer(name,raw):
 start=time.monotonic();result=atomic_writer(root/name,raw);transactions.append(dict(name=name,start=start,end=time.monotonic(),sha256=result['sha256']));return result

source=inspect.getsource(original.original_code_fixture)
def replace_once(old,new):
 global source
 assert source.count(old)==1,old;source=source.replace(old,new)
replace_once("permit,initial,controller,witness=sample();store={};root=VirtualPath(store);current=[copy.deepcopy(initial)]","permit,initial,controller,witness=sample();store={};current=[copy.deepcopy(initial)]")
replace_once("fixture.p=types.SimpleNamespace(directory=lambda ignored:root);fixture.CONTROL=VirtualPath(store,'control')","fixture.p=types.SimpleNamespace(directory=lambda ignored:root);fixture.CONTROL=control")
replace_once("fixture.read=lambda path:copy.deepcopy(store[str(path)])","fixture.read=lambda path:json.loads(life.read_bytes(path))")
replace_once("        require_new=str(path)not in store\n        if not require_new:raise RuntimeError('exclusive fixture record')\n        store[str(path)]=copy.deepcopy(value)","        raw=(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\\n').encode()\n        result=atomic_writer(path,raw)\n        transactions.append(dict(name=Path(path).name,end=time.monotonic(),sha256=result['sha256']))")
replace_once("self.title_witness=None;runtime.init_context(self);self.polls=0;self.native_calls=0","self.title_witness=None;runtime.init_context(self);self.polls=0;self.native_calls=0;self.callback_lock=threading.RLock();self.external_writer=external_writer")
replace_once("            self.polls+=1","            self.polls+=1;guard_work()")
replace_once("        def record(self,name,value):\n            self.poll();fixture.record(root/name,value);self.poll()","        write=actual_context_write\n        record=protocol.RuntimeContext.record\n        record_engine_title_before=actual_title_write")
namespace=dict(original.__dict__);namespace.update(runtime=runtime,life=life,root=root,control=control,json=json,Path=Path,time=time,threading=threading,sample=sampled,check=fixture_check,guard_work=guard_work,atomic_writer=atomic_writer,transactions=transactions,external_writer=external_writer,actual_context_write=actual_context_write,actual_title_write=actual_title_write,protocol=protocol)
compiled_path=out/'fixture-source.py'
with compiled_path.open('x')as f:f.write(source)
exec(compile(source,str(compiled_path),'exec'),namespace)
start=time.monotonic();value=namespace['original_code_fixture'](commit_early=not args.late_controller);elapsed=time.monotonic()-start
receipts={p.name:dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)for p in root.glob('*.json')}
error=value['failure'];before_path=root/'title-before-101.json';before=json.loads(before_path.read_text())if before_path.exists()else None
v=dict(schema='smollm3_title_transaction_source_probe_v2',controller_order=run_label,owner=owner,observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_seconds=elapsed,
 title_state_machine='smollm3_title_deadline_candidate_v1',dedicated_before_writer='smollm3_title_publication_candidate_v1',original_context_writer='RuntimeContext.write/record from protocol_v18',original_atomic_writer=True,original_ray_dispatcher=True,synthetic_process_observation=True,native_setter_is_stub=True,controller_ack_is_in_process=True,complete_live_guard_executed=False,
 workload_each_poll=dict(permit_reads=3,runtime_source_stat_count=len(spec['runtime_source_sha256']),source_stat_count=len(admission['source_stats']),payload_stat_count=len(admission['payload_stats']),full_permit_copies=2,source_plan_and_base_digest=True),
 permit_source_map_count=len(real_permit['root_runtime_source_sha256']),poll_times=poll_times,publications=transactions,receipts=receipts,source_checks=checks,
 original_deadline_seconds=15,created_monotonic=None if before is None else before['created_monotonic'],deadline_monotonic=None if before is None else before['deadline_monotonic'],
 title_committed=value['observer'].committed,native_stub_calls=value['context'].native_calls,polls=value['context'].polls,
 failure=None if error is None else dict(type=type(error).__name__,message=str(error)),model_requests=0,model_speedup_claim=False,native_inference_imports=False,
 limitation='Full title source control flow and original real writer with representative retained guard workload; no actual live worker/service/environment/lock proof or independently scheduled controller, not runtime admission.',peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
commit_publication=next((x['end']for x in transactions if x['name']=='title-commit-101.json'),None)
v['commit_publication_elapsed']=None if commit_publication is None or before is None else commit_publication-before['created_monotonic']
v['commit_publication_within_original_deadline']=commit_publication is not None and commit_publication<before['deadline_monotonic']
result=Q/('smollm3-title-transaction-source-probe-20261108-v2-'+run_label+'.json')
with result.open('x')as f:json.dump(v,f,indent=2);f.write('\n')
print(json.dumps(dict(path=str(result),sha256=hashlib.sha256(result.read_bytes()).hexdigest(),title_committed=v['title_committed'],failure=v['failure'],polls=v['polls'],wall_seconds=elapsed,commit_publication_elapsed=v['commit_publication_elapsed'],commit_publication_within_original_deadline=v['commit_publication_within_original_deadline'])),flush=True)
