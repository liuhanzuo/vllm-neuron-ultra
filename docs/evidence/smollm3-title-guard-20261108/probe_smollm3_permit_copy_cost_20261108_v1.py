"""Actual permit copy and hashing cost under the retained dispatcher fixture."""
from pathlib import Path
import copy,datetime,hashlib,json,os,resource,signal,sys,time,types
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
resource.setrlimit(resource.RLIMIT_AS,(512*2**20,512*2**20));resource.setrlimit(resource.RLIMIT_CPU,(175,175));resource.setrlimit(resource.RLIMIT_FSIZE,(16*2**20,16*2**20))
os.nice(19);os.sched_setaffinity(0,{7})
def deadline(sig,frame):raise TimeoutError('180second permit-copy diagnostic')
signal.signal(signal.SIGALRM,deadline);signal.alarm(180)
import smollm3_observer_invariance_lifecycle_v22 as life
import smollm3_observer_metadata_store_v3 as store
import smollm3_json_tree_clone_v1 as candidate
import smollm3_ray_import_witness_v5 as ray
from smollm3_enginecore_title_hook_v3 import EngineCoreTitleHook
owner=life.process(os.getpid());print(json.dumps(dict(owner=owner)),flush=True)
root=R/'sio13';path=root/'r00/permit-server-2626813.json';permit=store.decode(root,path,life.read_bytes(path))['permit'];wanted=life.p.digest(permit)
def fixture():
 title=EngineCoreTitleHook.__new__(EngineCoreTitleHook);title.trace_owner=None
 state=ray.RayImportWitness.__new__(ray.RayImportWitness);state.context=types.SimpleNamespace(title_witness=title);state.busy=False;state.before=None;state.after=None;state.returned=False;state.route='unresolved';state.route_frames={};state.modules={};state.install();return state
rows=[]
for label,fn in [('deepcopy',copy.deepcopy),('json_tree',candidate.clone),('json_tree',candidate.clone),('deepcopy',copy.deepcopy)]:
 state=fixture();start=time.monotonic();cpu=time.process_time()
 try:
  a=fn(permit);b=fn(a);digests=[life.p.digest(a),life.p.digest(b)]
 finally:sys.settrace(None)
 row=dict(implementation=label,copies=2,digests=2,wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu);assert digests==[wanted,wanted]and a==b==permit and a is not b and a is not permit
 rows.append(row);print(json.dumps(row),flush=True);del a,b
v=dict(schema='smollm3_permit_copy_diagnostic_v1',owner=owner,observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),samples=rows,permit_json_bytes=len(json.dumps(permit).encode()),permit_digest=wanted,actual_dispatcher_with_fixture_state=True,complete_live_guard_executed=False,complete_title_transaction_executed=False,native_imports=False,model_payload_reads=False,model_speedup_claim=False,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
out=R/'research/smollm3-permit-copy-cost-20261108-v1.json'
with out.open('x')as f:json.dump(v,f,indent=2);f.write('\n')
print(json.dumps(dict(path=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest())))
