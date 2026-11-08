"""Retained source/metadata replay under the actual dispatch callback.

Uses stdlib fixture state for the real Ray dispatcher; no Ray/vLLM/native code
executes. Three permit reads and two source-stat scans mirror dominant guard
components, not its complete live process/service/environment proof.
"""
from pathlib import Path
import datetime,hashlib,json,os,resource,signal,sys,time,types
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
resource.setrlimit(resource.RLIMIT_AS,(512*2**20,512*2**20));resource.setrlimit(resource.RLIMIT_CPU,(175,175));resource.setrlimit(resource.RLIMIT_FSIZE,(16*2**20,16*2**20))
os.nice(19);os.sched_setaffinity(0,{7})
def deadline(sig,frame):raise TimeoutError('180second component diagnostic')
signal.signal(signal.SIGALRM,deadline);signal.alarm(180)
import smollm3_observer_invariance_lifecycle_v22 as life
import smollm3_observer_metadata_store_v1 as original
import smollm3_observer_metadata_store_v4 as candidate
import smollm3_stat_identity_v2 as fast_stat
import smollm3_ray_import_witness_v5 as ray
from smollm3_enginecore_title_hook_v3 import EngineCoreTitleHook
root=R/'sio13';owner=life.process(os.getpid());expected=life.read(root/'control/source-admission.json')['source_stats'];assert len(expected)==37315
parent=root/'r00/permit-server-2622787.json';child=root/'r00/permit-server-2626813.json'
# Golden normalized hashes computed outside diagnostic timing.
expected_json={str(path):hashlib.sha256(original.codec.encode_json(life.read(path))).hexdigest()for path in [parent,child]}
def trace_fixture():
 title=EngineCoreTitleHook.__new__(EngineCoreTitleHook);title.trace_owner=None
 state=ray.RayImportWitness.__new__(ray.RayImportWitness)
 state.context=types.SimpleNamespace(title_witness=title);state.busy=False;state.before=None;state.after=None;state.returned=False;state.route='unresolved';state.route_frames={};state.modules={}
 state.install();assert sys.gettrace()is state.installed_trace and title.trace_owner is state.installed_trace
 return state
rows=[]
for label,store,statfn,traced in [('original',original,life.file_stat,True),('candidate',candidate,fast_stat.file_stat,True),('candidate',candidate,fast_stat.file_stat,True),('original',original,life.file_stat,True)]:
 state=trace_fixture()if traced else None
 start=time.monotonic();cpu=time.process_time();loaded=[]
 try:
  for path in [parent,child,parent]:
   raw=life.read_bytes(path);value=store.decode(root,path,raw);loaded.append((str(path),value))
  metadata_seconds=time.monotonic()-start
  scan_start=time.monotonic()
  for repeat in range(2):
   for name,wanted in expected.items():assert statfn(name)==wanted,name
  stat_seconds=time.monotonic()-scan_start
 finally:sys.settrace(None)
 elapsed=time.monotonic()-start;cpu_elapsed=time.process_time()-cpu
 for name,value in loaded:assert hashlib.sha256(original.codec.encode_json(value)).hexdigest()==expected_json[name]
 del loaded,value
 rows.append(dict(implementation=label,trace_installed=traced,source_stat_observations=2*len(expected),permit_reads=3,wall_seconds=elapsed,cpu_seconds=cpu_elapsed,metadata_seconds=metadata_seconds,stat_seconds=stat_seconds))
 print(json.dumps(rows[-1]),flush=True)
v=dict(schema='smollm3_guard_components_v3',owner=owner,observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),samples=rows,real_trace_source=str(Path(ray.__file__)),trace_state_is_fixture=True,full_live_guard_executed=False,
 observed_model_requests=0,native_imports=False,model_payload_reads=False,model_speedup_claim=False,original_sources_unchanged=True,memory_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
 limitation='Read-only harness component diagnostic during Nemotron nonmeasurement work. Actual active worker guards/complete title transaction not reproduced, no runtime admission.')
out=R/'research/smollm3-guard-components-20261108-v3.json'
with out.open('x')as f:json.dump(v,f,indent=2);f.write('\n')
print(json.dumps(dict(path=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),owner=owner)))
