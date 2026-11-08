"""Retained-metadata-only guard cost diagnostic; no native imports or work."""
from pathlib import Path
import cProfile,datetime,hashlib,io,json,os,pstats,resource,signal,sys,time
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
resource.setrlimit(resource.RLIMIT_AS,(512*2**20,512*2**20));resource.setrlimit(resource.RLIMIT_CPU,(100,100));resource.setrlimit(resource.RLIMIT_FSIZE,(16*2**20,16*2**20))
os.nice(19);os.sched_setaffinity(0,{7})
def deadline(sig,frame):raise TimeoutError('120second metadata diagnostic')
signal.signal(signal.SIGALRM,deadline);signal.alarm(120)
import smollm3_observer_invariance_lifecycle_v22 as life
owner=life.process(os.getpid());root=R/'sio13';rows=[]
for name in ['r00/permit-server-2622787.json','r00/permit-server-2626813.json','r00/spawn-before-2622787-1.json','r00/spawn-after-2622787-1.json']:
 path=root/name;started=time.monotonic();value=life.read(path);elapsed=time.monotonic()-started
 rows.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),seconds=elapsed,expanded_json_bytes=len(json.dumps(value))))
 del value
profile=cProfile.Profile();profile.enable();value=life.read(root/'r00/permit-server-2622787.json');profile.disable();del value
buf=io.StringIO();pstats.Stats(profile,stream=buf).sort_stats('cumulative').print_stats(20)
out=R/'research/smollm3-metadata-guard-cost-20261108-v1.json'
v=dict(schema='smollm3_metadata_guard_cost_v1',owner=owner,observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=rows,profile_top=buf.getvalue(),source_only=True,model_payload_reads=False,native_imports=False,new_speedup_claim=False,
 limitation='Independent retained metadata reads while Nemotron nonmeasurement stage runs, not a full live guard replay.',memory_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
with out.open('x')as f:json.dump(v,f,indent=2);f.write('\n')
print(json.dumps(dict(path=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),files=rows,profile_top=buf.getvalue(),owner=owner)))
