"""Bounded stdlib-only stat diagnostic of the retained source closure.

No model payload read, framework import, service operation or native inference.
Timing is harness diagnostic under concurrent non-measurement work, not a model
speedup or isolated benchmark. Does not modify any runtime source.
"""
from pathlib import Path
import datetime,hashlib,json,os,resource,signal,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
resource.setrlimit(resource.RLIMIT_AS,(512*2**20,512*2**20))
resource.setrlimit(resource.RLIMIT_CPU,(100,100))
resource.setrlimit(resource.RLIMIT_FSIZE,(16*2**20,16*2**20))
os.nice(19);os.sched_setaffinity(0,{7})
def deadline(signum,frame):raise TimeoutError('120second source-stat diagnostic wall bound')
signal.signal(signal.SIGALRM,deadline);signal.alarm(120)
import smollm3_observer_invariance_lifecycle_v22 as life
R=ROOT/'research';out=R/'smollm3-stat-guard-cost-20261108-v1.json'
assert not out.exists()
owner=life.process(os.getpid())
source=ROOT/'sio13/control/source-admission.json';wanted=''
base=json.loads((R/'smollm3-observer-sio13-root-terminal-audit-20261108-v1.json').read_text())
ref=base['references']['source'];assert ref['path']==str(source)
assert hashlib.sha256(source.read_bytes()).hexdigest()==ref['sha256']
expected=life.read(source)['source_stats'];assert len(expected)==37315
# Equivalent per-path resolution primitive; no cross-call/path caching.
def alternative(name):
 value=os.stat(name)
 return dict(device=value.st_dev,inode=value.st_ino,size=value.st_size,mtime_ns=value.st_mtime_ns,
             ctime_ns=value.st_ctime_ns,realpath=os.path.realpath(name,strict=True))
rows=[]
for label,fn in [('original',life.file_stat),('stdlib_realpath',alternative),('stdlib_realpath',alternative),('original',life.file_stat)]:
 started=time.monotonic();cpu=time.process_time();count=0
 for name,wanted in expected.items():
  assert fn(name)==wanted,name
  count+=1
 rows.append(dict(implementation=label,source_paths=count,wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu))
value=dict(schema='smollm3_stat_guard_cost_v1',reviewer='/root',observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),owner=owner,source_admission=ref,source_paths=len(expected),samples=rows,
 memory_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,no_cross_call_cache=True,model_payloads_read=False,native_imports=False,performance_claim=False,concurrent_native_run='nemotron9b_r4_nonmeasurement',original_sources_unchanged=True,
 limitation='Stat-only harness diagnostic while Nemotron nonmeasurement phase may run; not a full live RuntimeContext poll or independent model benchmark.')
with out.open('x')as f:json.dump(value,f,indent=2);f.write('\n')
print(json.dumps(dict(path=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),samples=rows,owner=owner)))
