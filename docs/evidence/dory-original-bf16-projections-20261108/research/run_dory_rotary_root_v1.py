"""Dory native arithmetic runner, exclusively admitted after actual Nemotron r6.

No runtime on import. --prepare authenticates completed history but never grants
source approval. --run requires a separately reviewed exact request and takes
the serial lock before native library reads/import or arithmetic.
"""
import argparse,ast,base64,csv,datetime,fcntl,hashlib,io,json,os,re,resource,signal,stat,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];Q=R/'research';sys.path.insert(0,str(R));sys.path.insert(0,str(Q))
import nemotron9b_package_protocol_v14 as p
import nemotron9b_package_r6_completed_history_v1 as history
import nemotron9b_r4_terminal_metadata_v2 as metadata
from nemotron9b_package_controller_v14 import acquire,release
PYTHON=Path('/home/ubuntu/venvs/model_reference_2026/bin/python')
BASE=PYTHON.parent.parent/'lib/python3.12/site-packages'
RECORD=BASE/'torch-2.9.1+cpu.dist-info/RECORD'
RECORD_SHA='b7b1a26641988654ae3f994cdfbf523200c5f08a9eb1a5fdac56c1f8493bd8e5'
RECORDS={
 'torch-2.9.1+cpu.dist-info/RECORD':RECORD_SHA,
 'filelock-4.2.5.dist-info/RECORD':'503b49547029e16192460d614f1047977a3f63583a17e1c7408907f9d33706b7',
 'typing_extensions-4.16.0.dist-info/RECORD':'9e07fea845cac76293d927afa7157a14ecba39b318bc8af236b8de530b7656c1',
 'setuptools-84.2.1.dist-info/RECORD':'03f770e022ed5af62052917e3441590b97d7f58ee66e514c343861cc38866d47',
 'sympy-1.14.0.dist-info/RECORD':'4c248fd145b914da11593f374a0f7ae81ef9269486355bba3a1506c33775d88a',
 'networkx-3.7.dist-info/RECORD':'0403c7ac1516dc6dc03a4f0d8256c3b5d333c5721e6ab116c9f2b37fdeb6eb30',
 'jinja2-3.1.6.dist-info/RECORD':'4551b3fb657b85a9f01e961c0f8a235da65d758149b619400211964b3ac02bdc',
 'fsspec-2026.9.0.dist-info/RECORD':'f7b3713c1750221155b0601147f3b68a24f11e342dca818b5fd551f5ef88a74b',
 'mpmath-1.3.0.dist-info/RECORD':'09cbd04e87d0c5ab64aab08291b31883e4ea27e8d2c640710e0beb1995e86b9b',
 'markupsafe-3.0.4.dist-info/RECORD':'ed48a71a5e151245a69cf5eded7adf5df093703a9cacd8df268042fce69ad544',
 'numpy-2.3.5.dist-info/RECORD':'46c02eaed00d395de9c0a53865fc7533b605116de2220acf3375f88da9125dc3',
}
EXECUTABLE=Path('/home/ubuntu/.local/share/uv/python/cpython-3.12.16-linux-x86_64-gnu/bin/python3.12')
NATIVE_ROOT=EXECUTABLE.parent.parent
BUDGET=dict(address_space_bytes=8*2**30,cpu_seconds=600,wall_seconds=900,file_bytes=16*2**20,output_bytes=16*2**20,root_floor_bytes=28*2**30,cpus=[6])
FILES=['dory_rotary_arithmetic_v1.py','dory_rotary_arithmetic_v2.py','dory_rotary_arithmetic_capture_v2.py','dory_rotary_capture_runtime_v1.py','review_dory_rotary_saved_boundaries_v1.py','check_dory_rotary_capture_v2.py','run_dory_rotary_root_v1.py','check_run_dory_rotary_root_v1.py','nemotron9b_r4_terminal_metadata_v2.py']

def ref(path):return metadata.ref(path)
def load(ref):return metadata.load(ref)
def once(path,value):
 # Compact metadata keeps the complete installed-file inventory below8MiB.
 path=Path(path);raw=(json.dumps(value,separators=(',',':'),allow_nan=False)+'\n').encode()
 p.require(len(raw)<=8*2**20,'Bounded individual Dory metadata')
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'wb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
 p.metadata.sync_parent(path)
 return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest())

def ensure_observer():
 from nemotron9b_package_window_v14 import external_observer,registry,gate_identity
 raw=os.environ.get('NEMOTRON_OBSERVER_GATE_FD','');p.require(raw.isdigit(),'Whole-command observer wrapper required')
 fd=int(raw);info=os.fstat(fd);gate=gate_identity()
 p.require((info.st_dev,info.st_ino)==(gate['device'],gate['inode']),'Inherited exact observer gate descriptor')
 value=registry();p.require(value['state']=='released'and value['performance_window']is False,'Previous Nemotron run released before Dory')
 return value

def source_paths():
 import dory_rotary_arithmetic_v2 as arithmetic
 paths={str(Q/name)for name in FILES}
 paths.update(str(path)for path,_ in arithmetic.PINS.values())
 paths.update(str(R/name)for name in ['nemotron9b_package_r6_completed_history_v1.py','nemotron9b_package_controller_v14.py'])
 # Resolve every explicit local import, including imports inside functions.
 pending=list(paths);seen=set()
 while pending:
  filename=pending.pop()
  if filename in seen:continue
  seen.add(filename)
  if not filename.endswith('.py'):continue
  for node in ast.walk(ast.parse(Path(filename).read_text())):
   modules=[a.name for a in node.names]if isinstance(node,ast.Import)else [node.module]if isinstance(node,ast.ImportFrom)and node.module and not node.level else []
   for module in modules:
    for root in (R,Q):
     target=root/(module.replace('.','/')+'.py')
     if target.is_file()and str(target)not in paths:paths.add(str(target));pending.append(str(target))
 return sorted(paths)


def validate_terminal(catalog_ref,exit_ref):
 p.require(catalog_ref['path']==str(history.CURRENT),'Only completed actual r6 history')
 catalog=history.load(catalog_ref);finish=load(exit_ref);ext=catalog['nemotron9b_package_r6_append']
 p.require(finish['schema']=='nemotron9b_r6_finalizer_external_exit_v1'and finish['catalog']==catalog_ref and finish['owner']==ext['finalizer_owner'],'Exact r6 finalizer completion')
 p.require(finish['actual_tool_exit_observed']is True and finish['exact_owner_gone']is True and finish['catalog_validated']is True,'Actual external finalizer exit and catalog validation')
 actual=finish['actual_exit'];p.require(type(actual['session_id'])is int and type(actual['exit_code'])is int and actual['exit_code']==0 and re.fullmatch('[0-9a-f]{6}',actual['tool_chunk']),'Real successful finalizer tool completion')
 p.low.gone(ext['finalizer_owner']);return catalog

def validate_request(request):
 p.require(request['schema']=='dory_rotary_runtime_request_v1'and re.fullmatch('[a-z0-9-]{1,32}',request['run_id']),'Exact runtime request')
 p.require(request['python']==str(PYTHON)and request['torch_record']==dict(path=str(RECORD),sha256=RECORD_SHA)and request['dependency_records']==RECORDS,'Fixed nativeTorchCPUinstallation and dependencies')
 p.require(request['runtime_identity']['executable']==str(EXECUTABLE)and request['runtime_identity']['python']==str(PYTHON),'Fixed prepared interpreter')
 validate_runtime_identity(request['runtime_identity'],hash_bytes=False)
 p.require(request['budgets']==BUDGET and request['science']==dict(unique_conditions=96,rounds=2,executed_comparisons=192,raw_tensors=53,raw_bytes=2230320,full_cache_positions=262144),'Unreduced original scientific and resource limits')
 p.require(request['output_root']==str(R/('dory-rotary-'+request['run_id'])),'Fresh exact output namespace')
 p.require(request['canonical_sha256']==p.canonical(),'Canonical main state unchanged')
 p.require(set(request['source_sha256'])==set(source_paths()),'Complete explicit new executable source closure')
 for path,wanted in request['source_sha256'].items():p.require(ref(path)['sha256']==wanted,'Frozen source changed: '+path)
 p.require(request['native_reference_qualified']is request['new_speedup_claim']is request['automatic_successor_launch']is False,'No premature model qualification or automatic launch')
 return request

def prepare(catalog_ref,exit_ref,run_id):
 ensure_observer();p.require(re.fullmatch('[a-z0-9-]{1,32}',run_id),'Bounded run identifier')
 catalog=validate_terminal(catalog_ref,exit_ref)
 fd,owner,lock=acquire()
 try:
  history.physical_drain(catalog);p.low.active_heavy();p.low.lock_proof(owner,lock)
  identity=capture_runtime_identity(lambda:p.low.lock_proof(owner,lock))
 finally:release(fd)
 out=R/('dory-rotary-'+run_id);p.require(not out.exists(),'Never reuse output roots')
 request=dict(schema='dory_rotary_runtime_request_v1',run_id=run_id,python=str(PYTHON),torch_record=dict(path=str(RECORD),sha256=RECORD_SHA),dependency_records=RECORDS,runtime_identity=identity,source_sha256={path:ref(path)['sha256']for path in source_paths()},budgets=BUDGET,science=dict(unique_conditions=96,rounds=2,executed_comparisons=192,raw_tensors=53,raw_bytes=2230320,full_cache_positions=262144),output_root=str(out),catalog=catalog_ref,finalizer_exit=exit_ref,canonical_sha256=p.canonical(),runtime_admitted=False,native_reference_qualified=False,new_speedup_claim=False,automatic_successor_launch=False)
 validate_request(request);path=Q/('dory-rotary-'+run_id+'-request.json');once(path,request)
 return dict(request=ref(path),review_created=False,native_started=False)

def file_identity(path):
 path=Path(path);s=path.lstat();p.require(path.resolve()==path and stat.S_ISREG(s.st_mode),'Canonical regular runtime file')
 return {key:getattr(s,key)for key in ['st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns']}

def verify_binary(path,expected,size,guard):
 path=Path(path);before=file_identity(path);p.require(before['st_size']==size,'Declared runtime file size')
 digest=hashlib.sha256();count=0;fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
 try:
  p.require(os.fstat(fd).st_ino==before['st_ino'],'Runtime file changed at open')
  while True:
   chunk=os.read(fd,2**20)
   if not chunk:break
   count+=len(chunk);p.require(count<=size,'Runtime file grew');digest.update(chunk);guard()
 finally:os.close(fd)
 p.require(file_identity(path)==before and count==size and digest.hexdigest()==expected,'Exact installed runtime bytes')
 return dict(path=str(path),sha256=expected,identity=before)

def parse_record(record,expected,*,torch=False):
 p.require(ref(record)['sha256']==expected,'Pinned installed dependency RECORD')
 rows=list(csv.reader(io.StringIO(record.read_text())))
 p.require(0<len(rows)<=12000 and (not torch or len(rows)==11721),'Bounded installed file count')
 result=[];seen=set();self_rows=0
 for row in rows:
  p.require(len(row)==3,'Exact three-column installed RECORD')
  name,digest,size=row;path=(BASE/name).resolve()
  p.require(path.is_relative_to(PYTHON.parent.parent)and path not in seen,'Unique runtime entry inside environment');seen.add(path)
  if not digest:
   p.require(path==record and not size,'Only RECORD self-hash omitted');self_rows+=1;continue
  p.require(re.fullmatch('sha256=[A-Za-z0-9_-]{43}',digest)and size.isdigit(),'Installed SHA256 and size')
  expected_bytes=base64.urlsafe_b64decode(digest[7:]+'=').hex()
  result.append((path,expected_bytes,int(size)))
 p.require(self_rows==1,'Exactly one RECORD self entry')
 return result

def verify_torch_install(guard):
 verified=[];seen=set();total=0
 for name,digest in RECORDS.items():
  record=BASE/name
  rows=parse_record(record,digest,torch=record==RECORD)
  for path,expected,size in rows:
   p.require(path not in seen,'No overlapping dependency files');seen.add(path)
   total+=size;p.require(total<=800*2**20,'Bounded dependency bytes')
   verified.append(verify_binary(path,expected,size,guard))
  p.require(ref(record)['sha256']==digest,'Installed RECORD unchanged')
 return verified

def runtime_paths():
 p.require(PYTHON.resolve()==EXECUTABLE,'Exact interpreter symlink target')
 stdlib=NATIVE_ROOT/'lib/python3.12'
 paths={EXECUTABLE,PYTHON.parent.parent/'pyvenv.cfg'}
 for root,dirs,files in os.walk(stdlib,followlinks=False):
  dirs[:]=[name for name in dirs if name not in ('site-packages','__pycache__')]
  for name in files:
   path=Path(root)/name
   if path.suffix in ('.py','.so','.zip'):paths.add(path)
 paths.update(x.resolve()for x in (NATIVE_ROOT/'lib').glob('libpython*.so*'))
 return sorted(paths)

def capture_runtime_identity(guard):
 rows=[];total=0
 for path in runtime_paths():
  identity=file_identity(path);total+=identity['st_size'];p.require(total<=256*2**20,'Bounded interpreter/stdlib closure')
  digest=hashlib.sha256()
  with path.open('rb')as stream:
   while chunk:=stream.read(2**20):digest.update(chunk);guard()
  p.require(file_identity(path)==identity,'Stable prepared runtime bytes')
  rows.append(dict(path=str(path),sha256=digest.hexdigest(),identity=identity))
 return dict(python=str(PYTHON),executable=str(EXECUTABLE),files=rows,origin='Local installed CPython identity; not an upstream signature',bytes=total)

def validate_runtime_identity(identity,hash_bytes,guard=lambda:None):
 p.require(identity['python']==str(PYTHON)and identity['executable']==str(EXECUTABLE),'Exact runtime identity')
 rows=identity['files'];p.require([row['path']for row in rows]==list(map(str,runtime_paths())),'Complete explicit interpreter and stdlib closure')
 for row in rows:
  p.require(re.fullmatch('[0-9a-f]{64}',row['sha256'])and file_identity(row['path'])==row['identity'],'Prepared interpreter/stdlib unchanged')
  if hash_bytes:verify_binary(row['path'],row['sha256'],row['identity']['st_size'],guard)
 return rows

def drain_children(deadline_seconds=10):
 # Only our actual direct children; pidfds prevent killing a reused PID.
 # This is an exceptional cleanup path; normal eager arithmetic spawns none.
 deadline=time.monotonic()+deadline_seconds
 while True:
  children=own_children()
  if not children:return
  p.require(time.monotonic()<deadline,'Unexpected native children not drained')
  for child in children:
   try:
    handle=os.pidfd_open(child)
    try:
     p.require(child in own_children(),'Child ownership changed before termination')
     signal.pidfd_send_signal(handle,signal.SIGKILL)
    finally:os.close(handle)
    try:os.waitpid(child,os.WNOHANG)
    except ChildProcessError:pass
   except ProcessLookupError:pass
  time.sleep(0.05)

def own_children():
 # Check all native threads, not only the Python main thread.
 children=set()
 for task in Path('/proc/self/task').iterdir():
  try:children.update(map(int,(task/'children').read_text().split()))
  except FileNotFoundError:continue
 return sorted(children)

class NoNativeChildren:
 def __init__(self):self.active=True
 def audit(self,event,args):
  if self.active and event in ('subprocess.Popen','os.system','os.fork','os.forkpty','os.posix_spawn','os.exec','pty.spawn'):
   raise RuntimeError('Dory eager arithmetic may not spawn or replace processes')
 def verify(self):p.require(not own_children(),'Native arithmetic left child processes')


def usage(root):
 total=0
 for path in root.rglob('*'):
  s=path.lstat();p.require(not path.is_symlink(),'No output symlinks')
  p.require(stat.S_ISREG(s.st_mode)or stat.S_ISDIR(s.st_mode),'Only ordinary output files/directories')
  if stat.S_ISREG(s.st_mode):p.require(s.st_nlink==1,'Ordinary output');total+=max(s.st_size,s.st_blocks*512)
 p.require(total<=BUDGET['output_bytes'],'Bounded whole output tree')
 fs=os.statvfs(R);p.require(fs.f_bavail*fs.f_frsize>=BUDGET['root_floor_bytes'],'Root free-space floor');return total

def run(request_ref,review_ref):
 ensure_observer();request=validate_request(load(request_ref));review=load(review_ref)
 p.require(review['schema']=='dory_rotary_runtime_source_review_v1'and review['reviewer']=='/root'and review['passed']is review['independent_review']is review['source_only']is True and review['blockers']==[],'Separate exact root source review')
 p.require(review['request']==request_ref and review['source_sha256']==request['source_sha256'],'Reviewed exact request/source')
 p.require(sys.executable==str(PYTHON)and sys.flags.no_site and sys.flags.isolated and sys.dont_write_bytecode and __debug__,'Fixed native interpreter; bytecode disabled')
 p.require('torch'not in sys.modules and 'numpy'not in sys.modules,'Authenticate before native import')
 out=Path(request['output_root']);p.require(not out.exists(),'Fresh runtime outputs')
 fd,owner,lock=acquire();started=time.monotonic();created=False;old_alarm=None;barrier=None;last_full_guard=[0.0]
 try:
  catalog=validate_terminal(request['catalog'],request['finalizer_exit']);history.physical_drain(catalog);p.low.active_heavy();p.low.lock_proof(owner,lock)
  os.sched_setaffinity(0,set(BUDGET['cpus']));os.nice(19)
  resource.setrlimit(resource.RLIMIT_AS,(BUDGET['address_space_bytes'],BUDGET['address_space_bytes']))
  resource.setrlimit(resource.RLIMIT_CPU,(BUDGET['cpu_seconds'],BUDGET['cpu_seconds']+5))
  resource.setrlimit(resource.RLIMIT_FSIZE,(BUDGET['file_bytes'],BUDGET['file_bytes']))
  def timeout(signum,frame):raise TimeoutError('Dory native arithmetic wall budget exceeded')
  old_alarm=signal.signal(signal.SIGALRM,timeout);signal.alarm(BUDGET['wall_seconds'])
  out.mkdir(mode=0o700);created=True
  def fast_guard():
   p.require(time.monotonic()-started<=BUDGET['wall_seconds'],'Bounded runtime duration')
   now=time.monotonic()
   if now-last_full_guard[0]>=0.5:
    p.low.lock_proof(owner,lock);ensure_observer();usage(out);last_full_guard[0]=now
    if barrier is not None:barrier.verify()
   p.require(os.sched_getaffinity(0)==set(BUDGET['cpus']),'Original CPU affinity')
  fast_guard();runtime_files=validate_runtime_identity(request['runtime_identity'],True,fast_guard);native_files=verify_torch_install(fast_guard)
  executable=PYTHON.resolve();binary_before=file_identity(executable)
  with executable.open('rb')as f:digest=hashlib.sha256(f.read()).hexdigest()
  p.require(file_identity(executable)==binary_before,'Interpreterstablebytes')
  environment={key:os.environ.get(key)for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONNOUSERSITE']}
  p.require(environment==dict(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONNOUSERSITE='1'),'Explicitone-threadnativeenvironment')
  once(out/'admission.json',dict(schema='dory_rotary_actual_admission_v1',request=request_ref,review=review_ref,owner=owner,lock_identity=lock,catalog=request['catalog'],all_predecessors_physically_drained=True,resources=BUDGET,environment=environment,torch_files_verified=len(native_files),interpreter=dict(path=str(executable),sha256=digest,identity=binary_before),actual_self_exit_claimed=False,native_reference_qualified=False,new_speedup_claim=False))
  once(out/'runtime-files.json',dict(files=native_files,runtime_files=runtime_files,records=RECORDS));usage(out)
  barrier=NoNativeChildren();barrier.verify();sys.addaudithook(barrier.audit)
  # -I -S excludes site hooks, editable Transformers and ambient PYTHONPATH.
  sys.path.append(str(BASE))
  import torch
  p.require(str(torch.__version__)=='2.9.1+cpu'and Path(torch.__file__).resolve()==BASE/'torch/__init__.py','ActualnativeTorchCPUmodule')
  torch.set_num_threads(1);torch.set_num_interop_threads(1)
  from dory_rotary_capture_runtime_v1 import run as compare
  result=compare(torch=torch,guard=fast_guard,output=out/'comparison')
  from review_dory_rotary_saved_boundaries_v1 import review as independent
  saved=independent(result['result'],fast_guard);once(out/'independent-review.json',saved)
  for row in native_files:p.require(file_identity(row['path'])==row['identity'],'Nativefilesunchangedafterarithmetic')
  validate_request(request);fast_guard();barrier.verify();barrier.active=False;p.low.active_heavy();barrier.verify();total=usage(out);actual=resource.getrusage(resource.RUSAGE_SELF)
  done=dict(schema='dory_rotary_actual_terminal_v1',complete=True,owner=owner,request=request_ref,admission=ref(out/'admission.json'),result=result['result'],independent_review=ref(out/'independent-review.json'),
   arithmetic_passed=result['passed'],reference_replay_passed=saved['all_reference_outputs_reproduced'],candidate_replay_passed=saved['all_candidate_outputs_reproduced'],
   passed=result['passed']and saved['all_reference_outputs_reproduced']and saved['all_candidate_outputs_reproduced'],
   measured_resources=dict(elapsed_seconds=time.monotonic()-started,peak_rss_bytes=actual.ru_maxrss*1024,user_seconds=actual.ru_utime,system_seconds=actual.ru_stime,output_bytes=total),
   actual_self_exit_claimed=False,native_reference_qualified=False,neuron_baseline_qualified=False,new_speedup_claim=False)
  once(out/'terminal.json',done);usage(out);print(json.dumps(dict(terminal=ref(out/'terminal.json'),passed=done['passed'],actual_self_exit_claimed=False)),flush=True)
  p.require(done['passed'],'Dory arithmetic or observer/replay check failed; all outputs retained')
 except BaseException as error:
  if created and not(out/'failure.json').exists():once(out/'failure.json',dict(error_type=type(error).__name__,error=str(error),owner=owner,request=request_ref,actual_self_exit_claimed=False,native_reference_qualified=False,new_speedup_claim=False))
  raise
 finally:
  signal.alarm(0)
  if old_alarm is not None:signal.signal(signal.SIGALRM,old_alarm)
  if barrier is not None:barrier.active=False
  try:drain_children()
  except BaseException:
   # Retain the lock for root recovery if an unexpected native child survives.
   # Do not turn a cleanup failure into an apparently released serial slot.
   print('Dory cleanup incomplete; serial lock retained for root recovery',file=sys.stderr,flush=True)
   while True:time.sleep(1)
  release(fd)

if __name__=='__main__':
 parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True);mode.add_argument('--prepare',action='store_true');mode.add_argument('--run',action='store_true')
 for name in ['catalog','catalog-sha256','finalizer-exit','finalizer-exit-sha256','run-id','request','request-sha256','review','review-sha256']:parser.add_argument('--'+name)
 args=parser.parse_args()
 if args.prepare:print(json.dumps(prepare(dict(path=args.catalog,sha256=args.catalog_sha256),dict(path=args.finalizer_exit,sha256=args.finalizer_exit_sha256),args.run_id)))
 else:run(dict(path=args.request,sha256=args.request_sha256),dict(path=args.review,sha256=args.review_sha256))
