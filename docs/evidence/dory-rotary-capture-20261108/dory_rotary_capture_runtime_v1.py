"""Guarded Dory arithmetic control/capture; caller owns serial runtime admission.

No native imports or CLI. Does not authorize a run, load weights, or launch jobs.
"""
import hashlib,json,math,os,re,resource,stat,time
from pathlib import Path
import dory_rotary_arithmetic_v2 as control
import dory_rotary_arithmetic_capture_v2 as captured

CAP=8*2**20
DTYPES={'torch.int64':(8,'int64'),'torch.float32':(4,'float32'),'torch.bfloat16':(2,'bfloat16')}
require=control.require

def once(path,value):
 raw=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode();require(len(raw)<2**20,'Bounded arithmetic metadata')
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'wb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
 return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))

class Capture:
 def __init__(self,root,torch,guard):
  require(callable(guard),'Actual external serial guard required');guard()
  self.root=Path(root);require(self.root.is_absolute()and self.root.resolve()==self.root and not self.root.exists(),'Fresh canonical capture root')
  self.root.mkdir(mode=0o700);self.torch=torch;self.guard=guard;self.rows={};self.total=0
 def __call__(self,name,tensor):
  self.guard();require(re.fullmatch('[A-Za-z0-9_]{1,160}',name)and name not in self.rows,'Unique bounded tensor name')
  require(tensor.device.type=='cpu'and str(tensor.dtype)in DTYPES,'CPU admitted tensor dtype')
  size,dtype=DTYPES[str(tensor.dtype)];shape=list(tensor.shape)
  require(shape and all(type(n)is int and 0<n<=256 for n in shape),'Bounded full tensor shape')
  length=math.prod(shape)*size;require(length<=128*1024 and self.total+length<=CAP,'Original small boundary capture caps')
  raw=bytes(tensor.detach().contiguous().view(self.torch.uint8).reshape(-1).tolist());require(len(raw)==length,'Exact packed tensor size')
  path=self.root/(name+'.raw');fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
  with os.fdopen(fd,'wb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  self.guard();self.total+=length;self.rows[name]=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=length,shape=shape,dtype=dtype)
 def verify(self):
  self.guard();require(set(self.root.iterdir())=={Path(v['path'])for v in self.rows.values()},'No undeclared saved tensor')
  for row in self.rows.values():
   self.guard();path=Path(row['path']);info=path.lstat();require(stat.S_ISREG(info.st_mode)and info.st_nlink==1 and info.st_size==row['bytes'],'Original regular boundary file')
   with path.open('rb')as f:raw=f.read(row['bytes']+1)
   require(len(raw)==row['bytes']and hashlib.sha256(raw).hexdigest()==row['sha256'],'Saved boundary integrity')
  return dict(tensors=len(self.rows),raw_bytes=self.total,all_files_rehashed=True)

def run(*,torch,guard,output):
 require(callable(guard),'Independently admitted serial guard required');guard()
 output=Path(output);require(output.is_absolute()and output.resolve()==output and not output.exists(),'Fresh canonical output root')
 require(control.plan()==captured.plan(),'Capture keeps exact original arithmetic plan')
 output.mkdir(mode=0o700);started=time.monotonic()
 original=control.run_comparison(torch=torch,guard=guard);original_ref=once(output/'control.json',original)
 writer=Capture(output/'raw',torch,guard)
 observed=captured.run_comparison(torch=torch,guard=guard,emit=writer);observed_ref=once(output/'capture.json',observed)
 # Preserve both outcomes even when instrumentation or arithmetic parity fails.
 proof=writer.verify();require(proof['tensors']==53,'All53complete boundary tensors')
 manifest_ref=once(output/'raw-manifest.json',dict(schema='dory_rotary_saved_boundaries_v1',rows=writer.rows,**proof))
 usage=resource.getrusage(resource.RUSAGE_SELF);invariant=original==observed
 value=dict(schema='dory_rotary_control_capture_result_v1',complete=True,control=original_ref,capture=observed_ref,manifest=manifest_ref,
  observer_invariant=invariant,original_comparisons=len(original['comparisons']),captured_comparisons=len(observed['comparisons']),
  original_arithmetic_passed=original['passed'],captured_arithmetic_passed=observed['passed'],
  passed=invariant and original['passed']and observed['passed'],
  measured_resources=dict(elapsed_seconds=time.monotonic()-started,user_seconds=usage.ru_utime,system_seconds=usage.ru_stime,peak_rss_bytes=usage.ru_maxrss*1024),
  unique_position_comparisons=96,executed_position_comparisons=192,saved_tensors=53,official_dispatch_executed=False,original_weights_loaded=False,
  native_reference_qualified=False,neuron_baseline_qualified=False,new_speedup_claim=False)
 guard();result_ref=once(output/'result.json',value);return dict(result=result_ref,**value)
