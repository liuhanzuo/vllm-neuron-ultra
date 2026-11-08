"""Finite original checkpoint download to existing NVMe; no native model work."""
from pathlib import Path
import hashlib,json,os,resource,signal,struct,sys,time,urllib.request
Q=Path(__file__).resolve().parent;R=Q.parent;sys.path[:0]=[str(Q),str(R)]
import run_dory_weight_storage_root_v2 as d
import dory_checkpoint_loader_v1 as loader
REV='f6c832e6963ddb74f62c75813a3e8c74b1af67b1'
URL='https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16/resolve/'+REV+'/model.safetensors?download=true'
OUT=Path('/mnt/nvme/models/dory1.0-4b-bf16-'+REV)
FINAL=OUT/'model.safetensors';PART=OUT/'model.safetensors.partial';PREFIX=Q/'dory-checkpoint-download-r1-20261108-v1'
PREDECESSOR=dict(path=str(Q/'dory-weight-storage-r1-20261108-v1-root-terminal.json'),sha256='ac648635e066c3dd0b2904b456c1a775d9953a5ce3fe1db36d03345badddacf1')
EXIT=dict(path=str(Q/'dory-weight-storage-r1-20261108-v1-finalizer-external-exit.json'),sha256='2d69489aa0ab984d4b332b085cc0dc1833e281544f4cba3caff47c216b24ea16')
BUDGET=dict(root_floor_bytes=28*2**30,nvme_floor_bytes=32*2**30,address_space_bytes=2*2**30,cpu_seconds=180,wall_seconds=1200,chunk_bytes=8*2**20)
def stream(source,destination,size,sha,guard,progress):
 total=0;digest=hashlib.sha256();checkpoint=0
 while True:
  guard(total);raw=source.read(min(BUDGET['chunk_bytes'],size-total+1))
  if not raw:break
  if total+len(raw)>size:raise ValueError('Checkpoint exceeds exact official length')
  if destination.write(raw)!=len(raw):raise OSError('Incomplete checkpoint write')
  digest.update(raw);total+=len(raw)
  if total-checkpoint>=256*2**20:progress(total);checkpoint=total
 if total!=size or digest.hexdigest()!=sha:raise ValueError('Incomplete or wrong official checkpoint bytes')
 progress(total);return total,digest.hexdigest()
def free(path):s=os.statvfs(path);return s.f_bavail*s.f_frsize
def main():
 contract=loader.source_contract();t=d.load(PREDECESSOR);e=d.load(EXIT)
 d.m.same(e['terminal'],PREDECESSOR,'Actual completed storage predecessor');d.m.same(e['owner'],t['verifier'],'Actual storage verifier owner');d.m.same(e['actual_exit'],dict(session_id=27138,exit_code=0,tool_chunk='f4e133'),'Actual storage verifier exit');d.require(e['actual_tool_exit_observed']is e['exact_owner_gone']is True and t['all18_saved_tensors_reverified']is True,'Storage actually complete')
 for path in (OUT.parent,*OUT.parent.parents):d.require(not path.is_symlink(),'Direct existing target parents')
 d.require(not OUT.exists(),'Fresh original-weight destination');d.require(free(R)>=BUDGET['root_floor_bytes']and free(OUT.parent)>=BUDGET['nvme_floor_bytes']+loader.FILE_BYTES+64*2**20,'Whole checkpoint fits while preserving both disk floors')
 fd,owner,lock=d.m.old.post.external.acquire_lock();start=time.monotonic();created=False;last_guard=[0.]
 try:
  d.m.same(lock,t['lock_identity'],'Same serial queue');d.m.p.clear_window();d.m.p.active_heavy()
  for p in (t['native_owner'],t['preparation_owner'],t['verifier']):d.m.p.gone(p)
  resource.setrlimit(resource.RLIMIT_AS,(BUDGET['address_space_bytes'],)*2);resource.setrlimit(resource.RLIMIT_CPU,(BUDGET['cpu_seconds'],BUDGET['cpu_seconds']+5));resource.setrlimit(resource.RLIMIT_FSIZE,(loader.FILE_BYTES,)*2);os.sched_setaffinity(0,{6});os.nice(19)
  def timeout(signum,frame):raise TimeoutError('Finite checkpoint download wall limit')
  signal.signal(signal.SIGALRM,timeout);signal.alarm(BUDGET['wall_seconds']);OUT.mkdir(mode=0o700);created=True
  def guard(total):
   now=time.monotonic();d.require(now-start<BUDGET['wall_seconds'],'Finite download duration')
   if now-last_guard[0]>=.5:
    d.m.p.lock_proof(owner,lock);d.m.p.clear_window();d.m.p.active_heavy();d.require(free(R)>=BUDGET['root_floor_bytes']and free(OUT)>=BUDGET['nvme_floor_bytes']+loader.FILE_BYTES-total,'Preserve root/NVMe free-space floors');last_guard[0]=now
  def progress(total):
   with Path(str(PREFIX)+'-progress.jsonl').open('a')as f:f.write(json.dumps(dict(bytes=total,total=loader.FILE_BYTES,elapsed_seconds=time.monotonic()-start))+'\n')
  admission=d.once(Path(str(PREFIX)+'-admission.json'),dict(owner=owner,lock_identity=lock,predecessor=PREDECESSOR,predecessor_finalizer_exit=EXIT,url=URL,revision=REV,target=str(FINAL),expected_sha256=loader.FULL_SHA,expected_bytes=loader.FILE_BYTES,budgets=BUDGET,source=d.ref(Path(__file__)),root_free_before=free(R),nvme_free_before=free(OUT),actual_self_exit_claimed=False))
  request=urllib.request.Request(URL,headers={'User-Agent':'trn1-dory-checkpoint-validation/1.0'})
  with urllib.request.urlopen(request,timeout=60)as response,PART.open('xb')as out:
   d.require(response.status==200,'Complete official checkpoint response');count,sha=stream(response,out,loader.FILE_BYTES,loader.FULL_SHA,guard,progress);out.flush();os.fsync(out.fileno())
  on_disk=hashlib.sha256();read=0
  with PART.open('rb')as source:
   while True:
    raw=source.read(BUDGET['chunk_bytes'])
    if not raw:break
    on_disk.update(raw);read+=len(raw);guard(loader.FILE_BYTES)
  d.require(read==loader.FILE_BYTES and on_disk.hexdigest()==loader.FULL_SHA,'Complete on-disk checkpoint independently rehashed')
  with PART.open('rb')as f:d.require(f.read(8)==struct.pack('<Q',loader.HEADER_BYTES)and f.read(loader.HEADER_BYTES)==contract['header'],'Exact original507-tensor header')
  d.require(not FINAL.exists(),'Never replace existing checkpoint');os.rename(PART,FINAL);dirfd=os.open(OUT,os.O_RDONLY|os.O_DIRECTORY)
  try:os.fsync(dirfd)
  finally:os.close(dirfd)
  d.m.same({n:d.m.p.sha(R/n)for n in d.m.CANONICAL},d.m.CANONICAL,'Canonical unchanged')
  result=d.once(Path(str(PREFIX)+'-result.json'),dict(schema='dory_original_checkpoint_download_v1',complete=True,owner=owner,admission=admission,checkpoint=loader.identity(FINAL),sha256=sha,bytes=count,official_revision=REV,network_hash_verified=True,whole_disk_hash_reverified=True,original507_tensor_header_exact=True,elapsed_seconds=time.monotonic()-start,root_free_after=free(R),nvme_free_after=free(OUT),original_weights_loaded=False,model_constructed=False,full_model_qualified=False,new_speedup_claim=False,actual_self_exit_claimed=False));print(json.dumps(dict(result=result,owner=owner,actual_self_exit_claimed=False)),flush=True)
 except BaseException as error:
  if created:d.once(Path(str(PREFIX)+'-failure.json'),dict(owner=owner,error_type=type(error).__name__,error=str(error),partial_path=str(PART),partial_bytes=PART.stat().st_size if PART.exists()else 0,actual_self_exit_claimed=False))
  raise
 finally:signal.alarm(0);d.m.old.post.external.release_lock(fd)
if __name__=='__main__':main()
