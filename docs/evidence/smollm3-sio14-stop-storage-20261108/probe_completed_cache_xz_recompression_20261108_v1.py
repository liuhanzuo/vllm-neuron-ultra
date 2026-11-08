"""Serial lossless recompression probe. Does not delete any original or old archive."""
import datetime,gzip,hashlib,json,lzma,os,resource,stat,sys,tarfile,time
from pathlib import Path
R=Path('/home/ubuntu/experiments/trn1-latest-models-20261031');Q=R/'research';sys.path.insert(0,str(R))
import nemotron9b_package_protocol_v12 as p
import minicpm5_python_trace_execute_v4 as life
from research.finalize_smollm3_sio9_root_20261107_v1 import physical_drain
from research.archive_completed_neuron_cache_20261108_v2 import check_candidate,identity
CATALOG=dict(path=str(Q/'root-post-smol-sio14-complete-predecessors-20261108-v1.json'),sha256='63a5c8bba41fb1e377d29029fa9fc212163774721e529c58f3d872952fc16980')
EXIT=dict(path=str(Q/'completed-cache-archival-r2-failure-root-exit-20261108-v1.json'),sha256='27b9eb59790fb35ac4203927546899c44f400727c96a0f7364e1fbe15b25a003')
OLD=Path('/mnt/nvme/retired-compile-cache-20261108-r2');DEST=Path('/mnt/nvme/retired-compile-cache-20261108-r3-probe');CAP=6*2**30

def digest(f):
 h=hashlib.sha256()
 while data:=f.read(2**20):h.update(data)
 return h.hexdigest()
def ref(path):
 with path.open('rb')as f:return dict(path=str(path),sha256=digest(f))
def once(path,value):
 raw=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode();assert len(raw)<4*2**20
 with path.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
 return ref(path)
def verify_members(path,manifest,guard):
 seen=set();dirs=set();root=Path('/mnt/nvme/neuron-cache/neuronxcc-2.23.6484.0+3b612583')
 with tarfile.open(path,'r:xz')as archive:
  for member in archive:
   guard()
   if member.isdir():
    name=str(root/member.name);assert name in manifest['directories']and name not in dirs;dirs.add(name);row=manifest['directories'][name]
   else:
    assert member.isfile()and member.name in manifest['files']and member.name not in seen;seen.add(member.name);row=manifest['files'][member.name]
    assert member.size==row['size']
    with archive.extractfile(member)as stream:assert digest(stream)==row['sha256']
   assert member.mode==stat.S_IMODE(row['mode'])and member.uid==row['uid']and member.gid==row['gid']
 assert len(seen)==1039 and len(dirs)==253 and seen==set(manifest['files'])and dirs==set(manifest['directories'])
 return dict(files_rehashed=len(seen),directories_verified=len(dirs))

def main():
 assert sys.dont_write_bytecode and __debug__ and not DEST.exists()
 prior=p.metadata.load(CATALOG);exited=p.metadata.load(EXIT);assert prior['root_finalized']and exited['exit_code']==1 and exited['actual_exit_observed_by_root']and not exited['deletion_performed']
 p.low.gone(exited['owner']);candidate,cache_root=check_candidate();manifest=p.metadata.load(exited['manifest'])
 assert len(manifest['files'])==1039 and len(manifest['directories'])==253
 fd,owner,lock=life.acquire();canonical=p.canonical();started=time.monotonic();created=False
 try:
  os.setpriority(os.PRIO_PROCESS,0,19);os.sched_setaffinity(0,{7});resource.setrlimit(resource.RLIMIT_AS,(2*2**30,2*2**30))
  physical_drain(prior);p.low.active_heavy();p.accelerator_idle();p.low.gone(exited['owner'])
  old=OLD/'cache.tar.gz';oldstat=p.file_stat(old);assert oldstat==exited['archive_file']['stat'];original=ref(old)
  assert os.statvfs('/mnt/nvme').f_bavail*os.statvfs('/mnt/nvme').f_frsize>=p.NVME_FLOOR+CAP
  DEST.mkdir(mode=0o700);created=True
  once(DEST/'start.json',dict(schema='completed_cache_recompression_probe_start_v1',owner=owner,lock=lock,source=ref(Path(__file__)),catalog=CATALOG,failed_maintenance=EXIT,original_archive=original,original_archive_stat=oldstat,manifest=exited['manifest'],canonical_sha256=canonical,xz_preset=9,output_cap_bytes=CAP,wall_limit_seconds=1800,deletion_performed=False))
  target=DEST/'cache.tar.xz';total=0;raw_hash=hashlib.sha256();check_time=time.monotonic();marker=0
  def guard():
   assert time.monotonic()<started+1800,'Recompression deadline'
   assert p.canonical()==canonical
   p.low.lock_proof(owner,lock)
   assert os.statvfs('/mnt/nvme').f_bavail*os.statvfs('/mnt/nvme').f_frsize>=p.NVME_FLOOR
   if target.exists():assert target.stat().st_size<=CAP
  with gzip.open(old,'rb')as source,lzma.open(target,'xb',preset=9)as output:
   while raw:=source.read(2**20):
    total+=len(raw);assert total<=CAP;raw_hash.update(raw);output.write(raw)
    if time.monotonic()>check_time+5:guard();check_time=time.monotonic()
    if total//(512*2**20)>marker:
     marker=total//(512*2**20);print(json.dumps(dict(uncompressed_bytes_processed=total,compressed_bytes_so_far=target.stat().st_size)),flush=True)
  with target.open('rb')as f:os.fsync(f.fileno())
  guard();assert p.file_stat(old)==oldstat and ref(old)==original
  members=verify_members(target,manifest,guard)
  h=hashlib.sha256();count=0
  with lzma.open(target,'rb')as f:
   while raw:=f.read(2**20):count+=len(raw);h.update(raw);guard()
  assert count==total and h.hexdigest()==raw_hash.hexdigest()
  check_candidate();physical_drain(prior);p.low.gone(exited['owner']);p.low.active_heavy();p.accelerator_idle();guard()
  allocated=target.stat().st_blocks*512
  result=dict(schema='completed_cache_recompression_probe_result_v1',observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),owner=owner,lock=lock,source=ref(Path(__file__)),catalog=CATALOG,failed_maintenance=EXIT,original_archive=original,original_archive_stat=oldstat,new_archive=ref(target),new_archive_stat=p.file_stat(target),manifest=exited['manifest'],uncompressed_tar_bytes=total,uncompressed_tar_sha256=raw_hash.hexdigest(),tar_stream_byte_identical=True,member_verification=members,new_archive_allocated_bytes=allocated,old_archive_allocated_bytes=old.stat().st_blocks*512,net_reclaimed_if_original_cache_and_old_duplicate_retired=candidate['allocated_bytes']-allocated,all253original_cache_entries_untouched=True,old_archive_untouched=True,deletion_performed=False,canonical_unchanged=True,canonical_sha256=canonical,external_self_exit_claimed=False,next_admission_must_verify_owner_exit=True,model_or_compiler_started=False,new_speedup_claim=False)
  print(json.dumps(dict(report=once(DEST/'result.json',result),new_archive_bytes=target.stat().st_size,net_original_cache_reclaim_bytes=result['net_reclaimed_if_original_cache_and_old_duplicate_retired'])),flush=True)
 except BaseException as e:
  if created:once(DEST/'failure.json',dict(error=repr(e),owner=owner,deletion_performed=False))
  raise
 finally:life.release(fd)
if __name__=='__main__':main()
