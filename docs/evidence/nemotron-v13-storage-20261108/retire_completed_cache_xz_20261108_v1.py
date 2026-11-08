"""Retire the exact 253 completed caches only after an externally completed XZ proof.
Preserves failure records and the verified lossless replacement. No model work.
"""
import argparse,ast,datetime,hashlib,json,lzma,os,resource,stat,subprocess,sys,tarfile,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];Q=R/'research';sys.path.insert(0,str(R))
import nemotron9b_package_protocol_v12 as p
import minicpm5_python_trace_execute_v4 as life
from research.finalize_smollm3_sio9_root_20261107_v1 import physical_drain
from research import archive_completed_neuron_cache_20261108_v2 as original
from research import probe_completed_cache_xz_recompression_20261108_v1 as probe
DEST=Path('/mnt/nvme/retired-compile-cache-20261108-r4-retirement')
PROBE_EXIT=Q/'completed-cache-xz-probe-root-exit-20261108-v1.json'
SCAN_SOURCE=Q/'archive_inactive_vscode_server_root_20261108_v5.py'
SCAN_SHA='4f59a4e778d66e664d292cc3c27dbbc129b3e1bd6aa36701d0f27859834b9de8'
GIB=2**30

def ref(path):return original.ref(Path(path))
def once(path,value):return original.record(path,value)
def free():
 s=os.statvfs('/mnt/nvme');return s.f_bavail*s.f_frsize

def verified_archive(path,manifest,raw_sha,raw_size,guard):
 before=original.identity(path)
 assert stat.S_ISREG(before['mode']) and before['nlink']==1
 files=manifest['files'];directories=manifest['directories'];seen=set();seen_dirs=set()
 root=Path('/mnt/nvme/neuron-cache/neuronxcc-2.23.6484.0+3b612583')
 with tarfile.open(path,'r:xz') as archive:
  for member in archive:
   guard()
   assert not member.name.startswith('/') and '..' not in Path(member.name).parts
   if member.isdir():
    name=str(root/member.name);assert name in directories and name not in seen_dirs
    seen_dirs.add(name);row=directories[name]
   else:
    assert member.isfile() and member.name in files and member.name not in seen
    seen.add(member.name);row=files[member.name];assert member.size==row['size']
    with archive.extractfile(member) as stream:assert original.hash_stream(stream)==row['sha256']
   assert member.mode==stat.S_IMODE(row['mode']) and member.uid==row['uid'] and member.gid==row['gid']
 assert seen==set(files) and seen_dirs==set(directories)
 count=0;h=hashlib.sha256()
 with lzma.open(path,'rb') as stream:
  while data:=stream.read(2**20):count+=len(data);h.update(data);guard()
 assert count==raw_size and h.hexdigest()==raw_sha
 assert original.identity(path)==before
 return dict(files_rehashed=len(seen),directories_verified=len(seen_dirs),uncompressed_tar_bytes=count,uncompressed_tar_sha256=h.hexdigest(),tar_stream_byte_identical=True,archive=ref(path),identity=before)

def bind_manifest(candidate,manifest):
 root=original.validate_candidate(candidate)
 wanted={str(Path(row['path']).relative_to(root)):row for entry in candidate['candidates'] for row in entry['files']}
 assert manifest['directories']=={entry['path']:entry['directory_identity'] for entry in candidate['candidates']}
 assert set(wanted)==set(manifest['files'])
 for name,row in manifest['files'].items():
  assert {k:v for k,v in row.items() if k!='sha256'}==wanted[name]
  assert isinstance(row['sha256'],str) and len(row['sha256'])==64 and all(x in '0123456789abcdef' for x in row['sha256'])
 return root

def budget(candidate_allocated,xz_allocated,retained_metadata,available,old_allocated):
 assert all(type(x)is int and x>=0 for x in [candidate_allocated,xz_allocated,retained_metadata,available,old_allocated])
 net=candidate_allocated-xz_allocated-retained_metadata
 # Reserve 16 MiB for all new maintenance metadata and accounting updates.
 reserve=16*2**20
 assert net>=2*GIB+reserve,'At least original 2 GiB net recovery plus metadata reserve'
 projected=available+candidate_allocated+old_allocated-reserve
 assert projected>=p.NVME_FLOOR+p.NVME_CAP,'Original 48 GiB admission floor'
 return dict(net_reclaimed_lower_bound=net-reserve,projected_free_lower_bound=projected,metadata_reserve_bytes=reserve)

def process_scan(root):
 assert ref(SCAN_SOURCE)['sha256']==SCAN_SHA
 s=SCAN_SOURCE.read_text();fn=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name=='unused')
 ns=dict(Path=Path,subprocess=subprocess,json=json,SOURCE=root)
 exec(compile(ast.Module(body=[fn],type_ignores=[]),'reviewed_process_checker','exec'),ns)
 return ns['unused']()

def retire_directory(entry):
 directory=Path(entry['path']);assert original.identity(directory)==entry['directory_identity']
 assert {str(f) for f in directory.iterdir()}=={row['path'] for row in entry['files']}
 fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  st=os.fstat(fd);assert (st.st_dev,st.st_ino)==(entry['directory_identity']['device'],entry['directory_identity']['inode'])
  for row in entry['files']:
   path=Path(row['path']);assert original.identity(path)==row
   os.unlink(path.name,dir_fd=fd)
  os.fsync(fd)
 finally:os.close(fd)
 directory.rmdir()

def main(args):
 assert sys.dont_write_bytecode and __debug__ and not DEST.exists()
 assert ref(Path(probe.__file__))['sha256']=='b7246ef058f32bcd3e87ceba7dab6ea5a7cf0a74b0430e04cdd3d349593e63ba'
 assert ref(Path(original.__file__))['sha256']=='770e6f524ad89bbba06861b90463dce31eb0c21c985ebf15ce0e2614537b8fbc'
 exit_ref=dict(path=str(PROBE_EXIT),sha256=args.probe_exit_sha256);exited=p.metadata.load(exit_ref)
 assert exited['schema']=='completed_cache_xz_probe_root_exit_v1' and exited['reviewer']=='/root'
 assert exited['actual_exit_observed_by_root'] is exited['gone'] is exited['terminal'] is True and exited['exit_code']==0
 handle=exited['execution_handle'];assert handle['session_id']==43750 and handle['exit_code']==0 and handle['actual_completion_observed'] is True
 assert exited['result']['path']==str(probe.DEST/'result.json') and exited['started']['path']==str(probe.DEST/'start.json')
 result=p.metadata.load(exited['result']);start=p.metadata.load(exited['started']);failed=p.metadata.load(probe.EXIT)
 assert exited['owner']==result['owner']==start['owner'] and result['owner']['pid']==2818967 and result['owner']['start_ticks']=='390606329'
 assert result['catalog']==start['catalog']==probe.CATALOG and result['failed_maintenance']==start['failed_maintenance']==probe.EXIT
 assert result['manifest']==start['manifest']==failed['manifest'] and result['canonical_sha256']==start['canonical_sha256']==failed['canonical_sha256']
 assert result['source']==start['source']==ref(Path(probe.__file__)) and result['lock']==start['lock']
 assert result['tar_stream_byte_identical'] and result['member_verification']==dict(files_rehashed=1039,directories_verified=253)
 assert result['deletion_performed'] is False and result['all253original_cache_entries_untouched'] and result['old_archive_untouched']
 assert not (probe.DEST/'failure.json').exists()
 candidate,root=original.check_candidate();manifest=p.metadata.load(result['manifest']);assert bind_manifest(candidate,manifest)==root
 prior=p.metadata.load(probe.CATALOG);assert prior['root_finalized'] and len(prior['rows'])==863 and len(prior['processes'])==165
 assert not any(name==str(root) or name.startswith(str(root)+'/') for name in prior['source_sha256'])
 archive=Path(result['new_archive']['path']);old=Path(result['original_archive']['path'])
 assert archive==probe.DEST/'cache.tar.xz' and old==probe.OLD/'cache.tar.gz'
 fd,owner,lock=life.acquire();created=False;retired=[];old_removed=False;deletion_started=False;started=time.monotonic()
 try:
  os.setpriority(os.PRIO_PROCESS,0,19);os.sched_setaffinity(0,{7});resource.setrlimit(resource.RLIMIT_AS,(2*GIB,2*GIB))
  def guard():
   assert time.monotonic()<started+1200,'Retirement wall bound'
   p.low.lock_proof(owner,lock);assert p.canonical()==result['canonical_sha256'];assert free()>=p.NVME_FLOOR
  def drained():
   p.low.gone(exited['owner']);p.low.gone(failed['owner']);physical_drain(prior);p.low.active_heavy();p.accelerator_idle();guard()
  drained();first=process_scan(root)
  assert p.file_stat(archive)==result['new_archive_stat'] and ref(archive)==result['new_archive']
  assert p.file_stat(old)==result['original_archive_stat']==failed['archive_file']['stat'] and ref(old)==result['original_archive']==start['original_archive']
  DEST.mkdir(mode=0o700);created=True
  once(DEST/'start.json',dict(schema='completed_cache_xz_retirement_start_v1',owner=owner,lock=lock,source=ref(Path(__file__)),catalog=probe.CATALOG,failed_maintenance=probe.EXIT,probe_exit=exit_ref,probe_result=exited['result'],probe_start=exited['started'],candidate=ref(original.CANDIDATE),canonical_sha256=result['canonical_sha256'],deletion_performed=False))
  verification=verified_archive(archive,manifest,result['uncompressed_tar_sha256'],result['uncompressed_tar_bytes'],guard)
  assert verification['archive']==result['new_archive']
  for row in manifest['files'].values():
   path=Path(row['path']);assert original.identity(path)=={k:v for k,v in row.items() if k!='sha256'}
   assert ref(path)['sha256']==row['sha256'];guard()
  original.check_candidate();drained();last=process_scan(root)
  metadata=sum(f.stat().st_blocks*512 for directory in [probe.OLD,probe.DEST,DEST] for f in directory.iterdir() if f!=archive and f!=old)
  before_free=free();space=budget(candidate['allocated_bytes'],archive.stat().st_blocks*512,metadata,before_free,old.stat().st_blocks*512)
  assert p.file_stat(old)==result['original_archive_stat'] and p.file_stat(archive)==result['new_archive_stat']
  intent=dict(schema='completed_cache_xz_retirement_verified_v1',owner=owner,lock=lock,source=ref(Path(__file__)),catalog=probe.CATALOG,failed_maintenance=probe.EXIT,probe_exit=exit_ref,probe_result=exited['result'],probe_start=exited['started'],candidate=ref(original.CANDIDATE),archive=result['new_archive'],manifest=result['manifest'],verification=verification,canonical_sha256=result['canonical_sha256'],first_process_scan=first,final_process_scan=last,storage=space,original_allocated_bytes=candidate['allocated_bytes'],old_archive=result['original_archive'],old_archive_allocated_bytes=old.stat().st_blocks*512,archive_allocated_bytes=archive.stat().st_blocks*512,retained_metadata_allocated_bytes=metadata,nvme_free_before=before_free,deletion_performed=False)
  once(DEST/'verified-before-retirement.json',intent)
  dfd=os.open(DEST,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
  try:os.fsync(dfd)
  finally:os.close(dfd)
  original.check_candidate();guard();deletion_started=True
  for entry in candidate['candidates']:
   retire_directory(entry);retired.append(entry['path']);guard()
  assert all(not Path(d).exists() for d in manifest['directories'])
  assert p.file_stat(old)==result['original_archive_stat'];old.unlink();old_removed=True
  for directory in [root,probe.OLD]:
   dfd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
   try:os.fsync(dfd)
   finally:os.close(dfd)
  assert ref(archive)==result['new_archive'] and p.file_stat(archive)==result['new_archive_stat']
  for keep in [failed['started'],failed['failure'],result['manifest'],exited['started'],exited['result']]:assert ref(keep['path'])==keep
  drained();after=free();assert after>=p.NVME_FLOOR+p.NVME_CAP
  output=dict(intent,schema='completed_cache_xz_retirement_result_v1',observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),deletion_performed=True,all_original_directories_absent=True,old_duplicate_archive_absent=True,canonical_unchanged=True,nvme_free_after=after,freed_since_before_retirement=after-before_free,retired_directories=len(retired),retired_files=1039,external_self_exit_claimed=False,next_admission_must_verify_owner_exit=True,maintenance_owners_must_be_appended_to_complete_history=True,experiment_outputs_deleted=False,model_inputs_changed=False,model_or_compiler_started=False,new_speedup_claim=False,full_model_qualified=False,restore='Extract verified cache.tar.xz into '+str(root)+' and verify all hashes and metadata from retained files.json')
  print(json.dumps(dict(report=once(DEST/'result.json',output),nvme_free_gib=after/GIB,retired_directories=len(retired))),flush=True)
 except BaseException as error:
  if created:once(DEST/'failure.json',dict(error=repr(error),owner=owner,retired_directories=retired,old_duplicate_removed=old_removed,deletion_may_have_started=deletion_started,original_failure_retained=True))
  raise
 finally:life.release(fd)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--probe-exit-sha256',required=True);main(parser.parse_args())
