"""Retire only hash-verified complete old cache entries into a recoverable archive."""
import argparse,ast,datetime,hashlib,json,os,re,shutil,stat,subprocess,sys,tarfile
from pathlib import Path
R=Path('/home/ubuntu/experiments/trn1-latest-models-20261031');Q=R/'research';sys.path.insert(0,str(R))
import minicpm5_python_trace_execute_v4 as lifecycle
import nemotron9b_package_protocol_v11 as p
from research.finalize_smollm3_sio9_root_20261107_v1 import physical_drain
CANDIDATE=Q/'storage-nvme-completed-cache-candidates-20261108-v2.json'
DEST=Path('/mnt/nvme/retired-compile-cache-20261108-r2');OUT=Q/'storage-nvme-completed-cache-retirement-20261108-v2.json'
def hash_stream(f):
 h=hashlib.sha256()
 while True:
  b=f.read(1024*1024)
  if not b:break
  h.update(b)
 return h.hexdigest()
def ref(f):
 with f.open('rb')as stream:return dict(path=str(f),sha256=hash_stream(stream))
def record(f,v):
 with f.open('x')as stream:json.dump(v,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
 return ref(f)
def identity(f):
 a=f.lstat();return dict(path=str(f),size=a.st_size,allocated=a.st_blocks*512,device=a.st_dev,inode=a.st_ino,mode=a.st_mode,uid=a.st_uid,gid=a.st_gid,nlink=a.st_nlink,mtime_ns=a.st_mtime_ns,ctime_ns=a.st_ctime_ns)
def validate_candidate(value):
 assert value['schema']=='completed_cache_archival_preparation_v2'and value['reviewer']=='/root'
 assert value['archive_started']is value['deletion_performed']is False
 root=Path(value['cache_root'])
 assert root==Path('/mnt/nvme/neuron-cache/neuronxcc-2.23.6484.0+3b612583')and root.resolve()==root
 assert value['planned_archive_root']==str(DEST)
 names=set();file_names=set();total=0
 required={'compile_flags.json','model.done','model.hlo_module.pb','model.neff'}
 for entry in value['candidates']:
  directory=Path(entry['path']);assert directory.parent==root and re.fullmatch(r'MODULE_[0-9a-f]+\+[0-9a-f]+',directory.name)
  assert str(directory)not in names;names.add(str(directory))
  assert entry['directory_identity']['path']==str(directory)and stat.S_ISDIR(entry['directory_identity']['mode'])
  actual={Path(row['path']).name for row in entry['files']};assert required<=actual<=required|{'wrapped_neff.hlo'}
  subtotal=0
  for row in entry['files']:
   path=Path(row['path']);assert path.parent==directory and str(path)not in file_names;file_names.add(str(path))
   assert stat.S_ISREG(row['mode'])and row['nlink']==1 and row['uid']==os.getuid()and row['size']>=0
   if path.name=='model.done':assert row['size']==0
   subtotal+=row['allocated']
  assert subtotal==entry['allocated']and entry['done_marker']is True;total+=subtotal
  assert entry['compile_flags_ref']['path']==str(directory/'compile_flags.json')
 assert len(names)==value['cache_entries']==253 and len(file_names)==value['files']==1039
 assert total==value['allocated_bytes']==4607754240
 return root

def check_candidate():
 assert ref(CANDIDATE)['sha256']=='9354c71921613e1bdc74b2c66281e66b48074d46f87487d6d61ea0396a4dd0bd'
 value=json.loads(CANDIDATE.read_text());root=validate_candidate(value)
 for entry in value['candidates']:
  directory=Path(entry['path']);assert identity(directory)==entry['directory_identity']
  assert {str(f)for f in directory.iterdir()}=={row['path']for row in entry['files']}
  for row in entry['files']:assert identity(Path(row['path']))==row
  assert ref(directory/'compile_flags.json')==entry['compile_flags_ref']
 return value,root

def completed_catalog(args):
 assert args.catalog and args.catalog_sha256,'Actual finalized sio14 catalog required'
 path=Path(args.catalog)
 assert path==Q/'root-post-smol-sio14-complete-predecessors-20261108-v1.json'
 current=dict(path=str(path),sha256=args.catalog_sha256);prior=p.metadata.load(current)
 assert prior['schema']=='root_post_smollm3_sio14_complete_predecessors_v1'and prior['reviewer']=='/root'
 assert prior['root_finalized']is prior['all_planned_launches_accounted_for']is True
 assert len(prior['rows'])==prior['complete_count']==863 and len(prior['processes'])==165
 audit=p.metadata.load(prior['appended_terminal_audit']);assert audit['passed']and audit['all_groups_drained']
 exited=p.metadata.load(audit['references']['exited']);assert exited['actual_exit_observed_by_root']and exited['gone']and exited['terminal']
 assert exited['execution_handle']['session_id']==57994 and exited['execution_handle']['actual_completion_observed']
 assert exited['owner']['pid']==2721488 and exited['owner']['start_ticks']=='389776497'
 p.low.gone(exited['owner'])
 return current,prior

def main(args):
 assert sys.dont_write_bytecode and __debug__
 value,root=check_candidate()
 if args.check:
  print(json.dumps(dict(source_check_passed=True,entries=value['cache_entries'],files=value['files'],allocated_bytes=value['allocated_bytes'],payload_bytes_read=0,archive_started=False,deletion_performed=False)))
  return
 assert args.apply
 catalog_ref,prior=completed_catalog(args)
 assert not any(name==str(root)or name.startswith(str(root)+'/')for name in prior['source_sha256'])
 assert not DEST.exists()and not OUT.exists()
 s=(Q/'archive_inactive_vscode_server_root_20261108_v5.py').read_text();fn=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)and n.name=='unused');ns=dict(Path=Path,subprocess=subprocess,json=json,SOURCE=root);exec(compile(ast.Module(body=[fn],type_ignores=[]),'process_checker','exec'),ns)
 fd,owner,lock=lifecycle.acquire();canonical=p.canonical();failure=None
 try:
  physical_drain(prior);p.low.active_heavy();p.accelerator_idle();first=ns['unused']()
  original_free=os.statvfs('/mnt/nvme').f_bavail*os.statvfs('/mnt/nvme').f_frsize
  assert original_free>=p.NVME_FLOOR+value['allocated_bytes'],'Temporary archive fits unreduced floor'
  check_candidate()
  DEST.mkdir(mode=0o700)
  record(DEST/'start.json',dict(owner=owner,lock=lock,candidate=ref(CANDIDATE),source=ref(Path(__file__)),canonical_sha256=canonical,actual_completed_catalog=catalog_ref))
  entries={};directories={}
  for entry in value['candidates']:
   directory=Path(entry['path']);assert directory.parent==root and not directory.is_symlink();directories[str(directory)]=identity(directory)
   assert {str(f)for f in directory.iterdir()}=={row['path']for row in entry['files']}
   for row in entry['files']:
    path=Path(row['path']);assert identity(path)==row and stat.S_ISREG(row['mode'])and row['nlink']==1
    with path.open('rb')as stream:digest=hash_stream(stream)
    assert identity(path)==row;entries[str(path.relative_to(root))]=dict(row,sha256=digest)
  archive_path=DEST/'cache.tar.gz'
  with tarfile.open(archive_path,'x:gz',compresslevel=1)as archive:
   for directory in directories:archive.add(directory,arcname=Path(directory).name,recursive=False)
   for name,row in sorted(entries.items()):archive.add(row['path'],arcname=name,recursive=False)
  with archive_path.open('rb')as stream:archive_sha=hash_stream(stream);os.fsync(stream.fileno())
  seen=set();seen_dirs=set()
  with tarfile.open(archive_path,'r:gz')as archive:
   for member in archive:
    if member.isdir():
     key=str(root/member.name);assert key in directories and key not in seen_dirs;seen_dirs.add(key);row=directories[key]
    else:
     assert member.isfile()and member.name in entries and member.name not in seen;seen.add(member.name);row=entries[member.name]
     assert member.size==row['size']
     with archive.extractfile(member)as stream:assert hash_stream(stream)==row['sha256']
    assert member.mode==stat.S_IMODE(row['mode'])and member.uid==row['uid']and member.gid==row['gid']
  assert seen==set(entries)and seen_dirs==set(directories)
  for name,row in entries.items():assert identity(Path(row['path']))=={k:v for k,v in row.items()if k!='sha256'}
  for name,row in directories.items():assert identity(Path(name))==row
  manifest=record(DEST/'files.json',dict(files=entries,directories=directories))
  before_free=os.statvfs('/mnt/nvme').f_bavail*os.statvfs('/mnt/nvme').f_frsize
  archived_allocated=sum(f.stat().st_blocks*512 for f in DEST.iterdir())
  required_recovery=max(2*2**30,p.NVME_FLOOR+p.NVME_CAP-original_free+256*p.MIB)
  assert value['allocated_bytes']-archived_allocated>=required_recovery,'Archive must recover sufficient actual allocated space'
  directory_fd=os.open(DEST,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
  try:os.fsync(directory_fd)
  finally:os.close(directory_fd)
  physical_drain(prior);p.low.active_heavy();p.accelerator_idle();p.low.lock_proof(owner,lock);second=ns['unused']();assert p.canonical()==canonical
  check_candidate()
  intent=dict(owner=owner,lock=lock,candidate=ref(CANDIDATE),archive=dict(path=str(archive_path),sha256=archive_sha),manifest=manifest,files_verified=len(entries),directories_verified=len(directories),first_process_scan=first,final_process_scan=second,archive_allocated_bytes=archived_allocated,original_allocated_bytes=value['allocated_bytes'],source=ref(Path(__file__)),canonical_sha256=canonical,actual_completed_catalog=catalog_ref,required_net_recovery_bytes=required_recovery,deletion_performed=False)
  record(DEST/'verified-before-retirement.json',intent)
  for name,row in entries.items():assert identity(Path(row['path']))=={k:v for k,v in row.items()if k!='sha256'}
  for directory in directories:shutil.rmtree(directory)
  assert all(not Path(d).exists()for d in directories);assert p.canonical()==canonical
  after_free=os.statvfs('/mnt/nvme').f_bavail*os.statvfs('/mnt/nvme').f_frsize
  assert after_free>=p.NVME_FLOOR+p.NVME_CAP
  result=dict(intent,deletion_performed=True,external_self_exit_claimed=False,next_admission_must_verify_owner_exit=True,maintenance_owner_must_be_appended_to_complete_history=True,all_original_directories_absent=True,canonical_unchanged=True,nvme_free_after=after_free,freed_since_archive_written=after_free-before_free,net_allocated_reclaimed_bytes=value['allocated_bytes']-archived_allocated,restore='Extract cache.tar.gz into '+str(root)+' and verify all hashes from files.json',experiment_outputs_deleted=False,model_inputs_changed=False,model_or_compiler_started=False,observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
  out=record(OUT,result);print(json.dumps(dict(report=out,net_allocated_reclaimed_bytes=result['net_allocated_reclaimed_bytes'],nvme_free_gib=after_free/2**30)))
 except BaseException as error:
  if DEST.exists():record(DEST/'failure.json',dict(error=repr(error),owner=owner,original_error_retained=True))
  raise
 finally:lifecycle.release(fd)
if __name__=='__main__':
 parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True);mode.add_argument('--check',action='store_true');mode.add_argument('--apply',action='store_true');parser.add_argument('--catalog');parser.add_argument('--catalog-sha256');main(parser.parse_args())
