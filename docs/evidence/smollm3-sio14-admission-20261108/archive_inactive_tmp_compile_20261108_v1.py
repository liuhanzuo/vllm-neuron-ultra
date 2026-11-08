"""Archive exact inactive temporary compiler output; preserve all file bytes."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,shutil,stat,sys,subprocess
R=Path(__file__).resolve().parents[1];Q=R/'research';sys.path.insert(0,str(R))
import nemotron9b_package_protocol_v10 as p
from nemotron9b_package_controller_v10 import acquire,release
ROOTS=[Path('/tmp/nxd_model'),Path('/tmp/torchinductor_ubuntu')]
DEST=Path('/mnt/nvme/finished-tmp-archives-20261108-smol14')
def digest(path):
 h=hashlib.sha256()
 with path.open('rb')as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def record(path,value):
 with path.open('x')as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def main():
 p.low.user_bus();fd,owner,lock=acquire()
 try:
  p.low.active_heavy();p.accelerator_idle();p.low.lock_proof(owner,lock)
  canonical=p.canonical();cat=p.read(Q/'nemotron9b-package-r4-20261108-complete-catalog-v1.json','d13b07ca6f7afbc0571806705b3f7b5a145cc780fcea58acd0cf737c4aac1f40')
  assert not DEST.exists();rows=[];before=shutil.disk_usage(R).free
  for root in ROOTS:
   assert root.resolve()==root and root.is_dir()and not any(n==str(root)or n.startswith(str(root)+'/')for n in cat['source_sha256'])
   for path in sorted(root.rglob('*')):
    s=path.lstat();assert s.st_uid==os.getuid()and not path.is_symlink()
    if path.is_dir():continue
    assert stat.S_ISREG(s.st_mode)and s.st_nlink==1
    rows.append(dict(source=str(path),destination=str(DEST/root.name/path.relative_to(root)),identity=list(p.file_identity(path)),sha256=digest(path),bytes=s.st_size,allocated_bytes=s.st_blocks*512))
  intent=dict(schema='inactive_temporary_compile_archive_v1',reviewer='/root',owner=owner,serial_lock=lock,roots=[str(x)for x in ROOTS],destination=str(DEST),files=rows,canonical_sha256=canonical,source_paths_not_pinned=True,prior_privileged_live_handle_scan_tool_chunk='757f47',all_root_processes_checked=True,live_handles_found=0,model_installations_unchanged=True)
  record(Q/'inactive-tmp-compile-archive-intent-20261108-v1.json',intent)
  DEST.mkdir(mode=0o700)
  for root in ROOTS:shutil.copytree(root,DEST/root.name,copy_function=shutil.copy2)
  for row in rows:
   source=Path(row['source']);target=Path(row['destination']);assert list(p.file_identity(source))==row['identity'];assert digest(target)==row['sha256']and target.stat().st_size==row['bytes']
   with target.open('rb')as stream:os.fsync(stream.fileno())
  for root in ROOTS:
   for directory in [DEST/root.name,*[x for x in(DEST/root.name).rglob('*')if x.is_dir()]]:
    d=os.open(directory,os.O_RDONLY|os.O_DIRECTORY);os.fsync(d);os.close(d)
  for row in rows:assert list(p.file_identity(Path(row['source'])))==row['identity']
  for root in ROOTS:shutil.rmtree(root)
  assert p.canonical()==canonical and all(not root.exists()for root in ROOTS)
  result=dict(intent,completed=True,all_archived_file_hashes_verified=True,source_files_removed_only_after_verified_copy=True,root_free_before_bytes=before,root_free_after_bytes=shutil.disk_usage(R).free,nvme_free_after_bytes=shutil.disk_usage('/mnt/nvme').free,original_root_admission_floor_met=shutil.disk_usage(R).free>=30*2**30,new_speedup_claim=False)
  record(Q/'inactive-tmp-compile-archive-result-20261108-v1.json',result)
  print(json.dumps(dict(files=len(rows),archived_bytes=sum(x['bytes']for x in rows),root_free_bytes=result['root_free_after_bytes'],admission_floor_met=result['original_root_admission_floor_met'],owner=owner)))
 finally:release(fd)
if __name__=='__main__':main()
