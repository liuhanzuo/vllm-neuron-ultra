"""Archive exact obsolete singleton mock directories, verify bytes, then remove."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile

Q = Path('/home/ubuntu/experiments/trn1-latest-models-20261031/research')
NAMES = ['mini-observer-full-plan-3xtlfzqn', 'mini-helper-complete-plan-5ls_r3yu',
 'mini-recovery-exact-review-c6a660xk', 'mini-recovery-terminal-memory-tx0eq5cb',
 'mini-recovery-terminal-memory-tkrj_hyr', 'mini-recovery-terminal-memory-c0db0mjn',
 'mini-recovery-terminal-memory-xb979xd4', 'mini-recovery-queue-memory-5rgxo_80',
 'mini-recovery-memory-cd1469h1']
REFS = ['qwen35-package-full-r1-20261112-spec.json',
 'minicpm5-operator-post-r2-serial-workloads-root-catalog-20261112-v112.json',
 'continuation-handoff-20261112-v584-qwen-package-precheck-failed.json']
def digest(data): return hashlib.sha256(data).hexdigest()
def ident(p):
 s=p.lstat()
 return dict(device=s.st_dev,inode=s.st_ino,size=s.st_size,nlink=s.st_nlink,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
def available(p):
 s=os.statvfs(p); return s.f_bavail*s.f_frsize
def check_holders():
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit(): continue
  try:
   cmd=(proc/'cmdline').read_bytes()
   if int(proc.name) not in (os.getpid(),os.getppid()):
    assert not any(n.encode() in cmd for n in NAMES),(proc.name,'cmdline')
   for f in [proc/'cwd',proc/'root',proc/'exe',*list((proc/'fd').iterdir())]:
    try: target=os.readlink(f)
    except OSError: continue
    assert not any('/tmp/'+n in target for n in NAMES),(proc.name,target)
   assert not any(('/tmp/'+n).encode() in (proc/'maps').read_bytes() for n in NAMES)
  except (PermissionError,ProcessLookupError,FileNotFoundError): pass

archive=Path('/mnt/nvme/neuron-work/completed-mock-tmp-20261112-v1.tar.gz')
receipt=Q/'completed-mock-tmp-archive-20261112-v1.json'
assert not archive.exists() and not receipt.exists()
references={}
for name in REFS:
 b=(Q/name).read_bytes(); assert not any(n.encode() in b for n in NAMES),name
 references[str(Q/name)]=digest(b)
files={}; dirs={}
for name in NAMES:
 root=Path('/tmp')/name; assert root.is_dir() and not root.is_symlink()
 dirs[str(root)]=ident(root)
 for p in sorted(root.rglob('*')):
  assert not p.is_symlink()
  if p.is_dir(): dirs[str(p)]=ident(p)
  else:
   assert p.is_file() and p.stat().st_nlink==1
   before=ident(p); b=p.read_bytes(); assert ident(p)==before
   files[str(p)]=dict(identity=before,sha256=digest(b))
check_holders(); before=available('/')
with archive.open('xb') as out:
 with tarfile.open(fileobj=out,mode='w:gz',compresslevel=6) as tar:
  for name in NAMES: tar.add(Path('/tmp')/name,arcname=name)
 out.flush(); os.fsync(out.fileno())
verified=set()
with tarfile.open(archive,'r:gz') as tar:
 for member in tar:
  if member.isdir(): continue
  assert member.isfile()
  original='/tmp/'+member.name; assert original in files
  assert digest(tar.extractfile(member).read())==files[original]['sha256']
  verified.add(original)
assert verified==set(files)
check_holders()
for p,row in files.items(): assert ident(Path(p))==row['identity']
for p,row in dirs.items(): assert ident(Path(p))==row
manifest=dict(schema='completed_mock_temp_archival_v1',observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 source=str(Path(__file__).resolve()),source_sha256=digest(Path(__file__).read_bytes()),
 archive=dict(path=str(archive),sha256=digest(archive.read_bytes()),bytes=archive.stat().st_size),
 files=files, directories=dirs,checked_metadata=references,archive_all_file_bytes_verified=True,
 no_current_source_input_or_catalog_references=True,no_active_fd_cwd_exe_mapping_or_command_holders=True,
 root_available_before=before,removal_pending=True)
pre=receipt.with_name(receipt.stem+'-before-removal.json')
with pre.open('x') as f: json.dump(manifest,f,sort_keys=True,indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
for name in NAMES: shutil.rmtree(Path('/tmp')/name)
manifest.update(removal_pending=False,root_available_after=available('/'),removed_directories=len(NAMES))
with receipt.open('x') as f: json.dump(manifest,f,sort_keys=True,indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
print(json.dumps(dict(receipt=str(receipt),archive=manifest['archive'],reclaimed_bytes=manifest['root_available_after']-before,root_available=manifest['root_available_after'])))
