"""Archive eight exact inactive historical runtime memory CSVs before removal."""
import datetime,hashlib,json,os,tarfile
from pathlib import Path
Q=Path('/home/ubuntu/experiments/trn1-latest-models-20261031/research')
NAMES=['nrt_mem_log_device_0_'+n+'.csv' for n in ('6ae6176f','6ae6e56c','6ae7953d','6ae7d668','6ae849a6','6ae84ab8','6ae8bfaa','6ae8fcbb')]
def sha(b):return hashlib.sha256(b).hexdigest()
def ident(p):
 s=p.lstat();return dict(dev=s.st_dev,ino=s.st_ino,size=s.st_size,nlink=s.st_nlink,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
def free():
 s=os.statvfs('/');return s.f_bavail*s.f_frsize
def holders():
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:
   if int(proc.name) not in (os.getpid(),os.getppid()):assert not any(n.encode() in (proc/'cmdline').read_bytes() for n in NAMES)
   assert not any(('/tmp/'+n).encode() in (proc/'maps').read_bytes() for n in NAMES)
   for f in [proc/'cwd',*list((proc/'fd').iterdir())]:
    try:t=os.readlink(f)
    except OSError:continue
    assert not any('/tmp/'+n in t for n in NAMES)
  except (PermissionError,ProcessLookupError,FileNotFoundError):pass
archive=Path('/mnt/nvme/neuron-work/completed-nrt-logs-20261112-v1.tar.gz')
out=Q/'completed-nrt-log-archive-20261112-v1.json';assert not archive.exists() and not out.exists()
refs={}
for name in ('qwen35-package-full-r2-20261112-spec.json','minicpm5-operator-post-r2-serial-workloads-root-catalog-20261112-v113.json','continuation-handoff-20261112-v585-qwen-package-partition-precheck-failed.json'):
 b=(Q/name).read_bytes();assert not any(n.encode() in b for n in NAMES);refs[str(Q/name)]=sha(b)
files={}
for name in NAMES:
 p=Path('/tmp')/name;assert p.is_file() and not p.is_symlink() and p.stat().st_nlink==1
 before=ident(p);b=p.read_bytes();assert ident(p)==before
 files[name]=dict(identity=before,sha256=sha(b))
holders();before=free()
with archive.open('xb') as stream:
 with tarfile.open(fileobj=stream,mode='w:gz',compresslevel=6) as tar:
  for name in NAMES:tar.add(Path('/tmp')/name,arcname=name)
 stream.flush();os.fsync(stream.fileno())
with tarfile.open(archive,'r:gz') as tar:
 assert {m.name for m in tar.getmembers()}==set(NAMES)
 for m in tar:
  assert m.isfile() and sha(tar.extractfile(m).read())==files[m.name]['sha256']
holders()
for name,row in files.items():assert ident(Path('/tmp')/name)==row['identity']
receipt=dict(schema='completed_nrt_log_archival_v1',observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,checked_metadata=refs,archive=dict(path=str(archive),sha256=sha(archive.read_bytes()),bytes=archive.stat().st_size),source=dict(path=str(Path(__file__).resolve()),sha256=sha(Path(__file__).read_bytes())),no_current_source_input_or_catalog_references=True,no_active_holders=True,all_archived_bytes_verified=True,root_available_before=before,removal_pending=True)
with out.with_name(out.stem+'-before-removal.json').open('x') as f:json.dump(receipt,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
for name in NAMES:(Path('/tmp')/name).unlink()
receipt.update(removal_pending=False,root_available_after=free())
with out.open('x') as f:json.dump(receipt,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps(dict(receipt=str(out),reclaimed_bytes=receipt['root_available_after']-before,root_available=receipt['root_available_after'],archive=receipt['archive'])))
