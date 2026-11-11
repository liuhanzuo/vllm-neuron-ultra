"""Retire exact unused obsolete tool releases and disposable APT binary caches."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import minicpm5_layer0_external_supervisor_v4 as external
p = external.p
CHILD = r'''
import fcntl,hashlib,json,os,stat,time
from pathlib import Path
codex=Path('/home/ubuntu/.codex/packages/app-server-daemon')
old=codex/'releases/0.167.0-x86_64-unknown-linux-musl'
claude=Path('/home/ubuntu/.local/share/claude/versions')
locks=[]
for name in [str(codex/'install.lock'),'/var/cache/apt/archives/lock','/var/lib/apt/lists/lock','/var/lib/dpkg/lock-frontend','/var/lib/dpkg/lock']:
 fd=os.open(name,os.O_RDWR|os.O_NOFOLLOW);locks.append(fd)
 if name.endswith('install.lock'):fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 else:fcntl.lockf(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert (codex/'current').resolve()!=old
assert Path('/home/ubuntu/.local/bin/claude').resolve()==claude/'2.1.296'
assert old.is_dir() and not old.is_symlink()
link=old/'codex';assert link.is_symlink() and os.readlink(link)=='bin/codex'
link_identity=(link.lstat().st_dev,link.lstat().st_ino,link.lstat().st_mtime_ns,link.lstat().st_ctime_ns)
files=[claude/'2.1.289',claude/'2.1.292',Path('/var/cache/apt/pkgcache.bin'),Path('/var/cache/apt/srcpkgcache.bin')]
dirs=[]
for folder,subdirs,names in os.walk(old,followlinks=False):
 folder=Path(folder);assert not folder.is_symlink();dirs.append(folder)
 for name in subdirs:assert not (folder/name).is_symlink()
 files.extend(folder/name for name in names if folder/name!=link)
assert 4<len(files)<512
def ident(s):return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode,s.st_nlink)
rows=[];targets=set()
for path in files:
 s=path.lstat();assert stat.S_ISREG(s.st_mode) and s.st_nlink==1 and time.time()-s.st_mtime>3600
 assert s.st_uid==(0 if str(path).startswith('/var/') else 1000)
 h=hashlib.sha256()
 with path.open('rb')as f:
  for data in iter(lambda:f.read(1048576),b''):h.update(data)
 assert ident(path.lstat())==ident(s)
 rows.append(dict(path=str(path),identity=ident(s),allocated=s.st_blocks*512,sha256=h.hexdigest()))
 targets.add((s.st_dev,s.st_ino))
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  for fd in (proc/'fd').iterdir():
   try:s=fd.stat()
   except (FileNotFoundError,ProcessLookupError):continue
   assert (s.st_dev,s.st_ino)not in targets,('open target',str(fd))
  for line in (proc/'maps').read_text().splitlines():
   fields=line.split(None,5)
   if len(fields)>=5:
    major,minor=fields[3].split(':')
    assert (os.makedev(int(major,16),int(minor,16)),int(fields[4]))not in targets,('mapped target',str(proc))
 except (FileNotFoundError,ProcessLookupError):continue
assert (codex/'current').resolve()!=old and Path('/home/ubuntu/.local/bin/claude').resolve()==claude/'2.1.296'
before=os.statvfs('/');before=before.f_bavail*before.f_frsize
for row in rows:
 path=Path(row['path']);assert ident(path.lstat())==tuple(row['identity']);path.unlink()
assert link.is_symlink() and os.readlink(link)=='bin/codex'
assert (link.lstat().st_dev,link.lstat().st_ino,link.lstat().st_mtime_ns,link.lstat().st_ctime_ns)==link_identity
link.unlink()
for path in sorted(dirs,key=lambda p:len(p.parts),reverse=True):path.rmdir()
after=os.statvfs('/');after=after.f_bavail*after.f_frsize
print(json.dumps(dict(removed=rows,directories_removed=len(dirs),free_before=before,free_after=after,
 reclaimed=after-before,current_tool_versions_unchanged=True,experiment_and_model_files_deleted=False)))
'''


def main():
    out = ROOT/'research/obsolete-tool-version-reclaim-20261111-v2.json'
    p.require(not out.exists(), 'Fresh retirement evidence')
    p.user_bus()
    fd, owner, lock = external.acquire_lock()
    try:
        p.require(lock == {'device': 66306, 'inode': 8294421}, 'Original serial lock')
        p.lock_proof(owner, lock); p.clear_window(); p.active_heavy()
        p.once(ROOT/'research/obsolete-tool-version-reclaim-20261111-v2-live.json',
               dict(owner=owner, lock_identity=lock, source=external.ref(Path(__file__).resolve())))
        child = subprocess.run(['sudo','-n','/usr/bin/python3','-B','-c',CHILD],
                               text=True,capture_output=True,timeout=90)
        p.require(child.returncode == 0, child.stderr)
        result = json.loads(child.stdout)
        p.lock_proof(owner,lock); p.clear_window(); p.active_heavy()
        value = dict(schema='obsolete_tool_version_retirement_v1',
                     observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     owner=owner,lock_identity=lock,child_exit=0,result=result,
                     source=external.ref(Path(__file__).resolve()),self_exit_claimed=False,
                     full_model_qualified=False,new_speedup_claim=False)
        p.once(out,value)
        print(json.dumps(dict(result=external.ref(out),owner=owner,reclaimed=result['reclaimed'])),flush=True)
    finally:
        external.release_lock(fd)


if __name__ == '__main__':
    main()
