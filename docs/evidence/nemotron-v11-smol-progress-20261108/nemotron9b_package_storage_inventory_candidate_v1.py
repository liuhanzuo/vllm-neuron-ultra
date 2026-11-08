"""Candidate output inventory with an explicit temporary UUID socket directory.

No service/admission behavior. Existing regular-file, owner, count, link and
size rules are retained. Sockets remain counted, and never read as file bodies.
"""
import os,re,stat
from pathlib import Path
MIB=2**20;GIB=2**30

def require(value,message):
 if not value:raise RuntimeError(message)

def _walk_error(error):raise error

def inventory(root,allow_view=False,*,ipc_root=None):
 root=Path(root);total=0;count=0;sockets=0
 if ipc_root is not None:
  ipc_root=Path(ipc_root)
  require(ipc_root.is_absolute()and ipc_root.resolve()==ipc_root and ipc_root.name=='tmp'and ipc_root.parent.name=='temporary','Explicit canonical temporary socket directory')
  require(root==ipc_root.parent.parent or root==ipc_root.parent,'Socket exception only for complete work or temporary inventory')
 for base,dirs,files in os.walk(root,followlinks=False,onerror=_walk_error):
  for name in dirs:require(not(Path(base)/name).is_symlink(),'No directory symlinks')
  for name in files:
   path=Path(base)/name;info=path.lstat();count+=1
   require(count<100000 and info.st_uid==os.getuid(),'Owned bounded output tree')
   if path.is_symlink():require(allow_view and path.parent==root/'view','Only prepared model view symlinks');continue
   if stat.S_ISSOCK(info.st_mode):
    require(ipc_root is not None and path.parent==ipc_root and re.fullmatch(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',path.name)is not None,'Only declared temporary UUID IPC sockets')
    require(info.st_nlink==1 and info.st_size==0,'Single-link zero-byte IPC socket')
    sockets+=1;require(sockets<=64,'Bounded temporary IPC socket count')
   else:require(stat.S_ISREG(info.st_mode)and info.st_nlink==1,'Ordinary new output file')
   require(info.st_size<=16*GIB and(path.suffix!='.log'or info.st_size<=64*MIB),'Per-file evidence/log cap')
   total+=max(info.st_size,info.st_blocks*512)
 return total
