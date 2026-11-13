"""Read-only source authentication for one already historical shared text pin.

All other source paths and all metadata/raw IO use the frozen singleton rules.
The one exception preserves its exact path, digest and eight stat fields; it
never creates, splits, relinks, replaces or removes a file. Import performs no IO.
"""
import os
from pathlib import Path
import stat

import minicpm5_retained_raw_transport_v1 as t

SHARED_PATH='/home/ubuntu/venvs/model_reference_2026/lib/python3.12/site-packages/__editable__.transformers-5.23.0.dev0.pth'
SHARED_SHA256='aa13fbb34e302ad14d5377603770dc22ca996b89b3dc92e3ac08fa33f0249f8f'
SHARED_IDENTITY=dict(realpath=SHARED_PATH,device=66306,inode=8822869,size=89,
    mtime_ns=1793420789350603196,ctime_ns=1793420789359602997,mode=0o100664,nlink=2)


def _shared_stat(path,value):
    actual=dict(t._stat_identity(path,value),mode=value.st_mode,nlink=value.st_nlink)
    t.require(stat.S_ISREG(value.st_mode)and actual==SHARED_IDENTITY,
        'Exact historical shared source identity including nlink2 is required')
    return actual


def identity(path):
    path=Path(path)
    if str(path)!=SHARED_PATH:return t.identity(path)
    t.require(path.is_absolute()and str(path)==str(path.resolve())
        and not any(p.is_symlink()for p in (path,*path.parents)),
        'Exact canonical nonsymlink historical shared source')
    return _shared_stat(path,path.lstat())


def _open(ident):
    if ident['realpath']!=SHARED_PATH:return t._open(ident)
    t.require(ident==SHARED_IDENTITY and identity(ident['realpath'])==ident,
        'Historical shared source identity changed before open')
    fd=os.open(ident['realpath'],os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:t.require(_shared_stat(ident['realpath'],os.fstat(fd))==ident,'Opened shared source descriptor changed')
    except BaseException:os.close(fd);raise
    return fd


def _unchanged(fd,ident):
    if ident['realpath']!=SHARED_PATH:return t._unchanged(fd,ident)
    t.require(_shared_stat(ident['realpath'],os.fstat(fd))==ident and identity(ident['realpath'])==ident,
        'Historical shared source changed across read')


def read(ref):
    """Authenticate source text only; never route prerequisite JSON refs here."""
    path=Path(ref['path'])
    if str(path)==SHARED_PATH:
        t.require(ref['sha256']==SHARED_SHA256,'Exact historical shared source SHA pin required')
    ident=identity(path);t.require(ident['size']<32<<20,'Original bounded source contract')
    fd=_open(ident)
    try:
        parts=[];count=0
        while True:
            block=os.read(fd,t.CHUNK)
            if not block:break
            count+=len(block);t.require(count<32<<20,'Source grew');parts.append(block)
        _unchanged(fd,ident)
    finally:os.close(fd)
    raw=b''.join(parts);t.require(t.digest(raw)==ref['sha256'],'Source digest changed')
    if path.suffix=='.json':return t.json.loads(raw)
    raw.decode('utf8');return raw
