"""Durable bounded worker/capture receipts, including exact binary output.

Requires a pre-admitted output directory and caller serial/resource guard.
Never creates an experiment namespace, grants admission, or releases its lock.
Failed partial writes are retained. A local receipt lock serializes writers;
the root still budgets concurrent systemd logs and all other output producers.
"""
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time

from nemotron4b_process_handshake_v1 import require

CAP=8*2**20


def pack(value):
    binary=[];nodes=0;byte_count=0
    def visit(x,path,depth=0):
        nonlocal nodes,byte_count
        nodes+=1
        require(nodes<=50000 and depth<=32,'Bounded receipt structure')
        if type(x)is bytes:
            byte_count+=len(x)
            require(byte_count<=5*2**20 and len(binary)<512,'Bounded original binary output')
            binary.append(dict(path=path,base64=base64.b64encode(x).decode('ascii')))
            return None
        if type(x)is dict:
            require(all(type(k)is str for k in x),'JSON object keys are exact strings')
            return {k:visit(v,path+[k],depth+1)for k,v in x.items()}
        if type(x)is list:return [visit(v,path+[i],depth+1)for i,v in enumerate(x)]
        require(type(x)in(str,int,float,bool,type(None)),'Explicit original receipt value types')
        return x
    framed=dict(schema='nemotron4b_binary_receipt_v1',value=visit(value,[]),binary=binary)
    raw=(json.dumps(framed,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    require(len(raw)<=CAP,'Original per-file eight-MiB cap')
    return raw


def unpack(raw):
    require(type(raw)is bytes and len(raw)<=CAP,'Bounded complete receipt bytes')
    def unique(items):
        result={}
        for k,v in items:
            require(k not in result,'No duplicate receipt fields');result[k]=v
        return result
    framed=json.loads(raw,object_pairs_hook=unique)
    require(type(framed)is dict and set(framed)=={'schema','value','binary'}
            and framed['schema']=='nemotron4b_binary_receipt_v1'
            and type(framed['binary'])is list and len(framed['binary'])<=512,'Exact binary receipt framing')
    value=framed['value'];seen=[]
    for row in framed['binary']:
        require(type(row)is dict and set(row)=={'path','base64'}and type(row['path'])is list
                and len(row['path'])<=32 and row['path']not in seen,'Unique bounded binary output path')
        seen.append(row['path']);parent=value
        for key in row['path'][:-1]:parent=parent[key]
        decoded=base64.b64decode(row['base64'],validate=True)
        require(base64.b64encode(decoded).decode()==row['base64'],'Canonical original byte encoding')
        if not row['path']:
            require(value is None,'Root binary placeholder');value=decoded
        else:
            key=row['path'][-1]
            require((type(parent)is dict and type(key)is str and key in parent)
                    or(type(parent)is list and type(key)is int and 0<=key<len(parent)),
                    'Exact object/list byte position')
            require(parent[key]is None,'Original binary placeholder');parent[key]=decoded
    require(pack(value)==raw,'Exact canonical complete receipt and preserved binary values')
    return value


class ReceiptWriter:
    def __init__(self,directory,*,allowed_names,guard,work_cap=64*2**20,reserve=2**20):
        self.directory=Path(directory);self.guard=guard
        require(callable(guard)and type(allowed_names)in(list,tuple)
                and 0<len(allowed_names)<=32 and len(set(allowed_names))==len(allowed_names)
                and all(type(n)is str and re.fullmatch('[a-z0-9][a-z0-9-]*\.json',n)for n in allowed_names),
                'Explicit fixed receipt names and resource guard')
        require(type(work_cap)is int and type(reserve)is int and 0<reserve<work_cap<=64*2**20,
                'Original bounded work cap with terminal reserve')
        self.allowed=frozenset(allowed_names);self.cap=work_cap;self.reserve=reserve
        guard();self.identity=self.directory_identity()

    def directory_identity(self):
        p=self.directory
        require(p.is_absolute()and p.resolve()==p and not any(x.is_symlink()for x in(p,*p.parents)),
                'Original admitted canonical output directory')
        info=p.lstat();require(stat.S_ISDIR(info.st_mode),'Preexisting admitted output directory')
        return info.st_dev,info.st_ino

    def used(self):
        require(self.directory_identity()==self.identity,'Original output directory unchanged')
        total=0;count=0
        for p in self.directory.rglob('*'):
            count+=1;require(count<=4096,'Bounded complete output inventory')
            info=p.lstat();require(not stat.S_ISLNK(info.st_mode),'No output symlinks')
            if stat.S_ISREG(info.st_mode):total+=max(info.st_size,info.st_blocks*512)
            else:require(stat.S_ISDIR(info.st_mode),'Ordinary output files/directories only')
        return total

    def write(self,name,value):
        require(name in self.allowed,'Receipt outside exact output plan')
        self.guard();raw=pack(value);digest=hashlib.sha256(raw).hexdigest()
        require(self.directory_identity()==self.identity,'Original output directory before write')
        lock_path=self.directory/'receipt-writer.lock'
        lock=os.open(lock_path,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW|os.O_NONBLOCK|os.O_CLOEXEC,0o600)
        target=self.directory/name;fd=None;directory_fd=None
        try:
            info=os.fstat(lock);current=lock_path.lstat()
            require(stat.S_ISREG(info.st_mode)and info.st_nlink==1
                    and(info.st_dev,info.st_ino)==(current.st_dev,current.st_ino),'Original receipt writer lock')
            deadline=time.monotonic()+5
            while True:
                self.guard()
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:
                    require(time.monotonic()<deadline,'Bounded receipt lock wait');time.sleep(.01)
            require(self.used()+len(raw)+8192+self.reserve<=self.cap,'Output reservation retains terminal budget')
            self.guard();fd=os.open(target,os.O_CREAT|os.O_EXCL|os.O_RDWR|os.O_NOFOLLOW|os.O_CLOEXEC,0o600)
            offset=0
            while offset<len(raw):
                self.guard();n=os.write(fd,raw[offset:offset+65536]);require(n>0,'Complete receipt write');offset+=n
            os.fsync(fd)
            directory_fd=os.open(self.directory,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC)
            require((os.fstat(directory_fd).st_dev,os.fstat(directory_fd).st_ino)==self.identity,
                    'Exact original output directory descriptor')
            os.fsync(directory_fd)
            os.lseek(fd,0,os.SEEK_SET);parts=[]
            while True:
                self.guard();part=os.read(fd,65536)
                if not part:break
                parts.append(part);require(sum(map(len,parts))<=CAP,'Bounded receipt readback')
            require(b''.join(parts)==raw,'Complete exact written receipt readback')
            actual=target.lstat();opened=os.fstat(fd)
            require((opened.st_dev,opened.st_ino)==(actual.st_dev,actual.st_ino)
                    and actual.st_nlink==1 and actual.st_size==len(raw),'Original written receipt identity')
            self.guard();require(self.used()+self.reserve<=self.cap,'Original work cap retained after write')
            return dict(path=str(target),sha256=digest)
        finally:
            if directory_fd is not None:os.close(directory_fd)
            if fd is not None:os.close(fd)
            os.close(lock)
