"""Bounded exclusive capture writer and immutable metadata/payload reader.

Import is stdlib-only. Runtime caller supplies admission_guard callable, which
must authenticate actual process/unit/nonce and return the exact writer permit.
This library authenticates bytes/identities and never constructs a model.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys


MIB=2**20


def require(value,message):
    if not value:raise RuntimeError(message)


def identity(path):
    p=Path(path);s=p.lstat();require(stat.S_ISREG(s.st_mode) and s.st_nlink==1,'Not exclusive regular payload')
    return dict(realpath=str(p.resolve()),device=s.st_dev,inode=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)


def no_symlinks(path):
    p=Path(path);require(p.is_absolute(),'Absolute path required')
    for part in (p,*p.parents):require(not part.is_symlink(),'Symlink path rejected')
    return p


def read_json(path,wanted,max_bytes=8*MIB):
    p=no_symlinks(path);fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        s=os.fstat(fd);require(stat.S_ISREG(s.st_mode) and s.st_size<=max_bytes,'Metadata size/type')
        with os.fdopen(fd,'rb',closefd=False) as f:data=f.read(max_bytes+1)
        require(len(data)<=max_bytes and hashlib.sha256(data).hexdigest()==wanted,'Metadata digest')
        after=os.fstat(fd);require(all(getattr(after,k)==getattr(s,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')),'Metadata changed during read');return json.loads(data)
    finally:os.close(fd)


class ExclusiveRawWriter:
    """One fresh context directory, one owned process; no append or overwrite.

    Whole acquisition is partitioned into fixed context caps by the plan; each
    process cannot consume another context's cap. Root independently enforces
    the whole output tree cap. No tensor metadata claims bypass byte checks.
    """
    def __init__(self,directory,permit,admission_guard,raw_cap=256*MIB,tensor_cap=16*MIB,metadata_cap=2*MIB):
        require(permit['schema']=='minicpm5_layer0_writer_permit_v1','Writer permit schema')
        require(permit['directory']==str(directory) and permit['pid']==os.getpid(),'Writer owner/directory')
        require(isinstance(permit['frame_indices'],list) and 1<=len(permit['frame_indices'])<=4 and len(set(permit['frame_indices']))==len(permit['frame_indices']),'Writer frame identities')
        require(re.fullmatch('[0-9a-f]{32}',permit['attempt_id']) and re.fullmatch('[0-9a-f]{64}',permit['plan_sha256']),'Writer nonce/plan')
        require(0<raw_cap<=256*MIB and 0<tensor_cap<=16*MIB and 0<metadata_cap<=2*MIB,'Writer cap')
        self.directory=Path(directory);self.permit=dict(permit);self.guard=admission_guard
        self.raw_cap=raw_cap;self.tensor_cap=tensor_cap;self.metadata_cap=metadata_cap
        self.raw_bytes=0;self.metadata_bytes=0;self.index=0;self.records=[];self.fd=None;self.finalized=False;self.closed=False;self._initializing=True
        self.authenticate();no_symlinks(self.directory.parent);require(not self.directory.exists(),'Never reuse capture directory')
        self.directory.mkdir(mode=0o700)
        try:
            self.fd=os.open(self.directory,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
            self.dir_identity=os.fstat(self.fd);require(self.dir_identity.st_uid==os.getuid(),'Capture directory owner')
            self._initializing=False
        except BaseException:
            self.close();raise

    def _open(self):
        require(not self.closed and type(self.fd) is int and self.fd>=0,'Writer is closed or not initialized')

    def _writable(self):
        self._open();require(not self.finalized,'Writer finalized')

    def authenticate(self):
        # Only the constructor may authenticate before opening the directory.
        # Closed writers fail before guard, /proc or any filesystem operation.
        require(not self.closed,'Writer is closed')
        require(self._initializing or type(self.fd) is int and self.fd>=0,'Writer has no directory descriptor')
        observed=self.guard();require(observed==self.permit,'Admission permit changed')
        require(os.getpid()==self.permit['pid'],'Writer process changed')
        ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]
        require(ticks==self.permit['start_ticks'],'Writer start identity changed')
        if self.fd is not None:
            now=self.directory.lstat();opened=os.fstat(self.fd)
            require(stat.S_ISDIR(now.st_mode) and now.st_ino==opened.st_ino==self.dir_identity.st_ino and now.st_dev==opened.st_dev==self.dir_identity.st_dev,'Capture directory replaced')

    def _exclusive(self,name,data):
        self._writable()
        require(re.fullmatch(r'[a-z0-9-]+\.(raw|json)',name),'Filename schema')
        self.authenticate();fd=os.open(name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=self.fd)
        try:
            offset=0
            while offset<len(data):
                count=os.write(fd,data[offset:min(offset+MIB,len(data))]);require(count>0,'Short write');offset+=count
            os.fsync(fd);s=os.fstat(fd);require(stat.S_ISREG(s.st_mode) and s.st_nlink==1 and s.st_size==len(data),'Written payload size/type')
        finally:os.close(fd)
        return self.directory/name

    def write_bytes(self,buffer,metadata):
        self._writable()
        require(not self.finalized and self.index<68,'Writer finalized/count cap')
        view=memoryview(buffer).cast('B');n=view.nbytes
        require(n<=self.tensor_cap and self.raw_bytes+n<=self.raw_cap,'Raw byte cap')
        require(metadata['selected_bytes']==n and isinstance(metadata['context'],dict),'Snapshot byte/context mismatch')
        require(metadata['context']['arm']==self.permit['arm'] and metadata['context']['rank']==self.permit['rank'] and metadata['context']['call']==self.permit['call'],'Snapshot belongs to different context')
        require([f['index'] for f in metadata['context']['frames']]==self.permit['frame_indices'],'Snapshot frame ordering differs')
        encoded=json.dumps(metadata,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        require(self.metadata_bytes+len(encoded)<=self.metadata_cap,'Metadata byte cap')
        self.authenticate();sha=hashlib.sha256(view).hexdigest();name='tensor-%04d.raw'%self.index
        path=self._exclusive(name,view);ref=dict(path=str(path),sha256=sha,identity=identity(path),encoding='raw_little_endian',bytes=n)
        # Read only this just-written bounded output to verify serialization.
        fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=self.fd)
        try:
            check=hashlib.sha256();seen=0
            while True:
                block=os.read(fd,MIB)
                if not block:break
                seen+=len(block);require(seen<=n,'Written payload grew');check.update(block)
            require(seen==n and check.hexdigest()==sha and identity(path)==ref['identity'],'Written payload digest mismatch')
        finally:os.close(fd)
        self.records.append(dict(reference=ref,metadata=metadata));self.raw_bytes+=n;self.metadata_bytes+=len(encoded);self.index+=1
        return ref

    def tensor(self,tensor,metadata):
        self._writable()
        require(sys.byteorder=='little' and tensor.device.type=='cpu' and tensor.is_contiguous(),'Tensor serialization contract')
        require(tensor.numel()*tensor.element_size()==metadata['selected_bytes']<=self.tensor_cap,'Tensor byte admission')
        # No tobytes copy: expose contiguous tensor bytes via a bounded view.
        import torch
        buffer=memoryview(tensor.detach().reshape(-1).view(torch.uint8).numpy())
        return self.write_bytes(buffer,metadata)

    def finish(self,observer_receipt,endpoint_receipt):
        self._writable();self.authenticate();require(not self.finalized,'Already finalized')
        require(observer_receipt['complete'] is True and endpoint_receipt['passed'] is True,'Original endpoint reproduction required')
        require(len(self.records)>0,'No payloads captured')
        require(observer_receipt['arm']==self.permit['arm'] and observer_receipt['rank']==self.permit['rank'] and observer_receipt['call']==self.permit['call'] and [f['index'] for f in observer_receipt['frames']]==self.permit['frame_indices'],'Observer receipt owner/frame binding')
        if self.permit['arm']=='HF_native':
            require(endpoint_receipt['schema']=='minicpm5_layer0_native_endpoint_v1' and [endpoint_receipt['frame']['index']]==self.permit['frame_indices'],'Native endpoint frame binding')
        else:
            require(endpoint_receipt['schema']=='minicpm5_layer0_cpu_endpoint_v1' and endpoint_receipt['arm']==self.permit['arm'] and endpoint_receipt['rank']==self.permit['rank'] and endpoint_receipt['call']['index']==self.permit['call'] and {f['index'] for f in endpoint_receipt['call']['frames']}==set(self.permit['frame_indices']),'CPU endpoint owner binding')
        require(endpoint_receipt['original_scope_preserved'] is True and endpoint_receipt['raw_exact'] is True,'Endpoint scope/equality')
        report=dict(complete=True,permit=self.permit,raw_bytes=self.raw_bytes,payloads=self.records,observer=observer_receipt,endpoint=endpoint_receipt)
        data=(json.dumps(report,indent=2,allow_nan=False)+'\n').encode();require(len(data)<=self.metadata_cap,'Final metadata cap')
        path=self._exclusive('capture.json',data);self.finalized=True
        return dict(path=str(path),sha256=hashlib.sha256(data).hexdigest(),raw_bytes=self.raw_bytes)

    def close(self):
        if self.closed:return
        # Clear ownership before close: even a failed close cannot reopen the
        # object or cause a second close of a possibly reused descriptor.
        self.closed=True;self._initializing=False
        fd=self.fd;self.fd=None
        if fd is not None:os.close(fd)
