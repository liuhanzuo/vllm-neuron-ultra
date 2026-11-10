"""Guarded immutable FP32 archive bytes, with no Torch/NumPy/model import.

The original restricted pickle metadata reader and float64 metric implementation
remain in their frozen module. A caller must separately obtain serial admission.
"""
import array
import hashlib
import io
import math
import os
from pathlib import Path
import stat
import sys
import zipfile
from research import minicpm5_layer1_payload_io_v1 as owned
from research import review_minicpm5_full_bf16_trajectory_raw_v4 as original

V=original.V
TOLERANCE=original.TOLERANCE
metrics=original.metrics
failed=original.failed
CAPS=(65536,3*2**20,32*2**20)
require=owned.require

def read_exact(ref,expected_path,cap,guard,expected_identity=None):
    require(callable(guard),'External serial audit guard')
    guard();path=owned.safe_path(expected_path)
    require(type(cap)is int and cap in CAPS,'Only original tap/logits/HF bounds')
    require(ref['path']==str(path) and type(ref['bytes'])is int and 0<ref['bytes']<=cap
        and owned.raw.hash_string(ref['sha256']),'Exact bounded admitted payload')
    def check(info,wanted):
        require(stat.S_ISREG(info.st_mode)and info.st_uid==1000 and info.st_nlink==1
            and info.st_size==ref['bytes']<=cap,'Original owned regular payload extent')
        require(owned.file_identity(path,info)==wanted,'Payload identity unchanged')
    before=path.lstat();identity=owned.file_identity(path,before)
    if expected_identity is not None:require(identity==expected_identity,'Frozen input identity unchanged')
    check(before,identity)
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        check(os.fstat(fd),identity);parts=[];count=0;digest=hashlib.sha256()
        while True:
            guard()
            piece=os.read(fd,min(65536,ref['bytes']-count+1))
            if not piece:break
            count+=len(piece);require(count<=ref['bytes'],'Payload grew during read')
            parts.append(piece);digest.update(piece)
        check(os.fstat(fd),identity)
        require(count==ref['bytes']and digest.hexdigest()==ref['sha256'],'Exact whole payload digest and length')
    finally:os.close(fd)
    guard();check(path.lstat(),identity)
    return b''.join(parts)

class Archive:
    """Decode already authenticated archive bytes in memory, under original caps."""
    def __init__(self,data,cap,guard):
        require(type(data)is bytes and type(cap)is int and cap in (3*2**20,32*2**20)
            and 0<len(data)<=cap and callable(guard),'Complete bounded archive')
        self.guard=guard;guard();self.buffer=io.BytesIO(data);self.z=None
        try:
            self.z=zipfile.ZipFile(self.buffer);entries=self.z.infolist()
            require(0<len(entries)<128 and len({e.filename for e in entries})==len(entries),
                    'Finite unique original archive entries')
            require(all(0<=e.file_size<=cap and e.compress_type==zipfile.ZIP_STORED
                and e.compress_size==e.file_size and not e.flag_bits&1 for e in entries)
                and sum(e.file_size for e in entries)<=cap,'Only bounded uncompressed archive data')
            names=[e.filename for e in entries if e.filename.endswith('/data.pkl')]
            require(len(names)==1,'One original pickle metadata record')
            name=names[0];self.prefix=name.rsplit('/',1)[0]
            require(self.prefix and all('/'+x+'/'not in '/'+e.filename+'/'for e in entries for x in ('.','..'))
                and all(e.filename.startswith(self.prefix+'/')for e in entries),'Single archive namespace')
            require(self.z.getinfo(name).file_size<2**20,'Original1MiB pickle metadata cap')
            require(self.z.read(self.prefix+'/byteorder')==b'little'and sys.byteorder=='little',
                    'Original little-endian FP32 bytes')
            guard();raw=self.z.read(name);stream=io.BytesIO(raw)
            self.data=original.Reader(stream).load()
            require(stream.tell()==len(raw),'No trailing pickle objects')
            guard()
        except BaseException:
            self.close();raise

    def tensor(self,description,shape):
        self.guard()
        require(type(shape)is list and shape in ([4,V],[1,1,V],[4,1,V]),'Original full-vocabulary geometry')
        require(type(description)is dict and description.get('tensor')is True
            and description['shape']==shape and description['offset']==0,'Complete original FP32 tensor')
        require(description['storage']['count']==math.prod(shape),'Complete storage, no view crop')
        stride=[];product=1
        for size in reversed(shape):stride.insert(0,product);product*=size
        actual=description['stride']
        require(type(actual)is list and len(actual)==len(shape)
            and all(type(v)is int and v>=0 for v in actual)
            and all(size==1 or got==wanted for size,got,wanted in zip(shape,actual,stride)),
            'Contiguous non-singleton original tensor dimensions')
        key=description['storage']['key']
        require(type(key)is str and key.isdecimal(),'Exact bounded tensor storage key')
        path=self.prefix+'/data/'+key
        require(self.z.getinfo(path).file_size==4*product,'Every FP32 tensor byte')
        data=self.z.read(path);require(len(data)==4*product,'Complete tensor read')
        values=array.array('f');require(values.itemsize==4,'Native FP32 representation');values.frombytes(data)
        require(all(math.isfinite(x)for x in values),'Every saved FP32 value finite')
        self.guard()
        return values,hashlib.sha256(data).hexdigest()

    def close(self):
        if self.z is not None:self.z.close()
        self.buffer.close()

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
