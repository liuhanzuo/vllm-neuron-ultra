"""Original Dory BF16 checkpoint loader; no imports of torch or work at import.

Only an independently admitted serial coordinator may call load_checkpoint.
The caller supplies its original torch module and a live lease/resource guard.
There is no downloader, runtime admission, service, CLI launcher or model forward.
"""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
import struct
import sys

Q=Path('/home/ubuntu/experiments/trn1-latest-models-20261031/research')
METADATA=Q/'dory-checkpoint-metadata-20261108-v1'
HEADER_SHA='17283cf0d92ee9381dbb083a64c06f3d8f527d395bc0010785363fbb5fb60428'
CONFIG_SHA='d2d086f5949598a043a00d70e2c28c45305bca7703682de3e4f990a7fd3279a4'
REVIEW_SHA='04125c5a8446c9d3c6dcb9cfe61b9dd36e4f6fed7630ae25897a6f66dabcf7eb'
CONTRACT_PATH=Q/'dory-native-reference-source-20261107-v1/dory_contract.py'
CONTRACT_SHA='4f0f9cdd3cb17ae23330944088dfc0334077b96463bb84e91b6194b8979fe95f'
FULL_SHA='207483664dfef54c4c4438863260d3ba423bf3d6009b43fe84264b281e9fda23'
FILE_BYTES=7950515472
DATA_BYTES=7950458880
HEADER_BYTES=56584
CHUNK_BYTES=2**20
MAX_TENSOR_BYTES=671088640
FIELDS=('device','inode','size','mtime_ns','ctime_ns')


def require(value,message):
    if not value:raise ValueError(message)


def unique(pairs):
    result={}
    for key,value in pairs:
        require(key not in result,'Duplicate JSON key')
        result[key]=value
    return result


def read_pinned(path,digest,cap):
    require(path.is_absolute()and path.resolve()==path and not any(x.is_symlink()for x in (path,*path.parents)),
            'Pinned source path without symlinks')
    raw=path.read_bytes()
    require(len(raw)<=cap and hashlib.sha256(raw).hexdigest()==digest,'Exact original metadata/source bytes')
    return raw


def source_contract():
    """Actual saved header and source only, not a future checkpoint identity."""
    raw=read_pinned(METADATA/'safetensors-header.json',HEADER_SHA,2**20)
    config_raw=read_pinned(METADATA/'config.json',CONFIG_SHA,16384)
    review=json.loads(read_pinned(METADATA/'metadata-review.json',REVIEW_SHA,65536),object_pairs_hook=unique)
    contract_raw=read_pinned(CONTRACT_PATH,CONTRACT_SHA,65536)
    # Import only this previously reviewed stdlib contract after hashing.
    name='_dory_checkpoint_loader_contract_v1'
    module=sys.modules.get(name)
    if module is None:
        spec=importlib.util.spec_from_file_location(name,CONTRACT_PATH)
        module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module
        try:exec(compile(contract_raw,str(CONTRACT_PATH),'exec',dont_inherit=True),module.__dict__)
        except BaseException:sys.modules.pop(name,None);raise
    config=json.loads(config_raw,object_pairs_hook=unique)
    contract=module.DoryContract.from_mapping(config,tensor_parallel_size=1)
    header=json.loads(raw,object_pairs_hook=unique)
    expected=contract.weight_shapes()
    require(set(header)==set(expected)and len(header)==507,'Exactly all original physical weights')
    rows=[]
    for name,row in header.items():
        require(set(row)=={'dtype','shape','data_offsets'}and row['dtype']=='BF16','Original BF16 tensor')
        require(row['shape']==list(expected[name]),'Original full tensor shape')
        begin,end=row['data_offsets']
        require(type(begin)is type(end)is int and 0<=begin<end and begin%2==0,
                'Original aligned tensor interval')
        require(end-begin==2*math.prod(expected[name])<=MAX_TENSOR_BYTES,'Complete bounded original tensor')
        rows.append(dict(name=name,shape=row['shape'],start=begin,end=end))
    rows.sort(key=lambda x:x['start'])
    cursor=0
    for row in rows:
        require(row['start']==cursor,'Original complete nonoverlapping intervals');cursor=row['end']
    require(cursor==DATA_BYTES and cursor+8+len(raw)==FILE_BYTES and len(raw)==HEADER_BYTES,
            'Whole original checkpoint extent')
    require(review['checkpoint_lfs']['sha256']==FULL_SHA and review['checkpoint_lfs']['size']==FILE_BYTES,
            'Official full checkpoint hash, not only header hash')
    return dict(header=raw,config=config,rows=rows,file_bytes=FILE_BYTES,data_bytes=DATA_BYTES,
                full_sha256=FULL_SHA,tensor_count=507,max_tensor_bytes=MAX_TENSOR_BYTES,
                parameters=3975229440,physical_weight_sharing='same 507 tensors for both original loops')


def identity(path,info=None):
    info=path.stat(follow_symlinks=False)if info is None else info
    return dict(path=str(path),device=info.st_dev,inode=info.st_ino,size=info.st_size,
                mtime_ns=info.st_mtime_ns,ctime_ns=info.st_ctime_ns)


def check_identity(path,fd,expected):
    info=os.fstat(fd)
    require(stat.S_ISREG(info.st_mode)and info.st_nlink==1,'Ordinary owned checkpoint')
    require(info.st_uid==os.getuid(),'Checkpoint belongs to current user')
    require(identity(path,info)==expected and identity(path)==expected,'Exact original checkpoint identity')


def _stream(fd,contract,guard,convert,validate,allocate=bytearray):
    """Single sequential read; callback tests may use tiny explicitly synthetic data."""
    h=hashlib.sha256();read_bytes=0;weights={};proof=[]
    guard()
    prefix=os.pread(fd,8,0)
    require(len(prefix)==8 and struct.unpack('<Q',prefix)[0]==len(contract['header']),'Exact original header prefix')
    raw=os.pread(fd,len(contract['header']),8)
    require(raw==contract['header'],'Original raw safetensors header, including padding')
    h.update(prefix);h.update(raw);read_bytes=8+len(raw)
    try:
        for row in contract['rows']:
            guard()
            length=row['end']-row['start']
            require(0<length<=contract['max_tensor_bytes'],'Finite full matrix allocation')
            require(read_bytes==8+len(raw)+row['start'],'Sequential read in official file order')
            buffer=allocate(length);require(len(buffer)==length,'One exact-sized backing buffer')
            view=memoryview(buffer);digest=hashlib.sha256()
            try:
                for offset in range(0,length,CHUNK_BYTES):
                    guard();wanted=min(CHUNK_BYTES,length-offset)
                    part=os.pread(fd,wanted,read_bytes)
                    require(len(part)==wanted,'Whole checkpoint tensor read without truncation')
                    view[offset:offset+wanted]=part
                    h.update(part);digest.update(part);read_bytes+=len(part)
            finally:view.release()
            tensor=convert(buffer,row['shape'])
            validate(tensor,row['shape'],guard)
            weights[row['name']]=tensor
            proof.append(dict(name=row['name'],shape=row['shape'],dtype='BF16',
                              bytes=length,sha256=digest.hexdigest()))
            del tensor,buffer
        guard()
        require(read_bytes==contract['file_bytes'] and not os.pread(fd,1,read_bytes),'Complete file without trailing bytes')
        require(h.hexdigest()==contract['full_sha256'],'Complete original LFS SHA256 verified before return')
        require(len(weights)==contract['tensor_count']and sum(x['bytes']for x in proof)==contract['data_bytes'],
                'Every original tensor loaded once')
        return weights,dict(full_sha256=h.hexdigest(),full_file_hash_verified=True,
                            file_bytes_read=read_bytes,weight_bytes=contract['data_bytes'],
                            tensors=proof,chunk_bytes=CHUNK_BYTES,
                            separately_qualified_model=False)
    except BaseException:
        weights.clear()
        raise


def load_checkpoint(checkpoint_path,expected_identity,*,torch,guard):
    """Future guarded runtime only. Returns physical BF16 tensors and load evidence.

    torch.frombuffer retains each bytearray owner, avoiding a full second copy.
    All file bytes are hashed in order while loading; no tensors are returned
    unless the entire official checkpoint digest matches. The lease guard must
    reject an unadmitted or expired run and enforce its existing resource budget.
    """
    require(callable(guard),'Actual coordinator lease/resource guard required')
    guard()
    require(sys.byteorder=='little','Packed BF16 little-endian host')
    contract=source_contract()
    path=Path(checkpoint_path)
    require(path.is_absolute()and path.resolve()==path and not any(x.is_symlink()for x in (path,*path.parents)),
            'Canonical original checkpoint, no symlinks')
    require(expected_identity['path']==str(path)and expected_identity['size']==FILE_BYTES,
            'Admitted original full checkpoint identity')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        check_identity(path,fd,expected_identity)
        def lease():
            guard();check_identity(path,fd,expected_identity)
        def convert(buffer,shape):
            return torch.frombuffer(buffer,dtype=torch.bfloat16).reshape(shape)
        def validate(tensor,shape,live):
            require(list(tensor.shape)==shape and tensor.dtype==torch.bfloat16 and tensor.device.type=='cpu',
                    'Actual original CPU BF16 tensor')
            require(tensor.is_contiguous()and tensor.storage_offset()==0,'Unpacked complete original tensor')
            flat=tensor.reshape(-1)
            for first in range(0,flat.numel(),CHUNK_BYTES//2):
                live()
                require(bool(torch.isfinite(flat[first:first+CHUNK_BYTES//2]).all().item()),'Finite actual checkpoint values')
        with torch.inference_mode():
            weights,proof=_stream(fd,contract,lease,convert,validate)
        lease()
        proof.update(checkpoint_identity=expected_identity,config_sha256=CONFIG_SHA,
                     header_sha256=HEADER_SHA,parameters=contract['parameters'],
                     physical_weight_sharing=contract['physical_weight_sharing'],
                     loader_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     model_forward_executed=False,peak_memory_fit_proven=False,new_speedup_claim=False)
        return weights,proof
    finally:os.close(fd)
