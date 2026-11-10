"""Exact layer1 official BF16 shard verification for a separately admitted worker.

Import is stdlib only. No launcher, standalone admission or model construction.
The caller must bind successful collection/audit lineage and the full source
closure in RuntimeContext before invoking this real-weight reader.
"""
import hashlib
import json
import os
import stat
import struct
from research import minicpm5_layer1_decode_boundary_v1 as boundary
from minicpm5_layer0_runtime_auth_v1 import identity
require=boundary.require
MIB=2**20
PROJECTIONS=('q_proj','k_proj','v_proj','o_proj')


def local_key(name):
    require(name in PROJECTIONS,'Original Q/K/V/O projection')
    return 'o_proj.o_proj.weight' if name=='o_proj' else 'qkv_proj.'+name+'.weight'


def shard_bytes(matrix,shape,axis,rank):
    """Select original row-major BF16 words without floating-point conversion."""
    require(type(matrix)is bytes and type(shape)is list and len(shape)==2
        and all(type(v)is int and 0<v<=2048 for v in shape)
        and type(axis)is int and axis in (0,1)
        and type(rank)is int and rank in (0,1),'Finite original matrix and TP2 axis')
    rows,cols=shape
    require(len(matrix)==rows*cols*2<=8*MIB and shape[axis]%2==0,'Exact complete matrix extent')
    if axis==0:
        width=len(matrix)//2
        return matrix[rank*width:(rank+1)*width]
    half=cols//2;out=bytearray(rows*half*2)
    for row in range(rows):
        start=2*(row*cols+rank*half)
        out[2*row*half:2*(row+1)*half]=matrix[start:start+2*half]
    return bytes(out)


def verify(context,rank,describe_shard,read_shard):
    """Compare all four actual shards byte-for-byte with authenticated official data.

    Context is the externally admitted original RuntimeContext. Callbacks expose
    actual native parameters only after its guard; tests provide synthetic bytes.
    """
    context.guard('cpu')
    require(context._sources_checked and type(rank)is int and rank in (0,1)
        and context.permit['rank']==rank,'Admitted source-authenticated CPU rank')
    require(callable(describe_shard)and callable(read_shard),'Actual model parameter accessors required')
    plan=context.plan;path=plan['official_layer0_weights']['path']
    record=context.verify_input(path)
    header_ref=plan['safetensors_header']
    header=context.read_json(header_ref['path'],header_ref['sha256'])
    slices=[row for row in boundary.weight_slices(plan,header)if row['rank']==rank]
    require(len(slices)==4,'All original layer1 official projections')
    context.guard('cpu');require(identity(path)==record['identity'],'Authenticated official file identity')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    def check_file():
        context.guard('cpu');info=os.fstat(fd);wanted=record['identity']
        require(stat.S_ISREG(info.st_mode)and (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns)==
            tuple(wanted[k]for k in ('device','inode','size','mtime_ns','ctime_ns')),
            'Exact opened original weight identity')
        require(identity(path)==wanted,'Official path unchanged')
    def read_span(offset,size,cap):
        require(type(offset)is int and type(size)is int and 0<=offset and 0<size<=cap
            and offset+size<=record['identity']['size'],'Bounded complete official byte range')
        out=bytearray(size);done=0
        while done<size:
            check_file();piece=os.pread(fd,min(65536,size-done),offset+done)
            require(piece,'Official matrix short read')
            out[done:done+len(piece)]=piece;done+=len(piece)
        check_file();return bytes(out)
    proofs=[];hashes={}
    try:
        check_file();header_length=struct.unpack('<Q',read_span(0,8,8))[0]
        require(0<header_length<=MIB,'Original bounded safetensors header')
        observed=json.loads(read_span(8,header_length,MIB))
        require(observed==header,'Complete official header equals authenticated metadata')
        for row in slices:
            context.guard('cpu');name=row['official_key'].split('.')[-2];key=local_key(name)
            entry=row['header_entry'];start,end=entry['data_offsets'];shape=entry['shape'];axis=row['shard_axis']
            require(type(start)is int and type(end)is int
                and 0<=start<end<=record['identity']['size']-8-header_length,
                'Official offsets include actual header length')
            matrix=read_span(8+header_length+start,end-start,8*MIB)
            expected=shard_bytes(matrix,shape,axis,rank)
            wanted_shape=list(shape);wanted_shape[axis]//=2
            require(describe_shard(key)==dict(shape=wanted_shape,dtype='torch.bfloat16',device='cpu'),
                'Actual original BF16 CPU shard geometry')
            context.guard('cpu');actual=read_shard(key)
            require(type(actual)is bytes and actual==expected,'Every official layer1 shard byte must match')
            check_file();digest=hashlib.sha256(expected).hexdigest();hashes[key]=digest
            proofs.append(dict(official_key=row['official_key'],local_key=key,rank=rank,
                shard_axis=axis,start=row['start'],stop=row['stop'],shape=wanted_shape,
                official_matrix_sha256=hashlib.sha256(matrix).hexdigest(),
                shard_sha256=digest,shard_bytes=len(expected),all_raw_bytes_equal=True))
            del matrix,expected,actual
        check_file()
    finally:os.close(fd)
    context.guard('cpu');require(identity(path)==record['identity'],'Original official file unchanged at completion')
    return dict(schema='minicpm5_layer1_official_shards_v1',complete=True,rank=rank,
        official_file=dict(path=path,sha256=record['sha256'],identity=record['identity']),
        header=header_ref,weights=proofs,hashes=hashes,
        q_head_ids=list(range(rank*8,(rank+1)*8)),kv_head_ids=[rank],
        all_four_projection_shards_raw_exact=True,precision_changed=False,
        attention_reexecuted=False,full_model_qualified=False,new_speedup_claim=False)


def check_parameter_contract(metadata,rank):
    require(type(rank)is int and rank in (0,1),'Original TP2 rank')
    expected={local_key(name):dict(shape=shape,dtype='torch.bfloat16',device='cpu')for name,shape in
        [('q_proj',[1024,2048]),('k_proj',[128,2048]),('v_proj',[128,2048]),('o_proj',[2048,1024])]}
    expected['rank_util.rank']=dict(shape=[1],dtype='torch.int32',device='cpu',values=[rank],requires_grad=False)
    require(metadata==expected,'Exact original attention projections plus SPMDRank parameter')
    return True


def verify_model(context,torch,model,rank):
    context.guard('cpu')
    require(context._sources_checked and context.permit['rank']==rank,'Actual admitted rank before model access')
    require(torch.is_inference_mode_enabled()and torch.get_num_threads()==1,'Original CPU inference mode')
    require(len(model.layers)==42,'Complete original model required')
    params=dict(model.layers[1].self_attn.named_parameters())
    metadata={}
    for key,value in params.items():
        row=dict(shape=list(value.shape),dtype=str(value.dtype),device=value.device.type)
        if key=='rank_util.rank':row.update(values=value.tolist(),requires_grad=value.requires_grad)
        metadata[key]=row
    check_parameter_contract(metadata,rank)
    def describe(key):
        context.guard('cpu');value=params[key]
        return dict(shape=list(value.shape),dtype=str(value.dtype),device=value.device.type)
    def read(key):
        context.guard('cpu');value=params[key]
        require(value.device.type=='cpu'and value.dtype==torch.bfloat16,'No tensor conversion across precision/device')
        return bytes(memoryview(value.detach().contiguous().reshape(-1).view(torch.uint8).numpy()))
    return verify(context,rank,describe,read)
