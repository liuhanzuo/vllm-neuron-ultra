"""Concrete frozen-input CPU adapter. Import is stdlib only."""
import hashlib
import json
import os
import struct
from pathlib import Path
import zlib
from minicpm5_layer0_runtime_auth_v1 import require,identity
import minicpm5_layer0_capture_endpoints_v1 as endpoint
CACHE_BYTES=4*1*4224*128*2
MIB=2**20

class CPUAdapter:
    def __init__(self,context,torch):
        self.context=context;self.torch=torch;self.inputs_before=None;self.placeholders=None
        context.guard('cpu');require(context._sources_checked,'Sources must be authenticated')
        self.rank=context.permit['rank'];self.arm=context.permit['arm']
    def guard(self):self.context.guard('cpu')
    def tensor_metadata(self,value):return endpoint.tensor_metadata(self.torch,value)
    def graph_inputs(self,graph):
        self.guard();require(graph in self.context.plan['graph_calls'],'Unknown graph record')
        values=self.context.load_pt(graph['payload'],32*MIB)['inputs'];require(len(values)==7,'All seven graph inputs')
        require([self.tensor_metadata(x) for x in values]==graph['input_metadata'],'Original graph metadata')
        for row,value in zip(graph['compressed_inputs'],values):
            packed=self.context.read_bytes(row['path'],16*MIB);decoder=zlib.decompressobj();raw=decoder.decompress(packed,row['raw_bytes']+1)
            require(decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail,'Graph zlib framing')
            require(len(raw)==row['raw_bytes'] and hashlib.sha256(raw).hexdigest()==row['raw_sha256']==self.tensor_metadata(value)['sha256'],'Graph reversible exact bytes')
        self.inputs_before=graph['input_metadata'];self.placeholders=values[5:];return values
    def cpu_indices(self,values):
        self.guard();require(len(values)==7,'Seven graph inputs');torch=self.torch;out=[];records=[]
        for i,value in enumerate(values):
            if i<4:
                require(value.dtype in (torch.int32,torch.int64),'Original index dtype');actual=value.to(torch.int64)
                require(endpoint.tensor_digest(torch,actual.to(value.dtype))==endpoint.tensor_digest(torch,value),'Index widening roundtrip')
                records.append(dict(argument=i,source_dtype=str(value.dtype),execution_dtype=str(actual.dtype),values_roundtrip_exact=True,reason='CPU scatter/gather index tensors use int64'))
            else:actual=value
            out.append(actual)
        require(all(a is b for a,b in zip(out[5:],self.placeholders)),'Placeholder object identity');return out,records
    def manifest(self,ref,rank):
        self.guard();require(rank==self.rank,'Rank');value=self.context.read_json(ref['path'],ref['sha256'])
        require(value['complete'] is True and len(value['rows']) in (84,168),'Complete manifest')
        rows=[r for r in value['rows'] if r['rank']==rank]
        require([(r['rank'],r['name']) for r in rows]==[(rank,'kv_mgr.past_key_values.%d'%i) for i in range(84)],'All 84 states in order')
        for row in rows:
            require(row['shape']==[4,1,4224,128] and row['dtype']=='torch.bfloat16' and row['raw_bytes']==CACHE_BYTES and row['nonfinite_elements']==0,'State layout')
            wanted=self.context.plan['inputs'][row['file']];require(wanted['identity']==row['identity'] and wanted['sha256']==row['payload_sha256'],'State inherited payload binding')
        return rows
    def load_state(self,row):
        self.guard();torch=self.torch;data=self.context.read_bytes(row['file'],CACHE_BYTES+2048)
        if row.get('encoding')=='raw_bf16_le':require(len(data)==CACHE_BYTES,'Raw cache bytes')
        else:
            require('encoding' not in row and Path(row['file']).suffix=='.zlib','Cache encoding')
            decoder=zlib.decompressobj();data=decoder.decompress(data,CACHE_BYTES+1)
            require(decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail,'Cache zlib framing')
        require(len(data)==CACHE_BYTES and hashlib.sha256(data).hexdigest()==row['raw_sha256'],'Cache original raw bytes')
        tensor=torch.frombuffer(bytearray(data),dtype=torch.uint8).view(torch.bfloat16).reshape(row['shape'])
        require(bool(torch.isfinite(tensor).all()),'Finite cache required');return tensor
    def build_original_model(self,rank,phase):
        self.guard();require(rank==self.rank,'CPU rank')
        import minicpm5_layer0_cpu_constructor_v1 as constructor
        return constructor.build(rank,phase)
    def select_original_bucket(self,model,bucket):
        self.guard();import minicpm5_layer0_cpu_constructor_v1 as constructor
        return constructor.select_bucket(model,bucket)
    def prediction_hooks(self,model):
        self.guard();from minicpm5_history_tensors_v1 import PredictionHooks
        return PredictionHooks(model)
    def attention_trace_source(self,module):
        from minicpm5_layer0_python_trace_source_v1 import source_text
        self.verify_attention_module(module)
        return source_text(self.context,module,'cpu')
    def verify_attention_module(self,module):
        self.guard();path=Path(module.__file__).resolve();wanted=self.context.plan['sdk_attention_source']
        require(str(path)==wanted and str(path) in self.context.plan['source_sha256'],'Official SDK attention path')
    def verify_original_positions_masks(self,inputs,call,fixtures):
        self.guard();ids,mask,positions,slots,sampling,hidden,adapters=inputs;seq=slots.tolist()
        require(seq==([call['frames'][0]['slot']] if call['phase']=='prefill' else [0,1,2,3]),'Physical seq ids')
        require(hidden.tolist()==adapters.tolist()==[0]*len(seq),'Original zero placeholders')
        for frame in call['frames']:
            row=seq.index(frame['slot']);fixture=fixtures[frame['fixture']];length=len(fixture['input_ids']);logical=length+frame['step']-1
            tokens=ids[row].tolist();pos=positions[row].tolist();m=mask[row].tolist();require(set(m)<={0,1},'Binary attention mask')
            if frame['step']==0:
                require(tokens[:length]==fixture['input_ids'] and pos[:length]==list(range(length)) and m==[1]*length+[0]*(len(m)-length),'Original prefill logical inputs')
            else:require(tokens==[fixture['continuation_ids'][frame['step']-1]] and pos==[logical] and m==[1]*(logical+1)+[0]*(len(m)-logical-1),'Original decode logical inputs')
    def inputs_original_roundtrip_exact(self,inputs,expected):
        self.guard();require(self.inputs_before==expected['input_metadata'],'Wrong remembered input')
        require(all(a is b for a,b in zip(inputs[5:],self.placeholders)),'Placeholder identity changed')
        return [endpoint.tensor_digest(self.torch,x.to(self.torch.int32) if i<4 else x) for i,x in enumerate(inputs)]==[x['sha256'] for x in self.inputs_before]
    def verify_head_shard_proof(self,model,rank):
        self.guard();require(rank==self.rank,'Head rank');torch=self.torch
        weight=self.context.plan['official_layer0_weights'];record=self.context.verify_input(weight['path'])
        header=self.context.read_json(self.context.plan['safetensors_header']['path'],self.context.plan['safetensors_header']['sha256'])
        # No mmap, safe_open or native model. Authenticate the file before reading
        # its bounded header, then preadv directly into one <=8 MiB bytearray.
        # Tensor shares that buffer; strided shard hashing needs <=4 MiB more.
        rows={};state=model.state_dict(keep_vars=True);fd=os.open(weight['path'],os.O_RDONLY|os.O_NOFOLLOW)
        try:
            prefix=os.pread(fd,8,0);require(len(prefix)==8,'Safetensors length prefix')
            header_len=struct.unpack('<Q',prefix)[0];require(0<header_len<=MIB,'Safetensors header cap')
            header_bytes=os.pread(fd,header_len,8);require(len(header_bytes)==header_len,'Safetensors header short read')
            require(json.loads(header_bytes)==header,'Exact inherited safetensors header')
            del header_bytes
            for name,width,axis in [('q_proj',1024,0),('k_proj',128,0),('v_proj',128,0),('o_proj',1024,1)]:
                key='model.layers.0.self_attn.'+name+'.weight';spec=weight['tensors'][key]
                require(header[key]==spec and spec['dtype']=='BF16','Official layer0 tensor header')
                start,end=spec['data_offsets'];size=end-start
                require(size==2*spec['shape'][0]*spec['shape'][1] and 0<size<=8*MIB and 0<=start<end<=record['identity']['size']-8-header_len,'Bounded official matrix offsets')
                data=bytearray(size);offset=0
                while offset<size:
                    count=os.preadv(fd,[memoryview(data)[offset:]],8+header_len+start+offset)
                    require(count>0,'Official matrix short read');offset+=count
                official=torch.frombuffer(data,dtype=torch.bfloat16).reshape(spec['shape'])
                localkey='layers.0.self_attn.'+('o_proj.o_proj.weight' if name=='o_proj' else 'qkv_proj.'+name+'.weight')
                shard=state[localkey];piece=official[rank*width:(rank+1)*width,:] if axis==0 else official[:,rank*width:(rank+1)*width]
                require(list(piece.shape)==list(shard.shape) and piece.dtype==shard.dtype and endpoint.tensor_digest(torch,piece)==endpoint.tensor_digest(torch,shard),'Exact official head shard')
                rows[name]=dict(axis=axis,start=rank*width,stop=(rank+1)*width,sha256=endpoint.tensor_digest(torch,shard));del official,piece,data
        finally:os.close(fd)
        del state
        require(identity(weight['path'])==record['identity'],'Official weight changed')
        return dict(passed=True,rank=rank,q_head_ids=list(range(rank*8,(rank+1)*8)),kv_head_ids=[rank],weights=rows,logical_head_mapping_raw_proven=True,native_model_constructed=False,additional_tensor_upper_bytes=12*MIB,additional_header_upper_bytes=MIB)


def run(context,writer_factory):
    """Original one-rank process lifecycle; only called inside future serial job."""
    context.guard('cpu');require(context._sources_checked,'Complete source verification required')
    from minicpm5_layer0_capture_cpu_driver_v3 import run as driver
    with context.activate():
        import torch
        import importlib
        importlib.import_module('neuronx_distributed_inference.modules.lora_serving')
        from neuronx_distributed.parallel_layers import parallel_state
        context.record_cpu_import();torch.set_num_threads(1)
        torch.distributed.init_process_group('gloo')
        try:
            parallel_state.initialize_model_parallel(2,skip_collective_init=True)
            try:
                rank=context.permit['rank'];arm=context.permit['arm']
                require(torch.distributed.get_rank()==rank and torch.distributed.get_world_size()==2,'Original distributed geometry')
                records=[context.read_json(row['path'],row['sha256']) for row in context.plan['cpu_records']]
                expected=sorted([r for r in records if r['arm']==arm and r['rank']==rank],key=lambda r:r['call']['index'])
                require(len(expected)==14 and [x['call'] for x in expected]==context.plan['calls'],'Original CPU endpoint sequence')
                return driver(torch,context.plan,arm,rank,expected,context.plan['graph_calls'],CPUAdapter(context,torch),writer_factory,context.guard)
            finally:parallel_state.destroy_model_parallel()
        finally:torch.distributed.destroy_process_group()
