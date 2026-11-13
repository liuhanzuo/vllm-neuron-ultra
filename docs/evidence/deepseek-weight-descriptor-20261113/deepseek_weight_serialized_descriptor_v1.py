"""Externally bound archive descriptor audit; no native imports or model load.

The original inert RuntimeReader streams embedded NEFF/graph-definition bytes.
This component compares exact saved WLT and named per-bucket weight interfaces;
it never promotes structural identity to loaded-value or layout qualification.
"""
import hashlib
import json
import os
from pathlib import Path
import pickle
import stat
import zipfile

from audit_qwen35_serialized_runtime import RuntimeReader, Payload, fields, tensor

BUCKETS=(128,256,512,1024,2048,4096)
TAGS=('context_encoding_model','token_generation_model')


def require(value,message):
    if not value:raise ValueError(message)


def file_identity(st):
    return dict(device=st.st_dev,inode=st.st_ino,size=st.st_size,mtime_ns=st.st_mtime_ns,
                ctime_ns=st.st_ctime_ns,mode=st.st_mode,nlink=st.st_nlink)


def blob(value):
    if isinstance(value,Payload):
        return dict(bytes=value.size,sha256=value.digest),value.interface
    require(type(value)is str,'Known serialized byte-string or streamed payload required')
    raw=value.encode('utf8','surrogateescape')
    return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()),raw


def interface(value):
    fingerprint,parsed=blob(value)
    if type(parsed)is dict:return fingerprint,parsed
    require(type(parsed)is bytes,'Complete MetaNeff interface required')
    result=dict(inputs=[],outputs=[],aliases={},graph_definition=None,scalars={})
    for number,wire,data in fields(parsed):
        if wire==0:
            require(str(number)not in result['scalars'],'Duplicate MetaNeff scalar')
            result['scalars'][str(number)]=data
        else:
            require(wire==2,'Known MetaNeff wire required')
            if number in (1,2):result['inputs'if number==1 else'outputs'].append(tensor(data))
            elif number==4:
                require(result['graph_definition']is None,'Single full serialized graph definition')
                result['graph_definition']=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
            elif number==6:
                pairs=fields(data);require(all(w==0 for n,w,v in pairs),'Integer aliases')
                d={n:v for n,w,v in pairs};require(len(d)==len(pairs)and set(d)<={1,2},'Unique alias fields')
                out,inp=d.get(1,0),d.get(2,0)
                require(out not in result['aliases'],'Unique alias output');result['aliases'][out]=inp
            elif number==3:
                require('model_config'not in result,'Unique model config')
                result['model_config']={str(n):v for n,w,v in fields(data)}
            elif number==5:
                require('name'not in result,'Unique model name');result['name']=data.decode('utf8')
            else:raise ValueError('Unsupported MetaNeff field '+str(number))
    return fingerprint,result


def descriptor(value,origin):
    require(getattr(value,'origin',None)==origin and hasattr(value,'state'),'Exact inert serialized class: '+origin)
    return value.state


def _tag(value):
    require(type(value)in (int,str)and len(str(value))<=256,'Finite original serialization tag')
    return value


def describe(obj,script_records,guard):
    """Pure descriptor checking; no live model or filesystem access."""
    guard()
    state=obj.state['nxd_model']
    state=descriptor(state,'__torch__.neuronx_distributed.trace.spmd.NxDModel')
    require(state['start_rank_id']==0,'Original rank zero start')
    loader=state['weight_loader']
    if loader is None:wlt=None
    else:
        saved=descriptor(loader,'__torch__.torch.classes.neuron.LayoutTransformation')
        require(type(saved)in (tuple,list)and len(saved)==4,'Complete actual WLT four-field state')
        neff,meta,tag,ranks=saved;require(type(ranks)is int and ranks==2,'Original WLT TP2')
        ninfo,nparsed=blob(neff);require(ninfo['bytes']>0 and (nparsed is None or type(nparsed)is bytes),'Complete NEFF bytes')
        minfo,mi=interface(meta)
        require(mi['inputs']and len(mi['inputs'])==len(mi['outputs']),'Complete WLT input/output records')
        for kind in ('inputs','outputs'):
            rows=mi[kind]
            require(all(x['name']==('input'if kind=='inputs'else'output')+str(i)for i,x in enumerate(rows)),
                    'Actual WLT ordered tensor names')
            require(all(x.get('checkpoint_key')for x in rows)and len({x['checkpoint_key']for x in rows})==len(rows),
                    'Complete unique WLT keys')
        require([x['checkpoint_key']for x in mi['inputs']]==[x['checkpoint_key']for x in mi['outputs']],
                'WLT retains complete input/output key order')
        wlt=dict(neff=ninfo,metaneff=minfo,serialization_tag=_tag(tag),local_ranks=ranks,interface=mi)
    routes={}
    for key,value in state['input_shape_map'].items():
        require(type(value)in (tuple,list)and len(value)==2,'Exact serialized route')
        tag,index=value;require(tag in TAGS and type(index)is int and 0<=index<6,'Original12 route coordinates')
        route=(tag,index);require(route not in routes,'Unique original graph route')
        shapes=json.loads(key);batch=1 if tag==TAGS[0]else 4;bucket=BUCKETS[index]
        require(shapes[0]==[batch,bucket if batch==1 else 1]and shapes[1]==[batch,bucket],
                'Original actual CTE/TKG input and mask shapes')
        routes[route]=shapes
    require(set(routes)=={(tag,i)for tag in TAGS for i in range(6)},'All12 original routes')
    models=state['models'].state;require(set(models)==set(TAGS),'Both original graph groups')
    graphs=[]
    for tag in TAGS:
        saved=models[tag].state['models'];require(len(saved)==6,'All six original buckets')
        for index,model in enumerate(saved):
            guard();parts=descriptor(model,'__torch__.torch.classes.neuron.SPMDModel')
            require(type(parts)in (tuple,list)and len(parts)==5,'Complete actual SPMD five-field state')
            neff,meta,version,local,world=parts;require(local==world==2,'Original TP2/world2')
            ninfo,_=blob(neff);minfo,mi=interface(meta)
            require(ninfo['bytes']>0 and all(x['name']=='input'+str(i)for i,x in enumerate(mi['inputs'])),
                    'Complete NEFF and ordered original graph inputs')
            weights=[dict(input_index=i,tensor=x)for i,x in enumerate(mi['inputs'])if x['type']==3]
            states=[dict(input_index=i,tensor=x)for i,x in enumerate(mi['inputs'])if x['type']==2]
            require(weights and all(x['tensor'].get('checkpoint_key')for x in weights),'Actual named weight inputs')
            require(len({x['tensor']['checkpoint_key']for x in weights})==len(weights),'Unique graph weight keys')
            require(len(states)==72 and {x['tensor']['checkpoint_key']for x in states}==
                    {'kv_mgr.past_key_values.'+str(i)for i in range(72)},'All72 original KV names per graph')
            require(len(mi['aliases'])==72 and set(mi['aliases'].values())=={x['input_index']for x in states},
                    'All72 actual state targets are aliased')
            require(all(x['name']=='output'+str(i)for i,x in enumerate(mi['outputs'])),
                    'Complete ordered actual graph outputs')
            for out,inp in mi['aliases'].items():
                require(0<=out<len(mi['outputs'])and mi['outputs'][out]['shape']==mi['inputs'][inp]['shape']
                    and mi['outputs'][out]['data_type']==mi['inputs'][inp]['data_type'],
                    'Actual alias output must preserve the corresponding full state geometry')
            graphs.append(dict(tag=tag,index=index,bucket=BUCKETS[index],route_shapes=routes[(tag,index)],
                neff=ninfo,metaneff=minfo,serialization_tag=_tag(version),local_ranks=local,world_size=world,
                weight_inputs=weights,state_inputs=states,aliases=mi['aliases']))
    common={x['tensor']['checkpoint_key']:{k:v for k,v in x['tensor'].items()if k!='name'}for x in graphs[0]['weight_inputs']}
    for graph in graphs:
        named={x['tensor']['checkpoint_key']:{k:v for k,v in x['tensor'].items()if k!='name'}for x in graph['weight_inputs']}
        require(named==common,'All12 graphs declare the same named weight interfaces')
    if wlt is not None:
        require({x['checkpoint_key']for x in wlt['interface']['inputs']}==set(common),'Complete WLT versus graph key coverage')
    require(type(script_records)is dict and script_records,'Full saved code identity inventory required')
    return dict(schema='deepseek_serialized_weight_descriptor_v1',wlt=wlt,graphs=graphs,
                all_serialized_script_records=script_records,all12_graphs_present=True,
                named_weights_per_graph=len(common),loaded_values_observed=False,
                graph_weight_ABI_semantics_proven=False,layout_semantics_proven=False,
                inference_admitted=False,full_model_qualified=False,new_speedup_claim=False)


class GuardedStream:
    def __init__(self,stream,guard):self.stream,self.guard=stream,guard
    def read(self,*args):
        self.guard();value=self.stream.read(*args);self.guard();return value
    def readline(self,*args):
        self.guard();value=self.stream.readline(*args);self.guard();return value
    def seek(self,*args):
        self.guard();value=self.stream.seek(*args);self.guard();return value
    def tell(self):return self.stream.tell()
    def seekable(self):return self.stream.seekable()


class BoundRuntimeReader(RuntimeReader):
    dispatch=RuntimeReader.dispatch.copy()
    def reject_extension(self):
        raise ValueError('Pickle extension registry cannot bypass inert find_class')


for _opcode in (pickle.EXT1,pickle.EXT2,pickle.EXT4):
    BoundRuntimeReader.dispatch[_opcode[0]]=BoundRuntimeReader.reject_extension


def read_archive(binding,guard,limits):
    """Runtime-only read; caller binds exact original SHA, full identity and caps.

This hashes the complete model.pt archive, including any storage-member bytes,
but never opens/deserializes tensor-storage members or executes model globals.
All output is returned; parent observes true exit
and seals it separately. No native/runtime/lock/model invocation occurs here.
"""
    require(__debug__ and callable(guard),'Assertions and authenticated owning guard required')
    require(set(binding)=={'path','sha256','identity'}and set(limits)=={'archive_bytes','pickle_bytes','members','code_bytes'},
            'Exact external artifact binding and finite reader limits')
    require(all(type(v)is int and v>0 for v in limits.values()),'Positive bounded reader limits')
    path=Path(binding['path']);require(path.is_absolute()and path.resolve()==path
        and not any(p.is_symlink()for p in (path,*path.parents)),'Ordinary canonical bound model archive')
    guard();fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        with os.fdopen(fd,'rb',closefd=False) as raw:
            observed=file_identity(os.fstat(fd));require(observed==binding['identity']
                and stat.S_ISREG(observed['mode'])and 0<observed['size']<=limits['archive_bytes'],
                'Exact authenticated original archive identity and cap')
            h=hashlib.sha256()
            while True:
                guard();data=raw.read(4<<20)
                if not data:break
                h.update(data)
            require(h.hexdigest()==binding['sha256'],'Complete original archive SHA256')
            raw.seek(0)
            with zipfile.ZipFile(GuardedStream(raw,guard))as archive:
                infos=archive.infolist();require(len(infos)<=limits['members']and len({i.filename for i in infos})==len(infos),
                        'Bounded unique archive members')
                data_members=[i for i in infos if i.filename.endswith('/data.pkl')]
                require(len(data_members)==1 and 0<data_members[0].file_size<=limits['pickle_bytes'],'One complete bounded data pickle')
                scripts={};total=0
                for info in infos:
                    if '/code/'not in info.filename or not info.filename.endswith('.py'):continue
                    total+=info.file_size;require(total<=limits['code_bytes'],'Bounded complete serialized code inventory')
                    digest=hashlib.sha256();count=0
                    with archive.open(info)as stream:
                        while True:
                            guard();data=stream.read(65536)
                            if not data:break
                            count+=len(data);digest.update(data)
                    require(count==info.file_size,'Complete saved script bytes')
                    scripts[info.filename]=dict(bytes=count,sha256=digest.hexdigest())
                with archive.open(data_members[0])as stream:
                    wrapped=GuardedStream(stream,guard)
                    require(wrapped.read(2)==b'\x80\x02','Pinned C++ protocol2 archive; no unparsed frame buffer')
                    wrapped.seek(0);obj=BoundRuntimeReader(wrapped).load()
                    require(wrapped.read(1)==b'','No trailing unparsed pickle content')
                result=describe(obj,scripts,guard)
            guard();require(file_identity(os.fstat(fd))==observed and file_identity(path.stat())==observed,
                            'Original archive identity unchanged after complete descriptor read')
            result.update(artifact_binding=binding,whole_archive_sha256_verified=True)
            return result
    finally:os.close(fd)


def compare_descriptors(off,on):
    """Pure strict saved-path comparison, never a device-layout proof."""
    for value in (off,on):
        require(value['schema']=='deepseek_serialized_weight_descriptor_v1'and value['all12_graphs_present']is True
            and len(value['graphs'])==12,'Complete original12 descriptor required')
        require(all(value[k]is False for k in ('loaded_values_observed','graph_weight_ABI_semantics_proven',
            'layout_semantics_proven','inference_admitted','full_model_qualified','new_speedup_claim')),
            'Descriptors cannot grant actual loaded-value/semantic/model qualification')
    fields=('tag','index','bucket','route_shapes','serialization_tag','local_ranks','world_size','weight_inputs')
    bindings_equal=[{k:g[k]for k in fields}for g in off['graphs']]==[{k:g[k]for k in fields}for g in on['graphs']]
    wlt_equal=off['wlt']==on['wlt']
    code_equal=off['all_serialized_script_records']==on['all_serialized_script_records']
    return dict(schema='deepseek_serialized_weight_descriptor_comparison_v1',
        complete_saved_WLT_equal=wlt_equal,all12_ordered_weight_interfaces_equal=bindings_equal,
        all_saved_script_bytes_equal=code_equal,strict_saved_components_equal=wlt_equal and bindings_equal and code_equal,
        loaded_values_observed=False,graph_weight_ABI_semantics_proven=False,layout_semantics_proven=False,
        inference_admitted=False,full_model_qualified=False,new_speedup_claim=False)
