"""Bounded selected prediction-tap comparison, callable only by a root serial wrapper.

No launcher, native imports, or file reads at import. prepare_inputs reads small
JSON only. analyze requires a live external admission guard; raw data are read
only there. Numeric comparisons describe prediction vectors, never full shapes.
"""
import collections
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import sys
import zipfile

from audit_nemotron_cached_trajectory import MetadataOnly

ROOT=Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
Q=ROOT/'research'
PLAN=dict(path=str(Q/'minicpm5-layer0-phase-cpu-input-plan-20261107-v1.json'),sha256='ce599a2beae22df0a05c13591854cfab8ba57519df11ee93a0d25ad4392306c8')
SPEC=dict(path=str(Q/'minicpm5-layer0-phase-cpu-execution-spec-20261107-v1.json'),sha256='4d7d017541ba2088a0d69fe10447d87046b9f466570f011edaf4a0388dd6ad58')
READER=dict(path=str(ROOT/'audit_nemotron_cached_trajectory.py'),sha256='0a6553732c41dd6b236c77384a6b41fdd562190504761614613e13a0d73076c5')
ARMS=('strict_anchor_s0','strict_layer0_oproj_fp32_prefill_only','strict_layer0_oproj_fp32_both_phases')
TAPS=('layers.0.self_attn','layers.0','layers.41.self_attn','layers.41')
RAW_CAP=64*2**20
VECTOR_BYTES=4096

def require(value,message):
    if not value:raise ValueError(message)
def sha(data):return hashlib.sha256(data).hexdigest()
def metadata(ref):
    path=Path(ref['path']);require(path.suffix=='.json','JSON metadata only')
    require(path.stat().st_size<8*2**20 and not path.is_symlink(),'Bounded regular JSON metadata')
    raw=path.read_bytes();require(len(raw)<8*2**20 and sha(raw)==ref['sha256'],'Immutable JSON reference')
    return json.loads(raw)
def file_identity(path):
    p=Path(path);require(p.is_absolute() and p.resolve()==p and not any(x.is_symlink()for x in(p,*p.parents)),'Canonical non-symlink payload path')
    s=p.lstat();require(stat.S_ISREG(s.st_mode)and s.st_uid==1000,'Regular owned payload')
    return dict(realpath=str(p),device=s.st_dev,inode=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)


class BF16MetadataOnly(MetadataOnly):
    """The existing restricted reader, plus one BF16 storage marker."""
    def find_class(self,module,name):
        if(module,name)==('torch','BFloat16Storage'):return 'BFLOAT16_STORAGE'
        return super().find_class(module,name)
    def persistent_load(self,identifier):
        require(type(identifier)is tuple and len(identifier)==5 and identifier[0]=='storage','Exact storage persistent ID')
        _,dtype,key,location,elements=identifier
        require(dtype in ('BFLOAT16_STORAGE','FLOAT32_STORAGE') and location=='cpu','Only CPU BF16/FP32 descriptor storage')
        require(type(key)is str and re.fullmatch('[0-9]{1,6}',key)is not None,'Finite storage key')
        require(type(elements)is int and 0<elements<=130560,'Finite storage element count')
        return dict(key=key,elements=elements,dtype='bfloat16'if dtype=='BFLOAT16_STORAGE'else'float32',location=location)


def native_vectors(raw,frame):
    """Decode four authenticated, contiguous prediction vectors from one .pt."""
    require(type(raw)is bytes and 0<len(raw)<2*2**20,'Bounded native archive bytes')
    with zipfile.ZipFile(io.BytesIO(raw))as archive:
        infos=archive.infolist();names=[item.filename for item in infos]
        require(len(names)==len(set(names))and 1<len(names)<=256,'Unique bounded archive members')
        require(all(item.compress_type==zipfile.ZIP_STORED and item.file_size<=2*2**20 for item in infos),'Only bounded stored archive members')
        require(sum(item.file_size for item in infos)<2*2**20,'Bounded uncompressed archive')
        picks=[name for name in names if name.endswith('/data.pkl')];require(len(picks)==1,'One metadata pickle')
        prefix=picks[0].rsplit('/',1)[0]
        require(all(name.startswith(prefix+'/')and '..'not in name.split('/')for name in names),'One safe archive namespace')
        require(archive.getinfo(picks[0]).file_size<128*2**10,'Finite metadata pickle')
        require(archive.read(prefix+'/byteorder')==b'little','Explicit little endian archive')
        stream=io.BytesIO(archive.read(picks[0]));obj=BF16MetadataOnly(stream).load();require(stream.read()==b'','No trailing pickle data')
        require(type(obj)is dict and set(obj)=={'frame','logits','taps'}and obj['frame']==frame['frame'],'Original native payload frame')
        require(type(obj['taps'])in(dict,collections.OrderedDict)and list(obj['taps'])==frame['stage_names'],'All213 original ordered tap descriptors')
        result={}
        for name in TAPS:
            desc=obj['taps'][name]
            require(desc['tensor_descriptor']is True and desc['shape']==[1,1,2048]and desc['offset']==0,'Owned native prediction vector')
            require(len(desc['stride'])==3 and all(type(x)is int and x>0 for x in desc['stride'])and desc['stride'][-1]==1,'Contiguous meaningful dimension; singleton strides may retain source geometry')
            storage=desc['storage'];require(storage['dtype']=='bfloat16'and storage['location']=='cpu'and storage['elements']==2048,'Exact native vector storage')
            member=prefix+'/data/'+storage['key'];require(archive.getinfo(member).file_size==VECTOR_BYTES,'Exact BF16 storage length')
            value=archive.read(member);require(sha(value)==frame['tap_sha256'][name],'Native vector matches original tap digest')
            result[name]=value
        return result


def bundle_vectors(raw,call):
    """Authenticate exact intervals and reorder physical rows into frame keys."""
    bundle=call['selected_tap_bundle'];mapping=sorted(call['physical_rows'],key=lambda x:x['physical_row']);rows=len(mapping)
    require(rows in(1,4)and [x['physical_row']for x in mapping]==list(range(rows)),'Complete physical row permutation')
    frames=[x['frame']['index']for x in mapping];require(len(set(frames))==rows,'Unique physical frames')
    require(type(raw)is bytes and len(raw)==bundle['raw_bytes']==rows*4*VECTOR_BYTES and sha(raw)==bundle['sha256'],'Complete bundle digest and length')
    require(bundle['encoding']=='raw_bf16_le'and len(raw)<=65536,'Finite little endian bundle')
    require([x['name']for x in bundle['entries']]==list(TAPS),'Exact four selected entries')
    result={}
    for index,entry in enumerate(bundle['entries']):
        name=entry['name'];size=rows*VECTOR_BYTES
        require(type(entry['offset'])is int and entry['offset']==index*size and type(entry['raw_bytes'])is int and entry['raw_bytes']==size,'Non-overlapping complete intervals')
        require(entry['shape']==[rows,1,2048]and entry['dtype']=='torch.bfloat16'and entry['frame_indices']==frames,'Physical-order prediction metadata')
        require(entry['source_metadata']==call['stage_metadata'][name],'Original source geometry retained')
        value=raw[entry['offset']:entry['offset']+size]
        require(sha(value)==entry['sha256']==call['stage_sha256'][name],'Entry matches gathered source tap digest')
        for physical,frame in enumerate(frames):result[frame,name]=value[physical*VECTOR_BYTES:(physical+1)*VECTOR_BYTES]
    require(len(result)==rows*4,'Every selected frame/tap covered')
    return result


def decode_bf16(raw):
    require(type(raw)is bytes and len(raw)==VECTOR_BYTES,'Exactly2048 BF16 elements')
    words=struct.unpack('<2048H',raw)
    values=tuple(struct.unpack('<f',struct.pack('<I',word<<16))[0]for word in words)
    require(all(math.isfinite(value)for value in values),'Finite selected values')
    return words,values


def metrics(reference,observed):
    aw,a=decode_bf16(reference);bw,b=decode_bf16(observed)
    error=[abs(x-y)for x,y in zip(a,b)];dot=math.fsum(x*y for x,y in zip(a,b))
    an=math.sqrt(math.fsum(x*x for x in a));bn=math.sqrt(math.fsum(x*x for x in b))
    cosine=max(-1.0,min(1.0,dot/(an*bn)))if an and bn else None
    return dict(elements=2048,raw_exact=reference==observed,bitwise_different_elements=sum(x!=y for x,y in zip(aw,bw)),
        different_elements=sum(x!=y for x,y in zip(a,b)),max_abs=max(error),mean_abs=math.fsum(error)/2048,
        cosine=cosine,cosine_defined=cosine is not None,zero_reference_norm=an==0,zero_observed_norm=bn==0)


def native_manifest(plan,spec):
    """Small JSON only; payload existence or bytes are not touched here."""
    final=metadata(spec['finalization'])
    require(final['passed']is True and final['runtime_successful']is True and final['completion']['all88_original_endpoints']is True,'Original88 completed endpoint binding')
    contexts={Path(x['directory']).name:x for x in final['completion']['contexts']if Path(x['directory']).name.startswith('native-frame-')}
    require(len(contexts)==32,'All32 native capture contexts')
    result=[]
    for frame,reference in zip(plan['frames'],plan['hf_frame_records']):
        old=metadata(reference);require(old['frame']==frame and old['original_logits_raw_exact']is True,'Original HF32 frame')
        ctx=contexts['native-frame-%02d'%frame['index']]
        receipt_ref=dict(path=str(Path(ctx['directory'])/'capture.json'),sha256=ctx['receipt_sha256']);receipt=metadata(receipt_ref)
        endpoint=receipt['endpoint']
        require(receipt['complete']is True and endpoint['passed']is True and endpoint['raw_exact']is True and endpoint['original_scope_preserved']is True,'Authentic original88 native endpoint')
        require(endpoint['frame']==frame and endpoint['original_v2_frame_payload_sha256']==old['payload_sha256'],'Original88 binds this exact native tap archive')
        require(endpoint['original_reference_payload_sha256']==plan['hf_reference']['sha256'],'Same original HF reference')
        require(old['hf_forward_order']==dict(fixture=frame['fixture'],step=frame['step'])and old['history']=='unchanged native independent fixture cached trajectory','Unchanged native cached trajectory')
        require(list(old['metadata'])==list(old['tap_raw_sha256'])and len(old['metadata'])==213,'Complete213 native tap metadata')
        require(old['payload_identity']['realpath']==old['payload']and 0<old['payload_identity']['size']<2*2**20,'Bounded native archive identity')
        positions={call['logical_positions'][i]for call in plan['calls']for i,f in enumerate(call['frames'])if f==frame};require(len(positions)==1,'Unique logical prediction position')
        position=positions.pop()
        for name in TAPS:
            m=old['metadata'][name]
            require(m['dtype']=='torch.bfloat16'and m['source_shape']==[1,position+1 if frame['step']==0 else 1,2048],'Native source prediction shape')
            require(m['logical_prediction_position']==position and m['local_prediction_position']==(position if frame['step']==0 else 0),'Native last-query selection')
        result.append(dict(frame=frame,metadata_ref=reference,original88_receipt=receipt_ref,
            payload=dict(path=old['payload'],sha256=old['payload_sha256'],identity=old['payload_identity']),
            stage_names=list(old['metadata']),tap_sha256={name:old['tap_raw_sha256'][name]for name in TAPS},
            source_metadata={name:old['metadata'][name]for name in TAPS},logical_position=position,history=old['history']))
    require(len(result)==32 and [x['frame']for x in result]==plan['frames'],'Complete native32 order')
    return result


def prepare_inputs(actual_exit_ref):
    """No raw reads: bind finished experiment reports to a complete analysis list."""
    plan=metadata(PLAN);spec=metadata(SPEC);require(spec['plan']==PLAN,'Fixed completed experiment plan')
    ended=metadata(actual_exit_ref);live=metadata(ended['root_live'])
    require(ended['reviewer']==live['reviewer']=='/root'and ended['actual_exit_observed']is True and type(ended['actual_exit_code'])is int and ended['actual_exit_code']==0,'Root actual exit0 required')
    require(ended['all_four_processes_gone']is ended['cgroup_empty']is ended['canonical_unchanged']is True,'Root drained predecessor required')
    require(live['actual_identity_verified']is True and live['spec']==SPEC and live['output_root']==plan['output_root'],'Live run exact spec/output binding')
    for key in ('owner','coordinator','ranks','actual_execution_handle'):require(ended[key]==live[key],'Live/exit identity '+key)
    root=Path(plan['output_root']);outer=metadata(ended['outputs']['outer_result']);coord=metadata(ended['outputs']['coordinator_result'])
    require(ended['outputs']['outer_result']['path']==str(root/'outer-result.json')and outer['complete']is True and outer['coordinator_result']==ended['outputs']['coordinator_result'],'Completed actual output tree')
    require(coord['complete']is True and coord['rank_exits']==[0,0]and coord['rank_owners']==ended['ranks'],'Both actual ranks complete')
    calls={};sources=[]
    for rank in(0,1):
        rd=metadata(ended['outputs']['rank%d_result'%rank]);require(rd==coord['results'][rank],'Coordinator embeds exact separately authenticated rank result')
        require(rd['process']==ended['ranks'][rank],'Rank result actual process identity')
        require(rd['complete']is True and rd['rank']==rank and rd['arms']==list(ARMS)and rd['anchor_all14_exact']is True,'Completed ordered three-arm rank')
        for index,arm in enumerate(ARMS):
            a=rd['arm_reports'][index];require(a['arm']==arm and a['report']==str(root/('cpu-rank%d'%rank)/arm/'result.json'),'Bound arm path')
            aref=dict(path=a['report'],sha256=a['sha256']);ar=metadata(aref);sources.append(aref)
            require(ar['complete']is True and ar['seed_import_count']==1 and ar['continuous_actual_cache_chain']is True,'Completed own-history arm')
            seen=[];rows=[]
            for ci,item in enumerate(ar['reports']):
                require(ci<14 and item['call']==ci and item['report']==str(root/('cpu-rank%d'%rank)/arm/('call-%02d'%ci)/'result.json'),'Bound ordered call')
                cref=dict(path=item['report'],sha256=item['sha256']);call=metadata(cref);sources.append(cref)
                require(call['complete']is True and call['arm']==arm and call['rank']==rank and call['call']==plan['calls'][ci],'Original call identity')
                anchor_ref=next(x for x in plan['layer0_phase_trajectory']['strict_anchor_reports']if x['rank']==rank and x['call']==ci)
                anchor=metadata(anchor_ref)
                require(call['physical_rows']==anchor['physical_rows']and call['stage_metadata']==anchor['stage_metadata'],'Authenticated original physical row selection and stage geometry')
                require(call['bucket']==anchor['bucket']and call['route_records']==anchor['route_records'],'Original graph geometry and mask routes')
                require(call['selected_tap_bundle']['path']==str(Path(item['report']).parent/'selected-taps.bf16'),'Dedicated bundle path')
                require(call['selected_tap_bundle']['identity']['size']==call['selected_tap_bundle']['raw_bytes']<=65536,'Bounded bundle identity')
                require(call['full_model_qualified']is call['new_speedup_claim']is False,'Diagnostic only')
                calls[rank,arm,ci]=call;seen+=call['call']['frames'];rows+=call['HF_metrics']
            require(len(ar['reports'])==14 and seen==plan['frames']and rows==ar['rows'],'Every14call32frame arm')
    natives=native_manifest(plan,spec)
    return dict(plan=plan,spec=spec,exit_ref=actual_exit_ref,exit=ended,calls=calls,natives=natives,metadata_sources=sources)


def read_payload(reference,cap,guard):
    guard();path=Path(reference['path']);before=file_identity(path)
    require(before==reference['identity']and 0<before['size']<=cap,'Bounded authenticated raw input')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        info=os.fstat(fd);require((info.st_dev,info.st_ino,info.st_size)==(before['device'],before['inode'],before['size']),'Same opened inode')
        with os.fdopen(fd,'rb',closefd=False)as stream:raw=stream.read(cap+1)
    finally:os.close(fd)
    require(len(raw)==before['size']and sha(raw)==reference['sha256']and file_identity(path)==before,'Raw bytes digest and identity')
    guard();return raw


def analyze(prepared,admission,external_guard):
    """Root wrapper owns lock, resource limits, prior drains, output and actual exit."""
    require(callable(external_guard),'External serial admission required')
    require(admission['serial_exclusive']is admission['actual_exit_verified']is True and admission['pid']==os.getpid(),'Root admitted current serial owner')
    require(admission['plan']==PLAN and admission['actual_exit']==prepared['exit_ref'],'Exact completed analysis inputs')
    source=Path(__file__).resolve();require(sha(source.read_bytes())==admission['source_sha256'],'Exact reviewed analyzer')
    require(sha(Path(READER['path']).read_bytes())==READER['sha256'],'Pinned reused restricted reader')
    def guard():
        require(external_guard()==admission,'Live external admission changed')
        require(not{'torch','numpy','torch_neuronx'}&set(sys.modules),'Stdlib-only saved-value analysis')
    guard();vectors={};payload_refs=[];read_bytes=0
    for arm in ARMS:
        for ci in range(14):
            paired=[]
            for rank in(0,1):
                call=prepared['calls'][rank,arm,ci];bundle=call['selected_tap_bundle']
                ref={k:bundle[k]for k in('path','sha256','identity')};raw=read_payload(ref,65536,guard);read_bytes+=len(raw);payload_refs.append(ref)
                paired.append(bundle_vectors(raw,call))
            require(paired[0]==paired[1],'Both ranks actual selected bytes agree before deduplication')
            for key,value in paired[0].items():require((arm,*key)not in vectors,'No duplicate prediction key');vectors[(arm,*key)]=value
    require(len(vectors)==3*32*4,'All3arms32frames4taps')
    native={}
    for frame in prepared['natives']:
        raw=read_payload(frame['payload'],2*2**20,guard);read_bytes+=len(raw);payload_refs.append(frame['payload'])
        require(read_bytes<=RAW_CAP,'Finite64MiB raw read budget')
        for name,value in native_vectors(raw,frame).items():native[frame['frame']['index'],name]=value
    require(len(native)==128 and read_bytes==3145728+sum(x['payload']['identity']['size']for x in prepared['natives']),'Exact full native and bundle input coverage')
    rows=[]
    comparisons=[('HF_native',arm)for arm in ARMS]+[(ARMS[0],ARMS[1]),(ARMS[0],ARMS[2]),(ARMS[1],ARMS[2])]
    for left,right in comparisons:
        for frame in prepared['plan']['frames']:
            guard();i=frame['index']
            for name in TAPS:
                reference=native[i,name]if left=='HF_native'else vectors[left,i,name]
                rows.append(dict(reference=left,observed=right,frame=frame,tap=name,domain='prediction_query_2048',metrics=metrics(reference,vectors[right,i,name])))
    require(len(rows)==768,'All six comparisons over32frames4taps')
    totals=[]
    for left,right in comparisons:
        for name in TAPS:
            selected=[x['metrics']for x in rows if x['reference']==left and x['observed']==right and x['tap']==name]
            totals.append(dict(reference=left,observed=right,tap=name,frames=32,elements=65536,
                exact_frames=sum(x['raw_exact']for x in selected),different_elements=sum(x['different_elements']for x in selected),
                bitwise_different_elements=sum(x['bitwise_different_elements']for x in selected),max_abs=max(x['max_abs']for x in selected),
                mean_abs=math.fsum(x['mean_abs']for x in selected)/32,min_frame_cosine=min((x['cosine']for x in selected if x['cosine_defined']),default=None)))
    geometry=[]
    for arm in ARMS:
        for ci in range(14):
            call=prepared['calls'][0,arm,ci]
            geometry.append(dict(arm=arm,call=call['call'],bucket=call['bucket'],physical_rows=call['physical_rows'],
                source_metadata={name:call['stage_metadata'][name]for name in TAPS}))
    guard()
    return dict(schema='minicpm5_layer0_phase_selected_tap_numeric_result_v1',complete=True,
        binding=dict(plan=PLAN,spec=SPEC,actual_exit=prepared['exit_ref'],analysis=dict(path=str(source),sha256=admission['source_sha256']),reader=READER),
        input_payloads=payload_refs,actual_raw_bytes_read=read_bytes,rows=rows,totals=totals,
        cpu_source_geometry=geometry,native_source_geometry=[dict(frame=x['frame'],source_metadata=x['source_metadata'],history=x['history'],original88_receipt=x['original88_receipt'])for x in prepared['natives']],
        rank_duplicates_counted_once=True,complete_frames=32,selected_taps=list(TAPS),comparisons=6,
        limits=['Gathered prediction queries only, not full sequence intermediates.','Native independent unsharded BF16 trajectory and padded CPU TP2 own-cache trajectories differ in geometry and histories; same semantic boundary/vector is comparable, not a same-input operator contrast.','Between candidate arms, calls0-3 and call4 incoming caches are controlled by the model trial; later effects include propagated history.','Native distance changes do not identify a unique downstream cause or prove additive layer errors.','CPU diagnostic only; all original device,cold,long,slot,quality96,paired-serving gates remain.'],
        self_exit_claimed=False,full_model_qualified=False,new_speedup_claim=False,causal_operator_proven=False)
