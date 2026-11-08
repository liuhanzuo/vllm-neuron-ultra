"""Independent logical head mapping and saved-byte checks, no model execution.

A future serial, admitted post-exit reviewer must supply authenticated reports
and a live guard. This file is source preparation, not a launch coordinator.
"""
from array import array
import hashlib,math,os,stat,struct,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1]
if str(R)not in sys.path:sys.path.insert(0,str(R))
import minicpm5_frame1_attention_observer_v1 as source
from review_minicpm5_frame1_raw_boundaries_v1 import Metric,VALUES
require=source.require
ROLES=('native','cpu-rank0','cpu-rank1')
REPLICATED={'normalized_input','position_ids','rope_cos','rope_sin','projection_output'}
HEAD_MAJOR={'q_rope_input','k_rope_input','q_post_rope','k_post_rope','v_heads'}


def same(a,b,message):require(a==b,message)


def original_layout(meta):
    shape,stride,offset=meta['shape'],meta['stride'],meta['offset']
    require(2<=len(shape)<=4 and len(stride)==len(shape),'Complete layout axes')
    require(all(type(s)is int and s>0 for s in shape)and all(type(s)is int and s>=0 for s in stride),'Positive integer geometry')
    require(type(offset)is int and 0<=offset<8*2**20,'Finite original storage offset')
    end=1
    for step,size in sorted((t,s)for t,s in zip(stride,shape)if s>1):
        require(step>=end,'Nonoverlapping captured layout');end+=step*(size-1)
    require(offset+end<=16*2**20,'Finite complete backing extent')


def read_payload(row,directory,shape,dtype,observed,guard):
    guard();path=Path(row['path']);same(path,Path(directory)/(row['name']+'.raw'),'Exact attention capture path')
    require(path.is_absolute()and path.resolve()==path and not any(x.is_symlink()for x in (path,*path.parents)),'Canonical saved payload')
    same(row['encoding'],'packed_logical_little_endian','Original logical byte encoding')
    require(row['observed_not_recomputed']is observed,'Observed versus own-input output distinguished')
    if not observed:require(row['own_input_recomputed']is True,'Actual recomputed control payload')
    meta=row['metadata'];same(meta['shape'],shape,'Complete original tensor shape');same(meta['dtype'],dtype,'Original scalar dtype');original_layout(meta)
    width={'torch.bfloat16':2,'torch.bool':1,'torch.int64':8}[dtype];size=math.prod(shape)*width
    require(0<size==row['bytes']==row['identity']['size']<=16*2**20,'Exact bounded tensor bytes')
    same(row['sha256'],meta['sha256'],'Metadata and payload digest agree')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        info=os.fstat(fd);require(stat.S_ISREG(info.st_mode)and info.st_nlink==1 and info.st_uid==os.getuid(),'Owned original regular tensor file')
        same(source.old.identity(path),row['identity'],'Exact path identity before read')
        expected=tuple(row['identity'][k]for k in ('device','inode','size','mtime_ns','ctime_ns'))
        def stable():
            st=os.fstat(fd);same((st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns),expected,'Opened payload unchanged')
        stable();raw=bytearray()
        while len(raw)<size:
            guard();part=os.read(fd,min(2**20,size-len(raw)));require(bool(part),'No truncated payload');raw.extend(part)
        same(hashlib.sha256(raw).hexdigest(),row['sha256'],'Whole captured tensor authenticated')
        data=array({'torch.bfloat16':'H','torch.bool':'B','torch.int64':'q'}[dtype]);data.frombytes(raw)
        require(sys.byteorder=='little','Exact little-endian raw decoding')
        if dtype=='torch.bool':require(all(x in (0,1)for x in data),'Every boolean mask cell encoded correctly')
        elif dtype=='torch.bfloat16':
            if row['name']=='attention_mask':require(all(VALUES[x]is not None or x==0xff80 for x in data),'Finite mask or negative infinity only')
            else:require(all(VALUES[x]is not None for x in data),'Finite complete tensor including padding')
        final_digest=hashlib.sha256()
        for offset in range(0,size,65536):
            guard();wanted=min(65536,size-offset);part=os.pread(fd,wanted,offset)
            same(len(part),wanted,'Complete final tensor read');final_digest.update(part)
        same(final_digest.hexdigest(),row['sha256'],'Whole tensor content unchanged after decoding')
        stable();same(source.old.identity(path),row['identity'],'Payload path retained')
        return data
    finally:os.close(fd)


def token_rows(data,shape,field,native):
    """Packed logical bytes: BHSD must be gathered per token before TP2 join."""
    require(shape[0]==1 and len(data)==math.prod(shape),'Complete logical array')
    head_major=field in HEAD_MAJOR or(field=='context_heads'and not native)
    if head_major:
        require(len(shape)==4,'Head-major four axes');_,heads,length,dim=shape
        for token in range(length):
            yield tuple(value for head in range(heads)for value in data[(head*length+token)*dim:(head*length+token+1)*dim])
    else:
        length=shape[1];width=math.prod(shape[2:])
        for token in range(length):yield tuple(data[token*width:(token+1)*width])


def boundary(data,shapes,field):
    require(field!='attention_mask','Masks use semantic comparison')
    rows=[iter(token_rows(x,shape,field,i==0))for i,(x,shape)in enumerate(zip(data,shapes))]
    lengths=[shape[2]if field in HEAD_MAJOR or(field=='context_heads'and i!=0)else shape[1]for i,shape in enumerate(shapes)]
    require(lengths[0]<=lengths[1]==lengths[2],'Real prefix and complete equal padded buckets')
    real,prediction=Metric(),Metric();exact=True;count=0;different=0
    for token in range(lengths[1]):
        left,right=next(rows[1]),next(rows[2])
        if field in REPLICATED:same(left,right,'Both CPU replicas equal including padding: '+field)
        joined=left if field in REPLICATED else left+right
        if token<lengths[0]:
            original=next(rows[0]);same(len(original),len(joined),'Complete per-token head mapping')
            if field=='position_ids':
                count+=len(original);different+=sum(a!=b for a,b in zip(original,joined));exact=exact and original==joined
            else:
                real.add(original,joined)
                if token==lengths[0]-1:prediction.add(original,joined)
    if field=='position_ids':return dict(raw_exact=exact,compared_elements=count,different_elements=different)
    return dict(real_tokens=real.result(),prediction=prediction.result(),cpu_padding_read=True)


def masks(native,rank0,rank1,length,bucket):
    same(len(native),length*length,'All native mask cells');same(len(rank0),bucket*bucket,'All CPU mask cells');same(rank0,rank1,'Both complete CPU masks equal')
    require(all(x in (0,1)for x in rank0),'Original binary CPU mask')
    require(length<=bucket,'Original right padding');mismatch=0;extra=0
    for query in range(length):
        for key in range(length):
            word=native[query*length+key];require(VALUES[word]is not None or word==0xff80,'Valid original additive mask')
            allowed=VALUES[word]==0
            if not allowed:require(word==0xff80 or VALUES[word]<0,'Only nonpositive mask bias')
            same(allowed,key<=query,'Original native full causal mask')
            mismatch+=allowed!=bool(rank0[query*bucket+key])
        extra+=sum(bool(x)for x in rank0[query*bucket+length:(query+1)*bucket])
    return dict(real_prefix_semantics_equal=mismatch==0 and extra==0,real_prefix_different_cells=mismatch,
                padded_keys_allowed_for_real_queries=extra,native_cells=length*length,cpu_cells_per_rank=bucket*bucket,
                padded_query_cells_preserved_and_read=True)


def control_outcome(observed,actual,observed_meta,actual_meta):
    return dict(raw_exact=observed==actual,layout_exact=all(observed_meta[k]==actual_meta[k]for k in ('shape','stride','offset','dtype')))


def validate_reports(reports):
    same(tuple(reports),ROLES,'All three original execution roles')
    for role,record in reports.items():
        native=role=='native';rank=-1 if native else int(role[-1]);obs=record['attention']
        require(record['schema']=='minicpm5_frame1_attention_capture_v1'and record['complete']is True and record['native']is native and record['rank']==rank,'Actual original full-prefix capture role')
        same(record['frame'],source.old.FRAME,'Same original frame1');require(record['anchor']['all_six_exact']is True and record['actual_prefix_calls']==1,'Original six prediction anchors')
        require(record['upstream_anchor']['all7_full_boundaries_exact']is True and record['precision_interventions']==[],'All prior full boundaries retained')
        same(record['upstream_anchor']['reference'],source.UPSTREAM[0 if native else rank+1],'Original upstream reference')
        require(record['official_head_weights']['passed']is True,'Official original projection weights')
        require(record['history_consumed']is False and record['downstream_layers_executed']==0,'Only original bounded prefix')
        require(record['full_model_qualified']is False and record['new_speedup_claim']is False,'No whole-model conclusion')
        require(obs['schema']=='minicpm5_frame1_attention_observation_v1'and obs['complete']is True and obs['native']is native and obs['rank']==rank,'Complete original internal observations')
        same([x['name']for x in obs['captures']],list(source.FIELDS),'All16 ordered original observations')
        require(obs['original_functions_replaced']is False and obs['attention_arithmetic_recomputed']is False,'Observed original functions')
        same(obs['calls'],{k:dict(calls=1,returns=1)for k in ('outer','rope','attention')},'Exactly one original complete route')
        controls=record['own_input_controls'];require(controls['complete']is True and controls['original_operators']is True,'Original operator controls complete')
        same(tuple(controls['controls']),source.CONTROL_FIELDS,'All seven own-input controls')


def audit_attention(root,reports,guard):
    """Only after capture workers exit; supplied admission is responsible for lifecycle."""
    require(callable(guard),'Admitted serial review guard');guard();validate_reports(reports);root=Path(root)
    metrics={};controls={role:{}for role in ROLES};bytes_read=0;files=0
    for field in source.FIELDS:
        data=[];shapes=[]
        for role in ROLES:
            guard();native=role=='native';row=next(x for x in reports[role]['attention']['captures']if x['name']==field)
            expected=source.geometry(native)[field];directory=root/role/'attention'
            observed=read_payload(row,directory,expected['shape'],expected['dtype'],True,guard)
            data.append(observed);shapes.append(expected['shape']);bytes_read+=row['bytes'];files+=1
            if field in source.CONTROL_FIELDS:
                declared=reports[role]['own_input_controls']['controls'][field];saved=declared['saved']
                same(saved['name'],field,'Same actual control output')
                actual=read_payload(saved,directory/'controls',expected['shape'],expected['dtype'],False,guard)
                proof=control_outcome(observed,actual,row['metadata'],saved['metadata']);controls[role][field]=proof
                same(proof,{k:declared[k]for k in proof},'Own-input equality independently checked from full saved bytes')
                bytes_read+=saved['bytes'];files+=1
        metrics[field]=masks(*data,source.LENGTH,source.BUCKET)if field=='attention_mask'else boundary(data,shapes,field)
    passed=all(source.controls_passed(c)for c in controls.values())
    for role in ROLES:
        same(source.controls_passed(controls[role]),reports[role]['own_input_controls']['passed'],'Actual per-role control verdict')
    same(files,69,'All48 observed plus21 recomputed tensors');same(bytes_read,source.science()['raw_capture_bytes']+source.science()['raw_control_bytes'],'Exact complete internal payload budget')
    return dict(complete=True,files=files,raw_bytes=bytes_read,controls=controls,all21_own_input_controls_exact=passed,
                observed_boundary_metrics=metrics,full_original_padding_read=True,all_payloads_authenticated=True,
                model_math_reexecuted=False,independently_recomputed_trigonometry=False,
                original_seven_upstream_raw_payloads_independently_checked=False,
                process_lifecycle_verified=False,unique_internal_or_final_KL_cause_proven=False,
                full_model_qualified=False,new_speedup_claim=False)
