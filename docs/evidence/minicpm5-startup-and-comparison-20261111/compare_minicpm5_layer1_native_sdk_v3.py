"""Pure saved-byte comparison after independently completed native/SDK audits.

No default reader, filesystem, native imports, CLI, launch or admission. Caller
authenticates all input metadata and actual audit terminals, then supplies the
original guarded stable no-follow full reader. This is not another raw auditor.
"""
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import struct
from research import minicpm5_layer1_operator_native_manifest_v1 as native_manifest
from research import minicpm5_layer1_operator_sdk_manifest_v1 as sdk_manifest
from research import minicpm5_layer1_operator_native_alignment_v1 as alignment
require=alignment.require
WIDTH={'torch.bfloat16':2,'torch.float32':4,'torch.int64':8,'torch.bool':1}


def sha(raw):return hashlib.sha256(raw).hexdigest()


def numbers(raw,dtype,guard):
    require(dtype in WIDTH and len(raw)%WIDTH[dtype]==0,'Exact complete scalar extent')
    fmt={'torch.bfloat16':'<H','torch.float32':'<f','torch.int64':'<q','torch.bool':'<B'}[dtype]
    for i,(x,)in enumerate(struct.iter_unpack(fmt,raw)):
        if i%4096==0:guard()
        if dtype=='torch.bfloat16':x=struct.unpack('<f',struct.pack('<I',x<<16))[0]
        require(math.isfinite(x)and (dtype!='torch.bool'or x in (0,1)),'Complete finite/canonical saved scalars')
        yield x


class Reader:
    def __init__(self,read_bytes,guard):
        require(callable(read_bytes)and callable(guard),'Explicit independently admitted reader and guard')
        self.read,self.guard=read_bytes,guard;self.files={};self.raw_counts={'native':0,'sdk':0}
    def get(self,ref,path,size,kind=None):
        self.guard();path=str(path)
        require(Path(path).is_absolute()and str(Path(path))==path and '..'not in Path(path).parts
            and ref['path']==path and path not in self.files and type(size)is int and 0<size<=16*2**20
            and ref.get('raw_bytes',ref.get('bytes'))==size,'Exact distinct full source reference/path/size')
        raw=self.read(ref,path,size);self.guard()
        require(type(raw)is bytes and len(raw)==size and sha(raw)==ref['sha256'],'Complete stable saved bytes/EOF/SHA required')
        self.files[path]=dict(reference=copy.deepcopy(ref),bytes=size,kind=kind or'metadata')
        if kind:self.raw_counts[kind]+=1
        return raw


def tensor(reader,leaf,wanted,path,context,kind):
    require(type(leaf)is dict and set(leaf)=={'metadata','reference'},'Exact saved tensor leaf')
    meta=leaf['metadata'];shape=wanted['shape'];dtype=wanted['dtype'];size=math.prod(shape)*WIDTH[dtype]
    require(meta['context']==context and meta['source_shape']==meta['selected_shape']==shape
        and meta['source_dtype']==dtype and meta['source_numel']==math.prod(shape)
        and meta['selected_bytes']==wanted['bytes']==size and meta['selection']==wanted['selection']
        and meta['cpu']is meta['owned_clone']is True and len(meta['source_stride'])==len(shape)
        and all(type(x)is int and x>=0 for x in meta['source_stride'])
        and type(meta['source_offset'])is int and meta['source_offset']>=0,'Full saved layout/selection/clone provenance')
    raw=reader.get(leaf['reference'],path,size,kind)
    for _ in numbers(raw,dtype,reader.guard):pass
    return dict(raw=raw,shape=list(shape),dtype=dtype,evidence=[dict(reference=copy.deepcopy(leaf['reference']),metadata=copy.deepcopy(meta))])


def leaves(value,path=()):
    if type(value)is dict and set(value)=={'metadata','reference'}:return [(path,value)]
    if type(value)is dict:return [x for key,child in value.items()for x in leaves(child,path+(key,))]
    if type(value)is list:return [x for i,child in enumerate(value)for x in leaves(child,path+(i,))]
    return []


def select(value,ranges):
    """Slice logical packed snapshot bytes, never reinterpret them with source strides."""
    shape=value['shape'];require(len(shape)==len(ranges),'One exact range per snapshot axis')
    for n,(lo,hi)in zip(shape,ranges):require(type(lo)is type(hi)is int and 0<=lo<=hi<=n,'Bounded complete logical selection')
    width=WIDTH[value['dtype']];chunks=[];strides=[math.prod(shape[i+1:])for i in range(len(shape))]
    for index in itertools.product(*(range(lo,hi)for lo,hi in ranges[:-1])):
        start=sum(i*s for i,s in zip(index,strides))+ranges[-1][0];size=ranges[-1][1]-ranges[-1][0]
        chunks.append(value['raw'][start*width:(start+size)*width])
    return dict(raw=b''.join(chunks),shape=[hi-lo for lo,hi in ranges],dtype=value['dtype'],
        evidence=[dict(row,logical_selection_chain=[*row.get('logical_selection_chain',[]),
            dict(input_shape=list(shape),ranges=copy.deepcopy(ranges))])for row in value['evidence']])


def row(value,index):return select(value,[[index,index+1],*[[0,n]for n in value['shape'][1:]]])


def reshape(value,shape):
    require(math.prod(shape)==math.prod(value['shape']),'Byte-only reshape preserves every selected element')
    return dict(value,shape=list(shape))


def join(left,right,axis):
    require(left['dtype']==right['dtype']and len(left['shape'])==len(right['shape'])
        and all(a==b for i,(a,b)in enumerate(zip(left['shape'],right['shape'])) if i!=axis),
        'Prior and observed active are exact compatible contiguous logical tensors')
    shape=list(left['shape']);shape[axis]+=right['shape'][axis];width=WIDTH[left['dtype']]
    a=math.prod(left['shape'][axis:])*width;b=math.prod(right['shape'][axis:])*width
    raw=b''.join(left['raw'][i*a:(i+1)*a]+right['raw'][i*b:(i+1)*b]for i in range(math.prod(shape[:axis])))
    return dict(raw=raw,shape=shape,dtype=left['dtype'],evidence=left['evidence']+right['evidence'])


def compare_values(left,right,guard):
    require(left['shape']==right['shape'],'Explicit aligned shapes before numerical comparison')
    a=list(numbers(left['raw'],left['dtype'],guard));b=list(numbers(right['raw'],right['dtype'],guard))
    require(len(a)==len(b)>0,'Complete nonempty aligned values')
    first=next((i for i,(x,y)in enumerate(zip(a,b))if x!=y),None)
    delta=[float(y)-float(x)for x,y in zip(a,b)];sq=math.fsum(x*x for x in delta);ref=math.fsum(float(x)*float(x)for x in a)
    byte_equal=left['raw']==right['raw']if left['dtype']==right['dtype']else None
    return dict(status='different'if first is not None or byte_equal is False else'equal',
        bytes_equal=byte_equal,values_equal=first is None,aligned_shape=left['shape'],
        native_dtype=left['dtype'],sdk_dtype=right['dtype'],elements=len(a),
        numeric_mismatch_elements=sum(x!=y for x,y in zip(a,b)),first_numeric_difference_flat_index=first,
        maximum_absolute_error=max(map(abs,delta)),relative_l2=math.sqrt(sq/ref)if ref else(0.0 if sq==0 else None),
        relative_l2_reference='native',relative_l2_undefined_zero_reference=ref==0 and sq!=0,
        native_selected_sha256=sha(left['raw']),sdk_selected_sha256=sha(right['raw']),
        native_evidence=left['evidence'],sdk_evidence=right['evidence'],all_selected_elements_compared=True)


def summary(value,guard):
    values=list(numbers(value['raw'],value['dtype'],guard))
    return dict(shape=value['shape'],dtype=value['dtype'],bytes=len(value['raw']),sha256=sha(value['raw']),
        elements=len(values),nonzero_elements=sum(x!=0 for x in values),minimum=min(values)if values else None,
        maximum=max(values)if values else None,evidence=value['evidence'],full_selection_read=True)


def admitted(native_result,native_plan,native_audit,sdk_results,sdk_plan,sdk_recovery):
    require(native_audit['schema']=='minicpm5_native_post_exit_independent_audit_v1'and native_audit['complete']is True
        and native_audit['actual_native_model_exit_verified']is native_audit['actual_native_recorder_exit_verified']is True
        and native_audit['every_raw_byte_reread_after_all_model_processes_exited']is True,
        'Complete independently audited native evidence required; no partial/live capture')
    proof=native_audit['result'];require(proof['schema']=='minicpm5_layer1_operator_native_independent_verification_v2'
        and proof['complete']is True and [proof[k]for k in ('frames','direct_attention_calls','raw_files','journals','arm_result_files')]==[24,72,936,72,72],
        'Original complete native24/72/936 independent audit')
    require(sdk_recovery['schema']=='minicpm5_operator_sdk_failed_run_recovery_audit_v1'and sdk_recovery['audit_complete']is True
        and type(sdk_recovery['original_exit_code'])is int and sdk_recovery['original_exit_code']==1
        and sdk_recovery['original_runtime_successful']is False and sdk_recovery['every_raw_byte_reread_after_all_model_processes_exited']is True,
        'Actually completed saved SDK recovery; original failed model exit1 remains1')
    sp=sdk_recovery['result'];require(sp['schema']=='minicpm5_layer1_operator_sdk_pair_verification_v1'and sp['complete']is True
        and [sp[k]for k in ('ranks','unique_checkpoints','physical_attention_invocations','raw_capture_files','raw_control_files')]==[2,24,60,564,60]
        and sp['every_raw_payload_reread']is sp['all_endpoint_controls_full_bytes_exact']is sp['both_rank_reduced_outputs_raw_exact']is True,
        'Original complete SDK2x6 calls/624raw saved-data audit')
    for value in (native_audit,sdk_recovery,proof,sp):
        require(value['full_model_qualified']is value['new_speedup_claim']is False,'Diagnostic scope and no speedup promotion')
    contract=native_plan['native_operator_manifest']
    require(native_result['schema']=='minicpm5_layer1_operator_native_driver_v4'and native_result['complete']is True
        and native_result['manifest_sha256']==native_manifest.digest(contract)and len(native_result['records'])==24
        and native_result['frames']==contract['checkpoints']==24 and native_result['raw_files']==contract['raw_files']==936
        and len(contract['frames'])==24 and len({c['frame']['index']for c in contract['frames']})==24,
        'Complete exact native driver and immutable manifest')
    require(sdk_plan['sdk_operator_manifest']==sdk_manifest.manifest()and len(sdk_results)==2
        and sp['raw_bytes']==sdk_plan['sdk_operator_manifest']['total_raw_bytes'],'Original complete SDK manifest')
    for result in sdk_results:
        shards=result['official_layer1_shards'];weights={row['official_key'].split('.')[-2]:row['official_matrix_sha256']for row in shards['weights']}
        require(len(shards['weights'])==4 and weights==native_result['projection_weight_sha256'],
            'Original complete BF16 projection matrix hashes agree across native and both SDK ranks')


def load_native(result,plan,proof,reader):
    frames={};audit_frames={x['frame']['index']:x for x in proof['frames_verified']}
    require(len(audit_frames)==24,'Every independently audited native frame')
    for report,case in zip(result['records'],plan['native_operator_manifest']['frames']):
        require(report['frame']==case['frame']and [a['arm']for a in report['arms']]==list(native_manifest.ARMS),'Original native frame/arm order')
        controls=[];events=None;operations={}
        for arm in report['arms']:
            folder=Path(plan['output_root'])/'native-operator'/('frame-%02d'%case['frame']['index'])/arm['arm']
            ref=arm['result'];saved=json.loads(reader.get(ref,folder/'result.json',ref['bytes']))
            require(saved['source_inputs']==case and saved['frame']==case['frame']and saved['call']==case['call']
                and saved['arm']==arm['arm']and saved['outputs']==arm['outputs']
                and saved['official_full_bytes_exact']==dict(attention=True,key=True,value=True),'Exact previously audited native arm result')
            outputs={}
            for name,wanted in case['control_outputs'].items():
                outputs[name]=reader.get(saved['outputs'][name],folder/(name+'.bf16'),wanted['bytes'],'native')
                for _ in numbers(outputs[name],wanted['dtype'],reader.guard):pass
            controls.append(outputs)
            if arm['arm']!='full_trace':continue
            cap=saved['capture'];require(cap['complete']is True and cap['frames']==[case['frame']]and cap['call']==case['call']
                and cap['events']==[x['label']for x in case['events']]and len(cap['snapshots'])==30,'Full native event/frame contract')
            events={};ordinal=0
            for index,(wanted,actual)in enumerate(zip(case['events'],cap['snapshots'])):
                require(actual['label']==wanted['label']and actual['context']['event_index']==index
                    and actual['context']['frames']==[case['frame']]and actual['context']['call']==case['call']
                    and actual['context']['arm']=='HF_native'and actual['context']['rank']==-1,'Exact native event identity')
                paths=[((),wanted['value'])]if'shape'in wanted['value']else(
                    [((i,),v)for i,v in enumerate(wanted['value'])]if type(wanted['value'])is list else[])
                require({p for p,_ in leaves(actual['value'])}=={p for p,_ in paths},'Exact native nested tensor locations')
                for path,shape in paths:
                    value=actual['value']
                    for part in path:value=value[part]
                    events[wanted['label'],path]=tensor(reader,value,shape,folder/('observed-%02d.bin'%ordinal),actual['context'],'native');ordinal+=1
            require(ordinal==30,'Every complete native capture tensor')
            operations={x['label']:copy.deepcopy(x['value'])for x in cap['snapshots']if x['label'].endswith('_scale_operation')}
        require(controls[0]==controls[1]==controls[2],'Original native three-arm endpoint/cache equality')
        require(events['attention_output',()]['raw']==controls[0]['attention']
            and events['cache_after',(0,)]['raw']==controls[0]['key']
            and events['cache_after',(1,)]['raw']==controls[0]['value'],
            'Observed native endpoint/full cache binds saved three-arm controls')
        ap=audit_frames[case['frame']['index']];require(ap['frame']==case['frame']and ap['all_three_arms_exact']is True
            and ap['official_attention_sha256']==sha(controls[0]['attention']),'Exact previously audited official native endpoint')
        frames[case['frame']['index']]=dict(case=case,events=events,scale_operations=operations)
    return frames


def load_sdk(result,cases,plan,reader):
    rank=result['rank'];require(type(rank)is int and rank in (0,1)and result['schema']=='minicpm5_layer1_operator_sdk_driver_v2'
        and result['complete']is True and result['manifest']==sdk_manifest.manifest()and len(result['records'])==len(cases)==6,
        'Exact complete SDK rank/call metadata')
    loaded={}
    for call,bucket,case,record in zip(sdk_manifest.CALLS,sdk_manifest.BUCKETS,cases,result['records']):
        value=record['result'];require(case['rank']==rank and case['call']['index']==call and case['bucket']==bucket
            and case['arm']==value['arm']=='strict_anchor_s0'and value['call']==case['call']
            and value['source_case_sha256']==native_manifest.digest(case),'Original authenticated complete AnchorStream case')
        folder=Path(plan['output_root'])/('cpu-rank%d'%rank)/'strict_anchor_s0'/('call-%02d'%call)
        require([x['variant']for x in record['outputs']]==list(sdk_manifest.MODES),'Five exact original SDK controls')
        controls={x['variant']:reader.get(x['output'],folder/x['variant']/'attention.bf16',16384,'sdk')for x in record['outputs']}
        require(controls['own']==controls['restore']and controls['official_unobserved']==controls['official_noop']==controls['official_full'],
            'Original audited SDK control identities remain exact')
        for raw in controls.values():
            for _ in numbers(raw,'torch.bfloat16',reader.guard):pass
        observations=value['observations'];require([x['mode']for x in observations]==list(sdk_manifest.MODES[1:-1]),'All original SDK observed controls')
        cap=observations[-1]['capture'];snap=cap['snapshots'];wanted=sdk_manifest.call_manifest(rank,call,bucket)
        require(snap['complete']is True and snap['manifest']==wanted and len(snap['records'])==wanted['event_count']
            and cap['summary']['frames']==case['physical_frame_order']and cap['summary']['call']==case['call'],
            'Exact complete SDK full capture and original physical slot mapping')
        events={}
        for index,(event,actual)in enumerate(zip(wanted['tensor_events'],snap['records'])):
            context=actual['context'];require(actual['label']==event['label']and context['event_index']==index
                and context['rank']==rank and context['arm']=='C_Hd_control'and context['call']==case['call']
                and context['frames']==case['physical_frame_order'],'Exact SDK event physical frame/rank/call')
            require({p for p,_ in leaves(actual['value'])}=={tuple(t['path'])for t in event['tensors']},'Exact SDK nested tensor locations')
            for j,t in enumerate(event['tensors']):
                value=actual['value']
                for part in t['path']:value=value[part]
                ctx=dict(context,event_label=event['label'],tensor_index=j,tensor_path=t['path'])
                events[event['label'],tuple(t['path'])]=tensor(reader,value,t,
                    folder/'official_full'/'capture'/('%02d-%s-%02d.bin'%(index,event['label'],j)),ctx,'sdk')
        require(events['attention_output',()]['raw']==events['o_proj_after_reduce',()]['raw']==controls['official_full'],
            'Actual observed SDK reduced endpoint binds saved controls')
        loaded[call]=dict(case=case,events=events,controls=controls,scale_operations={
            x['label']:copy.deepcopy(x['value'])for x in snap['records']if x['label'].endswith('_scale_operation')})
    return loaded


SDK_PRIOR_SCORE_OR_MASK_LABELS=frozenset(('attention_mask','prior_actual_mask','qk0_unscaled','qk0_scaled',
    'prior_mask_input','prior_mask_output','softmax_input','probabilities_fp32','probabilities_bf16'))


def padding_axis(label,path):
    """Semantic source axes only: a 128-wide head is never a padded key axis."""
    if label=='effective_prior_kv'and path in ((0,),(1,)):return 2
    if label=='route_inputs_sdk_decode':
        if path in (('past_key_value',0),('past_key_value',1)):return 2
        if path==('attention_mask',):return 3
    if path==()and label in SDK_PRIOR_SCORE_OR_MASK_LABELS:return 3
    return None


def compare_frame(native,sdk,shards,guard):
    case=native['case'];mapping=alignment.frame_alignment(sdk['case'],case,shards);rank=mapping['rank'];slot=mapping['sdk_physical_row']
    p=case['logical_position'];length=p+1;bucket=mapping['sdk_full_prior_bucket'];ne=native['events'];se=sdk['events'];out=[]
    def n(label,path=()):return ne[label,path]
    def s(label,path=()):return row(se[label,path],slot)
    def heads(value,kv=False):
        ranges=[[0,x]for x in value['shape']];ranges[1]=[rank,rank+1]if kv else[rank*8,(rank+1)*8];return select(value,ranges)
    def columns(value,kv=False):
        width=128 if kv else 1024;ranges=[[0,x]for x in value['shape']];ranges[-1]=[rank*width,(rank+1)*width];return select(value,ranges)
    def prefix(value,axis,stop):
        ranges=[[0,x]for x in value['shape']];ranges[axis]=[0,stop];return select(value,ranges)
    def add(name,left,right,note=None):
        guard();out.append(dict(boundary=name,**compare_values(left,right,guard),note=note))
    for label in ('normalized_input','position_ids','rope_position_ids'):add(label,n(label),s(label))
    for i,name in enumerate(('key','value')):add('cache_before_'+name,heads(n('cache_before',(i,)),True),prefix(s('effective_prior_kv',(i,)),2,p))
    for label in ('q_projection','k_projection','v_projection'):add(label,columns(n(label),label!='q_projection'),s(label))
    for label in ('q_rope_input','k_rope_input','q_post_rope','k_post_rope','v_heads'):
        if label in ('q_post_rope','k_post_rope'):continue
        add(label,heads(n(label),label!='q_rope_input'),s(label))
    for label in ('rope_cos','rope_sin'):add(label,n(label),s(label))
    for label in ('q_post_rope','k_post_rope'):add(label,heads(n(label),label=='k_post_rope'),s(label))
    for target,source in [('q_after_prep','q_post_rope'),('k_after_prep','k_post_rope'),('v_after_prep','v_heads')]:
        add(target,heads(n(source),source!='q_post_rope'),s(target),'Compare existing SDK prep output with existing native post-RoPE/value-head output.')
    for name,i,active in [('key',0,'k_after_prep'),('value',1,'v_after_prep')]:
        full=join(prefix(s('effective_prior_kv',(i,)),2,p),s(active),2)
        add('native_effective_'+name,heads(n('native_effective_'+name),True),full)
        add('cache_after_'+name,heads(n('cache_after',(i,)),True),full)
    add('cache_update_active_key',heads(n('cache_update_active_key'),True),s('k_after_prep'))
    for label,active in [('qk0_unscaled','qk1_unscaled'),('qk0_scaled','qk1_scaled')]:
        add(label,heads(n(label)),join(prefix(s(label),3,p),s(active),3),
            'Native multiply versus SDK divide are observed source operations; this comparison performs no scaling.')
    prior_mask=s('attention_mask');actual_mask=s('prior_actual_mask')
    require(prior_mask['raw']==actual_mask['raw']==bytes([1]*p+[0]*(bucket-p)),'Original full valid-prior and masked-padding mask')
    native_mask=n('attention_mask');require(native_mask['raw']==n('actual_eager_mask')['raw'],'Original native mask equals eager input')
    encoded=list(numbers(native_mask['raw'],native_mask['dtype'],guard))
    known=all(x in (0.0,-3.3895313892515355e38)for x in encoded)
    out.append(dict(boundary='mask',status=('equal'if all(x==0 for x in encoded)else'different')if known else'unknown',
        bytes_equal=None,values_equal=all(x==0 for x in encoded)if known else None,
        reason='Native additive BF16 mask and SDK bool prior mask have different encodings; active token is separate and unmasked.',
        native=summary(native_mask,guard),sdk_prior=summary(prior_mask,guard),sdk_active_mask=None,
        sdk_valid_prior_range=[0,p],sdk_masked_padding_range=[p,bucket],native_current_range=[0,length]))
    add('prior_mask_input',prefix(heads(n('qk0_scaled')),3,p),prefix(s('prior_mask_input'),3,p))
    add('prior_mask_output',prefix(heads(n('softmax_input')),3,p),prefix(s('prior_mask_output'),3,p),
        'Only valid prior positions have a native counterpart; full masked SDK suffix remains separately reported.')
    for label,active in [('softmax_input','softmax_active_input'),('probabilities_fp32','probabilities_active_fp32'),
                         ('probabilities_bf16','probabilities_active_bf16')]:
        add(label,heads(n(label)),join(prefix(s(label),3,p),s(active),3),
            'Different dtype reports numeric equality separately; no softmax or conversion rounding is replayed.')
    for label in ('pv_prior','pv_active'):
        out.append(dict(boundary=label,status='unknown',reason='Native has no captured prior/active PV split; SDK values are not added or substituted.',
            native=summary(heads(n('pv_context')),guard),sdk=summary(s(label),guard)))
    add('pv_merged_observed',heads(n('pv_context')),reshape(s('context_before_o'),[1,8,1,128]),
        'SDK observed post-merge context is byte-reshaped; prior and active outputs are not recomputed or summed.')
    add('context_before_o',columns(n('context_before_o')),s('context_before_o'))
    out.append(dict(boundary='o_proj_local_before_reduce',status='unknown',
        reason='No native rank-local output projection was captured; SDK partials are not summed.',sdk=summary(s('o_proj_local_before_reduce'),guard)))
    add('attention_output',n('attention_output'),s('o_proj_after_reduce'),'Compare actual observed complete reduced SDK output.')
    padding=[]
    for (label,path),value in se.items():
        shape=value['shape'];axis=padding_axis(label,path)
        if axis is None:continue
        require(len(shape)==4 and shape[axis]==bucket,'Original semantic prior padding axis matches complete bucket')
        selected=s(label,path);ranges=[[0,x]for x in selected['shape']];ranges[axis]=[p,bucket]
        padding.append(dict(label=label,tensor_path=list(path),range=[p,bucket],comparison='unknown',
            reason='Full original SDK masked prior padding has no native counterpart; retained, never treated as valid keys.',
            **summary(select(selected,ranges),guard)))
    different=[x['boundary']for x in out if x['status']=='different'];unknown=[x['boundary']for x in out if x['status']=='unknown']
    return dict(frame=case['frame'],call=case['call'],rank=rank,slot=slot,alignment=mapping,boundaries=out,sdk_padding=padding,
        first_observed_difference=different[0]if different else None,unknown_boundaries=unknown,
        first_difference_is_causal_localization=False,declared_comparison_scope_complete=True,
        comparison_coverage=[dict(boundary=x['boundary'],status=x['status'])for x in out],
        authenticated_sdk_leaves_not_individually_cross_compared=[
            dict(label=label,tensor_path=list(path),reason='Complete bytes authenticated; duplicate SDK snapshot, no separate cross-boundary equality claim.')
            for label,path in se if label in ('route_inputs_sdk_decode','returned_current_kv','attention_output')],
        observed_scale_operations=dict(native=copy.deepcopy(native['scale_operations']),sdk=copy.deepcopy(sdk['scale_operations'])))


def compare(native_result,native_plan,native_audit,sdk_results,sdk_cases,sdk_plan,sdk_recovery,read_bytes,guard):
    guard();admitted(native_result,native_plan,native_audit,sdk_results,sdk_plan,sdk_recovery)
    require(len(sdk_cases)==2 and [x['rank']for x in sdk_results]==[0,1],'Both original rank cases in order')
    reader=Reader(read_bytes,guard);native=load_native(native_result,native_plan,native_audit['result'],reader)
    sdk=[load_sdk(result,cases,sdk_plan,reader)for result,cases in zip(sdk_results,sdk_cases)]
    frames=[];seen=set()
    for case in native_plan['native_operator_manifest']['frames']:
        index=case['frame']['index'];call=case['call']
        for rank in (0,1):
            frames.append(compare_frame(native[index],sdk[rank][call],sdk_results[rank]['official_layer1_shards'],guard))
            seen.add((rank,call,frames[-1]['slot']))
    require(seen=={(r,c,s)for r in (0,1)for c in sdk_manifest.CALLS for s in range(4)},'Every rank/call/physical slot compared exactly')
    for call in sdk_manifest.CALLS:
        require(sdk[0][call]['case']['physical_frame_order']==sdk[1][call]['case']['physical_frame_order']
            and sdk[0][call]['case']['positions']==sdk[1][call]['case']['positions']
            and sdk[0][call]['controls']==sdk[1][call]['controls'],'Both actual rank complete reduced controls and physical cases agree')
    require(reader.raw_counts==dict(native=936,sdk=624)and len(reader.files)==936+624+72,'Every native/SDK raw and native arm metadata read once')
    guard();return dict(schema='minicpm5_layer1_native_sdk_saved_comparison_v3',complete=True,frames=24,rank_frame_comparisons=48,
        raw_files=reader.raw_counts,metadata_files=72,source_files=list(reader.files.values()),comparisons=frames,
        first_observed_difference=next((dict(frame=x['frame'],rank=x['rank'],boundary=x['first_observed_difference'])
            for x in frames if x['first_observed_difference']is not None),None),
        complete_source_tensors_read_before_selection=True,arithmetic_reexecuted=False,actual_exits_validated_here=False,
        caller_must_authenticate_audit_terminals=True,original_sdk_model_exit_code=1,
        attention_causality_proven=False,full_model_qualified=False,new_speedup_claim=False)
