"""Pure-byte recipes for a fixed physical official state, varying SDK batch/layout.

This is a diagnostic slice of the existing full B4 model, not a B1-configured
model or official native forward. No I/O, native imports, model calls or lease.
Only strict_anchor_s0 is selected; its complete cache padding remains intact.
"""
import copy
import itertools
import json
import math
from research import minicpm5_layer1_current_history_factorial_v1 as factorial
boundary=factorial.boundary
require=boundary.require
ANCHOR='strict_anchor_s0'
VARIANTS=('b4_strided','b4_contiguous','b1_strided','b1_contiguous')


def prepare(case,own_normalized_raw,own_cache_raw,prior_donors,current_donors):
    require(case['arm']==ANCHOR and case['seq_ids']==[0,1,2,3]
        and case['call']['index'] in boundary.DECODE_CALLS and case['bucket'] in (128,2048),
        'Original complete anchor decode call only')
    require(case['physical_frame_order']==sorted(case['call']['frames'],key=lambda f:f['slot'])
        and [f['slot']for f in case['physical_frame_order']]==[0,1,2,3],
        'Original physical rows, independent of logical fixture ordering')
    normalized,packed,caches,proof=factorial.select(case,own_normalized_raw,own_cache_raw,
        prior_donors,current_donors,'official_both')
    return dict(case=copy.deepcopy(case),normalized_raw=normalized,packed_raw=packed,
        cache_raw=caches,intervention=proof,
        source_case_sha256=boundary.sha(json.dumps(case,sort_keys=True,separators=(',',':')).encode()))


def _view(source,raw,shape,stride,offset):
    require(type(raw)is bytes and len(shape)==len(stride) and all(type(v)is int and v>0 for v in shape+stride)
        and type(offset)is int and offset>=0,'Exact positive BF16 view dimensions')
    end=offset+sum((size-1)*step for size,step in zip(shape,stride))+1
    require(2*end<=len(raw),'Selected view is within complete authenticated storage')
    return dict(source=source,shape=shape,stride=stride,offset=offset,dtype='torch.bfloat16',
        storage_bytes=len(raw),storage_sha256=boundary.sha(raw),logical_bytes=2*math.prod(shape),
        highest_referenced_element_exclusive=end)


def logical_bytes(raw,view):
    """Byte interpretation of a recipe, including non-contiguous and offset views.

    Backing bytes outside the view are retained by the caller; they are not
    tensor elements and are never counted as selected logical bytes.
    """
    expected=_view(view['source'],raw,view['shape'],view['stride'],view['offset'])
    require(expected==view,'Unchanged complete source and exact view recipe')
    output=bytearray(view['logical_bytes']);position=0
    for index in itertools.product(*(range(size)for size in view['shape'])):
        word=view['offset']+sum(i*s for i,s in zip(index,view['stride']))
        output[position:position+2]=raw[2*word:2*word+2];position+=2
    return bytes(output)


def tasks(prepared,variant):
    require(variant in VARIANTS,'Only same-SDK batch and normalized-layout variants')
    case=prepared['case'];proof=prepared['intervention'];bucket=case['bucket']
    require(prepared['source_case_sha256']==boundary.sha(json.dumps(case,sort_keys=True,separators=(',',':')).encode()),
        'Original source case remains unchanged after official state preparation')
    require(case['arm']==ANCHOR and proof['mode']=='official_both' and proof['arm']==ANCHOR
        and proof['call']==case['call'] and proof['rank']==case['rank']
        and proof['original_call']==case['original_call']
        and proof['original_normalized_input']==case['normalized_input']
        and [c['sdk_view']['shape']for c in case['caches']]==[[4,1,bucket,128]]*2
        and proof['selected_layout']==boundary.NORM_META
        and proof['selected_normalized_sha256']==boundary.sha(prepared['normalized_raw'])
        and proof['selected_storage_sha256']==boundary.sha(prepared['packed_raw'])
        and prepared['packed_raw']==boundary.normalized_storage(prepared['normalized_raw'],boundary.NORM_META,
            proof['selected_normalized_sha256']), 'Frozen official selection on exact original source case')
    require(set(prepared['cache_raw'])==set(boundary.CACHE_NAMES)
        and all(boundary.sha(prepared['cache_raw'][name])==proof['history_intervention']['output_cache_sha256'][name]
            for name in boundary.CACHE_NAMES),'Full anchor cache bytes including all masked/padding positions')
    contiguous=variant.endswith('contiguous');batch=1 if variant.startswith('b1_')else 4
    groups=[[slot]for slot in range(4)]if batch==1 else [list(range(4))]
    source='normalized_logical' if contiguous else 'normalized_packed'
    raw=prepared['normalized_raw']if contiguous else prepared['packed_raw']
    stride=[2048,2048,1]if contiguous else [1,4,4]
    rows=[]
    for slots in groups:
        slot=slots[0];positions=[case['positions'][s]for s in slots]
        require(all(type(p)is int and 0<p<bucket for p in positions),'Original positive valid history lengths')
        view=_view(source,raw,[batch,1,2048],stride,slot*2048 if contiguous else slot)
        normalized=b''.join(prepared['normalized_raw'][s*4096:(s+1)*4096]for s in slots)
        require(logical_bytes(raw,view)==normalized,'Exact row bytes after explicit source-stride selection')
        cache={}
        for name in boundary.CACHE_NAMES:
            cache[name]=_view(name,prepared['cache_raw'][name],[batch,1,bucket,128],
                list(boundary.CACHE_STRIDE),slot*boundary.CACHE_STRIDE[0])
        mask=bytes(int(column<position)for position in positions for column in range(bucket))
        rows.append(dict(schema='minicpm5_layer1_sdk_batch_layout_task_v1',variant=variant,
            original_call=copy.deepcopy(case['original_call']),rank=case['rank'],arm=ANCHOR,call=copy.deepcopy(case['call']),
            slots=slots,physical_frames=[copy.deepcopy(case['physical_frame_order'][s])for s in slots],
            normalized=view,normalized_logical_sha256=boundary.sha(normalized),caches=cache,
            positions=positions,seq_ids=slots,mask=dict(shape=[batch,1,1,bucket],stride=[bucket,bucket,bucket,1],
                offset=0,dtype='torch.bool',sha256=boundary.sha(mask),logical_bytes=len(mask)),
            mask_raw=mask,bucket=bucket,valid_prior_lengths=positions,
            native_total_lengths=[p+1 for p in positions],sdk_prior_width=bucket,sdk_active_width=1,
            rope=dict(source='same_full_B4_SDK_RoPE_once_per_original_call',operation='narrow_dim0'if batch==1 else 'identity',
                slots=slots,shape=[batch,1,128],source_stride_and_offset='observe_and_retain_at_runtime',
                require_same_selected_cos_sin_bytes=True,recompute_for_B1=False),
            cache_selection='original_B4_get_kv_by_layer_id_then_narrow_dim0',
            normalized_unselected_backing_bytes=len(raw)-len(normalized),
            backing_bytes_outside_view_are_not_logical_inputs=True,complete_anchor_cache_storage_retained=True,
            qkv_projection=dict(reduction_width=2048,rank_query_heads=8,rank_kv_heads=1),
            attention_arithmetic=dict(qk_reduction_width=128,softmax='original_FP32_manual_prior_plus_active',
                pv_prior_reduction_width=bucket,pv_active_reduction_width=1,pv_add_dtype='torch.bfloat16'),
            output_projection=dict(rank_reduction_width=1024,rank_partial_dtype='torch.bfloat16',
                collective_ranks=2,collective_dtype='torch.bfloat16',collective_elements=batch*2048),
            output_shape=[batch,1,2048],model_config_changed=False,full_model_reexecuted=False,
            native_reference_reexecuted=False,full_model_qualified=False,new_speedup_claim=False))
    return rows


def combine(case,variant,rows):
    """Assemble all four output rows without assuming fixture order equals slot order."""
    require(case['arm']==ANCHOR and variant in VARIANTS and type(rows)is list,'Original anchor output assembly')
    expected=[[s]for s in range(4)]if variant.startswith('b1_')else [[0,1,2,3]]
    require([row['slots']for row in rows]==expected,'All original physical slots exactly once in execution order')
    output=[]
    for row,slots in zip(rows,expected):
        require(row['rank']==case['rank'] and row['call']==case['call'] and row['variant']==variant
            and row['physical_frames']==[case['physical_frame_order'][s]for s in slots]
            and type(row['raw'])is bytes and len(row['raw'])==4096*len(slots)
            and factorial.history.finite_bf16(row['raw']), 'Complete finite original output rows with explicit provenance')
        output.append(row['raw'])
    raw=b''.join(output);require(len(raw)==16384,'Complete four-row output retained for every variant')
    return raw


def verify_controls(original_raw,previous_own,own_raw,previous_both,b4_strided_raw,restore_raw):
    require(all(type(raw)is bytes and len(raw)==16384 and factorial.history.finite_bf16(raw)
        for raw in (original_raw,previous_own,own_raw,previous_both,b4_strided_raw,restore_raw)),
        'All controls are complete finite four-row raw tensors')
    require(original_raw==previous_own==own_raw==restore_raw,'Original, previous own and restored raw outputs exact')
    require(previous_both==b4_strided_raw,'Previous actual both-official output exactly reproduced')
    return dict(original_own_restore_full_bytes_exact=True,previous_both_official_full_bytes_exact=True,
        full_model_qualified=False,new_speedup_claim=False)
