"""Exact source-derived SDK B4 decode capture manifest; no tensor/native IO."""
import copy
import math
import minicpm5_layer1_operator_capture_observer_v1 as observer
require=observer.require
SCHEMA='minicpm5_layer1_operator_sdk_manifest_v1'
CALLS=(4,5,6,11,12,13)
BUCKETS=(2048,2048,2048,128,128,128)
MODES=('own','official_unobserved','official_noop','official_full','restore')
DTYPE_BYTES={'torch.bfloat16':2,'torch.float32':4,'torch.int64':8,'torch.bool':1}
OUTPUT_CAP=256*2**20
RANK_CAP=128*2**20
TENSOR_CAP=16*2**20
METADATA_CAP_PER_CALL=2**20


def tensor(path,shape,dtype='torch.bfloat16',scores=False):
    return dict(path=path,shape=shape,dtype=dtype,bytes=math.prod(shape)*DTYPE_BYTES[dtype],
        selection=dict(kind='prediction_query_rows',axis=2,positions=[0]*4) if scores else None,
        source_query_axis_complete=True)


def events(bucket):
    require(type(bucket)is int and bucket in BUCKETS,'Original SDK decode bucket')
    norm=[4,1,2048];q=[4,8,1,128];kv=[4,1,1,128];cache=[4,1,bucket,128]
    positions=[4,1];mask=[4,1,1,bucket];rope=[4,1,128];score=[4,8,1,bucket];active=[4,8,1,1]
    rows=[]
    def add(label,*leaves):rows.append(dict(label=label,tensors=list(leaves)))
    def one(label,shape,dtype='torch.bfloat16',scores=False):add(label,tensor([],shape,dtype,scores))
    one('normalized_input',norm);one('attention_mask',mask,'torch.bool');add('active_mask')
    one('position_ids',positions,'torch.int64');add('cache_backing_before_reference');add('dispatch_flags')
    add('effective_prior_kv',tensor([0],cache),tensor([1],cache))
    one('q_projection',[4,1,1024]);one('k_projection',[4,1,128]);one('v_projection',[4,1,128])
    one('rope_position_ids',positions,'torch.int64');one('v_heads',kv);add('rotary_route')
    one('q_rope_input',q);one('k_rope_input',kv);one('rope_cos',rope);one('rope_sin',rope)
    one('q_post_rope',q);one('k_post_rope',kv)
    one('q_after_prep',q);one('k_after_prep',kv);one('v_after_prep',kv)
    add('route_inputs_sdk_decode',tensor(['Q'],q),tensor(['K'],kv),tensor(['V'],kv),
        tensor(['position_ids'],positions,'torch.int64'),tensor(['past_key_value',0],cache),
        tensor(['past_key_value',1],cache),tensor(['attention_mask'],mask,'torch.bool'))
    one('qk0_unscaled',score,scores=True);one('qk0_scaled',score,scores=True);add('qk0_scale_operation')
    one('prior_actual_mask',mask,'torch.bool');one('prior_mask_input',score,scores=True);one('prior_mask_output',score,scores=True)
    one('qk1_unscaled',active,scores=True);one('qk1_scaled',active,scores=True);add('qk1_scale_operation')
    one('softmax_input',score,'torch.float32',True);one('softmax_active_input',active,'torch.float32',True)
    one('probabilities_fp32',score,'torch.float32',True);one('probabilities_active_fp32',active,'torch.float32',True)
    one('probabilities_bf16',score,scores=True);one('pv_prior',q)
    one('probabilities_active_bf16',active,scores=True);one('pv_active',q)
    one('context_before_o',[4,1,1024]);one('o_proj_local_before_reduce',norm);one('o_proj_after_reduce',norm)
    one('attention_output',norm);add('returned_current_kv',tensor([0],kv),tensor([1],kv));add('cache_expected_after_reference')
    require(len(rows)<=64 and len({r['label']for r in rows})==len(rows),'Complete bounded unique event order')
    require(all(0<t['bytes']<=TENSOR_CAP for row in rows for t in row['tensors']),'Original per-tensor byte bound')
    return rows


def call_manifest(rank,call,bucket):
    require(type(rank)is int and rank in (0,1) and call in CALLS and bucket==BUCKETS[CALLS.index(call)],
        'Exact original TP2 six-call bucket mapping')
    rows=events(bucket);raw=sum(t['bytes']for r in rows for t in r['tensors'])
    allocated=sum(((t['bytes']+4095)//4096)*4096 for r in rows for t in r['tensors'])
    tensors=sum(len(r['tensors'])for r in rows)
    return dict(schema=SCHEMA,rank=rank,call=call,bucket=bucket,physical_batch=4,query_length=1,layer=1,
        tensor_events=rows,event_count=len(rows),tensor_count=tensors,
        capture_raw_files=tensors,control_raw_files=len(MODES),total_raw_files=tensors+len(MODES),
        capture_allocated_4k_bytes=allocated,control_allocated_4k_bytes=len(MODES)*16384,
        total_raw_allocated_4k_bytes=allocated+len(MODES)*16384,
        capture_raw_bytes=raw,control_raw_bytes=len(MODES)*16384,total_raw_bytes=raw+len(MODES)*16384,
        metadata_cap_bytes=METADATA_CAP_PER_CALL,per_tensor_cap_bytes=TENSOR_CAP,
        full_score_key_axis_preserved=True,original_cache_backing_bytes_per_tensor=4325376,
        original_backing_authenticated_before_after_not_reserialized=True)


def manifest():
    rows=[call_manifest(rank,ci,b) for rank in (0,1) for ci,b in zip(CALLS,BUCKETS)]
    raw=sum(r['total_raw_bytes']for r in rows);metadata=len(rows)*METADATA_CAP_PER_CALL
    allocated=sum(r['total_raw_allocated_4k_bytes']for r in rows)
    require(raw+metadata<OUTPUT_CAP and (raw+metadata)//2<RANK_CAP,'Original total/rank output caps')
    return dict(schema=SCHEMA,stage='sdk_only',layer=1,ranks=[0,1],checkpoints=24,calls=list(CALLS),
        buckets=list(BUCKETS),modes=list(MODES),records=rows,total_capture_raw_bytes=sum(r['capture_raw_bytes']for r in rows),
        total_control_raw_bytes=sum(r['control_raw_bytes']for r in rows),total_raw_bytes=raw,
        raw_capture_files=sum(r['capture_raw_files']for r in rows),raw_control_files=sum(r['control_raw_files']for r in rows),
        total_raw_files=sum(r['total_raw_files']for r in rows),allocation_block_bytes=4096,
        total_raw_allocated_4k_bytes=allocated,
        allocation_is_exact_for_one_4k_block_per_file_extent_model=True,
        filesystem_metadata_and_actual_st_blocks_require_outer_usage_check=True,
        old_512_file_outer_cap_insufficient=True,
        outer_must_bind_exact_additional_runtime_metadata_file_catalog=True,
        bounded_call_metadata_bytes=metadata,total_raw_plus_call_metadata_bytes=raw+metadata,
        original_total_output_cap_bytes=OUTPUT_CAP,original_rank_output_cap_bytes=RANK_CAP,
        actual_physical_attention_invocations=60,full_capture_invocations=12,noop_invocations=12,
        unknown_events_rejected=True,unused_byte_allowance_does_not_allow_extra_events=True,
        full_model_qualified=False,new_speedup_claim=False)


class StrictSink(observer.SnapshotSink):
    """Clone sink enforcing exact event, nested leaf, shape, dtype and byte order."""
    def __init__(self,torch,writer,expected):
        self.expected=copy.deepcopy(expected);self.event=None;self.leaf=0
        super().__init__(torch,writer,max_call_bytes=expected['capture_raw_bytes'],max_tensor_bytes=TENSOR_CAP)
    def tensor(self,value,selection,context):
        require(self.event is not None and self.leaf<len(self.event['tensors']),'No extra capture tensor')
        wanted=self.event['tensors'][self.leaf]
        require(list(value.shape)==wanted['shape'] and str(value.dtype)==wanted['dtype']
            and selection==wanted['selection'] and value.numel()*value.element_size()==wanted['bytes'],
            'Exact full source tensor shape, dtype, selection and bytes before cloning')
        bound=dict(context,event_label=self.event['label'],tensor_index=self.leaf,tensor_path=wanted['path'])
        self.leaf+=1
        return super().tensor(value,selection,bound)
    def __call__(self,label,value,selection,context):
        index=len(self.records)
        require(index<len(self.expected['tensor_events']) and context['event_index']==index,
                'Exact capture event index')
        self.event=self.expected['tensor_events'][index];self.leaf=0
        require(label==self.event['label'],'Exact original event order; no omitted or extra boundary')
        # Validate leaf placement, not only equal-shape encounter order.
        for leaf in self.event['tensors']:
            actual=value
            for part in leaf['path']:actual=actual[part]
            require(isinstance(actual,self.torch.Tensor),'Every exact manifest leaf is a tensor')
        super().__call__(label,value,selection,context)
        require(self.leaf==len(self.event['tensors']),'Every expected tensor captured in full')
        self.event=None
    def finish(self):
        require(len(self.records)==self.expected['event_count'] and self.bytes==self.expected['capture_raw_bytes'],
                'Exact full event and raw byte manifest exhausted')
        return dict(complete=True,manifest=copy.deepcopy(self.expected),raw_bytes=self.bytes,records=copy.deepcopy(self.records))
