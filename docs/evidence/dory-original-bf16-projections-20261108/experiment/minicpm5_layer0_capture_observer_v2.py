"""Draft transparent layer0 observer library; import is stdlib-only.

No CLI, model construction, compilation, admission or file output. Runtime caller
supplies torch, exact model/functions and a bounded authenticated sink. Original
functions run once; original objects are returned. Tests use inert fake tensors.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import functools
import inspect
import json
import math
import threading

_ACTIVE=ContextVar('minicpm5_layer0_capture_active',default=None)
_SUPPRESS=ContextVar('minicpm5_layer0_capture_suppress',default=False)


def require(value,message):
    if not value:raise RuntimeError(message)


def tensor_shape(value):return tuple(int(x) for x in value.shape)


class Capture:
    """One process/rank, one dynamic original call; sink snapshots synchronously.

    sink(label,value,selection,context) must clone/hash without mutating value,
    bound bytes and preserve original shape/stride/offset in metadata. For
    score/probability matrices selection identifies physical prediction rows.
    No clone is retained by this observer after sink returns.
    """
    def __init__(self,torch,sink,arm,rank,frames,call,phase,prediction_positions):
        require(arm in ('HF_native','C_Hd_control','C_Hc_history_generation'),'Arm')
        require((arm=='HF_native' and rank==-1) or (arm!='HF_native' and rank in (0,1)),'Rank')
        require(phase in ('prefill','decode') and len(frames)==len(prediction_positions),'Frame positions')
        require(len({f['index'] for f in frames})==len(frames),'Duplicate frames')
        self.torch=torch;self.sink=sink;self.arm=arm;self.rank=rank;self.frames=frames;self.call=call;self.phase=phase
        self.positions=list(prediction_positions);self.thread=threading.get_ident();self.layer_depth=0
        self.route=None;self.route_calls=[];self.counts={};self.events=[];self.score_role=None;self.matmul_count=0
        self.forward_calls=0;self.error=None;self._token=None
        self.pending_qk=None;self.last_qk=None;self.scaling_seen=set();self.rope_depth=0

    def context(self):
        return dict(arm=self.arm,rank=self.rank,call=self.call,phase=self.phase,frames=self.frames,
            thread=self.thread,route=self.route,event_index=len(self.events))

    def check_thread(self):require(threading.get_ident()==self.thread,'Capture crossed Python threads')

    def emit(self,label,value,selection=None):
        self.check_thread();require(self.layer_depth==1,'Capture outside layer0')
        require(label not in self.counts,'Duplicate capture label: '+label);self.counts[label]=1
        token=_SUPPRESS.set(True)
        try:self.sink(label,value,selection,self.context())
        finally:_SUPPRESS.reset(token)
        self.events.append(label)

    def scores(self,label,value):
        require(hasattr(value,'shape') and len(value.shape)==4,'Score must be BHQK tensor')
        require(value.shape[0]==len(self.positions),'Score batch mapping')
        require(all(0<=p<value.shape[2] for p in self.positions),'Score query selection')
        self.emit(label,value,dict(kind='prediction_query_rows',axis=2,positions=self.positions))

    @contextmanager
    def activate(self):
        require(_ACTIVE.get() is None,'Concurrent/nested capture context')
        self._token=_ACTIVE.set(self)
        try:yield self
        finally:
            _ACTIVE.reset(self._token);self._token=None

    @contextmanager
    def layer(self):
        self.check_thread();require(self.layer_depth==0,'Nested layer0 forward')
        self.forward_calls+=1;require(self.forward_calls==1,'More than one layer0 forward')
        self.layer_depth=1
        try:yield
        finally:self.layer_depth=0

    def enter_route(self,name):
        require(self.route is None,'Nested attention math route')
        require(name in ('hf_eager','sdk_prefill','sdk_decode'),'Unreviewed attention route')
        require(name==('hf_eager' if self.arm=='HF_native' else 'sdk_prefill' if self.phase=='prefill' else 'sdk_decode'),'Wrong phase/arm dispatch')
        self.route=name;self.route_calls.append(name);self.matmul_count=0;self.score_role=None;self.scaling_seen=set()

    def exit_route(self):
        required=2 if self.route in ('hf_eager','sdk_prefill') else 4
        require(self.matmul_count==required,'Unexpected attention matmul dispatch count')
        require(self.pending_qk is None and len(self.scaling_seen)==(2 if self.route=='sdk_decode' else 1),'Missing actual QK scaling')
        self.route=None;self.score_role=None

    def record_qk(self,role,result):
        require(self.pending_qk is None,'Previous QK not scaled')
        self.score_role=role;self.pending_qk=result;self.last_qk=result
        self.scores(role+'_unscaled',result)

    def observe_scale(self,operator,args,result):
        # Scalar arithmetic on any other object is not this QK operation.
        if not args or args[0] is not self.last_qk:return False
        expected=('aten.mul.Tensor','aten.mul.Scalar') if self.route=='hf_eager' else ('aten.div.Tensor','aten.div.Scalar')
        if operator not in ('aten.mul.Tensor','aten.mul.Scalar','aten.div.Tensor','aten.div.Scalar'):return False
        require(self.pending_qk is args[0] and self.score_role not in self.scaling_seen,'Duplicate QK scaling')
        require(operator in expected and len(args)==2 and type(args[1]) in (float,int),'Wrong QK scaling operator')
        scalar=128**-.5 if self.route=='hf_eager' else math.sqrt(128)
        require(args[1]==scalar,'Wrong QK scaling scalar')
        require(tensor_shape(result)==tensor_shape(self.pending_qk),'Scaling changed score geometry')
        self.scores(self.score_role+'_scaled',result)
        self.emit(self.score_role+'_scale_operation',dict(operator=operator,scalar=args[1],operand_is_recorded_qk=True))
        self.scaling_seen.add(self.score_role);self.pending_qk=None
        return True

    def complete(self):
        require(self.forward_calls==1 and self.layer_depth==0 and self.route is None,'Incomplete layer capture')
        require(len(self.route_calls)==1,'Missing/repeated actual attention route')
        wanted={'normalized_input','attention_mask','position_ids','q_projection','k_projection','v_projection','q_rope_input','k_rope_input','v_heads','rope_position_ids','rotary_route',
            'q_post_rope','k_post_rope','rope_cos','rope_sin','context_before_o','attention_output',
            'qk0_unscaled','qk0_scaled','qk0_scale_operation','softmax_input','probabilities_bf16'}
        if self.arm=='HF_native':wanted|={'cache_before','cache_after','native_effective_key','native_effective_value','probabilities_fp32'}
        else:wanted|={'cache_backing_before_reference','cache_expected_after_reference','returned_current_kv','effective_prior_kv','active_mask','dispatch_flags'}
        if self.route_calls[0]=='sdk_decode':wanted|={'qk1_unscaled','qk1_scaled','qk1_scale_operation','prior_mask_input','prior_mask_output','softmax_active_input','probabilities_active_fp32','probabilities_active_bf16'}
        else:wanted|={'probabilities_fp32'}
        require(wanted<=set(self.events),'Missing observed values: '+str(sorted(wanted-set(self.events))))
        return dict(complete=True,arm=self.arm,rank=self.rank,call=self.call,phase=self.phase,frames=self.frames,
            actual_route=self.route_calls[0],events=list(self.events),original_layer_forward_calls=1,
            wrappers_return_original_objects=True,full_model_endpoint_gate_still_required=True,
            production_runtime_tested=False)


def current():
    c=_ACTIVE.get()
    if c is None or _SUPPRESS.get() or c.layer_depth!=1:return None
    c.check_thread();return c


def bind(original,args,kwargs):
    bound=inspect.signature(original).bind(*args,**kwargs);bound.apply_defaults();return bound.arguments


class Patches:
    """Patch exact supplied attributes, restoring original own/inherited shape."""
    def __init__(self):self.rows=[]
    def replace(self,obj,name,make):
        original=getattr(obj,name);owned=name in getattr(obj,'__dict__',{})
        wrapper=make(original);setattr(obj,name,wrapper);self.rows.append((obj,name,original,wrapper,owned));return wrapper
    def close(self):
        conflicts=[]
        for obj,name,original,wrapper,owned in reversed(self.rows):
            try:
                if getattr(obj,name) is not wrapper:
                    conflicts.append(name+': externally replaced');continue
                if owned:setattr(obj,name,original)
                else:delattr(obj,name)
            except BaseException as error:conflicts.append(name+': '+type(error).__name__)
        self.rows=[]
        require(not conflicts,'Patch restoration conflicts: '+', '.join(conflicts))
    def __enter__(self):return self
    def __exit__(self,*args):self.close()


def simple_wrapper(before=None,after=None,only_active=True):
    def make(original):
        @functools.wraps(original)
        def wrapped(*args,**kwargs):
            c=current() if only_active else _ACTIVE.get()
            if c is None or _SUPPRESS.get():return original(*args,**kwargs)
            bound=bind(original,args,kwargs)
            if before:before(c,bound)
            result=original(*args,**kwargs)
            if after:after(c,bound,result)
            return result
        return wrapped
    return make


def native_cache(cache):
    if cache is None:return None
    require(hasattr(cache,'layers'),'Native cache structure')
    if len(cache.layers)==0:return None
    layer=cache.layers[0];keys=getattr(layer,'keys',None);values=getattr(layer,'values',None)
    require((keys is None)==(values is None),'Partial native cache')
    return None if keys is None else (keys,values)


def install_native(patches,attention,llama_module,torch):
    """Apply only to exact layer0 attention instance supplied by admitted caller."""
    require(getattr(attention,'layer_idx',None)==0,'Native wrong layer')
    def layer_make(original):
        @functools.wraps(original)
        def wrapped(*args,**kwargs):
            c=_ACTIVE.get()
            if c is None:return original(*args,**kwargs)
            require(c.arm=='HF_native','Wrong native arm');b=bind(original,args,kwargs)
            with c.layer():
                c.emit('normalized_input',b['hidden_states']);c.emit('attention_mask',b['attention_mask'])
                # Position IDs are supplied by the exact driver context, separately
                # from the precomputed native RoPE pair passed at this boundary.
                require(hasattr(c,'position_ids'),'Driver must bind actual native position IDs')
                c.emit('position_ids',c.position_ids)
                cache=b.get('past_key_values');c.emit('cache_before',native_cache(cache))
                require(cache is not None,'Original cached native trajectory required')
                def cache_update(original_update):
                    @functools.wraps(original_update)
                    def capture_update(*ua,**uk):
                        values=bind(original_update,ua,uk);require(values['layer_idx']==0,'Unexpected cache layer in layer0 scope')
                        c.emit('v_heads',values['value_states']);c.emit('cache_update_active_key',values['key_states'])
                        return original_update(*ua,**uk)
                    return capture_update
                with Patches() as cache_patches:
                    cache_patches.replace(cache,'update',cache_update)
                    result=original(*args,**kwargs)
                c.emit('cache_after',native_cache(cache));c.emit('attention_output',result[0])
                return result
        return wrapped
    patches.replace(attention,'forward',layer_make)
    for module,label in [(attention.q_proj,'q_projection'),(attention.k_proj,'k_projection'),(attention.v_proj,'v_projection')]:
        patches.replace(module,'forward',simple_wrapper(after=lambda c,b,out,label=label:c.emit(label,out)))
    def rotary_before(c,b):
        c.emit('q_rope_input',b['q']);c.emit('k_rope_input',b['k']);c.emit('rope_position_ids',c.position_ids)
        c.emit('rotary_route',dict(function='native.apply_rotary_pos_emb',position_source='actual_original_model_call'))
        c.emit('rope_cos',b['cos']);c.emit('rope_sin',b['sin'])
    def rotary_after(c,b,out):c.emit('q_post_rope',out[0]);c.emit('k_post_rope',out[1])
    patches.replace(llama_module,'apply_rotary_pos_emb',simple_wrapper(rotary_before,rotary_after))
    def eager_make(original):
        @functools.wraps(original)
        def wrapped(*args,**kwargs):
            c=current()
            if c is None:return original(*args,**kwargs)
            b=bind(original,args,kwargs);require(b['module'] is attention,'Unexpected native attention module')
            c.emit('native_effective_key',b['key']);c.emit('native_effective_value',b['value'])
            c.emit('actual_eager_mask',b['attention_mask']);c.enter_route('hf_eager')
            result=original(*args,**kwargs);c.exit_route();return result
        return wrapped
    patches.replace(llama_module,'eager_attention_forward',eager_make)
    patches.replace(attention.o_proj,'forward',simple_wrapper(before=lambda c,b:c.emit('context_before_o',b['input'])))
    install_math_observation(patches,torch,None)


def install_sdk(patches,attention,attention_base_module,torch):
    """Exact layer0 instance; actual route is observed rather than inferred flags."""
    def layer_make(original):
        @functools.wraps(original)
        def wrapped(*args,**kwargs):
            c=_ACTIVE.get()
            if c is None:return original(*args,**kwargs)
            require(c.arm in ('C_Hd_control','C_Hc_history_generation'),'Wrong SDK arm');b=bind(original,args,kwargs)
            with c.layer():
                require(hasattr(c,'cache_backing_before') and hasattr(c,'cache_backing_after'),'Driver must bind authenticated layer0 cache references')
                c.emit('normalized_input',b['hidden_states']);c.emit('attention_mask',b['attention_mask'])
                c.emit('active_mask',b['active_mask']);c.emit('position_ids',b['position_ids'])
                c.emit('cache_backing_before_reference',c.cache_backing_before)
                flags={k:getattr(attention,k,None) for k in ('attn_tkg_nki_kernel_enabled','attn_tkg_builtin_kernel_enabled','attn_block_tkg_nki_kernel_enabled','attn_block_cte_nki_kernel_enabled','flash_decoding_enabled','k_cache_transposed','sequence_parallel_enabled','num_heads','num_key_value_heads','head_dim')}
                c.emit('dispatch_flags',flags)
                result=original(*args,**kwargs)
                c.emit('attention_output',result.hidden_states)
                # Actual returned K/V is captured; post-global-cache hashes must be
                # supplied after the full original CPU model call, by the driver.
                c.emit('returned_current_kv',result.present_key_value)
                c.emit('cache_expected_after_reference',c.cache_backing_after)
                return result
        return wrapped
    patches.replace(attention,'forward',layer_make)
    def qkv_after(c,b,out):
        c.emit('q_projection',out[0]);c.emit('k_projection',out[1]);c.emit('v_projection',out[2])
    patches.replace(attention.qkv_proj,'forward',simple_wrapper(after=qkv_after))
    def prep_before(c,b):c.emit('effective_prior_kv',b['past_key_value'])
    def prep_after(c,b,out):
        c.emit('q_after_prep',out[0]);c.emit('k_after_prep',out[1]);c.emit('v_after_prep',out[2])
    patches.replace(attention,'prep_qkv_tensors',simple_wrapper(prep_before,prep_after))
    def rotary_scope(original):
        @functools.wraps(original)
        def wrapped(*args,**kwargs):
            c=current()
            if c is None:return original(*args,**kwargs)
            b=bind(original,args,kwargs);require(c.rope_depth==0 and b['use_polar_compatible_rope'] is False,'Unexpected rotary branch')
            c.emit('rope_position_ids',b['position_ids']);c.emit('v_heads',b['V']);c.rope_depth=1
            try:return original(*args,**kwargs)
            finally:c.rope_depth=0
        return wrapped
    patches.replace(attention,'apply_rotary_embedding',rotary_scope)
    def immediate_before(c,b):
        require(c.rope_depth==1,'SDK rotary helper outside expected scope')
        c.emit('rotary_route',dict(function='SDK.apply_rotary_embedding/apply_rotary_pos_emb',polar=False))
        c.emit('q_rope_input',b['q']);c.emit('k_rope_input',b['k']);c.emit('rope_cos',b['cos']);c.emit('rope_sin',b['sin'])
    def immediate_after(c,b,out):c.emit('q_post_rope',out[0]);c.emit('k_post_rope',out[1])
    patches.replace(attention_base_module,'apply_rotary_pos_emb',simple_wrapper(immediate_before,immediate_after))
    def route_make(name):
        def make(original):
            @functools.wraps(original)
            def wrapped(*args,**kwargs):
                c=current()
                if c is None:return original(*args,**kwargs)
                b=bind(original,args,kwargs);c.emit('route_inputs_'+name,dict(b));c.enter_route(name)
                result=original(*args,**kwargs);c.exit_route();return result
            return wrapped
        return make
    patches.replace(attention,'perform_prefill',route_make('sdk_prefill'))
    patches.replace(attention,'compute_for_token_gen',route_make('sdk_decode'))
    patches.replace(attention.o_proj,'forward',simple_wrapper(before=lambda c,b:c.emit('context_before_o',b['attention_output'])))
    install_math_observation(patches,torch,attention_base_module)


def install_math_observation(patches,torch,attention_base_module):
    """Observe original top-level matmul/where/softmax and actual scaling ops."""
    def matmul_make(original):
        @functools.wraps(original)
        def wrapped(*args,**kwargs):
            c=current()
            if c is None or c.route is None:return original(*args,**kwargs)
            require(len(args)==2 and not kwargs,'Unexpected attention matmul call form')
            i=c.matmul_count;c.matmul_count+=1
            qk=i<(2 if c.route=='sdk_decode' else 1)
            if not qk:
                label='probabilities_bf16' if i==(2 if c.route=='sdk_decode' else 1) else 'probabilities_active_bf16'
                c.scores(label,args[0])
            result=original(*args,**kwargs)
            if qk:
                c.record_qk('qk%d'%i,result)
            else:c.score_role=None
            return result
        return wrapped
    patches.replace(torch,'matmul',matmul_make)
    def where_make(original):
        @functools.wraps(original)
        def wrapped(*args,**kwargs):
            c=current()
            if c is None or c.route is None:return original(*args,**kwargs)
            require(len(args)==3 and not kwargs,'Unexpected attention where form')
            # Only score masks at the currently observed QK stage are captured.
            require(c.score_role is not None,'Unexpected where outside QK stage')
            prefix='prior' if c.route=='sdk_decode' else 'prefill'
            c.emit(prefix+'_actual_mask',args[0]);c.scores(prefix+'_mask_input',args[1])
            result=original(*args,**kwargs);c.scores(prefix+'_mask_output',result);return result
        return wrapped
    patches.replace(torch,'where',where_make)
    def softmax_make(original):
        @functools.wraps(original)
        def wrapped(*args,**kwargs):
            c=current()
            if c is None or c.route is None:return original(*args,**kwargs)
            b=bind(original,args,kwargs);c.scores('softmax_input',b['input'])
            result=original(*args,**kwargs);c.scores('probabilities_fp32',result);return result
        return wrapped
    patches.replace(torch.nn.functional,'softmax',softmax_make)
    if attention_base_module is not None:
        def manual_make(original):
            @functools.wraps(original)
            def wrapped(*args,**kwargs):
                c=current()
                if c is None or c.route is None:return original(*args,**kwargs)
                require(c.route=='sdk_decode','Manual softmax unexpected route');b=bind(original,args,kwargs)
                require(b['is_speculation'] is False,'Unexpected speculation/prefix')
                c.scores('softmax_input',b['prior_scores']);c.scores('softmax_active_input',b['active_scores'])
                result=original(*args,**kwargs)
                c.scores('probabilities_fp32',result[0]);c.scores('probabilities_active_fp32',result[1]);return result
            return wrapped
        patches.replace(attention_base_module,'manual_softmax',manual_make)


def scaling_dispatch_mode(torch):
    """Runtime-only dispatch observer for actual QK scalar multiply/divide.

    The admitted caller must enter this mode around the original model call.
    It returns each original aten result without rewriting arithmetic.
    """
    from torch.utils._python_dispatch import TorchDispatchMode
    class ScalingObserver(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            result=func(*args,**(kwargs or {}));c=current()
            if c is not None and c.route is not None and c.score_role is not None:
                c.observe_scale(str(func),args,result)
            return result
    return ScalingObserver()


def version_counter(value):
    try:return value._version
    except RuntimeError:return None  # Inference-mode tensors may have no counter.


class SnapshotSink:
    """Bounded synchronous clone sink, supplied writer owns exclusive file output.

    This does not open files. writer(tensor_clone,metadata) must immediately
    serialize to a fresh admitted path, drop the tensor and return a JSON-only
    reference. Observer payload clones never share storage with model tensors.
    """
    def __init__(self,torch,writer,max_call_bytes=256*2**20,max_tensor_bytes=16*2**20):
        self.torch=torch;self.writer=writer;self.max_call_bytes=max_call_bytes;self.max_tensor_bytes=max_tensor_bytes
        self.bytes=0;self.records=[]

    def tensor(self,value,selection,context):
        torch=self.torch;require(value.device.type=='cpu','CPU capture only')
        shape=list(value.shape);stride=list(value.stride());offset=value.storage_offset();version=version_counter(value)
        original_numel=value.numel();selected_shape=list(shape)
        if selection is not None:
            require(isinstance(selection,dict) and selection.get('axis')==2 and value.ndim==4,'Selection schema')
            kind=selection.get('kind')
            if kind=='prediction_query_rows':
                require(set(selection)=={'kind','axis','positions'},'Unexpected selection keys')
                positions=selection['positions']
                require(isinstance(positions,list) and len(positions)==shape[0] and
                    all(type(p) is int and 0<=p<shape[2] for p in positions),'Invalid prediction positions')
                selected_shape=[shape[0],shape[1],1,shape[3]]
            else:
                require(kind=='active_value_suffix' and set(selection)=={'kind','axis','length'},'Selection kind')
                length=selection['length'];require(type(length) is int and 0<length<=shape[2],'Invalid active suffix')
                selected_shape=[shape[0],shape[1],length,shape[3]]
        size=math.prod(selected_shape)*value.element_size()
        require(size<=self.max_tensor_bytes and self.bytes+size<=self.max_call_bytes,'Capture byte cap')
        self.bytes+=size
        if selection is None:cloned=value.detach().clone(memory_format=torch.contiguous_format)
        elif selection['kind']=='prediction_query_rows':
            cloned=torch.stack([value[b,:,pos:pos+1,:] for b,pos in enumerate(selection['positions'])],dim=0).detach()
        else:cloned=value[:,:, -selection['length']:,:].detach().clone(memory_format=torch.contiguous_format)
        require(list(cloned.shape)==selected_shape and cloned.is_contiguous(),'Snapshot layout differs')
        require(version_counter(value)==version and list(value.shape)==shape and list(value.stride())==stride and value.storage_offset()==offset,'Capture changed original tensor')
        metadata=dict(source_shape=shape,source_stride=stride,source_offset=offset,source_dtype=str(value.dtype),
            source_numel=original_numel,selected_shape=list(cloned.shape),selected_bytes=size,selection=selection,
            cpu=True,owned_clone=True,original_version=version,version_counter_available=version is not None,context=context)
        reference=self.writer(cloned,metadata)
        require(isinstance(reference,dict),'Writer must return metadata reference')
        require(len(json.dumps(reference,allow_nan=False))<=65536,'Writer metadata reference cap')
        return dict(metadata=metadata,reference=reference)

    def value(self,value,selection,context,depth=0):
        require(depth<=8,'Nested capture depth')
        if isinstance(value,self.torch.Tensor):return self.tensor(value,selection,context)
        require(selection is None,'Selection on nontensor')
        if value is None or type(value) in (bool,int,float,str):return value
        if isinstance(value,(tuple,list)):
            require(len(value)<=32,'Capture sequence too large')
            return [self.value(v,None,context,depth+1) for v in value]
        if isinstance(value,dict):
            require(len(value)<=64 and all(type(k) is str for k in value),'Capture mapping schema')
            return {k:self.value(v,None,context,depth+1) for k,v in value.items()}
        raise RuntimeError('Unsupported capture value type: '+type(value).__name__)

    def __call__(self,label,value,selection,context):
        require(len(self.records)<64,'Capture event cap')
        record=dict(label=label,context=context,value=self.value(value,selection,context))
        self.records.append(record)
