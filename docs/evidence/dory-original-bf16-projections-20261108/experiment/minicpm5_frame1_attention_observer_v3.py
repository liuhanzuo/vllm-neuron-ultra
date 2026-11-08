"""Full original frame1 attention boundaries; no native imports or execution at import.

Observe exact original Python calls/returns with sys.setprofile and tensor
module hooks. No model method or arithmetic function is replaced. Actual model
use requires a separate admitted serial worker and original upstream anchors.
"""
import ast,hashlib,math,os,struct,sys,threading,types
from pathlib import Path
from contextlib import contextmanager
import minicpm5_frame1_upstream_worker_v1 as old
import minicpm5_attention_k_consumer_layout_v1 as k_layout
require,same=old.require,old.same
LENGTH,BUCKET=old.LENGTH,old.BUCKET
FIELDS=('normalized_input','position_ids','q_projection','k_projection','v_projection',
        'q_rope_input','k_rope_input','q_post_rope','k_post_rope','v_heads',
        'rope_cos','rope_sin','attention_mask','context_heads','context_before_o','projection_output')
FILE_CAP=16*2**20
OUTPUT_CAP=384*2**20
UPSTREAM=({'path': '/home/ubuntu/experiments/trn1-latest-models-20261031/minicpm5-frame1-upstream-r1-20261108-v1/native/capture.json', 'sha256': '40418292e270997d077028405c470e7f4fc8839019744e95e7c7c89aa3ae8fc2'}, {'path': '/home/ubuntu/experiments/trn1-latest-models-20261031/minicpm5-frame1-upstream-r1-20261108-v1/cpu-rank0/capture.json', 'sha256': '9520927adc3544137b02d297bb1250c349fd3e1982177bf8aff2448744641d2c'}, {'path': '/home/ubuntu/experiments/trn1-latest-models-20261031/minicpm5-frame1-upstream-r1-20261108-v1/cpu-rank1/capture.json', 'sha256': 'd032f15fd76814fb72f8b365602409e0b7a4b78b787243bf33b8400c1c150483'})

def geometry(native):
    n=LENGTH if native else BUCKET;heads=16 if native else 8;kv=2 if native else 1
    shapes=dict(normalized_input=[1,n,2048],position_ids=[1,n],
      q_projection=[1,n,heads*128],k_projection=[1,n,kv*128],v_projection=[1,n,kv*128],
      q_rope_input=[1,heads,n,128],k_rope_input=[1,kv,n,128],
      q_post_rope=[1,heads,n,128],k_post_rope=[1,kv,n,128],v_heads=[1,kv,n,128],
      rope_cos=[1,n,128],rope_sin=[1,n,128],attention_mask=[1,1,n,n],
      context_heads=[1,n,heads,128]if native else[1,heads,n,128],
      context_before_o=[1,n,heads*128],projection_output=[1,n,2048])
    return {name:dict(shape=shapes[name],dtype='torch.int64'if name=='position_ids'else
                     'torch.bool'if name=='attention_mask'and not native else'torch.bfloat16')
            for name in FIELDS}

def capture_bytes(native):
    return sum(math.prod(x['shape'])*({'torch.int64':8,'torch.bool':1}.get(x['dtype'],2))for x in geometry(native).values())

def layout(shape,stride,offset):
    require(2<=len(shape)<=4 and len(shape)==len(stride)and all(type(x)is int and x>0 for x in shape),'Finite full tensor shape')
    require(type(offset)is int and 0<=offset<8*2**20 and all(type(x)is int and x>=0 for x in stride),'Nonnegative original layout')
    span=1
    for step,size in sorted((step,size)for step,size in zip(stride,shape)if size>1):
        require(step>=span,'Nonoverlapping original axes');span+=step*(size-1)
    extent=offset+sum((s-1)*t for s,t in zip(shape,stride))+1
    require(extent<=16*2**20,'Bounded original backing extent')
    return extent

def source_code(text,path,qualname):
    code=compile(ast.parse(text,filename=path),path,'exec',dont_inherit=True)
    for name in qualname.split('.'):
        options=[x for x in code.co_consts if isinstance(x,types.CodeType)and x.co_name==name]
        require(len(options)==1,'Unique original source callable: '+qualname);code=options[0]
    return code

def authenticated_function(fn,path,wanted,qualname):
    path=Path(path);raw=path.read_bytes()
    require(len(raw)<8*2**20 and hashlib.sha256(raw).hexdigest()==wanted,'Exact admitted attention source')
    function=fn.__func__ if isinstance(fn,types.MethodType)else fn
    require(isinstance(function,types.FunctionType)and function.__closure__ is None,'Original unwrapped Python function')
    require(function.__code__==source_code(raw.decode(),str(path),qualname)and function.__code__.co_filename==str(path),'Actual original attention code and filename')
    return function

class Calls:
    """Exact owner/caller scope and one balanced invocation of each boundary."""
    def __init__(self,bindings,callback):
        require(set(bindings)=={'outer','rope','attention'},'All original attention call boundaries')
        self.bindings=bindings;self.callback=callback
        self.codes={row['function'].__code__:role for role,row in bindings.items()}
        require(len(self.codes)==3,'Three distinct original function codes')
        self.live={};self.counts={role:dict(calls=0,returns=0)for role in bindings};self.outer=None;self.thread=threading.get_ident()
    def __call__(self,frame,event,arg):
        if event not in ('call','return')or frame.f_code not in self.codes:return
        require(threading.get_ident()==self.thread,'Original Python observer thread')
        role=self.codes[frame.f_code];row=self.bindings[role]
        if event=='call':
            owns=row['owner_key']is None or frame.f_locals.get(row['owner_key'])is row['owner']
            if role=='outer':
                if not owns:return
                require(self.outer is None and self.counts[role]['calls']==0,'Exactly one original layer0 attention')
                self.outer=frame
            else:
                if self.outer is None:return
                require(owns,'Original nested attention owner')
                caller=frame.f_back
                require(caller is not None and caller.f_code is row['caller_code']and
                        caller.f_locals.get(row['caller_owner_key'])is row['owner'],'Actual original immediate caller')
                require(self.counts[role]['calls']==0,'Exactly one original internal call')
            require(frame not in self.live,'No recursive observed frame')
            self.live[frame]=role;self.counts[role]['calls']+=1
            self.callback(role,event,frame.f_locals,arg)
        elif frame in self.live:
            require(self.live[frame]==role and arg is not None,'Original successful matching return frame')
            self.callback(role,event,frame.f_locals,arg)
            self.counts[role]['returns']+=1;del self.live[frame]
            if role=='outer':self.outer=None
    def complete(self):
        require(not self.live and self.outer is None and all(x==dict(calls=1,returns=1)for x in self.counts.values()),'Every original call/return observed exactly once')
        return dict(self.counts)

@contextmanager
def profile(calls):
    previous=sys.getprofile();require(previous is None,'No existing Python profiler may be replaced')
    sys.setprofile(calls)
    try:
        yield calls
        require(sys.getprofile()is calls,'Observer identity retained')
        calls.complete()
    finally:
        current=sys.getprofile();sys.setprofile(previous)
        require(current in (calls,None),'Unexpected profiler replacement')

def save(torch,directory,name,value,native,guard):
    guard();expected=geometry(native)[name]
    same(list(value.shape),expected['shape'],'Complete original tensor shape: '+name)
    same(str(value.dtype),expected['dtype'],'Original tensor dtype: '+name)
    require(value.device.type=='cpu','CPU-only observed tensor')
    meta=old.metadata(torch,value);extent=layout(meta['shape'],meta['stride'],meta['offset'])
    if name!='attention_mask':require(bool(torch.isfinite(value).all()),'Finite original observation')
    else:require(not bool(torch.isnan(value).any()),'No NaN attention mask')
    raw=old.raw(torch,value);require(len(raw)<=FILE_CAP,'Complete capture file cap')
    path=Path(directory)/(name+'.raw')
    with path.open('xb')as stream:require(stream.write(raw)==len(raw),'Full actual tensor bytes');stream.flush();os.fsync(stream.fileno())
    same(old.metadata(torch,value),meta,'Observation leaves original tensor unchanged')
    return dict(name=name,metadata=meta,path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),identity=old.identity(path),
                original_backing_elements=extent,encoding='packed_logical_little_endian',
                observed_not_recomputed=True,unaddressed_backing_storage_captured=False)

def verify_upstream(context,values,native,rank):
    require(type(native)is bool and type(rank)is int and ((native and rank==-1)or(not native and rank in (0,1))),'Exact original observation role')
    ref=UPSTREAM[0 if native else rank+1];record=context.read_json(ref['path'],ref['sha256'])
    require(record['schema']=='minicpm5_frame1_upstream_capture_v1'and record['complete']is True,'Actual completed upstream reference')
    same(record['frame'],old.FRAME,'Same full frame1')
    require(record['native']is native and record['rank']==rank,'Same original observation role')
    same([r['name']for r in record['captures']],list(old.FIELDS),'Exactly seven original ordered captures')
    require(set(values)==set(old.FIELDS),'Exactly seven current boundaries')
    same({r['name']:r['metadata']for r in record['captures']},{k:v['metadata']for k,v in values.items()},'All seven full upstream boundaries and layouts unchanged')
    return dict(reference=ref,all7_full_boundaries_exact=True,whole_model_endpoint_claimed=False)

def verify_cpu_output_projection(torch,oproj):
    require(type(oproj).__name__=='RowParallelLinear','Original SDK row-parallel output projection')
    require(oproj.input_is_parallel is True and oproj.reduce_output is True and oproj.reduce_dtype==torch.bfloat16 and oproj.tensor_parallel_group.size()==2,'Original BF16 TP2 output reduction')
    require(oproj.pad is False and oproj.sequence_parallel_enabled is False,'Original unpadded ordinary output projection')
    same((oproj.input_size,oproj.output_size,oproj.input_size_per_partition),(2048,2048,1024),'Original exact output projection sizes')
    same(list(oproj.weight.shape),[2048,1024],'Original exact unpadded output weight shard')
    require(oproj.bias is None and oproj.weight.dtype==torch.bfloat16,'Original bias-free BF16 output matrix')
    return True

class Observer:
    def __init__(self,torch,attention,module,native,rank,plan,directory,guard):
        self.torch,self.attention,self.native,self.rank,self.plan,self.directory,self.guard=torch,attention,native,rank,plan,Path(directory),guard
        self.values={};self.k_consumer_boundary=None;self.handles=[];self.context_flat_hash=None;self.dispatch=[];self.function_dispatch=[];self.closed=False
        self.source=Path(plan['native_provenance']['source']if native else plan['sdk_attention_source'])
        require(module.__file__==str(self.source),'Actual original attention module path')
        wanted=plan['source_sha256'][str(self.source)]
        def auth(fn,name):
            function=authenticated_function(fn,self.source,wanted,name)
            require(function.__globals__ is vars(module),'Original attention module globals')
            owner=fn.__self__ if isinstance(fn,types.MethodType)else module
            require(name.split('.')[-1]not in getattr(owner,'__dict__',{})or owner is module,'No instance function replacement')
            self.function_dispatch.append((owner,name.split('.')[-1],fn))
            return function
        if native:
            require(attention.layer_idx==0 and attention.config._attn_implementation=='eager'and not attention.training and attention.head_dim==128 and attention.scaling==128**-.5 and attention.num_key_value_groups==8,'Original native eager attention configuration')
            outer=auth(attention.forward,'LlamaAttention.forward')
            rope=auth(module.apply_rotary_pos_emb,'apply_rotary_pos_emb')
            compute=auth(module.eager_attention_forward,'eager_attention_forward')
            bindings=dict(outer=dict(function=outer,owner_key='self',owner=attention),rope=dict(function=rope,owner_key=None,owner=attention,caller_code=outer.__code__,caller_owner_key='self'),attention=dict(function=compute,owner_key='module',owner=attention,caller_code=outer.__code__,caller_owner_key='self'))
            projections={k:getattr(attention,k+'_proj')for k in ('q','k','v')};oproj=attention.o_proj
        else:
            require(rank in (0,1)and attention.head_dim==128 and attention.num_heads==8 and attention.num_key_value_heads==1 and attention.cp_degree==1 and not attention.use_qk_norm and not attention.flash_decoding_enabled,'Original CPU TP2 ordinary attention')
            qkv=attention.qkv_proj
            require(qkv.fused_qkv is False and qkv.qkv_kernel_enabled is False and qkv.qkv_nki_kernel_enabled is False and qkv.clip_qkv is None,'Original separate unfused unclipped projections')
            outer=auth(attention.standard_causal_attention_forward,'NeuronAttentionBase.standard_causal_attention_forward')
            rope=auth(attention.apply_rotary_embedding,'NeuronAttentionBase.apply_rotary_embedding')
            compute=auth(attention.perform_prefill,'NeuronAttentionBase.perform_prefill')
            prep=auth(attention.prep_qkv_tensors,'NeuronAttentionBase.prep_qkv_tensors')
            context=auth(attention.attention_context_encode,'NeuronAttentionBase.attention_context_encode')
            bindings=dict(outer=dict(function=outer,owner_key='self',owner=attention),rope=dict(function=rope,owner_key='self',owner=attention,caller_code=prep.__code__,caller_owner_key='self'),attention=dict(function=compute,owner_key='self',owner=attention,caller_code=context.__code__,caller_owner_key='self'))
            projections={k:getattr(qkv,k+'_proj')for k in ('q','k','v')};oproj=attention.o_proj.o_proj
            require(attention.o_proj.out_proj_kernel_enabled is False,'Original non-kernel output projection')
            verify_cpu_output_projection(torch,oproj)
        self.calls=Calls(bindings,self.event)
        try:
            for label,obj in projections.items():
                require('forward'not in obj.__dict__ and obj.bias is None,'Original bias-free projection forward')
                self.dispatch.append((obj,obj.forward))
                def pre(m,args,label=label):
                    require(self.calls.outer is not None and len(args)==1,'Original active projection input')
                    same(old.metadata(torch,args[0]),self.values['normalized_input']['metadata'],'All projection inputs preserve original full layout')
                def post(m,args,output,label=label):self.emit(label+'_projection',output)
                self.handles.append(obj.register_forward_pre_hook(pre));self.handles.append(obj.register_forward_hook(post))
            require('forward'not in oproj.__dict__ and oproj.bias is None,'Original output projection forward')
            self.dispatch.append((oproj,oproj.forward))
            def before_o(m,args):
                require(len(args)==1 and self.context_flat_hash is not None,'Original complete context precedes projection')
                self.emit('context_before_o',args[0]);same(self.values['context_before_o']['sha256'],self.context_flat_hash,'Original context reshape bytes')
            self.handles.append(oproj.register_forward_pre_hook(before_o))
            self.handles.append(oproj.register_forward_hook(lambda m,a,o:self.emit('projection_output',o)))
        except BaseException:self.close();raise
    def emit(self,name,value):
        require(name in FIELDS and name not in self.values,'Single actual observation: '+name)
        require(not self.closed and self.calls.outer is not None,'Only active original attention capture')
        self.values[name]=save(self.torch,self.directory,name,value,self.native,self.guard)
    def same_tensor(self,name,value):
        same(old.metadata(self.torch,value),self.values[name]['metadata'],'Exact observed tensor/layout through boundary '+name)
    def observe_k_consumer(self,value):
        require(self.k_consumer_boundary is None,'Exactly one original K consumer boundary')
        self.guard()
        producer=self.values['k_post_rope']['metadata'];consumer=old.metadata(self.torch,value)
        boundary=k_layout.compare(producer,consumer,self.native)
        # Keep actual before/after metadata even when the equality gate fails.
        reference=old.once(self.directory/'k-consumer-boundary.json',boundary)
        k_layout.validate(boundary,self.native)
        self.k_consumer_boundary=dict(record=boundary,reference=reference)
    def event(self,role,event,args,result):
        if role=='outer':
            if event=='call':
                self.emit('normalized_input',args['hidden_states'])
                positions=args['kwargs']['position_ids']if self.native else args['position_ids']
                self.emit('position_ids',positions)
                if not self.native:
                    require(args['past_key_value']is None and args['residual']is None and not args['get_kv_per_layer']and not args['update_kv_per_layer'],'Original fresh unfused CPU prefill')
            else:
                if self.native:
                    require(type(result)is tuple and len(result)==2,'Original native attention return');out=result[0]
                else:
                    require(type(result).__name__=='NeuronAttentionBaseOutput'and result.residual is None,'Original SDK return');out=result.hidden_states
                self.same_tensor('projection_output',out)
        elif role=='rope':
            if event=='call':
                self.emit('q_rope_input',args['q'if self.native else'Q']);self.emit('k_rope_input',args['k'if self.native else'K'])
                if self.native:
                    require(args['unsqueeze_dim']==1,'Original rotary broadcast dimension')
                    self.emit('rope_cos',args['cos']);self.emit('rope_sin',args['sin'])
                else:
                    require(args['use_polar_compatible_rope']is False,'Original ordinary rotary route')
                    self.same_tensor('position_ids',args['position_ids'])
                    self.emit('v_heads',args['V'])
            else:
                require(type(result)is tuple and len(result)==(2 if self.native else 4),'Original complete rotary return')
                self.emit('q_post_rope',result[0]);self.emit('k_post_rope',result[1])
                if not self.native:self.emit('rope_cos',result[2]);self.emit('rope_sin',result[3])
        else:
            if event=='call':
                self.same_tensor('q_post_rope',args['query'if self.native else'Q']);self.observe_k_consumer(args['key'if self.native else'K'])
                if self.native:
                    require(args['dropout']==0.0 and args['scaling']==128**-.5,'Original eager arithmetic parameters')
                    self.emit('v_heads',args['value'])
                else:
                    require(args['q_len']==BUCKET and args['bsz']==1,'Original full CPU physical geometry')
                    self.same_tensor('v_heads',args['V'])
                self.emit('attention_mask',args['attention_mask'])
            else:
                require(type(result)is tuple and len(result)==2,'Original attention computation pair')
                if not self.native:require(result[1].name=='NONE','Original non-flash attention route')
                value=result[0];self.emit('context_heads',value)
                folded=value if self.native else value.transpose(1,2)
                self.context_flat_hash=old.metadata(self.torch,folded.reshape(1,LENGTH if self.native else BUCKET,-1))['sha256']
    def summary(self):
        self.calls.complete();require(set(self.values)==set(FIELDS),'Every complete full-prefix attention boundary')
        require(sum(x['bytes']for x in self.values.values())==capture_bytes(self.native),'Exact all-boundary byte count')
        self.check_dispatch();require(self.k_consumer_boundary is not None,'Actual K consumer boundary retained');k_layout.validate(self.k_consumer_boundary['record'],self.native)
        return dict(schema='minicpm5_frame1_attention_observation_v1',complete=True,native=self.native,rank=self.rank,frame=old.FRAME,
                    source=dict(path=str(self.source),sha256=self.plan['source_sha256'][str(self.source)]),
                    captures=[self.values[k]for k in FIELDS],calls=self.calls.counts,k_consumer_boundary=self.k_consumer_boundary,
                    instrumentation='original_python_call_return_profile_and_module_hooks',
                    original_functions_replaced=False,attention_arithmetic_recomputed=False,
                    full_model_qualified=False,new_speedup_claim=False)
    def check_dispatch(self):
        for obj,fn in self.dispatch:require('forward'not in obj.__dict__ and obj.forward==fn,'Projection dispatch unchanged')
        for owner,name,fn in self.function_dispatch:require(getattr(owner,name)==fn,'Original attention callable retained')
    def close(self):
        self.closed=True
        handles,self.handles=self.handles,[]
        old.phase.base.cleanup_actions([h.remove for h in handles],'attention observer hook cleanup')

def restore(torch,row,directory,native,guard):
    guard();name=row['name'];expected=geometry(native)[name];path=Path(row['path'])
    require(path==Path(directory)/(name+'.raw')and path.resolve()==path,'Exact saved attention payload path')
    same(old.identity(path),row['identity'],'Original capture file identity')
    same(row['metadata']['shape'],expected['shape'],'Original full replay shape')
    same(row['metadata']['dtype'],expected['dtype'],'Original replay dtype')
    raw=path.read_bytes()
    require(len(raw)==row['bytes']<=FILE_CAP and hashlib.sha256(raw).hexdigest()==row['sha256']==row['metadata']['sha256'],'All observed attention bytes')
    dtype=getattr(torch,expected['dtype'].split('.')[1]);meta=row['metadata'];extent=layout(meta['shape'],meta['stride'],meta['offset'])
    packed=torch.frombuffer(bytearray(raw),dtype=dtype).reshape(meta['shape'])
    storage=torch.empty(extent,dtype=dtype)
    value=storage.as_strided(meta['shape'],meta['stride'],meta['offset']);value.copy_(packed)
    same(old.metadata(torch,value),meta,'Original full shape/stride/offset/dtype/bytes restored')
    same(old.identity(path),row['identity'],'Capture unchanged through restore')
    return value

CONTROL_FIELDS=('q_projection','k_projection','v_projection','q_post_rope','k_post_rope','context_heads','projection_output')
def own_input_controls(torch,attention,module,native,rank,report,directory,guard):
    """Replay each original operator on its own complete saved input, no interventions."""
    require(report['complete']is True and report['native']is native and report['rank']==rank,'Actual full observations precede controls')
    require(sys.getprofile()is None,'Capture profiler removed before own-input replay')
    directory=Path(directory);control_dir=directory/'controls';control_dir.mkdir(exist_ok=False)
    rows={x['name']:x for x in report['captures']}
    same([x['name']for x in report['captures']],list(FIELDS),'All original observed inputs retained exactly once')
    require(torch.is_inference_mode_enabled()and torch.get_num_threads()==(2 if native else 1),'Original inference mode and threads')
    inputs={name:restore(torch,row,directory,native,guard)for name,row in rows.items()}
    boundary=report['k_consumer_boundary']['record'];consumer_row=k_layout.replay_row(rows['k_post_rope'],boundary,native)
    consumer_k=restore(torch,consumer_row,directory,native,guard)
    controls={}
    projections=attention if native else attention.qkv_proj
    def check(name,value):
        row=save(torch,control_dir,name,value,native,guard)
        row.update(observed_not_recomputed=False,own_input_recomputed=True)
        target=rows[name]['metadata'];actual=row['metadata']
        controls[name]=dict(saved=row,raw_exact=actual['sha256']==target['sha256'],layout_exact=all(actual[k]==target[k]for k in ('shape','stride','offset','dtype')))
    for name in ('q','k','v'):
        check(name+'_projection',getattr(projections,name+'_proj')(inputs['normalized_input']))
    if native:
        q,k=module.apply_rotary_pos_emb(inputs['q_rope_input'],inputs['k_rope_input'],inputs['rope_cos'],inputs['rope_sin'])
    else:
        q,k,cos,sin=attention.apply_rotary_embedding(inputs['q_rope_input'],inputs['k_rope_input'],inputs['v_heads'],inputs['position_ids'],inputs['rope_cos'],inputs['rope_sin'],False)
        same(old.metadata(torch,cos),rows['rope_cos']['metadata'],'Original cosine retained')
        same(old.metadata(torch,sin),rows['rope_sin']['metadata'],'Original sine retained')
    check('q_post_rope',q);check('k_post_rope',k)
    if native:
        output,_=module.eager_attention_forward(attention,inputs['q_post_rope'],consumer_k,inputs['v_heads'],inputs['attention_mask'],scaling=128**-.5,dropout=0.0)
    else:
        output,strategy=attention.perform_prefill(inputs['q_post_rope'],consumer_k,inputs['v_heads'],BUCKET,1,inputs['attention_mask'])
        require(strategy.name=='NONE','Original non-flash replay')
    check('context_heads',output)
    out=attention.o_proj(inputs['context_before_o'])if native else attention.o_proj.o_proj(inputs['context_before_o'])
    check('projection_output',out)
    for name,value in inputs.items():same(old.metadata(torch,value),rows[name]['metadata'],'Original own-input tensor unchanged')
    same(old.metadata(torch,consumer_k),boundary['consumer'],'Exact original attention K consumer layout and values retained')
    same(tuple(controls),CONTROL_FIELDS,'All original operator controls')
    return dict(complete=True,passed=controls_passed(controls),controls=controls,k_consumer_boundary=boundary,
                original_operators=True,original_interpreter_and_thread_settings=True,
                cpu_output_reduction_uses_original_gloo=not native,independently_recomputed_trigonometry=False,
                precision_interventions=[],whole_model_endpoint_claimed=False,new_speedup_claim=False)

def science():
    controls=sum(math.prod(geometry(native)[k]['shape'])*2 for native in (True,False,False)for k in CONTROL_FIELDS)
    captures=capture_bytes(True)+2*capture_bytes(False)
    return dict(frame=old.FRAME,native_length=LENGTH,cpu_bucket=BUCKET,fields=list(FIELDS),
                geometry={kind:geometry(kind=='native')for kind in ('native','cpu')},
                own_input_control_fields=list(CONTROL_FIELDS),raw_capture_bytes=captures,raw_control_bytes=controls,
                original_upstream_capture_bytes=old.science()['actual_capture_bytes'],
                total_raw_bytes=captures+controls+old.science()['actual_capture_bytes']+old.science()['norm_weight_bytes'],
                original_upstream_anchors=list(UPSTREAM),output_cap_bytes=OUTPUT_CAP,
                file_cap_bytes=FILE_CAP,profiled_original_calls=['outer','rope','attention'],
                original_function_dispatch_retained=True,precision_interventions=[],full_model_qualified=False,new_speedup_claim=False)


def controls_passed(controls):
    same(tuple(controls),CONTROL_FIELDS,'Exactly seven saved own-input controls')
    for value in controls.values():
        require(all(type(value.get(k))is bool for k in ('raw_exact','layout_exact')),'Actual boolean control outcomes')
    return all(value['raw_exact']and value['layout_exact']for value in controls.values())


def projection_parameters(attention,native):
    projections=attention if native else attention.qkv_proj
    return {**{name:getattr(projections,name+'_proj').weight for name in ('q','k','v')},
            'o':(attention.o_proj if native else attention.o_proj.o_proj).weight}


def verify_native_projection_weights(torch,attention,context):
    """Check each actual native Q/K/V/O matrix against its bounded official interval."""
    ref=context.plan['official_layer0_weights'];record=context.verify_input(ref['path'])
    header=context.read_json(context.plan['safetensors_header']['path'],context.plan['safetensors_header']['sha256'])
    fd=os.open(ref['path'],os.O_RDONLY|os.O_NOFOLLOW);rows={}
    def unchanged():
        info=os.fstat(fd)
        same([info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns],
             [record['identity'][k]for k in ('device','inode','size','mtime_ns','ctime_ns')],'Opened exact official native checkpoint')
    try:
        unchanged();prefix=os.pread(fd,8,0);require(len(prefix)==8,'Official header prefix')
        hlen,=struct.unpack('<Q',prefix);require(0<hlen<=2**20,'Bounded native header')
        same(__import__('json').loads(os.pread(fd,hlen,8)),header,'Exact native official header')
        for name,value in projection_parameters(attention,True).items():
            context.guard();key='model.layers.0.self_attn.'+name+'_proj.weight';row=header[key]
            same(row,ref['tensors'][key],'Original native projection interval')
            shape=[256,2048]if name in ('k','v')else[2048,2048]
            same(row['shape'],shape,'Original native projection shape');same(row['dtype'],'BF16','Original native projection dtype')
            same(list(value.shape),shape,'Actual native full matrix');require(value.dtype==torch.bfloat16,'Actual native BF16 matrix')
            begin,end=row['data_offsets'];size=math.prod(shape)*2
            require(end-begin==size<=8*2**20 and 0<=begin<end<=record['identity']['size']-8-hlen,'Bounded native matrix interval')
            raw=os.pread(fd,size,8+hlen+begin);require(len(raw)==size,'Complete native projection interval')
            old.same_bytes(old.raw(torch,value),raw,'Actual native projection equals original checkpoint')
            rows[name]=dict(official_key=key,metadata=old.metadata(torch,value),bytes=size)
        unchanged()
    finally:os.close(fd)
    same(old.identity(ref['path']),record['identity'],'Official checkpoint unchanged')
    return dict(passed=True,weights=rows,all_original_full_matrices=True,maximum_interval_bytes=8*2**20)
