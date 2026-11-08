"""Strict frame1 upstream attention/residual/postnorm observations; no model execution at import.

Runtime requires a separately reviewed serial runner and fresh admission.
"""
import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import struct
import sys
import types

from minicpm5_layer0_runtime_auth_v1 import identity, require
import minicpm5_original32_layer0_phase_cpu_worker_v1 as phase
from minicpm5_history_tensors_v1 import PredictionHooks

ROOT = Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
MIB = 2**20
TAPS = ('embed_tokens', 'layers.0.input_layernorm', 'layers.0.self_attn',
        'layers.0.post_attention_layernorm', 'layers.0.mlp', 'layers.0')
FIELDS = ('layer_input', 'inputnorm_output', 'attention_input', 'attention_output',
          'postnorm_input', 'postnorm_output', 'mlp_input')
ARM = 'strict_anchor_s0'
FRAME = dict(index=1, fixture=7, slot=1, step=0)
LENGTH, BUCKET = 1231, 2048
CAPTURE_FILE_CAP = 16*MIB
OUTPUT_CAP = 192*MIB
same = phase.same
ACTIVATIONS = dict(native=str(ROOT/'research/transformers-reference/src/transformers/activations.py'),
                   cpu='/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/transformers/activations.py')


def same_bytes(left,right,message):
    require(type(left)is bytes and type(right)is bytes,'Exact raw byte operands')
    require(left==right,message)


def activation_source(text):
    tree=ast.parse(text)
    classes=[n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='SiLUActivation']
    require(len(classes)==1,'One actual SiLUActivation source class')
    forward=[n for n in classes[0].body if isinstance(n,ast.FunctionDef)and n.name=='forward']
    expected_decorator=ast.parse('@use_kernel_forward_from_hub("SiLU")\nclass C: pass').body[0].decorator_list
    same([ast.dump(n)for n in classes[0].decorator_list],[ast.dump(n)for n in expected_decorator],
         'Original SiLU kernel-dispatch decorator')
    expected=ast.parse('def forward(self, input):\n    return nn.functional.silu(input)').body[0].body
    require(len(forward)==1 and ast.dump(ast.Module(body=forward[0].body,type_ignores=[]))==
            ast.dump(ast.Module(body=expected,type_ignores=[])),'Original functional SiLU body, no replacement')
    mappings=[(k.value,v.id)for node in ast.walk(tree)if isinstance(node,ast.Dict)
              for k,v in zip(node.keys,node.values)if isinstance(k,ast.Constant)and k.value=='silu'and isinstance(v,ast.Name)]
    same(mappings,[('silu','SiLUActivation')],'Actual ACT2CLS silu mapping')
    return True


def activation_dispatch(torch,act,cls,text,path):
    """Prove the active original fallback; never assume an nn.SiLU instance."""
    activation_source(text)
    require(type(act)is cls and 'forward'not in act.__dict__,'Original SiLUActivation instance')
    forward=act.forward
    require(isinstance(forward,types.MethodType)and forward.__self__ is act and
            forward.__func__ is cls.forward,'Original bound forward dispatch')
    tree=ast.parse(text,filename=path)
    code=compile(tree,path,'exec',dont_inherit=True)
    class_code=next(x for x in code.co_consts if isinstance(x,types.CodeType)and x.co_name=='SiLUActivation')
    source_code=next(x for x in class_code.co_consts if isinstance(x,types.CodeType)and x.co_name=='forward')
    function=forward.__func__
    require(function.__code__==source_code and function.__code__.co_filename==path and
            function.__closure__ is None and function.__defaults__ is None and
            function.__globals__.get('nn')is torch.nn,'Active dispatch is pinned functional SiLU fallback')
    return dict(class_name=cls.__name__,source=path,forward_line=source_code.co_firstlineno,
                dispatch='original_functional_silu_fallback',runtime_code_exact=True)


def once(path, value):
    raw = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
    require(len(raw) < 2*MIB, 'Bounded metadata')
    with Path(path).open('xb') as f:
        require(f.write(raw) == len(raw), 'Complete metadata write')
        f.flush();os.fsync(f.fileno())
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())


def raw(torch, value):
    require(value.device.type == 'cpu' and value.numel()*value.element_size() <= 24*MIB,
            'Bounded CPU tensor')
    return bytes(memoryview(value.detach().contiguous().reshape(-1).view(torch.uint8).numpy()))


def metadata(torch, value):
    return dict(shape=list(value.shape), stride=list(value.stride()), offset=value.storage_offset(),
                dtype=str(value.dtype), sha256=hashlib.sha256(raw(torch, value)).hexdigest())


def checked_layout(shape, stride, offset):
    require(len(shape) == len(stride) == 3 and shape[0] == 1 and shape[1] in (LENGTH,BUCKET)
            and shape[2] in (2048,3072,6144), 'Finite tensor geometry')
    require(type(offset) is int and offset == 0 and all(type(s) is int and s > 0 for s in stride),
            'Finite positive layout with zero offset')
    # Singleton dimensions need no stride restriction. Other axes must form a
    # non-overlapping dense layout; allow the actual SDK feature-major input.
    axes = sorted((stride[i], shape[i]) for i in range(3) if shape[i] > 1)
    span = 1
    for step, size in axes:
        require(step == span, 'Dense non-overlapping tensor layout')
        span *= size
    return span


def expected_shapes(native):
    return {k:[1,LENGTH if native else BUCKET,2048] for k in FIELDS}



CPU_ANCHORS = tuple(dict(path=str(ROOT/f'minicpm5-original32-layer0-phase-cpu-r1-20261107-v1/cpu-rank{rank}/strict_anchor_s0/call-01/result.json'),sha256=sha)
    for rank,sha in enumerate(('04a6334a2950b16f9fb4c8dc3870603584b5c6e0da010bc8fab7583c98c7ffc0',
                              'd1cac9b1a0f1c6095a4dd6a885a782251da9cab7bed3d0f254e14201aa357d02')))
FULL_CPU_ANCHORS = tuple(dict(path=str(ROOT/f'minicpm5-original32-full-down-cpu-r1-20261107/cpu-rank{rank}/strict_anchor_s0/call-01/result.json'),sha256=sha)
    for rank,sha in enumerate(('9141f970060fb3ba0857b482c4f79505e72da48734cfae763ceb473e8684c74a',
                              '6e315aa0a88a7d52fb87f362a882e50b06d8089d92e5d6d6113b621f54b18523')))
FULL_FIELDS = dict(mlp_input='postnorm')
R3_CAPTURES = tuple(dict(path=str(ROOT/('minicpm5-frame1-mlp-r3-20261108-v1/'+part+'/capture.json')),sha256=sha) for part,sha in [('native','296c9a721f17c3118d18cb3cc9f929e648eb9d673c29e69aedf227b192d16348'),('cpu-rank0','38c8ba9a5e11ec366ab570e908f0b56e9a6e8fb389cd142cb7637ba94c7a0ca3'),('cpu-rank1','9f18c3163775cdbab4cb5789758d61f63472b0b3aea2dfd370e3d123d7908c45')])
NORM_NAMES=('input_layernorm','post_attention_layernorm')
NORM_SOURCES=dict(native=ROOT/'research/transformers-reference/src/transformers/models/llama/modeling_llama.py',cpu=Path('/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py'))


def capture_bytes(native):
    return sum(2*shape[0]*shape[1]*shape[2] for shape in expected_shapes(native).values())


def science():
    return dict(frame=dict(FRAME),capture_fields=list(FIELDS),cpu_anchors=[dict(x) for x in CPU_ANCHORS],
                cpu_full_anchors=[dict(x) for x in FULL_CPU_ANCHORS],r3_captures=[dict(x)for x in R3_CAPTURES],arm=ARM,precision_interventions=[],norm_names=list(NORM_NAMES),norm_eps=1e-6,norm_weight_bytes=3*2*4096,
                native_length=LENGTH,cpu_bucket=BUCKET,actual_capture_bytes=capture_bytes(True)+2*capture_bytes(False),
                capture_file_cap_bytes=CAPTURE_FILE_CAP,output_cap_bytes=OUTPUT_CAP,upstream_only=True)


def validate_science(plan):
    spec=plan['frame1_upstream'];same(spec,science(),'Finite strict upstream frame1 science')
    graph=plan['graph_calls'][1]
    same(graph['call'],dict(index=1,group=0,phase='prefill',frames=[FRAME],logical_positions=[LENGTH-1]),'Original long prefill')
    same(graph['bucket'],BUCKET,'Original physical bucket')
    fixture=plan['fixtures'][7]
    require(fixture['id']==7 and len(fixture['input_ids'])==LENGTH,'Original complete prompt')
    for ref in (*CPU_ANCHORS,*FULL_CPU_ANCHORS,*R3_CAPTURES):
        same(plan['inputs'][ref['path']]['sha256'],ref['sha256'],'All strict anchors admitted')
    return spec


def validate_baseline(record,rank,full):
    require(record['complete'] is True and record['rank']==rank and rank in (0,1),'Actual complete rank')
    same(record['arm'],ARM,'Unmodified strict baseline only')
    same(record['call'],dict(index=1,group=0,phase='prefill',frames=[FRAME],logical_positions=[LENGTH-1]),'Exact original frame1 call')
    same(record['bucket'],BUCKET,'Original full CPU bucket')
    same(record['changed_mask_cells'],[],'No strict mask changes at this prefill')
    same(record['candidate_targets'],[],'No candidate targets')
    require(record['candidate_full_down_enabled' if full else 'candidate_layer0_phase_enabled'] is False,'No candidate intervention')
    require(all(record['strict_anchor_exact'].get(k) is True for k in
                ('all7_inputs_exact','all213_taps_exact','all84_before_input_post_exact','full_logits_exact')),
            'Recorded complete strict baseline equality')
    return True


def original_dispatch(core):
    # The SDK wraps the actual row-parallel linear in an output-projection module.
    attention=core.layers[0].self_attn
    require(hasattr(attention.o_proj,'o_proj'),'Original SDK output-projection wrapper')
    modules=[core.layers[0],attention,attention.o_proj,attention.o_proj.o_proj,
             core.layers[0].mlp,*[getattr(core.layers[0].mlp,n+'_proj') for n in ('gate','up','down')]]
    require(all('forward' not in m.__dict__ for m in modules),'No instance precision overrides')
    return [(m,m.forward) for m in modules]


def check_dispatch(core,before):
    after=original_dispatch(core)
    require(len(before)==len(after) and all(m is n and f==g for (m,f),(n,g) in zip(before,after)),
            'Original attention and MLP dispatch unchanged')


def check_full_anchor(context,values,prediction_anchor,rank,native):
    reference=R3_CAPTURES[0 if native else rank+1]
    record=context.read_json(reference['path'],reference['sha256'])
    require(record['complete']is True and record['native']is native and record['rank']==rank,'Actual original r3 capture')
    same(record['frame'],FRAME,'Original r3 frame')
    require(record['actual_prefix_calls']==1 and record['precision_interventions']==[] and record['anchor']['all_six_exact']is True,'Unmodified original r3 prefix')
    original,=[row for row in record['captures']if row['name']=='input']
    same(values['mlp_input']['metadata'],original['metadata'],'Fullshape MLP input unchanged from r3')
    same(values['postnorm_output']['metadata'],values['mlp_input']['metadata'],'Postnorm output is actual full MLP input')
    same(values['inputnorm_output']['metadata'],values['attention_input']['metadata'],'Inputnorm output is actual full attention input')
    if not native:
        full=context.read_json(FULL_CPU_ANCHORS[rank]['path'],FULL_CPU_ANCHORS[rank]['sha256'])
        validate_baseline(full,rank,True)
        same(full['stage_sha256'],prediction_anchor['stage_sha256'],'Original full prediction anchors')
        same(values['mlp_input']['metadata'],full['layer0_full_observations']['postnorm'],'Original full CPU postnorm retained')
    return dict(reference=reference,full_mlp_input_exact=True,adjacent_boundaries_exact=True,source_shape_preserved=True)


def norm_source(text):
    tree=ast.parse(text);cls,=[n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='LlamaRMSNorm']
    forward,=[n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='forward']
    expected=ast.parse("""def forward(self,hidden_states):
 input_dtype=hidden_states.dtype
 hidden_states=hidden_states.to(torch.float32)
 variance=hidden_states.pow(2).mean(-1,keepdim=True)
 hidden_states=hidden_states*torch.rsqrt(variance+self.variance_epsilon)
 return self.weight*hidden_states.to(input_dtype)
""").body[0]
    same(ast.dump(ast.Module(body=forward.body,type_ignores=[])),ast.dump(ast.Module(body=expected.body,type_ignores=[])),'Original RMSNorm arithmetic/order')
    expected_decorator=ast.parse('@use_kernel_forward_from_hub("RMSNorm")\nclass C: pass').body[0].decorator_list
    same([ast.dump(n)for n in cls.decorator_list],[ast.dump(n)for n in expected_decorator],'Original norm dispatch decorator')
    return dict(arithmetic_ast_sha256=hashlib.sha256(ast.dump(ast.Module(body=forward.body,type_ignores=[])).encode()).hexdigest(),line=forward.lineno)


def norm_dispatch(torch,core,native,context):
    path=NORM_SOURCES['native'if native else'cpu'];text=path.read_text()
    same(hashlib.sha256(text.encode()).hexdigest(),context.plan['source_sha256'][str(path)],'Pinned norm source')
    proof=norm_source(text);module=importlib.import_module('transformers.models.llama.modeling_llama')
    same(str(Path(module.__file__).resolve()),str(path),'Actual original norm module');cls=module.LlamaRMSNorm
    code=compile(ast.parse(text),str(path),'exec',dont_inherit=True)
    class_code,=[x for x in code.co_consts if isinstance(x,types.CodeType)and x.co_name=='LlamaRMSNorm']
    source_code,=[x for x in class_code.co_consts if isinstance(x,types.CodeType)and x.co_name=='forward']
    rows={}
    for name in NORM_NAMES:
        obj=getattr(core.layers[0],name);function=obj.forward
        require(type(obj)is cls and 'forward'not in obj.__dict__ and isinstance(function,types.MethodType)and function.__self__ is obj and function.__func__ is cls.forward,'Actual original bound norm dispatch')
        require(function.__func__.__code__==source_code and function.__func__.__closure__ is None and function.__func__.__globals__.get('torch')is torch,'Actual original RMSNorm fallback code')
        require(obj.variance_epsilon==1e-6 and list(obj.weight.shape)==[2048]and obj.weight.dtype==torch.bfloat16,'Original2048BF16norm/epsilon')
        rows[name]=dict(source=str(path),**proof,epsilon=obj.variance_epsilon,weight=metadata(torch,obj.weight),dispatch='original_rmsnorm_functional_fallback')
    return rows


def norm_weights(torch,core,context,directory):
    reference=next(x for x in context.plan['model_files']if x['path'].endswith('.safetensors'))
    header=context.read_json(context.plan['safetensors_header']['path'],context.plan['safetensors_header']['sha256'])
    same(identity(reference['path']),reference['identity'],'Official norm checkpoint unchanged')
    fd=os.open(reference['path'],os.O_RDONLY|os.O_NOFOLLOW);out={}
    try:
        info=os.fstat(fd);same([info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns],[reference['identity'][k]for k in ('device','inode','size','mtime_ns','ctime_ns')],'Opened exact norm checkpoint')
        prefix=os.pread(fd,8,0);require(len(prefix)==8,'Norm checkpoint header prefix');hlen,=struct.unpack('<Q',prefix);require(0<hlen<MIB,'Norm header bounds')
        same(json.loads(os.pread(fd,hlen,8)),header,'Exact original norm header')
        for name in NORM_NAMES:
            context.guard();key='model.layers.0.'+name+'.weight';row=header[key]
            same(row['shape'],[2048],'Norm full2048shape');same(row['dtype'],'BF16','Norm original dtype');start,end=row['data_offsets']
            require(end-start==4096 and 0<=start<end<=info.st_size-8-hlen,'Exact small norm interval')
            expected=os.pread(fd,4096,8+hlen+start);require(len(expected)==4096,'Complete official norm interval')
            value=getattr(core.layers[0],name).weight;actual=raw(torch,value);same_bytes(actual,expected,'Official complete norm weight')
            require(bool(torch.isfinite(value).all()),'Finite original norm weight');path=Path(directory)/(name+'.weight.bf16')
            with path.open('xb')as stream:require(stream.write(actual)==4096,'Complete saved norm weight')
            out[name]=dict(path=str(path),sha256=hashlib.sha256(actual).hexdigest(),bytes=4096,shape=[2048],dtype='torch.bfloat16',official_key=key,official_tensor=row)
    finally:os.close(fd)
    same(identity(reference['path']),reference['identity'],'Norm checkpoint stable after small reads')
    return out


def capture_value(torch, directory, name, value, native):
    same(list(value.shape), expected_shapes(native)[name], 'Actual full capture shape '+name)
    require(value.dtype == torch.bfloat16 and bool(torch.isfinite(value).all()), 'Finite BF16 observation')
    meta = metadata(torch,value); checked_layout(meta['shape'],meta['stride'],meta['offset'])
    data = raw(torch,value); require(len(data)<=CAPTURE_FILE_CAP,'Finite full capture file'); path = directory/(name+'.bf16')
    with path.open('xb') as f: require(f.write(data) == len(data), 'Complete observed tensor write')
    return dict(name=name,metadata=meta,path=str(path),sha256=hashlib.sha256(data).hexdigest(),
                bytes=len(data),encoding='packed_bf16_le',identity=identity(path),observed_not_recomputed=True)


def weights(torch, plan, guard):
    """Same bounded pread pattern as saved-oproj; three explicit 24 MiB intervals."""
    reference = next(x for x in plan['model_files'] if x['path'].endswith('.safetensors'))
    header_ref = plan['safetensors_header']; header = json.loads(Path(header_ref['path']).read_text())
    require(hashlib.sha256(Path(header_ref['path']).read_bytes()).hexdigest() == header_ref['sha256'], 'Official header pin')
    guard(); same(identity(reference['path']),reference['identity'],'Official file retained')
    expected = [reference['identity'][k] for k in ('device','inode','size','mtime_ns','ctime_ns')]
    fd = os.open(reference['path'],os.O_RDONLY|os.O_NOFOLLOW); out={}; proofs={}
    try:
        def unchanged():
            s=os.fstat(fd)
            same([s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns],expected,'Opened original checkpoint')
        unchanged(); prefix=os.pread(fd,8,0); require(len(prefix)==8,'Complete header prefix')
        hlen,=struct.unpack('<Q',prefix);require(0<hlen<MIB,'Bounded header')
        data=os.pread(fd,hlen,8);require(len(data)==hlen,'Full header');same(json.loads(data),header,'Original exact header')
        for label in ('gate','up','down'):
            guard(); name='model.layers.0.mlp.'+label+'_proj.weight'; spec=header[name]
            shape=[2048,6144] if label=='down' else [6144,2048]
            same(spec['shape'],shape,'Official MLP matrix');same(spec['dtype'],'BF16','Official BF16')
            start,end=spec['data_offsets'];require(end-start==24*MIB and end+8+hlen<=expected[2],'24 MiB matrix bounds')
            buf=bytearray()
            while len(buf)<24*MIB:
                guard(); part=os.pread(fd,min(MIB,24*MIB-len(buf)),8+hlen+start+len(buf))
                require(bool(part),'Matrix interval short read');buf.extend(part)
            out[label]=torch.frombuffer(buf,dtype=torch.bfloat16).reshape(shape)
            require(bool(torch.isfinite(out[label]).all()),'Finite official matrix')
            proofs[label]=dict(name=name,tensor=spec,bytes_read=len(buf),sha256=hashlib.sha256(buf).hexdigest())
        unchanged()
    finally:os.close(fd)
    same(identity(reference['path']),reference['identity'],'Official checkpoint unchanged')
    return out,dict(file=reference,header=header_ref,matrices=proofs,bytes_read=72*MIB,full_file_rehashed=False)


class PrefixDone(BaseException):
    pass


def cpu_route(kwargs):
    require(kwargs.get('past_key_value') is None, 'CPU prefix cannot consume history')
    require(kwargs.get('get_kv_per_layer') is False and kwargs.get('update_kv_per_layer') is False,
            'No per-layer KV read/write')
    require(kwargs.get('is_for_context_encoding') is True and kwargs.get('idx') == 0,
            'Exact layer0 prefill route')
    require(kwargs.get('windowed_context_encoding_window_idx',-1) in (-1,0), 'No later context window')


class Observers:
    def __init__(self,torch,core,native,directory,guard):
        self.torch,self.core,self.native,self.directory,self.guard=torch,core,native,directory,guard
        self.handles=[];self.replaced=[];self.values={};self.routes=0;self.completed=0;self.done=PrefixDone()
        self.prediction=None
        try:
            self.prediction=PredictionHooks(core,list(TAPS))
            layer=core.layers[0]
            def save(name,value):
                self.guard();require(name not in self.values,'One actual observation '+name)
                self.values[name]=capture_value(torch,directory,name,value,native)
            def input_value(args,kwargs):
                require(not(args and 'hidden_states'in kwargs),'Unambiguous original hidden states')
                require(bool(args)or 'hidden_states'in kwargs,'Actual original hidden states')
                return args[0]if args else kwargs['hidden_states']
            def entry(module,args,kwargs):
                self.route(module,args,kwargs);save('layer_input',input_value(args,kwargs))
            def attention(module,args,output):
                if native:
                    require(type(output)is tuple and len(output)==2,'Original native attention pair')
                    value=output[0]
                else:
                    require(type(output).__name__=='NeuronAttentionBaseOutput'and type(output).__module__=='neuronx_distributed_inference.modules.attention.attention_base','Original SDK attention result')
                    require(output.residual is None,'No fused residual output');value=output.hidden_states
                save('attention_output',value)
            self.handles.append(layer.register_forward_pre_hook(entry,with_kwargs=True))
            self.handles.append(layer.input_layernorm.register_forward_hook(lambda m,a,o:save('inputnorm_output',o)))
            self.handles.append(layer.self_attn.register_forward_pre_hook(lambda m,a,k:save('attention_input',input_value(a,k)),with_kwargs=True))
            self.handles.append(layer.self_attn.register_forward_hook(attention))
            self.handles.append(layer.post_attention_layernorm.register_forward_pre_hook(lambda m,a,k:save('postnorm_input',input_value(a,k)),with_kwargs=True))
            self.handles.append(layer.post_attention_layernorm.register_forward_hook(lambda m,a,o:save('postnorm_output',o)))
            self.handles.append(layer.mlp.register_forward_pre_hook(lambda m,a,k:save('mlp_input',input_value(a,k)),with_kwargs=True))
            # This hook is registered after the prediction hook: capture the
            # actual complete block output before ending the bounded prefix.
            self.handles.append(core.layers[0].register_forward_hook(self.stop))
            def forbidden(*args,**kwargs):raise RuntimeError('Forbidden downstream/history execution')
            self.handles.append(core.layers[1].register_forward_pre_hook(forbidden))
            self.handles.append(core.norm.register_forward_pre_hook(forbidden))
            if not native:
                require(not core.neuron_config.is_prefix_caching and
                        core.neuron_config.windowed_context_encoding_size is None,'Ordinary non-prefix prefill')
                for name in ('get_cache','update_cache'):
                    require(name not in core.kv_mgr.__dict__,'Original KV manager method')
                    original=getattr(core.kv_mgr,name)
                    replacement=types.MethodType(forbidden,core.kv_mgr)
                    setattr(core.kv_mgr,name,replacement)
                    self.replaced.append((core.kv_mgr,name,original,replacement))
        except BaseException:
            self.close();raise

    def route(self,module,args,kwargs):
        self.guard();require(self.routes==0,'Exactly one layer0 entry')
        if self.native:
            cache=kwargs.get('past_key_values')
            require(cache is not None and cache.get_seq_length(0)==0,'Fresh native prefill cache')
            require(kwargs.get('position_ids') is not None,'Native original positions')
        else:
            cpu_route(kwargs);require(kwargs.get('residual')is None,'Layer0 has no previous fused residual')
            config=module.neuron_config
            for name in ('qkv_kernel_enabled','mlp_kernel_enabled','qkv_kernel_fuse_residual_add','mlp_kernel_fuse_residual_add','sequence_parallel_enabled','is_eagle_draft'):
                require(getattr(config,name)is False,'Original unfused layer0 route: '+name)
        self.routes+=1

    def stop(self,module,args,output):
        require(self.routes==1 and self.completed==0 and list(self.values)==list(FIELDS), 'Complete bounded layer0')
        self.completed+=1
        raise self.done

    def close(self):
        actions=[h.remove for h in self.handles];self.handles=[]
        if self.prediction is not None:actions.append(self.prediction.close);self.prediction=None
        for owner,name,original,replacement in reversed(self.replaced):
            def restore(o=owner,n=name,b=original,r=replacement):
                valid=o.__dict__.get(n) is r
                if n in o.__dict__:delattr(o,n)
                require(valid and getattr(o,n)==b,'KV method identity restored')
            actions.append(restore)
        self.replaced=[]
        phase.base.cleanup_actions(actions,'prefix observer cleanup')


def anchor(torch,taps,meta,old,native):
    same(list(taps),list(TAPS),'Six original prefix prediction taps')
    hashes={n:metadata(torch,taps[n])['sha256']for n in TAPS}
    expected=old['tap_raw_sha256'] if native else old['stage_sha256']
    same(hashes,{n:expected[n]for n in TAPS},'Frozen six prediction anchors')
    for name in TAPS:
        if native:
            m=old['metadata'][name]
            same(meta[name]['shape'],m['source_shape'],'Native source shape')
            same(meta[name]['stride'],m['source_stride'],'Native source stride')
            same(meta[name]['dtype'],m['dtype'],'Native source dtype')
            same(meta[name]['prediction_positions'],[m['local_prediction_position']],'Native prediction row')
        else:same(meta[name],old['stage_metadata'][name],'CPU original source geometry')
    return dict(hashes=hashes,source_metadata=meta,all_six_exact=True,whole_model_endpoint_claimed=False)


def execute(context,directory,native,rank):
    context.verify_sources();spec=validate_science(context.plan);directory=Path(directory);directory.mkdir(exist_ok=False)
    require(sys.byteorder=='little','Explicit captured BF16 little-endian host')
    activation_path=ACTIVATIONS['native'if native else'cpu']
    require(activation_path in context.plan['source_sha256'],'Actual activation source admitted before model load')
    activation_source(Path(activation_path).read_text())
    with context.activate():
        if native:
            from minicpm5_layer0_native_adapter_v2 import load
            torch,model,_,records=load(context);core=model.model;old=records[1];proof=context.plan['native_provenance'];transition=None
        else:
            import torch
            importlib.import_module('neuronx_distributed_inference.modules.lora_serving')
            from neuronx_distributed.parallel_layers import parallel_state
            from minicpm5_layer0_cpu_adapter_v2 import CPUAdapter
            transition=context.record_cpu_import();torch.set_num_threads(1)
            torch.distributed.init_process_group('gloo');initialized=False
            try:
                require(torch.distributed.get_rank()==rank and torch.distributed.get_world_size()==2,'Actual TP2')
                parallel_state.initialize_model_parallel(2,skip_collective_init=True);initialized=True
                runtime=CPUAdapter(context,torch);model,proof=runtime.build_original_model(rank,'prefill');core=model
                old=context.read_json(**dict(path=spec['cpu_anchors'][rank]['path'],wanted=spec['cpu_anchors'][rank]['sha256']))
                return capture(context,torch,model,core,old,proof,directory,False,rank,runtime,transition)
            finally:
                phase.base.cleanup_actions(([parallel_state.destroy_model_parallel]if initialized else[])+
                                           [torch.distributed.destroy_process_group],'TP cleanup')
        return capture(context,torch,model,core,old,proof,directory,True,-1,None,transition)


def capture(context,torch,model,core,old,proof,directory,native,rank,runtime,transition):
    observer=None;official=None;params=None;dispatch=None
    with torch.inference_mode():
        try:
            official,weight_proof=weights(torch,context.plan,context.guard)
            weight_rows={};mlp=core.layers[0].mlp
            from transformers.activations import SiLUActivation
            activation_path=ACTIVATIONS['native'if native else'cpu']
            same(str(Path(importlib.import_module(SiLUActivation.__module__).__file__).resolve()),
                 activation_path,'Original activation runtime source')
            activation_proof=activation_dispatch(torch,mlp.act_fn,SiLUActivation,Path(activation_path).read_text(),activation_path)
            require(all(getattr(mlp,n+'_proj').bias is None for n in ('gate','up','down')),'Original bias-free MLP')
            if not native:
                require(mlp.mlp_kernel_enabled is False and mlp.sequence_parallel_enabled is False,
                        'Original unfused native SDK MLP')
                down=mlp.down_proj
                require(down.input_is_parallel is True and down.reduce_output is True and
                        down.reduce_dtype==torch.bfloat16 and down.sequence_parallel_enabled is False and
                        down.tensor_parallel_group.size()==2 and down.pad_size==0,'Actual BF16 TP2 down path')
            for name,full in official.items():
                local=getattr(mlp,name+'_proj').weight
                wanted=full if native else (full[:,rank*3072:(rank+1)*3072]if name=='down'else full[rank*3072:(rank+1)*3072])
                same_bytes(raw(torch,local),raw(torch,wanted),'Exact official MLP '+name+' shard')
                weight_rows[name]=metadata(torch,local)
            params={name:getattr(mlp,name+'_proj').weight for name in official};del official
            if native:
                same(old['frame'],FRAME,'Native exact frame1')
                fixture=context.plan['fixtures'][7];require(fixture['id']==7 and len(fixture['input_ids'])==LENGTH,'Original long fixture')
                args=dict(input_ids=torch.tensor([fixture['input_ids']]),attention_mask=torch.ones((1,LENGTH),dtype=torch.long),
                          position_ids=torch.arange(LENGTH)[None],use_cache=True,logits_to_keep=1)
                original_inputs={k:metadata(torch,v)for k,v in args.items()if hasattr(v,'dtype')}
                invoke=lambda:model(**args)
            else:
                graph=context.plan['graph_calls'][1];same(graph['call'],old['call'],'Original call1');same(graph['bucket'],BUCKET,'Original bucket')
                validate_baseline(old,rank,full=False)
                original=runtime.graph_inputs(graph);inputs,widening=runtime.cpu_indices(original)
                same(widening,old['integer_widening'],'Lossless original integer widening')
                original_inputs=[runtime.tensor_metadata(x)for x in inputs]
                same(original_inputs,old['execution_input_metadata'],'All seven original argument values/layouts')
                runtime.verify_original_positions_masks(inputs,graph['call'],context.plan['fixtures'])
                runtime.select_original_bucket(model,BUCKET)
                same(phase.selected_targets(ARM,'prefill'),(),'Strict baseline has no precision intervention')
                dispatch=original_dispatch(core)
                invoke=lambda:model(*inputs)
            norms=norm_dispatch(torch,core,native,context);norm_saved=norm_weights(torch,core,context,directory)
            observer=Observers(torch,core,native,directory,context.guard)
            observer.prediction.begin(torch.tensor([LENGTH]),True)
            try:invoke()
            except PrefixDone as done:require(done is observer.done,'Only own terminal prefix signal')
            else:raise RuntimeError('Prefix failed to stop before downstream layers')
            require(observer.completed==observer.routes==1,'Exactly one completed prefix')
            taps,tapmeta=observer.prediction.finish();checked=anchor(torch,taps,tapmeta,old,native)
            if native:same({k:metadata(torch,v)for k,v in args.items()if hasattr(v,'dtype')},original_inputs,'Native args unchanged')
            else:same([runtime.tensor_metadata(x)for x in inputs],original_inputs,'Seven CPU args unchanged')
            for name,parameter in params.items():
                require(getattr(mlp,name+'_proj').weight is parameter,'Original MLP parameter object')
                same(metadata(torch,parameter),weight_rows[name],'Original MLP parameter values')
            if not native: check_dispatch(core,dispatch)
            full_anchor=check_full_anchor(context,observer.values,old,rank,native)
            same(norm_dispatch(torch,core,native,context),norms,'Original norm dispatch/weights unchanged')
            captures=list(observer.values.values());same(sum(x['bytes']for x in captures),capture_bytes(native),'Exact raw capture budget')
            result=dict(schema='minicpm5_frame1_upstream_capture_v1',complete=True,native=native,rank=rank,frame=FRAME,
                        anchor=checked,captures=captures,weights=weight_proof,actual_weight_shards=weight_rows,constructor=proof,
                        oproj_witnesses=[],precision_interventions=[],cpu_full_anchor=full_anchor,environment_transition=transition,actual_prefix_calls=1,
                        activation_dispatch=activation_proof,norm_dispatch=norms,norm_weights=norm_saved,
                        history_consumed=False,original_full_history_replayed=False,downstream_layers_executed=0,
                        full_model_qualified=False,new_speedup_claim=False)
        finally:
            phase.base.cleanup_actions(([observer.close]if observer else[]),'capture cleanup')
        once(directory/'capture.json',result)
        return result
