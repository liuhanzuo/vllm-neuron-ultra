"""Inert stdlib contract for the original Nemotron-3-Nano-4B BF16 baseline.

No model allocation, tokenizer load, payload read, registration or I/O at import.
Shape contracts are necessary checks, not evidence of native/Neuron execution.
"""
import math

MODEL = 'nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16'
REVISION = 'dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f'
WEIGHT_FILE = 'model.safetensors'
WEIGHT_SHA256 = '55d4e2519456c4a9bddf596b0748d630e3b2ce6ff6f4c2b7ed3e07e2b00dad42'
WEIGHT_BYTES = 7947142640
HEADER_BYTES = 28968
PATTERN = 'M-M-M-MM-M-M*-M-M*-M-M-M*-M-M-MM*-MMM-M-M-'
MODEL_TYPE = 'nemotron4b_neuron_text_v1'
NXDI_KEY = 'nemotron4bneurontextv1'
ARCHITECTURE = 'Nemotron4BNeuronTextV1ForCausalLM'
EXACT = dict(hidden_size=3136, intermediate_size=12544, vocab_size=131072,
    num_hidden_layers=42, num_attention_heads=40, num_key_value_heads=8, head_dim=128,
    mamba_num_heads=96, mamba_head_dim=80, n_groups=8, ssm_state_size=128,
    conv_kernel=4, chunk_size=256, pad_token_id=0, bos_token_id=1, eos_token_id=2,
    layer_norm_epsilon=1e-5, mamba_hidden_act='silu', mlp_hidden_act='relu2',
    use_conv_bias=True, hybrid_override_pattern=PATTERN)
CRITERIA = dict(cosine_min=.995, kl_max_nats=.02, max_abs_error_max=2.,
    argmax_match_fraction_min=.9, all_reference_margins_gt_0_5_must_match=True)
STOP_IDS = (2, 11)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_config(config):
    """Accept the exact official geometry, independent of runtime class aliases."""
    c = config if isinstance(config, dict) else vars(config)
    for name,wanted in EXACT.items():
        require(name in c and c[name] == wanted, 'Unexpected original 4B config: '+name)
        if type(wanted) is int: require(type(c[name]) is int, 'Noninteger 4B geometry: '+name)
    for flag in ('tie_word_embeddings','residual_in_fp32','attention_bias','mlp_bias',
                 'use_bias','mamba_proj_bias','attention_dropout','hidden_dropout'):
        require(not c.get(flag,False), 'Unsupported 4B setting: '+flag)
    require(c.get('sliding_window') is None and not c.get('num_nextn_predict_layers',0), 'Unexpected sliding/MTP route')
    require(tuple(c.get('time_step_limit',(0.,float('inf')))) == (0.,float('inf')), 'Unexpected runtime dt clamp')
    require(c.get('model_type') in ('nemotron_h', MODEL_TYPE, None), 'Wrong model family')
    for alias,key in [('attention_head_dim','head_dim'),('mamba_n_groups','n_groups'),
                      ('mamba_d_conv','conv_kernel'),('mamba_chunk_size','chunk_size')]:
        if alias in c: require(c[alias] == c[key], 'Conflicting config alias: '+alias)
    return c


def original_inventory():
    """Generate every original key/shape without importing native libraries."""
    result = {'backbone.embeddings.weight':[131072,3136], 'backbone.norm_f.weight':[3136],
              'lm_head.weight':[131072,3136]}
    for layer,kind in enumerate(PATTERN):
        prefix='backbone.layers.'+str(layer)+'.'
        result[prefix+'norm.weight']=[3136]
        if kind=='M':
            shapes={'in_proj.weight':[17504,3136], 'conv1d.weight':[9728,1,4],
                'conv1d.bias':[9728], 'A_log':[96], 'D':[96], 'dt_bias':[96],
                'norm.weight':[7680], 'out_proj.weight':[3136,7680]}
        elif kind=='*':
            shapes={'q_proj.weight':[5120,3136], 'k_proj.weight':[1024,3136],
                    'v_proj.weight':[1024,3136], 'o_proj.weight':[3136,5120]}
        else:
            shapes={'up_proj.weight':[12544,3136], 'down_proj.weight':[3136,12544]}
        result.update({prefix+'mixer.'+name:shape for name,shape in shapes.items()})
    require(len(result)==263 and sum(math.prod(x) for x in result.values())==3973556832, 'Inventory algebra changed')
    return result


def checkpoint_key_to_local(name):
    require(name=='lm_head.weight' or name.startswith('backbone.'), 'Non-original checkpoint key')
    return name if name=='lm_head.weight' else name[len('backbone.'):]


def learned_entries():
    """Map 431 logical learned entries to exact original rectangular slices.

    Split offsets are global before TP. Keeping gate/x/B/C/dt independent avoids
    the incorrect operation of splitting the entire packed projection in half.
    """
    entries={}
    for original,shape in original_inventory().items():
        key=checkpoint_key_to_local(original)
        if key.endswith('.mixer.in_proj.weight'):
            fields=[('gate',7680),('x',7680),('B',1024),('C',1024),('dt',96)]
            offset=0
            for field,width in fields:
                name=key.removesuffix('weight')+field+'.weight'
                entries[name]=dict(original=original,axis=0,offset=offset,width=width,shape=[width]+shape[1:])
                offset+=width
        elif '.mixer.conv1d.' in key:
            stem,param=key.rsplit('.',1);offset=0
            for field,width in [('x',7680),('B',1024),('C',1024)]:
                entries[stem+'.'+field+'.'+param]=dict(original=original,axis=0,offset=offset,width=width,shape=[width]+shape[1:])
                offset+=width
        else:
            axis=None
            if key=='embeddings.weight':axis=1
            elif key=='lm_head.weight' or key.endswith(('.q_proj.weight','.k_proj.weight','.v_proj.weight','.up_proj.weight')):axis=0
            elif key.endswith(('.o_proj.weight','.out_proj.weight','.down_proj.weight')):axis=1
            elif key.endswith(('.A_log','.D','.dt_bias','.mixer.norm.weight')):axis=0
            entries[key]=dict(original=original,axis=axis,offset=0,width=None if axis is None else shape[axis],shape=shape)
    require(len(entries)==431, 'Wrong learned entry count')
    return entries


def sharded_entries(rank, tp=2):
    require(tp==2 and type(rank) is int and rank in (0,1), 'Baseline requires TP2 rank0/1')
    result={}
    for name,spec in learned_entries().items():
        shape=list(spec['shape']);axis=spec['axis']
        start=spec['offset']
        if axis is not None:
            require(spec['width']%tp==0,'Nondivisible split')
            shape[axis]//=tp;start+=rank*shape[axis]
        result[name]=dict(spec,shape=shape,global_start=start,
            global_stop=None if axis is None else start+shape[axis],dtype='BF16')
    result['rank_util.rank']=dict(shape=[1],dtype='I32',value=rank,original=None)
    return result


def state_contract(batch=4, context=4096, tp=2):
    require(type(batch) is int and batch==4 and type(context) is int and context==4096 and tp==2,
            'Initial baseline state scope is B4/context4096/TP2')
    states=[];offsets=[]
    for layer,kind in enumerate(PATTERN):
        start=len(states)
        if kind=='M':specs=[('conv',[4,4864,4],'BF16'),('ssm',[4,48,80,128],'F32')]
        elif kind=='*':specs=[('key',[4,4,4096,128],'BF16'),('value',[4,4,4096,128],'BF16')]
        else:specs=[]
        for name,shape,dtype in specs:
            states.append(dict(index=len(states),layer=layer,kind=name,shape=shape,dtype=dtype,
                               bytes=math.prod(shape)*(4 if dtype=='F32' else 2)))
        offsets.append((start,len(states)))
    require(len(states)==50 and sum(x['bytes'] for x in states)==302637056,'State layout changed')
    return dict(per_rank=states,layer_offsets=offsets,buffers_per_rank=50,buffers_all_ranks=100,
                bytes_per_rank=302637056,bytes_all_ranks=605274112)


def native_config_values(raw):
    """Use the pinned native implementation, never checkpoint remote Python."""
    validate_config(raw)
    c=dict(raw)
    c.pop('auto_map',None);c.pop('architectures',None)
    # Preserve official initialization bounds. The pinned native mixer ignores
    # config.time_step_limit, so apply_native_runtime_contract must still run.
    c.update(model_type='nemotron_h',head_dim=128,time_step_limit=(0.,float('inf')))
    return c


def view_config_values(raw):
    validate_config(raw)
    c=dict(raw);c.pop('auto_map',None)
    c.update(model_type=MODEL_TYPE,architectures=[ARCHITECTURE])
    # Omit Infinity from JSON. validate_config and the port explicitly require
    # the original unbounded dt contract; the native override is separate.
    c.pop('time_step_limit',None)
    return c


REMAINING_GATES = [
    'Fresh serial/storage admission and full pinned weight SHA256 verification; no automatic download or enrollment.',
    'Original-weight native BF16 eager cached prefill/decode with all21 mixers explicitly dt0..inf and FP32 recurrent state.',
    'All263 original tensor values covered by431 learned entries per rank+rank utility; no shape-only substitute.',
    'Real-weight primitive/TP2 split/alias/state composition at4B geometry; no9B result transfer.',
    'Numeric32; all6 prefill/decode routes128..4096;7 cold prefixes32/128/256/512/1024/2048/3070,84frames,700zero checks.',
    'Long2041teacher-forced steps/93checkpoints, permuted slots,mixed lengths,one-token prefill,reuse,partial/inactive resume.',
    'Fresh96quality cases under exact official template with thinkingfalse and explicitEOS2/11; actual engine stop semantics.',
    'Five matched workloads x3rounds only after correctness/quality; standard CPU sampler on baseline; isolated timing.',
]
