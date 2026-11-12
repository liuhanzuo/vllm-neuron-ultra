"""Stdlib contract for the exact pinned Spark-X2.6 architecture and cache ABI."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REVISION = 'aa074aa851209a683aeaf5f99a8e25a642d94541'
MODEL_ID = 'XHToken/Spark-X2.6-4B'
SOURCE_DIR = ROOT/'research/new-small-model-code-prerequisites-20261104-v72'
CONFIG = ROOT/'research/new-small-model-source-refresh-20261104-v69/Spark-X2.6-4B/config.json'
MODELING = SOURCE_DIR/'spark-modeling.py'
CONFIGURATION = SOURCE_DIR/'spark-configuration.py'
LICENSE = SOURCE_DIR/'spark-LICENSE.txt'
READINESS = SOURCE_DIR/'source-porting-readiness.json'
PINS = {str(MODELING):'9cf0d1ad2b54b9f7088792779ddd8b5cb4d4fe63bfea054da7f2dd16362bcaf4',
 str(CONFIGURATION):'02218597240490f490659b184052db940b08163ff9c3af7b7e59323f6f963722',
 str(CONFIG):'767161ece8ed2891e44345afdf98d29c1cacb964116cb6db7d20dc1c757490ad',
 str(LICENSE):'525c465ab952779d2d360a9ff36f7f49ae018a23d80302ff839a850a77784c36',
 str(READINESS):'69ef02ed6aaed7f534eb01f54534260d4c22620da68d72f5dcc3854149484c28'}
PHASES = ('constructor_state_dict_meta', 'full_model_forward_meta', 'cache_mask_cpu', 'rope_gelu_cpu')
LAYER_TYPES = ['sliding_attention']*3+['full_attention']
LAYER_TYPES *= 9
UNIQUE_PARAMETERS = 4112079360
META_STEPS = (1, 511, 1)
CPU_SCENARIOS = tuple(dict(name=name+('-pad2' if pad else '-unpadded'),chunks=list(chunks),leading_pad=pad)
    for name,chunks in [('boundary',(510,1,1,1,17)),('long_prefill',(513,1,17))] for pad in (0,2))


def config():
    for path,digest in PINS.items():
        p=Path(path);assert p.stat().st_size<4*1024**2
        assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,path
    value=json.loads(CONFIG.read_text())
    expected=dict(architectures=['Spark2_5ForCausalLM'],model_type='spark2_5',vocab_size=131072,
        hidden_size=2560,intermediate_size=10240,num_hidden_layers=36,num_attention_heads=16,
        num_key_value_heads=4,head_dim=256,hidden_act='gelu',headwise_attn_output_gate=True,
        gate_attn_act_mode='sigmoid',sliding_window=512,layer_types=LAYER_TYPES,
        tie_word_embeddings=True,attention_bias=False,mlp_bias=False,dtype='bfloat16',
        max_position_embeddings=1048576,attention_dropout=0.0)
    assert all(value[key]==wanted for key,wanted in expected.items())
    assert value['rope_parameters']=={'full_attention':{'partial_rotary_factor':.25,'rope_theta':5000000},
        'sliding_attention':{'partial_rotary_factor':1.0,'rope_theta':10000}}
    return value


def state_schema():
    cfg=config(); h=cfg['hidden_size']; inter=cfg['intermediate_size']; vocab=cfg['vocab_size']
    result={'model.embedding.weight':[vocab,h],'model.norm.weight':[h],'lm_head.weight':[vocab,h]}
    weights={'self_attn.q_k_v_proj.weight':[6144,h],'self_attn.g_proj.weight':[16,h],
        'self_attn.out_proj.weight':[h,4096],'mlp.gate_proj.weight':[inter,h],
        'mlp.up_proj.weight':[inter,h],'mlp.down_proj.weight':[h,inter],
        'input_layernorm.weight':[h],'post_attention_layernorm.weight':[h]}
    for layer in range(36):
        result.update({f'model.layers.{layer}.{key}':shape for key,shape in weights.items()})
    assert len(result)==291
    count=sum(__import__('math').prod(shape) for name,shape in result.items() if name!='lm_head.weight')
    assert count==UNIQUE_PARAMETERS
    return result


def cache_layout(seen,query,layer_type):
    assert type(seen) is int and seen>=0 and type(query) is int and query>0
    assert layer_type in ('sliding_attention','full_attention')
    sliding=layer_type=='sliding_attention';offset=max(0,seen-511) if sliding else 0
    end=seen+query;stored_offset=max(0,end-511) if sliding else 0
    return dict(kv_offset=offset,returned_length=end-offset,stored_offset=stored_offset,
                stored_length=end-stored_offset,cumulative_length=end)


def mask_oracle(seen,query,layer_type,leading_pad=0):
    layout=cache_layout(seen,query,layer_type);offset=layout['kv_offset'];end=seen+query
    sliding=layer_type=='sliding_attention'
    raw=bytes(int(key<=position and key>=leading_pad and (not sliding or key>position-512))
              for position in range(seen,end) for key in range(offset,end))
    assert len(raw)==query*layout['returned_length']
    return layout,raw


def specification():
    return dict(model=MODEL_ID,revision=REVISION,config=config(),state_schema=state_schema(),
        unique_parameters=UNIQUE_PARAMETERS,state_dict_keys=291,phases=list(PHASES),
        meta_steps=list(META_STEPS),cpu_scenarios=list(CPU_SCENARIOS),
        real_cache_shape_prefix=[1,4],head_dim=256,full_model_layers=36,
        full_meta_forward_uses_prebuilt_meta_masks=True,official_mask_creation_tested_separately_on_cpu=True,
        no_weights_download=True,no_original_weight_numeric_claim=True,no_device_or_compilation=True,
        sdk_model_port_implemented=False,full_baseline_qualified=False,speedup_claim=False)
