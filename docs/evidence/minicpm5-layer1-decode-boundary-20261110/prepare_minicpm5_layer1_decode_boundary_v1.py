"""Build all36 source-only decode boundaries from authenticated original JSON."""
import ast
import hashlib
import json
from pathlib import Path
import sys
Q=Path(__file__).resolve().parent;R=Q.parent;sys.path[:0]=[str(R),str(Q)]
from research import minicpm5_layer1_decode_boundary_v1 as boundary
from research.prepare_minicpm5_layer1_observation_source_v1 import Reader
require=boundary.require
PRIOR=dict(path=str(Q/'minicpm5-layer1-runner-source-plan-20261110-v1.json'),
 sha256='122cf50cac06022356a0681712ff762f343b80c74ac53520964e02b9451537de')
SDK=Path('/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/neuronx_distributed_inference')
CONFIG=Path('/mnt/nvme/neuron-artifacts/latest-20261031-minicpm5-2b-bf16-cpu-tp2-b4-ctx4096-baseline-v4/neuron_config.json')
SOURCE_FUNCTIONS={
 str(R/'minicpm5_history_tensors_v1.py'):[('PredictionHooks','hook')],
 str(R/'minicpm5_original32_oproj_cpu_worker_v1.py'):[(None,'strict_inputs'),(None,'route_hooks')],
 str(SDK/'models/llama/modeling_llama.py'):[('NeuronLlamaAttention','get_rope'),('NeuronLlamaDecoderLayer','forward')],
 str(SDK/'models/model_base.py'):[('NeuronBaseModel','get_model_output')],
 str(SDK/'modules/attention/attention_base.py'):[('NeuronAttentionBase','standard_causal_attention_forward'),('NeuronAttentionBase','prep_qkv_tensors'),('NeuronAttentionBase','apply_rotary_embedding')],
 str(SDK/'modules/attention/utils.py'):[('RotaryEmbedding','forward')],
 str(SDK/'modules/kvcache/kv_cache_manager.py'):[(None,'_slice_kv_cacheline'),('KVCacheManager','get_kv_by_layer_id'),('KVCacheManager','get_cache')],
}


def ref(path):
    path=Path(path);return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def prepare():
    reader=Reader();plan=reader.read(PRIOR['path'],PRIOR['sha256'])
    rows=boundary.all_boundaries(plan,reader.read)
    header=reader.read(plan['safetensors_header']['path'],plan['safetensors_header']['sha256'])
    weights=boundary.weight_slices(plan,header)
    config=reader.read(str(CONFIG),plan['inputs'][str(CONFIG)]['sha256']);nc=config['neuron_config']
    require(config['rope_scaling']is None and config['rope_theta']==5000000
        and config['head_dim']==128 and config['num_hidden_layers']==42,
        'Original layer1/RoPE model geometry')
    disabled=('qkv_kernel_enabled','mlp_kernel_enabled','sequence_parallel_enabled','is_prefix_caching',
        'flash_decoding_enabled','kv_cache_tiling','k_cache_transposed','kv_cache_quant','is_medusa',
        'layer_boundary_markers','attn_block_cte_nki_kernel_enabled','attn_block_tkg_nki_kernel_enabled',
        'attn_tkg_builtin_kernel_enabled','attn_tkg_nki_kernel_enabled')
    require(all(nc[name]is False for name in disabled),'Original unfused direct attention route')
    require(nc.get('moe_mask_padded_tokens',False)is False,'Original SDK padding-mask default')
    require(nc['attention_dp_degree']==nc['cp_degree']==1 and nc['tp_degree']==2
        and nc['padding_side']=='right'and nc['max_batch_size']==4 and nc['kv_cache_padding_size']==0,
        'Original full batch/right padding/TP2 route')
    snippets=[]
    for path,functions in SOURCE_FUNCTIONS.items():
        raw=Path(path).read_bytes();require(hashlib.sha256(raw).hexdigest()==plan['source_sha256'][path],
            'Exact original SDK/capture source')
        tree=ast.parse(raw);lines=raw.splitlines(keepends=True)
        for class_name,name in functions:
            container=tree
            if class_name:
                found=[n for n in tree.body if isinstance(n,ast.ClassDef)and n.name==class_name]
                require(len(found)==1,'One actual original class '+str(class_name));container=found[0]
            found=[n for n in container.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))and n.name==name]
            require(len(found)==1,'One actual source function '+str(class_name)+'.'+name)
            node=found[0];snippet=b''.join(lines[node.lineno-1:node.end_lineno])
            snippets.append(dict(source=ref(path),class_name=class_name,function=name,start_line=node.lineno,
                end_line=node.end_lineno,function_sha256=hashlib.sha256(snippet).hexdigest()))
    require(not {'torch','numpy','neuronxcc','torch_neuronx','transformers','vllm'}&set(sys.modules),
        'Source preparation has no model runtime import')
    return dict(schema='minicpm5_layer1_decode_boundary_source_plan_v1',inherited_plan=PRIOR,
        source_only=True,runtime_admitted=False,cases=rows,total_rank_decode_calls=36,
        physical_prediction_rows=144,replicated_rank_rows_must_be_counted_once=72,
        own_history_replay_not_yet_executed=True,paired_history_intervention_allowed=False,
        current_input_layout_restore_required=True,full_caches_retained_before_sdk_slice=True,
        sdk_functions=snippets,finished_config=dict(path=str(CONFIG),sha256=plan['inputs'][str(CONFIG)]['sha256']),
        disabled_route_flags={key:nc[key]for key in disabled},
        moe_padding_mask=dict(config_key_present='moe_mask_padded_tokens'in nc,effective_value=nc.get('moe_mask_padded_tokens',False),
            sdk_source_default=False),official_layer1_weight_slices=weights,
        original_complete_precision_and_acceptance_gates=plan['remaining_gates'],
        original_HF32_and_full_collection_required=True,
        source_refs=[ref(Q/name)for name in ('minicpm5_layer1_decode_boundary_v1.py',
            'check_minicpm5_layer1_decode_boundary_v1.py','prepare_minicpm5_layer1_decode_boundary_v1.py')],
        missing_runtime_work=[
            'Actual complete original84call collection and independent full cache/full logits/HF audit with external exits.',
            'Native replay loader restores original normalized tensor stride, uses actual SDK full-cache view, authenticates all8 layer1 projection shards against original weights.',
            'Reproduce original carried RoPE caches with original SDK RotaryEmbedding; same dtype/device/positions/base and BF16 cos/sin. Source shows no dependence on value content, but actual replay output must still match.',
            'Admitted TP2 own-history replay must reproduce all36 complete4row attention outputs exactly before any paired-history test.',
            'Keep all cold-cache4096/long-decode/slot-reuse/generation/RNG/stop/text/quality/HTTP-C4/paired-performance gates for model qualification.'
        ],real_tensor_payloads_read=False,native_imported=False,attention_causality_proven=False,
        full_model_qualified=False,new_speedup_claim=False)


if __name__=='__main__':
    require(sys.dont_write_bytecode and __debug__,'Python -B with assertions')
    plan=prepare();target=Q/'minicpm5-layer1-decode-boundary-source-plan-20261110-v1.json'
    with target.open('x')as f:json.dump(plan,f,indent=2);f.write(chr(10))
    print(json.dumps(dict(plan=ref(target),decode_calls=36,physical_rows=144,sdk_functions=len(plan['sdk_functions']),
        runtime_admitted=False)))
