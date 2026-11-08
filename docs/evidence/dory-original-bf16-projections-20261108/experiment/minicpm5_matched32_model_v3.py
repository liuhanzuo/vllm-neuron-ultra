"""MiniCPM5 matched32 model contract; importing this module is stdlib-only.

The admitted caller owns source/payload authentication, resources and TP2 process
groups. Runtime entries never compile, reshard, patch SDK math or invoke forward.
"""
import copy
import importlib
import json
import os
from pathlib import Path


ROOT = Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
SDK = Path('/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages')
LLAMA = SDK / 'neuronx_distributed_inference/models/llama/modeling_llama.py'
ARTIFACT = Path('/mnt/nvme/neuron-artifacts/latest-20261031-minicpm5-2b-bf16-cpu-tp2-b4-ctx4096-baseline-v4')
LAYERS = 42
TP = 2
VOCAB = 130560
HIDDEN = 2048
HEAD_DIM = 128
BUCKETS = (128, 256, 512, 1024, 2048, 4096)
CACHE_SHAPE = (4, 1, 4224, 128)
CACHES_PER_RANK = 84
LEARNED_TENSORS = 381
SHARD_KEYS = 424
MODEL_KEYS = 509
RANK_ALIAS = 'kv_mgr.global_rank.rank'
RANK_KEYS = frozenset({'rank_util.rank'} | {f'layers.{i}.self_attn.rank_util.rank' for i in range(LAYERS)})
CACHE_KEYS = frozenset(f'kv_mgr.past_key_values.{i}' for i in range(CACHES_PER_RANK))
EXPECTED_MISSING = CACHE_KEYS | {RANK_ALIAS}


def stages():
    names = ['embed_tokens']
    for index in range(LAYERS):
        names.extend(f'layers.{index}.{name}' for name in (
            'input_layernorm', 'self_attn', 'post_attention_layernorm', 'mlp'))
        names.append(f'layers.{index}')
    names.extend(('norm', 'lm_head'))
    assert len(names) == len(set(names)) == 213
    return names


def learned_shapes():
    result = {'embed_tokens.weight': [VOCAB, HIDDEN // TP],
              'lm_head.weight': [VOCAB // TP, HIDDEN], 'norm.weight': [HIDDEN]}
    per_layer = {
        'input_layernorm.weight': [HIDDEN],
        'post_attention_layernorm.weight': [HIDDEN],
        'self_attn.qkv_proj.q_proj.weight': [1024, HIDDEN],
        'self_attn.qkv_proj.k_proj.weight': [HEAD_DIM, HIDDEN],
        'self_attn.qkv_proj.v_proj.weight': [HEAD_DIM, HIDDEN],
        'self_attn.o_proj.o_proj.weight': [HIDDEN, 1024],
        'mlp.gate_proj.weight': [3072, HIDDEN],
        'mlp.up_proj.weight': [3072, HIDDEN],
        'mlp.down_proj.weight': [HIDDEN, 3072],
    }
    for index in range(LAYERS):
        result.update({f'layers.{index}.{name}': list(shape) for name, shape in per_layer.items()})
    assert len(result) == LEARNED_TENSORS
    return result


def expected_state_metadata(kind='shard'):
    assert kind in ('shard', 'model')
    result = {name: dict(shape=shape, dtype='torch.bfloat16') for name, shape in learned_shapes().items()}
    result.update({name: dict(shape=[1], dtype='torch.int32') for name in RANK_KEYS})
    if kind == 'model':
        result.update({name: dict(shape=list(CACHE_SHAPE), dtype='torch.bfloat16') for name in CACHE_KEYS})
        result[RANK_ALIAS] = dict(shape=[1], dtype='torch.int32')
    assert len(result) == (SHARD_KEYS if kind == 'shard' else MODEL_KEYS)
    return result


def check_state_metadata(metadata, kind='shard'):
    """Validate every name, dtype and dimension, without importing tensor code."""
    expected = expected_state_metadata(kind)
    assert isinstance(metadata, dict) and set(metadata) == set(expected), 'Exact state key set required'
    for name, wanted in expected.items():
        row = metadata[name]
        assert isinstance(row, dict) and row.get('shape') == wanted['shape'], (name, row, wanted)
        assert row.get('dtype') == wanted['dtype'], (name, row, wanted)
        assert all(type(size) is int for size in row['shape']), name
    return True


def check_load_metadata(value, rank):
    assert type(rank) is int and rank in (0, 1)
    assert value['missing_keys'] == sorted(EXPECTED_MISSING)
    assert value['unexpected_keys'] == []
    assert value['rank_values'] == {name: [rank] for name in sorted(RANK_KEYS)}
    assert value['rank_alias_value'] == [rank]
    assert all(type(item) is int for values in value['rank_values'].values() for item in values)
    assert type(value['rank_alias_value'][0]) is int
    assert all(value[key] is True for key in (
        'shared_rank_module_identity', 'shared_rank_parameter_identity',
        'state_dict_shared_parameter_identity', 'all_shard_values_raw_exact',
        'all_state_on_cpu', 'all_buffers_on_cpu', 'all_cache_parameters_nontrainable'))
    assert value['alias_repair_performed'] is False
    check_state_metadata(value['shard_metadata'], 'shard')
    check_state_metadata(value['model_metadata'], 'model')
    return True


def stage_overrides(phase):
    assert phase in ('prefill', 'decode')
    return dict(batch_size=1 if phase == 'prefill' else 4,
                n_active_tokens=4096 if phase == 'prefill' else 1,
                bucket_n_active_tokens=phase == 'prefill',
                is_prefill_stage=phase == 'prefill', sequence_parallel_enabled=False)


def check_config_metadata(value, phase=None):
    """Check the finished compiler contract and official per-stage overrides."""
    assert phase in (None, 'prefill', 'decode')
    top = dict(architectures=['LlamaForCausalLM'], model_type='llama',
               hidden_size=HIDDEN, intermediate_size=6144, num_hidden_layers=LAYERS,
               num_attention_heads=16, num_key_value_heads=2, head_dim=HEAD_DIM,
               vocab_size=VOCAB, tie_word_embeddings=False, rope_theta=5000000,
               rope_scaling=None, rms_norm_eps=1e-6, hidden_act='silu', pad_token_id=1,
               num_cores_per_group=1, max_position_embeddings=131072)
    nc = dict(tp_degree=TP, batch_size=4, ctx_batch_size=1, tkg_batch_size=4,
              max_batch_size=4, kv_cache_batch_size=4, kv_cache_padding_size=0,
              seq_len=4096, max_length=4096, max_context_length=4096,
              n_positions=4096, n_active_tokens=4096, bucket_n_active_tokens=False,
              is_prefill_stage=None, buckets=list(BUCKETS), enable_bucketing=True,
              is_continuous_batching=True, apply_seq_ids_mask=True,
              padding_side='right', on_device_sampling_config=None,
              cast_type='config', save_sharded_checkpoint=True, quantized=False,
              vocab_parallel=False, fused_qkv=False, mlp_kernel_enabled=False,
              qkv_kernel_enabled=False, fused_rmsnorm_skip_gamma=False,
              is_block_kv_layout=False, is_prefix_caching=False,
              kv_cache_tiling=False, k_cache_transposed=False, kv_cache_quant=False,
              flash_decoding_enabled=False, sequence_parallel_enabled=False,
              async_mode=False, on_cpu=False, output_logits=False,
              attention_dp_degree=1, cp_degree=1, pp_degree=1, ep_degree=1,
              tensor_capture_config=None, tensor_replacement_config=None,
              attn_cls='NeuronLlamaAttention', attention_dtype=None,
              lora_config=None, is_medusa=False, speculation_length=0,
              enable_eagle_speculation=False, enable_fused_speculation=False,
              enable_token_tree=False, qkv_kernel_fuse_residual_add=False,
              mlp_kernel_fuse_residual_add=False, windowed_context_encoding_size=None)
    if phase is not None:
        nc.update(stage_overrides(phase))
    for name, wanted in top.items():
        assert name in value and value[name] == wanted, (name, value.get(name), wanted)
        assert type(value[name]) is type(wanted), name
    for name, wanted in nc.items():
        assert name in value['neuron_config'] and value['neuron_config'][name] == wanted, (
            name, value['neuron_config'].get(name), wanted)
        assert type(value['neuron_config'][name]) is type(wanted), name
    assert value['torch_dtype'] in ('bfloat16', 'torch.bfloat16')
    assert value['neuron_config']['torch_dtype'] in ('bfloat16', 'torch.bfloat16')
    return True


def _regular(path):
    path = Path(path)
    assert path.is_absolute() and path.is_file(), str(path)
    assert not any(item.is_symlink() for item in (path, *path.parents)), str(path)
    return path


def _serialized(config):
    # This SDK method normalizes torch.dtype without importing anything here.
    return json.loads(config.to_json_string())


def _guarded_runtime():
    import minicpm5_matched32_protocol_v3 as protocol
    return protocol.guard('matched_cpu', child=True)


def configuration(phase):
    """Runtime only. Reproduce the installed application's two stage configs."""
    _guarded_runtime()
    assert os.environ.get('NXD_CPU_MODE') == '1'
    assert phase in ('prefill', 'decode')
    importlib.import_module('neuronx_distributed_inference.modules.lora_serving')
    from neuronx_distributed_inference.models.llama.modeling_llama import LlamaInferenceConfig
    from neuronx_distributed_inference.modules import autobucketing
    path = _regular(ARTIFACT / 'neuron_config.json')
    finished = json.loads(path.read_text())
    check_config_metadata(finished)
    config = LlamaInferenceConfig.load(str(ARTIFACT))
    loaded = _serialized(config)
    # Check every serialized compiler field, not only the selected invariants.
    for name, value in finished.items():
        assert loaded.get(name) == value, ('Finished compiler config changed on load', name)
    config = copy.deepcopy(config)
    for name, value in stage_overrides(phase).items():
        setattr(config.neuron_config, name, value)
    generate = autobucketing.generate_buckets_for_cte if phase == 'prefill' else autobucketing.generate_buckets_for_tkg
    config.neuron_config.buckets = generate(config)
    expected = copy.deepcopy(finished)
    expected['neuron_config'].update(stage_overrides(phase))
    expected['neuron_config']['buckets'] = list(BUCKETS)
    current = _serialized(config)
    check_config_metadata(current, phase)
    for name, value in expected.items():
        assert current.get(name) == value, ('Unapproved per-stage config change', name)
    return config


def _tensor_metadata(state):
    return {name: dict(shape=list(value.shape), dtype=str(value.dtype)) for name, value in state.items()}


def build(rank, phase):
    """Return (official CPU model, load proof) inside admitted initialized TP2."""
    _guarded_runtime()
    assert type(rank) is int and rank in (0, 1)
    assert phase in ('prefill', 'decode') and os.environ.get('NXD_CPU_MODE') == '1'
    import torch
    importlib.import_module('neuronx_distributed_inference.modules.lora_serving')
    from safetensors.torch import load_file
    from neuronx_distributed.parallel_layers import parallel_state
    from neuronx_distributed.utils.model_utils import init_on_device
    from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaForCausalLM, NeuronLlamaModel
    from neuronx_distributed_inference.models.model_wrapper import DecoderModelInstance
    assert Path(importlib.import_module(NeuronLlamaModel.__module__).__file__).resolve() == LLAMA
    assert NeuronLlamaForCausalLM._model_cls is NeuronLlamaModel
    assert torch.distributed.is_initialized() and torch.distributed.get_world_size() == TP
    assert torch.distributed.get_rank() == rank and torch.distributed.get_backend() == 'gloo'
    assert parallel_state.model_parallel_is_initialized()
    assert parallel_state.get_tensor_model_parallel_size() == TP
    assert parallel_state.get_tensor_model_parallel_rank() == rank
    config = configuration(phase)
    config_before = _serialized(config)
    instance = DecoderModelInstance(NeuronLlamaModel, config)
    # Leave original small rotary buffers on CPU; meta buffers cannot be restored
    # from these shards because they are nonpersistent and absent from state_dict.
    with init_on_device(torch.device('meta'), include_buffers=False, force_custom_init_on_device=True):
        instance.load_module()
    model = instance.module
    assert type(model) is NeuronLlamaModel and not model.training
    assert _serialized(config) == config_before, 'Model construction changed compiler config'
    assert all(value.device.type == 'meta' for value in model.parameters())
    check_state_metadata(_tensor_metadata(model.state_dict()), 'model')
    assert model.kv_mgr.global_rank is model.rank_util
    assert model.kv_mgr.global_rank.rank is model.rank_util.rank
    path = _regular(ARTIFACT / 'weights' / f'tp{rank}_sharded_checkpoint.safetensors')
    state = load_file(str(path), device='cpu')
    shard_metadata = _tensor_metadata(state)
    check_state_metadata(shard_metadata, 'shard')
    assert all(value.device.type == 'cpu' and value.is_contiguous() for value in state.values())
    rank_values = {name: state[name].tolist() for name in sorted(RANK_KEYS)}
    assert rank_values == {name: [rank] for name in sorted(RANK_KEYS)}
    loaded = model.load_state_dict(state, strict=False, assign=True)
    assert sorted(loaded.missing_keys) == sorted(EXPECTED_MISSING)
    assert not loaded.unexpected_keys
    # The official KV manager shares the rank MODULE, so assign=True replaces its
    # parameter through both names. No repair or additional state key is needed.
    assert model.kv_mgr.global_rank is model.rank_util
    assert model.kv_mgr.global_rank.rank is model.rank_util.rank
    actual = model.state_dict(keep_vars=True)
    for name, value in state.items():
        target = actual[name]
        assert target.device.type == 'cpu' and target.dtype == value.dtype
        assert list(target.shape) == list(value.shape)
        # Byte comparison preserves distinctions such as NaN encodings and -0.
        assert torch.equal(target.detach().contiguous().view(torch.uint8), value.view(torch.uint8)), name
    for index, parameter in enumerate(model.kv_mgr.past_key_values):
        assert parameter.device.type == 'meta' and tuple(parameter.shape) == CACHE_SHAPE
        assert parameter.dtype == torch.bfloat16 and not parameter.requires_grad
        model.kv_mgr.past_key_values[index] = torch.nn.Parameter(
            torch.zeros(CACHE_SHAPE, dtype=torch.bfloat16, device='cpu'), requires_grad=False)
    assert len(model.kv_mgr.past_key_values) == CACHES_PER_RANK
    actual = model.state_dict(keep_vars=True)
    proof = dict(phase=phase, rank=rank, shard=str(path),
                 missing_keys=sorted(loaded.missing_keys), unexpected_keys=sorted(loaded.unexpected_keys),
                 shard_metadata=shard_metadata, model_metadata=_tensor_metadata(actual),
                 rank_values={name: actual[name].tolist() for name in sorted(RANK_KEYS)},
                 rank_alias_value=actual[RANK_ALIAS].tolist(),
                 shared_rank_module_identity=model.kv_mgr.global_rank is model.rank_util,
                 shared_rank_parameter_identity=model.kv_mgr.global_rank.rank is model.rank_util.rank,
                 state_dict_shared_parameter_identity=actual[RANK_ALIAS] is actual['rank_util.rank'],
                 alias_repair_performed=False, all_shard_values_raw_exact=True,
                 all_state_on_cpu=all(value.device.type == 'cpu' for value in actual.values()),
                 all_buffers_on_cpu=all(value.device.type == 'cpu' for value in model.buffers()),
                 all_cache_parameters_nontrainable=all(not value.requires_grad for value in model.kv_mgr.past_key_values),
                 learned_tensors=LEARNED_TENSORS, canonical_rank_tensors=len(RANK_KEYS),
                 cache_tensors=CACHES_PER_RANK,
                 configuration=dict(source=str(ARTIFACT / 'neuron_config.json'),
                                    all_finished_fields_preserved=True,
                                    official_stage_overrides=stage_overrides(phase),
                                    buckets=list(BUCKETS), is_continuous_batching=True,
                                    apply_seq_ids_mask=True, physical_cache_length=CACHE_SHAPE[2],
                                    logical_cache_length=4096, dtype='torch.bfloat16'),
                 official_model_class=f'{type(model).__module__}.{type(model).__name__}',
                 compiled=False, resharded=False, sdk_math_modified=False)
    check_load_metadata(proof, rank)
    assert model.lm_head.weight is not model.embed_tokens.weight
    assert len(model.layers) == LAYERS and all(name in dict(model.named_modules()) for name in stages())
    assert all(layer.is_prefill_stage is (phase == 'prefill') for layer in model.layers)
    assert model.kv_mgr.is_continuous_batching is True
    assert model.neuron_config.apply_seq_ids_mask is True
    assert all(not actual[name].requires_grad for name in RANK_KEYS | {RANK_ALIAS})
    assert _serialized(config) == config_before, 'State loading changed compiler config'
    # Keep no secondary model/shard owner or reference cycle on the returned model.
    del state, actual, instance
    return model, proof


def select_bucket(model, bucket):
    """Use installed DecoderModelInstance.get; only model.n_positions may change."""
    _guarded_runtime()
    assert type(bucket) is int and bucket in BUCKETS
    from neuronx_distributed_inference.models.model_wrapper import DecoderModelInstance
    before = _serialized(model.config)
    phase = 'prefill' if model.neuron_config.is_prefill_stage else 'decode'
    check_config_metadata(before, phase)
    active = model.n_active_tokens
    instance = DecoderModelInstance(type(model), model.config)
    instance.module = model
    selected, aliases = instance.get(BUCKETS.index(bucket))
    assert selected is model and model.n_positions == bucket
    assert model.n_active_tokens == active == stage_overrides(phase)['n_active_tokens']
    assert _serialized(model.config) == before, 'Bucket selection changed compiler config'
    caches = list(model.kv_mgr.past_key_values)
    assert {id(value): index for value, index in aliases.items()} == {
        id(value): index + 1 for index, value in enumerate(caches)}
    assert len(caches) == CACHES_PER_RANK
    return dict(bucket=bucket, bucket_index=BUCKETS.index(bucket), phase=phase,
                n_positions=model.n_positions, n_active_tokens=model.n_active_tokens,
                cache_aliases=CACHES_PER_RANK, configuration_unchanged=True)


def check_contract_controls():
    """Inert controls reject omitted learned/rank/cache keys and false alias proof."""
    shard = expected_state_metadata('shard')
    model = expected_state_metadata('model')
    check_state_metadata(shard, 'shard')
    check_state_metadata(model, 'model')
    good = dict(missing_keys=sorted(EXPECTED_MISSING), unexpected_keys=[],
                rank_values={name: [1] for name in sorted(RANK_KEYS)}, rank_alias_value=[1],
                shared_rank_module_identity=True, shared_rank_parameter_identity=True,
                state_dict_shared_parameter_identity=True, all_shard_values_raw_exact=True,
                all_state_on_cpu=True, all_buffers_on_cpu=True,
                all_cache_parameters_nontrainable=True, alias_repair_performed=False,
                shard_metadata=shard, model_metadata=model)
    check_load_metadata(good, 1)
    rejected = []

    def rejects(name, function):
        try:
            function()
        except AssertionError:
            rejected.append(name)
        else:
            raise AssertionError(('Negative control accepted', name))

    for name, kind, key in [('missing_learned', 'shard', 'norm.weight'),
                            ('missing_rank', 'shard', 'rank_util.rank'),
                            ('missing_cache', 'model', 'kv_mgr.past_key_values.83'),
                            ('missing_alias', 'model', RANK_ALIAS)]:
        value = expected_state_metadata(kind)
        del value[key]
        rejects(name, lambda value=value, kind=kind: check_state_metadata(value, kind))
    for name, kind, key, field, value in [
        ('rank_float', 'shard', 'rank_util.rank', 'dtype', 'torch.bfloat16'),
        ('learned_float32', 'shard', 'norm.weight', 'dtype', 'torch.float32'),
        ('unsharded_embedding', 'shard', 'embed_tokens.weight', 'shape', [VOCAB, HIDDEN]),
        ('lost_physical_padding', 'model', 'kv_mgr.past_key_values.0', 'shape', [4, 1, 4096, 128])]:
        bad = expected_state_metadata(kind)
        bad[key][field] = value
        rejects(name, lambda bad=bad, kind=kind: check_state_metadata(bad, kind))
    for key, value in [('missing_keys', sorted(CACHE_KEYS)), ('unexpected_keys', ['unrecognized']),
                       ('rank_alias_value', [0]), ('shared_rank_module_identity', False),
                       ('state_dict_shared_parameter_identity', False), ('all_shard_values_raw_exact', False)]:
        bad = copy.deepcopy(good)
        bad[key] = value
        rejects(key, lambda bad=bad: check_load_metadata(bad, 1))
    bad = copy.deepcopy(good)
    bad['rank_values']['layers.41.self_attn.rank_util.rank'] = [0]
    rejects('wrong_rank_value', lambda: check_load_metadata(bad, 1))
    assert len(rejected) == 15 and len(stages()) == 213
    return dict(passed=True, learned_tensors=LEARNED_TENSORS, shard_keys=SHARD_KEYS,
                model_keys=MODEL_KEYS, cache_tensors=CACHES_PER_RANK,
                prediction_stages=len(stages()), negative_controls=rejected,
                native_imported=False, model_executed=False, payload_read=False)
