"""Exact shared-rank alias contract; stdlib-only and independent of runtime imports."""
LAYERS = 42
LEARNED_TENSORS = 381
CACHE_KEYS = frozenset(f'kv_mgr.past_key_values.{i}' for i in range(84))
RANK_KEYS = frozenset({'rank_util.rank'} | {f'layers.{i}.self_attn.rank_util.rank' for i in range(LAYERS)})
ALIAS_KEY = 'kv_mgr.global_rank.rank'
EXPECTED_MISSING = CACHE_KEYS | {ALIAS_KEY}


def observe(model, loaded):
    """Collect exact names/identity metadata before accepting any missing key."""
    state = model.state_dict(keep_vars=True)
    module = model.kv_mgr.global_rank
    rank = module.rank
    return dict(actual_missing_keys=sorted(loaded.missing_keys),
        actual_unexpected_keys=sorted(loaded.unexpected_keys), model_state_keys=sorted(state),
        model_state_key_count=len(state), shared_rank_module_identity=module is model.rank_util,
        shared_rank_parameter_identity=rank is model.rank_util.rank,
        state_dict_shared_parameter_identity=state.get(ALIAS_KEY) is state.get('rank_util.rank') is rank,
        rank_alias_dtype=str(rank.dtype), rank_alias_shape=list(rank.shape),
        rank_alias_requires_grad=rank.requires_grad, rank_alias_device=rank.device.type,
        loaded_meta_parameter_values_verified=False)


def validate(value, learned_keys, canonical_rank_keys):
    """Accept exactly 84 absent caches and one proven alias; no wildcard filters."""
    learned_keys, canonical_rank_keys = set(learned_keys), set(canonical_rank_keys)
    assert len(learned_keys) == LEARNED_TENSORS and canonical_rank_keys == RANK_KEYS
    assert not learned_keys & (RANK_KEYS | CACHE_KEYS | {ALIAS_KEY})
    assert value['actual_missing_keys'] == sorted(EXPECTED_MISSING), value['actual_missing_keys']
    assert len(value['actual_missing_keys']) == 85
    assert value['actual_unexpected_keys'] == [], value['actual_unexpected_keys']
    expected_state = learned_keys | RANK_KEYS | CACHE_KEYS | {ALIAS_KEY}
    assert len(expected_state) == 509
    assert value['model_state_keys'] == sorted(expected_state)
    assert value['model_state_key_count'] == 509
    for key in ('shared_rank_module_identity', 'shared_rank_parameter_identity', 'state_dict_shared_parameter_identity'):
        assert value[key] is True, key
    assert value['rank_alias_dtype'] == 'torch.int32' and value['rank_alias_shape'] == [1]
    assert value['rank_alias_requires_grad'] is False and value['rank_alias_device'] == 'meta'
    assert value['loaded_meta_parameter_values_verified'] is False


def synthetic(learned_keys):
    """Inert checker fixture; never claimed as actual meta execution."""
    return dict(actual_missing_keys=sorted(EXPECTED_MISSING), actual_unexpected_keys=[],
        model_state_keys=sorted(set(learned_keys) | RANK_KEYS | CACHE_KEYS | {ALIAS_KEY}),
        model_state_key_count=509, shared_rank_module_identity=True,
        shared_rank_parameter_identity=True, state_dict_shared_parameter_identity=True,
        rank_alias_dtype='torch.int32', rank_alias_shape=[1], rank_alias_requires_grad=False,
        rank_alias_device='meta', loaded_meta_parameter_values_verified=False)
