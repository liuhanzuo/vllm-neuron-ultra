"""Meta-harness allocation context; never patches an installed SDK file.

The original GQA helper runs unchanged. Only its non-None meta-input call gets
a temporary default meta device. Conversion and sharding retain their original
allocation context, including the real CPU int32 rank values.
"""
from contextlib import contextmanager
from unittest.mock import patch


@contextmanager
def scoped_meta_padding(torch, gqa):
    original = gqa._maybe_pad_interleaved
    evidence = dict(meta_calls=[], cpu_passthrough_calls=0, none_passthrough_calls=0,
                    helper_restored=False, production_sdk_modified=False)

    def call(tensor, pad_dim, source_heads, target_heads, source_group_size):
        args = (tensor, pad_dim, source_heads, target_heads, source_group_size)
        if tensor is None:
            evidence['none_passthrough_calls'] += 1
            return original(*args)
        if tensor.device.type != 'meta':
            assert tensor.device.type == 'cpu', tensor.device
            evidence['cpu_passthrough_calls'] += 1
            return original(*args)
        assert tensor.dtype == torch.bfloat16
        before = list(tensor.shape)
        with torch.device('meta'):
            result = original(*args)
        assert result.device.type == 'meta' and result.dtype == tensor.dtype
        evidence['meta_calls'].append(dict(input_shape=before, output_shape=list(result.shape),
            pad_dim=pad_dim, source_heads=source_heads, target_heads=target_heads,
            source_group_size=source_group_size, dtype=str(result.dtype), device='meta'))
        return result

    try:
        with patch.object(gqa, '_maybe_pad_interleaved', call):
            yield evidence
    finally:
        evidence['helper_restored'] = gqa._maybe_pad_interleaved is original
        assert evidence['helper_restored']


def rank_values(state, keys, torch, expected):
    values = {}
    for name in sorted(keys):
        value = state[name]
        assert value.device.type == 'cpu' and value.dtype == torch.int32, name
        assert list(value.shape) == [len(expected)] and value.tolist() == expected, name
        values[name] = value.tolist()
    return values


def runtime_controls(torch, gqa):
    """Run only inside the future admitted SDK worker; inputs contain 24 values."""
    original = gqa._maybe_pad_interleaved
    assert torch.zeros(1).device.type == 'cpu'
    cases = []
    for dim, shape in ((0, (8, 3)), (1, (3, 8))):
        for target in (4, 8):
            tensor = torch.arange(24, dtype=torch.bfloat16).reshape(shape)
            reference = original(tensor, dim, 4, target, 2)
            rank = torch.arange(0, 2, dtype=torch.int32)
            with scoped_meta_padding(torch, gqa) as evidence:
                actual = gqa._maybe_pad_interleaved(tensor, dim, 4, target, 2)
                meta = torch.empty(shape, dtype=torch.bfloat16, device='meta')
                shaped = gqa._maybe_pad_interleaved(meta, dim, 4, target, 2)
                assert gqa._maybe_pad_interleaved(None, dim, 4, target, 2) is None
                assert torch.equal(actual, reference) and actual.device.type == 'cpu'
                assert list(shaped.shape) == list(reference.shape) and shaped.device.type == 'meta'
                # A factory after the helper call must still allocate real CPU values.
                assert torch.arange(0, 2, dtype=torch.int32).tolist() == [0, 1]
                assert rank.device.type == 'cpu' and rank.tolist() == [0, 1]
            assert evidence['helper_restored'] and gqa._maybe_pad_interleaved is original
            assert torch.zeros(1).device.type == 'cpu'
            cases.append(dict(pad_dim=dim, source_heads=4, target_heads=target,
                source_group_size=2, input_shape=list(shape), output_shape=list(reference.shape),
                cpu_original_equal=True, meta_shape_equal=True, cpu_rank_values=[0, 1],
                helper_restored=True, default_device_restored=True))
    result = dict(passed=True, cases=cases, actual_cpu_and_meta_tensors=True,
        model_forward_executed=False, device_execution=False, production_sdk_modified=False)
    validate_controls(result)
    return result


def validate_controls(value):
    assert value['passed'] is True and value['actual_cpu_and_meta_tensors'] is True
    assert value['model_forward_executed'] is False and value['device_execution'] is False
    assert value['production_sdk_modified'] is False and len(value['cases']) == 4
    assert [(x['pad_dim'], x['target_heads']) for x in value['cases']] == [(0,4),(0,8),(1,4),(1,8)]
    for case in value['cases']:
        dim, target = case['pad_dim'], case['target_heads']
        shape = [8, 3] if dim == 0 else [3, 8]
        expected = list(shape); expected[dim] *= target // 4
        assert case['input_shape'] == shape and case['output_shape'] == expected
        assert case['source_heads'] == 4 and case['source_group_size'] == 2
        assert case['cpu_rank_values'] == [0, 1]
        for key in ('cpu_original_equal','meta_shape_equal','helper_restored','default_device_restored'):
            assert case[key] is True


def validate_stage(value, tp, rank, keys):
    assert value['helper_restored'] is True and value['production_sdk_modified'] is False
    assert value['allocation_context_scope'] == 'only_original_gqa_helper_meta_input_call'
    assert value['rank_values_before_sharding'] == {name:list(range(tp)) for name in sorted(keys)}
    assert value['rank_values_after_sharding'] == {name:[rank] for name in sorted(keys)}
    assert value['rank_tensors_cpu_int32_before_and_after'] is True
    assert value['cpu_passthrough_calls'] == 0
    assert value['none_passthrough_calls'] == (42 if tp == 2 else 0)
    calls = value['meta_calls']
    assert len(calls) == (84 if tp == 2 else 0)
    assert sum(x['pad_dim'] == 0 for x in calls) == (42 if tp == 2 else 0)
    assert sum(x['pad_dim'] == 1 for x in calls) == (42 if tp == 2 else 0)
    for row in calls:
        assert row['device'] == 'meta' and row['dtype'] == 'torch.bfloat16'
        assert row['source_heads'] == row['target_heads'] == 16
        assert row['source_group_size'] == 8
        assert row['input_shape'] == row['output_shape'] == [2048, 2048]


def synthetic_controls():
    cases = []
    for dim, shape in ((0, [8,3]), (1, [3,8])):
        for target in (4,8):
            output = list(shape); output[dim] *= target // 4
            cases.append(dict(pad_dim=dim, source_heads=4, target_heads=target, source_group_size=2,
                input_shape=shape, output_shape=output, cpu_original_equal=True, meta_shape_equal=True,
                cpu_rank_values=[0,1], helper_restored=True, default_device_restored=True))
    return dict(passed=True, cases=cases, actual_cpu_and_meta_tensors=True,
        model_forward_executed=False, device_execution=False, production_sdk_modified=False)


def synthetic_stage(tp, rank, keys):
    calls = [dict(input_shape=[2048,2048], output_shape=[2048,2048], pad_dim=dim,
        source_heads=16, target_heads=16, source_group_size=8, dtype='torch.bfloat16', device='meta')
        for _ in range(42) for dim in (0,1)] if tp == 2 else []
    return dict(meta_calls=calls, cpu_passthrough_calls=0, none_passthrough_calls=42 if tp == 2 else 0,
        helper_restored=True, production_sdk_modified=False,
        allocation_context_scope='only_original_gqa_helper_meta_input_call',
        rank_values_before_sharding={name:list(range(tp)) for name in sorted(keys)},
        rank_values_after_sharding={name:[rank] for name in sorted(keys)},
        rank_tensors_cpu_int32_before_and_after=True)
