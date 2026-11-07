"""Inert contract for the full32 layer0 gate_up decode-row1 intervention."""
from pathlib import Path
from phi4_decode_batch_contract_v3 import CRITERIA, VARIANTS
from phi4_mixed_reference_protocol import frame_keys, replay_sha

ROOT = Path(__file__).resolve().parent
PROJECTION = 'layers.0.mlp.gate_up_proj'
BACKEND = 'dense_port'
ARMS = ('unchanged', 'wrapper_batch4', 'decode_row1_view')
REPLAY_SHA256 = 'afdd0744b78bf01191887edeaac85c5b76c5ad92de1ae8eae2f9ae261fd6bab8'
PHASES = ('admit', 'unchanged-instruct', 'unchanged-reasoning',
          'wrapper-instruct', 'wrapper-reasoning', 'control-admission',
          'candidate-instruct', 'candidate-reasoning', 'analyze')
PREFIX=('embed','norm','qkv_input','qkv_output','oproj_input','oproj_output',
        'post_attention_norm_input','post_attention_norm_output','gateup_input')
PROPAGATION={'gateup_output':16384,'down_input':8192,'down_output':3072,'layer0_output':3072,
             'layer1_norm':3072,'layer1_qkv':5120,'layer1_oproj_input':3072,'final_norm':3072}


def calls(fixtures):
    assert [row['id'] for row in fixtures] == list(range(6))
    assert all(len(row['continuation_ids']) == 3 for row in fixtures)
    slots = [3, 1, 0, 2]
    result = []
    for order in ([0, 5, 2, 4], [1, 3, 0, 2]):
        for fixture, slot in zip(order, slots):
            length = len(fixtures[fixture]['input_ids'])
            bucket = 1 << (max(128, length) - 1).bit_length()
            result.append(dict(kind='prefill', fixtures=[fixture], slots=[slot],
                input_shape=[1, bucket], logical_lengths=[length],
                keys=[dict(fixture=fixture, phase='prefill', slot=slot)]))
        for step in (1, 2, 3):
            result.append(dict(kind='decode', fixtures=list(order), slots=list(slots),
                input_shape=[4, 1], positions=[len(fixtures[f]['input_ids']) + step - 1 for f in order],
                keys=[dict(fixture=f, phase='decode', step=step, slot=s) for f,s in zip(order,slots)]))
    result.append(dict(kind='inactive', fixtures=[1], slots=[-1], input_shape=[1, 1], keys=[]))
    assert len(result) == 15 and [key for call in result for key in call['keys']] == frame_keys()
    return result


def dispatch(arm, kind, input_shape, slots):
    """Which unchanged bound Linear.forward calls the proposed wrapper may make.

    A None return means the original forward is called once with the original
    input object. Integer pairs are Python row-view slices in original order.
    The worker must provide an authenticated call context; shape alone is not
    a decode discriminator. Prefill and inactive calls remain direct passthrough.
    """
    assert arm in ARMS and kind in ('prefill', 'decode', 'inactive')
    assert len(input_shape) == 3 and input_shape[-1] == 3072
    assert all(type(x) is int and x > 0 for x in input_shape)
    assert len(slots) == input_shape[0] and all(type(s) is int for s in slots)
    if kind == 'prefill':
        assert input_shape[0] == 1 and input_shape[1] in (128, 2048) and 0 <= slots[0] < 4
    elif kind == 'inactive':
        assert input_shape == [1, 1, 3072] and slots == [-1]
    else:
        assert input_shape == [4, 1, 3072] and slots == [3, 1, 0, 2]
    if arm == 'decode_row1_view' and kind == 'decode':
        return [(i, i + 1) for i in range(4)]
    return None


def check():
    import json
    assert replay_sha() == REPLAY_SHA256
    assert CRITERIA == {'cosine_min':.995,'kl_max_nats':.02,'max_abs_error_max':2.,
        'argmax_match_fraction_min':.9,'all_reference_margins_gt_0.5_must_match':True}
    positive = negative = 0
    for variant in VARIANTS:
        fixtures = json.loads((ROOT/('phi4-'+variant)/'hf-reference.json').read_text())['fixtures']
        schedule = calls(fixtures)
        assert sum(len(call['keys']) for call in schedule) == 32
        assert [sum(call['kind']==kind for call in schedule) for kind in ('prefill','decode','inactive')] == [8,6,1]
        for arm in ARMS:
            invocation_count = altered_calls = 0
            for call in schedule:
                selected = dispatch(arm, call['kind'], [*call['input_shape'],3072],call['slots'])
                invocation_count += len(selected) if selected else 1
                altered_calls += selected is not None
            assert invocation_count == (33 if arm=='decode_row1_view' else 15)
            assert altered_calls == (6 if arm=='decode_row1_view' else 0)
            positive += 1
    for arguments in [('foreign','decode',[4,1,3072],[3,1,0,2]),
        ('decode_row1_view','prefill',[4,1,3072],[3,1,0,2]),
        ('decode_row1_view','decode',[4,2,3072],[3,1,0,2]),
        ('decode_row1_view','decode',[1,1,3072],[3]),
        ('decode_row1_view','decode',[4,1,1536],[3,1,0,2]),
        ('decode_row1_view','decode',[4,1,3072],[0,1,2,3]),
        ('decode_row1_view','inactive',[1,1,3072],[0]),
        ('decode_row1_view','decode',[4,1,3072],[3,1,-1,2])]:
        try: dispatch(*arguments)
        except AssertionError:negative+=1
        else:raise AssertionError('Invalid intervention scope admitted')
    return dict(passed=True,source_only=True,positive=positive,rejected=negative,
        original32_frames_and15_calls_per_variant=True,three_arms_both_variants=True,
        intervention_only_six_decode_calls=True,intervention_original_linear_invocations=33,
        original_replay_ast_preserved=True,native_execution=False,payloads_read=False,
        runnable_worker_implemented=True,prepared=False,enrolled=False)


if __name__=='__main__':
    import json
    print(json.dumps(check(),indent=2))
