"""Add actual host-method and physical-row-order checks to the frozen plan.

All 534 original cache events and 216 QKV diagnostic rows remain unchanged.
This contract grants no native/runtime authority and imports only stdlib sources.
"""
import copy
import itertools

import spark26_native_port_diagnostic_contract_v1 as previous

original = previous.original
ROOT, Q, PYTHON = previous.ROOT, previous.Q, previous.PYTHON
LOCK_IDENTITY = previous.LOCK_IDENTITY
LOGIT_CRITERIA, LOGIT_VALIDATOR = previous.LOGIT_CRITERIA, previous.LOGIT_VALIDATOR
require, digest = previous.require, previous.digest
HOST_LOADER = '/home/ubuntu/src/vllm-neuron/vllm_neuron/worker/neuronx_distributed_model_loader.py'
NATIVE_SOURCES = {**previous.NATIVE_SOURCES,
    HOST_LOADER: '328ddd70fa0c47939400d6275612a37a790d82a0aa1d27421414b641e395c994'}
SNAPSHOT_STEPS = (4, 5, 6, 7)


def plan():
    result = copy.deepcopy(previous.plan())
    result['schema'] = 'spark26_native_port_diagnostic_plan_v2'
    result['original_plan_sha256'] = digest(previous.plan())
    result['host_reorder_check'] = {
        'provider': 'actual NeuronModelBase class imported from fixed vllm_neuron loader',
        'source': {'path': HOST_LOADER, 'sha256': NATIVE_SOURCES[HOST_LOADER]},
        'methods': ['_reordered', '_sort_inputs'],
        'calls': 'Every original decode event before physical B4 construction',
        'receiver': 'Minimal receiver with is_reorder_needed=True and actual static _sort_inputs',
        'input_fields': ['position_ids', 'key', 'value', 'request_markers'],
        'output_restore_check': 'Actual yielded restore over sorted request markers and K/V tensors',
        'not_full_serving_model_or_SDK_padding_execution': True,
    }
    result['physical_row_order_check'] = {
        'snapshots': [{'step': step, 'event': copy.deepcopy(result['cache_trace'][step]),
                       'active_rows': len(result['cache_trace'][step]['slots'])}
                      for step in SNAPSHOT_STEPS],
        'permutations': [list(row) for row in itertools.permutations(range(4))],
        'per_snapshot_calls': 24,
        'total_additional_calls': 4 * 24 * len(result['cache_layers']) * len(result['tp_ranks']),
        'input': 'Same original prior-cache snapshot; permute K/V, position and seq_ids together',
        'output': 'Inverse permutation restores all returned rows; persistent physical-slot caches unchanged',
        'dummy_policy': 'Exact same diagnostic rows after inverse permutation; never model-output qualification',
        'snapshot_retention': 'Check at original event, discard temporaries; no persistent tensors written',
    }
    return result


def validate_plan(value):
    require(value == plan(), 'Complete original cases plus exact host/order additions; no changed model gates')
    inherited = copy.deepcopy(value)
    for key in ('original_plan_sha256', 'host_reorder_check', 'physical_row_order_check'):
        inherited.pop(key)
    inherited['schema'] = 'spark26_native_port_diagnostic_plan_v1'
    previous.validate_plan(inherited)


def validate_authority(value, expected_plan, checkpoint_root):
    # Preserve the original concrete authority/identity checks and additionally
    # require the real host-method source in the authenticated source map.
    result = previous.validate_authority(value, expected_plan, checkpoint_root)
    require(all(result.get('source_sha256', {}).get(path) == sha for path, sha in NATIVE_SOURCES.items()),
            'Original and actual host loader sources authenticated before native imports')
    return result
