"""Pinned input/trace contract for a future serial Spark native diagnostic.

This is an intermediate tensor diagnostic, not the full model qualification.
Import is stdlib-only. Metadata reads are explicit; no payloads or native code.
"""
import copy
import hashlib
import json
from pathlib import Path

import spark26_native_contract_v1 as original

ROOT = original.ROOT
Q = ROOT / 'research'
METADATA = Q / 'spark26-original-weight-metadata-20261112-v1'
METADATA_PINS = {
    'model-info.json': 'fdeccb42b4efcfa9b281e09f2f20592948d8c27c2aec10a569b62e94f58c19d0',
    'model.safetensors.index.json': '77a11d49b279b5e44fcb1f2453f71a8023237f93992fe5275d842e149d896edd',
}
PYTHON = '/home/ubuntu/venvs/vllm_neuron/bin/python'
LOCK_IDENTITY = {'device': 66306, 'inode': 8294421}
LOGIT_CRITERIA = {'cosine_min': .995, 'kl_max_nats': .02, 'max_abs_error_max': 2.,
                  'argmax_match_fraction_min': .9,
                  'all_reference_margins_gt_0.5_must_match': True}
LOGIT_VALIDATOR = {'path': str(ROOT / 'validate_real_logits.py'),
                   'sha256': '1f77e72f1cfcd8d6ee7253daea78e8d598cb7c07d9c98acc7ece9e1e4b41161f'}
NATIVE_SOURCES = {
    str(ROOT / 'spark26_text_port_v2.py'): '8eae6d77a3017d33ce9882cb2c98907e2057c4877597e7ac1d7edec86ed91399',
    str(original.MODELING): original.PINS[str(original.MODELING)],
    str(original.CONFIGURATION): original.PINS[str(original.CONFIGURATION)],
    '/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/neuronx_distributed/trace/trace.py':
        'c5aff7fc700a89505b7698314e7f9daf3527057ac5090b50a1b3541b074bef2e',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def fixed_metadata():
    values = {}
    for name, sha in METADATA_PINS.items():
        raw = (METADATA / name).read_bytes()
        require(len(raw) < 1024**2 and hashlib.sha256(raw).hexdigest() == sha,
                'Exact bounded original repository metadata')
        values[name] = json.loads(raw)
    info, index = values['model-info.json'], values['model.safetensors.index.json']
    require(info['id'] == original.MODEL_ID and info['sha'] == original.REVISION,
            'Original official model and revision')
    expected = set(original.state_schema()) - {'lm_head.weight'}
    require(set(index['weight_map']) == expected and len(expected) == 290,
            'All original tensors; only tied lm_head omitted from safetensors')
    require(index['metadata'] == {'total_parameters': original.UNIQUE_PARAMETERS,
                                  'total_size': 8224158720}, 'Original tensor storage')
    names = set(index['weight_map'].values())
    files = {}
    for row in info['siblings']:
        if row['rfilename'] in names:
            require(row['size'] == row['lfs']['size'], 'Original shard LFS size')
            files[row['rfilename']] = {'sha256': row['lfs']['sha256'], 'bytes': row['size']}
    require(len(files) == 5 and sum(row['bytes'] for row in files.values()) == 8224192408,
            'All five original weight shards')
    return {'model': original.MODEL_ID, 'revision': original.REVISION,
            'metadata_sha256': dict(METADATA_PINS), 'weight_map': index['weight_map'],
            'files': files, 'tied_head_source': 'model.embedding.weight'}


def cache_trace():
    """B1 prefill and B4 physical decode; real slots deliberately unordered."""
    trace = []
    seen, epoch = {}, {}
    def prefill(slot, length, generation):
        seen[slot], epoch[slot] = length, generation
        trace.append({'op': 'prefill', 'slots': [slot], 'positions': [0],
                      'lengths': [length], 'epochs': [generation]})
    def decode(slots):
        trace.append({'op': 'decode', 'slots': list(slots),
                      'positions': [seen[s] for s in slots],
                      'lengths': [1] * len(slots), 'epochs': [epoch[s] for s in slots]})
        for slot in slots:
            seen[slot] += 1
    for slot, length in ((3, 3), (1, 510), (0, 511), (2, 513)):
        prefill(slot, length, 0)
    for slots in ((3, 1), (2, 0, 3), (1,), (2, 3, 0, 1)):
        decode(slots)
    for _ in range(520):
        decode((3,))
    prefill(2, 1, 1)
    decode((2, 0))
    prefill(1, 7, 2)
    decode((3, 1, 2, 0))
    prefill(0, 4095, 3)
    decode((0,))
    require(len(trace) == 534 and all(0 <= p < 4096 for row in trace for p in row['positions']),
            'Finite original context trace')
    return trace


def plan():
    metadata = fixed_metadata()
    weights = [{'name': f'model.layers.{layer}.self_attn.q_k_v_proj.weight',
                'layer': layer, 'shape': [6144, 2560], 'dtype': 'torch.bfloat16',
                'shard': metadata['weight_map'][f'model.layers.{layer}.self_attn.q_k_v_proj.weight']}
               for layer in range(36)]
    return {'schema': 'spark26_native_port_diagnostic_plan_v1',
            'model': original.MODEL_ID, 'revision': original.REVISION,
            'checkpoint': metadata, 'qkv_weights': weights,
            'projection_shapes': [[1, 1, 2560], [1, 17, 2560], [4, 1, 2560]],
            'projection_input': 'BF16 exact codes ((arange + layer*7) % 29 - 14) / 64; synthetic hidden inputs',
            'tp_ranks': [0, 1], 'qkv_full_rows': [4096, 1024, 1024],
            'qkv_local_rows': [2048, 512, 512],
            'cache_layers': [{'layer': 0, 'kind': 'sliding_attention', 'capacity': 511},
                             {'layer': 3, 'kind': 'full_attention', 'capacity': 4096}],
            'cache_trace': cache_trace(), 'physical_batch': 4, 'local_kv_heads': 2,
            'head_dim': 256, 'cache_dtype': 'torch.bfloat16',
            'cache_and_weight_partition_gate': 'exact logical bytes and shape/dtype',
            'projection_numerics': 'Record BF16 differences; exactness is a diagnostic category, not a new model gate',
            'original_logit_criteria': dict(LOGIT_CRITERIA), 'original_logit_validator': dict(LOGIT_VALIDATOR),
            'full_model_validation_pending': ['original logits32', 'fresh-process cold-cache',
                'long-decode', 'slot reuse/isolation', 'quality96', 'matched HTTP'],
            'full_model_qualified': False, 'new_speedup_claim': False}


def validate_plan(value):
    require(value == plan(), 'Complete original diagnostic plan; no smaller cases or changed thresholds')


def validate_authority(value, expected_plan, checkpoint_root):
    """Shape only; actual live ownership and native authentication are host work.

    Every call to the host guard must revalidate its original retained lock,
    current owned service and complete authenticated native/source environment.
    An arbitrary dict matching this shape is not runtime authorization.
    """
    require(type(value) is dict and value.get('schema') == 'spark26_native_port_diagnostic_authority_v1',
            'Explicit future authenticated host authority')
    require(value.get('model') == original.MODEL_ID and value.get('revision') == original.REVISION
            and value.get('plan_sha256') == digest(expected_plan)
            and value.get('checkpoint_root') == str(checkpoint_root)
            and value.get('native_python') == PYTHON and value.get('lock_identity') == LOCK_IDENTITY,
            'Exact future model, plan, checkpoint, interpreter and original lock')
    require(value.get('native_sources_authenticated') is True and value.get('native_libraries_authenticated') is True
            and value.get('actual_owned_serial_worker') is True,
            'External native authentication and serial worker still required')
    require(all(value.get('source_sha256', {}).get(p) == sha for p, sha in NATIVE_SOURCES.items()),
            'Original candidate, official and SDK sources retained')
    require(Path(checkpoint_root).is_absolute(), 'Explicit absolute original checkpoint directory')
    return copy.deepcopy(value)
