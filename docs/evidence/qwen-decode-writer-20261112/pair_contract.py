"""Pure future writer-pair binding checks; no launch, graph load or qualification.

The existing sampler-only runner intentionally remains unchanged. A separate
writer runner must call these checks and preserve every original scientific gate.
"""
import copy
from pathlib import Path

ARMS = ('baseline', 'candidate')
MODES = ('torch_where', 'nki_predicated_decode_v1')
TREE_KEYS = ('artifact', 'compiler_cache', 'compiler_workdir')
BUCKETS = [128, 256, 512, 1024, 2048, 4096]


def require(value, message):
    if not value:
        raise ValueError(message)


def validate_pair(arms):
    """Reject sampler confounding, shared build trees and cross-source pairs."""
    require(type(arms) is dict and set(arms) == set(ARMS), 'Exactly two writer arms required')
    normalized = []
    paths = []
    samplers = []
    for arm, mode in zip(ARMS, MODES):
        value = arms[arm]
        require(type(value) is dict and set(value) == {'identity', 'sampler_fast', *TREE_KEYS},
                'Exact future writer-arm fields required')
        require(type(value['sampler_fast']) is str and value['sampler_fast'] in ('0', '1'),
                'Explicit unchanged CPU sampler setting required')
        samplers.append(value['sampler_fast'])
        identity = copy.deepcopy(value['identity'])
        writer = identity.get('cache_writer')
        require(type(writer) is dict and writer.get('schema') == 'qwen35-cache-writer-v1'
                and writer.get('mode') == mode and writer.get('decode_writer') == mode
                and writer.get('prefill_writer') == MODES[0]
                and writer.get('slots_dtype') == 'torch.int32'
                and writer.get('graphs_per_artifact') == {'prefill': 6, 'decode': 6},
                'Exact decode-only writer artifact identity required')
        writer['mode'] = writer['decode_writer'] = MODES[0]
        normalized.append(identity)
        for key in TREE_KEYS:
            path = Path(value[key])
            require(path.is_absolute() and str(path) == value[key] and '..' not in path.parts,
                    'Exact absolute independent build tree required')
            paths.append(path)
    require(samplers[0] == samplers[1], 'Writer comparison cannot change CPU sampler')
    # Import-free structural comparison is sufficient here: identities have
    # already passed the package's canonical SDK Infinity-aware validation.
    require(normalized[0] == normalized[1], 'Both writers require identical model/config/source/environment identity')
    require(len(set(paths)) == 6 and not any(a in b.parents for a in paths for b in paths if a != b),
            'Both arms require separate nonnested artifact/cache/workdir trees')
    return dict(pair_bindings_checked=True, sampler_fast=samplers[0],
                fresh_tree_checks_still_required=True, graphs_required_per_arm=12,
                full_model_qualified=False, new_speedup_claim=False)


def validate_graph_inventory(rows):
    """Check a future observed inventory's complete two-phase/six-bucket scope.

    The caller must separately authenticate files, routes, actual build exits,
    original states and every original cold/long/slot/quality/HTTP validator.
    """
    require(type(rows) is list and len(rows) == 12, 'Twelve actual graph records required per arm')
    seen = set()
    for row in rows:
        require(type(row) is dict and type(row.get('tag')) is str
                and type(row.get('index')) is int and type(row.get('bucket')) is int,
                'Exact graph tag/index/bucket types required')
        key = (row.get('tag'), row.get('index'), row.get('bucket'))
        require(key not in seen, 'Repeated graph cannot fill an unobserved bucket')
        seen.add(key)
    require(seen == {(tag, i, bucket) for tag in ('context_encoding_model', 'token_generation_model')
                     for i, bucket in enumerate(BUCKETS)}, 'Both complete six-bucket graph phases required')
