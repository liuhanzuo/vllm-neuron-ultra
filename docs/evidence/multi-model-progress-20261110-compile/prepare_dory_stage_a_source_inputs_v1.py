"""Read original pinned JSON into unbound Stage A source inputs only.

No output writes, checkpoint opens, model imports, spec/admission, serial lease,
actual-exit record or launch. Historical runtime metadata is read solely to
locate the already-pinned original loader proof; its old admission and checkpoint
inode/path are never copied into these inputs or accepted as current admission.
"""
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import stat

from dory_stage_a_capture_core_v1 import bind_cases, WIDE, CACHE_IDS, sha
from dory_stage_a_artifact_transaction_v2 import _plan, digest, STAGE_A_BYTES, SCOPE_BYTES
from dory_stage_a_archive_core_v1 import require
from dory_stage_a_saved_files_v1 import identity

Q = Path(__file__).resolve().parent
DESIGN = dict(path=str(Q/'dory-targeted-mixer-residual-design-20261110-v1.json'),
              sha256='cb1a5b9be6733e8bff370bbb70db17b3d148bc64c188fd6cb640e863aeee55fe', bytes=976742)
LOADER_SHA = '61e20085087318d79b94e98fac6f82ada245818a2f61723de4b4afa888659cf8'
FAILURES = [3607, 4105, 4113, 6160, 6164, 6176, 6181]
GATES = dict(cosine_min=0.995, max_abs_max=2.0, KL_max=0.02, relaxations_allowed=False)
FULL_SHA = '207483664dfef54c4c4438863260d3ba423bf3d6009b43fe84264b281e9fda23'
MAX_JSON_BYTES = 2 * 1024**2


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'No duplicate metadata keys')
        result[key] = value
    return result


def read_pinned_json(ref, *, guard):
    require(callable(guard) and type(ref) is dict and set(ref) <= {'path','sha256','bytes'}
            and {'path','sha256'} <= set(ref), 'Explicit original JSON pin and guard')
    sha(ref['sha256'])
    path = Path(ref['path'])
    require(path.suffix == '.json' and path.is_absolute() and path.resolve() == path,
            'Canonical original JSON only')
    guard()
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid() and before.st_nlink == 1
                and 0 < before.st_size <= MAX_JSON_BYTES
                and ('bytes' not in ref or type(ref['bytes']) is int and ref['bytes'] == before.st_size),
                'Bounded complete owned original JSON')
        blocks, consumed = [], 0
        while consumed < before.st_size:
            guard()
            block = os.read(fd, min(65536, before.st_size-consumed))
            require(bool(block), 'No truncated original JSON')
            blocks.append(block); consumed += len(block)
        require(os.read(fd,1) == b'', 'No appended original JSON')
        guard()
        require(identity(os.fstat(fd)) == identity(before)
                and identity(path.stat(follow_symlinks=False)) == identity(before)
                and path.resolve() == path, 'Stable complete original JSON identity')
        raw = b''.join(blocks)
        require(hashlib.sha256(raw).hexdigest() == ref['sha256'], 'Exact original JSON SHA256')
        return json.loads(raw, object_pairs_hook=unique,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))
    finally:
        os.close(fd)


def validate_source_inputs(prepared):
    require(type(prepared) is dict and set(prepared) == {'schema','source_only','runtime_bound',
            'runtime_admitted','actual_exit_or_review_claimed','design','loader_source_sha256',
            'original_references','cases','case_set_sha256','previous_loader_tensors','expected_model',
            'original_failures','quality_gates','scope_bytes','stage_a_bytes','input_sha256'},
            'Exact source-input schema without injected runtime fields')
    require(prepared['schema'] == 'dory_stage_a_source_inputs_v1'
            and prepared['source_only'] is True and prepared['runtime_bound'] is False
            and prepared['runtime_admitted'] is False
            and prepared['actual_exit_or_review_claimed'] is False
            and prepared['design'] == DESIGN and prepared['loader_source_sha256'] == LOADER_SHA,
            'Original source inputs; never a production binding or admission')
    body = {k:v for k,v in prepared.items() if k != 'input_sha256'}
    require(prepared['input_sha256'] == digest(body), 'Unchanged prepared source-input content')
    cases = prepared['cases']
    _plan(cases)
    require(prepared['case_set_sha256'] == digest(cases) and prepared['scope_bytes'] == SCOPE_BYTES
            and prepared['stage_a_bytes'] == STAGE_A_BYTES
            and prepared['original_failures'] == FAILURES and prepared['quality_gates'] == GATES,
            'Original complete cases, budgets, failures and numeric gates')
    rows = prepared['previous_loader_tensors']
    require(isinstance(rows,list) and len(rows) == 507 and len({r['name'] for r in rows}) == 507
            and sum(r['bytes'] for r in rows) == 7950458880, 'All original507 BF16 tensors')
    for row in rows:
        require(set(row) == {'name','shape','dtype','bytes','sha256'} and row['dtype'] == 'BF16'
                and type(row['name']) is str and bool(row['name']) and type(row['shape']) is list
                and len(row['shape']) in (1,2) and all(type(x) is int and x > 0 for x in row['shape'])
                and type(row['bytes']) is int and row['bytes'] == 2*math.prod(row['shape']),
                'Original complete BF16 loader tensor descriptors')
        sha(row['sha256'])
    expected = prepared['expected_model']
    require(type(expected) is dict and set(expected) == {'config','physical_parameters','parameters',
                'backends','cache_ids','rotary_tables'}
            and type(expected['config']) is dict
            and type(expected['physical_parameters']) is int and expected['physical_parameters'] == 399
            and type(expected['parameters']) is int and expected['parameters'] == 3975229440
            and type(expected['backends']) is dict and len(expected['backends']) == 182
            and all(type(name) is str and type(names) is list and all(type(x) is str for x in names)
                    for name,names in expected['backends'].items())
            and all('onednn_mm' in names for names in expected['backends'].values())
            and type(expected['cache_ids']) is list and all(type(x) is int for x in expected['cache_ids'])
            and expected['cache_ids'] == list(CACHE_IDS)
            and expected['rotary_tables'] == [dict(shape=[262144,256],dtype='torch.bfloat16',base=10000),
                                            dict(shape=[262144,256],dtype='torch.bfloat16',base=10000000)],
            'Original official physical model/backend/cache/RoPE requirements')
    require(not any(k in prepared for k in ('checkpoint','checkpoint_identity','output','admission','spec',
                                           'producer_exit','reviewer_exit','runtime_spec')),
            'No new production binding or future terminal evidence')
    return prepared


def prepare_source_inputs(*, guard):
    """Callable metadata preparation only, returns in memory; no CLI or writes."""
    require(callable(guard), 'Explicit source observation guard')
    references = []
    def read(ref):
        result = read_pinned_json(ref, guard=guard)
        references.append(deepcopy(ref))
        return result
    design = read(DESIGN)
    selection = read(design['prior_evidence']['selection_plan'])
    prior_runtime = read(design['prior_evidence']['previous_runtime_spec'])
    loader_path = prior_runtime['previous_loader_proof']
    loader_ref = dict(path=loader_path,sha256=prior_runtime['source_sha256'][loader_path])
    prior_loader = read(loader_ref)['loader']
    require(prior_runtime['source_sha256'][str(Q/'dory_checkpoint_loader_v1.py')] == LOADER_SHA
            and prior_loader['loader_source_sha256'] == LOADER_SHA
            and prior_loader['full_file_hash_verified'] is True
            and prior_loader['full_sha256'] == FULL_SHA
            and prior_loader['file_bytes_read'] == 7950515472
            and prior_loader['weight_bytes'] == 7950458880
            and prior_loader['parameters'] == 3975229440, 'Original complete checkpoint proof only')
    model = read(design['prior_evidence']['model_metadata'])
    chosen = {r[0] for r in WIDE}
    manifest = [x for x in design['selected_call_manifest'] if x['event'] in chosen]
    originals = {x['event']:read(x['original_event']) for x in manifest}
    replays = {x['event']:read(x['residual_event']) for x in manifest}
    cases = bind_cases(design, selection, originals, replays)
    require([x['event'] for x in design['selected_call_manifest'] if x['original_pointwise_failed']] == FAILURES,
            'Original seven failures preserved')
    body = dict(schema='dory_stage_a_source_inputs_v1',source_only=True,runtime_bound=False,
        runtime_admitted=False,actual_exit_or_review_claimed=False,design=deepcopy(DESIGN),
        loader_source_sha256=LOADER_SHA,original_references=references,cases=list(cases),
        case_set_sha256=digest(cases),previous_loader_tensors=deepcopy(prior_loader['tensors']),
        expected_model=dict(config=deepcopy(model['config']),physical_parameters=399,parameters=3975229440,
            backends=deepcopy(model['backends']),cache_ids=deepcopy(model['cache_ids']),
            rotary_tables=deepcopy(model['rotary_tables'])),
        original_failures=list(FAILURES),quality_gates=dict(GATES),
        scope_bytes=dict(SCOPE_BYTES),stage_a_bytes=STAGE_A_BYTES)
    body['input_sha256'] = digest(body)
    return validate_source_inputs(body)


if __name__ == '__main__':
    raise RuntimeError('Source preparation API only; no production binding, spec, admission or runner')
