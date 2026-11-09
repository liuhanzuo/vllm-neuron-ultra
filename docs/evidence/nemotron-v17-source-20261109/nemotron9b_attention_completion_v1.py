"""Bind the completed attention audit as an explicit extension to the old catalog.

This is metadata validation, not a new physical drain or runtime admission.
Original base source/payload partitioning remains intact. All r8 and later
Dory owners are included in the extension; missing identities stay missing.
"""
import copy
import hashlib
import json
from pathlib import Path

import nemotron9b_post_dory_history_v1 as base_history
from nemotron9b_package_metadata_v16 import stable_bytes

ROOT = Path(__file__).resolve().parent
Q = ROOT/'research'
BASE = Path('/mnt/nvme/dory-official-cpu-20261108-v1')
EXIT = dict(path=str(Q/'dory-attention-factors-saved-review-actual-exit-20261109-v1.json'),
            sha256='2f8c008db3ba7bb3a10f52364b57cac1df658ae2965442a5cb8252ecbc66401e')
require = base_history.require
MISSING_ACK_UNIT = base_history.MISSING_ACK_UNIT


def completed():
    pins = {}

    def load(ref):
        raw = stable_bytes(ref['path'], 8*2**20)
        require(hashlib.sha256(raw).hexdigest() == ref['sha256'], 'Exact completed attention metadata')
        pins[ref['path']] = ref['sha256']
        return json.loads(raw)

    receipt = load(EXIT)
    require(receipt['reviewer'] == '/root' and receipt['actual_external_exit_observed'] is True
            and (receipt['session_id'], receipt['exit_code'], receipt['tool_chunk']) == (75402, 0, '2b7e26'),
            'Actual successful independent attention review exit')
    spec, admission, result = [load(receipt[k]) for k in ('spec', 'admission', 'result')]
    require(admission['owner'] == result['owner'] == receipt['owner']
            and admission['spec_sha256'] == receipt['spec']['sha256']
            and result['complete'] is True and result['original_row_shape_exact'] == 249
            and result['full_query_selected_rows_exact'] == 198
            and result['raw_output_bytes_verified'] == 25141248
            and result['original_seven_failures_preserved'] is True
            and result['full_model_qualified'] is result['new_speedup_claim'] is False,
            'Independent completed attention scope, original failures retained')
    history = load(admission['predecessor_history'])
    require(len(history['rows']) == history['complete_count'] == 1096
            and len(history['processes']) == 289,
            'Every original predecessor service and known owner')
    native = load(spec['native_exit'])
    native_admission = load(native['admission'])
    native_spec = load(native['spec'])
    require(native['actual_external_exit_observed'] is True
            and (native['session_id'], native['exit_code'], native['tool_chunk']) == (99315, 0, 'c0c997')
            and native_admission['owner'] == native['owner'] == history['processes'][-1]
            and native_admission['spec_sha256'] == native['spec']['sha256'],
            'Actual native attention exit and its already recorded owner')
    require(receipt['owner'] not in history['processes'], 'Review owner appended exactly once')
    failed = load(receipt['failed_preflight'])
    require(spec['failed_metadata_preflight'] == receipt['failed_preflight']
            and failed['actual_external_exit_observed'] is True and failed['exit_code'] == 1
            and failed['owner_start_recorded'] is failed['serial_lock_acquired'] is False
            and failed['output_created'] is failed['native_started'] is False,
            'Preflight failure retained without inventing an owner')
    outputs = [Path(native_spec['output']), Path(spec['output'])]
    require(all(p.parent == BASE for p in outputs)
            and [p.name for p in outputs] == ['attention-factors-r1', 'attention-factors-review-r1'],
            'Exact completed attention directories')
    known = set(history['known_admitted_directories'])
    require(len(known) == 31 and not known.intersection(p.name for p in outputs),
            'Explicit two-directory extension of original snapshot')
    known.update(p.name for p in outputs)
    actual = {p.name for p in BASE.iterdir() if p.is_dir() and (p/'admission.json').is_file()}
    require(actual == known and len(known) == 33, 'No unaccounted new Dory admission')
    history = copy.deepcopy(history)
    history['processes'].append(receipt['owner'])
    history['known_admitted_directories'] = sorted(known)
    history['completed_attention_review'] = dict(EXIT)
    history['attention_metadata_preflight_owner_missing'] = dict(receipt['failed_preflight'])
    history['current_physical_release_verified'] = False
    return history, pins


def validate_base(base):
    require(base == base_history.load(base_history.CATALOG),
            'Exact immutable completed base catalog required')
    base_history.finalizer(base, base_history.load(base_history.EXIT))
    history, _ = completed()
    require(history['base_catalog'] == base_history.CATALOG
            and history['rows'][:len(base['rows'])] == base['rows']
            and history['processes'][:len(base['processes'])] == base['processes']
            and len(base['rows']) == 1043 and len(base['processes']) == 275,
            'Complete authenticated extension of the unchanged base catalog')
    require(history['r8_outcome']['producer_qualified'] is False
            and history['r8_outcome']['failed_release'] is True,
            'Original r8 failed outcome remains failed')
    return history


def validate_spec(spec, base):
    require(spec.get('required_attention_completion') == EXIT,
            'Latest actual completed review required for any successor')
    history, pins = completed()
    validate_base(base)
    require(all(spec['source_sha256'].get(path) == sha for path, sha in pins.items()),
            'Every attention completion metadata pin in new source closure')
    return history
