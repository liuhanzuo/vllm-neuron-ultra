"""Extend the authenticated attention snapshot with actual journal maintenance.

Lossless journal relocation is not model qualification or a new physical drain.
The old snapshot and its original missing identities remain unchanged.
"""
import copy
import hashlib
import json
from pathlib import Path

import nemotron9b_attention_completion_v1 as attention
from nemotron9b_package_metadata_v17 import stable_bytes

Q = Path(__file__).resolve().parent/'research'
EXIT = dict(path=str(Q/'closed-journals-compaction-actual-exit-20261109-v1.json'),
            sha256='8d1189bf301e91dc7dd283da7aa8683ad2fe4120290125313f941f26d27101e5')
require = attention.require


def completed():
    history, pins = attention.completed()

    def load(ref):
        raw = stable_bytes(ref['path'], 8*2**20)
        require(hashlib.sha256(raw).hexdigest() == ref['sha256'], 'Exact maintenance metadata')
        pins[ref['path']] = ref['sha256']
        return json.loads(raw)

    receipt = load(EXIT)
    require(receipt['actual_external_exit_observed'] is True and receipt['reviewer'] == '/root'
            and (receipt['session_id'], receipt['exit_code'], receipt['tool_chunk']) == (57954, 0, 'f89821'),
            'Actual successful maintenance exit')
    intent, result, plan, verified, review = [load(receipt[key])
        for key in ('intent', 'result', 'plan', 'verification', 'source_review')]
    predecessor = load(intent['predecessor_history'])
    require(predecessor == history and len(history['rows']) == 1096 and len(history['processes']) == 290,
            'Exact complete pre-maintenance history')
    require(intent['owner'] == result['owner'] == receipt['owner']
            and intent['plan'] == result['plan'] == review['plan'] == receipt['plan']
            and result['verification'] == receipt['verification']
            and review['passed'] is True and result['complete'] is True,
            'One admitted, reviewed and externally completed maintenance owner')
    require(all(result[key] is True for key in ('protected_trees_unchanged',
            'original_payloads_and_binaries_unchanged', 'canonical_unchanged', 'every_original_byte_retained'))
            and result['native_started'] is result['runtime_admitted'] is result['new_speedup_claim'] is False,
            'Preserved model/runtime/evidence scope')
    require(verified['every_archive_decompressed'] is True and len(verified['files']) == 16
            and result['original_bytes'] == 2147483648 and result['compressed_bytes'] == 251842473,
            'Complete lossless sixteen-journal result')
    require(plan['completion_exit'] == attention.EXIT and receipt['owner'] not in history['processes'],
            'Explicit unique post-attention owner')
    for path, digest in plan['source_sha256'].items():
        require(path not in pins or pins[path] == digest, 'Never rebind historical source')
        pins[path] = digest
    # Keep the action log as UTF8 provenance; no archived journal payload is read.
    action = receipt['actions']
    raw = stable_bytes(action['path'], 2*2**20)
    require(hashlib.sha256(raw).hexdigest() == action['sha256'], 'Exact lossless retirement actions')
    require(len(raw.splitlines()) == 16, 'Every exact retirement recorded')
    pins[action['path']] = action['sha256']
    for row in verified['files']:
        require(not Path(row['path']).exists() and Path(row['archive_path']).is_file()
                and Path(row['archive_path']).stat().st_size == row['gzip_bytes'],
                'Preserved local archive and retired original locations')
    history = copy.deepcopy(history)
    history['processes'].append(receipt['owner'])
    history['completed_journal_compaction'] = dict(EXIT)
    history['current_physical_release_verified'] = False
    return history, pins


def validate_base(base):
    previous = attention.validate_base(base)
    history, _ = completed()
    require(history['rows'] == previous['rows'] and history['processes'][:-1] == previous['processes']
            and len(history['processes']) == 291, 'Only the actual completed maintenance owner appended')
    return history


def validate_spec(spec, base):
    attention.validate_spec(spec, base)
    require(spec.get('required_journal_compaction') == EXIT, 'Latest lossless maintenance is required')
    history, pins = completed()
    validate_base(base)
    require(all(spec['source_sha256'].get(path) == digest for path,digest in pins.items()),
            'Complete maintenance source and metadata closure')
    return history
