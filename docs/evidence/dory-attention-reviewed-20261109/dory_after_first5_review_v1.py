"""Authenticate the completed prefix/review snapshot; never rerun old discovery."""
import copy
from pathlib import Path

from dory_layer0_serial_contract_v2 import read_pinned, require
import review_dory_layer0_serial_snapshot_v1 as prior

Q = Path(__file__).resolve().parent
REVIEW_EXIT = dict(path=str(Q/'dory-first5-combined-saved-review-actual-exit-20261109-v1.json'),
    sha256='abcd0a2c8af3b344b7f9a895872416569da0386472b0109d6d21e13484ad6536')


def load(ref):
    return read_pinned(ref['path'], ref['sha256'])


def completed():
    receipt = load(REVIEW_EXIT)
    require(receipt['reviewer'] == '/root' and receipt['actual_external_exit_observed'] is True
            and (receipt['session_id'], receipt['exit_code'], receipt['tool_chunk']) == (46992, 0, 'bc36db'),
            'Actual independent prefix review completed')
    spec, admission, result = [load(receipt[k]) for k in ('review_spec', 'admission', 'result')]
    review = load(receipt['review'])
    require(admission['owner'] == receipt['owner'] == result['owner']
            and admission['spec_sha256'] == receipt['review_spec']['sha256']
            and result['complete'] and review['complete']
            and review['original_calls'] == 6185 and review['original_marked_frames'] == 87
            and review['original_pointwise_failures'] == 7
            and review['total_raw_output_bytes_verified'] == 191883264
            and review['checkpoint_review']['checkpoint_bytes_read'] == 7950515472
            and review['recorded_source_checkpoint_exit_bound'] is True,
            'Complete unchanged original prefix provenance and failed model acceptance')
    history = load(admission['predecessor_history'])
    require(len(history['rows']) == 1096 and len(history['processes']) == 287,
            'Complete actual prefix-review history')
    native = load(spec['native_exit'])
    require(native['actual_external_exit_observed'] is True
            and (native['session_id'], native['exit_code'], native['tool_chunk']) == (44682, 0, '460472')
            and history['processes'][-2:] == [native['owner'], native['child_owner']],
            'Both actual prefix owners retained')
    known = set(history['known_admitted_directories']) | {Path(spec['review_output']).name}
    base = Path(spec['base'])
    actual = {p.name for p in base.iterdir() if p.is_dir() and (p/'admission.json').is_file()}
    require(len(known) == 31 and actual == known, 'Every completed admission, no unknown new job')
    history = copy.deepcopy(history)
    history['processes'].append(receipt['owner'])
    history['known_admitted_directories'] = sorted(known)
    history['completed_first5_review'] = REVIEW_EXIT
    history['current_physical_release_verified'] = False
    return history, receipt, spec


def drain(history):
    require(len(history['rows']) == 1096 and len(history['processes']) == 288,
            'All actual completed services and processes')
    return prior.physical_drain(history, [])
