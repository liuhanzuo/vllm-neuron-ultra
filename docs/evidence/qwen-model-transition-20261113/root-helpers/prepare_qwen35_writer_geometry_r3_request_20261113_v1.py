"""Create the r3 metadata request only from the actual finalized r2 receipt.

This does not resolve History, acquire the original lock, or launch native work.
The unchanged preparer and complete concrete preflight remain mandatory.
"""
import hashlib
import json
from pathlib import Path
import sys

Q = Path('/home/ubuntu/experiments/trn1-latest-models-20261031/research')


def read_pin(name, sha=None):
    path = Q / name
    assert not path.is_symlink() and path.resolve() == path
    payload = path.read_bytes()
    assert len(payload) <= 8 * 2**20
    digest = hashlib.sha256(payload).hexdigest()
    assert sha is None or digest == sha
    return dict(path=str(path), sha256=digest), json.loads(payload)


def main():
    assert sys.dont_write_bytecode and __debug__
    _, future = read_pin('qwen35-writer-geometry-r2-prepare-request-20261112-v1.json')
    final_ref, final = read_pin('qwen35-writer-geometry-r2-root-finalization-20261112-v1.json')
    assert final['schema'] == 'qwen35_writer_geometry_root_finalization_v1'
    assert final['reviewer'] == '/root' and final['all_prior163_rows_unchanged']
    assert final['total_rows'] == 164 and final['original_8MiB_cap_preserved']
    actual = dict(kind='tool_session', session_id=72363, terminal_chunk='129c0c', exit_code=1)
    assert final['actual_outer_tool'] == actual
    assert final['actual_independent_auditor_tool']['session_id'] == 9791
    assert final['actual_independent_auditor_tool']['exit_code'] == 0
    for key in ('catalog', 'idle_handoff', 'actual_exit'):
        row = final[key]
        assert Path(row['path']).parent == Q
        observed, value = read_pin(Path(row['path']).name, row['sha256'])
        assert observed == row
        if key == 'catalog':
            catalog = value
        elif key == 'idle_handoff':
            idle = value
        else:
            ended = value
    assert catalog['current_handoff'] == final['idle_handoff']
    assert len(catalog['runs']) == 164
    assert catalog['runs'][-1]['terminal'] == final['actual_exit']
    assert catalog['runs'][-1]['tool'] == ended['actual_tool'] == actual
    assert ended['owners_gone'] and ended['physically_drained']
    assert not ended['geometry_qualified']
    assert (ended['completed_layouts'], ended['completed_calls'], ended['compiled_graphs']) == (1, 7865, 7)
    assert idle['running'] is False and idle['active_serial'] is None
    assert idle['active_heavy_units'] == idle['live_tool_sessions'] == []
    delivery_ref, delivery = read_pin('qwen35-geometry-audit-freshness-source-delivery-20261113-v5.json',
        'b64a8cb5cbcae14250ab9b45c5ab7e70eaeaab151c892b30f362f13679c5d2e8')
    review_ref, review = read_pin('qwen35-geometry-audit-v5-root-source-review-20261113-v1.json',
        '11fdc94f311e15ff01d78bcbd8262b0c02a5e750e9c8ca8031eb3b501db79f28')
    assert review['source_accepted'] and review['blockers'] == []
    assert review['source_delivery'] == delivery_ref
    assert delivery['schema'] == 'qwen35_writer_geometry_binding_source_delivery_v1'
    assert len(delivery['source_sha256']) == 657
    future.update(catalog=final['catalog'], idle_handoff=final['idle_handoff'],
        prerequisite_exit=final['actual_exit'], prerequisite_tool=actual,
        preparer_delivery=delivery_ref, preparer_source_review=review_ref,
        run_name='writer-r3-20261113')
    for key in ('candidate', 'overlay', 'review', 'review_draft', 'spec'):
        future['paths'][key] = future['paths'][key].replace('r2-20261112', 'r3-20261113')
    future['paths']['output'] = '/mnt/nvme/neuron-work/qwen35-writer-geometry-writer-r3-20261113'
    future['paths']['overlay_output'] = '/mnt/nvme/minicpm5-layer1-operator-native-writer-geometry-r3-20261113'
    future['paths']['running'] = str(Q / 'continuation-handoff-20261113-v595-qwen35-writer-geometry-r3-running.json')
    assert all(not Path(p).exists() and not Path(p).is_symlink() for p in future['paths'].values())
    # Full freshness, resources and original History are checked by the unchanged
    # prepare/accept stages. Never mistake this metadata construction for them.
    output = Q / 'qwen35-writer-geometry-r3-prepare-request-20261113-v1.json'
    payload = (json.dumps(future, sort_keys=True, ensure_ascii=False, indent=2) + '\n').encode()
    with output.open('xb') as stream:
        stream.write(payload)
    print(json.dumps(dict(request=dict(path=str(output), sha256=hashlib.sha256(payload).hexdigest()),
        actual_finalization=final_ref, full_preflight_pending=True, runtime_launched=False)))


if __name__ == '__main__':
    main()
