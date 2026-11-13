"""Finalize the actual r3 audit, preserving every previous catalog row."""
import argparse
import copy
import datetime
import json
from pathlib import Path
import re
import sys

R = Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
Q = R / 'research'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-terminal', required=True)
    parser.add_argument('--audit-session-id', required=True, type=int)
    parser.add_argument('--outer-session-id', required=True, type=int)
    parser.add_argument('--outer-terminal', required=True)
    parser.add_argument('--outer-exit-code', required=True, type=int)
    args = parser.parse_args()
    assert re.fullmatch('[a-f0-9]{3,64}', args.audit_terminal)
    assert re.fullmatch('[a-f0-9]{3,64}', args.outer_terminal)
    assert args.audit_session_id > 0 and args.outer_session_id > 0
    assert sys.dont_write_bytecode and __debug__
    sys.path[:0] = [str(Q), str(R), '/home/ubuntu']
    from qwen35_writer_geometry_serial_v3 import run as stage
    from qwen35_sampler_runtime_io_v12 import read, ref, once, RetainedLock, process_api
    from prepare_qwen35_sampler_r6_candidate_v1 import catalog_shape
    from audit_qwen35_sampler_r10_terminal_root_20261112_v2 import verify_current_terminal, terminal_unit
    from research import minicpm5_layer1_operator_sdk_runtime_completion_v4 as latest
    from qwen35_package_full_r5_post_exit_20261112_v1 import source_delta, predicted

    binding_ref = ref(Q / 'qwen35-writer-geometry-r3-root-binding-20261113-v1.json')
    binding = read(binding_ref)
    live_ref = ref(Q / 'qwen35-writer-geometry-r3-root-live-20261113-v1.json')
    live = read(live_ref)
    assert live['root_binding'] == binding_ref
    assert binding['reviewer'] == '/root' and binding['passed']
    assert binding['full_source_history_resolution_passed'] and binding['catalog_rows'] == 164
    assert binding['compatibility_source_review']['sha256'] == '11fdc94f311e15ff01d78bcbd8262b0c02a5e750e9c8ca8031eb3b501db79f28'
    spec_ref, review_ref = binding['spec'], binding['review']
    spec = stage.load_spec(spec_ref, review_ref)
    exit_ref = ref(Q / 'qwen35-writer-geometry-writer-r3-20261113-root-actual-exit.json')
    ended = read(exit_ref)
    actual = dict(kind='tool_session', session_id=args.outer_session_id,
                  terminal_chunk=args.outer_terminal, exit_code=args.outer_exit_code)
    assert live['actual_tool'] == dict(kind='tool_session', session_id=args.outer_session_id)
    assert ended['schema'] == 'qwen35_writer_geometry_root_actual_exit_v1'
    assert ended['reviewer'] == '/root' and ended['actual_tool'] == actual
    assert ended['spec'] == spec_ref and ended['review'] == review_ref
    assert ended['owners_gone'] and ended['physically_drained'] and ended['original_lock_held_during_audit']
    assert ended['lock_identity'] == stage.LOCK_IDENTITY
    assert ended['full_model_qualified'] is False and ended['new_speedup_claim'] is False
    qualified = ended['geometry_qualified']
    assert type(qualified) is bool and ended['reconciliation']['passed'] is qualified
    counts = tuple(ended[k] for k in ('completed_layouts', 'completed_calls', 'compiled_graphs'))
    assert all(type(x) is int and x >= 0 for x in counts)
    assert all(x <= limit for x, limit in zip(counts, (8, 40000, 32)))
    if qualified:
        assert args.outer_exit_code == 0 and counts == (8, 40000, 32)
    else:
        assert args.outer_exit_code != 0
    assert ended['auditor_source'] == ref(Q / 'qwen35_writer_geometry_audit_v5.py')
    assert ended['auditor_source']['sha256'] == '562ba7b655c331d59948812d68dca8fb0e29f22a4bd04234f95db82f5d2913ac'
    qualification_path = Q / 'qwen35-writer-geometry-writer-r3-20261113-root-qualification.json'
    qualification_ref = None
    if qualified:
        qualification_ref = ref(qualification_path)
        proof = read(qualification_ref)
        assert proof['schema'] == 'qwen35_writer_geometry_independent_qualification_v1'
        assert proof['reviewer'] == '/root' and proof['passed'] is True and proof['blockers'] == []
        assert proof['actual_exit'] == exit_ref and proof['reports'] == ended['reports']
        assert proof['source_sha256'] == ended['source_sha256'] and proof['full_model_qualified'] is False
    else:
        assert not qualification_path.exists() and not qualification_path.is_symlink()
    running_ref = ended['running_handoff']; running = read(running_ref)
    assert running_ref['path'] == spec['running_handoff_path'] and running_ref == live['running_handoff']
    tip = latest.latest_handoff(); assert tip['path'] == running_ref['path']
    old = read(spec['current_serial_catalog']); catalog_shape(old)
    predecessor = read(ref(Q / 'qwen35-writer-geometry-r2-root-finalization-20261112-v1.json'))
    assert spec['current_serial_catalog'] == predecessor['catalog'] and len(old['runs']) == 164
    assert old['runs'][-1]['tool'] == dict(kind='tool_session', session_id=72363, terminal_chunk='129c0c', exit_code=1)
    overlay = read(spec['history_overlay'])
    external_paths = [Path(__file__).resolve(),
        Path('/home/ubuntu/ack_qwen35_writer_geometry_root_r3_20261113_v1.py'),
        Path('/home/ubuntu/accept_qwen35_writer_geometry_root_r3_20261113_v2.py'),
        Path('/home/ubuntu/prepare_qwen35_writer_geometry_r3_request_20261113_v1.py')]
    external_refs = [ref(p) for p in external_paths]
    delta, composition = source_delta(spec['source_sha256'], overlay['source_extension'], old,
        {r['path']: r['sha256'] for r in external_refs})
    terminal = read(ended['outer_terminal'])
    audit_tool = dict(kind='tool_session', session_id=args.audit_session_id,
                      terminal_chunk=args.audit_terminal, exit_code=0)
    idle_path = Q / 'continuation-handoff-20261113-v596-qwen-writer-geometry-r3-ended.json'
    cat_path = Q / 'minicpm5-operator-post-r2-serial-workloads-root-catalog-20261113-v118.json'
    receipt_path = Q / 'qwen35-writer-geometry-r3-root-finalization-20261113-v1.json'
    assert all(not p.exists() and not p.is_symlink() for p in (idle_path, cat_path, receipt_path))
    lock = RetainedLock(); lock.acquire()
    try:
        observations = []
        p = process_api()
        for _ in range(2):
            lock.check(); assert latest.latest_handoff() == tip
            p.active_heavy()
            for owner in ended['owners']: p.gone(owner)
            observed = {}
            for saved in terminal['complete_owned_unit_terminals']:
                unit = terminal_unit(saved); props = stage.properties(unit)
                verify_current_terminal(saved, props, p)
                observed[unit] = dict(properties=props,
                    recursive_group=p.empty_group(p.GROUP_PREFIX + unit + '.service'))
            assert set(observed) == set(ended['units'])
            observations.append(observed)
        assert observations[0] == observations[1]
        for row in external_refs: assert ref(row['path']) == row
        assert ref(exit_ref['path']) == exit_ref and ref(running_ref['path']) == running_ref
        if qualification_ref is not None:
            assert ref(qualification_path) == qualification_ref
        else:
            assert not qualification_path.exists() and not qualification_path.is_symlink()
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        idle = copy.deepcopy(running)
        idle.update(updated_at=now, previous=running_ref, previous_handoff=running_ref,
            running=False, active_serial=None, active_heavy_units=[], live_tool_sessions=[],
            goal_status='active', this_turn_classification='progress',
            complete_current_serial_catalog=spec['current_serial_catalog'],
            complete_current_serial_catalog_next=str(cat_path),
            qwen35_sampler_current_execution_ended=True,
            qwen35_sampler_current_execution_actual_exit=exit_ref,
            publication_current_status_stale=True, new_speedup_claim=False,
            next_serial_work='Review actual geometry result and preserve every failed predecessor. Mini needs exact completed writer-source compatibility; package v8 needs fresh binding and original storage checks.')
        idle['qwen35_writer_geometry_current'] = dict(actual_exit=exit_ref, actual_tool=actual,
            independent_auditor_tool=audit_tool, status='geometry_qualified' if qualified else 'failed',
            completed_layouts=counts[0], completed_calls=counts[1], compiled_graphs=counts[2],
            geometry_qualified=qualified, qualification=qualification_ref, full_model_qualified=False, new_speedup_claim=False)
        idle_ref = predicted(idle_path, idle)
        catalog = copy.deepcopy(old)
        catalog.update(current_handoff=idle_ref, runs=old['runs'] + [dict(
            terminal=exit_ref, tool=actual, owners=ended['owners'], units=ended['units'], source_sha256=delta)])
        assert catalog['runs'][:-1] == old['runs']; catalog_shape(catalog)
        cat_ref = predicted(cat_path, catalog)
        receipt = dict(schema='qwen35_writer_geometry_root_finalization_v1', reviewer='/root',
            observed_at=now, actual_exit=exit_ref, actual_outer_tool=actual,
            actual_independent_auditor_tool=audit_tool, source_composition=composition,
            external_sources=external_refs, complete_old_catalog=spec['current_serial_catalog'],
            catalog=cat_ref, idle_handoff=idle_ref, two_current_drain_observations=observations,
            all_prior164_rows_unchanged=True, total_rows=165, original_8MiB_cap_preserved=True,
            original_lock_identity=stage.LOCK_IDENTITY, geometry_qualified=qualified, qualification=qualification_ref,
            full_model_qualified=False, new_speedup_claim=False)
        receipt_ref = predicted(receipt_path, receipt)
        lock.check(); assert latest.latest_handoff() == tip
        assert once(idle_path, idle) == idle_ref
        assert once(cat_path, catalog) == cat_ref
        assert once(receipt_path, receipt) == receipt_ref
    finally:
        lock.release(True)
    print(json.dumps(dict(actual_exit=exit_ref, idle_handoff=idle_ref, catalog=cat_ref, finalization=receipt_ref)))


if __name__ == '__main__':
    main()
