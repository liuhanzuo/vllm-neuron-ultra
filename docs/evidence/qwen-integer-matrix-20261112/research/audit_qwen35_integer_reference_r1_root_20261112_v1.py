"""Reconcile an actual external terminal and retain the complete matrix outcome.

No native imports. Preserve old catalog rows and prove exact source composition
before storing only sources not already present in the inherited catalog/base.
"""
import argparse
import copy
import datetime
import hashlib
import json
from pathlib import Path
import sys

R = Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
Q = R / 'research'
sys.path[:0] = [str(Q), str(R)]
import qwen35_integer_reference_serial_v1 as stage
from qwen35_integer_reference_serial_contract_v1 import expected_calls
from qwen35_sampler_runtime_io_v12 import read, ref, once, process_api, RetainedLock
from qwen35_sampler_runtime_history_v12 import expected_running
from audit_qwen35_sampler_r10_terminal_root_20261112_v2 import verify_current_terminal, terminal_unit
from prepare_qwen35_sampler_r6_candidate_v1 import catalog_shape
from research import minicpm5_layer1_operator_sdk_runtime_completion_v4 as latest

assert sys.dont_write_bytecode and __debug__
parser = argparse.ArgumentParser()
parser.add_argument('--session-id', type=int, required=True)
parser.add_argument('--terminal-chunk', required=True)
parser.add_argument('--exit-code', type=int, required=True)
args = parser.parse_args()
actual = dict(kind='tool_session', session_id=args.session_id,
              terminal_chunk=args.terminal_chunk, exit_code=args.exit_code)
binding_ref = ref(Q / 'qwen35-integer-reference-r1-root-binding-20261112-v1.json')
binding = read(binding_ref)
spec = read(binding['spec'], 64 * 2**20)
stage.validate(spec, read(binding['review'], 64 * 2**20), binding['spec'])
root = Path(spec['output_root'])
admission_ref = ref(root / 'admission.json'); admission = read(admission_ref)
terminal_ref = ref(root / 'outer-terminal.json'); terminal = read(terminal_ref)
running_ref = ref(spec['running_handoff_path']); running = read(running_ref)
live_ref = ref(Q / 'qwen35-integer-reference-r1-root-live-20261112-v1.json'); live = read(live_ref)
owner_ref = ref(root / 'supervisor-owner.json'); owner = read(owner_ref)
assert live['actual_tool'] == dict(kind='tool_session', session_id=args.session_id)
assert admission['outer_owner'] == terminal['outer_owner'] == live['outer_owner']
assert admission['spec'] == binding['spec'] and admission['review'] == binding['review']
assert running == expected_running(read(admission['freshness']['latest_handoff']), running['updated_at'],
    live['actual_tool'], admission_ref, admission, owner_ref, owner)
assert latest.latest_handoff()['path'] == running_ref['path']
ack = read(ref(root / 'running-ack.json'))
assert ack['running']['reference'] == running_ref and ack['running']['value'] == running
assert ack['running']['observation']['path'] == running_ref['path']
assert ack['admission'] == admission_ref and ack['coordinator'] == owner
ends = {terminal_unit(t): t for t in terminal['complete_owned_unit_terminals']}
assert set(ends) == set(admission['units'].values())
owners = [admission['outer_owner'], owner['process']]
layouts = []
counts = dict(correctness=0, warmup=0, measured=0, reset=0)
graphs = 0
lock = RetainedLock(); lock.acquire()
try:
    p = process_api()
    for name in stage.LAYOUTS:
        work = root / name
        if not work.exists(): continue
        native_ref = ref(work / 'native-process-exit.json') if (work / 'native-process-exit.json').exists() else None
        native = read(native_ref) if native_ref else None
        if native is not None:
            assert native['popen_wait_observed'] and native['self_exit_claimed'] is False
            if 'parent_observer' in native:
                assert native['parent_observer'] == owner['process']
            else:
                assert native['interrupted_or_failed'] is True
            if native['owner'] is not None:
                assert native['owner']['ppid'] == owner['process']['pid']
                assert native['owner']['cgroup'] == owner['process']['cgroup']
                owners.append(native['owner'])
        if (work / 'native-owner.json').exists():
            observed_owner = read(ref(work / 'native-owner.json'))
            assert observed_owner['ppid'] == owner['process']['pid']
            assert observed_owner['cgroup'] == owner['process']['cgroup']
            if native is not None: assert native['owner'] == observed_owner
            if observed_owner not in owners: owners.append(observed_owner)
        if native is None or native['owner'] is None:
            assert actual['exit_code'] != 0 and terminal['failed']
        artifacts = []
        for mode in stage.protocol.MODES:
            for variant in stage.protocol.VARIANTS:
                artifact_path = work / ('artifact-%s-%s.json' % (mode, variant))
                if artifact_path.exists():
                    artifacts.append(stage.artifact_record(read(ref(artifact_path)), name, work))
        graphs += len(artifacts)
        row = dict(layout=name, native_exit=native_ref,
                   native_returncode=native['actual_returncode'] if native else None,
                   native_log=ref(work / 'native.log') if (work / 'native.log').exists() else None,
                   report=None, summary=None, compiled_graphs=len(artifacts), artifacts=artifacts)
        if (work / 'report.json').exists():
            report_ref = ref(work / 'report.json', 16 * 2**20); report = read(report_ref, 16 * 2**20)
            assert report['schema'] == 'qwen35_integer_reference_primitive_result_v1'
            assert report['case'] == stage.protocol.layout(name)
            assert report['reference_implementation'] == stage.protocol.REFERENCE_IMPLEMENTATION
            assert report['required_resources'] == stage.protocol.RESOURCES
            assert report['timing_scope'] == stage.protocol.TIMING_SCOPE
            assert all(report[k] is False for k in ('full_model_qualified', 'model_speedup_claim', 'lfm105_tasks_replaced'))
            assert report['source_sha256'] == stage.protocol.FROZEN_SOURCES
            assert [g['artifact'] for g in report['graphs']] == artifacts[:len(report['graphs'])]
            expected = list(expected_calls())
            assert len(report['checks']) <= len(expected)
            for check, want in zip(report['checks'], expected):
                assert all(check[k] == v for k, v in want.items())
                assert check['passed'] == all(item['passed'] for item in check['checks'].values())
            for phase in counts:
                assert type(report['runtime_counts'][phase]) is int
                assert 0 <= report['runtime_counts'][phase] <= stage.protocol.counts()['per_layout'][phase + '_calls']
                assert report['runtime_counts'][phase] >= sum(c['phase'] == phase for c in report['checks'])
                counts[phase] += report['runtime_counts'][phase]
            row.update(report=report_ref, complete=report['complete'], passed=report['passed'],
                       runtime_counts=report['runtime_counts'], compiled_graphs=len(artifacts),
                       artifacts=artifacts, failure=report.get('error'),
                       failed_checks=[c for c in report['checks'] if not c['passed']], timing=report['timing'])
            if report['complete'] and native is not None and native['actual_returncode'] == 0 and (work / 'leaf-result.json').exists():
                row['summary'] = stage.validate_layout(root, name, read(ref(work / 'leaf-result.json')), native)
        layouts.append(row)
    assert [r['layout'] for r in layouts] == list(stage.LAYOUTS)[:len(layouts)]
    observations = []
    for _ in range(2):
        p.active_heavy()
        for current in owners: p.gone(current)
        observed = {}
        for unit, saved in ends.items():
            props = stage.properties(unit); assert p.terminal(props)
            verify_current_terminal(saved, props, p)
            observed[unit] = dict(properties=props, recursive_group=p.empty_group(p.GROUP_PREFIX + unit + '.service'))
        observations.append(observed)
    assert observations[0] == observations[1]
    if actual['exit_code'] == 0:
        result = read(ref(root / 'supervisor-result.json'))
        stage.audit_results(root, result)
        assert len(layouts) == 8 and all(r['summary'] is not None for r in layouts)
        assert result['layouts'] == [r['summary'] for r in layouts]
    else:
        assert terminal['failed']
    reconciliation = stage.reconcile(terminal, running, actual, actual_owner_gone=True, actual_service_drained=True)
    receipt_path = Q / 'qwen35-integer-reference-r1-root-actual-exit-20261112-v1.json'
    receipt_value = dict(schema='qwen35_integer_reference_root_actual_exit_v1', reviewer='/root',
        observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), actual_tool=actual,
        actual_tool_exit_observed=True, actual_outer_exit_observed=True, root_live=live_ref, root_binding=binding_ref,
        spec=binding['spec'], review=binding['review'], admission=admission_ref, outer_terminal=terminal_ref,
        running_handoff=running_ref, reconciliation=reconciliation, owners=owners, owners_gone=True,
        units=sorted(ends), physically_drained=True, two_current_service_observations=observations,
        original_lock_reacquired_and_released=True, lock_identity=stage.LOCK_IDENTITY,
        layouts=layouts, runtime_counts=counts, runtime_calls=sum(counts.values()), compiled_graphs=graphs,
        completed_layouts=sum(r['summary'] is not None for r in layouts),
        primitive_qualified=reconciliation['passed'], failure=terminal['failure'], model_executed=False,
        full_model_qualified=False, new_speedup_claim=False, auditor_source=ref(Path(__file__).resolve()))
    # Exact composition: no previous row is changed or removed, and no source
    # already present in the retained catalog/base needs a duplicated hash row.
    overlay = read(spec['history_overlay']); oldcat = read(spec['current_serial_catalog'])
    recovered = {path: sha for path, sha in spec['source_sha256'].items() if path not in overlay['source_extension']}
    assert len(recovered) == 44532
    prior = dict(recovered)
    for row in oldcat['runs']:
        for path, sha in row['source_sha256'].items():
            assert path not in prior or prior[path] == sha
            prior[path] = sha
    assert all(spec['source_sha256'].get(path) == sha for path, sha in prior.items())
    delta = {path: sha for path, sha in spec['source_sha256'].items() if path not in prior}
    assert dict(prior, **delta) == spec['source_sha256']
    executed_sources = dict(spec['source_sha256'])
    for script in (Path(__file__).resolve(), Q / 'ack_qwen35_integer_reference_r1_root_20261112_v1.py'):
        source = ref(script)
        assert source['path'] not in executed_sources or executed_sources[source['path']] == source['sha256']
        executed_sources[source['path']] = source['sha256']
        if source['path'] not in prior: delta[source['path']] = source['sha256']
    assert dict(prior, **delta) == executed_sources
    receipt_value['source_composition'] = dict(recovered_sources=len(recovered), retained_prior_sources=len(prior),
        runtime_sources=len(spec['source_sha256']), post_audit_sources=len(executed_sources),
        new_row_sources=len(delta), all_previous_rows_retained=True, exact_full_mapping_reconstructed=True)
    def predicted(path, value):
        payload = (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
        assert len(payload) <= 8 * 2**20 and not path.exists()
        return dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest())
    receipt = predicted(receipt_path, receipt_value)
    idle = copy.deepcopy(running)
    catalog_path = Q / 'minicpm5-operator-post-r2-serial-workloads-root-catalog-20261112-v110.json'
    idle_path = Q / 'continuation-handoff-20261112-v581-qwen35-integer-reference-r1-ended.json'
    idle.update(updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), previous=running_ref,
        previous_handoff=running_ref, running=False, active_serial=None, live_tool_sessions=[], active_heavy_units=[],
        goal_status='active', this_turn_classification='progress', complete_current_serial_catalog=spec['current_serial_catalog'],
        complete_current_serial_catalog_next=str(catalog_path), publication_current_status_stale=True, new_speedup_claim=False)
    idle['qwen35_integer_reference_current'] = dict(actual_exit=receipt,
        status='complete_matrix_passed' if reconciliation['passed'] else 'matrix_failed',
        runtime_calls=sum(counts.values()), compiled_graphs=graphs, completed_layouts=receipt_value['completed_layouts'],
        primitive_qualified=reconciliation['passed'], full_model_qualified=False, new_speedup_claim=False)
    idle['qwen35_package_serial_runtime_v3'] = dict(source_review=ref(Q / 'qwen35-package-serial-runtime-v3-root-source-review-20261112-v1.json'), actual_full_run_pending=True)
    idle['next_serial_work'] = 'Review complete saved integer-reference matrix outcome; retain every failed path and original full model gates. Prepare exact package v3 binding after independent preparer v2 fixes and live storage recheck. Mini formal successor remains pending.'
    idle_ref = predicted(idle_path, idle)
    catalog = dict(oldcat, current_handoff=idle_ref, runs=oldcat['runs'] + [dict(
        terminal=receipt, tool=actual, owners=owners, units=sorted(ends), source_sha256=delta)])
    catalog_shape(catalog)
    assert catalog['runs'][:-1] == oldcat['runs']
    catalog_ref = predicted(catalog_path, catalog)
    assert once(receipt_path, receipt_value) == receipt
    assert once(idle_path, idle) == idle_ref
    assert once(catalog_path, catalog) == catalog_ref
finally:
    lock.release(True)
print(json.dumps(dict(actual_exit=receipt, catalog=catalog_ref, handoff=idle_ref, actual_tool=actual,
    counts=counts, graphs=graphs, completed_layouts=receipt_value['completed_layouts'], passed=reconciliation['passed'])))
