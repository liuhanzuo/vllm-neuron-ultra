"""Independent root audit after an externally observed geometry tool exit.

The caller supplies the actual tool result, as in the reviewed integer-reference
auditor. Importing this module never selects a run, resolves History, acquires
the original lock, reads output payloads or changes the current catalog.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

Q = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(Q), str(Q.parent)]
from qwen35_writer_geometry_serial_v2 import run as stage
from qwen35_writer_geometry_serial_v1 import protocol, geometry
from qwen35_sampler_runtime_io_v12 import read, ref, once, process_api, RetainedLock
from qwen35_sampler_runtime_history_v12 import expected_running
from qwen35_package_preflight_history_v3 import History, Authority
from audit_qwen35_sampler_r10_terminal_root_20261112_v2 import verify_current_terminal, terminal_unit

require = stage.require


def latest_handoff():
    from research import minicpm5_layer1_operator_sdk_runtime_completion_v4 as latest
    return latest.latest_handoff()


def observe_layouts(root, owner, actual, terminal, guard):
    """Read every saved report/artifact, retaining failed and interrupted prefixes."""
    owners, layouts, reports = [], [], {}
    for name in stage.LAYOUTS:
        guard()
        work = root / name
        if not work.exists(): continue
        binding_ref = ref(work / 'layout-binding.json') if (work / 'layout-binding.json').exists() else None
        binding = read(binding_ref) if binding_ref else None
        if binding is not None:
            require(binding['schema'] == 'qwen35_writer_geometry_layout_binding_v1'
                    and binding['admission'] == terminal['admission']
                    and binding['layout'] == name and binding['supervisor'] == owner,
                    'Actual layout binding must name the observed supervisor')
        native_ref = ref(work / 'native-process-exit.json') if (work / 'native-process-exit.json').exists() else None
        native = read(native_ref) if native_ref else None
        if native is not None:
            require(binding_ref is not None and native['popen_wait_observed'] is True
                    and native['self_exit_claimed'] is False
                    and native['argv'] == stage.entry_argv('layout', binding_ref, work), 'Actual waited leaf command')
            if 'parent_observer' in native:
                require(native['parent_observer'] == owner['process'], 'Exact external parent observation')
            else:
                require(native['interrupted_or_failed'] is True, 'Original interrupted child receipt')
            if native['owner'] is not None:
                require(native['owner']['ppid'] == owner['process']['pid']
                        and native['owner']['cgroup'] == owner['process']['cgroup']
                        and native['owner']['argv'] == native['argv'], 'Actual leaf parent/cgroup/command')
                owners.append(native['owner'])
        if (work / 'native-owner.json').exists():
            observed = read(ref(work / 'native-owner.json'))
            require(observed['ppid'] == owner['process']['pid']
                    and observed['cgroup'] == owner['process']['cgroup'], 'Saved observed child identity')
            if native is not None: require(native['owner'] == observed, 'Receipt cannot replace observed child')
            if observed not in owners: owners.append(observed)
        if native is None or native['owner'] is None or binding is None:
            require(actual['exit_code'] != 0 and terminal['failed'] is True, 'Missing native receipt is failure only')
        artifacts = []
        for mode in protocol.MODES:
            for variant in protocol.VARIANTS:
                path = work / ('artifact-%s-%s.json' % (mode, variant))
                if path.exists(): artifacts.append(stage.artifact_record(read(ref(path)), name, work))
        order = [(mode, variant) for mode in protocol.MODES for variant in protocol.VARIANTS]
        require([(info['mode'], info['variant']) for info in artifacts] == order[:len(artifacts)],
                'Artifacts emitted before report updates still form the original compile prefix')
        row = dict(layout=name, binding=binding_ref, native_exit=native_ref,
            native_returncode=native['actual_returncode'] if native else None,
            native_log=ref(work / 'native.log') if (work / 'native.log').exists() else None,
            report=None, summary=None, artifacts=artifacts, compiled_graphs=len(artifacts),
            completed_calls=0, invocations_started=None, invocations_returned=None)
        if (work / 'report.json').exists():
            report_ref = ref(work / 'report.json', 16 * stage.MIB)
            report = read(report_ref, 16 * stage.MIB)
            saved = protocol.validate_prefix(report, name, artifacts)
            reports[name] = report_ref
            row.update(report=report_ref, complete=report['complete'],
                completed_calls=saved['calls'], original_raw_failures=saved['original_raw_failures'],
                required_failure_observed=saved['required_failure_observed'],
                invocations_started=report['invocations_started'], invocations_returned=report['invocations_returned'],
                failure=report.get('error'))
            # A complete calculation can precede SDK.close/guard/receipt failure.
            if report['complete'] and native is not None and native['actual_returncode'] == 0 and (work / 'leaf-result.json').exists():
                row['summary'] = stage.validate_layout(root, name, read(ref(work / 'leaf-result.json')), native)
        layouts.append(row)
    require([row['layout'] for row in layouts] == list(stage.LAYOUTS)[:len(layouts)], 'Only an ordered layout prefix')
    return dict(owners=owners, layouts=layouts, reports=reports)


def saved_binding(spec_ref, review_ref, spec, actual):
    root = Path(spec['output_root'])
    admission_ref = ref(root / 'admission.json'); admission = read(admission_ref)
    terminal_ref = ref(root / 'outer-terminal.json'); terminal = read(terminal_ref)
    running_ref = ref(spec['running_handoff_path']); running = read(running_ref)
    owner_ref = ref(root / 'supervisor-owner.json'); owner = read(owner_ref)
    require(admission['schema'] == 'qwen35_writer_geometry_serial_admission_v1'
            and admission['spec'] == spec_ref and admission['review'] == review_ref
            and admission['outer_owner'] == terminal['outer_owner']
            and admission['units'] == stage.names(spec)
            and admission['current_serial_catalog'] == spec['current_serial_catalog']
            and admission['history_overlay'] == spec['history_overlay']
            and admission['workload'] == protocol.workload(), 'Exact admitted geometry run')
    tool = dict(kind=actual['kind'], session_id=actual['session_id'])
    require(running == expected_running(read(admission['freshness']['latest_handoff']), running['updated_at'],
                tool, admission_ref, admission, owner_ref, owner), 'Exact actual external session/running binding')
    require(latest_handoff()['path'] == running_ref['path'], 'No unknown successor before independent audit')
    ack = read(ref(root / 'running-ack.json'))
    require(ack['running']['reference'] == running_ref and ack['running']['value'] == running
            and ack['running']['observation']['path'] == running_ref['path']
            and ack['admission'] == admission_ref and ack['coordinator'] == owner, 'Original complete running ACK')
    ends = {terminal_unit(t): t for t in terminal['complete_owned_unit_terminals']}
    require(set(ends) == set(admission['units'].values()), 'Every declared unit terminal retained')
    return dict(root=root, admission_ref=admission_ref, admission=admission, terminal_ref=terminal_ref,
                terminal=terminal, running_ref=running_ref, running=running, owner_ref=owner_ref, owner=owner, ends=ends)


def predicted(path, value):
    payload = (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
    require(len(payload) <= 8 * stage.MIB and not path.exists() and not path.is_symlink(), 'Fresh complete bounded audit publication')
    return dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest())


def audit(spec_ref, review_ref, actual):
    """Root supplies an already observed tool terminal; this does not launch a run."""
    spec = stage.load_spec(spec_ref, review_ref)
    saved = saved_binding(spec_ref, review_ref, spec, actual)
    root, admission, terminal, owner = (saved[k] for k in ('root', 'admission', 'terminal', 'owner'))
    exit_path = Q / ('qwen35-writer-geometry-' + spec['run_name'] + '-root-actual-exit.json')
    qualification_path = Q / ('qwen35-writer-geometry-' + spec['run_name'] + '-root-qualification.json')
    require(not exit_path.exists() and not qualification_path.exists(), 'Fresh independent audit outputs')
    lock = RetainedLock(); lock.acquire()
    try:
        def guard():
            lock.check()
            require(latest_handoff()['path'] == saved['running_ref']['path'], 'Running binding changed during audit')
        history = History(spec, spec_ref, review_ref, stage.names(spec), guard).resolve()
        retained = observe_layouts(root, owner, actual, terminal, guard)
        owners = [admission['outer_owner'], owner['process'], *retained['owners']]
        p = process_api(); auditor = p.proc(os.getpid())
        require(auditor is not None and auditor not in owners, 'Independent post-exit audit process')
        observations = []
        for _ in range(2):
            guard(); p.active_heavy()
            for current in owners: p.gone(current)
            observed = {}
            for unit, ended in saved['ends'].items():
                props = stage.properties(unit); require(p.terminal(props), 'Actual current unit terminal')
                verify_current_terminal(ended, props, p)
                observed[unit] = dict(properties=props, recursive_group=p.empty_group(p.GROUP_PREFIX + unit + '.service'))
            observations.append(observed)
        require(observations[0] == observations[1], 'Two stable independent complete drains')
        reconciliation = stage.reconcile(terminal, saved['running'], actual,
            actual_owner_gone=True, actual_service_drained=True)
        layouts, reports = retained['layouts'], retained['reports']
        if reconciliation['passed']:
            result = read(ref(root / 'supervisor-result.json'))
            stage.audit_results(root, result)
            require(len(layouts) == 8 and all(row['summary'] is not None for row in layouts)
                    and result['layouts'] == [row['summary'] for row in layouts], 'Eight completed independently audited actual leaves')
            geometry.check_complete_geometry({name: read(report, 16 * stage.MIB) for name, report in reports.items()})
            require(terminal.get('storage_error') is None and terminal['storage'] is not None, 'Successful terminal storage proof')
        else:
            require(terminal['failed'] is True, 'Failed actual outer exit remains failed')
        full_history = history.physical_pair(Authority(history, guard, lambda: []))
        usage, storage_error, storage_exception = stage.terminal_storage(root, None)
        require(not reconciliation['passed'] or storage_error is None, 'New storage failure prevents qualification')
        history.unchanged(); stage.loaded_sources(spec['source_sha256']); guard()
        sources = protocol.verify_sources()
        value = dict(schema='qwen35_writer_geometry_root_actual_exit_v1', reviewer='/root',
            observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), actual_tool=actual,
            actual_tool_exit_observed=True, actual_outer_exit_observed=True, spec=spec_ref, review=review_ref,
            admission=saved['admission_ref'], outer_terminal=saved['terminal_ref'], running_handoff=saved['running_ref'],
            actual_observer=auditor, reconciliation=reconciliation, owners=owners, owners_gone=True,
            units=sorted(saved['ends']), physically_drained=True, two_current_service_observations=observations,
            full_history_pair=full_history, original_lock_held_during_audit=True, lock_identity=stage.LOCK_IDENTITY,
            layouts=layouts, reports=reports, source_sha256=sources,
            completed_layouts=sum(row['summary'] is not None for row in layouts),
            completed_calls=sum(row['completed_calls'] for row in layouts),
            compiled_graphs=sum(row['compiled_graphs'] for row in layouts),
            geometry_qualified=reconciliation['passed'], failure=terminal['failure'],
            storage=usage, storage_error=storage_error, full_model_qualified=False, new_speedup_claim=False,
            catalog_append_by_root_required=True, auditor_source=ref(Path(__file__).resolve()))
        actual_exit = predicted(exit_path, value)
        qualified = None
        if reconciliation['passed']:
            qualified = dict(schema='qwen35_writer_geometry_independent_qualification_v1', reviewer='/root',
                passed=True, blockers=[], reports=reports, actual_exit=actual_exit,
                source_sha256=sources, full_model_qualified=False)
            qualification_ref = predicted(qualification_path, qualified)
        require(once(exit_path, value) == actual_exit, 'Exact actual-exit publication')
        if qualified is not None:
            require(once(qualification_path, qualified) == qualification_ref, 'Exact independent qualification publication')
        return dict(actual_exit=actual_exit, qualification=qualification_ref if qualified is not None else None,
                    passed=reconciliation['passed'], catalog_append_by_root_required=True, full_model_qualified=False)
    finally:
        # This audit creates no children or services. It only holds the same original lock.
        lock.release(True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('spec', 'spec-sha256', 'review', 'review-sha256', 'terminal-chunk'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--session-id', required=True, type=int)
    parser.add_argument('--exit-code', required=True, type=int)
    args = parser.parse_args(argv)
    require(sys.dont_write_bytecode and __debug__, 'Explicit source-only bytecode policy')
    spec = stage.reference(dict(path=args.spec, sha256=args.spec_sha256))
    review = stage.reference(dict(path=args.review, sha256=args.review_sha256))
    actual = dict(kind='tool_session', session_id=args.session_id, exit_code=args.exit_code, terminal_chunk=args.terminal_chunk)
    print(json.dumps(audit(spec, review, actual), indent=2))


if __name__ == '__main__': main()
