"""Read-only saved JSON audit of the exact ongoing LFM r1; no native imports.

Uses the frozen scientific validators. A stage audit never invents a full result.
Process termination and physical drain must be checked separately by root.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import stat
import sys
from types import SimpleNamespace

ROOT = Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
Q = ROOT / 'research'
OUTPUT = Path('/mnt/nvme/lfm3-nki05-full-cpu-r1-20261112')
PRIOR = Q / 'lfm3-full-cpu-original-fixture-root-partial-audit-20261112-v1.json'
PRIOR_SHA = 'eaf36501f753981e3ab69c427eb42d89bef51825ed3de4957dda985cfa4fc04b'
ADMISSION = dict(path=str(OUTPUT / 'admission.json'),
                 sha256='dad9b67a8ac407238f9ef50928f1ed006a96cce2c3c1104835ca6090ded30e8b')
SELF_RUNNER = str(ROOT / 'run_lfm3_nki05_full_cpu_serial_v2.py')


def require(value, message):
    if not value:
        raise RuntimeError(message)


def identity(st):
    return (st.st_dev, st.st_ino, st.st_mode, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def stable(path, cap):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path, 'Exact real path')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_size <= cap, 'Bounded regular file')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            data = stream.read(cap + 1)
        require(len(data) == before.st_size <= cap and
                identity(before) == identity(os.fstat(fd)) == identity(path.stat()), 'Stable whole bytes')
        return data
    finally:
        os.close(fd)


class SavedJSON:
    def __init__(self):
        self.hashes = {}

    def read(self, path, wanted=None):
        path = Path(path)
        require(path.suffix == '.json' and path.is_relative_to(OUTPUT), 'Only exact output JSON')
        data = stable(path, 256 * 1024)
        sha = hashlib.sha256(data).hexdigest()
        require(wanted is None or sha == wanted, 'Saved report pin')
        require(str(path) not in self.hashes or self.hashes[str(path)] == sha, 'Saved JSON cannot change')
        self.hashes[str(path)] = sha
        def unique(pairs):
            result = {}
            for key, value in pairs:
                require(key not in result, 'Unique JSON keys')
                result[key] = value
            return result
        return json.loads(data, object_pairs_hook=unique)

    def reference(self, path):
        self.read(path)
        return dict(path=str(path), sha256=self.hashes[str(path)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('new_fixture', 'full', 'partial_failure'), required=True)
    args = parser.parse_args()
    require(sys.dont_write_bytecode and __debug__, 'Bytecode writes disabled, assertions retained')
    os.sched_setaffinity(0, {4})
    os.nice(19)
    resource.setrlimit(resource.RLIMIT_AS, (512 * 2**20,) * 2)
    resource.setrlimit(resource.RLIMIT_CPU, (30, 35))
    resource.setrlimit(resource.RLIMIT_FSIZE, (2**20,) * 2)
    def timeout(*_):
        raise TimeoutError('Saved JSON audit exceeded 60 seconds')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(60)

    prior_raw = stable(PRIOR, 2**20)
    require(hashlib.sha256(prior_raw).hexdigest() == PRIOR_SHA, 'Exact earlier original-fixture audit')
    pins = json.loads(prior_raw)['loaded_frozen_sources']
    require(len(pins) == 7, 'Exact seven frozen source modules')
    for path, sha in pins.items():
        require(hashlib.sha256(stable(path, 4 * 2**20)).hexdigest() == sha, 'Frozen source pin')
    before_modules = set(sys.modules)
    sys.path.insert(0, str(ROOT))
    import lfm3_nki05_full_cpu_leaf_v2 as science
    reader = SavedJSON()
    host = SimpleNamespace(p=reader, reference=reader.reference, SELF=SELF_RUNNER)
    reader.read(ADMISSION['path'], ADMISSION['sha256'])

    failed_children = []
    if args.stage == 'partial_failure':
        diagnostic = reader.read(OUTPUT / 'diagnostic-result.json')
        require(diagnostic['admission'] == ADMISSION and diagnostic['complete'] is False
                and diagnostic['passed'] is False and diagnostic['cpu_full_gate_passed'] is False,
                'Preserve actual incomplete failed result')
        require(diagnostic['stage_counts'] == {'original_fixture': dict(cases=3716, batches=291)},
                'Only original fixture stage completed')
        require(not (OUTPUT / 'controls').exists() and not (OUTPUT / 'candidate').exists(),
                'Controls and candidate did not start')
        science.stage_complete(host, OUTPUT, 'original_fixture', ADMISSION)
        matrices = science.matrices()
        parent = reader.read(OUTPUT / 'worker-owner.json')
        counts = {}
        for plan in science.schedule():
            stage, phase = plan['stage'], plan['phase']
            if stage not in ('original_fixture', 'new_fixture'):
                continue
            stats = counts.setdefault(stage, dict(reports=0, report_cases=0, zero_exit_batches=0, zero_exit_cases=0))
            phase_result = reader.read(OUTPUT / stage / (phase + '-outcomes.json'))
            require(phase_result['admission'] == ADMISSION and phase_result['stage'] == stage
                    and phase_result['phase'] == phase, 'Exact phase binding')
            outcomes = phase_result['outcomes']
            require([(row['begin'], row['end']) for row in outcomes] ==
                    [tuple(pair) for pair in plan['bounds'][:len(outcomes)]], 'Exact original ordered prefix')
            for row in outcomes:
                begin, end = row['begin'], row['end']
                path = science.result_path(OUTPUT, stage, phase, begin, end)
                require(row['stage'] == stage and row['phase'] == phase and row['secondary_errors'] == []
                        and row['report'] == reader.reference(path), 'Exact parent outcome/report pin')
                report = reader.read(path, row['report']['sha256'])
                science.validate_report(report, stage, phase, begin, end, matrices[phase][begin:end], ADMISSION)
                ack = reader.read(science.launch_path(OUTPUT, stage, phase, begin, end))
                require(ack['pid'] == report['pid'] == row['pid'] and
                        ack['start_ticks'] == report['start_ticks'] == row['start_ticks'] and
                        ack['parent'] == dict(parent, cgroup=parent['cgroup'].removeprefix('0::')) and
                        ack['argv'] == science.command(host, ADMISSION, stage, phase, begin, end),
                        'Every actual child ACK identity')
                if stage == 'new_fixture':
                    old = reader.read(science.result_path(OUTPUT, 'original_fixture', phase, begin, end))
                    require(report['checks'] == old['checks'], 'All saved new fixture hashes match originals')
                stats['reports'] += 1
                stats['report_cases'] += end - begin
                if row['returncode'] == 0:
                    require(row['passed'] is True and row['error'] is None, 'Actual zero-exit outcome')
                    stats['zero_exit_batches'] += 1
                    stats['zero_exit_cases'] += end - begin
                else:
                    require(stage == 'new_fixture' and phase == 'extended_cpu'
                            and (begin, end, row['pid'], row['start_ticks'], row['returncode']) ==
                            (1319, 1320, 2717091, '422981941', 120) and row['passed'] is False
                            and row['error'] == 'Actual finite child nonzero exit preserved',
                            'Exact failed last child remains failed despite complete report')
                    failed_children.append(row)
            finished = len(outcomes) == len(plan['bounds']) and all(row['returncode'] == 0 for row in outcomes)
            require(phase_result['complete'] is finished and phase_result['passed'] is finished,
                    'Actual phase completion is not inferred from reports')
        require(counts == {
            'original_fixture': dict(reports=291, report_cases=3716, zero_exit_batches=291, zero_exit_cases=3716),
            'new_fixture': dict(reports=232, report_cases=2968, zero_exit_batches=231, zero_exit_cases=2967)},
            'Exact complete original and partial new fixture counts')
        require(len(failed_children) == 1, 'Exactly one failed actual last child')
        complete_cpu = False
    elif args.stage == 'full':
        result = reader.read(OUTPUT / 'diagnostic-result.json')
        require(result['admission'] == ADMISSION, 'Exact r1 result admission')
        require(science.audit_full_success(host, result) is True, 'Frozen complete 874-child audit')
        counts = result['stage_counts']
        complete_cpu = True
    else:
        stage = 'new_fixture'
        summary = science.stage_complete(host, OUTPUT, stage, ADMISSION)
        matrices = science.matrices()
        parent = reader.read(OUTPUT / 'worker-owner.json')
        planned = [row for row in science.schedule() if row['stage'] == stage]
        require([ref['path'] for ref in summary['phases']] ==
                [str(OUTPUT / stage / (plan['phase'] + '-outcomes.json')) for plan in planned],
                'All original phase paths/order')
        checked = cases = 0
        for plan, ref in zip(planned, summary['phases']):
            phase = plan['phase']
            outcomes = reader.read(ref['path'], ref['sha256'])['outcomes']
            require([(row['begin'], row['end']) for row in outcomes] ==
                    [tuple(pair) for pair in plan['bounds']], 'Every original batch retained')
            for row in outcomes:
                begin, end = row['begin'], row['end']
                path = science.result_path(OUTPUT, stage, phase, begin, end)
                require(row['stage'] == stage and row['phase'] == phase and row['passed'] is True
                        and row['returncode'] == 0 and row['error'] is None and row['secondary_errors'] == []
                        and row['report'] == reader.reference(path), 'Parent-recorded successful exact child')
                report = reader.read(path, row['report']['sha256'])
                science.validate_report(report, stage, phase, begin, end, matrices[phase][begin:end], ADMISSION)
                ack = reader.read(science.launch_path(OUTPUT, stage, phase, begin, end))
                require(ack['pid'] == report['pid'] == row['pid'] and
                        ack['start_ticks'] == report['start_ticks'] == row['start_ticks'] and
                        ack['parent'] == dict(parent, cgroup=parent['cgroup'].removeprefix('0::')) and
                        ack['argv'] == science.command(host, ADMISSION, stage, phase, begin, end),
                        'Every actual child ACK identity')
                old = reader.read(science.result_path(OUTPUT, 'original_fixture', phase, begin, end))
                require(report['checks'] == old['checks'], 'Exact original typed input and expected hashes')
                checked += 1
                cases += end - begin
        require(checked == 291 and cases == 3716, 'Complete new-environment fixture scope')
        counts = {stage: dict(cases=cases, batches=checked)}
        complete_cpu = False

    loaded = {}
    for name in set(sys.modules) - before_modules:
        path = getattr(sys.modules[name], '__file__', None)
        if path and Path(path).is_relative_to(ROOT):
            require(path in pins, 'No unpinned local source import')
            loaded[path] = pins[path]
    require(not any(name.split('.')[0] in ('torch', 'numpy', 'nki', 'neuronxcc') for name in sys.modules),
            'No native framework import')
    require(loaded == pins, 'All and only the seven scientific source modules')
    result = dict(schema='lfm3_full_cpu_r1_root_saved_json_audit_v1', reviewer='/root',
        observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), stage=args.stage,
        admission=ADMISSION, stage_counts=counts, passed=True, failed_children=failed_children,
        cpu_full_gate_passed=complete_cpu, saved_json_sha256=reader.hashes, loaded_frozen_sources=loaded,
        native_framework_imported=False, raw_tensor_payload_read=False,
        physical_drain_checked=False, actual_outer_tool_exit_checked=False,
        device_behavior_proven=False, alias_gate_passed=False, full_model_qualified=False,
        new_speedup_claim=False, original_simulator_failure_preserved=True)
    output = Q / ('lfm3-full-cpu-r1-root-' + args.stage.replace('_', '-') + '-saved-json-audit-20261112-v1.json')
    raw = (json.dumps(result, ensure_ascii=False, indent=2) + '\n').encode()
    require(len(raw) <= 2**20, 'Finite complete audit report')
    with output.open('xb') as stream:
        stream.write(raw)
    signal.alarm(0)
    print(json.dumps(dict(path=str(output), sha256=hashlib.sha256(raw).hexdigest(),
                         stage=args.stage, stage_counts=counts, saved_json_files=len(reader.hashes))))


if __name__ == '__main__':
    main()
