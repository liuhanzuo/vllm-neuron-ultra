"""Bounded serial saved-tap analysis; never launches a model or compiles a graph."""
import argparse
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
SELF = Path(__file__).resolve()
ANALYSIS_PATH = ROOT / 'minicpm5_layer0_phase_selected_tap_analysis_v1.py'
RUNNER_PATH = ROOT / 'minicpm5_original32_layer0_phase_cpu_runner_v1.py'
RUNNER_SHA = '304962388c6ad09bd2da39555714dee6678c7bc20a2a165875988445c6b3dd9b'
OUTPUT_CAP = 16 * 2**20


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    return dict(path=str(path), sha256=sha(path))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(path, value):
    data = json.dumps(value, indent=2, allow_nan=False) + '\n'
    require(len(data.encode()) < 8 * 2**20, 'Bounded JSON output')
    with path.open('x') as stream:
        stream.write(data)
    return ref(path)


def run(exit_ref, review_ref, source_sha, analysis_sha, output_prefix):
    require(sys.dont_write_bytecode and __debug__, 'Python -B required')
    require(sha(SELF) == source_sha and sha(ANALYSIS_PATH) == analysis_sha,
            'Exact reviewed new sources')
    require(sha(RUNNER_PATH) == RUNNER_SHA, 'Frozen completed model runner')
    prefix = Path(output_prefix)
    require(prefix.parent == ROOT / 'research'
            and prefix.name.startswith('minicpm5-layer0-phase-selected-taps-r'),
            'Finite output namespace')
    paths = {key: Path(str(prefix) + suffix) for key, suffix in
             [('start', '-start.json'), ('report', '.json'), ('rows', '.csv'),
              ('result', '-runner-result.json'), ('failure', '-failure.json')]}
    require(all(not path.exists() for path in paths.values()), 'Fresh immutable output paths')
    require(os.sched_getaffinity(0) == {6, 7}, 'Bounded CPU affinity')
    require(all(os.environ.get(key) == '1' for key in
                ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')),
            'One native thread environment')
    address_soft, address_hard = resource.getrlimit(resource.RLIMIT_AS)
    require(0 < address_soft <= address_hard <= 4 * 2**30,
            'Four GiB address-space soft and hard admission bounds')
    runner = importlib.import_module('minicpm5_original32_layer0_phase_cpu_runner_v1')
    analysis = importlib.import_module('minicpm5_layer0_phase_selected_tap_analysis_v1')
    review = runner.p.read_ref(review_ref)
    require(review['reviewer'] == '/root' and review['passed'] is True and not review['blockers'],
            'Concrete root execution review required')
    require(review['runner'] == ref(SELF) and review['analysis'] == ref(ANALYSIS_PATH)
            and review['actual_exit'] == exit_ref and review['output_prefix'] == str(prefix),
            'Review binds source, actual predecessor and fresh output')
    prepared = analysis.prepare_inputs(exit_ref)
    ended = prepared['exit']
    require(ended['actual_execution_handle'] == dict(kind='tool_session', session_id=46999),
            'Actual targeted predecessor session')
    fd, owner, lock = runner.post.external.acquire_lock()
    started = time.monotonic()
    admission = dict(pid=os.getpid(), serial_exclusive=True, actual_exit_verified=True,
                     plan=analysis.PLAN, actual_exit=exit_ref, source_sha256=analysis_sha)

    def guard():
        runner.p.clear_window()
        runner.p.lock_proof(owner, lock)
        runner.p.active_heavy()
        require(time.monotonic() - started < 600, 'Ten-minute raw analysis deadline')
        require(os.sched_getaffinity(0) == {6, 7}, 'CPU affinity retained')
        require(sha(SELF) == source_sha and sha(ANALYSIS_PATH) == analysis_sha
                and sha(RUNNER_PATH) == RUNNER_SHA, 'Sources unchanged')
        require(not any(name.split('.')[0] in ('torch', 'numpy', 'torch_neuronx', 'neuronxcc')
                        for name in sys.modules), 'Stdlib-only analysis')
        fs = os.statvfs(ROOT)
        require(fs.f_bavail * fs.f_frsize >= 28 * 2**30, 'Original root free-space floor')
        total = 0
        for path in paths.values():
            if path.exists():
                info = path.lstat()
                require(not path.is_symlink() and info.st_nlink == 1 and info.st_uid == 1000
                        and info.st_size < 8 * 2**20, 'Bounded owned report')
                total += max(info.st_size, info.st_blocks * 512)
        require(total < OUTPUT_CAP, 'Finite report output cap')
        for name, expected in prepared['spec']['canonical_sha256'].items():
            require(sha(ROOT / name) == expected, 'Canonical state unchanged')
        return admission

    try:
        guard()
        fs = os.statvfs(ROOT)
        require(fs.f_bavail * fs.f_frsize >= 28 * 2**30 + OUTPUT_CAP,
                'Reserve complete report budget')
        bound, owners, priors = runner.check_prerequisites(prepared['spec'])
        runner.p.gone(bound['finalizer_owner'])
        runner.p.predecessor_drain(bound['catalog'])
        for previous in owners:
            runner.p.gone(previous)
        prior_drains = runner.predecessor_drains(priors)
        for previous in [ended['owner'], ended['coordinator'], *ended['ranks']]:
            runner.p.gone(runner.fullshape.previous.process(previous))
        group = ended['coordinator']['cgroup'].split('::', 1)[1]
        targeted_drain = runner.p.empty_group(group)
        guard()
        start_ref = write(paths['start'], dict(schema='minicpm5_selected_tap_runner_start_v1',
            owner=owner, lock_identity=lock, runner=ref(SELF), analysis=ref(ANALYSIS_PATH),
            review=review_ref, actual_exit=exit_ref, output_prefix=str(prefix),
            checked_previous_owners=owners, predecessor_drains=prior_drains,
            targeted_group_drain=targeted_drain, self_exit_claimed=False))
        print(json.dumps(dict(stage='admitted_saved_tap_analysis', owner=owner, start=start_ref)), flush=True)
        result = analysis.analyze(prepared, admission, guard)
        guard()
        result['execution_owner'] = owner
        report_ref = write(paths['report'], result)
        with paths['rows'].open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['reference', 'observed', 'frame', 'fixture',
                'step', 'tap', 'raw_exact', 'bitwise_different_elements', 'different_elements',
                'max_abs', 'mean_abs', 'cosine'])
            writer.writeheader()
            for row in result['rows']:
                metric = row['metrics']
                writer.writerow(dict(reference=row['reference'], observed=row['observed'],
                    frame=row['frame']['index'], fixture=row['frame']['fixture'],
                    step=row['frame']['step'], tap=row['tap'], **{key: metric[key] for key in
                        ('raw_exact', 'bitwise_different_elements', 'different_elements',
                         'max_abs', 'mean_abs', 'cosine')}))
        guard()
        result_ref = write(paths['result'], dict(schema='minicpm5_selected_tap_runner_result_v1',
            complete=True, execution_owner=owner, start=start_ref,
            outputs=dict(report=report_ref, rows=ref(paths['rows'])), actual_predecessor_exit=exit_ref,
            raw_bytes_read=result['actual_raw_bytes_read'], frames=32, taps=4, comparisons=6,
            self_exit_claimed=False, canonical_unchanged=True, full_model_qualified=False,
            new_speedup_claim=False))
        guard()
        return result_ref
    except BaseException as error:
        try:
            write(paths['failure'], dict(schema='minicpm5_selected_tap_runner_failure_v1',
                execution_owner=owner, error_type=type(error).__name__, error=str(error),
                self_exit_claimed=False))
        except BaseException as secondary:
            runner.secondary_error(error, secondary, 'analysis_failure_receipt')
        raise
    finally:
        primary = sys.exc_info()[1]
        try:
            runner.post.external.release_lock(fd)
        except BaseException as secondary:
            if primary is None:
                raise
            runner.secondary_error(primary, secondary, 'analysis_lock_release')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('actual-exit', 'actual-exit-sha256', 'review', 'review-sha256',
                'source-sha256', 'analysis-sha256', 'output-prefix'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    print(json.dumps(run(dict(path=args.actual_exit, sha256=args.actual_exit_sha256),
        dict(path=args.review, sha256=args.review_sha256), args.source_sha256,
        args.analysis_sha256, args.output_prefix)), flush=True)
