"""Serial metadata preflight, full official residual replay, or saved review.

Each mode is a separately reviewed invocation. No final spec or automatic
successor is created here; the root must observe each real external tool exit.
Never launch this lock owner through the ordinary observer wrapper.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

Q = Path(__file__).resolve().parent
sys.path[:0] = [str(Q), str(Q.parent)]
import dory_after_nemotron_r9_v2 as history
from dory_residual_serial_contract_v2 import (
    require, load, validate, metadata_history, check_metadata_result, source_accounting)
from nemotron9b_package_controller_v18 import acquire, release
from run_dory_layer0_serial_v2 import (
    save, free, memory_free, group_members, terminate_owned_group, validate_child)
from dory_generation_execution_provenance_v1 import review_source_pins
p = history.p


def environment(spec, ready_fd):
    base, venv = Path(spec['base']), Path(spec['environment'])
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', PYTHONOPTIMIZE='0',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        XDG_CACHE_HOME=str(base/'cache'), HF_HOME=str(base/'cache/huggingface'), TMPDIR=str(base/'tmp'),
        TRITON_CACHE_DIR=str(base/'cache/triton'), TORCH_EXTENSIONS_DIR=str(base/'cache/torch'),
        LD_PRELOAD='/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:'+str(venv/'lib/libiomp5.so'),
        VLLM_TARGET_DEVICE='cpu', VLLM_PLUGINS='', VLLM_LOGGING_LEVEL='INFO', VLLM_CPU_KVCACHE_SPACE='1',
        PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', TOKENIZERS_PARALLELISM='false',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', DORY_RESIDUAL_READY_FD=str(ready_fd))
    env.pop('PYTHONPATH', None)
    return env


def metadata(spec, guard, owner, source_review):
    from dory_residual_event_stream_v1 import OriginalEventStream
    stream = OriginalEventStream(spec['selection_plan'], guard)
    for _ in stream:
        pass
    require(stream.completed_summary is not None, 'No partial original event stream completion')
    return check_metadata_result(dict(complete=True, mode='metadata', owner=owner,
        summary=stream.completed_summary, actual_self_exit_claimed=False,
        native_started=False, tensor_payload_validation_performed=False,
        source_file_authentication=source_accounting(source_review),
        full_model_qualified=False, new_speedup_claim=False))


def main():
    parser = argparse.ArgumentParser()
    for name in ('spec', 'spec-sha256', 'review', 'review-sha256'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--mode', choices=('metadata', 'native', 'review'), required=True)
    a = parser.parse_args()
    require(sys.flags.optimize == 0, 'Original assertions enabled')
    spec_ref = dict(path=a.spec, sha256=a.spec_sha256)
    spec = validate(load(spec_ref), a.mode)
    review = load(dict(path=a.review, sha256=a.review_sha256))
    require(review['passed'] is True and review['reviewer'] == '/root' and review['blockers'] == []
            and review['spec_sha256'] == a.spec_sha256 and review['source_sha256'] == spec['source_sha256'],
            'Final exact specification and complete source review')
    field = dict(metadata='metadata_output', native='output', review='review_output')[a.mode]
    out = Path(spec[field])
    require(not out.exists(), 'Fresh output namespace; retain every failure')
    budget = spec['budget']
    p.low.user_bus()
    fd, owner, lock = acquire()
    started, initial = time.monotonic(), free('/mnt/nvme')
    child = child_owner = readfd = writefd = None
    created, handlers = False, {}
    def interrupted(signum, frame):
        raise InterruptedError('Serial residual diagnostic interrupted')
    def guard():
        p.low.lock_proof(owner, lock)
        p.low.clear_window()
        p.low.active_heavy()
        require(time.monotonic()-started < budget['wall_seconds']
                and free('/') >= budget['root_floor']
                and free('/mnt/nvme') >= max(budget['nvme_floor'], initial-budget['nvme_new_bytes'])
                and memory_free() >= budget['memory_available_floor_bytes'], 'Full original resource bounds')
    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            handlers[sig] = signal.signal(sig, interrupted)
        guard()
        require(not out.exists() and initial >= budget['nvme_floor']+budget['nvme_new_bytes']
                and memory_free() >= budget['initial_memory_available_bytes'], 'Fresh resources under serial lock')
        os.sched_setaffinity(0, set(budget['cpus']))
        os.nice(10)
        if a.mode != 'native':
            resource.setrlimit(resource.RLIMIT_AS, (budget['address_space_bytes'],)*2)
            resource.setrlimit(resource.RLIMIT_CPU, (budget['cpu_seconds'], budget['cpu_seconds']+5))
            resource.setrlimit(resource.RLIMIT_FSIZE, (budget['single_file_bytes'],)*2)
        sources = review_source_pins(spec['source_sha256'], spec['essential_sources'],
            max_total_bytes=spec['source_read_cap_bytes'], guard=guard)
        if a.mode == 'metadata':
            h = history.reconstruct(spec['r9_actual_exit'], spec['r9_additional_actual_exits'],
                                    spec['r9_recovery_operations'], guard)
        else:
            h = metadata_history(spec, p.low.gone)
        # All history pins must already be in the reviewed source closure.
        for path, digest in h['predecessor_source_sha256'].items():
            require(spec['source_sha256'].get(path) == digest, 'No late unreviewed history dependency')
        from review_dory_layer0_serial_snapshot_v1 import physical_drain
        drain = physical_drain(h, [])
        require(drain['complete'] is True and drain['services'] == 1149
                and drain['processes'] == len(h['processes']), 'All actual services and owners physically released')
        canonical = p.canonical()
        require(canonical == h['canonical_sha256'], 'Original frozen canonical files')
        guard()
        out.mkdir()
        created = True
        href = save(out/'predecessor-history.json', h)
        dref = save(out/'predecessor-drain.json', drain)
        save(out/'source-files.json', sources)
        admission = dict(owner=owner, lock=lock, spec_sha256=a.spec_sha256, mode=a.mode,
            review=a.mode != 'native', budget=budget, canonical_sha256=canonical,
            predecessor_history=href, predecessor_drain_record=dref, predecessor_drain=drain,
            performance_measurement=False)
        if a.mode == 'native':
            readfd, writefd = os.pipe()
            command = [str(Path(spec['environment'])/'bin/python'), '-B', spec['worker'],
                       '--spec', a.spec, '--admission', str(out/'admission.json')]
            admission['command'] = command
        save(out/'admission.json', admission)
        print(json.dumps(dict(owner=owner, mode=a.mode, admission=str(out/'admission.json'))), flush=True)
        if a.mode == 'metadata':
            result = metadata(spec, guard, owner, sources)
        elif a.mode == 'review':
            from dory_residual_saved_complete_v1 import freeze, review_complete
            contract = freeze(spec['producer_spec'], spec['native_exit'], guard)
            save(out/'post-exit-contract.json', contract)
            checked = review_complete(contract, guard)
            save(out/'saved-residual-review.json', checked)
            result = dict(complete=True, mode='review', review=True, owner=owner,
                actual_self_exit_claimed=False, full_model_qualified=False, new_speedup_claim=False)
        else:
            with (out/'worker.log').open('xb') as log:
                child = subprocess.Popen(command, cwd=spec['base'], env=environment(spec, readfd),
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(fd, readfd))
                os.close(readfd)
                readfd = None
                child_owner = p.low.proc(child.pid)
                save(out/'child.json', dict(pid=child.pid, command=command))
                deadline = time.monotonic()+10
                while True:
                    actual = p.low.proc(child.pid)
                    if actual and actual['argv'] == command:
                        break
                    require(child.poll() is None and time.monotonic() < deadline, 'Actual live child before ACK')
                    time.sleep(.01)
                child_owner = validate_child(actual, owner, command)
                save(out/'child-owner.json', child_owner)
                os.write(writefd, b'R')
                os.close(writefd)
                writefd = None
                while child.poll() is None:
                    guard()
                    require(os.fstat(log.fileno()).st_size <= budget['log_bytes'], 'Native worker log cap')
                    time.sleep(2)
                code = child.returncode
                save(out/'child-exit.json', dict(returncode=code, child_pid=child.pid, child_owner=child_owner))
                require(code == 0 and not group_members(child.pid), 'Successful actual child exit and drained descendants')
                child = None
            result = dict(complete=True, mode='native', review=False, owner=owner, child_exit=code,
                actual_self_exit_claimed=False, full_model_qualified=False, new_speedup_claim=False)
        guard()
        require(p.canonical() == canonical, 'Original canonical files unchanged')
        result['elapsed_seconds'] = time.monotonic()-started
        save(out/'result.json', result)
        print(json.dumps(dict(complete=True, mode=a.mode, result=str(out/'result.json'))), flush=True)
    except BaseException as error:
        if created:
            save(out/'failure.json', dict(error=repr(error), mode=a.mode))
        raise
    finally:
        for descriptor in (readfd, writefd):
            if descriptor is not None:
                os.close(descriptor)
        for sig in handlers:
            signal.signal(sig, lambda signum, frame: None)
        if child is not None:
            warned = False
            while True:
                try:
                    terminate_owned_group(child, child_owner)
                    break
                except BaseException as error:
                    if not warned:
                        save(out/'cleanup-pending.json', dict(error=repr(error), lock_retained=True,
                                                            child_owner=child_owner))
                        warned = True
                    time.sleep(.5)
            if not (out/'child-exit.json').exists():
                save(out/'child-exit.json', dict(returncode=child.returncode, child_pid=child.pid,
                    child_owner=child_owner, terminated_after_parent_failure=True))
        release(fd)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    main()
