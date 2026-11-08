"""Bounded serial full original Dory CPU reference and saved-byte review."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

Q = Path(__file__).resolve().parent
sys.path[:0] = [str(Q), str(Q.parent)]
import run_dory_weight_storage_root_v2 as d


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def free(path):
    s = os.statvfs(path)
    return s.f_bavail*s.f_frsize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', required=True)
    ap.add_argument('--spec-sha256', required=True)
    ap.add_argument('--review', action='store_true')
    args = ap.parse_args()
    path = Path(args.spec)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == args.spec_sha256
    spec = json.loads(path.read_text())
    for name, sha in spec['source_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == sha, name
    budget = spec['budget']
    base, env_path = Path(spec['base']), Path(spec['environment'])
    out = Path(spec['review_output'] if args.review else spec['output'])
    assert not out.exists()
    fd, owner, lock = d.m.old.post.external.acquire_lock()
    child = None
    started = time.monotonic()
    initial_free = free('/mnt/nvme')
    try:
        d.m.p.clear_window()
        d.m.p.active_heavy()
        d.m.same({n:d.m.p.sha(Q.parent/n) for n in d.m.CANONICAL}, d.m.CANONICAL, 'Canonical unchanged')
        assert free('/') >= budget['root_floor']
        assert initial_free >= budget['nvme_floor']+budget['nvme_new_bytes']
        memory = dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
        assert int(memory['MemAvailable'].split()[0])*1024 >= budget['initial_memory_available_bytes']
        out.mkdir()
        os.sched_setaffinity(0, set(budget['cpus']))
        os.nice(10)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
            OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
            XDG_CACHE_HOME=str(base/'cache'), HF_HOME=str(base/'cache/huggingface'),
            TMPDIR=str(base/'tmp'), TRITON_CACHE_DIR=str(base/'cache/triton'),
            TORCH_EXTENSIONS_DIR=str(base/'cache/torch'),
            LD_PRELOAD='/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:'+str(env_path/'lib/libiomp5.so'),
            VLLM_TARGET_DEVICE='cpu', VLLM_PLUGINS='', VLLM_LOGGING_LEVEL='INFO',
            VLLM_CPU_KVCACHE_SPACE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', TOKENIZERS_PARALLELISM='false',
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        env.pop('PYTHONPATH', None)
        (base/'tmp').mkdir(exist_ok=True)
        worker = spec['reviewer'] if args.review else spec['worker']
        cmd = [str(env_path/'bin/python'), '-B', worker, '--spec', str(path),
               '--admission', str(out/'admission.json')]
        save(out/'admission.json', dict(spec_sha256=args.spec_sha256, owner=owner, lock=lock,
            command=cmd, review=args.review, budget=budget, canonical_sha256=d.m.CANONICAL,
            root_free=free('/'), nvme_free=initial_free, performance_measurement=False))
        print(json.dumps(dict(admission=str(out/'admission.json'), owner=owner)), flush=True)
        log = out/'worker.log'
        with log.open('xb') as f:
            child = subprocess.Popen(cmd, cwd=base, env=env, stdout=f, stderr=subprocess.STDOUT,
                                     start_new_session=True, pass_fds=(fd,))
            save(out/'child.json', dict(pid=child.pid, command=cmd))
            while child.poll() is None:
                d.m.p.lock_proof(owner, lock)
                d.m.p.clear_window()
                d.m.p.active_heavy()
                assert time.monotonic()-started <= budget['wall_seconds'], 'Wall budget'
                assert free('/') >= budget['root_floor'], 'Root floor'
                assert free('/mnt/nvme') >= max(budget['nvme_floor'], initial_free-budget['nvme_new_bytes']), 'NVMe budget'
                assert log.stat().st_size <= budget['log_bytes'], 'Log budget'
                memory = dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
                assert int(memory['MemAvailable'].split()[0])*1024 >= budget['memory_available_floor_bytes'], 'Host memory floor'
                time.sleep(2)
            code = child.returncode
            save(out/'child-exit.json', dict(returncode=code, child_pid=child.pid))
            child = None
            assert code == 0, f'Worker failed with {code}; see {log}'
        d.m.same({n:d.m.p.sha(Q.parent/n) for n in d.m.CANONICAL}, d.m.CANONICAL, 'Canonical unchanged')
        save(out/'result.json', dict(complete=True, review=args.review, child_exit=code,
             elapsed_seconds=time.monotonic()-started, root_free=free('/'), nvme_free=free('/mnt/nvme'),
             actual_self_exit_claimed=False, full_model_qualified=False, new_speedup_claim=False))
        print(json.dumps(dict(result=str(out/'result.json'))), flush=True)
    except BaseException as error:
        if out.is_dir():
            save(out/'failure.json', dict(error_type=type(error).__name__, error=str(error)))
        raise
    finally:
        if child is not None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
        d.m.old.post.external.release_lock(fd)


if __name__ == '__main__':
    main()
