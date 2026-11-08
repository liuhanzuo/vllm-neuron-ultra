"""Finite serial preparation of the pinned official Dory CPU runtime.

This creates no model performance claim. Each stage needs a fresh run directory.
"""
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
R = Q.parent
sys.path[:0] = [str(Q), str(R)]
import run_dory_weight_storage_root_v2 as d

BASE = Path('/mnt/nvme/dory-official-cpu-20261108-v1')
ENV = Path('/mnt/nvme/venvs/dory-official-cpu-20261108-v1')
TMP = Path('/mnt/nvme/tmp-dory-build-20261108-v1')
SOURCE = BASE / 'vllm'
COMMIT = '75fc05a53a366068f534a61ab3ee6b50485419f2'
FETCHED = Q / 'dory-official-cpu-build-source-20261108-v1'
BUDGET = dict(root_floor=28*2**30, nvme_floor=8*2**30,
              nvme_new_bytes=24*2**30, wall_seconds=3600,
              cpus=[4, 5, 6, 7], build_jobs=2, log_bytes=64*2**20)

def free(path):
    s = os.statvfs(path)
    return s.f_bavail * s.f_frsize

def save(path, row):
    with path.open('x') as f:
        json.dump(row, f, indent=2)
        f.write('\n')

def commands(stage, out):
    py = str(ENV / 'bin/python')
    pip = [py, '-m', 'pip', '--isolated']

    if stage == 'setup-tests':
        wheels=list((BASE/'wheels').glob('vllm-*.whl'))
        if len(wheels)!=1 or not json.loads((BASE/'build-r1/result.json').read_text())['complete']:
            raise RuntimeError('One completed official CPU build required')
        return [(pip+['install','--no-deps','--no-cache-dir','--report',str(out/'wheel-installation.json'),str(wheels[0])],BASE),
            (pip+['install','--no-deps','--no-cache-dir','--require-hashes','--report',str(out/'test-tools-installation.json'),'-r',str(BASE/'test-tools.lock')],BASE),
            (pip+['check'],BASE)]
    if stage == 'qualify':
        source=Q/'dory_official_cpu_qualify_v2.py'
        if hashlib.sha256(source.read_bytes()).hexdigest() != '011e98fa69fbd0151a2f6c34e3f720ba74371f4cdaf10841ec5cf5018bfda8ee':
            raise RuntimeError('Qualification source differs from review')
        return [([py,'-B',str(source),'--output',str(out/'qualification')],BASE)]

    if stage == 'two-loop':
        prior=BASE/'qualify-r2/qualification/result.json'
        if not prior.is_file() or not json.loads(prior.read_text())['passed']:
            raise RuntimeError('Successful unchanged official tests required')
        source=Q/'dory_official_two_loop_diagnostic_v1.py'
        if hashlib.sha256(source.read_bytes()).hexdigest() != '23385984a97094074900c5ffa48b5c21929a6f3e875b3230c74107cd6b818aa3':
            raise RuntimeError('Two-loop diagnostic differs from review')
        return [([py,'-B',str(source),'--output',str(out/'diagnostic')],BASE)]

    if stage == 'decode-boundary':
        prior=BASE/'two-loop-r1-saved-review-v1.json'
        if not json.loads(prior.read_text())['passed']:
            raise RuntimeError('Independent saved-state review required')
        source=Q/'dory_first_decode_projection_v1.py'
        if hashlib.sha256(source.read_bytes()).hexdigest() != 'b8eec9e74b5a812992073f61f1fdb32aa83b92cba07f4b202e794d394bbb22aa':
            raise RuntimeError('Boundary source differs from review')
        return [([py,'-B',str(source),'--output',str(out/'diagnostic')],BASE)]
    if stage == 'resolve':
        return [(pip + ['install', '--dry-run', '--report', str(out/'resolution.json'),
            '--no-cache-dir', '--only-binary=:all:', '--index-url', 'https://pypi.org/simple',
            '--extra-index-url', 'https://download.pytorch.org/whl/cpu',
            '-r', str(FETCHED/'requirements/cpu.txt')], BASE)]
    if stage == 'fetch':
        if SOURCE.exists():
            raise RuntimeError('Source destination must be fresh')
        return [(['git', 'init', str(SOURCE)], BASE),
            (['git', 'remote', 'add', 'origin', 'https://github.com/vllm-project/vllm.git'], SOURCE),
            (['git', 'fetch', '--depth=1', 'origin', COMMIT], SOURCE),
            (['git', 'checkout', '--detach', COMMIT], SOURCE),
            (['git', 'rev-parse', 'HEAD'], SOURCE)]
    if stage == 'install':
        lock = BASE/'runtime-requirements.lock'
        if not lock.is_file():
            raise RuntimeError('Reviewed wheel URL/hash lock required')
        return [(pip + ['install', '--no-cache-dir', '--no-deps', '--require-hashes',
            '--report', str(out/'installation.json'), '-r', str(lock)], BASE),
            (pip+['check'], BASE), (pip+['freeze', '--all'], BASE)]
    if stage == 'build':
        actual = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != COMMIT:
            raise RuntimeError('Wrong official source commit')
        if subprocess.check_output(['git', '-C', str(SOURCE), 'diff', '--name-only']).strip():
            raise RuntimeError('Official sources must be unchanged before build')
        return [(pip + ['wheel', '--no-deps', '--no-build-isolation', '--no-cache-dir',
            '--wheel-dir', str(BASE/'wheels'), '.'], SOURCE)]
    raise ValueError(stage)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['resolve', 'fetch', 'install', 'build', 'setup-tests', 'qualify', 'two-loop', 'decode-boundary'])
    ap.add_argument('--run', required=True)
    ap.add_argument('--source-sha256', required=True)
    a = ap.parse_args()
    if not a.run.replace('-', '').isalnum():
        raise ValueError('Simple fresh run name required')
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != a.source_sha256:
        raise RuntimeError('Runner source hash mismatch')
    fd, owner, lock = d.m.old.post.external.acquire_lock()
    child = None
    out = BASE / a.run
    initial_free = free('/mnt/nvme')
    started = time.monotonic()
    try:
        d.m.p.clear_window()
        d.m.p.active_heavy()
        d.m.same({n:d.m.p.sha(R/n) for n in d.m.CANONICAL}, d.m.CANONICAL, 'Canonical unchanged')
        if free(R) < BUDGET['root_floor'] or initial_free < BUDGET['nvme_floor']+BUDGET['nvme_new_bytes']:
            raise RuntimeError('Insufficient initial storage budget')
        BASE.mkdir(exist_ok=True)
        out.mkdir()
        TMP.mkdir(exist_ok=True)
        os.sched_setaffinity(0, set(BUDGET['cpus']))
        os.nice(10)
        env = dict(os.environ, TMPDIR=str(TMP), PIP_CONFIG_FILE='/dev/null',
            PIP_NO_CACHE_DIR='1', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
            VLLM_TARGET_DEVICE='cpu', CC='/usr/bin/gcc-12', CXX='/usr/bin/g++-12',
            MAX_JOBS=str(BUDGET['build_jobs']), CMAKE_BUILD_PARALLEL_LEVEL=str(BUDGET['build_jobs']),
            OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
            XDG_CACHE_HOME=str(BASE/'cache'), HF_HOME=str(BASE/'cache/huggingface'),
            TRITON_CACHE_DIR=str(BASE/'cache/triton'), TORCH_EXTENSIONS_DIR=str(BASE/'cache/torch'),
            PATH=str(ENV/'bin')+':'+os.environ['PATH'])
        env.pop('PIP_EXTRA_INDEX_URL', None)
        env.pop('PYTHONPATH', None)
        env.update(CARGO_HOME=str(BASE/'cache/cargo'),RUSTUP_HOME=str(BASE/'cache/rustup'))
        if a.stage in ('qualify','two-loop','decode-boundary'):
            iomp=ENV/'lib/libiomp5.so'
            tcmalloc=Path('/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4')
            if not iomp.is_file() or not tcmalloc.is_file():
                raise RuntimeError('Official Intel OpenMP/TCMalloc preload missing')
            env.update(LD_PRELOAD=str(tcmalloc)+':'+str(iomp),
                PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',VLLM_PLUGINS='',
                VLLM_LOGGING_LEVEL='DEBUG',VLLM_CPU_KVCACHE_SPACE='1',
                HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')

        cmds = commands(a.stage, out)
        save(out/'admission.json', dict(stage=a.stage, owner=owner, lock=lock,
            source_sha256=a.source_sha256, budget=BUDGET, commands=[(c,str(w)) for c,w in cmds],
            canonical_sha256=d.m.CANONICAL, root_free=free(R), nvme_free=initial_free,
            official_commit=COMMIT, native_model_execution=False, performance_measurement=False))
        print(json.dumps(dict(admission=str(out/'admission.json'), owner=owner)), flush=True)
        results=[]
        for i,(cmd,cwd) in enumerate(cmds):
            log=out/f'{i:02d}.log'
            with log.open('xb') as f:
                child=subprocess.Popen(cmd,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,
                    start_new_session=True,pass_fds=(fd,))
                save(out/f'{i:02d}-child.json', dict(pid=child.pid, command=cmd, cwd=str(cwd)))
                while child.poll() is None:
                    d.m.p.lock_proof(owner, lock)
                    d.m.p.clear_window()
                    d.m.p.active_heavy()
                    if time.monotonic()-started > BUDGET['wall_seconds']:
                        raise TimeoutError('Maintenance wall budget exceeded')
                    if free(R)<BUDGET['root_floor'] or free('/mnt/nvme')<max(BUDGET['nvme_floor'],initial_free-BUDGET['nvme_new_bytes']):
                        raise RuntimeError('Storage guard reached')
                    if log.stat().st_size>BUDGET['log_bytes']:
                        raise RuntimeError('Log budget reached')
                    time.sleep(2)
                code=child.returncode
                results.append(dict(command=cmd,returncode=code,log=str(log)))
                print(json.dumps(results[-1]),flush=True)
                if code:
                    raise RuntimeError(f'Stage command failed with {code}; see {log}')
                child=None
        d.m.same({n:d.m.p.sha(R/n) for n in d.m.CANONICAL}, d.m.CANONICAL, 'Canonical unchanged')
        save(out/'result.json',dict(complete=True,stage=a.stage,owner=owner,commands=results,
            elapsed_seconds=time.monotonic()-started,root_free=free(R),nvme_free=free('/mnt/nvme'),
            full_model_qualified=False,new_speedup_claim=False,actual_self_exit_claimed=False))
    except BaseException as e:
        if out.is_dir():
            save(out/'failure.json',dict(error_type=type(e).__name__,error=str(e),owner=owner,
                elapsed_seconds=time.monotonic()-started,actual_self_exit_claimed=False))
        raise
    finally:
        if child is not None:
            try: os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError: pass
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL)
                child.wait(timeout=10)
        d.m.old.post.external.release_lock(fd)

if __name__=='__main__':
    main()
