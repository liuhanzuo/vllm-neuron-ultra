"""Finite cache-isolated compile comparison; no model or target ABI probe."""
import ast
import datetime
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import signal
import stat
import struct
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
Q = ROOT / 'research'
sys.path[:0] = [str(ROOT), str(Q)]
import minicpm5_layer0_external_supervisor_v4 as external
import dory_descriptor_inventory_runtime_v2 as descriptor
import run_dory_serial_stage_root_v5 as lifecycle

p = external.p
OUT = Path('/mnt/nvme/dory-four-timestamp-caches-r2-20261111')
PYTHON = '/mnt/nvme/venvs/dory-official-cpu-20261108-v1/bin/python'
PLAN = Q / 'dory-four-timestamp-caches-r2-20261111-plan-v3.json'
REVIEW = Q / 'dory-four-timestamp-caches-r2-20261111-review-v3.json'
PYTHON_ALIAS = Path('/home/ubuntu/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu')
PYTHON_CANONICAL = Path('/home/ubuntu/.local/share/uv/python/cpython-3.12.16-linux-x86_64-gnu')
TWO_LINKS = {'/usr/lib/x86_64-linux-gnu/libc.so.6', '/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30'}
PINS = {
    'gaps': (Q / 'dory-remaining-gaps-after-package-replay-20261111-v1.json', 'fdea581b6127e48774ce235cc5a3969eb922c99ee6581a49a0cabc177dea8bb0'),
    'python_review': (Q / 'dory-torch-cpython-archive-r2-post-exit-root-review-20261111-v1.json', '26456bac52441bd98c201e43ac8481076286cfd43f43e0ac0f47bd645b4a7e04'),
    'sysconfig': (Q / 'dory-uv-sysconfig-root-confirmation-20261111-v1.json', '0868d594a99a83330fe7719561e0d869f36a5d5f2ae0ccad8f6d1f07bf8475c8'),
}
CHILD = r'''
import sys, marshal
assert sys.version_info[:3] == (3, 12, 16)
assert sys.flags.isolated == 1 and sys.flags.no_site == 1
assert sys.dont_write_bytecode and sys.flags.optimize == 0 and marshal.version == 4
assert sys.pycache_prefix == sys.argv[1]
assert len(sys.argv) == 6
compiled = []
for filename in sys.argv[2:]:
    with open(filename, 'rb') as f:
        source = f.read(1048577)
    assert len(source) <= 1048576
    code = compile(source, filename, 'exec', dont_inherit=True, optimize=0)
    compiled.append((filename, code.co_filename, marshal.dumps(code, 4).hex()))
origins = [(name, getattr(getattr(module, '__spec__', None), 'origin', None),
            getattr(module, '__file__', None)) for name, module in sorted(sys.modules.items())]
print(repr(dict(version=sys.version, executable=sys.executable,
               cache_prefix=sys.pycache_prefix, magic=sys.modules['_frozen_importlib_external'].MAGIC_NUMBER.hex(),
               marshal_version=marshal.version, compiled=compiled, module_origins=origins)))
'''


def ident(s):
    return {k: getattr(s, k) for k in ('st_dev', 'st_ino', 'st_mode', 'st_nlink', 'st_size', 'st_mtime_ns', 'st_ctime_ns')}


def read_file(path, cap, guard, sha=None, size=None):
    path = Path(path); guard()
    p.require(path.resolve() == path, 'Canonical finite input')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        links = 2 if str(path) in TWO_LINKS else 1
        p.require(stat.S_ISREG(before.st_mode) and before.st_nlink == links and before.st_size <= cap, 'Bounded ordinary input with exact link count')
        if links == 2:
            p.require(before.st_uid == before.st_gid == 0 and not before.st_mode & 0o022, 'Original two root-owned library exceptions')
        p.require(size is None or before.st_size == size, 'Expected complete input size')
        chunks = []; count = 0
        while True:
            guard(); block = os.read(fd, 65536)
            if not block: break
            chunks.append(block); count += len(block)
            p.require(count <= cap, 'Input cap while reading')
        raw = b''.join(chunks)
        p.require(len(raw) == before.st_size and ident(before) == ident(os.fstat(fd)) == ident(path.lstat()), 'Stable complete input')
        digest = hashlib.sha256(raw).hexdigest()
        p.require(sha is None or digest == sha, 'Authenticated input SHA256')
        return dict(path=str(path), bytes=len(raw), sha256=digest, identity=ident(before)), raw
    finally:
        os.close(fd)


def reconstruct_header(magic, source_identity, source_size):
    p.require(magic == bytes.fromhex('cb0d0d0a'), 'Exact CPython 3.12 magic')
    return magic + struct.pack('<III', 0, (source_identity['st_mtime_ns'] // 10**9) & 0xffffffff, source_size & 0xffffffff)


def main():
    p.require(sys.dont_write_bytecode and not OUT.exists(), 'Fresh -B diagnostic')
    plan_ref = external.ref(PLAN); review_ref = external.ref(REVIEW)
    plan = json.loads(PLAN.read_bytes()); review = json.loads(REVIEW.read_bytes())
    p.require(review['plan'] == plan_ref and review['passed'] is True and review['blockers'] == [], 'Exact accepted diagnostic plan')
    p.require(plan['output_root'] == str(OUT) and plan['source_sha256'][str(Path(__file__).resolve())] == p.sha(Path(__file__)), 'Exact source and output binding')
    p.user_bus(); fd, owner, lock, first = descriptor.acquire(p, plan_ref, review_ref)
    started = time.monotonic(); child = None; observed = []; last = [0.0]
    result = dict(schema='dory_four_timestamp_cache_comparison_v1', owner=owner, lock_identity=lock,
                  source=external.ref(Path(__file__)), plan=plan_ref, review=review_ref, self_exit_claimed=False,
                  native_ABI_qualified=False, full_model_qualified=False, new_speedup_claim=False)
    def guard():
        p.require(time.monotonic() - started < 900, 'Original wall budget')
        if time.monotonic() - last[0] > 0.5:
            descriptor.observe(p, owner, lock, fd); p.clear_window(); p.active_heavy()
            for path, floor in [(ROOT, 28 * 2**30), (Path('/mnt/nvme'), 30 * 2**30)]:
                s = os.statvfs(path); p.require(s.f_bavail * s.f_frsize >= floor, 'Original storage floor')
            last[0] = time.monotonic()
    def pinned(reference, cap=8 * 2**20):
        actual, raw = read_file(reference['path'], cap, guard, reference['sha256'], reference.get('bytes'))
        observed.append(actual)
        return json.loads(raw)
    saved_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    def interrupted(sig, frame):
        raise InterruptedError('Parent termination signal ' + str(sig))
    try:
        for sig in saved_handlers: signal.signal(sig, interrupted)
        p.require(lock == {'device': 66306, 'inode': 8294421}, 'Original serial lock')
        os.sched_setaffinity(0, {5}); os.nice(19)
        resource.setrlimit(resource.RLIMIT_AS, (2 * 2**30,) * 2)
        resource.setrlimit(resource.RLIMIT_CPU, (600, 605))
        resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 2**20,) * 2)
        guard(); OUT.mkdir(mode=0o700); prefix = OUT / 'empty-pycache'; prefix.mkdir(mode=0o700)
        p.once(OUT / 'initial-lock-proof.json', descriptor.initial_record(first, owner, lock, fd, plan_ref, review_ref))
        for path, sha in plan['source_sha256'].items():
            actual, _ = read_file(path, 8 * 2**20, guard, sha); observed.append(actual)
        serial = pinned(plan['current_serial_catalog'])
        p.require(serial['all_later_serial_runs_included'] is True and serial['current_handoff'] == plan['current_handoff'], 'Complete current history binding')
        latest = max(Q.glob('continuation-handoff-*.json'), key=lambda path: (path.stat().st_mtime_ns, str(path)))
        p.require(str(latest) == plan['current_handoff']['path'] and external.ref(latest) == plan['current_handoff'], 'Fresh latest handoff')
        for row in serial['runs']:
            pinned(row['terminal'])
            for previous in row['owners']: p.gone(previous)
        history = lifecycle.preflight_history(plan, p)
        p.once(OUT / 'owner.json', dict(owner=owner, lock_identity=lock, source=result['source'], history=history))
        inputs = {name: pinned(dict(path=str(path), sha256=sha)) for name, (path, sha) in PINS.items()}
        gaps = inputs['gaps']; rows = gaps['remaining']['cpython_timestamp_caches']
        p.require(len(rows) == 4 and gaps['byte_origin_gaps_closed'] == 11, 'Exact four caches after package verification')
        replay = pinned(gaps['actual_replay_exit']); recorder = pinned(gaps['recorder_exit'])
        p.require(replay['stage_success_observed'] and replay['actual_tool']['exit_code'] == 0 and recorder['owner_gone'], 'Actual package replay and recorder complete')
        package = pinned(gaps['package_result'])
        for item in package['packages']:
            for comparison in item['comparisons']:
                ref = comparison['current']; actual, _ = read_file(ref['path'], 16 * 2**20, guard, ref['sha256'], ref['bytes']); observed.append(actual)
        review = inputs['python_review']; saved = review['full_comparison']
        actual, compressed = read_file(saved['path'], 2 * 2**20, guard, saved['sha256'], saved['bytes']); observed.append(actual)
        with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as stream:
            raw = stream.read(saved['uncompressed_bytes'] + 1)
            p.require(len(raw) == saved['uncompressed_bytes'] and stream.read(1) == b'', 'Complete bounded saved archive comparison')
        p.require(hashlib.sha256(raw).hexdigest() == saved['uncompressed_sha256'], 'Saved comparison raw SHA256')
        archived = json.loads(raw)['findings']['cpython']
        source_rows = {row['path']: row for row in archived['files']}
        interpreter = next(row for row in archived['venv_interpreter_copies'] if row['path'] == PYTHON)
        p.require(interpreter['status'] == 'exact' and interpreter['sha256'] == interpreter['expected_sha256'], 'Archived interpreter identity')
        actual, _ = read_file(PYTHON, 64 * 2**20, guard, interpreter['sha256'], interpreter['bytes']); observed.append(actual)
        sources, caches = [], []
        for index, row in enumerate(rows):
            source = row['source']; proof = source_rows[source['path']]
            if index < 3:
                p.require(proof['status'] == 'exact' and proof['sha256'] == proof['archive_sha256'] == source['sha256'], 'Exact upstream startup source')
            else:
                transformed = inputs['sysconfig']
                p.require(proof['archive_sha256'] == transformed['upstream_sha256'] and transformed['complete_bytes_match'] and transformed['predicted_installed_sha256'] == source['sha256'], 'Verified sysconfig relocation source')
            actual, raw = read_file(source['path'], 2**20, guard, source['sha256'], source['bytes']); sources.append(actual); observed.append(actual)
            cache, raw_cache = read_file(row['path'], 2**20, guard); observed.append(cache)
            p.require(cache['identity'] == row['cache_identity'] and raw_cache[:16].hex() == row['header_hex'], 'Original observed cache unchanged')
            header = reconstruct_header(bytes.fromhex('cb0d0d0a'), actual['identity'], len(raw))
            p.require(raw_cache[:16] == header, 'Full timestamp header matches current authenticated source')
            caches.append((cache, raw_cache, header))
        p.require(PYTHON_ALIAS.is_symlink() and os.readlink(PYTHON_ALIAS) == str(PYTHON_CANONICAL) and PYTHON_ALIAS.resolve() == PYTHON_CANONICAL, 'Exact observed uv minor-version alias before child startup')
        alias_identity = ident(PYTHON_ALIAS.lstat())
        argv = [PYTHON, '-I', '-S', '-B', '-X', 'pycache_prefix=' + str(prefix), '-c', CHILD, str(prefix)] + [s['path'] for s in sources]
        env = dict(PATH='/usr/bin:/bin', LANG='C.UTF-8', LC_ALL='C.UTF-8', OMP_NUM_THREADS='1')
        result.update(argv=argv, environment=env, started_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        with (OUT / 'stdout.txt').open('xb') as stdout, (OUT / 'stderr.txt').open('xb') as stderr:
            child = subprocess.Popen(argv, env=env, cwd=OUT, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, close_fds=True, start_new_session=True)
            deadline = time.monotonic() + 5
            while True:
                candidate = p.proc(child.pid)
                if candidate is not None and candidate['argv'] == argv and candidate['ppid'] == owner['pid']: break
                p.require(child.poll() is None and time.monotonic() < deadline, 'Actual child identity before completion')
                time.sleep(0.01)
            result['child_owner'] = candidate
            p.once(OUT / 'child-owner.json', dict(owner=result['child_owner']))
            try:
                deadline = time.monotonic() + 30
                while child.poll() is None:
                    guard(); p.require(time.monotonic() < deadline, 'Finite isolated child deadline'); time.sleep(0.1)
                child.wait()
            except BaseException:
                lifecycle.drain(child, result.get('child_owner'), p); raise
        p.require(not lifecycle.group_members(child.pid), 'Entire successful child group gone')
        result['child_exit_code'] = child.returncode
        stdout, raw = read_file(OUT / 'stdout.txt', 2 * 2**20, guard)
        stderr, _ = read_file(OUT / 'stderr.txt', 2 * 2**20, guard)
        result.update(stdout=stdout, stderr=stderr)
        p.require(child.returncode == 0, 'Isolated verifier child exited zero')
        returned = ast.literal_eval(raw.decode('utf-8'))
        allowed = {s['path']: s['path'] for s in sources[:3]}
        allowed.update({str(PYTHON_ALIAS / Path(s['path']).relative_to(PYTHON_CANONICAL)): s['path'] for s in sources[:3]})
        p.require(ident(PYTHON_ALIAS.lstat()) == alias_identity and os.readlink(PYTHON_ALIAS) == str(PYTHON_CANONICAL), 'Startup alias unchanged after child exit')
        for name, origin, filename in returned['module_origins']:
            p.require(origin in ('built-in', 'frozen') or origin in allowed and filename == origin and str(Path(origin).resolve()) == allowed[origin] or name == '__main__' and origin is filename is None, 'Only preauthenticated startup source imports: ' + name)
        p.require(returned['cache_prefix'] == str(prefix) and not list(prefix.iterdir()) and returned['magic'] == 'cb0d0d0a' and returned['marshal_version'] == 4, 'Exact isolated cache configuration')
        p.require(len(returned['compiled']) == 4, 'Four complete compiled results')
        comparisons = []
        for source, (cache, existing, header), value in zip(sources, caches, returned['compiled']):
            filename, co_filename, payload_hex = value
            p.require(filename == co_filename == source['path'], 'Original exact absolute compile filename')
            generated = header + bytes.fromhex(payload_hex)
            comparisons.append(dict(source=source, cache=cache, compile_filename=co_filename,
                                    regenerated_bytes=len(generated), regenerated_sha256=hashlib.sha256(generated).hexdigest(),
                                    complete_bytes_equal=generated == existing))
        for row in observed:
            p.require(ident(Path(row['path']).lstat()) == row['identity'], 'All inputs retain identity after child exit')
        result.update(complete=True, comparisons=comparisons, matched=sum(row['complete_bytes_equal'] for row in comparisons),
                      module_origins=returned['module_origins'], interpreter=interpreter,
                      startup_alias=dict(path=str(PYTHON_ALIAS), target=str(PYTHON_CANONICAL), identity=alias_identity),
                      all_inputs=observed, regenerated_code_objects_executed=False,
                      authenticated_startup_sources_executed=True, child_group_empty=True,
                      target_caches_imported=False, original_environment_modified=False)
        guard()
    except BaseException as error:
        result.update(complete=False, error=repr(error))
        raise
    finally:
        for sig in saved_handlers: signal.signal(sig, signal.SIG_IGN)
        lifecycle.release_after_drain(child, result.get('child_owner'), p,
                                      SimpleNamespace(release_lock=lambda retained: None), fd, OUT if OUT.exists() else None)
        if OUT.exists():
            result['elapsed_seconds'] = time.monotonic() - started
            p.once(OUT / 'result.json', result)
            print(json.dumps(dict(result=external.ref(OUT / 'result.json'), complete=result.get('complete'), matched=result.get('matched'))), flush=True)
        os.close(fd)
        for sig, handler in saved_handlers.items(): signal.signal(sig, handler)


if __name__ == '__main__':
    main()
