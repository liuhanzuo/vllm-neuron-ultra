"""Lossless archived-journal recovery after actual Qwen r8 terminal.

Reuse established full gzip roundtrip and unused/archive checks. No model launch.
"""
import os
import sys
import json
import fcntl
import hashlib
import time
import resource
import subprocess
from pathlib import Path

ROOT = Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
Q = ROOT / 'research'
OUT = Path('/mnt/nvme/closed-journals-after-qwen-r8-20261112-v1')
LOCK = ROOT / 'serial-enrollment.lock'
CAT = Q / 'minicpm5-operator-post-r2-serial-workloads-root-catalog-20261112-v87.json'
CAT_SHA = '952196878ed23815265ea5e576af4691c27ecb578cfb2a5ab7af39d52f308934'
AUDIT = Q / 'qwen35-sampler-r8-root-actual-exit-20261112-v1.json'
AUDIT_SHA = 'b9f5f037e67b6ef7f792b51c8abdc1afc227632a48e92f4fdb50e2069c751394'
FILES_JSON = Q / 'closed-journals-after-qwen-r8-files-20261112-v1.json'


def ref(path):
    return dict(path=str(path), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())


def main():
    assert sys.dont_write_bytecode and __debug__ and os.getuid() == 1000
    fd = os.open(LOCK, os.O_RDWR | os.O_NOFOLLOW)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        st = os.fstat(fd)
        assert (st.st_dev, st.st_ino) == (66306, 8294421)
        sys.path[:0] = [str(ROOT), str(Q)]
        import compact_closed_journals_after_r11_v3 as a
        p = a.p
        p.user_bus(); p.active_heavy(); p.clear_window()
        assert ref(CAT)['sha256'] == CAT_SHA and ref(AUDIT)['sha256'] == AUDIT_SHA
        catalog = json.loads(CAT.read_text()); audit = json.loads(AUDIT.read_text())
        assert len(catalog['runs']) == 134 and catalog['runs'][-1]['terminal'] == ref(AUDIT)
        handoff = a.load(catalog['current_handoff'])
        assert handoff['active_serial'] is None and not handoff['running'] and not handoff['live_tool_sessions']
        assert audit['actual_tool_exit_observed'] and audit['physically_drained'] and audit['owners_gone']
        for row in audit['owners']:
            cur = p.proc(row['pid'])
            assert cur is None or str(cur['start_ticks']) != str(row['start_ticks'])
        files = json.loads(FILES_JSON.read_text())
        assert 0 < len(files) <= 64 and len(files) == len(set(files))
        for path in files:
            target = Path(path)
            assert target.parent == a.JOURNALS and '@' in target.name and target.suffix == '.journal'
            assert target.name.startswith(('user-1000@', 'system@'))
        assert not set(files) & (a.absolute_refs(catalog) | a.absolute_refs(handoff))
        canonical = {n: ref(ROOT/n)['sha256'] for n in ('continuation-state.json', 'roadmap.json', 'best-known-configurations.json')}
        assert canonical == handoff['canonical_sha256'] and not OUT.exists()
        os.umask(0o077); os.sched_setaffinity(0, {4}); os.nice(15)
        resource.setrlimit(resource.RLIMIT_AS, (2*2**30,)*2)
        resource.setrlimit(resource.RLIMIT_CPU, (900, 905))
        resource.setrlimit(resource.RLIMIT_FSIZE, (a.ARCHIVE_CAP,)*2)
        owner = p.proc(os.getpid()); started = time.monotonic()
        before = dict(root=a.free('/'), nvme=a.free('/mnt/nvme'))
        assert before['root'] >= a.ROOT_FLOOR and before['nvme'] >= a.NVME_FLOOR+a.ARCHIVE_CAP+a.META_CAP
        OUT.mkdir(mode=0o700)
        sources = {str(x): ref(x)['sha256'] for x in [Path(__file__).resolve(), Path(a.__file__).resolve(), Path(p.__file__).resolve(), ROOT/'minicpm5_layer0_runtime_auth_v1.py']}
        a.save(OUT/'intent.json', dict(owner=owner, lock_identity=dict(device=st.st_dev, inode=st.st_ino), catalog=ref(CAT), audit=ref(AUDIT), before_space=before, source_sha256=sources, file_list=ref(FILES_JSON), files=files, canonical_sha256=canonical))

        def guard():
            current = os.stat(LOCK, follow_symlinks=False)
            assert (current.st_dev, current.st_ino) == (st.st_dev, st.st_ino)
            assert time.monotonic()-started < 1200
            assert a.free('/') >= a.ROOT_FLOOR and a.free('/mnt/nvme') >= a.NVME_FLOOR
            assert sum(x.stat().st_size for x in OUT.iterdir()) <= a.ARCHIVE_CAP+a.META_CAP

        a.unused(files); plan = []
        for path in files:
            guard(); a.archived(path); ident = a.identity(path)
            assert ident['st_uid'] == 0 and 0 < ident['st_size'] <= 128*2**20
            plan.append(dict(path=path, identity=ident, sha256=a.sha(path, guard, 128*2**20)))
        a.save(OUT/'plan.json', dict(files=plan, source_sha256=sources))
        rows = []
        for row in plan:
            guard(); rows.append(a.archive(row, OUT/(Path(row['path']).name+'.gz'), guard))
        verification = a.save(OUT/'verified-before-retirement.json', dict(files=rows, every_archive_decompressed=True))
        a.unused(files); p.active_heavy(); p.clear_window()
        for row in rows:
            assert a.identity(row['source']) == row['source_identity']; a.archived(row['source'])
        for i, row in enumerate(rows):
            guard(); a.unused([row['source']]); a.archived(row['source'])
            assert a.identity(row['source']) == row['source_identity']
            assert a.identity(row['archive']) == row['archive_identity'] and a.sha(row['archive'], guard, a.ARCHIVE_CAP) == row['archive_sha256']
            subprocess.run(['sudo', '-n', '/usr/bin/rm', '--', row['source']], check=True, timeout=15)
            assert not Path(row['source']).exists()
            a.save(OUT/('retired-%02d.json' % i), row)
        assert canonical == {n: ref(ROOT/n)['sha256'] for n in canonical}
        after = dict(root=a.free('/'), nvme=a.free('/mnt/nvme')); guard()
        result = dict(complete=True, owner=owner, source_sha256=sources, catalog=ref(CAT), verification=verification, before_space=before, after_space=after, original_bytes=sum(r['original_bytes'] for r in rows), compressed_bytes=sum(r['compressed_bytes'] for r in rows), file_count=len(rows), every_original_byte_retained=True, canonical_unchanged=True, new_speedup_claim=False, automatic_successor_launch=False)
        a.save(OUT/'result.json', result)
        print(json.dumps({k: result[k] for k in ['complete', 'file_count', 'original_bytes', 'compressed_bytes', 'before_space', 'after_space']}), flush=True)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN); os.close(fd)


if __name__ == '__main__':
    main()
