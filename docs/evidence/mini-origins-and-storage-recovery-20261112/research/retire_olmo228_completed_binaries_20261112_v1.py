"""Retire six unused binaries from the completed isolated OLMo compiler trial."""
import os
import sys
import json
import fcntl
import hashlib
import time
from pathlib import Path

ROOT = Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
Q = ROOT/'research'
CAT = Q/'minicpm5-operator-post-r2-serial-workloads-root-catalog-20261112-v89.json'
CAT_SHA = 'c1b5dccb57a3079bce25e8ed32fce9bd5890494f3b9652516af3b19d839c421d'
PLAN = Q/'olmo228-completed-binary-retirement-candidates-20261112-v1.json'
OUT = Path('/mnt/nvme/olmo228-completed-binary-retirement-20261112-v1')


def ref(p):
    return dict(path=str(p), sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())


def main():
    assert sys.dont_write_bytecode and __debug__
    fd = os.open(ROOT/'serial-enrollment.lock', os.O_RDWR | os.O_NOFOLLOW)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        st = os.fstat(fd)
        assert (st.st_dev, st.st_ino) == (66306, 8294421)
        sys.path[:0] = [str(ROOT), str(Q)]
        import compact_closed_journals_after_r11_v3 as a
        import retire_olmo228_download_archives_20261106_v1 as old
        a.p.user_bus(); a.p.active_heavy(); a.p.clear_window()
        assert ref(CAT)['sha256'] == CAT_SHA
        catalog = json.loads(CAT.read_text()); handoff = a.load(catalog['current_handoff'])
        assert len(catalog['runs']) == 136 and not handoff['running'] and not handoff['live_tool_sessions']
        plan = json.loads(PLAN.read_text()); rows = plan['files']
        assert len(rows) == 6 and len({r['path'] for r in rows}) == 6
        base = ROOT/'olmo3-compiler228/env/lib/python3.12/site-packages/neuronxcc/starfish/bin'
        assert {Path(r['path']).name for r in rows} == {'xla_infergoldens','hlo2penguin','hlo-opt','hlo-neff-wrapper','hlo_convert','snapshot-unpack'}
        for row in rows:
            assert Path(row['path']).parent == base and a.identity(row['path']) == row['identity']
        assert not set(r['path'] for r in rows) & a.absolute_refs(catalog)
        services = old.terminal()
        old.open_users([r['path'] for r in rows])
        canonical = {n: ref(ROOT/n)['sha256'] for n in ('continuation-state.json','roadmap.json','best-known-configurations.json')}
        assert canonical == handoff['canonical_sha256']
        evidence = [old.D/n for n in ('status.json','summary.json','plan.json','provision-result.json','installed-payloads.json','download-progress.json','compile-result.json')]
        lockpath = Path(json.loads((old.D/'plan.json').read_text())['package_lock'])
        evidence.append(lockpath)
        preserved = {str(p): ref(p)['sha256'] for p in evidence}
        owner = a.p.proc(os.getpid()); started = time.monotonic()
        before = dict(root=a.free('/'), nvme=a.free('/mnt/nvme'))
        assert before['root'] >= a.ROOT_FLOOR and before['nvme'] >= a.NVME_FLOOR
        assert not OUT.exists(); OUT.mkdir(mode=0o700)
        sources = {str(p): ref(p)['sha256'] for p in [Path(__file__).resolve(),Path(a.__file__).resolve(),Path(old.__file__).resolve(),Path(a.p.__file__).resolve(),ROOT/'minicpm5_layer0_runtime_auth_v1.py']}
        a.save(OUT/'intent.json',dict(owner=owner,catalog=ref(CAT),plan=ref(PLAN),source_sha256=sources,terminal_services=services,preserved_metadata_sha256=preserved,before_space=before))

        def guard():
            assert time.monotonic()-started < 300
            current = os.stat(ROOT/'serial-enrollment.lock',follow_symlinks=False)
            assert (current.st_dev,current.st_ino) == (st.st_dev,st.st_ino)
            assert a.free('/') >= a.ROOT_FLOOR and a.free('/mnt/nvme') >= a.NVME_FLOOR

        for row in rows:
            assert a.sha(row['path'],guard,256*2**20) == row['expected_sha256']
            assert a.identity(row['path']) == row['identity']
        a.save(OUT/'verified-before-retirement.json',dict(files=rows,current_full_hashes_match_historical_installation=True))
        old.terminal(); scan = old.open_users([r['path'] for r in rows]); a.p.active_heavy(); a.p.clear_window()
        removed = []
        try:
            for row in rows:
                guard(); assert a.identity(row['path']) == row['identity']
                Path(row['path']).unlink(); removed.append(row['path'])
                a.save(OUT/('retired-%02d.json' % len(removed)),row)
        finally:
            a.save(OUT/'retirement-receipt.json',dict(removed=removed,old_trial_environment_executable=False,restoration=plan['reconstruction']))
        assert len(removed) == 6 and all(not Path(p).exists() for p in removed)
        assert preserved == {p:ref(p)['sha256'] for p in preserved}
        assert canonical == {n:ref(ROOT/n)['sha256'] for n in canonical}
        after = dict(root=a.free('/'),nvme=a.free('/mnt/nvme'))
        result = dict(complete=True,owner=owner,plan=ref(PLAN),source_sha256=sources,removed=removed,removed_allocated_bytes=plan['reclaimable_allocated_bytes'],before_space=before,after_space=after,terminal_services=services,unused=scan,preserved_metadata_sha256=preserved,canonical_unchanged=True,old_trial_environment_executable=False,restoration=plan['reconstruction'],new_speedup_claim=False)
        a.save(OUT/'result.json',result)
        print(json.dumps({k:result[k] for k in ['complete','removed_allocated_bytes','after_space']}),flush=True)
    finally:
        fcntl.flock(fd,fcntl.LOCK_UN); os.close(fd)


if __name__ == '__main__':
    main()
