"""Lossless two-file recovery after r12 failed, under the original serial lock.

No model launch, registry mutation, changed model gates, or Nemotron admission.
The original below-floor condition is explicitly allowed only for this recovery.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
sys.path[:0] = [str(ROOT), str(ROOT/'research')]
import compact_closed_journals_after_r11_v3 as a
p = a.p
Q = ROOT/'research'
OUT = Path('/mnt/nvme/closed-journals-after-r12-failure-20261111-v1')
SPEC = Q/'nemotron9b-package-runtime-v27-r12-20261110-spec.json'
SPEC_SHA = '4186c9a29921b96fc3a55fb04040cf70977ca921130df288466c8d27e65feaaa'
CONTROL = Path('/mnt/nvme/nemotron9b-package-r12-20261110-control')
NAMES = tuple('user-1000@72cf4ba58fce415ab780db3f965e741b-'+s+'.journal' for s in (
    '000000000061d427-00065fd8da131da4', '000000000063b38b-00065fe3ffada253'))
FILES = [str(a.JOURNALS/n) for n in NAMES]
PLAN = Q/'closed-journals-after-r12-failure-plan-20261111-v1.json'
REVIEW = Q/'closed-journals-after-r12-failure-root-review-20261111-v1.json'
SOURCES = [Path(__file__).resolve(), Path(a.__file__).resolve(), Path(p.__file__).resolve(),
           ROOT/'minicpm5_layer0_runtime_auth_v1.py']
LOCK = dict(device=66306, inode=8294421)
require = a.require


def ref(path):
    return dict(path=str(path), sha256=a.sha(path))


def terminal():
    require(a.sha(SPEC) == SPEC_SHA, 'Original fixed r12 specification')
    spec=json.loads(SPEC.read_text()); failed=json.loads((CONTROL/'failed-terminal.json').read_text())
    require(failed['passed'] is False and failed['all_started_groups_physically_drained'] is True
            and failed['model_qualified'] is False, 'Original failure retained')
    require(not (CONTROL/'root-release.json').exists(), 'Recovery precedes original supported failed release')
    require(p.proc(727516) is None and p.proc(723810) is None, 'Original controller and external launcher gone')
    require(len(spec['units']) == 53, 'All 53 units')
    rows={}
    for stage,unit in spec['units'].items():
        value=p.properties(unit)
        require(p.terminal(value), 'No running r12 unit: '+stage)
        rows[stage]=dict(properties=value,group=p.empty_group(p.GROUP_PREFIX+unit+'.service'))
    value=rows['controller']['properties']
    require(value['InvocationID']=='1eaac4fd6907434688e74b4a9cd91bd2'
        and value['ExecMainPID']=='727516' and value['ExecMainCode']=='1'
        and value['ExecMainStatus']=='1', 'Authoritative original failed exit')
    p.active_heavy();p.clear_window()
    return dict(rows=rows,failed_terminal=ref(CONTROL/'failed-terminal.json'),
                old_external_exit_code=None,old_external_exit_observed=False)


def main(execute, digest):
    require(sys.dont_write_bytecode and __debug__ and os.getuid()==os.geteuid()==1000, 'UID1000 Python -B')
    p.user_bus()
    fd=os.open(p.LOCK,os.O_RDWR|os.O_NOFOLLOW);locked=False
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);locked=True
        owner=p.proc(os.getpid());p.lock_proof(owner,LOCK)
        prior=terminal()
        pins={str(x):a.sha(x) for x in SOURCES}
        canonical={n:a.sha(ROOT/n) for n in ('continuation-state.json','roadmap.json','best-known-configurations.json')}
        before=dict(root=a.free('/'),nvme=a.free('/mnt/nvme'))
        if not execute:
            a.unused(FILES); rows=[]
            for path in FILES:
                a.archived(path); ident=a.identity(path)
                require(ident['st_size']==128*2**20 and ident['st_uid']==0, 'Exact archived extent and owner')
                rows.append(dict(path=path,identity=ident,sha256=a.sha(path,limit=128*2**20)))
            print(json.dumps(a.save(PLAN,dict(schema='closed_journals_after_r12_failure_plan_v1',
                files=rows,source_sha256=pins,canonical_sha256=canonical,terminal=prior,
                output=str(OUT),lock=LOCK,before_space=before,automatic_successor_launch=False,
                root_floor_bytes=a.ROOT_FLOOR,nvme_floor_bytes=a.NVME_FLOOR,
                every_original_byte_retained=True,fresh_nemotron_admitted=False))),flush=True)
            return
        require(a.sha(PLAN)==digest, 'Exact reviewed plan')
        plan=json.loads(PLAN.read_text());review=json.loads(REVIEW.read_text())
        require(review['plan']==ref(PLAN) and review['source_sha256']==pins and review['passed'] is True
                and review['reviewer']=='/root' and review['blockers']==[], 'Separate root review')
        require(plan['source_sha256']==pins and plan['canonical_sha256']==canonical
                and plan['terminal']==prior and plan['output']==str(OUT) and plan['lock']==LOCK
                and [r['path'] for r in plan['files']]==FILES, 'Unchanged exact plan')
        require(before['nvme']>=a.NVME_FLOOR+a.ARCHIVE_CAP+a.META_CAP, 'Archive space reserved')
        require(not OUT.exists(), 'Fresh maintenance output')
        started=time.monotonic();os.umask(0o077);os.sched_setaffinity(0,{4});os.nice(15)
        resource.setrlimit(resource.RLIMIT_AS,(2*2**30,)*2)
        resource.setrlimit(resource.RLIMIT_CPU,(900,905))
        resource.setrlimit(resource.RLIMIT_FSIZE,(a.ARCHIVE_CAP,)*2)
        OUT.mkdir(mode=0o700)
        a.save(OUT/'intent.json',dict(plan=ref(PLAN),review=ref(REVIEW),owner=owner,lock=LOCK,
              before_space=before,original_below_floor_preserved=True,automatic_successor_launch=False))
        def guard():
            p.lock_proof(owner,LOCK);p.clear_window()
            require(time.monotonic()-started<1200, 'Recovery deadline')
            require(a.free('/')>=before['root']-a.META_CAP and a.free('/mnt/nvme')>=a.NVME_FLOOR,
                    'Bounded recovery writes and NVMe floor')
            require(sum(x.stat().st_size for x in OUT.iterdir())<=a.ARCHIVE_CAP+a.META_CAP, 'Output cap')
        a.unused(FILES);rows=[]
        for row in plan['files']:
            guard();a.archived(row['path'])
            rows.append(a.archive(row,OUT/(Path(row['path']).name+'.gz'),guard))
        verified=a.save(OUT/'verified-before-retirement.json',dict(files=rows,every_archive_decompressed=True))
        terminal();a.unused(FILES)
        for row in rows:
            require(a.identity(row['source'])==row['source_identity'], 'All sources stable before any unlink')
            a.archived(row['source'])
        for i,row in enumerate(rows):
            guard();a.unused([row['source']]);a.archived(row['source'])
            require(a.identity(row['source'])==row['source_identity'], 'Exact source identity at unlink')
            require(a.identity(row['archive'])==row['archive_identity'] and
                    a.sha(row['archive'],guard,a.ARCHIVE_CAP)==row['archive_sha256'], 'Verified archive unchanged')
            subprocess.run(['sudo','-n','/usr/bin/rm','--',row['source']],check=True,timeout=15)
            require(not Path(row['source']).exists(), 'Allowlisted source retired')
            a.save(OUT/('retired-%02d.json'%i),row)
        require({n:a.sha(ROOT/n) for n in canonical}==canonical, 'Frozen canonical state')
        guard();after=dict(root=a.free('/'),nvme=a.free('/mnt/nvme'))
        require(after['root']>=a.ROOT_FLOOR+128*2**20, 'Restore original root floor plus recovery headroom')
        a.save(OUT/'result.json',dict(complete=True,plan=ref(PLAN),review=ref(REVIEW),owner=owner,
            verification=verified,before_space=before,after_space=after,
            original_bytes=sum(r['original_bytes'] for r in rows),
            compressed_bytes=sum(r['compressed_bytes'] for r in rows),
            every_original_byte_retained=True,canonical_unchanged=True,
            fresh_nemotron_admitted=False,automatic_successor_launch=False,new_speedup_claim=False))
        print(json.dumps(dict(complete=True,output=str(OUT),after_space=after)),flush=True)
    finally:
        if locked:fcntl.flock(fd,fcntl.LOCK_UN)
        os.close(fd)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true')
    parser.add_argument('--plan-sha256');args=parser.parse_args()
    main(args.execute,args.plan_sha256)
