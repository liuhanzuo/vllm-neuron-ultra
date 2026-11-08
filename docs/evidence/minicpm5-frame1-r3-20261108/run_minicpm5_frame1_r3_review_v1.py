"""Bounded serial post-exit metadata and raw-byte review; no model imports."""
from pathlib import Path
import argparse
import datetime
import json
import os
import resource
import signal
import sys
import time
R=Path(__file__).resolve().parents[1];Q=R/'research';sys.path.insert(0,str(R))
import minicpm5_frame1_mlp_runner_v5 as m
TERMINAL=dict(path=str(Q/'minicpm5-frame1-r3-root-terminal-20261108-v1.json'),sha256='6dd0904acb31baa784905bb3b6ec74dbf17e3e390c432e16f16e7447b341e5ef')
REVIEWER_PINS={'review_minicpm5_frame1_metadata_v1.py': '8d5243d29480870b97b3821a2a54bceee3228d05eb72817c61ad872500e38ce3', 'review_minicpm5_frame1_raw_boundaries_v1.py': '22a5e192c31ad6fe838e5ab9bb6e274e982b2a3d1239c1776d934bb2ba5d0895'}

def main():
    terminal=m.p.read_ref(TERMINAL)
    assert terminal['actual_exit']==dict(session_id=15892,exit_code=0,tool_chunk='aa5edd')
    assert terminal['actual_tool_exit_observed']and terminal['all_three_groups_physically_drained']
    for name,sha in REVIEWER_PINS.items():assert m.p.sha(Q/name)==sha
    import review_minicpm5_frame1_metadata_v1 as meta
    import review_minicpm5_frame1_raw_boundaries_v1 as raw
    spec=m.p.read_ref(terminal['spec']);root=Path(spec['output_root']);a=m.p.read_ref(terminal['references']['admission.json'])
    fd,owner,lock=m.old.post.external.acquire_lock();started=time.monotonic()
    out=Q/'minicpm5-frame1-r3-independent-review-run-20261108-v1.json'
    try:
        assert lock==terminal['lock_identity'];m.p.active_heavy()
        for prior in [terminal['outer_owner'],terminal['terminal_observer'],*terminal['phase_owners'].values(),*terminal['rank_owners']]:m.p.gone(prior)
        for phase,unit in spec['units'].items():
            props=m.p.properties(unit);m.service_identity(spec,terminal['references']['admission.json'],a['attempt_id'],phase,props,terminal['phase_owners'][phase])
            assert m.p.terminal(props);m.p.empty_group(m.p.GROUP_PREFIX+unit+'.service')
        os.sched_setaffinity(0,{6});os.nice(19)
        resource.setrlimit(resource.RLIMIT_AS,(2*2**30,2*2**30))
        resource.setrlimit(resource.RLIMIT_CPU,(600,605))
        resource.setrlimit(resource.RLIMIT_FSIZE,(16*2**20,16*2**20))
        def timeout(signum,frame):raise TimeoutError('Independent review exceeded 900-second wall budget')
        signal.signal(signal.SIGALRM,timeout);signal.alarm(900)
        admission=dict(terminal=TERMINAL,owner=owner,lock_identity=lock,reviewers=REVIEWER_PINS,
            source=m.old.reference(Path(__file__)),resources=dict(cpus=[6],address_space_bytes=2*2**30,cpu_seconds=600,wall_seconds=900,file_bytes=16*2**20),
            actual_self_exit_claimed=False,full_model_qualified=False)
        m.worker.once(Q/'minicpm5-frame1-r3-independent-review-admission-20261108-v1.json',admission)
        meta_path=Q/'minicpm5-frame1-r3-independent-metadata-review-20261108-v1.json'
        meta.main(argparse.Namespace(spec=terminal['spec']['path'],spec_sha256=terminal['spec']['sha256'],
            result=terminal['references']['result.json']['path'],result_sha256=terminal['references']['result.json']['sha256'],output=str(meta_path)))
        raw_path=Q/'minicpm5-frame1-r3-independent-raw-review-20261108-v1.json'
        raw.main(argparse.Namespace(metadata_review=str(meta_path),metadata_review_sha256=m.p.sha(meta_path),output=str(raw_path)))
        usage=resource.getrusage(resource.RUSAGE_SELF)
        value=dict(complete=True,owner=owner,terminal=TERMINAL,metadata_review=m.old.reference(meta_path),raw_review=m.old.reference(raw_path),
            measured_resources=dict(elapsed_seconds=time.monotonic()-started,user_seconds=usage.ru_utime,system_seconds=usage.ru_stime,peak_rss_bytes=usage.ru_maxrss*1024),
            torch_imported='torch'in sys.modules,actual_self_exit_claimed=False,full_model_qualified=False,new_speedup_claim=False)
        assert not value['torch_imported'];m.worker.once(out,value);print(json.dumps(value),flush=True)
    except BaseException as error:
        m.old.guarded_receipt(Q/'minicpm5-frame1-r3-independent-review-failure-20261108-v1.json',
            dict(owner=owner,terminal=TERMINAL,error_type=type(error).__name__,error=str(error),full_model_qualified=False),error)
        raise
    finally:
        signal.alarm(0);m.old.post.external.release_lock(fd)
if __name__=='__main__':main()
