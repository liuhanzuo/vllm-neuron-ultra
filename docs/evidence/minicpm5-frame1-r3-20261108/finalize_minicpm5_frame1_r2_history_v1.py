"""Root r2 failed-startup history append, only actual known process identities."""
from pathlib import Path
import copy
import datetime
import hashlib
import json
import os
import sys
R=Path(__file__).resolve().parents[1];Q=R/'research';sys.path.insert(0,str(R))
import minicpm5_frame1_completed_history_v2 as h
import minicpm5_frame1_mlp_runner_v4 as runner

def compose(prior,terminal,spec,owner):
    require=h.require
    value=copy.deepcopy(prior);value['rows']+=h.new_rows(terminal,spec)
    value['processes']+=[dict(terminal['outer_owner'],role='minicpm5_frame1_r2_outer'),dict(terminal['terminal_observer'],role='minicpm5_frame1_r2_terminal_observer'),dict(owner,role='minicpm5_frame1_r2_history_finalizer')]
    if 'process_predecessors'in value:value['process_predecessors']=copy.deepcopy(value['processes'])
    source_refs=[h.PRIOR,h.TERMINAL,terminal['spec'],terminal['review'],terminal['live'],terminal['admission'],terminal['failure'],terminal['native_journal'],h.ref(Path(__file__)),h.ref(Path(h.__file__))]
    for item in source_refs:
        require(item['path']not in value['source_sha256']or value['source_sha256'][item['path']]==item['sha256'],'No replaced historical source')
        value['source_sha256'][item['path']]=item['sha256']
    value.update(schema='root_post_minicpm5_frame1_r2_complete_predecessors_v1',complete_count=922,appended_from=h.PRIOR,
        appended_terminal_audit=h.TERMINAL,latest_producer='minicpm5_frame1_r2',latest_producer_qualified=False,
        latest_producer_exit=h.TERMINAL,latest_producer_terminal_audit=h.TERMINAL,root_finalized=True,all_planned_launches_accounted_for=True)
    value['minicpm5_frame1_r2_append']=dict(services=value['rows'][919:],processes=value['processes'][170:],terminal=h.TERMINAL,owner_capture_gap_retained=True,full_model_qualified=False)
    return value

def main():
    prior=h.previous.load(h.PRIOR);terminal,admission,spec=h.evidence()
    require=h.require
    fd,owner,lock=runner.old.post.external.acquire_lock()
    try:
        require(lock==terminal['lock_identity'],'Original serial lock retained')
        runner.p.active_heavy();runner.p.gone(terminal['outer_owner']);runner.p.gone(terminal['terminal_observer'])
        sys.path.insert(0,str(Q))
        from research.finalize_smollm3_sio14_root_20261108_v1 import physical_drain
        physical_drain(prior)
        samples=[]
        for _ in range(2):
            observed=[]
            for phase,unit in spec['units'].items():
                props=runner.p.properties(unit)
                require(props['Id']==unit+'.service'and props['LoadState']=='not-found'and runner.p.terminal(props)and
                        not props['InvocationID']and props['MainPID']==props['ControlPID']==props['ExecMainPID']=='0',
                        'Actual failed stage absent')
                group=runner.p.empty_group(runner.p.GROUP_PREFIX+unit+'.service')
                observed.append(dict(phase=phase,properties=props,group=group))
            samples.append(observed)
        require(samples[0]==samples[1],'Two current stable failed-stage snapshots')
        value=compose(prior,terminal,spec,owner)
        h.validate_catalog(value,prior,terminal,spec)
        require({n:runner.p.sha(R/n)for n in spec['canonical_sha256']}==spec['canonical_sha256'],'Original canonical files')
        raw=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode();require(len(raw)<=32*2**20,'Original named envelope bound')
        with h.PATH.open('xb')as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
        print(json.dumps(dict(catalog=dict(path=str(h.PATH),sha256=hashlib.sha256(raw).hexdigest()),rows=922,processes=173,known_finalizer_owner=owner,missing_native_owner_not_fabricated=True,full_model_qualified=False)))
    finally:runner.old.post.external.release_lock(fd)
if __name__=='__main__':main()
