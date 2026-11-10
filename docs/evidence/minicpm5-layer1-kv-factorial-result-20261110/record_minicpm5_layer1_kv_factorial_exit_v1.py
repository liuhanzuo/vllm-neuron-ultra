"""External receipt after the calling tool reports an actual terminal exit."""
import argparse
import datetime
import json
from pathlib import Path
import sys
R=Path(__file__).resolve().parents[1];Q=R/'research';sys.path.insert(0,str(R))
import minicpm5_layer1_kv_factorial_cpu_runner_v1 as runner
p=runner.p

def run(session,chunk):
    p.user_bus();fd,owner,lock=runner.post.external.acquire_lock()
    try:
        livepath=Q/'minicpm5-layer1-kv-factorial-r1-actual-live-20261110-v1.json';live=p.read(livepath)
        assert live['actual_execution_handle']==dict(kind='tool_session',session_id=session)
        p.gone(live['owner']);spec=p.read_ref(live['spec']);root=Path(spec['output_root'])
        names=dict(admission='admission.json',intent='launch-intent.json',drains='predecessor-drain.json',
            coordinator_owner='coordinator-owner.json',rank0_owner='rank0-owner.json',rank1_owner='rank1-owner.json',
            rank0_result='rank0-result.json',rank1_result='rank1-result.json',children='children-exits.json',
            coordinator='coordinator-result.json',service_exit='service-exit.json',outer='outer-result.json',cleanup='cleanup-drained.json')
        refs={k:runner.reference(root/v)for k,v in names.items()};records={k:p.read_ref(v)for k,v in refs.items()}
        a=records['admission'];co=records['coordinator_owner'];ranks=[records['rank%d_owner'%i]for i in (0,1)]
        assert a['outer_owner']==live['owner']==records['outer']['owner'] and a['lock_identity']==lock
        assert records['outer']['complete'] and records['children']['exit_codes']==records['coordinator']['rank_exits']==[0,0]
        assert records['service_exit']['successful'] and records['cleanup']['physically_drained']
        assert records['cleanup']['original_failure']is None
        for old in [co,*ranks]:p.gone(old)
        drain=runner.verify_owned_drain(spec,refs['admission'],a['attempt_id'],co,True)
        p.active_heavy();p.clear_window();p.lock_proof(owner,lock)
        path=Q/'minicpm5-layer1-kv-factorial-r1-actual-exit-20261110-v1.json'
        p.once(path,dict(schema='minicpm5_layer1_kv_factorial_actual_exit_v1',reviewer='/root',
            observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),actual_exit_observed=True,actual_exit_code=0,
            actual_execution_handle=live['actual_execution_handle'],tool_chunk=chunk,live=runner.reference(livepath),
            spec=live['spec'],review=live['review'],owner=live['owner'],coordinator=co,ranks=ranks,outputs=refs,
            lock_identity=lock,all_four_processes_gone=True,cgroup_empty=True,physical_drain=drain,
            independent_post_exit_raw_reread_pending=True,full_model_qualified=False,new_speedup_claim=False))
        print(json.dumps(runner.reference(path)))
    finally:runner.post.external.release_lock(fd)
if __name__=='__main__':
    args=argparse.ArgumentParser();args.add_argument('--session',type=int,required=True);args.add_argument('--chunk',required=True)
    a=args.parse_args();run(a.session,a.chunk)
