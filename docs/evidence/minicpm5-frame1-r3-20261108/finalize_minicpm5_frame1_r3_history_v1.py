"""Finalize actual r3 diagnostic and independent review after both tool exits."""
from pathlib import Path
import copy
import datetime
import json
import os
import sys
R=Path(__file__).resolve().parents[1];Q=R/'research';sys.path.insert(0,str(R))
import minicpm5_frame1_completed_history_v3 as h
import minicpm5_frame1_mlp_runner_v5 as m

def main():
    prior=h.previous.load(h.PRIOR);terminal=h.previous.load(h.TERMINAL)
    review_ref=m.old.reference(Q/'minicpm5-frame1-r3-independent-review-run-20261108-v1.json');review=m.p.read_ref(review_ref)
    metadata=m.p.read_ref(review['metadata_review']);raw=m.p.read_ref(review['raw_review'])
    assert review['complete']and review['terminal']==h.TERMINAL and raw['checks']['all15_full_payload_hashes_verified']
    assert raw['checks']['observed_boundary_metrics_recomputed']and not raw['checks']['model_math_reexecuted']
    spec=m.p.read_ref(terminal['spec']);root=Path(spec['output_root']);a=m.p.read_ref(terminal['references']['admission.json'])
    fd,owner,lock=m.old.post.external.acquire_lock()
    try:
        assert lock==terminal['lock_identity'];m.p.active_heavy();h.previous.physical_drain(prior)
        known=[terminal['outer_owner'],terminal['terminal_observer'],*terminal['rank_owners'],review['owner']]
        for p in [*known,*terminal['phase_owners'].values()]:m.p.gone(p)
        for phase,unit in spec['units'].items():
            props=m.p.properties(unit);m.service_identity(spec,terminal['references']['admission.json'],a['attempt_id'],phase,props,terminal['phase_owners'][phase])
            assert m.p.terminal(props)and props['Result']=='success'and props['ExecMainStatus']=='0';m.p.empty_group(m.p.GROUP_PREFIX+unit+'.service')
        assert {n:m.p.sha(R/n)for n in m.CANONICAL}==m.CANONICAL
        final=dict(schema='minicpm5_frame1_r3_final_root_audit_v1',reviewer='/root',observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            terminal=h.TERMINAL,actual_model_exit=terminal['actual_exit'],actual_review_exit=dict(session_id=14042,exit_code=0,tool_chunk='c537bd'),
            independent_review=review_ref,finalizer_owner=owner,lock_identity=lock,all_known_predecessors_gone=True,all_groups_physically_drained=True,
            diagnostic_complete=True,all15_full_captures_reproduced=True,independent_saved_boundary_metrics_verified=True,
            independent_fixed_input_counterfactual_reexecution=False,unique_final_KL_cause_proven=False,
            finalizer_self_exit_claimed=False,full_model_qualified=False,new_speedup_claim=False)
        final_ref=m.worker.once(Q/'minicpm5-frame1-r3-final-root-audit-20261108-v1.json',final)
        value=copy.deepcopy(prior);value['rows']+=h.added_rows(terminal)
        roles=['outer','terminal_observer','cpu_rank0','cpu_rank1','independent_review']
        value['processes'] += [dict(p,role='minicpm5_frame1_r3_'+role)for p,role in zip(known,roles)]
        value['processes'].append(dict(owner,role='minicpm5_frame1_r3_history_finalizer'))
        if 'process_predecessors'in value:value['process_predecessors']=copy.deepcopy(value['processes'])
        refs=[h.PRIOR,h.TERMINAL,final_ref,review_ref,review['metadata_review'],review['raw_review'],terminal['spec'],terminal['review'],terminal['live'],
              m.old.reference(Path(__file__)),m.old.reference(Path(h.__file__)),m.old.reference(Q/'run_minicpm5_frame1_r3_review_v1.py'),
              m.old.reference(Q/'minicpm5-frame1-r3-independent-review-admission-20261108-v1.json'),
              m.old.reference(Q/'minicpm5-frame1-startup-source-preparation-20261108-v1.json')]
        refs += [m.old.reference(p)for p in sorted(root.rglob('*'))if p.suffix in ('.json','.jsonl')]
        refs += [dict(path=p,sha256=sha)for p,sha in spec['new_sources'].items()]
        for ref in refs:
            assert ref['path']not in value['source_sha256']or value['source_sha256'][ref['path']]==ref['sha256']
            value['source_sha256'][ref['path']]=ref['sha256']
        value.update(schema='root_post_minicpm5_frame1_r3_complete_predecessors_v1',complete_count=925,appended_from=h.PRIOR,
            appended_terminal_audit=final_ref,latest_producer='minicpm5_frame1_r3',latest_producer_qualified=False,
            latest_producer_exit=final_ref,latest_producer_terminal_audit=final_ref)
        h.validate(value);raw=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode();assert len(raw)<=32*2**20
        with h.PATH.open('xb')as f:assert f.write(raw)==len(raw);f.flush();os.fsync(f.fileno())
        print(json.dumps(dict(catalog=m.old.reference(h.PATH),final_audit=final_ref,services=925,processes=179,finalizer_owner=owner,full_model_qualified=False)),flush=True)
    finally:m.old.post.external.release_lock(fd)
if __name__=='__main__':main()
