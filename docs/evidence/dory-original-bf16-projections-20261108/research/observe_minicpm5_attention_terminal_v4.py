"""Inspect actual completed Mini attention lifecycle before independent byte review.

Source import does no process/device/weight work. Runtime requires genuine root
external-exit evidence and a separate source review. Never starts/stops a model.
"""
import argparse,datetime,hashlib,json,os,re,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];Q=R/'research';sys.path[:0]=[str(R),str(Q)]
import minicpm5_frame1_attention_runner_v4 as m
import run_minicpm5_attention_review_v4 as post
import review_minicpm5_frame1_attention_metadata_v2 as meta
SELF=Path(__file__).resolve()
PROCESS_KEYS=('pid','start_ticks','ppid','argv','cgroup')
ROLES=('native','cpu-rank0','cpu-rank1')


def process(owner):
    value={k:owner[k]for k in PROCESS_KEYS}
    m.p.require(type(value['pid'])is int and value['pid']>0 and type(value['ppid'])is int and value['ppid']>0,'Actual positive process IDs')
    m.p.require(type(value['start_ticks'])is str and value['start_ticks'].isdigit()and int(value['start_ticks'])>0,'Actual process start tick')
    m.p.require(type(value['argv'])is list and bool(value['argv'])and all(type(x)is str for x in value['argv']),'Actual nonempty process command')
    m.p.require(type(value['cgroup'])is str and value['cgroup'].startswith('0::/'),'Actual process group')
    return value


def external_exit(exited,live):
    m.p.require(exited['schema']=='minicpm5_frame1_attention_actual_external_exit_v1'and exited['reviewer']=='/root'and exited['actual_tool_exit_observed']is True,'Actual external tool exit required')
    m.p.require(live['schema']=='minicpm5_frame1_attention_actual_live_v1'and live['reviewer']=='/root'and live['actual_identity_verified']is True and live['terminal_exit_observed']is False,'Original observed live identity')
    actual=exited['actual_exit']
    m.p.require(type(actual['session_id'])is int and actual['session_id']>0 and type(actual['exit_code'])is int and actual['exit_code']==0 and re.fullmatch('[0-9a-f]{6}',actual['tool_chunk']),'Successful actual tool session/code/chunk')
    m.p.require(type(live['actual_tool_session'])is int,'Original actual session type')
    m.same(actual['session_id'],live['actual_tool_session'],'Exit belongs to original live session')
    for key in ('spec','review','outer_owner'):m.same(exited[key],live[key],'Same original live '+key)
    m.p.require(exited['new_speedup_claim']is False,'External exit makes no speed claim')
    return process(exited['outer_owner'])


def terminal_metadata(root,spec,exited,records,refs):
    """Pure metadata binding; caller must separately prove all owners/groups drained."""
    root=Path(root);a=records['admission.json'];outer=process(exited['outer_owner'])
    for key in ('spec','review'):m.same(a[key],exited[key],'Actual admitted '+key)
    m.same(a['outer_owner'],outer,'Original coordinator admitted');m.same(a['current_catalog'],spec['current_catalog'],'Actual predecessor catalog')
    m.p.require(a['current_history_drained']is True,'Recorded original admission drain')
    phase_owners={};rank_owners=[]
    for phase in m.PHASES:
        owner=records[phase+'-owner.json'];phase_owners[phase]=owner;proc=process(owner)
        m.same(owner['unit'],spec['units'][phase],'Actual phase unit');m.same(proc['argv'],m.arguments('phase',refs['admission.json'],phase),'Exact original phase command')
        m.same(proc['cgroup'],'0::'+m.p.GROUP_PREFIX+spec['units'][phase]+'.service','Actual phase group')
        m.p.require(re.fullmatch('[0-9a-f]{32}',owner['invocation_id']),'Actual phase invocation')
        ended=records[phase+'-exit.json'];m.same(ended['owner'],owner,'Exact retained exit owner');m.same(ended['observed_by_outer'],outer,'Exit recorded by original coordinator')
        m.p.require(ended['successful']is True and m.p.terminal(ended['properties']),'Retained phase actually terminal')
        props=ended['properties'];m.same((props['Result'],props['ExecMainCode'],props['ExecMainStatus']),('success','1','0'),'Actual successful phase result')
        m.same((props['InvocationID'],props['ExecMainPID']),(owner['invocation_id'],str(owner['pid'])),'Retained original invocation and PID')
        m.service_identity(spec,refs['admission.json'],a['attempt_id'],phase,props,owner)
        result=records[phase+'-result.json'];m.p.require(result['complete']is True,'Actual phase result complete');m.same(result['process'],proc,'Actual phase result owner')
        launch=records[phase+'-launch-intent.json'];m.same(launch,dict(unit=spec['units'][phase],command=m.service_command(spec,refs['admission.json'],a['attempt_id'],phase),admission=refs['admission.json'],outer_owner=outer,attempt_id=a['attempt_id']),'Exact actual phase launch intent')
    cpu=records['cpu-result.json'];m.p.require(type(cpu['actual_rank_exits'])is list and len(cpu['actual_rank_exits'])==2 and all(type(x)is int and x==0 for x in cpu['actual_rank_exits']),'Two actual successful rank exits')
    for rank in (0,1):
        owner=records['rank'+str(rank)+'-owner.json'];proc=process(owner);m.same(owner,proc,'Rank exact process fields')
        m.same(proc['argv'],m.arguments('rank',refs['admission.json'],'cpu'),'Actual rank command')
        m.same((proc['ppid'],proc['cgroup']),(phase_owners['cpu']['pid'],phase_owners['cpu']['cgroup']),'Rank is original CPU coordinator child')
        result=records['cpu-rank'+str(rank)+'-result.json'];m.p.require(result['complete']is True,'Complete actual rank capture');m.same(result['process'],proc,'Exact rank result owner');rank_owners.append(owner)
    m.same(cpu['ranks'],rank_owners,'Both original rank identities retained')
    owners=[outer,*phase_owners.values(),*rank_owners];m.p.require(len({(o['pid'],o['start_ticks'])for o in owners})==6,'Six distinct original processes')
    captures={role:refs[role+'/capture.json']for role in ROLES}
    for role in ROLES:m.same(records[role+'-result.json']['capture'],captures[role],'Exact actual role capture')
    m.same(cpu['captures'],[captures[role]for role in ROLES[1:]],'Both actual rank captures')
    report=records['attention-review.json'];m.same(report['captures'],captures,'All captures bound by original full review');post.compare_review(report,report)
    replay=records['replay-result.json'];m.same(replay['report'],refs['attention-review.json'],'Actual replay report');m.p.require(type(replay['scientific_passed'])is bool and replay['scientific_passed']is report['passed'],'Retain actual scientific result, including false')
    result=records['result.json'];m.p.require(result['complete']is True and result['self_exit_claimed']is False and result['full_model_qualified']is False and result['new_speedup_claim']is False,'Completed bounded diagnostic without self-exit claim')
    m.same(result['owner'],outer,'Final original coordinator');m.same(result['admission'],refs['admission.json'],'Final actual admission');m.same(result['report'],refs['attention-review.json'],'Final actual report')
    return dict(outer_owner=outer,phase_owners=phase_owners,rank_owners=rank_owners,all_own_input_controls_exact=report['passed'],lock_identity=a['lock_identity'],captures=captures)


def names():
    return ['admission.json','result.json','attention-review.json']+[phase+'-'+kind+'.json'for phase in m.PHASES for kind in ('owner','exit','result','launch-intent')]+['rank'+str(rank)+'-owner.json'for rank in (0,1)]+['cpu-rank'+str(rank)+'-result.json'for rank in (0,1)]+[role+'/capture.json'for role in ROLES]


def loaded_sources():
    return {str(Path(module.__file__).resolve()):m.p.sha(module.__file__)for module in tuple(sys.modules.values())if getattr(module,'__file__',None)and Path(module.__file__).is_relative_to(R)}


def observe(exit_ref,approval_ref):
    exited=m.p.read_ref(exit_ref);live=m.p.read_ref(exited['live']);external_exit(exited,live)
    approval=m.p.read_ref(approval_ref)
    m.p.require(approval['schema']=='minicpm5_attention_terminal_source_review_v1'and approval['reviewer']=='/root'and approval['passed']is approval['source_only']is approval['independent_review']is True and approval['blockers']==[],'Separate exact terminal-observer source review')
    m.same(approval['actual_exit'],exit_ref,'Approval binds genuine actual external exit')
    pins=approval['source_sha256'];m.p.require(pins.get(str(SELF))==m.p.sha(SELF),'Exact reviewed terminal observer')
    for path,wanted in pins.items():m.same(m.p.sha(path),wanted,'Unchanged reviewed terminal source')
    for path,wanted in loaded_sources().items():m.same(pins.get(path),wanted,'Every loaded local terminal helper reviewed')
    spec=m.p.read_ref(exited['spec']);review=m.p.read_ref(exited['review']);m.same(spec['schema'],'minicpm5_frame1_attention_spec_v1','Actual attention execution specification')
    m.same(review['spec'],exited['spec'],'Original source review binding');m.p.require(review['reviewer']=='/root'and review['passed']is review['source_only']is review['independent_review']is True and review['blockers']==[],'Original exact model source review')
    m.same(spec['runner'],m.SELF,'Original exact runner');m.same(spec['canonical_sha256'],m.CANONICAL,'Original canonical hashes')
    root=Path(spec['output_root']);m.p.require(root.parent==R and root.resolve()==root and not root.is_symlink()and re.fullmatch('minicpm5-frame1-attention-r[0-9]+-20261108-v[0-9]+',root.name),'Exact actual output namespace')
    target=Q/(root.name+'-root-terminal.json');m.p.require(not target.exists(),'Never rewrite terminal evidence')
    base=m.p.read_ref(m.BASE_PLAN)
    for group in (base['source_sha256'],spec['new_sources']):
        for path,wanted in group.items():m.same(m.p.sha(path),wanted,'Original admitted model/helper source retained')
    m.p.user_bus();fd,observer,lock=m.old.post.external.acquire_lock()
    try:
        m.p.clear_window();m.p.active_heavy();m.p.gone(exited['outer_owner']);m.p.lock_proof(observer,lock)
        m.p.require(not(root/'failure.json').exists(),'Original runner did not report failure')
        refs={name:m.old.reference(root/name)for name in names()};records={name:m.p.read_ref(reference)for name,reference in refs.items()}
        bound=terminal_metadata(root,spec,exited,records,refs);m.same(lock,bound['lock_identity'],'Original serial lock identity')
        reports={role:records[role+'/capture.json']for role in ROLES};metadata=meta.validate(root=root,reports=reports,plan=base)
        m.same(metadata,records['attention-review.json']['metadata'],'Full metadata independently rederived')
        observations=[]
        for _ in range(2):
            m.p.lock_proof(observer,lock);m.p.clear_window();m.p.active_heavy()
            for owner in [bound['outer_owner'],*bound['phase_owners'].values(),*bound['rank_owners']]:m.p.gone(owner)
            sample=[]
            for phase in m.PHASES:
                owner=bound['phase_owners'][phase];value=m.p.properties(spec['units'][phase]);m.service_identity(spec,refs['admission.json'],records['admission.json']['attempt_id'],phase,value,owner)
                m.p.require(m.p.terminal(value)and value['Result']=='success'and value['ExecMainStatus']=='0','Actual phase still successfully terminal')
                sample.append(dict(phase=phase,properties=value,group=m.p.empty_group(m.p.GROUP_PREFIX+spec['units'][phase]+'.service')))
            observations.append(sample)
        m.same({name:m.p.sha(R/name)for name in m.CANONICAL},m.CANONICAL,'Canonical files remain unchanged')
        for name,reference in refs.items():m.same(m.old.reference(root/name),reference,'Actual metadata unchanged during terminal observation')
        terminal=dict(schema='minicpm5_frame1_attention_root_terminal_v1',reviewer='/root',observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),spec=exited['spec'],review=exited['review'],live=exited['live'],actual_external_exit=exit_ref,terminal_source_review=approval_ref,output_root=str(root),actual_exit=exited['actual_exit'],actual_tool_exit_observed=True,exact_outer_owner_gone=True,terminal_observer=observer,physical_drain_samples=observations,all_three_groups_physically_drained=True,all_three_phases_succeeded=True,references=refs,captured_boundary_files=21,norm_weight_files=6,internal_attention_files=48,computed_control_files=21,raw_bytes=metadata['raw_bytes'],output_usage=m.usage(root,spec['root_floor_bytes']),independent_saved_byte_review_complete=False,unique_final_KL_cause_proven=False,full_model_qualified=False,new_speedup_claim=False,**bound)
        post.validate_terminal(terminal);reference=m.worker.once(target,terminal)
        print(json.dumps(dict(terminal=reference,observer=observer,observer_self_exit_claimed=False,independent_saved_byte_review_complete=False,new_speedup_claim=False)),flush=True)
    finally:m.old.post.external.release_lock(fd)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('actual-exit','actual-exit-sha256','review','review-sha256'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();m.p.require(sys.dont_write_bytecode and __debug__,'Bytecode disabled/assertions enabled')
    observe(dict(path=args.actual_exit,sha256=args.actual_exit_sha256),dict(path=args.review,sha256=args.review_sha256))
