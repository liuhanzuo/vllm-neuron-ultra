"""Explicit future root admission, actual-session ACK and catalog finalization.

Inert on import. All current references/tool observations/output paths are
caller arguments; no current catalog, run, handoff or session is selected here.
The original geometry root lifecycle is retained for one diagnostic identity.
"""
import argparse
import copy
import datetime
import json
import os
from pathlib import Path
import re
import sys
import time

Q=Path(__file__).resolve().parent
R=Q.parent
SELF=Path(__file__).resolve()
AUDITOR=Q/'qwen35_kv256_s16_pair_audit_v1.py'
HELPERS=(SELF,AUDITOR,Q/'prepare_qwen35_kv256_s16_pair_binding_v1.py',
    Q/'qwen35_geometry_audit_physical_v5.py',Path('/home/ubuntu/qwen35_package_full_r5_post_exit_20261112_v1.py'))


def require(ok,message):
    if not ok:raise ValueError(message)


class Latest:
    @staticmethod
    def latest_handoff():
        # The original collector transitively imports lifecycle ctypes code;
        # defer it to an explicitly admitted actual observation, never api().
        from research import minicpm5_layer1_operator_sdk_runtime_completion_v4 as latest
        return latest.latest_handoff()


def api():
    for path in(Q,R,Path('/home/ubuntu')):
        if str(path)not in sys.path:sys.path.insert(0,str(path))
    from qwen35_kv256_s16_pair_serial_v1 import run as stage
    from qwen35_sampler_runtime_io_v12 import read,ref,once,RetainedLock,process_api,descriptor
    return stage,read,ref,once,RetainedLock,process_api,descriptor,Latest


def exact_request(request,schema,fields):
    require(type(request)is dict and set(request)=={'schema',*fields}and request['schema']==schema,
        'Exact explicit future root-helper request')


def fresh_output(path,kind):
    p=Path(path)
    expressions=dict(binding=r'qwen35-kv256-s16-pair-[a-z0-9-]+-root-binding\.json',
        live=r'qwen35-kv256-s16-pair-[a-z0-9-]+-root-live\.json',
        idle=r'continuation-handoff-[a-z0-9-]+\.json',
        catalog=r'minicpm5-operator-post-r2-serial-workloads-root-catalog-[a-z0-9-]+\.json',
        receipt=r'qwen35-kv256-s16-pair-[a-z0-9-]+-root-finalization\.json')
    require(p.parent==Q and str(p)==path and p.resolve()==p and not p.exists()and not p.is_symlink()
        and re.fullmatch(expressions[kind],p.name),'Fresh explicit root output path')
    return p


def actual_tool(value,terminal=True):
    keys={'kind','session_id'}|({'terminal_chunk','exit_code'}if terminal else set())
    require(type(value)is dict and set(value)==keys and value['kind']=='tool_session'
        and type(value['session_id'])is int and value['session_id']>0,'Actual external tool identity required')
    if terminal:
        require(type(value['exit_code'])is int and 0<=value['exit_code']<=255
            and type(value['terminal_chunk'])is str and re.fullmatch('[a-f0-9]{3,64}',value['terminal_chunk']),
            'Actual external tool terminal required')
    return copy.deepcopy(value)


def pinned_helpers(spec,ref):
    refs=[ref(path)for path in HELPERS]
    require(all(spec['source_sha256'].get(row['path'])==row['sha256']for row in refs),
        'Admission, ACK, independent auditor and finalizer enrolled before preparation; no source retrofit')
    return refs


def accept(request):
    """Concrete source review plus original full History precheck; no launch."""
    exact_request(request,'qwen35_kv256_s16_pair_accept_request_v1',('candidate','binding_output'))
    import resource
    import signal
    resource.setrlimit(resource.RLIMIT_AS,(2*2**30,2*2**30))
    resource.setrlimit(resource.RLIMIT_CPU,(600,605));signal.alarm(600)
    stage,read,ref,once,RetainedLock,process_api,_,latest=api()
    from qwen35_package_preflight_history_v3 import History
    from prepare_qwen35_sampler_v9_candidate_v3 import encode,catalog_shape
    candidate_ref=stage.reference(request['candidate']);candidate=read(candidate_ref)
    require(candidate['schema']=='qwen35_kv256_s16_pair_binding_candidate_v1'
        and candidate['passed']is False and candidate['launch_authorized']is False,'Unadmitted exact candidate')
    for key in('source_review','preparer_source_review'):
        review=read(candidate[key]);require(review['reviewer']=='/root'and review['source_accepted']is True
            and review['blockers']==[],'Independent actual source review required before acceptance')
    spec_ref=candidate['spec_ref'];spec=read(spec_ref,64*2**20);helpers=pinned_helpers(spec,ref)
    catalog=read(spec['current_serial_catalog']);catalog_shape(catalog);tip=latest.latest_handoff()
    require(tip['path']==catalog['current_handoff']['path']and read(catalog['current_handoff'])['running']is False,
        'Actual then-current saved idle catalog/handoff')
    require(candidate['catalog']==spec['current_serial_catalog']and candidate['idle_handoff']==catalog['current_handoff']
        and candidate['catalog_rows']==len(catalog['runs'])and candidate['source_files']==len(spec['source_sha256']),
        'Exact complete candidate lineage and source count')
    require(catalog['runs'][-1]['terminal']==candidate['prerequisite_exit']
        and catalog['runs'][-1]['tool']==candidate['prerequisite_tool'],'Exact final prerequisite including failures')
    draft=read(candidate['review_draft_ref'],64*2**20)
    require(draft['passed']is False and draft['launch_authorized']is False and draft['spec']==spec_ref
        and draft['source_sha256']==spec['source_sha256'],'Exact unadmitted full-source draft')
    review=dict(draft,passed=True,blockers=[],launch_authorized=True)
    stage.validate(spec,review,spec_ref);stage.verify_delivery(spec)
    output=fresh_output(request['binding_output'],'binding')
    target=Path(candidate['paths']['review']);require(not target.exists()and not target.is_symlink(),'Fresh concrete review')
    lock=RetainedLock();lock.acquire();started=time.monotonic();last=[-1.0]
    try:
        def guard():
            lock.check();now=time.monotonic();require(now-started<600,'Original full-source precheck deadline')
            if now-last[0]>=.5:
                require(latest.latest_handoff()==tip,'Current handoff unchanged through admission');last[0]=now
            require(not any(n.split('.')[0]in('torch','numpy','neuronxcc','torch_neuronx','nki')for n in sys.modules),
                'No native imports during original root precheck')
        guard();process_api().active_heavy();stage.resources(spec,reserve=True)
        with target.open('xb')as stream:stream.write(encode(review));stream.flush();os.fsync(stream.fileno())
        review_ref=ref(target,64*2**20);require(stage.load_spec(spec_ref,review_ref)==spec,'Exact actual written review')
        history=History(spec,spec_ref,review_ref,stage.names(spec),guard).resolve()
        history.unchanged();freshness=history.freshness()
        require(freshness['latest_handoff']==catalog['current_handoff'],'Full History current handoff')
        children=len(history.supplement['scope']['children'])
        require(children>=874 and children>=candidate['preservation']['seed_lfm_children'],'All original LFM children retained')
        stage.loaded_sources(spec['source_sha256']);resources=stage.resources(spec,reserve=True);guard()
        require(pinned_helpers(spec,ref)==helpers,'All pre-enrolled root helpers unchanged')
        value=dict(schema='qwen35_kv256_s16_pair_root_concrete_binding_v1',reviewer='/root',passed=True,
            observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),candidate=candidate_ref,
            source_review=candidate['source_review'],preparer_source_review=candidate['preparer_source_review'],
            spec=spec_ref,review=review_ref,history_overlay=spec['history_overlay'],catalog=spec['current_serial_catalog'],
            idle_handoff=catalog['current_handoff'],catalog_rows=len(catalog['runs']),source_files=len(spec['source_sha256']),
            full_source_history_resolution_passed=True,lfm_children=children,original_failed_statuses_preserved=True,
            original_lock_reacquired=True,resources=resources,physical_pair_performed=False,native_executed=False,
            runtime_launched=False,full_geometry_qualified=False,full_model_qualified=False,new_speedup_claim=False)
        return dict(binding=once(output,value),spec=spec_ref,review=review_ref)
    finally:lock.release(True)


def ack(request):
    """Publish running only after the root supplies and observes the real session."""
    exact_request(request,'qwen35_kv256_s16_pair_ack_request_v1',('binding','actual_tool','live_output'))
    stage,read,ref,once,_,process_api,descriptor,latest=api()
    from qwen35_sampler_runtime_history_v12 import expected_running
    binding_ref=stage.reference(request['binding']);binding=read(binding_ref)
    require(binding['schema']=='qwen35_kv256_s16_pair_root_concrete_binding_v1'
        and binding['reviewer']=='/root'and binding['passed']is True,'Exact root concrete binding')
    spec=stage.load_spec(binding['spec'],binding['review']);pinned_helpers(spec,ref)
    output=fresh_output(request['live_output'],'live');root=Path(spec['output_root'])
    admission_ref=ref(root/'admission.json');admission=read(admission_ref)
    owner_ref=ref(root/'supervisor-owner.json');owner=read(owner_ref)
    require(admission['spec']==binding['spec']and admission['review']==binding['review']
        and admission['schema']=='qwen35_kv256_s16_pair_serial_admission_v1','Exact admitted diagnostic')
    require(process_api().proc(admission['outer_owner']['pid'])==admission['outer_owner']
        and process_api().proc(owner['process']['pid'])==owner['process'],'Actual live outer/supervisor owners')
    descriptor(admission['outer_owner'],admission['retained_fd']);unit=stage.names(spec)['coordinator']
    service=stage.Service(unit,stage.supervisor_argv(admission_ref),stage.environment(root),stage.PROPERTIES,
        root/'service.log',owner=owner);service.live();process_api().active_heavy((unit,))
    previous_ref=admission['freshness']['latest_handoff'];previous=read(previous_ref);tip=latest.latest_handoff()
    require(tip['path']==previous_ref['path']and ref(previous_ref['path'])==previous_ref
        and previous['serial_execution']is True and previous['running']is False
        and previous['active_serial']is None and previous['active_heavy_units']==previous['live_tool_sessions']==[],
        'Original exact actual idle handoff, complete digest and no live work')
    tool=actual_tool(request['actual_tool'],terminal=False);now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    running=expected_running(previous,now,tool,admission_ref,admission,owner_ref,owner)
    require(latest.latest_handoff()==tip and ref(previous_ref['path'])==previous_ref,
        'Full idle path and metadata identity unchanged immediately before running publication')
    descriptor(admission['outer_owner'],admission['retained_fd']);service.live()
    require(process_api().proc(admission['outer_owner']['pid'])==admission['outer_owner']
        and process_api().proc(owner['process']['pid'])==owner['process'],'Same actual owners before ACK publication')
    running_ref=once(spec['running_handoff_path'],running)
    live_ref=once(output,dict(schema='qwen35_kv256_s16_pair_root_live_v1',reviewer='/root',observed_at=now,
        actual_tool=tool,root_binding=binding_ref,outer_owner=admission['outer_owner'],coordinator_owner=owner_ref,
        running_handoff=running_ref,ack_source=ref(SELF),device_execution_not_yet_claimed=True,
        full_geometry_qualified=False,full_model_qualified=False,new_speedup_claim=False))
    return dict(running=running_ref,root_live=live_ref,actual_tool=tool)


def publication_values(old,running,spec,ended,exit_ref,completion_ref,actual,audit_tool,outputs,delta,composition,observations,now):
    """Pure exact append builder, also usable with labelled synthetic test objects."""
    from qwen35_package_full_r5_post_exit_20261112_v1 import predicted
    from prepare_qwen35_sampler_r6_candidate_v1 import catalog_shape
    done=ended['diagnostic_completed'];require(type(done)is bool,'Explicit diagnostic completion')
    require(ended['geometry_qualified']is False and ended['full_model_qualified']is False
        and ended['new_speedup_claim']is False,'Diagnostic cannot qualify geometry or model')
    idle=copy.deepcopy(running)
    idle.update(updated_at=now,previous=ended['running_handoff'],previous_handoff=ended['running_handoff'],
        running=False,active_serial=None,active_heavy_units=[],live_tool_sessions=[],goal_status='active',
        this_turn_classification='progress',complete_current_serial_catalog=spec['current_serial_catalog'],
        complete_current_serial_catalog_next=outputs['catalog'],qwen35_sampler_current_execution_ended=True,
        qwen35_sampler_current_execution_actual_exit=exit_ref,publication_current_status_stale=True,new_speedup_claim=False,
        next_serial_work='Review retained KV256 diagnostic differences; original geometry failure remains unqualified.')
    idle['qwen35_kv256_s16_pair_current']=dict(actual_exit=exit_ref,actual_tool=actual,independent_auditor_tool=audit_tool,
        status='diagnostic_completed'if done else'failed',diagnostic_completed=done,completion=completion_ref,
        completed_calls=ended['completed_calls'],retained_mismatch_calls=ended['retained_mismatch_calls'],
        full_geometry_qualified=False,full_model_qualified=False,new_speedup_claim=False)
    idle_ref=predicted(outputs['idle'],idle);catalog=copy.deepcopy(old)
    catalog.update(current_handoff=idle_ref,runs=old['runs']+[dict(terminal=exit_ref,tool=actual,
        owners=ended['owners'],units=ended['units'],source_sha256=delta)])
    require(catalog['runs'][:-1]==old['runs']and len(catalog['runs'])==len(old['runs'])+1,'Every old row unchanged plus exactly one true delta')
    catalog_shape(catalog);catalog_ref=predicted(outputs['catalog'],catalog)
    receipt=dict(schema='qwen35_kv256_s16_pair_root_finalization_v1',reviewer='/root',observed_at=now,
        actual_exit=exit_ref,actual_outer_tool=actual,actual_independent_auditor_tool=audit_tool,source_composition=composition,
        complete_old_catalog=spec['current_serial_catalog'],catalog=catalog_ref,idle_handoff=idle_ref,
        two_current_drain_observations=observations,all_prior_rows_unchanged=True,prior_rows=len(old['runs']),
        total_rows=len(catalog['runs']),original_8MiB_cap_preserved=True,original_lock_identity=ended['lock_identity'],
        diagnostic_completed=done,completion=completion_ref,full_geometry_qualified=False,
        full_model_qualified=False,new_speedup_claim=False,no_post_run_source_retrofit=True)
    return dict(idle=idle,catalog=catalog,receipt=receipt),dict(idle=idle_ref,catalog=catalog_ref,receipt=predicted(outputs['receipt'],receipt))


def finalize(request):
    """After both actual external exits, repeat current drains and append one row."""
    exact_request(request,'qwen35_kv256_s16_pair_finalize_request_v1',
        ('binding','live','outer_tool','audit_tool','outputs'))
    stage,read,ref,once,RetainedLock,process_api,_,latest=api()
    from prepare_qwen35_sampler_r6_candidate_v1 import catalog_shape
    from qwen35_package_full_r5_post_exit_20261112_v1 import source_delta
    from audit_qwen35_sampler_r10_terminal_root_20261112_v2 import verify_current_terminal,terminal_unit
    from qwen35_kv256_s16_pair_audit_v1 import FreshnessGuard
    binding_ref=stage.reference(request['binding']);binding=read(binding_ref);live=read(stage.reference(request['live']))
    require(binding['reviewer']=='/root'and binding['passed']is True and binding['full_source_history_resolution_passed']is True
        and live['root_binding']==binding_ref,'Exact accepted source/history and actual live binding')
    spec_ref,review_ref=binding['spec'],binding['review'];spec=stage.load_spec(spec_ref,review_ref)
    helpers=pinned_helpers(spec,ref);actual=actual_tool(request['outer_tool']);audited=actual_tool(request['audit_tool'])
    require(audited['exit_code']==0 and audited['session_id']!=actual['session_id'],'Distinct actual successful external auditor terminal')
    require(live['actual_tool']=={k:actual[k]for k in('kind','session_id')},'Actual original tool session')
    exit_ref=ref(Q/('qwen35-kv256-s16-pair-'+spec['run_name']+'-root-actual-exit.json'));ended=read(exit_ref)
    require(ended['schema']=='qwen35_kv256_s16_pair_root_actual_exit_v1'and ended['reviewer']=='/root'
        and ended['actual_tool']==actual and ended['spec']==spec_ref and ended['review']==review_ref
        and ended['owners_gone']is True and ended['physically_drained']is True and ended['original_lock_held_during_audit']is True
        and ended['lock_identity']==stage.LOCK_IDENTITY and ended['auditor_source']==ref(AUDITOR),'Exact actual independent audit')
    require(ended['full_history_pair']['full_physical_passes']==2 and len(ended['two_current_service_observations'])==2
        and ended['two_current_service_observations'][0]==ended['two_current_service_observations'][1],
        'Original full two-pass History and complete current drain audit retained')
    done=ended['diagnostic_completed'];counts=tuple(ended[k]for k in('completed_layouts','completed_calls','compiled_graphs'))
    require(type(done)is bool and ended['reconciliation']['passed']is done and all(type(x)is int and 0<=x<=cap
        for x,cap in zip(counts,(1,6,2))),'Exact finite diagnostic completion counters')
    require((actual['exit_code']==0 and counts==(1,6,2))if done else actual['exit_code']!=0,'Actual completion matches actual outer exit')
    completion_path=Q/('qwen35-kv256-s16-pair-'+spec['run_name']+'-root-completion.json');completion_ref=None
    if done:
        completion_ref=ref(completion_path);proof=read(completion_ref)
        require(proof['schema']=='qwen35_kv256_s16_pair_independent_completion_v1'and proof['reviewer']=='/root'
            and proof['passed']is True and proof['blockers']==[] and proof['actual_exit']==exit_ref
            and proof['reports']==ended['reports']and proof['source_sha256']==ended['source_sha256']
            and proof['full_geometry_qualified']is False and proof['full_model_qualified']is False
            and proof['new_speedup_claim']is False and proof['diagnostic_completed']is True
            and proof['retained_mismatch_calls']==ended['retained_mismatch_calls'],'Diagnostic-only independent completion')
    else:require(not completion_path.exists()and not completion_path.is_symlink(),'No completion receipt after failed actual exit')
    running_ref=ended['running_handoff'];running=read(running_ref)
    require(running_ref==live['running_handoff']and running_ref['path']==spec['running_handoff_path'],'Exact observed running handoff')
    tip=latest.latest_handoff();require(tip['path']==running_ref['path'],'No unknown successor before finalization')
    old=read(spec['current_serial_catalog']);catalog_shape(old)
    require(binding['catalog']==spec['current_serial_catalog']and binding['catalog_rows']==len(old['runs']),
        'Entire exact predecessor catalog retained')
    overlay=read(spec['history_overlay'])
    # Every executed helper is already in spec. No new source may be attached here.
    delta,composition=source_delta(spec['source_sha256'],overlay['source_extension'],old,{})
    require(composition['runtime_sources']==composition['post_audit_sources'],'No post-run source retrofit')
    require(type(request['outputs'])is dict and set(request['outputs'])=={'idle','catalog','receipt'},'All exact final publication paths')
    outputs={key:str(fresh_output(value,key))for key,value in request['outputs'].items()}
    terminal=read(ended['outer_terminal']);lock=RetainedLock();lock.acquire()
    try:
        guard=FreshnessGuard(lock,tip);observations=[];p=process_api()
        for _ in range(2):
            guard(force=True);p.active_heavy();p.gone(ended['actual_observer'])
            for owner in ended['owners']:p.gone(owner)
            observed={}
            for saved in terminal['complete_owned_unit_terminals']:
                unit=terminal_unit(saved);props=stage.properties(unit);verify_current_terminal(saved,props,p)
                observed[unit]=dict(properties=props,recursive_group=p.empty_group(p.GROUP_PREFIX+unit+'.service'))
            require(set(observed)==set(ended['units']),'Every original declared unit observed');observations.append(observed);guard(force=True)
        require(observations[0]==observations[1],'Two stable complete current finalization drains')
        require(pinned_helpers(spec,ref)==helpers and ref(exit_ref['path'])==exit_ref and ref(running_ref['path'])==running_ref,
            'Actual terminal, handoff and enrolled sources unchanged')
        if completion_ref is not None:require(ref(completion_path)==completion_ref,'Completion unchanged')
        stage.loaded_sources(spec['source_sha256']);guard(force=True)
        values,refs=publication_values(old,running,spec,ended,exit_ref,completion_ref,actual,audited,outputs,delta,composition,
            observations,datetime.datetime.now(datetime.timezone.utc).isoformat())
        for key in('idle','catalog','receipt'):
            # Discovery must run immediately before the first handoff change;
            # subsequent atomic publications retain the same original lock.
            if key=='idle':guard(force=True)
            else:lock.check()
            require(once(outputs[key],values[key])==refs[key],'Exact bounded final publication')
        return dict(actual_exit=exit_ref,idle_handoff=refs['idle'],catalog=refs['catalog'],finalization=refs['receipt'])
    finally:lock.release(True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('accept','ack','finalize'),required=True)
    parser.add_argument('--request',required=True);parser.add_argument('--request-sha256',required=True)
    args=parser.parse_args(argv);require(sys.dont_write_bytecode and __debug__,'Original -B and assertions')
    stage,read,*_=api();request=read(stage.reference(dict(path=args.request,sha256=args.request_sha256)))
    print(json.dumps(dict(accept=accept,ack=ack,finalize=finalize)[args.mode](request)))


if __name__=='__main__':main()
