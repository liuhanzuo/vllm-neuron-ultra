"""Pure, finite integration for the already accepted raw retirement transaction."""
import copy
import re
from pathlib import PurePosixPath as Path
import minicpm5_retained_replay_contract_v2 as base

ROOT,Q,LOCK,PLAN=base.ROOT,base.Q,base.LOCK,base.PLAN
require,reference,metadata_reference,canonical,sha=base.require,base.reference,base.metadata_reference,base.canonical,base.sha
SOURCES=tuple(str(Q/name)for name in ('minicpm5_retained_retirement_contract_v3.py',
    'minicpm5_retained_retirement_caller_v3.py','minicpm5_retained_retirement_holders_v3.py','minicpm5_retained_retirement_scope_v2.py',
    'minicpm5_retained_raw_transaction_v2.py','minicpm5_retained_raw_source_io_v1.py'))
REMOVER=str(Q/'minicpm5_retained_raw_transaction_v2.py')
LIMITS=dict(seconds=21600,ack_seconds=180,root_floor_bytes=28<<30,nvme_floor_bytes=5<<30,
    control_bytes=128<<20,journal_bytes=128<<20,control_entries=128,journal_entries=4706,
    file_bytes=32<<20,source_bytes=32<<20,memory_address_bytes=6<<30)
REFS=('history_spec','history_review','idle','replay_outer_exit','replay_observer_exit','replay_completed',
    'verification','replay_equality','writer_exclusion')


def paths(run):
    require(type(run)is str and re.fullmatch('[a-z0-9][a-z0-9-]{0,47}',run),'Finite future retirement run')
    root=ROOT/('minicpm5-retained-retirement-control-'+run)
    return dict(control=str(root),journal=str(ROOT/('minicpm5-retained-raw-retirement-'+run)),
        running=str(Q/('continuation-handoff-mini-retirement-'+run+'-running.json')),
        idle=str(Q/('continuation-handoff-mini-retirement-'+run+'-complete.json')),
        actual_exit=str(root/'actual-exit.json'),catalog=str(Q/('minicpm5-retirement-'+run+'-serial-catalog.json')))


def render_spec(run,references,history_spec,original_plan,new_sources,execution_sources):
    """All arguments are authenticated values supplied by the future root."""
    require(set(references)==set(REFS),'Complete explicit future references')
    for key,ref in references.items():
        (metadata_reference if key in ('verification','replay_equality')else reference)(ref)
    pins=history_spec['source_sha256']
    require(len(original_plan['source_sha256'])==1869,'Original complete replay source map')
    require(set(new_sources)==set(SOURCES)and all(pins.get(p)==h for p,h in new_sources.items()),
        'Every retirement source pre-enrolled in the full History overlay')
    require(set(SOURCES)<=set(execution_sources),'Complete explicitly reviewed execution import closure')
    text=dict(original_plan['source_sha256'])
    for p,h in execution_sources.items():
        require(p.endswith('.py')and Path(p).is_relative_to(ROOT)and pins.get(p)==h,'Every executing Python source is globally bound')
        require(p not in text or text[p]==h,'No original source conflict');text[p]=h
    require(all(pins.get(p)==h for p,h in text.items()),'Full text source subset exactly bound by global map')
    spec=dict(schema='minicpm5_retained_retirement_spec_v3',run_name=run,paths=paths(run),references=copy.deepcopy(references),
        storage_plan=copy.deepcopy(PLAN),source_sha256=copy.deepcopy(pins),text_source_sha256=text,
        execution_source_sha256=copy.deepcopy(execution_sources),
        lock_identity=copy.deepcopy(LOCK),limits=copy.deepcopy(LIMITS),automatic_successor=False)
    validate(spec);return spec


def validate(spec):
    require(type(spec)is dict and set(spec)=={'schema','run_name','paths','references','storage_plan','source_sha256',
        'text_source_sha256','execution_source_sha256','lock_identity','limits','automatic_successor'}and spec['schema']=='minicpm5_retained_retirement_spec_v3',
        'Exact finite retirement binding')
    require(spec['paths']==paths(spec['run_name'])and spec['storage_plan']==PLAN and spec['lock_identity']==LOCK
        and spec['limits']==LIMITS and spec['automatic_successor']is False,'Original lock, exact paths and unchanged finite limits')
    require(set(spec['references'])==set(REFS),'Complete future retirement prerequisite refs')
    for key,ref in spec['references'].items():
        (metadata_reference if key in ('verification','replay_equality')else reference)(ref)
    for p,h in spec['source_sha256'].items():reference(dict(path=p,sha256=h))
    require(set(SOURCES)<=set(spec['text_source_sha256'])and all(spec['source_sha256'].get(p)==h
        for p,h in spec['text_source_sha256'].items()),'Complete removal sources in unchanged full global map')
    require(set(SOURCES)<=set(spec['execution_source_sha256'])and all(p.endswith('.py')and Path(p).is_relative_to(ROOT)
        and spec['text_source_sha256'].get(p)==h for p,h in spec['execution_source_sha256'].items()),
        'Every explicitly reviewed executing local source retained in text submap')
    return spec


def running(previous,previous_ref,spec_ref,spec,owner,tool,permit_ref,updated_at):
    require(previous['running']is False and previous['active_serial']is None
        and previous['live_tool_sessions']==previous['active_heavy_units']==[]and previous['serial_execution']is True,
        'Exact complete idle predecessor before actual root ACK')
    require(set(tool)=={'kind','session_id'}and tool['kind']=='tool_session'and type(tool['session_id'])is int
        and tool['session_id']>0,'Actual external root tool session')
    value=copy.deepcopy(previous);current=dict(spec=spec_ref,owner=copy.deepcopy(owner),tool=copy.deepcopy(tool),
        permit=permit_ref,lock_identity=LOCK,automatic_successor=False)
    value.update(schema='minicpm5_retained_retirement_running_v1',running=True,active_serial=copy.deepcopy(current),
        live_tool_sessions=[tool['session_id']],active_heavy_units=[],previous=previous_ref,updated_at=updated_at,
        publication_only_since_bound_terminal=False,minicpm5_retained_retirement_current=current)
    return value


def compatibility(spec,history_spec,resolved,original_plan,plan):
    """Check every full current map; never replace global validation by text pins."""
    require(history_spec['source_sha256']==spec['source_sha256']==resolved['plan']['source_sha256'],
        'Full resolved current source map exactly equals reviewed global map')
    require(all(spec['text_source_sha256'].get(p)==h for p,h in original_plan['source_sha256'].items()),
        'Entire original 1869 replay source map retained')
    require(spec['text_source_sha256']==dict(original_plan['source_sha256'],**spec['execution_source_sha256']),
        'Exactly original source map plus complete executing closure; no mixed binary map substitution')
    maps=[resolved['plan']['inputs'],history_spec['input_extension'],resolved['plan']['source_sha256'],spec['source_sha256']]
    candidates={r['original']['realpath']for r in plan['entries']};keepers={r['keeper']['realpath']for r in plan['entries']}
    intersections=[len(candidates&set(m))for m in maps]
    require(not any(intersections),'Removal candidate intersects a complete current input/source map')
    require(not(candidates&keepers)and len(candidates)==len(keepers)==2352,'All finite distinct candidate and keeper paths')
    return dict(schema='minicpm5_retained_raw_full_history_compatibility_v1',plan_sha256=sha(canonical(plan)),passed=True,
        complete_current_input_and_source_maps_checked=True,all_predecessor_owner_service_cgroup_scope_preserved=True,
        candidate_current_input_intersection=0,map_sizes=[len(m)for m in maps],candidate_intersections=intersections,
        keeper_intersections=[len(keepers&set(m))for m in maps],complete_global_sources=len(spec['source_sha256']),
        global_source_sha256=sha(canonical(spec['source_sha256'])),current_serial_catalog=history_spec['current_serial_catalog'],
        history_overlay=history_spec['history_overlay'],original_plan=base.ORIGINAL_PLAN,retirement_spec_sources=list(SOURCES))


def validate_exclusion(value,spec_ref,spec):
    require(value['schema']=='minicpm5_retained_retirement_writer_exclusion_v1'and value['reviewer']=='/root'
        and value['spec_path']==spec_ref['path']and value['run_name']==spec['run_name']
        and value['storage_plan']==PLAN and value['original_lock_identity']==LOCK
        and value['other_task_agents_excluded']is value['external_namespace_writers_excluded']is True
        and value['retained_for_entire_transaction_and_finalization']is True,
        'Explicit root exclusion of every task agent and external namespace writer; never inferred from idle')


def validate_scope_drain(value,owner=None):
    require(type(value)is dict and value.get('schema')=='minicpm5_retained_retirement_helper_scope_drain_v1'
        and value.get('physically_drained')is True and value.get('pending_scope')is False
        and value.get('original_lock_identity')==LOCK and value.get('unrelated_processes_signalled')is False
        and type(value.get('completed_scopes'))is int and value['completed_scopes']>=0
        and type(value.get('scope_chain_sha256'))is str and re.fullmatch('[0-9a-f]{64}',value['scope_chain_sha256'])
        and (owner is None or value.get('owner')==owner),'Complete retained helper-scope drainage evidence')
    if value['completed_scopes']==0:
        require(value.get('last_scope')is None and value['scope_chain_sha256']=='0'*64,'Empty scope history is explicit')
    else:require(value.get('last_scope',{}).get('physically_drained')is True,'Last registered scope independently drained')
    return value


def completed_outputs(spec,history_spec,extension,oldcat,running_value,running_ref,terminal_ref,terminal,actual_tool,final_proof,updated_at):
    from minicpm5_retained_replay_observer_v2 import tool
    tool(actual_tool);current=running_value['minicpm5_retained_retirement_current']
    require(current['tool']=={k:actual_tool[k]for k in ('kind','session_id')}and current['owner']==terminal['owner']
        and terminal['self_exit_claimed']is False and terminal['actual_exit_pending']is True,'Exact actual remover owner and tool exit')
    require(actual_tool['exit_code']!=0 or terminal['failed']is False,'Actual exit zero cannot hide recorded retirement failure')
    require(final_proof['owners_gone']is final_proof['physically_drained']is True and final_proof['complete_history'],
        'Fresh complete final History and actual owner drain')
    validate_scope_drain(terminal['helper_scope_drain'],terminal['owner'])
    validate_scope_drain(final_proof['helper_scope_drain'])
    require(oldcat['current_handoff']==spec['references']['idle'],'Current pre-retirement catalog remains exact')
    pins=spec['source_sha256'];seen={}
    for row in oldcat['runs']:
        for p,h in row['source_sha256'].items():
            require(pins.get(p)==h and(p not in seen or seen[p]==h),'All complete prior source pins preserved');seen[p]=h
    require(len(oldcat['runs'])<256,'Original complete catalog row limit')
    require(all(pins.get(p)==h for p,h in extension.items())
        and all(extension.get(p)==pins[p] for p in SOURCES if p not in seen),
        'Authenticated overlay extension binds new sources; all prior pins remain in the complete global map')
    delta={p:h for p,h in extension.items()if p not in seen}
    receipt=dict(schema='minicpm5_retained_retirement_actual_exit_v1',reviewer='/root',actual_tool=copy.deepcopy(actual_tool),
        actual_tool_exit_observed=True,owners=[terminal['owner']],units=[],owners_gone=True,physically_drained=True,
        lock_identity=LOCK,outer_terminal=terminal_ref,final_proof=copy.deepcopy(final_proof),
        passed=actual_tool['exit_code']==0 and terminal['failed']is False,transaction_result=terminal['transaction_result'],
        original_failure_preserved=actual_tool['exit_code']!=0 or terminal['failed'],full_model_qualified=False,new_speedup_claim=False)
    receipt_ref=base.publication_ref(spec['paths']['actual_exit'],receipt)
    idle=copy.deepcopy(running_value);idle.pop('minicpm5_retained_retirement_current',None)
    idle.update(schema='minicpm5_retained_retirement_complete_v1',updated_at=updated_at,running=False,active_serial=None,
        previous=running_ref,live_tool_sessions=[],active_heavy_units=[],publication_only_since_bound_terminal=False,
        complete_current_serial_catalog=history_spec['current_serial_catalog'],complete_current_serial_catalog_next=spec['paths']['catalog'],
        minicpm5_retained_retirement_completed=dict(actual_exit=receipt_ref,passed=receipt['passed'],transaction_result=terminal['transaction_result'],
            full_model_qualified=False,new_speedup_claim=False,automatic_successor=False))
    idle_ref=base.publication_ref(spec['paths']['idle'],idle)
    newcat=copy.deepcopy(oldcat);newcat['current_handoff']=idle_ref
    newcat['runs'].append(dict(terminal=receipt_ref,tool=copy.deepcopy(actual_tool),owners=[terminal['owner']],units=[],source_sha256=delta))
    catref=base.publication_ref(spec['paths']['catalog'],newcat)
    return [(receipt_ref,receipt),(catref,newcat),(idle_ref,idle)]
