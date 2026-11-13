"""Independent actual-exit/complete-History observer and pure final publisher.

The observer does not claim its own exit and does not publish an idle handoff.
After its actual tool exit, root can render_completed() with the real observed
receipt and use publish_completed() for a fresh lock/History/owner check before
publication. No sessions are inferred or invented.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

Q=Path(__file__).resolve().parent;ROOT=Q.parent
for directory in (Q,ROOT):
    if str(directory)not in sys.path:sys.path.insert(0,str(directory))
import minicpm5_retained_replay_contract_v1 as c
import minicpm5_retained_replay_caller_v1 as caller


def tool(value):
    c.require(type(value)is dict and set(value)=={'kind','session_id','exit_code','terminal_chunk'}
        and value['kind']=='tool_session'and type(value['session_id'])is int and value['session_id']>0
        and type(value['exit_code'])is int and 0<=value['exit_code']<=255
        and type(value['terminal_chunk'])is str and 3<=len(value['terminal_chunk'])<=64,
        'Exact externally observed actual tool terminal required')
    return value


def reconcile(terminal,running,actual_tool):
    tool(actual_tool);execution=running['minicpm5_retained_replay_current_execution']
    c.require(terminal['schema']=='minicpm5_retained_replay_outer_terminal_v1'
        and terminal['self_exit_claimed']is False and terminal['actual_external_exit_pending']is True
        and terminal['full_owned_scope_drained']is True and terminal['admission']==execution['admission']
        and terminal['outer_owner']==execution['outer_owner']
        and execution['tool']=={k:actual_tool[k]for k in ('kind','session_id')},'Actual terminal belongs to this exact running outer')
    passed=actual_tool['exit_code']==0 and terminal['failed']is False
    c.require(actual_tool['exit_code']!=0 or passed,'Exit zero cannot hide recorded failure')
    if passed:
        c.require([r['phase']for r in terminal['phase_states']]==['pair','replay'],'Both actual phases required')
        for row in terminal['phase_states']:
            actual=row['exit'];props=actual['properties']
            c.require(actual['actual_service_exit_observed']is True and actual['owner']==row['owner']
                and props['ExecMainCode']=='1'and props['ExecMainStatus']=='0'and props['Result']=='success',
                'Both actual captured service exits must succeed')
    return dict(actual_tool=copy.deepcopy(actual_tool),passed=passed,original_failure_preserved=not passed,
        actual_outer_exit_observed=True,deletion_authorized=False,full_model_qualified=False)


def drain_phases(spec,admission_ref,admission,terminal):
    """Physical drain only; safe cleanup does not depend on an expired deadline."""
    io,life=caller.api();io.process_api().gone(terminal['outer_owner']);rows=[]
    c.require(len(terminal['phase_states'])<=2 and len({r['phase']for r in terminal['phase_states']})==len(terminal['phase_states']),
        'Finite original phases without duplicates')
    for row in terminal['phase_states']:
        c.require(row['phase']in ('pair','replay')and row['unit']==spec['units'][row['phase']]
            and row['nonce']==admission['attempts'][row['phase']], 'Exact finite original unit attempt')
        props=life.properties(row['unit'])
        if props['LoadState']!='not-found':
            sampled=life.sample(row['unit']);life.contract(sampled,row['unit'],c.phase_argv(row['phase'],admission_ref),None,row['nonce'])
            c.require(sampled['InvocationID']==row['state'].get('invocation_id'),'Original retained invocation')
        if row['owner']is not None:io.process_api().gone(row['owner']['process'])
        rows.append(dict(phase=row['phase'],state=copy.deepcopy(row['state']),samples=life.drained(row['unit'],row['state'])))
    io.process_api().active_heavy()
    return dict(outer_owner_gone=True,all_actual_phase_states=rows,physically_drained=True)


def observe(request_ref):
    io,life=caller.api();request=io.read(request_ref,c.LIMITS['spec_bytes'])
    c.require(set(request)=={'spec','review','admission','outer_terminal','running','actual_outer_tool'},'Exact external observation request')
    spec=caller.refs(request['spec'],request['review']);admission=io.read(request['admission'],c.LIMITS['spec_bytes'])
    terminal=io.read(request['outer_terminal']);running=io.read(request['running']);checked=reconcile(terminal,running,request['actual_outer_tool'])
    c.require(admission['spec']==request['spec']and admission['review']==request['review']
        and terminal['admission']==request['admission'],'Complete original caller binding')
    c.require(request['outer_terminal']['path']==str(Path(spec['output_root'])/'outer-terminal.json')
        and request['running']['path']==spec['running_paths'][running['minicpm5_retained_replay_current_execution']['phase']], 'Exact completed namespaces')
    observed=dict(path=request['running']['path'],identity=io.identity(request['running']['path']))
    c.require(caller.latest_observation()==observed,'Original running handoff remains latest')
    io.process_api().gone(terminal['outer_owner'])
    observer_owner=io.process_api().proc(os.getpid())
    print(json.dumps(dict(event='mini_retained_independent_observer_live',request=request_ref,
        observer_owner=observer_owner,actual_external_tool_session_required=True)),flush=True)
    from minicpm5_retained_replay_history_v1 import History,Authority
    from qwen35_package_preflight_serial_v2 import save_history
    from research.minicpm5_layer1_drain_hold_v1 import retain_until_drained
    root=Path(spec['observer_root']);c.require(not root.exists(),'Fresh independent observation namespace')
    lock=io.RetainedLock();lock.acquire();start=time.monotonic();last=[-1.0]
    def guard(force=False):
        lock.check();c.require(time.monotonic()-start<7200,'Finite independent observer')
        if force or time.monotonic()-last[0]>=.5:
            c.require(caller.latest_observation()==observed,'Unknown newer handoff during independent audit')
            caller.source_unchanged(spec);caller.storage(spec);last[0]=time.monotonic()
        c.require(io.identity(request['running']['path'])==observed['identity'],'Original running bytes unchanged')
    def owned():
        guard(True);value=drain_phases(spec,request['admission'],admission,terminal);guard(True);return value
    proof=None
    try:
        guard(True)
        live_path=Path(spec['output_root'])/'observer-live.json'
        io.wait_until(lambda:True if live_path.exists()else None,guard,c.LIMITS['ack_seconds'],message='Independent live observer capture missing')
        live=io.read(io.ref(live_path))
        c.require(live['owner']==observer_owner and live['request']==request_ref
            and live['independent_actual_live_observation']is True,'Exact root-captured observer identity')
        owned_first=owned()
        history=History(spec,request['spec'],request['review'],spec['units'],guard).resolve()
        guard(True);full_first=history.physical_pair(Authority(history,guard,lambda:[]));guard(True)
        history.unchanged();owned_second=owned();guard(True)
        root.mkdir(mode=0o700)
        history_refs=save_history(root,dict(full_history=full_first,first_owned=owned_first,second_owned=owned_second,
            current_serial_catalog=spec['current_serial_catalog'],history_overlay=spec['history_overlay'],
            metadata_identities=history.meta.identities,actual_outer_terminal=request['outer_terminal']))
        equality=None
        if checked['passed']:
            import minicpm5_retained_raw_compare_v1 as comparator
            verification=caller.bounded_ref(Path(spec['output_root'])/'paired-verification.json')
            result=caller.bounded_ref(Path(spec['replay_root'])/'compare-result.json')
            equality=comparator.compare_original_replay(io.read(c.PLAN),verification,result,root/'full-replay-equality.json',guard)
        guard(True);history.unchanged();owned_final=owned()
        output=dict(schema='minicpm5_retained_replay_independent_exit_observation_v1',spec=request['spec'],review=request['review'],
            admission=request['admission'],request=request_ref,outer_terminal=request['outer_terminal'],running=request['running'],
            running_identity=observed['identity'],independent_live=io.ref(live_path),
            actual_outer_exit=checked,complete_history=history_refs,final_owned=owned_final,
            full_replay_equality=equality,observer_owner=observer_owner,lock_identity=c.LOCK,
            actual_observer_exit_pending=True,self_exit_claimed=False,all_prior_fields_preserved=True,
            full_model_qualified=False,new_speedup_claim=False,deletion_authorized=False,automatic_successor=False)
        io.loaded_local_sources(spec['source_sha256']);guard(True)
        output_ref=io.once(root/'observation.json',output);proof=owned_final
        print(json.dumps(dict(observation=output_ref,observer_actual_exit_pending=True,deletion_authorized=False)),flush=True)
        return output_ref
    finally:
        # The observer launches no service. Retain the original lock until all
        # supplied actual outer/phase owners are freshly known drained again.
        def cleanup_drain():
            lock.check();return drain_phases(spec,request['admission'],admission,terminal)
        proof=retain_until_drained(lambda:lock.check(),cleanup_drain,
            lambda value:print(json.dumps(dict(event='mini_retained_observer_drain_blocked',details=value)),flush=True),
            lambda:time.sleep(1))
        lock.release(proof['physically_drained'])


def capture_live(request_ref,actual_tool,observer_owner,output_path):
    """External root supplies its actual running handle and observed live owner."""
    io,_=caller.api();request=io.read(request_ref,c.LIMITS['spec_bytes'])
    spec=caller.refs(request['spec'],request['review'])
    c.require(type(actual_tool)is dict and set(actual_tool)=={'kind','session_id'}and actual_tool['kind']=='tool_session'
        and type(actual_tool['session_id'])is int and actual_tool['session_id']>0,'Actual root tool handle required')
    c.require(io.process_api().proc(observer_owner['pid'])==observer_owner,'Independent root observes exact live observer')
    argv=observer_owner['argv']
    c.require(argv==['/usr/bin/python3','-B',c.FILES[3],'--request',request_ref['path'],'--request-sha256',request_ref['sha256']],
        'Actual finite observer entry argv')
    c.require(Path(output_path)==Path(spec['output_root'])/'observer-live.json','Exact externally published observer evidence')
    return io.once(Path(output_path),dict(schema='minicpm5_retained_replay_observer_live_v1',request=request_ref,
        owner=copy.deepcopy(observer_owner),actual_tool=copy.deepcopy(actual_tool),
        source=dict(path=c.FILES[3],sha256=spec['source_sha256'][c.FILES[3]]),independent_actual_live_observation=True))


def render_completed(previous,observation_ref,observation,observer_live_ref,observer_live,actual_observer_tool,updated_at,finalization):
    """Pure final handoff; root supplies the real observer exit after it occurs."""
    tool(actual_observer_tool)
    c.require(observer_live['schema']=='minicpm5_retained_replay_observer_live_v1'
        and observer_live['independent_actual_live_observation']is True
        and observer_live['owner']==observation['observer_owner']and observer_live['request']==observation['request']
        and observer_live['actual_tool']=={k:actual_observer_tool[k]for k in ('kind','session_id')},
        'Actual observer exit must match independently captured owner/request/tool session')
    c.require(actual_observer_tool['exit_code']==0 and observation['actual_observer_exit_pending']is True
        and observation['self_exit_claimed']is False and observation['final_owned']['physically_drained']is True,
        'Independent observer must actually exit successfully after full drain')
    c.require(previous['schema']=='minicpm5_retained_replay_running_v1'
        and previous['minicpm5_retained_replay_current_execution']['admission']==observation['admission']
        and previous['minicpm5_retained_replay_current_execution']['tool']==
            {k:observation['actual_outer_exit']['actual_tool'][k]for k in ('kind','session_id')}, 'Exact complete previous running handoff')
    c.require(finalization['schema']=='minicpm5_retained_replay_fresh_final_publication_v1'
        and finalization['running']==observation['running']and finalization['running_identity']==observation['running_identity']
        and finalization['observer_owner']==observation['observer_owner']and finalization['actual_observer_tool']==actual_observer_tool
        and finalization['observer_owner_gone']is True and finalization['final_owned']['physically_drained']is True
        and finalization['lock_identity']==c.LOCK and finalization['complete_history'], 'Fresh physical final publication required')
    completed=copy.deepcopy(previous)
    completed.update(schema='minicpm5_retained_replay_complete_handoff_v1',updated_at=updated_at,
        live_tool_sessions=[],active_heavy_units=[],
        minicpm5_retained_replay_completed=dict(observation=copy.deepcopy(observation_ref),
            actual_outer_exit=copy.deepcopy(observation['actual_outer_exit']),actual_observer_exit=copy.deepcopy(actual_observer_tool),
            independent_observer_live=copy.deepcopy(observer_live_ref),
            final_publication=copy.deepcopy(finalization),
            equality=copy.deepcopy(observation['full_replay_equality']),deletion_authorized=False,
            full_model_qualified=False,new_speedup_claim=False,automatic_successor=False))
    return completed


def publish_completed(request_ref):
    """Explicit external root entry after observing the actual observer exit.

    This function never starts a worker. It verifies the same latest running
    identity and all complete inherited/own owners under the original lock,
    then publishes the pure rendering before releasing that lock.
    """
    io,_=caller.api();request=io.read(request_ref)
    c.require(set(request)=={'observation','observer_live','actual_observer_tool','updated_at'},'Exact final root request')
    observation=io.read(request['observation']);live=io.read(request['observer_live']);tool(request['actual_observer_tool'])
    spec=caller.refs(observation['spec'],observation['review']);admission=io.read(observation['admission'])
    terminal=io.read(observation['outer_terminal']);previous=io.read(observation['running'])
    c.require(reconcile(terminal,previous,observation['actual_outer_exit']['actual_tool'])==observation['actual_outer_exit'],
        'Original actual outer terminal and every phase exit remain bound')
    c.require(request['observation']['path']==str(Path(spec['observer_root'])/'observation.json')
        and request['observer_live']==observation['independent_live'],'Exact accepted observer evidence')
    expected=dict(path=observation['running']['path'],identity=observation['running_identity'])
    io.process_api().gone(observation['observer_owner'])
    root=Path(spec['output_root'])/'final-publication';c.require(not root.exists(),'Fresh final publication namespace')
    from minicpm5_retained_replay_history_v1 import History,Authority
    from qwen35_package_preflight_serial_v2 import save_history
    from research.minicpm5_layer1_drain_hold_v1 import retain_until_drained
    lock=io.RetainedLock();lock.acquire();started=time.monotonic();last=[-1.0]
    def guard(force=False):
        lock.check();c.require(time.monotonic()-started<7200,'Finite final physical publication')
        c.require(io.identity(expected['path'])==expected['identity'],'Same final running metadata identity')
        if force or time.monotonic()-last[0]>=.5:
            c.require(caller.latest_observation()==expected,'Unknown newer handoff before final publication')
            io.process_api().gone(observation['observer_owner']);caller.source_unchanged(spec);caller.storage(spec);last[0]=time.monotonic()
    def drained():
        lock.check();io.process_api().gone(observation['observer_owner'])
        return drain_phases(spec,observation['admission'],admission,terminal)
    try:
        guard(True);drained()
        history=History(spec,observation['spec'],observation['review'],spec['units'],guard).resolve()
        guard(True);pair=history.physical_pair(Authority(history,guard,lambda:[]));guard(True);history.unchanged()
        root.mkdir(mode=0o700);history_refs=save_history(root,pair)
        finalization=dict(schema='minicpm5_retained_replay_fresh_final_publication_v1',running=observation['running'],
            running_identity=observation['running_identity'],observer_owner=observation['observer_owner'],observer_owner_gone=True,
            actual_observer_tool=request['actual_observer_tool'],final_owned=drained(),complete_history=history_refs,lock_identity=c.LOCK)
        value=render_completed(previous,request['observation'],observation,request['observer_live'],live,
            request['actual_observer_tool'],request['updated_at'],finalization)
        history.unchanged();guard(True);drained()
        return io.once(Path(spec['terminal_handoff_path']),value)
    finally:
        proof=retain_until_drained(lambda:lock.check(),drained,
            lambda value:print(json.dumps(dict(event='mini_retained_final_publication_drain_blocked',details=value)),flush=True),
            lambda:time.sleep(1))
        lock.release(proof['physically_drained'])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=('observe','capture','finalize'),default='observe')
    parser.add_argument('--request',required=True);parser.add_argument('--request-sha256',required=True)
    args=parser.parse_args(argv);c.require(sys.dont_write_bytecode and __debug__,'Explicit -B/assertions')
    os.sched_setaffinity(0,{7});os.setpriority(os.PRIO_PROCESS,0,19)
    request_ref=dict(path=args.request,sha256=args.request_sha256)
    if args.mode=='observe':observe(request_ref)
    elif args.mode=='finalize':print(json.dumps(publish_completed(request_ref)),flush=True)
    else:
        io,_=caller.api();request=io.read(request_ref)
        c.require(set(request)=={'request','actual_tool','observer_owner','output_path'},'Exact root capture request')
        print(json.dumps(capture_live(request['request'],request['actual_tool'],request['observer_owner'],request['output_path'])),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
