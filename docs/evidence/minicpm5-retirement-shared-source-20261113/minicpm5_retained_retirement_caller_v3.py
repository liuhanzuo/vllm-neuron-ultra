"""One finite root retirement owner and its separate actual-exit finalizer.

No current refs are selected. The sole deletion call is frozen transaction.retire.
Import/--help are inert. Run, ACK and finalization are explicit external entries.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import resource
import signal
import stat
import sys
import time

Q=Path(__file__).resolve().parent;ROOT=Q.parent
for directory in (Q,ROOT):
    if str(directory)not in sys.path:sys.path.insert(0,str(directory))
import minicpm5_retained_retirement_contract_v3 as c
from minicpm5_retained_retirement_scope_v2 import ForegroundScopes


def api():
    import qwen35_sampler_runtime_io_v12 as io
    return io


class Saved:
    def __init__(self,io):self.io=io;self.identities={}
    def read(self,ref,cap=8<<20):
        c.metadata_reference(ref);before=self.io.identity(ref['path'])
        c.require(ref['path']not in self.identities or self.identities[ref['path']]==before,'No input identity rebound')
        c.require(before['size']<=cap and('bytes'not in ref or ref['bytes']==before['size']),'Exact bounded metadata bytes')
        value=self.io.read({k:ref[k]for k in ('path','sha256')},cap)
        c.require(self.io.identity(ref['path'])==before,'Authenticated metadata changed while reading')
        self.identities[ref['path']]=before;return value
    def unchanged(self):
        for path,ident in self.identities.items():c.require(self.io.identity(path)==ident,'Retirement prerequisite metadata changed')


def source_check(spec):
    io=api()
    for path in (*c.SOURCES,*c.base.FROZEN):
        c.require(c.sha(io.raw(path,1<<20))==spec['source_sha256'][path],'Actual retirement/frozen source changed')


def loaded_text_sources(spec):
    io=api();io.loaded_local_sources(spec['source_sha256'])
    for module in list(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if isinstance(path,str)and Path(path).is_absolute()and Path(path).is_relative_to(ROOT):
            c.require(path.endswith('.py')and path in spec['text_source_sha256'],
                'Every actual local execution source retained in transaction text submap')


def load(spec_ref,review_ref,saved):
    spec=saved.read(spec_ref,64<<20);c.validate(spec);review=saved.read(review_ref,32<<20)
    c.require(review['schema']=='minicpm5_retained_raw_replay_and_removal_review_v1'
        and review['reviewer']=='/root'and review['spec']==spec_ref and review['passed']is True
        and review['blockers']==[]and review['launch_authorized']is review['production_deletion_admitted']is True
        and review['source_sha256']==spec['text_source_sha256']
        and review['verification']==spec['references']['verification']and review['replay_equality']==spec['references']['replay_equality']
        and all(review[k]is True for k in ('full11859_logical_inputs_preserved','full32_replay_and_all_scientific_outputs_equal',
            'all_original_failures_preserved','original_weights_precision_and_remaining_gates_unchanged')),
        'Independent root source/full-replay/removal review and explicit exact caller admission')
    for key,stem in (('remover_source','transaction'),('replay_source','legacy')):
        path=c.REMOVER if key=='remover_source'else str(Q/('minicpm5_retained_raw_'+stem+'_v1.py'))
        expected=spec['source_sha256'][path]if key=='remover_source'else c.base.FROZEN[path]
        c.require(review[key]==dict(path=path,sha256=expected),'Exact source-reviewed remover and unchanged replay sources')
    c.require(review['plan_sha256']==c.PLAN['sha256'],'Exact immutable complete raw map')
    history_spec=saved.read(spec['references']['history_spec'],64<<20)
    history_review=saved.read(spec['references']['history_review'],64<<20)
    c.require(history_spec['source_sha256']==spec['source_sha256']and history_spec['lock_identity']==c.LOCK
        and history_review['reviewer']=='/root'and history_review['passed']is True and history_review['blockers']==[]
        and history_review['spec']==spec['references']['history_spec']and history_review['source_sha256']==spec['source_sha256'],
        'Exact separately reviewed complete History spec/map')
    c.require(review['execution_source_sha256']==spec['execution_source_sha256'],
        'Independent root review retains the complete explicit execution-source closure')
    exclusion=saved.read(spec['references']['writer_exclusion']);c.validate_exclusion(exclusion,spec_ref,spec)
    source_check(spec);return spec,review,history_spec


def prerequisites(spec,history_spec,saved):
    refs=spec['references'];outer=saved.read(refs['replay_outer_exit']);observer=saved.read(refs['replay_observer_exit'])
    completed=saved.read(refs['replay_completed']);idle=saved.read(refs['idle'])
    for row in (outer,observer):
        c.require(row['reviewer']=='/root'and row['passed']is True and row['actual_tool_exit_observed']is True
            and row['actual_tool']['exit_code']==0 and row['owners_gone']is row['physically_drained']is True
            and row['lock_identity']==c.LOCK,'Actual successful independently drained replay and observer exits')
    c.require(outer['schema']=='minicpm5_retained_replay_root_actual_outer_exit_v1'
        and observer['schema']=='minicpm5_retained_replay_root_actual_observer_exit_v1'
        and observer['actual_outer_exit']==refs['replay_outer_exit']and outer['full_replay_equality']==refs['replay_equality'],
        'Exact completed Mini outer/observer/equality lineage')
    done=completed['minicpm5_retained_replay_completed']
    c.require(completed['running']is False and completed['active_serial']is None
        and completed['live_tool_sessions']==completed['active_heavy_units']==[]
        and done['actual_outer_terminal']==refs['replay_outer_exit']and done['actual_observer_terminal']==refs['replay_observer_exit']
        and done['equality']==refs['replay_equality'],'Actual complete Mini finalization exists before retirement')
    equality=saved.read(refs['replay_equality'],16<<20)
    c.require(equality['schema']=='minicpm5_retained_raw_complete_replay_equality_v1'
        and equality['verification']==refs['verification']and equality['storage_plan_sha256']==c.PLAN['sha256']
        and all(equality[k]is True for k in ('complete','full11859_logical_inputs_preserved',
            'full32_replay_and_all_scientific_outputs_equal','all_original_failures_preserved',
            'layouts98_byte_identical','scientific_reports14_byte_identical'))
        and tuple(equality[k]for k in ('full_frames32','cache_pairs9408','cpu_hook_pairs5964','stage_vectors13632'))==(32,9408,5964,13632)
        and equality['full_model_qualified']is equality['new_speedup_claim']is False,'Every original actual scientific condition retained')
    oldcat=saved.read(history_spec['current_serial_catalog']);c.require(oldcat['current_handoff']==refs['idle'],'Exact current idle/catalog binding')
    for ref,value in ((refs['replay_outer_exit'],outer),(refs['replay_observer_exit'],observer)):
        rows=[r for r in oldcat['runs']if r['terminal']==ref]
        c.require(len(rows)==1 and rows[0]['tool']==value['actual_tool']and rows[0]['owners']==value['owners']
            and rows[0]['units']==value['units'],'Current complete catalog retains both actual replay tools and all owners')
    c.require(idle['running']is False and idle['active_serial']is None and idle['live_tool_sessions']==idle['active_heavy_units']==[],
        'Root-selected current idle metadata; actual freshness and all-holder proof still required')
    return oldcat,idle


def latest():
    from research import minicpm5_layer1_operator_sdk_runtime_completion_v4 as current
    return current.latest_handoff()


def holders(permit_ref,scopes):
    value,drain=scopes.invoke(permit_ref)
    c.require(value['schema']=='minicpm5_retained_retirement_all_uid_holders_v2'and value['permit']==permit_ref
        and value['complete']is value['read_only']is True and value['lock_identity']==c.LOCK
        and len(value['samples'])==2 and all(s['complete']is s['all_uid_processes_checked']is s['zero_other_argv_fd_map_holders']is True
            for s in value['samples']),'Both actual complete all-UID holder inspections')
    value['helper_scope_drain']=drain;return value


def storage(spec):
    """Separate bounded journal; no replay 512-entry limit is changed."""
    totals={}
    for key,cap,count_cap in (('control',c.LIMITS['control_bytes'],c.LIMITS['control_entries']),
        ('journal',c.LIMITS['journal_bytes'],c.LIMITS['journal_entries'])):
        root=Path(spec['paths'][key]);total=count=0;pending=[root]if os.path.lexists(root)else[]
        while pending:
            path=pending.pop();s=path.lstat();c.require(not stat.S_ISLNK(s.st_mode),'No output symlink')
            if stat.S_ISDIR(s.st_mode):
                c.require(path==root or key=='control'and path.parent==root,'Bounded output nesting');pending.extend(path.iterdir())
            else:
                count+=1;c.require(stat.S_ISREG(s.st_mode)and s.st_nlink==1 and s.st_size<=c.LIMITS['file_bytes']
                    and path.suffix=='.json','Bounded singleton JSON output')
                if key=='journal':
                    allowed=path.name in ('complete.json','postcondition.json')or bool(__import__('re').fullmatch(r'[0-9]{4}-(intent|done)\.json',path.name))
                    c.require(allowed and(path.name[0].isalpha()or int(path.name[:4])<2352),'Only exact first-run finite pair journal')
            total+=max(s.st_size,s.st_blocks*512)
        c.require(count<=count_cap and total<=cap,'Finite dedicated output entries and bytes');totals[key]=total
    for root,floor in ((ROOT,c.LIMITS['root_floor_bytes']),(Path('/mnt/nvme'),c.LIMITS['nvme_floor_bytes'])):
        fs=os.statvfs(root);reserve=sum(max(0,c.LIMITS[k+'_bytes']-totals[k])for k in ('control','journal'))if root==ROOT else 0
        c.require(fs.f_bavail*fs.f_frsize>=floor+reserve,'Original disk floor and remaining bounded retirement reservation')
    return totals


def run(spec_ref,review_ref):
    io=api();saved=Saved(io);spec,review,hs=load(spec_ref,review_ref,saved);oldcat,idle=prerequisites(spec,hs,saved)
    import minicpm5_retained_raw_transport_v1 as transport
    import minicpm5_retained_raw_transaction_v2 as transaction
    from minicpm5_retained_replay_history_v1 import History,Authority
    from qwen35_package_preflight_serial_v2 import save_history
    plan=saved.read(c.PLAN,16<<20);transport.validate_plan(plan)
    original=saved.read(c.base.ORIGINAL_PLAN,16<<20)
    paths=spec['paths'];root=Path(paths['control'])
    c.require(all(not os.path.lexists(p)for p in paths.values()),'Entire fresh first-run retirement namespace')
    lock=io.RetainedLock();lock.acquire();start=time.monotonic();last=[-1.0];last_holders=[-1.0]
    scopes=ForegroundScopes(io,lock)
    stopped=[False];handlers={};history=None;running_observation=None;permit_ref=None;intent_ref=None;lease_ref=None;transaction_result=None
    failure=None;published_terminal=None
    for sig in(signal.SIGINT,signal.SIGTERM):handlers[sig]=signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    def guard(force=False):
        lock.check();c.require(not stopped[0]and time.monotonic()-start<c.LIMITS['seconds'],'Finite retirement owner interruption/deadline')
        if force or time.monotonic()-last[0]>=.5:
            saved.unchanged();source_check(spec);storage(spec);io.process_api().active_heavy()
            expected=running_observation or dict(path=spec['references']['idle']['path'],identity=saved.identities[spec['references']['idle']['path']])
            c.require(latest()==expected,'No newer/replaced serial handoff during retirement')
            if history is not None:history.unchanged()
            last[0]=time.monotonic()
        if permit_ref is not None and(force or time.monotonic()-last_holders[0]>=10):
            holders(permit_ref,scopes);last_holders[0]=time.monotonic()
            # A privileged scan is blocking: force freshness again after it.
            lock.check();saved.unchanged();source_check(spec)
            expected=running_observation or dict(path=spec['references']['idle']['path'],identity=saved.identities[spec['references']['idle']['path']])
            c.require(latest()==expected,'Serial tip changed during blocking all-UID inspection')
            if history is not None:history.unchanged()
    try:
        guard(True)
        history=History(hs,spec['references']['history_spec'],spec['references']['history_review'],{},guard).resolve()
        first=history.physical_pair(Authority(history,guard,lambda:[]));fresh=history.freshness()
        c.require(fresh['latest_handoff']==spec['references']['idle'],'Exact current complete History idle tip')
        compat=c.compatibility(spec,hs,history.resolved,original,plan);guard(True);loaded_text_sources(spec)
        root.mkdir(mode=0o700);history_refs=save_history(root,first)
        expected_argv=[c.base.PYTHON,'-B','-u',c.SOURCES[1],'--mode','run','--spec',spec_ref['path'],
            '--spec-sha256',spec_ref['sha256'],'--review',review_ref['path'],'--review-sha256',review_ref['sha256']]
        c.require(lock.owner['argv']==expected_argv,'Exact prebound actual remover invocation')
        permit=dict(schema='minicpm5_retained_retirement_owner_permit_v1',mode='run',spec=spec_ref,review=review_ref,owner=lock.owner,
            retained_fd=lock.fd,lock_identity=c.LOCK,previous=spec['references']['idle'],complete_history=history_refs,
            writer_exclusion=spec['references']['writer_exclusion'])
        permit_ref=io.once(root/'permit.json',permit)
        before_holders=holders(permit_ref,scopes);guard(True)
        print(json.dumps(dict(event='mini_retirement_actual_tool_ack_required',permit=permit_ref,owner=lock.owner,
            running_handoff_path=paths['running'],no_unlink_before_actual_ack=True)),flush=True)
        # While waiting for the one externally observed ACK, it may replace idle
        # only with the exact expected running object; no deletion is released.
        def wait_guard():
            lock.check();saved.unchanged();source_check(spec);storage(spec)
            c.require(not stopped[0]and time.monotonic()-start<c.LIMITS['seconds'],'Finite actual ACK wait')
        io.wait_until(lambda:True if Path(paths['running']).exists()else None,wait_guard,c.LIMITS['ack_seconds'],message='Actual retirement tool ACK missing')
        running_ref=io.ref(paths['running']);running_value=saved.read(running_ref)
        current=running_value['minicpm5_retained_retirement_current']
        c.require(running_value==c.running(idle,spec['references']['idle'],spec_ref,spec,lock.owner,current['tool'],permit_ref,running_value['updated_at']),
            'Exact actual root retirement ACK')
        running_observation=dict(path=paths['running'],identity=saved.identities[paths['running']]);guard(True)
        compat.update(complete_history=history_refs,actual_running=running_ref)
        history_ref=io.once(root/'history-compatibility.json',compat)
        lease_ref=io.once(root/'exclusive-lease.json',dict(schema='minicpm5_retained_raw_exclusive_retirement_lease_v1',
            plan_sha256=c.PLAN['sha256'],all_writers_and_holders_drained=True,retirement_source=review['remover_source'],
            actual_owner=lock.owner,original_lock_identity=c.LOCK,permit=permit_ref,actual_running=running_ref,
            writer_exclusion=spec['references']['writer_exclusion'],all_uid_holder_evidence=before_holders))
        admission=dict(plan_sha256=c.PLAN['sha256'],independent_verification=review_ref,remover_source=review['remover_source'],
            replay_source=review['replay_source'],serial_lease=lease_ref,full_history_compatibility=history_ref)
        guard(True);intent_ref=transport.prepare_retirement_intent(plan,spec['references']['verification'],root/'intent.json',admission,guard)
        def exclusive():
            guard();return dict(plan_sha256=c.PLAN['sha256'],intent=intent_ref,serial_lease=lease_ref,
                all_writers_and_holders_drained=True,exclusive_namespace_write_lease=True)
        guard(True);loaded_text_sources(spec)
        transaction_result=transaction.retire(plan,spec['references']['verification'],intent_ref,paths['journal'],exclusive)
        guard(True);history.unchanged()
        after=history.physical_pair(Authority(history,guard,lambda:[]));guard(True)
        after_root=root/'post-history';after_root.mkdir();save_history(after_root,after)
    except BaseException as error:failure=error
    finally:
        drain=scopes.drain()
        try:
            lock.check()
            # Every privileged helper and foreground sudo/monitor is independently
            # gone before terminal publication; the original owner is still live.
            if root.exists():
                published_terminal=io.once(root/'terminal.json',dict(schema='minicpm5_retained_retirement_owner_terminal_v2',
                    spec=spec_ref,review=review_ref,owner=lock.owner,permit=permit_ref,intent=intent_ref,serial_lease=lease_ref,
                    running=None if running_observation is None else io.ref(paths['running']),transaction_result=transaction_result,
                    failed=failure is not None,failure=None if failure is None else dict(type=type(failure).__name__,message=str(failure)[:2048]),
                    self_exit_claimed=False,actual_exit_pending=True,spawned_model_workers=False,
                    helper_scope_drain=drain,original_lock_identity=c.LOCK,full_model_qualified=False,new_speedup_claim=False))
        finally:
            lock.release(drain['physically_drained'])
            for sig,handler in handlers.items():signal.signal(sig,handler)
    if failure is not None:raise failure
    return published_terminal


def acknowledge(request_ref):
    io=api();saved=Saved(io);request=saved.read(request_ref)
    c.require(set(request)=={'permit','actual_tool','updated_at'},'Exact root-observed retirement ACK request')
    permit=saved.read(request['permit']);spec,_,hs=load(permit['spec'],permit['review'],saved)
    c.require(request['permit']['path']==str(Path(spec['paths']['control'])/'permit.json'),'Exact actual live owner permit')
    io.descriptor(permit['owner'],permit['retained_fd'])
    c.require(io.process_api().proc(permit['owner']['pid'])==permit['owner'],'Independent actual live remover observation')
    previous=saved.read(spec['references']['idle'])
    expected=dict(path=spec['references']['idle']['path'],identity=saved.identities[spec['references']['idle']['path']])
    c.require(latest()==expected,'Exact original idle tip before external ACK')
    value=c.running(previous,spec['references']['idle'],permit['spec'],spec,permit['owner'],request['actual_tool'],request['permit'],request['updated_at'])
    saved.unchanged();io.descriptor(permit['owner'],permit['retained_fd']);c.require(latest()==expected,'Fresh actual ACK immediately before write')
    return io.once(Path(spec['paths']['running']),value)


def finalize(request_ref):
    io=api();saved=Saved(io);request=saved.read(request_ref)
    c.require(set(request)=={'spec','review','terminal','actual_tool','updated_at'},'Exact externally observed retirement terminal request')
    spec,review,hs=load(request['spec'],request['review'],saved);oldcat,_=prerequisites(spec,hs,saved)
    terminal=saved.read(request['terminal']);c.require(request['terminal']['path']==str(Path(spec['paths']['control'])/'terminal.json')
        and terminal['spec']==request['spec']and terminal['review']==request['review']
        and terminal['helper_scope_drain']['physically_drained']is True
        and terminal['helper_scope_drain']['pending_scope']is False,'Exact producer terminal with complete helper drainage')
    c.validate_scope_drain(terminal['helper_scope_drain'],terminal['owner'])
    from minicpm5_retained_replay_observer_v2 import tool
    tool(request['actual_tool']);io.process_api().gone(terminal['owner'])
    c.require(terminal['running']is not None,'Unacknowledged preflight failure needs explicit root failure publication')
    running=saved.read(terminal['running']);expected=dict(path=terminal['running']['path'],identity=saved.identities[terminal['running']['path']])
    from minicpm5_retained_replay_history_v1 import History,Authority
    from qwen35_package_preflight_serial_v2 import save_history
    import minicpm5_retained_raw_transport_v1 as transport
    lock=io.RetainedLock();lock.acquire();start=time.monotonic();history=None;permit_ref=None;last_scan=[-1.0]
    scopes=ForegroundScopes(io,lock);stopped=[False];handlers={}
    for sig in(signal.SIGINT,signal.SIGTERM):handlers[sig]=signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    def guard(force=False):
        lock.check();c.require(not stopped[0]and time.monotonic()-start<7200,'Finite independent retirement finalizer')
        saved.unchanged();source_check(spec);storage(spec);io.process_api().gone(terminal['owner'])
        c.require(latest()==expected,'Exact same latest actual retirement running handoff')
        if history is not None:history.unchanged()
        if permit_ref is not None and(force or time.monotonic()-last_scan[0]>=10):
            holders(permit_ref,scopes);last_scan[0]=time.monotonic()
            lock.check();saved.unchanged();source_check(spec);io.process_api().gone(terminal['owner'])
            c.require(latest()==expected,'Serial tip changed during final blocking all-UID inspection')
            if history is not None:history.unchanged()
    try:
        guard();history=History(hs,spec['references']['history_spec'],spec['references']['history_review'],{},guard).resolve()
        pair=history.physical_pair(Authority(history,guard,lambda:[]));guard()
        root=Path(spec['paths']['control'])/'final-history';c.require(not root.exists(),'Fresh independent retirement finalization namespace')
        root.mkdir();refs=save_history(root,pair);guard()
        plan=saved.read(c.PLAN,16<<20);transport.validate_plan(plan)
        # Finalizer has its own original lock owner and a separate read-only scan
        # permit; no deletion entry is used after the remover's actual exit.
        permit_ref=io.once(Path(spec['paths']['control'])/'finalizer-permit.json',dict(
            schema='minicpm5_retained_retirement_owner_permit_v1',mode='finalize',spec=request['spec'],review=request['review'],
            final_request=request_ref,owner=lock.owner,retained_fd=lock.fd,lock_identity=c.LOCK))
        final_holders=holders(permit_ref,scopes);guard(True)
        final_proof=dict(owners_gone=True,physically_drained=True,complete_history=refs,actual_owner=terminal['owner'],
            all_uid_holder_evidence=final_holders,helper_scope_drain=scopes.drain(),lock_identity=c.LOCK)
        if request['actual_tool']['exit_code']==0:
            c.require(not terminal['failed']and terminal['transaction_result']is not None,'Successful actual transaction result required')
            result=saved.read(terminal['transaction_result'],16<<20)
            verified=transport._verification(plan,spec['references']['verification'])
            c.require(result['schema']=='minicpm5_retained_raw_transaction_complete_v1'and result['complete']is True
                and result['plan_sha256']==c.PLAN['sha256']
                and result['intent']==terminal['intent']and len(result['pairs'])==2352
                and all(row['path']==str(Path(spec['paths']['journal'])/('%04d-done.json'%i))for i,row in enumerate(result['pairs']))
                and result['original_allocated_bytes_retired']==sum(row['original_allocated_bytes']for row in verified['records'])
                and result['all_keeper_bytes_retained']is result['original_capture_identities_preserved']is True,
                'Complete actual original transaction result')
            independent=transport.audit_retired_state(plan,spec['references']['verification'],terminal['intent'],
                Path(spec['paths']['control'])/'independent-retired-postcondition.json',guard)
            final_proof.update(independent_postcondition=independent,original_allocated_bytes_retired=result['original_allocated_bytes_retired'])
        extension=history.meta.read(hs['history_overlay'])['source_extension']
        objects=c.completed_outputs(spec,hs,extension,oldcat,running,terminal['running'],request['terminal'],terminal,
            request['actual_tool'],final_proof,request['updated_at'])
        from research import minicpm5_layer1_operator_sdk_runtime_completion_v4 as consumer
        virtual={ref['path']:(ref,value)for ref,value in objects}
        def virtual_read(ref):
            if ref['path']in virtual:
                expected_ref,value=virtual[ref['path']];c.require(ref==expected_ref,'Exact composed retirement publication reference');return value
            return history.meta.read(ref)
        consumer.serial_catalog(dict(serial_predecessors=objects[1][0],source_sha256=spec['source_sha256']),virtual_read,guard)
        guard(True)
        for ref,value in objects:
            guard(True);c.require(io.once(Path(ref['path']),value)==ref,'Exact complete durable retirement publication bytes')
            saved.read(ref)
        return dict(actual_exit=objects[0][0],catalog=objects[1][0],idle=objects[2][0],passed=objects[0][1]['passed'])
    finally:
        drain=scopes.drain()
        try:
            if Path(spec['paths']['control']).exists():
                io.once(Path(spec['paths']['control'])/'finalizer-helper-drain.json',drain)
        finally:
            lock.release(drain['physically_drained'])
            for sig,handler in handlers.items():signal.signal(sig,handler)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=('run','ack','finalize'),required=True)
    for key in ('spec','spec-sha256','review','review-sha256','request','request-sha256'):parser.add_argument('--'+key)
    args=parser.parse_args(argv);c.require(sys.dont_write_bytecode and __debug__,'Explicit -B/assertions')
    os.sched_setaffinity(0,{7});os.setpriority(os.PRIO_PROCESS,0,19)
    resource.setrlimit(resource.RLIMIT_AS,(c.LIMITS['memory_address_bytes'],c.LIMITS['memory_address_bytes']))
    resource.setrlimit(resource.RLIMIT_FSIZE,(c.LIMITS['file_bytes'],c.LIMITS['file_bytes']))
    if args.mode=='run':result=run(dict(path=args.spec,sha256=args.spec_sha256),dict(path=args.review,sha256=args.review_sha256))
    else:result=(acknowledge if args.mode=='ack'else finalize)(dict(path=args.request,sha256=args.request_sha256))
    print(json.dumps(result),flush=True);return 0


if __name__=='__main__':raise SystemExit(main())
