"""Source-reviewed, prepublication-only reconciliation of failed retirement.

No deletion/model/replay entry. Old v2 spec, sources and entire control tree stay
unchanged. Only a real failed finalizer terminal can support a future request.
The original failed remover row keeps its original execution source delta.
Recovery sources are separately identified post-exit audit provenance only.
"""
import argparse
import copy
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import signal
import stat
import sys
import time

Q=Path(__file__).resolve().parent;ROOT=Q.parent
for directory in (Q,ROOT):
    if str(directory)not in sys.path:sys.path.insert(0,str(directory))
import minicpm5_retained_retirement_contract_v2 as c
import minicpm5_retained_retirement_caller_v2 as old
from minicpm5_retained_retirement_recovery_scope_v3 import ForegroundScopes,identity as process_shape

SELF=str(Q/'minicpm5_retained_retirement_recovery_v3.py')
HELPER=str(Q/'minicpm5_retained_retirement_recovery_holders_v3.py')
SCOPE=str(Q/'minicpm5_retained_retirement_recovery_scope_v3.py')
EXTRA_SOURCES=(SELF,HELPER,SCOPE)
RECOVERY_BYTES=128<<20
DIRECTORY_KEYS=('device','inode','mode','mtime_ns','ctime_ns')


def paths(name):
    c.require(type(name)is str and re.fullmatch('[a-z0-9][a-z0-9-]{0,47}',name),'Bounded fresh recovery name')
    root=ROOT/('minicpm5-retained-retirement-finalization-recovery-'+name)
    return dict(root=str(root),history=str(root/'history'),permit=str(root/'permit.json'),
        drain=str(root/'helper-drain.json'),actual_exit=str(root/'actual-exit.json'),
        catalog=str(Q/('minicpm5-retirement-recovery-'+name+'-serial-catalog.json')),
        idle=str(Q/('continuation-handoff-mini-retirement-recovery-'+name+'-complete.json')))


def verify_sources(io,supplement):
    c.require(set(supplement)==set(EXTRA_SOURCES)and str(Path(__file__).resolve())==SELF,
        'Exactly three independent post-exit audit sources')
    for path,sha in supplement.items():
        c.reference(dict(path=path,sha256=sha))
        c.require(c.sha(io.raw(path,1<<20))==sha,'Recovery audit source changed')


def validate_review(review,request,original,spec):
    pins=review['source_sha256'];name=review['recovery_name']
    c.require(review==dict(schema='minicpm5_retirement_failed_finalizer_recovery_review_v1',reviewer='/root',
        passed=True,blockers=[],original_request=request['original_request'],interruption=request['interruption'],
        original_spec=original['spec'],original_review=original['review'],source_sha256=pins,recovery_name=name,
        paths=paths(name),launch_authorized=True,deletion_authorized=False,model_or_raw_replay_authorized=False,
        original_writer_exclusion_retained=True),'Exact independent recovery review and original writer exclusion')
    c.require(set(pins)==set(EXTRA_SOURCES)and not set(pins)&set(spec['source_sha256']),
        'Recovery supplement adds only new audit sources; old execution pins are immutable')
    return dict(spec['source_sha256'],**pins)


def validate_observation(value,request,original,spec,terminal,running):
    from minicpm5_retained_replay_observer_v2 import tool
    keys={'schema','reviewer','original_request','failed_finalizer_tool','failed_finalizer_owner',
        'original_finalizer_source','original_lock_identity','running','running_identity','control_tree',
        'journal_absent','publication_paths_absent','finalizer_helper_drain','finalizer_log',
        'actual_exit_observed','termination_cause','retirement_deletion_not_resumed'}
    c.require(set(value)==keys and value['schema']=='minicpm5_retirement_finalizer_failed_observation_v1'
        and value['reviewer']=='/root'and value['original_request']==request['original_request']
        and value['original_lock_identity']==c.LOCK and value['actual_exit_observed']is True
        and value['retirement_deletion_not_resumed']is True,'Only a root-observed actual failed finalizer')
    tool(value['failed_finalizer_tool']);tool(original['actual_tool']);process_shape(value['failed_finalizer_owner'])
    c.require(value['failed_finalizer_tool']['exit_code']!=0 and original['actual_tool']['exit_code']!=0
        and value['failed_finalizer_tool']['session_id']!=original['actual_tool']['session_id'],
        'Real failed publisher is distinct from the failed remover')
    owner=value['failed_finalizer_owner'];expected=[c.base.PYTHON,'-B','-u',c.SOURCES[1],'--mode','finalize',
        '--request',request['original_request']['path'],'--request-sha256',request['original_request']['sha256']]
    c.require(owner['argv']==expected and value['original_finalizer_source']==
        dict(path=c.SOURCES[1],sha256=spec['source_sha256'][c.SOURCES[1]]),'Exact failed original v2 executable/argv')
    c.require(terminal['spec']==original['spec']and terminal['review']==original['review']
        and terminal['failed']is True and terminal['transaction_result']is None
        and terminal['running']==value['running']and value['running']['path']==spec['paths']['running']
        and terminal['original_lock_identity']==c.LOCK,'Original failed terminal and running binding retained')
    c.validate_scope_drain(terminal['helper_scope_drain'],terminal['owner'])
    current=running['minicpm5_retained_retirement_current']
    c.require(current['owner']==terminal['owner']and current['tool']=={k:original['actual_tool'][k]for k in ('kind','session_id')},
        'Retirement catalog must describe the original real remover')
    c.require(value['journal_absent']==spec['paths']['journal']and value['publication_paths_absent']==
        [spec['paths'][k]for k in ('actual_exit','catalog','idle')],
        'Only zero-journal and entirely prepublication failure is supported')
    c.require(value['finalizer_helper_drain']['path']==str(Path(spec['paths']['control'])/'finalizer-helper-drain.json'),
        'Actual failed finalizer must have retained its independent helper drain')
    c.require(type(value['termination_cause'])is str and len(value['termination_cause'])<=4096,
        'Bounded observed diagnostic cause; no timeout inferred from elapsed time')
    tree=value['control_tree'];root=Path(spec['paths']['control'])
    c.require(type(tree)is dict and str(root)in tree and len(tree)<=132,'Complete bounded old control tree snapshot')
    for name,row in tree.items():
        path=Path(name);c.require(path==root or root in path.parents,'Snapshot belongs to exact old control root')
        c.require(type(row)is dict and row.get('kind')in ('directory','file'),'Exact snapshot node kind')
        if row['kind']=='directory':
            c.require(set(row)=={'kind','identity'}and set(row['identity'])==set(DIRECTORY_KEYS)
                and(path==root or path.parent==root),'Exact bounded old directory identity')
        else:
            c.require(set(row)=={'kind','identity','sha256'}and path.suffix=='.json'
                and type(row['identity']['size'])is int and 0<=row['identity']['size']<=c.LIMITS['file_bytes'],
                'Bounded unchanged original JSON evidence')
            c.reference(dict(path=name,sha256=row['sha256']))
    for ref in (value['running'],value['original_finalizer_source'],value['finalizer_helper_drain']):c.reference(ref)
    log=value['finalizer_log'];c.require(set(log)=={'path','sha256','identity'}and type(log['identity']['size'])is int
        and 0<=log['identity']['size']<=c.LIMITS['file_bytes'],'Complete bounded real finalizer transcript')
    c.reference({k:log[k]for k in ('path','sha256')})


def directory_identity(st):
    return dict(device=st.st_dev,inode=st.st_ino,mode=st.st_mode,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns)


def preserved(io,spec,observation,full=False):
    """All old names/identities immutable; hashes at entry/exit/blocking/write boundaries."""
    root=Path(spec['paths']['control']);c.require(root.resolve()==root and not any(p.is_symlink()for p in (root,*root.parents)),
        'Canonical unchanged old control root')
    expected=observation['control_tree'];seen=set();pending=[root]
    while pending:
        path=pending.pop();name=str(path);c.require(name in expected and name not in seen,'No extra/repeated old control entry')
        seen.add(name);row=expected[name];st=path.lstat()
        if row['kind']=='directory':
            c.require(stat.S_ISDIR(st.st_mode)and directory_identity(st)==row['identity'],'Old directory identity changed')
            with os.scandir(path)as entries:
                for entry in entries:
                    c.require(len(pending)+len(seen)<133,'Bounded complete old namespace');pending.append(Path(entry.path))
        else:
            c.require(io.identity(name)==row['identity'],'Old partial metadata identity changed')
            if full:c.require(c.sha(io.raw(name,c.LIMITS['file_bytes']))==row['sha256'],'Old partial bytes changed')
    c.require(seen==set(expected),'An old partial entry is missing')
    log=observation['finalizer_log'];c.require(io.identity(log['path'])==log['identity'],'Old finalizer transcript identity changed')
    if full:c.require(c.sha(io.raw(log['path'],c.LIMITS['file_bytes']))==log['sha256'],'Old finalizer transcript bytes changed')
    absent=[observation['journal_absent'],*observation['publication_paths_absent']]
    c.require(all(not os.path.lexists(p)and not os.path.lexists(p+'.publishing')for p in absent),
        'Retirement journal or old partial publication exists; this narrow recovery must stop')


def old_owners_gone(io,terminal,observation):
    io.process_api().gone(terminal['owner']);io.process_api().gone(observation['failed_finalizer_owner'])
    count=0
    with os.scandir('/proc')as entries:
        for entry in entries:
            count+=1;c.require(count<=65536,'Bounded complete process namespace')
            if entry.name.isdigit():
                actual=io.process_api().proc(int(entry.name))
                c.require(actual is None or c.SOURCES[1]not in actual['argv'],'No original caller/finalizer process may remain')


def no_unlink_state(plan,verified,guard,identity_reader,stat_reader):
    c.require(len(plan['entries'])==len(verified['records'])==2352,'Entire original pair metadata set')
    allocated=0;chain=hashlib.sha256()
    for row,record in zip(plan['entries'],verified['records']):
        guard()
        for side in ('original','keeper'):
            actual=identity_reader(row[side]['realpath']);c.require(actual==row[side],'Selected original/keeper identity changed')
            chain.update(c.canonical(dict(index=row['index'],side=side,identity=actual)))
        blocks=stat_reader(row['original']['realpath']).st_blocks*512
        c.require(blocks==record['original_allocated_bytes'],'Original allocation changed; zero reclaim cannot be assumed')
        c.require(identity_reader(row['original']['realpath'])==row['original'],'Original identity changed across allocation observation')
        allocated+=blocks
    guard()
    return dict(all2352_originals_and_keepers_present_unchanged=True,selected_identity_chain_sha256=chain.hexdigest(),
        original_allocated_bytes_present=allocated,reclaimed_bytes=0,raw_payload_bytes_read=0)


def validate_helper_permit(io,permit_ref,permit,spec):
    c.require(set(permit)=={'schema','mode','spec','review','recovery_request','recovery_review','owner','retained_fd','lock_identity'}
        and permit['schema']=='minicpm5_retirement_recovery_holder_permit_v1'and permit['mode']=='recovery_finalize'
        and permit['lock_identity']==c.LOCK,'Exact read-only recovery holder permit')
    request=io.read(permit['recovery_request']);original=io.read(request['original_request'])
    review=io.read(permit['recovery_review'])
    c.require(request['recovery_review']==permit['recovery_review']and original['spec']==permit['spec']
        and original['review']==permit['review'],'Exact original inputs and independently reviewed recovery request')
    full=validate_review(review,request,original,spec);verify_sources(io,review['source_sha256'])
    c.require(permit_ref['path']==review['paths']['permit']and permit['owner']['argv']==
        [c.base.PYTHON,'-B','-u',SELF,'--request',permit['recovery_request']['path'],
         '--request-sha256',permit['recovery_request']['sha256']],'Exact new recovery owner argv')
    old.source_check(spec);io.loaded_local_sources(full)


def rebind(composed,spec,review,provenance,read,guard):
    """Pure fresh output chain; old remover execution source delta NEVER changes."""
    c.require([ref['path']for ref,value in composed]==[spec['paths'][k]for k in ('actual_exit','catalog','idle')],
        'Original three-object composition and semantic order')
    receipt,catalog,idle=copy.deepcopy([value for ref,value in composed]);p=review['paths']
    c.require(receipt['passed']is False and receipt['actual_tool']['exit_code']!=0
        and receipt['transaction_result']is None,'Failed retirement remains failed')
    receipt['final_proof']['recovery']=copy.deepcopy(provenance)
    receipt['reclaimed_bytes']=0
    receipt_ref=c.base.publication_ref(p['actual_exit'],receipt)
    idle['minicpm5_retained_retirement_completed']['actual_exit']=receipt_ref
    idle['minicpm5_retained_retirement_completed']['recovery_audit']=copy.deepcopy(provenance)
    idle['complete_current_serial_catalog_next']=p['catalog']
    idle_ref=c.base.publication_ref(p['idle'],idle)
    catalog['current_handoff']=idle_ref;catalog['runs'][-1]['terminal']=receipt_ref
    c.require(catalog['runs'][:-1]==composed[1][1]['runs'][:-1]
        and catalog['runs'][-1]['source_sha256']==composed[1][1]['runs'][-1]['source_sha256']
        and all(catalog['runs'][-1][k]==composed[1][1]['runs'][-1][k]for k in ('tool','owners','units')),
        'No recovery source or identity can be retroactively attributed to the remover')
    catalog_ref=c.base.publication_ref(p['catalog'],catalog)
    objects=[(receipt_ref,receipt),(catalog_ref,catalog),(idle_ref,idle)]
    pending={ref['path']:(ref,value)for ref,value in objects}
    def virtual(ref):
        if ref['path']not in pending:return read(ref)
        expected,value=pending[ref['path']];c.require(ref==expected,'Exact recovered object reference');return value
    from research import minicpm5_layer1_operator_sdk_runtime_completion_v4 as consumer
    full=dict(spec['source_sha256'],**review['source_sha256'])
    consumer.serial_catalog(dict(serial_predecessors=catalog_ref,source_sha256=full),virtual,guard)
    guard();return objects


def recovery_storage(spec,p,published):
    old_totals=old.storage(spec);root=Path(p['root']);total=count=nodes=0;pending=[root]if root.exists()else[]
    while pending:
        nodes+=1;c.require(nodes<=c.LIMITS['control_entries']+2,'Bounded complete recovery namespace')
        path=pending.pop();st=path.lstat();c.require(not stat.S_ISLNK(st.st_mode),'No recovery output symlink')
        if stat.S_ISDIR(st.st_mode):
            c.require(path==root or path==Path(p['history']),'Only exact new History subdirectory')
            for child in path.iterdir():
                c.require(nodes+len(pending)<c.LIMITS['control_entries']+2,'Bounded recovery directory enumeration')
                pending.append(child)
        else:
            count+=1;c.require(stat.S_ISREG(st.st_mode)and st.st_nlink==1 and st.st_size<=c.LIMITS['file_bytes']
                and path.suffix=='.json','Only bounded singleton recovery JSON')
        total+=max(st.st_size,st.st_blocks*512)
    for ref in published:
        if Path(ref['path']).parent==Q:
            st=Path(ref['path']).lstat();total+=max(st.st_size,st.st_blocks*512);count+=1
    c.require(count<=c.LIMITS['control_entries']and total<=RECOVERY_BYTES,'Separate finite recovery output envelope')
    fs=os.statvfs(ROOT)
    remaining=sum(max(0,c.LIMITS[k+'_bytes']-old_totals[k])for k in ('control','journal'))+max(0,RECOVERY_BYTES-total)
    c.require(fs.f_bavail*fs.f_frsize>=c.LIMITS['root_floor_bytes']+remaining,'Original root floor plus ALL remaining reservations')


def recover(request_ref):
    io=old.api();saved=old.Saved(io);request=saved.read(request_ref)
    c.require(set(request)=={'original_request','interruption','recovery_review','updated_at'},'Exact independent recovery request')
    for key in ('original_request','interruption','recovery_review'):c.reference(request[key])
    c.require(type(request['updated_at'])is str and 0<len(request['updated_at'])<=128,'Bounded actual recovery timestamp')
    original=saved.read(request['original_request'])
    c.require(set(original)=={'spec','review','terminal','actual_tool','updated_at'},'Exact original finalizer request')
    spec,old_review,hs=old.load(original['spec'],original['review'],saved)
    oldcat,_=old.prerequisites(spec,hs,saved)
    terminal=saved.read(original['terminal']);observation=saved.read(request['interruption']);review=saved.read(request['recovery_review'])
    running=saved.read(terminal['running']);full=validate_review(review,request,original,spec)
    validate_observation(observation,request,original,spec,terminal,running)
    c.require(original['terminal']['path']==str(Path(spec['paths']['control'])/'terminal.json'), 'Exact retained original terminal path')
    c.validate_scope_drain(saved.read(observation['finalizer_helper_drain']),observation['failed_finalizer_owner'])
    verify_sources(io,review['source_sha256']);io.loaded_local_sources(full)
    expected=dict(path=observation['running']['path'],identity=observation['running_identity'])
    c.require(saved.identities[expected['path']]==expected['identity'],'Exact original running identity from actual failed observation')
    p=review['paths'];outputs=[p[k]for k in ('actual_exit','catalog','idle')];published=[]
    c.require(all(not os.path.lexists(path)and not os.path.lexists(path+'.publishing')for path in p.values()),
        'Entire fresh recovery namespace')
    preserved(io,spec,observation,True)
    lock=io.RetainedLock();lock.acquire();started=time.monotonic();history=None;permit_ref=None;last_scan=[-1.0]
    scopes=ForegroundScopes(io,lock);stopped=[False];handlers={}
    for sig in (signal.SIGINT,signal.SIGTERM):handlers[sig]=signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    def outputs_unchanged():
        written={r['path']for r in published}
        c.require(all(not os.path.lexists(path)for path in outputs if path not in written)
            and all(not os.path.lexists(path+'.publishing')for path in outputs),'No concurrent or partial new publication')
        for ref in published:saved.read(ref)
    def fast_guard():
        lock.check();c.require(not stopped[0]and time.monotonic()-started<7200,'Original finite recovery finalizer deadline')
        io.process_api().gone(terminal['owner']);io.process_api().gone(observation['failed_finalizer_owner'])
    guard_stats=dict(count=0,seconds=0.0);telemetry_failures=[0]
    def full_guard(force=False,*,force_holders=None):
        if force_holders is None:force_holders=force
        fast_guard();saved.unchanged();old.source_check(spec);verify_sources(io,review['source_sha256'])
        recovery_storage(spec,p,published);old_owners_gone(io,terminal,observation)
        preserved(io,spec,observation,force);outputs_unchanged()
        c.require(old.latest()==expected,'Exact current recovery tip; no other serial successor')
        if history is not None:history.unchanged()
        if permit_ref is not None and(force_holders or time.monotonic()-last_scan[0]>=10):
            old.holders(permit_ref,scopes);last_scan[0]=time.monotonic()
            fast_guard();saved.unchanged();old.source_check(spec);verify_sources(io,review['source_sha256'])
            preserved(io,spec,observation,True);outputs_unchanged()
            c.require(old.latest()==expected,'Serial tip changed during blocking holder scan')
            if history is not None:history.unchanged()
        fast_guard()
    def guard(force=False,*,force_holders=None):
        before=time.monotonic();guard_stats['count']+=1
        try:return full_guard(force,force_holders=force_holders)
        finally:guard_stats['seconds']+=max(0.0,time.monotonic()-before)
    @contextmanager
    def stage(name):
        began=time.monotonic()
        def emit(status,success,error_type=None):
            now=time.monotonic()
            value=dict(event='mini_retirement_recovery_stage',stage=name,status=status,success=success,
                error_type=error_type,elapsed_seconds=max(0.0,now-started),stage_elapsed_seconds=max(0.0,now-began),
                global_guard_count=guard_stats['count'],global_guard_seconds=guard_stats['seconds'],
                telemetry_write_failures=telemetry_failures[0])
            # Observational stdout failure must not bypass checks or drainage.
            try:print(json.dumps(value),flush=True)
            except(OSError,ValueError):telemetry_failures[0]+=1
        emit('enter',None)
        try:yield
        except BaseException as error:
            emit('exit',False,type(error).__name__);raise
        else:emit('exit',True)
    # Global fences cycle only inside these explicit read-only stages. Direct
    # History/Authority proc/service/cgroup observations and selected-file
    # identity/allocation checks remain unchanged. In particular r12 metadata
    # calls Authority.check rather than history.meta.guard: both need the same
    # reviewed global cadence. This changes detection latency, not predicates.
    readonly_phase=[None];last_full=[None]
    def history_guard(force=False):
        if readonly_phase[0]is None:return guard(force)
        fast_guard();now=time.monotonic()
        if force or last_full[0]is None or now-last_full[0]>=.5:
            # Periodic full source/snapshot checks do not force the costly
            # holder scan. Its original 10-second completion-based due test
            # remains; explicit force and every existing boundary force both.
            last_full[0]=now;guard(True,force_holders=force);fast_guard()
    try:
        with stage('initial_fence'):
            guard(True)
            c.require(lock.owner['argv']==[c.base.PYTHON,'-B','-u',SELF,'--request',request_ref['path'],
                '--request-sha256',request_ref['sha256']],'Exact independently reviewed recovery invocation')
            Path(p['root']).mkdir(mode=0o700)
            print(json.dumps(dict(event='mini_retirement_recovery_live',owner=lock.owner,request=request_ref,lock_identity=c.LOCK,
                old_finalizer_tool=observation['failed_finalizer_tool'],old_remover_tool=original['actual_tool'],self_exit_claimed=False)),flush=True)
            from minicpm5_retained_replay_history_v1 import History,Authority
            from qwen35_package_preflight_serial_v2 import save_history
            import minicpm5_retained_raw_transport_v1 as transport
            history_spec=copy.deepcopy(hs);history_spec['source_sha256']=copy.deepcopy(full)
        with stage('resolve'):
            readonly_phase[0]='resolve'
            try:history=History(history_spec,spec['references']['history_spec'],spec['references']['history_review'],{},history_guard).resolve()
            finally:readonly_phase[0]=None;guard(True)
            c.require(history.resolved['plan']['source_sha256']==spec['source_sha256'],'Resolved historical map is exactly the old reviewed map')
        with stage('physical'):
            guard(True);readonly_phase[0]='physical'
            try:pair=history.physical_pair(Authority(history,history_guard,lambda:[]))
            finally:readonly_phase[0]=None;guard(True)
        with stage('history_snapshot'):
            Path(p['history']).mkdir();history_refs=save_history(Path(p['history']),pair);guard(True)
        with stage('holder_permit'):
            permit_ref=io.once(Path(p['permit']),dict(schema='minicpm5_retirement_recovery_holder_permit_v1',mode='recovery_finalize',
                spec=original['spec'],review=original['review'],recovery_request=request_ref,recovery_review=request['recovery_review'],
                owner=lock.owner,retained_fd=lock.fd,lock_identity=c.LOCK))
            holder_proof=old.holders(permit_ref,scopes);guard(True)
        with stage('verification'):
            plan=saved.read(c.PLAN,16<<20);transport.validate_plan(plan)
            saved_verification=saved.read(spec['references']['verification'],16<<20)
            verified=transport._verification(plan,spec['references']['verification'])
            c.require(verified==saved_verification,'Independent complete verification is held to its original metadata identity')
        with stage('selected_metadata_first'):
            guard(True);readonly_phase[0]='selected_metadata_first'
            try:first=no_unlink_state(plan,verified,history_guard,transport.identity,lambda path:Path(path).stat())
            finally:readonly_phase[0]=None;guard(True)
        with stage('selected_metadata_second'):
            guard(True);readonly_phase[0]='selected_metadata_second'
            try:second=no_unlink_state(plan,verified,history_guard,transport.identity,lambda path:Path(path).stat())
            finally:readonly_phase[0]=None;guard(True)
        c.require(first==second,'Both independent selected-file metadata passes remain identical')
        final_proof=dict(owners_gone=True,physically_drained=True,complete_history=history_refs,actual_owner=terminal['owner'],
            all_uid_holder_evidence=holder_proof,helper_scope_drain=scopes.drain(),lock_identity=c.LOCK,zero_reclaim_observation=second)
        provenance=dict(schema='minicpm5_retirement_failed_finalizer_recovery_v1',request=request_ref,
            original_request=request['original_request'],interruption=request['interruption'],recovery_review=request['recovery_review'],
            failed_finalizer_tool=observation['failed_finalizer_tool'],post_exit_audit_source_sha256=review['source_sha256'],
            audit_sources_attributed_to_original_remover=False,original_execution_source_sha256=spec['execution_source_sha256'],
            complete_audit_source_map_sha256=c.sha(c.canonical(full)),recovery_paths=p,old_control_preserved=True,
            deletion_authorized=False,model_or_raw_replay_authorized=False,new_speedup_claim=False)
        with stage('composition_consumer'):
            guard(True);readonly_phase[0]='composition_consumer'
            try:
                extension=history.meta.read(hs['history_overlay'])['source_extension']
                composed=c.completed_outputs(spec,hs,extension,oldcat,running,terminal['running'],original['terminal'],terminal,
                    original['actual_tool'],final_proof,request['updated_at'])
                objects=rebind(composed,spec,review,provenance,history.meta.read,history_guard)
            finally:readonly_phase[0]=None;guard(True)
        with stage('publication'):
            guard(True);io.loaded_local_sources(full)
            for ref,value in objects:
                guard(True);c.require(io.once(Path(ref['path']),value)==ref,'Exact fresh recovery publication bytes')
                published.append(ref);saved.read(ref)
                if ref['path']==p['idle']:expected=dict(path=ref['path'],identity=saved.identities[ref['path']])
                saved.unchanged();preserved(io,spec,observation,True);outputs_unchanged();history.unchanged()
            return dict(actual_exit=objects[0][0],catalog=objects[1][0],idle=objects[2][0],retirement_passed=False,
                reclaimed_bytes=0,original_remover_tool=original['actual_tool'],recovery_actual_exit_pending=True)
    finally:
        with stage('final_drain'):
            drain=scopes.drain()
            try:
                if Path(p['root']).exists():io.once(Path(p['drain']),drain)
            finally:
                lock.release(drain['physically_drained'])
                for sig,handler in handlers.items():signal.signal(sig,handler)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',required=True);parser.add_argument('--request-sha256',required=True)
    args=parser.parse_args(argv);c.require(sys.dont_write_bytecode and __debug__,'Explicit -B and assertions')
    os.sched_setaffinity(0,{7});os.setpriority(os.PRIO_PROCESS,0,19)
    resource.setrlimit(resource.RLIMIT_AS,(c.LIMITS['memory_address_bytes'],)*2)
    resource.setrlimit(resource.RLIMIT_FSIZE,(c.LIMITS['file_bytes'],)*2)
    print(json.dumps(recover(dict(path=args.request,sha256=args.request_sha256))),flush=True);return 0


if __name__=='__main__':raise SystemExit(main())
