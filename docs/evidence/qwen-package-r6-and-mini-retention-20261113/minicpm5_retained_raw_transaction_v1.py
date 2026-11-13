"""Finite reviewed retirement transaction. No IO on import and no CLI.

Production entry requires the exact frozen2352 map, a persisted intent, source
and full-replay reviews, and an external exclusive serial lease guard. It does
not acquire a lock, launch a process, execute History, or infer permission.
All incomplete and failed journals remain evidence; no recursive cleanup exists.
"""
import copy
import os
from pathlib import Path
import stat
import sys

import minicpm5_retained_raw_transport_v1 as t
import minicpm5_retained_raw_compare_v1 as comparison


def expected_ref(path, value):
    raw=t.canonical(value)+b'\n'
    return dict(path=str(path),sha256=t.digest(raw),bytes=len(raw))


def durable(path, value):
    ref=expected_ref(path,value)
    if os.path.lexists(path):
        t.require(t.read_receipt(ref)==value,'Existing receipt must match exactly; partial receipts require explicit recovery review')
        return ref
    return t._write_once(path,value)


def _load_evidence(ref, suffix):
    path=Path(ref['path']);t.require(path.suffix==suffix,'Exact prerequisite file kind')
    ident=t.identity(path);t.require(ident['size']<32<<20,'Original bounded source/prerequisite contract')
    fd=t._open(ident)
    try:
        parts=[];count=0
        while True:
            block=os.read(fd,t.CHUNK)
            if not block:break
            count+=len(block);t.require(count<32<<20,'Prerequisite grew');parts.append(block)
        t._unchanged(fd,ident)
    finally:os.close(fd)
    raw=b''.join(parts);t.require(t.digest(raw)==ref['sha256'],'Prerequisite digest changed')
    if suffix=='.json':return t.json.loads(raw)
    raw.decode('utf8');return raw


def _admission(plan, verification_ref, intent_ref, guard):
    """Verify root-produced prerequisite contents; never execute referenced code."""
    intent=t.read_receipt(intent_ref);a=intent['admission'];pin=t.digest(t.canonical(plan))
    t.require(intent['plan_sha256']==a['plan_sha256']==pin
        and intent['verification']==verification_ref,'Same original intent, plan and full verification')
    executing={str(t.Q/'minicpm5_retained_raw_transaction_v1.py'):Path(__file__),
        str(t.Q/'minicpm5_retained_raw_transport_v1.py'):Path(t.__file__),
        str(t.Q/'minicpm5_retained_raw_compare_v1.py'):Path(comparison.__file__)}
    t.require(all(str(p.resolve())==expected for expected,p in executing.items()),'Exact executing successor module paths')
    t.require(a['remover_source']['path']==str(Path(__file__).resolve())
        and a['replay_source']['path']==str(t.Q/'minicpm5_retained_raw_legacy_v1.py'),
        'Admission must bind this actual remover and the concrete legacy successor')
    for key in ('remover_source','replay_source'):_load_evidence(a[key],'.py')
    review=_load_evidence(a['independent_verification'],'.json')
    t.require(review['schema']=='minicpm5_retained_raw_replay_and_removal_review_v1'
        and review['plan_sha256']==pin and review['verification']==verification_ref
        and review['remover_source']==a['remover_source'] and review['replay_source']==a['replay_source']
        and review['passed']is True and review['blockers']==[]
        and review['full11859_logical_inputs_preserved']is True
        and review['full32_replay_and_all_scientific_outputs_equal']is True
        and review['all_original_failures_preserved']is True
        and review['original_weights_precision_and_remaining_gates_unchanged']is True,
        'Independent source and complete real replay review is required before deletion')
    source_map=review['source_sha256']
    old_plan=_load_evidence(t.REFERENCES['recovery_plan'],'.json')
    t.require(type(source_map)is dict and all(source_map.get(p)==h for p,h in old_plan['source_sha256'].items()),
        'Entire unchanged original replay source map required')
    own_paths={*executing,a['replay_source']['path']}
    t.require(all(p in source_map for p in own_paths)
        and source_map[a['remover_source']['path']]==a['remover_source']['sha256']
        and source_map[a['replay_source']['path']]==a['replay_source']['sha256'],
        'Exact complete transport, remover, comparator and legacy source closure')
    # Authenticate every reviewed source, including the actual external caller.
    # Original sources were already text-only under the original recovery gate.
    source_identities={}
    for path,sha in source_map.items():
        _load_evidence(dict(path=path,sha256=sha),Path(path).suffix)
        source_identities[path]=t.identity(path)
    executed_paths=set(own_paths)
    for module in list(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if isinstance(filename,str) and Path(filename).is_absolute() and Path(filename).is_relative_to(t.ROOT):
            t.require(filename in source_map and filename.endswith('.py'), 'Every actually loaded local source in reviewed complete map')
            executed_paths.add(filename)
    history=_load_evidence(a['full_history_compatibility'],'.json')
    t.require(history['schema']=='minicpm5_retained_raw_full_history_compatibility_v1'
        and history['plan_sha256']==pin and history['passed']is True
        and history['complete_current_input_and_source_maps_checked']is True
        and history['all_predecessor_owner_service_cgroup_scope_preserved']is True
        and history['candidate_current_input_intersection']==0,
        'Current complete History compatibility proof required')
    lease=_load_evidence(a['serial_lease'],'.json')
    t.require(lease['schema']=='minicpm5_retained_raw_exclusive_retirement_lease_v1'
        and lease['plan_sha256']==pin and lease['all_writers_and_holders_drained']is True
        and lease['retirement_source']==a['remover_source'],'Reviewed exclusive writer/holder lease required')
    expected=dict(plan_sha256=pin,intent=intent_ref,serial_lease=a['serial_lease'],
        all_writers_and_holders_drained=True,exclusive_namespace_write_lease=True)
    def admitted():
        t.require(guard()==expected,'External exclusive lease lost or replaced')
        for path in executed_paths:
            t.require(t.identity(path)==source_identities[path],'Executed source changed after admission')
            _load_evidence(dict(path=path,sha256=source_map[path]),'.py')
    def complete_sources():
        admitted()
        for path,ident in source_identities.items():
            t.require(t.identity(path)==ident,'Original reviewed source changed during transaction')
    admitted.check_complete_sources=complete_sources
    admitted()
    equality=t.read_receipt(review['replay_equality'])
    expected_equality=comparison._compare(plan,verification_ref,comparison.ORIGINAL_RESULT,
        equality['replay_result'],admitted)
    t.require(equality==expected_equality and equality['complete']is True,
        'Recompute the complete independent original/replay comparison before deletion')
    return admitted


def retire(plan, verification_ref, intent_ref, journal, guard):
    """Only this production entry admits unlink. Caller owns real serial lease.

    Linux has no ordinary unlink-if-inode API: the external lease must exclude
    ALL writers, including pathname replacement, throughout this transaction.
    Rechecks and dirfd anchoring detect substitutions at every observable step;
    they do not claim protection against a writer violating that lease between
    final stat and unlink. Without that guarantee this entry must not be called.
    """
    t.validate_plan(plan)
    t.require(Path(journal).parent==t.ROOT and Path(journal).name.startswith('minicpm5-retained-raw-retirement-'),
        'New dedicated production journal namespace')
    admitted=_admission(plan,verification_ref,intent_ref,guard)
    return _retire(plan,verification_ref,intent_ref,journal,admitted)


def _unlink_exact(row, guard):
    path=Path(row['original']['realpath']);guard()
    fd=t._open(row['original'])
    try:
        parent=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try:
            guard();t._unchanged(fd,row['original'])
            current=os.stat(path.name,dir_fd=parent,follow_symlinks=False)
            t.require(stat.S_ISREG(current.st_mode) and current.st_nlink==1
                and t._stat_identity(path,current)==row['original'],'Unlink target changed')
            os.unlink(path.name,dir_fd=parent)
            os.fsync(parent)
        finally:os.close(parent)
    finally:os.close(fd)


def _retire(plan, verification_ref, intent_ref, journal, guard):
    (t.synthetic_plan(plan)if plan['scope']=='synthetic-tests-only'else t.validate_plan(plan))
    verified=t._verification(plan,verification_ref);intent=t.read_receipt(intent_ref)
    t.require(intent['plan_sha256']==t.digest(t.canonical(plan)) and intent['verification']==verification_ref
        and intent['complete_original_selection']==[r['original']['realpath']for r in plan['entries']],
        'Full durable intent before any pair retirement')
    journal=Path(journal)
    t.require(journal.is_absolute() and not any(p.is_symlink()for p in (journal,*journal.parents)), 'Canonical journal')
    guard();journal.mkdir(exist_ok=True)
    replay=t.Replay(plan,verification_ref,guard,_synthetic=plan['scope']=='synthetic-tests-only')
    completed=[]
    for row,record in zip(plan['entries'],verified['records']):
        guard();index=row['index'];ip=journal/('%04d-intent.json'%index);dp=journal/('%04d-done.json'%index)
        iv=dict(schema='minicpm5_retained_raw_pair_intent_v1',plan_sha256=intent['plan_sha256'],
            whole_intent=intent_ref,verification=verification_ref,pair=copy.deepcopy(record))
        ir=expected_ref(ip,iv)
        if os.path.lexists(dp):
            t.require(t.read_receipt(ir)==iv,'No completion without exact durable pair intent')
            candidates=[dict(schema='minicpm5_retained_raw_pair_done_v1',intent=ir,
                original_path=row['original']['realpath'],unlink_observed_in_this_attempt=observed,
                recovered_after_unrecorded_unlink=not observed)for observed in (True,False)]
            accepted=None
            for value in candidates:
                ref=expected_ref(dp,value)
                try:
                    if t.read_receipt(ref)==value:accepted=ref;break
                except (ValueError,KeyError):pass
            t.require(accepted is not None,'Invalid or partial completion receipt')
            t.require(not os.path.lexists(row['original']['realpath']),'Completed original path reappeared')
            replay.read_state(row['original_row']);completed.append(accepted);continue
        present=os.path.lexists(row['original']['realpath'])
        if present:
            t.require(t._pair(row,guard)==record,'Pair must remain byte-exact before retirement')
            durable(ip,iv);guard()
            _unlink_exact(row,guard);observed=True
        else:
            # Crash after unlink but before completion is recoverable only from
            # its durable exact intent; absence alone never proves a prior unlink.
            t.require(t.read_receipt(ir)==iv,'Missing original without durable pair intent')
            replay.read_state(row['original_row']);observed=False
        guard();t.require(not os.path.lexists(row['original']['realpath']),'Retired original reappeared')
        done=dict(schema='minicpm5_retained_raw_pair_done_v1',intent=ir,
            original_path=row['original']['realpath'],unlink_observed_in_this_attempt=observed,
            recovered_after_unrecorded_unlink=not observed)
        completed.append(durable(dp,done))
    post=journal/'postcondition.json'
    if post.exists():
        # Re-audit into an explicit fresh file rather than trusting stale live
        # state from a previous successful postcondition.
        post=journal/('postcondition-recheck-%d.json'%len(list(journal.glob('postcondition-recheck-*.json'))))
    proof=t._audit_retired(plan,verification_ref,intent_ref,post,guard)
    final=dict(schema='minicpm5_retained_raw_transaction_complete_v1',plan_sha256=intent['plan_sha256'],
        intent=intent_ref,pairs=completed,postcondition=proof,complete=True,
        original_allocated_bytes_retired=sum(r['original_allocated_bytes']for r in verified['records']),
        original_capture_identities_preserved=True,all_keeper_bytes_retained=True,
        full_model_qualified=False,new_speedup_claim=False)
    path=journal/'complete.json'
    if path.exists():path=journal/('complete-recheck-%d.json'%len(list(journal.glob('complete-recheck-*.json'))))
    guard()
    if hasattr(guard,'check_complete_sources'):guard.check_complete_sources()
    return durable(path,final)
