"""One saved-data comparison after actual completed native and SDK audits.

No import-time IO, new service, model execution or automatic successor launch.
Only run() acquires the original serial descriptor; pure helpers are mockable.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import signal
import sys
ROOT=Path(__file__).resolve().parents[1];Q=ROOT/'research';sys.path[:0]=[str(Q),str(ROOT)]
from research import audit_minicpm5_layer1_operator_native_result_v3 as a
from research import compare_minicpm5_layer1_native_sdk_v3 as comparison
from research import minicpm5_native_sdk_report_transport_v1 as transport
from research import minicpm5_layer1_decode_boundary_v1 as boundary
from research import minicpm5_layer1_operator_sdk_results_v2 as sdk_reader
r=a.r;require=r.require;LIMITS=copy.deepcopy(a.LIMITS);SELF=str(Path(__file__).resolve())


def validate_spec(ref,review_ref,spec,review):
    require(set(spec)=={'schema','native_audit_exit','current_serial_catalog','source_extension','output','limits'}
        and spec['schema']=='minicpm5_native_sdk_saved_comparison_spec_v1'and spec['limits']==LIMITS,
        'Exact small saved-comparison spec and unchanged audit limits')
    for key in ('native_audit_exit','current_serial_catalog'):r.life.ref(spec[key])
    require(set(review)=={'schema','reviewer','passed','blockers','spec','source_extension'}
        and review['schema']=='minicpm5_native_sdk_saved_comparison_source_review_v1'and review['reviewer']=='/root'
        and review['passed']is True and review['blockers']==[]and review['spec']==ref
        and review['source_extension']==spec['source_extension'],'Separate exact concrete comparison source review')
    r.life.ref(review_ref);pins=spec['source_extension']
    require(type(pins)is dict and 0<len(pins)<=2048,'Finite additive source extension')
    for path,sha in pins.items():
        require(type(path)is str and Path(path).is_absolute()and str(Path(path))==path and '..'not in Path(path).parts
            and path.startswith(str(ROOT)+'/')and Path(path).suffix in ('.py','.json','.md','.txt','.sh')
            and type(sha)is str and re.fullmatch('[a-f0-9]{64}',sha),'Only exact new local text pins')
    require(all(str(Path(module.__file__).resolve())in pins for module in (sys.modules[__name__],comparison,transport)),
        'Entry, comparator and transport explicitly pinned in separate extension')
    path=Path(spec['output']);require(path.parent==Q and re.fullmatch(
        r'minicpm5-native-sdk-saved-comparison-[a-z0-9-]+\.json',path.name),'Fresh separate comparison output namespace')


def audit_terminal(ended,result,spec,review):
    require(set(ended)=={'schema','reviewer','audit_spec','audit_review','result','owner','actual_tool',
        'actual_tool_exit_observed','exact_owner_gone','lock_identity','self_exit_claimed','full_model_qualified','new_speedup_claim'}
        and ended['schema']=='minicpm5_native_independent_audit_actual_exit_v1'and ended['reviewer']=='/root'
        and ended['actual_tool_exit_observed']is ended['exact_owner_gone']is True
        and ended['lock_identity']==r.ORIGINAL_LOCK,'Exact root-observed native auditor external terminal')
    tool=a.core.actual_tool(ended['actual_tool'])
    for key in ('audit_spec','audit_review','result'):r.life.ref(ended[key])
    a.validate_spec(ended['audit_spec'],ended['audit_review'],spec,review)
    require(ended['result']['path']==spec['output']and result['schema']=='minicpm5_native_post_exit_independent_audit_v1'
        and result['complete']is True and result['reviewer']=='/root'
        and result['audit_spec']==ended['audit_spec']and result['audit_review']==ended['audit_review']
        and result['model_exit']==spec['model_exit']and result['recorder_exit']==spec['recorder_exit']
        and result['auditor_owner']==ended['owner']and result['lock_identity']==r.ORIGINAL_LOCK
        and result['source']==dict(path=a.SELF,sha256=spec['source_extension'][a.SELF]),
        'Exact original completed audit result/spec/review/owner/source bindings')
    require(ended['owner']['argv']==['python3','-B',a.SELF,'--spec',ended['audit_spec']['path'],
        '--spec-sha256',ended['audit_spec']['sha256'],'--review',ended['audit_review']['path'],
        '--review-sha256',ended['audit_review']['sha256']],'Exact original auditor command')
    for value in (ended,result):
        require(value['self_exit_claimed']is value['full_model_qualified']is value['new_speedup_claim']is False,
            'Independent terminal and result retain diagnostic flags')
    require(result['actual_auditor_tool_exit_verified']is False
        and result['actual_native_model_exit_verified']is result['actual_native_recorder_exit_verified']
        is result['every_raw_byte_reread_after_all_model_processes_exited']is True
        and type(result['original_sdk_model_exit_code'])is int and result['original_sdk_model_exit_code']==1,
        'Original audit complete guarantees and external-exit pending field preserved')
    proof=result['result'];require(proof['schema']=='minicpm5_layer1_operator_native_independent_verification_v2'
        and proof['complete']is True and [proof[k]for k in ('frames','direct_attention_calls','raw_files','journals','arm_result_files')]==[24,72,936,72,72]
        and proof['full_model_qualified']is proof['new_speedup_claim']is False,'Complete original native audit proof before any output read')
    return tool


def catalog_binding(serial,prior,required):
    rows={row['terminal']['path']:row for row in serial['runs']}
    require(all(rows.get(row['terminal']['path'])==row for row in prior['runs']),
        'Every original audit-era serial record retained unchanged')
    for ref,owner,tool in required:
        row=rows.get(ref['path']);require(row is not None and row['terminal']==ref
            and row['tool']==tool and owner in row['owners'],'Exact model/recorder/auditor terminal and owner in full current catalog')


def historical_pairs(resolved,records,ended,recorder,checked,prior,audit_result):
    scope=resolved['current_scope'];required={row['unit']for row in scope['nemotron']['history']['rows']}
    saved=records['drains'];pair=a.prior_audit.unpack(saved['paired_physical'])
    require(saved['schema']=='minicpm5_native_original_predecessor_pair_v1'and saved['resolved_plan_sha256']==resolved['plan_sha256']
        and saved['outer_owner']==ended['owner']and saved['lock_identity']==r.ORIGINAL_LOCK
        and saved['physical_pair_schema']==r.completion.physical.PAIR_SCHEMA
        and r.digest(pair)==saved['complete_observation_sha256'],'Exact original saved model predecessor pair')
    r.completion.physical.validate_pair(pair,required,scope['owners'],r.ORIGINAL_LOCK,known_units=scope['complete_known_units'])
    owners=copy.deepcopy(scope['owners'])+copy.deepcopy(prior['owners'])+checked['model_owners']+[recorder['owner']]
    units=set(scope['complete_known_units'])|set(prior['units'])|{checked['contract']['unit']}
    for owner in owners:
        group=owner.get('cgroup')
        if type(group)is str and group.startswith('0::'+r.p.GROUP_PREFIX):units.add(r.completion.physical.group_unit(group[3:]))
    pair=a.prior_audit.unpack(audit_result['current_physical_pair'])
    require(audit_result['physical_pair_schema']==r.completion.physical.PAIR_SCHEMA
        and audit_result['current_owner_count']==len(owners)and audit_result['current_unit_count']==len(units)
        and r.completion.physical.validate_pair(pair,required,owners,r.ORIGINAL_LOCK,known_units=units)==units,
        'Completed auditor retains entire exact original historical pair, owners and units')


def anchor_cases(recovered,results,read,guard):
    """Twelve extra original-call metadata reads, no historical payload reread."""
    collector=recovered['done']['collector'];plan=collector['collector_plan'];refs=plan['layer1_observation']['original_calls']
    require(len(refs)==84 and len(results)==2,'Original complete collector and both SDK ranks')
    output=[];used=[]
    for rank,result in enumerate(results):
        guard();stream=result['source_stream'];expected=[dict(rank=rank,arm=arm,call=ci)
            for arm in sdk_reader.inputs.stream.raw.ARMS for ci in boundary.DECODE_CALLS]
        original=stream['original_stream'];count=original['read_payload_bytes']
        require(type(count)is int and 0<count<=128*2**20 and original==dict(
            schema='minicpm5_layer1_rank_decode_stream_v1',complete=True,rank=rank,all_history_calls=42,decode_calls=18,
            physical_decode_rows=72,full_cache_records=84,restored_full_cache_bytes=84*sdk_reader.inputs.stream.raw.CACHE_BYTES,
            read_payload_bytes=count,all_read_bytes_match_saved_audit=True,prefill_replayed=False,attention_reexecuted=False,
            full_model_qualified=False,new_speedup_claim=False),'Original complete authenticated full-history stream')
        require(stream==dict(schema='minicpm5_layer1_sdk_batch_layout_anchor_stream_v1',complete=True,rank=rank,
            original_stream=original,selected_anchor_calls=expected[:6],verified_not_reexecuted_calls=expected[6:],
            selected_decode_calls=6,skipped_decode_calls=12,all_other_arms_verified_not_reexecuted=True,
            prefill_reexecuted=False,full_model_qualified=False,new_speedup_claim=False)
            and result['rank']==rank and len(result['records'])==6,'Exact original AnchorStream metadata, including other arms')
        cases=[]
        for index,ci in enumerate(boundary.DECODE_CALLS):
            guard();source=refs[rank*42+ci]
            require([source[k]for k in ('rank','arm','call')]==[rank,'strict_anchor_s0',ci],'Exact original anchor reference order')
            ref={k:source[k]for k in ('path','sha256')};case=boundary.boundary(plan,read(ref),ref)
            record=result['records'][index]['result']
            require(case['arm']=='strict_anchor_s0'and case['rank']==rank and case['call']['index']==ci
                and record['source_case_sha256']==r.digest(case)and record['original_call']==ref
                and record['call']==case['call'],'Same original boundary() case authenticated by completed SDK recovery')
            cases.append(case);used.append(ref)
        output.append(cases)
    require(len({ref['path']for ref in used})==12,'All twelve distinct original call metadata references')
    return output,used


class ReadIO:
    def __init__(self,plan,native,guard,sdk_root):
        self.native=a.audit_io.NativeAuditIO(plan,native,guard);self.guard=guard;self.root=plan['output_root']
        self.sdk_root=sdk_root;self.sdk={}
        require(self.root!=sdk_root and not self.root.startswith(sdk_root+'/')and not sdk_root.startswith(self.root+'/'),
            'Distinct original native and SDK output namespaces')
    def read_bytes(self,ref,path,size):
        self.guard()
        if path.startswith(self.root+'/'):return self.native.read_bytes(ref,path,size)
        require(path.startswith(self.sdk_root+'/')and path not in self.sdk,'Only distinct original SDK output payloads')
        raw=sdk_reader.payload(ref,Path(path),size,self.guard);self.sdk[path]=copy.deepcopy(ref['identity']);return raw
    def finish(self):
        require(len(self.native._read)==1008 and len(self.sdk)==624,'All936 native raw +72 metadata +624 SDK raw reads')
        value=self.native.unchanged()
        for path,wanted in self.sdk.items():
            self.guard();require(r.identity(path)==wanted,'Every SDK input identity remains unchanged after comparison')
        return value


def compare_saved(spec_ref,review_ref,lease):
    guard=a.AuditGuard(lease);guard(force=True);metadata=r.Metadata(guard);lease.metadata=metadata;read=metadata.read
    spec=read(spec_ref);review=read(review_ref);validate_spec(spec_ref,review_ref,spec,review)
    for path,sha in spec['source_extension'].items():guard();r.sources.text(path,sha)
    external=read(spec['native_audit_exit']);audit_result=read(external['result'])
    audit_spec=read(external['audit_spec']);audit_review=read(external['audit_review'])
    audit_tool=audit_terminal(external,audit_result,audit_spec,audit_review);r.p.gone(external['owner'])
    ended=read(audit_spec['model_exit']);recorder=read(audit_spec['recorder_exit'])
    a.core.actual_tool(ended['actual_tool']);a.core.actual_tool(recorder['actual_tool'],allow_call=True)
    require(ended['actual_exit_observed']is True and type(ended['actual_exit_code'])is int and ended['actual_exit_code']==0,
        'No partial/failed native model can reach saved comparison')
    require(audit_tool['session_id']!=ended['actual_tool']['session_id']and
        (recorder['actual_tool']['kind']=='tool_call'or audit_tool['session_id']!=recorder['actual_tool']['session_id']),
        'Separate actual model/recorder/native auditor terminal handles')
    for owner in (ended['owner'],ended['coordinator'],ended['native'],recorder['owner']):r.p.gone(owner)
    live=read(ended['live']);model_spec=read(ended['spec']);model_review=read(ended['review'])
    require(set(model_spec)=={'schema','overlay','runner','unit','lock_policy'}
        and model_spec['schema']=='minicpm5_native_process_spec_v1'and model_spec['lock_policy']==r.LOCK_POLICY
        and model_review['schema']=='minicpm5_native_process_source_review_v1'and model_review['reviewer']=='/root'
        and model_review['passed']is True and model_review['blockers']==[]and model_review['spec']==ended['spec']
        and model_review['overlay']==model_spec['overlay']and model_review['lock_policy']==r.LOCK_POLICY,'Original exact model source review')
    recovered=r.completion.completed(r.completion.BINDING,read,guard,lambda path,sha:r.sources.text(path,sha))
    resolved=r.overlays.resolve(read(model_spec['overlay']),recovered,read,guard)
    require(model_review['resolved_plan_sha256']==resolved['plan_sha256'],'Original resolved model plan unchanged')
    records={key:read(ref)for key,ref in ended['outputs'].items()}
    checked=a.core.validate(audit_spec['model_exit'],ended,audit_spec['recorder_exit'],recorder,live,records,resolved)
    admission=records['admission'];guard.bind(checked['contract'],resolved);r.validate_initial_descriptor(admission)
    r.execution.validate_intent(records['intent'],admission)
    require(admission['overlay']==model_spec['overlay']and admission['source_contract']==resolved['plan']['source_contract']
        and admission['sdk_operator_recovery']==r.completion.BINDING and admission['latest_completion']==resolved['plan']['latest_completion']
        and admission['completed_sdk_layout']==resolved['plan']['layer1_operator_boundary']
        and admission['serial_predecessors']==resolved['plan']['serial_predecessors'],'Original complete model admission lineage')
    model_partition=r.sources.classify(resolved['plan']['source_sha256'],resolved['plan']['source_contract'],read)
    inherited=r.sources.validate_native_receipt(records['source_authentication'],model_partition,ended['owner'],r.ORIGINAL_LOCK,admission['attempt_id'])
    require(inherited==audit_result['inherited_native_source_authentication'],'Original completed native source authentication')
    audit_plan=a.merged_plan(resolved['plan'],audit_spec['source_extension']);pins=a.merged_plan(audit_plan,spec['source_extension'])
    partition=r.sources.classify(pins['source_sha256'],pins['source_contract'],read)
    def sources():
        r.sources.verify_text_sources(partition,guard);r.sources.check_native_identities(records['source_authentication'],model_partition,guard)
        r.verify_loaded(pins,guard)
    sources()
    old_view=dict(latest_completion=pins['latest_completion'],serial_predecessors=audit_spec['current_serial_catalog'],source_sha256=pins['source_sha256'])
    prior=r.completion.latest.serial_catalog(old_view,read,guard)
    view=dict(old_view,serial_predecessors=spec['current_serial_catalog']);serial=r.completion.latest.serial_catalog(view,read,guard)
    required=[(audit_spec['model_exit'],ended['owner'],ended['actual_tool']),
        (audit_spec['recorder_exit'],recorder['owner'],recorder['actual_tool']),
        (spec['native_audit_exit'],external['owner'],audit_tool)]
    catalog_binding(prior,resolved['current_scope']['later_serial'],required[:2]);catalog_binding(serial,prior,required)
    historical_pairs(resolved,records,ended,recorder,checked,prior,audit_result)
    freshness=r.completion.latest.require_latest_terminal(view,read,guard);observed=r.completion.latest.latest_handoff()
    require(observed['path']==freshness['latest_handoff']['path'],'Exact current latest idle handoff');lease.execution_observation=copy.deepcopy(observed)
    owners=copy.deepcopy(resolved['current_scope']['owners'])+copy.deepcopy(serial['owners'])+checked['model_owners']+[recorder['owner'],external['owner']]
    units=set(resolved['current_scope']['complete_known_units'])|set(serial['units'])|{checked['contract']['unit']}
    for owner in owners:
        group=owner.get('cgroup')
        if type(group)is str and group.startswith('0::'+r.p.GROUP_PREFIX):units.add(r.completion.physical.group_unit(group[3:]))
    scope=dict(resolved['current_scope'],owners=owners,later_serial=serial,complete_known_units=sorted(units));lease.scope=scope
    authority=r.current_authority.Authority(r,lease);fresh_pair=a.pair(scope,guard,authority)
    owned=a.outer.OwnedService(checked['contract'],ended['outputs']['admission'],admission,metadata)
    owned.attempted=True;owned.owner=records['coordinator_owner'];drain=owned.verify_drained()
    require(drain==audit_result['model_drain'],'Exact original model remains physically drained')
    rank_refs=[recovered['recovery']['original_model_outputs']['rank%d_result'%rank]for rank in (0,1)]
    require(rank_refs==audit_result['sdk_rank_references'],'Same independently audited original SDK ranks')
    sdk=[read(ref)for ref in rank_refs];cases,case_refs=anchor_cases(recovered,sdk,read,guard)
    io=ReadIO(resolved['plan'],resolved['native_bundle'],guard,recovered['base_plan']['output_root'])
    io.native.final_inventory();guard(force=True)
    report=comparison.compare(checked['driver'],resolved['plan'],audit_result,sdk,cases,recovered['base_plan'],recovered['recovery'],io.read_bytes,guard)
    inventory=io.finish();require(inventory==audit_result['inventory'],'Same entire frozen native audited output inventory')
    require(a.pair(scope,guard,authority)==fresh_pair and owned.verify_drained()==drain,'Full historical/current physical scope unchanged')
    sources();metadata.unchanged();guard(force=True)
    packed=json.loads(transport.encode_report(report));guard(force=True)
    result=dict(schema='minicpm5_native_sdk_saved_comparison_execution_v1',complete=True,spec=spec_ref,review=review_ref,
        owner=lease.admission['outer_owner'],lock_identity=r.ORIGINAL_LOCK,native_audit_exit=spec['native_audit_exit'],native_audit=external['result'],
        native_model_exit=audit_spec['model_exit'],native_recorder_exit=audit_spec['recorder_exit'],sdk_recovery=r.completion.BINDING,
        native_driver=ended['outputs']['native_result'],sdk_rank_results=rank_refs,extra_anchor_call_metadata=case_refs,
        current_serial_catalog=spec['current_serial_catalog'],serial_freshness=freshness,source=dict(path=SELF,sha256=pins['source_sha256'][SELF]),
        complete_source_map_sha256=r.digest(pins['source_sha256']),native_plan_sha256=resolved['plan_sha256'],
        sdk_plan=recovered['base_plan_ref'],current_physical_pair=a.prior_audit.packed_physical(fresh_pair),
        native_inventory=inventory,storage=guard.storage,report=packed,comparator_reads=1632,extra_anchor_metadata_reads=12,
        actual_prior_audit_terminals_verified=True,original_sdk_model_exit_code=1,actual_comparator_tool_exit_verified=False,
        self_exit_claimed=False,full_model_qualified=False,new_speedup_claim=False)
    require(len((json.dumps(result,indent=2,allow_nan=False)+'\n').encode())<LIMITS['output_file_bytes'],
        'Complete wrapper including full report and physical pair below original2MiB')
    metadata.unchanged();guard(force=True);return result


def run(spec_ref,review_ref):
    require(sys.dont_write_bytecode and __debug__,'Original -B and assertions')
    r.p.user_bus();fd,owner,first=a.support.acquire()
    lease=r.Lease(dict(outer_owner=owner,lock_identity=r.ORIGINAL_LOCK,retained_fd=fd,lock_policy=r.LOCK_POLICY,
        initial_descriptor_proof=first,unit='trn1-mini-native-sdk-saved-comparator'))
    try:
        os.sched_setaffinity(0,{LIMITS['cpu']});os.nice(LIMITS['nice'])
        resource.setrlimit(resource.RLIMIT_AS,(LIMITS['address_space_bytes'],)*2)
        resource.setrlimit(resource.RLIMIT_CPU,(LIMITS['cpu_seconds'],LIMITS['hard_cpu_seconds']))
        resource.setrlimit(resource.RLIMIT_FSIZE,(LIMITS['output_file_bytes'],)*2)
        signal.signal(signal.SIGALRM,lambda *args:(_ for _ in ()).throw(TimeoutError('Original audit wall bound')))
        signal.alarm(LIMITS['wall_seconds']);result=compare_saved(spec_ref,review_ref,lease)
        target=Path(lease.metadata.read(spec_ref)['output']);require(not target.exists(),'Never overwrite comparison evidence')
        lease.check('comparison.publish');r.atomic_once(target,result);print(json.dumps(a.outer.reference(target)),flush=True)
        return a.outer.reference(target)
    finally:
        signal.alarm(0);a.support.release_after_physical_drain(fd,dict(physically_drained=True,no_comparator_children_or_services_started=True))


if __name__=='__main__':
    cli=argparse.ArgumentParser()
    for key in ('spec','review'):cli.add_argument('--'+key,required=True);cli.add_argument('--'+key+'-sha256',required=True)
    args=cli.parse_args();run(dict(path=args.spec,sha256=args.spec_sha256),dict(path=args.review,sha256=args.review_sha256))
