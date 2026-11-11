"""Independent native post-exit raw auditor. No model launch or exit invention.

Actual model and recorder terminal references are mandatory reviewed inputs.
Only run() performs real operations; all verification is independently mockable.
V4 authenticates entry sources in the complete inherited-plus-new map. It never
duplicates inherited package pins into the bounded additive source extension.
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
import time
ROOT=Path(__file__).resolve().parents[1];Q=ROOT/'research';sys.path[:0]=[str(Q),str(ROOT)]
from research import minicpm5_layer1_operator_native_process_adapter_v7 as r
from research import minicpm5_layer1_operator_native_outer_support_v3 as support
from research import minicpm5_layer1_operator_native_post_exit_core_v5 as core
from research import minicpm5_layer1_operator_native_audit_io_v1 as audit_io
from research import minicpm5_layer1_operator_native_results_v2 as verifier
from research import audit_minicpm5_layer1_operator_sdk_result_v3 as prior_audit
import minicpm5_layer1_operator_native_outer_runner_v4 as outer
LIMITS=copy.deepcopy(prior_audit.LIMITS);SELF=str(Path(__file__).resolve());require=r.require


def validate_spec(spec_ref,review_ref,spec,review):
    require(set(spec)=={'schema','model_exit','recorder_exit','current_serial_catalog','source_extension','output','limits'}
        and spec['schema']=='minicpm5_native_post_exit_audit_spec_v1'and spec['limits']==LIMITS,
        'Small exact independent native audit spec and original audit resource envelope')
    for key in ('model_exit','recorder_exit','current_serial_catalog'):r.life.ref(spec[key])
    require(set(review)=={'schema','reviewer','passed','blockers','spec','source_extension'}
        and review['schema']=='minicpm5_native_post_exit_audit_source_review_v1'and review['reviewer']=='/root'
        and review['passed']is True and review['blockers']==[]and review['spec']==spec_ref
        and review['source_extension']==spec['source_extension'],'Independent concrete audit source review')
    path=Path(spec['output'])
    require(path.parent==Q and re.fullmatch(r'minicpm5-layer1-operator-native-[a-z0-9-]+-post-exit-audit-[a-z0-9-]+\.json',path.name),
        'Separate immutable independent report outside model output tree')
    pins=spec['source_extension'];require(type(pins)is dict and 0<=len(pins)<=2048,'Bounded additional text-source map')
    for name,sha in pins.items():
        require(type(name)is str and Path(name).is_absolute()and str(Path(name))==name and '..'not in Path(name).parts
            and name.startswith(str(ROOT)+'/')and Path(name).suffix in ('.py','.json','.md','.txt','.sh')
            and type(sha)is str and re.fullmatch('[a-f0-9]{64}',sha),'Only exact new local text pins')
    # Entry/core pins may be inherited; source_binding verifies the complete
    # immutable model plan plus the disjoint extension before source verification.


def merged_plan(plan,extension):
    value=copy.deepcopy(plan);pins=value['source_sha256']
    for path,sha in extension.items():
        require(path not in pins,'Audit source extension cannot replace or reclassify original scientific pins');pins[path]=sha
    return value


def source_binding(plan,extension,model_spec,model_review,model_overlay,producer):
    """Pure complete-source binding, preserving all inherited keys and origins.

    This receipt is only published after the existing full source classification,
    text verification, native-identity and loaded-source gates pass. The exact
    model spec/review/overlay references allow future consumers to reconstruct
    the complete inherited map; digests/counts never replace that reconstruction.
    """
    require(type(extension)is dict and 0<=len(extension)<=2048,'Bounded only-new post-exit source extension')
    for path,sha in extension.items():
        require(type(path)is str and Path(path).is_absolute()and str(Path(path))==path
            and '..'not in Path(path).parts and path.startswith(str(ROOT)+'/')
            and Path(path).suffix in ('.py','.json','.md','.txt','.sh')
            and type(sha)is str and re.fullmatch('[a-f0-9]{64}',sha),'Exact new local text source binding')
    for reference in (model_spec,model_review,model_overlay):r.life.ref(reference)
    allowed={SELF,str(Q/'record_minicpm5_layer1_operator_native_model_exit_v3.py')}
    require(producer in allowed,'Exact reviewed current post-exit producer path')
    inherited=plan['source_sha256'];r.sources.metadata.pin_shape(inherited)
    value=merged_plan(plan,extension);pins=value['source_sha256']
    paths={producer,SELF,str(Path(core.__file__).resolve()),
        str(Q/'minicpm5_layer1_operator_native_post_exit_core_v4.py')}
    require(all(path in pins and type(pins[path])is str and re.fullmatch('[a-f0-9]{64}',pins[path])for path in paths),
        'Every executed post-exit entry/core retained in complete inherited-or-new map')
    require(all(pins.get(path)==sha for path,sha in inherited.items())
        and len(pins)==len(inherited)+len(extension),'Complete inherited source map preserved exactly without duplicate pins')
    proof=dict(schema='minicpm5_native_post_exit_complete_source_binding_v1',
        model_spec=copy.deepcopy(model_spec),model_review=copy.deepcopy(model_review),model_overlay=copy.deepcopy(model_overlay),
        inherited_plan_sha256=r.digest(plan),inherited_source_map_sha256=r.digest(inherited),
        inherited_source_files=len(inherited),extension_source_map_sha256=r.digest(extension),extension_source_files=len(extension),
        complete_source_map_sha256=r.digest(pins),complete_source_files=len(pins),
        producer=producer,entry_sources={path:pins[path]for path in sorted(paths)},
        inherited_sources_replaced=False,inherited_sources_omitted=False,
        complete_source_reconstruction_required=True)
    return value,proof


def validate_source_binding(proof,plan,extension,model_spec,model_review,model_overlay,producer):
    """Future metadata consumer must reconstruct the full original plan first."""
    value,wanted=source_binding(plan,extension,model_spec,model_review,model_overlay,producer)
    require(proof==wanted,'Exact complete inherited-source reference/count/digest/entry binding')
    return value


def validate_recorded_source_binding(ended,model_exit_ref,recorder_spec,recorder_review,
                                    plan,model_overlay,audit_plan):
    """Consume the actual recorder receipt using its authenticated spec/review.

    Caller reads both references through the same stable Metadata reader and
    has already revalidated the model/recorder true exits and original model
    plan. The complete recorder map must also be covered by the auditor map.
    """
    for reference in (model_exit_ref,ended['recorder_spec'],ended['recorder_review'],ended['recorder_source']):
        r.life.ref(reference)
    producer=str(Q/'record_minicpm5_layer1_operator_native_model_exit_v3.py')
    require(ended['recorder_source']['path']==producer,'Exact actually executed recorder_v3 source')
    require(type(ended.get('source_binding'))is dict,'Actual recorder complete source-binding receipt required')
    require(set(recorder_spec)=={'schema','live','model_spec','model_review','actual_tool','source_extension','output','limits'}
        and recorder_spec['schema']=='minicpm5_native_model_exit_recorder_spec_v1'
        and recorder_spec['limits']==LIMITS and recorder_spec['output']==model_exit_ref['path']
        and recorder_spec['live']==ended['live'] and recorder_spec['model_spec']==ended['spec']
        and recorder_spec['model_review']==ended['review'] and recorder_spec['actual_tool']==ended['actual_tool'],
        'Actual recorder spec binds this complete model terminal and unchanged resource/true-exit contract')
    core.actual_tool(recorder_spec['actual_tool'])
    require(set(recorder_review)=={'schema','reviewer','passed','blockers','spec','source_extension'}
        and recorder_review['schema']=='minicpm5_native_model_exit_recorder_source_review_v1'
        and recorder_review['reviewer']=='/root' and recorder_review['passed']is True
        and recorder_review['blockers']==[] and recorder_review['spec']==ended['recorder_spec']
        and recorder_review['source_extension']==recorder_spec['source_extension'],
        'Actual recorder reviewed extension, not a caller-invented source map')
    recorded=validate_source_binding(ended['source_binding'],plan,recorder_spec['source_extension'],
        ended['spec'],ended['review'],model_overlay,producer)
    require(ended['recorder_source']==dict(path=producer,sha256=recorded['source_sha256'][producer])
        and all(audit_plan['source_sha256'].get(path)==sha for path,sha in recorded['source_sha256'].items()),
        'Entire actual recorder source map is retained and verified by the auditor source closure')
    return dict(verified=True,recorder_spec=copy.deepcopy(ended['recorder_spec']),
        recorder_review=copy.deepcopy(ended['recorder_review']),producer=producer,
        source_binding_sha256=r.digest(ended['source_binding']),
        complete_recorder_sources=ended['source_binding']['complete_source_files'],
        complete_inherited_plan_reconstructed=True,all_recorded_sources_covered_by_auditor=True)


def pair(scope,guard,authority):
    proof=r.current_physical.collect_pair(scope,guard,authority)
    required={row['unit']for row in scope['nemotron']['history']['rows']};known=set(scope['complete_known_units'])
    require(r.completion.physical.validate_pair(proof,required,scope['owners'],r.ORIGINAL_LOCK,known_units=known)==known,
        'Full original current and historical double physical scope')
    return proof


def verify_saved_driver(driver,plan,native_bundle,sdk_results,guard,*,io_factory=audit_io.NativeAuditIO):
    """The accepted exact order; wrapper is already removed by terminal core."""
    io=io_factory(plan,native_bundle,guard)
    entries=io.final_inventory()
    proof=verifier.verify(driver,plan,native_bundle,sdk_results,io.read_bytes,
        lambda path,sha:r.sources.text(path,sha),guard,entries)
    unchanged=io.unchanged()
    require(proof['complete']is True and proof['frames']==24 and proof['direct_attention_calls']==72
        and proof['raw_files']==936 and proof['journals']==proof['arm_result_files']==72,
        'Complete accepted original native24/72/936 scope')
    return proof,unchanged


class AuditGuard:
    """Original lease every call; full storage bounds at most one second apart.

    Construction is pure. The exact native output catalog becomes available
    only after authenticated original admission/contract validation.
    """
    def __init__(self,lease):
        self.lease=lease;self.contract=None;self.resolved=None;self.last=None;self.storage=None
    def bind(self,contract,resolved):
        require(self.contract is None,'Bind original output contract exactly once')
        r.life.binding(contract,resolved);self.contract=contract;self.resolved=resolved
        self(force=True)
    def __call__(self,*,force=False):
        self.lease.check('audit.complete');now=time.monotonic()
        if not force and self.last is not None and now-self.last<1:return
        available={name:(lambda fs:fs.f_bavail*fs.f_frsize)(os.statvfs(path))
            for name,path in [('root',ROOT),('output','/mnt/nvme')]}
        require(available['root']>=28*2**30 and available['output']>=30*2**30,
            'Original root28GiB and NVMe30GiB floors throughout independent audit')
        r.p.active_heavy()  # Standalone auditor never exempts any service.
        usage=None
        if self.contract is not None:
            entries,root,writers=r.output_inventory(self.contract['output_root'],
                self.resolved['plan']['native_operator_manifest'],{})
            usage=r.life.usage(self.contract,self.resolved,entries,root,successful_driver=True,writers=writers)
        self.lease.check('audit.storage.after')
        self.storage=dict(available_bytes=available,usage=usage,maximum_check_interval_seconds=1)
        self.last=now


def audit(spec_ref,review_ref,lease):
    guard=AuditGuard(lease);guard(force=True)
    metadata=r.Metadata(guard);lease.metadata=metadata
    spec=metadata.read(spec_ref);review=metadata.read(review_ref)
    validate_spec(spec_ref,review_ref,spec,review)
    for path,sha in spec['source_extension'].items():
        guard();require(hashlib.sha256(r.sources.text(path,sha)).hexdigest()==sha,'Complete independently reviewed audit text extension')
    ended=metadata.read(spec['model_exit']);recorder=metadata.read(spec['recorder_exit']);live=metadata.read(ended['live'])
    # Deny absent/failed/bool actual exit before any output or payload path is opened.
    core.actual_tool(ended['actual_tool']);core.actual_tool(recorder['actual_tool'],allow_call=True)
    require(ended['actual_exit_observed']is True and type(ended['actual_exit_code'])is int and ended['actual_exit_code']==0,
        'Explicit independently observed successful native model exit required')
    for owner in (ended['owner'],ended['coordinator'],ended['native'],recorder['owner']):r.p.gone(owner)
    r.p.active_heavy()
    model_spec=metadata.read(ended['spec']);model_review=metadata.read(ended['review'])
    require(set(model_spec)=={'schema','overlay','runner','unit','lock_policy'}
        and model_spec['schema']=='minicpm5_native_process_spec_v1'and model_spec['lock_policy']==r.LOCK_POLICY
        and model_review['schema']=='minicpm5_native_process_source_review_v1'and model_review['reviewer']=='/root'
        and model_review['passed']is True and model_review['blockers']==[]and model_review['spec']==ended['spec']
        and model_review['overlay']==model_spec['overlay']and model_review['lock_policy']==r.LOCK_POLICY,
        'Original exact model source review and original descriptor policy')
    recovered=r.completion.completed(r.completion.BINDING,metadata.read,guard,lambda path,sha:r.sources.text(path,sha))
    resolved=r.overlays.resolve(metadata.read(model_spec['overlay']),recovered,metadata.read,guard)
    require(model_review['resolved_plan_sha256']==resolved['plan_sha256'],'Original model plan digest unchanged')
    records={key:metadata.read(ref)for key,ref in ended['outputs'].items()}
    checked=core.validate(spec['model_exit'],ended,spec['recorder_exit'],recorder,live,records,resolved)
    a=records['admission'];contract=checked['contract'];guard.bind(contract,resolved);r.validate_initial_descriptor(a)
    r.execution.validate_intent(records['intent'],a)
    require(a['overlay']==model_spec['overlay']and a['source_contract']==resolved['plan']['source_contract']
        and a['sdk_operator_recovery']==r.completion.BINDING and a['latest_completion']==resolved['plan']['latest_completion']
        and a['completed_sdk_layout']==resolved['plan']['layer1_operator_boundary']
        and a['serial_predecessors']==resolved['plan']['serial_predecessors'],'Complete original model admission/source lineage')
    model_partition=r.sources.classify(resolved['plan']['source_sha256'],resolved['plan']['source_contract'],metadata.read)
    inherited=r.sources.validate_native_receipt(records['source_authentication'],model_partition,ended['owner'],r.ORIGINAL_LOCK,a['attempt_id'])
    pins_plan,source_proof=source_binding(resolved['plan'],spec['source_extension'],ended['spec'],ended['review'],model_spec['overlay'],SELF)
    recorder_spec=metadata.read(ended['recorder_spec']);recorder_review=metadata.read(ended['recorder_review'])
    recorder_source_proof=validate_recorded_source_binding(ended,spec['model_exit'],recorder_spec,recorder_review,
        resolved['plan'],model_spec['overlay'],pins_plan)
    partition=r.sources.classify(pins_plan['source_sha256'],pins_plan['source_contract'],metadata.read)
    r.sources.verify_text_sources(partition,guard);r.sources.check_native_identities(records['source_authentication'],model_partition,guard)
    r.verify_loaded(pins_plan,guard)
    # Current catalog additionally enrolls exact external model and recorder exits.
    view=dict(latest_completion=resolved['plan']['latest_completion'],serial_predecessors=spec['current_serial_catalog'],source_sha256=pins_plan['source_sha256'])
    serial=r.completion.latest.serial_catalog(view,metadata.read,guard);rows={row['terminal']['path']:row for row in serial['runs']}
    require(all(rows.get(row['terminal']['path'])==row for row in resolved['current_scope']['later_serial']['runs']),
        'Every complete model-predecessor serial record retained unchanged')
    for reference,owner,tool in [(spec['model_exit'],ended['owner'],ended['actual_tool']),(spec['recorder_exit'],recorder['owner'],recorder['actual_tool'])]:
        row=rows.get(reference['path']);require(row is not None and row['terminal']==reference and row['tool']==tool and owner in row['owners'],
            'Current complete catalog retains exact model and recorder actual tool exits/owners')
    freshness=r.completion.latest.require_latest_terminal(view,metadata.read,guard)
    observed=r.completion.latest.latest_handoff()
    require(observed['path']==freshness['latest_handoff']['path'],'Lock actual latest idle terminal during standalone audit')
    lease.execution_observation=copy.deepcopy(observed)
    owners=copy.deepcopy(resolved['current_scope']['owners'])+copy.deepcopy(serial['owners'])+checked['model_owners']+[recorder['owner']]
    units=set(resolved['current_scope']['complete_known_units'])|set(serial['units'])|{contract['unit']}
    for owner in owners:
        group=owner.get('cgroup')
        if type(group)is str and group.startswith('0::'+r.p.GROUP_PREFIX):units.add(r.completion.physical.group_unit(group[3:]))
    scope=dict(resolved['current_scope'],owners=owners,later_serial=serial,complete_known_units=sorted(units));lease.scope=scope
    authority=r.current_authority.Authority(r,lease)
    saved=records['drains'];saved_pair=r.completion.audit.unpack(saved['paired_physical'])
    require(saved['schema']=='minicpm5_native_original_predecessor_pair_v1'and saved['resolved_plan_sha256']==resolved['plan_sha256']
        and saved['outer_owner']==ended['owner']and saved['lock_identity']==r.ORIGINAL_LOCK
        and saved['physical_pair_schema']==r.completion.physical.PAIR_SCHEMA
        and r.digest(saved_pair)==saved['complete_observation_sha256'],'Original complete saved predecessor pair binding')
    old_scope=resolved['current_scope'];required={row['unit']for row in old_scope['nemotron']['history']['rows']}
    r.completion.physical.validate_pair(saved_pair,required,old_scope['owners'],r.ORIGINAL_LOCK,known_units=old_scope['complete_known_units'])
    owned=outer.OwnedService(contract,ended['outputs']['admission'],a,metadata);owned.attempted=True;owned.owner=records['coordinator_owner']
    model_drain=owned.verify_drained();fresh_pair=pair(scope,guard,authority)
    rank_refs=[recovered['recovery']['original_model_outputs']['rank%d_result'%rank]for rank in (0,1)]
    require([ref['path']for ref in rank_refs]==[
        '/mnt/nvme/minicpm5-layer1-operator-sdk-r2-20261111/rank0-result.json',
        '/mnt/nvme/minicpm5-layer1-operator-sdk-r2-20261111/rank1-result.json'],'Only original recovered SDK operator rank references')
    sdk_results=[metadata.read(ref)for ref in rank_refs]
    guard(force=True);proof,inventory=verify_saved_driver(checked['driver'],resolved['plan'],resolved['native_bundle'],sdk_results,guard)
    guard(force=True);after_pair=pair(scope,guard,authority);require(after_pair==fresh_pair,'Complete model and historical physical pair unchanged across raw audit')
    require(owned.verify_drained()==model_drain,'Exact native model physical drain unchanged across raw audit')
    r.sources.verify_text_sources(partition,guard);r.sources.check_native_identities(records['source_authentication'],model_partition,guard)
    r.verify_loaded(pins_plan,guard);metadata.unchanged();guard(force=True)
    return dict(schema='minicpm5_native_post_exit_independent_audit_v1',complete=True,reviewer='/root',
        audit_spec=spec_ref,audit_review=review_ref,model_exit=spec['model_exit'],recorder_exit=spec['recorder_exit'],
        source_binding=source_proof,
        recorder_source_binding_verified=recorder_source_proof,
        auditor_owner=lease.admission['outer_owner'],lock_identity=r.ORIGINAL_LOCK,source=dict(path=SELF,sha256=pins_plan['source_sha256'][SELF]),
        original_sdk_model_exit_code=1,actual_native_model_exit_verified=True,actual_native_recorder_exit_verified=True,
        model_drain=model_drain,current_physical_pair=r.completion.audit.packed_physical(fresh_pair),
        physical_pair_schema=r.completion.physical.PAIR_SCHEMA,current_unit_count=len(units),current_owner_count=len(owners),
        inherited_native_source_authentication=inherited,sdk_rank_references=rank_refs,inventory=inventory,result=proof,storage=guard.storage,
        every_raw_byte_reread_after_all_model_processes_exited=True,actual_auditor_tool_exit_verified=False,
        self_exit_claimed=False,full_model_qualified=False,new_speedup_claim=False)


def run(spec_ref,review_ref):
    require(sys.dont_write_bytecode and __debug__,'Original audit -B and assertions')
    r.p.user_bus();fd,owner,first=support.acquire()
    lease=r.Lease(dict(outer_owner=owner,lock_identity=r.ORIGINAL_LOCK,retained_fd=fd,lock_policy=r.LOCK_POLICY,
        initial_descriptor_proof=first,unit='trn1-mini-layer1-operator-native-independent-auditor'))
    try:
        os.sched_setaffinity(0,{LIMITS['cpu']});os.nice(LIMITS['nice'])
        resource.setrlimit(resource.RLIMIT_AS,(LIMITS['address_space_bytes'],)*2)
        resource.setrlimit(resource.RLIMIT_CPU,(LIMITS['cpu_seconds'],LIMITS['hard_cpu_seconds']))
        resource.setrlimit(resource.RLIMIT_FSIZE,(LIMITS['output_file_bytes'],)*2)
        signal.signal(signal.SIGALRM,lambda *args:(_ for _ in ()).throw(TimeoutError('Original independent audit wall bound')))
        signal.alarm(LIMITS['wall_seconds'])
        result=audit(spec_ref,review_ref,lease);spec=r.sources.load(spec_ref);target=Path(spec['output'])
        require(not target.exists(),'Never overwrite an independent audit result')
        raw=(json.dumps(result,indent=2,allow_nan=False)+'\n').encode();require(len(raw)<LIMITS['output_file_bytes'],'Original full independent audit output bound')
        r.atomic_once(target,result);print(json.dumps(outer.reference(target)),flush=True)
        return outer.reference(target)
    finally:
        signal.alarm(0)
        # This auditor starts no child or service. Other jobs remain guarded;
        # no acquired descriptor is retained beyond this serial read-only owner.
        support.release_after_physical_drain(fd,dict(physically_drained=True,no_auditor_children_or_services_started=True))


if __name__=='__main__':
    cli=argparse.ArgumentParser()
    for key in ('spec','review'):
        cli.add_argument('--'+key,required=True);cli.add_argument('--'+key+'-sha256',required=True)
    args=cli.parse_args();run(dict(path=args.spec,sha256=args.spec_sha256),dict(path=args.review,sha256=args.review_sha256))
