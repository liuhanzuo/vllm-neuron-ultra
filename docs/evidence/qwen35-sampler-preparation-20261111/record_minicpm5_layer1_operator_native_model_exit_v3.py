"""External native model terminal recorder; only bounded metadata and source IO.

Root supplies genuine observed tool terminal/live/spec/review. This process owns
the original descriptor, verifies full source/physical bindings, and records its
actual own identity. It never reads tensor/native bytes or claims its own exit.
V3 binds entry/core sources through the complete inherited model plan plus only
new local text pins; it does not duplicate the inherited package source map.
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
from research import audit_minicpm5_layer1_operator_native_result_v4 as audit
from research import minicpm5_layer1_operator_native_post_exit_core_v5 as core
r=audit.r;outer=audit.outer;support=audit.support;require=r.require
SELF=str(Path(__file__).resolve());LIMITS=copy.deepcopy(audit.LIMITS)


def actual_tool(value):
    require(type(value)is dict and set(value)=={'kind','session_id','terminal_chunk','exit_code'}
        and value['kind']=='tool_session'and type(value['session_id'])is int and value['session_id']>0
        and type(value['terminal_chunk'])is str and re.fullmatch('[a-f0-9]{6,64}',value['terminal_chunk'])
        and type(value['exit_code'])is int and -255<=value['exit_code']<=255,
        'Exact root-observed model tool terminal, including true nonzero failure')
    return value


def validate_spec(spec_ref,spec,review):
    require(set(spec)=={'schema','live','model_spec','model_review','actual_tool','source_extension','output','limits'}
        and spec['schema']=='minicpm5_native_model_exit_recorder_spec_v1'and spec['limits']==LIMITS,
        'Exact small external metadata recorder spec with original bounded audit resources')
    for key in ('live','model_spec','model_review'):r.life.ref(spec[key])
    actual_tool(spec['actual_tool'])
    require(set(review)=={'schema','reviewer','passed','blockers','spec','source_extension'}
        and review['schema']=='minicpm5_native_model_exit_recorder_source_review_v1'and review['reviewer']=='/root'
        and review['passed']is True and review['blockers']==[]and review['spec']==spec_ref
        and review['source_extension']==spec['source_extension'],'Exact independently reviewed recorder source and inputs')
    target=Path(spec['output'])
    require(target.parent==Q and re.fullmatch(r'minicpm5-layer1-operator-native-[a-z0-9-]+-model-actual-exit-[a-z0-9-]+\.json',target.name),
        'Fresh external model terminal namespace outside model output tree')
    pins=spec['source_extension'];require(type(pins)is dict and 0<=len(pins)<=2048,'Bounded additive recorder source extension')
    for path,sha in pins.items():
        require(type(path)is str and Path(path).is_absolute()and str(Path(path))==path and '..'not in Path(path).parts
            and path.startswith(str(ROOT)+'/')and Path(path).suffix in ('.py','.json','.md','.txt','.sh')
            and type(sha)is str and re.fullmatch('[a-f0-9]{64}',sha),'Exact local new text pins only')
    # Entry/core may already be pinned by the original model plan. The complete
    # inherited-or-new source binding is checked before the existing source gates.


class Guard(audit.AuditGuard):
    def __init__(self,lease):
        super().__init__(lease);self.successful=False;self.writers={}
    def __call__(self,*,force=False):
        self.lease.check('model_recorder.complete');now=time.monotonic()
        if not force and self.last is not None and now-self.last<1:return
        available={name:(lambda fs:fs.f_bavail*fs.f_frsize)(os.statvfs(path))
            for name,path in [('root',ROOT),('output','/mnt/nvme')]}
        require(available['root']>=28*2**30 and available['output']>=30*2**30,'Original root28GiB/NVMe30GiB recorder floors')
        r.p.active_heavy();usage=None
        if self.contract is not None:
            entries,root,writers=r.output_inventory(self.contract['output_root'],self.resolved['plan']['native_operator_manifest'],self.writers)
            usage=r.life.usage(self.contract,self.resolved,entries,root,successful_driver=self.successful,writers=writers)
        self.lease.check('model_recorder.storage.after');self.last=now
        self.storage=dict(available_bytes=available,usage=usage,maximum_check_interval_seconds=1)


def read_receipts(contract,metadata,guard,successful):
    """Only fixed original runtime JSON, never driver journals/results/raw/log."""
    wanted=dict(core.OUTPUTS)
    if not successful:
        for name in r.catalog.RUNTIME_FILES:
            if name.endswith('.json')and name not in wanted.values():wanted[name[:-5].replace('-','_')]=name
    refs={};records={};missing=[]
    for key,name in wanted.items():
        guard();path=Path(contract['output_root'])/name
        if not path.exists():missing.append(name);continue
        ref=outer.reference(path);value=metadata.read(ref);refs[key]=ref;records[key]=value;guard()
    require(not successful or not missing,'Every original successful native receipt required')
    require({'admission','intent','source_authentication','drains'}<=set(refs),
        'Recorder scope requires actual admitted run; never synthesize pre-admission receipts')
    return refs,records,missing


def terminal_handoff(resolved,a,records,refs,metadata,lease,guard,successful):
    intent=records['intent'];fresh=r.execution.validate_intent(intent,a)
    catalog=metadata.read(a['serial_predecessors'])
    require(catalog['current_handoff']==fresh['bound_terminal'],'Original admission catalog/intent terminal binding')
    chain_refs=[*fresh['publication_only_successors'],fresh['bound_terminal']]
    chain=[(ref,metadata.read(ref))for ref in chain_refs]
    previous=r.execution.validate_predecessor_chain(fresh,a['serial_predecessors'],chain[-1][1],chain)
    observed=r.completion.latest.latest_handoff()
    if observed['path']==a['current_execution']['handoff_path']:
        require('coordinator_owner'in records,'Original running handoff requires acknowledged coordinator')
        raw=r.sources.text(observed['path']);require(len(raw)<=2*2**20,'Original bounded complete running handoff')
        ref=dict(path=observed['path'],sha256=hashlib.sha256(raw).hexdigest());value=metadata.read(ref)
        require(value==r.execution.expected_running(previous,value['updated_at'],refs['admission'],a,intent,
            refs['coordinator_owner'],records['coordinator_owner']),
            'Exact original complete model running handoff preserved after externally observed exit')
        mode='original_model_running_tip_after_actual_exit'
    else:
        require(not successful and observed['path']==fresh['latest_handoff']['path'],
            'Only admitted failure before running publication may retain original exact predecessor tip')
        proof=r.completion.latest.require_latest_terminal(resolved['current_serial_view'],metadata.read,guard)
        require(proof==fresh,'Original full pre-admission freshness, no invented idle successor')
        ref=fresh['latest_handoff'];mode='original_pre_admission_tip_after_failed_launch'
    require(r.completion.latest.latest_handoff()==observed,'No newer or replaced handoff during terminal recording')
    lease.execution_observation=copy.deepcopy(observed);guard()
    return dict(reference=ref,mode=mode,no_new_handoff_published=True)


def record(spec_ref,review_ref,lease):
    guard=Guard(lease);guard(force=True);metadata=r.Metadata(guard);lease.metadata=metadata
    spec=metadata.read(spec_ref);review=metadata.read(review_ref);validate_spec(spec_ref,spec,review)
    require(not Path(spec['output']).exists(),'Never overwrite actual model terminal')
    for path,sha in spec['source_extension'].items():guard();r.sources.text(path,sha)
    tool=actual_tool(spec['actual_tool']);successful=tool['exit_code']==0;live=metadata.read(spec['live'])
    require(live['schema']=='minicpm5_native_model_actual_live_v1'and live['reviewer']=='/root'
        and live['actual_identity_verified']is True
        and live['actual_execution_handle']==dict(kind='tool_session',session_id=tool['session_id'])
        and live['spec']==spec['model_spec']and live['review']==spec['model_review'],
        'Root observed actual model live/tool/spec/review chain')
    r.life.process(live['owner'],live['owner']['argv']);r.p.gone(live['owner']);r.p.active_heavy()
    model=metadata.read(spec['model_spec']);model_review=metadata.read(spec['model_review'])
    require(set(model)=={'schema','overlay','runner','unit','lock_policy'}and model['schema']=='minicpm5_native_process_spec_v1'
        and model['lock_policy']==r.LOCK_POLICY and model_review['schema']=='minicpm5_native_process_source_review_v1'
        and model_review['reviewer']=='/root'and model_review['passed']is True and model_review['blockers']==[]
        and model_review['spec']==spec['model_spec']and model_review['overlay']==model['overlay']
        and model_review['lock_policy']==r.LOCK_POLICY,'Original exact model review and descriptor policy')
    recovered=r.completion.completed(r.completion.BINDING,metadata.read,guard,lambda path,sha:r.sources.text(path,sha))
    resolved=r.overlays.resolve(metadata.read(model['overlay']),recovered,metadata.read,guard)
    require(model_review['resolved_plan_sha256']==resolved['plan_sha256'],'Original reviewed plan digest')
    contract=r.life.build_contract(resolved,model['overlay'],model['runner'],model['unit'])
    refs,records,missing=read_receipts(contract,metadata,guard,successful);a=records['admission']
    require(refs['admission']==live['admission']and a['contract']==contract and a['outer_owner']==live['owner']
        and a['spec']==spec['model_spec']and a['review']==spec['model_review']and a['overlay']==model['overlay']
        and a['lock_policy']==r.LOCK_POLICY and a['lock_identity']==r.ORIGINAL_LOCK
        and a['resolved_plan_sha256']==resolved['plan_sha256']and a['unit']==contract['unit']
        and a['source_contract']==resolved['plan']['source_contract']and a['sdk_operator_recovery']==r.completion.BINDING
        and a['latest_completion']==resolved['plan']['latest_completion']
        and a['completed_sdk_layout']==resolved['plan']['layer1_operator_boundary']
        and a['serial_predecessors']==resolved['plan']['serial_predecessors']
        and a['source_authentication']==refs['source_authentication']and a['observed_drains']==refs['drains']
        and a['runtime_admitted']is a['predecessors_checked']is True,'Exact complete original admitted lineage')
    require(a['current_execution']['intent']==refs['intent'],'Original exact launch intent bytes')
    r.validate_initial_descriptor(a);r.execution.validate_intent(records['intent'],a)
    release=a['outer_release'];require(release['reviewer']=='/root'and release['challenge']['owner']==live['owner']
        and release['challenge']['spec']==spec['model_spec']and release['challenge']['review']==spec['model_review']
        and release['tool']==a['current_execution']['tool']==live['actual_execution_handle'],'Root challenge binds actual terminal model session')
    co=records.get('coordinator_owner');native=records.get('native_owner');owners=[live['owner']]
    if co is not None:r.life.coordinator_process(contract,refs['admission'],co);owners.append(co)
    if native is not None:
        require(co is not None,'Native owner requires actual coordinator acknowledgment')
        r.life.process(native,r.life.command(contract,'native',refs['admission']),co['pid'],co['cgroup']);owners.append(native)
    for owner in owners:r.p.gone(owner)
    guard.writers=dict(outer=live['owner']['pid'])
    if co is not None:guard.writers['coordinator']=co['pid']
    if native is not None:guard.writers['native']=native['pid']
    guard.successful=successful;guard.bind(contract,resolved)
    freshness=terminal_handoff(resolved,a,records,refs,metadata,lease,guard,successful)
    partition=r.sources.classify(resolved['plan']['source_sha256'],resolved['plan']['source_contract'],metadata.read)
    inherited=r.sources.validate_native_receipt(records['source_authentication'],partition,live['owner'],r.ORIGINAL_LOCK,a['attempt_id'])
    pins_plan,source_proof=audit.source_binding(resolved['plan'],spec['source_extension'],spec['model_spec'],spec['model_review'],model['overlay'],SELF)
    extended=r.sources.classify(pins_plan['source_sha256'],pins_plan['source_contract'],metadata.read)
    r.sources.verify_text_sources(extended,guard);r.sources.check_native_identities(records['source_authentication'],partition,guard)
    r.verify_loaded(pins_plan,guard)
    saved=records['drains'];pair=r.completion.audit.unpack(saved['paired_physical']);old=resolved['current_scope']
    require(saved['schema']=='minicpm5_native_original_predecessor_pair_v1'and saved['resolved_plan_sha256']==resolved['plan_sha256']
        and saved['outer_owner']==live['owner']and saved['lock_identity']==r.ORIGINAL_LOCK
        and saved['physical_pair_schema']==r.completion.physical.PAIR_SCHEMA
        and r.digest(pair)==saved['complete_observation_sha256'],'Original complete saved predecessor pair')
    r.completion.physical.validate_pair(pair,{row['unit']for row in old['nemotron']['history']['rows']},old['owners'],r.ORIGINAL_LOCK,known_units=old['complete_known_units'])
    units=set(old['complete_known_units'])|{contract['unit']}
    for owner in owners:
        group=owner.get('cgroup')
        if type(group)is str and group.startswith('0::'+r.p.GROUP_PREFIX):units.add(r.completion.physical.group_unit(group[3:]))
    scope=dict(old,owners=copy.deepcopy(old['owners'])+owners,complete_known_units=sorted(units));lease.scope=scope
    authority=r.current_authority.Authority(r,lease)
    owned=outer.OwnedService(contract,refs['admission'],a,metadata);owned.attempted=True;owned.owner=co
    drain=owned.verify_drained();current=audit.pair(scope,guard,authority)
    recorder=lease.admission['outer_owner'];require(r.p.proc(os.getpid())==recorder,'Actual current recorder identity under original retained descriptor')
    ended=dict(schema='minicpm5_native_model_actual_exit_v1'if successful else'minicpm5_native_model_failed_actual_exit_v1',
        reviewer='/root',actual_tool=copy.deepcopy(tool),actual_exit_observed=True,actual_exit_code=tool['exit_code'],
        exact_owner_gone=True,self_exit_claimed=False,live=spec['live'],spec=spec['model_spec'],review=spec['model_review'],
        owner=live['owner'],coordinator=co,native=native,recorder=copy.deepcopy(recorder),outputs=refs,
        resolved_plan_sha256=resolved['plan_sha256'],lock_identity=r.ORIGINAL_LOCK,unit=contract['unit'],
        recorder_spec=spec_ref,recorder_review=review_ref,recorder_source=dict(path=SELF,sha256=pins_plan['source_sha256'][SELF]),
        source_binding=source_proof,
        original_sdk_model_exit_code=1,actual_recorder_tool_exit_verified=False,independent_raw_audit_pending=True,
        inherited_native_source_authentication=inherited,model_drain=drain,serial_freshness=freshness,
        current_physical_pair=r.completion.audit.packed_physical(current),physical_pair_schema=r.completion.physical.PAIR_SCHEMA,
        full_model_qualified=False,new_speedup_claim=False,raw_payload_bytes_read=0,native_binary_bytes_read=0)
    if successful:core.validate_model(ended,live,records,resolved)
    else:ended.update(original_nonzero_exit_preserved=True,missing_runtime_receipts=missing,successful_model_terminal=False)
    require(owned.verify_drained()==drain,'Exact full model service drain stable across terminal recording')
    refs_after,records_after,missing_after=read_receipts(contract,metadata,guard,successful)
    require((refs_after,records_after,missing_after)==(refs,records,missing),'All original present and absent runtime receipts unchanged')
    r.sources.verify_text_sources(extended,guard);r.sources.check_native_identities(records['source_authentication'],partition,guard)
    r.verify_loaded(pins_plan,guard);metadata.unchanged();guard(force=True);ended['storage']=guard.storage
    return ended


def run(spec_ref,review_ref):
    require(sys.dont_write_bytecode and __debug__,'Original -B/assertions required')
    r.p.user_bus();fd,owner,first=support.acquire()
    lease=r.Lease(dict(outer_owner=owner,lock_identity=r.ORIGINAL_LOCK,retained_fd=fd,lock_policy=r.LOCK_POLICY,
        initial_descriptor_proof=first,unit='trn1-mini-layer1-operator-native-model-exit-recorder'))
    try:
        os.sched_setaffinity(0,{LIMITS['cpu']});os.nice(LIMITS['nice'])
        resource.setrlimit(resource.RLIMIT_AS,(LIMITS['address_space_bytes'],)*2)
        resource.setrlimit(resource.RLIMIT_CPU,(LIMITS['cpu_seconds'],LIMITS['hard_cpu_seconds']))
        resource.setrlimit(resource.RLIMIT_FSIZE,(LIMITS['output_file_bytes'],)*2)
        signal.signal(signal.SIGALRM,lambda *args:(_ for _ in ()).throw(TimeoutError('Original bounded metadata recorder wall time')))
        signal.alarm(LIMITS['wall_seconds']);value=record(spec_ref,review_ref,lease)
        spec=r.sources.load(spec_ref);target=Path(spec['output'])
        raw=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
        require(len(raw)<LIMITS['output_file_bytes'],'Original independent metadata result bound')
        r.atomic_once(target,value);ref=outer.reference(target);print(json.dumps(ref),flush=True);return ref
    finally:
        signal.alarm(0)
        support.release_after_physical_drain(fd,dict(physically_drained=True,no_recorder_children_or_services_started=True))


if __name__=='__main__':
    cli=argparse.ArgumentParser()
    for key in ('spec','review'):
        cli.add_argument('--'+key,required=True);cli.add_argument('--'+key+'-sha256',required=True)
    args=cli.parse_args();run(dict(path=args.spec,sha256=args.spec_sha256),dict(path=args.review,sha256=args.review_sha256))
