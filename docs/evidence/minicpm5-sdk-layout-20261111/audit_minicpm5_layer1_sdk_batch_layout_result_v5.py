"""Independent paired-history SDK batch/layout audit v2. No automatic execution.

Actual tool exit receipt must exist before entry. All native/model processes
must be gone, original lock owned, source authentication inherited and every
saved output reread. Full fresh physical evidence is losslessly retained in
compressed JSON, under the original 2MiB individual audit-file bound.
"""
import argparse
import base64
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import sys
import zlib
ROOT=Path(__file__).resolve().parents[1];Q=ROOT/'research'
sys.path.insert(0,str(ROOT))
from research import minicpm5_layer1_sdk_batch_layout_post_exit_core_v3 as core
from research import minicpm5_layer1_sdk_batch_layout_sources_v2 as sources
from minicpm5_layer0_runtime_auth_v1 import identity
from research import minicpm5_sdk_physical_evidence_v3 as physical_evidence
from research import minicpm5_sdk_post_exit_source_supplement_v1 as supplement
require=core.require
LIMITS=dict(address_space_bytes=2*2**30,cpu_seconds=600,hard_cpu_seconds=605,
    output_file_bytes=2*2**20,wall_seconds=900,cpu=6,nice=19)


def output_path(path):
    path=Path(path)
    require(path.parent==Q and re.fullmatch(r'minicpm5-layer1-sdk-batch-layout-[a-z0-9-]+-post-exit-audit-[a-z0-9-]+\.json',path.name),
        'Fresh named independent audit result')
    require(not path.exists(),'Never overwrite an actual audit result')
    return path


def packed_physical(proof):
    raw=json.dumps(proof,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    require(len(raw)<=32*2**20,'Bounded complete historical physical observation')
    compressed=zlib.compress(raw,9)
    require(len(compressed)<2*2**20,'Compressed complete pair within original file bound')
    encoded=base64.b64encode(compressed).decode('ascii')
    require(len(encoded)<2*2**20,'Base64 complete pair within original file bound')
    require(zlib.decompress(compressed)==raw,'Lossless complete physical evidence retention')
    return dict(encoding='zlib-base64-json-v1',uncompressed_bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
        compressed_bytes=len(compressed),data=encoded,all_samples_retained=True)


def verify_outer_drain(records,done,owner,lock):
    a=records['admission'];d=records['drains'];f=d['latest_physical']
    require(d['complete']is True and d['latest_completion']==done['binding']==a['latest_completion']
        and d['outer_owner']==owner and d['lock_identity']==lock==done['lock_identity']
        and f['complete']is f['physical_checks_performed']is f['all_current_and_prior_services_preserved']is True
        and f['lock_identity']==lock and f['all_additional_owners_checked']==done['owners']
        and f['nemotron']['complete']is f['nemotron']['fresh_physical_drain_verified']is True
        and f['nemotron']['services']==1308,'Complete saved outer predecessor drain binds all actual owners')
    require(type(d['complete_observation_sha256'])is str and re.fullmatch('[a-f0-9]{64}',d['complete_observation_sha256']),
        'Retained digest of complete original outer observation')
    return dict(saved_outer_observation_sha256=d['complete_observation_sha256'],all_recorded_owners_retained=True)


def legacy_drain_units(value):
    """Extract every named unit from complete nested saved/fresh observations."""
    units=set()
    def visit(row):
        if type(row)is dict:
            if 'samples'in row:
                samples=row['samples']
                require(type(samples)is list and len(samples)==2 and samples[0]==samples[1],
                    'Every retained original double physical observation remains complete and stable')
                for sample in samples:
                    require(type(sample)is dict and type(sample.get('properties'))is dict
                        and (type(sample.get('group'))is dict or type(sample.get('groups'))is list and sample['groups']),
                        'Both service samples retain complete property and group evidence')
            if type(row.get('unit'))is str and row['unit'].startswith('trn1-'):
                units.add(row['unit'].removesuffix('.service'))
            if type(row.get('group'))is str:
                # Binding-only extraction from exact original owned group path.
                # The saved legacy tree is not promoted to strict new semantics.
                units.add(physical_evidence.group_unit(row['group']))
            props=row.get('properties')
            if type(props)is dict and type(props.get('Id'))is str and props['Id'].startswith('trn1-'):
                require(props['MainPID']==props['ControlPID']=='0'
                    and (props['ActiveState']in ('inactive','failed')or
                        (props['ActiveState'],props['SubState'])==('active','exited')),
                    'Every saved/fresh retained unit observation is terminal')
                units.add(props['Id'].removesuffix('.service'))
            for child in row.values():visit(child)
        elif type(row)is list:
            for child in row:visit(child)
    visit(value);return units


def drain_units(value):
    """New decoded paired envelope only; legacy lookup must be explicit."""
    return physical_evidence.validate_pair(value)


GUARD_BOUNDARIES=('before_source_verification','after_source_verification','completed_metadata_authenticated',
    'before_native_import','before_model_replay','after_model_replay','after_worker','before_rank_output')


def verify_rank_guards(records):
    result=[]
    for rank in (0,1):
        g=records['rank%d_result'%rank]['guard_checks'];events=g['events']
        require(g['schema']=='minicpm5_layer1_sdk_batch_layout_guard_checks_v2'
            and type(g['fast_checks'])is int and type(g['full_checks'])is int
            and g['fast_checks']>=g['full_checks']==len(events)>=len(GUARD_BOUNDARIES)
            and type(g['period_seconds'])in (int,float) and g['period_seconds']==1.0
            and g['every_source_fully_authenticated']is g['full_source_checks_not_sampled']
            is g['every_call_keeps_original_serial_lock_process_and_performance_checks']is True,
            'Actual complete source verification and continuous original fast checks')
        require(tuple(e['boundary']for e in events if e['boundary']!='periodic')==GUARD_BOUNDARIES,
            'All mandatory full guard boundaries occur once and in order')
        times=[]
        for event in events:
            require(type(event['monotonic'])in (int,float)and math.isfinite(event['monotonic'])and event['monotonic']>=0
                and type(event['owners'])is int and event['owners']>0
                and type(event['metadata'])is int and event['metadata']>=0,'Actual bounded full guard observation')
            times.append(event['monotonic'])
        require(times==sorted(times),'Original full guard event order')
        require(len({e['owners']for e in events})==1,'Full historical owner set never reduced')
        authenticated=next(i for i,e in enumerate(events)if e['boundary']=='completed_metadata_authenticated')
        require(events[authenticated]['metadata']>0 and all(e['metadata']==events[authenticated]['metadata']for e in events[authenticated:]),
            'Complete authenticated metadata watch retained through rank output')
        result.append(dict(rank=rank,fast_checks=g['fast_checks'],full_checks=g['full_checks'],
            mandatory_boundaries=list(GUARD_BOUNDARIES),source_verification_complete=True))
    return result


def audit(runner,exit_ref,owner,lock,supplement_ref):
    """Requires caller-owned original serial lock; tests inject only doubles."""
    p=runner.p;observed={}
    def lease():p.lock_proof(owner,lock);p.clear_window()
    def load(ref):
        lease();sources.reference(ref);before=identity(ref['path']);value=sources.load(ref)
        require(identity(ref['path'])==before,'Independent metadata read remains stable')
        observed[ref['path']]=before;return value
    lease();ended=load(exit_ref)
    require(ended['actual_exit_observed']is True and type(ended['actual_exit_code'])is int
        and ended['actual_exit_code']==0 and ended['lock_identity']==lock,'Actual model terminal and same independent lock')
    live=load(ended['live'])
    for old in [ended['owner'],ended['coordinator'],*ended['ranks']]:p.gone(old)
    p.active_heavy()
    spec,plan=supplement.read_bundle(runner,ended['spec'],ended['review'],lease,supplement_ref,exit_ref);review=load(ended['review'])
    records={key:load(ref)for key,ref in ended['outputs'].items()}
    lineage=core.validate(ended,live,spec,plan,review,records);admission=records['admission']
    guard_evidence=verify_rank_guards(records)
    def guard():
        lease()
        for old in lineage['owners']:p.gone(old)
        p.empty_group(lineage['group'])
    guard()
    model_drain=runner.verify_owned_drain(spec,ended['outputs']['admission'],admission['attempt_id'],ended['coordinator'],True)
    require(model_drain['physically_drained']is True,'Actual current model service drained')
    partition=sources.classify(spec['source_sha256'],spec['source_contract'])
    native=records['source_authentication']
    inherited=sources.validate_native_receipt(native,partition,ended['owner'],lock,admission['attempt_id'])
    sources.check_native_identities(native,partition,guard)
    done,read_json=supplement.admitted_bundles(runner,plan,guard,supplement_ref,exit_ref)
    require(done['lock_identity']==lock,'Every predecessor uses same original serial lock')
    require(all(e['owners']==len(done['owners'])for rank in (0,1)
        for e in records['rank%d_result'%rank]['guard_checks']['events']),
        'Every rank full check covered the entire actual historical owner set')
    saved=verify_outer_drain(records,done,ended['owner'],lock)
    # All original services and double samples are rechecked under this audit's
    # own lock. No property/name from a future or replacement service is exempt.
    # Preserve the complete legacy tree and its limitations. New strict proof
    # comes only from two original physical passes plus exact missing fields.
    legacy_units=legacy_drain_units(records['drains']['latest_physical'])
    saved_scan=physical_evidence.inspect_tree(records['drains']['latest_physical'])
    saved_units=saved_scan['units']
    require(legacy_units <= saved_units, 'Every original legacy name retained in complete owner-cgroup scope')
    saved=dict(saved,legacy_named_unit_count=len(legacy_units),complete_owner_cgroup_unit_count=len(saved_units),
        additional_owner_cgroup_units=sorted(saved_units-legacy_units),
        owner_only_historical_invocation_unavailable=sorted(name for name,owners in saved_scan['owner_only'].items()
            if not any('unit' in old for old in owners)),
        historical_invocation_match_claimed_for_unanchored_units=False)
    actual=physical_evidence.collect_pair(done,guard,runner.latest.physical,p.properties,p.empty_group,known_units=saved_units)
    required_units={row['unit']for row in done['nemotron']['history']['rows']}
    fresh_units=physical_evidence.validate_pair(actual,required_units,done['owners'],lock,known_units=saved_units)
    require(len(required_units)==1308 and required_units<=legacy_units<=saved_units==fresh_units,
        'Every original historical and supplemental unit retained in legacy and new paired evidence')
    encoded=packed_physical(actual);del actual
    def pair(results,exact_plan,exact_guard):
        require(exact_plan is plan and exact_guard is guard,'Original independent complete verifier arguments')
        return runner.layer1_results.validate_rank_pair(results,plan,done['collector'],done['native'],
            done['current_history'],read_json,guard)
    proof=core.recompute(records,plan,pair,guard)
    require(not {'torch','numpy','torch_neuronx','neuronxcc'}&set(sys.modules),'Independent stdlib audit only')
    sources.verify_text_sources(partition,guard)
    sources.check_native_identities(native,partition,guard)
    loaded=supplement.verify_loaded_sources(runner,plan,guard,supplement_ref,exit_ref)
    require(all(identity(path)==old for path,old in observed.items()),'All complete model receipt metadata remains stable')
    require({n:p.sha(ROOT/n)for n in spec['canonical_sha256']}==spec['canonical_sha256'],
        'Original canonical scientific state unchanged')
    runner.usage(spec['output_root'],spec['root_floor_bytes']);p.active_heavy();guard()
    return dict(schema='minicpm5_layer1_sdk_batch_layout_post_exit_audit_v2',complete=True,reviewer='/root',
        observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=runner.reference(__file__),
        model_exit=exit_ref,post_exit_source_supplement=supplement_ref,owner=owner,lock_identity=lock,result=proof,descriptive=proof['descriptive'],
        actual_model_drain=model_drain,actual_predecessor_drain=encoded,saved_outer_drain=saved,rank_guard_verification=guard_evidence,
        actual_predecessor_pair_schema=physical_evidence.PAIR_SCHEMA,
        full_predecessor_passes=2,strict_paired_physical_semantics_verified=True,
        legacy_saved_physical_semantics_not_requalified=True,
        source_authentication=ended['outputs']['source_authentication'],inherited_native_authentication=inherited,
        loaded_source_closure=loaded,every_raw_byte_reread_after_all_model_processes_exited=True,
        all_previous_controls_full_bytes_exact=True,every_view_formula_reconstructed=True,
        rope_payload_independently_reread=False,other_arms_reexecuted=False,attention_causality_proven=False,
        full_model_qualified=False,new_speedup_claim=False,self_exit_claimed=False)


def run(exit_ref,target,supplement_ref):
    sources.reference(supplement_ref)
    sources.reference(exit_ref);require(Path(exit_ref['path']).suffix=='.json','Actual model JSON exit reference')
    target=output_path(target)
    import minicpm5_layer1_sdk_batch_layout_cpu_runner_v4 as runner
    require(sys.dont_write_bytecode and __debug__,'Original python -B required')
    runner.p.user_bus();fd,owner,lock=runner.post.external.acquire_lock()
    try:
        os.sched_setaffinity(0,{LIMITS['cpu']});os.nice(LIMITS['nice'])
        resource.setrlimit(resource.RLIMIT_AS,(LIMITS['address_space_bytes'],)*2)
        resource.setrlimit(resource.RLIMIT_CPU,(LIMITS['cpu_seconds'],LIMITS['hard_cpu_seconds']))
        resource.setrlimit(resource.RLIMIT_FSIZE,(LIMITS['output_file_bytes'],)*2)
        signal.signal(signal.SIGALRM,lambda *args:(_ for _ in ()).throw(TimeoutError('Original independent audit bound')))
        signal.alarm(LIMITS['wall_seconds'])
        result=audit(runner,exit_ref,owner,lock,supplement_ref)
        raw=(json.dumps(result,indent=2,allow_nan=False)+'\n').encode()
        require(len(raw)<LIMITS['output_file_bytes'],'Complete independent audit within original2MiB bound')
        runner.p.once(target,result);print(json.dumps(runner.reference(target)))
    finally:
        signal.alarm(0);runner.post.external.release_lock(fd)


if __name__=='__main__':
    cli=argparse.ArgumentParser();cli.add_argument('--model-exit',required=True);cli.add_argument('--model-exit-sha256',required=True)
    cli.add_argument('--source-supplement',required=True);cli.add_argument('--source-supplement-sha256',required=True)
    cli.add_argument('--output',required=True);args=cli.parse_args()
    run(dict(path=args.model_exit,sha256=args.model_exit_sha256),args.output,
        dict(path=args.source_supplement,sha256=args.source_supplement_sha256))
