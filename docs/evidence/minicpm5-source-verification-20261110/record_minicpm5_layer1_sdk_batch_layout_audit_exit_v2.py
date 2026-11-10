"""External actual audit exit recorder with strict full historical evidence.

Requires a root-observed actual successful tool session/chunk after terminal,
then independently acquires the original lock, checks the recorded auditor is
gone, authenticates its result and losslessly decodes complete physical proof.
This metadata review does not repeat or claim a new tensor-byte experiment.
"""
import argparse
import base64
import datetime
import hashlib
import json
from pathlib import Path
import re
import sys
import zlib
ROOT=Path(__file__).resolve().parents[1];Q=ROOT/'research';sys.path[:0]=[str(ROOT),str(Q)]
from research import minicpm5_layer1_sdk_batch_layout_sources_v2 as sources
from research import minicpm5_layer1_sdk_batch_layout_post_exit_core_v1 as core
from research import audit_minicpm5_layer1_sdk_batch_layout_result_v2 as auditor
from research import minicpm5_sdk_physical_evidence_v2 as physical
from research.record_minicpm5_layer1_sdk_batch_layout_exit_v1 import actual_tool
from minicpm5_layer0_runtime_auth_v1 import identity
require=core.require
AUDITOR=Q/'audit_minicpm5_layer1_sdk_batch_layout_result_v2.py'


def unpack(proof):
    require(type(proof)is dict and set(proof)=={'encoding','uncompressed_bytes','sha256','compressed_bytes','data','all_samples_retained'}
        and proof['encoding']=='zlib-base64-json-v1' and proof['all_samples_retained']is True,
        'Exact complete physical evidence envelope')
    require(type(proof['uncompressed_bytes'])is int and 0<proof['uncompressed_bytes']<=32*2**20
        and type(proof['compressed_bytes'])is int and 0<proof['compressed_bytes']<2*2**20
        and type(proof['data'])is str and len(proof['data'])<2*2**20
        and type(proof['sha256'])is str and re.fullmatch('[a-f0-9]{64}',proof['sha256']),
        'Original bounded full physical metadata')
    encoded=base64.b64decode(proof['data'],validate=True)
    require(len(encoded)==proof['compressed_bytes'],'Exact stored compressed bytes')
    decoder=zlib.decompressobj();raw=decoder.decompress(encoded,proof['uncompressed_bytes']+1)
    require(len(raw)==proof['uncompressed_bytes']and decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail,
        'One complete stream, exact original length, no trailing or excess data')
    require(hashlib.sha256(raw).hexdigest()==proof['sha256'],'Complete original physical JSON digest')
    return json.loads(raw)


def validate(live,result,model,session,chunk,exit_code,result_ref,lock):
    handle=actual_tool(session,chunk,exit_code)
    require(live['schema']=='minicpm5_sdk_batch_layout_audit_live_v1' and live['reviewer']=='/root'
        and live['actual_identity_verified']is True and live['actual_execution_handle']==handle,
        'Exact root-observed auditor live-to-terminal tool session')
    owner=live['owner'];core.same_owner(owner,result['owner'])
    model_ref=live['model_actual_exit'];sources.reference(model_ref);sources.reference(result_ref)
    expected=['python3','-B',str(AUDITOR),'--model-exit',model_ref['path'],'--model-exit-sha256',model_ref['sha256'],
        '--output',result_ref['path']]
    require(owner['argv']==expected,'Exact actual independent auditor command')
    require(result['schema']=='minicpm5_layer1_sdk_batch_layout_post_exit_audit_v2' and result['reviewer']=='/root'
        and result['complete']is True and result['self_exit_claimed']is False
        and result['model_exit']==model_ref and result['lock_identity']==lock==model['lock_identity']
        and result['source']['path']==str(AUDITOR),'Same original model, actual auditor, lock and source')
    require(model['schema']=='minicpm5_layer1_sdk_batch_layout_actual_exit_v1'
        and model['actual_exit_observed']is True and type(model['actual_exit_code'])is int and model['actual_exit_code']==0
        and model['actual_execution_handle']['session_id']!=session,'Separate actually completed model tool session')
    for row in (result,model):
        require(row['full_model_qualified']is row['new_speedup_claim']is False,'No local diagnostic promotion')
    require(result['every_raw_byte_reread_after_all_model_processes_exited']is result['all_previous_controls_full_bytes_exact']
        is result['every_view_formula_reconstructed']is True and result['rope_payload_independently_reread']
        is result['other_arms_reexecuted']is result['attention_causality_proven']is False,
        'Complete audit guarantees retain all original limits')
    require(result['actual_predecessor_pair_schema']==physical.PAIR_SCHEMA and result['full_predecessor_passes']==2
        and result['strict_paired_physical_semantics_verified']is True
        and result['legacy_saved_physical_semantics_not_requalified']is True,'Strict paired successor evidence and explicit legacy limitations')
    proof=result['result']
    require(proof['schema']=='minicpm5_layer1_sdk_batch_layout_pair_verification_v1' and proof['complete']is True
        and [proof[k]for k in ('ranks','anchor_decode_calls','physical_attention_invocations','assembled_four_row_outputs',
            'unique_decode_frames','raw_output_bytes')]==[2,12,144,72,24,1179648]
        and proof['descriptive']==result['descriptive'],'Exact complete original output audit and descriptive result')
    require(proof['every_output_independently_reread']is proof['all_previous_controls_full_bytes_exact']
        is proof['every_view_formula_reconstructed']is proof['all_other_arms_verified_not_reexecuted']is True
        and proof['other_arms_reexecuted']is proof['rope_payload_independently_reread']
        is proof['full_model_qualified']is proof['new_speedup_claim']is False,'Original full verifier guarantees')
    require(result['source_authentication']==model['outputs']['source_authentication']
        and result['inherited_native_authentication']==dict(native_hash_authentication_inherited=True,
            native_bytes_read_here=0,historical_payload_bytes_read_here=0),'Exact inherited outer source authentication')
    pair=unpack(result['actual_predecessor_drain']);units=physical.validate_pair(pair,lock=lock)
    actual=pair['first']['tree']
    require(actual['complete']is actual['physical_checks_performed']is actual['all_current_and_prior_services_preserved']is True
        and actual['lock_identity']==lock and actual['nemotron']['complete']is actual['nemotron']['fresh_physical_drain_verified']is True
        and actual['nemotron']['services']==1308 and len(units)>=1308,
        'Complete original historical drain remains available for independent readback')
    return dict(owner=owner,handle=handle,full_predecessor_observation_verified=True,
        historical_unit_count=len(units),physical_metadata_sha256=result['actual_predecessor_drain']['sha256'],units=units,physical=actual,pair=pair)


def record(runner,live_ref,result_ref,session,chunk,exit_code,recorder,lock):
    actual_tool(session,chunk,exit_code);p=runner.p;seen={}
    def guard():p.lock_proof(recorder,lock);p.clear_window()
    def load(ref):
        guard();before=identity(ref['path']);value=sources.load(ref)
        require(identity(ref['path'])==before,'Stable complete audit terminal metadata')
        seen[ref['path']]=before;return value
    live=load(live_ref);result=load(result_ref);model=load(live['model_actual_exit'])
    checked=validate(live,result,model,session,chunk,exit_code,result_ref,lock)
    p.gone(checked['owner']);p.active_heavy();guard()
    require(hashlib.sha256(sources.text(str(AUDITOR),result['source']['sha256'])).hexdigest()==result['source']['sha256'],
        'Exact executed independent auditor source retained')
    for owner in [model['owner'],model['coordinator'],*model['ranks']]:p.gone(owner)
    spec,plan=runner.read_bundle(model['spec'],model['review'],guard)
    review=load(model['review']);model_live=load(model['live'])
    records={key:load(ref)for key,ref in model['outputs'].items()}
    core.validate(model,model_live,spec,plan,review,records)
    a=records['admission']
    owned=runner.verify_owned_drain(spec,model['outputs']['admission'],a['attempt_id'],model['coordinator'],True)
    require(owned['physically_drained']is True,'Current model service actually drained')
    done,_=runner.admitted_bundles(plan,guard)
    auditor.verify_outer_drain(records,done,model['owner'],lock)
    required={row['unit']for row in done['nemotron']['history']['rows']}
    saved=auditor.legacy_drain_units(records['drains']['latest_physical'])
    physical.validate_pair(checked['pair'],required_units=required,expected_owners=done['owners'],lock=lock)
    require(len(required)==1308 and required<=saved==checked['units']
        and checked['physical']['all_additional_owners_checked']==done['owners'],
        'Exact full saved/actual unit names and historical owners, never count-only')
    partition=sources.classify(spec['source_sha256'],spec['source_contract'])
    sources.validate_native_receipt(records['source_authentication'],partition,model['owner'],lock,a['attempt_id'])
    sources.check_native_identities(records['source_authentication'],partition,guard)
    runner.verify_loaded_sources(plan,guard)
    require(all(identity(path)==old for path,old in seen.items()),'All actual terminal evidence remains unchanged')
    p.active_heavy();guard()
    return dict(schema='minicpm5_layer1_sdk_batch_layout_post_exit_audit_actual_exit_v2',reviewer='/root',
        observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),live=live_ref,result=result_ref,
        model_exit=live['model_actual_exit'],owner=checked['owner'],actual_tool_exit_observed=True,
        actual_execution_handle=checked['handle'],actual_exit_code=exit_code,tool_chunk=chunk,
        exact_owner_gone=True,serial_lock_reacquired=True,lock_identity=lock,
        complete_audit_metadata_independently_verified=True,full_physical_metadata_decoded=True,
        decoded_historical_unit_count=checked['historical_unit_count'],
        physical_metadata_sha256=checked['physical_metadata_sha256'],raw_tensor_bytes_reread_here=False,
        full_model_qualified=False,new_speedup_claim=False)


def run(live_ref,result_ref,session,chunk,exit_code,target):
    actual_tool(session,chunk,exit_code);sources.reference(live_ref);sources.reference(result_ref);target=Path(target)
    require(target.parent==Q and re.fullmatch(r'minicpm5-layer1-sdk-batch-layout-[a-z0-9-]+-audit-actual-exit-[a-z0-9-]+\.json',target.name)
        and not target.exists(),'Fresh independent auditor actual-exit receipt')
    import minicpm5_layer1_sdk_batch_layout_cpu_runner_v2 as runner
    require(sys.dont_write_bytecode and __debug__,'Original python -B')
    runner.p.user_bus();fd,recorder,lock=runner.post.external.acquire_lock()
    try:
        value=record(runner,live_ref,result_ref,session,chunk,exit_code,recorder,lock)
        runner.p.once(target,value);print(json.dumps(runner.reference(target)))
    finally:runner.post.external.release_lock(fd)


if __name__=='__main__':
    cli=argparse.ArgumentParser()
    for name in ('live','live-sha256','result','result-sha256','chunk','output'):cli.add_argument('--'+name,required=True)
    cli.add_argument('--session',type=int,required=True);cli.add_argument('--exit-code',type=int,required=True);a=cli.parse_args()
    run(dict(path=a.live,sha256=a.live_sha256),dict(path=a.result,sha256=a.result_sha256),a.session,a.chunk,a.exit_code,a.output)
