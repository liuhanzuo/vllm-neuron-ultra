"""Record independently observed actual model tool exit; never infer from files.

The caller supplies its actual tool session/chunk/exit code after the tool has
returned terminal. Live receipt must name the same session. This executable
then reacquires the original lock and verifies model service/owner drainage.
"""
import argparse
import datetime
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1];Q=ROOT/'research';sys.path.insert(0,str(ROOT))
from research import minicpm5_layer1_sdk_batch_layout_post_exit_core_v1 as core
from research import minicpm5_layer1_sdk_batch_layout_sources_v2 as sources
from research.audit_minicpm5_layer1_sdk_batch_layout_result_v1 import verify_outer_drain
require=core.require


def actual_tool(session,chunk,exit_code):
    require(type(session)is int and session>0 and type(chunk)is str and re.fullmatch('[a-f0-9]{6,64}',chunk)
        and type(exit_code)is int and exit_code==0,'Externally observed successful tool session/chunk/code required')
    return dict(kind='tool_session',session_id=session)


def record(runner,live_ref,session,chunk,exit_code,recorder,lock):
    handle=actual_tool(session,chunk,exit_code);p=runner.p
    def guard():p.lock_proof(recorder,lock);p.clear_window()
    guard();live=sources.load(live_ref)
    require(live['schema']==core.SCHEMA+'_actual_live_v1' and live['reviewer']=='/root'
        and live['actual_identity_verified']is True and live['actual_execution_handle']==handle,
        'Exact actually observed live-to-terminal tool session')
    p.gone(live['owner']);p.active_heavy()
    spec,plan=runner.read_bundle(live['spec'],live['review'],guard);review=sources.load(live['review'])
    root=Path(spec['output_root']);refs={key:runner.reference(root/name)for key,name in core.NAMES.items()}
    records={key:sources.load(ref)for key,ref in refs.items()};admission=records['admission']
    co=records['coordinator_owner'];ranks=[records['rank0_owner'],records['rank1_owner']]
    require(admission['lock_identity']==lock and admission['outer_owner']==live['owner'],'Original outer serial ownership')
    for owner in [co,*ranks]:p.gone(owner)
    physical=runner.verify_owned_drain(spec,refs['admission'],admission['attempt_id'],co,True)
    require(physical['physically_drained']is True,'Actual complete owned model service drain')
    ended=dict(schema=core.SCHEMA+'_actual_exit_v1',reviewer='/root',
        observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),actual_exit_observed=True,
        actual_exit_code=exit_code,actual_execution_handle=handle,tool_chunk=chunk,live=live_ref,
        spec=live['spec'],review=live['review'],owner=live['owner'],coordinator=co,ranks=ranks,outputs=refs,
        lock_identity=lock,all_four_processes_gone=True,cgroup_empty=True,physical_drain=physical,
        independent_post_exit_raw_reread_pending=True,full_model_qualified=False,new_speedup_claim=False)
    core.validate(ended,live,spec,plan,review,records)
    partition=sources.classify(spec['source_sha256'],spec['source_contract'])
    sources.validate_native_receipt(records['source_authentication'],partition,live['owner'],lock,admission['attempt_id'])
    sources.check_native_identities(records['source_authentication'],partition,guard)
    done,_=runner.admitted_bundles(plan,guard);verify_outer_drain(records,done,live['owner'],lock)
    runner.verify_loaded_sources(plan,guard);guard();p.active_heavy()
    return ended


def run(live_ref,session,chunk,exit_code,target):
    actual_tool(session,chunk,exit_code);sources.reference(live_ref)
    target=Path(target)
    require(target.parent==Q and re.fullmatch(r'minicpm5-layer1-sdk-batch-layout-[a-z0-9-]+-actual-exit-[a-z0-9-]+\.json',target.name)
        and not target.exists(),'Fresh actual model terminal receipt namespace')
    import minicpm5_layer1_sdk_batch_layout_cpu_runner_v2 as runner
    require(sys.dont_write_bytecode and __debug__,'Original python -B')
    runner.p.user_bus();fd,recorder,lock=runner.post.external.acquire_lock()
    try:
        value=record(runner,live_ref,session,chunk,exit_code,recorder,lock)
        runner.p.once(target,value);print(json.dumps(runner.reference(target)))
    finally:runner.post.external.release_lock(fd)


if __name__=='__main__':
    cli=argparse.ArgumentParser()
    for name in ('live','live-sha256','chunk','output'):cli.add_argument('--'+name,required=True)
    cli.add_argument('--session',type=int,required=True);cli.add_argument('--exit-code',type=int,required=True)
    a=cli.parse_args();run(dict(path=a.live,sha256=a.live_sha256),a.session,a.chunk,a.exit_code,a.output)
