"""Root serial runner for finite saved prefill analysis; no model launches."""
import argparse
import json
import os
from pathlib import Path
import re
import time
import minicpm5_original32_mask_device_runner_v2 as previous
import minicpm5_original88_prefill_attention_analysis_v1 as analysis

p, five, same, post, ROOT = previous.p, previous.five, previous.same, previous.post, previous.ROOT
SELF = Path(__file__).resolve()
SPEC = previous.masks.ref('minicpm5-original32-mask-device-execution-spec-20261107-v2.json', '50d1b9002fde217908e13fcabf80909ab66215f2fcae8faef89916d40127ada6')
R2_EXIT = previous.masks.ref('minicpm5-original32-mask-device-root-exit-r2-20261107-v1.json', 'c044c8105ae85d7ae5d4606cad491d85e609c8ec6268c65fe999c4574540a847')
R2_REVIEW = previous.masks.ref('minicpm5-original32-mask-device-independent-result-review-20261107-v1.json', '904b66da76ca18b222366b859fe87cf242ea84c1ae512035b8e1c5d9489c9612')
PREVIOUS_SHA = '953d57012fd1ad56c5a6cb74c64846cecd5794a2b1d6e8a2d66d695a073268b9'

def r2_predecessors():
    ended = p.read_ref(R2_EXIT); live = p.read_ref(ended['root_live'])
    review = p.read_ref(R2_REVIEW)
    p.require(review['passed'] is True, 'Independent device32 result review')
    p.require(ended['reviewer'] == '/root' and ended['actual_exit_observed'] is True and type(ended['actual_exit_code']) is int and ended['actual_exit_code'] == 0, 'Actual device r2 exit0')
    p.require(live['reviewer'] == '/root' and live['actual_identity_verified'] is True, 'Actual device r2 identity')
    same(ended['owner'], live['owner'], 'Device r2 live/exit owner')
    same(ended['actual_execution_handle'], dict(kind='tool_session', session_id=26497), 'Actual device r2 session')
    same(live['actual_execution_handle'], ended['actual_execution_handle'], 'Device r2 same session')
    service = p.read_ref(ended['outputs']['service-exit.json'])
    owner = p.read_ref(ended['outputs']['device-owner.json'])
    admission = p.read_ref(ended['outputs']['admission.json'])
    result = p.read_ref(ended['outputs']['device-result.json'])
    outer = p.read_ref(ended['outputs']['outer-result.json'])
    same(service['owner'], owner, 'Device r2 service owner')
    same(service['observed_by_outer'], ended['owner'], 'Device r2 actual service observer')
    same(admission['outer_owner'], ended['owner'], 'Device r2 admitted owner')
    same(outer['owner'], ended['owner'], 'Device r2 completed owner')
    same(result['process'], previous.process(owner), 'Device r2 worker identity')
    p.require(outer['complete'] is result['complete'] is service['successful'] is True, 'Device r2 completed')
    p.require(p.terminal(service['properties']), 'Device r2 retained terminal state')
    same([service['properties'][key] for key in ('Result','ExecMainCode','ExecMainStatus')], ['success','1','0'], 'Actual device r2 service exit0')
    same(service['properties']['InvocationID'], owner['invocation_id'], 'Device r2 invocation')
    same(service['properties']['ExecMainPID'], str(owner['pid']), 'Device r2 service PID')
    group = p.GROUP_PREFIX+owner['unit']+'.service'
    p.require(ended['outer_and_device_gone'] is True, 'Root observed device r2 drain')
    for record in (ended['service_group_after_exit'], service['group']):
        p.require(record['group'] == group and record['absent'] is True, 'Device r2 recorded group drain')
    return [previous.process(ended['owner']), previous.process(owner)], dict(group=group, owner=owner)

def run(prefix, source_sha256, analysis_sha256):
    p.require(p.sha(SELF) == source_sha256 and p.sha(Path(analysis.__file__)) == analysis_sha256, 'Exact reviewed new sources')
    p.require(p.sha(Path(previous.__file__)) == PREVIOUS_SHA, 'Frozen predecessor helper')
    prefix = Path(prefix)
    p.require(prefix.parent == ROOT/'research' and re.fullmatch('minicpm5-original88-prefill-attention-[a-z0-9-]+', prefix.name), 'Finite fresh output prefix')
    paths = [Path(str(prefix)+suffix) for suffix in ('.json','-summary.json','-failure.json','-runner-start.json','-runner-result.json','-runner-failure.json')]
    p.require(all(not x.exists() for x in paths), 'Never overwrite saved evidence')
    p.user_bus()
    fd, owner, lock = post.external.acquire_lock()
    try:
        started = time.monotonic()
        spec = p.read_ref(SPEC)
        def guard():
            p.clear_window(); p.lock_proof(owner,lock); p.active_heavy()
            p.require(time.monotonic()-started < 1800, 'Bounded saved-data runtime')
            fs = os.statvfs(ROOT); p.require(fs.f_bavail*fs.f_frsize >= 28*p.GIB, 'Original 28GiB floor')
            same({n:p.sha(ROOT/n) for n in spec['canonical_sha256']}, spec['canonical_sha256'], 'Canonical state unchanged')
            p.require(p.sha(SELF) == source_sha256 and p.sha(Path(analysis.__file__)) == analysis_sha256 and p.sha(Path(previous.__file__)) == PREVIOUS_SHA, 'Analysis sources unchanged')
        guard()
        p.once(paths[3], dict(schema='minicpm5_prefill_attention_runner_start_v1', owner=owner, lock=lock,
                             source=dict(path=str(SELF),sha256=source_sha256), analysis=dict(path=analysis.__file__,sha256=analysis_sha256),
                             output_prefix=str(prefix), self_exit_claimed=False))
        print(json.dumps(dict(stage='admitted', owner=owner)), flush=True)
        bound, owners, prior = previous.check_prerequisites(spec)
        extra, r2 = r2_predecessors(); owners += extra
        p.control_usage(bound['spec']); p.gone(bound['finalizer_owner']); p.predecessor_drain(bound['catalog'])
        for old in owners: p.gone(old)
        drains = dict(call4=previous.drain_call4(prior), r1=previous.drain_failed_r1(prior['failed_r1']), r2=previous.drain_call4(r2))
        # All loaded original local helpers must retain the frozen source closure.
        import sys
        closure = {}
        for mapping in (bound['spec']['source_sha256'], bound['catalog']['source_sha256'], spec['source_sha256']):
            for path, wanted in mapping.items():
                p.require(path not in closure or closure[path] == wanted, 'Consistent inherited source pin '+path)
                closure[path] = wanted
        for module in tuple(sys.modules.values()):
            path = getattr(module, '__file__', None)
            if path and str(Path(path).resolve()).startswith(str(ROOT)+'/') and Path(path).resolve() not in (SELF, Path(analysis.__file__).resolve()):
                path = str(Path(path).resolve())
                p.require(closure.get(path) == p.sha(path), 'Frozen loaded helper '+path)
        guard()
        print(json.dumps(dict(stage='saved_payload_analysis', previous_processes=len(owners))), flush=True)
        outputs = analysis.run(bound, str(prefix), analysis_sha256, guard, owner)
        guard(); p.control_usage(bound['spec'])
        p.once(paths[4], dict(schema='minicpm5_prefill_attention_runner_result_v1', complete=True, execution_owner=owner,
                             inputs=dict(device_spec=SPEC,device_r2_exit=R2_EXIT,device_r2_review=R2_REVIEW),
                             checked_previous_owners=owners, observed_drains=drains, outputs=outputs,
                             self_exit_claimed=False, canonical_unchanged=True, full_model_qualified=False, new_speedup_claim=False))
        return dict(result=dict(path=str(paths[4]),sha256=p.sha(paths[4])), analysis_outputs=outputs)
    except BaseException as error:
        p.once(paths[5], dict(schema='minicpm5_prefill_attention_runner_failure_v1', execution_owner=owner,
                             error_type=type(error).__name__,error=str(error),self_exit_claimed=False))
        raise
    finally:
        post.external.release_lock(fd)

if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    for name in ('output-prefix','source-sha256','analysis-sha256'): cli.add_argument('--'+name,required=True)
    args = cli.parse_args()
    print(json.dumps(run(args.output_prefix,args.source_sha256,args.analysis_sha256)),flush=True)
