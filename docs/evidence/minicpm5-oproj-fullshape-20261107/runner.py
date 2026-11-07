"""Root serial runner for finite full-shape saved projection interventions; no model launches."""
import argparse
import json
import os
from pathlib import Path
import re
import time
import sys
import minicpm5_original32_mask_device_runner_v2 as previous
import minicpm5_saved_oproj_fullshape_intervention_worker_v1 as analysis
import minicpm5_saved_oproj_intervention_root_runner_v1 as original_projection
import minicpm5_original88_prefill_attention_root_runner_v1 as prefill

p, five, same, post, ROOT = previous.p, previous.five, previous.same, previous.post, previous.ROOT
SELF = Path(__file__).resolve()
SPEC = previous.masks.ref('minicpm5-original32-mask-device-execution-spec-20261107-v2.json', '50d1b9002fde217908e13fcabf80909ab66215f2fcae8faef89916d40127ada6')
R2_EXIT = previous.masks.ref('minicpm5-original32-mask-device-root-exit-r2-20261107-v1.json', 'c044c8105ae85d7ae5d4606cad491d85e609c8ec6268c65fe999c4574540a847')
R2_REVIEW = previous.masks.ref('minicpm5-original32-mask-device-independent-result-review-20261107-v1.json', '904b66da76ca18b222366b859fe87cf242ea84c1ae512035b8e1c5d9489c9612')
PREVIOUS_SHA = '953d57012fd1ad56c5a6cb74c64846cecd5794a2b1d6e8a2d66d695a073268b9'

PREFILL_RUNNER_SHA = '1135415aed72e1cc79ced60f30cbb767915da92499e9300ff59e49953cb837d7'
PREFILL_ANALYSIS_SHA = '08b927ef894a63cfab59105abc9b48ffeb9ffa38d6dde4273f801c40a1246261'
PREFILL_EXIT = previous.masks.ref('minicpm5-original88-prefill-attention-root-exit-r1-20261107-v1.json', '142551d148e1916e4de2bcb451da8733e7808c979ed091979c1a7e75926d75c8')
PREFILL_REVIEW = previous.masks.ref('minicpm5-original88-prefill-attention-independent-result-review-20261107-v1.json', 'ead16138391a124a2032c295156e8d0abbf058931b64e4c86882ba01704cbc66')
WORKER_REVIEW = previous.masks.ref('minicpm5-saved-oproj-fullshape-worker-independent-source-review-20261107-v1.json', '10b36891c822db87e5557e435c1eaa01693dd9eedb9eb1457415e629c1f50425')


def prefill_predecessor(bound):
    ended = p.read_ref(PREFILL_EXIT); live = p.read_ref(ended['root_live']); review = p.read_ref(PREFILL_REVIEW)
    p.require(review['passed'] is True and review['blockers'] == [], 'Independent saved prefill result')
    same(review['root_actual_exit_ref'], PREFILL_EXIT, 'Same reviewed actual prefill exit')
    p.require(ended['reviewer'] == live['reviewer'] == '/root' and ended['actual_exit_observed'] is True and
              type(ended['actual_exit_code']) is int and ended['actual_exit_code'] == 0 and
              ended['process_gone'] is True and live['actual_identity_verified'] is True, 'Actual prefill exit0 and identity')
    same(ended['owner'], live['owner'], 'Prefill live/exit owner')
    same(ended['actual_execution_handle'], dict(kind='tool_session', session_id=82651), 'Actual prefill session')
    same(live['actual_execution_handle'], ended['actual_execution_handle'], 'Same prefill handle')
    result = p.read_ref(ended['outputs']['report']); runner = p.read_ref(ended['outputs']['runner_result'])
    same(review['report'], ended['outputs']['report'], 'Reviewed prefill report')
    same(result['execution_owner'], ended['owner'], 'Prefill report owner')
    same(runner['execution_owner'], ended['owner'], 'Prefill wrapper owner')
    p.require(runner['complete'] is result['complete_selected_coverage'] is True, 'Completed saved analysis')
    p.require(result['self_exit_claimed'] is runner['self_exit_claimed'] is False, 'No self-certified exit')
    for key,value in bound['binding'].items(): same(result['binding'][key], value, 'Same original88 '+key)
    same(live['source'], dict(path=prefill.__file__,sha256=PREFILL_RUNNER_SHA), 'Actual prefill runner source')
    same(live['analysis'], dict(path=prefill.analysis.__file__,sha256=PREFILL_ANALYSIS_SHA), 'Actual prefill analyzer source')
    same(result['binding']['code'], live['analysis'], 'Prefill analyzer binding')
    expected = ['/home/ubuntu/venvs/vllm_neuron/bin/python','-B',prefill.__file__,'--output-prefix',live['output_prefix'],
                '--source-sha256',PREFILL_RUNNER_SHA,'--analysis-sha256',PREFILL_ANALYSIS_SHA]
    same(ended['owner']['argv'], expected, 'Actual saved prefill command')
    return previous.process(ended['owner'])


OPROJ_RUNNER_SHA = '8aeb08261c09bbdad4c087ed0f79e73e425be857494244f98ac8932873c61cb4'
OPROJ_WORKER_SHA = 'a24e6382c69c0d420e9d03c6a882a9c8ca40d0ce74947d10d27e355e00c531b8'
OPROJ_EXIT = previous.masks.ref('minicpm5-saved-oproj-intervention-root-exit-r1-20261107-v1.json','031eacc6babb8b812a7d2ae83d7dad937a57dde754fe819494acc53a603092e3')
OPROJ_REVIEW = previous.masks.ref('minicpm5-saved-oproj-intervention-independent-result-review-20261107-v1.json','932f7be451311f88a77570169ee4e642e8e991fd7ab1c10f47eea2ae5914f69d')


def projection_predecessor(bound):
    ended = p.read_ref(OPROJ_EXIT); live = p.read_ref(ended['root_live']); review = p.read_ref(OPROJ_REVIEW)
    p.require(review['passed'] is True and review['blockers'] == [], 'Independent projection result review')
    same(review['root_actual_exit_ref'], OPROJ_EXIT, 'Reviewed projection actual exit')
    same(review['report'], ended['outputs']['report'], 'Reviewed projection report')
    same(review['runner_result'], ended['outputs']['runner_result'], 'Reviewed projection wrapper result')
    p.require(ended['reviewer'] == live['reviewer'] == '/root' and ended['actual_exit_observed'] is True and
              type(ended['actual_exit_code']) is int and ended['actual_exit_code'] == 0 and
              ended['process_gone'] is True and live['actual_identity_verified'] is True, 'Actual projection exit and identity')
    same(ended['owner'], live['owner'], 'Projection live/exit owner')
    same(ended['actual_execution_handle'], dict(kind='tool_session',session_id=73920), 'Actual projection session')
    same(live['actual_execution_handle'], ended['actual_execution_handle'], 'Projection same handle')
    report = p.read_ref(ended['outputs']['report']); wrapper = p.read_ref(ended['outputs']['runner_result'])
    same(report['execution_owner'], ended['owner'], 'Projection report owner')
    same(wrapper['execution_owner'], ended['owner'], 'Projection wrapper owner')
    p.require(wrapper['complete'] is report['summary']['local_math_trial_complete'] is True, 'Completed projection')
    same([report['summary']['full_shape_controls'],report['summary']['fixed_input_cases'],report['summary']['formula_results'],report['summary']['original_execution_eligible_cases']],[24,24,144,19],'Preserved prior coverage and five failures')
    for key,value in bound['binding'].items():same(report['binding'][key],value,'Same finalized projection '+key)
    same(live['source'],dict(path=original_projection.__file__,sha256=OPROJ_RUNNER_SHA),'Actual projection runner pin')
    same(live['worker'],dict(path=original_projection.analysis.__file__,sha256=OPROJ_WORKER_SHA),'Actual projection worker pin')
    same(report['binding']['code'],live['worker'],'Projection report code')
    expected=['/home/ubuntu/venvs/vllm_neuron/bin/python','-B',original_projection.__file__,'--output-prefix',live['output_prefix'],
              '--source-sha256',OPROJ_RUNNER_SHA,'--analysis-sha256',OPROJ_WORKER_SHA]
    same(ended['owner']['argv'],expected,'Actual projection command')
    return previous.process(ended['owner'])


def run(prefix, source_sha256, analysis_sha256):
    p.require(p.sha(SELF) == source_sha256 and p.sha(Path(analysis.__file__)) == analysis_sha256, 'Exact reviewed new sources')
    p.require(p.sha(Path(previous.__file__)) == PREVIOUS_SHA, 'Frozen predecessor helper')
    p.require(p.sha(Path(prefill.__file__)) == PREFILL_RUNNER_SHA and p.sha(Path(prefill.analysis.__file__)) == PREFILL_ANALYSIS_SHA, 'Frozen prefill helpers')
    p.require(p.sha(original_projection.__file__) == OPROJ_RUNNER_SHA and p.sha(original_projection.analysis.__file__) == OPROJ_WORKER_SHA, 'Frozen completed projection helpers')
    accepted = p.read_ref(WORKER_REVIEW)
    same(accepted['worker'], dict(path=analysis.__file__,sha256=analysis_sha256), 'Reviewed exact worker')
    p.require(accepted['new_source_blockers'] == [], 'Worker independently reviewed')
    prefix = Path(prefix)
    p.require(prefix.parent == ROOT/'research' and re.fullmatch('minicpm5-saved-oproj-fullshape-intervention-[a-z0-9-]+', prefix.name), 'Finite fresh output prefix')
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
            p.require(p.sha(Path(prefill.__file__)) == PREFILL_RUNNER_SHA and p.sha(Path(prefill.analysis.__file__)) == PREFILL_ANALYSIS_SHA, 'Frozen prefill helpers unchanged')
            p.require(p.sha(original_projection.__file__) == OPROJ_RUNNER_SHA and p.sha(original_projection.analysis.__file__) == OPROJ_WORKER_SHA, 'Completed projection sources unchanged')
            p.require(os.sched_getaffinity(0) == {6,7}, 'Bounded CPU affinity')
            p.require(all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')), 'Fixed native thread environment')
            mem = dict((a.split(':')[0], int(a.split()[1])*1024) for a in Path('/proc/meminfo').read_text().splitlines() if a.startswith(('MemAvailable:','MemFree:')))
            p.require(mem['MemAvailable'] >= 4*p.GIB, 'Four GiB available for bounded saved operator')
            for path in paths:
                if path.exists(): p.require(path.stat().st_size < 16*p.MIB, 'Finite report file cap')
        guard()
        p.once(paths[3], dict(schema='minicpm5_saved_oproj_fullshape_runner_start_v1', owner=owner, lock=lock,
                             source=dict(path=str(SELF),sha256=source_sha256), analysis=dict(path=analysis.__file__,sha256=analysis_sha256),
                             output_prefix=str(prefix), self_exit_claimed=False))
        print(json.dumps(dict(stage='admitted', owner=owner)), flush=True)
        bound, owners, prior = previous.check_prerequisites(spec)
        extra, r2 = prefill.r2_predecessors(); owners += extra
        owners.append(prefill_predecessor(bound))
        owners.append(projection_predecessor(bound))
        p.control_usage(bound['spec']); p.gone(bound['finalizer_owner']); p.predecessor_drain(bound['catalog'])
        for old in owners: p.gone(old)
        drains = dict(call4=previous.drain_call4(prior), r1=previous.drain_failed_r1(prior['failed_r1']), r2=previous.drain_call4(r2))
        # All loaded original local helpers must retain the frozen source closure.
        closure = {}
        for mapping in (bound['spec']['source_sha256'], bound['catalog']['source_sha256'], spec['source_sha256']):
            for path, wanted in mapping.items():
                p.require(path not in closure or closure[path] == wanted, 'Consistent inherited source pin '+path)
                closure[path] = wanted
        closure.update({prefill.__file__:PREFILL_RUNNER_SHA,prefill.analysis.__file__:PREFILL_ANALYSIS_SHA,original_projection.__file__:OPROJ_RUNNER_SHA,original_projection.analysis.__file__:OPROJ_WORKER_SHA})
        design = p.read_ref(analysis.DESIGN)
        for ref in design['source_refs'].values():
            p.require(p.sha(ref['path']) == ref['sha256'], 'Original operator source unchanged')
        for ref in design['references'].values():
            # Design references are source/smallJSON only, never tensor files.
            p.read_ref(ref)
        for module in tuple(sys.modules.values()):
            path = getattr(module, '__file__', None)
            if path and str(Path(path).resolve()).startswith(str(ROOT)+'/') and Path(path).resolve() not in (SELF, Path(analysis.__file__).resolve()):
                path = str(Path(path).resolve())
                p.require(closure.get(path) == p.sha(path), 'Frozen loaded helper '+path)
        guard()
        print(json.dumps(dict(stage='saved_operator_intervention', previous_processes=len(owners))), flush=True)
        outputs = analysis.run(bound, str(prefix), analysis_sha256, guard, owner)
        guard(); p.control_usage(bound['spec'])
        p.once(paths[4], dict(schema='minicpm5_saved_oproj_fullshape_runner_result_v1', complete=True, execution_owner=owner,
                             inputs=dict(device_spec=SPEC,device_r2_exit=R2_EXIT,device_r2_review=R2_REVIEW,prefill_exit=PREFILL_EXIT,prefill_review=PREFILL_REVIEW,worker_review=WORKER_REVIEW,projection_exit=OPROJ_EXIT,projection_review=OPROJ_REVIEW),
                             checked_previous_owners=owners, observed_drains=drains, outputs=outputs,
                             self_exit_claimed=False, canonical_unchanged=True, full_model_qualified=False, new_speedup_claim=False))
        return dict(result=dict(path=str(paths[4]),sha256=p.sha(paths[4])), analysis_outputs=outputs)
    except BaseException as error:
        try:
            p.once(paths[5], dict(schema='minicpm5_saved_oproj_fullshape_runner_failure_v1', execution_owner=owner,
                                 error_type=type(error).__name__,error=str(error),self_exit_claimed=False))
        except BaseException as secondary:
            try:
                print(json.dumps(dict(original_error_type=type(error).__name__, original_error=str(error), failure_receipt_error_type=type(secondary).__name__, failure_receipt_error=str(secondary))), flush=True)
            except BaseException: pass
        raise
    finally:
        primary = sys.exc_info()[1]
        try:
            post.external.release_lock(fd)
        except BaseException as secondary:
            if primary is None: raise
            try:
                print(json.dumps(dict(original_error_type=type(primary).__name__, original_error=str(primary), lock_release_error_type=type(secondary).__name__, lock_release_error=str(secondary))), flush=True)
            except BaseException: pass

if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    for name in ('output-prefix','source-sha256','analysis-sha256'): cli.add_argument('--'+name,required=True)
    args = cli.parse_args()
    print(json.dumps(run(args.output_prefix,args.source_sha256,args.analysis_sha256)),flush=True)
