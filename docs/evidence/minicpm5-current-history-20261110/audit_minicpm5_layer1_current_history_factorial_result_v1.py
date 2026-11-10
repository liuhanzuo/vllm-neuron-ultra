"""Post-exit metadata and full raw-output reread, without native dependencies."""
import datetime
import json
import os
from pathlib import Path
import resource
import signal
import sys
R = Path(__file__).resolve().parents[1]
Q = R / 'research'
sys.path.insert(0, str(R))
import minicpm5_layer1_current_history_factorial_cpu_runner_v1 as runner
from research import minicpm5_layer1_current_history_factorial_analysis_v1 as analysis
p = runner.p


def run():
    p.user_bus()
    fd, owner, lock = runner.post.external.acquire_lock()
    try:
        os.sched_setaffinity(0, {6})
        os.nice(19)
        resource.setrlimit(resource.RLIMIT_AS, (2 * 2**30,) * 2)
        resource.setrlimit(resource.RLIMIT_CPU, (600, 605))
        resource.setrlimit(resource.RLIMIT_FSIZE, (2 * 2**20,) * 2)
        signal.signal(signal.SIGALRM, lambda *args: (_ for _ in ()).throw(TimeoutError('Post-exit audit bound')))
        signal.alarm(900)
        exit_path = Q / 'minicpm5-layer1-current-history-factorial-r1-actual-exit-20261110-v1.json'
        ended = p.read(exit_path)
        assert ended['actual_exit_observed'] and ended['actual_exit_code'] == 0
        assert ended['all_four_processes_gone'] and ended['cgroup_empty'] and ended['lock_identity'] == lock
        live = p.read_ref(ended['live'])
        assert ended['owner'] == live['owner'] and ended['actual_execution_handle'] == live['actual_execution_handle']
        spec = p.read_ref(ended['spec'])
        plan = p.read_ref(spec['plan'])
        review = p.read_ref(ended['review'])
        assert review['passed'] and review['spec'] == ended['spec']
        assert spec['source_sha256'] == review['source_sha256'] == plan['source_sha256']
        for path, digest in spec['source_sha256'].items():
            assert p.sha(path) == digest
        records = {k: p.read_ref(ref) for k, ref in ended['outputs'].items()}
        admission = records['admission']
        assert records['outer']['owner'] == admission['outer_owner'] == ended['owner']
        assert records['children']['exit_codes'] == records['coordinator']['rank_exits'] == [0, 0]
        assert records['service_exit']['successful'] is True and records['cleanup']['physically_drained'] is True
        assert records['cleanup']['original_failure'] is None
        def guard():
            p.lock_proof(owner, lock)
            p.clear_window()
            for old in [ended['owner'], ended['coordinator'], *ended['ranks']]:
                p.gone(old)
            p.empty_group(p.GROUP_PREFIX + spec['unit'] + '.service')
        guard()
        p.active_heavy()
        runner.verify_owned_drain(spec, ended['outputs']['admission'], admission['attempt_id'], ended['coordinator'], True)
        ranks = [records['rank0_result'], records['rank1_result']]
        for index, result in enumerate(ranks):
            assert result['process'] == ended['ranks'][index]
            assert result['admission'] == ended['outputs']['admission']
        proof = runner.validate_rank_pair(ranks, plan, guard)
        assert proof == records['coordinator']['validated']
        assert proof['decode_calls'] == 180 and proof['raw_bytes'] == 2949120
        assert proof['unique_replicated_output_rows'] == 360
        assert proof['previous_real_controls_full_bytes_exact'] is True
        assert proof['previous_control_calls'] == 108
        assert proof['official_current_intervention_performed'] is True
        assert proof['current_qkv_change_together'] is True
        assert not {'torch', 'numpy', 'neuronxcc', 'torch_neuronx'} & set(sys.modules)
        guard()
        for path, digest in spec['source_sha256'].items():
            assert p.sha(path) == digest
        descriptive=analysis.analyze(proof,plan,p.read_ref,guard)
        target = Q / 'minicpm5-layer1-current-history-factorial-post-exit-audit-20261110-v1.json'
        p.once(target, dict(schema='minicpm5_layer1_current_history_factorial_post_exit_audit_v1', complete=True,
            reviewer='/root', observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            source=runner.reference(__file__), model_exit=runner.reference(exit_path), owner=owner,
            lock_identity=lock, result=proof, descriptive=descriptive,analysis_source=runner.reference(analysis.__file__), every_raw_byte_reread_after_all_model_processes_exited=True,
            own_history_reproduced=True, official_history_intervention_performed=True,
            official_current_intervention_performed=True, current_qkv_change_together=True,
            attention_causality_proven=False, full_model_qualified=False, new_speedup_claim=False))
        print(json.dumps(runner.reference(target)))
    finally:
        signal.alarm(0)
        runner.post.external.release_lock(fd)


if __name__ == '__main__':
    run()
