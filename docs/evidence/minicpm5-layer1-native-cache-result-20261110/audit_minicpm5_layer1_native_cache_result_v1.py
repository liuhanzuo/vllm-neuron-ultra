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
import minicpm5_layer1_native_cache_runner_v1 as runner
p = runner.p


def run():
    p.user_bus()
    fd, owner, lock = runner.post.external.acquire_lock()
    try:
        os.sched_setaffinity(0, {6})
        os.nice(19)
        resource.setrlimit(resource.RLIMIT_AS, (2 * 2**30,) * 2)
        resource.setrlimit(resource.RLIMIT_CPU, (120, 125))
        resource.setrlimit(resource.RLIMIT_FSIZE, (2 * 2**20,) * 2)
        signal.signal(signal.SIGALRM, lambda *args: (_ for _ in ()).throw(TimeoutError('Post-exit audit bound')))
        signal.alarm(180)
        exit_path = Q / 'minicpm5-layer1-native-cache-r1-actual-exit-20261110-v1.json'
        ended = p.read(exit_path)
        assert ended['actual_exit_observed'] and ended['actual_exit_code'] == 0
        assert ended['all_three_processes_gone'] and ended['cgroup_empty'] and ended['lock_identity'] == lock
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
        assert records['children']['exit_codes'] == records['coordinator']['native_exits'] == [0]
        assert records['service_exit']['successful'] is True and records['cleanup']['physically_drained'] is True
        assert records['cleanup']['original_failure'] is None
        def guard():
            p.lock_proof(owner, lock)
            p.clear_window()
            for old in [ended['owner'], ended['coordinator'], ended['native']]:
                p.gone(old)
            p.empty_group(p.GROUP_PREFIX + spec['unit'] + '.service')
        guard()
        p.active_heavy()
        runner.verify_owned_drain(spec, ended['outputs']['admission'], admission['attempt_id'], ended['coordinator'], True)
        result=records['native_result']
        assert result['process']==ended['native'] and result['admission']==ended['outputs']['admission']
        proof=runner.validate_native_result(result,plan,guard)
        assert proof==records['coordinator']['validated']
        assert proof['frames']==32 and proof['full_cache_bytes']==5894144 and proof['full_logits_bytes']==16711680
        assert proof['selected_taps']==6816 and proof['all_native_cache_prefix_links_exact']
        assert not {'torch', 'numpy', 'neuronxcc', 'torch_neuronx'} & set(sys.modules)
        guard()
        for path, digest in spec['source_sha256'].items():
            assert p.sha(path) == digest
        target = Q / 'minicpm5-layer1-native-cache-post-exit-audit-20261110-v1.json'
        p.once(target, dict(schema='minicpm5_layer1_native_cache_post_exit_audit_v1', complete=True,
            reviewer='/root', observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            source=runner.reference(__file__), model_exit=runner.reference(exit_path), owner=owner,
            lock_identity=lock, result=proof, every_raw_byte_reread_after_all_model_processes_exited=True,
            original_native_history_preserved=True, all213_selected_taps_raw_exact=True,
            attention_causality_proven=False, full_model_qualified=False, new_speedup_claim=False))
        print(json.dumps(runner.reference(target)))
    finally:
        signal.alarm(0)
        runner.post.external.release_lock(fd)


if __name__ == '__main__':
    run()
