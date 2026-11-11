"""One finite, post-exit all-byte audit of the completed native own phase.

No model/native import or experiment launch. Retains the original serial lock
while reusing the existing descriptor restorer and independent output checker.
The external caller must record this auditor's actual exit separately.
"""
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
Q = ROOT / 'research'
sys.path[:0] = [str(Q), str(ROOT)]


class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'nki', 'transformers', 'vllm', 'torch_neuronx'}:
            raise AssertionError('Post-exit partial audit never imports native packages')


sys.meta_path.insert(0, NoNative())
from research import minicpm5_layer1_fixed_input_pv_adapter_v2 as adapter


def read(reference):
    raw = Path(reference['path']).read_bytes()
    assert len(raw) <= 16 * 2**20 and hashlib.sha256(raw).hexdigest() == reference['sha256']
    if 'bytes' in reference:
        assert len(raw) == reference['bytes']
    return json.loads(raw)


def reference(path):
    raw = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())


def main():
    assert sys.dont_write_bytecode and __debug__
    p = adapter.r.p; p.user_bus(); p.active_heavy()
    ended_ref = dict(path=str(Q/'minicpm5-fixed-input-pv-r4-root-actual-exit-20261111-v1.json'),
                     sha256='060478d8c4a25da7571bcafd5d825d099e5556e41f4392b71e6df1100186d3e2')
    ended = read(ended_ref)
    assert ended['actual_tool']['exit_code'] == 1 and ended['owners_gone']
    for owner in ended['owners']:
        p.gone(owner)
    fd, owner, first = adapter.prior.audited.support.acquire()
    began = time.monotonic()
    bound = dict(outer_owner=owner, lock_identity=adapter.r.ORIGINAL_LOCK,
                 retained_fd=fd, lock_policy=adapter.r.LOCK_POLICY)

    def guard():
        assert time.monotonic() - began < 120
        adapter.r.descriptor_observation(bound)

    try:
        os.sched_setaffinity(0, {6}); os.setpriority(os.PRIO_PROCESS, 0, 19)
        resource.setrlimit(resource.RLIMIT_AS, (2 * 2**30,) * 2)
        resource.setrlimit(resource.RLIMIT_CPU, (120, 125))
        resource.setrlimit(resource.RLIMIT_FSIZE, (2 * 2**20,) * 2)
        admission = read(ended['admission'])
        assert admission['lease']['outer_owner'] == ended['owners'][0]
        assert admission['witness']['spec'] == ended['spec']
        assert admission['witness']['review'] == ended['review']
        plan = adapter.restore_descriptor_plan(admission['witness'], read)
        store = adapter.Store(ended['output_root'], guard); inventory = store.inventory()
        for name, ref in ended['receipts'].items():
            assert reference(store.root/name) == ref
        terminal, terminal_ref = store.load('phase-0-exit.json')
        assert terminal['owner'] == ended['owners'][1]
        spec = read(ended['spec'])
        adapter.serial.actual_exit(terminal, terminal['owner'],
            adapter.serial.phase_argv(spec['worker'], ended['admission'], 0), 0)
        allowed = adapter.core.saved.validate_plan(plan)
        capture_io = adapter.core.saved.StableCaptureIO(allowed, guard, filesystem=os)
        summary, summary_ref = store.load('phase-0-summary.json')
        verifier = adapter.serial.IndependentOutputs(plan, 0,
            read_input=capture_io.read_bytes, read_output=store.read,
            event_at=lambda index: store.load('0-e-%04d.json' % index),
            guard=guard, path_for=lambda key: store.path_for(0, key))
        result = verifier.verify(summary)
        result.update(summary=summary_ref, terminal_sha256=adapter.r.digest(terminal),
            independent_original_files_read=len(capture_io._read),
            independent_original_bytes_read=sum(allowed[path]['raw_bytes'] for path in capture_io._read))
        prior_result, _ = store.load('phase-0-audit.json')
        assert adapter.serial.exact(result, prior_result)
        expected = {Path(r['path']).name for r in [*verifier.raw_refs.values(), *verifier.event_refs]}
        assert {name for name in inventory['names'] if name.startswith('0-')} == expected
        assert inventory == store.inventory()
        assert len(inventory['names']) == 54
        assert not (store.root/'phase-1-exit.json').exists()
        assert not (store.root/'own-gate.json').exists()
        assert not (store.root/'final.json').exists()
        for old_owner in ended['owners']:
            p.gone(old_owner)
        guard()
        out = Q/'minicpm5-fixed-input-pv-r4-native-partial-independent-audit-20261111-v1.json'
        report = dict(schema='minicpm5_pv_native_partial_independent_audit_v1',
            reviewer='/root', complete_for_native_phase_only=True, actual_model_terminal=ended_ref,
            auditor_owner=owner, lock_identity=adapter.r.ORIGINAL_LOCK, initial_descriptor=first,
            source=reference(Path(__file__)), admission=ended['admission'],
            reconstructed_descriptor_plan_sha256=adapter.r.digest(plan),
            phase0_audit_exactly_reproduced=result, inventory=inventory,
            original_model_failed=True, sdk_own_phase_completed=False,
            causal_operator_proven=False, full_model_qualified=False, new_speedup_claim=False,
            actual_auditor_tool_exit_observed=False, self_exit_claimed=False)
        with out.open('x') as handle:
            json.dump(report, handle, indent=2, sort_keys=True); handle.write('\n')
        print(json.dumps(reference(out)), flush=True)
    finally:
        adapter.prior.audited.support.release_after_physical_drain(fd,
            dict(physically_drained=True, workload_children_started=0))


if __name__ == '__main__':
    main()
