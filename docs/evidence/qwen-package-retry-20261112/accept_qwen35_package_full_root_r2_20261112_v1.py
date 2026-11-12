"""Root r2 exact package acceptance; original complete History and 874-child verification retained.

Import and --help perform no preparation, History, lock or native work.
"""
import argparse
from pathlib import Path
import re

R = Path('/home/ubuntu/experiments/trn1-latest-models-20261031'); Q = R / 'research'
EXACT_REFS = {'catalog': ('/home/ubuntu/experiments/trn1-latest-models-20261031/research/minicpm5-operator-post-r2-serial-workloads-root-catalog-20261112-v112.json', '7d3d1ac0b8ca22ec6ce334baffb8599e967081624f9caf658116905665c0a256'), 'idle': ('/home/ubuntu/experiments/trn1-latest-models-20261031/research/continuation-handoff-20261112-v584-qwen-package-precheck-failed.json', 'fdd95f3f235e3c4c16daf0c60372977f7bf9b01c8bfbbed40309a1460b02e7dc'), 'runtime-review': ('/home/ubuntu/experiments/trn1-latest-models-20261031/research/qwen35-package-serial-runtime-v4-root-source-review-20261112-v1.json', None), 'preparer-review': ('/home/ubuntu/experiments/trn1-latest-models-20261031/research/qwen35-package-serial-preparer-v3-root-source-review-20261112-v1.json', None), 'budget': ('/home/ubuntu/experiments/trn1-latest-models-20261031/research/qwen35-package-full-r2-20261112-resource-budget.json', None), 'candidate': ('/home/ubuntu/experiments/trn1-latest-models-20261031/research/qwen35-package-full-r2-20261112-binding-candidate.json', None)}

def parser():
    value = argparse.ArgumentParser(description=__doc__)
    for name in EXACT_REFS:
        value.add_argument('--' + name, required=True)
        value.add_argument('--' + name + '-sha256', required=True)
    return value


def selected_refs(args):
    result = {}
    for name, (expected_path, fixed_sha) in EXACT_REFS.items():
        path = getattr(args, name.replace('-', '_'))
        sha = getattr(args, name.replace('-', '_') + '_sha256')
        if path != expected_path or type(sha) is not str or re.fullmatch('[a-f0-9]{64}', sha) is None:
            raise ValueError('exact r2 reference required: ' + name)
        if fixed_sha is not None and sha != fixed_sha:
            raise ValueError('frozen r2 prerequisite digest changed: ' + name)
        result[name] = dict(path=path, sha256=sha)
    return result


def main(argv=None):
    refs = selected_refs(parser().parse_args(argv))
    import datetime
    import json
    import os
    from pathlib import Path
    import resource
    import signal
    import sys
    import time

    sys.path[:0] = [str(Q), str(R)]
    assert sys.dont_write_bytecode and __debug__
    resource.setrlimit(resource.RLIMIT_AS, (2 * 2**30, 2 * 2**30))
    resource.setrlimit(resource.RLIMIT_CPU, (600, 605)); signal.alarm(600)
    from qwen35_package_serial_runtime_v4 import contract, io, budget
    from qwen35_package_serial_runtime_v4.history import Bundle
    from research import minicpm5_layer1_operator_sdk_runtime_completion_v4 as latest

    candidate_ref = refs['candidate']
    candidate = io.read(candidate_ref)
    assert not candidate['passed'] and not candidate['launch_authorized']
    assert candidate['catalog'] == refs['catalog'] and candidate['idle_handoff'] == refs['idle']
    assert candidate['runtime_source_review'] == refs['runtime-review'] and candidate['preparer_source_review'] == refs['preparer-review']
    assert candidate['review_path'] == str(Q / 'qwen35-package-full-r2-20261112-review.json')
    assert candidate['spec_ref']['path'] == str(Q / 'qwen35-package-full-r2-20261112-spec.json')
    assert candidate['review_draft_ref']['path'] == str(Q / 'qwen35-package-full-r2-20261112-review-draft.json')
    assert io.read(candidate['runtime_source_review'])['passed']
    assert io.read(candidate['preparer_source_review'])['source_accepted']
    spec_ref = candidate['spec_ref']; spec = io.read(spec_ref, 64 * 2**20)
    assert spec['run_name'] == 'pkg-r2-20261112' and spec['output_root'] == '/mnt/nvme/neuron-work/qwp-pkg-r2-20261112'
    assert spec['resource_budget'] == refs['budget']
    assert spec['running_handoff_path'] == str(Q / 'continuation-handoff-20261112-v585-qwen35-package-full-r2-running.json')
    catalog = io.read(spec['current_serial_catalog']); idle = io.read(catalog['current_handoff'])
    assert spec['current_serial_catalog'] == candidate['catalog'] and catalog['current_handoff'] == candidate['idle_handoff']
    assert not idle['running'] and idle['active_serial'] is None and idle['active_heavy_units'] == idle['live_tool_sessions'] == []
    tip = latest.latest_handoff(); assert tip['path'] == catalog['current_handoff']['path']
    draft = io.read(candidate['review_draft_ref'], 64 * 2**20)
    assert draft['spec'] == spec_ref and draft['source_sha256'] == spec['source_sha256']
    assert draft['passed'] is draft['launch_authorized'] is False
    review = dict(draft, passed=True, blockers=[], launch_authorized=True)
    contract.validate_spec(spec, review, spec_ref)
    budget.fresh_paths(spec); budget.unchanged_inputs(spec); io.resources(spec, reserve=True)
    lock = io.RetainedLock(); lock.acquire()
    try:
        io.process_api().active_heavy()
        assert latest.latest_handoff() == tip
        payload = (json.dumps(review, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
        assert len(payload) <= 64 * 2**20
        with Path(candidate['review_path']).open('xb') as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        review_ref = io.ref(candidate['review_path'], 64 * 2**20)
        started = time.monotonic(); last = [-1.]
        def guard():
            assert time.monotonic() - started < 600
            lock.check()
            if time.monotonic() - last[0] >= .5:
                assert latest.latest_handoff() == tip
                last[0] = time.monotonic()
            assert not any(n.split('.')[0] in ('torch','torch_neuronx','numpy','neuronxcc','vllm','transformers') for n in sys.modules)
        print('Exact package source acceptance passed; checking full inherited History', flush=True)
        bound = Bundle(spec_ref, review_ref, guard).resolve()
        bound.unchanged(); freshness = bound.freshness()
        assert freshness['latest_handoff'] == catalog['current_handoff']
        assert len(bound.supplement['scope']['children']) == 874
        io.loaded_local_sources(spec['source_sha256']); resources = io.resources(spec, reserve=True)
        assert latest.latest_handoff() == tip
        result = io.once(Q / 'qwen35-package-full-r2-root-binding-20261112-v1.json', dict(
            schema='qwen35_package_full_root_concrete_binding_v1', reviewer='/root', passed=True,
            observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), candidate=candidate_ref,
            spec=spec_ref, review=review_ref, history_overlay=spec['history_overlay'],
            catalog=spec['current_serial_catalog'], idle_handoff=catalog['current_handoff'],
            source_files=len(spec['source_sha256']), catalog_rows=len(catalog['runs']),
            input_extension_count=len(spec['input_extension']), preservation=candidate['preservation'],
            full_source_history_resolution_passed=True, lfm_children=874, original_failed_statuses_preserved=True,
            original_lock_reacquired=True, resources=resources, physical_pair_performed=False,
            native_executed=False, runtime_launched=False, full_model_qualified=False, new_speedup_claim=False))
    finally:
        lock.release(True)
    print(json.dumps(dict(binding=result, spec=spec_ref, review=review_ref)), flush=True)


if __name__ == '__main__':
    main()
