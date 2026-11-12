"""Root r2 metadata-only publication using frozen runtime v4 and preparer v3.

Import and --help perform no preparation, History, lock or native work.
"""
import argparse
from pathlib import Path
import re

R = Path('/home/ubuntu/experiments/trn1-latest-models-20261031'); Q = R / 'research'
EXACT_REFS = {'catalog': ('/home/ubuntu/experiments/trn1-latest-models-20261031/research/minicpm5-operator-post-r2-serial-workloads-root-catalog-20261112-v112.json', '7d3d1ac0b8ca22ec6ce334baffb8599e967081624f9caf658116905665c0a256'), 'idle': ('/home/ubuntu/experiments/trn1-latest-models-20261031/research/continuation-handoff-20261112-v584-qwen-package-precheck-failed.json', 'fdd95f3f235e3c4c16daf0c60372977f7bf9b01c8bfbbed40309a1460b02e7dc'), 'runtime-review': ('/home/ubuntu/experiments/trn1-latest-models-20261031/research/qwen35-package-serial-runtime-v4-root-source-review-20261112-v1.json', None), 'preparer-review': ('/home/ubuntu/experiments/trn1-latest-models-20261031/research/qwen35-package-serial-preparer-v3-root-source-review-20261112-v1.json', None), 'stat-inputs': ('/home/ubuntu/experiments/trn1-latest-models-20261031/research/qwen35-package-full-r2-20261112-stat-inputs.json', None), 'budget': ('/home/ubuntu/experiments/trn1-latest-models-20261031/research/qwen35-package-full-r2-20261112-resource-budget.json', None)}

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
    import hashlib
    import json
    import os
    from pathlib import Path
    import sys

    sys.path[:0] = [str(Q), str(R)]
    assert sys.dont_write_bytecode and __debug__
    import prepare_qwen35_package_serial_candidate_v3 as prep
    from qwen35_package_serial_runtime_v4 import contract, io, budget
    from research import minicpm5_layer1_operator_sdk_runtime_completion_v4 as latest

    cat_ref = refs['catalog']; idle_ref = refs['idle']
    tip = latest.latest_handoff()
    assert tip['path'] == idle_ref['path']
    def guard():
        assert latest.latest_handoff() == tip
        assert not any(n.split('.')[0] in ('torch','torch_neuronx','numpy','neuronxcc','vllm','transformers') for n in sys.modules)
    readers = prep.StableReaders(guard)
    def at(name): return io.ref(Q / name)
    def read(reference): return readers.metadata(reference, 8 * 2**20)['value']
    root_review_ref = refs['preparer-review']
    root_review = read(root_review_ref)
    assert root_review['source_accepted'] and root_review['passed'] and not root_review['blockers']
    assert root_review['schema'] == 'qwen35_package_serial_preparer_v3_root_source_review_v1'
    assert root_review['reviewer'] == '/root' and root_review['source_only'] is True and root_review['runtime_admitted'] is False
    preparer_delivery_ref = root_review['delivery']; preparer_delivery = read(preparer_delivery_ref)
    assert preparer_delivery_ref == dict(path=str(Q / 'qwen35-package-serial-preparer-v3-source-delivery-20261112.json'), sha256='9bb1a6d0968925c02a30a589a8c570ce8c9760f9e7a31af7346d24c70720d2bf')
    stat_ref = refs['stat-inputs']; stat_inputs = read(stat_ref)
    assert stat_inputs['resource_budget'] == refs['budget']
    assert stat_inputs['run_name'] == 'pkg-r2-20261112' and stat_inputs['output_root'] == '/mnt/nvme/neuron-work/qwp-pkg-r2-20261112'
    seed_exit_ref = at('qwen35-sampler-r10-root-actual-exit-20261112-v1.json'); seed_exit = read(seed_exit_ref)
    helpers = [Path(__file__).resolve(), Q / 'accept_qwen35_package_full_root_r2_20261112_v1.py']
    sources = {str(path): io.ref(path)['sha256'] for path in helpers}
    sources.update({path: row['sha256'] for path, row in preparer_delivery['files'].items()})
    for name in ('qwen35_package_serial_runtime_v2/__init__.py', 'qwen35_package_serial_runtime_v2/contract.py',
                 'qwen35_sampler_pair_protocol_v1.py', 'qwen35_package_full_validation_v2/__init__.py',
                 'qwen35_package_full_validation_v2/runtime.py'):
        expected = preparer_delivery['import_dependency_sources'][str(Q / name)]
        assert at(name)['sha256'] == expected['sha256'] and (Q / name).stat().st_size == expected['bytes']
        sources[str(Q / name)] = expected['sha256']
    runtime_review_ref = refs['runtime-review']
    runtime_review = read(runtime_review_ref)
    assert runtime_review['source_delivery'] == dict(path=str(Q / 'qwen35_package_serial_runtime_v4/source_delivery.json'), sha256='58e8c65df2c38f8e3b83632f2a1e09dd3477dfef9d726e9530d3a6efb73b92b0')
    prefix = 'qwen35-package-full-r2-20261112'
    future = dict(run_name=stat_inputs['run_name'], output_root=stat_inputs['output_root'],
        overlay_output_root='/mnt/nvme/minicpm5-layer1-operator-native-package-full-r2-20261112',
        running_handoff_path=str(Q / 'continuation-handoff-20261112-v585-qwen35-package-full-r2-running.json'),
        overlay_path=str(Q / (prefix + '-history-overlay.json')), spec_path=str(Q / (prefix + '-spec.json')),
        review_candidate_path=str(Q / (prefix + '-review-draft.json')))
    request = dict(seed_spec=seed_exit['spec'], seed_review=seed_exit['review'], seed_actual_exit=seed_exit_ref,
        current_catalog=cat_ref, idle_handoff=idle_ref, runtime_directory=str(Q / 'qwen35_package_serial_runtime_v4'),
        runtime_delivery=runtime_review['source_delivery'], runtime_review=runtime_review_ref,
        resource_budget=stat_inputs['resource_budget'], case=stat_inputs['case'], binary_inputs=stat_inputs['binary_inputs'],
        preparation_sources=sources, required_helpers=[str(p) for p in helpers],
        extra_metadata=[root_review_ref, preparer_delivery_ref, stat_ref,
            at('qwen35-package-build-storage-source-review-20261112-v1.json'),
            root_review['independent_review']], future=future)
    candidate = prep.prepare(request, readers.metadata, readers.source, prep.file_identity, guard)
    spec = candidate['spec']
    budget.fresh_paths(spec); budget.unchanged_inputs(spec); resources = io.resources(spec, reserve=True)
    paths = [future[k] for k in ('overlay_path','spec_path','review_candidate_path','running_handoff_path')]
    review_path = Q / (prefix + '-review.json'); candidate_path = Q / (prefix + '-binding-candidate.json')
    assert all(not Path(p).exists() and not Path(p).is_symlink() for p in [*paths, review_path, candidate_path])
    for path, row in {**candidate['observed_metadata'], **candidate['observed_sources']}.items():
        assert prep.file_identity(path) == row['identity']
    guard()
    for name in ('overlay','spec','review'):
        target = candidate[name + '_ref']
        with Path(target['path']).open('xb') as stream:
            stream.write(candidate[name + '_bytes']); stream.flush(); os.fsync(stream.fileno())
        assert io.ref(target['path'], 64 * 2**20) == target
    receipt = dict(schema='qwen35_package_full_root_binding_candidate_v1', reviewer='/root',
        spec_ref=candidate['spec_ref'], overlay_ref=candidate['overlay_ref'], review_draft_ref=candidate['review_ref'],
        review_path=str(review_path), runtime_source_review=runtime_review_ref, preparer_source_review=root_review_ref,
        catalog=cat_ref, idle_handoff=idle_ref, preservation=candidate['preservation'], resources=resources,
        catalog_source_extension_location=dict(reference=candidate['overlay_ref'], field='source_extension'),
        preparation_only_inputs=candidate['preparation_only_references'], root_source=io.ref(Path(__file__).resolve()),
        passed=False, launch_authorized=False, full_history_precheck_pending=True, physical_checks_passed=False,
        native_executed=False, runtime_launched=False, full_model_qualified=False, new_speedup_claim=False)
    guard()
    result = io.once(candidate_path, receipt)
    print(json.dumps(dict(candidate=result, spec=candidate['spec_ref'], preservation=candidate['preservation'])))


if __name__ == '__main__':
    main()
