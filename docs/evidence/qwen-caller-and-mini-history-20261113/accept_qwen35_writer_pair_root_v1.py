"""Explicit root acceptance of a prepared writer candidate; no defaults.

No import-time IO. Only an actual root call after independent source review
resolves full current History under the original lock. This does not launch.
"""
import argparse
from pathlib import Path

R = Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
Q = R / 'research'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    parser.add_argument('--request-sha256', required=True)
    args = parser.parse_args(argv)
    import datetime
    import json
    import os
    import resource
    import signal
    import sys
    import time
    sys.path[:0] = [str(Q), str(R)]
    from qwen35_writer_pair_serial_runtime_v2 import contract, io, budget
    from qwen35_writer_pair_serial_runtime_v2.history import Bundle
    from prepare_qwen35_writer_pair_candidate_v1 import completed_prerequisites
    from research import minicpm5_layer1_operator_sdk_runtime_completion_v4 as latest
    require = contract.require
    require(sys.dont_write_bytecode and __debug__, 'python -B with assertions required')
    resource.setrlimit(resource.RLIMIT_AS, (2 * 2**30, 2 * 2**30))
    resource.setrlimit(resource.RLIMIT_CPU, (600, 605)); signal.alarm(600)
    request_ref = contract.reference(dict(path=args.request, sha256=args.request_sha256))
    request = io.read(request_ref)
    require(set(request) == {'schema', 'candidate', 'caller_source_review', 'review_path', 'binding_path'}
            and request['schema'] == 'qwen35_writer_pair_root_accept_request_v1', 'explicit exact root request required')
    candidate = io.read(request['candidate'])
    require(candidate['schema'] == 'qwen35_writer_pair_root_candidate_envelope_v1'
            and candidate['passed'] is False and candidate['launch_authorized'] is False,
            'non-authorizing writer preparation envelope required')
    caller_review = io.read(request['caller_source_review'])
    require(caller_review['reviewer'] == '/root' and caller_review['source_accepted'] is True
            and caller_review['blockers'] == [] and caller_review['runtime_admitted'] is False,
            'independent root caller-source review required')
    runtime_review = io.read(candidate['runtime_source_review'])
    require(runtime_review['schema'] == 'qwen35_writer_pair_runtime_v2_root_source_review_v1'
            and runtime_review['source_accepted'] is True and runtime_review['runtime_admitted'] is False,
            'accepted exact writer runtime v2 source required')
    spec_ref = candidate['spec_ref']; spec = io.read(spec_ref, 64 * 2**20)
    draft = io.read(candidate['review_draft_ref'], 64 * 2**20)
    require(draft['passed'] is False and draft['launch_authorized'] is False
            and draft['spec'] == spec_ref and draft['source_sha256'] == spec['source_sha256'],
            'candidate review binds the exact complete source map')
    require(spec['history_overlay'] == candidate['overlay_ref']
            and spec['current_serial_catalog'] == candidate['catalog']
            and spec['geometry_qualification'] == candidate['completed_prerequisites']['geometry_qualification'],
            'preparation and actual source bindings agree')
    self_ref = io.ref(Path(__file__).resolve())
    require(spec['source_sha256'].get(self_ref['path']) == self_ref['sha256'],
            'root acceptance source must be pinned before admission')
    caller_delivery = io.read(caller_review['source_delivery'])
    for path, row in caller_delivery['files'].items():
        require(spec['source_sha256'].get(path) == row['sha256'], 'complete reviewed caller source closure required')
    catalog = io.read(spec['current_serial_catalog']); idle_ref = catalog['current_handoff']; idle = io.read(idle_ref)
    require(idle_ref == candidate['idle_handoff'] and idle['running'] is False
            and idle['active_serial'] is None and idle['live_tool_sessions'] == idle['active_heavy_units'] == [],
            'complete exact idle predecessor required')
    tip = latest.latest_handoff()
    require(tip == dict(path=idle_ref['path'], identity=io.identity(idle_ref['path'])), 'current idle identity changed')
    output_paths = [Path(request[key]) for key in ('review_path', 'binding_path')]
    require(len(set(output_paths)) == 2, 'distinct fresh acceptance outputs required')
    for path in output_paths:
        require(path.parent == Q and path.resolve() == path and path.suffix == '.json'
                and not path.exists() and not path.is_symlink()
                and str(path) not in spec['source_sha256'] and str(path) not in spec['input_extension'],
                'fresh canonical acceptance output required')
    review = dict(draft, passed=True, blockers=[], launch_authorized=True)
    contract.validate_spec(spec, review, spec_ref)
    # All prerequisite reads must remain in the exact immutable input binding.
    def prerequisite_read(ref, cap=8 * 2**20):
        bound = spec['input_extension'].get(ref['path'])
        if cap == 64 * 2**20:
            bound = candidate['preparation_only_references'].get(ref['path'])
        require(bound is not None and bound['sha256'] == ref['sha256']
                and io.identity(ref['path']) == bound['identity'], 'immutable completed prerequisite required')
        value = io.read(ref, cap)
        require(io.identity(ref['path']) == bound['identity'], 'completed prerequisite changed during read')
        return value
    completed_prerequisites(candidate['completed_prerequisites'], catalog, prerequisite_read)
    budget.fresh_paths(spec); budget.unchanged_inputs(spec); io.resources(spec, reserve=True)
    lock = io.RetainedLock(); lock.acquire()
    try:
        io.process_api().active_heavy()
        require(latest.latest_handoff() == tip, 'unknown successor before source acceptance')
        payload = (json.dumps(review, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
        require(len(payload) <= 64 * 2**20, 'unchanged finite review envelope')
        with output_paths[0].open('xb') as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        review_ref = io.ref(output_paths[0], 64 * 2**20)
        started = time.monotonic(); last = [-1.]
        def guard():
            require(time.monotonic() - started < 600, 'original full History acceptance deadline')
            lock.check()
            if time.monotonic() - last[0] >= .5:
                require(latest.latest_handoff() == tip, 'unknown later handoff during full History')
                last[0] = time.monotonic()
            require(not any(n.split('.')[0] in ('torch', 'torch_neuronx', 'numpy', 'neuronxcc', 'vllm', 'transformers')
                    for n in sys.modules), 'no native imports during root source acceptance')
        bound = Bundle(spec_ref, review_ref, guard).resolve()
        bound.unchanged(); freshness = bound.freshness()
        require(freshness['latest_handoff'] == idle_ref and len(bound.supplement['scope']['children']) == 874,
                'complete original History and all 874 inherited children retained')
        io.loaded_local_sources(spec['source_sha256']); resources = io.resources(spec, reserve=True)
        require(latest.latest_handoff() == tip, 'unknown successor before binding publication')
        result = io.once(output_paths[1], dict(schema='qwen35_writer_pair_root_concrete_binding_v1',
            reviewer='/root', passed=True, observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            candidate=request['candidate'], spec=spec_ref, review=review_ref, history_overlay=spec['history_overlay'],
            catalog=spec['current_serial_catalog'], idle_handoff=idle_ref,
            source_files=len(spec['source_sha256']), catalog_rows=len(catalog['runs']),
            input_extension_count=len(spec['input_extension']), preservation=candidate['preservation'],
            completed_prerequisites=candidate['completed_prerequisites'], caller_source_review=request['caller_source_review'],
            full_source_history_resolution_passed=True, lfm_children=874, original_failed_statuses_preserved=True,
            original_lock_reacquired=True, resources=resources, physical_pair_performed=False,
            native_executed=False, runtime_launched=False, full_model_qualified=False, new_speedup_claim=False))
    finally:
        lock.release(True)
    print(json.dumps(dict(binding=result, spec=spec_ref, review=review_ref)))


if __name__ == '__main__':
    main()
