"""Root's concrete metadata binding using the accepted existing helpers."""
import copy
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time

R = Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
Q = R / 'research'
sys.path[:0] = [str(Q), str(R)]


def main():
    assert sys.dont_write_bytecode and __debug__
    resource.setrlimit(resource.RLIMIT_AS, (2 * 2**30, 2 * 2**30))
    resource.setrlimit(resource.RLIMIT_CPU, (600, 605))
    signal.alarm(600)
    fd = os.open(R / 'serial-enrollment.lock', os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        assert (os.fstat(fd).st_dev, os.fstat(fd).st_ino) == (66306, 8294421)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        import prepare_qwen35_sampler_r6_candidate_v1 as saved
        import prepare_qwen35_sampler_v9_candidate_v3 as objects
        import prepare_minicpm5_layer1_operator_native_v7 as witnesses
        import qwen35_sampler_runtime_contract_v12 as contract
        import qwen35_sampler_runtime_io_v12 as io
        import qwen35_package_preflight_serial_v2 as stage
        from qwen35_package_preflight_history_v2 import History
        from qwen35_sampler_history_supplement_v1 import build as supplement
        from research import minicpm5_layer1_operator_sdk_runtime_completion_v4 as latest

        def known(name, sha):
            return dict(path=str(Q / name), sha256=sha)

        old_ref = known('qwen35-sampler-r10-20261112-spec-v12.json', '86a4c46dafcd3c16c74101caef26604af8e4b4041bcc3c6406faeab60d8d433f')
        overlay_seed_ref = known('qwen35-sampler-r10-20261112-history-overlay-v12.json', 'e8210a2a9c4010583cbba719255967b57c06d09110c15509762a863996e7440e')
        catalog_ref = known('minicpm5-operator-post-r2-serial-workloads-root-catalog-20261112-v96.json', '6b2f567254f85ffba9fd1565034ff064486bb5c79e97c0baf69627635e5ef32f')
        handoff_ref = known('continuation-handoff-20261112-v558-mini-formal-import-r2-ended.json', '4566aa6e7f28cfcb9824bc83d77e337a989b1c8736360506478dda87fb88bf33')
        delivery_ref = known('qwen35-package-preflight-serial-source-delivery-20261112-v2.json', '8927a4b1b177aafb65cad7ad36e3899c84a2dad1f1f33329ef6f6277c5765652')
        acceptance_ref = known('qwen35-package-preflight-serial-root-source-review-20261112-v2.json', '96cfa3726ff32d585cd02de25ad2f88d4709db22e5250397663365ecba580f0d')
        tip = latest.latest_handoff()
        assert tip['path'] == handoff_ref['path']
        started = time.monotonic()
        last_tip_check = [-1.0]

        def guard():
            assert time.monotonic() - started < 600
            now = time.monotonic()
            if now - last_tip_check[0] >= .5:
                assert latest.latest_handoff() == tip
                last_tip_check[0] = now
            assert (os.fstat(fd).st_dev, os.fstat(fd).st_ino) == (66306, 8294421)
            for p, floor in ((R, 28 * 2**30), (Path('/mnt/nvme'), 30 * 2**30)):
                st = os.statvfs(p)
                assert st.f_bavail * st.f_frsize >= floor + 64 * 2**20
            assert not any(n.split('.')[0] in ('torch', 'numpy', 'neuronxcc', 'torch_neuronx', 'nki') for n in sys.modules)

        seed_reader, reader = saved.JsonMetadata(guard), witnesses.Reader(guard)

        def read(reference):
            exact = {k: reference[k] for k in ('path', 'sha256')}
            return seed_reader.read(exact)['value'] if exact == old_ref else reader(exact)

        old = read(old_ref)
        assert old['history_overlay'] == overlay_seed_ref
        overlay_seed, catalog, handoff = read(overlay_seed_ref), read(catalog_ref), read(handoff_ref)
        delivery, acceptance, native = read(delivery_ref), read(acceptance_ref), read(stage.BUNDLE)
        old_catalog = read(old['current_serial_catalog'])
        base = read(overlay_seed['base_plan'])
        assert len(base['inputs']) == 8791
        assert acceptance['source_accepted'] and acceptance['blockers'] == []
        assert {k: acceptance['delivery'][k] for k in ('path', 'sha256')} == delivery_ref
        assert not handoff['running'] and handoff['active_serial'] is None
        assert handoff['live_tool_sessions'] == handoff['active_heavy_units'] == []
        saved.catalog_shape(old_catalog)
        saved.catalog_shape(catalog)
        assert catalog['runs'][:len(old_catalog['runs'])] == old_catalog['runs']
        assert catalog['completed_sdk'] == old_catalog['completed_sdk'] and catalog['current_handoff'] == handoff_ref
        assert len(catalog['runs']) == 145
        for row in catalog['runs']:
            read(row['terminal'])
        native_maps = [
            {row['path']: row['sha256'] for row in native['worker_sources']},
            {str(Path(native['checkout']) / p): sha for p, sha in native['package_sources'].items()},
            {str(Path(native['site_packages']) / p.split('/', 1)[1]): sha for p, sha in native['sdk_sources'].items()},
            {row['path']: row['sha256'] for row in native['tokenizer']['sources'].values()},
        ]
        pins = contract.merge_sources(old['source_sha256'], *[r['source_sha256'] for r in catalog['runs'] if r['source_sha256']],
                                      delivery['static_local_source_sha256'], *native_maps)
        assert len(pins) == 48154
        pins = contract.merge_sources(pins, {str(Q / 'prepare_qwen35_sampler_r6_candidate_v1.py'): 'ee529887d9b81bf3f506be1eeafe12f8161235afe6634be59b7672b640529b6e', str(Q / 'prepare_qwen35_sampler_v9_candidate_v3.py'): '56ffe488298721a35b2bc6a5aa49557580e0fd3c7f48b078b2eda1f1235159c0'})
        for r in (delivery_ref, acceptance_ref, stage.BUNDLE, catalog_ref, handoff_ref):
            pins = contract.merge_sources(pins, {r['path']: r['sha256']})
        for r in [native['official_config'], native['source_delivery'], *native['tokenizer']['sources'].values()]:
            if Path(r['path']).suffix == '.json':
                read(r)
        print('source maps merged; resolving complete historical witnesses', flush=True)
        fresh = witnesses.current_history_freshness(catalog_ref, pins, read, guard)
        cache = {}
        for row in catalog['runs']:
            terminal = read(row['terminal'])
            if terminal.get('schema') in ('minicpm5_native_model_exit_recorder_actual_exit_v1', 'minicpm5_native_independent_audit_actual_exit_v1'):
                pins = contract.merge_sources(pins, witnesses.executed_extension(terminal, row, read, guard, cache))
        print('complete historical source witnesses resolved', flush=True)
        lfm = supplement(read, old['lfm_r2_scope'], catalog)
        assert len(lfm['scope']['children']) == 874 and len(lfm['scope']['predecessor_owners']) == 876
        assert witnesses.current_history_freshness(catalog_ref, pins, read, guard) == fresh
        recovered = {p: sha for p, sha in old['source_sha256'].items() if p not in overlay_seed['source_extension']}
        assert len(recovered) == 44532
        observed = copy.deepcopy(reader.inputs)
        objects.merge_inputs(observed, seed_reader.observed)
        inherited = copy.deepcopy(base['inputs'])
        for additions in (old['input_extension'], overlay_seed['input_extension'], observed):
            objects.merge_inputs(inherited, additions)
        ordinary = {p: row for p, row in observed.items() if row['identity']['size'] <= 8 * 2**20}
        preparation_only = {p: row for p, row in observed.items() if p not in ordinary}
        assert old_ref['path'] in preparation_only
        prefix = 'qwen35-package-preflight-r1b-20261112'
        run_name = 'package-r1b-v2-20261112'
        paths = {k: str(Q / (prefix + '-' + k + '-v2.json')) for k in ('history-overlay', 'spec', 'review')}
        running = str(Q / 'continuation-handoff-20261112-v559-qwen-package-r1-running.json')
        overlay_root = '/mnt/nvme/minicpm5-layer1-operator-native-qwen-package-r1b-v2-20261112'
        out = '/mnt/nvme/neuron-work/qwen35-package-preflight-' + run_name
        assert all(not Path(p).exists() for p in [*paths.values(), running, overlay_root, out])
        overlay = copy.deepcopy(overlay_seed)
        overlay.update(current_serial_catalog=catalog_ref, output_root=overlay_root,
                       source_extension={p: sha for p, sha in pins.items() if p not in recovered})
        objects.merge_inputs(overlay['input_extension'], {p: row for p, row in ordinary.items() if p not in base['inputs']})
        assert not set(overlay['input_extension']) & set(base['inputs'])
        spec_inputs = copy.deepcopy(old['input_extension'])
        objects.merge_inputs(spec_inputs, ordinary)
        overlay_ref = objects.object_ref(paths['history-overlay'], overlay, ordinary=True)
        spec = dict(schema='qwen35_package_preflight_serial_spec_v2', run_name=run_name, output_root=out,
            history_overlay=overlay_ref, current_serial_catalog=catalog_ref, source_sha256=pins, input_extension=spec_inputs,
            running_handoff_path=running, lfm_r2_scope=old['lfm_r2_scope'], bundle=stage.BUNDLE, limits=stage.LIMITS,
            lock_identity=stage.LOCK_IDENTITY, native_python=stage.PYTHON, automatic_successor=False)
        spec_ref = objects.object_ref(paths['spec'], spec)
        review = dict(schema='qwen35_sampler_runtime_source_review_v1', reviewer='/root', passed=True, blockers=[],
                      spec=spec_ref, source_sha256=pins, launch_authorized=True)
        review_ref = objects.object_ref(paths['review'], review)
        stage.validate(spec, review, spec_ref)
        reader.unchanged()
        assert latest.latest_handoff() == tip
        for key, value in (('history-overlay', overlay), ('spec', spec), ('review', review)):
            data = objects.encode(value)
            with open(paths[key], 'xb') as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            assert io.ref(paths[key], 64 * 2**20) == objects.object_ref(paths[key], value)
        print(json.dumps(dict(stage='bound_metadata', spec=spec_ref, review=review_ref, sources=len(pins), inputs=len(spec_inputs))), flush=True)
        assert stage.load_spec(spec_ref, review_ref) == spec
        bound = History(spec, spec_ref, review_ref, stage.names(spec), guard).resolve()
        bound.unchanged()
        assert bound.freshness() == fresh and len(bound.supplement['scope']['children']) == 874
        io.loaded_local_sources(pins)
        assert latest.latest_handoff() == tip
        receipt = dict(schema='qwen35_package_preflight_root_concrete_binding_v1', reviewer='/root', passed=True,
            observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), spec=spec_ref, review=review_ref,
            history_overlay=overlay_ref, catalog=catalog_ref, idle_handoff=handoff_ref, source_acceptance=acceptance_ref,
            source_files=len(pins), input_extension_count=len(spec_inputs), catalog_rows=len(catalog['runs']),
            full_source_history_resolution_passed=True, lfm_children=874, original_failed_statuses_preserved=True,
            native_executed=False, physical_pair_performed=False, preparation_only_inputs=preparation_only,
            runtime_launched=False, full_model_qualified=False)
        receipt_path = Q / (prefix + '-root-binding-v2.json')
        with receipt_path.open('xb') as stream:
            stream.write(objects.encode(receipt))
        print(json.dumps(dict(binding=io.ref(receipt_path), spec=spec_ref, review=review_ref)), flush=True)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


if __name__ == '__main__':
    main()
