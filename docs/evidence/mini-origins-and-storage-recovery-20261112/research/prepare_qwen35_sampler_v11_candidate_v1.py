"""Add storage failure observations to an explicit complete v10 candidate.

Only JSON/source metadata and in-memory candidate objects are used. The caller
selects current history and fresh paths. No input is written or launch approved.
"""
import copy
from pathlib import Path

import prepare_qwen35_sampler_v10_candidate_v1 as previous
from qwen35_sampler_runtime_contract_v11 import LIMITS, Q, ipc_paths, merge_sources, reference, require

NAMES = tuple('qwen35_sampler_runtime_' + part + '_v11.py'
              for part in ('contract', 'io', 'history', 'service', 'adapter')) + ('run_qwen35_sampler_runtime_v11.py',)


def prepare(future, short_ipc_ref, successor_ref, read_metadata, guard):
    """Original v10 preparation, then six additive v11 pins and one manifest."""
    ipc_paths(future)
    reference(short_ipc_ref); reference(successor_ref)
    require(Path(successor_ref['path']).parent == Q and Path(successor_ref['path']).suffix == '.json',
            'Explicit bounded source-manifest JSON')
    require(successor_ref['path'] not in {short_ipc_ref['path'], future['successor_sources']['path'],
            *(future[key] for key in ('running_handoff_path', 'overlay_path', 'spec_path', 'review_candidate_path'))},
            'New source manifest is distinct from inherited inputs and candidate outputs')
    guard(); saved = read_metadata(successor_ref)
    require(type(saved) is dict and set(saved) == {'value', 'identity'}
            and type(saved['identity']) is dict and set(saved['identity']) == previous.previous.IDENTITY_KEYS
            and saved['identity']['realpath'] == successor_ref['path']
            and all(type(saved['identity'][key]) is int and saved['identity'][key] >= 0
                    for key in previous.previous.IDENTITY_KEYS - {'realpath'})
            and saved['identity']['size'] <= LIMITS['metadata_bytes'],
            'Stable full source-manifest identity within unchanged ordinary metadata cap')
    manifest = saved['value']
    require(type(manifest) is dict and set(manifest) == {'schema', 'previous_source_manifest', 'source_files', 'limits'}
            and manifest['schema'] == 'qwen35_sampler_storage_diagnostic_v11_sources_v1'
            and manifest['previous_source_manifest'] == short_ipc_ref
            and set(manifest['source_files']) == set(NAMES)
            and manifest['limits'] == LIMITS,
            'Exact additive storage diagnostic runtime and unchanged limits')
    additions = {}
    for name, ref in manifest['source_files'].items():
        reference(ref); require(ref['path'] == str(Q/name), 'Exact v11 source path')
        additions[ref['path']] = ref['sha256']
    guard()
    result = previous.prepare(future, short_ipc_ref, read_metadata, guard)
    artifacts = result['artifacts']
    spec, overlay, review = (artifacts[future[key]] for key in ('spec_path', 'overlay_path', 'review_candidate_path'))
    old_pins = copy.deepcopy(spec['source_sha256'])
    pins = merge_sources(old_pins, additions)
    row = {successor_ref['path']: dict(sha256=successor_ref['sha256'], identity=copy.deepcopy(saved['identity']))}
    previous.previous.merge_inputs(spec['input_extension'], row)
    previous.previous.merge_inputs(overlay['input_extension'], row)
    overlay['source_extension'] = merge_sources(overlay['source_extension'], additions)
    spec.update(history_overlay=previous.previous.object_ref(future['overlay_path'], overlay, ordinary=True),
                source_sha256=pins)
    paths = ipc_paths(spec)
    review.update(spec=previous.previous.object_ref(future['spec_path'], spec), source_sha256=copy.deepcopy(pins))
    require(review['passed'] is False and review['launch_authorized'] is False,
            'Source preparation never creates root acceptance or admission')
    result['references'] = {key: previous.previous.object_ref(future[key], artifacts[future[key]])
                            for key in ('overlay_path', 'spec_path', 'review_candidate_path')}
    result['schema'] = 'qwen35_sampler_v11_metadata_candidate_v1'
    result['evidence'].update(full_candidate_sources=len(pins), previous_candidate_sources=len(old_pins),
        all_previous_candidate_sources_retained=True, storage_diagnostic_manifest=copy.deepcopy(successor_ref),
        ipc_path_preflight=paths, runtime_admitted=False, launch_authorized=False)
    guard()
    return result


if __name__ == '__main__':
    raise RuntimeError('Explicit future metadata API only; no implicit preparation or launch')
