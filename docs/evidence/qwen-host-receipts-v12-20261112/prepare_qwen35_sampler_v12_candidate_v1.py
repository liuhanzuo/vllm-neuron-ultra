"""Add observed host settings and strict usage accounting to a v11 candidate.

Explicit metadata API only; preserves all old sources/inputs, selects no
current history, writes no plan/review and grants no runtime admission.
"""
import copy
from pathlib import Path

import prepare_qwen35_sampler_v11_candidate_v1 as previous
from qwen35_sampler_runtime_contract_v12 import LIMITS, Q, ipc_paths, merge_sources, reference, require

legacy = previous.previous.previous
NAMES = tuple('qwen35_sampler_runtime_' + part + '_v12.py'
              for part in ('contract', 'io', 'history', 'service', 'adapter')) + (
    'run_qwen35_sampler_runtime_v12.py', 'qwen35_sampler_pair_driver_v3.py',
    'qwen35_sampler_host_settings_v1.py')
USAGE_FIX = dict(path=str(Q/'qwen35_sampler_pair_driver_v2.py'),
    sha256='b4c8a4451acfd12ee2a8d0254f1e3c94f32199c18b76ed502a1b62e08c74052f')


def prepare(future, short_ipc_ref, storage_ref, successor_ref, read_metadata, guard):
    """Retain complete v11 preparation, then add eight source pins in memory."""
    ipc_paths(future)
    for ref in (short_ipc_ref, storage_ref, successor_ref):
        reference(ref)
    require(Path(successor_ref['path']).parent == Q and Path(successor_ref['path']).suffix == '.json',
            'Explicit bounded source-manifest JSON')
    require(successor_ref['path'] not in {short_ipc_ref['path'], storage_ref['path'],
            future['successor_sources']['path'], *(future[key] for key in (
                'running_handoff_path', 'overlay_path', 'spec_path', 'review_candidate_path'))},
            'New source manifest distinct from all inherited inputs and outputs')
    guard(); saved = read_metadata(successor_ref)
    require(type(saved) is dict and set(saved) == {'value', 'identity'}
            and type(saved['identity']) is dict and set(saved['identity']) == legacy.IDENTITY_KEYS
            and saved['identity']['realpath'] == successor_ref['path']
            and all(type(saved['identity'][key]) is int and saved['identity'][key] >= 0
                    for key in legacy.IDENTITY_KEYS - {'realpath'})
            and saved['identity']['size'] <= LIMITS['metadata_bytes'],
            'Stable full source-manifest identity within original metadata cap')
    manifest = saved['value']
    require(type(manifest) is dict and set(manifest) == {
                'schema', 'previous_source_manifest', 'source_files', 'limits', 'initial_usage_source'}
            and manifest['schema'] == 'qwen35_sampler_observed_host_v12_sources_v1'
            and manifest['previous_source_manifest'] == storage_ref
            and manifest['initial_usage_source'] == USAGE_FIX
            and set(manifest['source_files']) == set(NAMES) and manifest['limits'] == LIMITS,
            'Exact additive host/settings successor and original limits')
    additions = {}
    for name, ref in manifest['source_files'].items():
        reference(ref); require(ref['path'] == str(Q/name), 'Exact v12 source path')
        additions[ref['path']] = ref['sha256']
    guard()
    result = previous.prepare(future, short_ipc_ref, storage_ref, read_metadata, guard)
    artifacts = result['artifacts']
    spec, overlay, review = (artifacts[future[key]] for key in ('spec_path', 'overlay_path', 'review_candidate_path'))
    old_pins = copy.deepcopy(spec['source_sha256'])
    pins = merge_sources(old_pins, additions)
    row = {successor_ref['path']: dict(sha256=successor_ref['sha256'], identity=copy.deepcopy(saved['identity']))}
    legacy.merge_inputs(spec['input_extension'], row)
    legacy.merge_inputs(overlay['input_extension'], row)
    overlay['source_extension'] = merge_sources(overlay['source_extension'], additions)
    spec.update(history_overlay=legacy.object_ref(future['overlay_path'], overlay, ordinary=True),
                source_sha256=pins)
    paths = ipc_paths(spec)
    review.update(spec=legacy.object_ref(future['spec_path'], spec), source_sha256=copy.deepcopy(pins))
    require(review['passed'] is False and review['launch_authorized'] is False,
            'Source preparation cannot create root acceptance or runtime admission')
    result['references'] = {key: legacy.object_ref(future[key], artifacts[future[key]])
                            for key in ('overlay_path', 'spec_path', 'review_candidate_path')}
    result['schema'] = 'qwen35_sampler_v12_metadata_candidate_v1'
    result['evidence'].update(full_candidate_sources=len(pins), previous_candidate_sources=len(old_pins),
        all_previous_candidate_sources_retained=True, observed_host_manifest=copy.deepcopy(successor_ref),
        ipc_path_preflight=paths, runtime_admitted=False, launch_authorized=False)
    guard()
    return result


if __name__ == '__main__':
    raise RuntimeError('Explicit future metadata API only; no implicit preparation or launch')
