"""Add the reviewed short-IPC runtime to an explicit complete v9 candidate.

Only authenticated JSON inputs and in-memory candidate objects are used. No
current catalog is selected, historical receipt relabelled, or output written.
The resulting review is deliberately unapproved and cannot admit a runtime.
"""
import copy
from pathlib import Path

import prepare_qwen35_sampler_v9_candidate_v3 as previous
from qwen35_sampler_runtime_contract_v10 import LIMITS, Q, ipc_paths, merge_sources, reference, require

NAMES = tuple('qwen35_sampler_runtime_' + part + '_v10.py'
              for part in ('contract', 'io', 'history', 'service', 'adapter')) + ('run_qwen35_sampler_runtime_v10.py',)
RPC_PATHS = tuple('/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/vllm/' + name
                  for name in ('envs.py', 'utils/__init__.py'))


def prepare(future, successor_ref, read_metadata, guard):
    """Same explicit future/history as v9 plus the separate six-file manifest."""
    ipc_paths(future)
    reference(successor_ref)
    require(Path(successor_ref['path']).parent == Q and Path(successor_ref['path']).suffix == '.json',
            'Explicit bounded source-manifest JSON')
    require(successor_ref['path'] not in {future[key] for key in
            ('running_handoff_path', 'overlay_path', 'spec_path', 'review_candidate_path')},
            'New candidate cannot overwrite its source manifest')
    guard(); saved = read_metadata(successor_ref)
    require(type(saved) is dict and set(saved) == {'value', 'identity'}
            and type(saved['identity']) is dict and set(saved['identity']) == previous.IDENTITY_KEYS
            and saved['identity']['realpath'] == successor_ref['path']
            and all(type(saved['identity'][key]) is int and saved['identity'][key] >= 0
                    for key in previous.IDENTITY_KEYS - {'realpath'})
            and saved['identity']['size'] <= LIMITS['metadata_bytes'],
            'Stable full source-manifest identity within unchanged ordinary metadata cap')
    manifest = saved['value']
    require(type(manifest) is dict and set(manifest) == {'schema', 'previous_source_manifest',
            'source_files', 'rpc_source_sha256', 'limits'}
            and manifest['schema'] == 'qwen35_sampler_short_ipc_v10_sources_v1'
            and manifest['previous_source_manifest'] == future['successor_sources']
            and set(manifest['source_files']) == set(NAMES)
            and set(manifest['rpc_source_sha256']) == set(RPC_PATHS)
            and manifest['limits'] == LIMITS, 'Exact additive short-IPC runtime and unchanged limits')
    additions = {}
    for name, ref in manifest['source_files'].items():
        reference(ref); require(ref['path'] == str(Q/name), 'Exact v10 source path')
        additions[ref['path']] = ref['sha256']
    guard()
    result = previous.prepare(future, read_metadata, guard)
    artifacts = result['artifacts']
    spec, overlay, review = (artifacts[future[key]] for key in ('spec_path', 'overlay_path', 'review_candidate_path'))
    old_pins = copy.deepcopy(spec['source_sha256'])
    require(all(old_pins.get(path) == sha for path, sha in manifest['rpc_source_sha256'].items()),
            'Actual vLLM IPC selector sources already retained in complete historical map')
    pins = merge_sources(old_pins, additions)
    row = {successor_ref['path']: dict(sha256=successor_ref['sha256'], identity=copy.deepcopy(saved['identity']))}
    previous.merge_inputs(spec['input_extension'], row)
    previous.merge_inputs(overlay['input_extension'], row)
    overlay['source_extension'] = merge_sources(overlay['source_extension'], additions)
    overlay_ref = previous.object_ref(future['overlay_path'], overlay, ordinary=True)
    spec.update(history_overlay=overlay_ref, source_sha256=pins)
    paths = ipc_paths(spec)
    spec_ref = previous.object_ref(future['spec_path'], spec)
    review.update(spec=spec_ref, source_sha256=copy.deepcopy(pins))
    require(review['passed'] is False and review['launch_authorized'] is False,
            'Source preparation never creates root acceptance or admission')
    result['references'] = {key: previous.object_ref(future[key], artifacts[future[key]])
                            for key in ('overlay_path', 'spec_path', 'review_candidate_path')}
    result['schema'] = 'qwen35_sampler_v10_metadata_candidate_v1'
    result['evidence'].update(full_candidate_sources=len(pins), previous_candidate_sources=len(old_pins),
        all_previous_candidate_sources_retained=True, short_ipc_manifest=copy.deepcopy(successor_ref),
        ipc_path_preflight=paths, runtime_admitted=False, launch_authorized=False)
    guard()
    return result


if __name__ == '__main__':
    raise RuntimeError('Explicit future metadata API only; no implicit preparation or launch')
