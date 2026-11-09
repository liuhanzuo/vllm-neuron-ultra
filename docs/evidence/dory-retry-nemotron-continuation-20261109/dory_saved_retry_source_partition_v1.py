"""Exact query-native dependency partition for future post-retry consumers.

Keep all original 14+6 historical payload pins and three non-UTF8 fixtures.
Only the nine observed query native libraries receive the additional binary
category. Classification reads JSON metadata only; authentication is explicit.
"""
import copy
from pathlib import Path

import dory_saved_review_retry_contract_v1 as retry
import nemotron9b_residual_source_partition_v1 as previous
from dory_service_query_auth_v1 import authenticated_bytes

require = previous.require
OBSERVATION = retry.QUERY_OBSERVATION
NATIVE_BYTES = 3920408


def native_pins(observation):
    require(observation['schema'] == 'dory_service_query_dependency_observation_v1'
            and observation['complete_observation'] is True
            and len(observation['source_and_binary_pins']) == 41
            and observation['source_and_binary_bytes_hashed'] == 4326816,
            'Exact complete recorded query dependency closure')
    mappings, pins = observation['new_native_file_mappings'], observation['source_and_binary_pins']
    require(len(mappings) == 9 and all(p in pins for p in mappings)
            and sum(pins[p]['bytes'] for p in mappings) == NATIVE_BYTES,
            'Exactly nine original observed native libraries and byte count')
    require(all(mappings[p] == {k:pins[p][k] for k in ('device','inode')} for p in mappings),
            'Native mappings retain their byte-authentication file identities')
    return {p:pins[p]['sha256'] for p in mappings}


def classify_maps(spec, catalog, parent, start, original_parts, observation):
    pins = spec['source_sha256']
    previous.original.pin_shape(pins)
    require(spec['required_saved_retry_query_observation'] == OBSERVATION
            and pins.get(OBSERVATION['path']) == OBSERVATION['sha256'],
            'Exact recorded query dependency observation is source-pinned')
    native = native_pins(observation)
    require(all(pins.get(p) == row['sha256'] for p,row in observation['source_and_binary_pins'].items()),
            'All41 query dependency pins retained without rebinding')
    require(not (set(native)&set(parent['source_sha256']))
            and not (set(native)&set(start['spec']['source_sha256'])),
            'New query libraries cannot reclassify any original source or evidence')
    projected = copy.deepcopy(spec)
    projected['source_sha256'] = {p:s for p,s in pins.items() if p not in native}
    original = previous.classify_maps(projected,catalog,parent,start,original_parts)
    require(not any('.so.' in Path(p).name or Path(p).suffix == '.so'
                    for p in original['text_sha256']),
            'Unreviewed native binary cannot enter the ordinary text category')
    result = dict(original, query_native_binary_sha256=native)
    combined = {}
    for part in result.values():
        require(not set(combined)&set(part), 'Every source category is disjoint')
        combined.update(part)
    require(combined == pins, 'All original and query dependency pins retained losslessly')
    return result


def classify_sources(spec,catalog):
    parent = previous.original.load(previous.PARENT)
    previous.original.classify_sources(parent,catalog)
    parts = previous.original.history.partition(catalog)
    reader = retry.original.Reader()
    start = retry.original.anchor(reader)
    observation = reader(OBSERVATION)
    return classify_maps(spec,catalog,parent,start,parts,observation)


def authenticate_native_sources(pins, observation, guard):
    """Explicit bounded native-file hashing, not ordinary text or tensor IO."""
    expected = native_pins(observation)
    require(pins == expected, 'Only the exact nine observed native libraries may be opened')
    total = 0
    for path in sorted(pins):
        guard()
        total += len(authenticated_bytes(path,observation['source_and_binary_pins'][path]))
    require(total == NATIVE_BYTES, 'Complete exact query native dependency bytes')
    return dict(complete=True,files=9,bytes=total,native_libraries_loaded_here=False,
                model_or_tensor_bytes_read=0,full_runtime_environment_verified=False)


if __name__ == '__main__':
    raise SystemExit('Source partition only; no model admission or automatic hashing')
