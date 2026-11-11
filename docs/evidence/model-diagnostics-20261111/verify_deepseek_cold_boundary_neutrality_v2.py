"""Callable full raw comparison after admitted capture; no launcher/CLI.

Original and captured logits must match byte-for-byte, as must every original
native/device full-history KV tensor. Caller-admitted SHA256 values bind all four raw-logit files and outer metadata before deserialization.
The original 14 failures remain failures. This never qualifies model performance.
"""
import hashlib
import json
from pathlib import Path

import deepseek_cpu_capture_bridge_v1 as b


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def admitted_digest(path, admitted_sha256):
    key = str(Path(path).resolve())
    expected = admitted_sha256.get(key)
    b.require(isinstance(expected, str) and len(expected) == 64, 'missing caller admission hash: ' + key)
    actual = digest(path)
    b.require(actual == expected, 'caller admission hash changed: ' + key)
    return actual


def pinned_read(path, admitted_sha256):
    admitted_digest(path, admitted_sha256)
    return read(path)


def _join_cache_frames(original_manifest, captured_manifest, backend):
    """Pure metadata pairing; whole sequences must agree before any payload IO."""
    import deepseek_cold_kv_contract_v2 as original
    old = original_manifest['frames']; new = captured_manifest['frames']
    expected = original.frames(backend)
    b.require(original_manifest['complete'] and captured_manifest['complete'], 'complete captures required')
    b.require([x['frame'] for x in old] == [x['frame'] for x in new] == expected,
              'no omitted/reordered original native42/device84 cache frames')
    return list(zip(old, new))


def _compare_cache_backend(original_root, destination, backend, admitted_sha256):
    import torch
    from deepseek_cold_kv_codec_v2 import decode, same
    import deepseek_cold_kv_contract_v2 as original
    old_manifest = pinned_read(original_root / 'cache-captures' / backend / 'manifest.json', admitted_sha256)
    new_manifest = pinned_read(destination / 'boundary-captures' / backend / 'manifest.json', admitted_sha256)
    pairs = _join_cache_frames(old_manifest, new_manifest, backend)
    old_previous = {}; new_previous = {}; checked = 0
    for old_entry, new_record in pairs:
        old_path = Path(old_entry['path'])
        b.require(digest(old_path) == old_entry['sha256'], 'original frame metadata hash')
        old_record = read(old_path); frame = old_entry['frame']
        original.check_frame(old_record, backend, frame)
        b.require(old_record['frame'] == new_record['frame'], 'paired frame')
        old_payload = Path(old_record['payload']); new_payload = Path(new_record['payload']['path'])
        b.require(digest(old_payload) == old_record['payload_file_sha256'], 'original raw KV payload hash')
        b.require(digest(new_payload) == new_record['payload']['sha256'], 'captured raw KV payload hash')
        old_values = torch.load(old_payload, map_location='cpu', weights_only=True)
        new_values = torch.load(new_payload, map_location='cpu', weights_only=True)
        names = original.stages()
        b.require(list(old_values) == list(new_values) == names, 'all72 KV tensors in original order')
        b.require([x['stage'] for x in new_record['rows']] == names, 'new stage identities')
        lineage = (frame['prefix'], frame.get('slot', frame['fixture']))
        a = old_previous.get(lineage); z = new_previous.get(lineage)
        b.require((a is None) == (z is None) == (frame['step'] == 0), 'cache chronology')
        next_a = {}; next_z = {}
        for left, right in zip(old_record['rows'], new_record['rows']):
            name = left['stage']
            original.check_record(right, frame, backend, name)
            va = decode(left, old_values[name], None if a is None else a[name])
            vz = decode(right, new_values[name], None if z is None else z[name])
            b.require(same(va, vz), 'capture changed full KV bytes: ' + str(frame) + ' ' + name)
            b.require(left['full_sha256'] == right['full_sha256'], 'full KV hash identity')
            next_a[name] = va; next_z[name] = vz; checked += 1
        if frame['step'] == 2:
            old_previous.pop(lineage, None); new_previous.pop(lineage, None)
        else:
            old_previous[lineage] = next_a; new_previous[lineage] = next_z
    b.require(not old_previous and not new_previous, 'all histories completed')
    b.require(checked == (3024 if backend == 'native' else 6048), 'complete cache denominator')
    return checked


def verify(original_root, destination, reference_directory, *, admitted_sha256):
    import torch
    from deepseek_cold_kv_codec_v2 import same
    from deepseek_cold_kv_protocol_v2 import check_copy, check_lifecycle
    original_root, destination, reference_directory = map(Path, (original_root, destination, reference_directory))
    b.require(original_root.resolve() != destination.resolve(), 'original evidence is immutable')
    result_path = destination / 'capture-neutrality.json'
    b.require(not result_path.exists(), 'refuse overwrite')
    # This mapping comes from the owning worker's prior admission snapshot;
    # do not generate it opportunistically from the files being verified.
    raw_paths = [original_root / 'cold-reference/cold-reference.pt',
                 reference_directory / 'cold-reference.pt',
                 original_root / 'cold-buckets.pt', destination / 'cold-buckets.pt']
    admitted_raw = {str(path.resolve()): admitted_digest(path, admitted_sha256) for path in raw_paths}
    expected_original = pinned_read(original_root / 'result.json', admitted_sha256)
    b.require(expected_original['diagnostic_complete'] and expected_original['all_original126_logits_and84_device_metrics_exact'],
              'completed original diagnostic required')
    b.require(expected_original['original14_failed_frames_preserved'], 'original failures required')
    before = pinned_read(original_root / 'cold-buckets.json', admitted_sha256); after = pinned_read(destination / 'cold-buckets.json', admitted_sha256)
    metadata = b.check_neutrality_metadata(before, after)
    check_copy(pinned_read(destination / 'state-copy.json', admitted_sha256))
    check_lifecycle(pinned_read(destination / 'state-lifecycle.json', admitted_sha256))
    b.require(pinned_read(destination / 'state-lifecycle.json', admitted_sha256)['state_tensors_changed_by_warmup'] == 144,
              'all original warmup transitions required')
    for report in (before, after):
        b.require(report['complete'] and report['all_duplicate_slots_bitwise_equal']
                  and report['all_cache_states_finite'] and report['all_initial_states_zero'], 'original state gates')

    counts = {}
    raw_hashes = {}
    for backend in ('native', 'device'):
        old_path = (original_root / 'cold-reference/cold-reference.pt' if backend == 'native'
                    else original_root / 'cold-buckets.pt')
        new_path = (reference_directory / 'cold-reference.pt' if backend == 'native'
                    else destination / 'cold-buckets.pt')
        raw_hashes[backend] = dict(original=admitted_raw[str(old_path.resolve())], captured=admitted_raw[str(new_path.resolve())])
        reproduction = original_root / ('native42-reproduction.json' if backend == 'native' else 'device84-reproduction.json')
        b.require(raw_hashes[backend]['original'] == pinned_read(reproduction, admitted_sha256)['new_sha256'], 'frozen original raw-logit reproduction hash')
        if backend == 'native':
            b.require(raw_hashes[backend]['original'] == pinned_read(old_path.with_suffix('.json'), admitted_sha256)['tensor_sha256'], 'old native logits hash')
            b.require(raw_hashes[backend]['captured'] == pinned_read(new_path.with_suffix('.json'), admitted_sha256)['tensor_sha256'], 'new native logits hash')
        else:
            b.require(raw_hashes[backend]['captured'] == after['raw_logits_sha256'], 'new device logits hash')
        a = torch.load(old_path, map_location='cpu', weights_only=True)
        z = torch.load(new_path, map_location='cpu', weights_only=True)
        b.require(len(a) == len(z) == (14 if backend == 'native' else 84), 'complete original raw logits')
        for left, right in zip(a, z):
            b.require(set(left) == set(right), 'logit metadata keys')
            b.require(all(left[k] == right[k] for k in left if k != 'logits'), 'same positions, fixtures and slots')
            b.require(left['logits'].dtype == right['logits'].dtype == torch.float32, 'original logits dtype')
            b.require(tuple(left['logits'].shape) == tuple(right['logits'].shape) ==
                      ((3, 151936) if backend == 'native' else (151936,)), 'complete original vocabulary')
            b.require(same(left['logits'], right['logits']), 'observer changed original raw logits')
        counts[backend] = _compare_cache_backend(original_root, destination, backend, admitted_sha256)
    result = dict(passed=True, metadata=metadata, raw_logits_hashes=raw_hashes,
                  caller_admitted_sha256=dict(admitted_sha256),
                  original_native42_and_device84_raw_logits_exact=True,
                  original_native3024_and_device6048_full_KV_tensors_exact=counts,
                  original_copy_warmup_reset_clear_and_four_slot_controls_retained=True,
                  observation_scope='All original prediction positions, full features and both device TP ranks; full KV histories.',
                  omitted_intermediates=['other prefill token activations', 'attention probabilities', 'MLP gate/up/product'],
                  first_causal_fault_proven=False, full_model_qualified=False, new_speedup_claim=False)
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result
