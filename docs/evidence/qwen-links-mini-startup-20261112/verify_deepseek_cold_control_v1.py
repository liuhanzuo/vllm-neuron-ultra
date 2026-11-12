"""Full original-to-fresh observer-off byte reproduction; root-admitted inputs only."""
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
    new_manifest = pinned_read(destination / 'cache-captures' / backend / 'manifest.json', admitted_sha256)
    original.check_manifest(old_manifest, backend); original.check_manifest(new_manifest, backend)
    pairs = _join_cache_frames(old_manifest, new_manifest, backend)
    old_previous = {}; new_previous = {}; checked = 0
    for old_entry, new_entry in pairs:
        old_path = Path(old_entry['path'])
        b.require(digest(old_path) == old_entry['sha256'], 'original frame metadata hash')
        new_path = Path(new_entry['path'])
        b.require(digest(new_path) == new_entry['sha256'], 'fresh control frame metadata hash')
        new_record = read(new_path)
        old_record = read(old_path); frame = old_entry['frame']
        original.check_frame(new_record, backend, frame)
        original.check_frame(old_record, backend, frame)
        b.require(old_record['frame'] == new_record['frame'], 'paired frame')
        old_payload = Path(old_record['payload']); new_payload = Path(new_record['payload'])
        b.require(digest(old_payload) == old_record['payload_file_sha256'], 'original raw KV payload hash')
        b.require(digest(new_payload) == new_record['payload_file_sha256'], 'captured raw KV payload hash')
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


def verify(original_root, destination, *, admitted_sha256, execution_guard):
    """Independently reproduce all original logits/KV, then publish control receipts.

The owning runtime supplies authenticated hashes of both arms' raw logits and
outer metadata/manifests. Each raw KV payload remains bound by its manifest
chain. Hashes are never invented here from the unverified inputs.
"""
    b.require(callable(execution_guard), 'owning serial guard required'); execution_guard()
    import torch
    from deepseek_cold_kv_codec_v2 import same
    from deepseek_cold_kv_protocol_v2 import check_copy, check_lifecycle, check_device_metadata
    original_root, destination = map(Path, (original_root, destination))
    b.require(original_root.resolve() != destination.resolve(), 'preserve original evidence')
    outputs = [destination/n for n in ('result.json', 'native42-reproduction.json', 'device84-reproduction.json')]
    b.require(all(not p.exists() and not p.is_symlink() for p in outputs), 'fresh reproduction receipts required')
    raw = [root/name for root in (original_root, destination)
           for name in ('cold-reference/cold-reference.pt', 'cold-buckets.pt')]
    hashes = {str(p.resolve()): admitted_digest(p, admitted_sha256) for p in raw}
    historical = pinned_read(original_root/'result.json', admitted_sha256)
    b.require(historical['diagnostic_complete'] and historical['all_original126_logits_and84_device_metrics_exact']
              and historical['original14_failed_frames_preserved'], 'completed original full diagnostic required')
    old = pinned_read(original_root/'cold-buckets.json', admitted_sha256)
    new = pinned_read(destination/'cold-buckets.json', admitted_sha256)
    metadata = check_device_metadata(new, old)
    check_copy(pinned_read(destination/'state-copy.json', admitted_sha256))
    lifecycle = pinned_read(destination/'state-lifecycle.json', admitted_sha256); check_lifecycle(lifecycle)
    b.require(lifecycle['state_tensors_changed_by_warmup'] == 144, 'all original warmup transitions')
    counts = {}; receipts = {}
    for backend in ('native', 'device'):
        execution_guard()
        tail = 'cold-reference/cold-reference.pt' if backend == 'native' else 'cold-buckets.pt'
        before, after = original_root/tail, destination/tail
        if backend == 'native':
            ainfo = pinned_read(before.with_suffix('.json'), admitted_sha256)
            zinfo = pinned_read(after.with_suffix('.json'), admitted_sha256)
            for info, path in ((ainfo, before), (zinfo, after)):
                b.require(info['complete'] and info['all_finite'] and info['frames'] == 42, 'full native42 reference')
                b.require(info['tensor_sha256'] == hashes[str(path.resolve())], 'native raw-logit hash')
            for key in ('revision', 'native_source_sha256', 'criteria', 'prefixes'):
                b.require(ainfo[key] == zinfo[key], 'original native metadata: ' + key)
        a = torch.load(before, map_location='cpu', weights_only=True)
        z = torch.load(after, map_location='cpu', weights_only=True)
        b.require(len(a) == len(z) == (14 if backend == 'native' else 84), 'complete original raw-logit rows')
        for left, right in zip(a, z):
            b.require(set(left) == set(right) and all(left[k] == right[k] for k in left if k != 'logits'),
                      'original fixture/slot/order/positions required')
            b.require(left['logits'].dtype == right['logits'].dtype == torch.float32, 'original full logits dtype')
            b.require(tuple(left['logits'].shape) == tuple(right['logits'].shape) ==
                      ((3, 151936) if backend == 'native' else (151936,)), 'full vocabulary and all frames')
            b.require(same(left['logits'], right['logits']), 'observer-off changed original raw logits')
        del a, z
        counts[backend] = _compare_cache_backend(original_root, destination, backend, admitted_sha256)
        receipts[backend] = dict(passed=True, frames=42 if backend == 'native' else 84,
            all_original_raw_logits_exact=True, original_sha256=hashes[str(before.resolve())],
            new_sha256=hashes[str(after.resolve())], all_full_KV_tensors_exact=counts[backend])
    execution_guard()
    result = dict(complete=True, diagnostic_complete=True, all_original126_logits_and84_device_metrics_exact=True,
        original14_failed_frames_preserved=True, original_failed_gate_preserved=True,
        original_copy_warmup_reset_clear_retained=True, compiled_model_unchanged=True,
        metadata_reproduction=metadata, full_KV_reproduction=counts, caller_admitted_sha256=dict(admitted_sha256),
        full_model_qualified=False, new_speedup_claim=False, first_global_causal_difference_proven=False)
    for path, value in zip(outputs, (result, receipts['native'], receipts['device'])):
        with path.open('x') as stream: stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')
    return result
