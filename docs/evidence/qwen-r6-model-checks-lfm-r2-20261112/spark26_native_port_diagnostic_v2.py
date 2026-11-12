"""Future native tensor diagnostic v2; adds original tensor mutation checks.

The serial host must authenticate its real worker/lock and complete runtime,
then supply load_native(). It returns actual torch, safe_open, official modules,
the unchanged v2 tensor module and SDK create_local_weight_qkv. This module does
not grant admission, implement a scheduler, download weights or enable v2 ports.
"""
import hashlib
import math
import os
from pathlib import Path
import stat

import spark26_native_port_diagnostic_contract_v1 as contract


def _identity(path):
    path = Path(path)
    contract.require(path.resolve() == path and all(not p.is_symlink() for p in (path, *path.parents)),
                     'Canonical nonsymlink original weight path')
    s = path.stat()
    contract.require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1, 'Singleton original weight shard')
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


class OriginalWeights:
    """Authenticate complete pinned shards before reading their actual tensors."""
    def __init__(self, root, plan, torch, safe_open, guard):
        self.root, self.plan, self.torch, self.safe_open, self.guard = Path(root), plan, torch, safe_open, guard
        self.verified = {}

    def authenticate_all(self):
        for filename, row in self.plan['checkpoint']['files'].items():
            self.guard()
            path = self.root / filename
            before = _identity(path)
            contract.require(before[2] == row['bytes'], 'Exact original complete shard length')
            digest = hashlib.sha256()
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
            try:
                s = os.fstat(fd)
                contract.require((s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns) == before,
                                 'Original shard open identity')
                count = 0
                while count < row['bytes']:
                    self.guard()
                    block = os.read(fd, min(1024**2, row['bytes'] - count))
                    contract.require(bool(block), 'Complete original shard bytes')
                    digest.update(block); count += len(block)
                contract.require(os.read(fd, 1) == b'', 'No shard trailing bytes')
            finally:
                os.close(fd)
            contract.require(_identity(path) == before and digest.hexdigest() == row['sha256'],
                             'Original official fixed-revision shard LFS hash')
            self.verified[filename] = before
        return {'files': len(self.verified), 'bytes': sum(v['bytes'] for v in self.plan['checkpoint']['files'].values()),
                'original_revision': self.plan['revision'], 'all_original_shard_hashes_match': True}

    def get(self, case):
        self.guard()
        contract.require(case in self.plan['qkv_weights'], 'Only declared original QKV tensor')
        filename = self.plan['checkpoint']['weight_map'][case['name']]
        path = self.root / filename
        contract.require(filename in self.verified and _identity(path) == self.verified[filename],
                         'Complete hash authentication precedes tensor reads')
        with self.safe_open(str(path), framework='pt', device='cpu') as stream:
            expected = {key for key, shard in self.plan['checkpoint']['weight_map'].items() if shard == filename}
            contract.require(set(stream.keys()) == expected, 'Every original shard tensor key retained')
            tensor = stream.get_tensor(case['name'])
        contract.require(_identity(path) == self.verified[filename], 'Original shard unchanged after tensor read')
        contract.require(list(tensor.shape) == case['shape'] and tensor.dtype == self.torch.bfloat16
                         and tensor.device.type == 'cpu', 'Original BF16 QKV shape/dtype/device')
        contract.require(self.torch.isfinite(tensor).all().item(), 'Original QKV weights must be finite')
        self.guard()
        return tensor


def bits_equal(torch, left, right):
    return (left.shape == right.shape and left.dtype == right.dtype
            and torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)))


def numeric_diagnostic(torch, reference, actual):
    contract.require(reference.shape == actual.shape and reference.dtype == actual.dtype == torch.bfloat16,
                     'Unchanged BF16 projection shape/dtype')
    finite = bool(torch.isfinite(reference).all().item() and torch.isfinite(actual).all().item())
    exact = bits_equal(torch, reference, actual)
    if not finite:
        return {'finite': False, 'bitwise_equal': exact, 'max_abs': None, 'mean_abs': None,
                'relative_l2': None, 'diagnostic_only_not_model_gate': True}
    a, b = reference.float(), actual.float()
    error = b - a
    norm = float(torch.linalg.vector_norm(a).item())
    return {'finite': True, 'bitwise_equal': exact,
            'max_abs': float(error.abs().max().item()), 'mean_abs': float(error.abs().mean().item()),
            'relative_l2': None if norm == 0 else float(torch.linalg.vector_norm(error).item()) / norm,
            'diagnostic_only_not_model_gate': True}


def qkv_diagnostic(torch, configuration, modeling, candidate, create_local_weight_qkv,
                   weights, plan, guard, emit):
    results = []
    cfg_dict = contract.original.config()
    cfg = configuration.Spark2_5Config(**cfg_dict)
    for case in plan['qkv_weights']:
        guard(); full = weights.get(case)
        full_before = full.clone()
        # Only these real QKV projections receive original weight tensors.
        # The unused official attention parameters stay meta, never random weights.
        with torch.device('meta'):
            official_projection = modeling.Spark2_5Attention(cfg, layer_idx=case['layer']).q_k_v_proj
            port_projection = candidate.SparkPackedProjection(cfg_dict)
        official_projection.weight = torch.nn.Parameter(full, requires_grad=False)
        port_projection.Wqkv.weight = torch.nn.Parameter(full, requires_grad=False)
        rank_weights = []
        for rank in plan['tp_ranks']:
            local = create_local_weight_qkv(rank, 2, full, 0, 4096, 1024)
            expected = torch.cat((full[rank * 2048:(rank + 1) * 2048],
                                  full[4096 + rank * 512:4096 + (rank + 1) * 512],
                                  full[5120 + rank * 512:5120 + (rank + 1) * 512]), 0)
            contract.require(bits_equal(torch, local, expected), 'Exact original SDK TP2 Q/K/V shard bytes')
            contract.require(bits_equal(torch, full, full_before), 'SDK sharding must preserve original QKV weight')
            rank_weights.append(local)
        for shape in plan['projection_shapes']:
            guard()
            x = ((torch.arange(math.prod(shape), dtype=torch.int64) + case['layer'] * 7)
                 .remainder(29).sub(14).to(torch.bfloat16).div(64).reshape(shape))
            original = official_projection(x)
            whole = (original[..., :4096], original[..., 4096:5120], original[..., 5120:])
            native_port = port_projection(x)
            contract.require(native_port[3] is None, 'Original v2 packed projection return contract')
            packed_metrics = {role: numeric_diagnostic(torch, ref, actual)
                              for role, ref, actual in zip(('Q', 'K', 'V'), whole, native_port[:3])}
            for rank, local in enumerate(rank_weights):
                actual = torch.nn.functional.linear(x, local)
                parts = torch.tensor_split(actual, (2048, 2560), dim=-1)
                metrics = {}
                for role, width, ref, got in zip(('Q', 'K', 'V'), (2048, 512, 512), whole, parts):
                    metrics[role] = numeric_diagnostic(torch, ref[..., rank * width:(rank + 1) * width], got)
                row = {'layer': case['layer'], 'weight': case['name'], 'shard': case['shard'],
                       'rank': rank, 'input_shape': list(shape), 'weight_partition_exact': True,
                       'metrics': metrics, 'v2_packed_metrics': packed_metrics, 'synthetic_hidden_input': True,
                       'distributed_collective_executed': False}
                results.append(row); emit('qkv_projection', row)
        contract.require(bits_equal(torch, full, full_before), 'Projection must preserve original QKV weights')
        guard()
    contract.require(len(results) == 36 * 3 * 2, 'All original-layer projection diagnostics completed')
    return results


def cache_codes(torch, start, length, layer, slot, epoch, value=False):
    # Exactly representable BF16 codes identify time, original head, slot and reuse.
    positions = torch.arange(start, start + length, dtype=torch.int64)
    result = torch.zeros((1, 4, length, 256), dtype=torch.bfloat16)
    result[..., 0] = positions.remainder(128).to(torch.bfloat16)
    result[..., 1] = torch.div(positions, 128, rounding_mode='floor').to(torch.bfloat16)
    result[..., 2] = layer; result[..., 3] = slot; result[..., 4] = epoch
    for head in range(4):
        result[:, head, :, 5] = head
    result[..., 6:] = torch.arange(250).remainder(128).to(torch.bfloat16)
    return -result if value else result


def stored_physical(torch, stored, cap, seen, rank, sliding):
    start = max(0, seen - 511) if sliding else 0
    expected = torch.zeros((1, 2, cap, 256), dtype=torch.bfloat16)
    positions = torch.arange(start, seen, dtype=torch.long)
    physical = positions.remainder(cap) if sliding else positions
    indices = physical[None, None, :, None].expand(1, 2, len(positions), 256)
    return torch.scatter(expected, 2, indices, stored[:, rank * 2:(rank + 1) * 2])


def cache_diagnostic(torch, configuration, modeling, candidate, plan, guard, emit):
    cfg = configuration.Spark2_5Config(**contract.original.config())
    cfg._attn_implementation = 'eager'
    totals = []
    for spec in plan['cache_layers']:
        layer, cap = spec['layer'], spec['capacity']
        sliding = spec['kind'] == 'sliding_attention'
        for rank in plan['tp_ranks']:
            guard()
            old_k = torch.zeros((4, 2, cap, 256), dtype=torch.bfloat16)
            old_v = torch.zeros_like(old_k)
            references, seen = {}, {}
            for step, event in enumerate(plan['cache_trace']):
                guard()
                before_k, before_v = old_k.clone(), old_v.clone()
                expected_k, expected_v = old_k.clone(), old_v.clone()
                slots = event['slots']
                if event['op'] == 'prefill':
                    slot, length, epoch = slots[0], event['lengths'][0], event['epochs'][0]
                    references[slot] = modeling.DynamicCache(config=cfg)
                    seen[slot] = length
                    k = cache_codes(torch, 0, length, layer, slot, epoch)
                    v = cache_codes(torch, 0, length, layer, slot, epoch, True)
                    input_k, input_v = k.clone(), v.clone()
                    references[slot].update(k, v, layer, {'cache_position': torch.arange(length)})
                    ids, lengths = torch.tensor([slot], dtype=torch.int32), torch.tensor([length], dtype=torch.int32)
                    new_k, new_v = candidate.prefill_cache(k[:, rank * 2:(rank + 1) * 2],
                        v[:, rank * 2:(rank + 1) * 2], old_k, old_v, ids, lengths, sliding)
                    # New buffers are a tensor-level cold comparison, not a fresh-process device test.
                    fresh_k, fresh_v = candidate.prefill_cache(k[:, rank * 2:(rank + 1) * 2],
                        v[:, rank * 2:(rank + 1) * 2], torch.zeros_like(old_k), torch.zeros_like(old_v), ids, lengths, sliding)
                    contract.require(bits_equal(torch, fresh_k[slot], new_k[slot])
                                     and bits_equal(torch, fresh_v[slot], new_v[slot]), 'Reused row matches fresh allocated cache')
                    contract.require(bits_equal(torch, k, input_k) and bits_equal(torch, v, input_v),
                                     'Original prefill K/V inputs unchanged')
                else:
                    # Emulate the documented physical B4 input; compare with independent official B1 caches.
                    by_slot = {slot: (pos, epoch) for slot, pos, epoch in zip(slots, event['positions'], event['epochs'])}
                    first = event['positions'][slots.index(min(slots))]
                    ids = torch.tensor([s if s in by_slot else -1 for s in range(4)], dtype=torch.int32)
                    positions = torch.tensor([[by_slot.get(s, (first, 0))[0]] for s in range(4)], dtype=torch.int32)
                    ks, vs, official = [], [], {}
                    for slot in range(4):
                        if slot in by_slot:
                            pos, epoch = by_slot[slot]
                            contract.require(seen[slot] == pos, 'Exact original per-slot history')
                            k = cache_codes(torch, pos, 1, layer, slot, epoch)
                            v = cache_codes(torch, pos, 1, layer, slot, epoch, True)
                            input_k, input_v = k.clone(), v.clone()
                            official[slot] = references[slot].update(k, v, layer, {'cache_position': torch.tensor([pos])})
                            contract.require(bits_equal(torch, k, input_k) and bits_equal(torch, v, input_v),
                                             'Official decode K/V inputs unchanged')
                            seen[slot] += 1
                        else:
                            k = torch.full((1, 4, 1, 256), -128, dtype=torch.bfloat16)
                            v = torch.full_like(k, 127)
                        ks.append(k[:, rank * 2:(rank + 1) * 2]); vs.append(v[:, rank * 2:(rank + 1) * 2])
                    batch_k, batch_v = torch.cat(ks, 0), torch.cat(vs, 0)
                    input_k, input_v = batch_k.clone(), batch_v.clone()
                    keys, values, key_positions, valid, new_k, new_v = candidate.decode_cache(
                        batch_k, batch_v, old_k, old_v, ids, positions, sliding)
                    contract.require(bits_equal(torch, batch_k, input_k) and bits_equal(torch, batch_v, input_v),
                                     'Candidate decode K/V inputs unchanged')
                    for slot in slots:
                        pos = by_slot[slot][0]
                        absolute = list(range(pos - cap, pos)) if sliding else list(range(cap))
                        expected_valid = torch.tensor([0 <= x < pos for x in absolute] + [True], dtype=torch.bool)
                        contract.require(torch.equal(key_positions[slot], torch.tensor(absolute + [pos]))
                                         and torch.equal(valid[slot], expected_valid), 'Independent chronological positions and validity')
                        rk, rv = official[slot]
                        contract.require(bits_equal(torch, keys[slot:slot + 1, :, expected_valid], rk[:, rank * 2:(rank + 1) * 2])
                                         and bits_equal(torch, values[slot:slot + 1, :, expected_valid], rv[:, rank * 2:(rank + 1) * 2]),
                                         'Actual v2 chronological read matches official DynamicCache')
                for slot in slots:
                    reference = references[slot].layers[layer]
                    contract.require(reference.get_seq_length() == seen[slot], 'Official cumulative cache position')
                    expected_k[slot:slot + 1] = stored_physical(torch, reference.keys, cap, seen[slot], rank, sliding)
                    expected_v[slot:slot + 1] = stored_physical(torch, reference.values, cap, seen[slot], rank, sliding)
                contract.require(bits_equal(torch, old_k, before_k) and bits_equal(torch, old_v, before_v),
                                 'Input physical caches unchanged')
                contract.require(bits_equal(torch, new_k, expected_k) and bits_equal(torch, new_v, expected_v),
                                 'Exact all-slot persistent bytes; dummy and inactive slots preserved')
                old_k, old_v = new_k, new_v
                emit('cache_step', {'layer': layer, 'rank': rank, 'step': step, 'op': event['op'],
                     'slots': slots, 'exact_bytes': True, 'positions': event['positions']})
            totals.append({'layer': layer, 'rank': rank, 'kind': spec['kind'],
                           'steps': len(plan['cache_trace']), 'all_cache_bytes_exact': True})
    return totals


def validate_native_objects(native):
    contract.require(set(native) == {'torch', 'safe_open', 'configuration', 'modeling', 'candidate', 'create_local_weight_qkv'},
                     'Exact externally authenticated native provider set')
    sources = {'configuration': str(contract.original.CONFIGURATION), 'modeling': str(contract.original.MODELING),
               'candidate': str(contract.ROOT / 'spark26_text_port_v2.py')}
    for name, expected in sources.items():
        contract.require(str(Path(native[name].__file__).resolve()) == expected, 'Actual unchanged native source module: ' + name)
    sharder = native['create_local_weight_qkv']
    contract.require(sharder.__name__ == 'create_local_weight_qkv'
                     and sharder.__code__.co_filename == '/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/neuronx_distributed/trace/trace.py',
                     'Actual pinned SDK sharding function, never AST-extracted substitute')


def run(plan, checkpoint_root, guard, load_native, emit):
    """Called only by a future authenticated original serial worker.

    guard must return the same bound authority and recheck actual lease/runtime.
    load_native must perform real imports after guard, never mock them at runtime.
    No admission receipt or standalone execution path is created by this core.
    """
    contract.validate_plan(plan)
    authority = contract.validate_authority(guard(), plan, checkpoint_root)
    def checked():
        contract.require(guard() == authority, 'Actual native authority cannot change during diagnostic')
    checked()
    native = load_native(); checked(); validate_native_objects(native)
    torch = native['torch']
    reader = OriginalWeights(checkpoint_root, plan, torch, native['safe_open'], checked)
    weight_auth = reader.authenticate_all()
    with torch.inference_mode():
        qkv = qkv_diagnostic(torch, native['configuration'], native['modeling'], native['candidate'],
                             native['create_local_weight_qkv'], reader, plan, checked, emit)
        cache = cache_diagnostic(torch, native['configuration'], native['modeling'], native['candidate'], plan, checked, emit)
    checked()
    return {'schema': 'spark26_native_port_diagnostic_result_v2', 'complete': True,
            'plan_sha256': contract.digest(plan), 'original_weight_authentication': weight_auth,
            'qkv': qkv, 'cache': cache, 'original_weight_shard_partition_exact': True,
            'cache_structural_gate_passed': True,
            'projection_bitwise_equal': all(m['bitwise_equal'] for row in qkv
                for group in ('metrics', 'v2_packed_metrics') for m in row[group].values()),
            'projection_all_finite': all(m['finite'] for row in qkv
                for group in ('metrics', 'v2_packed_metrics') for m in row[group].values()),
            'projection_differences_are_diagnostic_not_new_model_gate': True,
            'synthetic_hidden_and_cache_inputs': True, 'distributed_collectives_executed': False,
            'neuron_cache_aliases_or_compilation_tested': False,
            'fresh_process_cold_model_tested': False, 'full_model_qualified': False,
            'original_logit_criteria': dict(contract.LOGIT_CRITERIA), 'new_speedup_claim': False}


if __name__ == '__main__':
    raise RuntimeError('No standalone native entry: future reviewed serial worker/runtime authentication is required')
