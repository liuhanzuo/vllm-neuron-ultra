"""Unexecuted native successor: real host methods and physical input permutations.

No native import on module load. The future serial host supplies the real
NeuronModelBase class in addition to all v2 native providers, after authenticating
its current worker, retained original lock, complete source and native closure.
There is no AST extraction or code rewrite in this runtime module.
"""
import inspect
from pathlib import Path
from types import SimpleNamespace

import spark26_native_port_diagnostic_contract_v2 as contract
import spark26_native_port_diagnostic_v2 as previous

OriginalWeights = previous.OriginalWeights
bits_equal = previous.bits_equal
cache_codes = previous.cache_codes
stored_physical = previous.stored_physical
qkv_diagnostic = previous.qkv_diagnostic


def validate_native_objects(native):
    contract.require(set(native) == {'torch', 'safe_open', 'configuration', 'modeling', 'candidate',
                                    'create_local_weight_qkv', 'NeuronModelBase'},
                     'All original native providers plus actual host loader class')
    previous.validate_native_objects({key: value for key, value in native.items() if key != 'NeuronModelBase'})
    cls = native['NeuronModelBase']
    contract.require(cls.__name__ == 'NeuronModelBase'
                     and cls.__module__ == 'vllm_neuron.worker.neuronx_distributed_model_loader',
                     'Actual imported fixed NeuronModelBase class')
    for name in ('_reordered', '_sort_inputs'):
        function = inspect.unwrap(getattr(cls, name))
        contract.require(function.__name__ == name
                         and str(Path(function.__code__.co_filename).resolve()) == contract.HOST_LOADER,
                         'Real fixed host method, not AST-extracted or replacement: ' + name)


def real_host_reorder(torch, model_base, event, key_by_slot, value_by_slot, guard):
    """Exercise actual host sort and inverse restore with real tensor inputs.

    A minimal receiver selects the documented ordinary-cache method branch.
    This executes the original two methods, not model initialization/forward or
    SDK padding. Request-linked K/V and positions are checked before B4 filling.
    """
    guard()
    slots = event['slots']
    ids = torch.tensor(slots, dtype=torch.int32)
    position_ids = torch.tensor([[position] for position in event['positions']], dtype=torch.int32)
    key = torch.cat([key_by_slot[slot] for slot in slots], dim=0)
    value = torch.cat([value_by_slot[slot] for slot in slots], dim=0)
    markers = torch.tensor([[slot, position, epoch] for slot, position, epoch
                            in zip(slots, event['positions'], event['epochs'])], dtype=torch.int32)
    inputs = {'position_ids': position_ids, 'key': key, 'value': value, 'request_markers': markers}
    original_inputs = {name: tensor.clone() for name, tensor in inputs.items()}
    original_ids = ids.clone()
    receiver = SimpleNamespace(is_reorder_needed=True, _sort_inputs=model_base._sort_inputs)
    # This is the original contextmanager/function object from the imported SDK.
    with model_base._reordered(receiver, ids, **inputs) as (sorted_ids, sorted_inputs, restore):
        order = sorted(range(len(slots)), key=lambda index: slots[index])
        index = torch.tensor(order, dtype=torch.long)
        expected_ids = torch.tensor([slots[i] for i in order], dtype=torch.int32)
        contract.require(bits_equal(torch, sorted_ids, expected_ids), 'Actual host sorts original request slot IDs')
        contract.require(set(sorted_inputs) == set(inputs), 'All linked host fields retained')
        for name, tensor in original_inputs.items():
            expected = torch.index_select(tensor, 0, index)
            contract.require(bits_equal(torch, sorted_inputs[name], expected), 'Actual host linked row order: ' + name)
            restored = restore(sorted_inputs[name])
            contract.require(bits_equal(torch, restored, tensor), 'Actual host inverse restore: ' + name)
        result = {'ids': sorted_ids.clone(), 'inputs': {name: tensor.clone() for name, tensor in sorted_inputs.items()}}
    contract.require(bits_equal(torch, ids, original_ids)
                     and all(bits_equal(torch, inputs[name], tensor) for name, tensor in original_inputs.items()),
                     'Host reorder/restore preserves all original input tensors')
    guard()
    return result


def physical_order_diagnostic(torch, candidate, before_k, before_v, key, value, ids, positions,
                              sliding, reference, permutations, guard, emit, context):
    """Reexecute one original cache call from the same prior state in 24 orders."""
    input_k, input_v = before_k.clone(), before_v.clone()
    inputs = (key, value, ids, positions)
    saved_inputs = tuple(tensor.clone() for tensor in inputs)
    checks = 0
    for permutation in permutations:
        guard()
        index = torch.tensor(permutation, dtype=torch.long)
        inverse = torch.tensor([permutation.index(i) for i in range(4)], dtype=torch.long)
        reordered = tuple(torch.index_select(tensor, 0, index) for tensor in inputs)
        saved_reordered = tuple(tensor.clone() for tensor in reordered)
        k, v, seq_ids, pos = reordered
        # Every call starts with the same original previous physical cache.
        actual = candidate.decode_cache(k, v, before_k, before_v, seq_ids, pos, sliding)
        contract.require(len(actual) == len(reference) == 6, 'Complete real cache-helper output tuple')
        for i in range(4):
            restored = torch.index_select(actual[i], 0, inverse)
            contract.require(bits_equal(torch, restored, reference[i]),
                             'Inverse physical row order restores K/V/position/mask output: ' + str(i))
        for i in (4, 5):
            contract.require(bits_equal(torch, actual[i], reference[i]),
                             'Persistent cache physical-slot bytes invariant to input row order')
        contract.require(bits_equal(torch, before_k, input_k) and bits_equal(torch, before_v, input_v),
                         'Permutation checks cannot mutate the original prior cache')
        contract.require(all(bits_equal(torch, a, b) for a, b in zip(inputs, saved_inputs))
                         and all(bits_equal(torch, a, b) for a, b in zip(reordered, saved_reordered)),
                         'Permutation calls preserve original and permuted input tensors')
        emit('cache_physical_order', {**context, 'permutation': permutation,
             'returned_rows_restored_exact': True, 'all_physical_cache_slots_exact': True,
             'dummy_rows_are_diagnostic_only': True})
        checks += 1
    contract.require(checks == 24, 'All fixed four-row permutations required')
    return checks


def cache_diagnostic(torch, configuration, modeling, candidate, model_base, plan, guard, emit):
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
            host_checks = order_checks = 0
            snapshot_steps = {row['step'] for row in plan['physical_row_order_check']['snapshots']}
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
                    active_key, active_value = {}, {}
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
                            active_key[slot] = k[:, rank * 2:(rank + 1) * 2]
                            active_value[slot] = v[:, rank * 2:(rank + 1) * 2]
                            seen[slot] += 1
                        else:
                            k = torch.full((1, 4, 1, 256), -128, dtype=torch.bfloat16)
                            v = torch.full_like(k, 127)
                        ks.append(k[:, rank * 2:(rank + 1) * 2]); vs.append(v[:, rank * 2:(rank + 1) * 2])
                    # Exercise the real host methods on request-order tensors, then
                    # consume their actual sorted outputs while constructing B4 rows.
                    host = real_host_reorder(torch, model_base, event, active_key, active_value, guard)
                    ordered_slots = host['ids'].tolist()
                    slot_to_row = {slot: index for index, slot in enumerate(ordered_slots)}
                    host_positions = host['inputs']['position_ids']
                    for slot, index in slot_to_row.items():
                        ks[slot] = host['inputs']['key'][index:index + 1]
                        vs[slot] = host['inputs']['value'][index:index + 1]
                    host_b4_positions = torch.cat([host_positions[slot_to_row[slot]:slot_to_row[slot] + 1]
                        if slot in slot_to_row else host_positions[:1] for slot in range(4)], dim=0)
                    contract.require(bits_equal(torch, host_b4_positions, positions),
                                     'Actual host position order matches independent physical-slot plan')
                    positions = host_b4_positions
                    host_checks += 1
                    emit('cache_host_reorder', {'layer': layer, 'rank': rank, 'step': step,
                        'request_slots': slots, 'sorted_slots': ordered_slots,
                        'actual_host_methods_called': True, 'actual_inverse_restore_checked': True,
                        'full_serving_forward_executed': False, 'SDK_padding_executed': False})
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
                if step in snapshot_steps:
                    contract.require(event['op'] == 'decode', 'Original selected snapshot is decode')
                    order_checks += physical_order_diagnostic(torch, candidate, before_k, before_v,
                        batch_k, batch_v, ids, positions, sliding,
                        (keys, values, key_positions, valid, new_k, new_v),
                        plan['physical_row_order_check']['permutations'], guard, emit,
                        {'layer': layer, 'rank': rank, 'step': step, 'active_rows': len(slots)})
                old_k, old_v = new_k, new_v
                emit('cache_step', {'layer': layer, 'rank': rank, 'step': step, 'op': event['op'],
                     'slots': slots, 'exact_bytes': True, 'positions': event['positions']})
            contract.require(host_checks == sum(row['op'] == 'decode' for row in plan['cache_trace'])
                             and order_checks == 96, 'All original host calls and 4x24 row permutations completed')
            totals.append({'layer': layer, 'rank': rank, 'kind': spec['kind'],
                           'steps': len(plan['cache_trace']), 'all_cache_bytes_exact': True,
                           'actual_host_reorder_restore_calls': host_checks,
                           'physical_input_permutation_calls': order_checks})
    return totals


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
        cache = cache_diagnostic(torch, native['configuration'], native['modeling'], native['candidate'],
                                  native['NeuronModelBase'], plan, checked, emit)
    checked()
    return {'schema': 'spark26_native_port_diagnostic_result_v3', 'complete': True,
            'plan_sha256': contract.digest(plan), 'original_weight_authentication': weight_auth,
            'qkv': qkv, 'cache': cache, 'original_weight_shard_partition_exact': True,
            'cache_structural_gate_passed': True,
            'actual_host_reorder_restore_executed': True,
            'physical_input_permutation_calls': sum(row['physical_input_permutation_calls'] for row in cache),
            'full_serving_forward_or_SDK_padding_executed': False,
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
