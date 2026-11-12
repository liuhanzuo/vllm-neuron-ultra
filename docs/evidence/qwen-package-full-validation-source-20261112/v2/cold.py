"""Observe newly compiled package Qwen graphs on cold state across every bucket.

This is diagnostic evidence after the failed smoke test, not a speed benchmark.
"""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
DEST = ROOT / 'qwen35-state-reset-full-v1'
PREFIXES = (32, 128, 256, 512, 1024, 2048, 2052)
CONTEXT_BUCKETS = (128, 256, 512, 1024, 2048, 4096)
DECODE_BUCKETS = (128, 256, 512, 1024, 2048, 4096)
def expected_bucket(length, buckets):
    # NxDI first_fit uses a strict boundary, except exact maximum length.
    for bucket in buckets:
        if length < bucket:
            return bucket
    if length == buckets[-1]:
        return buckets[-1]
    raise ValueError(f'Input length {length} exceeds largest bucket {buckets[-1]}')


def main(directory):
    DEST = Path(directory)
    import torch
    from .runtime import load_admitted
    from vllm_neuron.worker.neuronx_distributed_model_loader import NeuronModelBase
    from .numeric import metrics
    from .long import passed, CRITERIA as original_criteria

    from .state_observation import raw_zero_states, graph_routes, verify_cold_reproduction
    from vllm_neuron.models.qwen35.state import clear_states
    torch.set_num_threads(1)
    case = json.loads((DEST / 'case.json').read_text())
    reference_path = ROOT / 'qwen35-baseline/long-decode-hf-reference.pt'
    source = torch.load(reference_path, weights_only=True, map_location='cpu')
    assert json.loads(reference_path.with_suffix('.json').read_text())['incremental_crosscheck_passed']
    for ref in source:
        assert all(pos in ref['positions'] for n in PREFIXES for pos in (n-1, n, n+1))
    model = load_admitted(case['model_path'], case['artifact'])
    cfg = model.config
    assert cfg.neuron_config.batch_size == 4 and cfg.neuron_config.tp_degree == 2
    assert not cfg.neuron_config.async_mode and cfg.neuron_config.cast_type == 'config'
    assert tuple(cfg.neuron_config.context_encoding_buckets) == CONTEXT_BUCKETS
    order = SimpleNamespace(is_reorder_needed=True, _sort_inputs=NeuronModelBase._sort_inputs)
    route = {}

    def hook(wrapper):
        original = wrapper._forward

        def observe(*inputs):
            signature = str([list(value.shape) for value in inputs])
            chosen = model.traced_model.nxd_model.input_shape_map[signature]
            route.update(tag=chosen[0], index=chosen[1], input_tokens=int(inputs[0].shape[1]),
                         mask_length=int(inputs[1].shape[1]))
            return original(*inputs)
        wrapper._forward = observe

    hook(model.context_encoding_model)
    hook(model.token_generation_model)
    output = DEST / 'cold-buckets.json'
    assert not output.exists(), 'Preserve previous diagnostic evidence'
    report = dict(complete=False, case=case, criteria=original_criteria, experiments=[],
                  scope='Newly compiled package tensor_split graphs, original weights and sequence protocol; seven in-place-cleared device-state '
                        'experiments, two native-HF causal streams repeated in slots0/2 and1/3. Prefill and '
                        'two teacher-forced B4 decode steps, 84 full-vocabulary frames. Covers all six context '
                        'and six decode bucket routes, with all96 state tensors raw-zero before each prefix. Native-HF full-causal oracle with its existing short '
                        'incremental crosscheck. Not an accuracy certification beyond these frames or a throughput test.')
    report['graph_routes'] = graph_routes(model)
    report['state_reset_receipts'] = []
    captured = []

    def save():
        temp = output.with_suffix('.json.tmp')
        temp.write_text(json.dumps(report, indent=2, allow_nan=False))
        temp.replace(output)

    def forward(ids, positions, slots, prefill):
        route.clear()
        mask = torch.ones_like(ids) if prefill else (torch.arange(int(positions.max())+1)[None] <= positions).long()
        with NeuronModelBase._reordered(order, slots, input_ids=ids, attention_mask=mask,
                position_ids=positions, sampling_params=model.default_sampling_params[:ids.shape[0]]) as (s, inputs, restore):
            return restore(model(seq_ids=s, **inputs).logits)

    def record(values, prefix, phase, step, slot):
        ref = source[slot % 2]
        position = prefix - 1 + step
        expected = ref['logits'][ref['positions'].index(position)]
        values = values.float().flatten()[:expected.numel()].contiguous().cpu()
        valid = values.numel() == expected.numel() and bool(torch.isfinite(values).all())
        row = dict(prefix=prefix, phase=phase, step=step, slot=slot, fixture=slot % 2,
                   reference_position=position, valid=valid, elements=values.numel(),
                   nan_elements=int(torch.isnan(values).sum()), inf_elements=int(torch.isinf(values).sum()),
                   route=dict(route), metrics=metrics(values, expected) if valid else None)
        captured.append({k: row[k] for k in ('prefix', 'phase', 'step', 'slot', 'fixture')} | dict(logits=values.clone()))
        return row

    try:
        with torch.inference_mode():
            for prefix in PREFIXES:
                expected_bindings = report['state_reset_receipts'][0]['bindings'] if report['state_reset_receipts'] else None
                report['state_reset_receipts'].append(clear_states(model.traced_model.nxd_model, expected_bindings))
                model.reset()
                item = dict(prefix=prefix, rows=[], duplicate_slot_checks=[])
                report['experiments'].append(item)
                item['initial_raw_zero_states'] = raw_zero_states(model)
                prefill_values = []
                for slot in range(4):
                    ids = torch.tensor([source[slot % 2]['input_ids'][:prefix]])
                    value = forward(ids, torch.arange(prefix)[None], torch.tensor([slot]), True)[0, -1].clone()
                    assert route['tag'] == 'context_encoding_model'
                    assert route['input_tokens'] == expected_bucket(prefix, CONTEXT_BUCKETS), route
                    prefill_values.append(value)
                    item['rows'].append(record(value, prefix, 'prefill', 0, slot))
                    save()
                for step in range(3):
                    if step:
                        position = prefix + step - 1
                        ids = torch.tensor([[source[slot % 2]['input_ids'][position]] for slot in range(4)])
                        values = forward(ids, torch.full((4, 1), position, dtype=torch.long), torch.arange(4), False)
                        assert route['tag'] == 'token_generation_model'
                        assert route['mask_length'] == expected_bucket(position+1, DECODE_BUCKETS), route
                        current = [values[slot, -1].clone() for slot in range(4)]
                        item['rows'].extend(record(value, prefix, 'decode', step, slot) for slot, value in enumerate(current))
                    else:
                        current = prefill_values
                    for a, b in ((0, 2), (1, 3)):
                        item['duplicate_slot_checks'].append(dict(step=step, slots=[a, b],
                            bitwise_equal=torch.equal(current[a].contiguous().view(torch.uint8), current[b].contiguous().view(torch.uint8))))
                    save()
                item['state_after_two_decode_steps'] = []
                for rank, tensors in enumerate(model.traced_model.nxd_model.state):
                    for name, tensor in sorted(tensors.items()):
                        cpu = tensor.detach().cpu()
                        item['state_after_two_decode_steps'].append(dict(rank=rank, name=name, shape=list(cpu.shape), dtype=str(cpu.dtype),
                            nan_elements=int(torch.isnan(cpu).sum()), inf_elements=int(torch.isinf(cpu).sum())))
                print(json.dumps(dict(prefix=prefix, finite_rows=sum(r['valid'] for r in item['rows']),
                                      duplicate_pairs_equal=sum(r['bitwise_equal'] for r in item['duplicate_slot_checks']))), flush=True)
                save()
        rows = [row for item in report['experiments'] for row in item['rows']]
        assert len(rows) == len(captured) == 84
        report.update(complete=True, all_outputs_finite=all(row['valid'] for row in rows),
                      numerical_gate_passed=all(row['valid'] for row in rows) and passed([row['metrics'] for row in rows]),
                      all_duplicate_slots_bitwise_equal=all(row['bitwise_equal'] for item in report['experiments'] for row in item['duplicate_slot_checks']))
        report['prior_partial_reproduction'] = verify_cold_reproduction(captured, report)
    except Exception as exc:
        report['error'] = repr(exc)
        raise
    finally:
        torch.save(captured, output.with_suffix('.pt'))
        report['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        save()


if __name__ == '__main__':
    raise SystemExit('Use entry.run_model_stage with the root execution guard')
