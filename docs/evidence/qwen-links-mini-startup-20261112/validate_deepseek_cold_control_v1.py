"""Exact native cached references and all-bucket qualification for original Qwen3."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
DEST = None  # explicit owning-worker destination only
REFERENCE = None
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


def main(model,case,destination,reference_directory,data_root,execution_guard):
    assert callable(execution_guard)
    execution_guard()
    DEST=Path(destination); REFERENCE=Path(reference_directory); DATA_ROOT=Path(data_root)
    import torch
    import server_with_bias_fix
    from deepseek_qwen3_neuron_adapter import NeuronDeepSeekQwen3ForCausalLM as cls
    from vllm_neuron.worker.neuronx_distributed_model_loader import NeuronModelBase
    from validate_real_logits import metrics
    from deepseek_lifecycle_full_protocol_v2 import passed, CRITERIA as original_criteria, decode_mask
    from deepseek_prior_mask_state_v1 import bindings, clear, exercise
    from qwen35_split_state_checks_v4 import graph_routes

    torch.set_num_threads(1)
    assert type(model) is cls, 'Original uncaptured model class required'
    assert model.neuron_config.tensor_capture_config is None
    assert model.neuron_config.on_device_sampling_config is None
    reference_path = REFERENCE/'cold-reference.pt'
    source = torch.load(reference_path, weights_only=True, map_location='cpu')
    info=json.loads(reference_path.with_suffix('.json').read_text())
    assert info['complete'] and info['all_finite'] and info['frames']==42
    assert hashlib.sha256(reference_path.read_bytes()).hexdigest()==info['tensor_sha256']
    from deepseek_lifecycle_full_protocol_v2 import build
    protocol=build()
    assert info['revision']==protocol['revision'] and info['criteria']==protocol['criteria']
    assert info['native_source_sha256']==protocol['native_source_sha256']
    assert info['source_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert [(r['prefix'],r['fixture']) for r in source]==[(n,f) for n in PREFIXES for f in range(2)]
    references={(r['prefix'],r['fixture']):r for r in source}
    assert set(references)=={(n,f) for n in PREFIXES for f in range(2)}
    for ref in source:
        assert ref['input_ids']==protocol['fixtures'][ref['fixture']]['input_ids']
        assert ref['logits'].shape==(3,151936) and bool(torch.isfinite(ref['logits']).all())
        assert ref['positions']==[ref['prefix']-1,ref['prefix'],ref['prefix']+1]
    cfg = model.config
    assert cfg.neuron_config.batch_size == 4 and cfg.neuron_config.tp_degree == 2
    assert not cfg.neuron_config.async_mode and cfg.neuron_config.cast_type == 'config'
    assert tuple(model.context_encoding_model.neuron_config.buckets)==CONTEXT_BUCKETS
    assert tuple(model.token_generation_model.neuron_config.buckets)==DECODE_BUCKETS
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
                  scope='Original compiled DeepSeek Qwen3 graph, BF16 weights/config; seven in-place-cleared device-state '
                        'experiments, two exact native-HF cached streams repeated in slots0/2 and1/3. Prefill and '
                        'two teacher-forced B4 decode steps, 84 full-vocabulary frames. Covers all six context '
                        'and six decode bucket routes. Native-HF prefill at each exact prefix and two cached updates; '
                        'all original strict thresholds retained. Not a throughput test.')
    report['graph_routes'] = graph_routes(model)
    report['reset_receipts'] = []
    captured = []
    from deepseek_cold_control_capture_v1 import Capture
    capture = Capture('device',DEST,DATA_ROOT)

    def save():
        temp = output.with_suffix('.json.tmp')
        temp.write_text(json.dumps(report, indent=2, allow_nan=False))
        temp.replace(output)

    def forward(ids, positions, slots, prefill):
        route.clear()
        mask = torch.ones_like(ids) if prefill else decode_mask(positions,model)
        with NeuronModelBase._reordered(order, slots, input_ids=ids, attention_mask=mask,
                position_ids=positions, sampling_params=model.default_sampling_params[:ids.shape[0]]) as (s, inputs, restore):
            result = restore(model(seq_ids=s, **inputs).logits)
            capture.device(model, ids, positions, slots, prefill, route)
            return result

    def record(values, prefix, phase, step, slot):
        ref = references[prefix,slot % 2]
        position = prefix - 1 + step
        expected = ref['logits'][ref['positions'].index(position)]
        values = values.float().flatten().contiguous().cpu()
        valid = values.numel() == expected.numel() and bool(torch.isfinite(values).all())
        row = dict(prefix=prefix, phase=phase, step=step, slot=slot, fixture=slot % 2,
                   reference_position=position, valid=valid, elements=values.numel(),
                   nan_elements=int(torch.isnan(values).sum()), inf_elements=int(torch.isinf(values).sum()),
                   route=dict(route), metrics=metrics(values, expected) if valid else None)
        captured.append({k: row[k] for k in ('prefix', 'phase', 'step', 'slot', 'fixture')} | dict(logits=values.clone()))
        return row

    try:
        exercise(model,DEST/'state-copy.json')
        from deepseek_cold_control_lifecycle_v1 import observe
        observe(model,DEST/'state-lifecycle.json',execution_guard)
        original_bindings = bindings(model)
        with torch.inference_mode():
            for prefix in PREFIXES:
                report['reset_receipts'].append(clear(model,original_bindings))
                model.reset()
                item = dict(prefix=prefix, rows=[], duplicate_slot_checks=[])
                report['experiments'].append(item)
                assert len(model.traced_model.nxd_model.state)==2
                item['initial_zero_state_checks']=[]
                for rank,tensors in enumerate(model.traced_model.nxd_model.state):
                    assert len(tensors)==72,'All36 independent KV pairs required on each rank'
                    for name,tensor in sorted(tensors.items()):
                        cpu=tensor.detach().cpu()
                        assert tuple(cpu.shape)==(4,4,4224,128) and cpu.dtype==torch.bfloat16
                        zero=int(torch.count_nonzero(cpu.contiguous().view(torch.uint8)))==0
                        item['initial_zero_state_checks'].append(dict(rank=rank,name=name,shape=list(cpu.shape),all_raw_bytes_zero=zero))
                        assert zero,'Device cache must be zero before cold prefill'
                del cpu
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
                        assert route['mask_length'] == expected_bucket(position, DECODE_BUCKETS), route
                        current = [values[slot, -1].clone() for slot in range(4)]
                        item['rows'].extend(record(value, prefix, 'decode', step, slot) for slot, value in enumerate(current))
                    else:
                        current = prefill_values
                    for a, b in ((0, 2), (1, 3)):
                        item['duplicate_slot_checks'].append(dict(step=step, slots=[a, b],
                            bitwise_equal=torch.equal(current[a].contiguous().view(torch.uint8), current[b].contiguous().view(torch.uint8))))
                    save()
                assert len(model.traced_model.nxd_model.state)==2
                assert all(len(tensors)>0 for tensors in model.traced_model.nxd_model.state)
                item['state_after_two_decode_steps'] = []
                for rank, tensors in enumerate(model.traced_model.nxd_model.state):
                    for name, tensor in sorted(tensors.items()):
                        cpu = tensor.detach().cpu()
                        item['state_after_two_decode_steps'].append(dict(rank=rank, name=name, shape=list(cpu.shape),
                            nan_elements=int(torch.isnan(cpu).sum()), inf_elements=int(torch.isinf(cpu).sum())))
                print(json.dumps(dict(prefix=prefix, finite_rows=sum(r['valid'] for r in item['rows']),
                                      duplicate_pairs_equal=sum(r['bitwise_equal'] for r in item['duplicate_slot_checks']))), flush=True)
                save()
        rows = [row for item in report['experiments'] for row in item['rows']]
        assert len(rows) == len(captured) == 84
        report.update(complete=True, all_outputs_finite=all(row['valid'] for row in rows),
                      numerical_gate_passed=all(row['valid'] for row in rows) and passed([row['metrics'] for row in rows]),
                      all_duplicate_slots_bitwise_equal=all(row['bitwise_equal'] for item in report['experiments'] for row in item['duplicate_slot_checks']))
        observed_context=sorted({r['route']['input_tokens'] for r in rows if r['phase']=='prefill'})
        observed_decode=sorted({r['route']['mask_length'] for r in rows if r['phase']=='decode'})
        assert observed_context==list(CONTEXT_BUCKETS) and observed_decode==list(DECODE_BUCKETS)
        report.update(observed_context_buckets=observed_context,observed_decode_buckets=observed_decode)
        report['all_cache_states_finite']=all(r['nan_elements']==r['inf_elements']==0 for item in report['experiments'] for r in item['state_after_two_decode_steps'])
        report['all_initial_states_zero']=all(r['all_raw_bytes_zero'] for item in report['experiments'] for r in item['initial_zero_state_checks'])
        report['passed']=report['numerical_gate_passed'] and report['all_duplicate_slots_bitwise_equal'] and report['all_cache_states_finite'] and report['all_initial_states_zero']
    except Exception as exc:
        report['error'] = repr(exc)
        raise
    finally:
        torch.save(captured, output.with_suffix('.pt'))
        report['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        save()
    capture.finish()
    if not report['passed']:raise SystemExit('Strict cold-bucket numerical/state gate failed')


def reference(model,protocol,source_hash,reference_directory,destination,data_root,execution_guard):
    assert callable(execution_guard)
    execution_guard()
    REFERENCE=Path(reference_directory)
    import torch
    path=REFERENCE/'cold-reference.pt';manifest=path.with_suffix('.json')
    assert not path.exists() and not manifest.exists(),'Preserve previous cold reference evidence'
    from deepseek_cold_control_capture_v1 import Capture
    capture = Capture('native',destination,data_root)
    rows=[];report=dict(complete=False,all_finite=False,frames=0,revision=protocol['revision'],
        native_source_sha256=source_hash,prefixes=list(PREFIXES),criteria=protocol['criteria'],
        scope='Original pinned native Qwen3 BF16/eager execution with seven exact prefill lengths and two cached decode updates per fixture;42frames.')
    try:
        with torch.inference_mode():
            for prefix in PREFIXES:
                for fixture in range(2):
                    ids=protocol['fixtures'][fixture]['input_ids'];positions=[prefix-1,prefix,prefix+1]
                    out=model(input_ids=torch.tensor([ids[:prefix]]),attention_mask=torch.ones((1,prefix),dtype=torch.long),
                              position_ids=torch.arange(prefix)[None],use_cache=True,logits_to_keep=1)
                    assert int(out.past_key_values.get_seq_length())==prefix
                    capture.native(out.past_key_values, prefix, fixture, 0, ids[:prefix])
                    values=[out.logits[0,-1].detach().float().cpu().clone()]
                    for pos in [prefix,prefix+1]:
                        out=model(input_ids=torch.tensor([[ids[pos]]]),attention_mask=torch.ones((1,pos+1),dtype=torch.long),
                                  position_ids=torch.tensor([[pos]]),past_key_values=out.past_key_values,use_cache=True,logits_to_keep=1)
                        assert int(out.past_key_values.get_seq_length())==pos+1
                        capture.native(out.past_key_values, prefix, fixture, pos-prefix+1, [ids[pos]])
                        values.append(out.logits[0,-1].detach().float().cpu().clone())
                    del out
                    logits=torch.stack(values);assert logits.shape==(3,151936) and bool(torch.isfinite(logits).all())
                    rows.append(dict(prefix=prefix,fixture=fixture,positions=positions,input_ids=ids,logits=logits))
                    print(json.dumps(dict(cold_reference_prefix=prefix,fixture=fixture,frames=3*len(rows))),flush=True)
            assert len(rows)==14;report.update(complete=True,all_finite=True,frames=42)
            capture.finish()
    except Exception as error:report['error']=repr(error);raise
    finally:
        torch.save(rows,path);report['tensor_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        report['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        manifest.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


