"""Two-arm call4 CPU worker. Import is stdlib-only; root owns serial admission."""
import ast
import hashlib
import importlib
import json
from pathlib import Path

ROOT = Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
MIB = 2**20
ARMS = ('original_inclusive_control', 'strict_prior_same_width')
CELLS = ((0, 28), (1, 1231), (2, 27), (3, 24))
ENDPOINT_SHAS = ('1902dc6d3233563b0ec8cc16eef97610f4aef21df97dec01cafd8bf80cd32b59',
                 'd8298e7da0fe32300341a7375a9d8d94cc12e45495054fc67b4c520ba5f6201d')
BEFORE_SHA = '86161d1a4ef9578085290e15ae6058248c22fa5df950a6ff32455c58999d046f'


def require(value, message):
    if not value: raise RuntimeError(message)


def same(left, right, message):
    require(json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True), message)


def mask_rows(rows, positions, strict):
    require(type(strict) is bool and len(rows) == len(positions) == 4, 'Four physical mask rows')
    same(positions, [[28], [1231], [27], [24]], 'Original physical position mapping')
    for index, row in enumerate(rows):
        pos = positions[index][0]; count = pos if strict else pos+1
        same(row, [1]*count+[0]*(2048-count), 'Exact full-width mask predicate')


def make_strict(inputs):
    mask_rows(inputs[1].tolist(), inputs[2].tolist(), False)
    changed = list(inputs); changed[1] = inputs[1].clone()
    for row, column in CELLS: changed[1][row, column] = 0
    mask_rows(changed[1].tolist(), changed[2].tolist(), True)
    return changed


def input_contract(inputs, original_metadata, original_values, runtime, strict):
    require(len(inputs) == len(original_metadata) == 7, 'Seven arguments retained')
    metadata = [runtime.tensor_metadata(x) for x in inputs]
    for index, (actual, expected) in enumerate(zip(metadata, original_metadata)):
        if index != 1 or not strict: same(actual, expected, 'Unchanged argument '+str(index))
        else:
            same({k:v for k,v in actual.items() if k != 'sha256'},
                 {k:v for k,v in expected.items() if k != 'sha256'}, 'Mask layout/dtype unchanged')
    require(all(inputs[i] is original_values[i] for i in range(7) if i != 1), 'Six original argument objects')
    mask_rows(inputs[1].tolist(), inputs[2].tolist(), strict)
    return metadata


def frame_map(call, seq_ids):
    same(seq_ids, [0,1,2,3], 'Original sorted physical slots')
    same(call['index'], 4, 'Only call4'); same(call['phase'], 'decode', 'Only decode')
    same(call['group'], 0, 'Original group0')
    expected = [dict(index=4, fixture=0, slot=3, step=1), dict(index=5, fixture=7, slot=1, step=1),
                dict(index=6, fixture=2, slot=0, step=1), dict(index=7, fixture=5, slot=2, step=1)]
    same(call['frames'], expected, 'Exact original four frames')
    same(call['logical_positions'], [24,1231,28,27], 'Original logical positions')
    return [dict(frame=f, physical_row=seq_ids.index(f['slot'])) for f in expected]


def expected_control(context, rank):
    path = str(ROOT/f'minicpm5-history-crossover-v1/cpu_control/cpu-rank{rank}/call-04/result.json')
    ref = dict(path=path, sha256=ENDPOINT_SHAS[rank])
    require(ref in context.plan['cpu_records'], 'Pinned original control reference')
    expected = context.read_json(path, ref['sha256']); graph = context.plan['graph_calls'][4]
    same(expected['rank'], rank, 'Rank-specific endpoint'); same(expected['arm'], 'C_Hd_control', 'Device-before lineage')
    require(expected['complete'] is True, 'Complete old endpoint')
    same(expected['call'], graph['call'], 'Same call4'); frame_map(expected['call'], [0,1,2,3])
    same(graph['call'], context.plan['calls'][4], 'Plan graph/call identity')
    same([graph['bucket'], expected['bucket']], [2048,2048], 'Same physical bucket2048')
    same(expected['input_metadata'], graph['input_metadata'], 'Original graph input metadata')
    same(expected['imported_manifest'], graph['before'], 'Same original device before')
    same(graph['before']['sha256'], BEFORE_SHA, 'Frozen call4 before manifest')
    same(expected['history']['origin'], dict(kind='frozen_v2_device_before', call=4), 'No generated or zero history')
    return ref, expected, graph


def references(context, torch):
    ref = context.plan['hf_reference']; original = context.load_pt(ref['path'], 32*MIB)
    require(len(original) == 8 and [x['id'] for x in original] == list(range(8)), 'Original eight HF fixtures')
    for fixture, saved in zip(context.plan['fixtures'], original):
        for key in ('id', 'input_ids', 'continuation_ids'): same(saved[key], fixture[key], 'Original HF '+key)
        require(len(saved['logits']) == 4, 'Four original reference steps')
        for value in saved['logits']:
            require(list(value.shape) == [130560] and value.dtype == torch.float32 and bool(torch.isfinite(value).all()), 'Finite full-vocabulary HF reference')
    path = ROOT/'validate_real_logits.py'
    context.guard('cpu'); source = path.read_bytes()
    require(len(source) < 8*MIB and hashlib.sha256(source).hexdigest() == context.plan['source_sha256'][str(path)], 'Frozen metric source')
    functions = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == 'metrics']
    require(len(functions) == 1, 'Exactly one frozen metrics function')
    namespace = dict(torch=torch)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    return original, namespace['metrics']


def score(output, original, mapping, metric):
    require(list(output.shape) == [4,1,130560], 'Never truncate full vocabulary')
    result = []
    for item in mapping:
        frame = item['frame']; ref = original[frame['fixture']]['logits'][frame['step']]
        values = metric(output[item['physical_row'],0], ref)
        checks = dict(cosine=values['cosine'] >= .995, KL=values['kl_reference_to_neuron'] <= .02,
                      max_abs=values['max_abs_error'] <= 2,
                      high_margin_argmax=values['reference_margin'] <= .5 or values['argmax_match'])
        result.append(dict(**item, metrics=values, per_frame_criteria=checks, failed_criteria=[k for k,v in checks.items() if not v]))
    return result


def tensor_delta(torch, left, right):
    require(list(left.shape) == list(right.shape) and left.dtype == right.dtype, 'Matched delta layout')
    require(bool(torch.isfinite(left).all()) and bool(torch.isfinite(right).all()), 'Finite delta operands')
    values = (right.float()-left.float()).abs()
    return dict(changed_elements=int((left != right).sum()), max_abs=float(values.max()), mean_abs=float(values.mean()))


def strict_comparison(torch, endpoint, output, taps, control, post_metadata, control_post, mapping):
    require(list(taps) == list(control['taps']), 'All213 matched tap names')
    stages = {name:dict(raw_exact=endpoint.tensor_digest(torch, value) == endpoint.tensor_digest(torch, control['taps'][name]),
                        **tensor_delta(torch, control['taps'][name], value)) for name,value in taps.items()}
    require(len(stages) == 213 and len(post_metadata) == len(control_post) == 84, 'Full213/84 comparison')
    return dict(output=tensor_delta(torch, control['output'], output),
                frames=[dict(**item, **tensor_delta(torch, control['output'][item['physical_row'],0], output[item['physical_row'],0]),
                             argmax_changed=int(control['output'][item['physical_row'],0].argmax()) != int(output[item['physical_row'],0].argmax())) for item in mapping],
                all213_taps=stages, layer0_changed_stages=[name for name,row in stages.items() if name.startswith('layers.0') and not row['raw_exact']],
                other_changed_stages=[name for name,row in stages.items() if not name.startswith('layers.0') and not row['raw_exact']],
                changed_post_state_indices=[i for i,(a,b) in enumerate(zip(control_post, post_metadata)) if a['sha256'] != b['sha256']],
                strict_original_endpoint_equality_required=False, full_model_qualified=False, unique_causal_layer_proven=False)


def route_hooks(model, inputs, runtime):
    """Read-only pre-hooks; do not wrap or replace SDK arithmetic methods."""
    records = {}; handles = []
    require(len(model.layers) == 42 and model.n_positions == 2048, 'Original42 layers and bucket')
    nc = model.neuron_config
    for flag in ('is_prefix_caching', 'flash_decoding_enabled', 'sequence_parallel_enabled', 'k_cache_transposed'):
        require(getattr(nc, flag) is False, 'Original route '+flag)
    expected_mask = [[[row]] for row in inputs[1].tolist()]
    expected_positions = inputs[2].tolist()
    def hook(index):
        def observe(module, args, kwargs):
            require(not args and index not in records, 'One keyword attention call/layer')
            mask = kwargs['attention_mask']; positions = kwargs['position_ids']
            require(list(mask.shape) == [4,1,1,2048] and str(mask.dtype) == 'torch.bool', 'Actual expanded prior mask')
            require(mask.tolist() == expected_mask and positions.tolist() == expected_positions, 'Actual mask/positions equal arm input')
            require(list(positions.shape) == [4,1] and kwargs.get('active_mask') is None, 'Single-token unmasked active path')
            records[index] = dict(layer=index, mask=runtime.tensor_metadata(mask), positions=runtime.tensor_metadata(positions),
                                  active_mask=None, is_prefix_caching=False, physical_key_width=2048)
        return observe
    try:
        for index, layer in enumerate(model.layers): handles.append(layer.self_attn.register_forward_pre_hook(hook(index), with_kwargs=True))
    except Exception:
        for handle in handles: handle.remove()
        raise
    return handles, records


def state_metadata(torch, endpoint, states, before_records):
    require(len(states) == len(before_records) == 84, 'Exactly84 returned states')
    records = []
    for value, before in zip(states, before_records):
        require(list(value.shape) == before['shape'] == [4,1,4224,128] and str(value.dtype) == before['dtype'] == 'torch.bfloat16', 'Post state layout')
        require(bool(torch.isfinite(value).all()), 'Finite post state')
        records.append(endpoint.tensor_metadata(torch, value))
    return records


def control_endpoint(torch, endpoint, output, taps, tap_meta, model, post, expected, before, after, rank):
    return endpoint.cpu_endpoint(torch, output, taps, tap_meta, model.kv_mgr.past_key_values, post, expected, before, after, rank)


def authenticate_constructor_inputs(context, rank):
    from minicpm5_layer0_cpu_constructor_v1 import ARTIFACT
    return [context.verify_input(str(path)) for path in
            (ARTIFACT/'neuron_config.json', ARTIFACT/'weights'/f'tp{rank}_sharded_checkpoint.safetensors')]


def collective_control_gate(torch, error):
    flag = torch.tensor([int(error is None)], dtype=torch.int32)
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
    require(int(flag.item()) == 1, 'Both rank control endpoints must pass; local='+str(error))
    torch.distributed.barrier()


def write_json(path, value):
    data = (json.dumps(value, indent=2, allow_nan=False)+'\n').encode()
    require(len(data) <= 2*MIB, 'Small JSON output cap')
    with path.open('xb') as stream: stream.write(data)


def save_payload(torch, path, value):
    require(sum(x.numel()*x.element_size() for x in [value['output'], *value['taps'].values()]) < 7*MIB, 'Bounded selected tensors only')
    with path.open('xb') as stream: torch.save(value, stream)
    require(path.stat().st_size < 8*MIB, 'Bounded output payload')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(MIB), b''): h.update(block)
    return dict(path=str(path), sha256=h.hexdigest(), bytes=path.stat().st_size)


def execute(context, output_dir, torch, runtime, endpoint):
    rank = context.permit['rank']; ref, expected, graph = expected_control(context, rank)
    call = graph['call']; original = runtime.graph_inputs(graph)
    inputs, widening = runtime.cpu_indices(original)
    same(widening, expected['integer_widening'], 'Original lossless integer widening')
    same([runtime.tensor_metadata(x) for x in inputs], expected['execution_input_metadata'], 'Original execution inputs')
    runtime.verify_original_positions_masks(inputs, call, context.plan['fixtures'])
    mapping = frame_map(call, inputs[3].tolist()); original_metadata = [runtime.tensor_metadata(x) for x in inputs]
    before = runtime.manifest(expected['imported_manifest'], rank); after = runtime.manifest(expected['after'], rank)
    hf, metric = references(context, torch)
    model = None; hooks = None; control = None; control_post = None; reports = []
    try:
        weight_inputs = authenticate_constructor_inputs(context, rank)
        model, proof = runtime.build_original_model(rank, 'decode')
        same(authenticate_constructor_inputs(context, rank), weight_inputs, 'Constructor artifact identity unchanged')
        require(proof['compiled'] is proof['resharded'] is proof['sdk_math_modified'] is False, 'Frozen original constructor')
        bucket_proof = runtime.select_original_bucket(model, 2048); hooks = runtime.prediction_hooks(model)
        for arm in ARMS:
            context.guard('cpu'); strict = arm == ARMS[1]; arm_inputs = make_strict(inputs) if strict else inputs
            arm_meta = input_contract(arm_inputs, original_metadata, inputs, runtime, strict)
            for parameter, record in zip(model.kv_mgr.past_key_values, before): parameter.copy_(runtime.load_state(record))
            endpoint.verify_states(torch, model.kv_mgr.past_key_values, before, rank, 'reimported original device-before')
            handles, routes = route_hooks(model, arm_inputs, runtime)
            try:
                hooks.begin(arm_inputs[2][:,0]+1, False); output, *post = model(*arm_inputs); taps, tap_meta = hooks.finish()
            finally:
                for handle in handles: handle.remove()
            error = None; checked = None
            try:
                require(list(routes) == list(range(42)), 'Actual route for all42 layers')
                require(list(output.shape) == [4,1,130560] and output.dtype == torch.float32 and bool(torch.isfinite(output).all()), 'Finite full-vocabulary output')
                same(list(taps), expected['stage_names'], 'All213 original taps'); same(tap_meta, expected['stage_metadata'], 'Original tap layouts')
                require(all(bool(torch.isfinite(x).all()) for x in taps.values()), 'Finite taps')
                same(input_contract(arm_inputs, original_metadata, inputs, runtime, strict), arm_meta, 'Inputs unchanged by forward')
                endpoint.verify_states(torch, model.kv_mgr.past_key_values, before, rank, 'unchanged original input')
                post_meta = state_metadata(torch, endpoint, post, before)
                if not strict: checked = control_endpoint(torch, endpoint, output, taps, tap_meta, model, post, expected, before, after, rank)
            except Exception as exc:
                if strict: raise
                error = type(exc).__name__+': '+str(exc)
            if not strict: collective_control_gate(torch, error)
            folder = output_dir/arm; folder.mkdir(exist_ok=False)
            payload = dict(output=output.detach().cpu().clone().contiguous(), taps=taps)
            report = dict(schema='minicpm5_call4_mask_rank_arm_v1', lineage_arm='C_Hd_control', intervention_arm=arm, rank=rank,
                          call=call, physical_rows=mapping, bucket=2048, input_metadata=arm_meta, integer_widening=widening,
                          changed_mask_cells=[dict(row=i,column=j,before=1,after=0) for i,j in CELLS] if strict else [],
                          original_device_before=expected['imported_manifest'], all84_original_before_reimported=True, all84_input_states_unchanged=True,
                          control_endpoint=checked, post_metadata=post_meta, post_payload_written=False, route_records=list(routes.values()),
                          output_metadata=endpoint.tensor_metadata(torch, output), stage_metadata=tap_meta,
                          stage_sha256={name:endpoint.tensor_digest(torch,value) for name,value in taps.items()},
                          HF_metrics=score(output, hf, mapping, metric), full_model_qualified=False, new_speedup_claim=False,
                          strict_original_output_tap_post_equality_required=False)
            if strict: report['contrast'] = strict_comparison(torch, endpoint, output, taps, control, post_meta, control_post, mapping)
            report['payload'] = save_payload(torch, folder/'output-taps.pt', payload)
            write_json(folder/'result.json', report); reports.append(dict(arm=arm, report=str(folder/'result.json'), payload=report['payload']))
            if not strict: control = payload; control_post = post_meta
            del output, post, taps, payload
            context.guard('cpu')
        result = dict(schema='minicpm5_call4_mask_rank_result_v1', complete=True, rank=rank, lineage_arm='C_Hd_control',
                      call=4, frames=4, arms=list(ARMS), original_endpoint=ref, constructor=proof, bucket=bucket_proof,
                      constructor_input_bindings=weight_inputs,
                      reports=reports, both_rank_control_gate_passed=True, model_forwards=2, full_post_payloads_written=0,
                      original_prefill_failure_retained=True, full_model_qualified=False, new_speedup_claim=False,
                      self_exit_claimed=False, root_must_observe_actual_exit=True)
        write_json(output_dir/'result.json', result); return result
    finally:
        if hooks is not None: hooks.close()


def run(context, output_dir):
    context.guard('cpu'); require(context.permit['arm'] == 'C_Hd_control' and type(context.permit['rank']) is int and context.permit['rank'] in (0,1), 'Original lineage/rank')
    output_dir = Path(output_dir)
    require(output_dir == Path(context.plan['output_root'])/('cpu-rank'+str(context.permit['rank'])), 'Fresh rank output directory')
    require(Path(context.plan['output_root']).parent == ROOT and Path(context.plan['output_root']).name.startswith('minicpm5-call4-mask-contrast-'), 'Dedicated fresh contrast output tree')
    require(str(Path(__file__).resolve()) in context.plan['source_sha256'], 'Worker included in authenticated source closure')
    context.verify_sources(); output_dir.mkdir(exist_ok=False)
    with context.activate():
        import torch
        importlib.import_module('neuronx_distributed_inference.modules.lora_serving')
        from neuronx_distributed.parallel_layers import parallel_state
        from minicpm5_layer0_cpu_adapter_v2 import CPUAdapter
        import minicpm5_layer0_capture_endpoints_v1 as endpoint
        witness = context.record_cpu_import(); torch.set_num_threads(1)
        torch.distributed.init_process_group('gloo')
        try:
            require(torch.distributed.get_world_size() == 2 and torch.distributed.get_rank() == context.permit['rank'], 'Actual TP2 rank')
            parallel_state.initialize_model_parallel(2, skip_collective_init=True)
            try:
                write_json(output_dir/'sdk-environment-transition.json', witness)
                with torch.inference_mode(): return execute(context, output_dir, torch, CPUAdapter(context,torch), endpoint)
            finally: parallel_state.destroy_model_parallel()
        finally: torch.distributed.destroy_process_group()
