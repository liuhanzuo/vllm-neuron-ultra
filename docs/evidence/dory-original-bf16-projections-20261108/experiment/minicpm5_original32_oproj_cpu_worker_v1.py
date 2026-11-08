"""Four finite original32 CPU trajectories; root owns admission and TP2 transport.

Import is stdlib/source only. Runtime uses original authenticated constructors,
teacher inputs and HF endpoints. No full-cache/tap payloads are written.
"""
import copy
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import struct
import sys
import types

import minicpm5_call4_mask_worker_v1 as shared
import minicpm5_layer0_capture_endpoints_v1 as endpoint
import minicpm5_original32_mask_device_worker_v1 as trajectory
from minicpm5_layer0_runtime_auth_v1 import identity

ROOT, MIB = shared.ROOT, 2**20
require, same = shared.require, shared.same
ARMS = ('inclusive_ownhistory_control', 'strict_cpu_s0', 'strict_cpu_s1', 'strict_fp32_oproj_candidate')
TARGETS = tuple(f'layers.{i}.self_attn.o_proj.o_proj' for i in range(42))
ORIGINAL = dict(path=str(ROOT/'research/minicpm5-layer0-runtime-inputs-20261107-v1.json'),
                sha256='b8d84596f9fa6b54c0fa599af6ad976f051e47544a2d8c76027136efe2254800')
SUCCESSOR = dict(path=str(ROOT/'research/minicpm5-original88-python-trace-inputs-20261107-v1.json'),
                 sha256='17849dbee01a34f232a72432919fb7d042d7052797fd1d25290382703ab6c196')
FULLSHAPE = dict(path=str(ROOT/'research/minicpm5-saved-oproj-fullshape-intervention-r1-20261107-v1.json'),
                 sha256='1161557d376394364b25ab05ef1da857a6827699d6eacaa95c35a5a0b7012509')
DESIGN = dict(path=str(ROOT/'research/minicpm5-saved-oproj-fullshape-intervention-design-20261107-v2.json'),
              sha256='dcfa57c14ff4bc3bbc05a7cbd74262c258ca6582ff9beb1d32af6fed3c3fe897')
CORE = ('schema', 'interpreters', 'fixtures', 'calls', 'frames', 'graph_calls', 'cpu_records', 'hf_frame_records',
        'hf_reference', 'model_files', 'native_provenance', 'safetensors_header', 'official_layer0_weights',
        'sdk_attention_source', 'remaining_gates')
REPEAT_FIELDS = ('call', 'bucket', 'input_metadata', 'execution_input_metadata', 'integer_widening',
                 'changed_mask_cells', 'output_metadata', 'stage_names', 'stage_metadata', 'stage_sha256',
                 'before_metadata', 'input_after_metadata', 'post_metadata', 'route_records')


def validate_plan(plan, read_json):
    spec = plan['oproj_trajectory']
    require(set(spec) == {'schema', 'arms', 'original_scientific_plan', 'fullshape_result', 'fullshape_exit', 'design',
                     'per_rank_output_cap_bytes', 'total_output_cap_bytes'}, 'Exact finite trajectory spec fields')
    require(spec['schema'] == 'minicpm5_original32_oproj_cpu_plan_v1', 'New finite trajectory plan')
    same(spec['arms'], list(ARMS), 'Fixed four arm order')
    same(spec['original_scientific_plan'], ORIGINAL, 'Original scientific plan')
    same(spec['fullshape_result'], FULLSHAPE, 'Actual fullshape prerequisite')
    same(spec['design'], DESIGN, 'Reviewed finite intervention design')
    same([spec['per_rank_output_cap_bytes'], spec['total_output_cap_bytes']], [192*MIB, 384*MIB], 'Finite output caps')
    original = read_json(ORIGINAL['path'], ORIGINAL['sha256'])
    successor = read_json(SUCCESSOR['path'], SUCCESSOR['sha256'])
    require(set(successor) == set(original), 'Original88 successor fields unchanged')
    for key in original:
        if key not in ('output_root', 'source_sha256', 'writer_contexts'):
            same(successor[key], original[key], 'Reviewed original88 scientific equivalence '+key)
    require(set(plan) == set(successor) | {'oproj_trajectory'}, 'Only finite plan extension')
    for key in successor:
        if key not in ('output_root', 'source_sha256', 'inputs'):
            same(plan[key], successor[key], 'Exact actual successor inheritance '+key)
    for key in CORE: same(plan[key], original[key], 'Unchanged scientific core '+key)
    require(all(plan['inputs'].get(k) == v for k, v in original['inputs'].items()), 'All original input bindings inherited')
    require(all(plan['source_sha256'].get(k) == v for k, v in successor['source_sha256'].items()), 'All actual successor source pins inherited')
    result = read_json(FULLSHAPE['path'], FULLSHAPE['sha256'])
    ended = read_json(spec['fullshape_exit']['path'], spec['fullshape_exit']['sha256'])
    require(ended['actual_exit_observed'] is True and type(ended['actual_exit_code']) is int and ended['actual_exit_code'] == 0, 'Actual fullshape exit0')
    same(ended['outputs']['report'], FULLSHAPE, 'Actual fullshape output'); same(ended['owner'], result['execution_owner'], 'Fullshape owner')
    require(result['summary']['all_full_shape_controls_exact'] is True and result['summary']['original_execution_eligible_cases'] == 24, 'All24 fullshape controls')
    same(result['binding']['design'], DESIGN, 'Fullshape design bound')
    require(len(plan['calls']) == 14 and len(plan['frames']) == 32, 'Original14/32')
    root = Path(plan['output_root'])
    require(root.parent == ROOT and root.name.startswith('minicpm5-original32-oproj-cpu-'), 'Fresh dedicated output namespace')
    return spec


def expected_records(context, rank):
    rows = []
    for ref in context.plan['cpu_records']:
        record = context.read_json(ref['path'], ref['sha256'])
        if record['arm'] == 'C_Hc_history_generation' and record['rank'] == rank:
            rows.append((ref, record))
    rows.sort(key=lambda x: x[1]['call']['index'])
    same([r['call'] for _, r in rows], context.plan['calls'], 'All14 original own-history endpoints')
    for index, (_, row) in enumerate(rows):
        require(row['complete'] is True and len(row['stage_names']) == 213, 'Complete original endpoint')
        expected = dict(kind='frozen_v2_device_before', call=0) if index == 0 else dict(kind='cpu_history_previous_post', arm='C_Hc_history_generation', call=index-1)
        same(row['history']['origin'], expected, 'Original same-arm history lineage')
    return rows


def strict_inputs(plan, graph, inputs, strict, runtime):
    original = [runtime.tensor_metadata(x) for x in inputs]
    mapping = trajectory.input_rows(plan, graph['call'], inputs, False)
    changed = list(inputs); cells = []
    if strict and graph['call']['phase'] == 'decode':
        changed[1] = inputs[1].clone()
        for item in mapping:
            row, column = item['physical_row'], item['current_position']
            changed[1][row, column] = 0; cells.append(dict(row=row, column=column, before=1, after=0))
    actual = [runtime.tensor_metadata(x) for x in changed]
    for index in range(7):
        if index == 1 and cells:
            same({k:v for k,v in actual[index].items() if k != 'sha256'},
                 {k:v for k,v in original[index].items() if k != 'sha256'}, 'Mask layout/dtype unchanged')
        else:
            require(changed[index] is inputs[index], 'Six argument objects and prefill unchanged')
            same(actual[index], original[index], 'Unchanged input bytes')
    trajectory.input_rows(plan, graph['call'], changed, strict)
    return changed, cells, mapping, actual


def state_metadata(torch, values):
    require(len(values) == 84, 'All84 cache tensors')
    rows = []
    for index, value in enumerate(values):
        require(list(value.shape) == [4,1,4224,128] and value.dtype == torch.bfloat16 and value.device.type == 'cpu', 'Original cache geometry')
        require(bool(torch.isfinite(value).all()), 'Finite actual cache')
        rows.append(dict(name=f'kv_mgr.past_key_values.{index}', **endpoint.tensor_metadata(torch, value)))
    return rows


def compare_repeat(current, baseline):
    for field in REPEAT_FIELDS: same(current[field], baseline[field], 'Strict S1 exact S0 '+field)
    return dict(all7_inputs_exact=True, all213_taps_exact=True, all84_before_input_post_exact=True, full_logits_exact=True)


def same_state_values(left, right, message):
    fields = ('name', 'shape', 'dtype', 'sha256')
    same([{k:r[k] for k in fields} for r in left], [{k:r[k] for k in fields} for r in right], message)


def target_contract(torch, module, group):
    from neuronx_distributed.parallel_layers.layers import RowParallelLinear
    require(type(module) is RowParallelLinear and module.tensor_parallel_group is group and group.size() == 2, 'Exact actual TP2 RowParallelLinear')
    require(module.input_is_parallel is True and module.reduce_output is True and module.sequence_parallel_enabled is False,
            'Original local input/full reduced output geometry')
    require(module.bias is None and module.skip_bias_add is False and module.pad is False and module.tile_cc is False,
            'Bias-free untiled original output projection')
    require(module.weight.dtype == torch.bfloat16 and module.weight.device.type == 'cpu' and
            list(module.weight.shape) == [2048,1024] and module.weight.is_contiguous(), 'Original BF16 contiguous official shard')
    require(module.input_size == 2048 and module.output_size == 2048 and module.input_size_per_partition == 1024, 'Original projection dimensions')
    require(module.rank_ordering is None or list(module.rank_ordering) == [0,1], 'No special TP weight rank ordering')
    require(module.reduce_dtype == torch.bfloat16, 'Original reduction dtype retained in module config')


def verify_all_oproj(context, torch, model, rank, modules):
    """Bounded official matrix reads at runtime, one8MiB matrix at a time."""
    path = context.plan['official_layer0_weights']['path']; file = context.verify_input(path)
    headerref = context.plan['safetensors_header']; header = context.read_json(headerref['path'], headerref['sha256'])
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW); rows = []
    try:
        sizebytes = os.pread(fd, 8, 0); require(len(sizebytes) == 8, 'Original safetensors prefix')
        header_len, = struct.unpack('<Q', sizebytes); require(0 < header_len <= MIB, 'Bounded original header')
        header_bytes = os.pread(fd, header_len, 8); require(len(header_bytes) == header_len, 'Original header length')
        same(json.loads(header_bytes), header, 'Official header exact')
        for layer, name in enumerate(TARGETS):
            context.guard('cpu'); key = f'model.layers.{layer}.self_attn.o_proj.weight'; spec = header[key]
            require(spec['dtype'] == 'BF16' and spec['shape'] == [2048,2048], 'All42 official output weights')
            lo, hi = spec['data_offsets']; require(type(lo) is int and type(hi) is int and 0 <= lo < hi and hi-lo == 8*MIB and 8+header_len+hi <= file['identity']['size'], 'Bounded8MiB interval')
            data = bytearray()
            while len(data) < hi-lo:
                context.guard('cpu'); part = os.pread(fd, min(MIB, hi-lo-len(data)), 8+header_len+lo+len(data))
                require(bool(part), 'Official interval not truncated'); data.extend(part)
            official = torch.frombuffer(data, dtype=torch.bfloat16).reshape(2048,2048)
            local = modules[name].weight
            wanted = endpoint.tensor_digest(torch, official[:,rank*1024:(rank+1)*1024])
            require(endpoint.tensor_digest(torch, local) == wanted, 'Official column shard bytes layer'+str(layer))
            rows.append(dict(layer=layer, target=name, official_name=key, column_range=[rank*1024,(rank+1)*1024], sha256=wanted))
            del official, local, data
    finally: os.close(fd)
    same(identity(path), file['identity'], 'Official weight file unchanged')
    return rows


class OProjPatch:
    """Instance-only 42 target forwards. No parameter/config/global mutation."""
    def __init__(self, context, torch, model, rank, candidate, phase, official_proofs):
        from neuronx_distributed.parallel_layers import parallel_state
        from neuronx_distributed.parallel_layers.mappings import reduce_from_tensor_model_parallel_region
        self.context, self.torch, self.candidate = context, torch, candidate
        self.group = parallel_state.get_tensor_model_parallel_group()
        modules = dict(model.named_modules()); require(all(name in modules for name in TARGETS), 'All42 exact targets')
        self.modules = {name:modules[name] for name in TARGETS}; self.originals = {}; self.records = {}; self.expected_shape = None
        for module in self.modules.values(): target_contract(torch, module, self.group)
        if phase not in official_proofs:
            official_proofs[phase] = verify_all_oproj(context, torch, model, rank, self.modules)
        self.weights = copy.deepcopy(official_proofs[phase])
        for row in self.weights:
            require(endpoint.tensor_digest(torch, self.modules[row['target']].weight) == row['sha256'], 'Rebuilt shard equals official proof')
        self.weight_objects = {name:module.weight for name,module in self.modules.items()}
        if candidate:
            try:
                for name, module in self.modules.items():
                    require('forward' not in module.__dict__, 'Original instance forward not replaced')
                    self.originals[name] = module.forward
                    def forward(owner, input_, slice_indices=None, *, target=name):
                        require(slice_indices is None and self.expected_shape is not None, 'Exact finite unindexed projection call')
                        require(target not in self.records and owner is self.modules[target], 'One call at exact layer instance')
                        require(owner.weight is self.weight_objects[target], 'Official parameter object retained')
                        require(list(input_.shape) == self.expected_shape and input_.dtype == torch.bfloat16 and input_.device.type == 'cpu', 'Original full projection input')
                        self.context.guard('cpu')
                        partial = torch.einsum('...m,mn->...n', input_.float(), owner.weight.float().t())
                        require(partial.dtype == torch.float32 and list(partial.shape) == self.expected_shape[:2]+[2048], 'Local FP32 product retained')
                        reduced = reduce_from_tensor_model_parallel_region(partial, process_group=self.group)
                        require(reduced.dtype == torch.float32 and list(reduced.shape) == self.expected_shape[:2]+[2048], 'Actual FP32 TP2 reduction')
                        output = reduced.to(input_.dtype)
                        self.records[target] = dict(target=target, input_shape=list(input_.shape), input_stride=list(input_.stride()),
                            input_dtype=str(input_.dtype), local_dtype=str(partial.dtype), reduction_dtype=str(reduced.dtype),
                            output_dtype=str(output.dtype), output_shape=list(output.shape), actual_tp_size=self.group.size(),
                            actual_reduce_function='neuronx_distributed.parallel_layers.mappings.reduce_from_tensor_model_parallel_region',
                            final_cast_count=1, original_parameter_object_unchanged=True)
                        return output
                    module.forward = types.MethodType(forward, module)
            except BaseException:
                cleanup(None, self); raise
    def begin(self, call, bucket):
        self.records = {}; self.expected_shape = [1,bucket,1024] if call['phase'] == 'prefill' else [4,1,1024]
    def finish(self):
        same(list(self.records), list(TARGETS) if self.candidate else [], 'Exactly42 candidate dtype witnesses')
        return list(self.records.values())
    def close(self):
        errors = []
        for name in tuple(self.originals):
            try:
                del self.modules[name].forward
                require(self.modules[name].forward == self.originals[name], 'Original instance forward restored')
            except BaseException as error: errors.append(error)
        self.originals.clear(); self.records.clear(); self.expected_shape = None
        for row in self.weights:
            try:
                module = self.modules[row['target']]
                require(module.weight is self.weight_objects[row['target']], 'Official weight object unchanged at cleanup')
                require(endpoint.tensor_digest(self.torch, module.weight) == row['sha256'], 'Official weight bytes unchanged at cleanup')
            except BaseException as error: errors.append(error)
        if errors: raise errors[0]


def route_hooks(torch, model, inputs, call, bucket, runtime):
    """Check actual masks at all42 attention entry points, including prefill."""
    require(len(model.layers) == 42 and model.n_positions == bucket, 'Original layer/bucket geometry')
    records = {}; handles = []; prefill = call['phase'] == 'prefill'
    for name in ('is_prefix_caching', 'flash_decoding_enabled', 'sequence_parallel_enabled', 'k_cache_transposed'):
        require(getattr(model.neuron_config,name) is False, 'Original non-special route '+name)
    def hook(index):
        def observe(module, args, kwargs):
            require(not args and index not in records, 'One keyword attention call per layer')
            mask = kwargs['attention_mask']; positions = kwargs['position_ids']
            same(positions.tolist(), inputs[2].tolist(), 'Actual positions exactly input')
            require(kwargs.get('active_mask') is None, 'Single-branch prefill/unmasked current decode')
            shape = [1,1,bucket,bucket] if prefill else [4,1,1,bucket]
            require(list(mask.shape) == shape and mask.dtype == torch.bool, 'Original expanded mask geometry')
            if prefill:
                length = int(inputs[1].sum()); valid = mask[0,0,:length,:]
                expected = torch.arange(bucket)[None,:] <= torch.arange(length)[:,None]
                require(torch.equal(valid,expected), 'All real queries causal, every physical padded key excluded')
            else:
                require(torch.equal(mask[:,0,0,:],inputs[1].bool()), 'Actual strict/inclusive prior bits')
            records[index] = dict(layer=index, mask=runtime.tensor_metadata(mask), positions=runtime.tensor_metadata(positions),
                                  active_mask=None, physical_key_width=bucket, phase=call['phase'])
        return observe
    try:
        for index,layer in enumerate(model.layers): handles.append(layer.self_attn.register_forward_pre_hook(hook(index),with_kwargs=True))
    except BaseException:
        cleanup_actions([handle.remove for handle in handles], 'partial route hook registration')
        raise
    return handles, records


def native_references(context, torch):
    hf, metric = shared.references(context, torch); rows = []
    for frame, ref in zip(context.plan['frames'], context.plan['hf_frame_records']):
        row = context.read_json(ref['path'], ref['sha256'])
        same(row['frame'], frame, 'Original native32 endpoint identity')
        require(row['original_logits_raw_exact'] is True, 'Retained native baseline exact witness')
        rows.append(dict(frame=frame, record=ref, original_logits_raw_exact=True,
                         hf_reference_logits_sha256=endpoint.tensor_digest(torch,hf[frame['fixture']]['logits'][frame['step']])))
    require(len(rows) == 32, 'Complete unchanged HF32 reference')
    return hf, metric, rows


def inactive_preservation(torch, before, post, call):
    inactive = sorted(set(range(4)) - {f['slot'] for f in call['frames']})
    for a,b in zip(before,post):
        for slot in inactive:
            require(torch.equal(a[slot].contiguous().view(torch.int16), b[slot].contiguous().view(torch.int16)), 'Inactive cache slot unchanged')
        require(torch.equal(a[:,:,4096:,:].contiguous().view(torch.int16),b[:,:,4096:,:].contiguous().view(torch.int16)), 'Physical safety padding unchanged')
    return dict(inactive_slots=inactive, all84_inactive_slots_raw_exact=True, all84_physical_padding_raw_exact=True)


def cleanup_actions(actions, label):
    """Try every finite cleanup; retain an active primary or the first failure."""
    primary = sys.exc_info()[1]; first = None
    for action in actions:
        if action is None: continue
        try: action()
        except BaseException as error:
            if primary is None and first is None: first = error
            else:
                try: print('Secondary '+label+' failure: '+repr(error), file=sys.stderr)
                except BaseException: pass
    if primary is None and first is not None: raise first


def cleanup(hooks, patch):
    cleanup_actions((hooks.close if hooks is not None else None, patch.close if patch is not None else None), 'model cleanup')


def run_arm(context, torch, runtime, rank, expected, hf, metric, arm, root, baseline, official_proofs):
    strict = arm != ARMS[0]; candidate = arm == ARMS[3]
    folder = root/arm; folder.mkdir(exist_ok=False)
    seedref = expected[0][1]['imported_manifest']; seed = runtime.manifest(seedref,rank)
    model = hooks = patch = None; phase = None; previous_post = previous_meta = None
    reports = []; scored = []; constructors = []; baseline_rows = []
    try:
        for index,(ref,old) in enumerate(expected):
            context.guard('cpu'); graph = context.plan['graph_calls'][index]; call = graph['call']
            same(call,old['call'],'Original call identity'); original = runtime.graph_inputs(graph)
            inputs,widening = runtime.cpu_indices(original)
            same(widening,old['integer_widening'],'Original lossless CPU index adaptation')
            same([runtime.tensor_metadata(x)for x in inputs],old['execution_input_metadata'],'Original widened inputs')
            runtime.verify_original_positions_masks(inputs,call,context.plan['fixtures'])
            actual,cells,mapping,inputmeta = strict_inputs(context.plan,graph,inputs,strict,runtime)
            if phase != call['phase']:
                cleanup(hooks,patch); hooks=patch=model=None; gc.collect()
                weight_inputs=shared.authenticate_constructor_inputs(context,rank)
                model,proof=runtime.build_original_model(rank,call['phase'])
                same(shared.authenticate_constructor_inputs(context,rank),weight_inputs,'Original constructor inputs unchanged')
                require(proof['compiled'] is proof['resharded'] is proof['sdk_math_modified'] is False,'Original model before instance-only patch')
                patch=OProjPatch(context,torch,model,rank,candidate,call['phase'],official_proofs); hooks=runtime.prediction_hooks(model); phase=call['phase']
                constructors.append(dict(before_call=index,proof=proof,official_all42_oproj=patch.weights,inputs=weight_inputs,
                                         candidate_patch_installed=candidate))
            if index == 0:
                require(previous_post is None,'Only first seed import')
                for parameter,row in zip(model.kv_mgr.past_key_values,seed): parameter.copy_(runtime.load_state(row))
                endpoint.verify_states(torch,model.kv_mgr.past_key_values,seed,rank,'initial seed')
            else:
                require(previous_post is not None,'Actual prior poststate required across groups/phases')
                same(state_metadata(torch,previous_post),previous_meta,'Previous actual post unchanged')
                for parameter,value in zip(model.kv_mgr.past_key_values,previous_post): parameter.copy_(value)
            before=state_metadata(torch,model.kv_mgr.past_key_values)
            if index: same_state_values(before,previous_meta,'Every next before equals previous actual post')
            previous_post=None
            bucket=runtime.select_original_bucket(model,graph['bucket']); patch.begin(call,graph['bucket'])
            prefill=call['phase']=='prefill'; lengths=actual[1].sum(-1).clamp_min(1)if prefill else actual[2][:,0]+1
            handles,routes=route_hooks(torch,model,actual,call,graph['bucket'],runtime)
            try:
                hooks.begin(lengths,prefill); output,*post=model(*actual); taps,tapmeta=hooks.finish()
            finally:
                cleanup_actions([handle.remove for handle in handles], 'hook removal')
            error=None; checked=None; repeat=None
            try:
                same(list(routes),list(range(42)),'All42 actual mask/position routes')
                witnesses=patch.finish()
                require(list(output.shape)==[len(mapping),1,130560]and output.dtype==torch.float32 and bool(torch.isfinite(output).all()),'Finite full logits')
                same(list(taps),old['stage_names'],'All213 ordered original tap names')
                require(all(bool(torch.isfinite(x).all())for x in taps.values()),'All213 finite taps')
                same([runtime.tensor_metadata(x)for x in actual],inputmeta,'Seven inputs unchanged by forward')
                input_after=state_metadata(torch,model.kv_mgr.past_key_values); same(input_after,before,'All84 input states unchanged')
                postmeta=state_metadata(torch,post); preserve=inactive_preservation(torch,model.kv_mgr.past_key_values,post,call)
                record=dict(call=call,bucket=graph['bucket'],input_metadata=graph['input_metadata'],execution_input_metadata=inputmeta,
                    integer_widening=widening,changed_mask_cells=cells,output_metadata=endpoint.tensor_metadata(torch,output),
                    stage_names=list(taps),stage_metadata=tapmeta,stage_sha256={n:endpoint.tensor_digest(torch,v)for n,v in taps.items()},
                    before_metadata=before,input_after_metadata=input_after,post_metadata=postmeta,route_records=list(routes.values()))
                if not strict:
                    oldbefore=runtime.manifest(old['imported_manifest'],rank); oldafter=runtime.manifest(old['after'],rank)
                    checked=endpoint.cpu_endpoint(torch,output,taps,tapmeta,model.kv_mgr.past_key_values,post,old,oldbefore,oldafter,rank)
                if arm==ARMS[2]: repeat=compare_repeat(record,baseline[index])
                scored_call=trajectory.scoring(output,hf,mapping,metric)
            except Exception as exc: error=type(exc).__name__+': '+str(exc)
            # Both ranks finish the current graph before deciding whether any
            # can enter the next graph/arm. Root owns termination for native hangs.
            call_dir=folder/f'call-{index:02d}';call_dir.mkdir(exist_ok=False)
            if error is not None:
                try: shared.write_json(call_dir/'failed-check.json',dict(arm=arm,rank=rank,call=call,error=error))
                except BaseException as secondary:
                    try: print('Secondary failed-check write: '+repr(secondary),file=sys.stderr)
                    except BaseException: pass
            shared.collective_control_gate(torch,error)
            scored.extend(scored_call)
            record.update(schema='minicpm5_original32_oproj_cpu_call_v1',complete=True,arm=arm,rank=rank,
                original_control=ref,inclusive_original_endpoint=checked,strict_repeat=repeat,
                history_origin='original_call0_seed'if index==0 else 'previous_actual_post_same_arm',
                seed_imported_this_call=index==0,group_reset_preserves_cache=index==7,
                preservation=preserve,physical_rows=mapping,HF_metrics=scored_call,oproj_dtype_records=witnesses,
                candidate_only42_instance_targets=candidate,full_model_qualified=False,new_speedup_claim=False)
            record['payload']=trajectory.save_output(torch,call_dir/'output.pt',output)
            shared.write_json(call_dir/'result.json',record)
            reports.append(dict(call=index,report=str(call_dir/'result.json'),sha256=hashlib.sha256((call_dir/'result.json').read_bytes()).hexdigest(),payload=record['payload']))
            if arm==ARMS[1]:baseline_rows.append({key:copy.deepcopy(record[key])for key in REPEAT_FIELDS})
            previous_post=post;previous_meta=postmeta
            del output,taps,actual,inputs,original
        require(len(reports)==14 and len(scored)==32,'Complete original trajectory')
        summary=trajectory.summarize(context.plan,scored)
        result=dict(schema='minicpm5_original32_oproj_cpu_arm_v1',complete=True,arm=arm,rank=rank,reports=reports,rows=scored,summary=summary,
            constructors=constructors,model_forwards=14,seed_import_count=1,continuous_actual_cache_chain=True,
            all14_inclusive_endpoints_exact=not strict,all14_strict_repeat_exact=arm==ARMS[2],
            full_cache_payloads_written=0,tap_payloads_written=0,full_model_qualified=False)
        shared.write_json(folder/'result.json',result)
        return result,baseline_rows
    finally: cleanup(hooks,patch)


def execute(context, output_dir, torch, runtime):
    rank=context.permit['rank']; expected=expected_records(context,rank); hf,metric,hf_evidence=native_references(context,torch)
    results=[]; baseline=None; official_proofs={}
    for index,arm in enumerate(ARMS):
        require(len(results)==index,'Fixed arm order')
        if index==1:require(results[0]['all14_inclusive_endpoints_exact'],'Inclusive own-history gate before strict')
        if index==2:require(baseline is not None and len(baseline)==14,'StrictS0 before S1')
        if index==3:require(results[2]['all14_strict_repeat_exact'],'Strict repeat before candidate')
        result,retained=run_arm(context,torch,runtime,rank,expected,hf,metric,arm,output_dir,baseline,official_proofs)
        results.append(result)
        if index==1:baseline=retained
        shared.collective_control_gate(torch,None)
    result=dict(schema='minicpm5_original32_oproj_cpu_rank_v1',complete=True,rank=rank,arms=list(ARMS),
        arm_reports=[dict(arm=a,report=str(output_dir/a/'result.json'),sha256=hashlib.sha256((output_dir/a/'result.json').read_bytes()).hexdigest())for a in ARMS],
        summaries={x['arm']:x['summary']for x in results},model_forwards=56,seed_import_count=4,S1_repeat_all14_exact=True,
        native32_evidence=hf_evidence,original_HF_reference=context.plan['hf_reference'],
        full_cache_payloads_written=0,tap_payloads_written=0,original_remaining_gates=context.plan['remaining_gates'],
        self_exit_claimed=False,root_must_observe_actual_exit=True,full_model_qualified=False,new_speedup_claim=False)
    shared.write_json(output_dir/'result.json',result);return result


def run(context, output_dir):
    context.guard('cpu'); require(context.permit['arm']=='C_Hc_history_generation'and type(context.permit['rank'])is int and context.permit['rank']in(0,1),'Original own-history rank permit')
    output_dir=Path(output_dir);require(output_dir==Path(context.plan['output_root'])/f'cpu-rank{context.permit["rank"]}','Dedicated rank output')
    context.verify_sources();validate_plan(context.plan,context.read_json)
    require(str(Path(__file__).resolve())in context.plan['source_sha256'],'Worker authenticated in new source closure')
    output_dir.mkdir(exist_ok=False)
    with context.activate():
        import torch
        importlib.import_module('neuronx_distributed_inference.modules.lora_serving')
        from neuronx_distributed.parallel_layers import parallel_state
        from minicpm5_layer0_cpu_adapter_v2 import CPUAdapter
        witness=context.record_cpu_import();torch.set_num_threads(1)
        torch.distributed.init_process_group('gloo')
        initialized=False
        try:
            require(torch.distributed.get_world_size()==2 and torch.distributed.get_rank()==context.permit['rank'],'Actual TP2 rank')
            parallel_state.initialize_model_parallel(2,skip_collective_init=True);initialized=True
            shared.write_json(output_dir/'sdk-environment-transition.json',witness)
            with torch.inference_mode():return execute(context,output_dir,torch,CPUAdapter(context,torch))
        finally:
            cleanup_actions(([parallel_state.destroy_model_parallel] if initialized else [])+[torch.distributed.destroy_process_group], 'distributed cleanup')
