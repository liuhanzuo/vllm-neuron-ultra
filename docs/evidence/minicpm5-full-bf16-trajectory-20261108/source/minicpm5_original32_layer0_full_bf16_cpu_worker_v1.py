"""Three finite layer-0 phase interventions; CPU runtime admission belongs to root.

Import and policy checks are stdlib/source only. Runtime retains the original
14-call/32-frame own-cache histories. Selected payloads are prediction queries,
not full intermediate tensors: four BF16 vectors/frame, 3 MiB across both ranks.
"""
import copy
import gc
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types

import minicpm5_original32_oproj_cpu_worker_v1 as base

ROOT, MIB = base.ROOT, base.MIB
require, same = base.require, base.same
shared, endpoint, trajectory, identity = base.shared, base.endpoint, base.trajectory, base.identity
ORIGINAL, SUCCESSOR, CORE, REPEAT_FIELDS = base.ORIGINAL, base.SUCCESSOR, base.CORE, base.REPEAT_FIELDS
ALL_TARGETS = base.TARGETS
PATCH_TARGETS = (ALL_TARGETS[0],)
ARMS = ('strict_anchor_s0', 'strict_layer0_full_bf16_prefill_only', 'strict_layer0_full_bf16_both_phases')
SELECTED_TAPS = ('layers.0.self_attn', 'layers.0', 'layers.41.self_attn', 'layers.41')
UPSTREAM_TAPS = ('embed_tokens', 'layers.0.input_layernorm')
PREDECESSOR_PLAN = dict(path=str(ROOT/'research/minicpm5-original32-oproj-cpu-input-plan-20261107-v1.json'),
    sha256='0037e8c1464953ddebba2c6ef20f5a5464125053d631fda417d56903eb8eed7c')
PREDECESSOR_EXIT = dict(path=str(ROOT/'research/minicpm5-original32-oproj-cpu-root-exit-r1-20261107-v1.json'),
    sha256='737d97c3b5675a060c05fef16d44548898812518327d75c37d707bbc94ad2b6d')
PREDECESSOR_SUMMARY = dict(path=str(ROOT/'research/minicpm5-original32-oproj-cpu-root-summary-r1-20261107-v1.json'),
    sha256='563a8df8425bece92afe29711214c4e727b05a0331715975a5005dae571610ab')
DESIGN = dict(path=str(ROOT/'research/minicpm5-layer0-full-bf16-next-design-20261108-v2.json'),
    sha256='f19895d9466ae42cd5e0a9b7022b9a68d3cf554669221dc18a2086935c56229d')


GATHER_TERMINAL = dict(path=str(ROOT/'research/minicpm5-full-bf16-gather-r1-20261108-v1-root-terminal.json'),sha256='6dda6b29935d8760c1e923cf9478a0134e0052d2fe5bf8c798028a271dcc659f')
GATHER_EXIT = dict(path=str(ROOT/'research/minicpm5-full-bf16-gather-r1-20261108-v1-finalizer-external-exit.json'),sha256='97517988cf617efa0774cc6a3b67d508c305589c077dd007638d5d8fbc04081f')
FULL_WEIGHT_SHA = 'f641e83dd6533cd4c9ea6f5b3689d4c952c86ca18c7e765db0399adfd259ed78'

def selected_targets(arm, phase):
    require(arm in ARMS and phase in ('prefill', 'decode'), 'Exact finite arm and phase')
    return PATCH_TARGETS if arm == ARMS[2] or (arm == ARMS[1] and phase == 'prefill') else ()


def validate_plan(plan, read_json):
    spec = plan['layer0_full_bf16_trajectory']
    require(set(spec) == {'schema', 'arms', 'predecessor_plan', 'predecessor_actual_exit', 'predecessor_summary',
                     'strict_anchor_reports', 'design', 'per_rank_output_cap_bytes', 'total_output_cap_bytes',
                     'selected_tap_names'}, 'Exact layer0 phase plan fields')
    same(spec['schema'], 'minicpm5_original32_layer0_full_bf16_cpu_plan_v1', 'Finite successor schema')
    same(spec['arms'], list(ARMS), 'Exact three arm order')
    for key, ref in [('predecessor_plan', PREDECESSOR_PLAN), ('predecessor_actual_exit', PREDECESSOR_EXIT),
                     ('predecessor_summary', PREDECESSOR_SUMMARY), ('design', DESIGN)]:
        same(spec[key], ref, 'Bound predecessor '+key)
    same(spec['selected_tap_names'], list(SELECTED_TAPS), 'Four finite selected prediction taps')
    same([spec['per_rank_output_cap_bytes'], spec['total_output_cap_bytes']], [128*MIB, 256*MIB], 'Finite caps')
    prior = read_json(PREDECESSOR_PLAN['path'], PREDECESSOR_PLAN['sha256']); base.validate_plan(prior, read_json)
    require(set(plan) == set(prior) | {'layer0_full_bf16_trajectory'}, 'Only finite extension added')
    for key in prior:
        if key not in ('output_root', 'source_sha256', 'inputs'):
            same(plan[key], prior[key], 'Original inherited plan '+key)
    for field in ('inputs', 'source_sha256'):
        require(all(plan[field].get(k) == v for k,v in prior[field].items()), 'Retain predecessor '+field)
    ended = read_json(PREDECESSOR_EXIT['path'], PREDECESSOR_EXIT['sha256'])
    require(ended['actual_exit_observed'] is True and type(ended['actual_exit_code']) is int and ended['actual_exit_code'] == 0,
            'Actual predecessor exit0')
    require(ended['all_four_processes_gone'] is True and ended['cgroup_empty'] is True and ended['canonical_unchanged'] is True,
            'Predecessor fully exited unchanged')
    summary = read_json(PREDECESSOR_SUMMARY['path'], PREDECESSOR_SUMMARY['sha256'])
    same(summary['actual_exit'], PREDECESSOR_EXIT, 'Summary bound actual exit')
    require(summary['strict_baseline_repeat_exact'] is True, 'Frozen S0/S1 repeat prerequisite')
    same(summary['newly_failed_frames'], [5,16], 'Negative candidate retained')
    require(summary['full_model_qualified'] is False and summary['new_speedup_claim'] is False, 'No predecessor qualification')
    design = read_json(DESIGN['path'], DESIGN['sha256'])
    require(design['full_model_qualified'] is False and design['source_only'] is True and design['local_gather_gate_complete'] is True, 'Full-model design after complete gather control')
    terminal=read_json(GATHER_TERMINAL['path'],GATHER_TERMINAL['sha256'])
    gather_exit=read_json(GATHER_EXIT['path'],GATHER_EXIT['sha256'])
    require(terminal['passed'] and terminal['all12_new_payloads_independently_compared'] and len(terminal['cases'])==4,'Actual complete gather proof')
    same(gather_exit['terminal'],GATHER_TERMINAL,'Actual gather verifier exit binding')
    same(gather_exit['owner'],terminal['verifier'],'Actual gather verifier identity')
    require(gather_exit['actual_tool_exit_observed'] and gather_exit['exact_owner_gone'] and gather_exit['actual_exit']==dict(session_id=93505,exit_code=0,tool_chunk='24dfe8'),'Observed complete actual gather verifier exit')
    same(design['actual_gather_terminal'],GATHER_TERMINAL,'Design binds completed gather')
    wanted = []
    for rank in (0,1):
        rankref = ended['outputs'][f'rank{rank}_result']; rank_result = read_json(rankref['path'], rankref['sha256'])
        require(rank_result['complete'] is True and rank_result['rank'] == rank and rank_result['S1_repeat_all14_exact'] is True,
                'Complete actual predecessor rank')
        arm_refs = [x for x in rank_result['arm_reports'] if x['arm'] == 'strict_cpu_s0']
        require(len(arm_refs) == 1, 'One frozen strict S0 arm')
        arm = read_json(arm_refs[0]['report'], arm_refs[0]['sha256'])
        require(arm['complete'] is True and arm['arm'] == 'strict_cpu_s0' and arm['rank'] == rank, 'Frozen anchor identity')
        same([x['call'] for x in arm['reports']], list(range(14)), 'All14 anchor calls')
        for row in arm['reports']:
            wanted.append(dict(rank=rank, call=row['call'], path=row['report'], sha256=row['sha256']))
    same(spec['strict_anchor_reports'], wanted, 'Exact actual S0 call references both ranks')
    require(len(plan['calls']) == 14 and len(plan['frames']) == 32, 'Original14/32 retained')
    root = Path(plan['output_root'])
    require(root.parent == Path('/mnt/nvme') and root.name.startswith('minicpm5-original32-layer0-full-bf16-cpu-'), 'New output namespace')
    return spec


def anchor_records(context, rank):
    refs = [x for x in context.plan['layer0_full_bf16_trajectory']['strict_anchor_reports'] if x['rank'] == rank]
    same([x['call'] for x in refs], list(range(14)), 'Ordered frozen S0 calls')
    rows = []
    for index, ref in enumerate(refs):
        row = context.read_json(ref['path'], ref['sha256'])
        require(row['complete'] is True and row['arm'] == 'strict_cpu_s0' and row['rank'] == rank, 'Exact frozen S0 record')
        same(row['call'], context.plan['calls'][index], 'Original call identity at anchor')
        require(len(row['stage_names']) == 213 and row['full_model_qualified'] is False, 'Complete original anchor')
        rows.append((ref, row))
    return rows


def compare_upstream(current, anchor):
    for name in UPSTREAM_TAPS:
        same(current['stage_metadata'][name], anchor['stage_metadata'][name], 'Upstream original geometry '+name)
        same(current['stage_sha256'][name], anchor['stage_sha256'][name], 'Upstream original prediction bytes '+name)
    return dict(names=list(UPSTREAM_TAPS), all_exact=True)


class OProjPatch:
    """One instance selected by phase; all42 weights and original methods checked."""
    def __init__(self, context, torch, model, rank, arm, phase, official_proofs):
        from neuronx_distributed.parallel_layers import parallel_state
        from research.minicpm5_full_bf16_gather_v1 import full_bf16_projection
        self.context, self.torch, self.arm, self.phase = context, torch, arm, phase
        self.targets = selected_targets(arm, phase)
        self.group = parallel_state.get_tensor_model_parallel_group()
        modules = dict(model.named_modules()); require(all(name in modules for name in ALL_TARGETS), 'All42 targets present')
        self.modules = {name:modules[name] for name in ALL_TARGETS}
        self.originals = {}; self.installed = {}; self.records = {}; self.expected_shape = None
        for module in self.modules.values():
            base.target_contract(torch, module, self.group)
            require('forward' not in module.__dict__, 'All42 original instance methods')
        self.all_originals = {name:module.forward for name,module in self.modules.items()}
        if phase not in official_proofs:
            official_proofs[phase] = base.verify_all_oproj(context, torch, model, rank, self.modules)
        self.weights = copy.deepcopy(official_proofs[phase])
        same([x['target'] for x in self.weights], list(ALL_TARGETS), 'All42 official proofs in layer order')
        for row in self.weights:
            require(endpoint.tensor_digest(torch, self.modules[row['target']].weight) == row['sha256'], 'Official rebuilt shard')
        self.weight_objects = {name:module.weight for name,module in self.modules.items()}
        self.full_weight = None
        self.full_weight_sha = FULL_WEIGHT_SHA
        require(torch.distributed.get_process_group_ranks(self.group)==[0,1] and torch.distributed.get_rank(self.group)==rank,'Actual NxD TP2 original rank order')
        if self.targets:
            # Authenticate the complete matrix reconstructed from unchanged official column shards.
            with torch.inference_mode(False):
                parts=[torch.empty_like(self.modules[PATCH_TARGETS[0]].weight) for _ in (0,1)]
            torch.distributed.all_gather(parts,self.modules[PATCH_TARGETS[0]].weight,group=self.group)
            self.full_weight=torch.cat(parts,dim=1).contiguous()
            require(list(self.full_weight.shape)==[2048,2048] and self.full_weight.dtype==torch.bfloat16,'Original full BF16 weight geometry')
            same(endpoint.tensor_digest(torch,self.full_weight),FULL_WEIGHT_SHA,'Original official full8MiB weight bytes')
            del parts
        try:
            for name in self.targets:
                module = self.modules[name]; self.originals[name] = module.forward
                def forward(owner, input_, slice_indices=None, *, target=name):
                    require(slice_indices is None and self.expected_shape is not None, 'Finite unindexed projection')
                    require(target in self.targets and target not in self.records and owner is self.modules[target], 'One selected actual instance')
                    require(owner.weight is self.weight_objects[target], 'Official parameter object unchanged')
                    require(list(input_.shape) == self.expected_shape and input_.dtype == torch.bfloat16 and input_.device.type == 'cpu', 'Original projection input')
                    self.context.guard('cpu')
                    same(endpoint.tensor_digest(torch,self.full_weight),FULL_WEIGHT_SHA,'Complete original weight before projection')
                    input_before=endpoint.tensor_digest(torch,input_)
                    output,gathered=full_bf16_projection(torch,input_,self.full_weight,self.group)
                    same(endpoint.tensor_digest(torch,input_),input_before,'Original projection input unchanged')
                    same(endpoint.tensor_digest(torch,self.full_weight),FULL_WEIGHT_SHA,'Complete original weight after projection')
                    self.records[target] = dict(target=target,input_shape=list(input_.shape),input_stride=list(input_.stride()),
                        input_dtype=str(input_.dtype),gathered_shape=list(gathered.shape),gathered_dtype=str(gathered.dtype),
                        weight_dtype=str(self.full_weight.dtype),full_weight_shape=list(self.full_weight.shape),full_weight_sha256=FULL_WEIGHT_SHA,
                        output_dtype=str(output.dtype),output_shape=list(output.shape),actual_tp_size=self.group.size(),
                        actual_group_ranks=torch.distributed.get_process_group_ranks(self.group),
                        actual_gather_function='torch.distributed.all_gather',full_linear_calls=1,output_reduction=False,
                        explicit_fp32_casts=0,original_parameter_object_unchanged=True)
                    return output
                installed = types.MethodType(forward, module)
                self.installed[name] = installed; module.forward = installed
            self.verify_methods()
        except BaseException:
            base.cleanup(None, self); raise

    def verify_methods(self):
        for name, module in self.modules.items():
            if name in self.targets:
                require(module.__dict__.get('forward') is self.installed[name], 'Selected method identity retained')
            else:
                require('forward' not in module.__dict__ and module.forward == self.all_originals[name], 'Unselected original method retained')

    def begin(self, call, bucket):
        same(call['phase'], self.phase, 'Patch constructed for actual phase')
        same(self.targets, selected_targets(self.arm, call['phase']), 'Actual per-call phase policy')
        self.verify_methods(); self.records = {}
        self.expected_shape = [1,bucket,1024] if call['phase'] == 'prefill' else [4,1,1024]

    def finish(self):
        self.verify_methods()
        same(list(self.records), list(self.targets), 'Exactly0or1 phase-selected dtype witnesses')
        return list(self.records.values())

    def close(self):
        errors = []
        for name in tuple(self.originals):
            try:
                require(self.modules[name].__dict__.get('forward') is self.installed[name], 'Selected method unchanged before restore')
            except BaseException as error: errors.append(error)
            try:
                del self.modules[name].forward
                require(self.modules[name].forward == self.originals[name], 'Original selected method restored')
            except BaseException as error: errors.append(error)
        self.originals.clear(); self.records.clear(); self.expected_shape = None
        for name,module in self.modules.items():
            try:
                require('forward' not in module.__dict__ and module.forward == self.all_originals[name], 'All42 original methods restored')
            except BaseException as error: errors.append(error)
        for row in self.weights:
            try:
                module = self.modules[row['target']]
                require(module.weight is self.weight_objects[row['target']], 'Official weight object at cleanup')
                require(endpoint.tensor_digest(self.torch, module.weight) == row['sha256'], 'Official weight bytes at cleanup')
            except BaseException as error: errors.append(error)
        if self.full_weight is not None:
            try: same(endpoint.tensor_digest(self.torch,self.full_weight),FULL_WEIGHT_SHA,'Original full weight retained at cleanup')
            except BaseException as error:errors.append(error)
            self.full_weight=None
        if errors: raise errors[0]


def save_selected_taps(context, torch, directory, taps, record, mapping):
    """One finite raw bundle/call, 4 gathered tensors, no full-shape export."""
    require(sys.byteorder == 'little', 'Explicit raw BF16 little-endian host')
    same(sorted(x['physical_row'] for x in mapping), list(range(len(mapping))), 'Complete physical-row permutation')
    physical_frames = [x['frame']['index'] for x in sorted(mapping, key=lambda x:x['physical_row'])]
    require(len(physical_frames) in (1,4) and len(set(physical_frames)) == len(physical_frames), 'Finite distinct physical frames')
    entries = []; parts = []; offset = 0
    for name in SELECTED_TAPS:
        context.guard('cpu'); value = taps[name]
        require(list(value.shape) == [len(mapping),1,2048] and value.dtype == torch.bfloat16 and value.device.type == 'cpu', 'Selected prediction geometry')
        raw = bytes(memoryview(value.detach().contiguous().reshape(-1).view(torch.uint8).numpy()))
        require(len(raw) == len(mapping)*4096 and 0 < len(raw) <= 16384, 'Finite raw selected entry')
        digest = hashlib.sha256(raw).hexdigest(); same(digest, record['stage_sha256'][name], 'Selected entry matches existing tap digest')
        entries.append(dict(name=name,offset=offset,raw_bytes=len(raw),sha256=digest,shape=list(value.shape),
                            dtype=str(value.dtype),source_metadata=record['stage_metadata'][name],
                            frame_indices=physical_frames))
        parts.append(raw); offset += len(raw)
    raw = b''.join(parts)
    require(len(raw) == 4*len(mapping)*4096 and 0 < len(raw) <= 65536, 'Finite selected bundle')
    path = directory/'selected-taps.bf16'
    with path.open('xb') as file:
        require(file.write(raw) == len(raw), 'Complete selected bundle write')
    context.guard('cpu')
    return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),identity=identity(path),
                encoding='raw_bf16_le',raw_bytes=len(raw),entries=entries)


def run_arm(context, torch, runtime, rank, expected, anchors, hf, metric, arm, root, prefill_only, official_proofs):
    require(arm in ARMS, 'Finite arm')
    folder = root/arm; folder.mkdir(exist_ok=False)
    seedref = expected[0][1]['imported_manifest']; seed = runtime.manifest(seedref,rank)
    model = hooks = patch = None; phase = None; previous_post = previous_meta = None
    reports = []; scored = []; constructors = []; retained = []
    try:
        for index,(ref,old) in enumerate(expected):
            context.guard('cpu'); graph = context.plan['graph_calls'][index]; call = graph['call']; anchor_ref,anchor = anchors[index]
            same(call,old['call'],'Original call identity'); original = runtime.graph_inputs(graph)
            inputs,widening = runtime.cpu_indices(original)
            same(widening,old['integer_widening'],'Original lossless CPU index adaptation')
            same([runtime.tensor_metadata(x)for x in inputs],old['execution_input_metadata'],'Original widened inputs')
            runtime.verify_original_positions_masks(inputs,call,context.plan['fixtures'])
            actual,cells,mapping,inputmeta = base.strict_inputs(context.plan,graph,inputs,True,runtime)
            if phase != call['phase']:
                base.cleanup(hooks,patch); hooks=patch=model=None; gc.collect()
                weight_inputs=shared.authenticate_constructor_inputs(context,rank)
                model,proof=runtime.build_original_model(rank,call['phase'])
                same(shared.authenticate_constructor_inputs(context,rank),weight_inputs,'Original constructor inputs unchanged')
                require(proof['compiled'] is proof['resharded'] is proof['sdk_math_modified'] is False,'Original model constructor')
                patch=OProjPatch(context,torch,model,rank,arm,call['phase'],official_proofs)
                hooks=runtime.prediction_hooks(model); phase=call['phase']
                constructors.append(dict(before_call=index,proof=proof,official_all42_oproj=patch.weights,inputs=weight_inputs,
                                         candidate_patch_installed=bool(patch.targets),candidate_targets=list(patch.targets),phase=phase))
            if index == 0:
                require(previous_post is None,'Only first seed import')
                for parameter,row in zip(model.kv_mgr.past_key_values,seed): parameter.copy_(runtime.load_state(row))
                endpoint.verify_states(torch,model.kv_mgr.past_key_values,seed,rank,'initial seed')
            else:
                require(previous_post is not None,'Actual prior poststate across groups/phases')
                same(base.state_metadata(torch,previous_post),previous_meta,'Previous actual post unchanged')
                for parameter,value in zip(model.kv_mgr.past_key_values,previous_post): parameter.copy_(value)
            before=base.state_metadata(torch,model.kv_mgr.past_key_values)
            if index: base.same_state_values(before,previous_meta,'Every before equals previous actual post')
            previous_post=None
            bucket=runtime.select_original_bucket(model,graph['bucket']); patch.begin(call,graph['bucket'])
            prefill=call['phase']=='prefill'; lengths=actual[1].sum(-1).clamp_min(1)if prefill else actual[2][:,0]+1
            handles,routes=base.route_hooks(torch,model,actual,call,graph['bucket'],runtime)
            try:
                hooks.begin(lengths,prefill); output,*post=model(*actual); taps,tapmeta=hooks.finish()
            finally:
                base.cleanup_actions([handle.remove for handle in handles], 'hook removal')
            error=None; anchor_exact=None; phase_repeat=None; phase_before=None
            try:
                same(list(routes),list(range(42)),'All42 mask/position routes'); witnesses=patch.finish()
                require(list(output.shape)==[len(mapping),1,130560]and output.dtype==torch.float32 and bool(torch.isfinite(output).all()),'Finite full logits')
                same(list(taps),old['stage_names'],'All213 ordered original tap names')
                require(all(bool(torch.isfinite(x).all())for x in taps.values()),'All213 finite taps')
                same([runtime.tensor_metadata(x)for x in actual],inputmeta,'Seven inputs unchanged by forward')
                input_after=base.state_metadata(torch,model.kv_mgr.past_key_values); same(input_after,before,'All84 input states unchanged')
                postmeta=base.state_metadata(torch,post); preserve=base.inactive_preservation(torch,model.kv_mgr.past_key_values,post,call)
                record=dict(call=call,bucket=graph['bucket'],input_metadata=graph['input_metadata'],execution_input_metadata=inputmeta,
                    integer_widening=widening,changed_mask_cells=cells,output_metadata=endpoint.tensor_metadata(torch,output),
                    stage_names=list(taps),stage_metadata=tapmeta,stage_sha256={n:endpoint.tensor_digest(torch,v)for n,v in taps.items()},
                    before_metadata=before,input_after_metadata=input_after,post_metadata=postmeta,route_records=list(routes.values()))
                if arm==ARMS[0]: anchor_exact=base.compare_repeat(record,anchor)
                upstream=compare_upstream(record,anchor)
                if arm==ARMS[2] and index<4:
                    phase_repeat=base.compare_repeat(record,prefill_only[index])
                if arm==ARMS[2] and index==4:
                    same(record['before_metadata'],prefill_only[index]['before_metadata'],'First decode equal actual incoming states')
                    phase_before=True
                scored_call=trajectory.scoring(output,hf,mapping,metric)
            except Exception as exc: error=type(exc).__name__+': '+str(exc)
            call_dir=folder/f'call-{index:02d}';call_dir.mkdir(exist_ok=False)
            if error is not None:
                try: shared.write_json(call_dir/'failed-check.json',dict(arm=arm,rank=rank,call=call,error=error))
                except BaseException as secondary:
                    try: print('Secondary failed-check write: '+repr(secondary),file=sys.stderr)
                    except BaseException: pass
            shared.collective_control_gate(torch,error)
            scored.extend(scored_call)
            record.update(schema='minicpm5_original32_layer0_full_bf16_cpu_call_v1',complete=True,arm=arm,rank=rank,
                original_control=ref,strict_anchor_ref=anchor_ref if arm==ARMS[0] else None,strict_anchor_exact=anchor_exact,
                history_origin='original_call0_seed'if index==0 else 'previous_actual_post_same_arm',
                seed_imported_this_call=index==0,group_reset_preserves_cache=index==7,
                preservation=preserve,physical_rows=mapping,HF_metrics=scored_call,oproj_dtype_records=witnesses,
                candidate_layer0_full_bf16_enabled=bool(patch.targets),candidate_targets=list(patch.targets),
                upstream_matches_anchor=upstream,first_group_prefill_repeat_exact=phase_repeat,
                call4_before_equal_prefill_only=phase_before,full_model_qualified=False,new_speedup_claim=False)
            record['payload']=trajectory.save_output(torch,call_dir/'output.pt',output)
            record['selected_tap_bundle']=save_selected_taps(context,torch,call_dir,taps,record,mapping)
            shared.write_json(call_dir/'result.json',record)
            reports.append(dict(call=index,report=str(call_dir/'result.json'),sha256=hashlib.sha256((call_dir/'result.json').read_bytes()).hexdigest(),payload=record['payload']))
            retained.append({key:copy.deepcopy(record[key])for key in REPEAT_FIELDS})
            previous_post=post;previous_meta=postmeta
            del output,taps,actual,inputs,original
        require(len(reports)==14 and len(scored)==32,'Complete original trajectory')
        summary=trajectory.summarize(context.plan,scored)
        result=dict(schema='minicpm5_original32_layer0_full_bf16_cpu_arm_v1',complete=True,arm=arm,rank=rank,reports=reports,rows=scored,summary=summary,
            constructors=constructors,model_forwards=14,seed_import_count=1,continuous_actual_cache_chain=True,
            anchor_all14_exact=arm==ARMS[0],first_group_prefill_repeat_all4_exact=arm==ARMS[2],call4_before_equal_prefill_only=arm==ARMS[2],
            full_cache_payloads_written=0,tap_payloads_written=14,selected_tap_payloads_written=14,selected_tap_raw_bytes=524288,
            selected_tap_names=list(SELECTED_TAPS),selected_taps_are_prediction_queries=True,full_model_qualified=False,new_speedup_claim=False)
        shared.write_json(folder/'result.json',result)
        return result,retained
    finally: base.cleanup(hooks,patch)


def execute(context, output_dir, torch, runtime):
    rank=context.permit['rank']; expected=base.expected_records(context,rank); anchors=anchor_records(context,rank)
    hf,metric,hf_evidence=base.native_references(context,torch)
    results=[]; prefill_only=None; official_proofs={}
    for index,arm in enumerate(ARMS):
        require(len(results)==index,'Exact arm order')
        if index:require(results[0]['anchor_all14_exact'],'Frozen S0 exact before candidates')
        if index==2:require(prefill_only is not None and len(prefill_only)==14,'Prefill-only control before both phases')
        result,retained=run_arm(context,torch,runtime,rank,expected,anchors,hf,metric,arm,output_dir,prefill_only,official_proofs)
        results.append(result)
        if index==1:prefill_only=retained
        shared.collective_control_gate(torch,None)
    result=dict(schema='minicpm5_original32_layer0_full_bf16_cpu_rank_v1',complete=True,rank=rank,arms=list(ARMS),
        arm_reports=[dict(arm=a,report=str(output_dir/a/'result.json'),sha256=hashlib.sha256((output_dir/a/'result.json').read_bytes()).hexdigest())for a in ARMS],
        summaries={x['arm']:x['summary']for x in results},model_forwards=42,seed_import_count=3,anchor_all14_exact=True,
        first_group_prefill_repeat_all4_exact=True,call4_before_equal_prefill_only=True,
        native32_evidence=hf_evidence,original_HF_reference=context.plan['hf_reference'],
        full_cache_payloads_written=0,tap_payloads_written=42,selected_tap_payloads_written=42,selected_tap_raw_bytes=1572864,
        selected_tap_names=list(SELECTED_TAPS),selected_taps_are_prediction_queries=True,original_remaining_gates=context.plan['remaining_gates'],
        self_exit_claimed=False,root_must_observe_actual_exit=True,full_model_qualified=False,new_speedup_claim=False)
    shared.write_json(output_dir/'result.json',result);return result


def run(context, output_dir):
    context.guard('cpu'); require(context.permit['arm']=='C_Hc_history_generation'and type(context.permit['rank'])is int and context.permit['rank']in(0,1),'Original rank permit')
    output_dir=Path(output_dir);require(output_dir==Path(context.plan['output_root'])/f'cpu-rank{context.permit["rank"]}','Dedicated rank output')
    context.verify_sources();validate_plan(context.plan,context.read_json)
    require(str(Path(__file__).resolve())in context.plan['source_sha256'],'Worker authenticated in closure')
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
            base.cleanup_actions(([parallel_state.destroy_model_parallel] if initialized else [])+[torch.distributed.destroy_process_group], 'distributed cleanup')
