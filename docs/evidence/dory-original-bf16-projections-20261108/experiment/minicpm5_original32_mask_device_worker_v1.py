"""Finite original32 compiled-device trajectories; root owns independent admission.

Local cache/graph readers copy the frozen algorithms without importing the old
history protocol or its global admission. No native import occurs at module load.
"""
import ast
from contextlib import contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import zlib
import minicpm5_call4_mask_worker_v1 as shared

ROOT = shared.ROOT
MIB = 2**20
CACHE_BYTES = 4*1*4224*128*2
ARMS = ('original_inclusive_control_trajectory', 'strict_prior_same_width_trajectory')
ORIGINAL_PLAN_SHA = 'b8d84596f9fa6b54c0fa599af6ad976f051e47544a2d8c76027136efe2254800'
COMPILE_SHA = '9351bdf9c062435f9249c3b2081c34d7cb8c351f59a4709eb7a5eeae314413fd'
ENV_ABSENT = ('NXD_CPU_MODE', 'TORCHELASTIC_RUN_ID', 'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE',
              'NEURON_PJRT_PROCESS_INDEX', 'PJRT_LOCAL_PROCESS_RANK', 'NEURON_PJRT_WORLD_SIZE', 'PJRT_LOCAL_PROCESS_COUNT',
              'NEURON_PT_MPMD_MODE', 'NEURON_PJRT_PROCESSES_NUM_DEVICES', 'NEURONCORE_NUM_DEVICES', 'XLA_USE_SPMD', 'XLA_AUTO_SPMD',
              'NXD_INFERENCE_CAPTURE_SNAPSHOT', 'TRN1_MINICPM5_NKI_CACHE', 'TRN1_SMOLLM3_MIN_DECODE_BUCKET', 'TRN1_CPU_SAMPLER_FAST')
require, same = shared.require, shared.same


def environment():
    require(os.environ.get('NEURON_RT_VISIBLE_CORES') == '0-1', 'Fixed device core pair')
    require(all(os.environ.get(key) is None for key in ENV_ABSENT), 'Single device process without CPU/rank/intervention environment')
    require(os.environ.get('PJRT_DEVICE') in (None, 'NEURON'), 'Neuron PJRT only')
    return {key:os.environ.get(key) for key in (*ENV_ABSENT, 'NEURON_RT_VISIBLE_CORES', 'PJRT_DEVICE')}


def authenticate_plan(context):
    plan = context.plan; ref = plan['original_plan']
    same(ref, dict(path=str(ROOT/'research/minicpm5-layer0-runtime-inputs-20261107-v1.json'), sha256=ORIGINAL_PLAN_SHA), 'Pinned original plan')
    original = context.read_json(ref['path'], ref['sha256'])
    for key in ('calls', 'frames', 'fixtures', 'graph_calls', 'hf_reference', 'hf_frame_records'):
        same(plan[key], original[key], 'Original32 unchanged '+key)
    require(len(plan['calls']) == 14 and len(plan['frames']) == 32 and len(plan['fixtures']) == 8, 'Complete original14/32/8')
    for path, binding in original['inputs'].items(): same(plan['inputs'].get(path), binding, 'Original input binding '+path)
    for path, sha in original['source_sha256'].items(): same(plan['source_sha256'].get(path), sha, 'Original source pin '+path)
    require([f['step'] for f in plan['frames']].count(0) == 8, 'All8 prefill frames retained')
    artifact = plan['device_artifact']; receipt = artifact['compile_receipt']
    same(receipt, dict(path=str(ROOT/'minicpm5-baseline-full-v5/compile.json'), sha256=COMPILE_SHA), 'Frozen compile receipt')
    compiled = context.read_json(receipt['path'], receipt['sha256'])
    require(compiled['complete'] is compiled['passed'] is True, 'Completed existing artifact')
    same(artifact['path'], compiled['case']['artifact'], 'Exact compiled artifact directory')
    require(compiled['compile_phase_only'] is compiled['cpu_sampler'] is compiled['original_sdk_math'] is True, 'Original graph math and host sampler')
    for path, binding in compiled['payloads'].items():
        same(plan['inputs'].get(path), binding, 'Compiled payload whitelist')
        context.verify_input(path)
    config = str(Path(artifact['path'])/'neuron_config.json')
    same(context.verify_input(config)['sha256'], compiled['config_sha256'], 'Compiled configuration pin')
    return compiled


def tensor_bytes(torch, value, cap=8*MIB):
    require(value.numel()*value.element_size() <= cap, 'Bounded one tensor read')
    return memoryview(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy())


def digest(torch, value):
    return hashlib.sha256(tensor_bytes(torch, value)).hexdigest()


def metadata(torch, value):
    return dict(shape=list(value.shape), stride=list(value.stride()), offset=value.storage_offset(),
                dtype=str(value.dtype), sha256=digest(torch, value))


def manifest(context, ref):
    value = context.read_json(ref['path'], ref['sha256']); rows = value['rows']
    require(value['complete'] is True and len(rows) == 168, 'Complete168 cache manifest')
    same([(row['rank'],row['name']) for row in rows], [(rank,f'kv_mgr.past_key_values.{index}') for rank in range(2) for index in range(84)], 'Exact cache rank/order')
    for row in rows:
        same([row['shape'],row['dtype'],row['raw_bytes'],row['nonfinite_elements']], [[4,1,4224,128],'torch.bfloat16',CACHE_BYTES,0], 'Original finite cache layout')
        same(context.plan['inputs'][row['file']], dict(identity=row['identity'],sha256=row['payload_sha256']), 'Cache payload binding')
    return rows


def load_cache(context, torch, row):
    data = context.read_bytes(row['file'], CACHE_BYTES+2048)
    if row.get('encoding') == 'raw_bf16_le': require(len(data) == CACHE_BYTES, 'Raw cache length')
    else:
        require('encoding' not in row and Path(row['file']).suffix == '.zlib', 'Original cache encoding')
        decoder = zlib.decompressobj(); data = decoder.decompress(data, CACHE_BYTES+1)
        require(decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail, 'Cache zlib framing')
    require(len(data) == CACHE_BYTES and hashlib.sha256(data).hexdigest() == row['raw_sha256'], 'Original cache bytes')
    value = torch.frombuffer(bytearray(data), dtype=torch.uint8).view(torch.bfloat16).reshape(row['shape'])
    require(bool(torch.isfinite(value).all()), 'Finite imported seed'); return value


def states(model, torch):
    state = model.traced_model.nxd_model.state
    require(len(state) == 2, 'Two device ranks')
    for rank in range(2):
        same(sorted(state[rank]), sorted(f'kv_mgr.past_key_values.{i}' for i in range(84)), 'All84 rank state keys')
        for index in range(84):
            name = f'kv_mgr.past_key_values.{index}'; value = state[rank][name]
            require(list(value.shape) == [4,1,4224,128] and value.dtype == torch.bfloat16, 'Original physical cache layout')
            require(str(value.device) == f'privateuseone:{rank}', 'Original rank device')
            yield rank, name, value


def bindings(model, torch):
    rows = [dict(rank=rank, name=name, tensor_impl=int(value._cdata)) for rank,name,value in states(model,torch)]
    require(len({row['tensor_impl'] for row in rows}) == 168, 'Unique168 device state bindings'); return rows


def cache_hashes(context, model, torch, expected_bindings):
    context.guard('device'); same(bindings(model,torch), expected_bindings, 'Device cache bindings unchanged')
    hashes = []
    for rank,name,value in states(model,torch):
        context.guard('device'); cpu = value.detach().cpu()
        require(bool(torch.isfinite(cpu).all()), 'Finite device cache '+name)
        hashes.append(dict(rank=rank,name=name,sha256=digest(torch,cpu))); del cpu
    return hashes


def expected_hashes(rows):
    return [dict(rank=row['rank'],name=row['name'],sha256=row['raw_sha256']) for row in rows]


def import_seed(context, model, torch, seed, expected_bindings):
    for (rank,name,value),row in zip(states(model,torch),seed):
        context.guard('device'); same([rank,name],[row['rank'],row['name']], 'Seed rank binding')
        cpu = load_cache(context,torch,row); value.copy_(cpu); del cpu
        require(digest(torch,value) == row['raw_sha256'], 'Seed device roundtrip')
    actual = cache_hashes(context,model,torch,expected_bindings)
    same(actual,expected_hashes(seed),'All168 original seed imported'); return actual


def graph_inputs(context, torch, graph):
    inputs = context.load_pt(graph['payload'],32*MIB)['inputs']; require(len(inputs) == 7, 'Seven original graph arguments')
    same([metadata(torch,x) for x in inputs],graph['input_metadata'],'Original graph bytes/layout')
    require(len(graph['compressed_inputs']) == 7, 'Seven reversible input records')
    for value,row in zip(inputs,graph['compressed_inputs']):
        packed = context.read_bytes(row['path'],16*MIB); decoder = zlib.decompressobj(); data = decoder.decompress(packed,row['raw_bytes']+1)
        require(decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail,'Graph input zlib framing')
        require(len(data) == row['raw_bytes'] and hashlib.sha256(data).hexdigest() == row['raw_sha256'] == digest(torch,value),'Original reversible graph input')
    return inputs


def input_rows(plan, call, inputs, strict):
    require(type(strict) is bool and len(inputs) == 7, 'One explicit intervention flag')
    prefill = call['phase'] == 'prefill'; require(prefill or call['phase'] == 'decode','Original phase')
    seq = inputs[3].tolist(); same(seq,[call['frames'][0]['slot']] if prefill else [0,1,2,3],'Original physical slots')
    same(inputs[5].tolist(),[0]*len(seq),'Original hidden placeholders'); same(inputs[6].tolist(),[0]*len(seq),'Original adapter placeholders')
    mapped = []
    for frame,logical in zip(call['frames'],call['logical_positions']):
        fixture = plan['fixtures'][frame['fixture']]; length = len(fixture['input_ids']); row = seq.index(frame['slot'])
        tokens = inputs[0][row].tolist(); positions = inputs[2][row].tolist(); mask = inputs[1][row].tolist()
        require(logical == length+frame['step']-1,'Original logical current position')
        if prefill:
            require(frame['step'] == 0,'Prefill step0'); same(tokens[:length],fixture['input_ids'],'Original prefill tokens')
            same(positions[:length],list(range(length)),'Original prefill positions'); allowed = length
        else:
            require(frame['step'] in (1,2,3),'Original decode step')
            same(tokens,[fixture['continuation_ids'][frame['step']-1]],'Original teacher token'); same(positions,[logical],'Original decode position')
            allowed = logical if strict else logical+1
        same(mask,[1]*allowed+[0]*(len(mask)-allowed),'Explicit full-width mask predicate')
        mapped.append(dict(frame=frame,physical_row=row,current_position=logical))
    require(len(mapped) == (1 if prefill else 4),'Complete active frame mapping'); return mapped


def intervention(context, torch, graph, inputs, strict):
    original_meta = [metadata(torch,x) for x in inputs]; mapping = input_rows(context.plan,graph['call'],inputs,False)
    changed = list(inputs); cells = []
    if strict and graph['call']['phase'] == 'decode':
        changed[1] = inputs[1].clone()
        for row in mapping:
            i,j = row['physical_row'],row['current_position']; changed[1][i,j] = 0; cells.append(dict(row=i,column=j,before=1,after=0))
    actual_meta = [metadata(torch,x) for x in changed]
    for index,(original,actual) in enumerate(zip(original_meta,actual_meta)):
        if index == 1 and cells:
            same({k:v for k,v in actual.items() if k!='sha256'}, {k:v for k,v in original.items() if k!='sha256'},'Same mask width/layout/dtype')
        else:
            require(changed[index] is inputs[index],'All unchanged argument objects'); same(original,actual,'Unchanged graph argument')
    input_rows(context.plan,graph['call'],changed,strict)
    return changed,cells,mapping,actual_meta


def scoring(output, hf, mapping, metric):
    require(list(output.shape) == [len(mapping),1,130560], 'Exact full-vocabulary output shape')
    rows = []
    for item in mapping:
        frame = item['frame']; values = metric(output[item['physical_row'],0],hf[frame['fixture']]['logits'][frame['step']])
        checks = dict(cosine=values['cosine'] >= .995, KL=values['kl_reference_to_neuron'] <= .02, max_abs=values['max_abs_error'] <= 2,
                      high_margin_argmax=values['reference_margin'] <= .5 or values['argmax_match'])
        rows.append(dict(**item,metrics=values,failed_criteria=[name for name,passed in checks.items() if not passed]))
    return rows


def references(context, torch):
    """Frozen call4 reference/metric algorithm with this device context guard."""
    ref=context.plan['hf_reference']; context.verify_input(ref['path']); original=context.load_pt(ref['path'],32*MIB)
    require(len(original)==8 and [x['id'] for x in original]==list(range(8)),'All original8 HF fixtures')
    for fixture,saved in zip(context.plan['fixtures'],original):
        for key in ('id','input_ids','continuation_ids'): same(saved[key],fixture[key],'Original HF '+key)
        require(len(saved['logits'])==4,'Four original HF steps')
        for value in saved['logits']:
            require(list(value.shape)==[130560] and value.dtype==torch.float32 and bool(torch.isfinite(value).all()),'Finite full-vocabulary HF')
        same(saved['continuation_ids'],[int(x.argmax()) for x in saved['logits'][:3]],'Unchanged HF teacher tokens')
    path=ROOT/'validate_real_logits.py'; context.guard('device'); source=path.read_bytes()
    require(len(source)<8*MIB and hashlib.sha256(source).hexdigest()==context.plan['source_sha256'][str(path)],'Frozen metric source')
    functions=[node for node in ast.parse(source).body if isinstance(node,ast.FunctionDef) and node.name=='metrics']
    require(len(functions)==1,'Exactly one original metrics function'); namespace=dict(torch=torch)
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),namespace)
    return original,namespace['metrics']


def summarize(plan, rows):
    same([row['frame'] for row in rows],plan['frames'],'All32 original frame order')
    require(len(rows) == 32,'All32 rows'); prefill = [r for r in rows if r['frame']['step']==0]; decode = [r for r in rows if r['frame']['step']>0]
    require(len(prefill)==8 and len(decode)==24,'All8 prefill and24 decode')
    fraction = sum(r['metrics']['argmax_match'] for r in rows)/32
    return dict(complete=True,frames=32,prefill_frames=8,decode_frames=24,passed=all(not r['failed_criteria'] for r in rows) and fraction>=.9,
                failed_frames=[r['frame']['index'] for r in rows if r['failed_criteria']],prefill_failed=sum(bool(r['failed_criteria']) for r in prefill),
                decode_failed=sum(bool(r['failed_criteria']) for r in decode),argmax_match_fraction=fraction,frame1=rows[1],full_model_qualified=False)


def save_output(torch, path, output):
    require(output.numel()*output.element_size() <= 4*130560*4,'Bounded full-vocabulary output')
    with path.open('xb') as stream: torch.save(dict(output=output.detach().cpu().clone().contiguous()),stream)
    require(path.stat().st_size <= 3*MIB,'Output payload cap')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(MIB),b''): h.update(block)
    return dict(path=str(path),sha256=h.hexdigest(),bytes=path.stat().st_size)


@contextmanager
def compilation_disabled(cls):
    had_local='compile' in cls.__dict__; original=cls.__dict__.get('compile')
    def forbidden(*args,**kwargs): raise RuntimeError('Existing compiled artifact only; compilation forbidden')
    cls.compile=forbidden
    try: yield
    finally:
        if had_local: cls.compile=original
        else: delattr(cls,'compile')


def artifact_identities(context, compiled):
    artifact=Path(context.plan['device_artifact']['path'])
    paths=[*compiled['payloads'],str(artifact/'neuron_config.json'),str(ROOT/'research/minicpm5-2b-readiness-20261104-v1/config.json')]
    return {path:context.verify_input(path) for path in paths}


def run_trajectory(context, torch, model, expected_bindings, seed, hf, metric, arm, output_dir):
    strict = arm == ARMS[1]; arm_dir = output_dir/arm; arm_dir.mkdir(exist_ok=False)
    previous = import_seed(context,model,torch,seed,expected_bindings); rows = []; reports = []; resets = []
    for index,graph in enumerate(context.plan['graph_calls']):
        context.guard('device'); call = graph['call']; same(call,context.plan['calls'][index],'Original call order')
        same(call['index'],index,'Original index')
        before = cache_hashes(context,model,torch,expected_bindings); same(before,previous,'Actual previous post is next before')
        if index in (0,7):
            model.reset(); require(model.kv_cache_populated is False,'Original bookkeeping reset')
            same(cache_hashes(context,model,torch,expected_bindings),before,'Reset preserves all168 cache values')
            resets.append(dict(before_call=index,all168_cache_hashes_unchanged=True))
        require(model.kv_cache_populated is (index not in (0,7)), 'Original logical cache populated flag')
        original = graph_inputs(context,torch,graph); inputs,cells,mapping,actual_meta = intervention(context,torch,graph,original,strict)
        wrapper = model.context_encoding_model if call['phase']=='prefill' else model.token_generation_model
        require(not wrapper.async_mode and not wrapper.pipeline_execution,'Synchronous original graph')
        route = list(model.traced_model.nxd_model.input_shape_map[str([list(x.shape) for x in inputs])])
        same(route,graph['graph'],'Original compiled graph route'); require(wrapper.tag==route[0] and inputs[1].shape[1]==graph['bucket'],'Same wrapper/bucket/mask width')
        output = wrapper._forward(*inputs); model.kv_cache_populated=True
        require(list(output.shape)==[len(mapping),1,130560] and output.dtype==torch.float32 and bool(torch.isfinite(output).all()),'Finite full-vocabulary output')
        same([metadata(torch,x) for x in inputs],actual_meta,'Inputs unchanged during forward'); post = cache_hashes(context,model,torch,expected_bindings)
        output_meta = metadata(torch,output)
        if not strict:
            same(before,expected_hashes(manifest(context,graph['before'])),'Control original before endpoint')
            same(output_meta,graph['output_metadata'],'Control original full output endpoint')
            same(post,expected_hashes(manifest(context,graph['after'])),'Control original168 post endpoint')
        scored = scoring(output,hf,mapping,metric); rows.extend(scored)
        folder = arm_dir/f'call-{index:02d}'; folder.mkdir(exist_ok=False)
        report = dict(schema='minicpm5_original32_mask_device_call_v1',arm=arm,call=call,graph=route,bucket=graph['bucket'],
                      input_metadata=actual_meta,changed_mask_cells=cells,physical_rows=mapping,output_metadata=output_meta,
                      before_hashes=before,post_hashes=post,history_origin='original_call0_seed' if index==0 else 'previous_actual_post_same_arm',
                      seed_imported_this_call=index==0,reset_preserved_cache=index in (0,7),all168_bindings_unchanged=True,
                      control_original_endpoint_exact=not strict,strict_original_output_post_equality_required=False,HF_metrics=scored,
                      payload=save_output(torch,folder/'output.pt',output),full_cache_payloads_written=0,full_model_qualified=False)
        shared.write_json(folder/'result.json',report); reports.append(dict(call=index,result=str(folder/'result.json'),payload=report['payload']))
        previous=post; del inputs,original,output
    summary=summarize(context.plan,rows)
    result=dict(schema='minicpm5_original32_mask_device_arm_v1',arm=arm,summary=summary,rows=rows,reports=reports,resets=resets,
                seed_import_count=1,actual_continuous_cache_chain=True,control_original14_endpoints_exact=not strict,full_model_qualified=False)
    shared.write_json(arm_dir/'result.json',result); return result


def run_pair(context, torch, model, expected_bindings, seed, hf, metric, output_dir):
    results=[]
    for arm in ARMS:
        if arm==ARMS[1]:
            require(len(results)==1 and results[0]['control_original14_endpoints_exact'] is True and results[0]['summary']['complete'] is True,
                    'Entire14-call control gate before strict')
            require(len(results[0]['reports'])==14 and len(results[0]['rows'])==32 and results[0]['summary']['frames']==32,
                    'Complete control trajectory before strict')
        results.append(run_trajectory(context,torch,model,expected_bindings,seed,hf,metric,arm,output_dir))
    return results


def run(context, output_dir):
    context.guard('device'); require(context.permit['role']=='device','Independent device permit')
    output_dir=Path(output_dir)
    output_root=Path(context.plan['output_root'])
    require(output_dir==output_root/'device' and output_root.parent==ROOT and output_root.name.startswith('minicpm5-original32-strict-prior-device-'),'Fresh dedicated device output tree')
    require(str(Path(__file__).resolve()) in context.plan['source_sha256'],'Worker in authenticated source closure')
    require(sys.dont_write_bytecode and __debug__ and 'torch' not in sys.modules and 'torch_neuronx' not in sys.modules,'Fresh -B process before native import')
    context.verify_sources(); compiled=authenticate_plan(context); before_env=environment(); output_dir.mkdir(exist_ok=False)
    model=None; results=[]
    try:
        import torch
        import minicpm5_baseline_neuron_v5 as baseline
        cls=baseline._official_class(); witness=context.record_device_import(); torch.set_num_threads(1); after_env=environment()
        initializer=sys.modules['torch_neuronx.initialization']._initializer
        require(initializer is not None and initializer.configured_pjrt_env is False and initializer.previous_pjrt_env_vars=={},'Original single-process SDK initialization')
        shared.write_json(output_dir/'sdk-environment.json',dict(before=before_env,after=after_env,witness=witness,configured_pjrt_env=False,diagnostic_environment_mutation=False))
        context.guard('device'); artifact=context.plan['device_artifact']['path']; same(compiled['case'],baseline.case(),'Original complete case')
        identities=artifact_identities(context,compiled)
        cfg=cls.get_config_cls().load(artifact); baseline._check_config(cfg)
        require(not cfg.neuron_config.skip_warmup and not cfg.neuron_config.async_mode,'Original warmup/synchronous config')
        with compilation_disabled(cls):
            model=cls(compiled['case']['model_path'],config=cfg); model.load(artifact)
            same(artifact_identities(context,compiled),identities,'Compiled/config/readiness inputs unchanged after load')
            require(not model.async_mode and model.is_loaded_to_neuron,'Official compiled load')
            for tag in ('context_encoding_model','token_generation_model'): same(list(getattr(model,tag).neuron_config.buckets),[128,256,512,1024,2048,4096],'Original compiled buckets')
            load_proof=dict(official_load_completed=True,default_official_warmup_enabled=True,skip_warmup_override=False,
                            new_compilation=False,warmup_in_replay_call_count=False,warmup_call_count_not_measured=True,
                            artifacts=identities,compiled_receipt=context.plan['device_artifact']['compile_receipt'])
            shared.write_json(output_dir/'official-load.json',load_proof)
            context.guard('device'); environment(); expected_bindings=bindings(model,torch)
            seed=manifest(context,context.plan['graph_calls'][0]['before']); hf,metric=references(context,torch)
            with torch.inference_mode(): results=run_pair(context,torch,model,expected_bindings,seed,hf,metric,output_dir)
        context.guard('device'); environment()
        result=dict(schema='minicpm5_original32_mask_device_result_v1',complete=True,arms=[dict(arm=x['arm'],summary=x['summary'],result=str(output_dir/x['arm']/'result.json')) for x in results],
                    replay_calls=28,replay_scored_frames=64,official_load=load_proof,original_prefill_retained=True,full_cache_payloads_written=0,compiled_new_model=False,
                    same_width_explicit_mask=True,serving_inferred_mask_tested=False,full_model_qualified=False,new_speedup_claim=False,
                    self_exit_claimed=False,root_must_observe_actual_exit=True)
        shared.write_json(output_dir/'result.json',result); return result
    except Exception as error:
        shared.write_json(output_dir/'failure.json',dict(schema='minicpm5_original32_mask_device_failure_v1',error_type=type(error).__name__,error=str(error),
                          completed_arms=[x['arm'] for x in results],full_model_qualified=False,self_exit_claimed=False))
        raise
