"""Original MiniCPM5 BF16 Neuron qualification; root owns admission and servers.

Draft library only: no CLI, queue, native import or payload read at import time.
All public execution entries call the supplied root guard before native imports.
"""
import ast
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from types import SimpleNamespace
from decimal import Decimal, InvalidOperation

import minicpm5_baseline_native_v4 as native
from minicpm5_forward_options_runtime_v5 import call_with_options

ROOT = native.ROOT
PHASES = ('numeric', 'cold', 'long', 'quality', 'workloads')
SDK = Path('/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages')
VLLM_NEURON = Path('/home/ubuntu/src/vllm-neuron/vllm_neuron')
LLAMA = SDK / 'neuronx_distributed_inference/models/llama/modeling_llama.py'
BUCKETS = native.BUCKETS
CRITERIA = native.CRITERIA
VOCAB = native.VOCAB
SCHEMA = 'minicpm5-baseline-v5'


def case():
    return dict(name='latest-minicpm5-2b-bf16-cpu-tp2-b4-ctx4096-baseline-v4',
        model_path=str(native.MODEL_PATH), served_name='minicpm5-2b',
        artifact='/mnt/nvme/neuron-artifacts/latest-20261031-minicpm5-2b-bf16-cpu-tp2-b4-ctx4096-baseline-v4',
        batch=4, tp=2, max_model_len=4096, cpu=True, adapter='llama_native', compile_workers=1,
        torchinductor_compile_threads=1, skip_sampling_probe=True,
        extra_overrides=dict(apply_seq_ids_mask=True, async_mode=False, cast_type='config',
            fused_qkv=False, vocab_parallel=False, mlp_kernel_enabled=False, qkv_kernel_enabled=False,
            fused_rmsnorm_skip_gamma=False, is_block_kv_layout=False, is_prefix_caching=False),
        host_limits=dict(memory_max_gib=26, memory_high_gib=24, swap_gib=8),
        startup_wait_limit_s=6900, startup_runtime_limit_s=7200)


def source_paths(require_meta_plan=True):
    from prepare_minicpm5_meta_v3 import source_paths as meta_sources
    paths = set(native.source_paths()) | set(meta_sources()) | {Path(__file__)}
    paths |= {ROOT / name for name in ('qwen3_4b_cached_json_v2_protocol.py', 'runtime_snapshot.py',
        'check_minicpm5_baseline_neuron_v4.py')}
    paths |= {VLLM_NEURON / name for name in ('worker/neuronx_distributed_model_loader.py',
        'worker/neuronx_distributed_model_runner.py', 'worker/neuron_worker.py',
        'platform.py', 'platform_overrides.py')}
    paths |= {SDK / name for name in ('neuronx_distributed/trace/model_builder.py',
        'neuronx_distributed_inference/utils/hf_adapter.py',
        'neuronx_distributed_inference/modules/attention/utils.py',
        'vllm/entrypoints/openai/api_server.py', 'vllm/model_executor/layers/logits_processor.py')}
    paths |= set((SDK/'vllm/v1/sample').glob('*.py'))
    paths |= {ROOT/'minicpm5_forward_options_runtime_v5.py',
              SDK/'neuronx_distributed_inference/models/model_base.py'}
    if require_meta_plan:
        paths.add(ROOT / 'minicpm5-meta-v3/plan.json')
    assert all(path.is_file() for path in paths), [str(x) for x in paths if not x.is_file()]
    return sorted(paths)


def meta_gate():
    """TP1 and both TP2 ranks must have passed under the frozen meta plan."""
    import minicpm5_meta_protocol_v3 as meta
    meta.verify()
    final = native.read(meta.DEST / 'result.json')
    assert final['passed'] and final['model'] == native.MODEL_ID
    assert final['checkpoint_revision'] == native.REVISION
    assert [r['phase'] for r in final['phases']] == list(meta.PHASES[1:])
    for row in final['phases']:
        path = meta.DEST / (row['phase'] + '-result.json')
        assert row['path'] == str(path) and row['sha256'] == native.text_sha(path)
        meta.check_result(row['phase'], native.read(path))
    assert final['sdk_rank_configurations'] == 3 and final['sdk_module_constructions'] == 6
    assert final['actual_sdk_sharder_invocations'] == 6 and final['learned_shape_checks'] == 2286
    assert final['cache_alias_bindings'] == 3024 and final['model_weight_payload_bytes_read'] == 0
    assert all(final[key] is False for key in ('numerical_execution', 'device_execution',
        'collectives_executed', 'full_model_qualified'))
    return dict(path=str(meta.DEST / 'result.json'), sha256=native.text_sha(meta.DEST / 'result.json'))


def _enter(context, phase):
    current = native._enter(context, phase)  # First action is root's ownership/resource guard.
    assert phase in PHASES and current['case'] == case()
    meta_gate()
    for path in source_paths():
        assert context['plan']['source_sha256'][str(path)] == native.text_sha(path), str(path)
    assert not any(os.environ.get(key) for key in ('TRN1_CPU_SAMPLER_FAST', 'TRN1_SMOLLM3_MIN_DECODE_BUCKET',
        'TRN1_MINICPM5_NKI_CACHE', 'NXD_CPU_MODE'))
    return current


def expected_bucket(length):
    # Installed ModelWrapper._get_seq_bucket uses strict first-fit, including equality boundaries.
    assert type(length) is int and 0 < length <= 4096
    return next((bucket for bucket in BUCKETS if length < bucket), BUCKETS[-1])


def passed(rows):
    from qwen3_4b_cached_json_v2_protocol import passed as original_passed
    return original_passed(rows)


def _payload_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def _protocol(context):
    protocol = native.read(context['DEST'] / 'input-protocol.json')
    assert protocol == native.build_protocol(protocol['fixtures'])
    tokenizer = native.read(context['DEST'] / 'tokenizers/runtime.json')
    original = native.read(context['DEST'] / 'tokenizers/native.json')
    ignored={'mode','full_vocabulary_compared'}
    assert {k:v for k,v in tokenizer.items() if k not in ignored} == {k:v for k,v in original.items() if k not in ignored}
    assert original['mode']=='native' and tokenizer['mode']=='runtime'
    assert original['full_vocabulary_compared'] is False
    assert tokenizer['passed'] and tokenizer['complete'] and tokenizer['full_vocabulary_compared']
    assert tokenizer['vocabulary_entries'] == VOCAB and tokenizer['enable_thinking'] is False
    assert tokenizer['protocol_sha256'] == native.json_digest(protocol)
    return protocol


def _reference(context, stem, frames):
    import torch
    protocol = _protocol(context)
    metadata = native.read(context['DEST'] / (stem + '.json'))
    assert metadata['complete'] and metadata['passed'] and metadata['all_finite']
    assert metadata['revision'] == native.REVISION and metadata['criteria'] == CRITERIA
    assert metadata['frames'] == frames and metadata['native_source_sha256'] == native.text_sha(native.NATIVE_LLAMA)
    path = ROOT / 'minicpm5-baseline-full-v4' / (stem + '.pt')
    assert metadata['tensor'] == str(path) and metadata['tensor_identity'] == native.identity(path)
    assert metadata['tensor_sha256'] == _payload_hash(path)
    if stem != 'hf-reference':
        assert metadata['protocol_sha256'] == native.json_digest(protocol)
    if stem == 'cached-reference':
        assert metadata['native_cached_steps'] == 8172 and metadata['native_forward_calls'] == 8177
        assert metadata['all_forward_calls_finite'] and metadata['device_operations'] == 2049
    rows = torch.load(path, weights_only=True, map_location='cpu')
    return rows, metadata


def _official_class():
    # The SDK's import ordering is also exercised by the admitted meta worker.
    import importlib
    importlib.import_module('neuronx_distributed_inference.modules.lora_serving')
    from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaForCausalLM
    from vllm_neuron.worker.neuronx_distributed_model_loader import _get_neuron_model_cls
    cls = NeuronLlamaForCausalLM
    assert _get_neuron_model_cls('LlamaForCausalLM') is cls
    assert Path(importlib.import_module(cls.__module__).__file__).resolve() == LLAMA.resolve()
    assert not any(hasattr(cls, key) for key in ('_minimum_decode_bucket_hook', '_minicpm5_nki_cache'))
    return cls


def _configuration(torch, cls):
    kwargs = dict(tp_degree=2, ctx_batch_size=1, batch_size=4, max_batch_size=4, kv_cache_batch_size=4,
        seq_len=4096, max_length=4096, max_context_length=4096, torch_dtype=torch.bfloat16,
        on_device_sampling_config=None, enable_bucketing=True, is_continuous_batching=True,
        padding_side='right', buckets=list(BUCKETS), quantized=False, save_sharded_checkpoint=True, **case()['extra_overrides'])
    nc = cls.get_neuron_config_cls()(**kwargs)
    cfg = cls.get_config_cls()(nc, **native.read(native.READINESS / 'config.json'))
    _check_config(cfg)
    return cfg


def _check_config(cfg):
    nc = cfg.neuron_config
    for key,value in dict(batch_size=4,tp_degree=2,seq_len=4096,max_length=4096,
        max_context_length=4096,ctx_batch_size=1,max_batch_size=4,kv_cache_batch_size=4,
        enable_bucketing=True,is_continuous_batching=True,padding_side='right',
        quantized=False,save_sharded_checkpoint=True,on_device_sampling_config=None,**case()['extra_overrides']).items():
        assert getattr(nc,key) == value, (key,getattr(nc,key),value)
    assert str(nc.torch_dtype) == 'torch.bfloat16'
    for key,value in native.read(native.READINESS / 'config.json').items():
        if key not in ('torch_dtype','transformers_version','architectures','model_type'):
            assert getattr(cfg,key) == value, (key,getattr(cfg,key),value)


def artifact_gate(context, hash_payloads=False):
    if hash_payloads:
        context['guard'](context['phase'])
    directory = context['DEST']; artifact = Path(context['case']['artifact'])
    receipt = native.read(directory / 'compile.json')
    assert receipt['complete'] and receipt['passed'] and receipt['case'] == case()
    assert receipt['revision'] == native.REVISION and receipt['model_class'] == 'NeuronLlamaForCausalLM'
    assert receipt['llama_source_sha256'] == native.text_sha(LLAMA)
    assert receipt['config_sha256'] == native.text_sha(artifact / 'neuron_config.json')
    assert receipt['compile_phase_only'] and receipt['cpu_sampler'] and receipt['original_sdk_math']
    config = native.read(artifact / 'neuron_config.json')
    nc = config['neuron_config']
    for key,value in dict(batch_size=4,tp_degree=2,seq_len=4096,quantized=False,
        save_sharded_checkpoint=True,on_device_sampling_config=None,**case()['extra_overrides']).items():
        assert nc[key] == value, (key,nc.get(key),value)
    assert nc['torch_dtype'] in ('torch.bfloat16','bfloat16')
    required_payloads={str(artifact/name) for name in ('model.pt',
        'weights/tp0_sharded_checkpoint.safetensors','weights/tp1_sharded_checkpoint.safetensors')}
    assert set(receipt['payloads']) >= required_payloads
    for name,row in receipt['payloads'].items():
        assert Path(name).is_relative_to(artifact) and native.identity(name) == row['identity']
        if hash_payloads:
            assert row['sha256'] == _payload_hash(name), name
    return receipt





def _outputs(context, *names):
    return dict(outputs={str(context['DEST']/name):native.text_sha(context['DEST']/name) for name in names})


def _load(context):
    artifact_gate(context,hash_payloads=True)
    native._verify_download(context)
    import torch
    from vllm_neuron.worker.neuronx_distributed_model_loader import NeuronModelBase
    cls = _official_class(); torch.set_num_threads(1)
    cfg = cls.get_config_cls().load(context['case']['artifact']); _check_config(cfg)
    model = cls(context['case']['model_path'],config=cfg)
    model.load(context['case']['artifact']); model.reset()
    assert tuple(model.context_encoding_model.neuron_config.buckets) == BUCKETS
    assert tuple(model.token_generation_model.neuron_config.buckets) == BUCKETS
    order = SimpleNamespace(is_reorder_needed=True,_sort_inputs=NeuronModelBase._sort_inputs)
    routes = []; expected = {}
    for tag in ('context_encoding_model','token_generation_model'):
        wrapper = getattr(model,tag); original = wrapper._forward
        def observe(*inputs,_original=original,_tag=tag):
            selected = model.traced_model.nxd_model.input_shape_map[str([list(value.shape) for value in inputs])]
            row = dict(tag=selected[0],index=selected[1],input_tokens=int(inputs[0].shape[1]),mask_length=int(inputs[1].shape[1]))
            assert row['tag'] == _tag == expected['tag']
            assert row['mask_length'] == expected['bucket'], (row,expected)
            if _tag == 'context_encoding_model': assert row['input_tokens'] == expected['bucket']
            routes.append(row)
            return _original(*inputs)
        wrapper._forward = observe
    def forward(ids,mask,positions,slots):
        prefill = ids.shape[1] > 1 or bool((positions == 0).all())
        expected.update(tag='context_encoding_model' if prefill else 'token_generation_model',
            bucket=expected_bucket(int(mask.shape[1])))
        before = len(routes)
        with NeuronModelBase._reordered(order,slots,input_ids=ids,attention_mask=mask,
                position_ids=positions,sampling_params=model.default_sampling_params[:ids.shape[0]]) as (seq,inputs,restore):
            values = restore(call_with_options(model.forward,seq_ids=seq,**inputs).logits)
        assert len(routes) == before+1
        assert values.shape[-1] == VOCAB, 'Full vocabulary only; no metric-side slicing'
        return values
    metrics = native._function(ROOT/'validate_real_logits.py','metrics',dict(torch=torch))
    return torch,model,forward,routes,metrics


def _capture_report(context, stem, report, captures):
    import torch
    tensor = context['DEST'] / (stem+'.pt')
    assert not tensor.exists()
    with tensor.open('xb') as stream: torch.save(captures,stream)
    report.update(revision=native.REVISION,case=case(),criteria=CRITERIA,source_sha256=native.text_sha(__file__),
        tensor=str(tensor),tensor_sha256=_payload_hash(tensor),tensor_identity=native.identity(tensor),
        original_sdk_math=True,new_speedup_claim=False)
    native.once(context['DEST']/(stem+'.json'),report)


def _numeric(context):
    torch,model,forward,routes,metrics = _load(context)
    evidence,_ = _reference(context,'hf-reference',32)
    assert len(evidence) == 8 and [r['id'] for r in evidence] == list(range(8))
    canonical = native.read(context['DEST']/'tokenizers/runtime.json')['canonical']
    for row,fixture in zip(evidence,canonical):
        assert all(row[key] == value for key,value in fixture.items())
        assert row['logits'].shape == (4,VOCAB) and bool(torch.isfinite(row['logits']).all())
        assert len(row['continuation_ids']) == 3
        assert row['continuation_ids'] == [int(x.argmax()) for x in row['logits'][:3]]
    rows=[]; captures=[]; report=dict(complete=False,passed=False,rows=rows,routes=routes)
    try:
        with torch.inference_mode():
            order=[0,7,2,5,4,1,6,3]
            for start in (0,4):
                group=[evidence[i] for i in order[start:start+4]]; slots=[3,1,0,2];model.reset()
                for item,slot in zip(group,slots):
                    ids=torch.tensor([item['input_ids']]);positions=torch.arange(ids.shape[1])[None]
                    value=forward(ids,torch.ones_like(ids),positions,torch.tensor([slot]))[0,-1].float().cpu().clone()
                    meta=dict(fixture=item['id'],phase='prefill',slot=slot,step=0)
                    captures.append(dict(**meta,logits=value));rows.append(dict(**meta,**metrics(value,item['logits'][0])))
                for step in range(3):
                    ids=torch.tensor([[item['continuation_ids'][step]] for item in group])
                    positions=torch.tensor([[len(item['input_ids'])+step] for item in group])
                    mask=(torch.arange(int(positions.max())+1)[None]<=positions).long()
                    values=forward(ids,mask,positions,torch.tensor(slots))
                    for index,item in enumerate(group):
                        value=values[index,-1].float().cpu().clone()
                        meta=dict(fixture=item['id'],phase='mixed_length_decode',slot=slots[index],step=step+1)
                        captures.append(dict(**meta,logits=value));rows.append(dict(**meta,**metrics(value,item['logits'][step+1])))
        assert len(rows)==32 and len(routes)==14
        report.update(complete=True,passed=passed(rows),frames=32,unique_prompts=8,
            serving_reorder_original=True,scope='Original eight prompts x prefill and three cached teacher-forced steps; all slots and full-vocabulary strict metrics.')
    except BaseException as error:
        report['error']=repr(error);raise
    finally:
        _capture_report(context,'real-logits-validation',report,captures)
    assert report['passed'],'Original 32-frame numerical gate failed'
    numeric_gate(context,'numeric')
    return _outputs(context,'real-logits-validation.json')


def _cold(context):
    torch,model,forward,routes,metrics = _load(context)
    source,_ = _reference(context,'cold-reference',42); protocol=_protocol(context)
    assert [(r['prefix'],r['fixture']) for r in source] == [(n,f) for n in native.PREFIXES for f in range(2)]
    refs={(r['prefix'],r['fixture']):r for r in source}
    for row in source:
        assert row['input_ids']==protocol['fixtures'][row['fixture']]['input_ids']
        assert row['positions']==[row['prefix']-1,row['prefix'],row['prefix']+1]
        assert row['logits'].shape==(3,VOCAB) and bool(torch.isfinite(row['logits']).all())
    captures=[]; report=dict(complete=False,passed=False,experiments=[],routes=routes)
    def state(zero):
        checks=[];assert len(model.traced_model.nxd_model.state)==2
        for rank,tensors in enumerate(model.traced_model.nxd_model.state):
            assert len(tensors)==84
            for name,tensor in sorted(tensors.items()):
                value=tensor.detach().cpu()
                assert list(value.shape)==[4,1,4224,128] and value.dtype==torch.bfloat16
                row=dict(rank=rank,name=name,shape=list(value.shape),nan_elements=int(torch.isnan(value).sum()),inf_elements=int(torch.isinf(value).sum()))
                if zero: row['all_raw_bytes_zero']=int(torch.count_nonzero(value.contiguous().view(torch.uint8)))==0
                checks.append(row)
        return checks
    try:
        with torch.inference_mode():
            for prefix in native.PREFIXES:
                model.traced_model.nxd_model.initialize_with_saved_weights(torch.tensor([0],dtype=torch.int32));model.reset()
                item=dict(prefix=prefix,rows=[],duplicate_slot_checks=[],initial_zero_state_checks=state(True))
                report['experiments'].append(item)
                assert all(x['all_raw_bytes_zero'] for x in item['initial_zero_state_checks'])
                initial=[]
                for slot in range(4):
                    ids=torch.tensor([refs[prefix,slot%2]['input_ids'][:prefix]])
                    values=forward(ids,torch.ones_like(ids),torch.arange(prefix)[None],torch.tensor([slot]))
                    initial.append((values[0,-1].float().cpu().clone(),dict(routes[-1])))
                for step in range(3):
                    if step:
                        position=prefix+step-1
                        ids=torch.tensor([[refs[prefix,slot%2]['input_ids'][position]] for slot in range(4)])
                        positions=torch.full((4,1),position,dtype=torch.long)
                        mask=(torch.arange(position+1)[None]<=positions).long()
                        values=forward(ids,mask,positions,torch.arange(4))
                        current=[(values[slot,-1].float().cpu().clone(),dict(routes[-1])) for slot in range(4)]
                    else: current=initial
                    for slot,(value,route) in enumerate(current):
                        valid=value.shape==(VOCAB,) and bool(torch.isfinite(value).all())
                        meta=dict(prefix=prefix,phase='prefill' if step==0 else 'decode',step=step,slot=slot,fixture=slot%2)
                        captures.append(dict(**meta,logits=value))
                        item['rows'].append(dict(**meta,valid=valid,route=route,metrics=metrics(value,refs[prefix,slot%2]['logits'][step]) if valid else None))
                    for a,b in ((0,2),(1,3)):
                        item['duplicate_slot_checks'].append(dict(step=step,slots=[a,b],bitwise_equal=torch.equal(current[a][0].view(torch.uint8),current[b][0].view(torch.uint8))))
                item['state_after_two_decode_steps']=state(False)
                native.progress(context['DEST']/'cold-progress.json',dict(completed_prefixes=len(report['experiments']),frames=len(captures)))
        rows=[row for item in report['experiments'] for row in item['rows']]
        assert len(rows)==len(captures)==84 and len(routes)==42
        context_buckets=sorted({r['route']['input_tokens'] for r in rows if r['phase']=='prefill'})
        decode_buckets=sorted({r['route']['mask_length'] for r in rows if r['phase']=='decode'})
        assert context_buckets==decode_buckets==list(BUCKETS)
        numerical=all(row['valid'] for row in rows) and passed([row['metrics'] for row in rows])
        duplicates=all(row['bitwise_equal'] for item in report['experiments'] for row in item['duplicate_slot_checks'])
        zero=all(row['all_raw_bytes_zero'] for item in report['experiments'] for row in item['initial_zero_state_checks'])
        finite=all(row['nan_elements']==row['inf_elements']==0 for item in report['experiments'] for row in item['state_after_two_decode_steps'])
        report.update(complete=True,passed=numerical and duplicates and zero and finite,frames=84,
            numerical_gate_passed=numerical,all_duplicate_slots_bitwise_equal=duplicates,
            all_initial_states_zero=zero,all_cache_states_finite=finite,
            observed_context_buckets=context_buckets,observed_decode_buckets=decode_buckets,
            physical_cache_shape=[4,1,4224,128],scope='Seven fresh device cache initializations; 84 full-vocabulary frames, 42 duplicate-slot checks and both ranks all84 KV states. Original thresholds and natural SDK routes.')
    except BaseException as error:
        report['error']=repr(error);raise
    finally:
        _capture_report(context,'cold-buckets',report,captures)
    assert report['passed'],'Strict cold numerical/cache-state gate failed'
    numeric_gate(context,'cold')
    return _outputs(context,'cold-buckets.json')


def _long(context):
    torch,model,forward,routes,metrics = _load(context)
    source,_=_reference(context,'cached-reference',93);protocol=_protocol(context)
    assert [{k:row[k] for k in protocol['checks'][0]} for row in source]==protocol['checks']
    refs={(r['trajectory'],r['position']):r['logits'] for r in source}
    assert len(refs)==93 and all(value.shape==(VOCAB,) and bool(torch.isfinite(value).all()) for value in refs.values())
    rows=[];captures=[];report=dict(complete=False,passed=False,rows=rows,routes=routes,all_forward_calls_finite=True)
    try:
        with torch.inference_mode():
            for index,op in enumerate(protocol['operations']):
                requests=op['requests'];prefill=op['phase'] in ('prefill','slot_reuse_prefill')
                if prefill:
                    req=requests[0];length=req['position']+1
                    ids=torch.tensor([protocol['fixtures'][req['fixture']]['input_ids'][:length]])
                    positions=torch.arange(length)[None];mask=torch.ones_like(ids)
                else:
                    ids=torch.tensor([[protocol['fixtures'][r['fixture']]['input_ids'][r['position']]] for r in requests])
                    positions=torch.tensor([[r['position']] for r in requests])
                    mask=(torch.arange(int(positions.max())+1)[None]<=positions).long()
                values=forward(ids,mask,positions,torch.tensor([r['slot'] for r in requests]))
                report['all_forward_calls_finite'] &= bool(torch.isfinite(values).all())
                for i,req in enumerate(requests):
                    if not req['record']: continue
                    value=values[i,-1].float().cpu().clone();valid=value.shape==(VOCAB,) and bool(torch.isfinite(value).all())
                    meta={key:req[key] for key in ('trajectory','fixture','position','slot')}|dict(phase=op['phase'])
                    rows.append(dict(**meta,finite=valid,metrics=metrics(value,refs[req['trajectory'],req['position']]) if valid else None))
                    captures.append(dict(**meta,logits=value))
                if index%128==0:
                    native.progress(context['DEST']/'long-progress.json',dict(completed_operations=index+1,frames=len(rows)))
        assert [{k:r[k] for k in protocol['checks'][0]} for r in rows]==protocol['checks']
        assert len(routes)==2049 and sum(r['tag']=='token_generation_model' for r in routes)==2044
        report.update(complete=True,passed=report['all_forward_calls_finite'] and all(r['finite'] for r in rows) and passed([r['metrics'] for r in rows]),
            frames=93,device_operations=2049,observed_decode_calls=2044,teacher_forced_decode_steps=2041,
            protocol_sha256=native.json_digest(protocol),operations_sha256=native.json_digest(protocol['operations']),
            scope='All2049 original calls and 93 marks against8172 native cached updates, including slot reuse, partial batch and inactive resume; natural original Llama routes.')
    except BaseException as error:
        report['error']=repr(error);raise
    finally:
        _capture_report(context,'long-decode-validation',report,captures)
    assert report['passed'],'Strict long/cache-slot gate failed'
    numeric_gate(context,'long')
    return _outputs(context,'long-decode-validation.json')


def numeric_gate(context, kind):
    name={'numeric':'real-logits-validation','cold':'cold-buckets','long':'long-decode-validation'}[kind]
    report=native.read(Path(context['DEST'])/(name+'.json'))
    assert report['complete'] and report['passed'] and report['criteria']==CRITERIA and report['case']==case()
    assert report['revision']==native.REVISION and report['source_sha256']==native.text_sha(__file__)
    assert native.identity(report['tensor'])==report['tensor_identity']
    if kind=='numeric':
        assert len(report['rows'])==report['frames']==32 and passed(report['rows'])
        assert len(report['routes'])==14 and report['unique_prompts']==8 and report['serving_reorder_original']
        expected=[]
        for group in ((0,7,2,5),(4,1,6,3)):
            for step in range(4):
                expected.extend((fixture,'prefill' if step==0 else 'mixed_length_decode',slot,step) for fixture,slot in zip(group,(3,1,0,2)))
        assert [(row['fixture'],row['phase'],row['slot'],row['step']) for row in report['rows']]==expected
    elif kind=='cold':
        assert [item['prefix'] for item in report['experiments']]==list(native.PREFIXES)
        rows=[r for item in report['experiments'] for r in item['rows']]
        assert len(rows)==report['frames']==84 and all(r['valid'] for r in rows) and passed([r['metrics'] for r in rows])
        assert report['observed_context_buckets']==report['observed_decode_buckets']==list(BUCKETS)
        for item in report['experiments']:
            assert len(item['rows'])==12 and len(item['duplicate_slot_checks'])==6
            assert [(row['prefix'],row['phase'],row['step'],row['slot'],row['fixture']) for row in item['rows']]==[
                (item['prefix'],'prefill' if step==0 else 'decode',step,slot,slot%2) for step in range(3) for slot in range(4)]
            assert [(row['step'],row['slots']) for row in item['duplicate_slot_checks']]==[(step,pair) for step in range(3) for pair in ([0,2],[1,3])]
            assert all(row['bitwise_equal'] for row in item['duplicate_slot_checks'])
            for key in ('initial_zero_state_checks','state_after_two_decode_steps'):
                assert len(item[key])==168 and all(row['shape']==[4,1,4224,128] for row in item[key])
                assert all(row['nan_elements']==row['inf_elements']==0 for row in item[key])
            assert all(row['all_raw_bytes_zero'] for row in item['initial_zero_state_checks'])
        assert all(report[key] for key in ('numerical_gate_passed','all_duplicate_slots_bitwise_equal','all_initial_states_zero','all_cache_states_finite'))
    else:
        protocol=_protocol(context)
        assert [{key:r[key] for key in protocol['checks'][0]} for r in report['rows']]==protocol['checks']
        assert len(report['rows'])==report['frames']==93 and report['all_forward_calls_finite']
        assert all(r['finite'] for r in report['rows']) and passed([r['metrics'] for r in report['rows']])
        assert report['device_operations']==len(report['routes'])==2049
        assert report['teacher_forced_decode_steps']==2041 and report['observed_decode_calls']==2044
        assert report['protocol_sha256']==native.json_digest(protocol)
        for operation,route in zip(protocol['operations'],report['routes']):
            prefill=operation['phase'] in ('prefill','slot_reuse_prefill')
            assert route['tag']==('context_encoding_model' if prefill else 'token_generation_model')
            assert route['mask_length']==expected_bucket(max(row['position'] for row in operation['requests'])+1)
    return report


def _full_numeric(context):
    for kind in ('numeric','cold','long'): numeric_gate(context,kind)


def server(context):
    """Root calls in API parent and spawn children before native vLLM bootstrap.

    Context phase is quality/workloads. Its guard must validate the server unit
    and allow only its owned process descendants; this function never launches.
    """
    current=_enter(context,context['phase'])
    assert current['phase'] in ('quality','workloads')
    _full_numeric(current);artifact_gate(current)
    if current['phase']=='workloads': quality_gate(current)
    import torch
    cls=_official_class();torch.set_num_threads(1)
    assert os.environ['NEURON_COMPILED_ARTIFACTS']==current['case']['artifact']
    assert os.environ['NEURON_RT_VISIBLE_CORES']=='0-1'
    assert not getattr(cls,'_minicpm5_baseline_observed',False), 'Bootstrap once per server process'
    original_load=cls.load
    original_forward=cls.forward
    def forward(self,*args,**kwargs):
        return call_with_options(original_forward,self,*args,**kwargs)
    cls.forward=forward
    def refuse_compile(*args,**kwargs):
        raise RuntimeError('Original MiniCPM5 serving may only load its qualified compiled artifact')
    cls.compile=refuse_compile
    def load(self,path,*args,**kwargs):
        assert Path(path).resolve()==Path(current['case']['artifact']).resolve()
        artifact_gate(current);_check_config(self.config)
        result=original_load(self,path,*args,**kwargs)
        route_file=current['DEST']/('serving-routes-'+current['phase']+'-'+str(os.getpid())+'.jsonl')
        for tag in ('context_encoding_model','token_generation_model'):
            wrapper=getattr(self,tag);original=wrapper._forward;seen=set()
            assert tuple(wrapper.neuron_config.buckets)==BUCKETS
            def observe(*inputs,_original=original,_tag=tag,_seen=seen):
                shapes=[list(value.shape) for value in inputs]
                selected=self.traced_model.nxd_model.input_shape_map[str(shapes)]
                assert selected[0]==_tag and shapes[1][1] in BUCKETS
                key=json.dumps([selected,shapes])
                if key not in _seen:
                    with route_file.open('a') as stream:
                        stream.write(json.dumps(dict(phase=current['phase'],pid=os.getpid(),tag=selected[0],index=selected[1],
                            input_shapes=shapes,artifact=current['case']['artifact'],llama_source_sha256=native.text_sha(LLAMA)))+'\n')
                    _seen.add(key)
                return _original(*inputs)
            wrapper._forward=observe
        return result
    cls.load=load;cls._minicpm5_baseline_observed=True


def _serving_routes(context):
    files=sorted(context['DEST'].glob('serving-routes-'+context['phase']+'-*.jsonl'))
    rows=[json.loads(line) for path in files for line in path.read_text().splitlines()]
    assert rows and {'context_encoding_model','token_generation_model'} <= {row['tag'] for row in rows}
    assert all(row['phase']==context['phase'] and row['artifact']==case()['artifact'] and row['llama_source_sha256']==native.text_sha(LLAMA) for row in rows)
    output=context['DEST']/(context['phase']+'-routes.json')
    native.once(output,dict(passed=True,rows=rows,files=[str(path) for path in files],official_vllm_neuron=True))


def _serving(context):
    _full_numeric(context);artifact_gate(context)
    if context['phase']=='workloads': quality_gate(context)
    import orchestrate as runner
    from unittest.mock import patch
    tokenizer=native.runtime_tokenizer(context)
    pin=native.read(context['DEST']/'tokenizers/runtime.json')
    _protocol(context)
    assert runner.port_ready(8010), 'Root must start and own the matching server before HTTP measurement'
    with patch.object(runner,'prompt_tokenizer',return_value=tokenizer):
        if context['phase']=='quality':
            from evaluate_quality import prepare_inputs,evaluate
            suite=native.read(ROOT/'quality-suite.json')
            assert [dict(id=item['id'],input_ids=ids,max_tokens=item['max_tokens']) for item,ids in prepare_inputs(case(),suite)]==pin['quality']
            smoke=runner.quality(case());native.once(context['DEST']/'smoke.json',smoke)
            assert smoke['passed'], 'Original serving smoke gate failed'
            assert not (context['DEST']/'quality96.json').exists()
            evaluate(case(),context['DEST']/'quality96.json')
            quality_gate(context);names=('smoke.json','quality96.json','quality-routes.json')
        else:
            import bench_http
            from benchmark_workloads import inputs,evaluate,verify
            assert {str(key):value for key,value in inputs(case()).items()}==pin['workloads']
            assert not (context['DEST']/'initial-http.json').exists() and not (context['DEST']/'workloads.json').exists()
            initial_ids=runner.prompt_ids(case(),'请详细介绍一下法国的首都。')
            assert initial_ids==pin['initial_http_input_ids']
            initial=bench_http.benchmark(8010,case()['served_name'],[initial_ids],4)
            verify(initial);native.once(context['DEST']/'initial-http.json',initial)
            evaluate(case(),context['DEST']/'workloads.json')
            workloads_gate(context)
            from runtime_snapshot import capture
            capture(context['DEST']/'runtime-memory.json')
            names=('initial-http.json','workloads.json','runtime-memory.json','workloads-routes.json')
    _serving_routes(context)
    return _outputs(context,*names)


def quality_gate(context):
    directory=Path(context['DEST']);report=native.read(directory/'quality96.json')
    suite=native.read(ROOT/'quality-suite.json');pin=native.read(directory/'tokenizers/runtime.json')
    assert report['technical_complete'] and report['requested_cases']==report['total']==96
    assert report['suite_sha256']==native.text_sha(ROOT/'quality-suite.json') and report['protocol']==suite['protocol']
    assert report['case']==case() and native.read(directory/'smoke.json')['passed']
    assert [row['id'] for row in report['rows']]==[row['id'] for row in pin['quality']]
    assert all(row['prompt_ids']==expected['input_ids'] and 'error' not in row for row,expected in zip(report['rows'],pin['quality']))
    num=r'[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'
    decimal_fn=native._function(ROOT/'evaluate_quality.py','decimal',dict(re=re,NUM=num,Decimal=Decimal,InvalidOperation=InvalidOperation))
    grader=native._function(ROOT/'evaluate_quality.py','grade',dict(re=re,json=json,NUM=num,decimal=decimal_fn))
    for row,item,expected in zip(report['rows'],suite['cases'],pin['quality']):
        assert row['id']==item['id'] and row['group']==item['group'] and row['expected']==item['expected']
        assert all(row[key]==value for key,value in grader(item,row['text']).items())
        assert row['usage']['prompt_tokens']==len(expected['input_ids'])
        assert 0 <= row['usage']['completion_tokens'] <= item['max_tokens']
    assert report['correct']==sum(bool(row['correct']) for row in report['rows'])
    return report


def _check_http(report, input_ids, output_tokens, sampling):
    assert report['model']==case()['served_name']
    assert report['batch']==4 and report['output_tokens_per_request']==output_tokens
    assert report['prompt_inputs']==[input_ids] and report['sampling']==sampling and len(report['rounds'])==3
    groups=[]
    for trial in report['rounds']:
        assert len(trial['single']['requests'])==1 and len(trial['concurrent']['requests'])==4
        groups += [trial['single'],trial['concurrent']]
    assert len(report['streaming']['requests'])==4;groups.append(report['streaming'])
    for group in groups:
        assert math.isfinite(group['elapsed_s']) and group['elapsed_s']>0
        assert group['output_tokens']==output_tokens*len(group['requests'])
        assert math.isclose(group['throughput_tok_s'],group['output_tokens']/group['elapsed_s'],rel_tol=1e-10)
        for row in group['requests']:
            assert row['usage']['prompt_tokens']==len(input_ids) and row['usage']['completion_tokens']==output_tokens
            assert math.isfinite(row['elapsed_s']) and row['elapsed_s']>0


def workloads_gate(context):
    directory=Path(context['DEST']);report=native.read(directory/'workloads.json')
    pin=native.read(directory/'tokenizers/runtime.json')
    assert report['complete'] and report['case']==case() and report['inputs']==pin['workloads']
    names=('short_decode','medium_prompt','long_prompt','short_stochastic','mixed_lengths')
    assert set(report['workloads'])==set(names)
    specs=((128,512,{}),(512,128,{}),(1536,128,{}),(128,512,dict(temperature=.7,top_p=.9,top_k=40)))
    for name,(length,output,sampling) in zip(names,specs):
        _check_http(report['workloads'][name],pin['workloads'][str(length)],output,sampling)
    mixed=report['workloads']['mixed_lengths']
    assert mixed['max_client_inflight']==4 and mixed['requests_per_round']==16 and len(mixed['rounds'])==3
    assert mixed['sampling']==dict(temperature=.7,top_p=.9,top_k=40)
    for trial in mixed['rounds']:
        assert len(trial['requests'])==16
        assert [(row['input_tokens'],row['output_tokens']) for row in trial['requests']]==[(32,32),(128,128),(512,256),(1536,128)]*4
        for row in trial['requests']:
            assert row['usage']['prompt_tokens']==row['input_tokens'] and row['usage']['completion_tokens']==row['output_tokens']
    initial=native.read(directory/'initial-http.json')
    _check_http(initial,pin['initial_http_input_ids'],120,{})
    return report


def run(phase, context):
    current=_enter(context,phase)
    function={'numeric':_numeric,'cold':_cold,'long':_long,
        'quality':_serving,'workloads':_serving}[phase]
    return function(current)
