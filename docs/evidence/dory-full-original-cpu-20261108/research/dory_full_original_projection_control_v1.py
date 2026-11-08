"""Full original Dory causal control with common official projection backend.

Official oneDNN is authoritative. Only candidate F.linear calls are routed to
the same official projection modules; original weights and non-projection code
are unchanged. The previous F.linear failure remains. This is not a new baseline.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

Q = Path(__file__).resolve().parent
sys.path[:0] = [str(Q), str(Q.parent)]
import dory_checkpoint_loader_v1 as loader
import run_dory_weight_storage_root_v2 as d


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', required=True)
    ap.add_argument('--admission', required=True)
    a = ap.parse_args()
    spec = json.loads(Path(a.spec).read_text())
    admission = json.loads(Path(a.admission).read_text())
    assert admission['spec_sha256'] == hashlib.sha256(Path(a.spec).read_bytes()).hexdigest()
    for name, sha in spec['source_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == sha, name
    budget = spec['budget']
    resource.setrlimit(resource.RLIMIT_AS, (budget['address_space_bytes'],)*2)
    resource.setrlimit(resource.RLIMIT_CPU, (budget['cpu_seconds'],budget['cpu_seconds']+5))
    resource.setrlimit(resource.RLIMIT_FSIZE, (budget['single_file_bytes'],)*2)
    out = Path(spec['output'])/'diagnostic'
    out.mkdir()
    start, last = time.monotonic(), [0.]

    def guard():
        now = time.monotonic()
        assert now-start < budget['wall_seconds']
        if now-last[0] > .5:
            d.m.p.lock_proof(admission['owner'], admission['lock'])
            d.m.p.clear_window()
            assert os.sched_getaffinity(0) == set(budget['cpus'])
            last[0] = now

    guard()
    import torch
    import pytest
    import transformers
    import tokenizers
    from transformers import PreTrainedTokenizerFast
    from run_dory_tokenizer_template_v1 import fixtures
    assert str(torch.__version__) == '2.14.0+cpu' and transformers.__version__ == '5.20.2'
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    metadata = Path(spec['tokenizer_directory'])
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(metadata), local_files_only=True)
    native_tokenizer = tokenizers.Tokenizer.from_file(str(metadata/'tokenizer.json'))
    prior = json.loads(Path(spec['previous_token_captures']).read_text())
    assert tokenizer.chat_template == (metadata/'chat_template.jinja').read_text()
    assert (len(tokenizer), tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id) == (131072,1,11,11)
    captures = []
    for fixture in fixtures():
        rendered = tokenizer.apply_chat_template(fixture['messages'], tokenize=False, **fixture['kwargs'])
        old = next(r for r in prior['cases'] if r['name'] == fixture['name'])
        assert rendered == old['rendered']
        ids = tokenizer.apply_chat_template(fixture['messages'], tokenize=True, **fixture['kwargs'])
        if hasattr(ids, 'keys'):
            ids = ids['input_ids']
        if ids and isinstance(ids[0],list):
            assert len(ids) == 1
            ids = ids[0]
        assert ids == old['input_ids'] == native_tokenizer.encode(rendered,add_special_tokens=False).ids
        assert tokenizer.encode(rendered,add_special_tokens=True) == ids
        assert tokenizer.decode(ids,skip_special_tokens=False) == rendered
        captures.append(dict(name=fixture['name'], rendered=rendered, input_ids=ids))
    for row in prior['specials']:
        assert tokenizer.encode(row['content'],add_special_tokens=False) == [row['id']]
        assert tokenizer.decode([row['id']],skip_special_tokens=False) == row['content']
    policy = json.loads((metadata/'generation_config.json').read_text())
    assert policy == prior['generation_policy']
    assert (policy['eos_token_id'],policy['do_sample'],policy['temperature'],policy['top_p'],policy['top_k']) == ([2,11],True,1.0,.95,0)
    save(out/'tokenizer.json', dict(complete=True, cases=captures, special_ids_checked=len(prior['specials']),
         versions=dict(transformers=transformers.__version__,tokenizers=tokenizers.__version__),
         all_previous_prompts_and_ids_exact=True,generation_policy=policy,generation_executed=False))
    print(json.dumps(dict(stage='tokenizer',cases=len(captures),specials=len(prior['specials']))),flush=True)

    source = Path(spec['official_source'])
    module_spec = importlib.util.spec_from_file_location('upstream_dory_cpu_reference',source/'tests/models/test_dory.py')
    upstream = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(upstream)
    from vllm.config import VllmConfig, DeviceConfig, CompilationConfig, set_current_vllm_config
    from vllm.distributed import parallel_state
    from vllm.model_executor.models import dory as official_source
    from vllm.model_executor.models.utils import extract_layer_index
    from vllm.transformers_utils.configs.dory import DoryConfig
    import vllm._custom_ops as ops
    sys.path.insert(0,str(Q/'dory-native-reference-source-20261107-v1'))
    from dory_reference_candidate import DoryReferenceCandidate
    from dory_contract import DoryContract
    config = loader.source_contract()['config']
    contract = DoryContract.from_mapping(config)
    counts, layer_calls = [0], {}
    original_op = ops.onednn_mm

    def count_op(*args,**kwargs):
        counts[0] += 1
        return original_op(*args,**kwargs)

    def raw(t):
        return t.detach().contiguous().view(torch.uint8).numpy().tobytes()

    def digest(t):
        return hashlib.sha256(raw(t)).hexdigest()

    def compare(a,b):
        assert a.dtype == b.dtype and a.shape == b.shape
        return dict(exact_bytes=digest(a)==digest(b), unequal_values=int((a!=b).sum()),
                    max_abs=float((a.double()-b.double()).abs().max()),
                    finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()))

    def snapshot(hidden, logits, histories):
        value = dict(hidden=hidden,logits=logits)
        assert set(histories) == {x['cache_id'] for x in contract.cache_plan()}
        for key,(k,v) in sorted(histories.items()):
            value[f'cache-{key}-key'] = k
            value[f'cache-{key}-value'] = v
        assert len(value) == 114
        return value

    def archive(name, tensors):
        path = out/(name+'.bin')
        rows, offset, h = {},0,hashlib.sha256()
        with path.open('xb') as f:
            for label,tensor in tensors.items():
                guard()
                assert tensor.dtype in (torch.bfloat16,torch.float32)
                data = raw(tensor)
                f.write(data)
                h.update(data)
                rows[label] = dict(shape=list(tensor.shape),dtype='BF16' if tensor.dtype==torch.bfloat16 else 'F32',
                                   offset=offset,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
                offset += len(data)
        return dict(file=path.name,bytes=offset,sha256=h.hexdigest(),tensors=rows)

    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(ops,'onednn_mm',count_op)
        mp.setattr(parallel_state,'_TP',SimpleNamespace(world_size=1,rank_in_group=0))
        mp.setattr(official_source,'Attention',upstream.ReferenceAttention)
        vc = VllmConfig(device_config=DeviceConfig(device='cpu'),compilation_config=CompilationConfig(mode=0,custom_ops=['none']))
        vc.model_config = SimpleNamespace(hf_config=DoryConfig(**config),head_dtype=None,dtype=torch.bfloat16)
        with torch.inference_mode(),set_current_vllm_config(vc):
            weights,proof = loader.load_checkpoint(Path(spec['checkpoint']['path']),spec['checkpoint'],torch=torch,guard=guard)
            assert proof['tensors'] == json.loads(Path(spec['previous_loader_proof']).read_text())['loader']['tensors']
            save(out/'loader-proof.json',proof)
            print(json.dumps(dict(stage='weights_loaded',tensors=len(weights))),flush=True)
            prior_dtype = torch.get_default_dtype()
            try:
                torch.set_default_dtype(torch.bfloat16)
                official = official_source.DoryForCausalLM(vllm_config=vc)
            finally:
                torch.set_default_dtype(prior_dtype)
            assert len(official.backbone.layers) == 72 and official.backbone.num_loops == 2
            assert all(p.dtype == torch.bfloat16 for p in official.parameters())
            params = dict(official.named_parameters())
            assert len(params) == 399 and sum(p.numel() for p in params.values()) == 3975229440
            loaded = official.load_weights(list(weights.items()))
            assert loaded == set(params)
            proof_by_name = {r['name']:r for r in proof['tensors']}
            mapped = []
            for name,tensor in weights.items():
                guard()
                target,begin,end = name,0,tensor.shape[0]
                for old,new,offset in [('q_proj','qkv_proj',0),('k_proj','qkv_proj',4096),('v_proj','qkv_proj',5120),
                                       ('gate_proj','gate_up_proj',0),('up_proj','gate_up_proj',7168)]:
                    if '.'+old+'.' in name:
                        target=name.replace('.'+old+'.','.'+new+'.')
                        begin,end=offset,offset+tensor.shape[0]
                        break
                actual = params[target][begin:end]
                assert list(actual.shape) == list(tensor.shape)
                sha = digest(actual)
                assert sha == proof_by_name[name]['sha256']
                mapped.append(dict(original=name,target=target,begin=begin,end=end,sha256=sha))
            save(out/'loaded-parameters.json',dict(all507_exact=True,physical_parameters=399,parameters=3975229440,mapping=mapped))
            del params
            mp.setattr(official.logits_processor,'_gather_logits',lambda logits:logits)
            upstream._prepare_for_inference(official)
            backends={name:list(m.cpu_linear.__code__.co_names) for name,m in official.named_modules() if hasattr(m,'cpu_linear')}
            assert len(backends) == 182 and all('onednn_mm' in names for names in backends.values())
            caches = [m for m in official.modules() if isinstance(m,upstream.ReferenceAttention)]
            assert len(caches)==56 and {extract_layer_index(m.prefix)for m in caches} == {x['cache_id']for x in contract.cache_plan()}
            ropes = {id(m.rotary_emb):m.rotary_emb for m in (x.mixer for x in official.backbone.layers) if hasattr(m,'rotary_emb')}
            assert len(ropes)==2
            rope_metadata=[]
            for rope in ropes.values():
                assert list(rope.cos_sin_cache.shape)==[262144,256] and rope.cos_sin_cache.dtype==torch.bfloat16
                rope_metadata.append(dict(shape=list(rope.cos_sin_cache.shape),dtype=str(rope.cos_sin_cache.dtype),base=rope.base))
            candidate = DoryReferenceCandidate(config,weights)
            # Causal intervention only: actual candidate source remains byte-pinned.
            # Route each original matrix by storage identity, never by its values.
            routes, route_calls = {}, {}
            for row in mapped:
                name = row['original']
                weight = weights[name]
                if weight.ndim != 2 or name == 'backbone.embeddings.weight':
                    continue
                layer = official.get_submodule(row['target'].rsplit('.',1)[0])
                key = (weight.data_ptr(), tuple(weight.shape), tuple(weight.stride()))
                assert key not in routes
                routes[key] = (name,layer,row['begin'],row['end'])
                route_calls[name] = 0
            assert len(routes) == 289
            control_active = [False]
            original_linear = torch.nn.functional.linear
            def routed_linear(x,weight,bias=None):
                if not control_active[0]:
                    return original_linear(x,weight,bias)
                key = (weight.data_ptr(),tuple(weight.shape),tuple(weight.stride()))
                name,layer,begin,end = routes[key]
                assert bias is None and x.ndim == 2 and x.dtype == weight.dtype == torch.bfloat16
                assert x.shape[-1] == weight.shape[-1]
                value = layer.quant_method.apply(layer,x,bias=None) if name == 'lm_head.weight' else layer(x)[0]
                selected = value[:,begin:end].contiguous()
                assert selected.shape == (x.shape[0],weight.shape[0])
                route_calls[name] += 1
                return selected
            mp.setattr(torch.nn.functional,'linear',routed_linear)

            def set_positions(model,args):
                for cache in caches:
                    cache.start_pos = int(args[1][0])

            official.register_forward_pre_hook(set_positions)
            for index,layer in enumerate(official.backbone.layers):
                def count_layer(module,args,output,_index=index):
                    guard()
                    layer_calls[_index]=layer_calls.get(_index,0)+1
                layer.register_forward_hook(count_layer)
            save(out/'model.json',dict(config=config,official_parameters_before_postload=399,all_original_weights_exact=True,
                backends=backends,cache_ids=sorted(extract_layer_index(m.prefix)for m in caches),rotary_tables=rope_metadata,
                attention='Unchanged upstream CPU ReferenceAttention; not production paged attention',
                candidate_source_unchanged=True,projection_backend_replaced_for_causal_control=True,full_model_constructed=True))
            tokens = torch.tensor(spec['tokens'],dtype=torch.long)
            assert len(tokens)==32 and spec['schedule']==[[0,7]]+[[i,i+1]for i in range(7,32)]
            epoch=candidate.start_request(0,'original-first')
            steps=[]
            first_official=first_candidate=None
            print(json.dumps(dict(stage='full_model_ready',physical_layers=72,histories=56)),flush=True)

            def call(start,end,epoch,request):
                layer_calls.clear()
                before=counts[0]
                h=official(tokens[start:end],torch.arange(start,end))
                logits=official.compute_logits(h)
                assert counts[0]-before==281
                expected_calls={i:2 if 16<=i<56 else 1 for i in range(72)}
                assert layer_calls==expected_calls
                observed={extract_layer_index(c.prefix):(c.keys,c.values)for c in caches}
                assert all(list(t.shape)==[4,end,256]for pair in observed.values()for t in pair)
                assert len({pair[0].data_ptr()for pair in observed.values()})==56
                official_state=snapshot(h,logits,observed)
                control_active[0] = True
                try:
                    ch,cl=candidate.forward(0,epoch,request,tokens[start:end],range(start,end))
                finally:
                    control_active[0] = False
                candidate_histories=candidate.slots[0][1]
                assert all(list(t.shape)==[4,end,256]for pair in candidate_histories.values()for t in pair)
                assert len({pair[0].data_ptr()for pair in candidate_histories.values()})==56
                candidate_state=snapshot(ch,cl,candidate_histories)
                checks={name:compare(official_state[name],candidate_state[name])for name in official_state}
                assert all(c['finite']for c in checks.values())
                return official_state,candidate_state,checks

            for start_pos,end in spec['schedule']:
                guard()
                osnap,csnap,checks=call(start_pos,end,epoch,'original-first')
                row=dict(start=start_pos,end=end,official=archive(f'official-{start_pos:03d}',osnap),
                    candidate=archive(f'candidate-{start_pos:03d}',csnap),comparisons=checks,
                    top1_matches=int((osnap['logits'].argmax(-1)==csnap['logits'].argmax(-1)).sum()),
                    official_top1=osnap['logits'].argmax(-1).tolist(),candidate_top1=csnap['logits'].argmax(-1).tolist(),
                    actual_onednn_calls=281,physical_layer_invocations=dict(layer_calls))
                if start_pos==0:
                    first_official={k:v.clone()for k,v in osnap.items()}
                    first_candidate={k:v.clone()for k,v in csnap.items()}
                steps.append(row)
                save(out/f'step-{start_pos:03d}.json',row)
                print(json.dumps(dict(stage='forward',start=start_pos,end=end,logits=checks['logits'],
                     top1_matches=row['top1_matches'],elapsed_seconds=time.monotonic()-start)),flush=True)
            candidate.release_request(0,epoch,'original-first')
            assert not candidate.slots
            for cache in caches:
                cache.keys=cache.values=None
            new_epoch=candidate.start_request(0,'original-reused')
            assert new_epoch>epoch
            rejected=False
            try:
                candidate.forward(0,epoch,'original-first',tokens[:7],range(7))
            except ValueError:
                rejected=True
            assert rejected
            osnap,csnap,checks=call(0,7,new_epoch,'original-reused')
            reuse=dict(official=archive('official-reused',osnap),candidate=archive('candidate-reused',csnap),
                official_checks={n:compare(first_official[n],t)for n,t in osnap.items()},
                candidate_checks={n:compare(first_candidate[n],t)for n,t in csnap.items()},
                stale_candidate_handle_rejected=True,old_epoch=epoch,new_epoch=new_epoch,
                production_slot_reuse=False)
            assert all(c['exact_bytes'] and c['finite'] for k in ('official_checks','candidate_checks')for c in reuse[k].values())
            save(out/'reuse.json',reuse)
            assert counts[0]==27*(281+449)==19710
            assert sum(route_calls.values())==27*449==12123
            for name,number in route_calls.items():
                layer_index=int(name.split('.')[2]) if name.startswith('backbone.layers.') else -1
                assert number==27*(2 if 16<=layer_index<56 else 1)
            all_checks=[c for row in steps for c in row['comparisons'].values()]
            summary=dict(complete=True,original_weight_full_forward_executed=True,physical_layers=72,loops=2,cache_histories=56,
                tokens=spec['tokens'],schedule=spec['schedule'],calls=26,compared_tensors=len(all_checks),
                exact_tensors=sum(c['exact_bytes']for c in all_checks),all_exact=all(c['exact_bytes']for c in all_checks),
                exact_logit_calls=sum(row['comparisons']['logits']['exact_bytes']for row in steps),
                max_logit_abs=max(row['comparisons']['logits']['max_abs']for row in steps),
                top1_matches=sum(row['top1_matches']for row in steps),top1_tokens=32,
                all_states_finite=True,actual_onednn_calls=counts[0],reference_restart_all_states_exact=True,
                causal_projection_control=True,candidate_projection_calls=sum(route_calls.values()),
                candidate_matrix_call_counts=route_calls,original_F_linear_candidate_still_unqualified=True,
                steps=steps,reuse=reuse,elapsed_seconds=time.monotonic()-start,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                generation_executed=False,neuron_baseline_executed=False,full_long_decode_executed=False,
                production_slot_reuse_executed=False,full_model_qualified=False,new_speedup_claim=False)
            save(out/'result.json',summary)
            print(json.dumps({k:v for k,v in summary.items()if k not in ('steps','reuse','tokens','schedule')}),flush=True)
    finally:
        mp.undo()


if __name__=='__main__':
    main()
