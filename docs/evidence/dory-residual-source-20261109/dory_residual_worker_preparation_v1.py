# Prepared complete residual replay worker; original full model setup retained.
# Entrypoint disabled pending full saved-byte review and fresh serial admission.
"""Long matrix original Dory CPU worker; fixed selected output rows.

Initialization copied from the completed full original projection control v1.

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
            head_rows, head_shapes = [None], []
            original_linear = torch.nn.functional.linear
            def routed_linear(x,weight,bias=None):
                if not control_active[0]:
                    return original_linear(x,weight,bias)
                key = (weight.data_ptr(),tuple(weight.shape),tuple(weight.stride()))
                name,layer,begin,end = routes[key]
                assert bias is None and x.ndim == 2 and x.dtype == weight.dtype == torch.bfloat16
                assert x.shape[-1] == weight.shape[-1]
                if name == 'lm_head.weight':
                    assert head_rows[0] is not None
                    x = x.index_select(0,head_rows[0]).contiguous()
                    head_shapes.append(list(x.shape))
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
            from dory_residual_replay_runtime_v1 import run
            run(dict(locals(), save=save))
    finally:
        mp.undo()


if __name__=='__main__':
    raise RuntimeError('Source preparation only: complete residual replay reviewer, fresh resource budget, source finalization and serial admission required')
