"""Synthetic two-loop official/candidate comparison; never a full-model gate.

Predeclared exact-value diagnostic, retaining mismatches without changing either
implementation. The official attention is its own upstream CPU test fixture.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

Q=Path('/home/ubuntu/experiments/trn1-latest-models-20261031/research')
BASE=Path('/mnt/nvme/dory-official-cpu-20261108-v1')
SOURCE=BASE/'vllm'
CANDIDATE=Q/'dory-native-reference-source-20261107-v1'
PINS={
    SOURCE/'tests/models/test_dory.py':'54b9d821d7509cb2bbceb2886ac581255b26c33f3170d2288bea928821ee021f',
    CANDIDATE/'dory_reference_candidate.py':'366dc0ab95bbeebb3bd2311f205e23afc63ce3fbbcfca321345185dcbc191716',
}

def save(p,j):
    with p.open('x') as f:json.dump(j,f,indent=2)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args()
    out=Path(a.output);out.mkdir()
    for p,h in PINS.items():assert hashlib.sha256(p.read_bytes()).hexdigest()==h
    import torch
    import pytest
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    assert str(torch.__version__)=='2.14.0+cpu'
    sys.path.insert(0,str(CANDIDATE))
    from dory_contract import DoryContract
    from dory_reference_candidate import DoryReferenceCandidate
    spec=importlib.util.spec_from_file_location('dory_official_original_test',SOURCE/'tests/models/test_dory.py')
    upstream=importlib.util.module_from_spec(spec);spec.loader.exec_module(upstream)
    import vllm._custom_ops as ops
    from vllm.model_executor.models.utils import extract_layer_index
    config=json.loads((CANDIDATE/'official-config.json').read_text())
    config.update(vocab_size=32,hidden_size=16,intermediate_size=24,
        num_hidden_layers=8,n_input_layers=2,n_recurrent_layers=4,n_output_layers=2,
        n_recurrent_loops=2,num_attention_heads=2,num_key_value_heads=1,head_dim=8,
        max_position_embeddings=128,sliding_window=3,
        layer_types=['sliding_attention','mlp']*3+['full_attention','mlp'])
    contract=DoryContract.from_mapping(config)
    save(out/'plan.json',dict(config=config,seed=17,prefill_tokens=6,total_tokens=33,
        dtypes=['float32','bfloat16'],acceptance='Exact finite values and bytes; descriptive alignment only',
        source_sha256={str(p):h for p,h in PINS.items()},slot_reuse=True,
        actual_model_dimensions=False,original_weights=False,full_long_decode_gate=False))
    rows=[]
    def digest(t):return hashlib.sha256(t.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    def compare(x,y):
        assert x.shape==y.shape and x.dtype==y.dtype
        delta=(x.double()-y.double()).abs()
        return dict(shape=list(x.shape),dtype=str(x.dtype),exact_bytes=digest(x)==digest(y),
            unequal_values=int((x!=y).sum()),max_abs=float(delta.max()),
            finite=bool(torch.isfinite(x).all() and torch.isfinite(y).all()),
            official_sha256=digest(x),candidate_sha256=digest(y))
    with torch.inference_mode():
        for dtype in [torch.float32,torch.bfloat16]:
            label=str(dtype).split('.')[-1];run=out/label;run.mkdir()
            generator=torch.Generator().manual_seed(17)
            weights={name:(torch.randint(-16,17,shape,generator=generator).float()/128).to(dtype)
                for name,shape in contract.weight_shapes().items()}
            torch.save(weights,run/'synthetic-weights.pt')
            candidate=DoryReferenceCandidate(config,weights,float32_fixture=dtype==torch.float32)
            op_counts={};mp=pytest.MonkeyPatch();factory_gen=None
            try:
                for name in ['onednn_mm','weight_packed_linear_cpu']:
                    if not hasattr(ops,name):continue
                    original=getattr(ops,name)
                    def wrapper(*args,_fn=original,_name=name,**kwargs):
                        op_counts[_name]=op_counts.get(_name,0)+1
                        return _fn(*args,**kwargs)
                    mp.setattr(ops,name,wrapper)
                factory_gen=upstream.model_factory.__wrapped__(mp);factory=next(factory_gen)
                official=factory(**config).to(dtype)
                for parameter in official.parameters():parameter.zero_()
                official.load_weights(list(weights.items()))
                mp.setattr(official.logits_processor,'_gather_logits',lambda logits:logits)
                upstream._prepare_for_inference(official)
                backends={name:list(getattr(mod.cpu_linear,'__code__').co_names)
                    for name,mod in official.named_modules()
                    if hasattr(mod,'cpu_linear') and hasattr(mod.cpu_linear,'__code__')}
                caches=[m for m in official.modules()if isinstance(m,upstream.ReferenceAttention)]
                assert len(caches)==len(contract.cache_plan())==6
                actual_ids={extract_layer_index(c.prefix)for c in caches}
                assert actual_ids=={c['cache_id']for c in contract.cache_plan()}
                primitive=[]
                for tag,layer,names in [
                    ('qkv',official.backbone.layers[0].mixer.qkv_proj,
                     ['backbone.layers.0.mixer.'+n+'.weight'for n in ['q_proj','k_proj','v_proj']]),
                    ('gate_up',official.backbone.layers[1].mixer.gate_up_proj,
                     ['backbone.layers.1.mixer.'+n+'.weight'for n in ['gate_proj','up_proj']])]:
                    for batch in [1,6]:
                        inp=(torch.arange(batch*16).reshape(batch,16).remainder(31).float()/32-0.5).to(dtype)
                        observed,_=layer(inp)
                        packed=torch.nn.functional.linear(inp,torch.cat([weights[n]for n in names]))
                        split=torch.cat([torch.nn.functional.linear(inp,weights[n])for n in names],dim=-1)
                        primitive.append(dict(projection=tag,batch=batch,official_vs_packed=compare(observed,packed),
                            official_vs_split=compare(observed,split),packed_vs_split=compare(packed,split)))
                        torch.save(dict(input=inp,official=observed,packed=packed,split=split),run/f'primitive-{tag}-{batch}.pt')
                tokens=(torch.arange(33)*7+3)%32
                epoch=candidate.start_request(0,'first');steps=[]
                first_official=first_candidate=None
                for start,end in [(0,6)]+[(i,i+1)for i in range(6,33)]:
                    h=official(tokens[start:end],torch.arange(start,end));logits=official.compute_logits(h)
                    ch,cl=candidate.forward(0,epoch,'first',tokens[start:end],range(start,end))
                    checks=[]
                    for cache in caches:
                        key=extract_layer_index(cache.prefix);ck,cv=candidate.slots[0][1][key]
                        checks.append(dict(cache_id=key,key=compare(cache.keys,ck),value=compare(cache.values,cv)))
                    steps.append(dict(start=start,end=end,hidden=compare(h,ch),logits=compare(logits,cl),caches=checks))
                    torch.save(dict(hidden=h.clone(),logits=logits.clone(),candidate_hidden=ch.clone(),candidate_logits=cl.clone(),
                        official_caches={extract_layer_index(c.prefix):(c.keys.clone(),c.values.clone())for c in caches},
                        candidate_caches={k:(v[0].clone(),v[1].clone())for k,v in candidate.slots[0][1].items()}),run/f'output-{start:03d}.pt')
                    if start==0:first_official=logits.clone();first_candidate=cl.clone()
                candidate.release_request(0,epoch,'first')
                assert 0 not in candidate.slots
                for cache in caches:cache.keys=cache.values=None
                new_epoch=candidate.start_request(0,'second');assert new_epoch>epoch
                stale_rejected=False
                try:candidate.forward(0,epoch,'first',tokens[:6],range(6))
                except ValueError:stale_rejected=True
                assert stale_rejected
                rh=official(tokens[:6],torch.arange(6));rl=official.compute_logits(rh)
                _,rcl=candidate.forward(0,new_epoch,'second',tokens[:6],range(6))
                reuse=dict(official=compare(first_official,rl),candidate=compare(first_candidate,rcl),stale_owner_rejected=stale_rejected)
                assert reuse['official']['exact_bytes'] and reuse['candidate']['exact_bytes']
                row=dict(dtype=label,steps=steps,primitives=primitive,backends=backends,actual_operator_calls=op_counts,
                    cache_ids=sorted(actual_ids),slot_reuse=reuse,
                    all_logits_exact=all(s['logits']['exact_bytes']for s in steps),
                    all_hidden_exact=all(s['hidden']['exact_bytes']for s in steps),
                    all_cache_values_exact=all(c[k]['exact_bytes']for s in steps for c in s['caches']for k in ['key','value']))
                save(run/'result.json',row);rows.append(row)
            finally:
                if factory_gen is not None:factory_gen.close()
                mp.undo()
    save(out/'result.json',dict(complete=True,dtypes=rows,all_exact=all(r['all_logits_exact']and r['all_hidden_exact']and r['all_cache_values_exact']for r in rows),
        synthetic_two_loop_fixture_executed=True,original_weight_forward=False,
        full_model_qualified=False,new_speedup_claim=False))

if __name__=='__main__':main()
