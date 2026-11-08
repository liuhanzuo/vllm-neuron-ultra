"""Replay tiny FP32 layer0 attention from saved state to locate the first delta."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

Q=Path(__file__).resolve().parent
BASE=Path('/mnt/nvme/dory-official-cpu-20261108-v1')
CAND=Q/'dory-native-reference-source-20261107-v1'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
    out=Path(args.output);out.mkdir()
    review=json.loads((BASE/'two-loop-r1-saved-review-v1.json').read_text());assert review['passed']
    hashes={f['path']:f['sha256']for f in review['files']}
    source=BASE/'two-loop-r1/diagnostic/float32'
    for name in ['synthetic-weights.pt','output-000.pt','output-006.pt']:
        p=source/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)]
    upstream_path=BASE/'vllm/tests/models/test_dory.py'
    assert hashlib.sha256(upstream_path.read_bytes()).hexdigest()=='54b9d821d7509cb2bbceb2886ac581255b26c33f3170d2288bea928821ee021f'
    import torch
    import pytest
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    assert str(torch.__version__)=='2.14.0+cpu'
    sys.path.insert(0,str(CAND))
    import dory_reference_candidate as candidate_source
    spec=importlib.util.spec_from_file_location('dory_original_fixture',upstream_path)
    upstream=importlib.util.module_from_spec(spec);spec.loader.exec_module(upstream)
    config=json.loads((BASE/'two-loop-r1/diagnostic/plan.json').read_text())['config']
    weights=torch.load(source/'synthetic-weights.pt',weights_only=True,map_location='cpu')
    prefill=torch.load(source/'output-000.pt',weights_only=True,map_location='cpu')
    previous_decode=torch.load(source/'output-006.pt',weights_only=True,map_location='cpu')
    captured={};phase=['official'];mp=pytest.MonkeyPatch();factory_gen=None
    def digest(t):return hashlib.sha256(t.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    def compare(a,b):
        assert a.shape==b.shape and a.dtype==b.dtype
        return dict(exact_bytes=digest(a)==digest(b),unequal_values=int((a!=b).sum()),
            max_abs=float((a.double()-b.double()).abs().max()),left_sha256=digest(a),right_sha256=digest(b))
    try:
        factory_gen=upstream.model_factory.__wrapped__(mp);factory=next(factory_gen)
        with torch.inference_mode():
            official=factory(**config).to(torch.float32)
            for p in official.parameters():p.zero_()
            official.load_weights(list(weights.items()));upstream._prepare_for_inference(official)
            candidate=candidate_source.DoryReferenceCandidate(config,weights,float32_fixture=True)
            epoch=candidate.start_request(0,'replay')
            cursor,histories=candidate.slots[0]
            cursor.length=6;cursor.begin(epoch,'replay',[6])
            histories.update({k:(v[0].clone(),v[1].clone())for k,v in prefill['candidate_caches'].items()})
            layer=official.backbone.layers[0].mixer;cache=layer.attn[0]
            cache.keys,cache.values=(t.clone()for t in prefill['official_caches'][0]);cache.start_pos=6
            x=weights['backbone.embeddings.weight'][13:14].clone()
            positions=torch.tensor([6],dtype=torch.long)
            original_sdpa=torch.nn.functional.scaled_dot_product_attention
            def sdpa(q,k,v,**kw):
                result=original_sdpa(q,k,v,**kw)
                captured[phase[0]+'_sdpa_q']=q.clone();captured[phase[0]+'_sdpa_k']=k.clone()
                captured[phase[0]+'_sdpa_v']=v.clone();captured[phase[0]+'_sdpa_out']=result.clone()
                return result
            mp.setattr(torch.nn.functional,'scaled_dot_product_attention',sdpa)
            original_linear=torch.nn.functional.linear
            ptr=candidate.weights['backbone.layers.0.mixer.o_proj.weight'].data_ptr()
            def linear(inp,weight,bias=None):
                result=original_linear(inp,weight,bias)
                if phase[0]=='candidate' and weight.data_ptr()==ptr:
                    captured['candidate_oproj_input']=inp.clone();captured['candidate_oproj_output']=result.clone()
                return result
            mp.setattr(torch.nn.functional,'linear',linear)
            def before(module,args):captured['official_oproj_input']=args[0].clone()
            def after(module,args,result):captured['official_oproj_output']=result[0].clone()
            prehook=layer.o_proj.register_forward_pre_hook(before)
            posthook=layer.o_proj.register_forward_hook(after)
            official_out=layer(positions,x)
            prehook.remove();posthook.remove()
            phase[0]='candidate'
            candidate_out=candidate._attention(x,positions,0,0,cursor,histories)
            phase[0]='controls'
            shared_input=captured['official_oproj_input']
            control_flinear=original_linear(shared_input,weights['backbone.layers.0.mixer.o_proj.weight'])
            control_onednn,_=layer.o_proj(shared_input)
            captured.update(input_embedding=x,control_flinear=control_flinear,control_onednn=control_onednn,
                original_weight=weights['backbone.layers.0.mixer.o_proj.weight'],official_layer_output=official_out,
                candidate_layer_output=candidate_out)
            comparisons={k:compare(captured['official_'+k],captured['candidate_'+k])
                for k in ['sdpa_q','sdpa_k','sdpa_v','sdpa_out','oproj_input','oproj_output']}
            comparisons.update(official_repeat=compare(official_out,control_onednn),
                candidate_same_input_control=compare(candidate_out,control_flinear),
                original_official_key_cache=compare(cache.keys,previous_decode['official_caches'][0][0]),
                original_official_value_cache=compare(cache.values,previous_decode['official_caches'][0][1]),
                original_candidate_key_cache=compare(histories[0][0],previous_decode['candidate_caches'][0][0]),
                original_candidate_value_cache=compare(histories[0][1],previous_decode['candidate_caches'][0][1]))
            for key in ['sdpa_q','sdpa_k','sdpa_v','sdpa_out','oproj_input','official_repeat','candidate_same_input_control',
                'original_official_key_cache','original_official_value_cache','original_candidate_key_cache','original_candidate_value_cache']:
                assert comparisons[key]['exact_bytes'],key
            assert not comparisons['oproj_output']['exact_bytes']
            torch.save(captured,out/'captured.pt')
            row=dict(complete=True,passed=True,position=6,token=13,dtype='float32',layer=0,
                comparisons=comparisons,backend=list(layer.o_proj.cpu_linear.__code__.co_names),
                observed_local_first_difference='oneDNN output projection versus F.linear on exactly identical input and weight',
                local_difference_caused_by_projection_path=True,full_model_logits_difference_fully_explained=False,
                source_state='Saved original tiny fixture weights and six-token cache; isolated first attention layer replay',
                original_weight_model_executed=False,full_model_qualified=False,new_speedup_claim=False)
            with (out/'result.json').open('x')as f:json.dump(row,f,indent=2)
    finally:
        if factory_gen is not None:factory_gen.close()
        mp.undo()

if __name__=='__main__':main()
