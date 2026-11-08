"""Independent stdlib byte/statistic check of the local projection replay."""
import hashlib
import json
from pathlib import Path
import sys
import time
import review_dory_two_loop_saved_v1 as reader

BASE=reader.BASE

def cross(a,aa,b,bb):
    assert a['shape']==b['shape'] and a['storage']['dtype']==b['storage']['dtype']
    ar,av=aa.tensor(a);br,bv=bb.tensor(b)
    return dict(exact_bytes=ar==br,unequal_values=sum(x!=y for x,y in zip(av,bv)),
        max_abs=max(abs(x-y)for x,y in zip(av,bv)),left_sha256=hashlib.sha256(ar).hexdigest(),right_sha256=hashlib.sha256(br).hexdigest())

def main():
    d=reader.d;fd,owner,lock=d.m.old.post.external.acquire_lock();started=time.monotonic()
    try:
        d.m.p.clear_window();d.m.p.active_heavy()
        end=json.loads((BASE/'decode-boundary-r1/result.json').read_text());d.m.p.gone(end['owner']);assert end['complete']
        root=BASE/'decode-boundary-r1/diagnostic';report=json.loads((root/'result.json').read_text())
        ar=reader.Archive(root/'captured.pt')
        old=reader.Archive(BASE/'two-loop-r1/diagnostic/float32/output-006.pt')
        weights=reader.Archive(BASE/'two-loop-r1/diagnostic/float32/synthetic-weights.pt')
        data=ar.data;found={}
        for key in ['sdpa_q','sdpa_k','sdpa_v','sdpa_out','oproj_input','oproj_output']:
            found[key]=cross(data['official_'+key],ar,data['candidate_'+key],ar)
        found['official_repeat']=cross(data['official_layer_output'],ar,data['control_onednn'],ar)
        found['candidate_same_input_control']=cross(data['candidate_layer_output'],ar,data['control_flinear'],ar)
        for phase in ['official','candidate']:
            for index,kind in enumerate(['key','value']):
                found[f'original_{phase}_{kind}_cache']=cross(data[phase+'_sdpa_'+('k'if index==0 else 'v')],ar,old.data[phase+'_caches'][0][index],old)
        assert found==report['comparisons'] and len(found)==12
        weight=cross(data['original_weight'],ar,weights.data['backbone.layers.0.mixer.o_proj.weight'],weights)
        assert weight['exact_bytes']
        assert not found['oproj_output']['exact_bytes'] and found['oproj_output']['unequal_values']==10
        files=[]
        for archive in [ar,old,weights]:
            files.append(dict(path=str(archive.path),sha256=archive.hash,bytes=archive.path.stat().st_size));archive.close()
        assert 'torch'not in sys.modules and 'numpy'not in sys.modules
        result=dict(complete=True,passed=True,owner=owner,comparisons=found,original_weight_exact=weight,
            files=files,elapsed_seconds=time.monotonic()-started,
            model_or_gemm_reexecuted=False,producer_local_projection_difference_reverified=True,
            full_model_logits_difference_fully_explained=False,full_model_qualified=False,new_speedup_claim=False)
        reader.save(BASE/'decode-boundary-r1-saved-review-v1.json',result)
        print(json.dumps(dict(passed=True,comparisons=12,original_weight_exact=True,projection_difference=found['oproj_output'])),flush=True)
    finally:d.m.old.post.external.release_lock(fd)

if __name__=='__main__':main()
