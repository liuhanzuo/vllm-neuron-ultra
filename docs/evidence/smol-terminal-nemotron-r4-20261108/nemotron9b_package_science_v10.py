"""Frozen metrics and expected route/state contracts; no native work on import."""
import ast
import collections
import json
from pathlib import Path
BUCKETS=(128,256,512,1024,2048,4096)
PREFIXES=(32,128,256,512,1024,2048,3070)
STEPS=2041
LENGTH=3072
MARKS=sorted(set([0,22,30,31,32,33,63,64,65,126,127,128,129,255,256,257,510,511,512,513,1022,1023,1024,1025,2042,2043,2047,2048,2049,2168,2169,3063,3064,3065]))
CRITERIA={'cosine_min':.995,'kl_max_nats':.02,'max_abs_error_max':2.,'argmax_match_fraction_min':.9,'all_reference_margins_gt_0.5_must_match':True}

def metrics(got,ref):
    import torch
    got=got.float().flatten()[:ref.numel()]
    ref=ref.float().flatten()
    assert got.shape==ref.shape and torch.isfinite(got).all()
    logp=torch.log_softmax(ref,dim=-1)
    logq=torch.log_softmax(got,dim=-1)
    top=torch.topk(ref,2)
    return {'max_abs_error':(got-ref).abs().max().item(),'mean_abs_error':(got-ref).abs().mean().item(),
        'cosine':torch.nn.functional.cosine_similarity(got,ref,dim=0).item(),
        'kl_reference_to_neuron':(logp.exp()*(logp-logq)).sum().item(),
        'argmax_match':bool(got.argmax()==ref.argmax()),'reference_margin':(top.values[0]-top.values[1]).item(),
        'reference_top1':int(ref.argmax()),'neuron_top1':int(got.argmax())}

def passed(rows):
    return bool(rows) and all(r['cosine']>=.995 and r['kl_reference_to_neuron']<=.02 and r['max_abs_error']<=2
               and (r['reference_margin']<=.5 or r['argmax_match']) for r in rows) and \
        sum(r['argmax_match'] for r in rows)/len(rows)>=.9

def graph_routes(model):
    rows = []
    mapping = model.traced_model.nxd_model.input_shape_map
    for signature, chosen in mapping.items():
        shapes = ast.literal_eval(signature)
        tag, index = str(chosen[0]), int(chosen[1])
        assert tag in ('context_encoding_model', 'token_generation_model')
        batch = 1 if tag == 'context_encoding_model' else 4
        assert shapes[0][0] == shapes[1][0] == batch
        bucket = shapes[0][1] if tag == 'context_encoding_model' else shapes[1][1]
        if tag == 'token_generation_model':
            assert shapes[0][1] == 1
        rows.append(dict(tag=tag, index=index, bucket=bucket, input_shapes=shapes))
    assert len(rows) == 12
    for tag in ('context_encoding_model', 'token_generation_model'):
        assert sorted((r['index'], r['bucket']) for r in rows if r['tag'] == tag) == list(enumerate(BUCKETS))
    return dict(complete=True, buckets=list(BUCKETS), graphs=rows)

def expected_bucket(length):
    for bucket in BUCKETS:
        if length<bucket:
            return bucket
    assert length==BUCKETS[-1]
    return BUCKETS[-1]

def validate_states(rows,zero=False):
    assert len(rows)==len({(r['rank'],r['name'])for r in rows})==124
    expected={((4,6144,4),'torch.bfloat16'):27,((4,64,80,128),'torch.float32'):27,
              ((4,4,4096,128),'torch.bfloat16'):8}
    for rank in (0,1):
        assert collections.Counter((tuple(r['shape']),r['dtype'])for r in rows if r['rank']==rank)==expected
    if zero:
        assert all(r['nonzero_bytes']==0 for r in rows)
    else:
        assert all(r['nan_elements']==r['inf_elements']==0 for r in rows)

def expected_long_frames(info):
    fixtures = [r['fixture'] for r in info['fixtures']];marks=info['positions']
    def row(ref,p,slot,phase):
        assert p in info['fixtures'][ref]['positions']
        return dict(trajectory=ref,fixture=fixtures[ref],position=p,slot=slot,phase=phase)
    refs,lengths,slots=[0,1,2,3],[23,127,1023,1],[3,1,0,2]
    rows=[row(r,n-1,s,'prefill')for r,n,s in zip(refs,lengths,slots)]
    for step in range(2041):
        rows += [row(r,n+step,s,'long_decode')for r,n,s in zip(refs,lengths,slots)if n+step in marks]
    rows.append(row(4,30,3,'slot_reuse_prefill'))
    rows += [row(r,p,s,'decode_after_slot_reuse')for r,p,s in zip([4,1,2,3],[31,2168,3064,2042],slots)]
    rows += [row(r,p,s,'partial_batch')for r,p,s in zip([1,2,3],[2169,3065,2043],slots[1:])]
    rows.append(row(4,32,3,'resume_inactive'))
    assert len(rows)==93
    return rows
