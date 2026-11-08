"""Reproduce full original Dory saved tensor metrics using the stdlib only."""
import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import resource
import struct
import sys
import time

Q=Path(__file__).resolve().parent
sys.path[:0]=[str(Q),str(Q.parent)]
import run_dory_weight_storage_root_v2 as d


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--spec',required=True)
    ap.add_argument('--admission',required=True)
    args=ap.parse_args()
    spec=json.loads(Path(args.spec).read_text())
    admission=json.loads(Path(args.admission).read_text())
    assert hashlib.sha256(Path(args.spec).read_bytes()).hexdigest()==admission['spec_sha256']
    for path,sha in spec['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
    resource.setrlimit(resource.RLIMIT_AS,(2*2**30,)*2)
    resource.setrlimit(resource.RLIMIT_CPU,(900,905))
    started=time.monotonic()
    last=[0.]
    def guard():
        now=time.monotonic()
        assert now-started<1000
        if now-last[0]>.5:
            d.m.p.lock_proof(admission['owner'],admission['lock'])
            d.m.p.clear_window()
            last[0]=now
    out=Path(spec['output'])/'diagnostic'
    result=json.loads((out/'result.json').read_text())
    assert result['complete'] and result['original_weight_full_forward_executed']
    assert result['schedule']==spec['schedule']==[[0,7]]+[[i,i+1]for i in range(7,32)]
    assert result['tokens']==spec['tokens']
    model=json.loads((out/'model.json').read_text())
    cache_ids=model['cache_ids']
    assert cache_ids==sorted(list(range(0,72,2))+list(range(88,128,2))) and len(cache_ids)==56
    proof=json.loads((out/'loader-proof.json').read_text())
    checkpoint=Path(spec['checkpoint']['path'])
    s=checkpoint.stat()
    assert dict(path=str(checkpoint),device=s.st_dev,inode=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)==spec['checkpoint']
    tensors={r['name']:r for r in proof['tensors']}
    with checkpoint.open('rb')as f:
        prefix=f.read(8)
        raw_header=f.read(struct.unpack('<Q',prefix)[0])
        header=json.loads(raw_header)
        base=8+len(raw_header)
        full=hashlib.sha256(prefix+raw_header)
        assert set(header)==set(tensors) and len(header)==507
        for name,row in sorted(header.items(),key=lambda x:x[1]['data_offsets'][0]):
            begin,end=row['data_offsets']
            assert f.tell()==base+begin
            h=hashlib.sha256()
            left=end-begin
            while left:
                guard()
                data=f.read(min(left,4*2**20))
                assert data
                left-=len(data)
                h.update(data)
                full.update(data)
            assert h.hexdigest()==tensors[name]['sha256']
        assert f.tell()==spec['checkpoint']['size'] and not f.read(1)
        assert full.hexdigest()==spec['checkpoint_sha256']==proof['full_sha256']
    loaded=json.loads((out/'loaded-parameters.json').read_text())
    assert len(loaded['mapping'])==507 and {r['original']for r in loaded['mapping']}==set(tensors)
    for row in loaded['mapping']:
        assert row['sha256']==tensors[row['original']]['sha256']
    tokenizer=json.loads((out/'tokenizer.json').read_text())
    previous=json.loads(Path(spec['previous_token_captures']).read_text())
    assert len(tokenizer['cases'])==13 and tokenizer['special_ids_checked']==1000
    for actual in tokenizer['cases']:
        old=next(r for r in previous['cases']if r['name']==actual['name'])
        assert actual=={k:old[k]for k in ('name','rendered','input_ids')}
    assert tokenizer['generation_policy']==previous['generation_policy']
    decoded={}
    def number(word):
        if word not in decoded:
            decoded[word]=struct.unpack('<f',struct.pack('<I',word<<16))[0]
        return decoded[word]
    archives={}
    def load(record,m,total):
        guard()
        p=out/record['file']
        assert p.name==record['file'] and sys.byteorder=='little'
        data=p.read_bytes()
        assert len(data)==record['bytes'] and hashlib.sha256(data).hexdigest()==record['sha256']
        wanted={'hidden','logits'}|{f'cache-{cid}-{kind}'for cid in cache_ids for kind in ('key','value')}
        assert set(record['tensors'])==wanted and len(wanted)==114
        cursor,values=0,{}
        for name,r in record['tensors'].items():
            shape=[m,131072]if name=='logits'else [m,2560]if name=='hidden'else [4,total,256]
            dtype='F32'if name=='logits'else 'BF16'
            size=4 if dtype=='F32'else 2
            assert r['shape']==shape and r['dtype']==dtype and r['offset']==cursor and r['bytes']==math.prod(shape)*size
            chunk=data[cursor:cursor+r['bytes']]
            cursor+=r['bytes']
            assert hashlib.sha256(chunk).hexdigest()==r['sha256']
            vals=array('f'if dtype=='F32'else 'H')
            vals.frombytes(chunk)
            assert all(math.isfinite(v)for v in vals)if dtype=='F32'else all((v&0x7f80)!=0x7f80 for v in vals)
            values[name]=(chunk,vals,dtype)
        assert cursor==len(data)
        archives[record['file']]=dict(bytes=len(data),sha256=record['sha256'])
        return values
    def compare(a,b):
        assert a[2]==b[2] and len(a[1])==len(b[1])
        unequal,maximum=0,0.
        for x,y in zip(a[1],b[1]):
            if x!=y:
                xx,yy=(number(x),number(y))if a[2]=='BF16'else (x,y)
                unequal+=xx!=yy
                maximum=max(maximum,abs(xx-yy))
        return dict(exact_bytes=a[0]==b[0],unequal_values=unequal,max_abs=maximum,finite=True)
    def argmaxes(value,m):
        vals=value[1]
        assert value[2]=='F32' and len(vals)==m*131072
        return [max(range(131072),key=lambda i:vals[row*131072+i])for row in range(m)]
    checks,exact,logits_exact,top1=0,0,0,0
    maximum=0.
    first=None
    assert len(result['steps'])==26
    for row,(start,end)in zip(result['steps'],spec['schedule']):
        assert (row['start'],row['end'])==(start,end)
        assert row==json.loads((out/f'step-{start:03d}.json').read_text())
        assert row['actual_onednn_calls']==281
        assert row['physical_layer_invocations']=={str(i):2 if 16<=i<56 else 1 for i in range(72)}
        oa=load(row['official'],end-start,end)
        ca=load(row['candidate'],end-start,end)
        assert set(row['comparisons'])==set(oa)==set(ca)
        for name in oa:
            actual=compare(oa[name],ca[name])
            assert actual==row['comparisons'][name],(start,name)
            checks+=1
            exact+=actual['exact_bytes']
            if name=='logits':
                logits_exact+=actual['exact_bytes']
                maximum=max(maximum,actual['max_abs'])
        ot,ct=argmaxes(oa['logits'],end-start),argmaxes(ca['logits'],end-start)
        assert ot==row['official_top1'] and ct==row['candidate_top1']
        matches=sum(x==y for x,y in zip(ot,ct))
        assert matches==row['top1_matches']
        top1+=matches
        if start==0:
            first=(oa,ca)
    reuse=result['reuse']
    assert reuse==json.loads((out/'reuse.json').read_text())
    assert reuse['new_epoch']>reuse['old_epoch'] and reuse['stale_candidate_handle_rejected']
    repeated=(load(reuse['official'],7,7),load(reuse['candidate'],7,7))
    reuse_checks=0
    for label,left,right in zip(('official_checks','candidate_checks'),first,repeated):
        for name in left:
            actual=compare(left[name],right[name])
            assert actual==reuse[label][name] and actual['exact_bytes']
            reuse_checks+=1
    assert checks==2964 and reuse_checks==228 and len(archives)==54
    assert (result['compared_tensors'],result['exact_tensors'],result['exact_logit_calls'],result['max_logit_abs'],result['top1_matches'])==(checks,exact,logits_exact,maximum,top1)
    assert result['all_exact']==(checks==exact) and result['actual_onednn_calls']==spec['expected_total_onednn_calls']
    assert not {'torch','numpy','transformers'}&set(sys.modules)
    guard()
    review=dict(passed=True,complete=True,original_checkpoint_sha256=full.hexdigest(),original_tensors_rehashed=507,
        saved_archives=len(archives),saved_raw_bytes=sum(x['bytes']for x in archives.values()),
        forward_comparisons_reproduced=checks,exact_forward_comparisons=exact,exact_logit_calls=logits_exact,
        max_logit_abs=maximum,top1_matches=top1,top1_tokens=32,restart_comparisons_reproduced=reuse_checks,
        saved_restart_all_states_exact=True,all_payloads_finite=True,tokenizer_saved_cases_verified=13,
        tokenizer_BPE_reexecuted=False,model_or_GEMM_reexecuted=False,
        stale_handle_rejection_independently_reexecuted=False,archives=archives,
        elapsed_seconds=time.monotonic()-started,full_model_qualified=False,new_speedup_claim=False)
    path=Path(spec['review_output'])/'saved-review.json'
    with path.open('x')as f:
        json.dump(review,f,indent=2)
        f.write('\n')
    print(json.dumps({k:v for k,v in review.items()if k!='archives'}),flush=True)


if __name__=='__main__':
    main()
