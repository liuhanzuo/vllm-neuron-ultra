"""Stdlib-only independent review of saved tiny Dory tensors, not model replay."""
import array
import collections
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import pickle
import resource
import signal
import struct
import sys
import time
import zipfile

Q=Path(__file__).resolve().parent;R=Q.parent
sys.path[:0]=[str(Q),str(R)]
import run_dory_weight_storage_root_v2 as d

BASE=Path('/mnt/nvme/dory-official-cpu-20261108-v1')
ROOT=BASE/'two-loop-r1/diagnostic'
OUT=BASE/'two-loop-r1-saved-review-v1.json'
CAP=2*2**20

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,j):
    encoded=json.dumps(j,indent=2)+'\n'
    with p.open('x')as f:f.write(encoded)

def rebuild(storage,offset,shape,stride,requires_grad,hooks,*extra):
    assert requires_grad is False and not hooks and not extra
    assert 1<=len(shape)<=3 and len(shape)==len(stride)
    assert all(type(x)is int and 0<x<=1024 for x in shape)
    assert all(type(x)is int and 0<=x<=65536 for x in stride)
    assert type(offset)is int and 0<=offset<=65536
    assert math.prod(shape)<=65536
    return dict(tensor=True,storage=storage,offset=offset,shape=list(shape),stride=list(stride))

class Reader(pickle.Unpickler):
    def find_class(self,module,name):
        allowed={('torch._utils','_rebuild_tensor_v2'):rebuild,
            ('torch','FloatStorage'):'float32',('torch','BFloat16Storage'):'bfloat16',
            ('collections','OrderedDict'):collections.OrderedDict}
        assert (module,name)in allowed,(module,name)
        return allowed[module,name]
    def persistent_load(self,value):
        assert len(value)==5 and value[0]=='storage'
        _,dtype,key,location,count=value
        assert dtype in ('float32','bfloat16') and location=='cpu'
        assert type(key)is str and key.isdecimal() and type(count)is int and 0<count<=65536
        return dict(key=key,count=count,dtype=dtype)

class Archive:
    def __init__(self,path):
        self.path=path;assert path.stat().st_size<=CAP
        self.hash=sha(path);self.z=zipfile.ZipFile(path)
        entries=self.z.infolist();assert len(entries)<256
        assert len({x.filename for x in entries})==len(entries)
        assert all(x.compress_type==zipfile.ZIP_STORED and x.file_size<=CAP for x in entries)
        assert sum(x.file_size for x in entries)<=CAP
        names=[x.filename for x in entries if x.filename.endswith('/data.pkl')];assert len(names)==1
        self.prefix=names[0].rsplit('/',1)[0]
        assert self.z.read(self.prefix+'/byteorder')==b'little' and sys.byteorder=='little'
        assert self.z.getinfo(names[0]).file_size<256*1024
        self.data=Reader(io.BytesIO(self.z.read(names[0]))).load()
        self.used=set()
    def tensor(self,t):
        assert t['tensor'];storage=t['storage'];width=4 if storage['dtype']=='float32' else 2
        name=self.prefix+'/data/'+storage['key'];raw=self.z.read(name)
        assert len(raw)==storage['count']*width
        assert t['offset']+sum((n-1)*s for n,s in zip(t['shape'],t['stride']))<storage['count']
        logical=b''.join(raw[index*width:(index+1)*width]
            for index in (t['offset']+sum(i*s for i,s in zip(ix,t['stride']))
                for ix in itertools.product(*[range(n)for n in t['shape']])))
        if width==4:
            values=array.array('f');assert values.itemsize==4;values.frombytes(logical)
        else:
            values=[struct.unpack('<f',struct.pack('<I',x[0]<<16))[0]for x in struct.iter_unpack('<H',logical)]
        assert all(math.isfinite(x)for x in values)
        self.used.add(name)
        return logical,values
    def close(self):
        self.z.close();assert sha(self.path)==self.hash

def comparison(a,b,archive):
    assert a['shape']==b['shape'] and a['storage']['dtype']==b['storage']['dtype']
    ar,av=archive.tensor(a);br,bv=archive.tensor(b)
    return dict(shape=a['shape'],dtype='torch.'+a['storage']['dtype'],exact_bytes=ar==br,
        unequal_values=sum(x!=y for x,y in zip(av,bv)),max_abs=max(abs(x-y)for x,y in zip(av,bv)),
        finite=True,official_sha256=hashlib.sha256(ar).hexdigest(),candidate_sha256=hashlib.sha256(br).hexdigest())

def main():
    fd,owner,lock=d.m.old.post.external.acquire_lock();started=time.monotonic()
    try:
        d.m.p.clear_window();d.m.p.active_heavy()
        assert 'torch'not in sys.modules and 'numpy'not in sys.modules
        resource.setrlimit(resource.RLIMIT_AS,(2**30,)*2)
        resource.setrlimit(resource.RLIMIT_CPU,(120,125))
        os.sched_setaffinity(0,{6});os.nice(19)
        def timeout(*_):raise TimeoutError('Saved-byte review wall limit')
        signal.signal(signal.SIGALRM,timeout);signal.alarm(120)
        ended=json.loads((BASE/'two-loop-r1/result.json').read_text())
        d.m.p.gone(ended['owner']);assert ended['complete']
        expected=json.loads((ROOT/'result.json').read_text());rows=[];files=[];comparisons=0
        for dtype_result in expected['dtypes']:
            label=dtype_result['dtype'];folder=ROOT/label
            step_rows=[]
            for step in dtype_result['steps']:
                d.m.p.lock_proof(owner,lock)
                p=folder/f"output-{step['start']:03d}.pt";ar=Archive(p);obj=ar.data
                h=comparison(obj['hidden'],obj['candidate_hidden'],ar)
                l=comparison(obj['logits'],obj['candidate_logits'],ar)
                assert h==step['hidden'] and l==step['logits']
                cache_rows=[]
                for cache in step['caches']:
                    key=cache['cache_id'];left=obj['official_caches'][key];right=obj['candidate_caches'][key]
                    for i,kind in enumerate(['key','value']):
                        found=comparison(left[i],right[i],ar);assert found==cache[kind]
                        cache_rows.append(found)
                        comparisons+=1
                comparisons+=2
                step_rows.append(dict(start=step['start'],logits_exact=l['exact_bytes'],hidden_exact=h['exact_bytes'],
                    cache_exact=all(c['exact_bytes']for c in cache_rows),max_logits_error=l['max_abs']))
                ar.close();files.append(dict(path=str(p),sha256=ar.hash,bytes=p.stat().st_size))
            for primitive in dtype_result['primitives']:
                p=folder/f"primitive-{primitive['projection']}-{primitive['batch']}.pt";ar=Archive(p);obj=ar.data
                ar.tensor(obj['input'])
                for name,left,right in [('official_vs_packed','official','packed'),('official_vs_split','official','split'),('packed_vs_split','packed','split')]:
                    assert comparison(obj[left],obj[right],ar)==primitive[name];comparisons+=1
                ar.close();files.append(dict(path=str(p),sha256=ar.hash,bytes=p.stat().st_size))
            p=folder/'synthetic-weights.pt';ar=Archive(p)
            for tensor in ar.data.values():ar.tensor(tensor)
            assert len(ar.used)==len(ar.data)
            ar.close();files.append(dict(path=str(p),sha256=ar.hash,bytes=p.stat().st_size))
            rows.append(dict(dtype=label,steps=len(step_rows),logits_exact_steps=sum(s['logits_exact']for s in step_rows),
                hidden_exact_steps=sum(s['hidden_exact']for s in step_rows),all_caches_exact_steps=sum(s['cache_exact']for s in step_rows),
                maximum_logits_absolute_error=max(s['max_logits_error']for s in step_rows)))
        assert len(files)==66 and comparisons==808
        assert 'torch'not in sys.modules and 'numpy'not in sys.modules
        d.m.same({n:d.m.p.sha(R/n)for n in d.m.CANONICAL},d.m.CANONICAL,'Canonical unchanged')
        result=dict(complete=True,passed=True,owner=owner,elapsed_seconds=time.monotonic()-started,
            producer_result=dict(path=str(ROOT/'result.json'),sha256=sha(ROOT/'result.json')),
            comparisons=comparisons,files=files,raw_archive_bytes=sum(f['bytes']for f in files),rows=rows,
            all_producer_saved_comparison_dicts_exact=True,producer_all_dtype_exact_gate=expected['all_exact'],
            model_reexecuted=False,matmul_reexecuted=False,torch_or_numpy_imported=False,
            slot_reuse_payloads_independently_reverified=False,full_model_qualified=False,new_speedup_claim=False)
        save(OUT,result);print(json.dumps(dict(result=str(OUT),rows=rows,comparisons=comparisons)),flush=True)
    finally:
        signal.alarm(0);d.m.old.post.external.release_lock(fd)

if __name__=='__main__':main()
