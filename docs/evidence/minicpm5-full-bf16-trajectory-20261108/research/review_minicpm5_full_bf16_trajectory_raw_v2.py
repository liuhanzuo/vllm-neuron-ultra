"""Independent saved FP32 logits review using restricted ZIP/pickle metadata.

No Torch/NumPy or model import. Metrics use Python float64 and math.fsum. This
checks saved endpoints, not an independently implemented model or GEMM.
"""
import argparse,array,collections,hashlib,io,json,math,os,pickle,resource,signal,sys,time,zipfile
from pathlib import Path
Q=Path(__file__).resolve().parent;R=Q.parent;sys.path.insert(0,str(R))
import minicpm5_original32_layer0_full_bf16_cpu_runner_v1 as runner
p=runner.p
# Declared before candidate results: accommodate independent float64 statistics
# versus original FP32 log_softmax/reductions, never loosen model thresholds.
TOLERANCE=dict(cosine=3e-6,kl_reference_to_neuron=2e-5,max_abs_error=2e-6,mean_abs_error=2e-6,reference_margin=2e-6)
V=130560

def rebuild(storage,offset,shape,stride,requires_grad,hooks,*extra):
    assert requires_grad is False and not hooks and not extra
    return dict(tensor=True,storage=storage,offset=offset,shape=list(shape),stride=list(stride))
class Reader(pickle.Unpickler):
    def find_class(self,module,name):
        allowed={('torch._utils','_rebuild_tensor_v2'):rebuild,('torch','FloatStorage'):'float32',('collections','OrderedDict'):collections.OrderedDict}
        assert (module,name) in allowed,(module,name);return allowed[module,name]
    def persistent_load(self,value):
        assert len(value)==5 and value[0]=='storage'
        _,dtype,key,location,count=value
        assert dtype=='float32' and location=='cpu' and isinstance(key,str) and key.isdecimal()
        assert type(count)is int and 0<count<=4*V
        return dict(key=key,count=count)
class Archive:
    def __init__(self,path,expected_hash,cap):
        self.path=Path(path);assert self.path.stat().st_size<=cap
        self.before=runner.identity(self.path);h=hashlib.sha256()
        with self.path.open('rb') as f:
            while chunk:=f.read(2**20):h.update(chunk)
        assert h.hexdigest()==expected_hash and runner.identity(self.path)==self.before
        self.z=zipfile.ZipFile(self.path);entries=self.z.infolist()
        assert len(entries)<128 and len(set(x.filename for x in entries))==len(entries)
        assert all(x.file_size<=cap and x.compress_size<=cap and x.compress_type==zipfile.ZIP_STORED for x in entries)
        assert sum(x.file_size for x in entries)<=cap
        names=[x.filename for x in entries if x.filename.endswith('/data.pkl')];assert len(names)==1
        name=names[0];assert self.z.getinfo(name).file_size<2**20;self.prefix=name.rsplit('/',1)[0]
        assert self.z.read(self.prefix+'/byteorder')==b'little' and sys.byteorder=='little'
        self.data=Reader(io.BytesIO(self.z.read(name))).load()
    def tensor(self,d):
        assert d['tensor'] and d['offset']==0 and d['storage']['count']==math.prod(d['shape'])
        stride=[];product=1
        for size in reversed(d['shape']):stride.insert(0,product);product*=size
        assert d['stride']==stride
        name=self.prefix+'/data/'+d['storage']['key'];size=4*product
        assert self.z.getinfo(name).file_size==size
        raw=self.z.read(name);assert len(raw)==size
        values=array.array('f');assert values.itemsize==4;values.frombytes(raw)
        assert all(math.isfinite(v) for v in values)
        return values,hashlib.sha256(raw).hexdigest()
    def close(self):
        self.z.close();assert runner.identity(self.path)==self.before

def metrics(got,ref):
    assert len(got)==len(ref)==V
    top_g=max(range(V),key=got.__getitem__);top_r=max(range(V),key=ref.__getitem__)
    hi_g=got[top_g];hi_r=ref[top_r];second=max(x for i,x in enumerate(ref) if i!=top_r)
    exp_r=[math.exp(x-hi_r) for x in ref];sum_r=math.fsum(exp_r)
    log_r=math.log(sum_r);log_g=math.log(math.fsum(math.exp(x-hi_g) for x in got))
    kl=math.fsum((ex/sum_r)*((a-hi_r-log_r)-(b-hi_g-log_g)) for ex,a,b in zip(exp_r,ref,got))
    norm_g=math.sqrt(math.fsum(x*x for x in got));norm_r=math.sqrt(math.fsum(x*x for x in ref));assert norm_g>0 and norm_r>0
    diffs=[abs(a-b) for a,b in zip(got,ref)]
    return dict(max_abs_error=max(diffs),mean_abs_error=math.fsum(diffs)/V,cosine=math.fsum(a*b for a,b in zip(got,ref))/(norm_g*norm_r),
                kl_reference_to_neuron=kl,argmax_match=top_g==top_r,reference_margin=hi_r-second,reference_top1=top_r,neuron_top1=top_g)
def failed(m):
    return [k for k,ok in [('cosine',m['cosine']>=.995),('KL',m['kl_reference_to_neuron']<=.02),('max_abs',m['max_abs_error']<=2),('high_margin_argmax',m['reference_margin']<=.5 or m['argmax_match'])] if not ok]

def run(exit_ref):
    p.user_bus();fd,owner,lock=runner.post.external.acquire_lock();started=time.monotonic()
    try:
        p.clear_window();p.active_heavy();ended=p.read_ref(exit_ref)
        assert ended['actual_exit_observed'] and ended['actual_exit_code']==0 and ended['all_four_processes_gone'] and ended['cgroup_empty']
        for old in [ended['owner'],ended['coordinator'],*ended['ranks']]:p.gone(old)
        spec=p.read_ref(ended['spec']);review=p.read_ref(ended['review']);plan=p.read_ref(spec['plan']);root=Path(spec['output_root'])
        assert review['passed'] and review['reviewer']=='/root' and review['spec']==ended['spec']
        assert plan['source_sha256']==spec['source_sha256'] and plan['output_root']==spec['output_root']
        for path,sha in spec['source_sha256'].items():assert p.sha(path)==sha
        assert ended['source_sha256']==spec['source_sha256']
        assert lock==ended['lock_identity'];assert ended['outputs']['outer_result']==runner.reference(root/'outer-result.json')
        os.sched_setaffinity(0,{6});os.nice(19);resource.setrlimit(resource.RLIMIT_AS,(2*2**30,)*2);resource.setrlimit(resource.RLIMIT_CPU,(600,605));resource.setrlimit(resource.RLIMIT_FSIZE,(2*2**20,)*2)
        def timeout(signum,frame):raise TimeoutError('Saved-logits verifier wall bound')
        signal.signal(signal.SIGALRM,timeout);signal.alarm(900)
        ranks=[p.read_ref(ended['outputs']['rank'+str(i)+'_result']) for i in (0,1)]
        controls=runner.validate_rank_pair(ranks,plan)
        native=Archive(plan['hf_reference']['path'],plan['hf_reference']['sha256'],32*2**20)
        assert len(native.data)==8;hf={};native_hashes={}
        for fixture,row in zip(plan['fixtures'],native.data):
            for k in ('id','input_ids','continuation_ids'):assert row[k]==fixture[k]
            assert len(row['logits'])==4
            for step,desc in enumerate(row['logits']):
                assert desc['shape']==[V];hf[row['id'],step],native_hashes[row['id'],step]=native.tensor(desc)
        native.close();raw_files=[];all_rows=[];arm_summary={};deviations={k:0. for k in TOLERANCE};tap_bytes=0
        for arm in runner.ARMS:
            computed=[]
            for ci in range(14):
                p.lock_proof(owner,lock);p.clear_window();runner.usage(root,spec['root_floor_bytes'])
                calls=[p.read(root/('cpu-rank'+str(rank))/arm/('call-%02d'%ci)/'result.json') for rank in (0,1)]
                arrays=[];hashes=[]
                for rank,call in enumerate(calls):
                    item=call['payload'];archive=Archive(item['path'],item['sha256'],3*2**20)
                    assert set(archive.data)=={'output'};desc=archive.data['output'];assert desc['shape']==call['output_metadata']['shape']
                    values,sha=archive.tensor(desc);archive.close();assert sha==call['output_metadata']['sha256']
                    arrays.append(values);hashes.append(sha);raw_files.append(item)
                    bundle=call['selected_tap_bundle'];data=Path(bundle['path']).read_bytes();assert len(data)==bundle['raw_bytes'] and hashlib.sha256(data).hexdigest()==bundle['sha256']
                    for entry in bundle['entries']:
                        raw=data[entry['offset']:entry['offset']+entry['raw_bytes']];assert hashlib.sha256(raw).hexdigest()==entry['sha256']
                    tap_bytes+=len(data)
                assert hashes[0]==hashes[1] and arrays[0]==arrays[1]
                for scored in calls[0]['HF_metrics']:
                    frame=scored['frame'];physical=scored['physical_row'];g=arrays[0][physical*V:(physical+1)*V];ref=hf[frame['fixture'],frame['step']]
                    m=metrics(g,ref);recorded=scored['metrics']
                    for k,tolerance in TOLERANCE.items():
                        delta=abs(m[k]-recorded[k]);assert delta<=tolerance,(arm,frame,k,delta,tolerance);deviations[k]=max(deviations[k],delta)
                    for k in ('argmax_match','reference_top1','neuron_top1'):assert m[k]==recorded[k]
                    bad=failed(m);assert bad==scored['failed_criteria'],'Independent float64 retains exact pass/fail classification'
                    for e in ranks[0]['native32_evidence']:
                        if e['frame']==frame:assert e['hf_reference_logits_sha256']==native_hashes[frame['fixture'],frame['step']]
                    computed.append(dict(frame=frame,metrics=m,failed_criteria=bad))
            assert [x['frame'] for x in computed]==plan['frames'];all_rows.append(dict(arm=arm,rows=computed))
            arm_summary[arm]=dict(passed_frames=sum(not x['failed_criteria'] for x in computed),top1_matches=sum(x['metrics']['argmax_match'] for x in computed),failed_frames=[x['frame']['index'] for x in computed if x['failed_criteria']],frame1=computed[1])
            print(json.dumps(dict(arm=arm,passed_frames=arm_summary[arm]['passed_frames'],top1_matches=arm_summary[arm]['top1_matches'])),flush=True)
        assert len(raw_files)==84 and tap_bytes==3145728
        assert not {'torch','numpy'}&set(sys.modules);assert {n:p.sha(R/n) for n in spec['canonical_sha256']}==spec['canonical_sha256']
        result=dict(schema='minicpm5_full_bf16_trajectory_raw_review_v1',passed=True,complete=True,reviewer='/root',verifier=owner,lock_identity=lock,actual_exit=exit_ref,
            verifier_source=runner.reference(Path(__file__).resolve()),plan=spec['plan'],metadata_controls=controls,summaries=arm_summary,rows=all_rows,
            logits_payloads=raw_files,logits_payload_count=84,selected_tap_bytes_checked=tap_bytes,HF_frames=32,candidate_metric_frames=96,
            metrics_independently_recomputed=True,independent_float64_tolerances=TOLERANCE,maximum_metric_deviations=deviations,
            exact_classification_and_argmax=True,model_reexecuted=False,matrix_multiplication_reexecuted=False,full_model_qualified=False,new_speedup_claim=False,
            measured_resources=dict(elapsed_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
        target=Q/'minicpm5-layer0-full-bf16-cpu-raw-terminal-r1-20261108-v1.json';p.once(target,result);print(json.dumps(dict(terminal=runner.reference(target),owner=owner)),flush=True)
    finally:signal.alarm(0);runner.post.external.release_lock(fd)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--exit',required=True);a.add_argument('--exit-sha256',required=True);a.add_argument('--source-sha256',required=True);args=a.parse_args();assert p.sha(Path(__file__).resolve())==args.source_sha256;run(dict(path=args.exit,sha256=args.exit_sha256))
