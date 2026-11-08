"""Replay original FP32 endpoint metrics with producer strides; no model run."""
import argparse,ast,hashlib,json,os,resource,signal,sys,time
from pathlib import Path
Q=Path(__file__).resolve().parent;R=Q.parent;sys.path[:0]=[str(Q),str(R)]
import run_minicpm5_frame1_fixed_oproj_v3 as old
import review_minicpm5_full_bf16_trajectory_raw_v4 as raw
D=old.d;SELF=Path(__file__).resolve();OUT=Path('/mnt/nvme/minicpm5-full-bf16-metric-replay-r1-20261108-v1')
NATIVE_EXIT=dict(path=str(Q/'minicpm5-layer0-full-bf16-cpu-root-exit-r1-20261108-v1.json'),sha256='727d92a3189bc671016701e53bfc982547c33917cb046dfd2ca29eb78f84e0ff')
FLOAT_EXIT=dict(path=str(Q/'minicpm5-full-bf16-float64-analysis-actual-exit-20261108-v1.json'),sha256='854eafc8eb19425cd02632a841f0f3772716a9809e4e04d743538976a52bcd03')
SDK_PLAN=dict(path=str(Q/'minicpm5-frame1-fixed-oproj-r2-20261108-v1-plan.json'),sha256='b636246604b1c8df2fd8fbc74aa7b74efd1313784dce7f5cf0cc52a30c0279a1')
BUDGET=dict(cpus=[6],address_space_bytes=16*2**30,cpu_seconds=300,wall_seconds=600,file_bytes=2*2**20,output_bytes=8*2**20,root_floor_bytes=28*2**30,nvme_floor_bytes=8*2**30)

def sources():
    spec=raw.p.read_ref(raw.p.read_ref(NATIVE_EXIT)['spec'])
    return dict(old.sources(),**spec['source_sha256'],**{str(x):raw.p.sha(x) for x in [SELF,Path(raw.__file__)]})

def run(review_ref):
    review=D.load(review_ref);assert review['reviewer']=='/root' and review['passed'] and review['sources']==sources() and review['budgets']==BUDGET
    assert sys.executable==str(old.PYTHON) and sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode
    assert not {'torch','numpy','torch_neuronx'}&set(sys.modules)
    D.m.p.user_bus();fd,owner,lock=D.m.old.post.external.acquire_lock();started=time.monotonic();barrier=None
    try:
        D.m.p.clear_window();D.m.p.active_heavy();ne=D.load(NATIVE_EXIT);fe=D.load(FLOAT_EXIT);analysis=D.load(fe['result'])
        assert ne['actual_exit_code']==0 and fe['actual_tool_exit_observed'] and fe['exact_owner_gone'];D.actual_exit(fe['actual_exit'])
        for previous in [ne['owner'],ne['coordinator'],*ne['ranks'],fe['owner']]:D.m.p.gone(previous)
        assert lock==ne['lock_identity'] and not OUT.exists();OUT.mkdir(mode=0o700)
        admission=D.once(OUT/'admission.json',dict(owner=owner,lock_identity=lock,review=review_ref,native_exit=NATIVE_EXIT,float64_exit=FLOAT_EXIT,budgets=BUDGET))
        os.sched_setaffinity(0,{6});os.nice(19);resource.setrlimit(resource.RLIMIT_AS,(BUDGET['address_space_bytes'],)*2);resource.setrlimit(resource.RLIMIT_CPU,(300,305));resource.setrlimit(resource.RLIMIT_FSIZE,(BUDGET['file_bytes'],)*2)
        def timeout(signum,frame):raise TimeoutError('Metric replay wall bound')
        signal.signal(signal.SIGALRM,timeout);signal.alarm(600);last=[0.]
        def guard():
            now=time.monotonic();assert now-started<600
            if now-last[0]>.5:
                D.m.p.lock_proof(owner,lock);D.m.p.clear_window()
                for path,floor in [(R,BUDGET['root_floor_bytes']),('/mnt/nvme',BUDGET['nvme_floor_bytes'])]:
                    fs=os.statvfs(path);assert fs.f_bavail*fs.f_frsize>=floor
                assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<=BUDGET['output_bytes']
                if barrier is not None:barrier.verify()
                last[0]=now
        identities,runtime=old.verify_dependencies(D.load(SDK_PLAN),guard);D.once(OUT/'runtime.json',runtime)
        os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',PYTHONNOUSERSITE='1')
        barrier=D.runtime.NoNativeChildren();sys.addaudithook(barrier.audit);sys.pycache_prefix=str(OUT/'unused-bytecode');assert not Path(sys.pycache_prefix).exists();sys.path.append(str(old.BASE))
        import torch
        assert str(torch.__version__)=='2.8.0+cu128' and Path(torch.__file__).resolve()==old.BASE/'torch/__init__.py'
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        spec=raw.p.read_ref(ne['spec']);plan=raw.p.read_ref(spec['plan']);root=Path(spec['output_root'])
        metric_path=R/'validate_real_logits.py';source=metric_path.read_bytes();assert hashlib.sha256(source).hexdigest()==spec['source_sha256'][str(metric_path)]
        funcs=[x for x in ast.parse(source).body if isinstance(x,ast.FunctionDef) and x.name=='metrics'];assert len(funcs)==1
        scope=dict(torch=torch);exec(compile(ast.Module(body=funcs,type_ignores=[]),str(metric_path),'exec'),scope);metric=scope['metrics']
        hf=[];native=raw.Archive(plan['hf_reference']['path'],plan['hf_reference']['sha256'],32*2**20)
        for row in native.data:
            values,_=native.tensor(row['logits']);hf.append(torch.frombuffer(values,dtype=torch.float32).reshape(4,raw.V).clone())
        native.close();rows=[];disagreements=[];layout_differences=[]
        with torch.inference_mode():
            for arm in raw.runner.ARMS:
                for ci in range(14):
                    guard();call=raw.p.read(root/'cpu-rank0'/arm/('call-%02d'%ci)/'result.json');item=call['payload']
                    archive=raw.Archive(item['path'],item['sha256'],3*2**20);values,sha=archive.tensor(archive.data['output']);archive.close();assert sha==call['output_metadata']['sha256']
                    md=call['output_metadata'];assert md['offset']==0
                    packed=torch.frombuffer(values,dtype=torch.float32).reshape(md['shape']);original=torch.empty_strided(md['shape'],md['stride'],dtype=torch.float32);original.copy_(packed)
                    assert old.meta(torch,original)==md
                    for scored in call['HF_metrics']:
                        frame=scored['frame'];got=original[scored['physical_row'],0];ref=hf[frame['fixture']][frame['step']]
                        reproduced=metric(got,ref);contiguous=metric(got.contiguous(),ref)
                        delta={k:reproduced[k]-scored['metrics'][k] for k in reproduced if type(reproduced[k])is float}
                        exact=reproduced==scored['metrics']
                        if not exact:disagreements.append(dict(arm=arm,frame=frame['index'],delta=delta,actual=reproduced,recorded=scored['metrics']))
                        if contiguous!=reproduced:layout_differences.append(dict(arm=arm,frame=frame['index'],producer_stride=list(got.stride()),producer=reproduced,contiguous=contiguous))
                        rows.append(dict(arm=arm,frame=frame['index'],producer_stride=list(got.stride()),reproduced_exact=exact,original=reproduced,contiguous=contiguous))
                    del original,packed,values
                print(json.dumps(dict(arm=arm,replayed=True)),flush=True)
        assert len(rows)==96
        for path,identity in identities.items():assert old.identity(path)==identity
        assert review['sources']==sources();guard();barrier.verify()
        result=D.once(OUT/'result.json',dict(schema='minicpm5_full_bf16_original_metric_replay_v1',complete=True,passed=not disagreements,owner=owner,admission=admission,review=review_ref,
            actual_original_metric_source=raw.runner.reference(metric_path),full_original_layout_restored=True,frames=96,rows=rows,all96_original_metrics_exact=not disagreements,original_metric_disagreements=disagreements,layout_differences=layout_differences,
            original_tensor_values_unchanged=True,model_executed=False,device_executed=False,new_speedup_claim=False,full_model_qualified=False,
            retained_float64_scalar_agreement_passed=analysis['original_metric_tolerance_gate_passed'],float64_classification_matches=analysis['exact_classification_and_argmax'],
            measured_resources=dict(elapsed_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)))
        print(json.dumps(dict(result=result,owner=owner,all96_original_metrics_exact=not disagreements,layout_different_frames=len(layout_differences))),flush=True)
        assert not disagreements,'Original producer-stride metrics failed exact reproduction'
    except BaseException as e:
        if OUT.exists() and not (OUT/'failure.json').exists():D.once(OUT/'failure.json',dict(owner=owner,error_type=type(e).__name__,error=str(e)))
        raise
    finally:
        signal.alarm(0)
        if barrier is not None:barrier.active=False
        D.m.old.post.external.release_lock(fd)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--review',required=True);p.add_argument('--review-sha256',required=True);a=p.parse_args();run(dict(path=a.review,sha256=a.review_sha256))
