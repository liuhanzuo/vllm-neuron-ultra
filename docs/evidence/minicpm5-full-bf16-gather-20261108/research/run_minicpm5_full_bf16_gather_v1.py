"""Bounded real TP2 full-context gather control; additive, no model execution."""
import argparse, hashlib, json, os, resource, signal, stat, subprocess, sys, time
from datetime import timedelta
from pathlib import Path
Q=Path(__file__).resolve().parent; R=Q.parent; sys.path[:0]=[str(Q),str(R)]
import run_minicpm5_frame1_fixed_oproj_v3 as old
from minicpm5_full_bf16_gather_v1 import full_bf16_projection
d=old.d; SELF=Path(__file__).resolve(); HELPER=Q/'minicpm5_full_bf16_gather_v1.py'
OUT=Path('/mnt/nvme/minicpm5-full-bf16-gather-r1-20261108-v1')
PLAN=Q/'minicpm5-full-bf16-gather-r1-20261108-v1-plan.json'
PREV=dict(path=str(Q/'minicpm5-frame1-fixed-oproj-r2-20261108-v1-root-terminal.json'),sha256='5d4d230354e9fc256e90d01b3a580a092dd6b4d529bab71bdc874f2e9b1c6a76')
EXIT=dict(path=str(Q/'minicpm5-frame1-fixed-oproj-r2-20261108-v1-finalizer-external-exit.json'),sha256='1a354ad07b6fbdd27f90f1f5fddd76e01b442fc29e56c5b14b5a75d3399886e9')
DESIGN=dict(path=str(Q/'minicpm5-layer0-full-bf16-next-design-20261108-v1.json'),sha256='4e5fc79112baba2c19d66c13f2fbd0052076d2e140c51828b9ab98f927c0a757')
BUDGET=dict(cpus=[6,7],address_space_per_rank=16*2**30,cpu_seconds_per_rank=300,wall_seconds=600,file_bytes=16*2**20,output_bytes=128*2**20,root_floor_bytes=28*2**30,nvme_floor_bytes=8*2**30,available_memory_floor_bytes=4*2**30)

def sources():
    return dict(old.sources(),**{str(p):d.ref(p)['sha256'] for p in (SELF,HELPER)})

def predecessor():
    t=d.load(PREV); e=d.load(EXIT); d.load(DESIGN)
    d.require(t['passed'] and t['all_saved_raw_files_authenticated'],'Complete predecessor')
    d.m.same(e['terminal'],PREV,'Actual terminal binding'); d.m.same(e['owner'],t['verifier'],'Actual verifier binding')
    d.require(e['actual_tool_exit_observed'] and e['exact_owner_gone'],'Actual external exit'); d.actual_exit(e['actual_exit'])
    for owner in (t['native_owner'],t['verifier']):d.m.p.gone(owner)
    result=d.load(t['result']); plan=d.load(t['plan']); d.m.same(plan['sources'],old.sources(),'Prior immutable sources')
    return t,result,plan

def usage():
    total=0
    for p in OUT.rglob('*'):
        s=p.lstat(); d.require(not p.is_symlink() and (stat.S_ISREG(s.st_mode) or stat.S_ISDIR(s.st_mode)),'Ordinary output tree')
        if stat.S_ISREG(s.st_mode):
            d.require(s.st_size<=BUDGET['file_bytes'],'Per-file bound');total+=max(s.st_size,s.st_blocks*512)
    d.require(total<=BUDGET['output_bytes'],'Aggregate output bound')
    for path,floor in [(R,BUDGET['root_floor_bytes']),('/mnt/nvme',BUDGET['nvme_floor_bytes'])]:
        fs=os.statvfs(path);d.require(fs.f_bavail*fs.f_frsize>=floor,'Retained filesystem floor')
    return total

def prepare():
    d.m.p.user_bus();fd,owner,lock=d.m.old.post.external.acquire_lock()
    try:
        t,result,prior=predecessor();d.m.same(lock,t['lock_identity'],'Same serial lock');d.m.p.clear_window();d.m.p.active_heavy()
        d.require(not OUT.exists() and not PLAN.exists(),'Fresh namespace');usage()
        d.once(PLAN,dict(schema='minicpm5_full_bf16_gather_plan_v1',source_sha256=sources(),predecessor=PREV,predecessor_exit=EXIT,design=DESIGN,
            original_plan=t['plan'],original_result=t['result'],preparation_owner=owner,lock_identity=lock,budgets=BUDGET,output_root=str(OUT),
            actual_group=dict(backend='gloo',global_ranks=[0,1],explicit_subgroup=True,init_method='unique file store'),
            cases=prior['cases'],controls=['complete original local partial','actual BF16 all_reduce versus saved split','actual BF16 all_gather rank order versus saved input','full BF16 projection versus saved full'],
            output_policy='Both ranks retain complete gathered input, full output and reduced split control for both full context cases.',
            remaining_gates=prior['remaining_gates'],model_execution=False,device_execution=False,full_model_qualified=False,new_speedup_claim=False))
        print(json.dumps(dict(plan=d.ref(PLAN),owner=owner)),flush=True)
    finally:d.m.old.post.external.release_lock(fd)

def bundle(plan_ref,review_ref):
    plan=d.load(plan_ref);review=d.load(review_ref)
    d.require(review['reviewer']=='/root' and review['passed'] and not review['blockers'],'Root source review')
    d.m.same(review['plan'],plan_ref,'Reviewed exact plan');d.m.same(plan['source_sha256'],sources(),'Frozen complete sources')
    d.m.same(plan['budgets'],BUDGET,'Resource bounds');d.m.same(plan['output_root'],str(OUT),'Fresh output binding')
    d.m.same([plan['predecessor'],plan['predecessor_exit'],plan['design']],[PREV,EXIT,DESIGN],'Pinned previous evidence')
    d.m.p.gone(plan['preparation_owner']);return plan

def isolation():
    d.require(sys.executable==str(old.PYTHON) and sys.flags.no_site and sys.flags.isolated and sys.dont_write_bytecode,'Isolated original SDK interpreter')
    d.require(not {'torch','numpy','torch_neuronx'}&set(sys.modules),'No framework import before authentication')
    os.sched_setaffinity(0,{6,7});os.nice(19)
    resource.setrlimit(resource.RLIMIT_AS,(BUDGET['address_space_per_rank'],)*2)
    resource.setrlimit(resource.RLIMIT_CPU,(BUDGET['cpu_seconds_per_rank'],BUDGET['cpu_seconds_per_rank']+5))
    resource.setrlimit(resource.RLIMIT_FSIZE,(BUDGET['file_bytes'],)*2)

def save(torch,folder,name,value):
    raw=old.raw(torch,value);path=folder/(name+'.raw');d.require(len(raw)<=BUDGET['file_bytes'],'Finite complete payload')
    with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),metadata=old.meta(torch,value),identity=old.identity(path))

def worker(plan_ref,review_ref,rank):
    plan=bundle(plan_ref,review_ref);isolation();ad=d.load(d.ref(OUT/'admission.json'))
    owner=d.m.p.proc(os.getpid());d.require(owner['ppid']==ad['owner']['pid'],'Actual admitted direct worker')
    lockfd=int(os.environ['MINI_GATHER_LOCK_FD']);s=os.fstat(lockfd);d.m.same(dict(device=s.st_dev,inode=s.st_ino),plan['lock_identity'],'Inherited original serial lock')
    folder=OUT/('rank'+str(rank));folder.mkdir();d.once(folder/'admission.json',dict(owner=owner,parent=ad['owner'],rank=rank,plan=plan_ref,review=review_ref))
    started=time.monotonic();last=[0.];barrier=None;initialized=False;group=None
    def guard():
        now=time.monotonic();d.require(now-started<540,'Worker wall bound')
        if now-last[0]>.5:
            d.m.p.lock_proof(ad['owner'],plan['lock_identity']);d.m.p.clear_window();usage()
            d.require(os.sched_getaffinity(0)=={6,7},'Worker CPU affinity');last[0]=now
            if barrier is not None:barrier.verify()
    def timeout(signum,frame):raise TimeoutError('Worker wall limit')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(540)
    try:
        prior=d.load(plan['original_plan']);result=d.load(plan['original_result'])
        identities,runtime=old.verify_dependencies(prior,guard);d.once(folder/'runtime.json',runtime)
        barrier=d.runtime.NoNativeChildren();sys.addaudithook(barrier.audit)
        sys.pycache_prefix=str(folder/'unused-bytecode');d.require(not Path(sys.pycache_prefix).exists(),'Fresh unused pycache prefix');sys.path.append(str(old.BASE))
        import torch
        d.m.same(str(torch.__version__),'2.8.0+cu128','Original SDK torch');d.require(Path(torch.__file__).resolve()==old.BASE/'torch/__init__.py','Original torch module')
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        torch.distributed.init_process_group('gloo',init_method='file://'+str(OUT/'gloo-store'),rank=rank,world_size=2,timeout=timedelta(seconds=120));initialized=True
        group=torch.distributed.new_group(ranks=[0,1],backend='gloo',timeout=timedelta(seconds=120))
        d.require(torch.distributed.get_process_group_ranks(group)==[0,1] and torch.distributed.get_rank(group)==rank,'Actual explicit rank-ordered subgroup')
        cases=[]
        with torch.inference_mode():
            weight=old.weight_tensor(torch,prior['weight'],guard);before_weight=old.meta(torch,weight)
            shard=weight[:,rank*1024:(rank+1)*1024].contiguous();d.m.same(old.meta(torch,shard)['sha256'],prior['weight']['shards'][rank]['sha256'],'Original column shard')
            for case in result['cases']:
                name=case['name'];torch.set_num_threads(case['threads']);guard()
                if name=='native_context':
                    full=old.restore(torch,prior['payloads']['native']['context_before_o'],guard)
                    local=full[...,rank*1024:(rank+1)*1024].contiguous();del full
                else:local=old.restore(torch,prior['payloads']['cpu-rank'+str(rank)]['context_before_o'],guard)
                local_before=old.meta(torch,local)
                partial=torch.einsum('...m,mn->...n',local,shard.t())
                d.m.same(old.meta(torch,partial),case['partials']['bf16_partial'][rank]['metadata'],'Complete original partial before actual reduce')
                # Ordinary buffers allow asynchronous collective completion under inference_mode.
                with torch.inference_mode(False):split=partial.clone()
                torch.distributed.all_reduce(split,group=group)
                d.m.same(old.meta(torch,split),case['captures']['bf16_split']['metadata'],'Actual BF16 TP2 sum matches saved complete split')
                output,gathered=full_bf16_projection(torch,local,weight,group)
                d.m.same(old.meta(torch,gathered),case['input'],'Actual gathered complete input and rank order')
                d.m.same(old.meta(torch,output),case['captures']['bf16_full']['metadata'],'Actual two-rank full BF16 projection')
                d.m.same(old.meta(torch,local),local_before,'Original local input unchanged');d.m.same(old.meta(torch,weight),before_weight,'Original weight unchanged')
                payloads={k:save(torch,folder,name+'-'+k,v) for k,v in [('input',gathered),('full',output),('split',split)]}
                cases.append(dict(name=name,threads=case['threads'],local_input=local_before,partial=old.meta(torch,partial),payloads=payloads,all_exact=True))
                torch.distributed.barrier(group=group);guard()
                print(json.dumps(dict(rank=rank,case=name,all_exact=True)),flush=True)
                del local,partial,split,output,gathered
        for path,identity in identities.items():d.m.same(old.identity(path),identity,'Dependency unchanged')
        d.m.same(plan['source_sha256'],sources(),'Frozen sources retained');guard();barrier.verify()
        torch.distributed.destroy_process_group(group);group=None;torch.distributed.destroy_process_group();initialized=False
        report=dict(schema='minicpm5_full_bf16_gather_rank_result_v1',complete=True,passed=True,rank=rank,owner=owner,parent=ad['owner'],plan=plan_ref,cases=cases,
            actual_backend='gloo',actual_global_rank_order=[0,1],explicit_subgroup=True,original_weight=before_weight,original_weight_unchanged=True,
            computation='all_gather BF16 local contexts, concatenate last dimension, one F.linear with unchanged full BF16 weight',
            projection_output_reduced=False,model_executed=False,device_executed=False,full_model_qualified=False,new_speedup_claim=False,
            measured_resources=dict(elapsed_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
        d.once(folder/'result.json',report)
    except BaseException as e:
        d.once(folder/'failure.json',dict(owner=owner,error_type=type(e).__name__,error=str(e)));raise
    finally:
        signal.alarm(0)
        if group is not None:torch.distributed.destroy_process_group(group)
        if initialized:torch.distributed.destroy_process_group()
        if barrier is not None:barrier.active=False
        os.close(lockfd)

def run(plan_ref,review_ref):
    plan=bundle(plan_ref,review_ref);isolation();d.m.p.user_bus();fd,owner,lock=d.m.old.post.external.acquire_lock()
    children=[];logs=[];started=time.monotonic()
    try:
        t,_,_=predecessor();d.m.same(lock,t['lock_identity'],'Same serial queue');d.m.p.clear_window();d.m.p.active_heavy();usage()
        d.require(not OUT.exists(),'Never restart or overwrite');OUT.mkdir(mode=0o700)
        admission=d.once(OUT/'admission.json',dict(owner=owner,lock_identity=lock,plan=plan_ref,review=review_ref,budgets=BUDGET))
        env=dict(os.environ,MINI_GATHER_LOCK_FD=str(fd),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES='',GLOO_SOCKET_IFNAME='lo')
        for rank in (0,1):
            log=(OUT/('rank'+str(rank)+'.log')).open('xb');logs.append(log)
            cmd=[str(old.PYTHON),'-I','-S','-B',str(SELF),'--mode','worker','--plan',plan_ref['path'],'--plan-sha256',plan_ref['sha256'],'--review',review_ref['path'],'--review-sha256',review_ref['sha256'],'--rank',str(rank)]
            child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(fd,));children.append(child)
        while any(child.poll() is None for child in children):
            d.require(time.monotonic()-started<BUDGET['wall_seconds'],'Complete experiment wall bound');d.m.p.lock_proof(owner,lock);d.m.p.clear_window();usage()
            available=int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])*1024
            d.require(available>=BUDGET['available_memory_floor_bytes'],'Host memory floor')
            d.require(all(c.returncode in (None,0) for c in children),'Worker failed; preserve outputs and drain peer');time.sleep(.25)
        d.require([c.wait() for c in children]==[0,0],'Both actual workers exited zero')
        refs=[d.ref(OUT/('rank'+str(rank))/'result.json') for rank in (0,1)];reports=[d.load(ref) for ref in refs]
        for rank,report in enumerate(reports):
            d.require(report['passed'] and report['complete'] and report['rank']==rank,'Both complete rank reports');d.m.p.gone(report['owner'])
        for i in range(2):
            for kind in ('input','full','split'):d.m.same(reports[0]['cases'][i]['payloads'][kind]['metadata'],reports[1]['cases'][i]['payloads'][kind]['metadata'],'Replicated complete rank values')
        d.m.same(plan['source_sha256'],sources(),'Sources unchanged');d.m.same({n:d.m.p.sha(R/n) for n in d.m.CANONICAL},d.m.CANONICAL,'Canonical unchanged')
        ref=d.once(OUT/'result.json',dict(schema='minicpm5_full_bf16_gather_result_v1',passed=True,complete=True,owner=owner,admission=admission,plan=plan_ref,
            ranks=refs,actual_child_exit_codes=[c.returncode for c in children],both_workers_gone=True,all4_case_rank_controls_exact=True,
            model_executed=False,device_executed=False,full_model_qualified=False,new_speedup_claim=False,actual_self_exit_claimed=False,
            measured_resources=dict(elapsed_seconds=time.monotonic()-started,output_bytes=usage())))
        print(json.dumps(dict(result=ref,owner=owner)),flush=True)
    except BaseException as e:
        if OUT.exists() and not (OUT/'failure.json').exists():d.once(OUT/'failure.json',dict(owner=owner,error_type=type(e).__name__,error=str(e)))
        raise
    finally:
        for c in children:
            if c.poll() is None:
                try:os.killpg(c.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        for c in children:
            try:c.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(c.pid,signal.SIGKILL);c.wait(timeout=5)
        for log in logs:log.close()
        d.m.old.post.external.release_lock(fd)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','run','worker'],required=True);p.add_argument('--plan');p.add_argument('--plan-sha256');p.add_argument('--review');p.add_argument('--review-sha256');p.add_argument('--rank',type=int,choices=[0,1]);a=p.parse_args()
    if a.mode=='prepare':prepare()
    else:
        refs=(dict(path=a.plan,sha256=a.plan_sha256),dict(path=a.review,sha256=a.review_sha256))
        if a.mode=='run':run(*refs)
        else:worker(*refs,a.rank)
