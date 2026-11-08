"""Original full-context projection controls and diagnostic precision contrasts.

Single SDK Torch process, no Gloo/device/model run. Original native endpoint is
reproduced as a byte gate in this interpreter, not claimed as native runtime reuse.
"""
import argparse,ast,base64,csv,hashlib,json,math,os,resource,signal,stat,struct,sys,time
from pathlib import Path
Q=Path(__file__).resolve().parent;R=Q.parent;sys.path[:0]=[str(Q),str(R)]
import run_dory_weight_storage_root_v2 as d
from minicpm5_layer0_runtime_auth_v1 import identity
SELF=Path(__file__).resolve();SDK=Path('/home/ubuntu/venvs/vllm_neuron');BASE=SDK/'lib/python3.10/site-packages';PYTHON=SDK/'bin/python'
OUT=R/'minicpm5-frame1-fixed-oproj-r1-20261108-v1';PLAN=Q/'minicpm5-frame1-fixed-oproj-r1-20261108-v1-plan.json'
PREDECESSOR=dict(path=str(Q/'dory-tokenizer-template-r1-20261108-v1-root-terminal.json'),sha256='7e017212ad9db6b0bf43f495a34352d445b8af4cc178566ef327870c7cf3b1cd')
PREDECESSOR_EXIT=dict(path=str(Q/'dory-tokenizer-template-r1-20261108-v1-finalizer-external-exit.json'),sha256='d0054f7d4fb66757b8b3138c9b505c3d07b0127f877ededa487701a631dc3ed9')
SAVED=dict(path=str(Q/'minicpm5-frame1-attention-r3-20261108-v1-recovered-review-result.json'),sha256='31faef21b1173b12a1b722e86f0d6752d23bdfd9395e97207a11991b6cbc2fc6')
SAVED_EXIT=dict(path=str(Q/'minicpm5-attention-r3-recovered-review-external-exit-v1.json'),sha256='1d988633e36d69beee968853a8fc56dc33944925c9c4de16b8ff4e88ba85105d')
BUDGET=dict(cpus=[6,7],address_space_bytes=16*2**30,cpu_seconds=900,wall_seconds=1200,file_bytes=16*2**20,output_bytes=256*2**20,root_floor_bytes=28*2**30,available_memory_floor_bytes=4*2**30)
FORMULAS=('bf16_full','bf16_split','bf16_split_wide_sum','fp32_full_final','fp32_split_final','fp32_split_round_local')


def sources():
    result=d.sources();result[str(SELF)]=d.ref(SELF)['sha256'];result[str(R/'minicpm5_layer0_runtime_auth_v1.py')]=d.ref(R/'minicpm5_layer0_runtime_auth_v1.py')['sha256'];return result


def predecessors():
    t=d.load(PREDECESSOR);e=d.load(PREDECESSOR_EXIT);s=d.load(SAVED);se=d.load(SAVED_EXIT)
    d.m.same((e['terminal'],e['owner']),(PREDECESSOR,t['verifier']),'Actual latest terminal/exit')
    d.require(t['passed'] and e['actual_tool_exit_observed'] and e['exact_owner_gone'],'Latest serial work completed');d.actual_exit(e['actual_exit'])
    for owner in (t['verifier'],t['native_owner']):d.m.p.gone(owner)
    d.require(s['passed'] and s['all96_full_payloads_authenticated'] and s['all21_own_input_controls_exact'],'Complete saved evidence before new intervention')
    run=d.load(se['run']);d.m.same(run['result'],SAVED,'Saved-byte audit result binding');d.m.same(se['owner'],run['owner'],'Saved-byte audit owner');d.m.p.gone(run['owner'])
    d.require(se['actual_tool_exit_observed'] and se['exact_owner_gone'],'Actual saved review exit');d.actual_exit(se['actual_exit'])
    return t,s


def runtime_paths():
    paths={PYTHON.resolve(),SDK/'pyvenv.cfg'}
    for folder in [Path('/usr/lib/python3.10')]:
        for base,dirs,files in os.walk(folder):
            dirs[:]=[x for x in dirs if x not in ('site-packages','__pycache__','dist-packages')]
            for name in files:
                p=Path(base)/name
                if p.suffix in ('.py','.so','.zip'):paths.add(p.resolve())
    paths.update(p.resolve()for p in Path('/usr/lib/x86_64-linux-gnu').glob('libpython3.10.so*'))
    return sorted(paths)


def prepare():
    d.m.p.user_bus();fd,owner,lock=d.m.old.post.external.acquire_lock()
    try:
        terminal,saved=predecessors();d.m.same(lock,terminal['lock_identity'],'Same serial queue');d.m.p.clear_window();d.m.p.active_heavy()
        d.require(not PLAN.exists() and not OUT.exists(),'Fresh plan namespace')
        captures={role:d.load(ref) for role,ref in saved['captures'].items()};rows={}
        for role,report in captures.items():
            d.require(report['complete'] and report['own_input_controls_passed'],'Actual original capture/control')
            rows[role]={x['name']:x for x in report['attention']['captures']if x['name']in ('context_before_o','projection_output')}
            for value in rows[role].values():d.m.same(identity(value['path']),value['identity'],'Original saved payload identity')
        weight=captures['native']['weights']['file'];header=captures['native']['weights']['header'];tensor=d.load(header)['model.layers.0.self_attn.o_proj.weight']
        d.m.same((tensor['shape'],tensor['dtype']),([2048,2048],'BF16'),'Original full output weight')
        d.m.same(identity(weight['path']),weight['identity'],'Original full checkpoint identity')
        prefixes=('torch','numpy','filelock','typing_extensions','sympy','networkx','jinja2','fsspec','mpmath','markupsafe','setuptools')
        record_paths=[]
        for prefix in prefixes:
            matches=list(BASE.glob(prefix+'-*.dist-info/RECORD'));d.require(len(matches)==1,'One actual dependency '+prefix);record_paths.extend(matches)
        record_paths.extend(sorted(BASE.glob('nvidia_*.dist-info/RECORD')))
        records={str(p):d.ref(p)['sha256']for p in record_paths}
        runtime=[]
        for path in runtime_paths():runtime.append(dict(path=str(path),sha256=d.ref(path)['sha256'],identity=identity(path)))
        base_ref=saved['base_plan'];base_path=Path(base_ref['path']);d.require(base_path.stat().st_size==4087706 and d.ref(base_path)==base_ref,'Exact unchanged original4MiB plan');base_plan=json.loads(base_path.read_text())
        plan=dict(schema='minicpm5_frame1_fixed_oproj_plan_v1',sources=sources(),predecessor=PREDECESSOR,predecessor_exit=PREDECESSOR_EXIT,saved_review=SAVED,saved_review_exit=SAVED_EXIT,
                  preparation_owner=owner,lock_identity=lock,budgets=BUDGET,python=str(PYTHON),records=records,runtime=runtime,captures=saved['captures'],payloads=rows,
                  weight=dict(file=weight,header=header,name='model.layers.0.self_attn.o_proj.weight',tensor=tensor,sha256=captures['native']['official_head_weights']['weights']['o']['metadata']['sha256'],
                              shards=[captures['cpu-rank'+str(rank)]['official_head_weights']['weights']['o_proj']for rank in (0,1)]),
                  formulas=list(FORMULAS),cases=[dict(name='native_context',length=1231,threads=2),dict(name='cpu_context',length=2048,threads=1)],real_tokens=1231,prediction_index=1230,
                  controls='Native1231 full F.linear and CPU2048 split einsum/BF16 sum must exactly match all original shape/stride/offset/dtype/bytes before contrasts.',
                  constraints=dict(full_context_before_metrics=True,no_changed_weights=True,no_device=True,no_model=True,no_gloo=True,no_whole_model_or_speedup_claim=True,native_interpreter_reused=False),
                  original_base_plan=base_ref,remaining_gates=base_plan['remaining_gates'])
        print(json.dumps(dict(plan=d.once(PLAN,plan),owner=owner)),flush=True)
    finally:d.m.old.post.external.release_lock(fd)


def usage():
    total=0
    for path in OUT.rglob('*'):
        s=path.lstat();d.require(not path.is_symlink() and (stat.S_ISREG(s.st_mode)or stat.S_ISDIR(s.st_mode)),'Ordinary output tree')
        if stat.S_ISREG(s.st_mode):total+=max(s.st_size,s.st_blocks*512);d.require(s.st_size<=BUDGET['file_bytes'],'Bounded output file')
    d.require(total<=BUDGET['output_bytes'],'Bounded output aggregate')
    fs=os.statvfs(R);d.require(fs.f_bavail*fs.f_frsize>=BUDGET['root_floor_bytes'],'Original root free floor');return total


def binary(path,sha,size,guard):
    p=Path(path);before=identity(p);d.require(before['size']==size,'Expected file extent');h=hashlib.sha256();count=0
    with p.open('rb')as f:
        while chunk:=f.read(2**20):h.update(chunk);count+=len(chunk);guard()
    d.require(count==size and h.hexdigest()==sha and identity(p)==before,'Actual complete pinned bytes: '+str(p));return before


def verify_dependencies(plan,guard):
    total=0;count=0;identities={}
    for name,wanted in plan['records'].items():
        path=Path(name);d.m.same(d.ref(path)['sha256'],wanted,'Pinned dependency RECORD')
        for relative,digest,size in csv.reader(path.open()):
            p=(BASE/relative).resolve();d.require(p.is_relative_to(SDK),'Dependency inside existing SDK environment')
            if not digest:
                d.require(relative.endswith(('.pyc','/RECORD')),'Known omitted RECORD digest');continue
            d.require(digest.startswith('sha256=') and size.isdigit(),'Dependency SHA256/length')
            sha=base64.urlsafe_b64decode(digest[7:]+'=').hex();total+=int(size);d.require(total<8*2**30,'Bounded SDK closure bytes')
            identities[str(p)]=binary(p,sha,int(size),guard);count+=1
    for row in plan['runtime']:
        d.m.same(identity(row['path']),row['identity'],'Pinned interpreter/stdlib identity');binary(row['path'],row['sha256'],row['identity']['size'],guard)
    return identities,dict(installed_files=count,installed_bytes=total,records=plan['records'],interpreter_files=len(plan['runtime']))


def raw(torch,x):return x.detach().contiguous().view(torch.uint8).numpy().tobytes()
def meta(torch,x):return dict(shape=list(x.shape),stride=list(x.stride()),offset=x.storage_offset(),dtype=str(x.dtype),sha256=hashlib.sha256(raw(torch,x)).hexdigest())
def restore(torch,row,guard):
    d.m.same(identity(row['path']),row['identity'],'Actual capture identity');binary(row['path'],row['sha256'],row['bytes'],guard)
    m=row['metadata'];d.m.same(m['dtype'],'torch.bfloat16','Original BF16 input');d.require(m['offset']==0,'Original zero storage offset')
    packed=torch.frombuffer(bytearray(Path(row['path']).read_bytes()),dtype=torch.bfloat16).reshape(m['shape'])
    value=torch.empty_strided(m['shape'],m['stride'],dtype=torch.bfloat16);value.copy_(packed);d.m.same(meta(torch,value),m,'Original full layout and bytes');return value

def save(torch,name,value):
    path=OUT/(name+'.raw');payload=raw(torch,value);d.require(len(payload)<=BUDGET['file_bytes'],'Bounded saved tensor')
    with path.open('xb')as f:f.write(payload);f.flush();os.fsync(f.fileno())
    return dict(path=str(path),bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest(),metadata=meta(torch,value),encoding='packed_logical_little_endian')

def weight_tensor(torch,spec,guard):
    f=spec['file'];d.m.same(identity(f['path']),f['identity'],'Original checkpoint identity');fd=os.open(f['path'],os.O_RDONLY|os.O_NOFOLLOW)
    try:
        opened=os.fstat(fd);d.m.same([opened.st_dev,opened.st_ino,opened.st_size,opened.st_mtime_ns,opened.st_ctime_ns],[f['identity'][k]for k in ('device','inode','size','mtime_ns','ctime_ns')],'Opened original checkpoint')
        hlen,=struct.unpack('<Q',os.pread(fd,8,0));d.require(0<hlen<2**20,'Bounded official header')
        header=json.loads(os.pread(fd,hlen,8));d.m.same(header,d.load(spec['header']),'Official header');d.m.same(header[spec['name']],spec['tensor'],'Official matrix interval')
        begin,end=spec['tensor']['data_offsets'];d.require(end-begin==8*2**20 and 0<=begin<end<=opened.st_size-8-hlen,'Exact8MiB matrix')
        data=os.pread(fd,end-begin,8+hlen+begin);d.require(len(data)==end-begin and hashlib.sha256(data).hexdigest()==spec['sha256'],'Actual full original output matrix bytes');after=os.fstat(fd);d.m.same([after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns],[opened.st_dev,opened.st_ino,opened.st_size,opened.st_mtime_ns,opened.st_ctime_ns],'Unchanged opened checkpoint')
    finally:os.close(fd)
    d.m.same(identity(f['path']),f['identity'],'Unchanged original checkpoint');guard()
    return torch.frombuffer(bytearray(data),dtype=torch.bfloat16).reshape(2048,2048)

def measure(torch,a,b):
    d.m.same(list(a.shape),list(b.shape),'Same metric shape');d.require(a.dtype==b.dtype==torch.bfloat16,'FinalBF16 comparisons')
    delta=(a.double()-b.double()).abs();d.require(bool(torch.isfinite(delta).all()),'Finite comparison')
    return dict(compared_elements=a.numel(),different_elements=int((a.contiguous().view(torch.int16)!=b.contiguous().view(torch.int16)).sum()),max_abs=float(delta.max()),mean_abs=float(delta.mean()))

def scopes(torch,a,b):return dict(full=measure(torch,a,b),real_tokens=measure(torch,a[:,:1231],b[:,:1231]),prediction=measure(torch,a[:,1230:1231],b[:,1230:1231]))

def formulas(torch,x,weight,shards):
    parts=[x[...,i*1024:(i+1)*1024].contiguous()for i in (0,1)]
    bf=[torch.einsum('...m,mn->...n',part,shard.t())for part,shard in zip(parts,shards)]
    fp=[torch.einsum('...m,mn->...n',part.float(),shard.float().t())for part,shard in zip(parts,shards)]
    results=dict(bf16_full=torch.nn.functional.linear(x,weight),bf16_split=bf[0]+bf[1],bf16_split_wide_sum=(bf[0].float()+bf[1].float()).bfloat16(),fp32_full_final=torch.nn.functional.linear(x.float(),weight.float()).bfloat16(),fp32_split_final=(fp[0]+fp[1]).bfloat16(),fp32_split_round_local=(fp[0].bfloat16().float()+fp[1].bfloat16().float()).bfloat16())
    d.m.same(tuple(results),FORMULAS,'All6 unchanged formulas');return results,bf,fp


def run(plan_ref,review_ref):
    plan=d.load(plan_ref);review=d.load(review_ref);d.require(review['passed'] and review['reviewer']=='/root' and review['plan']==plan_ref,'Root reviewed exact scientific plan')
    d.m.same(plan['sources'],sources(),'Frozen local sources');d.m.same(plan['budgets'],BUDGET,'Frozen resource bounds')
    d.require(sys.executable==str(PYTHON) and sys.flags.no_site and sys.flags.isolated and sys.dont_write_bytecode,'Original isolated SDK interpreter')
    d.require(not{'torch','numpy','torch_neuronx'}&set(sys.modules),'Authenticate before framework import')
    d.m.p.user_bus();fd,owner,lock=d.m.old.post.external.acquire_lock();start=time.monotonic();barrier=None;created=False
    try:
        terminal,saved=predecessors();d.m.same(lock,terminal['lock_identity'],'Original serial lock');d.m.p.gone(plan['preparation_owner']);d.m.p.clear_window();d.m.p.active_heavy()
        d.require(not OUT.exists(),'Never restart or overwrite experiment');os.sched_setaffinity(0,{6,7});os.nice(19)
        resource.setrlimit(resource.RLIMIT_AS,(BUDGET['address_space_bytes'],)*2);resource.setrlimit(resource.RLIMIT_CPU,(BUDGET['cpu_seconds'],BUDGET['cpu_seconds']+5));resource.setrlimit(resource.RLIMIT_FSIZE,(BUDGET['file_bytes'],)*2)
        def timeout(signum,frame):raise TimeoutError('Fixed-context projection wall budget')
        signal.signal(signal.SIGALRM,timeout);signal.alarm(BUDGET['wall_seconds']);OUT.mkdir(mode=0o700);created=True;last=[0.]
        admission=d.once(OUT/'admission.json',dict(owner=owner,lock_identity=lock,plan=plan_ref,review=review_ref,budgets=BUDGET))
        def guard():
            now=time.monotonic();d.require(now-start<BUDGET['wall_seconds'],'Bounded full-context replay')
            if now-last[0]>.5:
                d.m.p.lock_proof(owner,lock);d.m.p.clear_window();usage();d.require(os.sched_getaffinity(0)=={6,7},'Original bounded affinity')
                available=int(next(x for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')).split()[1])*1024;d.require(available>=BUDGET['available_memory_floor_bytes'],'Host memory headroom')
                if barrier is not None:barrier.verify()
                last[0]=now
        identities,runtime=verify_dependencies(plan,guard);d.once(OUT/'runtime-summary.json',runtime)
        os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES='')
        barrier=d.runtime.NoNativeChildren();sys.addaudithook(barrier.audit);sys.path.append(str(BASE))
        import torch
        d.m.same(str(torch.__version__),'2.8.0+cu128','Original SDK Torch version');d.require(Path(torch.__file__).resolve()==BASE/'torch/__init__.py','Original Torch source path');torch.set_num_threads(1);torch.set_num_interop_threads(1)
        d.require(not torch.distributed.is_initialized(),'No distributed process group')
        with torch.inference_mode():
            weight=weight_tensor(torch,plan['weight'],guard);shards=[weight[:,i*1024:(i+1)*1024].contiguous()for i in (0,1)]
            for i,shard in enumerate(shards):d.m.same(meta(torch,shard)['sha256'],plan['weight']['shards'][i]['sha256'],'Actual original TP2 column shard')
            d.require(bool(torch.isfinite(weight).all()),'Finite original weight');weight_saved=save(torch,'original_weight',weight)
            observed={role:{field:restore(torch,row,guard)for field,row in rows.items()}for role,rows in plan['payloads'].items()}
            native=observed['native'];ranks=[observed['cpu-rank0'],observed['cpu-rank1']]
            d.m.same(meta(torch,ranks[0]['projection_output']),meta(torch,ranks[1]['projection_output']),'Replicated original CPU outputs')
            torch.set_num_threads(2);native_control=torch.nn.functional.linear(native['context_before_o'],weight)
            d.m.same(meta(torch,native_control),plan['payloads']['native']['projection_output']['metadata'],'Original1231 full output reproduced')
            torch.set_num_threads(1);partials=[torch.einsum('...m,mn->...n',ranks[i]['context_before_o'],shards[i].t())for i in (0,1)];cpu_control=partials[0]+partials[1]
            for rank in (0,1):d.m.same(meta(torch,cpu_control),plan['payloads']['cpu-rank'+str(rank)]['projection_output']['metadata'],'Original2048 TP2 output reproduced')
            controls=d.once(OUT/'controls.json',dict(all3_fullshape_exact=True,native=meta(torch,native_control),cpu=meta(torch,cpu_control),native_threads=2,cpu_threads=1,native_interpreter_reused=False,gloo_reexecuted=False))
            inputs=dict(native_context=native['context_before_o'],cpu_context=torch.cat([v['context_before_o']for v in ranks],dim=-1).contiguous());cases=[];saved_outputs={}
            for case in plan['cases']:
                guard();torch.set_num_threads(case['threads']);name=case['name'];x=inputs[name];d.m.same(list(x.shape),[1,case['length'],2048],'Whole original length')
                before=meta(torch,x);outputs,bf,fp=formulas(torch,x,weight,shards);d.m.same(meta(torch,x),before,'Fixed input unchanged')
                endpoint=outputs['bf16_full']if name=='native_context'else outputs['bf16_split'];target=native_control if name=='native_context'else cpu_control;d.m.same(meta(torch,endpoint),meta(torch,target),'Case original control retained')
                captures={key:save(torch,name+'-'+key,value)for key,value in outputs.items()}
                partial_captures={precision:[save(torch,name+'-'+precision+'-rank'+str(i),value)for i,value in enumerate(values)]for precision,values in [('bf16_partial',bf),('fp32_partial',fp)]}
                pairs=[('bf16_split','bf16_full'),('bf16_split_wide_sum','bf16_split'),('fp32_full_final','bf16_full'),('fp32_split_final','fp32_full_final'),('fp32_split_round_local','fp32_split_final'),('fp32_split_round_local','bf16_split')]
                metrics={a+'_vs_'+b:scopes(torch,outputs[a],outputs[b])for a,b in pairs}
                cases.append(dict(name=name,length=case['length'],threads=case['threads'],input=before,captures=captures,partials=partial_captures,comparisons=metrics,local_bf16_vs_fp32_cast=[scopes(torch,b,f.bfloat16())for b,f in zip(bf,fp)]))
                saved_outputs[name]={k:v.clone()for k,v in outputs.items()if k in ('bf16_full','bf16_split')};guard();d.once(OUT/(name+'-result.json'),cases[-1]);print(json.dumps(dict(stage='case_complete',name=name)),flush=True)
            comparisons=dict(input=measure(torch,inputs['native_context'],inputs['cpu_context'][:,:1231]),original_output=measure(torch,native_control,cpu_control[:,:1231]),cpu_context_full_vs_native=measure(torch,native_control,saved_outputs['cpu_context']['bf16_full'][:,:1231]))
            for role,rows in plan['payloads'].items():
                for field,row in rows.items():d.m.same(meta(torch,observed[role][field]),row['metadata'],'All original observed values retained')
        for path,old in identities.items():d.m.same(identity(path),old,'Installed dependency unchanged')
        d.m.same(plan['sources'],sources(),'Sources unchanged');d.m.same({n:d.m.p.sha(R/n)for n in d.m.CANONICAL},d.m.CANONICAL,'Canonical retained');guard();barrier.verify();resources=resource.getrusage(resource.RUSAGE_SELF)
        result=d.once(OUT/'result.json',dict(schema='minicpm5_frame1_fixed_oproj_result_v1',complete=True,passed=True,owner=owner,admission=admission,plan=plan_ref,controls=controls,weight=weight_saved,cases=cases,observed_comparisons=comparisons,torch_version=str(torch.__version__),native_interpreter_reused=False,gloo_reexecuted=False,model_executed=False,device_executed=False,unique_final_KL_cause_proven=False,full_model_qualified=False,new_speedup_claim=False,actual_self_exit_claimed=False,measured_resources=dict(elapsed_seconds=time.monotonic()-start,peak_rss_bytes=resources.ru_maxrss*1024,output_bytes=usage())))
        print(json.dumps(dict(result=result,owner=owner)),flush=True)
    except BaseException as error:
        if created:d.once(OUT/'failure.json',dict(owner=owner,error_type=type(error).__name__,error=str(error),actual_self_exit_claimed=False))
        raise
    finally:
        signal.alarm(0)
        if barrier is not None:barrier.active=False
        d.runtime.drain_children();d.m.old.post.external.release_lock(fd)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--plan');parser.add_argument('--plan-sha256');parser.add_argument('--review');parser.add_argument('--review-sha256');args=parser.parse_args()
    if args.prepare:prepare()
    else:run(dict(path=args.plan,sha256=args.plan_sha256),dict(path=args.review,sha256=args.review_sha256))
