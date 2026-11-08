"""Run the original Dory storage probe only after finalized r7 and actual Mini.

No runtime on import. Preparation requires real completed predecessor evidence;
execution additionally requires a separate exact source/request review. This is
not a checkpoint loader, full-model reference, Neuron baseline or speed benchmark.
"""
import argparse,ast,hashlib,json,os,re,resource,signal,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];Q=R/'research';sys.path[:0]=[str(R),str(Q)]
import run_dory_rotary_root_v1 as runtime
import observe_minicpm5_attention_terminal_v1 as terminal
import dory_weight_storage_capture_v1 as capture
m=terminal.m;require=m.p.require;once=runtime.once;ref=runtime.ref;load=runtime.load
SELF=Path(__file__).resolve()
BUDGET=dict(runtime.BUDGET)
PINS={Q/'run_dory_rotary_root_v1.py':'3db2bd1f728d81fb20c7c98b16232f14557c4370d8ef91bccc560b82afffb511',
      Q/'observe_minicpm5_attention_terminal_v1.py':'db3a1c591fa3ebeb0735c42fb9a757ed4a4bb3ada4e084e29bc6504436e5d9d5',
      Q/'run_minicpm5_attention_review_v1.py':'0f73218d44fdd0813e78e879b410bc1ae9a5035d6291807856fce618c5cfa4c6',
      Q/'dory_weight_storage_capture_v1.py':'a09cca64327292a4371d35ef5a5d806961fc1633e0d54393b4c559aad5f483a1'}


def actual_exit(value):
    require(type(value['session_id'])is int and value['session_id']>0 and type(value['exit_code'])is int and value['exit_code']==0 and re.fullmatch('[0-9a-f]{6}',value['tool_chunk']),'Actually observed successful tool exit')
    return value


def bind_review_exit(run,run_ref,exit_value,terminal_ref,t):
    require(run['schema']=='minicpm5_attention_independent_review_run_v1'and run['complete']is True,'Actual completed Mini independent review')
    require(run['terminal']==terminal_ref and run['actual_model_exit']==t['actual_exit'],'Independent review binds actual Mini terminal')
    require(run['model_exit_verified']is run['physical_drain_verified']is run['complete_report_reproduced']is True,'Actual complete Mini review')
    require(run['all21_own_input_controls_exact']is t['all_own_input_controls_exact'],'Preserve genuine failed scientific controls')
    require(run['actual_self_exit_claimed']is run['full_model_qualified']is run['new_speedup_claim']is False,'No self-exit or model promotion')
    terminal.process(run['owner'])
    require(exit_value['schema']=='minicpm5_attention_independent_review_actual_external_exit_v1'and exit_value['reviewer']=='/root','Actual root-observed Mini review exit')
    require(exit_value['run']==run_ref and exit_value['owner']==run['owner']and exit_value['terminal']==terminal_ref,'Exact original review owner and result')
    actual_exit(exit_value['actual_exit'])
    require(exit_value['actual_tool_exit_observed']is exit_value['exact_owner_gone']is True and exit_value['new_speedup_claim']is False,'Observed real exit and no speed promotion')
    return run['owner']


def predecessor(terminal_ref,review_run_ref,review_exit_ref):
    t=terminal.post.validate_terminal(load(terminal_ref));run=load(review_run_ref);ended=load(review_exit_ref)
    review_owner=bind_review_exit(run,review_run_ref,ended,terminal_ref,t)
    spec=load(t['spec']);review=load(t['review']);root=Path(t['output_root'])
    require(spec['schema']=='minicpm5_frame1_attention_spec_v1'and spec['runner']==m.SELF and spec['output_root']==str(root),'Original actual Mini runtime specification')
    require(root.parent==R and root.resolve()==root and re.fullmatch('minicpm5-frame1-attention-r[0-9]+-20261108-v[0-9]+',root.name),'Exact original Mini output namespace')
    require(review['spec']==t['spec']and review['reviewer']=='/root'and review['passed']is review['source_only']is review['independent_review']is True and review['blockers']==[],'Exact original Mini source approval')
    require(spec['canonical_sha256']==m.CANONICAL,'Original canonical inputs')
    # Do not call load_spec from a successor process: its loaded-module check
    # deliberately admits only Mini's own source closure, not this new runner.
    base=load(m.BASE_PLAN)
    for group in (base['source_sha256'],spec['new_sources']):
        for path,digest in group.items():require(ref(path)['sha256']==digest,'Frozen original model/helper source')
    exited=load(t['actual_external_exit']);live=load(exited['live']);terminal.external_exit(exited,live)
    require(exited['spec']==t['spec']and exited['review']==t['review']and exited['actual_exit']==t['actual_exit'],'Actual original Mini external exit')
    records={name:load(reference)for name,reference in t['references'].items()}
    for name in terminal.names():require(ref(root/name)==t['references'][name],'Actual original terminal metadata unchanged')
    bound=terminal.terminal_metadata(root,spec,exited,records,t['references'])
    for key,value in bound.items():require(t[key]==value,'Original terminal metadata independently rebound')
    post_approval=load(run['source_review'])
    require(post_approval['schema']=='minicpm5_attention_post_exit_source_review_v1'and post_approval['reviewer']=='/root'and post_approval['passed']is post_approval['source_only']is post_approval['independent_review']is True and post_approval['blockers']==[],'Actual separately reviewed Mini byte verifier')
    require(post_approval['terminal']==terminal_ref and post_approval['budgets']==terminal.post.BUDGET,'Original post-exit review binding/budgets')
    for path,digest in post_approval['source_sha256'].items():require(ref(path)['sha256']==digest,'Frozen actual post-exit reviewer')
    require(post_approval['source_sha256'].get(str(terminal.post.SELF))==ref(terminal.post.SELF)['sha256']and post_approval['source_sha256'].get(str(terminal.post.CALLBACK))==terminal.post.CALLBACK_SHA,'Actual independent verifier source pinned')
    admission=load(run['admission']);require(admission['owner']==review_owner and admission['terminal']==terminal_ref and admission['source_review']==run['source_review']and admission['lock_identity']==t['lock_identity'],'Original independent review admission')
    require(run['original_phase_report']==t['references']['attention-review.json'],'Actual original full review report')
    terminal.post.compare_review(load(run['result']),records['attention-review.json'])
    m.load_history_helpers();catalog=m.current_history(spec)
    owners=[t['outer_owner'],t['terminal_observer'],*t['phase_owners'].values(),*t['rank_owners'],review_owner]
    require(len({(o['pid'],o['start_ticks'])for o in owners})==len(owners),'Distinct actual Mini owners')
    return dict(terminal=t,spec=spec,catalog=catalog,owners=owners,admission=records['admission.json'],review_run=run)


def physical_drain(checked):
    m.load_history_helpers();m.history_metadata.physical_drain(checked['catalog'],checked['spec'])
    for owner in checked['owners']:m.p.gone(owner)
    t=checked['terminal'];spec=checked['spec'];samples=[]
    for phase in m.PHASES:
        unit=spec['units'][phase];props=m.p.properties(unit)
        m.service_identity(spec,t['references']['admission.json'],checked['admission']['attempt_id'],phase,props,t['phase_owners'][phase])
        require(m.p.terminal(props)and props['Result']=='success'and props['ExecMainStatus']=='0','Original Mini service successfully terminal')
        samples.append(dict(phase=phase,properties=props,group=m.p.empty_group(m.p.GROUP_PREFIX+unit+'.service')))
    return samples


def source_paths():
    loader=capture.probe.bridge.loader
    paths={SELF,Q/'check_run_dory_weight_storage_root_v1.py',*PINS,capture.probe.bridge.CANDIDATE,loader.CONTRACT_PATH}
    paths.update(loader.METADATA/name for name in ('safetensors-header.json','config.json','metadata-review.json'))
    todo=list(paths);seen=set()
    while todo:
        path=todo.pop()
        if path in seen:continue
        seen.add(path)
        if path.suffix!='.py':continue
        for node in ast.walk(ast.parse(path.read_text())):
            names=[a.name for a in node.names]if isinstance(node,ast.Import)else[node.module]if isinstance(node,ast.ImportFrom)and node.module and not node.level else[]
            for name in names:
                for folder in (R,Q):
                    found=folder/(name.replace('.','/')+'.py')
                    if found.is_file()and found not in paths:paths.add(found);todo.append(found)
    require(len(paths)<=256,'Bounded original complete local source closure')
    return sorted(map(str,paths))


def sources():
    for path,digest in PINS.items():require(ref(path)['sha256']==digest,'Unchanged original runtime/terminal/capture helper')
    capture.source_check();return {path:ref(path)['sha256']for path in source_paths()}


def validate_request(request):
    require(request['schema']=='dory_weight_storage_runtime_request_v1'and re.fullmatch('r[0-9]+-20261108-v[0-9]+',request['run_id']),'Finite fresh storage request')
    require(request['output_root']==str(R/('dory-weight-storage-'+request['run_id'])),'Exact fresh storage output namespace')
    require(request['budgets']==BUDGET and request['science']==capture.probe.plan(),'Full unchanged fixture and resource plan')
    require(request['python']==str(runtime.PYTHON)and request['dependency_records']==runtime.RECORDS,'Original CPU Torch installation')
    runtime.validate_runtime_identity(request['runtime_identity'],False)
    require(request['source_sha256']==sources(),'Exact complete approved source closure')
    require(request['canonical_sha256']==m.CANONICAL=={name:m.p.sha(R/name)for name in m.CANONICAL},'Canonical state unchanged')
    require(request['runtime_admitted']is request['full_model_qualified']is request['new_speedup_claim']is request['automatic_successor_launch']is False,'Preparation grants no runtime/model approval')
    terminal.process(request['preparation_owner']);return request


def prepare(terminal_ref,review_run_ref,review_exit_ref,run_id):
    require(re.fullmatch('r[0-9]+-20261108-v[0-9]+',run_id),'Finite original run namespace')
    checked=predecessor(terminal_ref,review_run_ref,review_exit_ref);m.p.user_bus();fd,owner,lock=m.old.post.external.acquire_lock()
    try:
        m.p.clear_window();m.p.active_heavy();require(lock==checked['terminal']['lock_identity'],'Original serial lock');drain=physical_drain(checked)
        def guard():m.p.lock_proof(owner,lock);m.p.clear_window()
        identity=runtime.capture_runtime_identity(guard);guard()
        request=dict(schema='dory_weight_storage_runtime_request_v1',run_id=run_id,output_root=str(R/('dory-weight-storage-'+run_id)),python=str(runtime.PYTHON),dependency_records=runtime.RECORDS,runtime_identity=identity,source_sha256=sources(),budgets=BUDGET,science=capture.probe.plan(),predecessor_terminal=terminal_ref,predecessor_review_run=review_run_ref,predecessor_review_exit=review_exit_ref,preparation_owner=owner,lock_identity=lock,preparation_physical_drain=drain,canonical_sha256=m.CANONICAL,runtime_admitted=False,full_model_qualified=False,new_speedup_claim=False,automatic_successor_launch=False)
        validate_request(request);require(not Path(request['output_root']).exists(),'Never reuse storage runtime output')
        target=Q/('dory-weight-storage-'+run_id+'-request.json');request_ref=once(target,request)
        print(json.dumps(dict(request=request_ref,owner=owner,actual_self_exit_claimed=False,runtime_admitted=False)),flush=True)
    finally:m.old.post.external.release_lock(fd)


def run(request_ref,review_ref):
    request=validate_request(load(request_ref));review=load(review_ref)
    require(review['schema']=='dory_weight_storage_runtime_source_review_v1'and review['reviewer']=='/root'and review['passed']is review['source_only']is review['independent_review']is True and review['blockers']==[],'Separate exact storage runtime source review')
    require(review['request']==request_ref and review['source_sha256']==request['source_sha256']and review['preparation_owner']==request['preparation_owner'],'Exact reviewed request/source/preparation owner');actual_exit(review['preparation_actual_exit']);m.p.gone(request['preparation_owner'])
    require(sys.executable==str(runtime.PYTHON)and sys.flags.no_site and sys.flags.isolated and sys.dont_write_bytecode and __debug__,'Fixed isolated original native interpreter')
    require(not {'torch','numpy','torch_neuronx','transformers'}&set(sys.modules),'Authenticate before native framework import')
    require({key:os.environ.get(key)for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONNOUSERSITE')}==dict(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONNOUSERSITE='1'),'Original explicit single-thread native environment')
    out=Path(request['output_root']);require(not out.exists(),'Fresh actual runtime output');m.p.user_bus();fd,owner,lock=m.old.post.external.acquire_lock();started=time.monotonic();barrier=None;created=False;old_alarm=None;last_guard=[0.]
    try:
        m.p.clear_window();m.p.active_heavy();require(lock==request['lock_identity'],'Same original serial lock')
        checked=predecessor(request['predecessor_terminal'],request['predecessor_review_run'],request['predecessor_review_exit']);drain=physical_drain(checked);m.p.lock_proof(owner,lock)
        os.sched_setaffinity(0,set(BUDGET['cpus']));os.nice(19)
        resource.setrlimit(resource.RLIMIT_AS,(BUDGET['address_space_bytes'],)*2);resource.setrlimit(resource.RLIMIT_CPU,(BUDGET['cpu_seconds'],BUDGET['cpu_seconds']+5));resource.setrlimit(resource.RLIMIT_FSIZE,(BUDGET['file_bytes'],)*2)
        def timeout(signum,frame):raise TimeoutError('Dory storage native wall budget')
        old_alarm=signal.signal(signal.SIGALRM,timeout);signal.alarm(BUDGET['wall_seconds']);out.mkdir(mode=0o700);created=True
        def guard():
            now=time.monotonic();require(now-started<BUDGET['wall_seconds'],'Actual bounded runtime duration');require(os.sched_getaffinity(0)==set(BUDGET['cpus']),'Original native CPU affinity')
            if now-last_guard[0]>=.5:
                m.p.lock_proof(owner,lock);m.p.clear_window();runtime.usage(out);last_guard[0]=now
                if barrier is not None:barrier.verify()
        guard();runtime_files=runtime.validate_runtime_identity(request['runtime_identity'],True,guard);native_files=runtime.verify_torch_install(guard)
        admission=once(out/'admission.json',dict(schema='dory_weight_storage_actual_admission_v1',request=request_ref,review=review_ref,owner=owner,lock_identity=lock,predecessor_terminal=request['predecessor_terminal'],predecessor_review_run=request['predecessor_review_run'],predecessor_review_exit=request['predecessor_review_exit'],preparation_owner=request['preparation_owner'],all_predecessors_physically_drained=True,mini_drain=drain,resources=BUDGET,actual_self_exit_claimed=False,new_speedup_claim=False))
        once(out/'runtime-files.json',dict(files=native_files,runtime_files=runtime_files,records=runtime.RECORDS));runtime.usage(out)
        barrier=runtime.NoNativeChildren();barrier.verify();sys.addaudithook(barrier.audit);sys.path.append(str(runtime.BASE))
        import torch
        require(str(torch.__version__)=='2.9.1+cpu'and Path(torch.__file__).resolve()==runtime.BASE/'torch/__init__.py','Original actual CPU Torch module')
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        with torch.inference_mode():result=capture.run(torch=torch,guard=guard,output=out/'capture')
        saved=capture.review(result['result'],guard);saved_ref=once(out/'independent-review.json',saved)
        for row in native_files:require(runtime.file_identity(row['path'])==row['identity'],'Native dependency files unchanged')
        validate_request(request);guard();barrier.verify();barrier.active=False;m.p.active_heavy();barrier.verify();total=runtime.usage(out);usage=resource.getrusage(resource.RUSAGE_SELF)
        terminal_ref=once(out/'terminal.json',dict(schema='dory_weight_storage_actual_terminal_v1',complete=True,passed=True,owner=owner,request=request_ref,admission=admission,result=result['result'],independent_review=saved_ref,saved_tensors=18,saved_raw_bytes=786864,all_saved_words_exact=saved['all_saved_words_exact'],native_storage_lifetime_observed=True,native_lifetime_independently_reexecuted=False,actual_self_exit_claimed=False,measured_resources=dict(elapsed_seconds=time.monotonic()-started,peak_rss_bytes=usage.ru_maxrss*1024,user_seconds=usage.ru_utime,system_seconds=usage.ru_stime,output_bytes=total),original_weights_loaded=False,model_constructed=False,full_model_memory_fit_proven=False,full_model_qualified=False,new_speedup_claim=False))
        runtime.usage(out);print(json.dumps(dict(terminal=terminal_ref,owner=owner,actual_self_exit_claimed=False,new_speedup_claim=False)),flush=True)
    except BaseException as error:
        if created and not(out/'failure.json').exists():once(out/'failure.json',dict(error_type=type(error).__name__,error=str(error),owner=owner,request=request_ref,actual_self_exit_claimed=False,full_model_qualified=False,new_speedup_claim=False))
        raise
    finally:
        signal.alarm(0)
        if old_alarm is not None:signal.signal(signal.SIGALRM,old_alarm)
        if barrier is not None:barrier.active=False
        try:runtime.drain_children()
        except BaseException:
            print('Dory native children not drained; retaining serial lock for root recovery',file=sys.stderr,flush=True)
            while True:time.sleep(1)
        m.old.post.external.release_lock(fd)


if __name__=='__main__':
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True);mode.add_argument('--prepare',action='store_true');mode.add_argument('--run',action='store_true')
    for name in ('terminal','terminal-sha256','review-run','review-run-sha256','review-exit','review-exit-sha256','run-id','request','request-sha256','review','review-sha256'):parser.add_argument('--'+name)
    args=parser.parse_args();require(sys.dont_write_bytecode and __debug__,'Bytecode disabled and assertions enabled')
    if args.prepare:prepare(dict(path=args.terminal,sha256=args.terminal_sha256),dict(path=args.review_run,sha256=args.review_run_sha256),dict(path=args.review_exit,sha256=args.review_exit_sha256),args.run_id)
    else:run(dict(path=args.request,sha256=args.request_sha256),dict(path=args.review,sha256=args.review_sha256))
