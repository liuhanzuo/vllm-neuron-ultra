"""Bounded local execution of the pinned official Dory tokenizer and template.

No weights or model forward. Shared-backend wrapper checks are not independent
model/tokenizer parity. Actual outputs and original generation policy are saved.
"""
import argparse, hashlib, json, os, resource, signal, sys, time
from pathlib import Path
Q = Path(__file__).resolve().parent
R = Q.parent
sys.path[:0] = [str(Q), str(R)]
import run_dory_weight_storage_root_v2 as d

SELF = Path(__file__).resolve()
META = Q / 'dory-checkpoint-metadata-20261108-v1'
HF = Q / 'transformers-reference/src'
OUT = R / 'dory-tokenizer-template-r1-20261108-v1'
PLAN = Q / 'dory-tokenizer-template-r1-20261108-v1-plan.json'
TERMINAL = dict(path=str(Q/'dory-original-weight-load-r2-20261108-v1-root-terminal.json'), sha256='f304bc4272a2a1347c380c92571820e3b455aaddc644500cdf0d365341bbdba8')
EXIT = dict(path=str(Q/'dory-original-weight-load-r2-20261108-v1-finalizer-external-exit.json'), sha256='6e0ab3f1232f6d91077f7f2e5a808f0a40b20d7019529a8cbe5e184fa9fa61df')
BUDGET = dict(cpus=[6], address_space_bytes=8*2**30, cpu_seconds=180, wall_seconds=300, file_bytes=16*2**20, output_bytes=16*2**20, root_floor_bytes=28*2**30)


def fixtures():
    def msg(role, text, **extra): return dict(role=role, content=text, **extra)
    def frame(role, text): return '<|im_start|>'+role+'\n'+text+'<|im_end|>\n'
    def case(name, messages, expected, **kwargs):
        return dict(name=name, messages=messages, kwargs=kwargs, expected=expected)
    empty_system = frame('system', '')
    prefix = '<|im_start|>assistant\n<think>\n'
    user = [msg('user', 'Hello')]
    base = empty_system + frame('user', 'Hello')
    history = [msg('user', 'First'), msg('assistant', 'Answer', reasoning_content='old reasoning'), msg('user', 'Next')]
    historical_base = empty_system + frame('user', 'First')
    rows = [
        case('default_thinking', user, base+prefix, add_generation_prompt=True),
        case('explicit_thinking', user, base+prefix, add_generation_prompt=True, enable_thinking=True),
        case('thinking_disabled', user, base+'<|im_start|>assistant\n<think></think>', add_generation_prompt=True, enable_thinking=False),
        case('no_generation_prompt', user, base, add_generation_prompt=False),
        case('system_message', [msg('system', 'Be concise.'), *user], frame('system','Be concise.')+frame('user','Hello')+prefix, add_generation_prompt=True),
        case('medium_effort', user, empty_system+frame('user','Hello\n\n{reasoning effort: efficient}')+prefix, add_generation_prompt=True, medium_effort=True),
        case('history_default_truncation', history, historical_base+frame('assistant','<think></think>Answer')+frame('user','Next')+prefix, add_generation_prompt=True),
        case('history_preserve_thinking', history, historical_base+frame('assistant','<think>\nold reasoning</think>Answer')+frame('user','Next')+prefix, add_generation_prompt=True, truncate_history_thinking=False),
        case('latest_assistant_reasoning', history[:2], historical_base+frame('assistant','<think>\nold reasoning</think>Answer'), add_generation_prompt=False),
        case('assistant_no_reasoning', [*user, msg('assistant','Answer')], base+frame('assistant','<think></think>Answer'), add_generation_prompt=False),
    ]
    unicode_text = '你好，世界！ Café e\u0301 🙂\n\t日本語 العربية 12345'
    rows.append(case('unicode_whitespace', [msg('user',unicode_text)], empty_system+frame('user',unicode_text)+prefix, add_generation_prompt=True))
    call = dict(function=dict(name='lookup', arguments=dict(city='北京', options=[1,2])))
    messages = [*user, msg('assistant', '', tool_calls=[call]), msg('tool','21°C'), msg('tool','sunny')]
    tool_body = '<think></think>\n<tool_call>\n<function=lookup>\n<parameter=city>\n北京\n</parameter>\n<parameter=options>\n[1, 2]\n</parameter>\n</function>\n</tool_call>\n'
    tool_response = '<tool_response>\n21°C\n</tool_response>\n<tool_response>\nsunny\n</tool_response>\n'
    rows.append(case('tool_call_and_two_responses', messages, base+frame('assistant',tool_body)+frame('user',tool_response)+prefix, add_generation_prompt=True))
    tools = [dict(type='function', function=dict(name='lookup', description=' Weather lookup ', parameters=dict(type='object',properties=dict(city=dict(type='string',description=' City ',enum=['北京','Paris'])),required=['city'])))]
    rows.append(dict(name='tool_schema', messages=user, kwargs=dict(add_generation_prompt=True, tools=tools), expected=None,
                     required_fragments=['<name>lookup</name>','<description>Weather lookup</description>','<name>city</name>', '<enum>["北京", "Paris"]</enum>', '<required>["city"]</required>'], required_suffix=frame('user','Hello')+prefix))
    return rows


def predecessors():
    terminal = d.load(TERMINAL); exit_record = d.load(EXIT)
    d.require(terminal['passed'] and terminal['complete'] and terminal['all507_tensor_hashes_independently_recomputed'], 'Actual completed weight-load audit')
    d.m.same((exit_record['terminal'], exit_record['owner']), (TERMINAL, terminal['verifier']), 'Actual audit exit binding')
    d.actual_exit(exit_record['actual_exit'])
    d.require(exit_record['actual_tool_exit_observed'] is exit_record['exact_owner_gone'] is True, 'Actual verifier gone')
    for owner in (terminal['native_owner'], terminal['verifier']): d.m.p.gone(owner)
    return terminal


def local_sources():
    return {**d.sources(), str(SELF): d.ref(SELF)['sha256']}


def prepare():
    d.m.p.user_bus(); fd, owner, lock = d.m.old.post.external.acquire_lock()
    try:
        terminal = predecessors(); d.m.same(lock,terminal['lock_identity'],'Same serial queue'); d.m.p.clear_window(); d.m.p.active_heavy()
        d.require(not OUT.exists() and not PLAN.exists(), 'Fresh preparation only')
        metadata = json.loads((META/'metadata-review.json').read_text())
        inputs = metadata['input_files']
        for row in inputs.values():
            p=Path(row['path']); d.require(p.stat().st_size==row['bytes'] and d.ref(p)['sha256']==row['sha256'],'Pinned official metadata')
        records = {str(p):d.ref(p)['sha256'] for p in sorted(d.runtime.BASE.glob('*.dist-info/RECORD'))}
        hf_files = {}
        for p in sorted((HF/'transformers').rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.json','.jinja'):
                hf_files[str(p)] = dict(sha256=d.ref(p)['sha256'], bytes=p.stat().st_size)
        request=d.load(dict(path=str(Q/'dory-weight-storage-r1-20261108-v1-request.json'),sha256='03feaed616557baa245b09dcd10bb88a711011508ac87c9d1c00b7607c639c28'))
        plan=dict(schema='dory_tokenizer_template_plan_v1', predecessor=TERMINAL, predecessor_exit=EXIT, preparation_owner=owner, lock_identity=lock,
                  sources=local_sources(), metadata_review=d.ref(META/'metadata-review.json'), inputs=inputs, dependency_records=records, hf_files=hf_files,
                  runtime_identity=request['runtime_identity'], budgets=BUDGET, fixtures=fixtures(),
                  numerical_policy='Exact expected prompt bytes, shared-backend IDs, Unicode roundtrip and 1000 special IDs; no numerical-model acceptance',
                  full_model_qualified=False, new_speedup_claim=False)
        print(json.dumps(dict(plan=d.once(PLAN,plan),owner=owner)),flush=True)
    finally: d.m.old.post.external.release_lock(fd)


def run(plan_ref, review_ref):
    plan=d.load(plan_ref); review=d.load(review_ref)
    d.require(review['passed'] and review['reviewer']=='/root' and review['plan']==plan_ref,'Root source review of exact plan')
    d.m.same(plan['sources'],local_sources(),'Unchanged local source closure')
    d.m.same((plan['budgets'],plan['fixtures']),(BUDGET,fixtures()),'Frozen bounds and cases')
    d.require(sys.executable==str(d.runtime.PYTHON) and sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode,'Isolated original runtime')
    d.require(not {'torch','transformers','tokenizers'} & set(sys.modules),'Authenticate before native imports')
    d.m.p.user_bus(); fd,owner,lock=d.m.old.post.external.acquire_lock(); start=time.monotonic(); barrier=None; created=False
    try:
        terminal=predecessors(); d.m.same(lock,terminal['lock_identity'],'Original queue lock'); d.m.p.gone(plan['preparation_owner'])
        d.m.p.clear_window(); d.m.p.active_heavy(); d.require(not OUT.exists(),'Never restart/reuse completed or failed run')
        os.sched_setaffinity(0,{6}); os.nice(19)
        resource.setrlimit(resource.RLIMIT_AS,(BUDGET['address_space_bytes'],)*2)
        resource.setrlimit(resource.RLIMIT_CPU,(BUDGET['cpu_seconds'],BUDGET['cpu_seconds']+5))
        resource.setrlimit(resource.RLIMIT_FSIZE,(BUDGET['file_bytes'],)*2)
        def timeout(signum,frame): raise TimeoutError('Tokenizer validation wall budget')
        signal.signal(signal.SIGALRM,timeout); signal.alarm(BUDGET['wall_seconds'])
        OUT.mkdir(mode=0o700);created=True
        admission=d.once(OUT/'admission.json',dict(owner=owner,lock_identity=lock,plan=plan_ref,review=review_ref,budgets=BUDGET,model_forward_permitted=False))
        last=[0.]
        def guard():
            now=time.monotonic(); d.require(now-start<BUDGET['wall_seconds'],'Bounded duration')
            if now-last[0]>.5:
                d.m.p.lock_proof(owner,lock);d.m.p.clear_window();d.runtime.usage(OUT)
                d.require(os.sched_getaffinity(0)=={6},'CPU affinity retained')
                if barrier is not None:barrier.verify()
                last[0]=now
        d.runtime.validate_runtime_identity(plan['runtime_identity'],True,guard)
        native_files=[]; known={}; total=0
        for name,digest in plan['dependency_records'].items():
            for path,sha,size in d.runtime.parse_record(Path(name),digest,torch=Path(name)==d.runtime.RECORD):
                total+=size; d.require(total<2*2**30,'Bounded installed dependency bytes')
                row=d.runtime.verify_binary(path,sha,size,guard);native_files.append(row);known[str(path)]=sha
        for name,row in plan['hf_files'].items():
            native_files.append(d.runtime.verify_binary(Path(name),row['sha256'],row['bytes'],guard));known[name]=row['sha256']
        for row in plan['inputs'].values():d.runtime.verify_binary(row['path'],row['sha256'],row['bytes'],guard)
        d.once(OUT/'runtime-summary.json',dict(files_verified=len(native_files),installed_dependency_bytes=total,dependency_records=plan['dependency_records'],transformers_source_files=len(plan['hf_files'])))
        os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONNOUSERSITE='1')
        barrier=d.runtime.NoNativeChildren(); sys.addaudithook(barrier.audit)
        def no_network(event,args):
            if event in ('socket.connect','socket.getaddrinfo'):raise RuntimeError('Local tokenizer task may not access network')
        sys.addaudithook(no_network);sys.path.extend([str(d.runtime.BASE),str(HF)])
        import transformers, tokenizers
        from transformers import PreTrainedTokenizerFast
        tokenizer=PreTrainedTokenizerFast.from_pretrained(str(META),local_files_only=True)
        raw=tokenizers.Tokenizer.from_file(str(META/'tokenizer.json'))
        template=(META/'chat_template.jinja').read_text()
        d.m.same(tokenizer.chat_template,template,'External official template loaded without substitution')
        d.m.same((len(tokenizer),tokenizer.bos_token_id,tokenizer.eos_token_id,tokenizer.pad_token_id),(131072,1,11,11),'Actual vocabulary and special IDs')
        cases=[]
        for fixture in plan['fixtures']:
            guard();kwargs=fixture['kwargs'];rendered=tokenizer.apply_chat_template(fixture['messages'],tokenize=False,**kwargs)
            if fixture['expected'] is not None:d.m.same(rendered,fixture['expected'],'Manually specified exact prompt: '+fixture['name'])
            else:
                for fragment in fixture['required_fragments']:d.require(fragment in rendered,'Tool schema fragment '+fragment)
                d.require(rendered.endswith(fixture['required_suffix']),'Tool schema suffix')
            ids=tokenizer.apply_chat_template(fixture['messages'],tokenize=True,**kwargs)
            # Transformers 5 returns BatchEncoding by default; save the actual IDs.
            if hasattr(ids,'keys'):ids=ids['input_ids']
            if ids and isinstance(ids[0],list):d.require(len(ids)==1,'Single conversation');ids=ids[0]
            d.m.same(ids,raw.encode(rendered,add_special_tokens=False).ids,'Same-backend wrapper encoding')
            d.m.same(tokenizer.encode(rendered,add_special_tokens=True),ids,'No implicit BOS/EOS')
            d.m.same(tokenizer.decode(ids,skip_special_tokens=False),rendered,'Exact text roundtrip')
            cases.append(dict(name=fixture['name'],rendered=rendered,input_ids=ids,tokens=len(ids),sha256=hashlib.sha256(rendered.encode()).hexdigest(),exact_expected_prompt=fixture['expected'] is not None))
        original=json.loads((META/'tokenizer.json').read_text()); specials=[]
        for token in original['added_tokens']:
            ids=tokenizer.encode(token['content'],add_special_tokens=False);d.m.same(ids,[token['id']],'Original added special token ID')
            d.m.same(tokenizer.decode(ids,skip_special_tokens=False),token['content'],'Special token decode')
            specials.append(dict(id=token['id'],content=token['content']))
        policy=json.loads((META/'generation_config.json').read_text())
        d.m.same((policy['eos_token_id'],policy['do_sample'],policy['temperature'],policy['top_p'],policy['top_k']),([2,11],True,1.0,.95,0),'Original generation policy')
        d.m.same([tokenizer.convert_ids_to_tokens(x)for x in policy['eos_token_id']],['</s>','<|im_end|>'],'Both generation stops retained')
        for row in native_files:d.m.same(d.runtime.file_identity(row['path']),row['identity'],'Dependencies unchanged')
        modules=[]
        for name,module in sorted(sys.modules.items()):
            path=getattr(module,'__file__',None)
            if path:
                p=Path(path).resolve()
                if p.is_relative_to(d.runtime.BASE) or p.is_relative_to(HF):
                    d.require(str(p) in known,'Imported source authenticated: '+str(p));modules.append(dict(module=name,path=str(p),sha256=known[str(p)]))
        d.once(OUT/'imported-modules.json',modules)
        capture=d.once(OUT/'captures.json',dict(cases=cases,specials=specials,generation_policy=policy))
        d.m.same(plan['sources'],local_sources(),'Local sources unchanged');guard();barrier.verify()
        d.m.same({n:d.m.p.sha(R/n)for n in d.m.CANONICAL},d.m.CANONICAL,'Canonical state retained')
        usage=resource.getrusage(resource.RUSAGE_SELF)
        result=d.once(OUT/'result.json',dict(schema='dory_tokenizer_template_result_v1',complete=True,passed=True,owner=owner,admission=admission,plan=plan_ref,capture=capture,
            versions=dict(transformers=transformers.__version__,tokenizers=tokenizers.__version__),cases=len(cases),exact_expected_prompts=sum(x['exact_expected_prompt']for x in cases),special_ids=len(specials),
            tokenizer_executed=True,chat_template_executed=True,shared_backend_wrapper_consistency=True,independent_tokenizer_implementation_parity=False,
            generation_policy_preserved=True,model_forward_executed=False,full_model_qualified=False,new_speedup_claim=False,actual_self_exit_claimed=False,
            measured_resources=dict(elapsed_seconds=time.monotonic()-start,peak_rss_bytes=usage.ru_maxrss*1024)))
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
