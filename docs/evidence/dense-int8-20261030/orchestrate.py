"""Serial accelerator experiment queue with bounded jobs and durable results."""
import concurrent.futures
import json
import os
import signal
import threading
import statistics
import subprocess
import sys
import time
import traceback
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from transformers import AutoTokenizer
from bench_http import benchmark, request

ROOT=Path(__file__).resolve().parent
PYTHON='/home/ubuntu/venvs/vllm_neuron/bin/python'
ART=Path('/mnt/nvme/neuron-artifacts')
BASE_MODEL='/mnt/nvme/models/Qwen3-8B'
DOWNLOAD=None
DOWNLOAD_LOG=None
RESULTS=json.loads((ROOT/'results.json').read_text()) if (ROOT/'results.json').exists() else []


def emit(event,**data):
    row={'time':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'event':event,**data}
    line=json.dumps(row,ensure_ascii=False)
    print(line,flush=True)
    with (ROOT/'events.jsonl').open('a') as f: f.write(line+'\n')
    (ROOT/'status.json').write_text(json.dumps(row,ensure_ascii=False,indent=2))


def save():
    temp=ROOT/'results.tmp'
    temp.write_text(json.dumps(RESULTS,ensure_ascii=False,indent=2))
    temp.replace(ROOT/'results.json')


def env(artifact=None,cores=None):
    out={'HF_HOME':'/mnt/nvme/hf','NEURON_COMPILE_CACHE_URL':'/mnt/nvme/neuron-cache',
         'TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'1',
         'PATH':'/home/ubuntu/venvs/vllm_neuron/bin:/opt/aws/neuron/bin:/usr/local/bin:/usr/bin:/bin'}
    if artifact: out['NEURON_COMPILED_ARTIFACTS']=str(artifact)
    if cores is not None: out['NEURON_RT_VISIBLE_CORES']=str(cores)
    return out


def unit_args(name,log,settings,timeout=2400,wait=False,memory_max_gib=22,memory_high_gib=20,swap_gib=8):
    args=['systemd-run','--user','--unit='+name,'--collect','--quiet','--working-directory='+str(ROOT),
          '-p',f'MemoryMax={memory_max_gib}G','-p',f'MemoryHigh={memory_high_gib}G','-p',f'MemorySwapMax={swap_gib}G',
          '-p','RuntimeMaxSec='+str(timeout),'-p','TimeoutStopSec=15s','-p','StandardOutput=append:'+str(log),'-p','StandardError=inherit']
    if wait: args.append('--wait')
    for k,v in settings.items(): args.append('--setenv='+k+'='+v)
    return args


def stop(unit):
    subprocess.run(['systemctl','--user','stop',unit],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)


def port_ready(port):
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models',timeout=2) as r: json.load(r)
        return True
    except Exception: return False


def wait_server(unit,port,limit=2100):
    start=time.monotonic()
    heartbeat=0
    while time.monotonic()-start<limit:
        if port_ready(port): return time.monotonic()-start
        active=subprocess.run(['systemctl','--user','is-active',unit],capture_output=True,text=True).stdout.strip()
        if active not in ['active','activating']: raise RuntimeError(f'{unit} ended before readiness: {active}')
        if time.monotonic()-heartbeat>45:
            emit('waiting_for_server',unit=unit,elapsed_s=round(time.monotonic()-start))
            heartbeat=time.monotonic()
        time.sleep(2)
    raise TimeoutError(f'{unit} did not become ready within {limit}s')


def start_downloads():
    global DOWNLOAD,DOWNLOAD_LOG
    if DOWNLOAD is None:
        catalog_path=ROOT/'model-catalog.json'
        if catalog_path.exists():
            catalog=json.loads(catalog_path.read_text())
            public=[x for x in catalog.get('models',[]) if x['repo_id']!='Qwen/Qwen3-8B' and x.get('gated') is False]
            if len(public)>=3 and all(x.get('status')=='downloaded' and Path(x.get('local_path','/nonexistent')).is_dir() for x in public):
                emit('model_downloads_reused')
                return
        DOWNLOAD_LOG=(ROOT/'downloads.log').open('a')
        settings={**os.environ,**env(),'HF_HUB_DISABLE_XET':'1'}
        DOWNLOAD=subprocess.Popen([PYTHON,str(ROOT/'catalog.py'),'--download'],env=settings,stdout=DOWNLOAD_LOG,stderr=subprocess.STDOUT,start_new_session=True)
        emit('downloads_started',pid=DOWNLOAD.pid)


def finish_downloads():
    if DOWNLOAD is None: return
    while DOWNLOAD.poll() is None:
        emit('waiting_for_downloads')
        time.sleep(30)
    if DOWNLOAD.returncode: raise RuntimeError('Model download job failed; see downloads.log')


def overrides(case):
    out={'on_device_sampling_config':None if case.get('cpu') else {'dynamic':False,'do_sample':False,'top_k':1}}
    if case.get('int8'):
        out.update(quantized=True,quantization_type='per_channel_symmetric',quantization_dtype='int8',
                   quantized_checkpoints_path=case['quant_path'],modules_to_not_convert=['lm_head'])
    out.update(case.get('extra_overrides',{}))
    return out


def validate_cache(case):
    path=Path(case['artifact'])/'neuron_config.json'
    if not path.is_file(): return False
    cfg=json.loads(path.read_text())
    nc=cfg.get('neuron_config',cfg)
    expected={'batch_size':case['batch'],'tp_degree':case.get('tp',2),'seq_len':2048,'quantized':bool(case.get('int8'))}
    for k,v in expected.items():
        if nc.get(k)!=v: raise RuntimeError(f'Cache mismatch {k}: {nc.get(k)!r} != {v!r}')
    for k,v in case.get('extra_overrides',{}).items():
        if nc.get(k)!=v:raise RuntimeError(f'Cache override mismatch {k}: {nc.get(k)!r} != {v!r}')
    if case.get('int8') and Path(nc.get('quantized_checkpoints_path') or '/nonexistent').resolve()!=Path(case['quant_path']).resolve():
        raise RuntimeError('Compiled artifact points to a different quantized checkpoint')
    ods=nc.get('on_device_sampling_config')
    if case.get('cpu'):
        if ods is not None: raise RuntimeError('CPU sampling case points to a device-sampling graph')
    elif not ods or ods.get('dynamic') is not False or ods.get('do_sample') is not False:
        raise RuntimeError('Greedy case does not contain the validated argmax sampler')
    return (Path(case['artifact'])/'model.pt').is_file()


def server(case,port=8010,suffix='',cores=None):
    unit='trn1-bench-'+case['name']+suffix
    if port_ready(port): raise RuntimeError(f'Port {port} already occupied; refusing to measure an unidentified server')
    cached=validate_cache(case)
    if not cached: start_downloads()
    settings=env(case['artifact'],cores)
    if case.get('compile_workers'):settings['TRN1_COMPILE_WORKERS']=str(case['compile_workers'])
    if case.get('sampler_diagnostic_dir'):settings['TRN1_SAMPLER_DIAGNOSTIC_DIR']=case['sampler_diagnostic_dir']
    if case.get('cpu_fast_sampler'):
        settings['TRN1_CPU_SAMPLER_FAST']='1'
        if case.get('cpu_sampler_stats_dir'):settings['TRN1_CPU_SAMPLER_STATS_DIR']=case['cpu_sampler_stats_dir']
    cold_tp1=not cached and case.get('tp',2)==1
    limits={'memory_max_gib':26,'memory_high_gib':24,'swap_gib':12} if cold_tp1 else {}
    args=unit_args(unit,ROOT/(case['name']+suffix+'-server.log'),settings,timeout=4200 if cold_tp1 else 3600,**limits)
    use_wrapper=case.get('bias_shim') or case.get('compile_workers') or case.get('sampler_diagnostic_dir') or case.get('cpu_fast_sampler')
    entry=[PYTHON,str(ROOT/'server_with_bias_fix.py')] if use_wrapper else [PYTHON,'-m','vllm.entrypoints.openai.api_server']
    args += entry+['--model',case['model_path'],
             '--served-model-name',case['served_name'],'--tensor-parallel-size',str(case.get('tp',2)),
             '--max-model-len','2048','--max-num-seqs',str(case['batch']),'--no-enable-prefix-caching',
             '--host','127.0.0.1','--port',str(port),'--additional-config',json.dumps({'override_neuron_config':overrides(case)})]
    (ROOT/(case['name']+suffix+'-launch.json')).write_text(json.dumps({'argv':args,'cache_reused':cached},indent=2))
    subprocess.run(args,check=True)
    return unit,wait_server(unit,port,limit=3600 if cold_tp1 else 2100)


def prompt_ids(case,text):
    tokenizer=AutoTokenizer.from_pretrained(case['model_path'])
    return tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=True,add_generation_prompt=True,enable_thinking=False)


def quality(case,port=8010):
    checks=[('法国的首都是哪里？只回答城市名。','巴黎'),('2加2等于几？只回答数字。','4'),
            ('日本的首都是哪里？只回答城市名。','东京'),('10乘以10等于几？只回答数字。','100')]
    def one(item):
        prompt,expected=item
        r=request(port,case['served_name'],prompt_ids(case,prompt),256 if case.get('reasoning') else 40,force=False)
        return {'prompt':prompt,'expected':expected,'text':r['text'],'matches_exactly':r['text'].strip()==expected,'contains_expected':expected in r['text']}
    serial=[one(x) for x in checks]
    with concurrent.futures.ThreadPoolExecutor(case['batch']) as pool:
        concurrent_rows=list(pool.map(one,(checks*case['batch'])[:case['batch']]))
    same=prompt_ids(case,'2加2等于几？只回答数字。')
    with concurrent.futures.ThreadPoolExecutor(case['batch']) as pool:
        identical=list(pool.map(lambda _:request(port,case['served_name'],same,40,force=False)['text'],range(case['batch'])))
    report={'serial':serial,'concurrent':concurrent_rows,'identical_prompt_outputs':identical,
            'all_identical_rows_match':len(set(identical))==1,'note':'Small smoke test only; does not establish overall model quality.'}
    if case['model_path']==BASE_MODEL:
        report['passed']=all(x['matches_exactly'] for x in serial+concurrent_rows) and report['all_identical_rows_match']
    else:
        # Distilled reasoning models may spend the token budget on reasoning. Save raw outputs instead of falsely certifying quality.
        report['passed']=all(x['text'].strip() for x in serial+concurrent_rows) and report['all_identical_rows_match']
    return report


def sampling_probe(case,port=8010):
    prompt=prompt_ids(case,'给一家猫咪开的太空咖啡馆取一个有创意的名字。只回答名字，每次尝试不同的构思。')
    results={}
    for label,params in [('greedy',{'temperature':0.0}),('topk1',{'temperature':1.0,'top_k':1}),('random',{'temperature':1.2,'top_p':0.95,'top_k':40})]:
        results[label]=[request(port,case['served_name'],prompt,24,force=False,seed=seed,**params)['text'] for seed in range(6)]
    return {'outputs':results,'greedy_unique':len(set(results['greedy'])),'topk1_unique':len(set(results['topk1'])),
            'random_unique':len(set(results['random'])),'note':'Behavioral sampling probe; diversity alone does not prove distributional correctness.'}


def direct(case):
    output=ROOT/(case['name']+'-direct.json')
    args=unit_args('trn1-direct-'+case['name'],ROOT/(case['name']+'-direct.log'),env(cores='0-1'),timeout=900,wait=True)
    args += [PYTHON,str(ROOT/'direct_bench.py'),'--model-path',case['model_path'],'--artifacts',case['artifact'],
             '--batch',str(case['batch']),'--output',str(output)]
    if case.get('device_profile'): args.append('--profile')
    subprocess.run(args,check=True)
    return json.loads(output.read_text())


def quantize(case):
    path=Path(case['quant_path'])
    if (path/'model.safetensors').is_file(): return
    emit('quantizing',model=case['served_name'])
    args=unit_args('trn1-quant-'+case['name'],ROOT/(case['name']+'-quantize.log'),env(),timeout=1800,wait=True)
    args += [PYTHON,str(ROOT/'quantize.py'),'--model-path',case['model_path'],'--output',str(path)]
    subprocess.run(args,check=True)
    if not (path/'model.safetensors').is_file(): raise RuntimeError('Quantization did not produce the expected checkpoint')


def run_case(case):
    if case.get('int8'):
        hf_config=json.loads((Path(case['model_path'])/'config.json').read_text())
        if hf_config.get('model_type')=='qwen2':case={**case,'bias_shim':True}
    previous=next((r for r in RESULTS if r['case']['name']==case['name']),None)
    if previous is not None and previous['status']=='passed':
        emit('case_reused',name=case['name'])
        return previous
    if previous is not None and previous['status']=='failed' and case.get('no_retry'):
        emit('known_failure_deferred_to_followup',name=case['name'])
        return previous
    unit=None
    rec={'case':case,'status':'running'}
    if previous is not None:
        with (ROOT/'attempt-history.jsonl').open('a') as f:f.write(json.dumps(previous,ensure_ascii=False)+'\n')
        RESULTS[RESULTS.index(previous)]=rec
    else:RESULTS.append(rec)
    save()
    emit('case_started',name=case['name'])
    try:
        if case.get('int8'): quantize(case)
        unit,startup=server(case,cores='0' if case.get('tp')==1 else '0-1')
        if DOWNLOAD is not None and DOWNLOAD.poll() is None: finish_downloads()
        rec.update(startup_s=startup,quality=quality(case))
        (ROOT/(case['name']+'-quality.json')).write_text(json.dumps(rec['quality'],ensure_ascii=False,indent=2))
        if not rec['quality']['passed']: raise RuntimeError('Correctness smoke test failed; refusing performance recommendation')
        http_path=ROOT/(case['name']+'-http.json')
        if http_path.is_file(): http=json.loads(http_path.read_text())
        else:
            http=benchmark(8010,case['served_name'],[prompt_ids(case,'请详细介绍一下法国的首都。')],case['batch'],sampling=case.get('http_sampling'))
            http_path.write_text(json.dumps(http,ensure_ascii=False,indent=2))
        rec['http']=http['summary']
        if case.get('cpu') and not case.get('skip_sampling_probe'): rec['sampling']=sampling_probe(case)
        if case.get('inspect_topk1'):
            from sampling_ties import evaluate as inspect_ties
            try:rec['topk1_diagnostic']=inspect_ties(case)
            except Exception as exc:rec['topk1_diagnostic']={'error':str(exc)}
        stop(unit);unit=None
        if case.get('direct'): rec['direct_output']=str(ROOT/(case['name']+'-direct.json'));direct(case)
        rec['status']='passed'
        emit('case_passed',name=case['name'],summary=rec['http'])
    except Exception as exc:
        rec.update(status='failed',error=str(exc),traceback=traceback.format_exc())
        emit('case_failed',name=case['name'],error=str(exc))
    finally:
        if unit: stop(unit)
        # Also stop failed starts whose name was not returned by server().
        stop('trn1-bench-'+case['name'])
        save()
    return rec


def replicas(case):
    if not any(r['case']['name']==case['name'] and r['status']=='passed' for r in RESULTS): return
    units=[]
    try:
        for idx in range(2):
            u,_=server(case,port=8011+idx,suffix='-replica'+str(idx),cores=str(idx));units.append(u)
        quality_rows=[quality(case,port=8011+idx) for idx in range(2)]
        if not all(q['passed'] for q in quality_rows):raise RuntimeError('Replica correctness smoke test failed')
        try:
            from runtime_snapshot import capture
            capture(ROOT/(case['name'].split('-int8')[0]+'-replicas-runtime-memory.json'))
        except Exception as exc:emit('replica_memory_snapshot_failed',error=str(exc))
        prompt=prompt_ids(case,'请详细介绍一下法国的首都。')
        for idx in range(2):request(8011+idx,case['served_name'],prompt,8)
        rows=[]
        concurrency=case['batch']*2
        for trial in range(3):
            barrier=threading.Barrier(concurrency)
            def one(i):
                barrier.wait()
                return request(8011+(i%2),case['served_name'],prompt,120)
            start=time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
                outputs=list(pool.map(one,range(concurrency)))
            elapsed=time.perf_counter()-start
            total=sum(o['usage']['completion_tokens'] for o in outputs)
            if total!=120*concurrency or len({o['text'] for o in outputs})!=1:
                raise RuntimeError('Replica token budget or greedy row consistency failed')
            rows.append({'throughput_tok_s':total/elapsed,'elapsed_s':elapsed,'requests':outputs})
        data={'case':case,'configuration':f'two TP=1 replicas, each batch={case["batch"]}, total concurrency={concurrency}',
              'prompt_ids':prompt,'quality':quality_rows,'rounds':rows,
              'best_tok_s':max(r['throughput_tok_s'] for r in rows),'median_tok_s':statistics.median(r['throughput_tok_s'] for r in rows)}
        (ROOT/(case['name'].split('-int8')[0]+'-int8-two-replicas.json')).write_text(json.dumps(data,ensure_ascii=False,indent=2))
        emit('replicas_passed',best_tok_s=data['best_tok_s'],median_tok_s=data['median_tok_s'])
        return data
    except Exception as exc: emit('replicas_failed',error=str(exc))
    finally:
        for unit in units: stop(unit)
        for idx in range(2): stop('trn1-bench-'+case['name']+'-replica'+str(idx))


def main():
    q={'model_path':BASE_MODEL,'served_name':'Qwen3-8B','batch':4,'tp':2}
    cases=[
      {**q,'name':'qwen3-bf16-tp2-b4','artifact':str(ART/'Qwen3-8B-ks-baseline'),'direct':True},
      {**q,'name':'qwen3-int8-tp2-b4','artifact':str(ART/'Qwen3-8B-ks-int8'),'int8':True,'quant_path':BASE_MODEL+'-int8','direct':True},
      {**q,'name':'qwen3-bf16-cpu-tp2-b4','artifact':str(ART/'Qwen3-8B-nodevsample'),'cpu':True},
      {**q,'name':'qwen3-int8-cpu-tp2-b4','artifact':str(ART/'exp-20261030-qwen3-int8-cpu-tp2-b4'),'cpu':True,'int8':True,'quant_path':BASE_MODEL+'-int8'},
      {**q,'name':'qwen3-int8-tp2-b8','artifact':str(ART/'Qwen3-8B-ks-int8-bs8'),'batch':8,'int8':True,'quant_path':BASE_MODEL+'-int8'},
      {**q,'name':'qwen3-int8-tp1-b4','artifact':str(ART/'exp-20261030-qwen3-int8-tp1-b4'),'tp':1,'int8':True,'quant_path':BASE_MODEL+'-int8','no_retry':True},
    ]
    for case in cases: run_case(case)
    replicas(cases[-1])
    start_downloads();finish_downloads()
    catalog=json.loads((ROOT/'model-catalog.json').read_text())
    for entry in catalog['models']:
        if entry['repo_id']=='Qwen/Qwen3-8B' or entry.get('status')!='downloaded': continue
        model=entry['repo_id'].split('/')[-1]
        key={'Qwen2.5-7B-Instruct':'qwen25','DeepSeek-R1-Distill-Qwen-7B':'deepseek-r1','Mistral-7B-Instruct-v0.3':'mistral'}[model]
        shared={'model_path':entry['local_path'],'served_name':model,'batch':4,'tp':2,'reasoning':key=='deepseek-r1'}
        for int8,batch in [(False,4),(True,4),(True,8)]:
            name=key+('-int8' if int8 else '-bf16')+'-tp2-b'+str(batch)
            case={**shared,'name':name,'batch':batch,'int8':int8,'artifact':str(ART/('exp-20261030-'+name))}
            if int8: case['quant_path']=entry['local_path']+'-int8-exp20261030'
            run_case(case)
    emit('queue_finished',passed=sum(r['status']=='passed' for r in RESULTS),failed=sum(r['status']=='failed' for r in RESULTS))


if __name__=='__main__':
    def terminate(signum,frame): raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,terminate)
    main()
