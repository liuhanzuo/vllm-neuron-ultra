"""Matched long-input/decode and completion-churn HTTP workloads.

Synthetic prompts are trimmed inside a native chat-template token sequence to
make exact input sizes. This file measures serving, not answer correctness.
"""
import concurrent.futures
import json
import math
import statistics
import time
from pathlib import Path
from nemotron9b_package_http_transport_v24 import benchmark,request

def percentile(values,p):
    return sorted(values)[max(0,math.ceil(p*len(values))-1)]

def inputs(case):
    from orchestrate import prompt_tokenizer
    tok=prompt_tokenizer(case)
    text='Describe the patterns in these records and explain your reasoning.\n'+''.join(f'Record {i}: inventory {100+i}, sales {20+i%17}, returns {i%4}.\n' for i in range(400))+'\nGive a detailed explanation.'
    full=tok.apply_chat_template([{'role':'user','content':text}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
    result={}
    for length in [32,128,512,1536]:
        ids=full[:length-16]+full[-16:]
        assert len(ids)==length
        result[length]=ids
    return result

def verify(report):
    groups=[r[k] for r in report['rounds'] for k in ['single','concurrent']]+[report['streaming']]
    for group in groups:
        for row in group['requests']:
            assert row['usage']['completion_tokens']==report['output_tokens_per_request']
            assert row['usage']['prompt_tokens']==len(report['prompt_inputs'][0])

def mixed(case,prompts,sampling,rounds=3):
    sizes=[(32,32),(128,128),(512,256),(1536,128)]*4
    results=[]
    for trial in range(rounds):
        start=time.perf_counter()
        def one(pair):
            n,m=pair
            row=request(8010,case['served_name'],prompts[n],m,stream=True,**sampling)
            assert row['usage']['prompt_tokens']==n and row['usage']['completion_tokens']==m
            return {'input_tokens':n,'output_tokens':m,**row}
        with concurrent.futures.ThreadPoolExecutor(max_workers=case['batch']) as pool:
            rows=list(pool.map(one,sizes))
        elapsed=time.perf_counter()-start
        latency=[r['elapsed_s'] for r in rows];ttft=[1000*r['ttft_s'] for r in rows if r['ttft_s'] is not None]
        result={'trial':trial,'output_tokens':sum(r['output_tokens'] for r in rows),'elapsed_s':elapsed,'throughput_tok_s':sum(r['output_tokens'] for r in rows)/elapsed,'latency_p50_s':statistics.median(latency),'latency_p95_s':percentile(latency,.95),'ttft_p50_ms':statistics.median(ttft),'ttft_p95_ms':percentile(ttft,.95),'requests':rows}
        results.append(result)
        print(json.dumps({'mixed_trial':trial,'throughput_tok_s':result['throughput_tok_s']}),flush=True)
    return {'rounds':results,'sampling':sampling,'max_client_inflight':case['batch'],'requests_per_round':len(sizes),'summary':{'median_tok_s':statistics.median(r['throughput_tok_s'] for r in results)},'note':'Closed-loop client: replace each completed request until 16 finish. Client worker queue time is excluded from per-request latency; aggregate elapsed includes all work. Streaming TTFT is time to first nonempty decoded text chunk, not device first-token latency. p95 uses nearest rank. Mixed cases intentionally reuse slots across different lengths.'}

def evaluate(case,output):
    prompts=inputs(case)
    path=Path(output)
    result={'case':case,'inputs':{str(k):v for k,v in prompts.items()},'workloads':{},'note':'Synthetic token-trimmed native chat prompts, exact input and forced output token counts; template trailing 16 tokens preserved. Throughput only, no quality score.'}
    specs=[('short_decode',128,512,{}),('medium_prompt',512,128,{}),('long_prompt',1536,128,{}),('short_stochastic',128,512,{'temperature':.7,'top_p':.9,'top_k':40})]
    for name,n,m,sampling in specs:
        assert n+m<=case.get('max_model_len',2048)
        print(json.dumps({'workload':name,'input':n,'output':m}),flush=True)
        report=benchmark(8010,case['served_name'],[prompts[n]],case['batch'],output_tokens=m,sampling=sampling)
        verify(report);result['workloads'][name]=report
        path.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    result['workloads']['mixed_lengths']=mixed(case,prompts,{'temperature':.7,'top_p':.9,'top_k':40})
    result['complete']=True;path.write_text(json.dumps(result,ensure_ascii=False,indent=2));return result
