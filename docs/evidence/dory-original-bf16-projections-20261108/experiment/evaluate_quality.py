"""Evaluate a pinned paired-quality suite through the running vLLM HTTP service."""
import argparse
import concurrent.futures
import hashlib
import json
import re
import time
import urllib.request
from collections import defaultdict
from decimal import Decimal,InvalidOperation
from pathlib import Path
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parent
NUM=r'[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'

def decimal(text):
    if not re.fullmatch(NUM,text.strip()):return None
    try:return Decimal(text.strip().replace(',',''))
    except InvalidOperation:return None

def grade(case,text):
    if case['grader']=='gsm8k_numeric_marker':
        matches=re.findall(r'^\s*####[ \t]*('+NUM+r')[ \t]*$',text,re.MULTILINE)
        value=decimal(matches[-1]) if matches else None
        return {'correct':value is not None and value==decimal(case['expected']),'extracted':str(value) if value is not None else None,'format_valid':bool(matches)}
    if case['grader']=='numeric_only':
        value=decimal(text.strip())
        return {'correct':value is not None and value==decimal(case['expected']),'extracted':str(value) if value is not None else None,'format_valid':value is not None}
    if case['grader']!='json_equal':raise ValueError('Unknown grader: '+case['grader'])
    def strict_object(pairs):
        out={}
        for key,value in pairs:
            if key in out:raise ValueError('Duplicate JSON key')
            out[key]=value
        return out
    def reject_constant(value):raise ValueError('Nonstandard JSON constant: '+value)
    try:
        value=json.loads(text.strip(),object_pairs_hook=strict_object,parse_constant=reject_constant)
        # bool is an int subclass; exact serialization distinguishes them.
        correct=json.dumps(value,sort_keys=True,ensure_ascii=False)==json.dumps(case['expected'],sort_keys=True,ensure_ascii=False)
        return {'correct':correct,'extracted':value,'format_valid':True}
    except (ValueError,TypeError):return {'correct':False,'extracted':None,'format_valid':False}

def prepare_inputs(case,suite):
    from orchestrate import prompt_tokenizer
    tok=prompt_tokenizer(case)
    prepared=[]
    for item in suite['cases']:
        ids=tok.apply_chat_template([{'role':'user','content':item['prompt']}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
        if len(ids)+item['max_tokens']>case.get('max_model_len',2048):
            raise ValueError('Quality input exceeds context; do not silently truncate '+item['id'])
        prepared.append((item,ids))
    reference_path={'ministral3_text':ROOT/'ministral3/tokenizer-reference.json',
                    'smollm3_text':ROOT/'smollm3/tokenizer-reference.json',
                    'olmo3_text':ROOT/'olmo3/tokenizer-reference.json',
                    'nanbeige42_text':ROOT/'nanbeige42/tokenizer-reference.json',
                    'rnj15_text':ROOT/'rnj15/tokenizer-reference.json',
                    'nemotron9b_text':ROOT/'nemotron9b/tokenizer-reference.json',
                    'qwen3_native':ROOT/'qwen3_4b/tokenizer-reference.json',
                    'deepseek_qwen3_yarn':ROOT/'deepseek_qwen3_8b/tokenizer-reference.json'}.get(case.get('adapter'))
    if case.get('adapter')=='phi4_text':
        from phi4_variants import identity
        reference_path=identity(case['phi4_variant'])['dest']/'tokenizer-reference.json'
    if reference_path is not None:
        reference=json.loads(reference_path.read_text())
        assert reference['suite_sha256']==hashlib.sha256((ROOT/'quality-suite.json').read_bytes()).hexdigest()
        expected={row['id']:row['input_ids'] for row in reference['fixtures']}
        assert all(ids==expected[item['id']] for item,ids in prepared), 'Quality inputs differ from the pinned native tokenizer'
    return prepared

def evaluate(case,output,port=8010,concurrency=4):
    path=ROOT/'quality-suite.json'
    raw_suite=path.read_bytes();suite=json.loads(raw_suite)
    if concurrency!=suite['protocol']['concurrency']:raise ValueError('Concurrency differs from fixed quality protocol')
    prepared=prepare_inputs(case,suite)
    rows=[]
    def one(pair):
        item,ids=pair
        body={'model':case['served_name'],'prompt':ids,'max_tokens':item['max_tokens'],'temperature':0}
        req=urllib.request.Request(f'http://127.0.0.1:{port}/v1/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        start=time.perf_counter()
        try:
            with urllib.request.urlopen(req,timeout=240) as response:raw=json.load(response)
            choice=raw['choices'][0];text=choice['text']
            return {'id':item['id'],'group':item['group'],'expected':item['expected'],'prompt_ids':ids,'text':text,'elapsed_s':time.perf_counter()-start,'usage':raw.get('usage'),'finish_reason':choice.get('finish_reason'),**grade(item,text)}
        except Exception as e:
            return {'id':item['id'],'group':item['group'],'error':str(e),'correct':False,'elapsed_s':time.perf_counter()-start}
    def save(final=False):
        groups=defaultdict(lambda:{'correct':0,'total':0,'request_errors':0,'length_capped':0})
        for row in rows:
            g=groups[row['group']];g['total']+=1;g['correct']+=int(row['correct']);g['request_errors']+=int('error' in row);g['length_capped']+=int(row.get('finish_reason')=='length')
        if path.read_bytes()!=raw_suite:raise RuntimeError('Quality suite changed during evaluation')
        report={'case':case,'suite_sha256':hashlib.sha256(raw_suite).hexdigest(),'protocol':suite['protocol'],'technical_complete':final and len(rows)==len(prepared) and not any('error' in row for row in rows),'requested_cases':len(prepared),'groups':dict(groups),'correct':sum(x['correct'] for x in rows),'total':len(rows),'rows':rows,'note':'Quality evaluation, not a controlled throughput benchmark. Score alone is not a general accuracy guarantee.'}
        tmp=Path(str(output)+'.tmp');tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2));tmp.replace(output)
        return report
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        for row in pool.map(one,prepared):
            rows.append(row)
            if len(rows)%8==0:save();print(json.dumps({'quality_completed':len(rows),'correct':sum(x['correct'] for x in rows)}),flush=True)
    return save(True)

def compare(left_path,right_path,output):
    left=json.loads(Path(left_path).read_text());right=json.loads(Path(right_path).read_text())
    assert left['technical_complete'] and right['technical_complete']
    assert left['suite_sha256']==right['suite_sha256'] and left['protocol']==right['protocol']
    a={x['id']:x for x in left['rows']};b={x['id']:x for x in right['rows']};assert a.keys()==b.keys()
    assert all(a[k]['prompt_ids']==b[k]['prompt_ids'] for k in a)
    lost=[k for k in a if a[k]['correct'] and not b[k]['correct']]
    gained=[k for k in a if not a[k]['correct'] and b[k]['correct']]
    result={'reference':str(left_path),'candidate':str(right_path),'suite_sha256':left['suite_sha256'],'reference_correct':left['correct'],'candidate_correct':right['correct'],'total':left['total'],'reference_groups':left['groups'],'candidate_groups':right['groups'],'lost_correct_cases':lost,'gained_correct_cases':gained,'identical_texts':sum(a[k]['text']==b[k]['text'] for k in a),'interpretation':'Paired finite-sample regression evidence only. Net score can hide individual losses; inspect both lost and gained cases. No universal quality-equivalence claim.'}
    Path(output).write_text(json.dumps(result,ensure_ascii=False,indent=2));return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case');ap.add_argument('--port',type=int,default=8010);ap.add_argument('--output',required=True);ap.add_argument('--compare',nargs=2);args=ap.parse_args()
    if args.compare:print(json.dumps(compare(*args.compare,args.output),ensure_ascii=False))
    else:
        result=evaluate(json.loads(Path(args.case).read_text()),Path(args.output),args.port)
        print(json.dumps({k:v for k,v in result.items() if k not in ['rows','case','protocol']}))
        if not result['technical_complete']:raise SystemExit(1)

if __name__=='__main__':main()
