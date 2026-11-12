"""Long teacher-forced state continuity, bucket transitions and request churn.

The CPU oracle is the official HF model's full causal sequence computation.
This is separately labelled from a token-by-token HF oracle; an incremental
cross-check at positions127..129 records the BF16 execution-order difference.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import torch
from .numeric import metrics

ROOT = Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
DEST = ROOT/'qwen35-baseline'
CRITERIA = {'cosine_min':.995,'kl_max_nats':.02,'max_abs_error_max':2.,
            'argmax_match_fraction_min':.9,'all_reference_margins_gt_0.5_must_match':True}
LENGTH = 2060
STEPS = 1541
MARKS = sorted(set([0,22,30,31,32,33,63,64,65,126,127,128,129,255,256,257,
                    510,511,512,513,1023,1024,1025,1542,1543,1668,1669,
                    2047,2048,2049,2051,2052,2053]))


def passed(rows):
    return all(r['cosine']>=.995 and r['kl_reference_to_neuron']<=.02 and r['max_abs_error']<=2
               and (r['reference_margin']<=.5 or r['argmax_match']) for r in rows) and \
        sum(r['argmax_match'] for r in rows)/len(rows)>=.9


def reference():
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
    torch.set_num_threads(4)
    source_hash = hashlib.sha256(Path(hf.__file__).read_bytes()).hexdigest()
    assert source_hash=='41a55adcfaebf92a762de7ca1c5c456310d3e03c64d4413a90ccf098c00dc6d8'
    path='/mnt/nvme/models/Qwen3.5-2B'
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
    model=hf.Qwen3_5ForConditionalGeneration.from_pretrained(path,dtype=torch.bfloat16,
        attn_implementation='eager',local_files_only=True).eval()
    prefixes=['Explain how a library organizes books and helps readers find them.',
              '请解释语言模型的输入处理、逐字生成和请求缓存，给出具体例子。']
    paragraphs=[
        'A library groups books by subject. Each shelf has a label. Readers can search the catalog by author, title, or topic. A librarian checks the request, finds the shelf, and records a loan. On Monday 12 books were borrowed; on Tuesday 7 were returned. The system keeps each reader record separate. ',
        '模型首先处理输入文本，把词元转换成向量。接着每一层更新隐藏状态，并保存后续步骤需要的缓存。生成时每次只加入一个词元。不同请求的长度可以不同，缓存必须按请求编号分别保存。假设甲请求已有128个词元，乙请求已有512个词元，新增内容不能覆盖另一条请求。']
    rows=[]
    with torch.inference_mode():
        for i,(prompt,paragraph) in enumerate(zip(prefixes,paragraphs)):
            tokenized=tok.apply_chat_template([{'role':'user','content':prompt}],
                tokenize=True,add_generation_prompt=True,enable_thinking=False)
            prefix=tokenized['input_ids'] if hasattr(tokenized,'keys') else tokenized
            if prefix and isinstance(prefix[0],list):prefix=prefix[0]
            body=tok.encode(paragraph,add_special_tokens=False)
            ids=(prefix+body*((LENGTH//len(body))+1))[:LENGTH]
            assert len(ids)==LENGTH
            tensor=torch.tensor([ids]);start=time.monotonic()
            output=model(input_ids=tensor,use_cache=False,logits_to_keep=torch.tensor(MARKS))
            selected=output.logits[0].float().cpu();del output
            # Same official model in its actual incremental mode, short window.
            out=model(input_ids=tensor[:,:127],use_cache=True,logits_to_keep=1)
            cross=[]
            for pos in (127,128,129):
                out=model(input_ids=tensor[:,pos:pos+1],past_key_values=out.past_key_values,
                          use_cache=True,logits_to_keep=1)
                cross.append({'position':pos,**metrics(out.logits[0,-1],selected[MARKS.index(pos)])})
            del out
            row={'id':i,'input_ids':ids,'positions':MARKS,'logits':selected,
                 'incremental_crosscheck':cross,'seconds':time.monotonic()-start}
            rows.append(row)
            print(json.dumps({'fixture':i,'seconds':row['seconds'],'incremental_crosscheck_passed':passed(cross)}),flush=True)
    DEST.mkdir(exist_ok=True)
    torch.save(rows,DEST/'long-decode-hf-reference.pt')
    manifest={'source_model':path,'revision':'15852e8c16360a2fea060d615a32b45270f8a8fc',
              'reference_source_sha256':source_hash,'sequence_length':LENGTH,'positions':MARKS,
              'criteria':CRITERIA,'scope':'Original HF full causal teacher-forced sequence, selected vocabulary logits. Separate incremental cross-check127..129 retained; not an oracle generated token-by-token for all2060positions.',
              'incremental_crosscheck_passed':all(passed(x['incremental_crosscheck']) for x in rows),
              'fixtures':[{k:v for k,v in x.items() if k!='logits'} for x in rows]}
    (DEST/'long-decode-hf-reference.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))


def neuron(case_path, result_dir):
    from types import SimpleNamespace
    from .runtime import load_admitted
    from vllm_neuron.worker.neuronx_distributed_model_loader import NeuronModelBase
    torch.set_num_threads(1)
    case=json.loads(Path(case_path).read_text())
    result_dir=Path(result_dir)
    result_dir.mkdir(exist_ok=True)
    evidence=torch.load(DEST/'long-decode-hf-reference.pt',weights_only=True)
    model=load_admitted(case['model_path'], case['artifact'])
    cfg=model.config
    assert cfg.neuron_config.batch_size==4 and cfg.neuron_config.max_length>=LENGTH
    model.reset()
    order=SimpleNamespace(is_reorder_needed=True,_sort_inputs=NeuronModelBase._sort_inputs)
    def forward(ids,positions,slots,prefill=False):
        mask=torch.ones_like(ids) if prefill else (torch.arange(int(positions.max())+1)[None,:]<=positions).long()
        with NeuronModelBase._reordered(order,slots,input_ids=ids,attention_mask=mask,
            position_ids=positions,sampling_params=model.default_sampling_params[:ids.shape[0]]) as (s,inputs,restore):
            return restore(model(seq_ids=s,**inputs).logits)
    refs=[0,1,0,1];lengths=[23,127,511,1];slots=[3,1,0,2]
    rows=[];captured=[];start=time.monotonic()
    def record(logits,ref,pos,slot,phase):
        expected=evidence[ref]['logits'][MARKS.index(pos)]
        row={'fixture':ref,'position':pos,'slot':slot,'phase':phase,**metrics(logits,expected)}
        rows.append(row);captured.append({**{k:row[k] for k in ('fixture','position','slot','phase')},'logits':logits.float().cpu()})
        print(json.dumps(row),flush=True)
    with torch.inference_mode():
        for ref,length,slot in zip(refs,lengths,slots):
            ids=torch.tensor([evidence[ref]['input_ids'][:length]])
            out=forward(ids,torch.arange(length)[None],torch.tensor([slot]),True)
            record(out[0,-1],ref,length-1,slot,'prefill')
        for step in range(STEPS):
            positions=[length+step for length in lengths]
            ids=torch.tensor([[evidence[ref]['input_ids'][pos]] for ref,pos in zip(refs,positions)])
            out=forward(ids,torch.tensor(positions)[:,None],torch.tensor(slots))
            for i,(ref,pos,slot) in enumerate(zip(refs,positions,slots)):
                if pos in MARKS:record(out[i,-1],ref,pos,slot,'long_decode')
            if step%128==0:
                print(json.dumps({'steps':step+1,'max_position':max(positions),'elapsed_s':time.monotonic()-start}),flush=True)
        # Reuse one slot, preserving the other three ongoing recurrent histories.
        refs[0]=1;lengths=[31]+[x+STEPS for x in lengths[1:]]
        ids=torch.tensor([evidence[1]['input_ids'][:31]])
        out=forward(ids,torch.arange(31)[None],torch.tensor([slots[0]]),True)
        record(out[0,-1],1,30,slots[0],'slot_reuse_prefill')
        ids=torch.tensor([[evidence[ref]['input_ids'][pos]] for ref,pos in zip(refs,lengths)])
        out=forward(ids,torch.tensor(lengths)[:,None],torch.tensor(slots))
        for i,(ref,pos,slot) in enumerate(zip(refs,lengths,slots)):
            record(out[i,-1],ref,pos,slot,'decode_after_slot_reuse')
        # Three active requests cause NxDI to pad the fourth with inactive ID-1.
        positions=[p+1 for p in lengths[1:]]
        ids=torch.tensor([[evidence[ref]['input_ids'][pos]] for ref,pos in zip(refs[1:],positions)])
        out=forward(ids,torch.tensor(positions)[:,None],torch.tensor(slots[1:]))
        for i,(ref,pos,slot) in enumerate(zip(refs[1:],positions,slots[1:])):
            record(out[i,-1],ref,pos,slot,'partial_batch')
        # Resume the temporarily inactive request and verify its state survived.
        ids=torch.tensor([[evidence[1]['input_ids'][32]]])
        out=forward(ids,torch.tensor([[32]]),torch.tensor([slots[0]]))
        record(out[0,-1],1,32,slots[0],'resume_inactive')
    reference_info=json.loads((DEST/'long-decode-hf-reference.json').read_text())
    result={'passed':passed(rows) and reference_info['incremental_crosscheck_passed'],
            'neuron_vs_full_causal_oracle_passed':passed(rows),
            'reference_incremental_crosscheck_passed':reference_info['incremental_crosscheck_passed'],
            'criteria':CRITERIA,'case':case,'rows':rows,
            'teacher_forced_decode_steps':STEPS,'elapsed_s':time.monotonic()-start,
            'scope':'Original HF full causal oracle vs Neuron iterative decoding; mixed lengths and permuted slots,128/256/512/1024/2048boundaries, single-token prefill, slot reuse, partial batch and inactive resume. This is a correctness check, not throughput benchmarking.'}
    (result_dir/'long-decode-validation.json').write_text(json.dumps(result,indent=2))
    torch.save(captured,result_dir/'long-decode-validation.pt')
    if not result['passed']:raise SystemExit('Long-decode strict numerical gate failed')


if __name__=='__main__':
    raise SystemExit('Use entry.run_model_stage with the root execution guard')
