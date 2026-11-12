"""Real-weight reference/Neuron logits, incremental decode and mixed-length cache tests.

Run CPU reference after compilation finishes; run Neuron phase with no server
occupying the chip. References and outputs are persisted for independent review.
"""
import argparse
import json
import random
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT=Path('/home/ubuntu/experiments/trn1-latest-models-20261031')

def fixtures(tokenizer):
    texts=[
        '法国的首都是哪里？只回答城市名。',
        'Calculate 37 * 8 - 119. Give only the number.',
        '将以下信息转为 JSON：姓名 Alice，年龄 27，城市上海。',
        'Write a Python function that removes duplicate integers while preserving first occurrence order.',
        'Explain the difference between prefill and autoregressive decoding in a language model.',
        'Translate into English: 昨天我和朋友一起去图书馆借了三本书。',
        'A box has 12 red balls and 8 blue balls. What fraction are blue? Simplify your answer.',
        'Read the records and return only the value of key target.\n'+''.join(f'item{i}: value{i}\n' for i in range(200))+'target: 483729\n',
    ]
    out=[]
    for i,text in enumerate(texts):
        ids=tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
        assert 1<len(ids)<1900
        out.append({'id':i,'text':text,'input_ids':ids})
    assert len(out[-1]['input_ids'])>1024, 'Long fixture must exercise the compiled 2048-token bucket'
    return out

def reference(args):
    tokenizer=AutoTokenizer.from_pretrained(args.model)
    cases=fixtures(tokenizer)
    torch.set_num_threads(4)
    model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,attn_implementation='eager').eval()
    evidence=[]
    with torch.inference_mode():
        for case in cases:
            start=time.perf_counter()
            ids=torch.tensor([case['input_ids']],dtype=torch.long)
            result=model(input_ids=ids,use_cache=True,logits_to_keep=1)
            cache=result.past_key_values
            frames=[result.logits[0,-1,:].float().cpu()]
            continuation=[]
            for step in range(3):
                token=int(frames[-1].argmax())
                continuation.append(token)
                result=model(input_ids=torch.tensor([[token]]),past_key_values=cache,use_cache=True,logits_to_keep=1)
                cache=result.past_key_values
                frames.append(result.logits[0,-1,:].float().cpu())
            evidence.append({**case,'continuation_ids':continuation,'logits':torch.stack(frames),'seconds':time.perf_counter()-start})
            print(json.dumps({'reference_fixture':case['id'],'prompt_tokens':len(case['input_ids']),'seconds':evidence[-1]['seconds']}),flush=True)
    torch.save(evidence,args.reference)
    manifest={'model':args.model,'reference':args.reference,'fixtures':len(evidence),'logit_frames':4*len(evidence),'dtype':'BF16 weights and execution, float32 exported logits','attention':'HF eager','source':'Installed Transformers GraniteForCausalLM; original official weights','case_details':[{k:v for k,v in row.items() if k!='logits'} for row in evidence]}
    Path(args.reference+'.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))

def metrics(got,ref):
    got=got.float().flatten()[:ref.numel()]
    ref=ref.float().flatten()
    assert got.shape==ref.shape and torch.isfinite(got).all()
    logp=torch.log_softmax(ref,dim=-1)
    logq=torch.log_softmax(got,dim=-1)
    top=torch.topk(ref,2)
    return {'max_abs_error':(got-ref).abs().max().item(),'mean_abs_error':(got-ref).abs().mean().item(),
        'cosine':torch.nn.functional.cosine_similarity(got,ref,dim=0).item(),
        'kl_reference_to_neuron':(logp.exp()*(logp-logq)).sum().item(),
        'argmax_match':bool(got.argmax()==ref.argmax()),'reference_margin':(top.values[0]-top.values[1]).item(),
        'reference_top1':int(ref.argmax()),'neuron_top1':int(got.argmax())}

def neuron(args):
    from .runtime import load_admitted
    from vllm_neuron.worker.neuronx_distributed_model_loader import NeuronModelBase
    from types import SimpleNamespace
    torch.set_num_threads(1)
    evidence=torch.load(args.reference,weights_only=True)
    model=load_admitted(args.model, args.artifacts)
    cfg=model.config
    batch=cfg.neuron_config.batch_size
    assert batch>=4 and cfg.neuron_config.on_device_sampling_config is None
    # Contiguous NxDI caches require canonical slot order at full batch size.
    # Use the exact vLLM-Neuron serving reorder/restore implementation; raw
    # NxDI forward does not perform it for a full batch. The initial deliberately
    # unsorted direct call is preserved as a diagnostic, not a model failure.
    serving_order=SimpleNamespace(is_reorder_needed=True,_sort_inputs=NeuronModelBase._sort_inputs)
    def forward_logits(ids,mask,positions,slots):
        with NeuronModelBase._reordered(serving_order,slots,input_ids=ids,attention_mask=mask,position_ids=positions,sampling_params=model.default_sampling_params[:ids.shape[0]]) as (ordered_slots,inputs,restore):
            output=model(seq_ids=ordered_slots,**inputs)
            return restore(output.logits)
    rows=[]
    captured=[]
    # Mix short and long prompts, then deliberately permute cache slots.
    base_order=[0,5,2,4,1,3,0,2] if args.adapter in ('qwen35_text','lfm3','gemma4_text','ministral3_text','smollm3_text','phi4_text','olmo3_text','nanbeige42_text','rnj15_text','nemotron9b_text') else [0,7,2,5,4,1,6,3]
    # At B>8 repeat fixtures to occupy every compiled cache slot. Distinct slots
    # with identical inputs also test that request state remains independent.
    order=(base_order*max(1,(batch+7)//8))[:max(8,batch)]
    for start in range(0,len(order),batch):
        group=[evidence[i] for i in order[start:start+batch]]
        slots=([3,1,0,2] if batch==4 else random.Random(20261031).sample(range(batch),batch))[:len(group)]
        model.reset()
        for case,slot in zip(group,slots):
            ids=torch.tensor([case['input_ids']],dtype=torch.long)
            output=forward_logits(ids,torch.ones_like(ids),torch.arange(ids.shape[1])[None,:],torch.tensor([slot]))
            logits=output[0,-1,:].float()
            captured.append({'fixture':case['id'],'phase':'prefill','step':0,'logits':logits})
            row={'fixture':case['id'],'phase':'prefill','slot':slot,'step':0,**metrics(logits,case['logits'][0])}
            rows.append(row);print(json.dumps(row),flush=True)
        for step in range(3):
            ids=torch.tensor([[case['continuation_ids'][step]] for case in group])
            positions=torch.tensor([[len(case['input_ids'])+step] for case in group])
            mask=(torch.arange(int(positions.max())+1)[None,:]<=positions).long()
            output=forward_logits(ids,mask,positions,torch.tensor(slots))
            for index,case in enumerate(group):
                logits=output[index,-1,:].float()
                captured.append({'fixture':case['id'],'phase':'mixed_length_decode','step':step+1,'logits':logits})
                row={'fixture':case['id'],'phase':'mixed_length_decode','slot':slots[index],'step':step+1,**metrics(logits,case['logits'][step+1])}
                rows.append(row);print(json.dumps(row),flush=True)
    criteria={'cosine_min':.995,'kl_max_nats':.02,'max_abs_error_max':2.,'argmax_match_fraction_min':.9,'all_reference_margins_gt_0.5_must_match':True}
    passed=all(r['cosine']>=.995 and r['kl_reference_to_neuron']<=.02 and r['max_abs_error']<=2 and (r['reference_margin']<=.5 or r['argmax_match']) for r in rows) and sum(r['argmax_match'] for r in rows)/len(rows)>=.9
    precision='INT8 weights with BF16 embedding/lm_head/activations' if cfg.neuron_config.quantized else 'BF16'
    report={'passed':passed,'precision':precision,'compiled_batch_size':batch,'requests_tested':len(order),'unique_prompts':len(set(order)),'criteria_set_before_measurement':criteria,'model':args.model,'artifacts':args.artifacts,'reference':args.reference,'serving_order_implementation':'vllm_neuron.worker.neuronx_distributed_model_loader.NeuronModelBase._reordered (unchanged installed implementation)','scope':precision+' full-vocabulary logits against original HF BF16, real prompts × prefill and 3 teacher-forced decode steps, repeating prompts to exercise all slots; mixed lengths and permuted request/cache slots using the mandatory vLLM serving reorder/restore. Identical strict numerical thresholds are retained for both precisions; failures must be reported and investigated with downstream quality evidence. This does not certify broad downstream model quality.','rows':rows}
    Path(args.output).write_text(json.dumps(report,indent=2))
    torch.save(captured,args.output+'.pt')
    if not passed:raise SystemExit('Full-model numerical gate failed; inspect evidence before performance recommendations')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('phase',choices=['reference','neuron'])
    ap.add_argument('--model',required=True)
    ap.add_argument('--reference',required=True)
    ap.add_argument('--artifacts')
    ap.add_argument('--output')
    ap.add_argument('--adapter',choices=['granite','granite_flash','qwen35_text','lfm3','gemma4_text','ministral3_text','smollm3_text','phi4_text','olmo3_text','nanbeige42_text','rnj15_text','nemotron9b_text','qwen3_native','deepseek_qwen3_yarn'],default='granite')
    args=ap.parse_args()
    (reference if args.phase=='reference' else neuron)(args)

if __name__=='__main__':raise SystemExit('Use entry.run_model_stage with the root execution guard')
