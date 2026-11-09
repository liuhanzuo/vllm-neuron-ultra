"""Frozen full-model stage loops. Native imports happen only after worker lease."""
import ast
import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace
import time
import nemotron9b_package_protocol_v19 as p
from nemotron9b_package_science_v19 import metrics,passed,graph_routes,validate_states,expected_bucket,CRITERIA,STEPS,LENGTH,MARKS,PREFIXES
SPEC=DEST=None

def configure(spec):
    global SPEC,DEST
    SPEC=spec;DEST=Path(spec['evidence_root'])

def context():
    global torch,NeuronModelBase
    import torch
    from vllm_neuron.models.nemotron9b import register
    register()
    from vllm_neuron.models.nemotron9b.modeling import NeuronNemotron9BForCausalLM as cls
    from vllm_neuron.models.nemotron9b.identity import build_identity
    from vllm_neuron.models.nemotron9b.artifacts import admit_existing
    from vllm_neuron.worker.neuronx_distributed_model_loader import NeuronModelBase
    c=p.case(SPEC);cfg=cls.get_config_cls().load(c['artifact'])
    body=json.loads(cfg.to_json_string());body['metadata']=None
    admit_existing(c['artifact'],build_identity(c['model_path'],body))
    torch.set_num_threads(1);model=cls(c['model_path'],config=cfg);model.load(c['artifact']);model.reset()
    return c,model,cfg

def save(path,value):p.save(path,value)

def binding_metadata(model):
    from vllm_neuron.models.nemotron9b.state import bindings
    return bindings(model)

def state_layout():
    from vllm_neuron.models.nemotron9b.state import state_layout as layout
    return layout()

def clear_states(model,expected_bindings=None):
    from vllm_neuron.models.nemotron9b.state import clear_for_cold_validation
    before=binding_metadata(model)
    if expected_bindings is not None:assert before==expected_bindings
    rows=[]
    for rank,tensors in enumerate(model.state):
        for name,value in sorted(tensors.items()):
            old=value.detach().cpu().contiguous()
            rows.append(dict(rank=rank,name=name,before_nonzero_bytes=int(torch.count_nonzero(old.view(torch.uint8))),before_sha256=digest(old)))
    receipt=clear_for_cold_validation(model,before)
    return dict(receipt,complete=True,bindings=before,rows=rows,states_nonzero_before_clear=sum(x['before_nonzero_bytes']>0 for x in rows))

def cold_jobs():
    metadata=p.read(SPEC['reference_metadata']['cold']['path'],SPEC['reference_metadata']['cold']['sha256'])
    return metadata['jobs']

def validate_copy(report):
    assert report['complete']and report['passed']and report['state_buffers']==124
    assert report['fill_roundtrip_comparisons']==report['raw_zero_comparisons']==248
    assert len(report['patterns'])==2 and report['same_tensor_bindings_preserved']and report['graph_models_loaded']==0
    for item in report['patterns']:
        assert len(item['fills'])==len(item['zero_checks'])==124
        assert all(x['passed']and x['actual_sha256']==x['expected_sha256']for x in item['fills'])
        assert all(x['passed']and x['nonzero_bytes']==0 for x in item['zero_checks'])

def validate_cold(proof):
    assert proof['complete']and proof['case']==p.case(SPEC)and proof['variant']=='nki'
    assert proof['graph_routes']['complete']and len(proof['graph_routes']['graphs'])==12
    assert [x['prefix']for x in proof['experiments']]==list(PREFIXES)
    rows=[]
    for item in proof['experiments']:
        validate_states(item['initial_raw_zero_states'],zero=True);validate_states(item['final_states'])
        assert len(item['rows'])==12 and {(r['slot'],r['step'])for r in item['rows']}=={(s,t)for s in range(4)for t in range(3)}
        assert len(item['duplicate_checks'])==6 and all(r['bitwise_equal']for r in item['duplicate_checks'])
        for row in item['rows']:
            assert row['prefix']==item['prefix']and row['valid']and row['fixture']==row['slot']%2
            assert row['position']==item['prefix']-1+row['step']
            expected=expected_bucket(item['prefix']+row['step'])
            assert row['route']['tag']==('token_generation_model'if row['step']else'context_encoding_model')
            assert row['route']['mask_length'if row['step']else'input_tokens']==expected
            rows.append(row['metrics'])
    assert len(rows)==84 and passed(rows)
    receipts=proof['state_reset_receipts'];assert len(receipts)==7
    assert all(x['states_written']==124 and x['bindings']==receipts[0]['bindings']for x in receipts)
    assert all(x['states_nonzero_before_clear']>0 for x in receipts[1:])
    assert proof['same124_tensor_bindings_across_all_clears']

def digest(value):
    import torch
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

def state_rows(model,zero=False):
    import torch
    rows=[]
    for rank,tensors in enumerate(model.traced_model.nxd_model.state):
        for name,value in sorted(tensors.items()):
            cpu=value.detach().cpu().contiguous()
            row=dict(rank=rank,name=name,shape=list(cpu.shape),dtype=str(cpu.dtype))
            if zero:
                row['nonzero_bytes']=int(torch.count_nonzero(cpu.view(torch.uint8)))
            else:
                row.update(nan_elements=int(torch.isnan(cpu).sum()),inf_elements=int(torch.isinf(cpu).sum()))
            rows.append(row);del cpu
    return rows

def state_copy():
    output = DEST / 'state-copy.json'; assert not output.exists()
    report = dict(complete=False, passed=False, patterns=[], source_sha256=SPEC['source_sha256'],
        full_model_qualified=False, performance_measured=False)
    try:
        global torch
        import torch
        import torch_neuronx
        torch.set_num_threads(1)
        layout = state_layout()
        states = [{}, {}]
        for row in layout:
            dtype = {'torch.bfloat16': torch.bfloat16, 'torch.float32': torch.float32}[row['dtype']]
            states[row['rank']][row['name']] = torch.zeros(row['shape'], dtype=dtype).to('privateuseone:' + str(row['rank']))
        model = SimpleNamespace(state=states); bindings = binding_metadata(model)
        for pattern in range(2):
            fills = []
            for rank, tensors in enumerate(states):
                for index, (name, target) in enumerate(sorted(tensors.items())):
                    words = torch.arange(target.numel(), dtype=torch.int64)
                    salt = 17 + pattern * 65537 + rank * 131 + index * 31
                    if target.dtype == torch.bfloat16:
                        source = ((words * 637 + salt) % 65536).to(torch.int16).view(torch.bfloat16).reshape(target.shape)
                    else:
                        source = (words * 2654435761 + salt).to(torch.int32).view(torch.float32).reshape(target.shape)
                    target.copy_(source)
                    actual = target.detach().cpu().contiguous()
                    assert torch.equal(actual.view(torch.uint8), source.contiguous().view(torch.uint8)), (pattern, rank, name)
                    fills.append(dict(rank=rank, name=name, shape=list(source.shape), dtype=str(source.dtype),
                        bytes_compared=source.numel() * source.element_size(), expected_sha256=digest(source),
                        actual_sha256=digest(actual), passed=True))
                    del words, source, actual
            assert binding_metadata(model) == bindings
            receipt = clear_states(model, bindings)
            assert receipt['states_nonzero_before_clear'] == 124
            assert [(x['rank'], x['name'], x['before_sha256']) for x in receipt['rows']] == [
                (x['rank'], x['name'], x['expected_sha256']) for x in fills]
            zero_checks = []
            for rank, tensors in enumerate(states):
                for name, target in sorted(tensors.items()):
                    actual = target.detach().cpu().contiguous()
                    nonzero = int(torch.count_nonzero(actual.view(torch.uint8)))
                    assert nonzero == 0, (pattern, rank, name)
                    zero_checks.append(dict(rank=rank, name=name, shape=list(actual.shape), dtype=str(actual.dtype),
                        bytes_checked=actual.numel() * actual.element_size(), nonzero_bytes=nonzero, passed=True))
                    del actual
            assert binding_metadata(model) == bindings
            report['patterns'].append(dict(pattern=pattern, fills=fills, reset=receipt, zero_checks=zero_checks))
            save(output, report)
        report.update(complete=True, passed=True, initial_bindings=bindings, patterns_tested=2,
            state_buffers=124, fill_roundtrip_comparisons=248, raw_zero_comparisons=248,
            same_tensor_bindings_preserved=True, graph_models_loaded=0)
        validate_copy(report)
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        save(output, report)

def numeric():
    case,model,cfg=context();batch=4
    args=SimpleNamespace(model=case['model_path'],artifacts=case['artifact'],
        reference=SPEC['references']['numeric'],output=str(DEST/'numeric.json'))
    evidence=torch.load(args.reference,weights_only=True)
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
    base_order=[0,5,2,4,1,3,0,2]
    # At B>8 repeat fixtures to occupy every compiled cache slot. Distinct slots
    # with identical inputs also test that request state remains independent.
    order=(base_order*max(1,(batch+7)//8))[:max(8,batch)]
    try:
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
    finally:
        save(DEST/'numeric-partial.json',dict(complete=False,rows=rows,retained_partial_evidence=True))
        torch.save(captured,DEST/'numeric-partial.pt')
    criteria={'cosine_min':.995,'kl_max_nats':.02,'max_abs_error_max':2.,'argmax_match_fraction_min':.9,'all_reference_margins_gt_0.5_must_match':True}
    passed=all(r['cosine']>=.995 and r['kl_reference_to_neuron']<=.02 and r['max_abs_error']<=2 and (r['reference_margin']<=.5 or r['argmax_match']) for r in rows) and sum(r['argmax_match'] for r in rows)/len(rows)>=.9
    precision='INT8 weights with BF16 embedding/lm_head/activations' if cfg.neuron_config.quantized else 'BF16'
    report={'passed':passed,'precision':precision,'compiled_batch_size':batch,'requests_tested':len(order),'unique_prompts':len(set(order)),'criteria_set_before_measurement':criteria,'model':args.model,'artifacts':args.artifacts,'reference':args.reference,'serving_order_implementation':'vllm_neuron.worker.neuronx_distributed_model_loader.NeuronModelBase._reordered (unchanged installed implementation)','scope':precision+' full-vocabulary logits against original HF BF16, real prompts × prefill and 3 teacher-forced decode steps, repeating prompts to exercise all slots; mixed lengths and permuted request/cache slots using the mandatory vLLM serving reorder/restore. Identical strict numerical thresholds are retained for both precisions; failures must be reported and investigated with downstream quality evidence. This does not certify broad downstream model quality.','rows':rows}
    Path(args.output).write_text(json.dumps(report,indent=2))
    torch.save(captured,args.output+'.pt')
    if not passed:raise SystemExit('Full-model numerical gate failed; inspect evidence before performance recommendations')

def cold():
    variant='nki'
    case,model,cfg=context()
    plan={'source_sha256':SPEC['source_sha256']}
    source=torch.load(SPEC['references']['cold'],weights_only=True,map_location='cpu')
    assert [r['job']for r in source]==cold_jobs()
    evidence={(r['job']['prefix'],r['job']['fixture']):r for r in source}
    dest=DEST
    order=SimpleNamespace(is_reorder_needed=True,_sort_inputs=NeuronModelBase._sort_inputs)
    output=dest/'cold-buckets.json';assert not output.exists()
    report=dict(complete=False,variant=variant,case=case,source_sha256=SPEC['source_sha256'],reset_source_sha256=plan['source_sha256'],criteria=CRITERIA,
        graph_routes=graph_routes(model),experiments=[],state_reset_receipts=[],scope='Seven in-place-cleared device caches after one model load, '
        '124 raw-zero states each;84 complete-vocabulary outputs vs exact native cached prefix oracle. '
        'Two fixtures repeated in slots0/2 and1/3;all12 graph routes;two B4 decode steps per prefix. '
        'Preserves original BF16/FP32 semantics and standard sampler; no throughput measured here.')
    captured=[];route={}
    def hook(wrapper):
        original=wrapper._forward
        def observe(*inputs):
            signature=str([list(v.shape)for v in inputs]);chosen=model.traced_model.nxd_model.input_shape_map[signature]
            route.update(tag=chosen[0],index=chosen[1],input_tokens=int(inputs[0].shape[1]),mask_length=int(inputs[1].shape[1]))
            return original(*inputs)
        wrapper._forward=observe
    hook(model.context_encoding_model);hook(model.token_generation_model)
    def forward(ids,positions,slots,prefill=False):
        route.clear();mask=torch.ones_like(ids) if prefill else (torch.arange(int(positions.max())+1)[None]<=positions).long()
        with NeuronModelBase._reordered(order,slots,input_ids=ids,attention_mask=mask,position_ids=positions,
            sampling_params=model.default_sampling_params[:ids.shape[0]])as(s,inputs,restore):
            return restore(model(seq_ids=s,**inputs).logits)
    def record(value,prefix,step,slot):
        expected=evidence[prefix,slot%2]['logits'][step]
        got=value.float().flatten()[:expected.numel()].cpu().contiguous()
        valid=got.numel()==expected.numel() and bool(torch.isfinite(got).all())
        row=dict(prefix=prefix,step=step,slot=slot,fixture=slot%2,position=prefix-1+step,
                 valid=valid,route=dict(route),metrics=metrics(got,expected) if valid else None,
                 nan_elements=int(torch.isnan(got).sum()),inf_elements=int(torch.isinf(got).sum()))
        captured.append({k:row[k]for k in ('prefix','step','slot','fixture','position')}|dict(logits=got.clone()))
        return row
    try:
        with torch.inference_mode():
            for prefix in PREFIXES:
                bindings = report['state_reset_receipts'][0]['bindings'] if report['state_reset_receipts'] else binding_metadata(model.traced_model.nxd_model)
                report['state_reset_receipts'].append(clear_states(model.traced_model.nxd_model, bindings));model.reset()
                item=dict(prefix=prefix,initial_raw_zero_states=state_rows(model,zero=True),rows=[],duplicate_checks=[])
                report['experiments'].append(item);save(output,report);validate_states(item['initial_raw_zero_states'],zero=True)
                current=[]
                for slot in range(4):
                    ids=torch.tensor([evidence[prefix,slot%2]['job']['input_ids'][:prefix]])
                    value=forward(ids,torch.arange(prefix)[None],torch.tensor([slot]),True)[0,-1].clone()
                    current.append(value);item['rows'].append(record(value,prefix,0,slot));save(output,report)
                for step in range(3):
                    if step:
                        position=prefix+step-1
                        ids=torch.tensor([[evidence[prefix,slot%2]['job']['input_ids'][position]]for slot in range(4)])
                        values=forward(ids,torch.full((4,1),position,dtype=torch.long),torch.arange(4))
                        current=[values[slot,-1].clone()for slot in range(4)]
                        item['rows'] += [record(value,prefix,step,slot)for slot,value in enumerate(current)]
                    for a,b in ((0,2),(1,3)):
                        item['duplicate_checks'].append(dict(step=step,slots=[a,b],
                            bitwise_equal=torch.equal(current[a].contiguous().view(torch.uint8),current[b].contiguous().view(torch.uint8))))
                    save(output,report)
                item['final_states']=state_rows(model);save(output,report)
                print(json.dumps(dict(prefix=prefix,frames=len(captured))),flush=True)
        assert len(captured)==84;report['complete']=True
        report['same124_tensor_bindings_across_all_clears'] = all(r['bindings'] == report['state_reset_receipts'][0]['bindings'] for r in report['state_reset_receipts'])
        validate_cold(report);report['passed']=True
    except BaseException as error:
        report.update(passed=False,error=repr(error));raise
    finally:
        torch.save(captured,output.with_suffix('.pt'));save(output,report)

def long_decode():
    case,model,cfg=context()
    evidence=torch.load(SPEC['references']['long'],weights_only=True,map_location='cpu')
    order=SimpleNamespace(is_reorder_needed=True,_sort_inputs=NeuronModelBase._sort_inputs)
    def forward(ids,positions,slots,prefill=False):
        mask=torch.ones_like(ids) if prefill else (torch.arange(int(positions.max())+1)[None,:]<=positions).long()
        with NeuronModelBase._reordered(order,slots,input_ids=ids,attention_mask=mask,
            position_ids=positions,sampling_params=model.default_sampling_params[:ids.shape[0]]) as (s,inputs,restore):
            return restore(model(seq_ids=s,**inputs).logits)
    refs=[0,1,2,3];lengths=[23,127,1023,1];slots=[3,1,0,2]
    rows=[];captured=[];start=time.monotonic()
    def record(logits,ref,pos,slot,phase):
        expected=evidence[ref]['logits'][evidence[ref]['positions'].index(pos)]
        row={'trajectory':ref,'fixture':evidence[ref]['fixture'],'position':pos,'slot':slot,'phase':phase,**metrics(logits,expected)}
        rows.append(row);captured.append({**{k:row[k] for k in ('trajectory','fixture','position','slot','phase')},'logits':logits.float().cpu()})
        print(json.dumps(row),flush=True)
    try:
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
            refs[0]=4;lengths=[31]+[x+STEPS for x in lengths[1:]]
            ids=torch.tensor([evidence[4]['input_ids'][:31]])
            out=forward(ids,torch.arange(31)[None],torch.tensor([slots[0]]),True)
            record(out[0,-1],4,30,slots[0],'slot_reuse_prefill')
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
            ids=torch.tensor([[evidence[4]['input_ids'][32]]])
            out=forward(ids,torch.tensor([[32]]),torch.tensor([slots[0]]))
            record(out[0,-1],4,32,slots[0],'resume_inactive')
    finally:
        save(DEST/'long-partial.json',dict(complete=False,rows=rows,retained_partial_evidence=True))
        torch.save(captured,DEST/'long-partial.pt')
    reference_info=json.loads((Path(SPEC['reference_metadata']['long']['path'])).read_text())
    result={'passed':passed(rows) and reference_info['cached_trajectories_complete'],
            'neuron_vs_cached_oracle_passed':passed(rows),
            'reference_cached_trajectories_complete':reference_info['cached_trajectories_complete'],
            'criteria':CRITERIA,'case':case,'rows':rows,
            'teacher_forced_decode_steps':STEPS,'elapsed_s':time.monotonic()-start,
            'scope':'Original native HF cached trajectories vs Neuron iterative decoding;exact prefill boundaries,mixed lengths/permutedslots,128/256/512/1024/2048bucket transitions,FP32SSMstate,NoPEattention,ReLU2MLP,single-tokenprefill,slotreuse,partialbatch/inactiveresume. 2041decode steps;correctness only,not throughput.'}
    (DEST/'long-decode-validation.json').write_text(json.dumps(result,indent=2))
    torch.save(captured,DEST/'long-decode-validation.pt')
    if not result['passed']:raise SystemExit('Long-decode strict numerical gate failed')
