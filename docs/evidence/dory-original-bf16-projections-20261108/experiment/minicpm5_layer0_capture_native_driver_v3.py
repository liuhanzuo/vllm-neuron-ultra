"""Original native32 capture loop; injected authenticated runtime only, no CLI.

This deliberately cannot load a model or invoke admission itself. Future root
worker supplies frozen native loader outputs, original payload and permit-bound
writer factory. The full original32 output gate applies to every frame.
"""
import importlib
import minicpm5_python_trace_observer_v2 as observe
from minicpm5_layer0_capture_observer_v2 import SnapshotSink
import minicpm5_layer0_capture_endpoints_v1 as endpoint


def run(torch,model,plan,original,hf_frame_records,native_cache_check,writer_factory,admission_guard,native_source):
    admission_guard();require=endpoint.require
    require(model.config._attn_implementation=='eager' and not model.training,'Original native eager eval route')
    require(len(original)==8 and [x['id'] for x in original]==list(range(8)),'Original8fixtures')
    require(len(plan['frames'])==32 and len(plan['calls'])==14,'Original32/14scope')
    require(len(hf_frame_records)==32,'All retained HFframe metadata')
    module=importlib.import_module(type(model.model.layers[0].self_attn).__module__)
    require(module.__file__==plan['native_provenance']['source'],'Original native source class path')
    mapping={(f['fixture'],f['step']):f for f in plan['frames']};reports=[]
    with observe.installed_native(model.model.layers[0].self_attn,module,torch,native_source['text'],native_source['path'],native_source['sha256'],full=True) as trace_counts:
        with torch.inference_mode():
            for fixture,saved in zip(plan['fixtures'],original):
                require(fixture['id']==saved['id'] and fixture['input_ids']==saved['input_ids'] and fixture['continuation_ids']==saved['continuation_ids'],'Original input fixture changed')
                length=len(fixture['input_ids']);output=None
                for step in range(4):
                    admission_guard();frame=mapping[(fixture['id'],step)];logical=length+step-1
                    if step==0:
                        args=dict(input_ids=torch.tensor([fixture['input_ids']]),attention_mask=torch.ones((1,length),dtype=torch.long),position_ids=torch.arange(length)[None],use_cache=True,logits_to_keep=1)
                    else:
                        token=fixture['continuation_ids'][step-1];require(token==int(saved['logits'][step-1].argmax()),'Original teacher-forced token changed')
                        args=dict(input_ids=torch.tensor([[token]]),attention_mask=torch.ones((1,length+step),dtype=torch.long),position_ids=torch.tensor([[logical]]),past_key_values=output.past_key_values,use_cache=True,logits_to_keep=1)
                    call=next(c['index'] for c in plan['calls'] if frame in c['frames'])
                    writer=writer_factory('HF_native',-1,call,frame_index=frame['index'])
                    try:
                        sink=SnapshotSink(torch,writer.tensor)
                        capture=observe.Capture(torch,sink,'HF_native',-1,[frame],call,'prefill' if step==0 else 'decode',[length-1 if step==0 else 0])
                        capture.position_ids=args['position_ids']
                        with capture.activate():output=model(**args)
                        native_cache_check(output.past_key_values,length+step)
                        receipt=capture.complete();receipt['snapshots']=sink.records
                        observe.validate_capture_summary(receipt,trace_counts['contract'])
                        require(trace_counts['calls']==len(reports)+1 and trace_counts['observed_scale_events']==len(reports)+1,'All original native trace calls and events')
                        logits=output.logits[0,-1].detach().float().cpu().clone()
                        checked=endpoint.native_endpoint(torch,logits,saved['logits'][step],frame,hf_frame_records[frame['index']],plan['hf_reference']['sha256'])
                        reports.append(writer.finish(receipt,checked));del logits
                    finally:writer.close()
                del output
    require(len(reports)==32,'Missing nativeframes');admission_guard()
    return dict(complete=True,frames=32,original_full_vocab_raw_exact=True,reports=reports,scale_observation_schema=observe.SCHEMA,scale_events=32,full_model_qualified=False)
