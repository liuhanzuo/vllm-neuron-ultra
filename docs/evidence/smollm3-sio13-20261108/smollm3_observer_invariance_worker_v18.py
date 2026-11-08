"""Callable server/client only; root future wrapper owns main/spawn admission.

No CLI, autonomous service start, serial lock, native import or writes on import.
Root wrapper calls server(context) in __main__, server(context,spawned=True)
in __mp_main__, binding the same exact run ID/attempt to the actual group.
"""
import concurrent.futures
import hashlib
import importlib
import json
import os
from pathlib import Path
import runpy
import sys
import threading
import urllib.error
import urllib.request
import smollm3_observer_invariance_protocol_v18 as p
import smollm3_observer_invariance_analysis_v18 as analysis
import run_smollm3_token_capture_v7 as old_worker

_SERVER_STARTED=False


def choose_observer(context,model_class,runner_class,torch):
    """Off does not import the observer module or replace the three methods."""
    context.poll();row=context.row
    p.require('smollm3_token_capture_observer_v4' not in sys.modules,'Fresh process without preinstalled observer')
    originals=(model_class.load,runner_class.execute_model,runner_class._sample)
    if row['mode']=='on':
        module=importlib.import_module('smollm3_token_capture_observer_v4')
        p.require(Path(module.__file__).resolve()==p.ROOT/'smollm3_token_capture_observer_v4.py','Pinned observer4 module')
        module.install(row['role'],p.directory(row['run_id']),model_class,runner_class,torch)
    else:p.require(row['mode']=='off','Exact mode')
    unchanged=all(a is b for a,b in zip(originals,(model_class.load,runner_class.execute_model,runner_class._sample)))
    imported='smollm3_token_capture_observer_v4' in sys.modules
    p.require(unchanged is (row['mode']=='off') and imported is (row['mode']=='on'),'Mode changed original callable identity')
    permit=context.authenticate()
    record=dict(schema='smollm3_observer_invariance_mode_v2',run=row,mode=row['mode'],pid=permit['pid'],start_ticks=permit['start_ticks'],spawned=context.spawned,
        observer_install_called=row['mode']=='on',original_methods_unchanged=unchanged,observer_module_imported=imported,
        api_return_token_ids_preserved=True,graph_slot_history_observed=row['mode']=='on',full_model_qualified=False)
    context.record('mode-'+str(permit['pid'])+'.json',record);return record


def native_setup(context):
    """Only admitted caller reaches this original package/artifact loader."""
    context.poll();role=context.row['role'];base=context.base
    p.require(not any(x=='vllm_neuron' or x.startswith('vllm_neuron.') for x in sys.modules),'Package not imported before admission')
    wheel_path=p.ROOT/'smollm3-cache-ablation-native-v3/wheel.json'
    wheel=json.loads(p.bounded_source(wheel_path,context.plan['source_sha256'][str(wheel_path)]))
    for name,wanted in wheel['installed_files'].items():
        p.require(hashlib.sha256(p.bounded_source(p.TARGET/name)).hexdigest()==wanted,'Original installed package source')
    # Root proves actual complete artifact/payload bytes before this entry.
    # Never call old native_setup: it binds sc7 and directly rehashes payloads.
    sys.path.insert(0,str(p.TARGET))
    context.advance_environment('import')
    import vllm.plugins
    vllm.plugins.load_general_plugins()
    from vllm_neuron.models.smollm3.identity import build_identity
    from vllm_neuron.models.smollm3.artifacts import admit_existing
    from vllm_neuron.models.smollm3.modeling import NeuronSmolLM3ForCausalLM as cls
    case=base['cases'][role];config_path=Path(case['artifact'])/'neuron_config.json'
    saved=json.loads(p.bounded_source(config_path,base['artifact_sha256'][role][str(config_path)]));saved['metadata']=None
    identity=build_identity(case['model_path'],saved);p.require(identity['variant']==role,'Exact retained variant');admit_existing(case['artifact'],identity)
    def forbidden(*args,**kwargs):raise RuntimeError('Invariance diagnostic must not compile or save artifacts')
    cls.compile=forbidden;cls.save=forbidden;context.advance_environment('native_ready');context.poll();return cls


def server(context,spawned=False):
    global _SERVER_STARTED
    p.require(not _SERVER_STARTED and spawned is context.spawned and context.kind=='server','One fresh server mode per process');_SERVER_STARTED=True
    context.poll();cls=native_setup(context)
    context.begin_runner_import()
    import torch
    from vllm_neuron.worker.neuronx_distributed_model_runner import NeuronxDistributedModelRunner
    context.complete_runner_import()
    choose_observer(context,cls,NeuronxDistributedModelRunner,torch)
    if spawned:
        context.advance_environment('serve');return
    context.advance_environment('api_loading')
    import orchestrate as runner
    case=context.base['cases'][context.row['role']]
    sys.argv=[context.permit['wrapper_source'],'--model',case['model_path'],'--served-model-name',case['served_name'],
        '--tensor-parallel-size','2','--max-model-len','2048','--max-num-seqs','4','--no-enable-prefix-caching',
        '--host','127.0.0.1','--port','8010','--additional-config',json.dumps({'override_neuron_config':runner.overrides(case)})]
    runpy.run_module('vllm.entrypoints.openai.api_server',run_name='__main__',alter_sys=False)


def request(context,group,index,opener=None):
    context.poll();opener=urllib.request.urlopen if opener is None else opener
    name=group['id']+'-request'+str(index);body=p.old.request_body(context.base,group);wire=json.dumps(body).encode()
    context.record(name+'-request.json',body);context.write(name+'-request.raw',wire)
    req=urllib.request.Request('http://127.0.0.1:8010/v1/completions',data=wire,headers={'Content-Type':'application/json'})
    partial=bytearray();status=None;response_saved=False;response_complete=False
    try:
        with opener(req,timeout=group['timeout']) as response:
            status=response.status;raw=old_worker.response_bytes(response,partial);response_complete=True
            p.require(status==200,'HTTP completion status')
        context.write(name+'-response.raw',raw);response_saved=True
        parser=old_worker.parser_module();kwargs=dict(prompt_ids=body['prompt'],output_tokens=body['max_tokens'],model=body['model'])
        value=parser.parse_sse(raw.splitlines(keepends=True),**kwargs) if group['stream'] else parser.parse_completion(raw,**kwargs)
        value.update(group=group['id'],index=index,request_body_sha256=hashlib.sha256(wire).hexdigest(),response_status=status)
        context.record(name+'-parsed.json',value);return value
    except BaseException as error:
        if isinstance(error,urllib.error.HTTPError):
            status=error.code
            try:old_worker.response_bytes(error,partial);response_complete=True
            except BaseException:pass
        # The bounded reader retains one overflow byte. Preserve every byte it
        # actually read in separate bounded files, without extending the read.
        # A raw-write failure must not skip the independent failure receipt.
        retained=bytes(partial);parts=[];write_errors=[]
        overflow=len(retained)>4*p.MIB
        if not response_saved:
            chunks=[(name+'-response.raw',retained)] if not overflow else [
                (name+'-response-part'+str(i//(4*p.MIB))+'.raw',retained[i:i+4*p.MIB])
                for i in range(0,len(retained),4*p.MIB)]
            offset=0
            for part_index,(filename,chunk) in enumerate(chunks):
                try:
                    receipt=context.write(filename,chunk)
                    parts.append(dict(index=part_index,offset=offset,name=filename,sha256=hashlib.sha256(chunk).hexdigest(),bytes=len(chunk),receipt=receipt))
                except BaseException as failure:
                    write_errors.append(dict(index=part_index,offset=offset,name=filename,bytes=len(chunk),error_type=type(failure).__name__,error=str(failure)[:1024]))
                offset+=len(chunk)
        try:
            context.record(name+'-failure.json',dict(passed=False,protocol_passed=False,error_type=type(error).__name__,error=str(error)[:4096],
                group=group['id'],index=index,response_status=status,raw_sha256=hashlib.sha256(retained).hexdigest(),
                observed_response_bytes=len(retained),response_complete=response_complete,response_cap_exceeded=overflow,
                response_previously_saved=response_saved,retained_parts=parts,raw_write_errors=write_errors,
                all_observed_bytes_durable=response_saved or sum(x['bytes'] for x in parts)==len(retained) and not write_errors,
                full_model_qualified=False))
        except BaseException as failure:
            if hasattr(error,'add_note'):error.add_note('Failure evidence unavailable: '+str(failure)[:1024])
        raise


def capture(context,requester=request):
    """All original groups; historical mismatch is stored and never raises."""
    rows=[]
    for group in p.old.groups():
        context.poll()
        if group['id'].endswith('warm8'):current=[requester(context,group,0)]
        else:
            barrier=threading.Barrier(group['count'])
            def one(index):barrier.wait();return requester(context,group,index)
            with concurrent.futures.ThreadPoolExecutor(group['count']) as executor:current=list(executor.map(one,range(group['count'])))
        rows.extend(current);context.record(group['id']+'-group.json',dict(complete=True,group=group,requests=current))
    result=dict(schema='smollm3_observer_invariance_capture_v2',run=context.row,role=context.row['role'],passed=True,protocol_passed=True,
        diagnostic_complete=True,request_scope=22,output_tokens=2808,requests=rows,historical=analysis.historical(context.base,context.row['role'],rows),
        full_model_qualified=False,new_speedup_claim=False,original76_requests_qualified=False,observer_invariance_pending=True)
    # Preserve whole evidence on protocol rejection as well; a false protocol
    # receipt must never retain passed=true merely because all HTTP calls ended.
    try:analysis.validate_capture(context.base,context.row['run_id'],result)
    except BaseException as error:
        result.update(passed=False,protocol_passed=False,diagnostic_complete=False,protocol_error=repr(error))
        context.record('capture-result.json',result);raise
    context.record('capture-result.json',result);context.poll();return result


def client(context):
    p.require(context.kind=='client' and not context.spawned,'Original client mode');context.poll()
    p.require(context.permit['actual_live_server_verified'] is True,'Root authenticated exact paired server before HTTP')
    context.advance_environment('import')
    context.record('tokenizer-proof.json',old_worker.tokenizer_proof(context.base,context.row['role']))
    context.advance_environment('serve')
    return capture(context)


def export_capture_view(base,run_id,capture):
    """Exact exporter3 input schema without restoring old historical gate.

    Root runs frozen export3 only after authenticated service drain and observer
    log verification. This adapter neither reads logs nor invents a v7 result.
    """
    analysis.validate_capture(base,run_id,capture)
    p.require(capture['run']['mode']=='on','Only complete on capture can export')
    return dict(passed=True,role=capture['role'],request_scope=capture['request_scope'],requests=capture['requests'])

