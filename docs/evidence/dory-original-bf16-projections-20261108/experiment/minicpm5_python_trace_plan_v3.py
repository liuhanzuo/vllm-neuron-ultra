"""Pure source-plan builder, not an execution spec or runtime admission.

Inputs and outputs are decoded metadata in memory. No files, directories,
services, processes, locks, native libraries or tensor payloads are touched.
"""
import copy
import hashlib
import json
from pathlib import Path
from minicpm5_layer0_runtime_auth_v1 import require,digest
from minicpm5_native_ablation_witness_v1 import VECTOR_BYTES
from minicpm5_python_trace_observer_v2 import ARMS,SCHEMA

ROOT=Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
OLD_ROOT=ROOT/'minicpm5-native-observer-ablation-v2'
OUTPUT_ROOT=ROOT/'minicpm5-native-python-trace-v3'
INPUT_PATH=ROOT/'research/minicpm5-python-trace-input-context-20261107-v3.json'
OLD_INPUT_SHA='8c19d0126a08ec43d1028aa901faf671a6af081c5a535ca6579b25e5c617c21e'
OLD_INPUT_DIGEST='72d670310ed44174996d05325c2b8fef35051ce7b72633b89b6659f9411fbd63'
OLD_PLAN_DIGEST='18fc10e4a6b9ad0d6f26f843aa04f0534fd735d32dd167c85c61aa8540db70d6'
MIB=2**20
NEW_SOURCES=tuple('minicpm5_python_trace_'+name+'.py' for name in
    ('observer_v2','driver_v3','analysis_v1','writer_v1','factory_v3','monitor_v3','adapter_v2','worker_v3','plan_v3'))+(
    'research/minicpm5_python_scale_trace_candidate_20261107_v2.py',
    'research/minicpm5_python_scale_trace_candidate_20261107_v3.py',
    'research/minicpm5-python-trace-integration-root-source-acceptance-20261107-v2.json')
ACCEPTED_SOURCE_PINS={
    str(ROOT/'research/minicpm5-python-trace-compile-context-root-source-review-20261107-v1.json'):'b058a5b442745ddf7b9a41369f8052fe550b30205cec2c8759dad35845e3da04',
    str(ROOT/'research/minicpm5-python-trace-compile-context-independent-source-review-20261107-v1.json'):'25d32aebe42a146416c380e6191a8a90d0ccd3f9b1cf81a6f1ddee4c41ee0540',
    str(ROOT/'research/minicpm5_python_scale_trace_candidate_20261107_v3.py'):'14828eb24a912c1b8b78c78585f3788588acf3cfa045329c0851b4da5ac6f2ab',
    str(ROOT/'minicpm5_python_trace_observer_v2.py'):'ff57d12506a3bf2f2eff5727d9cf4f4fdf9bd4156444dce24817856907966e56',
    str(ROOT/'minicpm5_python_trace_driver_v3.py'):'1e68a1f5776fa4aa9768065059076c8922032e204ee42deab784d2c022802ac1',
    str(ROOT/'research/minicpm5_python_scale_trace_candidate_20261107_v2.py'):'61eda326850d41c9c996ea3d62c9709cb9407d02616aa2c42663ae6162a77b93',
    str(ROOT/'research/minicpm5-python-trace-integration-root-source-acceptance-20261107-v2.json'):'a1bd4c66291efd56546c35c839ee7f741fc6365844617955d5a51b2e4f2b423e',
    str(ROOT/'minicpm5_python_trace_observer_v1.py'):'1b87bee0be455d1789e1f20aad3ccd1e655f9f91f5c1c9b300e35588ba07153d',
    str(ROOT/'minicpm5_python_trace_driver_v2.py'):'bcadf561ff589b7f4c96ad68c89f7a0e928f0c42e6078235b6bf74258effe933',
    str(ROOT/'minicpm5_python_trace_analysis_v1.py'):'7e65e59a9c8c73c7e0d7cb1569e6bfc60e50261c9349b77282124ea63594f98c',
}

def encoded(value):return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()

def translate_input(original):
    require(digest(original)==OLD_INPUT_DIGEST and original['output_root']==str(OLD_ROOT),'Exact frozen original input')
    value=copy.deepcopy(original);value['output_root']=str(OUTPUT_ROOT)
    for row in value['writer_contexts']:
        relative=Path(row['directory']).relative_to(OLD_ROOT)
        row['directory']=str(OUTPUT_ROOT/relative)
    validate_input(value)
    return value

def validate_input(value):
    require(value['output_root']==str(OUTPUT_ROOT) and value['writer_contexts_executable_in_ablation'] is False,'Retained88 context not executable')
    original=copy.deepcopy(value);original['output_root']=str(OLD_ROOT)
    for row in original['writer_contexts']:
        relative=Path(row['directory']).relative_to(OUTPUT_ROOT)
        require('..' not in relative.parts,'Canonical retained context path')
        row['directory']=str(OLD_ROOT/relative)
    require(digest(original)==OLD_INPUT_DIGEST,'Only retained88 output namespace may change')
    return True

def build(original_input,original_plan,new_source_sha256):
    """Return source-only input/plan metadata; caller reviews before publication."""
    require(digest(original_plan)==OLD_PLAN_DIGEST,'Exact prior native160 scientific plan')
    require(isinstance(new_source_sha256,dict),'New source closure mapping')
    for name in NEW_SOURCES:
        require(str(ROOT/name) in new_source_sha256,'Every new callable source required')
    require(all(new_source_sha256.get(path)==wanted for path,wanted in ACCEPTED_SOURCE_PINS.items()),'Exact root-accepted trace integration bytes')
    for path,wanted in new_source_sha256.items():
        require(type(path)is str and Path(path).is_absolute() and Path(path).suffix in ('.py','.json','.md'),'Small source/metadata closure only')
        require(type(wanted)is str and len(wanted)==64 and all(c in '0123456789abcdef' for c in wanted),'Exact source digest')
    inputs=translate_input(original_input)
    plan=copy.deepcopy(original_plan)
    plan.update(schema='minicpm5_python_trace_source_plan_v1',output_root=str(OUTPUT_ROOT),arms=list(ARMS),
        input_context=dict(path=str(INPUT_PATH),sha256=hashlib.sha256(encoded(inputs)).hexdigest(),canonical_sha256=digest(inputs)),
        source_revision='Python local trace native160 source-only successor; all original gates and budgets retained',
        worker_entrypoint='minicpm5_python_trace_worker_v3.run',scale_observation_schema=SCHEMA,
        runtime_admitted=False,root_lifecycle_implemented=False,
        source_only_guard='Writer/factory/monitor/adapter and fresh-arm callable source only. New lifecycle, external admission, raw audit and actual native160 remain required.',
        actual_admission_catalog='Root must inherit all347 failed-sio3 catalog rows/23 processes plus every intervening finalized unit/process; no stale source-plan catalog is admission.',
        original_input_context=dict(path=str(ROOT/'research/minicpm5-native-ablation-input-context-20261107-v2.json'),sha256=OLD_INPUT_SHA,canonical_sha256=OLD_INPUT_DIGEST))
    plan['writer_contexts']=[]
    native={row['frame_index']:row for row in inputs['writer_contexts'] if row['arm']=='HF_native'}
    for arm in ARMS:
        for frame in inputs['frames']:
            call=next(c['index']for c in inputs['calls']if frame in c['frames'])
            cap=native[frame['index']]['raw_cap'] if arm=='full_trace' else 0
            plan['writer_contexts'].append(dict(arm=arm,frame=copy.deepcopy(frame),call=call,
                directory=str(OUTPUT_ROOT/(arm.replace('_','-')+'-frame-%02d'%frame['index'])),
                raw_cap=VECTOR_BYTES+cap,observer_raw_cap=cap,tensor_cap=16*MIB,metadata_cap=2*MIB,
                max_observer_payloads=64 if cap else 0,max_payloads=68 if cap else 4,allocation_slack=72*4096))
    for path,wanted in new_source_sha256.items():
        require(path not in plan['source_sha256'] or plan['source_sha256'][path]==wanted,'Never overwrite inherited source pin')
        plan['source_sha256'][path]=wanted
    for key,field in [('partition_raw_bytes','raw_cap'),('partition_metadata_bytes','metadata_cap'),('partition_allocation_slack','allocation_slack')]:
        require(sum(row[field]for row in plan['writer_contexts'])==plan[key]==original_plan[key],'Unchanged aggregate reservation')
    require(plan['partition_allocated_bytes']==730110232 and len(plan['writer_contexts'])==160,'All160 unchanged allocation')
    return dict(input_context=inputs,source_plan=plan,source_only=True,runtime_admitted=False,actual_execution_spec_created=False)
