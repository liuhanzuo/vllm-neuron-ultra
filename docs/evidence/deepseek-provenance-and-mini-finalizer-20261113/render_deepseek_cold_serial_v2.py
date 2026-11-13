"""Pure mechanical successor; frozen v1 and Mini lifecycle bodies are inputs."""
import ast
import hashlib

PINS={'contract.py':'73768e69b8020918db0578ca1b9ee213053e24df93d1bfeed4d180d816628cb4',
 'service.py':'1a8ada788523e4e5363003fac2a4d3b93c7b52a7abd22ab9f8434798032823c2',
 'phase.py':'5d5f2188356c70f6b84d70284642e1103357147c22cb35ec15987d18c33c6ee8',
 'bootstrap.py':'5d46bc431f4150e6555d1d57238434d4c4fba2a96779f250f64ef65cd64b12b3',
 'runtime.py':'b3fd1f9d087903fdc5c093231d27ac524a25b79ef61d2ff97d4ed2036f19edd4',
 'run.py':'7abd007babfb28b274a08848604c7655ea1bf358e2c19a233d623637f4fae53c'}
MINI_SHA='cb74a2dc076b33d09161c28c070dbd702c5498dddda8da8372b42b9322ee0d57'

def replace(text,before,after):
    if text.count(before)!=1:raise ValueError('Exact successor substitution required: '+before)
    return text.replace(before,after,1)

def render(inputs,mini):
    if set(inputs)!=set(PINS):raise ValueError('All frozen six integration sources')
    for name,text in inputs.items():
        if hashlib.sha256(text.encode()).hexdigest()!=PINS[name]:raise ValueError('Frozen source changed: '+name)
    if hashlib.sha256(mini.encode()).hexdigest()!=MINI_SHA:raise ValueError('Frozen accepted Mini lifecycle changed')
    result=dict(inputs)
    value=result['contract.py']
    value=replace(value,"'running_handoff_path','outer_seconds'","'running_handoff_path','outer_seconds','original_checkpoint_manifest'")
    value=value.replace('deepseek_cold_serial_spec_v1','deepseek_cold_serial_spec_v2').replace(
        'deepseek_cold_serial_root_review_v1','deepseek_cold_serial_root_review_v2').replace('deepseek_cold_phase_spec_v1','deepseek_cold_phase_spec_v2')
    value=replace(value,"for key in ('original_cold_metadata','resource_review','history_overlay','current_serial_catalog','lfm_r2_scope'):",
        "for key in ('original_checkpoint_manifest','original_cold_metadata','resource_review','history_overlay','current_serial_catalog','lfm_r2_scope'):")
    value=replace(value,"launch_authorized=True),'Independent exact source and resource admission before invocation')",
        "original_checkpoint_manifest=spec['original_checkpoint_manifest'],launch_authorized=True),'Independent exact source/resource/checkpoint binding before invocation')")
    result['contract.py']=value
    value=result['bootstrap.py']
    value=replace(value,'from .service import Service,argv,properties','from .service import Service,argv,properties\nfrom .checkpoint import load_bound_proof')
    value=replace(value,"'source_map_sha256','full_model_qualified'","'source_map_sha256','checkpoint_provenance','full_model_qualified'")
    value=value.replace('deepseek_cold_serial_admission_v1','deepseek_cold_serial_admission_v2')
    value=replace(value,'self.coordinator=None;self.current=None;self.running=None;self.history=None',
        'self.coordinator=None;self.current=None;self.running=None;self.history=None;self.checkpoint=None')
    value=replace(value,'resources(self.spec,self.policy);allowed=[]',
        'resources(self.spec,self.policy);allowed=[]\n                require(self.checkpoint is not None,\'Actual complete checkpoint byte proof must precede monitoring\')\n                self.checkpoint.check()')
    value=replace(value,"observer=Observer(admission,spec,load_policy(spec));observer.history=load_history(admission)",
        "observer=Observer(admission,spec,load_policy(spec));observer.history=load_history(admission)\n    observer.checkpoint=load_bound_proof(spec,admission['spec'],admission['outer_owner'],admission['checkpoint_provenance'],\n        lambda:descriptor(admission['outer_owner'],admission['retained_fd']))")
    value=replace(value,'from qwen35_sampler_runtime_history_v12 import expected_running','from .lifecycle import expected_running')
    value=value.replace("value['qwen35_sampler_current_execution']","value['deepseek_cold_serial_current_execution']")
    result['bootstrap.py']=value
    value=result['runtime.py']
    value=replace(value,'from .readiness import require_loaded_layout_producer',
        'from .readiness import require_loaded_layout_producer\nfrom .checkpoint import authenticate_checkpoint')
    value=replace(value,'from qwen35_sampler_runtime_history_v12 import check_running','from .lifecycle import check_running')
    value=replace(value,"history.unchanged();guard()\n        for path,expected", "history.unchanged();guard()\n        checkpoint=authenticate_checkpoint(spec,spec_ref,lock.owner,guard)\n        for path,expected")
    value=replace(value,"(control/'tmp').mkdir(mode=0o700)","(control/'tmp').mkdir(mode=0o700)\n        checkpoint_ref=once(control/'original-checkpoint-provenance.json',checkpoint.proof)")
    value=value.replace('deepseek_cold_serial_admission_v1','deepseek_cold_serial_admission_v2')
    value=replace(value,"source_map_sha256=digest(spec['source_sha256']),full_model_qualified=False", 
        "source_map_sha256=digest(spec['source_sha256']),checkpoint_provenance=checkpoint_ref,full_model_qualified=False")
    value=replace(value,"observer=Observer(admission,spec,policy);observer.history=transported;observer.cancelled=lambda:interrupted[0]",
        "observer=Observer(admission,spec,policy);observer.history=transported;observer.cancelled=lambda:interrupted[0]\n        observer.checkpoint=checkpoint")
    result['runtime.py']=value
    result['run.py']=result['run.py'].replace('deepseek_cold_serial_v1','deepseek_cold_serial_v2')
    # Keep the original accepted Mini descriptor and whole deepcopy/update
    # publication bodies; only names and admitted worker type are adapted.
    tree=ast.parse(mini)
    descriptor=ast.get_source_segment(mini,next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='active_descriptor'))
    descriptor=descriptor.replace("kind='minicpm5_retained_replay'","kind='deepseek_cold_serial'").replace(
        "execution['worker']['unit']","execution['coordinator']['service_owner']['unit']").replace(
        "worker_ref=copy.deepcopy(execution['worker_ref'])","coordinator_owner=copy.deepcopy(execution['coordinator_owner'])")
    function=ast.get_source_segment(mini,next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='running_value'))
    tail=function[function.index('    execution=dict('):]
    tail=tail.replace("worker=copy.deepcopy(worker),worker_ref=copy.deepcopy(worker_ref)",
        "coordinator=copy.deepcopy(coordinator),coordinator_owner=copy.deepcopy(coordinator_ref)")
    tail=tail.replace('copy.deepcopy(LOCK)','copy.deepcopy(LOCK_IDENTITY)').replace(
        'minicpm5_retained_replay_running_v2','deepseek_cold_serial_running_v2').replace(
        "[worker['unit']]","[coordinator['service_owner']['unit']]").replace(
        'minicpm5_retained_replay_current_execution=execution','deepseek_cold_serial_current_execution=execution')
    tail=replace(tail,'    result=copy.deepcopy(previous)',
        "    execution.update(spec=copy.deepcopy(admission['spec']),review=copy.deepcopy(admission['review']),\n"
        "        phase_units=copy.deepcopy(admission['units']),checkpoint_provenance=copy.deepcopy(admission['checkpoint_provenance']))\n"
        '    result=copy.deepcopy(previous)')
    result['lifecycle_shape.py']='''"""Accepted Mini v2 complete running semantics, adapted to DeepSeek coordinator."""
import copy
from datetime import datetime
import re
from .contract import LOCK_IDENTITY,reference,require
from .service import argv
'''+descriptor+'''

def expected_running(previous,updated_at,tool,admission_ref,admission,coordinator_ref,coordinator):
    phase='cold-diagnostic';previous_ref=admission['freshness']['latest_handoff']
    require(type(tool)is dict and set(tool)=={'kind','session_id'}and tool['kind']=='tool_session'
        and type(tool['session_id'])is int and tool['session_id']>0,'Externally observed actual outer tool session')
    require(type(updated_at)is str and len(updated_at)<48 and datetime.fromisoformat(updated_at).utcoffset()is not None,
        'Zoned external running observation')
    for value in(admission_ref,coordinator_ref,previous_ref):reference(value)
    process=coordinator['process'];owner=coordinator['service_owner']
    require(admission['schema']=='deepseek_cold_serial_admission_v2'and admission['lock_identity']==LOCK_IDENTITY
        and owner['unit']==admission['units']['coordinator']and process['argv']==argv('coordinator',admission_ref)
        and owner['pid']==process['pid']and str(owner['start_ticks'])==str(process['start_ticks'])
        and type(process['pid'])is int and process['pid']>0 and type(process['ppid'])is int and process['ppid']>0
        and re.fullmatch('[0-9a-f]{32}',owner['invocation_id'])
        and process['cgroup'].endswith('/'+owner['unit']+'.service'),'Exact admitted real coordinator identity')
    require(previous['serial_execution']is True and previous['running']is False and previous['active_serial']is None
        and previous['live_tool_sessions']==previous['active_heavy_units']==[],
        'Actual complete idle predecessor; no running-to-fake-idle transition')
'''+tail+'\n'
    for name,text in result.items():ast.parse(text,filename=name)
    return result
