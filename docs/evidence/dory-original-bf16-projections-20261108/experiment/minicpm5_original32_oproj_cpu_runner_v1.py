"""Bounded TP2 CPU four-arm original32 trajectory; root approval required before launch."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time

import minicpm5_original88_decode_mask_analysis_v1 as masks
import minicpm5_saved_oproj_fullshape_root_runner_v1 as fullshape
from minicpm5_layer0_runtime_auth_v1 import RuntimeContext, digest, identity

p, five, post, same, ROOT = masks.p, masks.five, masks.post, masks.same, masks.ROOT
SELF = str(Path(__file__).resolve())
PYTHON = '/home/ubuntu/venvs/vllm_neuron/bin/python'
CAP = 384 * 2**20
RANK_CAP = 192 * 2**20
ARMS = ['inclusive_ownhistory_control', 'strict_cpu_s0', 'strict_cpu_s1', 'strict_fp32_oproj_candidate']
FULLSHAPE_EXIT = masks.ref('minicpm5-saved-oproj-fullshape-root-exit-r1-20261107-v1.json', 'a7b37e6710efadb775a99d70af25b7a3768193f4cf5c12fe6e076ef3a80e12e9')
FULLSHAPE_REVIEW = masks.ref('minicpm5-saved-oproj-fullshape-independent-result-review-20261107-v1.json', '4978e951d439313be81e16c450e2b9ccc9200a970c29bb0dbe40496b96744231')
FULLSHAPE_RESULT = masks.ref('minicpm5-saved-oproj-fullshape-intervention-r1-20261107-v1.json', '1161557d376394364b25ab05ef1da857a6827699d6eacaa95c35a5a0b7012509')
FULLSHAPE_SHA = 'c771d99d5bdc12b81a6e6828ee5b51eb6f2a4ce605feadbcc1f6d92942d93e1b'
FULLSHAPE_WORKER_SHA = '452e7e08fdf35feaa8becec7e0d6c971a28a8f57cb8d27b10597eb5d4739685e'
SCIENCE_PLAN = masks.ref('minicpm5-layer0-runtime-inputs-20261107-v1.json', 'b8d84596f9fa6b54c0fa599af6ad976f051e47544a2d8c76027136efe2254800')
MASK_EXIT = masks.ref('minicpm5-original88-decode-mask-root-exit-r1-20261107-v1.json', '580dcdff25da4a194d5c797303a076f8faf0d64ae86e5f3e301a5f2355684f21')
MASK_REVIEW = masks.ref('minicpm5-original88-decode-mask-independent-metadata-review-20261107-v1.json', '2eaee60f4513053af1358e154ac67ad3e004e4b173416661275824e986bb2eb7')
PROPS = dict(MemoryMax=str(16*2**30), MemoryHigh=str(14*2**30), MemorySwapMax='0', TasksMax='128', AllowedCPUs='6-7',
             CPUAffinity='6-7', Nice='19', KillMode='control-group', RemainAfterExit='yes', SendSIGKILL='yes',
             WorkingDirectory=str(ROOT), LimitFSIZE=str(16*2**20), RuntimeMaxSec='7200', TimeoutStopSec='15')

def reference(path):
    return dict(path=str(path), sha256=p.sha(path))

def read_bundle(spec_ref, review_ref):
    spec = p.read_ref(spec_ref); review = p.read_ref(review_ref)
    p.require(spec['schema'] == 'minicpm5_original32_oproj_cpu_execution_spec_v1', 'Finite four-arm trajectory spec')
    p.require(review['reviewer'] == '/root' and review['passed'] is True and not review['blockers'], 'Root source approval')
    same(review['spec'], spec_ref, 'Reviewed exact execution spec')
    p.require(spec['worker'] == str(ROOT/'minicpm5_original32_oproj_cpu_worker_v1.py') and spec['runner'] == SELF, 'Finite executable paths')
    for path, sha in spec['source_sha256'].items(): p.require(p.sha(path) == sha, 'Reviewed source changed: '+path)
    p.require(spec['source_sha256'].get(SELF) == p.sha(SELF) and spec['source_sha256'].get(spec['worker']) == p.sha(spec['worker']), 'Both entrypoints pinned')
    plan = p.read_ref(spec['plan'])
    same(plan['source_sha256'], spec['source_sha256'], 'Plan source closure')
    same(plan['output_root'], spec['output_root'], 'Fresh output binding')
    p.require(Path(spec['output_root']).parent == ROOT and Path(spec['output_root']).name.startswith('minicpm5-original32-oproj-cpu-'), 'Output scope')
    p.require(spec['output_cap_bytes'] == CAP and spec['root_floor_bytes'] == 28*2**30, 'Original floor/finite output cap')
    same(spec['resources'], PROPS, 'Fixed bounded resource profile')
    p.require(type(spec['master_port']) is int and 29000 < spec['master_port'] < 30000, 'Finite rendezvous port')
    p.require(re.fullmatch(r'trn1-mini-original32-oproj-[a-z0-9-]{1,80}',spec['unit']) is not None, 'Named finite job')
    p.require(spec['model_qualification_claim'] is spec['automatic_full_gates_launch'] is False, 'Diagnostic CPU experiment only')
    for module in tuple(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if path and str(path).startswith(str(ROOT)+'/'):
            path=str(Path(path).resolve())
            p.require(spec['source_sha256'].get(path)==p.sha(path),'Every loaded local helper pinned: '+path)
    return spec, plan

def usage(root, floor):
    total = 0; count = 0; ranks = {'cpu-rank0':0, 'cpu-rank1':0}
    for entry in Path(root).rglob('*'):
        info = entry.lstat(); p.require(not stat.S_ISLNK(info.st_mode), 'Output symlink')
        if stat.S_ISDIR(info.st_mode): continue
        p.require(stat.S_ISREG(info.st_mode) and info.st_uid == 1000 and info.st_nlink == 1, 'Regular owned output')
        size=max(info.st_size, info.st_blocks*512)
        p.require(info.st_size <= 16*2**20, 'Bounded single output file')
        total += size; count += 1
        top=entry.relative_to(root).parts[0]
        if top in ranks:ranks[top]+=size
    p.require(total <= CAP and count <= 512, 'Finite output tree cap')
    p.require(all(size <= RANK_CAP for size in ranks.values()), 'Per-rank output cap')
    fs = os.statvfs(ROOT); p.require(fs.f_bavail*fs.f_frsize >= floor, 'Original root floor')
    return dict(allocated_bytes=total, files=count, per_rank_allocated_bytes=ranks)

def env(attempt, port, rank=None):
    result = p.environment({}, 'cpu_control', attempt)
    result.update(OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    if rank is not None:
        result.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE='2', LOCAL_WORLD_SIZE='2', MASTER_ADDR='127.0.0.1',
                      MASTER_PORT=str(port), TORCHELASTIC_RUN_ID='mini-original32-oproj-'+attempt)
    return result

def argv(mode, admission_ref):
    return [PYTHON, '-B', SELF, '--mode', mode, '--admission', admission_ref['path'], '--admission-sha256', admission_ref['sha256']]

def service_command(spec, admission_ref, attempt):
    return ['systemd-run', '--user', '--quiet', '--unit='+spec['unit'], '--service-type=exec',
            *[item for key,value in PROPS.items() for item in ('-p', key+'='+value)],
            '-p', 'StandardOutput=append:'+str(Path(spec['output_root'])/'runtime.log'), '-p', 'StandardError=inherit',
            '/usr/bin/env', '-i', *[k+'='+v for k,v in env(attempt,spec['master_port']).items()], *argv('coordinator', admission_ref)]

def validate_service(spec, admission_ref, attempt, value, owner=None):
    p.require(value['LoadState'] == 'loaded' and p.invocation(value['InvocationID']), 'Actual service invocation')
    for key, expected in PROPS.items():
        if key in ('RuntimeMaxSec', 'TimeoutStopSec'):
            actual_key = key.replace('Sec','USec')
            p.require(p.low.duration_seconds(value[actual_key]) == int(expected), 'Service duration '+key)
        else: p.require(value[key] == expected, 'Actual service property '+key)
    validate_service_identity(spec,admission_ref,attempt,value,owner)
    p.require(value['ExecStopPost'] == '', 'No extra service cleanup')
    return value

def validate_service_identity(spec,admission_ref,attempt,value,owner=None):
    p.require(value['LoadState']=='loaded' and p.invocation(value['InvocationID']),'Identifiable current service')
    command = service_command(spec,admission_ref,attempt)
    same(re.findall(r'argv\[\]=(.*?)\s*;', value['ExecStart']), [' '.join(command[command.index('/usr/bin/env'):])], 'Exact service command')
    if owner is not None:
        p.require(value['InvocationID'] == owner['invocation_id'] and value['ExecMainPID'] == str(owner['pid']), 'Same service owner')
    return value

def check_plan_extension(plan, original, added_inputs):
    allowed = {'output_root', 'source_sha256', 'inputs', 'oproj_trajectory'}
    same({k:v for k,v in plan.items() if k not in allowed},
         {k:v for k,v in original.items() if k not in allowed}, 'All original scientific and interpreter fields retained')
    for name in ('source_sha256', 'inputs'):
        p.require(all(plan[name].get(k) == v for k,v in original[name].items()), 'No frozen '+name+' entry replaced')
    same({k:v for k,v in plan['inputs'].items() if k not in original['inputs']}, added_inputs, 'Only declared metadata inputs added')
    extension = plan['oproj_trajectory']
    same(extension, dict(schema='minicpm5_original32_oproj_cpu_plan_v1', arms=ARMS,
                        original_scientific_plan=SCIENCE_PLAN, fullshape_result=FULLSHAPE_RESULT,
                        fullshape_exit=FULLSHAPE_EXIT, design=fullshape.analysis.DESIGN,
                        per_rank_output_cap_bytes=RANK_CAP, total_output_cap_bytes=CAP), 'Finite four-arm trajectory contract')
    for ref in (SCIENCE_PLAN, FULLSHAPE_RESULT, FULLSHAPE_EXIT, fullshape.analysis.DESIGN):
        same(plan['inputs'][ref['path']]['sha256'], ref['sha256'], 'Admitted trajectory metadata reference')
    for path, row in added_inputs.items():
        p.require(Path(path).suffix == '.json' and row['identity']['size'] < 8*2**20, 'Only bounded added JSON metadata')
        same(identity(path), row['identity'], 'Added metadata identity unchanged')
        p.require(p.sha(path) == row['sha256'], 'Added metadata digest unchanged')


def fullshape_predecessor(bound, owners):
    ended=p.read_ref(FULLSHAPE_EXIT);live=p.read_ref(ended['root_live']);review=p.read_ref(FULLSHAPE_REVIEW)
    p.require(review['passed'] is True and review['blockers']==[], 'Independent fullshape result review')
    same(review['root_actual_exit_ref'], FULLSHAPE_EXIT, 'Reviewed actual fullshape exit')
    same(review['report'], FULLSHAPE_RESULT, 'Reviewed fullshape report')
    same(ended['outputs']['report'], FULLSHAPE_RESULT, 'Actual fullshape report')
    p.require(ended['reviewer']==live['reviewer']=='/root' and ended['actual_exit_observed'] is True and
              type(ended['actual_exit_code']) is int and ended['actual_exit_code']==0 and ended['process_gone'] is True and
              live['actual_identity_verified'] is True, 'Root observed fullshape lifecycle')
    same(ended['owner'],live['owner'],'Fullshape live/exit owner')
    same(ended['actual_execution_handle'],dict(kind='tool_session',session_id=34798),'Actual fullshape session')
    same(live['actual_execution_handle'],ended['actual_execution_handle'],'Same fullshape handle')
    report=p.read_ref(FULLSHAPE_RESULT);wrapper=p.read_ref(ended['outputs']['runner_result'])
    same(review['runner_result'],ended['outputs']['runner_result'],'Reviewed fullshape wrapper')
    for value in (report,wrapper):
        same(value['execution_owner'],ended['owner'],'Fullshape result owner')
        p.require(value['self_exit_claimed'] is value['full_model_qualified'] is value['new_speedup_claim'] is False,'Diagnostic fullshape only')
    same(wrapper['checked_previous_owners'],owners,'Every historical owner bound by fullshape retained')
    p.require(wrapper['complete'] is report['summary']['local_math_trial_complete'] is True,'Fullshape local trial completed')
    same([report['summary'][key] for key in ('full_shape_controls','fixed_input_cases','formula_results','original_execution_eligible_cases')],[24,24,144,24],'Fullshape coverage')
    for key,value in bound['binding'].items():same(report['binding'][key],value,'Same original finalized '+key)
    same(live['source'],dict(path=fullshape.__file__,sha256=FULLSHAPE_SHA),'Actual fullshape runner')
    same(live['worker'],dict(path=fullshape.analysis.__file__,sha256=FULLSHAPE_WORKER_SHA),'Actual fullshape worker')
    same(report['binding']['code'],live['worker'],'Fullshape result code')
    same(ended['owner']['argv'],[PYTHON,'-B',fullshape.__file__,'--output-prefix',live['output_prefix'],
                              '--source-sha256',FULLSHAPE_SHA,'--analysis-sha256',FULLSHAPE_WORKER_SHA],'Actual fullshape command')
    return fullshape.previous.process(ended['owner'])


def check_prerequisites(spec):
    previous_spec=p.read_ref(fullshape.SPEC)
    bound,owners,prior=fullshape.previous.check_prerequisites(previous_spec)
    more,r2=fullshape.prefill.r2_predecessors();owners+=more
    owners.append(fullshape.prefill_predecessor(bound))
    owners.append(fullshape.projection_predecessor(bound))
    owners.append(fullshape_predecessor(bound,owners))
    for key in ('finalization','catalog','finalizer_exit'):
        same(spec[key],previous_spec[key],'Same original finalized '+key)
    same(spec['canonical_sha256'],previous_spec['canonical_sha256'],'Canonical files retained')
    inherited={}
    # Finalized archive provenance includes unrelated model metadata aliases.
    # Keep the complete actual runtime closures plus every loaded helper pin.
    for mapping in (bound['plan']['source_sha256'],previous_spec['source_sha256']):
        for path,wanted in mapping.items():
            p.require(path not in inherited or inherited[path]==wanted,'Consistent inherited source')
            inherited[path]=wanted
    inherited.update({fullshape.__file__:FULLSHAPE_SHA,fullshape.analysis.__file__:FULLSHAPE_WORKER_SHA,
                      fullshape.previous.__file__:fullshape.PREVIOUS_SHA,
                      fullshape.prefill.__file__:fullshape.PREFILL_RUNNER_SHA,
                      fullshape.prefill.analysis.__file__:fullshape.PREFILL_ANALYSIS_SHA,
                      fullshape.original_projection.__file__:fullshape.OPROJ_RUNNER_SHA,
                      fullshape.original_projection.analysis.__file__:fullshape.OPROJ_WORKER_SHA})
    p.require(all(spec['source_sha256'].get(path)==wanted for path,wanted in inherited.items()),'All predecessor sources retained')
    for module in tuple(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if path and str(path).startswith(str(ROOT)+'/'):
            path=str(Path(path).resolve())
            for archive in (bound['spec'],bound['catalog']):
                if path in archive['source_sha256']:
                    same(spec['source_sha256'][path],archive['source_sha256'][path],'Loaded helper retains archive pin')
    check_plan_extension(p.read_ref(spec['plan']),bound['plan'],spec['added_inputs'])
    return bound,owners,dict(call4=prior,failed_r1=prior['failed_r1'],r2=r2)


def predecessor_drains(priors):
    return dict(call4=fullshape.previous.drain_call4(priors['call4']),
                r1=fullshape.previous.drain_failed_r1(priors['failed_r1']),
                r2=fullshape.previous.drain_call4(priors['r2']))


def secondary_error(primary, secondary, stage):
    # Reporting must never replace the exception that caused cleanup.
    try:
        print(json.dumps(dict(primary_error_type=type(primary).__name__,primary_error=str(primary),
                              secondary_error_type=type(secondary).__name__,secondary_error=str(secondary),stage=stage)),flush=True)
    except BaseException:pass


def guarded_receipt(path,value,primary):
    try:p.once(path,value)
    except BaseException as secondary:
        if primary is None:raise
        secondary_error(primary,secondary,'failure_receipt')


def cleanup_children(children,records,root,success,primary):
    errors=[]
    for child in children:
        try:
            if not success and child.poll() is None:child.terminate()
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
        except BaseException as secondary:errors.append(secondary)
    try:
        p.once(root/'children-exits.json',dict(complete=all(c.returncode is not None for c in children),
                                              owners=records,exit_codes=[c.returncode for c in children],successful=success))
    except BaseException as secondary:errors.append(secondary)
    for secondary in errors:secondary_error(primary or errors[0],secondary,'children_cleanup')
    if errors and primary is None:raise errors[0]


def checked_output_json(ref, expected):
    same(ref['path'],str(expected),'Exact bounded output JSON path')
    info=Path(expected).lstat()
    p.require(stat.S_ISREG(info.st_mode) and info.st_uid==1000 and info.st_nlink==1 and info.st_size<8*2**20,
              'Bounded regular output JSON')
    return p.read_ref(ref)


def state_values(rows):
    same([row['name'] for row in rows],['kv_mgr.past_key_values.'+str(n) for n in range(84)],'All84 ordered state witnesses')
    for row in rows:
        same(row['shape'],[4,1,4224,128],'Original state geometry')
        same(row['dtype'],'torch.bfloat16','Original state dtype')
        p.require(re.fullmatch('[0-9a-f]{64}',row['sha256']) is not None,'State byte digest')
    return [{k:row[k] for k in ('name','shape','dtype','sha256')} for row in rows]


def validate_rank_result(result,plan,rank_id):
    p.require(result['schema']=='minicpm5_original32_oproj_cpu_rank_v1' and result['complete'] is True and
              type(result['rank']) is int and result['rank']==rank_id,'Complete original trajectory rank')
    same(result['arms'],ARMS,'Four arms in order')
    same([result[k] for k in ('model_forwards','seed_import_count','S1_repeat_all14_exact')],[56,4,True],'All56 forwards with four initial seeds and strict repeat')
    same([result[k] for k in ('full_cache_payloads_written','tap_payloads_written')],[0,0],'No cache or tap payload dumps')
    same(result['original_remaining_gates'],plan['remaining_gates'],'Original remaining scientific gates')
    same(result['original_HF_reference'],plan['hf_reference'],'Original HF reference')
    p.require(result['full_model_qualified'] is result['new_speedup_claim'] is False,'Diagnostic trajectory only')
    same([row['arm'] for row in result['arm_reports']],ARMS,'Every arm report in order')
    originals={}
    for ref in plan['cpu_records']:
        old=p.read_ref(ref)
        if old['arm']=='C_Hc_history_generation' and old['rank']==rank_id:originals[old['call']['index']]=old
    same(sorted(originals),list(range(14)),'All14 original CPU endpoints')
    output=[];repeat_baseline=None
    for arm_index,entry in enumerate(result['arm_reports']):
        arm=ARMS[arm_index];folder=Path(plan['output_root'])/('cpu-rank'+str(rank_id))/arm
        report=checked_output_json(dict(path=entry['report'],sha256=entry['sha256']),folder/'result.json')
        p.require(report['schema']=='minicpm5_original32_oproj_cpu_arm_v1' and report['complete'] is True,'Complete arm')
        same([report['arm'],report['rank'],report['model_forwards'],report['seed_import_count']],[arm,rank_id,14,1],'Finite arm identity and seed count')
        p.require(report['continuous_actual_cache_chain'] is True,'Actual own-history cache chain')
        same([row['call'] for row in report['reports']],list(range(14)),'All14 calls ordered')
        same([report[k] for k in ('full_cache_payloads_written','tap_payloads_written')],[0,0],'No large hidden outputs')
        if arm_index==0:p.require(report['all14_inclusive_endpoints_exact'] is True,'Original inclusive control gate')
        if arm_index==2:p.require(report['all14_strict_repeat_exact'] is True,'Strict repeated trajectory gate')
        calls=[];scores=[];previous=None
        for call_index,call_entry in enumerate(report['reports']):
            call=checked_output_json(dict(path=call_entry['report'],sha256=call_entry['sha256']),folder/('call-%02d'%call_index)/'result.json')
            old=originals[call_index]
            p.require(call['schema']=='minicpm5_original32_oproj_cpu_call_v1' and call['complete'] is True,'Complete original call')
            same([call['arm'],call['rank'],call['call']],[arm,rank_id,plan['calls'][call_index]],'Original call identity')
            same(call['bucket'],plan['graph_calls'][call_index]['bucket'],'Original physical bucket')
            p.require(len(call['execution_input_metadata'])==7,'Seven actual inputs')
            same(call['stage_names'],old['stage_names'],'Original213 ordered taps')
            same(list(call['stage_sha256']),call['stage_names'],'All tap digests present')
            p.require(len(call['stage_metadata'])==len(call['stage_names'])==213,'All213 tap metadata')
            p.require(all(re.fullmatch('[0-9a-f]{64}',v) is not None for v in call['stage_sha256'].values()),'Valid tap digests')
            if arm_index==0:
                control=call['inclusive_original_endpoint']
                p.require(all(control[k] is True for k in ('passed','raw_exact','original_scope_preserved','input_state_unchanged','original_poststate_reproduced')),
                          'Actual inclusive endpoint exact against original')
                same([control['arm'],control['rank'],control['call']],['C_Hc_history_generation',rank_id,plan['calls'][call_index]],'Inclusive endpoint original identity')
                for key in ('execution_input_metadata','output_metadata','stage_metadata'):
                    same(call[key],old[key],'Inclusive original '+key)
                same(call['stage_sha256'],old['stage_raw_sha256'],'Inclusive all213 original tap bytes')
                same(control['output_sha256'],call['output_metadata']['sha256'],'Inclusive output proof digest')
                same(control['all213_tap_sha256'],call['stage_sha256'],'Inclusive tap proof digests')
            before=state_values(call['before_metadata']);state_values(call['input_after_metadata']);poststate=state_values(call['post_metadata'])
            same(call['input_after_metadata'],call['before_metadata'],'All84 input states unchanged')
            if call_index:same(before,previous,'Actual previous post to next before, including call7')
            previous=poststate
            same(call['seed_imported_this_call'],call_index==0,'Only call0 imports seed')
            same(call['history_origin'],'original_call0_seed' if call_index==0 else 'previous_actual_post_same_arm','Actual same-arm lineage')
            same(call['group_reset_preserves_cache'],call_index==7,'Group reset preserves own history')
            same([row['layer'] for row in call['route_records']],list(range(42)),'All42 actual mask routes')
            witnesses=call['oproj_dtype_records']
            if arm_index==3:
                same([row['target'] for row in witnesses],['layers.%d.self_attn.o_proj.o_proj'%i for i in range(42)],'Exactly42 ordered candidate targets')
                input_shape=[1,call['bucket'],1024] if call['call']['phase']=='prefill' else [4,1,1024]
                for row in witnesses:
                    same(row['input_shape'],input_shape,'Original complete projection shape')
                    same(row['output_shape'],input_shape[:2]+[2048],'Full candidate output shape')
                    same([row[k] for k in ('input_dtype','local_dtype','reduction_dtype','output_dtype')],
                         ['torch.bfloat16','torch.float32','torch.float32','torch.bfloat16'],'FP32 local product and TP sum before single BF16 cast')
                    p.require(row['actual_tp_size']==2 and row['final_cast_count']==1 and row['original_parameter_object_unchanged'] is True,'Actual TP2 and original parameter object')
                    same(row['actual_reduce_function'],'neuronx_distributed.parallel_layers.mappings.reduce_from_tensor_model_parallel_region','Actual original TP reduction')
            else:same(witnesses,[],'Controls retain original projection without instance patch')
            if arm_index==2:
                same(call['strict_repeat'],dict(all7_inputs_exact=True,all213_taps_exact=True,all84_before_input_post_exact=True,full_logits_exact=True),'S1 repeat gates')
                for key in ('call','bucket','input_metadata','execution_input_metadata','integer_widening','changed_mask_cells','output_metadata',
                            'stage_names','stage_metadata','stage_sha256','before_metadata','input_after_metadata','post_metadata','route_records'):
                    same(call[key],repeat_baseline[call_index][key],'S1 equals S0 actual '+key)
            same(call['candidate_only42_instance_targets'],arm_index==3,'Only candidate changes projection instances')
            same(call['payload'],call_entry['payload'],'Bound full logits payload reference')
            same(call['payload']['path'],str(folder/('call-%02d'%call_index)/'output.pt'),'Exact full logits output path')
            p.require(type(call['payload']['bytes']) is int and 0<call['payload']['bytes']<=3*2**20 and
                      re.fullmatch('[0-9a-f]{64}',call['payload']['sha256']) is not None,'Bounded payload metadata')
            same(call['output_metadata']['shape'],[len(call['physical_rows']),1,130560],'Full vocabulary logits')
            same(call['output_metadata']['dtype'],'torch.float32','Full logits FP32 metadata')
            p.require(re.fullmatch('[0-9a-f]{64}',call['output_metadata']['sha256']) is not None,'Full logits tensor digest')
            same([row['frame'] for row in call['HF_metrics']],plan['calls'][call_index]['frames'],'All call HF frames')
            for row in call['HF_metrics']:
                values=row['metrics']
                p.require(all(math.isfinite(values[k]) for k in ('cosine','kl_reference_to_neuron','max_abs_error','reference_margin')),'Finite HF metrics')
                checks=dict(cosine=values['cosine']>=.995,KL=values['kl_reference_to_neuron']<=.02,max_abs=values['max_abs_error']<=2,
                            high_margin_argmax=values['reference_margin']<=.5 or values['argmax_match'])
                same(row['failed_criteria'],[k for k,v in checks.items() if not v],'Original HF thresholds unchanged')
            calls.append(call);scores.extend(call['HF_metrics'])
        same([row['frame'] for row in scores],plan['frames'],'All32 original HF scores in order')
        same(report['rows'],scores,'Arm rows derived from all14 call records')
        fraction=sum(row['metrics']['argmax_match'] for row in scores)/32
        summary=report['summary']
        same([summary[k] for k in ('complete','frames','prefill_frames','decode_frames')],[True,32,8,24],'Complete HF scoring coverage')
        same(summary['argmax_match_fraction'],fraction,'Exact aggregate argmax fraction')
        same(summary['passed'],all(not row['failed_criteria'] for row in scores) and fraction>=.9,'Original aggregate gate')
        same(summary,result['summaries'][arm],'Rank summary bound to per-call scores')
        if arm_index==1:repeat_baseline=calls
        output.append(dict(arm=arm,calls=calls,summary=summary))
    return output


def validate_rank_pair(results,plan):
    evidence=[validate_rank_result(result,plan,rank_id) for rank_id,result in enumerate(results)]
    p.require(len(evidence)==2,'Exactly two ranks')
    for left,right in zip(*evidence):
        same(left['arm'],right['arm'],'Same paired arm')
        same(left['summary'],right['summary'],'Cross-rank HF summary')
        for a,b in zip(left['calls'],right['calls']):
            same(a['output_metadata'],b['output_metadata'],'Cross-rank complete logits tensor bytes/layout')
            same(a['HF_metrics'],b['HF_metrics'],'Cross-rank all32 HF metrics')
    return dict(complete=True,rank_count=2,arm_count=4,calls_per_rank=56,HF_frames_per_arm=32,
                cross_rank_full_logits_tensor_hashes_equal=True,cross_rank_HF_metrics_equal=True,raw_payloads_read=False)


def wait_file(path, guard, seconds=30):
    end = time.monotonic()+seconds
    while not Path(path).exists():
        guard(); p.require(time.monotonic() < end, 'Parent identity acknowledgement timeout'); time.sleep(.05)
    return p.read(path)

def child_admission(admission_ref):
    admission = p.read_ref(admission_ref); spec, plan = read_bundle(admission['spec'], admission['review'])
    same(admission['plan'], spec['plan'], 'Current run plan binding')
    p.lock_proof(admission['outer_owner'], admission['lock_identity'])
    return admission, spec, plan

def coordinator(admission_ref):
    a,spec,plan = child_admission(admission_ref); root=Path(spec['output_root'])
    guard=lambda:p.lock_proof(a['outer_owner'],a['lock_identity'])
    owner=wait_file(root/'coordinator-owner.json',guard)
    same({k:owner[k] for k in ('pid','start_ticks','ppid','argv','cgroup')},p.proc(os.getpid()),'Actual coordinator parent acknowledgement')
    children=[]; records=[]; success=False
    def interrupted(signum, frame): raise InterruptedError('Coordinator signal '+str(signum))
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,interrupted)
    try:
        for rank in (0,1):
            child=subprocess.Popen(argv('rank',admission_ref),env=env(a['attempt_id'],spec['master_port'],rank),cwd=ROOT);children.append(child)
            deadline=time.monotonic()+10;actual=None
            while actual is None or actual['argv']!=argv('rank',admission_ref):
                p.require(child.poll() is None and time.monotonic()<deadline,'Rank exec deadline')
                actual=p.proc(child.pid)
                if actual is None or actual['argv']!=argv('rank',admission_ref):time.sleep(.01)
            p.require(actual['ppid']==owner['pid'] and actual['cgroup']==owner['cgroup'] and actual['argv']==argv('rank',admission_ref),'Direct original rank process')
            records.append(actual);p.once(root/('rank'+str(rank)+'-owner.json'),actual)
        while any(c.poll() is None for c in children):
            guard();usage(root,spec['root_floor_bytes']);p.require(all(c.poll() in (None,0) for c in children),'Rank failed');time.sleep(.5)
        p.require(all(c.wait()==0 for c in children),'Both actual rank exits0')
        results=[p.read(root/('rank'+str(r)+'-result.json'))for r in (0,1)]
        for rank,result in enumerate(results):
            same(result['process'],records[rank],'Rank result identity');p.require(result['complete']is True and result['rank']==rank,'Complete rank result')
        validated=validate_rank_pair(results,plan)
        p.once(root/'coordinator-result.json',dict(complete=True,rank_exits=[0,0],rank_owners=records,results=results,validated=validated,self_exit_claimed=False))
        success=True
    finally:
        cleanup_children(children,records,root,success,sys.exc_info()[1])

def rank(admission_ref):
    a,spec,plan=child_admission(admission_ref);root=Path(spec['output_root']);rank_id=int(os.environ['RANK'])
    p.require(rank_id in (0,1),'TP2 rank')
    lease=lambda:p.lock_proof(a['outer_owner'],a['lock_identity'])
    process=wait_file(root/('rank'+str(rank_id)+'-owner.json'),lease);same(process,p.proc(os.getpid()),'Parent observed actual rank')
    coordinator_owner=p.read(root/'coordinator-owner.json');same(process['ppid'],coordinator_owner['pid'],'Parent coordinator')
    stable=env(a['attempt_id'],spec['master_port'],rank_id);stable.pop('NEURON_RT_VISIBLE_CORES')
    permit=dict(schema='minicpm5_layer0_runtime_permit_v1',plan_sha256=digest(plan),role='cpu',rank=rank_id,arm='C_Hc_history_generation',
                attempt_id=a['attempt_id'],serial_exclusive=True,source_review_accepted=True,all_predecessors_drained=True,uid=1000,
                **process,python=PYTHON,environment=stable,output_root=spec['output_root'])
    def guard():
        lease();same(p.proc(coordinator_owner['pid']),{k:coordinator_owner[k]for k in process},'Coordinator remains original')
        p.clear_window();return permit
    context=RuntimeContext(plan,digest(plan),permit,guard);context.verify_sources()
    import minicpm5_original32_oproj_cpu_worker_v1 as worker
    try:
        result=worker.run(context,root/('cpu-rank'+str(rank_id)))
        p.require(result['complete']is True and result['rank']==rank_id,'Finite worker complete')
        validate_rank_result(result,plan,rank_id)
        result=dict(result,process=process,admission=admission_ref,self_exit_claimed=False)
        p.once(root/('rank'+str(rank_id)+'-result.json'),result)
    except BaseException as error:
        guarded_receipt(root/('rank'+str(rank_id)+'-failure.json'),dict(process=process,error_type=type(error).__name__,error=str(error),self_exit_claimed=False),error);raise

def outer(spec_ref,review_ref):
    spec,plan=read_bundle(spec_ref,review_ref);bound,owners,priors=check_prerequisites(spec);root=Path(spec['output_root'])
    p.require(not root.exists(),'Fresh contrast output directory');p.user_bus()
    fd,owner,lock=post.external.acquire_lock();launch_attempted=False;drained=False;service_owner=None;terminal=None;ar=None;a=None;failure=None
    def interrupted(signum,frame):raise InterruptedError('Outer signal '+str(signum))
    handlers={}
    try:
        for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):handlers[sig]=signal.signal(sig,interrupted)
        p.clear_window();p.active_heavy();p.gone(bound['finalizer_owner']);p.predecessor_drain(bound['catalog'])
        for previous in owners:p.gone(previous)
        drains=predecessor_drains(priors)
        p.control_usage(bound['spec'])
        same({n:p.sha(ROOT/n) for n in spec['canonical_sha256']},spec['canonical_sha256'],'Canonical state unchanged')
        p.require(p.properties(spec['unit'])['LoadState']=='not-found','Fresh service name')
        root.mkdir(mode=0o700);usage(root,spec['root_floor_bytes'])
        a=dict(schema='minicpm5_original32_oproj_cpu_admission_v1',spec=spec_ref,review=review_ref,plan=spec['plan'],outer_owner=owner,
               lock_identity=lock,attempt_id=os.urandom(16).hex(),predecessors_checked=True,later_owners=owners,fullshape_exit=FULLSHAPE_EXIT,observed_drains=drains)
        p.once(root/'admission.json',a);ar=reference(root/'admission.json')
        launch_attempted=True
        p.command(service_command(spec,ar,a['attempt_id']),timeout=30)
        deadline=time.monotonic()+30
        while service_owner is None:
            value=validate_service(spec,ar,a['attempt_id'],p.properties(spec['unit']))
            pid=int(value['MainPID']);actual=p.proc(pid) if pid else None
            if actual:
                same(actual['argv'],argv('coordinator',ar),'Actual service coordinator argv')
                same(actual['cgroup'],'0::'+p.GROUP_PREFIX+spec['unit']+'.service','Dedicated service group')
                service_owner=dict(actual,unit=spec['unit'],invocation_id=value['InvocationID']);p.once(root/'coordinator-owner.json',service_owner)
            else:p.require(not p.terminal(value) and time.monotonic()<deadline,'No live coordinator');time.sleep(.05)
        deadline=time.monotonic()+7230
        while True:
            p.lock_proof(owner,lock);p.clear_window();usage(root,spec['root_floor_bytes']);p.active_heavy((spec['unit'],))
            value=validate_service(spec,ar,a['attempt_id'],p.properties(spec['unit']),service_owner)
            if p.terminal(value):terminal=value;break
            p.require(time.monotonic()<deadline,'Bounded four-arm trajectory deadline');time.sleep(1)
        p.gone(service_owner);drain=p.empty_group(p.GROUP_PREFIX+spec['unit']+'.service')
        drained=True
        p.once(root/'service-exit.json',dict(owner=service_owner,properties=terminal,group=drain,observed_by_outer=owner,
                                          successful=terminal['Result']=='success' and terminal['ExecMainCode']=='1' and terminal['ExecMainStatus']=='0'))
        p.require(terminal['Result']=='success' and terminal['ExecMainCode']=='1' and terminal['ExecMainStatus']=='0','Actual model job failed')
        result=p.read(root/'coordinator-result.json');p.require(result['complete']is True and result['rank_exits']==[0,0],'Actual two-rank completion')
        same({n:p.sha(ROOT/n) for n in spec['canonical_sha256']},spec['canonical_sha256'],'Canonical state unchanged at completion')
        p.once(root/'outer-result.json',dict(complete=True,owner=owner,service_exit=reference(root/'service-exit.json'),coordinator_result=reference(root/'coordinator-result.json'),
                                            usage=usage(root,spec['root_floor_bytes']),self_exit_claimed=False,full_model_qualified=False,new_speedup_claim=False))
        return reference(root/'outer-result.json')
    except BaseException as error:
        failure=dict(error_type=type(error).__name__,error=str(error),owner=owner,self_exit_claimed=False)
        try:guarded_receipt(root/'outer-failure.json',failure,error)
        except BaseException as secondary:secondary_error(error,secondary,'outer_failure_receipt')
        raise
    finally:
        primary=sys.exc_info()[1];cleanup_errors=[]
        try:
            if launch_attempted and not drained:
                value=p.properties(spec['unit'])
                if value['LoadState']!='not-found':
                    validate_service_identity(spec,ar,a['attempt_id'],value,service_owner)
                    p.command(['systemctl','--user','stop',spec['unit']],timeout=30)
                    group=p.empty_group(p.GROUP_PREFIX+spec['unit']+'.service')
                    after=p.properties(spec['unit'])
                    same(after['InvocationID'],value['InvocationID'],'Cleanup retained same invocation')
                    p.require(p.terminal(after),'Cleanup actual terminal')
                    p.once(root/'cleanup.json',dict(before=value,after=after,group=group,owner=service_owner,actual_stop_completed=True,outer_failure=failure))
        except BaseException as secondary:cleanup_errors.append(secondary)
        try:post.external.release_lock(fd)
        except BaseException as secondary:cleanup_errors.append(secondary)
        for sig,handler in handlers.items():
            try:signal.signal(sig,handler)
            except BaseException as secondary:cleanup_errors.append(secondary)
        for secondary in cleanup_errors:secondary_error(primary or cleanup_errors[0],secondary,'outer_cleanup')
        if cleanup_errors and primary is None:raise cleanup_errors[0]

if __name__=='__main__':
    cli=argparse.ArgumentParser();cli.add_argument('--mode',choices=['outer','coordinator','rank'],required=True)
    for key in ['spec','review','admission']:
        cli.add_argument('--'+key);cli.add_argument('--'+key+'-sha256')
    args=cli.parse_args();p.require(sys.dont_write_bytecode and __debug__,'python -B required')
    refs={k:dict(path=getattr(args,k),sha256=getattr(args,k+'_sha256'))for k in ['spec','review','admission']}
    if args.mode=='outer':print(json.dumps(outer(refs['spec'],refs['review']),indent=2))
    elif args.mode=='coordinator':coordinator(refs['admission'])
    else:rank(refs['admission'])
