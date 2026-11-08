"""Bounded TP2 CPU three-arm original32 trajectory; root approval required before launch."""
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
import minicpm5_original32_oproj_cpu_runner_v1 as predecessor
from minicpm5_layer0_runtime_auth_v1 import RuntimeContext, digest, identity

p, five, post, same, ROOT = masks.p, masks.five, masks.post, masks.same, masks.ROOT
SELF = str(Path(__file__).resolve())
PYTHON = '/home/ubuntu/venvs/vllm_neuron/bin/python'
CAP = 256 * 2**20
RANK_CAP = 128 * 2**20
ARMS = ['strict_anchor_s0', 'strict_layer0_full_bf16_prefill_only', 'strict_layer0_full_bf16_both_phases']
FULLSHAPE_EXIT = masks.ref('minicpm5-saved-oproj-fullshape-root-exit-r1-20261107-v1.json', 'a7b37e6710efadb775a99d70af25b7a3768193f4cf5c12fe6e076ef3a80e12e9')
FULLSHAPE_REVIEW = masks.ref('minicpm5-saved-oproj-fullshape-independent-result-review-20261107-v1.json', '4978e951d439313be81e16c450e2b9ccc9200a970c29bb0dbe40496b96744231')
FULLSHAPE_RESULT = masks.ref('minicpm5-saved-oproj-fullshape-intervention-r1-20261107-v1.json', '1161557d376394364b25ab05ef1da857a6827699d6eacaa95c35a5a0b7012509')
FULLSHAPE_SHA = 'c771d99d5bdc12b81a6e6828ee5b51eb6f2a4ce605feadbcc1f6d92942d93e1b'
FULLSHAPE_WORKER_SHA = '452e7e08fdf35feaa8becec7e0d6c971a28a8f57cb8d27b10597eb5d4739685e'
SCIENCE_PLAN = masks.ref('minicpm5-layer0-runtime-inputs-20261107-v1.json', 'b8d84596f9fa6b54c0fa599af6ad976f051e47544a2d8c76027136efe2254800')
MASK_EXIT = masks.ref('minicpm5-original88-decode-mask-root-exit-r1-20261107-v1.json', '580dcdff25da4a194d5c797303a076f8faf0d64ae86e5f3e301a5f2355684f21')
MASK_REVIEW = masks.ref('minicpm5-original88-decode-mask-independent-metadata-review-20261107-v1.json', '2eaee60f4513053af1358e154ac67ad3e004e4b173416661275824e986bb2eb7')
PREVIOUS_PLAN = masks.ref('minicpm5-original32-oproj-cpu-input-plan-20261107-v1.json','0037e8c1464953ddebba2c6ef20f5a5464125053d631fda417d56903eb8eed7c')
PREVIOUS_EXIT = masks.ref('minicpm5-original32-oproj-cpu-root-exit-r1-20261107-v1.json','737d97c3b5675a060c05fef16d44548898812518327d75c37d707bbc94ad2b6d')
PREVIOUS_SUMMARY = masks.ref('minicpm5-original32-oproj-cpu-root-summary-r1-20261107-v1.json','563a8df8425bece92afe29711214c4e727b05a0331715975a5005dae571610ab')
PREVIOUS_SPEC = masks.ref('minicpm5-original32-oproj-cpu-execution-spec-20261107-v1.json','c6960316d9e8f8b0c926cf8936fb3eaafe9117b4c85d0f01387e94d9a50e858d')
PREVIOUS_REVIEW = masks.ref('minicpm5-original32-oproj-cpu-independent-result-review-20261107-v1.json','5a178148905729242763a3d22b55e247162ab78d1473bf82656606cfed0a571e')
PREVIOUS_RUNNER_SHA = '4b4f9d27cf54aae1a66139df8241497aff6a9cc96907f498f4d4eb1662cef0d8'
DESIGN = masks.ref('minicpm5-layer0-full-bf16-next-design-20261108-v2.json','f19895d9466ae42cd5e0a9b7022b9a68d3cf554669221dc18a2086935c56229d')
SELECTED_TAPS = ['layers.0.self_attn','layers.0','layers.41.self_attn','layers.41']
TARGET = 'layers.0.self_attn.o_proj.o_proj'
REPEAT_FIELDS = ('call','bucket','input_metadata','execution_input_metadata','integer_widening','changed_mask_cells','output_metadata','stage_names','stage_metadata','stage_sha256','before_metadata','input_after_metadata','post_metadata','route_records')

GATHER_TERMINAL = dict(path=str(ROOT/'research/minicpm5-full-bf16-gather-r1-20261108-v1-root-terminal.json'),sha256='6dda6b29935d8760c1e923cf9478a0134e0052d2fe5bf8c798028a271dcc659f')
GATHER_EXIT = dict(path=str(ROOT/'research/minicpm5-full-bf16-gather-r1-20261108-v1-finalizer-external-exit.json'),sha256='97517988cf617efa0774cc6a3b67d508c305589c077dd007638d5d8fbc04081f')
FULL_WEIGHT_SHA = 'f641e83dd6533cd4c9ea6f5b3689d4c952c86ca18c7e765db0399adfd259ed78'

PROPS = dict(MemoryMax=str(16*2**30), MemoryHigh=str(14*2**30), MemorySwapMax='0', TasksMax='128', AllowedCPUs='6-7',
             CPUAffinity='6-7', Nice='19', KillMode='control-group', RemainAfterExit='yes', SendSIGKILL='yes',
             WorkingDirectory=str(ROOT), LimitFSIZE=str(16*2**20), RuntimeMaxSec='7200', TimeoutStopSec='15')

def reference(path):
    return dict(path=str(path), sha256=p.sha(path))

def read_bundle(spec_ref, review_ref):
    spec = p.read_ref(spec_ref); review = p.read_ref(review_ref)
    p.require(spec['schema'] == 'minicpm5_original32_layer0_full_bf16_cpu_execution_spec_v1', 'Finite three-arm trajectory spec')
    p.require(review['reviewer'] == '/root' and review['passed'] is True and not review['blockers'], 'Root source approval')
    same(review['spec'], spec_ref, 'Reviewed exact execution spec')
    p.require(spec['worker'] == str(ROOT/'minicpm5_original32_layer0_full_bf16_cpu_worker_v1.py') and spec['runner'] == SELF, 'Finite executable paths')
    for path, sha in spec['source_sha256'].items(): p.require(p.sha(path) == sha, 'Reviewed source changed: '+path)
    p.require(spec['source_sha256'].get(SELF) == p.sha(SELF) and spec['source_sha256'].get(spec['worker']) == p.sha(spec['worker']), 'Both entrypoints pinned')
    plan = p.read_ref(spec['plan'])
    same(plan['source_sha256'], spec['source_sha256'], 'Plan source closure')
    same(plan['output_root'], spec['output_root'], 'Fresh output binding')
    p.require(Path(spec['output_root']).parent == Path('/mnt/nvme') and Path(spec['output_root']).name.startswith('minicpm5-original32-layer0-full-bf16-cpu-'), 'Output scope')
    p.require(spec['output_cap_bytes'] == CAP and spec['root_floor_bytes'] == 28*2**30 and spec['output_floor_bytes']==8*2**30, 'Original root floor and finite NVMe output cap')
    same(spec['resources'], PROPS, 'Fixed bounded resource profile')
    p.require(type(spec['master_port']) is int and 29000 < spec['master_port'] < 30000, 'Finite rendezvous port')
    p.require(re.fullmatch(r'trn1-mini-original32-layer0-full-bf16-[a-z0-9-]{1,80}',spec['unit']) is not None, 'Named finite job')
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
    fs=os.statvfs('/mnt/nvme');p.require(fs.f_bavail*fs.f_frsize>=8*2**30,'NVMe output floor')
    return dict(allocated_bytes=total, files=count, per_rank_allocated_bytes=ranks)

def env(attempt, port, rank=None):
    result = p.environment({}, 'cpu_control', attempt)
    result.update(OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    if rank is not None:
        result.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE='2', LOCAL_WORLD_SIZE='2', MASTER_ADDR='127.0.0.1',
                      MASTER_PORT=str(port), TORCHELASTIC_RUN_ID='mini-original32-layer0-full-bf16-'+attempt)
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

def strict_anchor_refs():
    summary=p.read_ref(PREVIOUS_SUMMARY); refs=[]
    for rank in (0,1):
        expected=Path(p.read_ref(PREVIOUS_PLAN)['output_root'])/('cpu-rank'+str(rank))/'strict_cpu_s0'/'result.json'
        matches=[ref for ref in summary['sources'] if ref['path']==str(expected)]
        p.require(len(matches)==1,'Unique frozen strict S0 arm')
        report=p.read_ref(matches[0]);p.require(report['complete'] is True and report['rank']==rank and report['arm']=='strict_cpu_s0','Complete frozen strict arm')
        same([row['call']for row in report['reports']],list(range(14)),'All14 frozen anchors')
        refs.extend(dict(rank=rank,call=row['call'],path=row['report'],sha256=row['sha256'])for row in report['reports'])
    return refs


def check_plan_extension(plan, original, added_inputs):
    allowed={'output_root','source_sha256','inputs','layer0_full_bf16_trajectory'}
    same({k:v for k,v in plan.items() if k not in allowed}, {k:v for k,v in original.items() if k not in allowed},'Original four-arm scientific plan retained')
    for name in ('source_sha256','inputs'):
        p.require(all(plan[name].get(k)==v for k,v in original[name].items()),'No inherited '+name+' replaced')
    same({k:v for k,v in plan['inputs'].items() if k not in original['inputs']},added_inputs,'Only declared metadata inputs added')
    anchors=strict_anchor_refs()
    same(plan['layer0_full_bf16_trajectory'],dict(schema='minicpm5_original32_layer0_full_bf16_cpu_plan_v1',arms=ARMS,
         predecessor_plan=PREVIOUS_PLAN,predecessor_actual_exit=PREVIOUS_EXIT,predecessor_summary=PREVIOUS_SUMMARY,
         strict_anchor_reports=anchors,design=DESIGN,per_rank_output_cap_bytes=RANK_CAP,total_output_cap_bytes=CAP,
         selected_tap_names=SELECTED_TAPS),'Exact three-arm layer0 phase contract')
    for ref in [PREVIOUS_PLAN,PREVIOUS_EXIT,PREVIOUS_SUMMARY,DESIGN,*anchors]:
        same(plan['inputs'][ref['path']]['sha256'],ref['sha256'],'Admitted immutable metadata reference')
    for path,row in added_inputs.items():
        p.require(Path(path).suffix=='.json' and row['identity']['size']<8*2**20,'Only bounded added JSON')
        same(identity(path),row['identity'],'Added metadata identity unchanged')
        same(p.sha(path),row['sha256'],'Added metadata hash unchanged')


def fourarm_predecessor(owners):
    ended=p.read_ref(PREVIOUS_EXIT);live=p.read_ref(ended['root_live']);review=p.read_ref(PREVIOUS_REVIEW)
    same(review['actual_exit']['root_exit_record']['sha256'],PREVIOUS_EXIT['sha256'],'Independent review binds actual exit')
    same(review['root_summary']['sha256'],PREVIOUS_SUMMARY['sha256'],'Independent review binds summary')
    p.require(review['passed'] is True and review['blockers']==[],'Independent predecessor review passed')
    p.require(ended['reviewer']==live['reviewer']=='/root' and live['actual_identity_verified'] is True,'Root predecessor live identity')
    p.require(ended['actual_exit_observed'] is True and type(ended['actual_exit_code'])is int and ended['actual_exit_code']==0,'Root observed actual predecessor exit0')
    p.require(ended['all_four_processes_gone'] is ended['cgroup_empty'] is ended['canonical_unchanged'] is True,'Predecessor drained and canonical retained')
    same(ended['actual_execution_handle'],dict(kind='tool_session',session_id=49173),'Actual predecessor execution handle')
    same(live['actual_execution_handle'],ended['actual_execution_handle'],'Predecessor live/exit handle')
    for key in ('owner','coordinator','ranks'):same(live[key],ended[key],'Predecessor live/exit '+key)
    admission=p.read_ref(live['admission']);same(admission['later_owners'],owners,'Every historical owner retained')
    same(admission['spec'],PREVIOUS_SPEC,'Actual predecessor spec');same(admission['plan'],PREVIOUS_PLAN,'Actual predecessor plan')
    same(admission['outer_owner'],ended['owner'],'Actual predecessor outer owner')
    results={k:p.read_ref(v)for k,v in ended['outputs'].items()}
    p.require(results['outer_result']['complete'] is results['coordinator_result']['complete'] is results['children_exits']['successful'] is results['service_exit']['successful'] is True,'Complete predecessor service tree')
    same(results['coordinator_result']['rank_exits'],[0,0],'Predecessor coordinator rank exits0')
    same(results['children_exits']['exit_codes'],[0,0],'Predecessor child exits0')
    same(results['children_exits']['owners'],ended['ranks'],'Predecessor child owners')
    same(results['coordinator_result']['rank_owners'],ended['ranks'],'Predecessor coordinator rank owners')
    same(results['outer_result']['owner'],ended['owner'],'Predecessor outer result owner')
    same(results['service_exit']['owner'],ended['coordinator'],'Predecessor service owner')
    same(results['outer_result']['service_exit'],ended['outputs']['service_exit'],'Predecessor service receipt binding')
    same(results['outer_result']['coordinator_result'],ended['outputs']['coordinator_result'],'Predecessor coordinator receipt binding')
    props=results['service_exit']['properties']
    p.require(props['Result']=='success' and props['ExecMainCode']=='1' and props['ExecMainStatus']=='0' and props['SubState']=='exited','Actual predecessor service exit0')
    same(props['InvocationID'],ended['coordinator']['invocation_id'],'Same predecessor service invocation')
    same(props['ExecMainPID'],str(ended['coordinator']['pid']),'Same predecessor service pid')
    same(p.read_ref(PREVIOUS_SUMMARY)['actual_exit'],PREVIOUS_EXIT,'Summary binds predecessor actual exit')
    same([p.read_ref(PREVIOUS_SUMMARY)['summaries'][arm]['passed_frames']for arm in predecessor.ARMS],[16,29,29,27],'Negative predecessor outcomes retained')
    return [fullshape.previous.process(owner)for owner in [ended['owner'],ended['coordinator'],*ended['ranks']]],ended


def check_prerequisites(spec):
    previous_spec=p.read_ref(PREVIOUS_SPEC)
    bound,owners,priors=predecessor.check_prerequisites(previous_spec)
    more,ended=fourarm_predecessor(owners);owners+=more
    for key in ('finalization','catalog','finalizer_exit','canonical_sha256'):
        same(spec[key],previous_spec[key],'Same original finalized '+key)
    p.require(all(spec['source_sha256'].get(k)==v for k,v in previous_spec['source_sha256'].items()),'Complete predecessor source closure retained')
    same(spec['source_sha256'].get(predecessor.__file__),PREVIOUS_RUNNER_SHA,'Pinned predecessor runner')
    check_plan_extension(p.read_ref(spec['plan']),p.read_ref(PREVIOUS_PLAN),spec['added_inputs'])
    terminal=p.read_ref(GATHER_TERMINAL);completed=p.read_ref(GATHER_EXIT)
    p.require(terminal['passed'] and terminal['all12_new_payloads_independently_compared'] and len(terminal['cases'])==4,'Complete actual local gather prerequisite')
    same(completed['terminal'],GATHER_TERMINAL,'Latest actual terminal');same(completed['owner'],terminal['verifier'],'Latest actual verifier')
    p.require(completed['actual_tool_exit_observed'] and completed['exact_owner_gone'] and completed['actual_exit']==dict(session_id=93505,exit_code=0,tool_chunk='24dfe8'),'Latest actual successful external exit')
    for previous in [terminal['native_owner'],terminal['verifier'],*terminal['worker_owners']]:p.gone(previous)
    return bound,owners,dict(inherited=priors,fourarm=ended,gather=terminal)


def predecessor_drains(priors):
    for owner in [priors['gather']['native_owner'],priors['gather']['verifier'],*priors['gather']['worker_owners']]:p.gone(owner)
    ended=priors['fourarm']
    for owner in [ended['owner'],ended['coordinator'],*ended['ranks']]:p.gone(fullshape.previous.process(owner))
    group=ended['coordinator']['cgroup'].split('::',1)[1]
    return dict(inherited=predecessor.predecessor_drains(priors['inherited']),fourarm=p.empty_group(group))


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


def patch_enabled(arm,phase):
    p.require(arm in ARMS and phase in ('prefill','decode'),'Finite arm/phase')
    return arm==ARMS[2] or (arm==ARMS[1] and phase=='prefill')


def validate_selected_bundle(call,folder):
    bundle=call['selected_tap_bundle'];rows=len(call['physical_rows']);size=rows*2048*2
    same(bundle['path'],str(folder/'selected-taps.bf16'),'Exact finite selected tap bundle path')
    p.require(bundle['encoding']=='raw_bf16_le' and bundle['raw_bytes']==4*size and 0<4*size<=65536,'Bounded raw selected taps')
    p.require(re.fullmatch('[0-9a-f]{64}',bundle['sha256']) is not None,'Bundle digest')
    same(bundle['identity'],identity(bundle['path']),'Actual selected bundle identity')
    info=Path(bundle['path']).lstat()
    p.require(stat.S_ISREG(info.st_mode) and info.st_uid==1000 and info.st_nlink==1 and info.st_size==4*size,'Regular bounded selected bundle')
    same([row['name']for row in bundle['entries']],SELECTED_TAPS,'Four selected taps in order')
    for index,row in enumerate(bundle['entries']):
        same(row['offset'],index*size,'Exact cumulative tap offset')
        same(row['raw_bytes'],size,'Exact tap interval length')
        same(row['shape'],[rows,1,2048],'Prediction-row tap geometry')
        same(row['dtype'],'torch.bfloat16','BF16 selected prediction values')
        same(row['sha256'],call['stage_sha256'][row['name']],'Selected entry binds gathered stage digest')
        same(row['source_metadata'],call['stage_metadata'][row['name']],'Original full-source tap metadata')
        mapping=sorted(call['physical_rows'],key=lambda item:item['physical_row'])
        same([item['physical_row']for item in mapping],list(range(rows)),'Complete physical prediction rows')
        same(row['frame_indices'],[item['frame']['index']for item in mapping],'Selected prediction rows map physical order')
    return bundle['raw_bytes']


def validate_constructors(report,rank,arm):
    same([row['before_call']for row in report['constructors']],[0,4,7,11],'Original phase constructor calls')
    first=None
    for row in report['constructors']:
        phase='prefill'if row['before_call']in (0,7)else'decode';enabled=patch_enabled(arm,phase)
        same(row['candidate_patch_installed'],enabled,'Actual phase dispatch')
        proof=row['proof']
        p.require(proof['compiled'] is proof['resharded'] is proof['sdk_math_modified'] is False,'Original CPU model constructor')
        p.require(proof['all_shard_values_raw_exact'] is proof['all_state_on_cpu'] is proof['all_buffers_on_cpu'] is True,'Original CPU shard proof')
        weights=row['official_all42_oproj']
        same([w['target']for w in weights],['layers.%d.self_attn.o_proj.o_proj'%i for i in range(42)],'All42 official projection proofs retained')
        for layer,w in enumerate(weights):
            same([w['layer'],w['official_name'],w['column_range']],[layer,'model.layers.%d.self_attn.o_proj.weight'%layer,[rank*1024,(rank+1)*1024]],'Official column shards')
            p.require(re.fullmatch('[0-9a-f]{64}',w['sha256'])is not None,'Official shard proof digest')
        if first is None:first=weights
        else:same(weights,first,'All42 weights unchanged across phase constructors')


def validate_rank_result(result,plan,rank_id):
    p.require(result['schema']=='minicpm5_original32_layer0_full_bf16_cpu_rank_v1' and result['complete'] is True and type(result['rank'])is int and result['rank']==rank_id,'Complete targeted rank')
    same(result['arms'],ARMS,'Three finite arms in order')
    same(result['selected_tap_names'],SELECTED_TAPS,'Rank four selected taps')
    p.require(result['selected_taps_are_prediction_queries']is True and result['first_group_prefill_repeat_all4_exact']is True and result['call4_before_equal_prefill_only']is True,'Complete phase contrast controls')
    same([result[k]for k in ('model_forwards','seed_import_count','anchor_all14_exact')],[42,3,True],'All42 forwards and three own seeds with exact anchor')
    same([result[k]for k in ('full_cache_payloads_written','tap_payloads_written','selected_tap_payloads_written','selected_tap_raw_bytes')],[0,42,42,1572864],'Only three-arm bounded selected taps')
    same(result['original_remaining_gates'],plan['remaining_gates'],'Original remaining gates')
    same(result['original_HF_reference'],plan['hf_reference'],'Original HF reference')
    p.require(result['full_model_qualified']is result['new_speedup_claim']is False,'Diagnostic only')
    same([row['arm']for row in result['arm_reports']],ARMS,'All arm reports ordered')
    anchors={}
    for ref in plan['layer0_full_bf16_trajectory']['strict_anchor_reports']:
        if ref['rank']==rank_id:
            old=p.read_ref(ref)
            same([old['rank'],old['arm'],old['call']['index']],[rank_id,'strict_cpu_s0',ref['call']],'Frozen strict anchor identity')
            anchors[ref['call']]=(ref,old)
    same(sorted(anchors),list(range(14)),'All14 immutable strict anchors')
    output=[];prefill_only=None
    for arm_index,entry in enumerate(result['arm_reports']):
        arm=ARMS[arm_index];folder=Path(plan['output_root'])/('cpu-rank'+str(rank_id))/arm
        report=checked_output_json(dict(path=entry['report'],sha256=entry['sha256']),folder/'result.json')
        p.require(report['schema']=='minicpm5_original32_layer0_full_bf16_cpu_arm_v1'and report['complete']is True,'Complete targeted arm')
        same([report['arm'],report['rank'],report['model_forwards'],report['seed_import_count']],[arm,rank_id,14,1],'Original arm identity and seed count')
        p.require(report['continuous_actual_cache_chain']is True,'Actual own-history cache chain')
        same([row['call']for row in report['reports']],list(range(14)),'All14 original calls ordered')
        same([report[k]for k in ('full_cache_payloads_written','tap_payloads_written','selected_tap_payloads_written','selected_tap_raw_bytes')],[0,14,14,524288],'Only bounded selected tap bundles')
        same(report['anchor_all14_exact'],arm_index==0,'Anchor gate')
        same(report['first_group_prefill_repeat_all4_exact'],arm_index==2,'Candidate first-group repeat gate')
        same(report['call4_before_equal_prefill_only'],arm_index==2,'Candidate first decode incoming gate')
        same(report['selected_tap_names'],SELECTED_TAPS,'Arm four selected taps')
        p.require(report['selected_taps_are_prediction_queries']is True,'Only gathered prediction taps')
        validate_constructors(report,rank_id,arm)
        calls=[];scores=[];previous=None;tap_bytes=0
        for ci,call_entry in enumerate(report['reports']):
            callfolder=folder/('call-%02d'%ci)
            call=checked_output_json(dict(path=call_entry['report'],sha256=call_entry['sha256']),callfolder/'result.json')
            oldref,old=anchors[ci]
            p.require(call['schema']=='minicpm5_original32_layer0_full_bf16_cpu_call_v1'and call['complete']is True,'Complete original targeted call')
            same([call['arm'],call['rank'],call['call']],[arm,rank_id,plan['calls'][ci]],'Original call identity')
            same(call['bucket'],plan['graph_calls'][ci]['bucket'],'Original bucket')
            for key in ('input_metadata','execution_input_metadata','integer_widening','changed_mask_cells','physical_rows','route_records'):
                same(call[key],old[key],'Original strict inputs mapping and routes '+key)
            p.require(len(call['execution_input_metadata'])==7,'All seven inputs')
            same(call['stage_names'],old['stage_names'],'All original ordered taps')
            same(list(call['stage_sha256']),call['stage_names'],'Every tap digest present')
            same(list(call['stage_metadata']),call['stage_names'],'Every tap layout present')
            p.require(len(call['stage_names'])==213 and all(re.fullmatch('[0-9a-f]{64}',v)is not None for v in call['stage_sha256'].values()),'All213 valid tap digests')
            before=state_values(call['before_metadata']);state_values(call['input_after_metadata']);post=state_values(call['post_metadata'])
            same(call['input_after_metadata'],call['before_metadata'],'All84 input states unchanged')
            same(before,previous if ci else state_values(old['before_metadata']),'Actual own-history chain including call7; original call0 seed')
            previous=post
            same(call['seed_imported_this_call'],ci==0,'Only call0 seed import')
            same(call['history_origin'],'original_call0_seed'if ci==0 else'previous_actual_post_same_arm','Actual own-cache lineage')
            same(call['group_reset_preserves_cache'],ci==7,'Call7 preserves actual cache')
            same([row['layer']for row in call['route_records']],list(range(42)),'All42 mask routes')
            preserve=call['preservation']
            same(preserve['inactive_slots'],sorted(set(range(4))-{f['slot']for f in call['call']['frames']}),'Inactive slot mapping')
            p.require(preserve['all84_inactive_slots_raw_exact']is preserve['all84_physical_padding_raw_exact']is True,'Inactive slots and padding preserved')
            enabled=patch_enabled(arm,call['call']['phase'])
            same(call['candidate_layer0_full_bf16_enabled'],enabled,'Finite phase-specific intervention')
            same(call['candidate_targets'],[TARGET]if enabled else[],'Only exact layer0 target')
            witnesses=call['oproj_dtype_records'];same([row['target']for row in witnesses],[TARGET]if enabled else[],'One or zero candidate witnesses')
            input_shape=[1,call['bucket'],1024]if call['call']['phase']=='prefill'else[4,1,1024]
            for row in witnesses:
                same(row['input_shape'],input_shape,'Original full projection input')
                same(row['output_shape'],input_shape[:2]+[2048],'Original full projection output')
                same([row[k]for k in ('input_dtype','gathered_dtype','weight_dtype','output_dtype')],['torch.bfloat16']*4,'All BF16 gather and full projection')
                same(row['gathered_shape'],input_shape[:2]+[2048],'Actual complete gathered input')
                same(row['full_weight_shape'],[2048,2048],'Complete original weight geometry');same(row['full_weight_sha256'],FULL_WEIGHT_SHA,'Complete original weight bytes')
                same(row['actual_group_ranks'],[0,1],'Actual NxD TP rank order')
                p.require(row['actual_tp_size']==2 and row['full_linear_calls']==1 and row['output_reduction']is False and row['explicit_fp32_casts']==0 and row['original_parameter_object_unchanged']is True,'One BF16 full projection and original parameter identity')
                same(row['actual_gather_function'],'torch.distributed.all_gather','Actual BF16 input gather')
            if arm_index==0:
                same(call['strict_anchor_ref'],oldref,'Frozen strict anchor reference')
                same(call['strict_anchor_exact'],dict(all7_inputs_exact=True,all213_taps_exact=True,all84_before_input_post_exact=True,full_logits_exact=True),'Anchor exact witness')
                for key in REPEAT_FIELDS+('HF_metrics','physical_rows','preservation'):same(call[key],old[key],'Frozen strict anchor '+key)
            for name in ('embed_tokens','layers.0.input_layernorm'):
                same(call['stage_metadata'][name],old['stage_metadata'][name],'Before selected projection original metadata')
                same(call['stage_sha256'][name],old['stage_sha256'][name],'Before selected projection original digest')
            same(call['upstream_matches_anchor'],dict(names=['embed_tokens','layers.0.input_layernorm'],all_exact=True),'Pre-projection anchor proof')
            if arm_index==2 and ci<4:
                same(call['first_group_prefill_repeat_exact'],dict(all7_inputs_exact=True,all213_taps_exact=True,all84_before_input_post_exact=True,full_logits_exact=True),'First-group candidate repeat proof')
                for key in REPEAT_FIELDS+('HF_metrics','physical_rows','preservation'):same(call[key],prefill_only[ci][key],'First group prefill candidates exact '+key)
            if arm_index==2 and ci==4:
                p.require(call['call4_before_equal_prefill_only']is True,'First decode incoming proof')
                same(call['before_metadata'],prefill_only[ci]['before_metadata'],'First decode equal candidate incoming caches')
            if arm_index!=0:
                same(call['strict_anchor_ref'],None,'No false exact-anchor claim on candidate')
                same(call['strict_anchor_exact'],None,'No false exact-anchor witness on candidate')
            if not(arm_index==2 and ci<4):same(call['first_group_prefill_repeat_exact'],None,'No unrelated first-group repeat witness')
            if not(arm_index==2 and ci==4):same(call['call4_before_equal_prefill_only'],None,'No unrelated incoming decode witness')
            same(call['payload'],call_entry['payload'],'Full logits payload bound')
            same(call['payload']['path'],str(callfolder/'output.pt'),'Exact logits payload path')
            p.require(type(call['payload']['bytes'])is int and 0<call['payload']['bytes']<=3*2**20 and re.fullmatch('[0-9a-f]{64}',call['payload']['sha256'])is not None,'Bounded logits payload metadata')
            same(call['output_metadata']['shape'],[len(call['physical_rows']),1,130560],'Complete vocabulary logits')
            same(call['output_metadata']['dtype'],'torch.float32','FP32 logits')
            p.require(re.fullmatch('[0-9a-f]{64}',call['output_metadata']['sha256'])is not None,'Complete logits digest')
            tap_bytes+=validate_selected_bundle(call,callfolder)
            same([row['frame']for row in call['HF_metrics']],plan['calls'][ci]['frames'],'Every original HF frame')
            for row in call['HF_metrics']:
                values=row['metrics'];p.require(all(math.isfinite(values[k])for k in ('cosine','kl_reference_to_neuron','max_abs_error','reference_margin')),'Finite HF metrics')
                checks=dict(cosine=values['cosine']>=.995,KL=values['kl_reference_to_neuron']<=.02,max_abs=values['max_abs_error']<=2,high_margin_argmax=values['reference_margin']<=.5 or values['argmax_match'])
                same(row['failed_criteria'],[k for k,v in checks.items()if not v],'Original thresholds')
            calls.append(call);scores.extend(call['HF_metrics'])
        same(tap_bytes,524288,'Exact bounded per-arm tap bytes')
        same([row['frame']for row in scores],plan['frames'],'All32 original frame order')
        same(report['rows'],scores,'Arm scores derive from calls')
        fraction=sum(row['metrics']['argmax_match']for row in scores)/32;summary=report['summary']
        same([summary[k]for k in ('complete','frames','prefill_frames','decode_frames')],[True,32,8,24],'Original scoring coverage')
        same(summary['failed_frames'],[row['frame']['index']for row in scores if row['failed_criteria']],'Exact failure frames')
        same(summary['prefill_failed'],sum(bool(row['failed_criteria'])and row['frame']['step']==0 for row in scores),'Exact prefill failures')
        same(summary['decode_failed'],sum(bool(row['failed_criteria'])and row['frame']['step']>0 for row in scores),'Exact decode failures')
        same(summary['argmax_match_fraction'],fraction,'Exact top1 fraction')
        same(summary['passed'],all(not row['failed_criteria']for row in scores)and fraction>=.9,'Original aggregate gate')
        same(summary,result['summaries'][arm],'Rank summary bound')
        if arm_index==1:prefill_only=calls
        output.append(dict(arm=arm,calls=calls,summary=summary))
    return output


def validate_rank_pair(results,plan):
    p.require(len(results)==2,'Exactly two ranks')
    evidence=[validate_rank_result(result,plan,rank)for rank,result in enumerate(results)]
    for left,right in zip(*evidence):
        same(left['arm'],right['arm'],'Same paired arm')
        same(left['summary'],right['summary'],'Cross-rank summary')
        for a,b in zip(left['calls'],right['calls']):
            same(a['output_metadata'],b['output_metadata'],'Cross-rank complete logits tensor bytes/layout')
            same(a['HF_metrics'],b['HF_metrics'],'Cross-rank HF metrics')
            aa,bb=a['selected_tap_bundle'],b['selected_tap_bundle']
            for key in ('sha256','encoding','raw_bytes','entries'):same(aa[key],bb[key],'Cross-rank selected tap bundle '+key)
    return dict(complete=True,rank_count=2,arm_count=3,calls_per_rank=42,HF_frames_per_arm=32,
                selected_tap_raw_bytes_total=3145728,cross_rank_full_logits_tensor_hashes_equal=True,
                cross_rank_HF_metrics_equal=True,cross_rank_selected_tap_bundles_equal=True,raw_payloads_read=False)


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
        p.clear_window();usage(root,spec['root_floor_bytes']);return permit
    context=RuntimeContext(plan,digest(plan),permit,guard);context.verify_sources()
    import minicpm5_original32_layer0_full_bf16_cpu_worker_v1 as worker
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
        fs=os.statvfs(ROOT);p.require(fs.f_bavail*fs.f_frsize>=spec['root_floor_bytes']+8*2**20,'Reserve root metadata headroom')
        fs=os.statvfs('/mnt/nvme');p.require(fs.f_bavail*fs.f_frsize>=spec['output_floor_bytes']+CAP,'Reserve full output cap above NVMe floor')
        root.mkdir(mode=0o700);usage(root,spec['root_floor_bytes'])
        a=dict(schema='minicpm5_original32_layer0_full_bf16_cpu_admission_v1',spec=spec_ref,review=review_ref,plan=spec['plan'],outer_owner=owner,
               lock_identity=lock,attempt_id=os.urandom(16).hex(),predecessors_checked=True,later_owners=owners,predecessor_actual_exit=PREVIOUS_EXIT,observed_drains=drains)
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
            p.require(time.monotonic()<deadline,'Bounded three-arm trajectory deadline');time.sleep(1)
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
