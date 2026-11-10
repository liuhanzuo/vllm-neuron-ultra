"""Partition-aware TP2 SDK batch/layout v2 runner; no admission without actual r12 exit."""
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
from minicpm5_layer0_runtime_auth_v1 import digest, identity
from minicpm5_layer1_sdk_batch_layout_runtime_context_v2 import RuntimeContext
from research import minicpm5_layer1_sdk_batch_layout_sources_v2 as source_auth
import minicpm5_original32_layer0_full_bf16_cpu_runner_v1 as original_runner
import minicpm5_original32_layer1_observation_cpu_runner_v2 as collector
from research import minicpm5_layer1_sdk_batch_layout_results_v1 as layer1_results
from research import minicpm5_layer1_after_saved_audit_v3 as after_audit
from research import minicpm5_layer1_after_own_history_v1 as own
from research import minicpm5_layer1_after_cross_history_v1 as crossed
from research import minicpm5_layer1_after_kv_factorial_v1 as factored
from research import minicpm5_layer1_after_native_cache_v1 as official
from research import minicpm5_layer1_after_official_history_v1 as latest_completed
from research.minicpm5_layer1_current_source_view_v1 import current_source_view
from research import minicpm5_layer1_sdk_batch_layout_completion_v1 as latest
from research.minicpm5_layer1_drain_hold_v1 import retain_until_drained
OBSERVED_EXIT=dict(path='/home/ubuntu/experiments/trn1-latest-models-20261031/research/minicpm5-layer0-full-bf16-cpu-root-exit-r1-20261108-v1.json',
 sha256='727d92a3189bc671016701e53bfc982547c33917cb046dfd2ca29eb78f84e0ff')
RESULTS_SOURCE_PLAN=dict(path='/home/ubuntu/experiments/trn1-latest-models-20261031/research/minicpm5-layer1-own-history-driver-source-plan-20261110-v1.json',
 sha256='bf3a581007ee578566f171335c1ae60dc3d799a9e951c8bc2336f69a6f6d0705')

p, five, post, same, ROOT = masks.p, masks.five, masks.post, masks.same, masks.ROOT
SELF = str(Path(__file__).resolve())
PYTHON = '/home/ubuntu/venvs/vllm_neuron/bin/python'
CAP = 256 * 2**20
RANK_CAP = 128 * 2**20


PROPS = dict(MemoryMax=str(16*2**30), MemoryHigh=str(14*2**30), MemorySwapMax='0', TasksMax='128', AllowedCPUs='6-7',
             CPUAffinity='6-7', Nice='19', KillMode='control-group', RemainAfterExit='yes', SendSIGKILL='yes',
             WorkingDirectory=str(ROOT), LimitFSIZE=str(16*2**20), RuntimeMaxSec='7200', TimeoutStopSec='15')

def reference(path):
    # Exact source/metadata transports, including named large JSON envelopes.
    path=str(path)
    if source_auth.envelope_kind(path)is not None:
        raw=source_auth.metadata.stable_bytes(path,source_auth.metadata.ENVELOPE_LIMIT)
        source_auth.envelope_shape(path,json.loads(raw.decode('utf8')))
    else:raw=source_auth.text(path)
    return dict(path=path,sha256=hashlib.sha256(raw).hexdigest())

def verify_loaded_sources(plan,guard=lambda:p.clear_window()):
    checked={}
    for module in tuple(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if path and str(path).startswith(str(ROOT)+'/'):
            guard();path=str(Path(path).resolve());wanted=plan['source_sha256'].get(path)
            p.require(wanted is not None,'Every loaded local helper belongs to complete reviewed closure')
            same(hashlib.sha256(source_auth.text(path,wanted)).hexdigest(),wanted,'Exact loaded local source pin')
            checked[path]=wanted
    guard();return dict(complete=True,count=len(checked),source_sha256=checked)


def read_bundle(spec_ref,review_ref,guard=lambda:p.clear_window()):
    guard();spec=source_auth.load(spec_ref);latest.require_complete_binding(spec['latest_completion'])
    review=source_auth.load(review_ref)
    p.require(spec['schema']=='minicpm5_layer1_sdk_batch_layout_cpu_execution_spec_v2','Exact new observation runtime')
    p.require(review['schema']=='minicpm5_layer1_sdk_batch_layout_runtime_source_review_v2'
        and review['reviewer']=='/root' and review['passed']is True and review['blockers']==[],
        'Final concrete runtime source review')
    same(review['spec'],spec_ref,'Exact reviewed runtime spec')
    same(review['source_sha256'],spec['source_sha256'],'Complete reviewed source closure')
    same([spec['runner'],spec['worker']],[SELF,str(ROOT/'minicpm5_layer1_sdk_batch_layout_cpu_worker_v2.py')],
         'Only new collector runner and worker')
    partition=source_auth.classify(spec['source_sha256'],spec['source_contract'])
    source_auth.verify_text_sources(partition,guard)
    for path in (SELF,spec['worker']):
        wanted=spec['source_sha256'].get(path)
        same(hashlib.sha256(source_auth.text(path,wanted)).hexdigest(),wanted,'Both actual entrypoints pinned')
    plan=source_auth.load(spec['plan'])
    same(plan['source_sha256'],spec['source_sha256'],'Scientific runtime source closure')
    same(plan['source_contract'],spec['source_contract'],'One exact reviewed source contract')
    same(plan['output_root'],spec['output_root'],'Exact output root')
    p.require(Path(spec['output_root']).parent==Path('/mnt/nvme')
        and Path(spec['output_root']).name.startswith('minicpm5-layer1-sdk-batch-layout-'),
        'Fresh layer1-only output namespace')
    same([spec['output_cap_bytes'],spec['root_floor_bytes'],spec['output_floor_bytes']],
         [CAP,28*2**30,30*2**30],'Original output cap and current serial storage floors')
    same(spec['resources'],PROPS,'Original16GiB/TP2 CPU resources')
    p.require(type(spec['master_port'])is int and 29000<spec['master_port']<30000,'Finite rendezvous port')
    p.require(re.fullmatch(r'trn1-mini-layer1-sdk-batch-layout-[a-z0-9-]{1,80}',spec['unit'])is not None,'Fresh layer1 service')
    p.require(spec['model_qualification_claim']is spec['automatic_full_gates_launch']is False,'Diagnostic only')
    same(spec['original_completed_collection'],OBSERVED_EXIT,'Actual completed original trajectory')
    same(spec['prepared_source_plan'],RESULTS_SOURCE_PLAN,'Reviewed complete observation/validator source plan')
    same(spec['latest_completion'],plan['latest_completion'],'Latest complete external r12 and MiniCPM lineage')
    same(spec['completed_current_history'],plan['layer1_sdk_batch_layout'],'Completed full180-call current/history prerequisite')
    same(spec['completed_native_cache'],plan['layer1_official_history'],'Completed native capture/audit/comparison')
    same(spec['completed_official_history'],plan['layer1_current_history_factorial'],'Completed original official history')
    same(spec['completed_storage'],plan['completed_storage_maintenance'],'Completed storage maintenance and audit')
    same(spec['completed_factorial'],plan['layer1_native_cache'],'Completed independent K/V matrix')
    same(spec['completed_cross_history'],plan['layer1_kv_factorial'],'Actual completed paired-history and audit')
    same(spec['completed_own_history'],plan['layer1_cross_history'],'Actual completed own-history plus post-exit audit')
    same(spec['completed_layer1_data'],plan['layer1_own_history_replay'],'Actual collection and audit binding')
    verify_loaded_sources(plan,guard)
    return spec,plan


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
    fs=os.statvfs('/mnt/nvme');p.require(fs.f_bavail*fs.f_frsize >=30*2**30,'Current serial NVMe floor')
    return dict(allocated_bytes=total, files=count, per_rank_allocated_bytes=ranks)

def env(attempt, port, rank=None):
    result = p.environment({}, 'cpu_control', attempt)
    result.update(OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    if rank is not None:
        result.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE='2', LOCAL_WORLD_SIZE='2', MASTER_ADDR='127.0.0.1',
                      MASTER_PORT=str(port), TORCHELASTIC_RUN_ID='mini-layer1-sdk-batch-layout-'+attempt)
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







def validate_current_history_extension(plan,done):
    previous=done['current_history']['plan']
    extensions={'source_sha256','inputs','output_root','layer1_sdk_batch_layout','latest_completion','source_contract'}
    same({k:v for k,v in plan.items()if k not in extensions},
        {k:v for k,v in previous.items()if k not in extensions},
        'Every completed current/history scientific and supplemental field retained')
    for key in ('inputs','source_sha256'):
        p.require(all(plan[key].get(path)==value for path,value in previous[key].items()),
            'Every completed current/history input and source retained')


def check_prerequisites(spec,guard):
    guard();latest.require_complete_binding(spec['latest_completion'])
    latest.require_latest_terminal(spec['latest_completion'])
    done=latest.completed(spec['latest_completion'],source_auth.load,guard)
    bundle=done['collector'];cs=bundle['collector_spec'];original=bundle['collector_plan']
    plan=source_auth.load(spec['plan']);validate_current_history_extension(plan,done)
    same(plan['latest_completion'],spec['latest_completion'],'Latest plan binding')
    same(done['current_history']['binding'],spec['completed_current_history'],'Completed current/history binding')
    same(done['native']['binding'],spec['completed_native_cache'],'Completed native binding')
    same(bundle['binding'],spec['completed_layer1_data'],'Completed collected data binding')
    allowed={'source_sha256','inputs','output_root','layer1_own_history_replay','layer1_cross_history',
        'layer1_kv_factorial','layer1_native_cache','layer1_official_history',
        'layer1_current_history_factorial','completed_storage_maintenance','layer1_sdk_batch_layout','latest_completion','source_contract'}
    same({k:v for k,v in plan.items()if k not in allowed},
        {k:v for k,v in original.items()if k not in allowed},'Original complete scientific trajectory')
    for field in ('inputs','source_sha256'):
        p.require(all(plan[field].get(path)==value for path,value in original[field].items()),
            'All actual collected scientific inputs and sources retained')
    prepared=source_auth.load(RESULTS_SOURCE_PLAN)
    for sources in (prepared['source_sha256'],bundle['saved_audit_spec']['source_sha256'],done['source_sha256']):
        p.require(all(plan['source_sha256'].get(path)==value for path,value in sources.items()),
            'Complete prepared and actual completed source closure retained')
    for key in ('finalization','catalog','finalizer_exit','canonical_sha256','nemotron_completion'):
        same(spec[key],cs[key],'Original finalized prerequisite '+key)
    verify_loaded_sources(plan,guard)
    guard();return done


def predecessor_drains(done,guard):
    # This interface covers the complete inherited chain plus all r12 services.
    guard();return latest.physical(done,guard)



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












def admitted_bundles(plan,guard):
    guard();latest.require_complete_binding(plan['latest_completion'])
    def read(ref):
        guard();row=plan['inputs'][ref['path']]
        same(row['sha256'],ref['sha256'],'Admitted completion metadata digest')
        before=identity(ref['path']);same(before,row['identity'],'Admitted completion metadata identity')
        value=source_auth.load(ref);same(identity(ref['path']),before,'Stable completion metadata read');return value
    done=latest.completed(plan['latest_completion'],read,guard)
    validate_current_history_extension(plan,done)
    same(done['current_history']['binding'],plan['layer1_sdk_batch_layout'],'Latest authenticated current/history')
    same(done['native']['binding'],plan['layer1_official_history'],'Latest authenticated native cache')
    for path,sha in done['source_sha256'].items():
        same(plan['source_sha256'].get(path),sha,'Completed sources retained in admitted plan')
    verify_loaded_sources(plan,guard)
    def read_json(path,sha):return read(dict(path=path,sha256=sha))
    return done,read_json


def validate_rank_result(result,plan,rank,guard):
    done,read=admitted_bundles(plan,guard)
    same(result['latest_completion'],plan['latest_completion'],'Worker latest completion unchanged')
    return layer1_results.validate_rank_result(result,plan,rank,done['collector'],done['native'],done['current_history'],read,guard)



def validate_rank_pair(results,plan,guard):
    done,read=admitted_bundles(plan,guard)
    for result in results:same(result['latest_completion'],plan['latest_completion'],'Both latest completion bindings unchanged')
    return layer1_results.validate_rank_pair(results,plan,done['collector'],done['native'],done['current_history'],read,guard)



def wait_file(path, guard, seconds=60):
    end = time.monotonic()+seconds
    while not Path(path).exists():
        guard(); p.require(time.monotonic() < end, 'Parent identity acknowledgement timeout'); time.sleep(.05)
    return source_auth.load(reference(path))

def child_admission(admission_ref):
    admission=source_auth.load(admission_ref)
    lease=lambda:p.lock_proof(admission['outer_owner'],admission['lock_identity'])
    lease();spec,plan=read_bundle(admission['spec'],admission['review'],lease)
    same(admission['plan'],spec['plan'],'Current run plan binding')
    same(admission['source_contract'],spec['source_contract'],'Actual admitted source contract')
    p.require(admission['schema']=='minicpm5_layer1_sdk_batch_layout_cpu_admission_v2','Exact successor admission')
    lease();return admission,spec,plan

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
        results=[source_auth.load(reference(root/('rank'+str(r)+'-result.json')))for r in (0,1)]
        for rank,result in enumerate(results):
            same(result['process'],records[rank],'Rank result identity');p.require(result['complete']is True and result['rank']==rank,'Complete rank result')
        validated=validate_rank_pair(results,plan,guard)
        p.once(root/'coordinator-result.json',dict(complete=True,rank_exits=[0,0],rank_owners=records,results=results,validated=validated,self_exit_claimed=False))
        success=True
    finally:
        cleanup_children(children,records,root,success,sys.exc_info()[1])

class RankGuard:
    PERIOD_SECONDS=1.0
    def __init__(self,admission,spec,plan,process,coordinator,done,permit):
        self.admission,self.spec,self.plan=admission,spec,plan
        self.process,self.coordinator,self.done,self.permit=process,coordinator,done,permit
        self.fast_checks=0;self.full_checks=0;self.last_full=None;self.events=[];self.metadata={};self.sources_verified=False
    def fast(self):
        a=self.admission
        p.lock_proof(a['outer_owner'],a['lock_identity'])
        same(p.proc(self.coordinator['pid']),{k:self.coordinator[k]for k in self.process},
            'Coordinator remains exact acknowledged original process')
        p.clear_window();self.fast_checks+=1
    def watch_metadata(self,observed):
        for path,wanted in observed.items():
            p.require(path not in self.metadata or self.metadata[path]==wanted,'No metadata identity rebound')
            self.metadata[path]=wanted
    def full(self,boundary):
        self.fast()
        for owner in self.done['owners']:p.gone(owner)
        for path,wanted in self.metadata.items():same(identity(path),wanted,'Every completed metadata identity unchanged')
        usage(Path(self.spec['output_root']),self.spec['root_floor_bytes'])
        self.last_full=time.monotonic();self.full_checks+=1
        self.events.append(dict(boundary=boundary,monotonic=self.last_full,owners=len(self.done['owners']),
            metadata=len(self.metadata)))
        return self.permit
    def __call__(self):
        self.fast()
        if self.last_full is None or time.monotonic()-self.last_full>=self.PERIOD_SECONDS:self.full('periodic')
        return self.permit
    def mark_sources_verified(self,context):
        p.require(context.sdk_guard is self and context._sources_checked is True,'Actual complete context source verification first')
        self.sources_verified=True
    def evidence(self):
        return dict(schema='minicpm5_layer1_sdk_batch_layout_guard_checks_v2',fast_checks=self.fast_checks,
            full_checks=self.full_checks,period_seconds=self.PERIOD_SECONDS,events=copy.deepcopy(self.events),
            every_source_fully_authenticated=self.sources_verified,full_source_checks_not_sampled=True,
            every_call_keeps_original_serial_lock_process_and_performance_checks=True)


def rank(admission_ref):
    a,spec,plan=child_admission(admission_ref);root=Path(spec['output_root']);rank_id=int(os.environ['RANK'])
    p.require(rank_id in (0,1),'TP2 rank')
    lease=lambda:p.lock_proof(a['outer_owner'],a['lock_identity'])
    process=wait_file(root/('rank'+str(rank_id)+'-owner.json'),lease);same(process,p.proc(os.getpid()),'Parent observed actual rank')
    coordinator_owner=source_auth.load(reference(root/'coordinator-owner.json'));same(process['ppid'],coordinator_owner['pid'],'Parent coordinator')
    stable=env(a['attempt_id'],spec['master_port'],rank_id);stable.pop('NEURON_RT_VISIBLE_CORES')
    permit=dict(schema='minicpm5_layer0_runtime_permit_v1',plan_sha256=digest(plan),role='cpu',rank=rank_id,arm='C_Hc_history_generation',
                attempt_id=a['attempt_id'],serial_exclusive=True,source_review_accepted=True,all_predecessors_drained=True,uid=1000,
                **process,python=PYTHON,environment=stable,output_root=spec['output_root'],
                runtime_admission=admission_ref,runtime_plan=a['plan'],outer_owner=a['outer_owner'],
                lock_identity=a['lock_identity'],observed_drains=a['observed_drains'],source_authentication=a['source_authentication'])
    done,_=admitted_bundles(plan,lease)
    same(done['lock_identity'],a['lock_identity'],'Latest lineage uses the actual outer lock')
    same(coordinator_owner['unit'],spec['unit'],'Exact owned service acknowledged by outer')
    same(process['cgroup'],'0::'+p.GROUP_PREFIX+spec['unit']+'.service','Exact owned rank service group')
    same(coordinator_owner['argv'],argv('coordinator',admission_ref),'Exact admitted coordinator command')
    guard=RankGuard(a,spec,plan,process,coordinator_owner,done,permit)
    guard.full('before_source_verification')
    context=RuntimeContext(plan,digest(plan),permit,guard);context.sdk_guard=guard
    context.verify_loaded_sources=lambda:verify_loaded_sources(plan,guard)
    context.verify_sources();guard.mark_sources_verified(context);guard.full('after_source_verification')
    import minicpm5_layer1_sdk_batch_layout_cpu_worker_v2 as worker
    try:
        verify_loaded_sources(plan,guard)
        result=worker.run(context,root/('cpu-rank'+str(rank_id)))
        verify_loaded_sources(plan,guard);guard.full('after_worker')
        p.require(result['complete']is True and result['rank']==rank_id,'Finite worker complete')
        validate_rank_result(result,plan,rank_id,guard)
        guard.full('before_rank_output')
        result=dict(result,process=process,admission=admission_ref,self_exit_claimed=False,guard_checks=guard.evidence())
        p.once(root/('rank'+str(rank_id)+'-result.json'),result)
    except BaseException as error:
        guarded_receipt(root/('rank'+str(rank_id)+'-failure.json'),dict(process=process,error_type=type(error).__name__,error=str(error),self_exit_claimed=False),error);raise

def compact_drain(proof):
    # Keep all units, owners and samples, compressing only systemd properties.
    fields={'Id','LoadState','ActiveState','SubState','MainPID','ControlPID',
        'ExecMainPID','ExecMainCode','ExecMainStatus','Result','InvocationID'}
    def bounded(value):
        if type(value)is list:return [bounded(v)for v in value]
        if type(value)is not dict:return copy.deepcopy(value)
        return {k:({f:copy.deepcopy(x)for f,x in v.items()if f in fields}
            if k=='properties' and type(v)is dict else bounded(v))for k,v in value.items()}
    return dict(complete=True,complete_observation_sha256=digest(proof),latest_physical=bounded(proof))

def verify_owned_drain(spec,admission_ref,attempt,service_owner,launch_attempted):
    root=Path(spec['output_root']);value=p.properties(spec['unit'])
    if value['LoadState']=='loaded':
        p.require(launch_attempted and admission_ref is not None,'No unknown service cleanup')
        validate_service_identity(spec,admission_ref,attempt,value,service_owner)
        p.require(p.terminal(value),'Original service must be terminal')
    else:
        p.require(value['LoadState']=='not-found' and not value['InvocationID']
            and value['MainPID']==value['ControlPID']==value['ExecMainPID']=='0','Original service fully collected')
    owners=[]
    if service_owner is not None:owners.append(service_owner)
    for rank_id in (0,1):
        path=root/('rank'+str(rank_id)+'-owner.json')
        if path.exists():owners.append(source_auth.load(reference(path)))
    for owner in owners:p.gone(owner)
    group=p.empty_group(p.GROUP_PREFIX+spec['unit']+'.service')
    return dict(physically_drained=True,properties=value,group=group,owners_checked=owners)

def cleanup_owned_service(spec,admission_ref,attempt,service_owner,launch_attempted):
    if not launch_attempted:return
    root=Path(spec['output_root']);intent=source_auth.load(reference(root/'launch-intent.json'))
    same(intent['command'],service_command(spec,admission_ref,attempt),'Exact durable owned launch')
    same(intent['admission'],admission_ref,'Cleanup binds original runtime admission')
    value=p.properties(spec['unit'])
    if value['LoadState']=='not-found':return
    validate_service_identity(spec,admission_ref,attempt,value,service_owner)
    p.command(['systemctl','--user','stop',spec['unit']],timeout=30)

def outer(spec_ref,review_ref):
    spec,plan=read_bundle(spec_ref,review_ref);latest.require_latest_terminal(spec['latest_completion'])
    root=Path(spec['output_root'])
    p.require(not root.exists(),'Fresh SDK batch/layout output');p.user_bus()
    fd,owner,lock=post.external.acquire_lock()
    lease=lambda:p.lock_proof(owner,lock)
    launch_attempted=False;service_owner=None;admission_ref=None;admission=None;failure=None;handlers={}
    def interrupted(signum,frame):raise InterruptedError('Layer1 SDK batch/layout signal '+str(signum))
    try:
        for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):handlers[sig]=signal.signal(sig,interrupted)
        lease();p.clear_window();p.active_heavy()
        done=check_prerequisites(spec,lease)
        same(lock,done['lock_identity'],'Same serial lock throughout the latest complete lineage')
        p.gone(done['bound']['finalizer_owner'])
        for previous in done['owners']:p.gone(previous)
        drains=predecessor_drains(done,lease)
        p.control_usage(done['bound']['spec'])
        same({n:p.sha(ROOT/n)for n in spec['canonical_sha256']},spec['canonical_sha256'],'Original canonical files')
        p.require(p.properties(spec['unit'])['LoadState']=='not-found','Fresh service name')
        fs=os.statvfs(ROOT);p.require(fs.f_bavail*fs.f_frsize>=spec['root_floor_bytes']+8*2**20,'Root metadata reserve')
        fs=os.statvfs('/mnt/nvme');p.require(fs.f_bavail*fs.f_frsize>=spec['output_floor_bytes']+CAP,'Full original output reserve')
        root.mkdir(mode=0o700);usage(root,spec['root_floor_bytes'])
        attempt_id=os.urandom(16).hex()
        partition=source_auth.classify(spec['source_sha256'],spec['source_contract'])
        native_receipt=source_auth.authenticate_native_sources(partition,owner,lock,attempt_id,lease)
        p.once(root/'source-authentication.json',native_receipt)
        verify_loaded_sources(plan,lease)
        p.once(root/'predecessor-drain.json',dict(compact_drain(drains),latest_completion=spec['latest_completion'],
            outer_owner=owner,lock_identity=lock))
        admission=dict(schema='minicpm5_layer1_sdk_batch_layout_cpu_admission_v2',spec=spec_ref,review=review_ref,plan=spec['plan'],
            outer_owner=owner,lock_identity=lock,attempt_id=attempt_id,predecessors_checked=True,
            source_contract=spec['source_contract'],source_authentication=reference(root/'source-authentication.json'),
            later_owners=done['owners'],original_completed_collection=OBSERVED_EXIT,
            latest_completion=spec['latest_completion'],completed_current_history=spec['completed_current_history'],
            completed_layer1_data=spec['completed_layer1_data'],completed_native_cache=spec['completed_native_cache'],
            completed_collector_service=done['collector']['collector_spec']['unit'],
            observed_drains=reference(root/'predecessor-drain.json'),runtime_admitted=True,
            full_model_qualified=False,new_speedup_claim=False)
        p.once(root/'admission.json',admission);admission_ref=reference(root/'admission.json')
        command=service_command(spec,admission_ref,admission['attempt_id'])
        p.once(root/'launch-intent.json',dict(command=command,admission=admission_ref))
        launch_attempted=True;p.command(command,timeout=30)
        deadline=time.monotonic()+60
        while service_owner is None:
            value=validate_service(spec,admission_ref,admission['attempt_id'],p.properties(spec['unit']))
            pid=int(value['MainPID']);actual=p.proc(pid)if pid else None
            if actual:
                same(actual['argv'],argv('coordinator',admission_ref),'Actual coordinator argv')
                same(actual['cgroup'],'0::'+p.GROUP_PREFIX+spec['unit']+'.service','Owned observation group')
                service_owner=dict(actual,unit=spec['unit'],invocation_id=value['InvocationID'])
                p.once(root/'coordinator-owner.json',service_owner)
            else:
                p.require(not p.terminal(value)and time.monotonic()<deadline,'Actual coordinator acknowledgement')
                time.sleep(.05)
        deadline=time.monotonic()+7230
        while True:
            lease();p.clear_window();usage(root,spec['root_floor_bytes']);p.active_heavy((spec['unit'],))
            value=validate_service(spec,admission_ref,admission['attempt_id'],p.properties(spec['unit']),service_owner)
            if p.terminal(value):break
            p.require(time.monotonic()<deadline,'Original two-hour full trajectory bound');time.sleep(1)
        drain=verify_owned_drain(spec,admission_ref,admission['attempt_id'],service_owner,True)
        success=value['Result']=='success'and value['ExecMainCode']=='1'and value['ExecMainStatus']=='0'
        p.once(root/'service-exit.json',dict(owner=service_owner,properties=value,group=drain['group'],
            observed_by_outer=owner,successful=success))
        p.require(success,'Actual complete SDK batch/layout job failed')
        result=source_auth.load(reference(root/'coordinator-result.json'))
        p.require(result['complete']is True and result['rank_exits']==[0,0],'Both original rank exits0')
        same({n:p.sha(ROOT/n)for n in spec['canonical_sha256']},spec['canonical_sha256'],'Canonical files unchanged')
        p.once(root/'outer-result.json',dict(complete=True,owner=owner,service_exit=reference(root/'service-exit.json'),
            coordinator_result=reference(root/'coordinator-result.json'),usage=usage(root,spec['root_floor_bytes']),
            self_exit_claimed=False,raw_output_independently_reread=True,full_model_qualified=False,new_speedup_claim=False))
        return reference(root/'outer-result.json')
    except BaseException as error:
        failure=dict(error_type=type(error).__name__,error=str(error),owner=owner,self_exit_claimed=False)
        if root.exists():guarded_receipt(root/'outer-failure.json',failure,error)
        raise
    finally:
        retry=[True]
        def retry_requested(signum,frame):retry[0]=True
        for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP,signal.SIGUSR1):
            if sig not in handlers:handlers[sig]=signal.getsignal(sig)
            signal.signal(sig,retry_requested)
        attempt=admission['attempt_id']if admission is not None else None
        def cleanup():
            lease();cleanup_owned_service(spec,admission_ref,attempt,service_owner,launch_attempted)
        def verify():
            lease();return verify_owned_drain(spec,admission_ref,attempt,service_owner,launch_attempted)
        def pending(value):
            if root.exists():p.once(root/'cleanup-pending.json',dict(value,owner=owner,original_failure=failure))
        def await_retry():
            while not retry[0]:lease();time.sleep(.5)
            retry[0]=False
        # No release in an exception/finally path can bypass physical proof.
        proof=retain_until_drained(cleanup,verify,pending,await_retry)
        if root.exists():
            try:p.once(root/'cleanup-drained.json',dict(proof,owner=owner,original_failure=failure))
            except BaseException as error:secondary_error(sys.exc_info()[1],error,'drained_receipt')
        post.external.release_lock(fd)
        for sig,handler in handlers.items():signal.signal(sig,handler)


if __name__=='__main__':
    cli=argparse.ArgumentParser();cli.add_argument('--mode',choices=['outer','coordinator','rank'],required=True)
    for key in ['spec','review','admission']:
        cli.add_argument('--'+key);cli.add_argument('--'+key+'-sha256')
    args=cli.parse_args();p.require(sys.dont_write_bytecode and __debug__,'python -B required')
    refs={k:dict(path=getattr(args,k),sha256=getattr(args,k+'_sha256'))for k in ['spec','review','admission']}
    if args.mode=='outer':print(json.dumps(outer(refs['spec'],refs['review']),indent=2))
    elif args.mode=='coordinator':coordinator(refs['admission'])
    else:rank(refs['admission'])
