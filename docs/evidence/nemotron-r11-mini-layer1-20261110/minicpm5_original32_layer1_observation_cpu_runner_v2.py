"""Complete layer1 observation collector. No launch without concrete source review and actual r11 release."""
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
import minicpm5_original32_layer0_full_bf16_cpu_runner_v1 as original_runner
from research import minicpm5_layer1_runtime_results_v1 as layer1_results
from research import minicpm5_after_nemotron_r11_v1 as history
from research.minicpm5_layer1_drain_hold_v1 import retain_until_drained
from research import minicpm5_layer1_legacy_sources_v1 as legacy_sources
OBSERVED_EXIT=dict(path='/home/ubuntu/experiments/trn1-latest-models-20261031/research/minicpm5-layer0-full-bf16-cpu-root-exit-r1-20261108-v1.json',
 sha256='727d92a3189bc671016701e53bfc982547c33917cb046dfd2ca29eb78f84e0ff')
RESULTS_SOURCE_PLAN=dict(path='/home/ubuntu/experiments/trn1-latest-models-20261031/research/minicpm5-layer1-results-source-plan-20261110-v1.json',
 sha256='53bf5ef6eb8205666de2ebb7d0ba4b7e8fb77adec7847d9020911b438a269c59')

p, five, post, same, ROOT = masks.p, masks.five, masks.post, masks.same, masks.ROOT
SELF = str(Path(__file__).resolve())
PYTHON = '/home/ubuntu/venvs/vllm_neuron/bin/python'
CAP = 256 * 2**20
RANK_CAP = 128 * 2**20


PROPS = dict(MemoryMax=str(16*2**30), MemoryHigh=str(14*2**30), MemorySwapMax='0', TasksMax='128', AllowedCPUs='6-7',
             CPUAffinity='6-7', Nice='19', KillMode='control-group', RemainAfterExit='yes', SendSIGKILL='yes',
             WorkingDirectory=str(ROOT), LimitFSIZE=str(16*2**20), RuntimeMaxSec='7200', TimeoutStopSec='15')

def reference(path):
    return dict(path=str(path), sha256=p.sha(path))

def read_bundle(spec_ref,review_ref):
    spec=p.read_ref(spec_ref);review=p.read_ref(review_ref)
    p.require(spec['schema']=='minicpm5_layer1_observation_cpu_execution_spec_v1','Exact new observation runtime')
    p.require(review['schema']=='minicpm5_layer1_observation_runtime_source_review_v1'
        and review['reviewer']=='/root' and review['passed']is True and review['blockers']==[],
        'Final concrete runtime source review')
    same(review['spec'],spec_ref,'Exact reviewed runtime spec')
    same(review['source_sha256'],spec['source_sha256'],'Complete reviewed source closure')
    same([spec['runner'],spec['worker']],[SELF,str(ROOT/'minicpm5_original32_layer1_observation_cpu_worker_v1.py')],
         'Only new collector runner and worker')
    for path,sha in spec['source_sha256'].items():
        info=identity(path)
        p.require(Path(path).suffix not in ('.pt','.safetensors','.bin','.neff','.so','.whl','.raw','.zlib','.pb')
            and info['size']<8*2**20,'Original bounded source and text metadata policy')
        same(p.sha(path),sha,'Pinned collector source unchanged')
    for path in (SELF,spec['worker']):
        same(spec['source_sha256'].get(path),p.sha(path),'Both actual entrypoints pinned')
    plan=p.read_ref(spec['plan'])
    same(plan['source_sha256'],spec['source_sha256'],'Scientific runtime source closure')
    same(plan['output_root'],spec['output_root'],'Exact output root')
    p.require(Path(spec['output_root']).parent==Path('/mnt/nvme')
        and Path(spec['output_root']).name.startswith('minicpm5-original32-layer0-full-bf16-cpu-layer1-'),
        'Fresh layer1-only output namespace')
    same([spec['output_cap_bytes'],spec['root_floor_bytes'],spec['output_floor_bytes']],
         [CAP,28*2**30,30*2**30],'Original output cap and current serial storage floors')
    same(spec['resources'],PROPS,'Original16GiB/TP2 CPU resources')
    p.require(type(spec['master_port'])is int and 29000<spec['master_port']<30000,'Finite rendezvous port')
    p.require(re.fullmatch(r'trn1-mini-layer1-observation-[a-z0-9-]{1,80}',spec['unit'])is not None,'Fresh layer1 service')
    p.require(spec['model_qualification_claim']is spec['automatic_full_gates_launch']is False,'Diagnostic only')
    same(spec['original_completed_collection'],OBSERVED_EXIT,'Actual completed original trajectory')
    same(spec['prepared_source_plan'],RESULTS_SOURCE_PLAN,'Reviewed complete observation/validator source plan')
    same(set(spec['nemotron_completion']),{'actual_exit','failed_release_exit'},'Actual future completion references only')
    for module in tuple(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if path and str(path).startswith(str(ROOT)+'/'):
            path=str(Path(path).resolve())
            same(spec['source_sha256'].get(path),p.sha(path),'Every loaded local helper pinned')
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
                      MASTER_PORT=str(port), TORCHELASTIC_RUN_ID='mini-layer1-observation-'+attempt)
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







def check_prerequisites(spec,guard):
    guard()
    completed=p.read_ref(OBSERVED_EXIT);original=p.read_ref(completed['spec'])
    p.require(completed['actual_exit_observed']is True and completed['actual_exit_code']==0
        and completed['all_four_processes_gone']is completed['cgroup_empty']is True,'Complete original CPU trajectory')
    for owner in (completed['owner'],completed['coordinator'],*completed['ranks']):p.gone(owner)
    for key in ('finalization','catalog','finalizer_exit','canonical_sha256'):
        same(spec[key],original[key],'Original finalized prerequisites retained')
    p.require(all(spec['source_sha256'].get(path)==sha for path,sha in original['source_sha256'].items()),
              'All original scientific sources retained')
    plan=p.read_ref(spec['plan']);prepared=p.read_ref(RESULTS_SOURCE_PLAN)
    same({k:v for k,v in plan.items()if k not in ('source_sha256','output_root')},
         {k:v for k,v in prepared.items()if k not in ('source_sha256','output_root')},
         'Only runtime source closure/output namespace extend reviewed scientific source plan')
    p.require(all(plan['source_sha256'].get(path)==sha for path,sha in prepared['source_sha256'].items()),
              'Every reviewed source is retained')
    import minicpm5_original32_layer1_observation_cpu_worker_v1 as worker
    worker.validate_plan(plan,lambda path,sha:p.read_ref(dict(path=path,sha256=sha)))
    bound,owners,legacy=legacy_sources.check_layer0(original, spec['source_sha256'])
    actual=spec['nemotron_completion']
    done=history.completed(actual['actual_exit'],actual['failed_release_exit'],guard)
    same(done['history']['canonical_sha256'],spec['canonical_sha256'],'Original canonical state throughout complete history')
    return bound,owners,dict(legacy=legacy,current=done)



def predecessor_drains(priors,guard):
    legacy=original_runner.predecessor_drains(priors['legacy'])
    current=history.recheck(priors['current'],guard)
    return dict(legacy=legacy,current=current)



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












validate_rank_result=layer1_results.validate_rank_result



validate_rank_pair=layer1_results.validate_rank_pair



def wait_file(path, guard, seconds=60):
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
    import minicpm5_original32_layer1_observation_cpu_worker_v1 as worker
    try:
        result=worker.run(context,root/('cpu-rank'+str(rank_id)))
        p.require(result['complete']is True and result['rank']==rank_id,'Finite worker complete')
        validate_rank_result(result,plan,rank_id)
        result=dict(result,process=process,admission=admission_ref,self_exit_claimed=False)
        p.once(root/('rank'+str(rank_id)+'-result.json'),result)
    except BaseException as error:
        guarded_receipt(root/('rank'+str(rank_id)+'-failure.json'),dict(process=process,error_type=type(error).__name__,error=str(error),self_exit_claimed=False),error);raise

def compact_drain(proof):
    # Retain every service and both samples, including all cgroup observations.
    # Omit unrelated systemd configuration values from this bounded receipt.
    fields=('Id','LoadState','ActiveState','SubState','MainPID','ControlPID',
            'ExecMainPID','ExecMainCode','ExecMainStatus','Result','InvocationID')
    def sample(value):
        output={k:copy.deepcopy(v)for k,v in value.items()if k!='properties'}
        output['properties']={k:value['properties'][k]for k in fields if k in value['properties']}
        return output
    current=proof['current'];rows=[]
    for entry in current['inherited']:
        rows.append(dict(unit=entry['unit'],samples=[sample(x)for x in entry['samples']]))
    for entry in current['current']:
        rows.append(dict(stage=entry['stage'],samples=[sample(x)for x in entry['samples']]))
    p.require(len(rows)==1255 and all(len(x['samples'])==2 for x in rows),'All1255 double observations retained')
    return dict(complete=True,services=1255,processes=current['processes'],
        complete_observation_sha256=digest(proof),observations=rows,legacy=proof['legacy'])

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
        if path.exists():owners.append(p.read(path))
    for owner in owners:p.gone(owner)
    group=p.empty_group(p.GROUP_PREFIX+spec['unit']+'.service')
    return dict(physically_drained=True,properties=value,group=group,owners_checked=owners)

def cleanup_owned_service(spec,admission_ref,attempt,service_owner,launch_attempted):
    if not launch_attempted:return
    root=Path(spec['output_root']);intent=p.read(root/'launch-intent.json')
    same(intent['command'],service_command(spec,admission_ref,attempt),'Exact durable owned launch')
    same(intent['admission'],admission_ref,'Cleanup binds original runtime admission')
    value=p.properties(spec['unit'])
    if value['LoadState']=='not-found':return
    validate_service_identity(spec,admission_ref,attempt,value,service_owner)
    p.command(['systemctl','--user','stop',spec['unit']],timeout=30)

def outer(spec_ref,review_ref):
    spec,plan=read_bundle(spec_ref,review_ref);root=Path(spec['output_root'])
    p.require(not root.exists(),'Fresh observation output');p.user_bus()
    fd,owner,lock=post.external.acquire_lock()
    lease=lambda:p.lock_proof(owner,lock)
    launch_attempted=False;service_owner=None;admission_ref=None;admission=None;failure=None;handlers={}
    def interrupted(signum,frame):raise InterruptedError('Layer1 collector signal '+str(signum))
    try:
        for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):handlers[sig]=signal.signal(sig,interrupted)
        lease();p.clear_window();p.active_heavy()
        bound,owners,priors=check_prerequisites(spec,lease)
        p.gone(bound['finalizer_owner']);p.predecessor_drain(bound['catalog'])
        for previous in owners:p.gone(previous)
        drains=predecessor_drains(priors,lease)
        p.control_usage(bound['spec'])
        same({n:p.sha(ROOT/n)for n in spec['canonical_sha256']},spec['canonical_sha256'],'Original canonical files')
        p.require(p.properties(spec['unit'])['LoadState']=='not-found','Fresh service name')
        fs=os.statvfs(ROOT);p.require(fs.f_bavail*fs.f_frsize>=spec['root_floor_bytes']+8*2**20,'Root metadata reserve')
        fs=os.statvfs('/mnt/nvme');p.require(fs.f_bavail*fs.f_frsize>=spec['output_floor_bytes']+CAP,'Full original output reserve')
        root.mkdir(mode=0o700);usage(root,spec['root_floor_bytes'])
        p.once(root/'predecessor-drain.json',compact_drain(drains))
        history=priors['current']['history']
        admission=dict(schema='minicpm5_layer1_observation_cpu_admission_v1',spec=spec_ref,review=review_ref,plan=spec['plan'],
            outer_owner=owner,lock_identity=lock,attempt_id=os.urandom(16).hex(),predecessors_checked=True,
            later_owners=owners,original_completed_collection=OBSERVED_EXIT,
            complete_history_services=len(history['rows']),complete_history_processes=len(history['processes']),
            complete_history_sha256=digest(history),nemotron_completion=spec['nemotron_completion'],
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
        p.require(success,'Actual complete observation job failed')
        result=p.read(root/'coordinator-result.json')
        p.require(result['complete']is True and result['rank_exits']==[0,0],'Both original rank exits0')
        same({n:p.sha(ROOT/n)for n in spec['canonical_sha256']},spec['canonical_sha256'],'Canonical files unchanged')
        p.once(root/'outer-result.json',dict(complete=True,owner=owner,service_exit=reference(root/'service-exit.json'),
            coordinator_result=reference(root/'coordinator-result.json'),usage=usage(root,spec['root_floor_bytes']),
            self_exit_claimed=False,raw_payload_reconstruction_pending=True,full_model_qualified=False,new_speedup_claim=False))
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
