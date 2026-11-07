"""Finite native-prefix / TP2-prefix / replay services under one existing lease."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time

import minicpm5_original32_layer0_phase_cpu_runner_v1 as old
from minicpm5_layer0_runtime_auth_v1 import RuntimeContext,digest,identity
import minicpm5_frame17_mlp_worker_v2 as worker

p,same,ROOT=old.p,old.same,old.ROOT
SELF=str(Path(__file__).resolve())
PHASES=('native','cpu','replay')
CAP=32*2**20
BASE_PLAN=old.masks.ref('minicpm5-layer0-phase-cpu-input-plan-20261107-v1.json','ce599a2beae22df0a05c13591854cfab8ba57519df11ee93a0d25ad4392306c8')
BASE_SPEC=old.masks.ref('minicpm5-layer0-phase-cpu-execution-spec-20261107-v1.json','4d7d017541ba2088a0d69fe10447d87046b9f466570f011edaf4a0388dd6ad58')
MODEL_EXIT=old.masks.ref('minicpm5-layer0-phase-cpu-root-exit-r1-20261107-v1.json','8a0acf6851550ada88bcc2b86d3d925b4c645d76e233dc54dd3fa0e78b23c967')
NUMERIC_EXIT=old.masks.ref('minicpm5-layer0-phase-selected-tap-root-exit-r1-20261107-v1.json','965ef604d89310bd15ec07515baade847e5bb7a41c31623b4f22138bc0fb8ca9')
DESIGN_REVIEW=old.masks.ref('minicpm5-frame17-layer0-mlp-root-design-review-20261107-v1.json','9646ab9caa222cfebd8277f3b16bb222d39cd2fbd4e803ab151c4f7d1b31cfd6')
FAILED_ATTEMPT=old.masks.ref('minicpm5-frame17-mlp-root-exit-r1-20261107-v1.json','39ef5348974579a4f0e397cfb7f11d5a0d2d034e9ea8a3494b20758446722ce9')


def props(phase):
    p.require(phase in PHASES,'Finite phase')
    result=dict(old.PROPS,MemoryMax=str((4 if phase=='replay' else 16)*2**30),
                MemoryHigh=str((3 if phase=='replay' else 14)*2**30),RuntimeMaxSec='120'if phase=='replay'else'600',
                LimitFSIZE=str(4*2**20))
    # MemoryMax bounds resident memory; no low RLIMIT_AS breaks torch mappings.
    if phase=='replay':result.update(AllowedCPUs='6',CPUAffinity='6')
    return result


def environment(phase,attempt,port,rank=None):
    result=p.environment({},'native' if phase=='native' else 'cpu_control',attempt)
    result.update(OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    if rank is not None:
        result.update(RANK=str(rank),LOCAL_RANK=str(rank),WORLD_SIZE='2',LOCAL_WORLD_SIZE='2',
                      MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port),TORCHELASTIC_RUN_ID='mini-f17-mlp-'+attempt)
    return result


def load_spec(ref,review_ref):
    spec=p.read_ref(ref);review=p.read_ref(review_ref)
    p.require(review.get('reviewer')=='/root' and review.get('passed')is True and review.get('blockers')==[],
              'Root reviewed exact executable source/spec')
    same(review['spec'],ref,'Root bound spec')
    same(spec['schema'],'minicpm5_frame17_mlp_spec_v2','Finite spec')
    same(spec['base_plan'],BASE_PLAN,'Inherited accepted scientific inputs')
    same(spec['base_spec'],BASE_SPEC,'Inherited predecessor lifecycle')
    same(spec['design_review'],DESIGN_REVIEW,'Exact design review')
    same(spec['failed_attempt_exit'],FAILED_ATTEMPT,'Original failed attempt retained')
    failed=p.read_ref(FAILED_ATTEMPT)
    p.require(failed['actual_exit_observed']is True and type(failed['actual_exit_code'])is int and failed['actual_exit_code']==1 and
              failed['native_model_forward_executed']is False and failed['captured_payloads']==0 and
              failed['cpu_and_replay_started']is False and failed['all_observed_owners_gone']is True,
              'Retained exact pre-forward byte-comparison failure')
    same(p.read_ref(DESIGN_REVIEW)['decision'],'approve_narrow_source_implementation;runtime_pending','Accepted design')
    p.require(spec['runner']==SELF and spec['worker']==str(ROOT/'minicpm5_frame17_mlp_worker_v2.py')
              and spec['replay']==str(ROOT/'minicpm5_frame17_mlp_replay_v2.py'),'Only finite implementation paths')
    root=Path(spec['output_root'])
    p.require(root.parent==ROOT and re.fullmatch('minicpm5-frame17-mlp-r[0-9]+-20261107-v[0-9]+',root.name),'Finite fresh output root')
    p.require(spec['output_cap_bytes']==CAP and spec['root_floor_bytes']==28*2**30,'Finite output budget/original floor')
    p.require(set(spec['units'])==set(PHASES) and len(set(spec['units'].values()))==3,'Three distinct dedicated units')
    for unit in spec['units'].values():p.require(re.fullmatch('trn1-mini-f17-mlp-[a-z0-9-]{1,80}',unit),'Finite new service')
    p.require(type(spec['master_port'])is int and 29000<spec['master_port']<30000,'Finite TP2 port')
    same(spec['resources'],{k:props(k)for k in PHASES},'Enforced per-phase resident bounds')
    plan=copy.deepcopy(p.read_ref(BASE_PLAN));plan['output_root']=str(root)
    allowed_new={spec[n]for n in ('runner','worker','replay')}|set(worker.ACTIVATIONS.values())
    same(set(spec['new_sources']),allowed_new,'Only three new modules and two actual activation sources')
    for path,sha in spec['new_sources'].items():
        p.require(Path(path).parent==ROOT or path in worker.ACTIVATIONS.values(),'Only two exact external activation paths')
        p.require(Path(path).suffix=='.py' and Path(path).stat().st_size<256*1024,'Small finite source additions only')
        same(p.sha(path),sha,'Frozen new source')
        p.require(path not in plan['source_sha256'],'Do not replace frozen source')
        plan['source_sha256'][path]=sha
    for name in ('runner','worker','replay'):p.require(spec[name]in spec['new_sources'],'Each executable pinned')
    for path,sha in plan['source_sha256'].items():same(p.sha(path),sha,'Complete inherited/new source closure')
    for path in worker.ACTIVATIONS.values():worker.activation_source(Path(path).read_text())
    for path,row in spec['added_json'].items():
        p.require(Path(path).suffix=='.json' and row['identity']['size']<2*2**20,'Small metadata additions only')
        same(identity(path),row['identity'],'Added metadata identity');same(p.sha(path),row['sha256'],'Added metadata digest')
        if path in plan['inputs']:same(plan['inputs'][path],row,'Existing metadata cannot change')
        plan['inputs'][path]=row
    plan['frame17_mlp']=spec['science']
    same(spec['science']['frame'],worker.FRAME,'One frame only')
    same(spec['science']['capture_fields'],list(worker.FIELDS),'Finite capture boundary set')
    for rank,reference in enumerate(spec['science']['cpu_anchors']):
        expected=ROOT/f'minicpm5-original32-layer0-phase-cpu-r1-20261107-v1/cpu-rank{rank}/{worker.ARM}/call-08/result.json'
        same(reference['path'],str(expected),'Actual candidate anchor path')
        same(plan['inputs'][str(expected)]['sha256'],reference['sha256'],'Candidate anchor admitted')
        rec=p.read_ref(reference);same([rec['rank'],rec['arm'],rec['call']['index']],[rank,worker.ARM,8],'Actual candidate identity')
    return spec,plan


def usage(root,floor):
    total=0;count=0
    for path in root.rglob('*'):
        s=path.lstat();p.require(not stat.S_ISLNK(s.st_mode),'No output symlinks')
        if stat.S_ISDIR(s.st_mode):continue
        p.require(stat.S_ISREG(s.st_mode)and s.st_uid==1000 and s.st_nlink==1 and s.st_size<=4*2**20,'Owned bounded output')
        total+=max(s.st_size,s.st_blocks*512);count+=1
    p.require(total<=CAP and count<=96,'Whole output cap')
    fs=os.statvfs(ROOT);p.require(fs.f_bavail*fs.f_frsize>=floor,'Original free-space floor')
    return dict(allocated_bytes=total,files=count)


def arguments(mode,admission_ref,phase):
    python='/home/ubuntu/venvs/model_reference_2026/bin/python' if phase=='native' else old.PYTHON
    return [python,'-B',SELF,'--mode',mode,'--phase',phase,'--admission',admission_ref['path'],'--admission-sha256',admission_ref['sha256']]


def service_command(spec,a_ref,attempt,phase):
    return ['systemd-run','--user','--quiet','--unit='+spec['units'][phase],'--service-type=exec',
            *[x for k,v in props(phase).items()for x in ('-p',k+'='+v)],
            '-p','StandardOutput=append:'+str(Path(spec['output_root'])/(phase+'.log')),'-p','StandardError=inherit',
            '/usr/bin/env','-i',*[k+'='+v for k,v in environment(phase,attempt,spec['master_port']).items()],
            *arguments('phase',a_ref,phase)]


def service_identity(spec,a_ref,attempt,phase,value,owner=None):
    p.require(value['LoadState']=='loaded' and p.invocation(value['InvocationID']),'Actual named service')
    cmd=service_command(spec,a_ref,attempt,phase)
    same(re.findall(r'argv\[\]=(.*?)\s*;',value['ExecStart']),[' '.join(cmd[cmd.index('/usr/bin/env'):])],'Exact service argv')
    same(value['ExecStopPost'],'','No unreviewed cleanup command')
    for key,wanted in props(phase).items():
        if key in ('RuntimeMaxSec','TimeoutStopSec'):
            p.require(p.low.duration_seconds(value[key.replace('Sec','USec')])==int(wanted),'Service bounded duration')
        else:same(value[key],wanted,'Service actual '+key)
    if owner:
        same([value['InvocationID'],value['ExecMainPID']],[owner['invocation_id'],str(owner['pid'])],'Same service invocation')
    return value


def phase_exec_ready(actual,spec,a_ref,attempt,phase):
    if actual is None:return False
    same(actual['cgroup'],'0::'+p.GROUP_PREFIX+spec['units'][phase]+'.service','Actual dedicated cgroup')
    if actual['argv']==arguments('phase',a_ref,phase):return True
    cmd=service_command(spec,a_ref,attempt,phase)
    same(actual['argv'],cmd[cmd.index('/usr/bin/env'):],'Only exact transient env exec may wait')
    return False


def child(admission_ref,phase,rank=None):
    a=p.read_ref(admission_ref);spec,plan=load_spec(a['spec'],a['review']);root=Path(spec['output_root'])
    lease=lambda:p.lock_proof(a['outer_owner'],a['lock_identity'])
    lease();same(a['plan_digest'],digest(plan),'Exact expanded runtime plan')
    owner=old.wait_file(root/(phase+'-owner.json'),lease)
    if rank is None:
        process={k:owner[k]for k in ('pid','start_ticks','ppid','argv','cgroup')}
    else:process=old.wait_file(root/('rank'+str(rank)+'-owner.json'),lease)
    same(process,p.proc(os.getpid()),'Parent acknowledged actual child')
    def guard():
        lease();p.clear_window();usage(root,spec['root_floor_bytes'])
        if rank is not None:
            same(p.proc(owner['pid']),{k:owner[k]for k in process},'Actual CPU coordinator alive')
        return True
    return a,spec,plan,root,owner,process,guard


def capture_child(admission_ref,phase,rank=None):
    a,spec,plan,root,owner,process,lease=child(admission_ref,phase,rank)
    native=phase=='native';actual_rank=-1 if native else rank
    stable=environment(phase,a['attempt_id'],spec['master_port'],rank)
    stable.pop('NEURON_RT_VISIBLE_CORES',None)
    permit=dict(schema='minicpm5_layer0_runtime_permit_v1',plan_sha256=digest(plan),role='native'if native else'cpu',
                rank=actual_rank,arm='HF_native'if native else'C_Hc_history_generation',attempt_id=a['attempt_id'],
                serial_exclusive=True,source_review_accepted=True,all_predecessors_drained=True,uid=1000,
                **process,python=plan['interpreters']['native'if native else'cpu'],environment=stable,output_root=str(root))
    def guard():lease();return permit
    context=RuntimeContext(plan,digest(plan),permit,guard)
    role='native'if native else'cpu-rank'+str(rank)
    result=worker.execute(context,root/role,native,actual_rank)
    p.require(result['complete']is True,'Actual capture complete')
    worker.once(root/(role+'-result.json'),dict(complete=True,process=process,capture=old.reference(root/role/'capture.json')))


def cpu_coordinator(admission_ref):
    a,spec,plan,root,owner,process,guard=child(admission_ref,'cpu');children=[];records=[];success=False
    try:
        for rank in (0,1):
            argv=arguments('rank',admission_ref,'cpu')
            c=subprocess.Popen(argv,env=environment('cpu',a['attempt_id'],spec['master_port'],rank),cwd=ROOT);children.append(c)
            end=time.monotonic()+10;observed=None
            while observed is None or observed['argv']!=argv:
                p.require(c.poll()is None and time.monotonic()<end,'Rank exec timeout');observed=p.proc(c.pid)
                if observed is None or observed['argv']!=argv:time.sleep(.01)
            same([observed['ppid'],observed['cgroup']],[process['pid'],process['cgroup']],'Direct CPU child')
            records.append(observed);worker.once(root/('rank'+str(rank)+'-owner.json'),observed)
        while any(c.poll()is None for c in children):
            guard();p.require(all(c.poll()in (None,0)for c in children),'One CPU rank failed');time.sleep(.2)
        p.require(all(c.wait()==0 for c in children),'Both actual ranks exited0')
        refs=[]
        for rank in (0,1):
            result=p.read(root/('cpu-rank'+str(rank)+'-result.json'));same(result['process'],records[rank],'Result actual rank identity')
            p.require(result['complete']is True,'Rank capture completed');refs.append(result['capture'])
        reports=[p.read_ref(ref)for ref in refs]
        for field in ('input','output'):
            pair=[next(x for x in r['captures']if x['name']==field)for r in reports]
            same(pair[0]['metadata'],pair[1]['metadata'],'Complete replicated '+field)
        worker.once(root/'cpu-result.json',dict(complete=True,process=process,ranks=records,captures=refs,actual_rank_exits=[0,0]))
        success=True
    finally:old.cleanup_children(children,records,root,success,sys.exc_info()[1])


def phase_main(admission_ref,phase):
    if phase=='native':return capture_child(admission_ref,'native')
    if phase=='cpu':return cpu_coordinator(admission_ref)
    a,spec,plan,root,owner,process,guard=child(admission_ref,'replay')
    refs={}
    for preceding in ('native','cpu'):
        ended=p.read(root/(preceding+'-exit.json'))
        p.require(ended['successful']is True,'Actual successful drained capture stage')
        p.gone(ended['owner']);p.empty_group(p.GROUP_PREFIX+spec['units'][preceding]+'.service')
    for role in ('native','cpu-rank0','cpu-rank1'):
        result=p.read(root/(role+'-result.json'));p.require(result['complete']is True,'Actual capture result');refs[role]=result['capture']
    import minicpm5_frame17_mlp_replay_v2 as replay
    result=replay.run(plan,root,refs,guard)
    worker.once(root/'replay-result.json',dict(complete=True,process=process,report=old.reference(root/'replay.json')))


def outer(spec_ref,review_ref):
    spec,plan=load_spec(spec_ref,review_ref);root=Path(spec['output_root'])
    p.require(not root.exists(),'Never reuse output namespace');p.user_bus()
    # The accepted previous harness supplies the entire historical drain chain.
    previous=p.read_ref(BASE_SPEC);bound,owners,priors=old.check_prerequisites(previous)
    ended=p.read_ref(MODEL_EXIT);numeric=p.read_ref(NUMERIC_EXIT)
    failed=p.read_ref(FAILED_ATTEMPT)
    for item in (ended,numeric):p.require(item['actual_exit_observed']is True and item['actual_exit_code']==0,'Actual predecessor exits')
    fd,owner,lock=old.post.external.acquire_lock();active=None;service_owner=None;a_ref=None;attempt=None;launched=False;drained=False
    handlers={}
    def interrupted(signum,frame):raise InterruptedError('Finite runner signal '+str(signum))
    try:
        for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):handlers[sig]=signal.signal(sig,interrupted)
        p.clear_window();p.active_heavy();p.gone(bound['finalizer_owner']);p.predecessor_drain(bound['catalog'])
        for prior in owners:p.gone(prior)
        old.predecessor_drains(priors)
        for prior in [ended['owner'],ended['coordinator'],*ended['ranks'],numeric['owner']]:p.gone(old.fullshape.previous.process(prior))
        p.empty_group(ended['coordinator']['cgroup'].split('::',1)[1])
        for prior in (failed['outer_owner'],failed['native_owner']):p.gone(old.fullshape.previous.process(prior))
        p.empty_group(failed['native_owner']['cgroup'].split('::',1)[1])
        for phase in ('cpu','replay'):
            same(p.properties('trn1-mini-f17-mlp-'+phase+'-r1-20261107-v1')['LoadState'],'not-found','Failed attempt never entered later phase')
        for unit in spec['units'].values():same(p.properties(unit)['LoadState'],'not-found','Fresh named phase')
        same({n:p.sha(ROOT/n)for n in spec['canonical_sha256']},spec['canonical_sha256'],'Canonical unchanged')
        fs=os.statvfs(ROOT);p.require(fs.f_bavail*fs.f_frsize>=spec['root_floor_bytes']+CAP,'Reserve all output bytes')
        root.mkdir(mode=0o700);attempt=os.urandom(16).hex()
        a=dict(spec=spec_ref,review=review_ref,outer_owner=owner,lock_identity=lock,attempt_id=attempt,plan_digest=digest(plan),
               predecessor_model_exit=MODEL_EXIT,predecessor_numeric_exit=NUMERIC_EXIT,predecessor_failed_attempt_exit=FAILED_ATTEMPT)
        a_ref=worker.once(root/'admission.json',a)
        for phase in PHASES:
            active=phase;service_owner=None;launched=False;drained=False
            p.lock_proof(owner,lock);p.active_heavy();launched=True
            p.command(service_command(spec,a_ref,attempt,phase),timeout=30)
            end=time.monotonic()+30
            while service_owner is None:
                value=service_identity(spec,a_ref,attempt,phase,p.properties(spec['units'][phase]))
                p.require(not p.terminal(value) and time.monotonic()<end,'Phase final exec identity timeout/terminal')
                actual=p.proc(int(value['MainPID']))if value['MainPID']!='0'else None
                if phase_exec_ready(actual,spec,a_ref,attempt,phase):
                    service_owner=dict(actual,unit=spec['units'][phase],invocation_id=value['InvocationID'])
                    worker.once(root/(phase+'-owner.json'),service_owner)
                else:time.sleep(.05)
            deadline=time.monotonic()+int(props(phase)['RuntimeMaxSec'])+15
            while True:
                p.lock_proof(owner,lock);p.clear_window();usage(root,spec['root_floor_bytes']);p.active_heavy((spec['units'][phase],))
                value=service_identity(spec,a_ref,attempt,phase,p.properties(spec['units'][phase]),service_owner)
                if p.terminal(value):break
                p.require(time.monotonic()<deadline,'Finite phase deadline');time.sleep(.5)
            p.gone(service_owner);group=p.empty_group(p.GROUP_PREFIX+spec['units'][phase]+'.service');drained=True
            ok=value['Result']=='success' and value['ExecMainCode']=='1' and value['ExecMainStatus']=='0'
            worker.once(root/(phase+'-exit.json'),dict(owner=service_owner,properties=value,group=group,successful=ok,observed_by_outer=owner))
            p.require(ok,'Actual phase failure: '+phase)
            result=p.read(root/(phase+'-result.json'));p.require(result['complete']is True,'Completed stage result')
        same({n:p.sha(ROOT/n)for n in spec['canonical_sha256']},spec['canonical_sha256'],'Canonical retained')
        worker.once(root/'result.json',dict(complete=True,owner=owner,admission=a_ref,report=old.reference(root/'replay.json'),
                    usage=usage(root,spec['root_floor_bytes']),self_exit_claimed=False,full_model_qualified=False,new_speedup_claim=False))
    except BaseException as error:
        old.guarded_receipt(root/'failure.json',dict(phase=active,error_type=type(error).__name__,error=str(error),owner=owner),error)
        raise
    finally:
        primary=sys.exc_info()[1]
        try:
            if launched and not drained:
                value=p.properties(spec['units'][active])
                if value['LoadState']!='not-found':
                    service_identity(spec,a_ref,attempt,active,value,service_owner)
                    p.command(['systemctl','--user','stop',spec['units'][active]],timeout=30)
                    group=p.empty_group(p.GROUP_PREFIX+spec['units'][active]+'.service')
                    after=p.properties(spec['units'][active]);same(after['InvocationID'],value['InvocationID'],'Same cleanup invocation')
                    p.require(p.terminal(after),'Actual cleanup terminal')
                    worker.once(root/'cleanup.json',dict(before=value,after=after,group=group))
        except BaseException as secondary:
            if primary is None:raise
            old.secondary_error(primary,secondary,'phase_stop')
        finally:
            worker.phase.base.cleanup_actions([lambda:old.post.external.release_lock(fd)]+
                [lambda s=sig,h=handler:signal.signal(s,h)for sig,handler in handlers.items()], 'outer final cleanup')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',required=True,choices=('outer','phase','rank'))
    parser.add_argument('--phase',choices=PHASES)
    for key in ('spec','spec-sha256','review','review-sha256','admission','admission-sha256'):parser.add_argument('--'+key)
    args=parser.parse_args()
    if args.mode=='outer':outer(dict(path=args.spec,sha256=args.spec_sha256),dict(path=args.review,sha256=args.review_sha256))
    else:
        def terminate(signum,frame):raise InterruptedError('Phase signal '+str(signum))
        for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,terminate)
        reference=dict(path=args.admission,sha256=args.admission_sha256)
        if args.mode=='rank':
            p.require(args.phase=='cpu','Rank belongs to CPU phase');capture_child(reference,'cpu',int(os.environ['RANK']))
        else:phase_main(reference,args.phase)
