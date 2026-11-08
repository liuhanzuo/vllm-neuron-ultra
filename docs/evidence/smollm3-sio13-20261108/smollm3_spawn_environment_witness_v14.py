"""Exact POSIX spawn boundary witness. Importing performs no process/file IO.

The pure Dispatcher is mockable. install() is an explicit admitted-worker call;
it wraps transport identically in off/on arms, without modifying environment,
arguments, file descriptors, preparation bytes, return values or exceptions.
"""
import hashlib
import marshal
import os
from pathlib import Path
import sys
import threading
import time
import types
import smollm3_observer_invariance_environment_v8 as env
import smollm3_enginecore_title_runtime_v11 as title_runtime
import smollm3_enginecore_title_transition_v1 as title

require=env.require
ROOT=Path(__file__).resolve().parent
TRANSPORT_FILES=('/usr/lib/python3.10/multiprocessing/popen_spawn_posix.py',
                 '/usr/lib/python3.10/multiprocessing/util.py')


def model_transport(args):
    return isinstance(args,(list,tuple)) and any(x in ('--multiprocessing-fork',b'--multiprocessing-fork')for x in args)


def error_type(error):return type(error).__name__


def observe_exec_child(pid,parent,argv,observe,deadline,now=time.monotonic,sleep=time.sleep):
    """Bounded fork→exec visibility race; identity may not change during wait."""
    first=None
    while True:
        child=observe(pid)
        require(child['ppid']==parent['pid'] and child['cgroup']==parent['cgroup'],'Spawn direct child same group')
        if first is None:first=child['start_ticks']
        require(child['start_ticks']==first,'Spawn child PID not reused during exec')
        if child['argv']==argv:return child
        require(now()<deadline,'Child exec identity deadline');sleep(.01)


def validate_dispatch(before,after,parent_permit,child):
    """Pure receipt binding; root verifies bytes, live processes and ancestors."""
    env.validate_lineage(parent_permit['environment_lineage'],parent_permit,parent_permit['spawned'])
    require(set(before)=={'schema','sequence','parent','parent_permit_sha256','run','attempt_id','stage','lineage',
                         'ld_source_state','ray_source_state','ray_observation','transport','source_sha256','exact_values_checked','values_recorded','parent_title'},'Exact before receipt fields')
    checked_title=title_runtime.validate_spawn_parent(parent_permit,before['parent_title'])
    require(checked_title['spawn_identity']==env.identity(parent_permit),'Spawn title projection retains original parent')
    require(before['schema']=='smollm3_spawn_before_v3' and before['parent']==env.identity(parent_permit),'Actual before parent')
    require(before['parent_permit_sha256']==env.digest(parent_permit) and before['run']==parent_permit['run'] and
            before['attempt_id']==parent_permit['attempt_id'],'Before permit run binding')
    require(before['stage']=='serve' and before['lineage']==parent_permit['environment_lineage'] and
            before['ld_source_state']==env.ld_state(before['lineage'],'serve'),'Before exact source state')
    env.ray.validate_state(before['ray_source_state'])
    require(before['ray_source_state']['kind']!='absent' and before['ray_observation']==env.ray.ZERO,'Actual parent exact zero with source receipt')
    require(before['exact_values_checked']is True and before['values_recorded']is False,'Before exact redacted check')
    require(type(before['sequence'])is int and 1<=before['sequence']<=7,'Bounded dispatch sequence')
    require(before['source_sha256']=={x:parent_permit['root_runtime_source_sha256'][x]for x in TRANSPORT_FILES},'Exact dispatch transport sources')
    transport=before['transport'];require(set(transport)=={'executable','argv','passfds','callsite'},'Exact transport fields')
    require(transport['executable']==parent_permit['python'] and transport['callsite']=='popen_spawn_posix.Popen._launch','Pinned spawn transport')
    require(transport['argv']==child['argv'] and child['argv'][0]==parent_permit['python'] and '-c'in child['argv']
            and model_transport(child['argv']) and child['argv'][child['argv'].index('-c')+1].startswith('from multiprocessing.spawn import spawn_main; spawn_main('),'Exact dispatched child argv')
    require(type(transport['passfds'])is list and all(type(x)is int and x>=0 for x in transport['passfds']),'Dispatch passfds')
    require(set(after)=={'schema','sequence','before_sha256','parent','child','run','attempt_id','parent_permit_sha256','parent_title_sha256'},'Exact after receipt fields')
    require(after==dict(schema='smollm3_spawn_after_v3',sequence=before['sequence'],before_sha256=env.digest(before),parent=before['parent'],
                       child=env.identity(child),run=before['run'],attempt_id=before['attempt_id'],parent_permit_sha256=before['parent_permit_sha256'],
                       parent_title_sha256=env.digest(before['parent_title'])),'Exact child dispatch result')
    require(child['ppid']==before['parent']['pid'] and child['cgroup']==before['parent']['cgroup'],'Exact immediate dispatched parent')
    return dict(parent=before['parent'],child=env.identity(child),before_sha256=env.digest(before),after_sha256=env.digest(after),
                parent_permit_sha256=before['parent_permit_sha256'],sequence=before['sequence'])


class Dispatcher:
    def __init__(self,original,before,after,failed):
        self.original=original;self.before=before;self.after=after;self.failed=failed
        self.lock=threading.RLock();self.sequence=0

    def call(self,callsite_ok,path,args,passfds):
        if not callsite_ok:
            require(not model_transport(args),'Model spawn outside pinned Popen callsite')
            return self.original(path,args,passfds)
        require(model_transport(args),'Pinned Popen callsite has nonmodel transport')
        # Serializes sequence/dispatch association only. Root ACK does not need
        # this lock, and the preparation pipe is still written by original code.
        with self.lock:
            self.sequence+=1;sequence=self.sequence
            require(sequence<=7,'Original eight-producer run cap')
            before=self.before(sequence,path,args,passfds)
            try:
                result=self.original(path,args,passfds)
            except BaseException as original_error:
                try:self.failed(sequence,before,original_error,None)
                except BaseException as secondary:
                    original_error.smollm3_witness_errors=(secondary,)
                raise
            try:self.after(sequence,before,result)
            except BaseException as evidence_error:
                try:self.failed(sequence,before,evidence_error,result)
                except BaseException as secondary:
                    evidence_error.smollm3_witness_errors=(secondary,)
                raise
            return result


def _code(code,path):
    if not path:return code
    matching=[x for x in code.co_consts if isinstance(x,types.CodeType)and x.co_name==path[0]]
    require(len(matching)==1,'Unique pinned transport function code')
    return _code(matching[0],path[1:])


def authenticate_transport(context,popen,util):
    originals=(popen.Popen._launch,util.spawnv_passfds)
    for filename,function,names in zip(TRANSPORT_FILES,originals,[('Popen','_launch'),('spawnv_passfds',)]):
        path=Path(filename);wanted=context.plan['source_sha256'][filename]
        raw=context.source_bytes(path,wanted)
        expected=_code(compile(raw,filename,'exec',dont_inherit=True,optimize=0),names)
        require(function.__code__.co_filename==filename and marshal.dumps(function.__code__)==marshal.dumps(expected),'Original exact CPython transport code')
        require(function.__defaults__ is None and function.__closure__ is None,'Original transport callable defaults')
    require(originals[0].__globals__ is popen.__dict__ and originals[1].__globals__ is util.__dict__,'Original transport globals')
    require(popen.util is util and popen.Popen.method=='spawn','Original POSIX spawn implementation')
    return originals


def install(context,observe_process):
    """Caller has an authenticated server context; no installation on clients."""
    import multiprocessing.popen_spawn_posix as popen
    import multiprocessing.util as util
    require(context.kind=='server' and not hasattr(context,'spawn_witness'),'One server spawn witness')
    launch,original=authenticate_transport(context,popen,util)
    def before(sequence,path,args,passfds):
        context.complete_pending_ray_before_spawn()
        context.poll();require(context.environment_stage=='serve','Model spawn after complete runner/serve')
        require(context.actual_environment().get('VLLM_USE_V1')=='1','Parent actually entered V1 before spawn')
        # Reauthenticate exact source and original bound low-level callable.
        require(popen.Popen._launch is launch and util.spawnv_passfds is wrapped,'Transport wrapper unchanged')
        for filename in TRANSPORT_FILES:context.source_bytes(Path(filename),context.plan['source_sha256'][filename])
        parent_title=context.title_spawn_parent()
        current=parent_title['initial']['identity']
        require(current==env.identity(context.permit) and observe_process(context.permit['pid'])==parent_title['live']['identity'],
                'Spawn actual parent and immutable source identity')
        executable=os.fsdecode(path);argv=[os.fsdecode(x)for x in args]
        require(executable==context.python_executable and len(argv)>=4 and argv[0]==executable and '-c' in argv,'Exact Python transport path')
        command=argv[argv.index('-c')+1]
        require(command.startswith('from multiprocessing.spawn import spawn_main; spawn_main('),'Exact CPython spawn entry')
        require(all(type(fd)is int and fd>=0 for fd in passfds),'Original integer passfds')
        row=dict(schema='smollm3_spawn_before_v3',sequence=sequence,parent=current,parent_title=parent_title,
                 parent_permit_sha256=env.digest(context.permit),run=context.row,attempt_id=context.permit['attempt_id'],
                 stage='serve',lineage=context.lineage,ld_source_state=env.ld_state(context.lineage,'serve'),
                  ray_source_state=dict(context.ray_state),ray_observation=env.ray.observe(context.actual_environment()),
                 transport=dict(executable=executable,argv=argv,passfds=list(passfds),callsite='popen_spawn_posix.Popen._launch'),
                 source_sha256={x:context.plan['source_sha256'][x] for x in TRANSPORT_FILES},
                 exact_values_checked=True,values_recorded=False)
        context.record('spawn-before-'+str(current['pid'])+'-'+str(sequence)+'.json',row)
        # The controller can acknowledge only an already durable after receipt.
        # No callback lock is held over the original fork/exec operation.
        context.poll()
        require(env.ray.observe(context.actual_environment())==row['ray_observation']==env.ray.ZERO and context.ray_state==row['ray_source_state'],'Parent actual Ray state immediately before original transport')
        return row
    def after(sequence,before,result):
        require(type(result)is int and result>0,'Original spawn returned real child PID')
        deadline=min(time.monotonic()+5,context.permit['shared_run_deadline_monotonic'],context.permit['matrix_deadline_monotonic'])
        child=observe_exec_child(result,before['parent'],before['transport']['argv'],observe_process,deadline)
        context.poll();require(context.title_spawn_parent()==before['parent_title'],'No title transition/reversion during child dispatch')
        row=dict(schema='smollm3_spawn_after_v3',sequence=sequence,before_sha256=env.digest(before),parent=before['parent'],child=child,
                 run=context.row,attempt_id=context.permit['attempt_id'],parent_permit_sha256=before['parent_permit_sha256'],
                 parent_title_sha256=env.digest(before['parent_title']))
        context.record('spawn-after-'+str(before['parent']['pid'])+'-'+str(sequence)+'.json',row)
    def failed(sequence,before,error,child_pid):
        # Only type is retained: exception messages may contain environment data.
        context.record('spawn-failure-'+str(context.permit['pid'])+'-'+str(sequence)+'.json',
                       dict(schema='smollm3_spawn_failure_v1',sequence=sequence,before_sha256=env.digest(before),
                            parent=before['parent'],child_pid=child_pid,error_type=error_type(error),passed=False))
    dispatcher=Dispatcher(original,before,after,failed)
    def wrapped(path,args,passfds):
        frame=sys._getframe(1)
        try:callsite=frame.f_code is launch.__code__ and frame.f_globals is launch.__globals__
        finally:del frame
        return dispatcher.call(callsite,path,args,passfds)
    util.spawnv_passfds=wrapped;context.spawn_witness=dispatcher
    context.record('spawn-hook-'+str(context.permit['pid'])+'.json',dict(schema='smollm3_spawn_hook_v3',
                   pid=context.permit['pid'],start_ticks=context.permit['start_ticks'],mode=context.row['mode'],
                   same_in_both_modes=True,original_transport_code_verified=True,title_identity_policy=title.POLICY,
                   source_sha256={x:context.plan['source_sha256'][x]for x in TRANSPORT_FILES}))
    return dispatcher
