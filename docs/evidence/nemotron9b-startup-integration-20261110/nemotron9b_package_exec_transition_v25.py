"""Bounded pre-exec observation; exact post-exec identity remains mandatory.

A transient empty /proc cmdline never qualifies. Only the same live child after
an authenticated exec intent can be resampled, and only for a short deadline.
"""
import math
import time

def await_exec(*,child_owner,target_argv,read_process,poll,guard,validate_intent,observe_target,save_diagnostic,deadline,clock=time.monotonic,sleep=time.sleep,empty_grace_seconds=.25,initial_empty_since=None):
    if not 0 < empty_grace_seconds <= 1:raise ValueError('Bounded empty-argv grace')
    if initial_empty_since is not None and (isinstance(initial_empty_since,bool) or not isinstance(initial_empty_since,(int,float)) or not math.isfinite(initial_empty_since) or initial_empty_since>clock()):raise ValueError("Valid initial empty observation time")
    stable={k:v for k,v in child_owner.items() if k!='argv'}
    empty_since=initial_empty_since;samples=[];sample_count=0;last=None
    try:
        while True:
            guard()
            if clock()>=deadline:raise RuntimeError('Post-exec observation deadline')
            if poll() is not None:raise RuntimeError('Server exited before post-exec observation')
            last=read_process()
            if last is None:raise RuntimeError('Server disappeared during exec')
            if {k:v for k,v in last.items() if k!='argv'}!=stable:raise RuntimeError('Server kernel identity changed during exec')
            if empty_since is not None and clock()-empty_since>=empty_grace_seconds:raise RuntimeError('Empty server argv exceeded exec grace')
            if last['argv']==target_argv:
                validate_intent()
                result=observe_target()
                if empty_since is not None:save_diagnostic(dict(status='transient_empty_argv_resolved',samples=samples,sample_count=sample_count,last=last,qualified=False))
                return result
            if last['argv']==child_owner['argv']:
                if empty_since is not None:raise RuntimeError('Server reverted to pre-exec argv')
            elif last['argv']==[]:
                validate_intent()
                now=clock()
                if empty_since is None:empty_since=now
                sample_count+=1
                if len(samples)<8:samples.append(dict(process=last,elapsed_since_empty=now-empty_since))
                if now-empty_since>=empty_grace_seconds:raise RuntimeError('Empty server argv exceeded exec grace')
            else:raise RuntimeError('Unexpected server argv during exec')
            sleep(.005)
    except BaseException as error:
        save_diagnostic(dict(status='failed',error=repr(error),samples=samples,sample_count=sample_count,last=last,qualified=False))
        raise
