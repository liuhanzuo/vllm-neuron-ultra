"""Separate bounded source preparation from the authenticated exec transition.

This source candidate waits for a separate atomic directory marker after the
existing child's immutable exec-intent file has been fully written and synced,
written only after bundle, parent identity, service identity and health checks.
No model precision, numeric gate, or existing exec grace is changed.
"""
import math
import time
import os
import stat
from nemotron9b_package_exec_transition_v25 import await_exec

PREPARATION_SECONDS = 300.
EXEC_SECONDS = 60.
POST_EXEC_ACK_SECONDS = 60.

def await_prepared_exec(*, child_owner, target_argv, read_process, poll, guard,
                        intent_ready, validate_intent, observe_target,
                        save_preparation, save_diagnostic,
                        clock=time.monotonic, sleep=time.sleep,
                        preparation_seconds=PREPARATION_SECONDS,
                        exec_seconds=EXEC_SECONDS):
    for value,limit in ((preparation_seconds,PREPARATION_SECONDS),(exec_seconds,EXEC_SECONDS)):
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0 < value <= limit:
            raise ValueError('Finite bounded startup budgets')
    if not child_owner['argv'] or not target_argv or child_owner['argv']==target_argv:
        raise ValueError('Distinct nonempty preparation and target argv')
    stable={k:v for k,v in child_owner.items() if k!='argv'}
    started=clock();deadline=started+preparation_seconds;last=None;initial_empty_since=None
    try:
        while True:
            guard()
            if clock()>=deadline:raise RuntimeError('Server preparation deadline')
            if poll() is not None:raise RuntimeError('Server exited during preparation')
            last=read_process()
            if last is None:raise RuntimeError('Server disappeared during preparation')
            if {k:v for k,v in last.items() if k!='argv'}!=stable:
                raise RuntimeError('Server kernel identity changed during preparation')
            if last['argv'] not in (child_owner['argv'],target_argv,[]):
                raise RuntimeError('Unexpected server argv during preparation')
            if last["argv"]==[]:initial_empty_since=clock()
            if intent_ready():
                validate_intent()
                if clock()>=deadline:raise RuntimeError('Server preparation deadline')
                save_preparation(dict(status='authenticated_exec_intent',owner=child_owner,
                    last=last,elapsed_seconds=clock()-started,budget_seconds=preparation_seconds,
                    full_model_qualified=False))
                break
            if last['argv']!=child_owner['argv']:
                raise RuntimeError('Exec transition without authenticated preparation intent')
            sleep(.005)
    except BaseException as error:
        save_diagnostic(dict(status='preparation_failed',error=repr(error),last=last,
            elapsed_seconds=clock()-started,budget_seconds=preparation_seconds,qualified=False))
        raise
    # The old transition retains empty-argv grace and a fresh double observation.
    return await_exec(child_owner=child_owner,target_argv=target_argv,
        read_process=read_process,poll=poll,guard=guard,validate_intent=validate_intent,
        observe_target=observe_target,save_diagnostic=save_diagnostic,
        deadline=clock()+exec_seconds,clock=clock,sleep=sleep,initial_empty_since=initial_empty_since)

def await_post_exec_ack(*, ready, poll, guard, observe_target, expected,
                        clock=time.monotonic, sleep=time.sleep):
    """Give the independently observed post-exec ACK its own bounded window."""
    deadline=clock()+POST_EXEC_ACK_SECONDS
    while True:
        guard()
        if clock()>=deadline:raise RuntimeError('Independent post-exec ACK timeout')
        if poll() is not None:raise RuntimeError('Server exited before independent post-exec ACK')
        if observe_target()!=expected:raise RuntimeError('Post-exec server identity changed')
        if ready():return
        sleep(.05)


def publish_prepared(path):
    """Publish only after the complete exec-intent write and fsync return."""
    os.mkdir(path, 0o700)

def prepared_ready(path):
    try:
        info=os.lstat(path)
    except FileNotFoundError:
        return False
    if not stat.S_ISDIR(info.st_mode) or info.st_uid!=os.getuid() or stat.S_IMODE(info.st_mode)!=0o700:
        raise RuntimeError('Invalid preparation-ready directory')
    return True
