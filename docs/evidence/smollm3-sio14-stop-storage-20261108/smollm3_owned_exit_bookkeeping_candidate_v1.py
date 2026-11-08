"""Inert source candidate: bounded wait for authenticated MainPID=0 stop bookkeeping.

Not installed into any runtime. The caller must still classify/authenticate the
same initial and refresh properties, including durable invocation/argv/resource
binding and authorized disappearance. Only two identical terminal observations
plus original owner absence may return from the MainPID=0 path. No success claim.
"""
import time
from smollm3_observer_invariance_lifecycle_v24 import require, process, gone, terminal

def owned_live_or_retained_exit(owner,initial,refresh,timeout=2):
    """Bounded exit transition: empty argv is never a replacement identity.

    Already authenticated PID/start/PPID/cgroup and invocation must still match.
    A nonempty different argv fails immediately. Only a matching empty-argv
    exit transition may wait for two identical retained terminal samples.
    """
    deadline=time.monotonic()+timeout;value=initial;previous_terminal=None
    while True:
        require(value['InvocationID']==owner['invocation_id'],'Exit transition invocation changed')
        if value['MainPID']!='0':
            require(value['MainPID']==str(owner['pid']),'Exit transition PID changed')
            try:actual=process(owner['pid'])
            except FileNotFoundError:actual=None
            if actual is not None:
                require(all(actual[k]==owner[k]for k in('pid','start_ticks','ppid','cgroup')),'Exit transition identity changed')
                if actual['argv']:
                    require(actual['argv']==owner['argv'],'Foreign nonempty argv')
                    return value
            previous_terminal=None
        else:
            require(value['ExecMainPID']==str(owner['pid']),'Exact retained exit PID')
            if terminal(value):
                try:gone(owner)
                except RuntimeError:previous_terminal=None
                else:
                    if previous_terminal==value:return value
                    previous_terminal=value
            else:
                # An authenticated stop may clear MainPID while stop-post/job
                # bookkeeping is still pending. This sample grants no return,
                # success or drain claim; the original deadline still applies.
                previous_terminal=None
        require(time.monotonic()<deadline,'Bounded exiting process transition expired')
        time.sleep(.02);value=refresh()
