"""Stdlib serving-process observations and terminal proof; no native imports.

A successful HTTP result alone is insufficient. The worker and serial controller
observe the executed process independently. Shutdown signals use a PID file
handle, so even an unexpected exit cannot redirect a signal to a reused PID.
"""
import os
from pathlib import Path
import signal
import subprocess
import time
import nemotron9b_package_protocol_v25 as p


def observe(pid):
    """Read bounded kernel identity twice, including exec, session and state."""
    first=p.low.proc(pid)
    p.require(first is not None,'Serving process disappeared')
    def extra():
        path=Path('/proc')/str(pid)
        with (path/'stat').open('rb')as stream:
            raw=stream.read(65537)
        p.require(len(raw)<=65536,'Bounded server stat')
        fields=raw.decode().rsplit(')',1)[1].split()
        return dict(pgid=int(fields[2]),sid=int(fields[3]),state=fields[0],
            exe=p.file_stat(path/'exe'))
    a=extra();second=p.low.proc(pid);b=extra()
    p.require(first==second and {k:v for k,v in a.items()if k!='state'}=={k:v for k,v in b.items()if k!='state'},
        'Serving process identity changed during observation')
    p.require(a['state']not in ('Z','X','x')and b['state']not in ('Z','X','x'),'Serving process already exited')
    return dict(process=first,**{k:v for k,v in b.items()if k!='state'},live=True)


def expected(spec,owner,child,argv,observed):
    process=observed['process']
    p.require(process==dict(child,argv=argv)and process['ppid']==owner['pid'],'Actual server PID/start/parent/argv/group')
    p.require(observed['live']is True and observed['pgid']==observed['sid']==process['pid'],'Actual private process group/session')
    p.require(observed['exe']==p.file_stat(p.PYTHON),'Actual serving interpreter identity')
    return observed


def observe_expected(spec,stage,owner,child,argv):
    p.require(child['cgroup']=='0::'+p.group(spec,stage)and child['ppid']==owner['pid'],'Server child owned group/parent')
    return expected(spec,owner,child,argv,observe(child['pid']))


def controller_observe(spec,stage,owner):
    """Called outside timed windows; ACK actual post-exec identity independently."""
    c=Path(spec['control_root']);path=c/(stage+'-server-post-exec.json')
    ack_path=c/(stage+'-server-observed.json')
    if not path.exists()or ack_path.exists():return
    saved=p.read(path);child=p.read(c/(stage+'-server-owner.json'))
    actual=observe_expected(spec,stage,owner,child,p.server_argv(spec))
    p.require(saved['observation']==actual and saved['parent']==owner,'Independent actual post-exec process')
    p.require(saved['exec_intent']==p.ref(c/(stage+'-server-exec.json'))and saved['ack']==p.ref(c/(stage+'-server-ack.json')),
        'Post-exec binds child ACK and exec intent')
    p.once(ack_path,dict(parent=owner,observation=actual,post_exec=p.ref(path),
        observer='serial-controller',controller=p.read(c/'admission.json')['controller']))


def validate_chain(spec,stage,owner,result,admission,require_gone=True):
    c=Path(spec['control_root'])
    names=('intent','owner','ack','exec','preparation','post-exec','observed','ready','request-complete','stop-intent','exit')
    refs={name:p.ref(c/(stage+'-server-'+name+'.json'))for name in names}
    p.require(result['server_lifecycle']==refs,'Worker result binds every server lifecycle receipt')
    rows={name:p.read(ref['path'],ref['sha256'])for name,ref in refs.items()}
    intent,child,ack,execution=(rows[x]for x in ('intent','owner','ack','exec'))
    command=p.argv(spec,stage,admission['execution_spec'],admission['source_review'])+['--server-child']
    p.require(intent==dict(argv=command,parent=owner)and child['argv']==command,'Original spawned child argv/parent')
    p.require(child['ppid']==owner['pid']and child['cgroup']=='0::'+p.group(spec,stage),'Owned child group/parent')
    p.require(ack==dict(parent=owner,child=child,execution_spec=admission['execution_spec'],source_review=admission['source_review']),
        'Exact owner/spec/review child ACK')
    p.require(execution==dict(original_owner=child,argv=p.server_argv(spec),ack=refs['ack']),'Bound pre-exec intent')
    from nemotron9b_package_startup_v25 import prepared_ready
    preparation=rows['preparation']
    p.require(prepared_ready(c/(stage+'-server-prepared')),'Complete intent readiness marker retained')
    p.require(preparation['status']=='authenticated_exec_intent' and preparation['owner']==child
              and preparation['budget_seconds']==300. and type(preparation['elapsed_seconds'])is float
              and 0<=preparation['elapsed_seconds']<300.
              and preparation['full_model_qualified'] is False,'Bound finite preparation observation')
    p.require({k:v for k,v in preparation['last'].items()if k!='argv'}=={k:v for k,v in child.items()if k!='argv'}
              and preparation['last']['argv'] in (child['argv'],p.server_argv(spec),[]),
              'Preparation retained same child kernel identity')
    post=rows['post-exec'];observation=expected(spec,owner,child,p.server_argv(spec),post['observation'])
    p.require(post['parent']==owner and post['exec_intent']==refs['exec']and post['ack']==refs['ack'],'Actual post-exec receipt binding')
    independent=rows['observed']
    p.require(independent==dict(parent=owner,observation=observation,post_exec=refs['post-exec'],
        observer='serial-controller',controller=admission['controller']),'Independent controller post-exec observation')
    ready=rows['ready'];request=rows['request-complete'];stop=rows['stop-intent'];exit=rows['exit']
    p.require(ready==dict(observation=observation,post_exec=refs['post-exec'],controller_observation=refs['observed'],health_status=200),
        'Observed ready process, not HTTP health alone')
    p.require(request==dict(observation=observation,ready=refs['ready'],workload_complete=True),'Observed process after final response')
    p.require(stop==dict(observation=observation,request_complete=refs['request-complete'],
        pre_stop_returncode=None,signal=int(signal.SIGTERM),reason='successful_requests',transport='pidfd'),
        'Authenticated deliberate stop intent after successful requests')
    p.require(exit['pid']==child['pid']and exit['start_ticks']==child['start_ticks']and exit['pre_stop_returncode']is None,
        'Original child stayed alive until deliberate shutdown')
    p.require(exit['stop_intent']==refs['stop-intent']and exit['signal_sent']is True and exit['signal_error']is None,
        'Successful signal bound to durable stop intent')
    p.require(exit['escalated']is False and exit['workload_complete']is True and exit['cleanup_only']is False,
        'Cleanup/timeout cannot qualify')
    p.require(type(exit['returncode'])is int and exit['returncode']in (0,-int(signal.SIGTERM)),
        'Only explicit graceful exit or requested SIGTERM qualifies')
    p.require(exit['qualification_passed']is True and exit['original_error']is None and exit['validation_error']is None
        and exit['controller_cgroup_drain_still_required']is True,'Server lifecycle actually qualified')
    if require_gone:p.low.gone(child)
    return refs


def shutdown(spec,stage,owner,child,child_owner,pidfd,observation,workload_complete,original_error,start):
    """Retain original status. Error cleanup is never a qualification path."""
    c=Path(spec['control_root']);pre=child.poll();stop_ref=None;sent=False;signal_error=None;escalated=False
    clean=False;validation_error=None
    try:
        if pre is None:
            current=observe(child.pid)
            # Before readiness the process may still be the ACK-gated wrapper.
            p.require(current['process']in (child_owner,dict(child_owner,argv=p.server_argv(spec))),
                'Refuse signal to changed server identity')
            expected(spec,owner,child_owner,current['process']['argv'],current)
            clean=workload_complete and original_error is None and observation==current
            if clean:
                intent=dict(observation=current,request_complete=p.ref(c/(stage+'-server-request-complete.json')),
                    pre_stop_returncode=None,signal=int(signal.SIGTERM),reason='successful_requests',transport='pidfd')
                p.once(c/(stage+'-server-stop-intent.json'),intent);stop_ref=p.ref(c/(stage+'-server-stop-intent.json'))
            else:
                p.once(c/(stage+'-server-cleanup-intent.json'),dict(observation=current,original_error=original_error,
                    workload_complete=workload_complete,signal=int(signal.SIGTERM),transport='pidfd',qualifying=False))
            # Re-poll after durable intent. A process which exited independently
            # between response and intent cannot be described as explicitly stopped.
            pre=child.poll()
            if pre is None:
                signal.pidfd_send_signal(pidfd,signal.SIGTERM,None,0);sent=True
                try:child.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    escalated=True
                    signal.pidfd_send_signal(pidfd,signal.SIGKILL,None,0)
                    child.wait(timeout=30)
            else:clean=False
    except BaseException as error:
        validation_error=repr(error);signal_error=repr(error);clean=False
    rc=child.poll()
    qualified=clean and pre is None and sent and signal_error is None and not escalated and type(rc)is int and rc in (0,-int(signal.SIGTERM))
    receipt=dict(pid=child.pid,start_ticks=child_owner['start_ticks'],pre_stop_returncode=pre,returncode=rc,
        elapsed_seconds=time.monotonic()-start,stop_intent=stop_ref,signal_sent=sent,signal_error=signal_error,
        escalated=escalated,workload_complete=workload_complete,cleanup_only=not clean,
        qualification_passed=qualified,original_error=original_error,validation_error=validation_error,
        controller_cgroup_drain_still_required=True)
    p.once(c/(stage+'-server-exit.json'),receipt)
    p.require(qualified,'Server lifecycle rejected: '+repr(receipt))
    return receipt
