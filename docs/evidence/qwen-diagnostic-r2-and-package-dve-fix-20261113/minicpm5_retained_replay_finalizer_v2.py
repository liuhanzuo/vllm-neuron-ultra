"""Source-only v2 final publisher: recheck after the last blocking drain.

Pair/replay and independent observation remain the frozen v1 interfaces. Only
future final publication uses this explicit, separately source-pinned successor.
Import and pure render_binding do not discover current state or perform IO.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

Q=Path(__file__).resolve().parent
for directory in (Q,Q.parent):
    if str(directory)not in sys.path:sys.path.insert(0,str(directory))
import minicpm5_retained_replay_contract_v1 as c
import minicpm5_retained_replay_caller_v1 as caller
from minicpm5_retained_replay_observer_v1 import tool,reconcile,drain_phases,render_completed

SELF=str(c.Q/'minicpm5_retained_replay_finalizer_v2.py')


def render_binding(context,source_bytes,original_plan_bytes,finalizer_bytes):
    """Pure extension: retain all v1 sources and add this exact finalizer pin."""
    c.require(type(finalizer_bytes)is bytes and 0<len(finalizer_bytes)<1<<20,'Complete finite v2 finalizer source bytes')
    context=copy.deepcopy(context);pins=context['inherited_source_sha256'];sha=c.sha(finalizer_bytes)
    c.require(SELF not in pins or pins[SELF]==sha,'No inherited finalizer source replacement')
    pins[SELF]=sha
    result=c.render_binding(context,source_bytes,original_plan_bytes)
    result['mandatory_source_sha256'][SELF]=sha
    result['final_publisher']=dict(path=SELF,sha256=sha)
    return result


def verify_self(spec):
    io,_=caller.api()
    c.require(str(Path(__file__).resolve())==SELF and SELF in spec['source_sha256'],
        'Separate v2 final publisher must be present in reviewed complete source closure')
    c.require(hashlib.sha256(io.raw(SELF,1<<20)).hexdigest()==spec['source_sha256'][SELF],
        'Executing v2 final publisher source changed')


def publish_completed(request_ref):
    """Explicit external root entry after observing the actual observer exit.

    This function never starts a worker. It verifies the same latest running
    identity and all complete inherited/own owners under the original lock,
    then publishes the pure rendering before releasing that lock.
    """
    io,_=caller.api();request=io.read(request_ref)
    c.require(set(request)=={'observation','observer_live','actual_observer_tool','updated_at'},'Exact final root request')
    observation=io.read(request['observation']);live=io.read(request['observer_live']);tool(request['actual_observer_tool'])
    spec=caller.refs(observation['spec'],observation['review']);verify_self(spec);admission=io.read(observation['admission'])
    terminal=io.read(observation['outer_terminal']);previous=io.read(observation['running'])
    c.require(reconcile(terminal,previous,observation['actual_outer_exit']['actual_tool'])==observation['actual_outer_exit'],
        'Original actual outer terminal and every phase exit remain bound')
    c.require(request['observation']['path']==str(Path(spec['observer_root'])/'observation.json')
        and request['observer_live']==observation['independent_live'],'Exact accepted observer evidence')
    expected=dict(path=observation['running']['path'],identity=observation['running_identity'])
    io.process_api().gone(observation['observer_owner'])
    root=Path(spec['output_root'])/'final-publication';c.require(not root.exists(),'Fresh final publication namespace')
    from minicpm5_retained_replay_history_v1 import History,Authority
    from qwen35_package_preflight_serial_v2 import save_history
    from research.minicpm5_layer1_drain_hold_v1 import retain_until_drained
    lock=io.RetainedLock();lock.acquire();started=time.monotonic();last=[-1.0]
    def guard(force=False):
        lock.check();c.require(time.monotonic()-started<7200,'Finite final physical publication')
        c.require(io.identity(expected['path'])==expected['identity'],'Same final running metadata identity')
        if force or time.monotonic()-last[0]>=.5:
            c.require(caller.latest_observation()==expected,'Unknown newer handoff before final publication')
            io.process_api().gone(observation['observer_owner']);caller.source_unchanged(spec);verify_self(spec);caller.storage(spec);last[0]=time.monotonic()
    def drained():
        lock.check();io.process_api().gone(observation['observer_owner'])
        return drain_phases(spec,observation['admission'],admission,terminal)
    try:
        guard(True);drained()
        history=History(spec,observation['spec'],observation['review'],spec['units'],guard).resolve()
        guard(True);pair=history.physical_pair(Authority(history,guard,lambda:[]));guard(True);history.unchanged()
        root.mkdir(mode=0o700);history_refs=save_history(root,pair)
        finalization=dict(schema='minicpm5_retained_replay_fresh_final_publication_v1',running=observation['running'],
            running_identity=observation['running_identity'],observer_owner=observation['observer_owner'],observer_owner_gone=True,
            actual_observer_tool=request['actual_observer_tool'],final_owned=drained(),complete_history=history_refs,lock_identity=c.LOCK)
        value=render_completed(previous,request['observation'],observation,request['observer_live'],live,
            request['actual_observer_tool'],request['updated_at'],finalization)
        history.unchanged();guard(True);drained()
        history.unchanged();guard(True)
        return io.once(Path(spec['terminal_handoff_path']),value)
    finally:
        proof=retain_until_drained(lambda:lock.check(),drained,
            lambda value:print(json.dumps(dict(event='mini_retained_final_publication_drain_blocked',details=value)),flush=True),
            lambda:time.sleep(1))
        lock.release(proof['physically_drained'])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',required=True);parser.add_argument('--request-sha256',required=True)
    args=parser.parse_args(argv);c.require(sys.dont_write_bytecode and __debug__,'Explicit -B/assertions')
    os.sched_setaffinity(0,{7});os.setpriority(os.PRIO_PROCESS,0,19)
    print(json.dumps(publish_completed(dict(path=args.request,sha256=args.request_sha256))),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
