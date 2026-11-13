"""Complete actual-exit/catalog/idle publication after fresh full History drain.

The accepted v2 final post-drain checks are retained. All runtime evidence is
also held to its first observed file identity. A catalog points to a not-yet
published idle until the final atomic write; interrupted publication fails closed.
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
import minicpm5_retained_replay_contract_v2 as c
import minicpm5_retained_replay_caller_v2 as caller
import minicpm5_retained_replay_catalog_v1 as catalog
from minicpm5_retained_replay_observer_v2 import tool,reconcile,drain_phases

SELF=str(c.Q/'minicpm5_retained_replay_finalizer_v3.py')
render_binding=c.render_binding


def metadata_cap(ref,spec=None):
    """Ordinary JSON stays at 8 MiB; exact reviewed source envelopes only."""
    if spec is None or ref['path']not in spec['exact_envelopes']:return c.LIMITS['ordinary_metadata_bytes']
    row=spec['exact_envelopes'][ref['path']]
    c.require(set(row)=={'sha256','bytes','role'}and row['role']=='complete_json_metadata_envelope'
        and Path(ref['path']).parent==Q and Path(ref['path']).suffix=='.json'
        and row['sha256']==ref['sha256']==spec['source_sha256'].get(ref['path'])
        and type(row['bytes'])is int and c.LIMITS['ordinary_metadata_bytes']<row['bytes']<=c.LIMITS['spec_bytes'],
        'Only exact reviewed complete source metadata envelope')
    return row['bytes']


def verify_self(spec):
    io,_=caller.api()
    c.require(str(Path(__file__).resolve())==SELF and SELF in spec['source_sha256'],
        'Separate v3 final publisher must be in the reviewed complete source closure')
    c.require(hashlib.sha256(io.raw(SELF,1<<20)).hexdigest()==spec['source_sha256'][SELF],
        'Executing v3 final publisher source changed')


def publish_completed(request_ref):
    io,_=caller.api();observed={};spec=None
    def read_saved(ref,cap=None):
        c.metadata_reference(ref);path=ref['path'];before=io.identity(path)
        if cap is None:cap=metadata_cap(ref,spec)
        c.require(before['size']<=cap,'Bounded finalization metadata; no arbitrary 64 MiB JSON')
        c.require('bytes'not in ref or before['size']==ref['bytes'],'Exact optional metadata reference byte length')
        if spec is not None and path in spec['exact_envelopes']:
            c.require(before['size']==spec['exact_envelopes'][path]['bytes'],'Exact source envelope size')
        c.require(path not in observed or observed[path]==before,'No finalization input identity rebound')
        value=io.read({k:ref[k]for k in ('path','sha256')},cap)
        c.require(io.identity(path)==before,'Finalization input changed during authenticated read')
        observed[path]=copy.deepcopy(before);return value
    def metadata_unchanged():
        for path,ident in observed.items():
            c.require(io.identity(path)==ident,'Finalization input metadata changed after its first read')
    request=read_saved(request_ref,c.LIMITS['ordinary_metadata_bytes'])
    c.require(set(request)=={'observation','observer_live','actual_observer_tool','updated_at'},'Exact final root request')
    observation=read_saved(request['observation']);live=read_saved(request['observer_live']);tool(request['actual_observer_tool'])
    spec=caller.refs(observation['spec'],observation['review']);verify_self(spec)
    c.require(read_saved(observation['spec'],c.LIMITS['spec_bytes'])==spec,'Same reviewed finalization spec')
    # These exact spec/review bindings already use the frozen caller's 64 MiB
    # complete-source-map limit; all runtime evidence/catalogs remain ordinary.
    read_saved(observation['review'],c.LIMITS['spec_bytes'])
    admission=read_saved(observation['admission']);terminal=read_saved(observation['outer_terminal'])
    previous=read_saved(observation['running'])
    c.require(reconcile(terminal,previous,observation['actual_outer_exit']['actual_tool'])==observation['actual_outer_exit'],
        'Original actual outer terminal and every phase exit remain bound')
    c.require(request['observation']['path']==str(Path(spec['observer_root'])/'observation.json')
        and request['observer_live']==observation['independent_live'],'Exact accepted observer evidence')
    expected=dict(path=observation['running']['path'],identity=observation['running_identity'])
    c.require(observed[expected['path']]==expected['identity'],'Exact independently observed running identity')
    io.process_api().gone(observation['observer_owner'])
    root=Path(spec['output_root'])/'final-publication'
    outputs=[*spec['publication_paths'].values(),spec['terminal_handoff_path']]
    c.require(not os.path.lexists(root)and all(not os.path.lexists(p)for p in outputs),'Fresh entire final publication namespace')
    from minicpm5_retained_replay_history_v1 import History,Authority
    from qwen35_package_preflight_serial_v2 import save_history
    from research.minicpm5_layer1_drain_hold_v1 import retain_until_drained
    lock=io.RetainedLock();lock.acquire();started=time.monotonic();last=[-1.0]
    def guard(force=False):
        lock.check();c.require(time.monotonic()-started<7200,'Finite final physical publication');metadata_unchanged()
        c.require(io.identity(expected['path'])==expected['identity'],'Same final running metadata identity')
        if force or time.monotonic()-last[0]>=.5:
            c.require(caller.latest_observation()==expected,'Unknown newer handoff before final publication')
            io.process_api().gone(observation['observer_owner'])
            caller.source_unchanged(spec);verify_self(spec);caller.storage(spec);last[0]=time.monotonic()
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
        composed=catalog.compose(spec,history.resolved,previous,request['observation'],observation,request['observer_live'],live,
            request['actual_observer_tool'],request['updated_at'],finalization,admission,terminal,read_saved,lambda:guard(True))
        # Recheck after the final blocking drain, exactly as the accepted v2 did.
        history.unchanged();guard(True);drained();history.unchanged();guard(True)
        c.require(all(not os.path.lexists(p)for p in outputs),'All final outputs still absent before the first write')
        published=[]
        for ref,value in composed['objects']:
            history.unchanged();guard(True)
            for written in published:read_saved(written,c.LIMITS['ordinary_metadata_bytes'])
            history.unchanged();guard(True)
            c.require(not os.path.lexists(ref['path']),'Exclusive append-only final publication')
            # The original writer serializes exactly publication_bytes(), fsyncs
            # and exclusively links. Only the final object enters handoff discovery.
            actual=io.once(Path(ref['path']),value)
            c.require(actual==ref,'Published complete bytes match the precomposed source/catalog chain')
            published.append(actual)
        return {k:composed[k]for k in ('outer_actual_exit','observer_actual_exit','catalog','idle_handoff',
            'catalog_rows','full_sources','retained_replay_passed','deletion_authorized','new_speedup_claim')}
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
