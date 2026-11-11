"""Native coordinator/worker entrypoints; no outer launch/admission producer.

Requires a separately reviewed actual outer admission and held descriptor. This
source increment cannot acquire/release a lock or create that admission itself.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'research'),str(ROOT)]
from research import minicpm5_layer1_operator_native_process_adapter_v6 as runtime
from research import minicpm5_layer1_operator_native_lifecycle_core_v2 as life
from minicpm5_layer1_operator_native_runtime_context_v2 import RuntimeContext
require=life.require
SELF=str(Path(__file__).resolve())
_PHASE_START=time.monotonic()


def phase_log(phase,process):
    require(phase in ('bootstrap.begin','bootstrap.end','source.begin','source.end','driver.begin','driver.end'),
        'Finite native startup phase label')
    require(type(process['pid'])is int and process['pid']==os.getpid()
        and type(process['start_ticks'])is str and process['start_ticks'].isdigit(), 'Exact acknowledged phase process')
    value=dict(phase=phase,elapsed_seconds=round(time.monotonic()-_PHASE_START,6),
        pid=process['pid'],start_ticks=process['start_ticks'])
    raw=json.dumps(value,separators=(',',':'),allow_nan=False)
    require(len(raw.encode())<=512,'Bounded phase progress line')
    print(raw,flush=True)


def wait_owner(path,metadata,lease,deadline):
    while True:
        lease.check('entry.await_owner')
        try:raw=runtime.sources.text(str(path))
        except FileNotFoundError:
            require(time.monotonic()<deadline,'Exact parent owner acknowledgment deadline');time.sleep(.05);continue
        ref=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest());return metadata.read(ref),ref


def bootstrap(admission_ref,mode):
    require(mode in ('coordinator','native')and sys.dont_write_bytecode and __debug__ and sys.executable==life.PYTHON,
        'Original native interpreter -B and assertions before runtime work')
    a=runtime.sources.load(admission_ref)
    require(a['contract']['runner']['path']==SELF and a['contract']['output_root']==str(Path(admission_ref['path']).parent),
        'Exact admitted runner/output namespace')
    runtime.validate_initial_descriptor(a)
    lease=runtime.Lease(a);metadata=runtime.Metadata(lambda:lease.check('entry.metadata'));lease.metadata=metadata
    require(metadata.read(admission_ref)==a,'Stable original admission')
    process=runtime.p.proc(os.getpid());require(process is not None and process['argv']==life.command(a['contract'],mode,admission_ref),
        'Actual child argv matches exact admitted native mode')
    require(all(os.environ.get(key)==value for key,value in runtime.environment(a).items()),'Original exact native environment')
    root=Path(a['contract']['output_root']);deadline=time.monotonic()+60
    co,co_ref=wait_owner(root/'coordinator-owner.json',metadata,lease,deadline)
    co_process=life.coordinator_process(a['contract'],admission_ref,co)
    if mode=='coordinator':require(process==co_process,'Outer acknowledged this actual coordinator')
    else:life.process(process,process['argv'],co['pid'],co['cgroup'])
    lease.processes=[co_process]if mode=='coordinator'else[co_process,process]
    runtime.service_properties(a['contract'],admission_ref,a,co)
    if mode=='native':
        owner,owner_ref=wait_owner(root/'native-owner.json',metadata,lease,deadline)
        require(owner==process,'Exact actual native identity acknowledged before release')
        runtime.receive_release(a['contract'],admission_ref,co,owner_ref,lease)
    # Outer publishes full running handoff only after real coordinator ACK.
    current=runtime.execution.declaration(a['current_execution'],a['contract'])
    wait_owner(Path(current['handoff_path']),metadata,lease,deadline)
    runtime.check_current_execution(admission_ref,a,co,lease,metadata)
    # Heavy completion/source/physical metadata work occurs after actual native gate.
    if mode=='native':phase_log('bootstrap.begin',process)
    resolved,recovered,rank_refs=runtime.resolve_admitted(admission_ref,lease,metadata)
    require(resolved['plan']['interpreters']['native']==life.PYTHON,'Original native interpreter retained')
    if mode=='native':phase_log('bootstrap.end',process)
    return a,resolved,recovered,rank_refs,process,co,lease,metadata


def coordinator(admission_ref):
    a,resolved,recovered,rank_refs,process,co,lease,metadata=bootstrap(admission_ref,'coordinator')
    ports=runtime.Ports(a['contract'],resolved,admission_ref,a,co,lease,metadata)
    return life.coordinator_step(a['contract'],resolved,admission_ref,co,lease,ports)


def native(admission_ref):
    a,resolved,recovered,rank_refs,process,co,lease,metadata=bootstrap(admission_ref,'native')
    plan=resolved['plan'];contract=a['contract'];ports=runtime.Ports(contract,resolved,admission_ref,a,co,lease,metadata)
    ports.read_owner=process
    permit=dict(schema='minicpm5_layer0_runtime_permit_v1',plan_sha256=resolved['plan_sha256'],role='native',rank=-1,arm='HF_native',
        attempt_id=a['attempt_id'],serial_exclusive=True,source_review_accepted=True,all_predecessors_drained=True,uid=1000,
        **process,python=life.PYTHON,environment=runtime.environment(a),output_root=plan['output_root'],
        runtime_admission=admission_ref,runtime_plan=a['overlay'],sdk_operator_recovery=runtime.completion.BINDING,
        outer_owner=a['outer_owner'],lock_identity=a['lock_identity'],observed_drains=a['observed_drains'],source_authentication=a['source_authentication'])
    def guard():
        lease.check('native.runtime');life.storage(contract,ports)
        entries,root,writers=ports.inventory();life.usage(contract,resolved,entries,root,writers=writers)
        return permit
    phase_log('source.begin',process)
    context=RuntimeContext(plan,resolved['plan_sha256'],permit,guard,
        contract=contract,resolved=resolved,lease=lease,ports=ports)
    context.verify_loaded_sources=lambda:runtime.verify_loaded(plan,guard)
    context.verify_sources();runtime.verify_loaded(plan,guard)
    phase_log('source.end',process)
    # The unchanged original driver imports native model code only after all gates.
    from research import minicpm5_layer1_operator_native_driver_v4 as driver
    try:
        phase_log('driver.begin',process)
        result=driver.run(context,guard,resolved['native_bundle']);runtime.verify_loaded(plan,guard);guard()
        phase_log('driver.end',process)
        wrapped=life.wrap_native_result(result,contract,resolved,process,admission_ref)
        return ports.publish('native-result.json',wrapped)
    except BaseException as error:
        lease.retain_after_failure(error)
        try:
            # Failure record bypasses the latched work guard but never admits IO
            # beyond one bounded owned receipt; the outer still owns all cleanup.
            runtime.atomic_once(Path(plan['output_root'])/'native-failure.json',dict(error_type=type(error).__name__,error=str(error)[:2048],
                process=process,admission=admission_ref,descriptor_observations=getattr(error,'observations',None),
                self_exit_claimed=False,full_model_qualified=False,new_speedup_claim=False))
        except BaseException as secondary:ports.secondary(error,secondary,'native_failure_receipt')
        raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['coordinator','native'],required=True)
    parser.add_argument('--admission',required=True);parser.add_argument('--admission-sha256',required=True);args=parser.parse_args()
    ref=dict(path=args.admission,sha256=args.admission_sha256)
    result=coordinator(ref)if args.mode=='coordinator'else native(ref)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
