"""Future exclusive saved-data analysis; launch directly, never observer-wrapped.

No real execution is possible without actual completed Nemotron v22/v23,
a concrete final specification/review and a fresh full physical drain.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time

Q=Path(__file__).resolve().parent
sys.path[:0]=[str(Q),str(Q.parent)]
import minicpm5_saved_tap_runtime_contract_v1 as c
from nemotron9b_package_controller_v18 import acquire,release
from run_dory_layer0_serial_v2 import free,memory_free
from review_dory_layer0_serial_snapshot_v1 import physical_drain
from dory_generation_execution_provenance_v1 import review_source_pins
from dory_service_query_auth_v1 import AuthenticatedServiceQuery,read_observation
from dory_saved_review_guard_v1 import CompleteReviewGuard
import minicpm5_layer0_execution_protocol_v4 as low


def execute(spec,spec_ref,review_ref):
    low.user_bus()
    fd,owner,lock=acquire()
    start,initial=time.monotonic(),free('/mnt/nvme')
    budget=spec['budget'];writer=query=None;handlers={}
    guard=CompleteReviewGuard(low=low,owner=owner,lock=lock,budget=budget,started=start,
        initial_nvme=initial,free=free,memory_free=memory_free,clock=time.monotonic)
    def interrupted(signum,frame):raise InterruptedError('Saved-tap analysis interrupted')
    try:
        print(json.dumps(dict(owner=owner,lock=lock,serial_lock_acquired=True,runtime_admitted=False)),flush=True)
        for sig in (signal.SIGTERM,signal.SIGINT):handlers[sig]=signal.signal(sig,interrupted)
        guard()
        c.require(not c.OUT.exists() and free('/')>=budget['root_floor']+budget['output_bytes']
            and initial>=budget['nvme_floor']+budget['nvme_new_bytes']
            and memory_free()>=budget['initial_memory_available_bytes'],'Fresh complete startup reservations')
        os.sched_setaffinity(0,set(budget['cpus']))
        os.setpriority(os.PRIO_PROCESS,0,budget['nice'])
        resource.setrlimit(resource.RLIMIT_AS,(budget['address_space_bytes'],)*2)
        resource.setrlimit(resource.RLIMIT_CPU,(budget['cpu_seconds'],budget['cpu_seconds']+5))
        resource.setrlimit(resource.RLIMIT_FSIZE,(budget['single_file_bytes'],)*2)
        sources=review_source_pins(spec['source_sha256'],spec['essential_sources'],
                                  max_total_bytes=spec['source_read_cap_bytes'],guard=guard)
        completed=c.history.completed(spec['profile'],spec['nemotron_actual_exit'],
                                      spec['nemotron_failed_release_exit'],guard)
        prior=completed['history']
        c.require(all(spec['source_sha256'].get(p)==digest for p,digest in completed['source_sha256'].items()),
                  'Every actual historical dependency is in the final reviewed closure')
        reader=c.history.Reader(guard)
        diagnosis,manifest=c.input_metadata(spec,reader,os.getuid())
        c.require(all(spec['source_sha256'].get(p)==digest for p,digest in reader.pins.items()),
                  'All original analysis metadata is reviewed')
        drain=physical_drain(prior,[])
        c.require(drain['complete'] is True and drain['services']==1202
                  and drain['processes']==len(prior['processes']),'Fresh full history physical drain')
        from nemotron9b_package_protocol_v23 import canonical
        original=canonical()
        c.require(original==prior['canonical_sha256'],'Original canonical files unchanged')
        loaded=c.loaded_local_sources(spec['source_sha256'])
        guard()
        writer=c.OutputWriter(c.OUT,file_cap=budget['single_file_bytes'],total_cap=budget['output_bytes'])
        href=writer.save('predecessor-history.json',prior)
        dref=writer.save('predecessor-drain.json',drain)
        sref=writer.save('source-files.json',sources)
        admission=dict(schema='minicpm5_saved_tap_admission_v1',owner=owner,lock=lock,
            execution_spec=spec_ref,source_review=review_ref,budget=budget,
            predecessor_history=href,predecessor_drain=dref,source_files=sref,
            canonical_sha256=original,input_manifest=spec['input_manifest'],loaded_local_sources=loaded,
            runtime_admitted=True,performance_measurement=False,actual_self_exit_claimed=False,
            full_model_qualified=False,new_speedup_claim=False)
        aref=writer.save('admission.json',admission)
        print(json.dumps(dict(owner=owner,admission=aref,mode='saved-tap-analysis')),flush=True)
        observation=read_observation(spec['query_dependency_observation'])
        c.require(all(spec['source_sha256'].get(p)==row['sha256']
            for p,row in observation['source_and_binary_pins'].items()),'Every query dependency source is bound')
        query=AuthenticatedServiceQuery(observation,before_check=guard)
        guard.attach_authenticated_query(query)
        guard()
        report=c.validate_report(c.inputs.analyze(diagnosis,manifest,guard,uid=os.getuid()))
        report_ref=writer.save('numeric-report.json',report)
        guard()
        c.require(canonical()==original,'Original canonical files unchanged after analysis')
        final_loaded=c.loaded_local_sources(spec['source_sha256'])
        guard.close()
        c.require(query.closed and not query.poisoned and query.restored,'Final query audit and cleanup')
        qref=writer.save('service-query-authentication.json',dict(complete=True,
            dependency_observation=spec['query_dependency_observation'],authenticated_files_before_import=query.startup_file_count,
            authenticated_bytes_before_import=query.startup_bytes,guard_accounting=guard.accounting(),
            final_dependency_audit=query.final_audit,original_import_policy_restored=query.restored))
        result=dict(schema='minicpm5_saved_tap_execution_result_v1',complete=True,owner=owner,
            execution_spec=spec_ref,source_review=review_ref,admission=aref,numeric_report=report_ref,
            service_query_authentication=qref,loaded_local_sources=final_loaded,
            elapsed_seconds=time.monotonic()-start,all84bundles_and384comparisons_complete=True,
            canonical_unchanged=True,actual_self_exit_claimed=False,model_reexecuted=False,
            full_model_qualified=False,new_speedup_claim=False)
        result_ref=writer.save('result.json',result)
        print(json.dumps(dict(complete=True,result=result_ref)),flush=True)
    except BaseException as error:
        if writer is not None:
            writer.save('failure.json',dict(error=repr(error),owner=owner,complete=False,
                actual_self_exit_claimed=False,full_model_qualified=False,new_speedup_claim=False))
        raise
    finally:
        for sig in handlers:signal.signal(sig,lambda signum,frame:None)
        try:
            try:guard.close()
            finally:
                if query is not None:query.close()
        except BaseException as error:
            if writer is not None:
                writer.save('query-cleanup-failure.json',dict(error=repr(error),full_model_qualified=False))
            raise
        finally:
            try:
                if writer is not None:writer.close()
            finally:
                release(fd)
                for sig,handler in handlers.items():signal.signal(sig,handler)


def main():
    parser=argparse.ArgumentParser()
    for name in ('spec','spec-sha256','review','review-sha256'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    c.require(sys.dont_write_bytecode and sys.flags.optimize==0,'Python -B with original assertions')
    spec_ref=dict(path=args.spec,sha256=args.spec_sha256)
    review_ref=dict(path=args.review,sha256=args.review_sha256)
    reader=c.history.Reader(lambda:None)
    spec=c.validate_spec(reader(spec_ref));review=reader(review_ref)
    c.validate_review(spec,spec_ref,review)
    execute(spec,spec_ref,review_ref)


if __name__=='__main__':main()
