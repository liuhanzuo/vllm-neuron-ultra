"""Separately reviewed retry entrypoint; never launch through observer wrapper.

Preparation only until a real failed prior-review exit, complete source closure,
concrete runtime spec and exact source review have been independently recorded.
No retry is launched automatically. Original raw-review routines are unchanged.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time

Q = Path(__file__).resolve().parent
sys.path[:0] = [str(Q), str(Q.parent)]
import dory_saved_review_retry_contract_v1 as contract
import dory_after_nemotron_r9_v2 as history
from nemotron9b_package_controller_v18 import acquire, release
from run_dory_layer0_serial_v2 import save, free, memory_free
from review_dory_layer0_serial_snapshot_v1 import physical_drain
from dory_generation_execution_provenance_v1 import review_source_pins
from dory_residual_saved_complete_v2 import freeze, review_complete
from dory_service_query_auth_v1 import AuthenticatedServiceQuery, read_observation
from dory_saved_review_guard_v1 import CompleteReviewGuard

p, require = history.p, contract.require


def main():
    parser = argparse.ArgumentParser()
    for name in ('spec', 'spec-sha256', 'review', 'review-sha256'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    require(sys.dont_write_bytecode and sys.flags.optimize == 0,
            'Original assertions enabled and bytecode writes disabled')
    spec_ref = dict(path=args.spec, sha256=args.spec_sha256)
    review_ref = dict(path=args.review, sha256=args.review_sha256)
    reader = contract.original.Reader()
    spec = reader(spec_ref)
    parent = contract.validate_spec(spec, reader)
    review = reader(review_ref)
    require(review['reviewer'] == '/root' and review['passed'] is True and review['blockers'] == []
            and review['spec_sha256'] == args.spec_sha256
            and review['source_sha256'] == spec['source_sha256'],
            'Final concrete specification and complete source review required')
    out = Path(spec['output'])
    require(not out.exists(), 'Fresh retry namespace; retain every previous output')
    p.low.user_bus()
    fd, owner, lock = acquire()
    started, initial = time.monotonic(), free('/mnt/nvme')
    budget, created, handlers, query = spec['budget'], False, {}, None
    guard = CompleteReviewGuard(low=p.low, owner=owner, lock=lock, budget=budget,
        started=started, initial_nvme=initial, free=free, memory_free=memory_free,
        clock=time.monotonic)

    def interrupted(signum, frame):
        raise InterruptedError('Saved-review retry interrupted')

    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            handlers[sig] = signal.signal(sig, interrupted)
        guard()
        require(not out.exists() and initial >= budget['nvme_floor'] + budget['nvme_new_bytes']
                and memory_free() >= budget['initial_memory_available_bytes'],
                'Full original startup reservations under the serial lock')
        os.sched_setaffinity(0, set(budget['cpus']))
        os.nice(10)
        resource.setrlimit(resource.RLIMIT_AS, (budget['address_space_bytes'],)*2)
        resource.setrlimit(resource.RLIMIT_CPU, (budget['cpu_seconds'], budget['cpu_seconds']+5))
        resource.setrlimit(resource.RLIMIT_FSIZE, (budget['single_file_bytes'],)*2)
        sources = review_source_pins(spec['source_sha256'], spec['essential_sources'],
                                    max_total_bytes=spec['source_read_cap_bytes'], guard=guard)
        previous = contract.failed_history(spec['failed_review_exit'], reader)
        external_inputs = {spec_ref['path']: spec_ref['sha256'], review_ref['path']: review_ref['sha256']}
        require(all(external_inputs.get(path) == digest or spec['source_sha256'].get(path) == digest
                    for path, digest in reader.pins.items()),
                'No unreviewed historical metadata read')
        prior = previous['history']
        require(all(spec['source_sha256'].get(path) == digest
                    for path, digest in prior['predecessor_source_sha256'].items()),
                'Complete inherited source closure retained')
        drain = physical_drain(prior, [])
        require(drain['complete'] is True
                and (drain['services'], drain['processes']) == (1149, 348),
                'All historical services and owners, including the failed review, physically drained')
        canonical = p.canonical()
        require(canonical == prior['canonical_sha256'], 'Original frozen canonical files')
        guard()
        out.mkdir()
        created = True
        href, dref = save(out/'predecessor-history.json', prior), save(out/'predecessor-drain.json', drain)
        save(out/'source-files.json', sources)
        admission = dict(owner=owner, lock=lock, spec_sha256=args.spec_sha256,
            source_review=review_ref, mode='review', review=True, budget=budget,
            canonical_sha256=canonical, predecessor_history=href, predecessor_drain_record=dref,
            predecessor_drain=drain, parent_review_spec=contract.PARENT_SPEC,
            failed_review_exit=spec['failed_review_exit'], performance_measurement=False)
        save(out/'admission.json', admission)
        print(json.dumps(dict(owner=owner, mode='review', admission=str(out/'admission.json'))), flush=True)
        observation_ref = spec['query_dependency_observation']
        observation = read_observation(observation_ref['path'], observation_ref['sha256'])
        require(all(spec['source_sha256'].get(path) == row['sha256']
                    for path, row in observation['source_and_binary_pins'].items()),
                'Every query dependency belongs to the reviewed full source closure')
        # All predecessor/drain imports above finish before the restrictive
        # query finder is installed. No model or tensor algorithm is replaced.
        query = AuthenticatedServiceQuery(observation, before_check=guard)
        guard.attach_authenticated_query(query)
        guard()
        frozen = freeze(spec['producer_spec'], spec['native_exit'], guard)
        save(out/'post-exit-contract.json', frozen)
        checked = review_complete(frozen, guard)
        save(out/'saved-residual-review.json', checked)
        guard()
        require(p.canonical() == canonical, 'Original canonical files unchanged')
        guard.close()
        require(query.closed and not query.poisoned and query.restored,
                'Successful final dependency audit and private connection close')
        query_record = dict(complete=True, dependency_observation=observation_ref,
            authenticated_files_before_import=query.startup_file_count,
            authenticated_bytes_before_import=query.startup_bytes,
            guard_accounting=guard.accounting(), final_dependency_audit=query.final_audit,
            original_import_policy_restored=query.restored, model_speedup_claim=False)
        save(out/'service-query-authentication.json', query_record)
        result = dict(complete=True, mode='review', review=True, owner=owner,
            elapsed_seconds=time.monotonic()-started, actual_self_exit_claimed=False,
            failed_review_exit=spec['failed_review_exit'], full_model_qualified=False, new_speedup_claim=False)
        save(out/'result.json', result)
        print(json.dumps(dict(complete=True, mode='review', result=str(out/'result.json'))), flush=True)
    except BaseException as error:
        if created:
            save(out/'failure.json', dict(error=repr(error), mode='review',
                                         failed_review_exit=spec['failed_review_exit']))
        raise
    finally:
        for sig in handlers:
            signal.signal(sig, lambda signum, frame: None)
        try:
            # The query may have been constructed but not attached if its
            # initial audit or binding failed. Its close must not be skipped.
            try:
                guard.close()
            finally:
                if query is not None:
                    query.close()
        except BaseException as error:
            if created and not (out/'query-cleanup-failure.json').exists():
                save(out/'query-cleanup-failure.json', dict(error=repr(error), full_model_qualified=False))
            raise
        finally:
            release(fd)
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


if __name__ == '__main__':
    main()
