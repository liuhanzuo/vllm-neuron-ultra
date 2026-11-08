"""Future exact parent/spawn wrapper; no service launch or native work on import."""
import argparse
import os
from pathlib import Path
import sys
import time

import smollm3_observer_invariance_lifecycle_v24 as life
import smollm3_observer_invariance_protocol_v20 as p


def entry(args, spawned=False):
    p.require(not p.native_loaded(), 'Wrapper runs before native import')
    spec, plan, base, handoff = life.load_spec(args.spec, args.spec_sha)
    accepted = life.reference(dict(path=args.review, sha256=args.review_sha))
    p.require(accepted['reviewer'] == '/root' and accepted['passed'] is True and accepted['independent_review'] is True,
              'Independent root execution-source acceptance')
    p.require(accepted['spec_sha256'] == args.spec_sha and accepted['runtime_source_sha256'] == spec['runtime_source_sha256'], 'Review/source/spec binding')
    row = p.run_row(args.run_id); nonce = spec['attempt_ids'][args.run_id]
    p.require(os.environ.get(p.SPAWN) == args.run_id+':'+nonce, 'Same run/nonce parent and spawn')
    p.require(not spawned or args.kind == 'server', 'Only server can spawn')
    current = life.process(os.getpid())
    hello = dict(schema='smollm3_observer_invariance_hello_v6', run=row, kind=args.kind,
                 attempt_id=nonce, spawned=spawned, process=current,
                 wrapper_source=str(life.WRAPPER), wrapper_sha256=life.digest_bytes(p.bounded_source(life.WRAPPER)),
                 spec_sha256=args.spec_sha,ray_clean_observation=p.environment_policy.ray.observe(os.environ))
    root = p.directory(args.run_id)
    stem = args.kind+'-'+str(current['pid'])
    life.record(root/('hello-'+stem+'.json'), hello)
    permit_path = root/('permit-'+stem+'.json')
    deadline = time.monotonic()+60
    while not permit_path.exists():
        p.performance_guard(); p.require(time.monotonic() < deadline, 'Actual owner ACK timeout')
        time.sleep(.1)
    bundle = life.read(permit_path)
    permit, admission, intent = bundle['permit'], bundle['admission'], bundle['intent']
    p.require(bundle['hello_sha256'] == p.digest(hello), 'Actual hello ACK binding')
    p.require(permit['pid'] == current['pid'] and permit['start_ticks'] == current['start_ticks'] and
              permit['spawned'] is spawned and permit['run'] == row, 'Actual permit process/run')
    context_box = {}
    def guard():
        return life.verify_live_permit(permit, intent, admission, spec,
            title_transaction=p.title_runtime.inflight(context_box.get('context')))
    def writer(name, raw):
        return life.write_once(root/name, raw)
    context = p.RuntimeContext(plan, base, args.run_id, args.kind, spawned,
                               guard, writer, lambda: life.monitor(args.run_id))
    context_box['context'] = context
    # Freeze a root-readable confirmation before any native entry. Frozen worker
    # subsequently writes mode receipt after verifying actual method identities.
    context.record('admitted-'+stem+'.json', dict(hello_sha256=p.digest(hello), permit_sha256=p.digest(permit),
                   actual_process=current, spawned=spawned, mode=row['mode']))
    context.record_environment();context.advance_environment('startup')
    if args.kind=='server':
        import smollm3_spawn_environment_witness_v16 as witness
        if spawned: context.install_title_witness()
        context.install_ray_witness()
        witness.install(context,life.process)
    import smollm3_observer_invariance_worker_v20 as worker
    try:
        return worker.server(context, spawned=spawned) if args.kind == 'server' else worker.client(context)
    except BaseException as error:
        # Best-effort evidence must never replace the original exception.
        try:
            life.record(root/('wrapper-failure-'+stem+'.json'), dict(passed=False, run=row,
                        process=current, original_error_type=type(error).__name__, original_error=str(error)[:4096],ray_mode_observation=p.environment_policy.ray.observe(os.environ)))
        except BaseException:
            pass
        raise


def parser():
    value = argparse.ArgumentParser()
    value.add_argument('--kind', required=True, choices=('server','client'))
    value.add_argument('--run-id', required=True)
    for name in ('spec','spec-sha','review','review-sha'): value.add_argument('--'+name, required=True)
    return value


if __name__ == '__main__':
    entry(parser().parse_args())
elif __name__ == '__mp_main__':
    # runpy's API server argv replaces sys.argv in the parent. Recover the
    # original authenticated args only from this exact run's retained intent.
    token = os.environ.get(p.SPAWN, '')
    p.require(token.count(':') == 1, 'Spawn token missing')
    run_id, nonce = token.split(':')
    p.run_row(run_id)
    intent = life.read(p.directory(run_id)/'server-launch.json')
    p.require(intent['attempt_id'] == nonce and intent['argv'][:4] == [p.PYTHON,'-B','-u',str(life.WRAPPER)], 'Spawn original wrapper intent')
    entry(parser().parse_args(intent['argv'][4:]), spawned=True)
