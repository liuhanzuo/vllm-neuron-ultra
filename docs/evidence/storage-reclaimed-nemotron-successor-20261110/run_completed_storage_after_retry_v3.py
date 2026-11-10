"""Run exact reviewed maintenance only after actual Dory native+review exits.

Launch directly, never through the observer wrapper. All source verification,
full predecessor drain, hashes, archives and retirement share the original lock.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import urllib.request

import completed_storage_contract_after_retry_v3 as c
import completed_storage_files_v1 as f
from nemotron9b_package_controller_v20 import acquire, release
from review_dory_layer0_serial_snapshot_v1 import physical_drain


def free(path):
    info = os.statvfs(path)
    return info.f_bavail*info.f_frsize


def privileged(request):
    result = subprocess.run(['sudo','-n','/usr/bin/python3','-B',str(c.PRIVILEGED)],
        input=json.dumps(request),capture_output=True,text=True,check=True,timeout=90)
    f.require(len(result.stdout) <= 2**20, 'Bounded privileged response')
    return json.loads(result.stdout)


def available(weights, guard):
    result = []
    for row in weights:
        guard()
        request = urllib.request.Request(row['restore_url'],method='HEAD')
        with urllib.request.urlopen(request,timeout=30) as response:
            f.require(response.status == 200 and int(response.headers['Content-Length']) == row['stat']['st_size'],
                      'Exact original shard remains publicly downloadable')
            result.append(dict(path=row['path'],status=200,bytes=row['stat']['st_size'],body_bytes_read=0))
    return result


def prepare_all(plan, guard, *, availability=available):
    """No unlink here; every target must verify before the separate commit loop."""
    weights, journals = [], []
    heads = availability(plan['weights'],guard)
    for row in plan['weights']:
        checked = f.file_hash(row['path'],row['stat'],guard)
        f.require(checked['sha256'] == row['sha256_from_official_metadata'], 'Complete official LFS weight hash')
        weights.append(dict(path=row['path'],stat=row['stat'],**checked))
        print(json.dumps(dict(verified_weight=Path(row['path']).name,bytes=checked['bytes'])),flush=True)
    used = 0
    for row in plan['journals']:
        target = Path(plan['output'])/(Path(row['path']).name+'.gz')
        archive = f.archive_journal(row['path'],target,row['stat'],guard,plan['budget']['gzip_cap']-used)
        used += archive['gzip_bytes']
        f.require(used <= plan['budget']['gzip_cap'], 'Total retained gzip cap')
        journals.append(archive)
        print(json.dumps(dict(verified_journal=Path(row['path']).name,gzip_bytes=archive['gzip_bytes'])),flush=True)
    return dict(weights=weights,journals=journals,availability=heads,
                all_five_weights_authenticated=True,all_eight_journals_decompressed=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ('plan','plan-sha256','review','review-sha256'): parser.add_argument('--'+name,required=True)
    args = parser.parse_args()
    f.require(sys.dont_write_bytecode and __debug__, 'Original assertions/no bytecode')
    plan_ref = dict(path=args.plan,sha256=args.plan_sha256)
    review_ref = dict(path=args.review,sha256=args.review_sha256)
    plan = c.load(plan_ref,plan=True)
    c.validation(plan)
    review = c.load(review_ref)
    f.require(review['reviewer'] == '/root' and review['passed'] is True and review['blockers'] == []
              and review['plan'] == plan_ref and review['source_sha256'] == plan['source_sha256'],
              'Exact final plan and complete separate source review')
    out = Path(plan['output'])
    f.require(not out.exists(), 'No automatic maintenance retry or overwrite')
    c.p.low.user_bus()
    fd,owner,lock = acquire()
    started,handlers,created = time.monotonic(),{},False
    budget = plan['budget']
    def interrupted(signum,frame): raise InterruptedError('Serial storage maintenance interrupted')
    last_full = [0.0]
    def guard(force=False):
        c.p.low.lock_proof(owner,lock)
        now = time.monotonic()
        f.require(now-started < budget['wall_seconds'] and free('/') >= budget['root_floor']
                  and free('/mnt/nvme') >= budget['nvme_floor'], 'Maintenance wall time and storage floors')
        if force or now-last_full[0] >= 2:
            c.p.low.clear_window();c.p.low.active_heavy()
            if created:
                files = list(out.iterdir())
                f.require(all(path.is_file() and not path.is_symlink() for path in files), 'Ordinary maintenance outputs')
                f.require(sum(path.stat().st_size for path in files if path.suffix == '.gz') <= budget['gzip_cap'],
                          'Whole gzip output cap')
                f.require(sum(path.stat().st_size for path in files if path.suffix != '.gz') <= budget['metadata_cap'],
                          'Whole metadata output cap')
            last_full[0] = now
    try:
        for sig in (signal.SIGINT,signal.SIGTERM): handlers[sig] = signal.signal(sig,interrupted)
        os.umask(0o077);os.sched_setaffinity(0,set(budget['cpus']));os.nice(10)
        resource.setrlimit(resource.RLIMIT_AS,(budget['address_space_bytes'],)*2)
        resource.setrlimit(resource.RLIMIT_CPU,(budget['cpu_seconds'],budget['cpu_seconds']+5))
        resource.setrlimit(resource.RLIMIT_FSIZE,(budget['gzip_cap'],)*2)
        guard(True)
        before = dict(root=free('/'),nvme=free('/mnt/nvme'))
        f.require(before['root'] >= budget['root_floor']+budget['gzip_cap']+budget['metadata_cap'],
                  'All root archive/metadata space reserved before admission')
        out.mkdir(mode=0o700);created=True
        f.once(out/'intent.json',dict(owner=owner,lock=lock,plan=plan_ref,source_review=review_ref,
                                    actual_self_exit_claimed=False))
        done = c.residual.completed(plan['completed_residual_review'])
        h = done['history']
        f.require((len(h['rows']),len(h['processes']),len(h['known_admitted_directories'])) == (1149,349,38),
                  'Actual complete predecessor chain')
        parent,effective = c.combined_sources(plan,done)
        c.source_checks(plan,effective,guard=guard)
        history_ref = f.once(out/'predecessor-history.json',h)
        drain = physical_drain(h,[])
        f.require(drain['complete'] and (drain['services'],drain['processes']) == (1149,349), 'Full physical drain')
        drain_ref = f.once(out/'predecessor-drain.json',drain)
        c.p.accelerator_idle()
        canonical = c.p.canonical()
        f.require(canonical == h['canonical_sha256'], 'Original canonical inputs unchanged')
        trees = c.protected_trees(plan['weights'])
        f.require(trees == plan['protected_trees'], 'All protected model/evidence trees unchanged since preparation')
        payloads = {path:c.p.file_stat(path) for path in parent['payloads']}
        binaries = {path:c.p.file_stat(path) for path in parent['runtime_binary_stat']}
        f.require(all(payloads[path] == row['stat'] for path,row in parent['payloads'].items())
                  and binaries == parent['runtime_binary_stat'], 'Original model payload and runtime identities')
        targets = plan['weights']+plan['journals']
        for row in targets: f.require(f.ordinary(row['path']) == row['stat'], 'Exact target identity before admission')
        unused = privileged(dict(operation='check-unused',rows=targets))
        f.once(out/'protected-before.json',dict(trees=trees,payloads=payloads,binaries=binaries,unused=unused))
        ad_ref = f.once(out/'admission.json',dict(owner=owner,lock=lock,plan=plan_ref,source_review=review_ref,
            predecessor_history=history_ref,predecessor_drain=drain_ref,fully_drained=True,
            canonical_sha256=canonical,before_space=before,maintenance_only=True,performance_measurement=False))
        print(json.dumps(dict(owner=owner,admission=ad_ref)),flush=True)
        verified = prepare_all(plan,guard)
        verified['plan'] = plan_ref
        verified_ref = f.once(out/'verified-before-retirement.json',verified)
        privileged(dict(operation='check-unused',rows=targets))
        guard(True)
        c.source_checks(plan,effective,guard=guard)
        for row in targets: f.require(f.ordinary(row['path']) == row['stat'], 'All targets stable before any deletion')
        # Archives and five complete LFS hashes are durable before the first unlink.
        for row in targets:
            guard(True)
            privileged(dict(operation='check-unused',rows=targets))
            f.append(out/'actions.jsonl',dict(phase='before',path=row['path'],verification=verified_ref))
            if row in plan['journals']:
                response = privileged(dict(operation='unlink-journal',path=row['path'],plan=plan_ref,
                                           admission=ad_ref,verification=verified_ref))
            else:
                f.unlink_exact(row['path'],row['stat'])
                response = dict(retired=row['path'],restore_url=row['restore_url'],
                                sha256=row['sha256_from_official_metadata'])
            f.append(out/'actions.jsonl',dict(phase='after',**response))
        f.require(all(not Path(row['path']).exists() for row in targets), 'All exact planned retirements complete')
        f.require(c.protected_trees(plan['weights']) == trees, 'All protected models, INT8, artifacts and evidence retained')
        f.require({path:c.p.file_stat(path) for path in payloads} == payloads
                  and {path:c.p.file_stat(path) for path in binaries} == binaries, 'Original active inputs retained')
        c.source_checks(plan,effective,guard=guard)
        f.require(c.p.canonical() == canonical, 'Canonical inputs unchanged')
        guard(True)
        after = dict(root=free('/'),nvme=free('/mnt/nvme'))
        result = dict(complete=True,owner=owner,plan=plan_ref,verification=verified_ref,
            before_space=before,after_space=after,retired_weight_files=5,archived_journal_files=8,
            weight_bytes_verified=sum(row['bytes'] for row in verified['weights']),
            journal_bytes_retained=sum(row['original_bytes'] for row in verified['journals']),
            gzip_bytes=sum(row['gzip_bytes'] for row in verified['journals']),
            protected_trees_unchanged=True,original_payloads_and_runtime_unchanged=True,canonical_unchanged=True,
            original_qwen_bf16_restore_required=True,elapsed_seconds=time.monotonic()-started,
            actual_self_exit_claimed=False,runtime_admitted=False,full_model_qualified=False,new_speedup_claim=False)
        f.once(out/'result.json',result)
        print(json.dumps(dict(complete=True,after_space=after)),flush=True)
    except BaseException as error:
        if created:
            f.once(out/'failure.json',dict(owner=owner,error=repr(error),partial_outputs_retained=True,
                                          automatic_retry=False,actual_self_exit_claimed=False))
        raise
    finally:
        release(fd)
        for sig,handler in handlers.items(): signal.signal(sig,handler)


if __name__ == '__main__': main()
