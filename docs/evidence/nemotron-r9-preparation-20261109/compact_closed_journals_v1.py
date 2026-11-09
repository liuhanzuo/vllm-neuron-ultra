"""Serial, lossless compaction of an exact reviewed set of closed journals.

No experiment payload, environment, active journal or service is removed.
Every complete gzip is independently decompressed and matched to the source
before any original is retired. Archives are private and remain on this host.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import stat
import subprocess
import sys
import time

R = Path(__file__).resolve().parents[1]
Q = R/'research'
sys.path[:0] = [str(R), str(Q)]
import nemotron9b_package_protocol_v17 as p
from nemotron9b_package_controller_v17 import acquire, release
import nemotron9b_attention_completion_v1 as completion
from retire_completed_storage_for_nemotron_v16 import check_unused, tree_snapshot

FIELDS = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_nlink', 'st_blocks', 'st_uid', 'st_mode')
ROOT_JOURNALS = Path('/var/log/journal/ec2139d93c43b899a9a1defcd3a54e08')
OLD_ARCHIVES = Path('/mnt/nvme/completed-storage-retirement-20261109-v1/system-journal')


def require(value, message):
    if not value:
        raise RuntimeError(message)


def identity(path):
    path = Path(path)
    value = path.lstat()
    require(path.resolve() == path and stat.S_ISREG(value.st_mode) and value.st_nlink == 1,
            'Canonical single-link regular journal')
    return {key: getattr(value, key) for key in FIELDS}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load(ref):
    raw = p.metadata.stable_bytes(ref['path'], 8*2**20)
    require(sha(raw) == ref['sha256'], 'Exact admitted metadata')
    return json.loads(raw)


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    return dict(path=str(path), sha256=sha(Path(path).read_bytes()))


def free(path):
    value = os.statvfs(path)
    return value.f_bavail*value.f_frsize


def validate(plan):
    require(plan['schema'] == 'closed_journals_compaction_plan_v1', 'Exact plan schema')
    require(plan['completion_exit'] == completion.EXIT, 'Latest completed experiment bound')
    require(len(plan['files']) == 16 and len({row['path'] for row in plan['files']}) == 16,
            'Exactly sixteen closed journals')
    require(Path(plan['output']) == Path('/mnt/nvme/closed-journals-compaction-20261109-r1'),
            'Fresh fixed archive output')
    counts = {str(ROOT_JOURNALS): 0, str(OLD_ARCHIVES): 0}
    for row in plan['files']:
        path = Path(row['path'])
        require(str(path.parent) in counts and path.name.startswith('user-1000@')
                and path.suffix == '.journal' and row['stat']['st_size'] == 128*2**20
                and row['original_closed_mtime_ns'] < 1793577600*10**9,
                'Only closed user journals older than November 2')
        if path.parent == ROOT_JOURNALS:
            require(row['original_closed_mtime_ns'] == row['stat']['st_mtime_ns'],
                    'Root journal close time uses the actual source timestamp')
        require(row['stat']['st_nlink'] == 1 and stat.S_ISREG(row['stat']['st_mode']),
                'No shared or special journal target')
        counts[str(path.parent)] += 1
    require(counts == {str(ROOT_JOURNALS): 4, str(OLD_ARCHIVES): 12}, 'Exact volume counts')
    require(plan['compressed_cap_bytes'] == 400*2**20, 'Bounded retained archive reservation')
    require(plan['root_floor_bytes'] == 28*2**30 and plan['nvme_floor_bytes'] == 30*2**30,
            'Original filesystem floors')


def archive(source, target, expected, guard):
    require(identity(source) == expected, 'Unchanged source before compression')
    digest = hashlib.sha256()
    size = 0
    with Path(source).open('rb') as src, Path(target).open('xb') as dst:
        with gzip.GzipFile(filename='', mode='wb', compresslevel=6, fileobj=dst, mtime=0) as compressed:
            while data := src.read(4*2**20):
                guard()
                digest.update(data)
                size += len(data)
                compressed.write(data)
        dst.flush()
        os.fsync(dst.fileno())
    require(identity(source) == expected and size == expected['st_size'], 'Stable complete original source')
    decoded = hashlib.sha256()
    decoded_bytes = 0
    # A distinct reader consumes the complete archive, including gzip CRC/footer.
    with gzip.open(target, 'rb') as stream:
        while data := stream.read(4*2**20):
            guard()
            decoded_bytes += len(data)
            require(decoded_bytes <= size, 'No decompressed size overrun')
            decoded.update(data)
    require(decoded_bytes == size and decoded.digest() == digest.digest(), 'Every decompressed byte verified')
    return dict(path=str(source), archive_path=str(target), original_bytes=size,
                original_sha256=digest.hexdigest(), gzip_bytes=Path(target).stat().st_size,
                gzip_sha256=sha(Path(target).read_bytes()), source_stat=expected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    require(sys.dont_write_bytecode and __debug__, 'No bytecode or disabled assertions')
    plan_ref = dict(path=args.plan, sha256=args.plan_sha256)
    plan = load(plan_ref)
    validate(plan)
    for path, digest in plan['source_sha256'].items():
        require(p.metadata.sha(path, digest) == digest, 'Frozen maintenance source')
    # The fixed runtime specification uses the existing named 32 MiB envelope.
    spec = p.metadata.load(plan['protected_nemotron_spec'])
    protected = set(spec['source_sha256']) | set(spec['payloads']) | set(spec['runtime_binary_stat'])
    targets = [row['path'] for row in plan['files']]
    require(not protected.intersection(targets), 'No experiment dependency selected')
    prior = load(plan['prior_retirement_exit'])
    require(prior['actual_tool_exit_observed'] is True and prior['actual_exit']['exit_code'] == 0,
            'Actual prior archive operation complete')
    verified = load(prior['verification'])
    prior_plan = load(prior['plan'])
    old_close_times = {row['archive_path']:row['stat']['st_mtime_ns'] for row in prior_plan['journals']}
    old_hashes = {row['archive_path']: row['sha256'] for row in verified['files'] if 'archive_path' in row}
    require({row['path']:row['previous_sha256'] for row in plan['files'] if row['previous_sha256']}
            == old_hashes, 'All twelve original journal archive hashes preserved')
    require({row['path']:row['original_closed_mtime_ns'] for row in plan['files']
             if Path(row['path']).parent == OLD_ARCHIVES} == old_close_times,
            'Archived copies retain the original closed-journal timestamp provenance')
    out = Path(plan['output'])
    require(not out.exists(), 'Never overwrite earlier maintenance')
    p.low.user_bus()
    fd, owner, lock = acquire()
    started = time.monotonic()
    created = False
    try:
        os.sched_setaffinity(0, {4})
        os.nice(10)
        os.umask(0o077)
        resource.setrlimit(resource.RLIMIT_AS, (2*2**30,)*2)
        resource.setrlimit(resource.RLIMIT_CPU, (900, 905))
        resource.setrlimit(resource.RLIMIT_FSIZE, (512*2**20,)*2)
        before = dict(root=free('/'), nvme=free('/mnt/nvme'))
        canonical = p.canonical()
        require(before['root'] >= plan['root_floor_bytes'] and
                before['nvme'] >= plan['nvme_floor_bytes']+plan['compressed_cap_bytes'], 'Archive capacity')
        out.mkdir(mode=0o700)
        created = True
        history, _ = completion.completed()
        history_ref = save(out/'predecessor-history.json', history)
        save(out/'intent.json', dict(owner=owner, lock=lock, plan=plan_ref,
             predecessor_history=history_ref, before_space=before, canonical_sha256=canonical,
             actual_self_exit_claimed=False, runtime_admitted=False))
        print(json.dumps(dict(owner=owner, output=str(out))), flush=True)
        p.quiet()
        drain = p.predecessor_drain(completion.base_history.load(completion.base_history.CATALOG))
        save(out/'predecessor-drain.json', dict(services=len(drain), processes=len(history['processes']),
             complete=True, observations=drain))
        p.accelerator_idle()
        snapshots = {path:tree_snapshot(path) for path in plan['protected_trees']}
        payloads = {path:p.file_stat(path) for path in spec['payloads']}
        binaries = {path:p.file_stat(path) for path in spec['runtime_binary_stat']}
        require(all(payloads[path] == row['stat'] for path,row in spec['payloads'].items())
                and binaries == spec['runtime_binary_stat'], 'Original payload/runtime identities')
        save(out/'protected-before.json', dict(trees=snapshots, payloads=payloads, binaries=binaries))
        for row in plan['files']:
            require(identity(row['path']) == row['stat'], 'Original journal identity at admission')
        check_unused(targets)

        def guard():
            p.low.lock_proof(owner, lock)
            p.low.clear_window()
            require(time.monotonic()-started < 1200, 'Bounded serial maintenance')
            require(free('/') >= plan['root_floor_bytes'] and free('/mnt/nvme') >= plan['nvme_floor_bytes'],
                    'Filesystem floors maintained')
            require(sum(f.stat().st_size for f in out.glob('*.gz')) <= plan['compressed_cap_bytes'],
                    'Bounded complete gzip output')

        rows = []
        for row in plan['files']:
            guard()
            archived = archive(row['path'], out/(Path(row['path']).name+'.gz'), row['stat'], guard)
            require(row['previous_sha256'] is None or archived['original_sha256'] == row['previous_sha256'],
                    'Prior independently recorded original bytes preserved')
            rows.append(archived)
            print(json.dumps(dict(archived=Path(row['path']).name, gzip_bytes=archived['gzip_bytes'])), flush=True)
        verified_ref = save(out/'verified-before-retirement.json', dict(files=rows, every_archive_decompressed=True))
        check_unused(targets)
        guard()
        p.low.active_heavy()
        for row in plan['files']:
            require(identity(row['path']) == row['stat'], 'All sources stable before any retirement')
        for row in rows:
            guard()
            require(identity(row['path']) == row['source_stat'], 'Exact source immediately before retirement')
            require(sha(Path(row['archive_path']).read_bytes()) == row['gzip_sha256'], 'Retained archive unchanged')
            if Path(row['path']).parent == ROOT_JOURNALS:
                subprocess.run(['sudo', '-n', '/usr/bin/rm', '--', row['path']], check=True)
            else:
                Path(row['path']).unlink()
            with (out/'actions.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(action='replace_closed_journal_with_verified_gzip', **row))+'\n')
                stream.flush()
                os.fsync(stream.fileno())
        require(all(not Path(name).exists() for name in targets), 'All exact planned sources retired')
        require({path:tree_snapshot(path) for path in snapshots} == snapshots, 'Protected experiment trees unchanged')
        require({path:p.file_stat(path) for path in payloads} == payloads
                and {path:p.file_stat(path) for path in binaries} == binaries, 'Original model/runtime unchanged')
        require(p.canonical() == canonical, 'Canonical progress inputs unchanged')
        after = dict(root=free('/'), nvme=free('/mnt/nvme'))
        save(out/'result.json', dict(complete=True, owner=owner, plan=plan_ref, verification=verified_ref,
             original_bytes=sum(row['original_bytes'] for row in rows),
             compressed_bytes=sum(row['gzip_bytes'] for row in rows), before_space=before, after_space=after,
             required_new_run_bytes=dict(root=p.ROOT_FLOOR+p.ROOT_CAP,nvme=p.NVME_FLOOR+p.NVME_CAP),
             protected_trees_unchanged=True, original_payloads_and_binaries_unchanged=True,
             canonical_unchanged=True, every_original_byte_retained=True,
             elapsed_seconds=time.monotonic()-started, actual_self_exit_claimed=False,
             runtime_admitted=False, native_started=False, new_speedup_claim=False))
        print(json.dumps(dict(complete=True, after_space=after)), flush=True)
    except BaseException as error:
        if created:
            save(out/'failure.json', dict(owner=owner, error=repr(error),
                 originals_may_only_be_retired_after_verified_archive=True, runtime_admitted=False))
        raise
    finally:
        release(fd)


if __name__ == '__main__':
    main()
