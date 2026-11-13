"""Root post-exit byte analysis of the saved 15-call prefix; never qualification.

Run only after the original independent observer and catalog finalization.
This adds analysis evidence, not sources retrofitted into the executed spec.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import stat
import struct
import sys
import time

Q = Path(__file__).resolve().parent
sys.path[:0] = [str(Q), str(Q.parent)]
from qwen35_sampler_runtime_io_v12 import RetainedLock, read, ref, once, identity, process_api
from qwen35_kv256_alias_diagnostic_serial_v1 import protocol
from qwen35_kv256_alias_diagnostic_bytes_v1 import pattern, compare, STATE_BYTES, TOKENS, WIDTH, PROFILES


def require(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('actual-exit', 'actual-exit-sha256', 'idle', 'idle-sha256', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    require(sys.dont_write_bytecode and __debug__, 'Explicit -B and assertions')
    resource.setrlimit(resource.RLIMIT_AS, (2 * 2**30, 2 * 2**30))
    resource.setrlimit(resource.RLIMIT_CPU, (600, 605))
    signal.alarm(600)
    exit_ref = dict(path=args.actual_exit, sha256=args.actual_exit_sha256)
    idle_ref = dict(path=args.idle, sha256=args.idle_sha256)
    ended, idle = read(exit_ref), read(idle_ref)
    require(ended['actual_tool'] == dict(kind='tool_session', session_id=18826,
        terminal_chunk='ea9360', exit_code=1), 'Actual failed r2 tool only')
    require(ended['owners_gone'] and ended['physically_drained']
        and not ended['diagnostic_completed'] and ended['completed_calls'] == 15,
        'Original failed prefix and independent drain remain required')
    require(idle['running'] is False and idle['active_serial'] is None
        and idle['active_heavy_units'] == idle['live_tool_sessions'] == [], 'Current finalized idle only')
    require(idle['qwen35_kv256_alias_diagnostic_current']['actual_exit'] == exit_ref,
        'Finalized idle belongs to this failed diagnostic')
    output = Path(args.output)
    require(output.parent == Q and not output.exists() and not output.is_symlink(), 'Fresh root evidence')
    spec = read(ended['spec'], 64 * 2**20)
    for source in (Path(__file__), Q/'qwen35_kv256_alias_diagnostic_bytes_v1.py',
                   Q/'qwen35_kv256_alias_diagnostic_leaf_v1.py',
                   Q/'qwen35_kv256_alias_diagnostic_serial_v1/protocol.py'):
        if source != Path(__file__):
            require(ref(source)['sha256'] == spec['source_sha256'][str(source)], 'Original executed pure analysis source')
    from research.minicpm5_layer1_operator_sdk_runtime_completion_v4 import latest_handoff
    expected = dict(path=idle_ref['path'], identity=identity(idle_ref['path']))
    lock = RetainedLock()
    lock.acquire()
    started, last = time.monotonic(), [-1.0]

    def guard():
        lock.check()
        now = time.monotonic()
        require(now - started < 600, 'Finite independent prefix analysis')
        require(identity(idle_ref['path']) == expected['identity'], 'Idle identity unchanged')
        if now - last[0] >= .5:
            require(latest_handoff() == expected, 'Unknown successor during raw analysis')
            process_api().active_heavy()
            last[0] = now

    try:
        guard()
        for owner in ended['owners'] + [ended['actual_observer']]:
            process_api().gone(owner)
        report_ref = ended['reports']['kv256_alias_diag']
        report = read(report_ref, 16 * 2**20)
        root = Path(report['output_root'])
        require(root == Path(spec['output_root'])/'kv256_alias_diag/artifacts'
            and root.resolve() == root, 'Exact retained capture root')
        layout = ended['layouts'][0]
        protocol.validate_prefix(report, 'kv256_alias_diag', layout['artifacts'])
        require(report['complete'] is False and len(report['calls']) == 15
            and all('all_expected_bytes_equal' in row for row in report['calls']),
            'Exactly saved complete calls; missing calls remain unknown')
        seen = {}

        def capture(reference, role):
            guard()
            require(set(reference) == {'path', 'sha256', 'bytes'}
                and 0 < reference['bytes'] <= STATE_BYTES, 'Bounded complete raw reference')
            path = Path(reference['path'])
            require(path.is_relative_to(root/role) and path.resolve() == path
                and not any(p.is_symlink() for p in path.parents), 'Exact no-symlink capture path')
            before = identity(path)
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
            try:
                info = os.fstat(fd)
                require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1
                    and info.st_size == reference['bytes'] and info.st_ino == before['inode']
                    and info.st_dev == before['device'], 'Exact singleton capture')
                with os.fdopen(fd, 'rb', closefd=False) as stream:
                    data = stream.read(reference['bytes'] + 1)
                require(len(data) == reference['bytes']
                    and hashlib.sha256(data).hexdigest() == reference['sha256']
                    and identity(path) == before, 'Full raw contents and identity unchanged')
            finally:
                os.close(fd)
            require(str(path) not in seen or seen[str(path)]['reference'] == reference,
                'No conflicting raw references')
            seen[str(path)] = dict(reference=reference, identity=before)
            guard()
            return data

        require(set(report['inputs']) == set(PROFILES), 'All three exact input profiles')
        for profile, refs in report['inputs'].items():
            expected_cache = pattern(profile, TOKENS*WIDTH, salt=17)
            require(capture(refs['cache'], 'inputs') == expected_cache, 'Exact original cache pattern')
            require(capture(refs['old'], 'inputs') == expected_cache[:STATE_BYTES//4]*4,
                'Exact clamped old gather oracle')
            require(capture(refs['slots'], 'inputs') == struct.pack('<iiii', -1, -1, -1, -1), 'All inactive int32')
            for bucket in (128, 256):
                require(capture(refs['updates'][str(bucket)], 'inputs') == pattern(profile, bucket*WIDTH, salt=31),
                    'Exact complete update pattern')
        comparisons, outcomes = 0, []
        for row in report['calls']:
            graph = next(g for g in protocol.GRAPHS if g['name'] == row['graph'])
            require(row['bucket'] == graph['bucket'] and row['alias'] == graph['alias']
                and row['inputs'] == report['inputs'][row['profile']]
                and row['physical_output_order'] == protocol.leaf.physical_output_order(graph),
                'Original graph geometry, profile inputs and logical output normalization')
            if graph['alias'] is not None:
                require(row['reset_exact'] is True and row['reset_sha256'] == row['inputs']['cache']['sha256'],
                    'Original complete persistent reset recorded')
            require(row['call_returned'] and all(row['host_inputs_unchanged'])
                and not row.get('changed_host_inputs'), 'This prefix records unchanged host inputs')
            expected_keys = {'old', 'new'} | ({'state'} if row['alias'] is not None else set())
            require(set(row['proofs']) == expected_keys, 'Every complete output/state retained')
            proofs = {}
            for name, proof in row['proofs'].items():
                folder = root/'calls'/(row['graph']+'-'+row['profile'])
                require(proof['actual']['path'] == str(folder/(name+'.raw'))
                    and proof['difference_bitmap']['path'] == str(folder/(name+'.diffbits'))
                    and proof['expected'] == row['inputs']['old' if name == 'old' else 'cache'],
                    'Complete exact raw/bitmap/oracle relationship')
                actual, oracle = capture(proof['actual'], 'calls'), capture(proof['expected'], 'inputs')
                bitmap = capture(proof['difference_bitmap'], 'calls')
                observed, derived_bits = compare(actual, oracle, row['bucket'], guard=guard)
                require(bitmap == derived_bits and observed == {k:v for k,v in proof.items()
                    if k not in ('actual','expected','difference_bitmap')}, 'Every saved difference reproduced')
                comparisons += 1
                proofs[name] = {k:observed[k] for k in ('identical','actual_sha256','expected_sha256',
                    'differing_bytes','differing_words','word_class_transitions','partitions')}
            require(row['all_expected_bytes_equal'] is all(x['identical'] for x in proofs.values()),
                'Honest saved equality flag')
            outcomes.append(dict(graph=row['graph'],profile=row['profile'],proofs=proofs,
                all_expected_bytes_equal=row['all_expected_bytes_equal']))
        require(comparisons == 42, 'All42 complete saved comparisons')
        inventory = []
        for path in sorted(root.rglob('*')):
            if path.suffix not in ('.raw','.diffbits'):
                continue
            info = path.lstat()
            require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, 'Ordinary retained capture')
            if str(path) not in seen:
                inventory.append(dict(path=str(path),bytes=info.st_size,
                    status='unreferenced_by_last_saved_report_not_analyzed'))
        require(all(identity(path) == row['identity'] for path,row in seen.items()), 'All analyzed inputs unchanged')
        require(ref(report_ref['path'],16*2**20) == report_ref and ref(exit_ref['path']) == exit_ref,
            'Original report and terminal unchanged')
        guard()
        result = dict(schema='qwen35_kv256_partial_bytes_independent_root_analysis_v1',reviewer='/root',
            actual_exit=exit_ref,idle=idle_ref,report=report_ref,analysis_source=ref(__file__),
            saved_calls=15,verified_comparisons=comparisons,verified_raw_files=len(seen),
            verified_unique_raw_bytes=sum(x['reference']['bytes'] for x in seen.values()),
            all_saved_prefix_bytes_and_bitmaps_reproduced=True,outcomes=outcomes,
            unreferenced_captures=inventory,remaining_call_outcomes_unknown=True,
            original_experiment_failed=True,diagnostic_completed=False,
            full_geometry_qualified=False,full_model_qualified=False,new_speedup_claim=False)
        print(json.dumps(dict(result=once(output,result),comparisons=comparisons,calls=15,
            unreferenced_captures=len(inventory))),flush=True)
    finally:
        lock.release(True)


if __name__ == '__main__':
    main()
