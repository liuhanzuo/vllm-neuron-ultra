"""Pure stdlib validation; synthetic worker failure and saved metadata only.

No actual worker admission, serial lock, service/proc/cgroup query, native import,
compiler, model, tensor payload or experiment is allowed by this checker.
"""
import base64
import copy
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch
import zlib

FORBIDDEN = {'numpy', 'ml_dtypes', 'nki', 'neuronxcc', 'torch', 'torch_xla', 'torch_neuronx',
             'neuronx_distributed', 'neuronx_distributed_inference', 'transformers'}


class RejectNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in FORBIDDEN:
            raise RuntimeError('Native import forbidden in source test: ' + fullname)


sys.meta_path.insert(0, RejectNative())
import run_lfm3_nki05_transport_serial_v1 as r


def rejected(function):
    try: function()
    except (RuntimeError, ValueError, KeyError, TypeError, AssertionError): return
    raise AssertionError('Invalid synthetic input accepted')


def valid_spec():
    return dict(schema='lfm3_nki05_serial_spec_v1', root_finalized=True, runner=r.SELF,
                diagnostic=str(Path(r.diagnostic.__file__).resolve()), source_sha256={r.SELF: '0' * 64},
                serial_catalog=dict(path=str(r.ROOT / 'research/synthetic-catalog.json'), sha256='1' * 64),
                terminal_handoff=dict(path=str(r.ROOT / 'research/synthetic-terminal.json'), sha256='2' * 64),
                output_root=str(r.ROOT / 'lfm3-nki05-transport-synthetic'), unit='trn1-lfm3-nki05-transport-synthetic',
                resources=copy.deepcopy(r.PROPS), lock_identity=copy.deepcopy(r.LOCK),
                canonical_sha256={str(r.ROOT / 'continuation-state.json'): '3' * 64},
                root_floor_bytes=r.FLOOR, output_cap_bytes=r.OUTPUT_CAP,
                model_qualification_claim=False, automatic_successor_launch=False)


def good_result():
    return dict(schema='lfm3_nki05_transport_diagnostic_v1', complete=True, passed=True,
                legacy_full3716_passed=False, full_model_qualified=False, device_behavior_proven=False,
                new_speedup_claim=False, checks=[dict(passed=True, inputs_unchanged=True, observer_coverage=True) for _ in range(32)],
                bf16_encoding_count=65536, exact_fixture_hashes_verified=True, arithmetic_control=True,
                round_to_nearest_even_cast_control=True, negative_controls=r.diagnostic.negative_controls())


def failed_worker_child(output):
    # Invoke the real CLI dispatch and actual execute_diagnostic failure boundary.
    # Only process setup/native callback is replaced; no admission or native code.
    r.diagnostic.diagnose = lambda admit: dict(schema='lfm3_nki05_transport_diagnostic_v1',
                                               complete=False, passed=False, error='synthetic expected failure')
    r.worker = lambda ref: r.execute_diagnostic(ref, dict(synthetic=True), Path(output), lambda: True)
    sys.argv = [r.SELF, '--mode', 'worker', '--admission', 'synthetic', '--admission-sha256', '0' * 64]
    r.main()


def metadata_compression():
    """Parent-authorized authenticated existing saved metadata, never fresh drain."""
    def read(ref):
        path = Path(ref['path']); raw = path.read_bytes()
        assert len(raw) <= 2 * 2**20 and hashlib.sha256(raw).hexdigest() == ref['sha256']
        return json.loads(raw)
    live = read(r.history.MINI_LIVE)
    admission = read(live['admission'])
    source_ref = admission['observed_drains']
    saved = read(source_ref); packed = saved['paired_physical']
    compressed = base64.b64decode(packed['data'], validate=True)
    assert len(compressed) == packed['compressed_bytes']
    decoder = zlib.decompressobj()
    raw = decoder.decompress(compressed, packed['uncompressed_bytes'] + 1)
    assert decoder.eof and not decoder.unconsumed_tail and not decoder.unused_data
    assert len(raw) == packed['uncompressed_bytes'] <= 32 * 2**20
    assert hashlib.sha256(raw).hexdigest() == packed['sha256']
    pair = json.loads(raw)
    units = r.history.physical.validate_pair(pair, lock=r.LOCK)
    catalog = read(r.history.MINIMUM)
    assert len(catalog['runs']) == 85
    provenance = {}
    for row in catalog['runs']:
        for path, wanted in row['source_sha256'].items():
            assert path not in provenance or provenance[path] == wanted
            provenance[path] = wanted
    owners = pair['first']['tree']['all_additional_owners_checked']
    # Conservatively retain the entire existing nested historical pair, plus
    # duplicated complete owners/source provenance as the new proof will do.
    representative = dict(historical_metadata_representative_only=True, source_ref=source_ref,
                          full_historical_pair=pair, owner_scope=owners, repeated_owner_proof=owners,
                          later_source_provenance=provenance, all_units=sorted(units),
                          full_current_catalog=catalog)
    manifest, parts = r.proof_parts(representative)
    assert r.join_proof(manifest, parts) == representative
    serialized = [len(r.encode(part)) for part in parts]
    assert max(serialized) <= r.FILE_CAP
    assert sum(serialized) + len(r.encode(manifest)) < r.OUTPUT_CAP - 1024 * 1024
    missing = parts[:-1]; rejected(lambda: r.join_proof(manifest, missing))
    reversed_parts = list(reversed(parts)); rejected(lambda: r.join_proof(manifest, reversed_parts))
    damaged = copy.deepcopy(parts); damaged[0]['data'] = 'AAAA' + damaged[0]['data'][4:]
    rejected(lambda: r.join_proof(manifest, damaged))
    return dict(source=source_ref, source_json_bytes=Path(source_ref['path']).stat().st_size,
                historical_units=len(units), historical_owners=len(owners), minimum_catalog_runs=85,
                source_provenance_entries=len(provenance), representative_raw_bytes=manifest['raw_bytes'],
                compressed_bytes=manifest['compressed_bytes'], chunks=len(parts), maximum_chunk_json_bytes=max(serialized),
                total_chunk_and_manifest_bytes=sum(serialized) + len(r.encode(manifest)),
                lossless_roundtrip=True, missing_reordered_corrupted_chunks_rejected=True,
                actual_current_physical_drain_claimed=False)


def check():
    assert not FORBIDDEN.intersection(name.split('.')[0] for name in sys.modules)
    spec = valid_spec(); r.validate_spec(spec)
    for key, wrong in [('AllowedCPUs', '7'), ('Nice', '0'), ('MemoryMax', str(16 * 2**30)),
                       ('MemoryHigh', str(10 * 2**30)), ('MemorySwapMax', '1'), ('TasksMax', '128'),
                       ('RuntimeMaxSec', '1801'), ('LimitCPU', '1801'), ('LimitAS', str(13 * 2**30)),
                       ('LimitFSIZE', str(r.FILE_CAP + 1))]:
        bad = copy.deepcopy(spec); bad['resources'][key] = wrong
        rejected(lambda: r.validate_spec(bad))
    for key, wrong in [('root_finalized', False), ('lock_identity', dict(device=66306, inode=99)),
                       ('serial_catalog', r.history.MINIMUM), ('model_qualification_claim', True),
                       ('automatic_successor_launch', True), ('output_cap_bytes', r.OUTPUT_CAP + 1)]:
        bad = copy.deepcopy(spec); bad[key] = wrong; rejected(lambda: r.validate_spec(bad))
    ref = dict(path=spec['output_root'] + '/admission.json', sha256='a' * 64); attempt = 'b' * 32
    command = r.service_command(spec, ref, attempt)
    env = r.environment(spec['output_root'], attempt)
    assert command[command.index('/usr/bin/env') + 1] == '-i'
    assert command[-len(r.worker_argv(ref)):] == r.worker_argv(ref)
    assert env['NEURON_PLATFORM_TARGET_OVERRIDE'] == 'trn1' and env['NKI_PRECISE_FP'] == '1'
    assert env['NEURON_RT_VISIBLE_CORES'] == '' and env['LFM3_NKI05_ATTEMPT'] == attempt
    assert all(env[key] == '1' for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'))
    assert not {'PYTHONPATH', 'PYTHONHOME', 'LD_PRELOAD', 'LD_LIBRARY_PATH'} & set(env)
    props = dict(r.PROPS, LoadState='loaded', Id=spec['unit'] + '.service', InvocationID='c' * 32,
                 ExecMainPID='123', ControlGroup=r.p.GROUP_PREFIX + spec['unit'] + '.service', ExecStopPost='')
    for key in ('RuntimeMaxSec', 'TimeoutStopSec'): props[key.replace('Sec', 'USec')] = props.pop(key) + 's'
    props['ExecStart'] = '{ path=/usr/bin/env ; argv[]=' + ' '.join(command[command.index('/usr/bin/env'):]) + ' ; }'
    owner = dict(pid=123, invocation_id='c' * 32)
    r.service_identity(spec, ref, attempt, props, owner)
    for key, wrong in [('InvocationID', 'd' * 32), ('ExecMainPID', '124'), ('ExecStart', 'wrong'),
                       ('MemoryMax', '1'), ('ExecStopPost', 'unapproved cleanup')]:
        bad = dict(props, **{key: wrong}); rejected(lambda: r.service_identity(spec, ref, attempt, bad, owner))
    # Resource mismatch cannot prevent authenticated cleanup, but changed identity can.
    r.service_identity(spec, ref, attempt, dict(props, MemoryMax='1'), owner, resources=False)
    rejected(lambda: r.service_identity(spec, ref, attempt, dict(props, InvocationID='d' * 32), owner, resources=False))
    good = good_result(); r.require_diagnostic_success(good)
    for key, wrong in [('complete', False), ('passed', False), ('complete', 1), ('passed', 1),
                       ('error', 'failed'), ('checks', good['checks'][:-1]), ('bf16_encoding_count', 65535),
                       ('full_model_qualified', True), ('arithmetic_control', False)]:
        bad = dict(good, **{key: wrong}); rejected(lambda: r.require_diagnostic_success(bad))
    with tempfile.TemporaryDirectory(prefix='lfm3-source-synthetic-') as directory:
        completed = subprocess.run([sys.executable, '-B', __file__, '--failed-worker-child', directory],
                                   capture_output=True, text=True, timeout=30)
        assert completed.returncode != 0 and 'Incomplete/failed diagnostic' in completed.stderr
        output = Path(directory)
        assert json.loads((output / 'diagnostic-result.json').read_text())['passed'] is False
        assert json.loads((output / 'worker-failure.json').read_text())['passed'] is False
        assert not (output / 'worker-result.json').exists()
        failed_worker_exit = completed.returncode
    events = []
    class Owned:
        calls = 0
        def cleanup(self): events.append('cleanup')
        def drained(self):
            self.calls += 1; events.append('verify')
            if self.calls == 1: raise RuntimeError('synthetic child still live')
            events.append('drain_proven'); return dict(physically_drained=True)
    with patch.object(r.fcntl, 'flock', side_effect=lambda *args: events.append('unlock')), \
         patch.object(r.os, 'close', side_effect=lambda *args: events.append('close')):
        r.drain_and_release(Owned(), 123, lambda value: events.append('pending'),
                            lambda: events.append('retry'), lambda value: events.append('record'))
    assert events == ['cleanup', 'verify', 'pending', 'retry', 'cleanup', 'verify', 'drain_proven', 'record', 'unlock', 'close']
    expected = dict(path='synthetic-latest', identity=dict(inode=1, mtime_ns=1))
    with patch.object(r.history.serial, 'latest_handoff', return_value=expected): r.check_latest(expected)
    for different in [dict(expected, path='new-latest'), dict(expected, identity=dict(inode=2, mtime_ns=1))]:
        with patch.object(r.history.serial, 'latest_handoff', return_value=different): rejected(lambda: r.check_latest(expected))
    # Exact 85-row minimum, real-current-session requirement, and failure retention.
    minimum = dict(runs=[dict(terminal=dict(path=f'/synthetic/{i}.json'), tool={}, owners=[], units=[], source_sha256={}) for i in range(85)])
    process = dict(pid=1, start_ticks='1', ppid=0, argv=['synthetic'], cgroup='0::/synthetic')
    co = dict(process, pid=2, unit='trn1-synthetic')
    live = dict(owner=process, coordinator=co, actual_execution_handle=dict(kind='tool_session', session_id=58352))
    row = dict(terminal=dict(path='/synthetic/latest.json'), tool=dict(kind='tool_session', session_id=58352, exit_code=1),
               owners=[process, co], units=['trn1-synthetic'], source_sha256={})
    current = dict(runs=copy.deepcopy(minimum['runs']) + [row])
    assert r.history.retained_runs(current, minimum, live)['tool']['exit_code'] == 1
    rejected(lambda: r.history.retained_runs(dict(runs=current['runs'][1:]), minimum, live))
    rejected(lambda: r.history.retained_runs(minimum, minimum, live))
    changed = copy.deepcopy(current); changed['runs'][-1]['owners'] = [process]
    rejected(lambda: r.history.retained_runs(changed, minimum, live))
    compression = metadata_compression()
    sources = r.execution_source_paths()
    assert len(sources) < 1024 and all(path.is_file() and path.suffix == '.py' for path in sources)
    assert not FORBIDDEN.intersection(name.split('.')[0] for name in sys.modules)
    return dict(passed=True, source_only=True, synthetic_failed_worker_exit_code=failed_worker_exit,
                failed_diagnostic_preserved=True, failure_never_promoted=True, original_resource_mutations_rejected=True,
                exact_clean_env_argv_verified=True, original85_and_actual_mini_terminal_required=True,
                newer_or_replaced_latest_handoff_rejected=True, physical_drain_before_unlock_verified_synthetically=True,
                metadata_compression=compression, import_only_local_closure_count=len(sources),
                native_imports=False, actual_runtime_admission=False, actual_lock_operations=False,
                service_queries=False, fresh_physical_checks=False, model_execution=False, payload_reads=False,
                source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources})


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--failed-worker-child': failed_worker_child(sys.argv[2])
    else:
        report = check()
        print(json.dumps({key: value for key, value in report.items() if key != 'source_sha256'}, indent=2))
        if len(sys.argv) == 3 and sys.argv[1] == '--report':
            with Path(sys.argv[2]).open('x') as stream:
                json.dump(report, stream, indent=2, sort_keys=True); stream.write('\n')
