"""Thin original-lock outer + one bounded NKI simulation worker; inert import.

Root supplies a concrete spec/review and actual post-Mini idle catalog. No
specification, terminal outcome, catalog or handoff is manufactured here.
"""
import argparse
import base64
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import signal
import stat
import sys
import time
import zlib

import minicpm5_layer0_execution_protocol_v4 as p
from minicpm5_layer0_runtime_auth_v1 import identity
from research import descriptor_serial_lock_proof_v1 as descriptor
from research.minicpm5_layer1_drain_hold_v1 import retain_until_drained
import lfm3_nki05_serial_history_v1 as history
import lfm3_nki05_transport_diagnostic_v3 as diagnostic
import lfm3_nki05_torch_namespace_sources_v1 as namespace_sources

ROOT = Path(__file__).resolve().parent
SELF = str(Path(__file__).resolve())
PYTHON = diagnostic.PREFIX + '/bin/python'
LOCK = dict(device=66306, inode=8294421)
FILE_CAP = 256 * 2**10
OUTPUT_CAP = 4 * 2**20
FLOOR = 28 * 2**30
AUTHOR = '/root/mini_operator_saved_comparison_design_20261111'
PROPS = dict(MemoryMax=str(10 * 2**30), MemoryHigh=str(8 * 2**30), MemorySwapMax='0', TasksMax='32',
             AllowedCPUs='6-7', CPUAffinity='6-7', Nice='19', KillMode='control-group',
             RemainAfterExit='yes', SendSIGKILL='yes', WorkingDirectory=str(ROOT),
             LimitFSIZE=str(FILE_CAP), LimitAS=str(12 * 2**30), LimitCPU='1800',
             RuntimeMaxSec='1800', TimeoutStopSec='15')
ORIGINAL_HELPERS = {
    str(Path(p.__file__).resolve()): '81e7d7dd913c6b85f0add7b946692d3270f80cc8f2fe980ecad543251657f185',
    str(Path(descriptor.__file__).resolve()): 'f3faa10356e16dce6b689298ceebf11441f3821ce391f54896cbffed1c82b003',
}
SPEC_KEYS = {'schema', 'root_finalized', 'runner', 'diagnostic', 'source_sha256', 'serial_catalog',
             'terminal_handoff', 'output_root', 'unit', 'resources', 'lock_identity', 'canonical_sha256',
             'root_floor_bytes', 'output_cap_bytes', 'model_qualification_claim', 'automatic_successor_launch'}


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def digest(value):
    return hashlib.sha256(encode(value)).hexdigest()


def reference(path):
    return dict(path=str(path), sha256=p.sha(path))


def packed(value):
    raw = encode(value)
    p.require(len(raw) <= 32 * 2**20, 'Finite complete metadata proof')
    compressed = zlib.compress(raw, 6)
    row = dict(encoding='zlib-base64-json-v1', uncompressed_bytes=len(raw),
               sha256=hashlib.sha256(raw).hexdigest(), data=base64.b64encode(compressed).decode('ascii'))
    p.require(len(encode(row)) < FILE_CAP, 'Complete proof must fit original report cap; never truncate')
    return row


def unpacked(value):
    p.require(set(value) == {'encoding', 'uncompressed_bytes', 'sha256', 'data'}
              and value['encoding'] == 'zlib-base64-json-v1'
              and type(value['uncompressed_bytes']) is int and 0 < value['uncompressed_bytes'] <= 32 * 2**20,
              'Exact bounded proof encoding')
    decoder = zlib.decompressobj()
    raw = decoder.decompress(base64.b64decode(value['data'], validate=True), value['uncompressed_bytes'] + 1)
    p.require(decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail
              and len(raw) == value['uncompressed_bytes'] and hashlib.sha256(raw).hexdigest() == value['sha256'],
              'Complete authenticated proof, no trailing or missing bytes')
    return json.loads(raw)


def proof_parts(value):
    """Finite lossless chunks, keeping the original 256 KiB per-file ceiling."""
    raw = encode(value)
    p.require(len(raw) <= 32 * 2**20, 'Original complete proof expansion limit')
    compressed = zlib.compress(raw, 6)
    p.require(len(compressed) <= 2 * 2**20, 'Finite complete proof within four-MiB total output')
    parts = [dict(schema='lfm3_nki05_metadata_chunk_v1', index=i // (128 * 2**10),
                  data=base64.b64encode(compressed[i:i + 128 * 2**10]).decode('ascii'))
             for i in range(0, len(compressed), 128 * 2**10)]
    p.require(1 <= len(parts) <= 16 and all(len(encode(part)) <= FILE_CAP for part in parts), 'Bounded complete chunks')
    manifest = dict(schema='lfm3_nki05_metadata_chunks_v1', complete=True, encoding='zlib-base64-json-v1',
                    raw_bytes=len(raw), raw_sha256=hashlib.sha256(raw).hexdigest(),
                    compressed_bytes=len(compressed), compressed_sha256=hashlib.sha256(compressed).hexdigest(),
                    chunks=[dict(filename=f'predecessor-drain-chunk-{i:03d}.json', sha256=digest(part))
                            for i, part in enumerate(parts)])
    return manifest, parts


def join_proof(manifest, parts):
    p.require(manifest['schema'] == 'lfm3_nki05_metadata_chunks_v1' and manifest['complete'] is True
              and manifest['encoding'] == 'zlib-base64-json-v1' and 1 <= len(parts) <= 16
              and len(parts) == len(manifest['chunks'])
              and type(manifest['raw_bytes']) is int and 0 < manifest['raw_bytes'] <= 32 * 2**20,
              'Complete bounded predecessor chunk manifest')
    blocks = []
    for i, (part, ref) in enumerate(zip(parts, manifest['chunks'])):
        p.require(set(part) == {'schema', 'index', 'data'} and part['schema'] == 'lfm3_nki05_metadata_chunk_v1'
                  and type(part['index']) is int and part['index'] == i
                  and ref == dict(filename=f'predecessor-drain-chunk-{i:03d}.json', sha256=digest(part))
                  and len(encode(part)) <= FILE_CAP, 'Every exact chunk present in order')
        block = base64.b64decode(part['data'], validate=True)
        p.require(0 < len(block) <= 128 * 2**10, 'Original chunk byte bound'); blocks.append(block)
    compressed = b''.join(blocks)
    p.require(len(compressed) == manifest['compressed_bytes'] <= 2 * 2**20
              and hashlib.sha256(compressed).hexdigest() == manifest['compressed_sha256'], 'Complete compressed proof hash')
    decoder = zlib.decompressobj(); raw = decoder.decompress(compressed, manifest['raw_bytes'] + 1)
    p.require(decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail
              and len(raw) == manifest['raw_bytes'] and hashlib.sha256(raw).hexdigest() == manifest['raw_sha256'],
              'Complete lossless proof after bounded decoding')
    return json.loads(raw)


def save_proof(root, value):
    manifest, parts = proof_parts(value)
    for ref, part in zip(manifest['chunks'], parts): once(root / ref['filename'], part)
    once(root / 'predecessor-drain.json', manifest)


def once(path, value):
    path = Path(path); raw = encode(value)
    p.require(len(raw) <= FILE_CAP and path.parent.resolve() == path.parent and not path.is_symlink(),
              'Bounded new report in exact output directory')
    temp = path.with_name('.' + path.name + '.tmp')
    p.require(not path.exists() and not temp.exists(), 'Never replace previous evidence')
    try:
        with temp.open('xb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.link(temp, path)
    finally:
        if temp.exists(): temp.unlink()


def report_failure(path, error, **context):
    try:
        once(path, dict(complete=False, passed=False, error_type=type(error).__name__, error=str(error)[:2048],
                       self_exit_claimed=False, full_model_qualified=False, new_speedup_claim=False, **context))
    except BaseException as secondary:
        print(json.dumps(dict(stage='failure_receipt', original=repr(error)[:1024], secondary=repr(secondary)[:1024])),
              file=sys.stderr, flush=True)


class Metadata:
    def __init__(self, guard):
        self.guard, self.observed = guard, {}

    def read(self, ref):
        self.guard(); before = identity(ref['path'])
        # Reuse the original named large-JSON envelope reader only for historical
        # metadata. No native/payload authentication or model source-map merge.
        value = history.serial.sources.load(ref)
        p.require(identity(ref['path']) == before, 'Stable authenticated historical metadata')
        p.require(ref['path'] not in self.observed or self.observed[ref['path']]['identity'] == before,
                  'No metadata identity rebound')
        self.observed[ref['path']] = dict(sha256=ref['sha256'], identity=before)
        self.guard(); return value

    def unchanged(self):
        p.require(all(identity(path) == row['identity'] for path, row in self.observed.items()),
                  'Every admitted metadata file unchanged')


def descriptor_proof(owner, fd):
    return descriptor.prove(owner, LOCK, fd, proc=p.proc, open_binary=lambda path: open(path, 'rb'),
                            lstat=os.lstat, stat=os.stat, readlink=os.readlink)


def acquire():
    path = Path(descriptor.LOCK_PATH)
    for parent in reversed(path.parents):
        p.require(stat.S_ISDIR(parent.lstat().st_mode), 'Original no-symlink lock ancestry')
    before = path.lstat()
    p.require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
              and dict(device=before.st_dev, inode=before.st_ino) == LOCK, 'Original singleton serial lock')
    fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        info = os.fstat(fd)
        p.require(not os.get_inheritable(fd) and info.st_dev == before.st_dev and info.st_ino == before.st_ino,
                  'Retained parent-only original descriptor')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        owner = p.proc(os.getpid())
        return fd, owner, descriptor_proof(owner, fd)
    except BaseException:
        os.close(fd); raise


def validate_spec(spec):
    p.require(type(spec) is dict and set(spec) == SPEC_KEYS and spec['schema'] == 'lfm3_nki05_serial_spec_v1'
              and spec['root_finalized'] is True, 'Concrete root-finalized LFM diagnostic spec')
    p.require(spec['runner'] == SELF and spec['diagnostic'] == str(Path(diagnostic.__file__).resolve()),
              'Exact two execution entrypoints')
    p.require(spec['resources'] == PROPS and spec['lock_identity'] == LOCK, 'Original lock and requested resource limits')
    p.require(spec['root_floor_bytes'] == FLOOR and spec['output_cap_bytes'] == OUTPUT_CAP,
              'Original root floor and finite metadata-only output')
    p.require(spec['model_qualification_claim'] is spec['automatic_successor_launch'] is False,
              'No promotion or automatic next model gate')
    root = Path(spec['output_root'])
    p.require(root.parent == ROOT and re.fullmatch(r'lfm3-nki05-transport-[a-z0-9-]{1,80}', root.name),
              'Fresh finite LFM output namespace')
    p.require(re.fullmatch(r'trn1-lfm3-nki05-transport-[a-z0-9-]{1,80}', spec['unit']), 'Fresh bounded service name')
    for key in ('serial_catalog', 'terminal_handoff'):
        history.serial.sources.reference(spec[key])
    p.require(spec['serial_catalog'] != history.MINIMUM, 'Pre-Mini catalog cannot admit a post-Mini worker')
    p.require(type(spec['source_sha256']) is dict and 0 < len(spec['source_sha256']) <= 1024,
              'True local/runtime dependency closure, not unrelated model library maps')
    p.require(type(spec['canonical_sha256']) is dict and spec['canonical_sha256'], 'Actual root-bound canonical snapshots')
    for name in spec['canonical_sha256']:
        p.require(Path(name).parent == ROOT and Path(name).suffix == '.json', 'Only exact root canonical metadata')
    return spec


def verify_sources(spec):
    p.require(all(spec['source_sha256'].get(path) == wanted for path, wanted in ORIGINAL_HELPERS.items()),
              'Original descriptor/process implementations retained exactly')
    for path, wanted in spec['source_sha256'].items():
        p.require(Path(path).is_absolute() and Path(path).suffix == '.py' and p.sha(path) == wanted,
                  'Exact execution dependency source: ' + path)
    for name, module in tuple(sys.modules.items()):
        if namespace_sources.known_namespace(name, module, sys.modules, spec['source_sha256'],
                                            lambda path: Path(path).read_bytes()):
            continue
        path = getattr(module, '__file__', None)
        if path and Path(path).resolve().is_relative_to(ROOT):
            resolved = str(Path(path).resolve())
            p.require(spec['source_sha256'].get(resolved) == p.sha(resolved), 'Every loaded local helper pinned: ' + resolved)
    for path in (SELF, str(Path(diagnostic.__file__).resolve()), str(Path(history.__file__).resolve())):
        p.require(spec['source_sha256'].get(path) == p.sha(path), 'Actual new execution sources reviewed')


def execution_source_paths(reviewed_source_sha256):
    """Root supplies the complete reviewed map, preserving the original 558 pins.

    Initial sys.modules is insufficient for lazy historical branches. This
    function never infers a new closure or discards previously reviewed pins.
    Root must supply its hash-authenticated additive closure report explicitly.
    """
    p.require(type(reviewed_source_sha256) is dict and 558 <= len(reviewed_source_sha256) <= 1024,
              'Explicit full preserved-plus-additive source map; never initial-import inference')
    verify_sources(dict(source_sha256=reviewed_source_sha256))
    return sorted(Path(path) for path in reviewed_source_sha256)


def check_latest(expected):
    p.require(history.serial.latest_handoff() == expected,
              'No replacement or newer latest handoff during this original serial lease')


def read_bundle(spec_ref, review_ref):
    spec, review = history.serial.sources.load(spec_ref), history.serial.sources.load(review_ref)
    validate_spec(spec)
    p.require(review['schema'] == 'lfm3_nki05_serial_source_review_v1' and review['reviewer'] == '/root'
              and review['source_author'] == AUTHOR and review['source_only'] is review['passed'] is True
              and review['blockers'] == [] and review['spec'] == spec_ref
              and review['source_sha256'] == spec['source_sha256'] and review['runtime_admitted'] is False,
              'Root independent review of exact concrete spec and sources')
    verify_sources(spec)
    return spec


def environment(root, attempt):
    return dict(PATH=diagnostic.PREFIX + '/bin:/usr/bin:/bin', HOME='/home/ubuntu',
                XDG_RUNTIME_DIR='/run/user/1000', DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1000/bus',
                PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONOPTIMIZE='0',
                TMPDIR=str(Path(root) / 'tmp'), NEURON_RT_VISIBLE_CORES='',
                NKI_PRECISE_FP='1', NEURON_PLATFORM_TARGET_OVERRIDE='trn1',
                OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                LFM3_NKI05_ATTEMPT=attempt)


def worker_argv(admission_ref):
    return [PYTHON, '-B', SELF, '--mode', 'worker', '--admission', admission_ref['path'],
            '--admission-sha256', admission_ref['sha256']]


def service_command(spec, admission_ref, attempt):
    return ['systemd-run', '--user', '--quiet', '--unit=' + spec['unit'], '--service-type=exec',
            *[item for key, value in PROPS.items() for item in ('-p', key + '=' + value)],
            '-p', 'StandardOutput=append:' + spec['output_root'] + '/runtime.log', '-p', 'StandardError=inherit',
            '/usr/bin/env', '-i', *[key + '=' + value for key, value in environment(spec['output_root'], attempt).items()],
            *worker_argv(admission_ref)]


def properties(unit):
    p.require(re.fullmatch(r'trn1-lfm3-nki05-transport-[a-z0-9-]{1,80}', unit), 'Exact LFM service namespace')
    return p.parse_properties(p.command(['systemctl', '--user', 'show', '--all', unit,
                                        '--property=' + p.KEYS + ',LimitAS,LimitCPU']))


def service_identity(spec, ref, attempt, value, owner=None, *, resources=True):
    p.require(value['LoadState'] == 'loaded' and value['Id'] == spec['unit'] + '.service'
              and p.invocation(value['InvocationID']), 'Exact loaded service invocation')
    command = service_command(spec, ref, attempt)
    p.require(re.findall(r'argv\[\]=(.*?)\s*;', value['ExecStart']) == [' '.join(command[command.index('/usr/bin/env'):])]
              and value['ExecStopPost'] == '', 'Exact durable launch command and no added cleanup')
    p.require(value['ControlGroup'] in ('', p.GROUP_PREFIX + spec['unit'] + '.service'), 'Exact service cgroup')
    if owner is not None:
        p.require(value['InvocationID'] == owner['invocation_id'] and value['ExecMainPID'] == str(owner['pid']),
                  'No replaced service/PID')
    if resources:
        for key, wanted in PROPS.items():
            if key in ('RuntimeMaxSec', 'TimeoutStopSec'):
                p.require(p.duration_seconds(value[key.replace('Sec', 'USec')]) == int(wanted), 'Exact service duration')
            else:
                p.require(value[key] == wanted, 'Exact service resource: ' + key)
    return value


def usage(root):
    paths = list(Path(root).rglob('*'))
    p.require(len(paths) <= 64, 'Bounded metadata-only output tree')
    size = 0
    for path in paths:
        p.require(not path.is_symlink() and (path.is_file() or path.is_dir()), 'Ordinary owned output only')
        info = path.stat(); size += max(info.st_size, info.st_blocks * 512)
        if path.is_file():
            p.require(info.st_size <= FILE_CAP and info.st_nlink in (1, 2), 'Original per-file cap')
            if info.st_nlink == 2:
                sibling = (path.with_name(path.name[1:-4]) if path.name.startswith('.') and path.name.endswith('.tmp')
                           else path.with_name('.' + path.name + '.tmp'))
                other = sibling.stat()
                p.require((info.st_dev, info.st_ino) == (other.st_dev, other.st_ino), 'Only the exact atomic report link pair')
    p.require(size <= OUTPUT_CAP, 'Finite complete output cap')
    fs = os.statvfs(ROOT)
    p.require(fs.f_bavail * fs.f_frsize >= FLOOR, 'Original root storage floor')
    return size


def require_diagnostic_success(result):
    p.require(type(result) is dict and result.get('schema') == 'lfm3_nki05_transport_diagnostic_v1'
              and result.get('complete') is True and result.get('passed') is True and 'error' not in result,
              'Incomplete/failed diagnostic is a failed worker, never success')
    p.require(result.get('legacy_full3716_passed') is result.get('full_model_qualified') is False
              and result.get('device_behavior_proven') is result.get('new_speedup_claim') is False,
              'Diagnostic does not qualify old gate/device/model')
    p.require(type(result.get('checks')) is list and len(result['checks']) == 32
              and all(row.get('passed') is True and row.get('inputs_unchanged') is True
                      and row.get('observer_coverage') is True for row in result['checks']), 'All 32 typed checks required')
    p.require(result.get('bf16_encoding_count') == 65536 and result.get('exact_fixture_hashes_verified') is True
              and result.get('arithmetic_control') is result.get('round_to_nearest_even_cast_control') is True
              and result.get('negative_controls') == diagnostic.negative_controls(), 'All original diagnostic controls required')


def require_characterization(result):
    """Completed causal characterization retains the original scientific failure."""
    p.require(type(result) is dict and result.get('schema') == 'lfm3_nki05_writeback_diagnostic_v2'
              and result.get('complete') is True and result.get('passed') is False and 'error' not in result
              and result.get('original_suite_passed') is False and result.get('candidate_suite_passed') is True
              and result.get('original_failure_preserved') is result.get('restored_original_reproduced') is True
              and result.get('writeback_precision_explanation_supported') is True,
              'Complete distinct original/candidate/restored characterization, never original success')
    p.require(all(result.get(key) is False for key in ('legacy_candidate_changed', 'legacy_full3716_passed',
              'device_behavior_proven', 'full_model_qualified', 'new_speedup_claim', 'installed_sdk_source_modified')),
              'No legacy/device/model promotion or installed SDK change')
    before, candidate, restored = [result[name]['suite'] for name in ('original','candidate','restored_original')]
    require_diagnostic_success(candidate)
    p.require(before['complete'] is before['passed'] is restored['complete'] is restored['passed'] is False
              and len(before['checks']) == len(restored['checks']) == 1
              and before['checks'] == restored['checks'], 'Original failed case retained exactly after restoration')
    check = before['checks'][0]
    p.require(check['name'] == 'exact_v5_first_case' and check['operation'] == 'identity'
              and check['passed'] is check['inputs_unchanged'] is False and check['observer_coverage'] is True
              and check['first_boundary_divergence'] is None
              and check['returned_difference']['mismatched_words'] == 0
              and check['input_sha256'] == list(diagnostic.OLD_HASHES[:3])
              and check['actual_sha256'] == check['expected_sha256'] == diagnostic.OLD_HASHES[0]
              and candidate['checks'][0]['input_sha256'] == check['input_sha256'],
              'Exact original fixture: identity output preserved while host input failure remains')
    for name, count, substitutions in (('original',4,0),('candidate',132,40),('restored_original',4,0)):
        probe = result[name]['probe']
        p.require(probe['installed'] is probe['restored'] is True and probe['installed_sdk_source_modified'] is False
                  and probe['case_errors'] == [] and len(probe['extraction_calls']) == count
                  and probe['substitutions'] == substitutions
                  and sum(event['substitution'] is True for event in probe['extraction_calls']) == substitutions,
                  'Complete actual extraction and restoration evidence: '+name)
    changed = [event for event in result['candidate']['probe']['extraction_calls'] if event['substitution']]
    p.require(all(event['writeback_call'] is True and event['requested_precise'] == 'omitted'
                  and event['effective_precise'] is True and event['tensor_dtype'] == event['storage_dtype'] == event['extracted_dtype'] == 'bfloat16'
                  and event['storage_matches_host_before'] is True
                  and event['precision_controls']['true']['difference']['changed_words'] == 0 for event in changed),
              'Only exact BF16 input writebacks substituted with preserving true-precision controls')
    return dict(characterization_complete=True, original_passed=False, candidate_passed=True,
                original_failure_preserved=True, restored_original_reproduced=True,
                movement_cases=32, bf16_encoding_count=65536, extraction_calls=132,
                precise_bf16_writeback_substitutions=40, actual_tool_exit_verified=False,
                nonzero_worker_exit_required=True, full_model_qualified=False, new_speedup_claim=False)


def execute_diagnostic(admission_ref, owner, root, admit):
    """Save full characterization and a distinct status, then retain real failure."""
    completed = False
    status_ref = None
    try:
        admit()
        result = diagnostic.diagnose(admit)
        once(root / 'diagnostic-result.json', result)
        status = require_characterization(result)
        admit()
        once(root / 'characterization-status.json', dict(status,
             schema='lfm3_nki05_writeback_characterization_status_v1', owner=owner, admission=admission_ref,
             diagnostic=reference(root / 'diagnostic-result.json'), self_exit_claimed=False))
        status_ref = reference(root / 'characterization-status.json')
        completed = True
        # No successful worker-result is published: the original implementation
        # still fails. The unchanged outer observes and preserves actual failure.
        raise RuntimeError('Characterization complete; original implementation failed; nonzero exit retained')
    except BaseException as error:
        report_failure(root / 'worker-failure.json', error, owner=owner, admission=admission_ref,
                       characterization_complete=completed, original_passed=False if completed else None,
                       characterization=status_ref)
        raise


def worker(admission_ref):
    p.require(sys.dont_write_bytecode and __debug__ and sys.executable == PYTHON, 'Pinned worker interpreter with -B')
    a = history.serial.sources.load(admission_ref)
    spec = read_bundle(a['spec'], a['review']); root = Path(spec['output_root'])
    p.require(admission_ref['path'] == str(root / 'admission.json') and a['schema'] == 'lfm3_nki05_serial_admission_v1'
              and a['lock_identity'] == LOCK and a['runtime_admitted'] is a['predecessors_checked'] is True,
              'Actual outer-produced admission')
    p.require(all(os.environ.get(key) == value for key, value in environment(root, a['attempt_id']).items()),
              'Pinned clean native environment')
    p.require(os.sched_getaffinity(0) == {6, 7} and os.getpriority(os.PRIO_PROCESS, 0) == 19, 'CPU6-7/nice19')
    for key, cap in ((resource.RLIMIT_AS, 12 * 2**30), (resource.RLIMIT_CPU, 1800), (resource.RLIMIT_FSIZE, FILE_CAP)):
        p.require(resource.getrlimit(key) == (cap, cap), 'Actual hard/soft worker limit')
    initial = unpacked(a['initial_descriptor'])
    p.require(initial['passed'] is True and initial['owner'] == a['outer_owner'] and initial['lock'] == LOCK
              and initial['retained_fd'] == a['retained_fd'], 'Original initial descriptor proof')
    manifest = p.read(a['observed_drains']['path'], a['observed_drains']['sha256'])
    proof = join_proof(manifest, [p.read(root / row['filename'], row['sha256']) for row in manifest['chunks']])
    p.require(proof['outer_owner'] == a['outer_owner'] and proof['scope']['catalog'] == spec['serial_catalog']
              and proof['scope']['terminal'] == spec['terminal_handoff']
              and proof['observed']['complete'] is proof['observed']['physical_checks_performed'] is True
              and proof['observed']['lock_identity'] == LOCK, 'Complete actual predecessor proof bound to this outer')
    deadline = time.monotonic() + 60
    while not (root / 'worker-owner.json').exists():
        descriptor_proof(a['outer_owner'], a['retained_fd'])
        p.require(time.monotonic() < deadline, 'Real outer owner ACK deadline'); time.sleep(.05)
    owner = p.read(root / 'worker-owner.json')
    process = p.proc(os.getpid())
    p.require(process == {key: owner[key] for key in ('pid', 'start_ticks', 'ppid', 'argv', 'cgroup')}
              and process['argv'] == worker_argv(admission_ref), 'Outer acknowledged actual worker')

    def admit():
        descriptor_proof(a['outer_owner'], a['retained_fd']); p.clear_window()
        check_latest(a['latest_handoff_identity'])
        p.require(p.proc(os.getpid()) == process, 'Actual worker identity remains')
        service_identity(spec, admission_ref, a['attempt_id'], properties(spec['unit']), owner)
        p.active_heavy((spec['unit'],)); usage(root); verify_sources(spec)
        p.require({path: p.sha(path) for path in spec['canonical_sha256']} == spec['canonical_sha256'],
                  'Root-bound canonical snapshots unchanged')
        return True

    return execute_diagnostic(admission_ref, owner, root, admit)


class OwnedService:
    def __init__(self, spec, ref, admission):
        self.spec, self.ref, self.a = spec, ref, admission
        self.attempted, self.owner = False, None

    def launch(self):
        command = service_command(self.spec, self.ref, self.a['attempt_id'])
        once(Path(self.spec['output_root']) / 'launch-intent.json', dict(command=command, admission=self.ref))
        self.attempted = True
        p.command(command, timeout=30)

    def sample(self, *, resources=True):
        return service_identity(self.spec, self.ref, self.a['attempt_id'], properties(self.spec['unit']),
                                self.owner, resources=resources)

    def acknowledge(self, guard):
        deadline = time.monotonic() + 60
        while self.owner is None:
            guard(); value = self.sample(); pid = int(value['MainPID'])
            actual = p.proc(pid) if pid else None
            if actual is not None:
                p.require(actual['argv'] == worker_argv(self.ref)
                          and actual['cgroup'] == '0::' + p.GROUP_PREFIX + self.spec['unit'] + '.service',
                          'Observed real worker command/cgroup')
                self.owner = dict(actual, unit=self.spec['unit'], invocation_id=value['InvocationID'])
                once(Path(self.spec['output_root']) / 'worker-owner.json', self.owner)
            else:
                p.require(not p.terminal(value) and time.monotonic() < deadline, 'Actual worker ACK required')
                time.sleep(.05)
        return self.owner

    def cleanup(self):
        if not self.attempted: return
        intent = p.read(Path(self.spec['output_root']) / 'launch-intent.json')
        p.require(intent == dict(command=service_command(self.spec, self.ref, self.a['attempt_id']), admission=self.ref),
                  'Cleanup binds durable actual launch attempt')
        value = properties(self.spec['unit'])
        if value['LoadState'] == 'not-found': return
        service_identity(self.spec, self.ref, self.a['attempt_id'], value, self.owner, resources=False)
        p.command(['systemctl', '--user', 'stop', self.spec['unit']], timeout=30)

    def drained(self):
        if not self.attempted:
            return dict(physically_drained=True, no_service_launch_attempted=True)
        samples = []
        for _ in range(2):
            value = properties(self.spec['unit'])
            if value['LoadState'] == 'loaded':
                service_identity(self.spec, self.ref, self.a['attempt_id'], value, self.owner, resources=False)
                p.require(p.terminal(value), 'Owned service physically terminal before unlock')
            else:
                p.require(value['LoadState'] == 'not-found' and not value['InvocationID']
                          and value['MainPID'] == value['ControlPID'] == value['ExecMainPID'] == '0', 'Collected exact service')
            if self.owner is not None: p.gone(self.owner)
            samples.append(dict(properties=value, group=p.empty_group(p.GROUP_PREFIX + self.spec['unit'] + '.service')))
        p.require(samples[0] == samples[1], 'Stable complete owned drain before unlock')
        return dict(physically_drained=True, owner=self.owner, samples=samples, self_exit_claimed=False)


def drain_and_release(owned, fd, pending, await_retry, record):
    proof = (dict(physically_drained=True, no_service_launch_attempted=True) if owned is None else
             retain_until_drained(owned.cleanup, owned.drained, pending, await_retry))
    try: record(proof)
    except BaseException as secondary:
        print(json.dumps(dict(stage='cleanup_receipt', error=repr(secondary)[:1024])), file=sys.stderr, flush=True)
    p.require(proof['physically_drained'] is True, 'No unlock before actual complete drain')
    fcntl.flock(fd, fcntl.LOCK_UN); os.close(fd)
    return proof


def outer(spec_ref, review_ref):
    p.require(sys.dont_write_bytecode and __debug__, 'Root outer must use -B with assertions')
    os.sched_setaffinity(0, {6, 7}); os.setpriority(os.PRIO_PROCESS, 0, 19)
    for key, cap in ((resource.RLIMIT_AS, 12 * 2**30), (resource.RLIMIT_CPU, 1800), (resource.RLIMIT_FSIZE, FILE_CAP)):
        resource.setrlimit(key, (cap, cap))
    p.user_bus(); spec = read_bundle(spec_ref, review_ref); root = Path(spec['output_root'])
    p.require(not root.exists(), 'Fresh output tree')
    fd, owner, initial = acquire()
    owned, primary, metadata, frozen_latest = None, None, None, None
    started = time.monotonic(); handlers = {}; retry = [True]

    def lease():
        descriptor_proof(owner, fd); p.clear_window()
        p.require(time.monotonic() - started < 1800, 'Bounded whole work window')

    def guard():
        lease()
        if frozen_latest is not None: check_latest(frozen_latest)
        p.active_heavy((spec['unit'],) if owned is not None and owned.attempted else ())
        p.require({path: p.sha(path) for path in spec['canonical_sha256']} == spec['canonical_sha256'],
                  'Canonical snapshots unchanged')
        if root.exists(): usage(root)

    def interrupted(signum, frame):
        raise InterruptedError('LFM serial work interrupted: ' + str(signum))

    try:
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            handlers[sig] = signal.signal(sig, interrupted)
        guard(); metadata = Metadata(lease)
        scope = history.resolve(spec['serial_catalog'], spec['terminal_handoff'], metadata.read, guard)
        frozen_latest = history.serial.latest_handoff()
        p.require(frozen_latest['path'] == scope['freshness']['latest_handoff']['path']
                  and frozen_latest['identity'] == metadata.observed[frozen_latest['path']]['identity'],
                  'Exact freshly authenticated latest terminal identity')
        drains = history.collect_drain(scope, guard)
        metadata.unchanged(); verify_sources(spec); guard()
        p.require(properties(spec['unit'])['LoadState'] == 'not-found', 'Unused actual service namespace')
        fs = os.statvfs(ROOT); p.require(fs.f_bavail * fs.f_frsize >= FLOOR + OUTPUT_CAP, 'Original floor plus full reserve')
        root.mkdir(mode=0o700); (root / 'tmp').mkdir(mode=0o700)
        save_proof(root, dict(outer_owner=owner, scope=scope, observed=drains, metadata=metadata.observed))
        attempt = os.urandom(16).hex()
        admission = dict(schema='lfm3_nki05_serial_admission_v1', spec=spec_ref, review=review_ref,
                         outer_owner=owner, retained_fd=fd, lock_identity=LOCK, initial_descriptor=packed(initial),
                         attempt_id=attempt, predecessors_checked=True, runtime_admitted=True,
                         latest_handoff_identity=frozen_latest,
                         observed_drains=reference(root / 'predecessor-drain.json'),
                         full_model_qualified=False, new_speedup_claim=False)
        once(root / 'admission.json', admission); admission_ref = reference(root / 'admission.json')
        owned = OwnedService(spec, admission_ref, admission)
        guard(); metadata.unchanged(); owned.launch(); worker_owner = owned.acknowledge(guard)
        print(json.dumps(dict(stage='worker_acknowledged', owner=owner, worker=worker_owner,
                              admission=admission_ref, self_exit_claimed=False)), flush=True)
        while True:
            guard(); metadata.unchanged(); value = owned.sample()
            if p.terminal(value): break
            time.sleep(.5)
        drain = owned.drained()
        success = value['Result'] == 'success' and value['ExecMainCode'] == '1' and value['ExecMainStatus'] == '0'
        once(root / 'service-exit.json', dict(owner=worker_owner, properties=value, observed_by_outer=owner,
                                             successful=success, drain=drain, self_exit_claimed=False))
        p.require(success, 'Actual worker service failed')
        result = p.read(root / 'diagnostic-result.json'); require_diagnostic_success(result)
        worker_result = p.read(root / 'worker-result.json')
        p.require(worker_result['complete'] is worker_result['passed'] is True
                  and worker_result['owner'] == worker_owner and worker_result['admission'] == admission_ref
                  and worker_result['diagnostic'] == reference(root / 'diagnostic-result.json'), 'Exact successful worker report')
        guard(); metadata.unchanged()
        once(root / 'outer-result.json', dict(complete=True, passed=True, owner=owner,
             service_exit=reference(root / 'service-exit.json'), worker_result=reference(root / 'worker-result.json'),
             self_exit_claimed=False, full_model_qualified=False, new_speedup_claim=False))
        return reference(root / 'outer-result.json')
    except BaseException as error:
        primary = error
        if root.exists(): report_failure(root / 'outer-failure.json', error, outer_owner=owner)
        raise
    finally:
        def retry_requested(signum, frame): retry[0] = True
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGUSR1):
            if sig not in handlers: handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, retry_requested)
        def await_retry():
            while not retry[0]: time.sleep(.5)
            retry[0] = False
        def pending(value):
            if root.exists(): once(root / 'cleanup-pending.json', dict(value, owner=owner, original_failure=repr(primary)))
        def record(proof):
            if root.exists(): once(root / 'cleanup-drained.json', proof)
        drain_and_release(owned, fd, pending, await_retry, record)
        for sig, handler in handlers.items(): signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('outer', 'worker'), required=True)
    for key in ('spec', 'review', 'admission'):
        parser.add_argument('--' + key); parser.add_argument('--' + key + '-sha256')
    args = parser.parse_args()
    refs = {key: dict(path=getattr(args, key), sha256=getattr(args, key + '_sha256'))
            for key in ('spec', 'review', 'admission')}
    if args.mode == 'outer': print(json.dumps(outer(refs['spec'], refs['review'])), flush=True)
    else: worker(refs['admission'])


if __name__ == '__main__':
    main()
