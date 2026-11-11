"""Full original CPU successor of frozen transport-v4 retained-lock outer.

Only the fixed scientific leaf, namespace, NVMe metadata placement and original
full-pilot budgets differ. No native/history/process activity on import.
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
import lfm3_nki05_full_cpu_leaf_v2 as science
import smollm3_record_bytecode_v1 as bytecode

ROOT = Path(__file__).resolve().parent
SELF = str(Path(__file__).resolve())
PYTHON = diagnostic.PREFIX + '/bin/python'
LOCK = dict(device=66306, inode=8294421)
FILE_CAP = 256 * 2**10
OUTPUT_CAP = 2**30
NVME_FLOOR = 30 * 2**30
MAX_FILES = 2048
FLOOR = 28 * 2**30
AUTHOR = '/root/lfm_after_writeback_design_20261111'
PROPS = dict(MemoryMax=str(10 * 2**30), MemoryHigh=str(8 * 2**30), MemorySwapMax='0', TasksMax='32',
             AllowedCPUs='6-7', CPUAffinity='6-7', Nice='19', KillMode='control-group',
             RemainAfterExit='yes', SendSIGKILL='yes', WorkingDirectory=str(ROOT),
             LimitFSIZE=str(FILE_CAP), LimitAS=str(12 * 2**30), LimitCPU='30000',
             RuntimeMaxSec='30000', TimeoutStopSec='15')
ORIGINAL_HELPERS = {
    str(Path(p.__file__).resolve()): '81e7d7dd913c6b85f0add7b946692d3270f80cc8f2fe980ecad543251657f185',
    str(Path(descriptor.__file__).resolve()): 'f3faa10356e16dce6b689298ceebf11441f3821ce391f54896cbffed1c82b003',
}
SPEC_KEYS = {'schema', 'root_finalized', 'runner', 'diagnostic', 'source_sha256', 'serial_catalog',
             'terminal_handoff', 'output_root', 'unit', 'resources', 'lock_identity', 'canonical_sha256',
             'root_floor_bytes', 'output_cap_bytes', 'nvme_floor_bytes', 'max_files', 'cpu_protocol',
             'model_qualification_claim', 'automatic_successor_launch'}


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
    p.require(type(spec) is dict and set(spec) == SPEC_KEYS and spec['schema'] == 'lfm3_nki05_full_cpu_spec_v1'
              and spec['root_finalized'] is True, 'Concrete root-finalized LFM diagnostic spec')
    p.require(spec['runner'] == SELF and spec['diagnostic'] == str(Path(science.__file__).resolve()),
              'Exact two execution entrypoints')
    p.require(spec['resources'] == PROPS and spec['lock_identity'] == LOCK, 'Original lock and requested resource limits')
    p.require(spec['root_floor_bytes'] == FLOOR and spec['output_cap_bytes'] == OUTPUT_CAP
              and spec['nvme_floor_bytes'] == NVME_FLOOR and spec['max_files'] == MAX_FILES
              and spec['cpu_protocol'] == science.protocol_manifest(),
              'Original root floor and finite metadata-only output')
    p.require(spec['model_qualification_claim'] is spec['automatic_successor_launch'] is False,
              'No promotion or automatic next model gate')
    root = Path(spec['output_root'])
    p.require(root.parent == Path('/mnt/nvme') and re.fullmatch(r'lfm3-nki05-full-cpu-[a-z0-9-]{1,80}', root.name),
              'Fresh finite LFM output namespace')
    p.require(re.fullmatch(r'trn1-lfm3-nki05-full-cpu-[a-z0-9-]{1,80}', spec['unit']), 'Fresh bounded service name')
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
    for path in (SELF, str(Path(diagnostic.__file__).resolve()), str(Path(history.__file__).resolve()), str(Path(science.__file__).resolve())):
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
    p.require(review['schema'] == 'lfm3_nki05_full_cpu_source_review_v1' and review['reviewer'] == '/root'
              and review['source_author'] == AUTHOR and review['source_only'] is review['passed'] is True
              and review['blockers'] == [] and review['spec'] == spec_ref
              and review['source_sha256'] == spec['source_sha256'] and review['runtime_admitted'] is False,
              'Root independent review of exact concrete spec and sources')
    verify_sources(spec)
    return spec


def pycache_prefix(root):
    prefix = Path(root) / 'tmp' / 'empty-pycache'
    p.require(prefix.is_absolute() and not prefix.exists() and not prefix.is_symlink(),
              'Original bytecode isolation prefix must remain absent and not a symlink')
    return prefix


def bytecode_guard(root):
    # Original predicate checks startup environment, sys prefix and both -B flags.
    # This destination is never created or exempted from original output limits.
    bytecode.isolated(pycache_prefix(root))


def environment(root, attempt):
    return dict(PATH=diagnostic.PREFIX + '/bin:/usr/bin:/bin', HOME='/home/ubuntu',
                XDG_RUNTIME_DIR='/run/user/1000', DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1000/bus',
                PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONOPTIMIZE='0',
                PYTHONPYCACHEPREFIX=str(pycache_prefix(root)),
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
    p.require(re.fullmatch(r'trn1-lfm3-nki05-full-cpu-[a-z0-9-]{1,80}', unit), 'Exact LFM service namespace')
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
    p.require(len(paths) <= MAX_FILES, 'Original full CPU finite file count')
    size = 0
    for path in paths:
        try:
            info = path.lstat()
        except FileNotFoundError:
            p.require(path.name.startswith('.') and path.name.endswith('.tmp'),
                      'Only our atomic temporary report may disappear during observation')
            continue
        p.require(stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode), 'Ordinary owned output only')
        size += max(info.st_size, info.st_blocks * 512)
        if stat.S_ISREG(info.st_mode):
            p.require(info.st_size <= FILE_CAP and info.st_nlink in (1, 2), 'Original per-file cap')
            if info.st_nlink == 2:
                sibling = (path.with_name(path.name[1:-4]) if path.name.startswith('.') and path.name.endswith('.tmp')
                           else path.with_name('.' + path.name + '.tmp'))
                try:
                    other = sibling.lstat()
                except FileNotFoundError:
                    # The original once() unlinks only its temporary hard link.
                    # A final report is never allowed to disappear or rebound.
                    p.require(not path.name.startswith('.') and sibling.name.startswith('.')
                              and sibling.name.endswith('.tmp'), 'Only completed atomic temporary unlink')
                    after = path.lstat()
                    p.require((after.st_dev,after.st_ino)==(info.st_dev,info.st_ino)
                              and after.st_nlink==1 and stat.S_ISREG(after.st_mode), 'Same final inode after atomic unlink')
                else:
                    p.require(stat.S_ISREG(other.st_mode) and (info.st_dev, info.st_ino) == (other.st_dev, other.st_ino),
                              'Only the exact atomic report link pair')
    p.require(size <= OUTPUT_CAP, 'Finite complete output cap')
    fs = os.statvfs(ROOT)
    p.require(fs.f_bavail * fs.f_frsize >= FLOOR, 'Original root storage floor')
    nvme = Path('/mnt/nvme'); fs = os.statvfs(nvme)
    p.require(nvme.is_mount() and nvme.stat().st_dev != ROOT.stat().st_dev
              and fs.f_bavail * fs.f_frsize >= NVME_FLOOR, 'Current NVMe storage floor and actual mount')
    return size


def require_diagnostic_success(result):
    science.audit_full_success(sys.modules[__name__], result)


def execute_diagnostic(admission_ref, owner, root, admit):
    return science.execute_full_cpu(sys.modules[__name__], admission_ref, owner, root, admit)


def worker(admission_ref):
    p.require(sys.dont_write_bytecode and __debug__ and sys.executable == PYTHON, 'Pinned worker interpreter with -B')
    a = history.serial.sources.load(admission_ref)
    spec = read_bundle(a['spec'], a['review']); root = Path(spec['output_root'])
    p.require(admission_ref['path'] == str(root / 'admission.json') and a['schema'] == 'lfm3_nki05_full_cpu_admission_v1'
              and a['lock_identity'] == LOCK and a['runtime_admitted'] is a['predecessors_checked'] is True,
              'Actual outer-produced admission')
    p.require(all(os.environ.get(key) == value for key, value in environment(root, a['attempt_id']).items()),
              'Pinned clean native environment')
    p.require(os.sched_getaffinity(0) == {6, 7} and os.getpriority(os.PRIO_PROCESS, 0) == 19, 'CPU6-7/nice19')
    for key, cap in ((resource.RLIMIT_AS, 12 * 2**30), (resource.RLIMIT_CPU, 30000), (resource.RLIMIT_FSIZE, FILE_CAP)):
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
    for key, cap in ((resource.RLIMIT_AS, 12 * 2**30), (resource.RLIMIT_CPU, 30000), (resource.RLIMIT_FSIZE, FILE_CAP)):
        resource.setrlimit(key, (cap, cap))
    p.user_bus(); spec = read_bundle(spec_ref, review_ref); root = Path(spec['output_root'])
    p.require(not root.exists(), 'Fresh output tree')
    fd, owner, initial = acquire()
    owned, primary, metadata, frozen_latest = None, None, None, None
    started = time.monotonic(); handlers = {}; retry = [True]

    def lease():
        descriptor_proof(owner, fd); p.clear_window()
        p.require(time.monotonic() - started < 30000, 'Original full CPU whole work window')

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
        fs = os.statvfs(ROOT); p.require(fs.f_bavail * fs.f_frsize >= FLOOR, 'Original root floor')
        nvme = Path('/mnt/nvme'); fs = os.statvfs(nvme)
        p.require(nvme.is_mount() and nvme.stat().st_dev != ROOT.stat().st_dev
                  and fs.f_bavail * fs.f_frsize >= NVME_FLOOR + OUTPUT_CAP, 'Current NVMe floor plus original full output reserve')
        root.mkdir(mode=0o700); (root / 'tmp').mkdir(mode=0o700)
        save_proof(root, dict(outer_owner=owner, scope=scope, observed=drains, metadata=metadata.observed))
        attempt = os.urandom(16).hex()
        admission = dict(schema='lfm3_nki05_full_cpu_admission_v1', spec=spec_ref, review=review_ref,
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
    parser.add_argument('--mode', choices=('outer', 'worker', 'batch'), required=True)
    for key in ('spec', 'review', 'admission'):
        parser.add_argument('--' + key); parser.add_argument('--' + key + '-sha256')
    parser.add_argument('--stage', choices=science.STAGES)
    parser.add_argument('--phase', choices=science.PHASES)
    parser.add_argument('--begin', type=int); parser.add_argument('--end', type=int)
    args = parser.parse_args()
    refs = {key: dict(path=getattr(args, key), sha256=getattr(args, key + '_sha256'))
            for key in ('spec', 'review', 'admission')}
    if args.mode == 'outer': print(json.dumps(outer(refs['spec'], refs['review'])), flush=True)
    elif args.mode == 'worker': worker(refs['admission'])
    else: science.run_batch(sys.modules[__name__], refs['admission'], args.stage, args.phase, args.begin, args.end)


if __name__ == '__main__':
    main()
