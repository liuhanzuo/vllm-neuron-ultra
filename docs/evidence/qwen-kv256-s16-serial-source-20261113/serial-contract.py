"""Exact slot experiment contract and bounded output helpers; inert on import."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import time

Q = Path(__file__).resolve().parents[1]
ROOT = Q.parent
for directory in (Q, ROOT):
    if str(directory) not in sys.path: sys.path.insert(0, str(directory))
from . import protocol
from qwen35_sampler_runtime_contract_v12 import LOCK_IDENTITY, pin_map, reference, require
from qwen35_sampler_runtime_io_v12 import observed_stat, PublicationRace, raw, read, ref, transient

MIB, GIB = 2**20, 2**30
PYTHON = protocol.RESOURCES['interpreter']
SITE = Path('/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages')
PACKAGE = Path('/home/ubuntu/src/vllm-neuron-ultra-qwen-writer-20261112')
SDK_ROOTS = ('torch', 'torch_xla', 'torch_neuronx', 'neuronxcc', 'numpy', 'ml_dtypes', 'libneuronxla')
LAYOUTS = tuple(row[0] for row in protocol.LAYOUTS)
MANIFEST = {'path': '/home/ubuntu/experiments/trn1-latest-models-20261031/research/qwen35_kv256_s16_pair_serial_v1/primitive_manifest.json', 'bytes': 3020, 'sha256': '043e69810896e5da17a2d2cc55bbaafd7a28eaa1ff752ad43f5131425d99512d'}
LIMITS = dict(protocol.RESOURCES, cpus=[6, 7], nice=19, ack_seconds=180, outer_seconds=9000,
    log_bytes=8*MIB, max_entries=65536, memory_available_floor_bytes=2*GIB,
    poll_seconds=.5, metadata_bytes=8*MIB, complete_history_bytes=64*MIB)
PROPERTIES = dict(KillMode='control-group', RemainAfterExit='yes', SendSIGKILL='yes',
    WorkingDirectory=str(ROOT), TimeoutStopSec='15', LimitFSIZE=str(GIB), TasksMax='256',
    MemoryMax=str(6*GIB), MemoryHigh=str(5*GIB), MemorySwapMax='0', RuntimeMaxSec='7200')


def names(spec):
    require(type(spec['run_name']) is str and re.fullmatch('[a-z0-9][a-z0-9-]{0,39}', spec['run_name']), 'Fresh bounded run name')
    return {'coordinator': 'trn1-qwen35-sampler-' + spec['run_name'] + '-kv256-s16-pair'}


def validate(spec, review, spec_ref):
    fields = {'schema', 'run_name', 'output_root', 'history_overlay', 'current_serial_catalog', 'source_sha256',
        'input_extension', 'running_handoff_path', 'lfm_r2_scope', 'source_delivery', 'primitive_manifest',
        'workload', 'limits', 'lock_identity', 'native_python', 'automatic_successor'}
    require(type(spec) is dict and set(spec) == fields and spec['schema'] == 'qwen35_kv256_s16_pair_serial_spec_v1', 'Exact slot spec')
    selected = names(spec)
    require(spec['output_root'] == '/mnt/nvme/neuron-work/qwen35-kv256-s16-pair-' + spec['run_name'], 'Fresh slot-only output namespace')
    require(spec['primitive_manifest'] == MANIFEST and spec['workload'] == protocol.workload()
        and spec['limits'] == LIMITS and spec['native_python'] == PYTHON
        and spec['lock_identity'] == LOCK_IDENTITY and spec['automatic_successor'] is False, 'Immutable workload and original resource policy')
    for key in ('history_overlay', 'current_serial_catalog', 'lfm_r2_scope', 'source_delivery'): reference(spec[key])
    target = Path(spec['running_handoff_path'])
    require(target.parent == Q and re.fullmatch('continuation-handoff-[a-z0-9-]+\\.json', target.name), 'Original running handoff namespace')
    pin_map(spec['source_sha256'])
    require(type(spec['input_extension']) is dict, 'Complete explicit metadata identities')
    for path, row in spec['input_extension'].items():
        require(type(row) is dict and set(row) == {'sha256', 'identity'} and type(row['identity']) is dict, 'Exact input identity')
        reference(dict(path=path, sha256=row['sha256']))
    require(type(review) is dict and set(review) == {'schema', 'reviewer', 'passed', 'blockers', 'spec', 'source_sha256', 'launch_authorized'}
        and review['schema'] == 'qwen35_sampler_runtime_source_review_v1' and review['reviewer'] == '/root'
        and review['passed'] is True and review['blockers'] == [] and review['spec'] == spec_ref
        and review['source_sha256'] == spec['source_sha256'] and review['launch_authorized'] is True, 'Original exact root review envelope')
    return selected


def verify_delivery(spec):
    manifest = read({key: MANIFEST[key] for key in ('path', 'sha256')})
    require(len(raw(MANIFEST['path'])) == MANIFEST['bytes'] and manifest['workload'] == protocol.workload(), 'Frozen primitive manifest')
    delivery = read(spec['source_delivery'], 16*MIB)
    require(delivery['schema'] == 'qwen35_kv256_s16_pair_serial_source_delivery_v1'
        and delivery['primitive_manifest'] == MANIFEST and delivery['source_only'] is True
        and delivery['workload'] == protocol.workload() and delivery['limits'] == LIMITS, 'Reviewed slot source inventory')
    pins = spec['source_sha256']
    for mapping in (delivery['static_local_source_sha256'], delivery['sdk_source_sha256']):
        require(mapping and all(pins.get(path) == sha for path, sha in mapping.items()), 'All integration and SDK pins retained')
    from qwen35_kv256_s16_pair_root_helpers_v1 import HELPERS
    require(all(str(path) in delivery['static_local_source_sha256']
        and pins.get(str(path)) == ref(path, 4*MIB)['sha256'] for path in HELPERS),
        'All root admission/ACK/audit/finalizer helpers pre-enrolled before future preparation')
    for row in manifest['source_files']:
        require(pins.get(row['path']) == row['sha256'] and len(raw(row['path'])) == row['bytes']
            and ref(row['path'])['sha256'] == row['sha256'], 'Frozen primitive source unchanged')
    for name in SDK_ROOTS:
        actual = {str(path) for path in (SITE / name).rglob('*.py')}
        expected = {path for path in delivery['sdk_source_sha256'] if Path(path).is_relative_to(SITE / name)}
        require(actual == expected and actual, 'Exact SDK Python source inventory: ' + name)
    for evidence in protocol.REFERENCE_EVIDENCE:
        row = spec['input_extension'].get(evidence['path'])
        require(row is not None and row['sha256'] == evidence['sha256'], 'All original failure and actual select evidence metadata retained')
        read(evidence)
    require(protocol.verify_sources() == {row['path']: row['sha256'] for row in manifest['source_files']},
            'Complete geometry source map matches its independently pinned manifest')
    return delivery


def environment(work):
    work = Path(work)
    return dict(HOME='/home/ubuntu', LANG='C.UTF-8', XDG_RUNTIME_DIR='/run/user/1000',
        DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1000/bus',
        PATH='/home/ubuntu/venvs/vllm_neuron/bin:/opt/aws/neuron/bin:/usr/local/bin:/usr/bin:/bin',
        PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', PYTHONOPTIMIZE='0',
        PYTHONPYCACHEPREFIX=str(bytecode_prefix(work)),
        NEURON_RT_VISIBLE_CORES='0', NEURON_LOGICAL_NC_CONFIG='1',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
        TORCHINDUCTOR_COMPILE_THREADS='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false', TMPDIR=str(work / 'tmp'),
        XDG_CACHE_HOME=str(work / 'cache'), HF_HOME=str(work / 'cache/hf'),
        TORCHINDUCTOR_CACHE_DIR=str(work / 'cache/torch'), NEURON_COMPILE_CACHE_URL=str(work / 'cache/neuron'))


def bytecode_prefix(root): return Path(root) / 'bytecode-never-created'


def check_environment(work):
    require(dict(os.environ) == environment(work), 'Exact clean slot environment')
    check_process_settings(work)


def check_process_settings(work):
    require(sys.dont_write_bytecode and __debug__ and sys.flags.isolated
        and sys.pycache_prefix == str(bytecode_prefix(work)) and not bytecode_prefix(work).exists()
        and not bytecode_prefix(work).is_symlink(), 'Absent isolated bytecode prefix')
    require(sorted(os.sched_getaffinity(0)) == LIMITS['cpus'] and os.getpriority(os.PRIO_PROCESS, 0) == LIMITS['nice'], 'Exactly two assigned CPUs and nice priority')


def update_json(path, value, cap):
    """One known writer; retain complete previous report until fsynced replacement."""
    path = Path(path)
    require(path.name in ('report.json', 'current-layout.json') and path.parent.resolve() == path.parent, 'Only two declared mutable receipts')
    if path.exists():
        info = path.lstat(); require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, 'Ordinary mutable receipt')
    payload = (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    require(len(payload) <= cap, 'Complete report/clock exceeds cap; never truncate')
    temporary = path.with_name(path.name + '.tmp')
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try: os.fsync(fd)
    finally: os.close(fd)


def current_layout(root):
    """Read one complete atomic clock version; old version remains conservative."""
    path = Path(root) / 'current-layout.json'
    if not path.exists(): return None
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        info = os.fstat(fd)
        require(stat.S_ISREG(info.st_mode) and info.st_size <= 4096, 'Bounded exact layout clock')
        payload = os.read(fd, 4097)
        require(len(payload) == info.st_size, 'Complete layout clock snapshot')
    finally: os.close(fd)
    value = json.loads(payload)
    require(set(value) == {'name', 'started', 'active'} and value['name'] in LAYOUTS
        and type(value['started']) in (float, int) and type(value['active']) is bool, 'Exact layout clock fields')
    if value['active']: require(0 <= time.monotonic() - value['started'] < 900, 'Per-layout 900 second wall deadline')
    return value


def scan_once(root):
    from . import storage
    return storage.scan_once(root)


def scan_output(root):
    from . import storage
    return storage.scan_output(root)


def resources(spec, reserve=True):
    from . import storage
    return storage.resources(spec, reserve=reserve)



expected_calls = protocol.expected_calls
validate_report = protocol.validate_report
