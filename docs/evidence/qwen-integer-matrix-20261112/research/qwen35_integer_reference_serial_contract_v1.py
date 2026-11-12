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

Q = Path(__file__).resolve().parent
ROOT = Q.parent
for directory in (Q, ROOT):
    if str(directory) not in sys.path: sys.path.insert(0, str(directory))
import qwen35_integer_reference_protocol_v1 as protocol
from qwen35_sampler_runtime_contract_v12 import LOCK_IDENTITY, pin_map, reference, require
from qwen35_sampler_runtime_io_v12 import observed_stat, PublicationRace, raw, read, ref, transient

MIB, GIB = 2**20, 2**30
PYTHON = protocol.RESOURCES['interpreter']
SITE = Path('/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages')
SDK_ROOTS = ('torch', 'torch_xla', 'torch_neuronx', 'neuronxcc', 'numpy', 'ml_dtypes', 'libneuronxla')
LAYOUTS = tuple(row[0] for row in protocol.LAYOUTS)
MANIFEST = {'path': '/home/ubuntu/experiments/trn1-latest-models-20261031/research/qwen35-integer-reference-primitive-source-manifest-20261112-v1.json', 'sha256': '790d937ada82db907fac735b43ceadaced1da33803d3816fddd696b3d64dd1e1', 'bytes': 50215}
LIMITS = dict(protocol.RESOURCES, cpus=[6, 7], nice=19, ack_seconds=180, outer_seconds=9000,
    log_bytes=8*MIB, max_entries=65536, memory_available_floor_bytes=2*GIB,
    poll_seconds=.5, metadata_bytes=8*MIB, complete_history_bytes=64*MIB)
PROPERTIES = dict(KillMode='control-group', RemainAfterExit='yes', SendSIGKILL='yes',
    WorkingDirectory=str(ROOT), TimeoutStopSec='15', LimitFSIZE=str(GIB), TasksMax='256',
    MemoryMax=str(6*GIB), MemoryHigh=str(5*GIB), MemorySwapMax='0', RuntimeMaxSec='7200')


def names(spec):
    require(type(spec['run_name']) is str and re.fullmatch('[a-z0-9][a-z0-9-]{0,39}', spec['run_name']), 'Fresh bounded run name')
    return {'coordinator': 'trn1-qwen35-sampler-' + spec['run_name'] + '-integer-reference-slot'}


def validate(spec, review, spec_ref):
    fields = {'schema', 'run_name', 'output_root', 'history_overlay', 'current_serial_catalog', 'source_sha256',
        'input_extension', 'running_handoff_path', 'lfm_r2_scope', 'source_delivery', 'primitive_manifest',
        'workload', 'limits', 'lock_identity', 'native_python', 'automatic_successor'}
    require(type(spec) is dict and set(spec) == fields and spec['schema'] == 'qwen35_integer_reference_serial_spec_v1', 'Exact slot spec')
    selected = names(spec)
    require(spec['output_root'] == '/mnt/nvme/neuron-work/qwen35-integer-reference-slot-' + spec['run_name'], 'Fresh slot-only output namespace')
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
    require(delivery['schema'] == 'qwen35_integer_reference_serial_source_delivery_v1'
        and delivery['primitive_manifest'] == MANIFEST and delivery['source_only'] is True
        and delivery['workload'] == protocol.workload() and delivery['limits'] == LIMITS, 'Reviewed slot source inventory')
    pins = spec['source_sha256']
    for mapping in (delivery['static_local_source_sha256'], delivery['sdk_source_sha256']):
        require(mapping and all(pins.get(path) == sha for path, sha in mapping.items()), 'All integration and SDK pins retained')
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
    protocol.verify_sources()
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
    root = Path(root); entries = 0; totals = [0, 0]; by_layout = {name: [0, 0] for name in LAYOUTS}
    pending = [root] if root.exists() else []
    while pending:
        path = pending.pop(); relative = path.relative_to(root)
        owner = relative.parts[0] if relative.parts and relative.parts[0] in LAYOUTS else None
        volatile = owner is not None and len(relative.parts) > 1 and relative.parts[1] in ('tmp', 'cache', 'artifacts')
        try: info = observed_stat(path)
        except FileNotFoundError:
            if volatile or transient(path): raise PublicationRace('Declared compiler temporary disappeared')
            raise
        entries += 1; require(entries <= LIMITS['max_entries'], 'Bounded output entries')
        require(not stat.S_ISLNK(info.st_mode) and info.st_dev == root.stat().st_dev, 'Ordinary same-volume output only')
        allocated = info.st_blocks*512; logical = 0
        if stat.S_ISDIR(info.st_mode):
            try: pending.extend(path.iterdir())
            except FileNotFoundError:
                if volatile: raise PublicationRace('Declared compiler temporary directory disappeared')
                raise
        else:
            require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, 'Singleton regular output only')
            cap = (128*MIB if path.name == 'model.pt' else 16*MIB if path.name in ('report.json', 'report.json.tmp')
                else LIMITS['log_bytes'] if path.suffix == '.log' else GIB)
            require(info.st_size <= cap, 'Per-file output cap: ' + str(path)); logical = info.st_size
        totals[0] += allocated; totals[1] += logical
        if owner:
            by_layout[owner][0] += allocated; by_layout[owner][1] += logical
    require(max(totals) <= 8*GIB, 'Total output including metadata, cache and logs <=8 GiB')
    require(all(max(row) <= GIB for row in by_layout.values()), 'Each layout complete output <=1 GiB')
    return dict(entries=entries, allocated_bytes=totals[0], logical_bytes=totals[1],
        layouts={key: dict(allocated_bytes=row[0], logical_bytes=row[1]) for key, row in by_layout.items()})


def scan_output(root):
    for attempt in range(8):
        try: return scan_once(root)
        except PublicationRace:
            if attempt == 7: raise ValueError('Declared atomic/compiler output did not settle')
            time.sleep(.01)


def resources(spec, reserve=True):
    root = Path(spec['output_root']); usage = scan_output(root); result = {}
    require(Path('/mnt/nvme').is_mount() and ROOT.stat().st_dev != Path('/mnt/nvme').stat().st_dev
        and root.parent.stat().st_dev == Path('/mnt/nvme').stat().st_dev, 'Actual distinct NVMe output volume')
    for path, floor in ((ROOT, 28*GIB), (Path('/mnt/nvme'), 30*GIB)):
        fs = os.statvfs(path); available = fs.f_bavail * fs.f_frsize
        extra = 2*GIB + max(0, 8*GIB - usage['allocated_bytes']) if reserve and path == Path('/mnt/nvme') else 0
        require(available >= floor + extra, 'Original floor plus remaining output reservation and headroom: ' + str(path))
        result[str(path)] = available
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))) * 1024
    require(available >= LIMITS['memory_available_floor_bytes'], 'Original memory available floor')
    current_layout(root)
    return dict(storage=usage, available_disk_bytes=result, available_memory_bytes=available)


def expected_calls():
    for mode in protocol.MODES:
        for sequence in protocol.sequences():
            phase, profile = sequence['phase'], sequence['profile']
            for variant in sequence['order']:
                if mode == 'persistent_alias':
                    yield dict(phase='reset', mode=mode, variant=variant, slots=list(protocol.TIMING_MAPPINGS[0]),
                        detail=dict(phase=phase, profile=profile, round=sequence['round']))
                for step, mapping in enumerate(sequence['mappings']):
                    yield dict(phase=phase, mode=mode, variant=variant, slots=list(mapping),
                        detail=dict(profile=profile, round=sequence['round'], step=step))


def validate_report(report, name, artifacts):
    require(report['schema'] == 'qwen35_integer_reference_primitive_result_v1' and report['case'] == protocol.layout(name)
        and report['reference_implementation'] == protocol.REFERENCE_IMPLEMENTATION
        and report['complete'] is report['passed'] is True and report['source_sha256'] == protocol.FROZEN_SOURCES
        and report['required_resources'] == protocol.RESOURCES and report['timing_scope'] == protocol.TIMING_SCOPE
        and all(report[key] is False for key in ('full_model_qualified', 'model_speedup_claim', 'lfm105_tasks_replaced')),
        'Complete exact primitive report')
    counts = protocol.counts()['per_layout']
    require(report['runtime_counts'] == {key: counts[key+'_calls'] for key in ('correctness','warmup','measured','reset')}
        and len(report['checks']) == counts['total_runtime_calls'], 'Full 5796 explicit calls per layout')
    words = ((0, 0x8000), (0x7fc1, 0x7fc2)) if name != 'recurrent' else ((0, 0x80000000), (0x7fc00001, 0x7fc00002))
    require(report['controls'] == [dict(left_word=a, right_word=b, passed=True) for a,b in words], 'Signed-zero and NaN payload negative controls')
    wanted_graphs = [(mode, variant) for mode in protocol.MODES for variant in protocol.VARIANTS]
    require([(row['mode'], row['variant']) for row in report['graphs']] == wanted_graphs
        and [row['artifact'] for row in report['graphs']] == artifacts and len(artifacts) == 4, 'All four compiled graph artifacts')
    for row in report['graphs']:
        require(row['input_output_alias_indices'] == ([1] if row['mode'] == 'persistent_alias' else [])
            and (type(row['state_name']) is str and bool(row['state_name']) if row['mode'] == 'persistent_alias' else row['state_name'] is None), 'Real MUST_ALIAS state and old-read output contract')
    for actual, expected in zip(report['checks'], expected_calls()):
        require(all(actual[key] == value for key, value in expected.items()) and actual['passed'] is True, 'Every original ordered mapping/profile/mode/variant call')
        keys = {'updated_output', 'updates_unchanged', 'slots_unchanged'} | ({'cache_input_unchanged'} if actual['mode'] == 'standalone' else {'old_read', 'persistent_state'})
        require(set(actual['checks']) == keys, 'All full-byte, input, old-read and actual-state checks present')
        for key, check in actual['checks'].items():
            require(all(check[field] is True for field in ('passed','shape_equal','dtype_equal','bitwise_equal'))
                and check['bytes_compared'] == (32 if key == 'slots_unchanged' else report['case']['tensor_bytes'])
                and re.fullmatch('[a-f0-9]{64}', check['actual_sha256'])
                and check['actual_sha256'] == check['expected_sha256'], 'Complete exact byte-check evidence')
        require(type(actual['elapsed_ns']) is int and actual['elapsed_ns'] > 0 if actual['phase'] == 'measured'
            else actual['elapsed_ns'] is None, 'Only measured calls carry elapsed timings')
    require(set(report['timing']) == set(protocol.MODES), 'Separate exact mode timing summaries')
    for mode in protocol.MODES:
        timing = report['timing'][mode]; rows = timing['paired_rows']
        require(timing == dict(paired_rows=rows, complete=True, **protocol.summarize_pairs(rows)), 'Exact paired timing summary')
        for row in rows:
            for variant in protocol.VARIANTS:
                elapsed = [check['elapsed_ns'] for check in report['checks'] if check['mode'] == mode and check['variant'] == variant
                    and check['phase'] == 'measured' and check['detail']['round'] == row['round']]
                require(row['elapsed_ns'][variant] == elapsed, 'Timing summaries refer to actual checked calls')
    return dict(layout=name, graphs=4, counts=counts, passed=True,
        reference_implementation=dict(protocol.REFERENCE_IMPLEMENTATION), model_speedup_claim=False)
