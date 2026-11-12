"""Concrete future serial Spark CPU diagnostic worker; no admission on import.

The external outer must retain the original lock, validate/drain the complete
current history, install this exact service and acknowledge its actual owner.
This worker independently checks those facts and authenticates real providers.
It never acquires/releases the lock, downloads weights, or publishes results.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import stat
import sys
import time

import spark26_native_port_diagnostic_contract_v2 as contract

ROOT, Q, PYTHON = contract.ROOT, contract.Q, contract.PYTHON
SELF = str(ROOT / 'spark26_native_serial_worker_v2.py')
GIB, MIB = 2**30, 2**20
PROPS = dict(MemoryMax=str(16*GIB), MemoryHigh=str(14*GIB), MemorySwapMax='0',
    TasksMax='256', AllowedCPUs='6-7', CPUAffinity='6-7', Nice='19',
    KillMode='control-group', RemainAfterExit='yes', SendSIGKILL='yes',
    WorkingDirectory=str(ROOT), LimitFSIZE=str(8*MIB), LimitAS=str(24*GIB),
    RuntimeMaxSec='10800', TimeoutStopSec='30')
BUDGET = dict(root_floor_bytes=28*GIB, nvme_floor_bytes=30*GIB,
    qwen_reserved_bytes=2*GIB, available_memory_floor_bytes=2*GIB,
    output_total_bytes=48*MIB, single_file_bytes=8*MIB, max_output_entries=64,
    event_row_bytes=3072, event_total_bytes=16*MIB, log_bytes=8*MIB,
    receipt_total_bytes=16*MIB, temporary_and_directory_bytes=4*MIB,
    max_runtime_authentication_read_bytes=16*GIB, total_wall_seconds=10800)
COUNTS = dict(qkv_projection=216, cache_step=2136, cache_host_reorder=2108, cache_physical_order=384)
HELPERS = {
    str(ROOT / 'minicpm5_layer0_execution_protocol_v4.py'):
        '81e7d7dd913c6b85f0add7b946692d3270f80cc8f2fe980ecad543251657f185',
    str(Q / 'descriptor_serial_lock_proof_v1.py'):
        'f3faa10356e16dce6b689298ceebf11441f3821ce391f54896cbffed1c82b003',
}
require = contract.require


def encode(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                       separators=(',', ':')) + '\n').encode()


def metadata_bytes(reference):
    require(type(reference) is dict and set(reference) == {'path', 'sha256'}, 'Exact hash-bound metadata reference')
    path = Path(reference['path'])
    require(path.is_absolute() and path.resolve() == path and not any(p.is_symlink() for p in (path, *path.parents))
            and path.suffix in ('.py', '.json', '.md', '.txt', '') and '.so' not in path.name,
            'Canonical source/metadata only')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_size <= 8*MIB, 'Bounded source/metadata input')
        with os.fdopen(os.dup(fd), 'rb') as stream:
            raw = stream.read(8*MIB + 1)
        after = os.fstat(fd)
        require(all(getattr(before, key) == getattr(after, key) for key in
                    ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns'))
                and len(raw) == before.st_size and hashlib.sha256(raw).hexdigest() == reference['sha256'],
                'Stable complete frozen source/metadata bytes')
        return raw
    finally:
        os.close(fd)


def read(reference):
    return json.loads(metadata_bytes(reference))


def once(path, value):
    data = encode(value)
    require(len(data) <= 8*MIB, 'Finite immutable worker receipt')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    return {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest()}


def environment(root):
    return dict(PATH='/home/ubuntu/venvs/vllm_neuron/bin:/usr/bin:/bin', HOME='/home/ubuntu',
        XDG_RUNTIME_DIR='/run/user/1000', DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1000/bus',
        PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONOPTIMIZE='0',
        PYTHONPYCACHEPREFIX=str(Path(root)/'tmp'/'absent-pycache'),
        TMPDIR=str(Path(root)/'tmp'), HF_HOME=str(Path(root)/'tmp'/'hf'),
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
        NEURON_RT_VISIBLE_CORES='', NXD_CPU_MODE='1', PJRT_DEVICE='CPU',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')


def argv(admission_ref):
    return [PYTHON, '-S', '-B', SELF, '--admission', admission_ref['path'],
            '--admission-sha256', admission_ref['sha256']]


def service_command(spec, admission_ref):
    return ['systemd-run', '--user', '--quiet', '--unit='+spec['unit'], '--service-type=exec',
        *[a for key, value in PROPS.items() for a in ('-p', key+'='+value)],
        '-p', 'StandardOutput=append:'+spec['output_root']+'/runtime.log', '-p', 'StandardError=inherit',
        '/usr/bin/env', '-i', *[key+'='+value for key, value in sorted(environment(spec['output_root']).items())],
        *argv(admission_ref)]


def validate_spec(spec):
    require(set(spec) == {'schema', 'diagnostic_plan', 'checkpoint_root', 'output_root', 'unit', 'properties',
                         'budget', 'source_sha256', 'native_manifest', 'serial_catalog', 'terminal_handoff',
                         'lock_identity', 'native_python', 'automatic_successor'}, 'Exact Spark worker spec')
    require(spec['schema'] == 'spark26_native_serial_worker_spec_v2'
            and spec['properties'] == PROPS and spec['budget'] == BUDGET
            and spec['lock_identity'] == contract.LOCK_IDENTITY and spec['native_python'] == PYTHON
            and spec['automatic_successor'] is False, 'Fixed resource/original serial and interpreter contract')
    contract.validate_plan(spec['diagnostic_plan'])
    root = Path(spec['output_root']); checkpoint = Path(spec['checkpoint_root'])
    require(root.parent == Path('/mnt/nvme') and re.fullmatch('spark26-native-diagnostic-[a-z0-9-]{1,80}', root.name)
            and checkpoint.is_absolute() and checkpoint != root and root not in checkpoint.parents
            and checkpoint not in root.parents, 'Separate original checkpoint and new finite output namespace')
    require(re.fullmatch('trn1-spark26-native-diagnostic-[a-z0-9-]{1,80}', spec['unit']), 'New exact Spark service')
    require(all(spec['source_sha256'].get(path) == wanted for path, wanted in
                {**HELPERS, **contract.NATIVE_SOURCES}.items()), 'Original proven helper and scientific sources')


def service_identity(spec, admission_ref, value, owner):
    import minicpm5_layer0_execution_protocol_v4 as p
    require(value['LoadState'] == 'loaded' and value['Id'] == spec['unit']+'.service'
            and p.invocation(value['InvocationID']) and value['InvocationID'] == owner['invocation_id']
            and value['ExecMainPID'] == value['MainPID'] == str(owner['process']['pid'])
            and value['ActiveState'] == 'active' and value['SubState'] == 'running'
            and value['ControlGroup'] == p.GROUP_PREFIX+spec['unit']+'.service', 'Actual acknowledged live service')
    command = service_command(spec, admission_ref)
    require(re.findall(r'argv\[\]=(.*?)\s*;', value['ExecStart']) == [' '.join(command[command.index('/usr/bin/env'):])]
            and value['Environment'] == value['ExecStopPost'] == '', 'Exact installed command and clean environment')
    for key, expected in PROPS.items():
        actual = value[key.replace('Sec', 'USec')] if key.endswith('Sec') else value[key]
        require((p.duration_seconds(actual) == int(expected)) if key.endswith('Sec') else actual == expected,
                'Actual enforced resource: '+key)


class WorkerContext:
    """Enforce real process/FD/service/history facts, independently of outer declarations."""
    metadata_bytes = staticmethod(metadata_bytes)

    def __init__(self, admission_ref):
        require(__debug__ and sys.version_info[:2] == (3, 10) and sys.executable == PYTHON
                and sys.flags.no_site and sys.dont_write_bytecode, 'Exact Python 3.10 -S -B worker')
        self.reference = copy.deepcopy(admission_ref)
        self.admission = read(admission_ref)
        a = self.admission
        self.spec = read(a['spec']); validate_spec(self.spec)
        review = read(a['review'])
        require(review['schema'] == 'spark26_native_serial_worker_root_review_v2' and review['reviewer'] == '/root'
                and review['spec'] == a['spec'] and review['source_sha256'] == self.spec['source_sha256']
                and review['passed'] is review['source_only'] is True and review['runtime_admitted'] is False
                and review['blockers'] == [], 'Root independent review of exact source/budget/manifest')
        self.root = Path(self.spec['output_root'])
        require(a['schema'] == 'spark26_native_serial_worker_admission_v2'
                and a['lock_identity'] == contract.LOCK_IDENTITY and a['runtime_admitted'] is True
                and admission_ref['path'] == str(self.root/'admission.json'), 'Concrete external serial admission')
        for path, wanted in self.spec['source_sha256'].items():
            metadata_bytes({'path': path, 'sha256': wanted})
        for name in (SELF, str(ROOT/'spark26_native_provider_v2.py'),
                     str(ROOT/'spark26_native_port_diagnostic_v3.py')):
            require(name in self.spec['source_sha256'], 'Actual worker/provider/core source must be reviewed')
        prefix = environment(self.root)['PYTHONPYCACHEPREFIX']
        require(sys.pycache_prefix == prefix and not os.path.lexists(prefix)
                and all(os.environ.get(key) == value for key, value in environment(self.root).items()),
                'Fresh absent bytecode cache and original clean offline CPU environment')
        require(os.sched_getaffinity(0) == {6, 7} and os.getpriority(os.PRIO_PROCESS, 0) == 19,
                'Actual fixed CPU affinity and nice')
        require(resource.getrlimit(resource.RLIMIT_FSIZE) == (8*MIB, 8*MIB)
                and resource.getrlimit(resource.RLIMIT_AS) == (24*GIB, 24*GIB), 'Actual hard/soft worker limits')
        import minicpm5_layer0_execution_protocol_v4 as p
        from research import descriptor_serial_lock_proof_v1 as descriptor
        import lfm3_nki05_serial_history_v1 as history
        self.p, self.descriptor, self.history = p, descriptor, history
        self.process = p.proc(os.getpid())
        self.deadline = time.monotonic()+BUDGET['total_wall_seconds']
        self.last_full_observation = 0.
        self.native_authenticated = False
        require(self.process['argv'] == argv(admission_ref), 'Real exact worker argv')
        deadline = time.monotonic()+60
        while not (self.root/'worker-owner.json').exists():
            self.descriptor_check()
            require(time.monotonic() < deadline, 'Independent actual outer owner acknowledgement required')
            time.sleep(.05)
        # This ACK is necessarily produced after admission/service creation;
        # never create a circular admission-hash <-> owner-hash requirement.
        raw_owner = p.text(self.root/'worker-owner.json')
        self.owner_ref = {'path': str(self.root/'worker-owner.json'), 'sha256': hashlib.sha256(raw_owner).hexdigest()}
        self.owner = read(self.owner_ref)
        require(self.owner['process'] == self.process and self.owner['admission'] == admission_ref
                and self.owner['outer_owner'] == a['outer_owner'], 'Real separately acknowledged worker')
        self.check(full=True)
        # Read only complete historical metadata; no narrower fabricated history list.
        # Retain historical envelope/alias transports and their original caps;
        # this new worker's 8 MiB output receipt cap cannot shrink old history.
        scope = history.resolve(self.spec['serial_catalog'], self.spec['terminal_handoff'],
                                history.serial.sources.load, self.descriptor_check)
        previous = read(a['predecessor_drain'])
        require(previous['outer_owner'] == a['outer_owner'] and previous['scope'] == scope
                and previous['observed']['complete'] is previous['observed']['physical_checks_performed'] is True,
                'Root outer observed exact complete historical scope before launch')
        # Independently repeat historical PID/cgroup observations while our own service owns the lease.
        self.current_history_observation = history.collect_drain(scope, self.descriptor_check)

    def descriptor_check(self):
        a = self.admission
        return self.descriptor.prove(a['outer_owner'], contract.LOCK_IDENTITY, a['retained_fd'],
            proc=self.p.proc, open_binary=lambda path: open(path, 'rb'), lstat=os.lstat, stat=os.stat,
            readlink=os.readlink)

    def check(self, full=False):
        self.descriptor_check()
        require(time.monotonic() < self.deadline and self.p.proc(os.getpid()) == self.process,
                'Same live process under original retained outer lease and finite deadline')
        # All heavy systemd/storage scans are paced, not the per-call retained-FD checks.
        if full or time.monotonic()-self.last_full_observation >= 1:
            self.p.clear_window()
            props = self.p.parse_properties(self.p.command(['systemctl', '--user', 'show', '--all', self.spec['unit'],
                '--property='+self.p.KEYS+',LimitAS,LimitFSIZE']))
            service_identity(self.spec, self.reference, props, self.owner)
            self.p.active_heavy((self.spec['unit'],))
            require(self.history.serial.latest_handoff() == self.admission['latest_handoff_identity'],
                    'Root-frozen terminal handoff must remain unchanged during this serial attempt')
            require(read(self.owner_ref) == self.owner, 'Independent owner acknowledgement remains immutable')
            require(shutil.disk_usage(ROOT).free >= BUDGET['root_floor_bytes']
                    and shutil.disk_usage('/mnt/nvme').free >= BUDGET['nvme_floor_bytes']+BUDGET['qwen_reserved_bytes'],
                    'Original storage floors and reserved Qwen budget retained')
            with Path('/proc/meminfo').open() as stream:
                info = stream.read(65537)
            require(len(info) <= 65536, 'Bounded actual memory observation')
            rows = [line.split() for line in info.splitlines() if line.startswith('MemAvailable:')]
            require(len(rows) == 1 and rows[0][2] == 'kB'
                    and int(rows[0][1])*1024 >= BUDGET['available_memory_floor_bytes'],
                    'Actual host available-memory floor')
            self.output_usage()
            if self.native_authenticated:
                self.runtime.check_identities()
                self.runtime.native_mappings()
            require(all(os.environ.get(key) == value for key, value in environment(self.root).items()),
                    'Stable CPU/offline/SDK environment during every monitored phase')
            self.last_full_observation = time.monotonic()
        return True

    def output_usage(self):
        paths = [self.root, *self.root.rglob('*')]
        require(len(paths) <= BUDGET['max_output_entries'], 'Finite actual worker output entries')
        total = 0
        categories = dict(events=0, logs=0, receipts=0, temporary=0)
        for path in paths:
            s = path.lstat()
            require(stat.S_ISREG(s.st_mode) or stat.S_ISDIR(s.st_mode), 'No output symlinks or devices')
            if stat.S_ISREG(s.st_mode):
                require(s.st_nlink == 1 and s.st_size <= BUDGET['single_file_bytes'], 'Finite singleton output')
                category = ('temporary' if self.root/'tmp' in path.parents else
                            'events' if path.name in {name+'.jsonl' for name in COUNTS} else
                            'logs' if path.name == 'runtime.log' else 'receipts' if path.suffix == '.json' else None)
                require(category is not None, 'No undeclared runtime output/model/compile artifacts')
            else:
                category = 'temporary'
            categories[category] += s.st_blocks*512
            total += s.st_blocks*512
        require(total <= BUDGET['output_total_bytes'], 'Finite actual allocated output bytes')
        for category, cap in [('events', 'event_total_bytes'), ('logs', 'log_bytes'),
                              ('receipts', 'receipt_total_bytes'), ('temporary', 'temporary_and_directory_bytes')]:
            require(categories[category] <= BUDGET[cap], 'Actual category output budget: '+category)

    def authority(self):
        self.check()
        require(self.native_authenticated, 'Full concrete native authentication precedes science authority')
        return dict(schema='spark26_native_port_diagnostic_authority_v1', model=contract.original.MODEL_ID,
            revision=contract.original.REVISION, plan_sha256=contract.digest(self.spec['diagnostic_plan']),
            checkpoint_root=self.spec['checkpoint_root'], native_python=PYTHON,
            lock_identity=contract.LOCK_IDENTITY, native_sources_authenticated=True,
            native_libraries_authenticated=True, actual_owned_serial_worker=True,
            source_sha256=copy.deepcopy(self.spec['source_sha256']))


class EventSink:
    """Four finite files; exclusive files, exact counts and bounded JSON rows."""
    def __init__(self, root):
        self.root, self.counts, self.total = Path(root), dict.fromkeys(COUNTS, 0), 0
        self.streams = {}

    def emit(self, kind, row):
        require(kind in COUNTS and self.counts[kind] < COUNTS[kind], 'Only declared diagnostic event counts')
        data = encode({'event': kind, 'row': row})
        require(len(data) <= BUDGET['event_row_bytes'] and self.total+len(data) <= BUDGET['event_total_bytes'],
                'Finite event row and total byte budget')
        if kind not in self.streams:
            path = self.root/(kind+'.jsonl')
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
            self.streams[kind] = os.fdopen(fd, 'wb')
        stream = self.streams[kind]
        require(stream.tell()+len(data) <= BUDGET['single_file_bytes'], 'Finite per-category event file')
        stream.write(data); stream.flush()
        self.total += len(data); self.counts[kind] += 1

    def close(self, complete=False):
        for stream in self.streams.values():
            stream.flush(); os.fsync(stream.fileno()); stream.close()
        if complete:
            require(self.counts == COUNTS, 'All original and added diagnostic records emitted')
        return {'counts': dict(self.counts), 'bytes': self.total, 'complete': complete}


def worker(admission_ref):
    context = WorkerContext(admission_ref)
    from spark26_native_provider_v2 import Runtime, SITE, EDITABLE, native_providers
    # -S prevents editable .pth execution before authentication. These exact
    # reviewed package roots are installed only after all dependency bytes hash.
    runtime = Runtime(context, read(context.spec['native_manifest']))
    context.runtime = runtime
    native_auth = runtime.authenticate()
    sys.path.extend([str(SITE), str(EDITABLE)])
    providers = native_providers(runtime)
    context.native_authenticated = True
    once(context.root/'native-authentication.json', native_auth)
    import spark26_native_port_diagnostic_v3 as diagnostic
    sink = EventSink(context.root)
    complete = False
    try:
        result = diagnostic.run(context.spec['diagnostic_plan'], context.spec['checkpoint_root'],
                                context.authority, lambda: providers, sink.emit)
        runtime.check_unchanged(); context.check(full=True)
        complete = True
    finally:
        event_receipt = sink.close(complete)
        once(context.root/'event-receipt.json', event_receipt)
    return once(context.root/'result.json', dict(result=result, admission=admission_ref,
                native_authentication=native_auth, event_receipt=event_receipt,
                worker_exit_or_outer_drain_proven=False))


if __name__ == '__main__':
    sys.modules['spark26_native_serial_worker_v2'] = sys.modules[__name__]
    parser = argparse.ArgumentParser()
    parser.add_argument('--admission', required=True)
    parser.add_argument('--admission-sha256', required=True)
    args = parser.parse_args()
    worker({'path': args.admission, 'sha256': args.admission_sha256})
