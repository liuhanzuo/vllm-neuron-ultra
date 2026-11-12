# SPDX-License-Identifier: Apache-2.0
"""Bounded, explicitly invoked native package preflight; imports are lazy.

This is a worker for the existing root serial runner. It never reserves/releases
its lock, installs a package, constructs a model, loads weights, or compiles.
Without --execute-native it only prints the input schema description.
"""
from __future__ import annotations
import argparse
import ast
import copy
import errno
import hashlib
import importlib
import json
import os
from pathlib import Path
import resource
import selectors
import signal
import stat
import sys
import threading
import time

SCHEMA = 'qwen35-package-native-preflight-v5'
CHECKOUT = '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107'
PYTHON = '/home/ubuntu/venvs/vllm_neuron/bin/python'
SITE_PACKAGES = '/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages'
SOURCE_COMMIT = 'e40fd753ba080a9a4640688f4d6beda9985b7099'
SOURCE_DELIVERY_SHA256 = 'e73b5d59d28be857f16cc9349d042f3e0687956bdf61f569c6c1e3ea4a3ff835'
LIMITS = dict(memory_max=12 * 2**30, memory_swap_max=0, pids_max=256,
              cpu_count=8, wall_seconds=600, cpu_seconds=480, file_bytes=16 * 2**20)
NEURON_KWARGS = dict(tp_degree=2, ctx_batch_size=1, batch_size=4,
    max_context_length=4096, seq_len=4096, enable_bucketing=True,
    is_continuous_batching=True, quantized=False, torch_dtype='bfloat16',
    padding_side='right', on_device_sampling_config=None, lora_config=None,
    is_block_kv_layout=False, is_prefix_caching=False, apply_seq_ids_mask=True,
    context_encoding_buckets=[128, 256, 512, 1024, 2048, 4096])
COMPILE_ENV = ('NEURON_CC_FLAGS', 'NEURON_RT_STOCHASTIC_ROUNDING_EN',
    'NEURON_LOGICAL_NC_CONFIG', 'XLA_USE_BF16', 'XLA_DOWNCAST_BF16',
    'XLA_DISABLE_FUNCTIONALIZATION')
NATIVE_ROOTS = ('torch', 'torch_xla', 'torch_neuronx', 'neuronx_distributed',
    'neuronx_distributed_inference', 'transformers', 'vllm')
NUMPY_SVE_SOURCE = '/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/numpy/testing/_private/utils.py'
NUMPY_SVE_SHA256 = 'dc5ad24cc8b439da43383043a27720883433768e4d280e9855f437b8a4524277'
PJRT_HELPER = '/home/ubuntu/venvs/vllm_neuron/bin/libneuronpjrt-path'
PJRT_INITIALIZER = SITE_PACKAGES + '/torch_neuronx/initialization/initializer.py'
PJRT_LIBRARY = SITE_PACKAGES + '/libneuronxla/libneuronpjrt.so'
PJRT_PATH = '/home/ubuntu/venvs/vllm_neuron/bin:/opt/aws/neuron/bin:/usr/local/bin:/usr/bin:/bin'
PJRT_SOURCE_REFS = [{'path': '/home/ubuntu/venvs/vllm_neuron/bin/libneuronpjrt-path', 'bytes': 204, 'sha256': 'ee7c03670b393965ca49f0103e96f462d7fc22a7daa250db3bf3de4b1f198dba'}, {'path': '/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/torch_neuronx/initialization/initializer.py', 'bytes': 13212, 'sha256': '34b4c73980c4513696569f4cf2ef51fe9de5952a9957a9af706e2457fcb90116'}, {'path': '/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/torch_neuronx/initialization/__init__.py', 'bytes': 420, 'sha256': '9318cd15d847544734b2d4732156860a5248b78cfd3f2492e2cf93ac5ad243d3'}, {'path': '/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/torch_neuronx/xla.py', 'bytes': 13504, 'sha256': 'cc5f040784603de5817817874bff18e7f79d469ed45d41836d72979e82d97a01'}, {'path': '/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/torch_neuronx/__init__.py', 'bytes': 5792, 'sha256': '1842b2414de6a85a259d04a6786385f67e78513345bd6935be4351171f42d8ca'}, {'path': '/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/torch_neuronx/libtorchneuron.py', 'bytes': 2111, 'sha256': 'da9edf2a6f12539a9ee5bdaffc45f5bc6f7ebdbb4e59e14c432ea093dd500d5e'}, {'path': '/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/torch_neuronx/utils/utils.py', 'bytes': 1988, 'sha256': '834f4ace222bd6c6079432b2c438bf040f5cda2843e8b87e7d727360c5d06f2a'}, {'path': '/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/torch_xla/__init__.py', 'bytes': 9898, 'sha256': 'cbde3e68a24209523c66db254ef3060fa77af3b8d50702fe45bf78c2dbd4e279'}, {'path': '/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/libneuronxla/__init__.py', 'bytes': 1795, 'sha256': 'ce74235a966860162ad075118461d96cc79d91cc88dc61cff3793290a51d7f54'}, {'path': '/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/libneuronxla/analyze_interface.py', 'bytes': 939, 'sha256': '57809f0b08156b05f80fc140d203058df08c23907cd81acaaee6d2a525822296'}, {'path': '/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/libneuronxla/hook.py', 'bytes': 919, 'sha256': 'ddb8b401223aba91b57b284be8e61c113cc0fe2a7d43c10ea44dafda642ae582'}, {'path': '/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/libneuronxla/libncc.py', 'bytes': 8227, 'sha256': '1545066b7fd8e443a8676d757043e99725cde5fdca0f2e82946f80c3588a51a5'}, {'path': '/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/libneuronxla/libneuronpjrt_path.py', 'bytes': 309, 'sha256': 'd2eb6ccf4796e396927243c57843d46f60f0457f87cc5da34a56076bd781c62c'}, {'path': '/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/libneuronxla/libnrt.py', 'bytes': 1597, 'sha256': '9d6f5fe2bfc2e879843bbf7b8bca3fd24cdcc58f06fccdfc927b68465a20ea8d'}, {'path': '/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/libneuronxla/logger.py', 'bytes': 1837, 'sha256': 'a0ddb03e9a323c3ed4aeb9ab6ae1d45467c0ffa0f68588aa09824297600a16a1'}, {'path': '/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/libneuronxla/neuron_cc_cache.py', 'bytes': 21272, 'sha256': 'cdfa0d6f5c2481da7d4331b36e14985363b6b29c05d257c27c866b8a1e841f90'}, {'path': '/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/libneuronxla/neuron_cc_wrapper.py', 'bytes': 17522, 'sha256': '33218bc87806d18277a47a2a7c5d9e7ddc78314d00f27c86816d2ec3731c961f'}, {'path': '/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/libneuronxla/profiler.py', 'bytes': 808, 'sha256': '732ec459ff488b7092d5761e5163e4b4e3d63b79151b34f3828495717403ed26'}, {'path': '/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/libneuronxla/version.py', 'bytes': 2550, 'sha256': '0e1ec41dac19b19e245c63704d5f20e1cfed265e03b9dab853008f02cc739ee8'}]



def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_small(path, cap=4 * 2**20):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path, 'Ordinary absolute metadata/source path required')
    require(path.suffix not in ('.so', '.pt', '.bin', '.safetensors', '.neff', '.whl', '.tar', '.gz'), 'Payload/binary read forbidden')
    before = path.stat()
    require(stat.S_ISREG(before.st_mode) and before.st_size <= cap, 'Metadata/source size/type bound')
    raw = path.read_bytes()
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), 'Input changed while reading')
    require(len(raw) == before.st_size, 'Incomplete source read')
    return raw


def checked(ref):
    require(type(ref) is dict and set(ref) == {'path', 'bytes', 'sha256'}, 'Exact metadata reference required')
    raw = read_small(ref['path'])
    require(type(ref['bytes']) is int and len(raw) == ref['bytes'] and sha(raw) == ref['sha256'], 'Reference bytes changed')
    return raw


def once(path, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    require(len(raw) <= LIMITS['file_bytes'], 'Result exceeds output limit')
    with Path(path).open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def schema():
    return dict(schema=SCHEMA, fields=['schema', 'bundle', 'work', 'binding'],
        bundle='Exact path/bytes/sha256 of the source-preparation bundle; source-only bundle is not admission.',
        work='Fresh absolute ordinary output directory under an existing parent; root owns cleanup.',
        binding=dict(owner='Exact pid/start_ticks of existing serial root lock owner',
            lock='Existing original lock path/device/inode',
            history='Nonempty list of exact current complete history/physical-drain metadata references, supplied by root',
            cgroup='Exact existing cgroup-v2 relative path of this worker and its spawn child'),
        python=PYTHON, limits=LIMITS, no_model_or_artifact_admission=True)


def validate_spec(spec):
    require(type(spec) is dict and set(spec) == {'schema', 'bundle', 'work', 'binding'}, 'Exact worker input fields required')
    require(spec['schema'] == SCHEMA, 'Wrong worker schema')
    work = Path(spec['work'])
    require(work.is_absolute() and '..' not in work.parts and work.parent.resolve() == work.parent, 'Ordinary absolute work parent required')
    b = spec['binding']
    require(set(b) == {'owner', 'lock', 'history', 'cgroup'}, 'Existing root-run binding fields required')
    require(set(b['owner']) == {'pid', 'start_ticks'} and all(type(b['owner'][k]) is int and b['owner'][k] > 0 for k in b['owner']), 'Exact live root identity required')
    require(set(b['lock']) == {'path', 'device', 'inode'}, 'Exact original lock identity required')
    require(Path(b['lock']['path']).is_absolute() and type(b['lock']['device']) is int and type(b['lock']['inode']) is int, 'Invalid original lock identity')
    require(type(b['history']) is list and bool(b['history']), 'Root must bind complete current history; no fixed predecessor floor')
    require(type(b['cgroup']) is str and b['cgroup'].startswith('/') and '..' not in Path(b['cgroup']).parts, 'Cgroup-v2 path required')
    return spec


def process_start(pid):
    raw = Path('/proc', str(pid), 'stat').read_text()
    return int(raw.rsplit(')', 1)[1].split()[19])


def check_binding(spec):
    """Observe the existing owner; never flock, acquire, release or mutate it."""
    b = spec['binding']; owner = b['owner']; lock = b['lock']
    require(process_start(owner['pid']) == owner['start_ticks'], 'Original serial owner is no longer live')
    path = Path(lock['path']); info = path.stat()
    require(path.resolve() == path and stat.S_ISREG(info.st_mode), 'Ordinary original lock required')
    require((info.st_dev, info.st_ino) == (lock['device'], lock['inode']), 'Original serial lock was replaced')
    key = f'{os.major(info.st_dev):02x}:{os.minor(info.st_dev):02x}:{info.st_ino}'
    found = False
    for row in Path('/proc/locks').read_text().splitlines():
        fields = row.split()
        if len(fields) == 8 and fields[1:4] == ['FLOCK', 'ADVISORY', 'WRITE']:
            found |= fields[4] == str(owner['pid']) and fields[5] == key and fields[6:] == ['0', 'EOF']
    require(found, 'Existing root must still own the original kernel FLOCK')
    history = [{'path': x['path'], 'sha256': sha(checked(x))} for x in b['history']]
    current = Path('/proc/self/cgroup').read_text().strip().splitlines()
    require(current == ['0::' + b['cgroup']], 'Worker escaped its root-assigned cgroup')
    root = Path('/sys/fs/cgroup') / b['cgroup'].lstrip('/')
    observed = {}
    for key, filename in [('memory_max', 'memory.max'), ('memory_swap_max', 'memory.swap.max'), ('pids_max', 'pids.max')]:
        value = (root / filename).read_text().strip()
        require(value.isdecimal() and int(value) <= LIMITS[key], 'Missing/loose cgroup ceiling: ' + key)
        observed[key] = int(value)
    cpus = sorted(os.sched_getaffinity(0))
    require(0 < len(cpus) <= LIMITS['cpu_count'], 'Root must bound CPU affinity before launch')
    return dict(owner=owner, lock=lock, history=history, cgroup=b['cgroup'], limits=observed, cpus=cpus)


def validate_bundle(bundle):
    require(bundle['schema'] == SCHEMA + '-source-bundle' and bundle['source_commit'] == SOURCE_COMMIT, 'Wrong source bundle')
    require(bundle['checkout'] == CHECKOUT and bundle['python'] == PYTHON and bundle['site_packages'] == SITE_PACKAGES, 'Wrong source/interpreter root')
    repo = Path(CHECKOUT)
    delivery = repo / 'test/qwen35/source_delivery_v2.json'
    require(sha(read_small(delivery)) == SOURCE_DELIVERY_SHA256, 'Frozen Qwen3.5 v2 snapshot changed')
    for relative, expected in bundle['package_sources'].items():
        require(not Path(relative).is_absolute() and '..' not in Path(relative).parts and relative.endswith('.py'), 'Invalid package source member')
        require(sha(read_small(repo / relative)) == expected, 'Package source changed: ' + relative)
    actual = {str(p.relative_to(repo)) for p in (repo / 'vllm_neuron').rglob('*.py')}
    require(actual == set(bundle['package_sources']), 'Package source inventory changed')
    for ref in bundle['worker_sources']:
        checked(ref)
    for name, expected in bundle['sdk_sources'].items():
        relative = name.split('/', 1)[1]
        require(sha(read_small(Path(SITE_PACKAGES) / relative)) == expected, 'Pinned SDK source changed: ' + name)
    require(bundle['optional_probe_source'] == dict(path=NUMPY_SVE_SOURCE, sha256=NUMPY_SVE_SHA256,
        bytes=Path(NUMPY_SVE_SOURCE).stat().st_size), 'Exact optional NumPy probe source reference')
    require(sha(checked(bundle['optional_probe_source'])) == NUMPY_SVE_SHA256, 'Pinned NumPy optional probe source changed')
    require(bundle['pjrt_helper_sources'] == PJRT_SOURCE_REFS, 'Exact required SDK initialization/helper sources')
    for source in PJRT_SOURCE_REFS: checked(source)
    require(bundle['pjrt_library_identity'] == library_identity(), 'Exact installed PJRT library location and identity')
    checked(bundle['official_config'])
    for reference in bundle['tokenizer']['sources'].values(): checked(reference)
    require(bundle['neuron_kwargs'] == NEURON_KWARGS, 'Original root Neuron settings changed')
    return bundle


def environment(work, inherited):
    """Pure environment helper for root's existing serial service launcher."""
    require(not any(k.startswith('TRN1_') for k in inherited), 'Do not inherit experiment switches')
    out = {k: v for k, v in inherited.items() if not k.startswith(('NEURON_', 'VLLM_', 'PYTHON', 'XLA_', 'HF_', 'TRANSFORMERS_', 'TORCH'))}
    out.update(NEURON_QWEN35='1', NEURON_ON_DEVICE_SAMPLING_DISABLED='1', NEURON_CPU_SAMPLER_FAST='0',
        XLA_DISABLE_FUNCTIONALIZATION='1', VLLM_WORKER_MULTIPROC_METHOD='spawn',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
        TORCHINDUCTOR_COMPILE_THREADS='1', PYTHONDONTWRITEBYTECODE='1',
        HF_HOME=str(Path(work) / 'hf-cache'), XDG_CACHE_HOME=str(Path(work) / 'cache'),
        TMPDIR=str(Path(work) / 'tmp'))
    return out


def check_environment(spec):
    expected = environment(spec['work'], {})
    for key, value in expected.items():
        require(os.environ.get(key) == value, 'Root must set bounded worker environment: ' + key)
    require(not any(k.startswith('TRN1_') or (k.startswith('NEURON_') and 'ALLOW_COMPILE' in k and os.environ[k] != '0') for k in os.environ), 'Compile/experiment switches forbidden')
    require(not os.environ.get('NEURON_COMPILED_ARTIFACTS'), 'Preflight must not receive a compiled artifact path')
    allowed_neuron = {'NEURON_QWEN35', 'NEURON_ON_DEVICE_SAMPLING_DISABLED', 'NEURON_CPU_SAMPLER_FAST'} | set(COMPILE_ENV)
    require(not any(k.startswith('NEURON_') and k not in allowed_neuron for k in os.environ),
            'Unexpected model/runtime switch outside bounded preflight environment')


def set_limits():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (LIMITS['file_bytes'], LIMITS['file_bytes']))
    resource.setrlimit(resource.RLIMIT_CPU, (LIMITS['cpu_seconds'], LIMITS['cpu_seconds']))
    def timeout(signum, frame):
        raise TimeoutError('Bounded package native preflight wall deadline')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(LIMITS['wall_seconds'])


def optional_sve_probe(args, caller):
    """Identify only the pinned x86 NumPy optional SVE probe before exec."""
    if os.uname().machine != 'x86_64' or args != ('lscpu', ['lscpu'], None, None):
        return None
    # Popen.__init__ -> _execute_child (audit) -> run -> exact NumPy function.
    # Walk a small stack because CPython audit callbacks add no Python wrapper.
    for _ in range(8):
        if caller is None: return None
        if (caller.f_code.co_filename == NUMPY_SVE_SOURCE and caller.f_code.co_name == 'check_support_sve'
                and caller.f_globals.get('__name__') == 'numpy.testing._private.utils'
                and caller.f_globals.get('__file__') == NUMPY_SVE_SOURCE):
            function = caller.f_globals.get('check_support_sve')
            if getattr(function, '__code__', None) is not caller.f_code: return None
            require(sha(read_small(NUMPY_SVE_SOURCE)) == NUMPY_SVE_SHA256, 'Optional probe exact source changed')
            return dict(source=dict(path=NUMPY_SVE_SOURCE, sha256=NUMPY_SVE_SHA256),
                function='check_support_sve', machine='x86_64', command='lscpu',
                subprocess_started=False, denied_before_exec=True, exception='FileNotFoundError',
                reason='Optional SVE discovery denied by the existing no-subprocess policy')
        caller = caller.f_back
    return None


def optional_probe_evidence(events):
    """Observe NumPy's own except-OSError fallback; never substitute a result."""
    module = sys.modules.get('numpy.testing._private.utils')
    require(module is not None and getattr(module, '__file__', None) == NUMPY_SVE_SOURCE
        and getattr(module, '_SUPPORTS_SVE', None) is False and len(events) == 1,
        'Exactly one denied optional probe with NumPy original false fallback')
    return dict(events=copy.deepcopy(events), supports_sve=False, original_numpy_fallback_observed=True,
        optional_probe_subprocess_remains_forbidden=True, fabricated_command_output=False)


def library_identity():
    path = Path(PJRT_LIBRARY); value = path.lstat()
    require(path.resolve() == path and stat.S_ISREG(value.st_mode) and value.st_size > 0,
        'Actual installed ordinary PJRT library; never substitute CPU')
    return dict(path=str(path), device=value.st_dev, inode=value.st_ino, bytes=value.st_size,
        mtime_ns=value.st_mtime_ns, ctime_ns=value.st_ctime_ns)


def helper_environment(work):
    return dict(environment(work, dict(HOME='/home/ubuntu', LANG='C.UTF-8', PATH=PJRT_PATH,
        XDG_RUNTIME_DIR='/run/user/1000', DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1000/bus')),
        PYTHONNOUSERSITE='1', PYTHONPYCACHEPREFIX=str(Path(work) / 'pjrt-helper-empty-pycache'))


def helper_process(pid):
    def small(name):
        with Path('/proc', str(pid), name).open('rb') as stream: value = stream.read(65537)
        require(len(value) <= 65536, 'Bounded required-helper process metadata')
        return value
    before = small('stat').decode().rsplit(')', 1)[1].split()
    command = [value.decode() for value in small('cmdline').split(b'\0') if value]
    group = small('cgroup').decode().strip()
    executable = os.readlink(Path('/proc', str(pid), 'exe'))
    after = small('stat').decode().rsplit(')', 1)[1].split()
    require(before[19] == after[19], 'Helper PID identity retained')
    return dict(pid=pid, start_ticks=int(before[19]), ppid=int(before[1]), argv=command, cgroup=group, executable=executable)


class RequiredPjrtHelper:
    """One exact SDK path-location command per fresh parent/spawn process."""
    def __init__(self, spec, bundle, label):
        self.spec, self.bundle, self.label = spec, bundle, label
        self.events = []; self.active = False; self.audit_count = 0; self.thread = None
        self.env = helper_environment(spec['work']); self.original_run = None

    def caller_matches(self, frame):
        if not (frame.f_code.co_filename == PJRT_INITIALIZER and frame.f_code.co_name == '__init__'
                and frame.f_globals.get('__name__') == 'torch_neuronx.initialization.initializer'
                and frame.f_globals.get('__file__') == PJRT_INITIALIZER): return False
        initializer = frame.f_globals.get('Initializer')
        return getattr(getattr(initializer, '__init__', None), '__code__', None) is frame.f_code

    def install(self):
        import subprocess
        require(self.original_run is None, 'Install the fixed helper adapter once')
        self.original_run = subprocess.run
        subprocess.run = self.run

    def run(self, *args, **kwargs):
        import subprocess
        if not self.caller_matches(sys._getframe(1)):
            return self.original_run(*args, **kwargs)  # Existing audit still rejects every other process.
        require(args == (['libneuronpjrt-path'],) and kwargs == {'stdout': subprocess.PIPE}, 'Exact pinned SDK helper invocation')
        require(not self.events and self.audit_count == 0, 'No repeated SDK path locator or retry')
        require(os.environ.get('PATH') == PJRT_PATH and os.environ.get('PJRT_DEVICE') == 'NEURON',
            'Original fixed PATH and actual NEURON initialization')
        for source in PJRT_SOURCE_REFS: checked(source)
        require(read_small(PJRT_HELPER).startswith(b'#!/home/ubuntu/venvs/vllm_neuron/bin/python3\n'), 'Original script interpreter')
        require(library_identity() == self.bundle['pjrt_library_identity'], 'Original library unchanged before helper')
        prefix = Path(self.env['PYTHONPYCACHEPREFIX'])
        require(not prefix.exists() and not prefix.is_symlink(), 'No stale helper Python bytecode')
        check_binding(self.spec)
        return self.capture()

    def permits_audit(self, args, caller):
        # Only the Popen called synchronously inside capture can consume this allowance.
        if not self.active or threading.get_ident() != self.thread or self.audit_count != 0: return False
        if args != (PJRT_HELPER, [PJRT_HELPER], str(Path(self.spec['work']) / 'tmp'), self.env): return False
        for _ in range(8):
            if caller is None: return False
            if caller.f_code is type(self).capture.__code__ and caller.f_locals.get('self') is self:
                self.audit_count += 1; return True
            caller = caller.f_back
        return False

    def capture(self):
        import subprocess
        started = time.monotonic(); owner = None; child = None; selector = selectors.DefaultSelector()
        streams = {'stdout': bytearray(), 'stderr': bytearray()}; failure = None; waited = False
        self.active = True; self.thread = threading.get_ident()
        try:
            child = subprocess.Popen([PJRT_HELPER], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=str(Path(self.spec['work']) / 'tmp'), env=self.env, close_fds=True)
            self.active = False
            require(self.audit_count == 1, 'Actual exact required-helper audit event observed')
            owner = helper_process(child.pid)
            require(owner['ppid'] == os.getpid() and owner['argv'] == ['/home/ubuntu/venvs/vllm_neuron/bin/python3', PJRT_HELPER]
                and owner['executable'] == str(Path(PYTHON).resolve())
                and owner['cgroup'] == '0::' + self.spec['binding']['cgroup'], 'Actual fixed helper argv/parent/inherited cgroup')
            require(os.sched_getaffinity(child.pid) == os.sched_getaffinity(0), 'Helper inherits assigned CPUs')
            for key in streams: selector.register(getattr(child, key), selectors.EVENT_READ, key)
            while selector.get_map() or child.poll() is None:
                require(time.monotonic() - started < 30, 'Required path helper 30 second cap within original worker budget')
                for key, _ in selector.select(.1):
                    block = os.read(key.fileobj.fileno(), 4096)
                    if not block: selector.unregister(key.fileobj); continue
                    streams[key.data].extend(block)
                    require(len(streams[key.data]) <= (4096 if key.data == 'stdout' else 16384), 'Complete helper output bound')
            child.wait(timeout=1); waited = True
            require(child.returncode == 0 and bytes(streams['stdout']) == (PJRT_LIBRARY+'\n').encode()
                and not streams['stderr'], 'Actual helper exit zero and exact installed library stdout')
            require(library_identity() == self.bundle['pjrt_library_identity'], 'Actual returned library unchanged')
            check_binding(self.spec)
            return subprocess.CompletedProcess(['libneuronpjrt-path'], child.returncode, bytes(streams['stdout']), bytes(streams['stderr']))
        except BaseException as error:
            failure = dict(type=type(error).__name__, message=str(error)[:1024]); raise
        finally:
            self.active = False; selector.close()
            if child is not None:
                if child.poll() is None:
                    child.terminate()
                    try: child.wait(timeout=5)
                    except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=5)
                    waited = True
                if not waited: child.wait(timeout=1); waited = True
                for key in streams:
                    if getattr(child, key) is not None: getattr(child, key).close()
            value = dict(label=self.label, owner=owner, parent_pid=os.getpid(), command=[PJRT_HELPER],
                sdk_command=['libneuronpjrt-path'], environment=self.env,
                actual_returncode=None if child is None else child.returncode, popen_wait_observed=waited,
                stdout=bytes(streams['stdout']).decode('utf-8', errors='backslashreplace'),
                stderr=bytes(streams['stderr']).decode('utf-8', errors='backslashreplace'),
                elapsed_seconds=time.monotonic()-started, library=self.bundle['pjrt_library_identity'],
                failure=failure, same_owned_cgroup_required=True, whole_recursive_cgroup_drain_pending=True,
                self_exit_claimed=False, fabricated_command_output=False)
            self.events.append(value)
            once(Path(self.spec['work']) / (self.label+'-pjrt-helper.json'), value)
            print(json.dumps(dict(event='qwen35_required_pjrt_helper_exit', **value)), flush=True)

    def evidence(self):
        require(len(self.events) == 1 and self.events[0]['failure'] is None
            and self.events[0]['actual_returncode'] == 0 and self.events[0]['popen_wait_observed'] is True
            and os.environ.get('PJRT_DEVICE') == 'NEURON' and os.environ.get('NEURON_LIBRARY_PATH') == PJRT_LIBRARY,
            'Required helper actually succeeded and SDK consumed its real result')
        return copy.deepcopy(self.events[0])


def start_fresh_spawn(child, clean_environment):
    """Spawn reruns initialization from the original clean parent input environment."""
    initialized = dict(os.environ)
    try:
        os.environ.clear(); os.environ.update(clean_environment); child.start()
    finally:
        os.environ.clear(); os.environ.update(initialized)


def install_runtime_guard(tokenizer_allowlist=(), optional_probe_events=None, required_helper=None):
    """Reject model payloads, subprocesses and network connects in this worker."""
    original_roots = ("/mnt/nvme/models/Qwen3.5-2B/", "/mnt/nvme/models/Qwen3.5-2B-neuron-text-exp20261031/")
    allowed = {os.path.abspath(os.fsdecode(path)) for path in tokenizer_allowlist}
    readonly = {name for name in allowed if name.startswith(original_roots) or Path(name).name != 'config.json'}
    def guard(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            name = os.path.abspath(os.fsdecode(args[0]))
            resolved = str(Path(name).resolve())
            for candidate in (name, resolved):
                if candidate.startswith(original_roots):
                    require(candidate in allowed, "Only exact pinned tokenizer/config paths are permitted")
            if name in readonly or resolved in readonly:
                flags = args[2] if len(args) > 2 and type(args[2]) is int else 0
                mode = args[1] if len(args) > 1 and isinstance(args[1], str) else ''
                require(not any(character in mode for character in 'wax+')
                        and not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND),
                        'Original/historical/prepared tokenizer files are read-only')
            require(not name.endswith(('.safetensors', '.pt', '.pth', '.neff', '.bin', '.whl')),
                    'Model/compiled/archive payload access is outside native preflight')
            require(not name.startswith('/dev/neuron'), 'Accelerator device access is outside native preflight')
        if event == 'subprocess.Popen' and optional_probe_events is not None:
            evidence = optional_sve_probe(args, sys._getframe(1))
            if evidence is not None:
                require(not optional_probe_events, 'Optional NumPy probe may be rejected only once per fresh process')
                optional_probe_events.append(evidence)
                print(json.dumps(dict(event='qwen35_optional_numpy_probe_denied', **evidence)), flush=True)
                raise FileNotFoundError(errno.ENOENT, 'Optional lscpu probe denied before execution', 'lscpu')
        if event == 'subprocess.Popen' and required_helper is not None and required_helper.permits_audit(args, sys._getframe(1)):
            return
        require(event not in ('subprocess.Popen', 'os.system', 'socket.connect'), 'Subprocess/network forbidden: ' + event)
    sys.addaudithook(guard)


def origins(bundle):
    result = {}
    for name, module in sorted(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if name == 'vllm_neuron' or name.startswith('vllm_neuron.'):
            require(path.is_relative_to(Path(CHECKOUT)) and path.suffix == '.py', 'Foreign package module: ' + name)
            relative = str(path.relative_to(CHECKOUT))
            require(relative in bundle['package_sources'] and sha(read_small(path)) == bundle['package_sources'][relative], 'Unpinned package import: ' + name)
            result[name] = str(path)
        elif name.split('.')[0] in NATIVE_ROOTS:
            require(path.is_relative_to(Path(SITE_PACKAGES)), 'Foreign native Python module: ' + name)
            # Do not hash native binaries. Record real module origins only.
            result[name] = str(path)
    return result


def negative_scope_checks(identity, body):
    cases = [('tp', ('neuron_config', 'tp_degree'), 1), ('context', ('neuron_config', 'max_length'), 2048),
        ('batch', ('neuron_config', 'batch_size'), 1), ('dtype', ('neuron_config', 'torch_dtype'), 'float32'),
        ('quantized', ('neuron_config', 'quantized'), True), ('prefix', ('neuron_config', 'is_prefix_caching'), True),
        ('buckets', ('neuron_config', 'context_encoding_buckets'), [128, 4096]),
        ('model_dimension', ('hidden_size',), 1024), ('type_confusion', ('neuron_config', 'ctx_batch_size'), True)]
    result = []
    for name, keys, value in cases:
        wrong = copy.deepcopy(body); target = wrong
        for key in keys[:-1]: target = target[key]
        target[keys[-1]] = value
        try:
            identity.validate_scope(wrong, wrong['neuron_config'])
        except identity.AdmissionError:
            result.append(name)
        else:
            raise RuntimeError('Invalid package configuration was admitted: ' + name)
    return result


def config_checks(modules, output, bundle):
    identity, modeling, hf, artifacts = (modules[k] for k in ('identity', 'modeling', 'hf_adapter', 'artifacts'))
    # Parent and spawn use the same config-only path: HF records _name_or_path.
    config_dir = output.parent / 'shared-hf-config'
    from vllm_neuron.models.qwen35.prepare_view import compatible_model_config
    original = json.loads(checked(bundle['official_config']))
    transformed = compatible_model_config(original)
    require(transformed == bundle['view_config'] and bundle['view_config_sha256'] == identity.CONFIG_VIEW_SHA256,
            'Official pinned model config/view transformation changed')
    config_raw = json.dumps(transformed, indent=2).encode()
    require(sha(config_raw) == bundle['view_config_sha256'], 'Pinned original text-view transformation changed')
    if config_dir.exists():
        require(read_small(config_dir / 'config.json') == config_raw, 'Shared config-only fixture changed')
    else:
        config_dir.mkdir(exist_ok=False)
        with (config_dir / 'config.json').open('xb') as stream:
            stream.write(config_raw)
    # Real AutoConfig -> pinned hf_adapter hook -> actual model InferenceConfig.
    config = modeling.Qwen35InferenceConfig(modeling.NeuronConfig(**bundle['neuron_kwargs']),
        load_config=hf.load_pretrained_config(str(config_dir)))
    serialized = config.to_json_string(); body = json.loads(serialized)
    require('dtype' not in body and 'torch_dtype' not in body, 'SDK dtype normalization changed')
    identity.canonical_configuration(body)
    identity.validate_scope(body, body['neuron_config'])
    root_path = output / 'root-config'; config.save(root_path)
    restored = modeling.Qwen35InferenceConfig.load(root_path)
    require(identity.canonical_configuration(json.loads(restored.to_json_string())) == identity.canonical_configuration(body), 'Full SDK config save/load changed')
    direct = modeling.Qwen35InferenceConfig.from_json_string(serialized)
    require(identity.canonical_configuration(json.loads(direct.to_json_string())) == identity.canonical_configuration(body), 'Full SDK JSON constructor changed')
    # Real saved-body admission only; this does not call artifact/model/payload admission.
    binding = {'configuration': body, 'preflight_config_only': True}
    config.metadata = {'qwen35_ultra': binding}
    bound_path = output / 'bound-config'; config.save(bound_path)
    artifacts.admit_saved_configuration(bound_path, binding)
    saved = json.loads((bound_path / 'neuron_config.json').read_text())
    mutated = copy.deepcopy(saved); mutated['neuron_config']['max_length'] = 2048
    wrong_path = output / 'wrong-config'; wrong_path.mkdir()
    (wrong_path / 'neuron_config.json').write_text(json.dumps(mutated))
    try: artifacts.admit_saved_configuration(wrong_path, binding)
    except identity.AdmissionError: pass
    else: raise RuntimeError('Saved-body admission accepted changed context')
    negatives = negative_scope_checks(identity, body)
    constructor_negatives = []
    for name, key, value in [('padding', 'padding_side', 'left'), ('quantization', 'quantized', True)]:
        wrong = copy.deepcopy(body); wrong['neuron_config'][key] = value
        try:
            candidate = modeling.Qwen35InferenceConfig.from_json_string(json.dumps(wrong))
            actual = json.loads(candidate.to_json_string())
            identity.validate_scope(actual, actual['neuron_config'])
        except (ValueError, AssertionError, identity.AdmissionError):
            constructor_negatives.append(name)
        else:
            raise RuntimeError('SDK/package path admitted invalid configuration: ' + name)
    return body, dict(configuration_sha256=sha(serialized.encode()),
        complete_configuration_canonical=identity.canonical_configuration(body),
        full_sdk_json_file=str(root_path / "neuron_config.json"),
        normalized_dtype_aliases_removed=True, root_save_load_equal=True, json_constructor_equal=True,
        saved_body_admission=True, changed_saved_body_rejected=True, invalid_scope_rejected=negatives,
        invalid_sdk_constructor_or_scope_rejected=constructor_negatives,
        full_build_identity_called=False, weights_admitted=False)


def native_checks(spec, bundle, label):
    """All native imports and tiny synthetic fixtures occur only here."""
    binding = check_binding(spec); check_environment(spec); set_limits()
    validate_bundle(bundle)
    sys.pycache_prefix = str(Path(spec["work"]) / "fresh-pycache")
    sys.dont_write_bytecode = True
    require(sys.version_info[:2] == (3, 10) and Path(sys.executable).absolute() == Path(PYTHON), 'Original pinned Python3.10 interpreter required')
    sys.path.insert(0, CHECKOUT)
    import importlib.util
    helper_ref = next(row for row in bundle["worker_sources"] if row["path"].endswith("qwen35_package_tokenizer_checks_v3.py"))
    checked(helper_ref)
    helper_spec = importlib.util.spec_from_file_location("qwen35_tokenizer_preflight_v3", helper_ref["path"])
    tokenizer_checks = importlib.util.module_from_spec(helper_spec); helper_spec.loader.exec_module(tokenizer_checks)
    optional_probe_events = []
    required_helper = RequiredPjrtHelper(spec, bundle, label)
    required_helper.install()
    install_runtime_guard(tokenizer_checks.tokenizer_paths(spec["work"]), optional_probe_events, required_helper)
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    import importlib.metadata
    for name, version in bundle['software'].items():
        require(importlib.metadata.version(name) == version, 'Native dependency version changed: ' + name)
    package = importlib.import_module('vllm_neuron.models.qwen35')
    package.register(); package.register()
    modules = {key: importlib.import_module('vllm_neuron.models.qwen35.' + key)
               for key in ('identity', 'modeling', 'registration', 'text_core', 'helpers', 'artifacts')}
    modules['hf_adapter'] = importlib.import_module('neuronx_distributed_inference.utils.hf_adapter')
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from vllm.model_executor.models.registry import ModelRegistry
    from neuronx_distributed_inference.utils.constants import MODEL_TYPES
    registration = modules['registration']; modeling = modules['modeling']
    require(CONFIG_MAPPING['qwen35_neuron_text'] is registration.Qwen35NeuronTextConfig, 'Real Transformers registration mismatch')
    require('Qwen35NeuronTextForCausalLM' in ModelRegistry.get_supported_archs(), 'Real vLLM registration missing')
    require(MODEL_TYPES['qwen35neurontext'] == {'causal-lm': modeling.NeuronQwen35ForCausalLM}, 'Real NxDI registration mismatch')
    require(modules['text_core'].split_last_dim is modules['helpers'].split_last_dim, 'Packaged corrected split was replaced')
    # Synthetic CPU-only BF16 input: 1*3072*2 = 6144 elements (12 KiB).
    source = torch.arange(6144, dtype=torch.float32).remainder(251).to(torch.bfloat16).reshape(1, 3072, 2)
    transposed = source.transpose(1, 2)
    pieces = modules['helpers'].split_last_dim(transposed, [1024, 1024, 1024])
    require(all(torch.equal(piece, transposed[..., n*1024:(n+1)*1024]) for n, piece in enumerate(pieces)), 'Native BF16 split changed coordinates')
    output = Path(spec['work']) / label; output.mkdir()
    body, configuration = config_checks(modules, output, bundle)
    tokenization = tokenizer_checks.run(bundle, spec['work'])
    from vllm.v1.sample.sampler import Sampler
    from vllm_neuron.worker.cpu_sampler import BoundedTopKTopPSampler, maybe_enable_cpu_sampler
    standard = Sampler(); original_operation = standard.topk_topp_sampler
    maybe_enable_cpu_sampler(standard)
    require(standard.topk_topp_sampler is original_operation, 'Standard CPU sampler was changed')
    graph_inputs = dict(sources=modules['identity'].source_identity(), configuration=body,
                        environment={key: os.environ.get(key) for key in COMPILE_ENV})
    os.environ['NEURON_CPU_SAMPLER_FAST'] = '1'
    try:
        fast = Sampler(); reference = fast.topk_topp_sampler
        maybe_enable_cpu_sampler(fast); maybe_enable_cpu_sampler(fast)
        require(isinstance(fast.topk_topp_sampler, BoundedTopKTopPSampler) and fast.topk_topp_sampler.reference is reference,
                'Fast host sampler did not retain the original per-instance reference')
        require(standard.topk_topp_sampler is original_operation, 'Fast opt-in changed another instance')
        fast_inputs = dict(sources=modules['identity'].source_identity(), configuration=body,
                           environment={key: os.environ.get(key) for key in COMPILE_ENV})
        require(modules['identity'].same_identity(graph_inputs, fast_inputs), 'Host sampler flag changed package graph inputs')
    finally:
        os.environ['NEURON_CPU_SAMPLER_FAST'] = '0'
    result = dict(label=label, pid=os.getpid(), ppid=os.getppid(), start_ticks=process_start(os.getpid()),
        binding=binding, origins=origins(bundle), registration_idempotent=True,
        split=dict(cpu=True, dtype='bfloat16', elements=6144, transpose=True, equal_coordinates=True),
        configuration=configuration, tokenization=tokenization, sampler=dict(standard_preserved=True, fast_per_instance_reference=True,
        same_source_configuration_compiler_environment=True, full_artifact_identity_not_computed=True,
        sampling_tokens_or_performance_not_tested=True), native_imports_executed=True,
        model_constructed=False, weights_read=False, device_graph_executed=False, compiled=False,
        optional_cpu_probe=optional_probe_evidence(optional_probe_events), required_pjrt_helper=required_helper.evidence())
    check_binding(spec)
    return result


def spawn_worker(spec, bundle):
    """Actual multiprocessing spawn target; module top-level remains stdlib-only."""
    result = native_checks(spec, bundle, 'spawn')
    once(Path(spec['work']) / 'spawn-result.json', result)


def execute(spec):
    validate_spec(spec)
    bundle = validate_bundle(json.loads(checked(spec['bundle'])))
    check_binding(spec); check_environment(spec)
    require(not any(name == root or name.startswith(root + '.') for root in NATIVE_ROOTS for name in sys.modules), 'Native preflight requires a fresh interpreter')
    work = Path(spec['work']); work.mkdir(exist_ok=False)
    for name in ('tmp', 'cache', 'hf-cache'): (work / name).mkdir()
    clean_environment = dict(os.environ)
    parent = native_checks(spec, bundle, 'parent')
    import multiprocessing
    context = multiprocessing.get_context('spawn')
    child = context.Process(target=spawn_worker, args=(spec, bundle), name='qwen35-package-preflight-spawn')
    start_fresh_spawn(child, clean_environment)
    try:
        deadline = time.monotonic() + LIMITS['wall_seconds']
        while child.is_alive() and time.monotonic() < deadline:
            child.join(timeout=1)
        require(not child.is_alive() and child.exitcode == 0, 'Native spawn preflight failed/timed out')
    finally:
        if child.is_alive():
            child.terminate(); child.join(timeout=10)
            if child.is_alive(): child.kill(); child.join(timeout=10)
    spawned = json.loads(read_small(work / 'spawn-result.json'))
    require(spawned['pid'] == child.pid and spawned['pid'] != parent['pid'] and spawned['ppid'] == parent['pid'], 'Actual spawn parent/child identity mismatch')
    require(spawned['configuration']['configuration_sha256'] == parent['configuration']['configuration_sha256'], 'Spawn SDK config differs')
    require(spawned['origins'] == parent['origins'], 'Spawn native/package origins differ')
    require(spawned['tokenization'] == parent['tokenization'], 'Spawn complete tokenizer evidence differs')
    check_binding(spec)
    result = dict(schema=SCHEMA + '-result', complete=True, passed=True, parent=parent, child=spawned,
        source_bundle=spec['bundle'], root_binding=spec['binding'], source_commit=SOURCE_COMMIT,
        full_model_qualification=False, inherited_performance_claim=False, lock_released=False)
    once(work / 'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-native', action='store_true')
    parser.add_argument('--input'); parser.add_argument('--input-sha256')
    args = parser.parse_args()
    if not args.execute_native:
        print(json.dumps(schema(), indent=2)); return
    require(args.input and args.input_sha256, 'Explicit root input and SHA256 required')
    raw = read_small(args.input)
    require(sha(raw) == args.input_sha256, 'Root input changed')
    execute(json.loads(raw))


if __name__ == '__main__':
    main()
