#!/usr/bin/env python3
"""Inert until a reviewed spec and actual external owner/service ACK are supplied.

This is a worker, not a queue, launcher, slot allocator, or qualification runner.
The future root launcher retains the serial lock and owns terminal/drain evidence.
Only wheel installation, plugin discovery, and configuration objects are exercised.
"""
import argparse
import configparser
import copy
import hashlib
import importlib
import importlib.metadata
import importlib.util
import io
import json
import multiprocessing
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
import traceback
import zipfile

ROOT = Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
REPOSITORY = Path('/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107')
COMMIT = 'e0a01d3bd9e1fec9faf2b6b0f37ecbe2d8bf92dc'
LIFECYCLE = 'smollm3_observer_invariance_lifecycle_v18'
PLUGINS = {'lfm3': 'vllm_neuron.models.lfm3:register',
           'neuron_smollm3': 'vllm_neuron.models.smollm3:register',
           'nemotron9b': 'vllm_neuron.models.nemotron9b:register'}
MODELS = {
    'lfm3': ('lfm3', 'Lfm3NeuronConfig', 'Lfm3ForCausalLM', 'lfm3',
             'NeuronLfm3ForCausalLM', 'Lfm3InferenceConfig', 2048),
    'smollm3': ('smollm3_neuron_text', 'SmolLM3NeuronTextConfig',
                'SmolLM3NeuronTextForCausalLM', 'smollm3neurontext',
                'NeuronSmolLM3ForCausalLM', 'SmolLM3InferenceConfig', 2048),
    'nemotron9b': ('nemotron9b_neuron_text', 'Nemotron9BNeuronTextConfig',
                   'Nemotron9BNeuronTextForCausalLM', 'nemotron9bneurontext',
                   'NeuronNemotron9BForCausalLM', 'Nemotron9BInferenceConfig', 4096),
}
MODES = {'default': (None, 'disabled'),
         'lfm3-reference': ('lfm3', 'reference'),
         'lfm3-nki': ('lfm3', 'nki_predicated_decode_v4'),
         'smollm3-reference': ('smollm3', 'reference'),
         'smollm3-read': ('smollm3', 'nki_read'),
         'smollm3-write': ('smollm3', 'nki_write'),
         'smollm3-read-write': ('smollm3', 'nki_read_write'),
         'nemotron9b-split-nki-v5': ('nemotron9b', 'split_nki_v5')}
PHASES = ('install', *MODES)
NATIVE_ROOTS = {'torch', 'torch_xla', 'torch_neuronx', 'transformers', 'vllm',
                'vllm_neuron', 'neuronxcc', 'neuronx_distributed',
                'neuronx_distributed_inference'}
RESOURCE_LIMITS = {'MemoryMax': str(12 * 2**30), 'MemoryHigh': str(10 * 2**30),
                   'MemorySwapMax': '0', 'AllowedCPUs': '0-7', 'CPUAffinity': '0-7',
                   'Nice': '19', 'KillMode': 'control-group', 'RemainAfterExit': 'yes',
                   'LimitFSIZE': str(64 * 2**20), 'RuntimeMaxUSec': '15min',
                   'WorkingDirectory': str(ROOT), 'TasksMax': '256'}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def ordinary(path, directory=False):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts, 'Absolute ordinary path required')
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'Symlink component rejected')
    value = path.stat()
    require(stat.S_ISDIR(value.st_mode) if directory else stat.S_ISREG(value.st_mode),
            'Ordinary file/directory required')
    require(directory or value.st_nlink == 1, 'Shared hardlink rejected')
    return value


def read(path, cap=8 * 2**20):
    path = Path(path)
    require(path.suffix not in {'.safetensors', '.bin', '.pt', '.pth', '.neff', '.so'},
            'Payload/native binary IO is outside this worker')
    before = ordinary(path)
    require(before.st_size <= cap, 'Metadata/source exceeds bound')
    with path.open('rb') as stream:
        raw = stream.read(cap + 1)
    after = ordinary(path)
    require(len(raw) == before.st_size == after.st_size and
            (before.st_ino, before.st_dev, before.st_mtime_ns, before.st_ctime_ns) ==
            (after.st_ino, after.st_dev, after.st_mtime_ns, after.st_ctime_ns), 'Input changed')
    return raw


def reference(path):
    raw = read(path)
    return {'path': str(path), 'bytes': len(raw), 'sha256': sha(raw)}


def checked(row, cap=8 * 2**20):
    require(set(row) == {'path', 'bytes', 'sha256'}, 'Exact file reference required')
    raw = read(row['path'], cap)
    require(len(raw) == row['bytes'] and sha(raw) == row['sha256'], 'File reference changed: ' + row['path'])
    return raw


def once(path, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    require(len(raw) < 2 * 2**20, 'Output bound')
    ordinary(Path(path).parent, directory=True)
    with Path(path).open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def relative(name):
    path = Path(name)
    require(name == path.as_posix() and not path.is_absolute() and '..' not in path.parts
            and bool(path.parts), 'Unsafe relative package path')
    return path


def entry_points(raw):
    parser = configparser.ConfigParser()
    parser.read_string(raw.decode())
    general = dict(parser.items('vllm.general_plugins'))
    platform = dict(parser.items('vllm.platform_plugins'))
    require(general == PLUGINS and platform == {'neuron': 'vllm_neuron:register'},
            'Wrong/missing/extra installed plugin entrypoint')
    return general


def environment(phase, work, inherited):
    """Root launcher uses this before starting each fresh isolated interpreter."""
    require(phase in PHASES, 'Unknown phase')
    require(not any(k.startswith('TRN1_') for k in inherited), 'Historical experiment flags rejected')
    env = {k: v for k, v in inherited.items() if not k.startswith(
        ('NEURON_LFM3', 'NEURON_SMOLLM3', 'NEURON_NEMOTRON9B')) and k not in
        {'PYTHONPATH', 'PYTHONPYCACHEPREFIX', 'PYTHONHOME', 'NEURON_CPU_SAMPLER_FAST'}}
    model, variant = MODES.get(phase, (None, 'disabled'))
    scratch = Path(work) / phase
    env.update(NEURON_LFM3='1' if model == 'lfm3' else '0',
               NEURON_SMOLLM3='1' if model == 'smollm3' else '0',
               NEURON_NEMOTRON9B='1' if model == 'nemotron9b' else '0',
               NEURON_LFM3_NKI_WRITER='1' if phase == 'lfm3-nki' else '0',
               NEURON_LFM3_ALLOW_COMPILE='0', NEURON_SMOLLM3_ALLOW_COMPILE='0',
               NEURON_NEMOTRON9B_ALLOW_COMPILE='0',
               NEURON_CPU_SAMPLER_FAST='1' if model == 'lfm3' else '0',
               XLA_DISABLE_FUNCTIONALIZATION='1', NEURON_RT_VISIBLE_CORES='0-1',
               VLLM_PLUGINS='neuron,lfm3,neuron_smollm3,nemotron9b',
               VLLM_WORKER_MULTIPROC_METHOD='spawn', PYTHONDONTWRITEBYTECODE='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PIP_NO_INDEX='1',
               PIP_DISABLE_PIP_VERSION_CHECK='1', PIP_NO_CACHE_DIR='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               TOKENIZERS_PARALLELISM='false', TMPDIR=str(scratch / 'tmp'),
               XDG_CACHE_HOME=str(scratch / 'cache'), HF_HOME=str(scratch / 'hf'),
               TORCHINDUCTOR_CACHE_DIR=str(scratch / 'torch-cache'),
               NEURON_COMPILE_CACHE_URL=str(scratch / 'forbidden-compile-cache'))
    if model == 'smollm3':
        env['NEURON_SMOLLM3_CACHE_VARIANT'] = variant
    return env


def verify_environment(phase, work):
    expected = environment(phase, work, {})
    require(all(os.environ.get(k) == v for k, v in expected.items()), 'Worker environment differs')
    require(not any(k.startswith('TRN1_') for k in os.environ), 'Historical flags present')
    require(not {'PYTHONPATH', 'PYTHONPYCACHEPREFIX', 'PYTHONHOME',
                 'NEURON_SMOLLM3_NKI_CACHE'} & set(os.environ), 'Competing environment setting')
    require('NEURON_SMOLLM3_CACHE_VARIANT' not in os.environ or phase.startswith('smollm3-'),
            'Smol selector leaked into another model')
    for prefix in ('NEURON_LFM3', 'NEURON_SMOLLM3', 'NEURON_NEMOTRON9B'):
        require(all(k in expected for k in os.environ if k.startswith(prefix)), 'Unknown model switch')


def validate_spec(spec, review, spec_sha, bundle):
    """Pure shape/content gate; actual ownership is separately observed below."""
    require(spec['schema'] == 'unified_package_native_spec_v1' and spec['commit'] == COMMIT,
            'Wrong spec/commit')
    require(review['schema'] == 'unified_package_native_root_review_v1' and
            review['spec_sha256'] == spec_sha and review['decision'] == 'approved',
            'Concrete root review does not bind spec')
    require(review['source_bundle'] == spec['source_bundle'], 'Review source bundle differs')
    require(bundle['schema'] == 'unified_package_native_source_bundle_v1' and bundle['commit'] == COMMIT
            and bundle['repository'] == str(REPOSITORY), 'Wrong source bundle')
    require(tuple(spec['phases']) == PHASES, 'All phases required in fixed serial order')
    require(spec['python'] == '/home/ubuntu/venvs/vllm_neuron/bin/python', 'Wrong interpreter')
    require(set(bundle['fixtures']) == set(MODELS), 'Every model needs an exact root config fixture')
    require(set(bundle['lifecycle_sources']) and str(ROOT / (LIFECYCLE + '.py')) in
            bundle['lifecycle_sources'], 'Lifecycle closure missing')
    require(bundle['worker']['path'] == str(Path(__file__).absolute()), 'Wrong worker source')
    work, control = Path(spec['work']), Path(spec['control'])
    require(work.parent == ROOT and work.name.startswith('unified-package-native-') and
            control.parent == ROOT / 'research' and control.name.startswith('unified-package-native-'),
            'Fresh dedicated work/control roots required')
    require(spec['resource_properties'] == RESOURCE_LIMITS, 'Resource scope changed')
    require(set(spec['intents']) == set(PHASES), 'Every phase requires exact service intent')
    for phase, intent in spec['intents'].items():
        require(re.fullmatch(r'trn1-unified-native-[a-z0-9-]+', intent['unit']), 'Dedicated unit required')
        require(intent['argv_prefix'] == [spec['python'], '-I', '-B', bundle['worker']['path']],
                'Fresh isolated worker argv required')
        require(intent['phase'] == phase, 'Phase intent differs')
        require(set(intent) == {'unit', 'phase', 'argv_prefix'}, 'Only reviewed intent template fields')
    require(spec['predecessor_catalog'] == review['predecessor_catalog'] and
            spec['physical_drain_receipt'] == review['physical_drain_receipt'],
            'Concrete current predecessor and physical drain references required')
    require(spec['wheel']['path'].endswith('.whl') and spec['wheel']['bytes'] < 32 * 2**20,
            'Bounded prebuilt wheel required')


def runtime_intent(spec, args):
    """Resolve hashes after review to avoid a self-referential spec/argv digest."""
    template = spec['intents'][args.phase]
    argv = template['argv_prefix'] + [
        '--spec', args.spec, '--spec-sha256', args.spec_sha256,
        '--review', args.review, '--review-sha256', args.review_sha256, '--phase', args.phase]
    return dict(unit=template['unit'], phase=args.phase, execargv=argv)


class PinnedSources:
    """Compile checked lifecycle source directly; never consume stale local pyc."""
    def __init__(self, rows):
        self.rows = {Path(path).stem: row for path, row in rows.items()}
        require(not set(self.rows) & set(sys.modules), 'Lifecycle modules must start fresh')

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.rows:
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def get_filename(self, fullname):
        return self.rows[fullname]['path']

    def is_package(self, fullname):
        return False

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        row = self.rows[module.__name__]
        exec(compile(checked(row), row['path'], 'exec'), module.__dict__)


class Session:
    def __init__(self, args):
        self.args = args
        raw = read(args.spec)
        require(sha(raw) == args.spec_sha256, 'Spec changed')
        self.spec = json.loads(raw)
        review_raw = read(args.review)
        require(sha(review_raw) == args.review_sha256, 'Review changed')
        self.review = json.loads(review_raw)
        self.bundle = json.loads(checked(self.spec['source_bundle']))
        validate_spec(self.spec, self.review, args.spec_sha256, self.bundle)
        require(sys.flags.isolated == 1 and sys.flags.optimize == 0 and sys.dont_write_bytecode,
                'Use python -I -B without optimization')
        require(sys.version_info[:2] == (3, 10) and sys.executable == self.spec['python'],
                'Pinned Python 3.10 required')
        checked(self.bundle['worker'])
        for path, row in self.bundle['lifecycle_sources'].items():
            require(row['path'] == path and Path(path).parent == ROOT and path.endswith('.py'),
                    'Lifecycle closure must contain only root Python sources')
            checked(row)
        for row in (self.spec['predecessor_catalog'], self.spec['physical_drain_receipt']):
            checked(row)
        self.source_loader = PinnedSources(self.bundle['lifecycle_sources'])
        sys.meta_path.insert(0, self.source_loader)
        self.life = importlib.import_module(LIFECYCLE)
        self.work, self.control = Path(self.spec['work']), Path(self.spec['control'])
        self.phase = args.phase
        self.intent = runtime_intent(self.spec, args)
        self.root_owner = self.spec['external_lock_owner']
        self.life.lock_proof(self.root_owner, self.spec['lock_identity'])
        require(self.root_owner['pid'] != os.getpid(), 'Worker cannot be its own external lock owner')
        verify_environment(self.phase, self.work)
        self.ack = None

    def ready(self):
        actual = self.life.process(os.getpid())
        require(actual['argv'] == self.intent['execargv'], 'Worker process argv differs from reviewed service')
        ordinary(self.control, directory=True)
        once(self.control / (self.phase + '.ready.json'),
             dict(schema='unified_package_native_ready_v1', phase=self.phase,
                  spec_sha256=self.args.spec_sha256, process=actual))
        # Root observes the actual service and writes ACK only after binding its
        # InvocationID, process identity and retained serial owner. No native IO yet.
        ack_path = self.control / (self.phase + '.root-ack.json')
        deadline = time.monotonic() + 60
        while not ack_path.exists():
            self.life.lock_proof(self.root_owner, self.spec['lock_identity'])
            require(time.monotonic() < deadline, 'Root service ACK timeout')
            time.sleep(.1)
        self.ack = json.loads(read(ack_path))
        self.ack_sha = sha(read(ack_path))
        self.check()

    def check(self, spawn=None):
        require(self.ack is not None, 'Actual root owner/service ACK required')
        require(sha(read(self.args.spec)) == self.args.spec_sha256 and
                sha(read(self.args.review)) == self.args.review_sha256, 'Reviewed binding changed')
        require(sha(read(self.control / (self.phase + '.root-ack.json'))) == self.ack_sha, 'Root ACK changed')
        ack = self.ack
        require(ack['schema'] == 'unified_package_native_owner_ack_v1' and
                ack['phase'] == self.phase and ack['spec_sha256'] == self.args.spec_sha256 and
                ack['review_sha256'] == self.args.review_sha256 and
                ack['external_lock_owner'] == self.root_owner and
                ack['lock_identity'] == self.spec['lock_identity'], 'ACK binding mismatch')
        lock = self.life.lock_proof(self.root_owner, self.spec['lock_identity'])
        value = self.life.props(self.intent['unit'])
        self.life.verify_binding(self.intent, value, ack['service_binding'])
        require(value['ActiveState'] == 'active' and value['SubState'] == 'running' and
                value['MainPID'] == str(ack['process']['pid']), 'Service is not actual running worker')
        require(value['ControlGroup'] == self.life.GROUP_PREFIX + self.intent['unit'] + '.service',
                'Service cgroup differs')
        require(all(value.get(k) == v for k, v in RESOURCE_LIMITS.items()), 'Service resource profile differs')
        parent = self.life.same_process(ack['process'])
        require(parent['argv'] == self.intent['execargv'] and
                parent['cgroup'] == '0::' + value['ControlGroup'], 'Service process proof differs')
        current = self.life.process(os.getpid())
        if spawn is None:
            require(current == parent, 'Only exact ACK process may run phase')
        else:
            require(spawn == current and current['ppid'] == parent['pid'] and
                    current['cgroup'] == parent['cgroup'] and current['pid'] != parent['pid'],
                    'Actual direct spawn child in same service required')
        verify_environment(self.phase, self.work)
        return {'lock': lock, 'service_binding': ack['service_binding'], 'process': current}

    def sources(self):
        for name, expected in self.bundle['package_sources'].items():
            require(name.startswith('vllm_neuron/'), 'Only package source inventory')
            require(sha(read(REPOSITORY / relative(name))) == expected, 'Pinned package source changed')
        for row in self.bundle['sdk_sources'] + self.bundle['metadata_sources']:
            checked(row)
        for row in self.bundle['fixtures'].values():
            require(row['path'].startswith(str(ROOT / 'research') + '/') and row['path'].endswith('.json'),
                    'Only copied config JSON fixtures allowed')
            checked(row)
        for name, version in self.bundle['versions'].items():
            require(importlib.metadata.version(name) == version, 'SDK version changed: ' + name)


def wheel_sources(raw, expected):
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = archive.namelist()
        require(len(set(names)) == len(names) and len(names) < 2048, 'Duplicate/excess wheel members')
        require(sum(item.file_size for item in archive.infolist()) < 32 * 2**20, 'Wheel expansion bound')
        for item in archive.infolist():
            relative(item.filename)
            require(not stat.S_ISLNK(item.external_attr >> 16), 'Wheel symlink rejected')
        packaged = {name: sha(archive.read(name)) for name in names if name.startswith('vllm_neuron/')}
        require(packaged == expected, 'Wheel does not contain exactly pinned package files')
        entries = [n for n in names if n.endswith('.dist-info/entry_points.txt')]
        require(len(entries) == 1, 'One distribution entrypoint file required')
        entry_points(archive.read(entries[0]))
        return {n: sha(archive.read(n)) for n in names}


def inventory(target):
    ordinary(target, directory=True)
    result = {}
    for path in sorted(target.rglob('*')):
        require(not path.is_symlink(), 'Installed symlink rejected')
        if path.is_dir():
            continue
        name = str(path.relative_to(target))
        require(path.suffix != '.pyc', 'Bytecode not allowed in exact wheel target')
        result[name] = sha(read(path))
    require(len(result) < 2048, 'Installed inventory bound')
    return result


def scratch(work, phase):
    base = work / phase
    base.mkdir(mode=0o700)
    for name in ('tmp', 'cache', 'hf', 'torch-cache'):
        (base / name).mkdir(mode=0o700)
    return base


def install(session):
    session.check()
    session.sources()
    wheel = checked(session.spec['wheel'], 32 * 2**20)
    expected = wheel_sources(wheel, session.bundle['package_sources'])
    require(not session.work.exists() and not session.work.is_symlink(), 'Work path must be entirely new')
    session.work.mkdir(mode=0o700)
    scratch(session.work, 'install')
    target = session.work / 'target'
    # Prebuilt exact wheel only; no build tool, index, dependencies or global installation.
    argv = [session.spec['python'], '-I', '-B', '-m', 'pip', 'install', '--no-deps',
            '--no-index', '--no-cache-dir', '--no-compile', '--target', str(target),
            session.spec['wheel']['path']]
    session.check()
    subprocess.run(argv, cwd=session.work, env=dict(os.environ), check=True, timeout=180)
    session.check()
    actual = inventory(target)
    for name, expected_sha in expected.items():
        if not name.endswith('.dist-info/RECORD'):
            require(actual.get(name) == expected_sha, 'Installed wheel member differs: ' + name)
    extra = set(actual) - set(expected)
    require(all('.dist-info/' in n and n.rsplit('/', 1)[1] in
                {'INSTALLER', 'REQUESTED', 'direct_url.json'} for n in extra), 'Unexpected installed file')
    require(sha(checked(session.spec['wheel'], 32 * 2**20)) == sha(wheel), 'Wheel changed during install')
    return dict(target=str(target), installed=actual, wheel=session.spec['wheel'],
                global_environment_modified=False, package_installed=True)


def activate(session):
    session.sources()
    receipt = json.loads(checked(session.ack['install_receipt']))
    require(receipt['schema'] == 'unified_package_native_result_v1' and receipt['phase'] == 'install'
            and receipt['spec_sha256'] == session.args.spec_sha256 and receipt['passed'] is True,
            'Root ACK must bind successful install receipt')
    body = receipt['details']
    target = session.work / 'target'
    require(body['target'] == str(target) and body['wheel'] == session.spec['wheel'], 'Wrong installed target')
    require(inventory(target) == body['installed'], 'Installed target changed')
    require(not any(n.split('.')[0] in NATIVE_ROOTS for n in sys.modules), 'Fresh native interpreter required')
    sys.path.insert(0, str(target))
    distributions = list(importlib.metadata.distributions(path=[str(target)]))
    require(len(distributions) == 1, 'One installed distribution required')
    dist = distributions[0]
    require(dist.metadata['Name'] == 'vllm-neuron' and dist.version == '0.2.2+lts', 'Wrong installed distribution')
    found = [ep for ep in dist.entry_points if ep.group == 'vllm.general_plugins']
    require(len(found) == len(PLUGINS) and {ep.name: ep.value for ep in found} == PLUGINS,
            'Installed distribution plugins differ')
    # Reject shadow distribution metadata as well as wrong Python import origin.
    discovered = [ep for ep in importlib.metadata.entry_points(group='vllm.general_plugins')
                  if ep.name in PLUGINS]
    require(len(discovered) == len(PLUGINS) and {ep.name: ep.value for ep in discovered} == PLUGINS,
            'Duplicate/shadow package entrypoints discovered')
    return target, found


def package_origins(target):
    modules = {n: str(Path(m.__file__).resolve()) for n, m in sys.modules.items()
               if n.startswith('vllm_neuron') and getattr(m, '__file__', None)}
    require(modules and all(Path(path).is_relative_to(target) for path in modules.values()),
            'Package escaped exact fresh wheel target')
    return modules


def disabled_absent(selected):
    for model in MODELS:
        if model != selected:
            prefix = 'vllm_neuron.models.' + model + '.'
            require(not any(n.startswith(prefix) for n in sys.modules), 'Disabled adapter imported: ' + model)


def plugin_probe(session):
    target, entries = activate(session)
    model, expected_variant = MODES[session.phase]
    if model is None:
        for ep in entries:
            require(ep.load()() is None, 'Disabled entrypoint result changed')
        require(not any(n.split('.')[0] in NATIVE_ROOTS - {'vllm_neuron'} for n in sys.modules),
                'Disabled entrypoint imported a native framework')
        disabled_absent(None)
    import vllm.plugins as plugins
    loaded = plugins.load_plugins_by_group('vllm.general_plugins')
    require(set(loaded) == set(PLUGINS) and all(loaded[ep.name] is ep.load() for ep in entries),
            'Actual vLLM general-plugin discovery differed')
    plugins.load_general_plugins()
    from vllm.model_executor.models.registry import ModelRegistry
    disabled_absent(model)
    for name, fields in MODELS.items():
        require((fields[2] in ModelRegistry.get_supported_archs()) == (name == model),
                'Wrong registered architecture: ' + name)
    if model is not None:
        from transformers import AutoConfig
        from neuronx_distributed_inference.utils.constants import MODEL_TYPES
        package = 'vllm_neuron.models.' + model
        registration = importlib.import_module(package + '.registration')
        modeling = importlib.import_module(package + '.modeling')
        identity = importlib.import_module(package + '.identity')
        hf_type, config_name, arch, nxdi_type, nxdi_class, _, _ = MODELS[model]
        require(identity.selected_variant() == expected_variant, 'Wrong selected variant')
        require(registration._INSTALLED and type(AutoConfig.for_model(hf_type)) is
                getattr(registration, config_name), 'Real AutoConfig registration differs')
        require(MODEL_TYPES[nxdi_type]['causal-lm'] is getattr(modeling, nxdi_class), 'NxDI registry differs')
        if model == 'lfm3':
            core = importlib.import_module(package + '.text_core')
            writer = importlib.import_module(package + ('.helpers' if expected_variant == 'reference'
                                                        else '.cache_writer'))
            require(core.store_slots is getattr(writer, 'store_slots' if expected_variant == 'reference'
                                                else 'store_slots_nki_predicated_decode'), 'LFM writer differs')
        elif model == 'nemotron9b':
            core = importlib.import_module(package + '.text_core')
            cache = importlib.import_module(package + '.cache')
            require(core.Mamba.forward is cache.mamba_forward, 'Nemotron split cache patch absent')
        # Smol selects attention during actual model construction, which this
        # preflight deliberately never invokes. This checks selector/registration only.
        registration.install()
    plugins.load_general_plugins()
    disabled_absent(model)
    return dict(pid=os.getpid(), model=model, variant=expected_variant,
                actual_general_plugin_loader=True, actual_autoconfig=model is not None,
                package_modules=package_origins(target),
                smollm3_attention_execution_checked=False, full_model_constructed=False)


def configurations(session, role):
    model, variant = MODES[session.phase]
    if model is None:
        return None
    package = 'vllm_neuron.models.' + model
    identity_module = importlib.import_module(package + '.identity')
    inference = getattr(importlib.import_module(package + '.modeling'), MODELS[model][5])
    admit_saved = importlib.import_module(package + '.artifacts').admit_saved_configuration
    canonical = identity_module.canonical_configuration
    original = json.loads(checked(session.bundle['fixtures'][model]))
    require(original['metadata'] is None, 'Historical fixture metadata must be null')
    require(original['neuron_config']['seq_len'] == MODELS[model][6], 'Original model context differs')
    # Real SDK config constructor, not a mock or whole model/application.
    config = inference.from_json_string(json.dumps(original))
    body = json.loads(config.to_json_string())
    require(canonical(body) == canonical(original), 'SDK constructor changed full root body')
    identity_module.validate_scope(body, body['neuron_config'])
    require(config.neuron_config.quantize_clamp_bound == float('inf'), 'SDK infinity default changed')
    if model == 'nemotron9b':
        require(config.time_step_limit[1] == float('inf'), 'Nemotron dt infinity changed')
    output = session.work / session.phase / ('configuration-' + role)
    output.mkdir(mode=0o700)
    saved = output / 'root'
    config.save(saved)
    loaded = inference.load(saved)
    require(canonical(json.loads(loaded.to_json_string())) == canonical(original), 'Root save/load changed full body')
    # The package's config-only full-body check, never build_identity/admit_existing.
    config_identity = dict(configuration=original, commit=COMMIT, variant=variant,
                           preflight_config_only=True, full_artifact_admitted=False)
    config.metadata = {model + '_ultra': config_identity}
    bound = output / 'metadata-root'
    config.save(bound)
    admit_saved(bound, config_identity)
    restored_body = json.loads(inference.load(bound).to_json_string())
    require(restored_body['metadata'] == config.metadata, 'Embedded config identity changed')
    restored_body['metadata'] = None
    require(canonical(restored_body) == canonical(original), 'Metadata roundtrip changed remaining root body')
    scope_rejected = []
    for field, value in [('tp_degree', 1), ('world_size', 4), ('local_ranks_size', 1),
                         ('ep_degree', 2), ('seq_len', 2048 if model == 'nemotron9b' else 4096)]:
        bad = copy.deepcopy(original)
        bad['neuron_config'][field] = value
        try:
            # Nemotron's low-level constructor permits TP1; the packaged route's
            # actual scope admission is authoritative, so exercise both in order.
            candidate = inference.from_json_string(json.dumps(bad))
            candidate_body = json.loads(candidate.to_json_string())
            identity_module.validate_scope(candidate_body, candidate_body['neuron_config'])
        except (identity_module.AdmissionError, ValueError, AssertionError) as error:
            scope_rejected.append(dict(field=field, error_type=type(error).__name__))
        else:
            raise RuntimeError('Invalid packaged configuration scope admitted: ' + field)
    baseline = json.loads(read(bound / 'neuron_config.json'))
    mutations = [('allow_input_truncation', True), ('quantize_clamp_bound', 1.),
                 ('quantize_clamp_bound', float('nan')), ('quantize_clamp_bound', -float('inf')),
                 ('unexpected_infinity', float('inf')), ('async_mode', 0), ('unknown_field', 'surprise')]
    rejected = []
    for index, (field, value) in enumerate(mutations):
        bad = copy.deepcopy(baseline)
        bad['neuron_config'][field] = value
        directory = output / ('rejected-' + str(index))
        directory.mkdir(mode=0o700)
        # Deliberately malformed config JSON only; never a tensor/model artifact.
        (directory / 'neuron_config.json').write_text(json.dumps(bad))
        try:
            admit_saved(directory, config_identity)
        except identity_module.AdmissionError as error:
            rejected.append(dict(field=field, error_type=type(error).__name__))
        else:
            raise RuntimeError('Changed saved configuration admitted: ' + field)
    return dict(actual_sdk_constructor=True, actual_sdk_root_save_load=True,
                complete_root_equal=True, context=MODELS[model][6], scope_rejections=scope_rejected,
                saved_body_rejections=rejected, full_artifact_admitted=False)


def spawn_child(args, connection):
    """Actual multiprocessing.spawn child waits for parent's real process ACK."""
    try:
        session = Session(args)
        actual = session.life.process(os.getpid())
        connection.send(dict(kind='ready', process=actual))
        require(connection.poll(60), 'Parent spawn ACK timeout')
        message = connection.recv()
        require(message['kind'] == 'spawn-ack' and message['child'] == actual and
                message['spec_sha256'] == args.spec_sha256, 'Spawn ACK differs')
        session.ack = json.loads(read(session.control / (session.phase + '.root-ack.json')))
        session.ack_sha = message['root_ack_sha256']
        require(message['parent'] == session.ack['process'], 'Spawn ACK parent differs')
        session.check(spawn=actual)
        plugin = plugin_probe(session)
        session.check(spawn=actual)
        config = configurations(session, 'spawn')
        proof = session.check(spawn=actual)
        connection.send(dict(kind='result', passed=True, plugin=plugin, configuration=config, final_proof=proof))
    except BaseException:
        connection.send(dict(kind='error', traceback=traceback.format_exc(limit=12)))
        raise
    finally:
        connection.close()


def native_phase(session):
    session.check()
    scratch(session.work, session.phase)
    parent_plugin = plugin_probe(session)
    session.check()
    parent_config = configurations(session, 'parent')
    session.check()
    context = multiprocessing.get_context('spawn')
    parent_pipe, child_pipe = context.Pipe()
    child = context.Process(target=spawn_child, args=(session.args, child_pipe))
    child.start()
    child_pipe.close()
    message = None
    try:
        require(parent_pipe.poll(60), 'Spawn readiness timeout')
        ready = parent_pipe.recv()
        require(ready['kind'] == 'ready', 'Spawn failed before ACK: ' + repr(ready))
        actual = session.life.process(child.pid)
        parent = session.life.process(os.getpid())
        require(ready['process'] == actual and actual['ppid'] == parent['pid'] and
                actual['cgroup'] == parent['cgroup'], 'Spawn process not actual direct child')
        session.check()
        ack = dict(kind='spawn-ack', child=actual, parent=parent, spec_sha256=session.args.spec_sha256,
                   root_ack_sha256=session.ack_sha)
        once(session.control / (session.phase + '.spawn-ack.json'), ack)
        parent_pipe.send(ack)
        deadline = time.monotonic() + 240
        while not parent_pipe.poll(2):
            require(time.monotonic() < deadline, 'Spawn native probe timeout')
            session.check()
        message = parent_pipe.recv()
        child.join(timeout=30)
        require(message.get('passed') is True and child.exitcode == 0, 'Spawn native probe failed: ' + repr(message))
        require(message['plugin']['pid'] == child.pid and child.pid != os.getpid() and
                message['plugin']['variant'] == parent_plugin['variant'], 'Spawn identity/variant differs')
    finally:
        if child.is_alive():
            child.terminate()
            child.join(timeout=10)
        if child.is_alive():
            child.kill()
            child.join(timeout=10)
        require(not child.is_alive(), 'Spawn child remains; root must retain slot and drain service')
        parent_pipe.close()
    session.check()
    return dict(parent=dict(plugin=parent_plugin, configuration=parent_config), spawn=message,
                spawn_exitcode=child.exitcode, actual_start_method='spawn')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec')
    parser.add_argument('--spec-sha256')
    parser.add_argument('--review')
    parser.add_argument('--review-sha256')
    parser.add_argument('--phase', choices=PHASES)
    args = parser.parse_args(argv)
    if not any(vars(args).values()):
        print(json.dumps(dict(source_only=True, native_executed=False,
              phases=PHASES, external_root_launcher_required=True,
              runtime_authorization='reviewed spec + real serial owner + actual service ACK')))
        return 0
    require(all(vars(args).values()), 'All concrete reviewed bindings and one phase required')
    session = Session(args)
    session.ready()
    details = install(session) if args.phase == 'install' else native_phase(session)
    proof = session.check()
    once(session.control / (args.phase + '.result.json'),
         dict(schema='unified_package_native_result_v1', passed=True, phase=args.phase,
              commit=COMMIT, spec_sha256=args.spec_sha256, root_ack_sha256=session.ack_sha,
              details=details, final_owner_service_proof=proof,
              full_model_qualified=False, new_speedup_claim=False, model_payload_read=False,
              tokenizer_payload_read=False, model_constructed=False, device_graph_executed=False,
              compile_executed=False, native_binaries_rehashed=False, serial_slot_released=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
