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
import hashlib
import importlib
import json
import os
from pathlib import Path
import resource
import signal
import stat
import sys
import time

SCHEMA = 'qwen35-package-native-preflight-v3'
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


def install_runtime_guard(tokenizer_allowlist=()):
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
    install_runtime_guard(tokenizer_checks.tokenizer_paths(spec["work"]))
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
        model_constructed=False, weights_read=False, device_graph_executed=False, compiled=False)
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
    parent = native_checks(spec, bundle, 'parent')
    import multiprocessing
    context = multiprocessing.get_context('spawn')
    child = context.Process(target=spawn_worker, args=(spec, bundle), name='qwen35-package-preflight-spawn')
    child.start()
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
