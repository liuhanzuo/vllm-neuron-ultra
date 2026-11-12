"""Concrete CPU native imports, deferred until an authenticated serial worker.

No native import or file IO occurs on import. RECORD bytes, local source
overrides and linked native files must be frozen in a root-reviewed manifest.
This authenticates the declared application/SDK closure, not the OS interpreter.
Unknown third-party imports fail before their loader executes. No AST lifting.
"""
import base64
import copy
import csv
import hashlib
import importlib
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import re
import stat
import sys
import sysconfig
import types

import spark26_native_port_diagnostic_contract_v2 as contract

SITE = Path('/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages')
EDITABLE = Path('/home/ubuntu/src/vllm-neuron')
REQUIRED = {
    'torch': '2.8.0', 'transformers': '4.57.6', 'numpy': '1.26.4',
    'tokenizers': '0.22.2', 'safetensors': '0.7.0', 'ml_dtypes': '0.5.4',
    'huggingface_hub': '0.36.2', 'torch_xla': '2.8.1',
    'torch_neuronx': '2.8.0.2.12.22436+0f1dac25',
    'neuronx_cc': '2.23.6484.0+3b612583', 'libneuronxla': '2.2.15515.0+50c26cbd',
    'neuronx_distributed': '0.17.26814+4b18de63',
    'neuronx_distributed_inference': '0.8.16251+f3ca5575',
    'vllm': '0.11.0', 'vllm_neuron': '0.2.2+lts', 'regex': '2026.2.19',
}
MAX_MANIFEST_ENTRIES = 100000
MAX_RUNTIME_AUTHENTICATION_READ = 16 * 2**30
require = contract.require


def identity(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path
            and not any(p.is_symlink() for p in (path, *path.parents)), 'Canonical immutable input path')
    s = path.lstat()
    require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1, 'Singleton regular runtime input')
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def authenticate_file(path, expected, guard):
    """Actual full-byte hash only inside the real owned worker, never preparation."""
    guard()
    before = identity(path)
    require(before[2] == expected['bytes'], 'Exact declared runtime file size')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    digest, count = hashlib.sha256(), 0
    try:
        s = os.fstat(fd)
        require([s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns] == before,
                'Runtime input open identity')
        while count < expected['bytes']:
            guard()
            block = os.read(fd, min(2**20, expected['bytes'] - count))
            require(bool(block), 'Complete runtime payload')
            count += len(block); digest.update(block)
        require(os.read(fd, 1) == b'', 'No runtime trailing bytes')
    finally:
        os.close(fd)
    require(identity(path) == before and digest.hexdigest() == expected['sha256'],
            'Frozen actual runtime bytes: ' + str(path))
    return before


def record_payloads(record_text, site=SITE):
    """Pure RECORD parser: package Python/native/data, no console scripts/pyc.

    Script paths outside site-packages are excluded, never normalized into
    package imports. Every actual import must independently have a pinned origin.
    """
    result = {}
    for relative, encoded, size in csv.reader(record_text.splitlines()):
        path = site / relative
        if '..' in Path(relative).parts or Path(relative).is_absolute():
            continue
        if path.suffix == '.pyc' or '__pycache__' in path.parts:
            continue
        if not encoded:
            require(path.name == 'RECORD', 'Unhashed package payload is not allowed')
            continue
        require(encoded.startswith('sha256=') and size.isdigit(), 'SHA256 RECORD payload')
        digest = base64.urlsafe_b64decode(encoded[7:] + '=' * (-len(encoded[7:]) % 4))
        require(len(digest) == 32 and str(path) not in result, 'Unique RECORD payload')
        result[str(path)] = {'sha256': digest.hex(), 'bytes': int(size)}
    return result


def merge_manifest(manifest, source_sha256, read_record):
    """Concrete RECORD reconstruction, rejecting native hash overrides."""
    require(set(manifest) == {'schema', 'distributions', 'python_overrides', 'additional_files',
                             'linked_native_files', 'payload_count', 'payload_bytes', 'payload_sha256'},
            'Exact native manifest schema')
    require(manifest['schema'] == 'spark26_native_provider_manifest_v1', 'Native manifest version')
    packages, payloads = {}, {}
    for row in manifest['distributions']:
        require(set(row) == {'name', 'version', 'record'}, 'Named frozen distribution')
        name, version = row['name'], row['version']
        require(name not in packages and name.replace('_', '').isalnum()
                and re.fullmatch(r'[0-9][A-Za-z0-9.+_-]{0,127}', version), 'Unique normalized package name/version')
        expected = str(SITE / (name + '-' + version + '.dist-info') / 'RECORD')
        require(row['record']['path'] == expected and source_sha256.get(expected) == row['record']['sha256'],
                'Frozen RECORD is in reviewed source metadata map')
        packages[name] = version
        for path, entry in record_payloads(read_record(row['record'])).items():
            require(path not in payloads or payloads[path] == entry, 'No conflicting installed payload')
            payloads[path] = entry
    require(all(packages.get(name) == version for name, version in REQUIRED.items()),
            'Original Python 3.10 native packages plus all new real SDK/vLLM roots')
    for path, entry in manifest['python_overrides'].items():
        require(path in payloads and Path(path).suffix == '.py' and source_sha256.get(path) == entry['sha256'],
                'Only independently reviewed Python edits may override wheel RECORD')
        payloads[path] = copy.deepcopy(entry)
    for path, entry in manifest['additional_files'].items():
        require(Path(path).suffix == '.py' and source_sha256.get(path) == entry['sha256'],
                'Editable/official/worker source must have an independent reviewed source pin')
        require(path not in payloads or payloads[path] == entry, 'Additional source cannot replace a payload')
        payloads[path] = copy.deepcopy(entry)
    for path, entry in manifest['linked_native_files'].items():
        require('.so' in Path(path).name and path not in payloads, 'Explicit extra linked native library')
        payloads[path] = copy.deepcopy(entry)
    require(0 < len(payloads) <= MAX_MANIFEST_ENTRIES and len(payloads) == manifest['payload_count']
            and sum(row['bytes'] for row in payloads.values()) == manifest['payload_bytes'] <= MAX_RUNTIME_AUTHENTICATION_READ
            and contract.digest(payloads) == manifest['payload_sha256'], 'Complete finite reconstructed manifest')
    require(all(payloads.get(path, {}).get('sha256') == sha for path, sha in contract.NATIVE_SOURCES.items()),
            'Every fixed diagnostic/official/host/SDK source in actual payload manifest')
    return payloads


def mapped_native_files(raw, authenticated, observe_identity=identity):
    """Pure parse + injected stat: mapped dev/inode must be the prehashed object.

    A same-name replacement on disk cannot authenticate an older loaded object.
    /lib and /usr/lib are the explicitly stated OS trust boundary; their native
    identities are not claimed by this experiment's application/SDK manifest.
    """
    require(type(raw) is str and len(raw.encode()) <= 2*2**20, 'Bounded native mapping text')
    mapped = {}
    for line in raw.splitlines():
        fields = line.split(None, 5)
        if len(fields) < 6 or not fields[5].startswith('/') or '.so' not in Path(fields[5]).name:
            continue
        require(not fields[5].endswith(' (deleted)') and '\x00' not in fields[5],
                'Deleted native mapping is not an authenticated live file')
        require(re.fullmatch('[0-9a-fA-F]+:[0-9a-fA-F]+', fields[3]) and fields[4].isdigit()
                and int(fields[4]) > 0, 'Actual native mapping device/inode fields')
        major, minor = (int(part, 16) for part in fields[3].split(':'))
        mapped_identity = [os.makedev(major, minor), int(fields[4])]
        path = Path(fields[5]).resolve()
        if path.is_relative_to(Path('/usr/lib')) or path.is_relative_to(Path('/lib')):
            continue
        before = authenticated.get(str(path))
        require(before is not None and before[:2] == mapped_identity
                and observe_identity(path) == before,
                'Loaded native dev/inode and current file must match the same prehashed object')
        observation = {'device': mapped_identity[0], 'inode': mapped_identity[1],
                       'file_identity': list(before)}
        require(str(path) not in mapped or mapped[str(path)] == observation,
                'All mappings of one native pathname name the same immutable object')
        mapped[str(path)] = observation
    require(mapped, 'Actual authenticated non-system native mapping required')
    return mapped


class OriginFinder:
    """Delegate to real finders, but authenticate origin before create/exec."""
    def __init__(self, runtime, finders):
        self.runtime, self.finders = runtime, tuple(finders)

    def find_spec(self, fullname, path=None, target=None):
        for finder in self.finders:
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                self.runtime.check_origin(spec.origin, spec.submodule_search_locations)
                return spec
        return None


class Runtime:
    """Constructed by WorkerContext only; never trusts an authority-only dict."""
    def __init__(self, context, manifest):
        from spark26_native_serial_worker_v2 import WorkerContext
        require(type(context) is WorkerContext, 'Concrete Spark worker context required')
        self.context, self.manifest = context, manifest
        self.payloads, self.identities, self.finder = {}, {}, None
        self.native_observed = False
        self.stdlib = Path(sysconfig.get_path('stdlib')).resolve()

    def authenticate(self):
        self.context.check()
        require(not {'torch', 'numpy', 'transformers', 'torch_xla', 'torch_neuronx',
                     'neuronx_distributed', 'vllm'} & set(sys.modules), 'Authenticate before any native imports')
        self.payloads = merge_manifest(self.manifest, self.context.spec['source_sha256'],
                                       lambda ref: self.context.metadata_bytes(ref).decode('utf8'))
        for path, expected in self.payloads.items():
            self.identities[path] = authenticate_file(path, expected, self.context.check)
        self.context.check()
        self.finder = OriginFinder(self, sys.meta_path)
        sys.meta_path[:] = [self.finder]
        for module in tuple(sys.modules.values()):
            spec = getattr(module, '__spec__', None)
            if spec is not None:
                self.check_origin(spec.origin, spec.submodule_search_locations)
        return {'files': len(self.payloads), 'bytes': self.manifest['payload_bytes'],
                'manifest_sha256': contract.digest(self.manifest),
                'scope': 'Declared application/SDK bytes; system Python/stdlib is the host trust boundary'}

    def check_origin(self, origin, namespace=None):
        self.context.check()
        if origin in ('built-in', 'frozen'):
            return
        if origin is None:
            require(namespace is not None and all(any(Path(p).is_relative_to(Path(name).parent)
                    or Path(name).is_relative_to(Path(p)) for name in self.payloads) for p in namespace),
                    'Only declared namespace paths')
            return
        path = Path(origin).resolve()
        # No site-packages may hide inside the stdlib trust boundary.
        if path.is_relative_to(self.stdlib) and 'site-packages' not in path.parts and 'dist-packages' not in path.parts:
            require(path.suffix != '.pyc', 'Isolated stdlib source/native imports only')
            return
        require(str(path) in self.identities and identity(path) == self.identities[str(path)]
                and path.suffix != '.pyc', 'No undeclared/changed third-party import: ' + str(path))

    def check_unchanged(self):
        self.context.check()
        require(sys.meta_path == [self.finder], 'Native code may not bypass the reviewed import-origin guard')
        self.check_identities()
        if self.native_observed:
            self.native_mappings()

    def check_identities(self):
        for path, before in self.identities.items():
            require(identity(path) == before, 'Authenticated runtime file changed')

    def native_mappings(self):
        """Before scientific reads, check actual loaded non-system shared objects.

        ELF system dependencies in /lib and /usr/lib are the same stated OS trust
        boundary as Python. Third-party dlopen dependencies must be prehashed.
        This is an observation of imports, not protection against arbitrary ELF
        initializer code; the root-approved dependency closure remains required.
        """
        with Path('/proc/self/maps').open() as stream:
            raw = stream.read(2*2**20+1)
        require(len(raw) <= 2*2**20, 'Bounded native mapping observation')
        mapped = mapped_native_files(raw, self.identities)
        self.native_observed = True
        return mapped


def load_official(runtime):
    """Same actual official-module loading mechanism as the successful meta worker."""
    name = '_spark26_pinned_' + contract.original.REVISION
    require(name not in sys.modules, 'Fresh official pinned package')
    package = types.ModuleType(name)
    package.__path__ = [str(contract.original.SOURCE_DIR)]
    sys.modules[name] = package
    loaded = []
    for suffix, path in [('configuration_spark', contract.original.CONFIGURATION),
                         ('modeling_spark', contract.original.MODELING)]:
        runtime.check_origin(str(path))
        spec = importlib.util.spec_from_file_location(name + '.' + suffix, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        loaded.append(module)
    return loaded


def native_providers(runtime):
    """Actual imports only. No mock providers, AST extraction, or port registration."""
    require(type(runtime) is Runtime and runtime.finder is not None, 'Completed concrete runtime authentication')
    runtime.check_unchanged()
    import torch
    from safetensors import safe_open
    from neuronx_distributed.trace.trace import create_local_weight_qkv
    from vllm_neuron.worker.neuronx_distributed_model_loader import NeuronModelBase
    import spark26_text_port_v2 as candidate
    configuration, modeling = load_official(runtime)
    require(torch.__version__.split('+')[0] == '2.8.0', 'Original Torch version')
    require(torch.get_default_device().type == 'cpu' and not torch.distributed.is_initialized(),
            'Single-process CPU diagnostic; no XLA default or initialized collective')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    runtime.check_unchanged()
    result = dict(torch=torch, safe_open=safe_open, configuration=configuration, modeling=modeling,
                  candidate=candidate, create_local_weight_qkv=create_local_weight_qkv,
                  NeuronModelBase=NeuronModelBase)
    import spark26_native_port_diagnostic_v3 as diagnostic
    diagnostic.validate_native_objects(result)
    runtime.native_mappings()
    return result
