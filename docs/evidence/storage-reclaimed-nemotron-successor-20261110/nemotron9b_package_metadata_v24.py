"""Bounded metadata transport for package v5; no framework/payload reads.

Ordinary UTF8 text remains 8 MiB. Only the enumerated preparation/runtime
JSON namespaces and structural schemas may use 32 MiB. The three original
non-UTF8 Python fixtures have a separate exact-path/exact-hash byte reader.
Historical raw pins are retained and proven from metadata, never opened here.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import minicpm5_layer0_execution_protocol_v4 as low
import historical_b3_payload_partition_v2 as history
import nemotron9b_completed_smol_metadata_v1 as completed_smol
import nemotron9b_post_smol_maintenance_v2 as maintenance
import nemotron9b_package_metadata_v23 as previous_metadata
from research import nemotron9b_completed_storage_metadata_after_retry_v2 as storage_metadata
from research import nemotron9b_residual_source_partition_v1 as residual_sources
from research import nemotron9b_completed_saved_review_metadata_v1 as saved_metadata
from research import dory_saved_retry_source_partition_v1 as query_sources
import nemotron9b_post_dory_history_v1 as latest_history
import nemotron9b_post_mini_history_v1 as mini_history

ROOT = Path(__file__).resolve().parent
RESEARCH = ROOT/'research'
LIMIT = 8*2**20
ENVELOPE_LIMIT = 32*2**20
LIMITS = dict(ordinary_file_bytes=LIMIT, named_json_envelope_bytes=ENVELOPE_LIMIT)
TEXT_SUFFIXES = frozenset(('.py', '.json', '.md', '.txt', '.jinja', '.log', '.jsonl',
                          '.patch', '.yaml', '.toml', '.ini', '.typed'))
TEXT_BASENAMES = frozenset(('.gitignore', 'LICENSE', 'METADATA', 'RECORD', 'WHEEL', 'PKG-INFO'))
SPECIAL_TEXT_PINS = {
    '/home/ubuntu/venvs/model_reference_2026/lib/python3.12/site-packages/__editable__.transformers-5.23.0.dev0.pth':
        'aa13fbb34e302ad14d5377603770dc22ca996b89b3dc92e3ac08fa33f0249f8f',
    '/mnt/nvme/models/gemma-4-E2B-it/.cache/huggingface/download/model.safetensors.metadata':
        'e5a2aeeeb239bae1c6cad98c36ac7842d5c4311287496321039d94ff7df41f88',
}
NON_UTF8 = {
    '/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/IPython/core/tests/nonascii.py':
        '74ba44a36438f5d10c9a8d777f8f5a7381f2d738b1883b155de85acc125f8b9e',
    '/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/IPython/core/tests/nonascii2.py':
        'aaecd61ad7eca1cb868fff6c83e75f131367fad3cde836b20e208faeee2c0e3e',
    '/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/joblib/test/test_func_inspect_special_encoding.py':
        'e7120b0e348efb1b6340030bbcc7950fe2fb206e195116f682f0624a168313bf',
}
ENVELOPE_SCHEMAS = {
    'nemotron9b_package_preparation_request_v2': {'source_sha256': dict, 'stages': list, 'disk_budget': dict},
    'nemotron9b_package_preparation_source_review_v2': {'source_sha256': dict, 'request': dict, 'reviewer': str},
    'nemotron9b_package_preparation_capture_v1': {'source_sha256': dict, 'runtime_python_source_sha256': dict,
        'runtime_binary_stat': dict, 'runtime_binary_sha256': dict, 'physical_drain': dict, 'payloads': dict},
    'nemotron9b_package_runtime_v4': {'source_sha256': dict, 'runtime_python_source_sha256': dict,
        'runtime_binary_stat': dict, 'payloads': dict, 'runtime_readiness': dict},
    'nemotron9b_package_runtime_v7': {'source_sha256': dict, 'runtime_python_source_sha256': dict, 'runtime_binary_stat': dict, 'payloads': dict, 'runtime_readiness': dict, 'metadata_limits': dict},
    'nemotron9b_package_runtime_source_review_v7': {'source_sha256': dict, 'execution_spec_sha256': str, 'reviewer': str, 'blockers': list, 'source_only': bool, 'independent_review': bool, 'passed': bool},
    'nemotron9b_package_runtime_v8': {'source_sha256': dict, 'runtime_python_source_sha256': dict,
        'runtime_binary_stat': dict, 'payloads': dict, 'runtime_readiness': dict, 'metadata_limits': dict},
    'nemotron9b_package_runtime_source_review_v8': {'source_sha256': dict, 'execution_spec_sha256': str,
        'reviewer': str, 'blockers': list, 'source_only': bool, 'independent_review': bool, 'passed': bool},
    'nemotron9b_package_runtime_v9': {'source_sha256': dict, 'runtime_python_source_sha256': dict,
        'runtime_binary_stat': dict, 'payloads': dict, 'runtime_readiness': dict, 'metadata_limits': dict},
    'nemotron9b_package_runtime_source_review_v9': {'source_sha256': dict, 'execution_spec_sha256': str,
        'reviewer': str, 'blockers': list, 'source_only': bool, 'independent_review': bool, 'passed': bool},
    'nemotron9b_package_runtime_v10': {'source_sha256': dict, 'runtime_python_source_sha256': dict,
        'runtime_binary_stat': dict, 'payloads': dict, 'runtime_readiness': dict, 'metadata_limits': dict},
    'nemotron9b_package_runtime_source_review_v10': {'source_sha256': dict, 'execution_spec_sha256': str,
        'reviewer': str, 'blockers': list, 'source_only': bool, 'independent_review': bool, 'passed': bool},
    'nemotron9b_package_runtime_v5': {'source_sha256': dict, 'runtime_python_source_sha256': dict,
        'runtime_binary_stat': dict, 'payloads': dict, 'runtime_readiness': dict, 'metadata_limits': dict},
    'nemotron9b_package_runtime_source_review_v5': {'source_sha256': dict, 'execution_spec_sha256': str,
        'reviewer': str, 'blockers': list, 'source_only': bool, 'independent_review': bool, 'passed': bool},
    'nemotron9b_package_runtime_v6': {'source_sha256': dict, 'runtime_python_source_sha256': dict,
        'runtime_binary_stat': dict, 'payloads': dict, 'runtime_readiness': dict, 'metadata_limits': dict},
    'nemotron9b_package_runtime_source_review_v6': {'source_sha256': dict, 'execution_spec_sha256': str,
        'reviewer': str, 'blockers': list, 'source_only': bool, 'independent_review': bool, 'passed': bool},
    'root_post_nemotron_package_complete_predecessors_v1': {'source_sha256': dict, 'rows': list, 'processes': list},
    'root_post_unified_native_complete_predecessors_v1': {'source_sha256': dict, 'rows': list, 'processes': list},
}
ENVELOPE_SCHEMAS.update({
    'nemotron9b_package_runtime_v11': dict(ENVELOPE_SCHEMAS['nemotron9b_package_runtime_v10']),
    'nemotron9b_package_runtime_source_review_v11': dict(ENVELOPE_SCHEMAS['nemotron9b_package_runtime_source_review_v10']),
})
ENVELOPE_SCHEMAS.update({
    'nemotron9b_package_runtime_v24': dict(ENVELOPE_SCHEMAS['nemotron9b_package_runtime_v11']),
    'nemotron9b_package_runtime_source_review_v24': dict(ENVELOPE_SCHEMAS['nemotron9b_package_runtime_source_review_v11']),
})
ENVELOPE_SCHEMAS.update({
    'nemotron9b_package_runtime_v12': dict(ENVELOPE_SCHEMAS['nemotron9b_package_runtime_v11']),
    'nemotron9b_package_runtime_source_review_v12': dict(ENVELOPE_SCHEMAS['nemotron9b_package_runtime_source_review_v11']),
})
require = low.require


ENVELOPE_SCHEMAS = dict(previous_metadata.ENVELOPE_SCHEMAS, **ENVELOPE_SCHEMAS)

def ordinary_path(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and
            not any(item.is_symlink() for item in (path, *path.parents)), 'Canonical ordinary metadata path')
    return path


def research_envelope(path):
    path = Path(path)
    return path.parent == RESEARCH and re.fullmatch(r'nemotron9b-package-[a-z0-9][a-z0-9.-]*\.json', path.name) is not None


def runtime_envelope(path):
    """Exact output classes that carry the unchanged complete source map."""
    path = Path(path)
    if path.parent.parent == ROOT and re.fullmatch(r'nemotron9b-package-[a-z0-9-]{1,40}-control', path.parent.name):
        match = re.fullmatch(r'([a-z0-9-]+)-raw-manifest\.json', path.name)
        if match: return 'phase_manifest', match[1]
        match = re.fullmatch(r'((?:alias|split)-[a-z0-9-]+)-worker-result\.json', path.name)
        if match: return 'primitive_worker', match[1]
    if path.parent.parent == ROOT and re.fullmatch(r'nemotron9b-package-[a-z0-9-]{1,40}-evidence', path.parent.name):
        if path.name in ('state-copy.json', 'cold-buckets.json', 'final-raw-manifest.json'):
            return path.name, None
    parts = path.parts
    if len(parts) >= 7 and path.is_relative_to('/mnt/nvme'):
        relative = path.relative_to('/mnt/nvme').parts
        if re.fullmatch(r'nemotron9b-package-[a-z0-9-]{1,40}', relative[0]):
            if len(relative) == 4 and relative[1:3] == ('primitives', 'device') and re.fullmatch(r'alias-[a-z0-9-]+\.json', relative[3]):
                return 'alias_report', path.stem
            if len(relative) == 5 and relative[1:3] == ('primitives', 'jobs') and re.fullmatch(r'split-[a-z0-9-]+', relative[3]) and relative[4] == 'result.json':
                return 'split_report', relative[3]
    return None


def pin_shape(value):
    require(type(value) is dict and len(value) <= 100000, 'Bounded source map')
    history.pin_shape(value)


def fields(value, shape):
    require(type(value) is dict and all(type(value.get(key)) is kind for key, kind in shape.items()),
            'Explicit metadata structural fields')


def envelope_shape(path, value):
    if research_envelope(path):
        require(type(value) is dict and value.get('schema') in ENVELOPE_SCHEMAS, 'Explicit named research envelope schema')
        fields(value, ENVELOPE_SCHEMAS[value['schema']])
        for key in ('source_sha256', 'runtime_python_source_sha256', 'runtime_binary_sha256'):
            if key in value: pin_shape(value[key])
        return True
    kind_stage = runtime_envelope(path)
    require(kind_stage is not None, 'Exact runtime envelope namespace')
    kind, stage = kind_stage
    if kind == 'primitive_worker':
        fields(value, dict(complete=bool, passed=bool, phase=str, producer=dict, lease=dict, result=dict))
        require(value['phase'] == stage, 'Worker filename/phase binding')
        report = value['result']
        fields(report, dict(source_sha256=dict, job=dict, complete=bool))
        require(report['job'].get('name') == stage, 'Worker primitive job binding')
        pin_shape(report['source_sha256'])
        return True
    fields(value, dict(source_sha256=dict, complete=bool))
    pin_shape(value['source_sha256'])
    if kind == 'phase_manifest':
        fields(value, dict(schema=str, phase=str, producer=dict, lease=dict, execution_spec=dict, files=dict))
        require(value['schema'] == 'nemotron_phase_raw_manifest_v3' and value['phase'] == stage, 'Original phase manifest schema/phase')
    elif kind in ('alias_report', 'split_report'):
        fields(value, dict(job=dict, checks=list) if kind == 'alias_report' else dict(job=dict, cpu_checks=list, device_checks=list))
        require(value['job'].get('name') == stage, 'Primitive filename/job binding')
    elif kind == 'state-copy.json':
        fields(value, dict(passed=bool, patterns=list))
    elif kind == 'cold-buckets.json':
        fields(value, dict(reset_source_sha256=dict, criteria=dict, experiments=list, state_reset_receipts=list))
        require(value['reset_source_sha256'] == value['source_sha256'], 'Unchanged reset source closure')
    elif kind == 'final-raw-manifest.json':
        fields(value, dict(producer=dict, lease=dict, execution_spec=dict, source_review=dict, scientific_results=dict, closure=dict))
    else: raise RuntimeError('Unknown runtime envelope class')
    return True


def stable_bytes(path, cap):
    """Caller establishes an explicit text class before any file is opened."""
    path = ordinary_path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_size <= cap, 'Ordinary bounded metadata')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            raw = stream.read(cap + 1)
        after, current = os.fstat(fd), path.stat()
        keys = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
        require(len(raw) == before.st_size and len(raw) <= cap and all(
            getattr(before, key) == getattr(after, key) == getattr(current, key) for key in keys), 'Metadata identity/length changed')
        return raw
    finally:
        os.close(fd)


def text(path, expected=None):
    low.clear_window()
    path = Path(path)
    if saved_metadata.accepts(path):return saved_metadata.text(path,expected,stable_bytes)
    if storage_metadata.accepts(path):return storage_metadata.text(path,expected,stable_bytes)
    if residual_sources.accepts(path):return residual_sources.text(path,expected)
    if latest_history.accepts(path):return latest_history.text(path,expected)
    if path in (mini_history.CURRENT,mini_history.prior.PATH,mini_history.prior.previous.PATH,mini_history.prior.previous.previous.PATH):
        return mini_history.text(path,expected)
    if completed_smol.accepts(path):
        return completed_smol.source_bytes(path, expected, stable_bytes)
    if maintenance.accepts(path):
        return maintenance.source_bytes(path, expected, stable_bytes)
    require(path.suffix in TEXT_SUFFIXES or (not path.suffix and path.name in TEXT_BASENAMES) or
            (str(path) in SPECIAL_TEXT_PINS and expected == SPECIAL_TEXT_PINS[str(path)]), 'Reviewed UTF8 text only')
    cap = ENVELOPE_LIMIT if research_envelope(path) or runtime_envelope(path) else LIMIT
    raw = stable_bytes(path, cap)
    raw.decode('utf8')
    require(expected is None or hashlib.sha256(raw).hexdigest() == expected, 'Metadata fingerprint changed: '+str(path))
    if len(raw) > LIMIT: envelope_shape(path, json.loads(raw))
    return raw


def sha(path, expected=None):
    return hashlib.sha256(text(path, expected)).hexdigest()


def read(path, wanted=None):
    return json.loads(text(path, wanted))


def load(ref):
    require(type(ref) is dict and set(ref) in ({'path', 'sha256'}, {'path', 'sha256', 'bytes'}), 'Exact metadata reference')
    require(type(ref['path']) is str and type(ref['sha256']) is str and re.fullmatch('[0-9a-f]{64}', ref['sha256']), 'Literal metadata reference')
    require('bytes' not in ref or type(ref['bytes']) is int and ref['bytes'] >= 0, 'Nonnegative integer reference bytes')
    raw = text(ref['path'], ref['sha256'])
    require('bytes' not in ref or len(raw) == ref['bytes'], 'Metadata reference byte count')
    return json.loads(raw)


def python_source_bytes(path, expected):
    low.clear_window()
    require(NON_UTF8.get(str(path)) == expected, 'Only exact three original non-UTF8 Python fixtures')
    raw = stable_bytes(path, LIMIT)
    require(hashlib.sha256(raw).hexdigest() == expected, 'Non-UTF8 Python source fingerprint changed')
    try: raw.decode('utf8')
    except UnicodeDecodeError: return raw
    raise RuntimeError('Original non-UTF8 fixture encoding changed')


def classify_sources(spec, catalog):
    """Exact original14 and admitted Dory6 historical pins, no payload reads."""
    return query_sources.classify_sources(spec,catalog)


def encode(path, value):
    path = ordinary_path(path)
    registry_intent = path == ROOT/'nemotron9b-package-window-v3.json.next'
    require(path.suffix == '.json' or registry_intent, 'JSON metadata output or exact registry intent only')
    if registry_intent:
        fields(value, dict(schema=str, gate=dict, state=str, performance_window=bool))
        require(value['schema'] == 'nemotron_external_window_v3', 'Original observer registry intent schema')
    raw = (json.dumps(value, indent=2, allow_nan=False)+'\n').encode()
    cap = ENVELOPE_LIMIT if research_envelope(path) or runtime_envelope(path) else LIMIT
    require(len(raw) <= cap and (not registry_intent or len(raw) < 16384), 'Bounded metadata output')
    if len(raw) > LIMIT: envelope_shape(path, value)
    return raw


def sync_parent(path):
    fd = os.open(Path(path).parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try: os.fsync(fd)
    finally: os.close(fd)


def once(path, value):
    low.clear_window()
    raw = encode(path, value)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    sync_parent(path)


def save(path, value):
    low.clear_window()
    path = Path(path); raw = encode(path, value)
    temporary = ordinary_path(path.with_suffix(path.suffix+'.tmp'))
    if temporary.exists():
        info = temporary.lstat()
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, 'Ordinary atomic metadata temporary')
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path); sync_parent(path)
