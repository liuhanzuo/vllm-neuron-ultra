"""Explicit package candidate preparation; stdlib/source only, never admission.

No latest lookup, file writes, History, lock, payload reads or native import.
The caller selects every binding and supplies guarded authenticated readers.
The only review returned has passed=False and launch_authorized=False.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import stat

from qwen35_package_serial_runtime_v2 import contract as c

IDENTITY_KEYS = {'realpath', 'device', 'inode', 'size', 'mtime_ns', 'ctime_ns'}
MIB = 1024**2
ORDINARY_CAP, ENVELOPE_CAP = 8*MIB, 64*MIB
SELF = str(Path(__file__).resolve())
TEXT_SUFFIXES = {'.py', '.json', '.md', '.txt', '.patch', '.sh', '.toml', '.cfg', '.ini', '.jinja'}
REQUEST_KEYS = {'seed_spec', 'seed_review', 'seed_actual_exit', 'current_catalog',
    'idle_handoff', 'runtime_directory', 'runtime_delivery', 'runtime_review', 'resource_budget', 'case', 'binary_inputs',
    'preparation_sources', 'required_helpers', 'extra_metadata', 'future'}
FUTURE_KEYS = {'run_name', 'output_root', 'overlay_output_root', 'running_handoff_path',
    'overlay_path', 'spec_path', 'review_candidate_path'}


def require(value, message):
    if not value:
        raise ValueError(message)


def encode(value, cap=ENVELOPE_CAP):
    payload = (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2,
                          allow_nan=False)+'\n').encode()
    require(len(payload) <= cap, 'original finite envelope exceeded')
    return payload


def reference(value):
    return c.reference(value)


def short_ref(value):
    require(type(value) is dict and set(value) in ({'path', 'sha256'}, {'path', 'sha256', 'bytes'}),
            'exact metadata or sized source reference required')
    return reference({k: value[k] for k in ('path', 'sha256')})


def identity_shape(path, value):
    require(type(value) is dict and set(value) == IDENTITY_KEYS
            and value['realpath'] == path, 'complete exact stat-only identity required')
    require(all(type(value[k]) is int and value[k] >= 0 for k in IDENTITY_KEYS-{'realpath'}),
            'finite nonnegative identity numbers required')


def file_identity(path):
    """Stat only; never opens model/checkpoint/graph payloads."""
    path = Path(path)
    require(path.is_absolute() and str(path) == str(path.resolve())
            and not any(p.is_symlink() for p in (path, *path.parents)), 'canonical ordinary input path required')
    s = path.lstat()
    require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1, 'singleton ordinary input required')
    return dict(realpath=str(path), device=s.st_dev, inode=s.st_ino, size=s.st_size,
                mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key: '+key)
        result[key] = value
    return result


class StableReaders:
    """Optional future root reader. Construction/import performs no IO.

    prepare explicitly selects 64 MiB only for seed spec/review and base plan.
    All other metadata uses 8 MiB. No aliases, payload decoding or retries that
    refresh an already observed identity. Full historical alias handling is
    retained in the unchanged runtime History, never replaced by this helper.
    """
    def __init__(self, guard):
        self.guard, self.observed = guard, {}

    def source(self, ref, cap=ORDINARY_CAP):
        reference(ref)
        require(Path(ref['path']).suffix in TEXT_SUFFIXES, 'explicit text sources only; never open binary payloads')
        require(type(cap) is int and 0 < cap <= ENVELOPE_CAP, 'finite explicit read cap')
        self.guard()
        before = file_identity(ref['path'])
        require(before['size'] <= cap, 'input exceeds original finite envelope')
        fd = os.open(ref['path'], os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            opened = os.fstat(fd)
            require((opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns)
                    == tuple(before[k] for k in ('device', 'inode', 'size', 'mtime_ns', 'ctime_ns')),
                    'opened input differs from observed identity')
            parts, count = [], 0
            while count <= cap:
                part = os.read(fd, min(MIB, cap+1-count))
                if not part:
                    break
                parts.append(part)
                count += len(part)
            payload = b''.join(parts)
        finally:
            os.close(fd)
        require(len(payload) == before['size'] and file_identity(ref['path']) == before
                and hashlib.sha256(payload).hexdigest() == ref['sha256'], 'complete stable authenticated read required')
        row = dict(sha256=ref['sha256'], identity=before)
        require(ref['path'] not in self.observed or self.observed[ref['path']] == row,
                'observed input cannot be rebound')
        self.observed[ref['path']] = copy.deepcopy(row)
        self.guard()
        return dict(payload=payload, identity=before)

    def metadata(self, ref, cap=ORDINARY_CAP):
        require(Path(ref['path']).suffix == '.json', 'JSON metadata only')
        saved = self.source(ref, cap)
        value = json.loads(saved['payload'], object_pairs_hook=_unique,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError('finite JSON only')))
        return dict(value=value, identity=saved['identity'])


def merge_sources(*maps):
    result = {}
    for values in maps:
        require(type(values) is dict, 'source map required')
        for path, sha in values.items():
            reference(dict(path=path, sha256=sha))
            require(path not in result or result[path] == sha, 'source pin conflict: '+path)
            result[path] = sha
    require(0 < len(result) <= c.LIMITS['source_map_entries'], 'original source map cap')
    return result


def merge_inputs(target, addition, *, json_only=True):
    require(type(addition) is dict, 'metadata map required')
    for path, row in addition.items():
        require(type(row) is dict and set(row) == {'sha256', 'identity'}
                and (not json_only or Path(path).suffix == '.json'),
                'all extension inputs must be JSON; do not drop incompatible evidence')
        reference(dict(path=path, sha256=row['sha256']))
        identity_shape(path, row['identity'])
        require(path not in target or target[path] == row, 'immutable metadata hash/identity conflict: '+path)
        target[path] = copy.deepcopy(row)


def catalog_shape(value):
    require(type(value) is dict and set(value) == {'schema', 'reviewer', 'passed', 'blockers',
            'completed_sdk', 'current_handoff', 'all_later_serial_runs_included', 'runs'}
            and value['schema'] == 'minicpm5_operator_later_serial_metadata_catalog_v1'
            and value['reviewer'] == '/root' and value['passed'] is True and value['blockers'] == []
            and value['all_later_serial_runs_included'] is True and type(value['runs']) is list
            and 0 < len(value['runs']) <= 256, 'complete original catalog contract required')
    reference(value['current_handoff'])
    result = {}
    for row in value['runs']:
        require(type(row) is dict and set(row) == {'terminal', 'tool', 'owners', 'units', 'source_sha256'},
                'complete unchanged serial row required')
        reference(row['terminal'])
        path = row['terminal']['path']
        require(path not in result, 'duplicate serial terminal')
        require(type(row['source_sha256']) is dict, 'catalog source extension required')
        result[path] = row
    return result


def manifest_pins(value):
    require(type(value) is dict and value, 'complete source manifest required')
    return merge_sources({path: row['sha256'] if type(row) is dict else row for path, row in value.items()})


def _future(future, seed, overlay, forbidden):
    require(type(future) is dict and set(future) == FUTURE_KEYS, 'explicit future output bindings required')
    c.units(future['run_name'])
    require(future['run_name'] != seed['run_name'] and future['output_root'] != seed['output_root']
            and future['output_root'] == '/mnt/nvme/neuron-work/qwp-'+future['run_name'], 'fresh exact package output namespace')
    path = Path(future['overlay_output_root'])
    require(path.parent == Path('/mnt/nvme') and re.fullmatch('minicpm5-layer1-operator-native-[a-z0-9-]{1,64}', path.name)
            and str(path) == future['overlay_output_root'] and future['overlay_output_root'] != overlay['output_root'],
            'fresh original overlay namespace')
    paths = []
    for key in ('running_handoff_path', 'overlay_path', 'spec_path', 'review_candidate_path'):
        path = Path(future[key])
        require(path.parent == c.Q and str(path) == future[key]
                and re.fullmatch('[a-z0-9][a-z0-9-]*\\.json', path.name), 'exact research JSON output path')
        paths.append(str(path))
    require(len(set(paths)) == len(paths) and not set(paths) & forbidden
            and seed['running_handoff_path'] not in paths, 'future paths cannot overwrite retained metadata')


def prepare(request, read_metadata, read_source, identity_only, guard):
    """Return objects/bytes/refs only; root owns publication and final review.

    Readers: metadata(ref, cap)->{value,identity}; source(ref,cap)->{payload,identity}.
    identity_only(path) performs stat only. guard() checks caller's independent
    source-only preparation window; it is NOT a runtime admission callback.
    """
    require(type(request) is dict and set(request) == REQUEST_KEYS, 'exact explicit preparation request required')
    require(all(callable(x) for x in (read_metadata, read_source, identity_only, guard)), 'explicit guarded readers required')
    request = copy.deepcopy(request)
    observed, source_observed, preparation_only = {}, {}, set()
    def read(ref, cap=ORDINARY_CAP):
        reference(ref)
        require(Path(ref['path']).suffix == '.json', 'metadata inputs must be JSON')
        guard()
        saved = read_metadata(copy.deepcopy(ref), cap)
        require(type(saved) is dict and set(saved) == {'value', 'identity'}, 'exact stable metadata reader receipt')
        identity_shape(ref['path'], saved['identity'])
        require(saved['identity']['size'] <= cap, 'original metadata cap exceeded')
        merge_inputs(observed, {ref['path']: dict(sha256=ref['sha256'], identity=saved['identity'])})
        if cap == ENVELOPE_CAP:
            preparation_only.add(ref['path'])
        guard()
        return copy.deepcopy(saved['value'])

    def source(ref):
        reference(ref)
        require(Path(ref['path']).suffix in TEXT_SUFFIXES, 'explicit text sources only; never open binary payloads')
        guard()
        saved = read_source(copy.deepcopy(ref), ORDINARY_CAP)
        require(type(saved) is dict and set(saved) == {'payload', 'identity'} and type(saved['payload']) is bytes,
                'exact stable text source reader receipt')
        identity_shape(ref['path'], saved['identity'])
        require(len(saved['payload']) == saved['identity']['size'] <= ORDINARY_CAP
                and hashlib.sha256(saved['payload']).hexdigest() == ref['sha256'], 'complete authenticated source')
        require(b'\x00' not in saved['payload'], 'text source only')
        saved['payload'].decode('utf-8')
        row = dict(sha256=ref['sha256'], identity=saved['identity'])
        require(ref['path'] not in source_observed or source_observed[ref['path']] == row,
                'source observation cannot refresh an identity')
        source_observed[ref['path']] = copy.deepcopy(row)
        guard()

    seed = read(request['seed_spec'], ENVELOPE_CAP)
    seed_review = read(request['seed_review'], ENVELOPE_CAP)
    actual = read(request['seed_actual_exit'])
    require(seed.get('schema') == 'qwen35_sampler_runtime_spec_v1'
            and seed_review.get('schema') == 'qwen35_sampler_runtime_source_review_v1'
            and seed_review.get('reviewer') == '/root' and seed_review.get('passed') is True
            and seed_review.get('launch_authorized') is True and seed_review.get('blockers') == []
            and seed_review.get('spec') == request['seed_spec']
            and seed_review.get('source_sha256') == seed['source_sha256'], 'executed seed source admission must match exactly')
    require(actual.get('reviewer') == '/root' and actual.get('spec') == request['seed_spec']
            and actual.get('review') == request['seed_review'] and actual.get('actual_tool_exit_observed') is True
            and actual.get('actual_exit_code') == 0 and actual.get('actual_tool', {}).get('exit_code') == 0
            and all(actual.get(k) is True for k in ('owners_gone', 'physically_drained', 'full_model_qualified', 'paired_throughput_qualified')),
            'actual independently observed successful seed required')
    require(seed['limits'] == c.LIMITS and seed['native_python'] == c.PYTHON
            and seed['lock_identity'] == c.LOCK_IDENTITY and seed['automatic_successor'] is False,
            'unchanged scientific limits, original lock and interpreter required')
    overlay = read(seed['history_overlay'])
    require(set(overlay) == {'schema', 'base_plan', 'sdk_operator_recovery', 'current_serial_catalog',
            'native_manifest', 'source_extension', 'input_extension', 'output_root'}
            and overlay['schema'] == 'minicpm5_layer1_operator_native_small_overlay_v2'
            and overlay['current_serial_catalog'] == seed['current_serial_catalog'], 'exact executed overlay required')
    base = read(overlay['base_plan'], ENVELOPE_CAP)
    old_catalog = read(seed['current_serial_catalog'])
    catalog = read(request['current_catalog'])
    catalog_shape(old_catalog)
    rows = catalog_shape(catalog)
    require(catalog['runs'][:len(old_catalog['runs'])] == old_catalog['runs']
            and catalog['completed_sdk'] == old_catalog['completed_sdk']
            and catalog['current_handoff'] == request['idle_handoff'], 'all historical rows must remain an exact ordered prefix')
    exit_row = rows.get(request['seed_actual_exit']['path'])
    require(exit_row is not None and exit_row['terminal'] == request['seed_actual_exit']
            and exit_row['tool'] == actual['actual_tool'] and exit_row['owners'] == actual['owners']
            and exit_row['units'] == actual['all_declared_units'], 'current catalog must retain actual executed seed row unchanged')
    idle = read(request['idle_handoff'])
    require(idle.get('serial_execution') is True and idle.get('running') is False
            and idle.get('live_tool_sessions') == [] and idle.get('active_heavy_units') == [], 'explicit root idle handoff required')
    read(seed['lfm_r2_scope'])

    all_inputs = {}
    # Base includes original binary/source receipts. Preserve these verbatim in
    # the base reference; do not read or migrate them into JSON extensions.
    merge_inputs(all_inputs, base['inputs'], json_only=False)
    for values in (overlay['input_extension'], seed['input_extension']):
        merge_inputs(all_inputs, values)
    require(not set(base['inputs']) & set(overlay['input_extension']), 'original overlay may not duplicate base inputs')
    old_sources = merge_sources(seed['source_sha256'], overlay['source_extension'], base['source_sha256'])
    require(old_sources == seed['source_sha256'], 'complete executed seed must contain base and overlay sources')
    recovered = {p: sha for p, sha in old_sources.items() if p not in overlay['source_extension']}
    require(all(recovered.get(p) == sha for p, sha in base['source_sha256'].items()), 'recovered partition retains every base source')

    runtime_directory = Path(request['runtime_directory'])
    require(str(runtime_directory) == request['runtime_directory']
            and runtime_directory in (c.Q/'qwen35_package_serial_runtime_v2', c.Q/'qwen35_package_serial_runtime_v3'),
            'only explicit reviewed package runtime v2 or v3 directory')
    runtime_version = runtime_directory.name.rsplit('_', 1)[1]
    runtime_review = read(request['runtime_review'])
    require(runtime_review.get('schema') == 'qwen35_package_serial_runtime_'+runtime_version+'_root_source_review_v1'
            and runtime_review.get('reviewer') == '/root' and runtime_review.get('passed') is True
            and runtime_review.get('source_only') is True and runtime_review.get('blockers') == []
            and runtime_review.get('runtime_admitted') is False, 'independent exact runtime source-only review required')
    runtime_ref = short_ref(runtime_review['source_delivery'])
    require(runtime_ref == request['runtime_delivery']
            and runtime_ref['path'] == str(runtime_directory/'source_delivery.json'), 'exact root-selected runtime source delivery')
    runtime = read(runtime_ref)
    require(runtime.get('schema') == 'qwen35_package_serial_runtime_source_delivery_'+runtime_version
            and runtime.get('source_only') is True and runtime.get('launch_authorized') is False
            and runtime.get('runtime_admitted') is False and runtime.get('frozen_science_delivery') == c.SCIENCE_DELIVERY,
            'runtime delivery is source only and retains frozen full science')
    runtime_pins = manifest_pins(runtime['files'])
    require(all(str(runtime_directory/name) in runtime_pins for name in (*c.RUNTIME_FILES, 'input_pins.json')), 'complete runtime file closure')
    for path, sha in runtime_pins.items():
        source(dict(path=path, sha256=sha))
    input_manifest = read(dict(path=str(runtime_directory/'input_pins.json'), sha256=runtime_pins[str(runtime_directory/'input_pins.json')]))
    require(input_manifest.get('source_only') is True and input_manifest.get('binary_payloads_read') is False,
            'runtime input manifest must remain source only')
    runtime_input_pins = manifest_pins(input_manifest['files'])
    science = read(c.SCIENCE_DELIVERY)
    science_pins = manifest_pins(science['files'])
    for name in ('input_sources.json', 'capture_sources.json'):
        path = str(c.Q/'qwen35_package_full_validation_v2'/name)
        require(path in science_pins, 'frozen science manifest missing')
        science_pins = merge_sources(science_pins, manifest_pins(read(dict(path=path, sha256=science_pins[path]))))

    prep = request['preparation_sources']
    require(type(prep) is dict and SELF in prep and type(request['required_helpers']) is list
            and request['required_helpers'] and len(set(request['required_helpers'])) == len(request['required_helpers'])
            and all(p in prep for p in request['required_helpers']), 'preparer and every explicit root helper must be source pinned')
    for path, sha in prep.items():
        source(dict(path=path, sha256=sha))
    pins = merge_sources(old_sources, *[r['source_sha256'] for r in catalog['runs']], runtime_pins,
        {runtime_ref['path']: runtime_ref['sha256']}, runtime_input_pins, science_pins,
        {c.SCIENCE_DELIVERY['path']: c.SCIENCE_DELIVERY['sha256']}, prep)
    import_sources = (str(c.HERE/'__init__.py'), str(c.HERE/'contract.py'),
        str(c.Q/'qwen35_sampler_pair_protocol_v1.py'),
        str(c.Q/'qwen35_package_full_validation_v2/__init__.py'),
        str(c.Q/'qwen35_package_full_validation_v2/runtime.py'))
    required = (*c.STARTUP_FILES, *import_sources, str(c.Q/'qwen35_sampler_history_supplement_v1.py'),
                str(c.Q/'qwen35_sampler_torch_namespace_sources_v1.py'))
    require(all(p in pins for p in required), 'complete startup and history helper closure required')
    for path in import_sources:
        source(dict(path=path, sha256=pins[path]))
    require(type(request['extra_metadata']) is list, 'explicit extra ordinary metadata list required')
    for ref in request['extra_metadata']:
        read(ref)
    budget = read(request['resource_budget'])
    originals_path = str(runtime_directory/'source_files.json')
    originals = read(dict(path=originals_path, sha256=runtime_pins[originals_path]))
    future = request['future']
    _future(future, seed, overlay, set(observed) | set(all_inputs) | set(pins))
    require(set(request['binary_inputs']) == set(c.BINARY_PATHS), 'all six exact binary paths required')
    stat_observed = {}
    for path, row in request['binary_inputs'].items():
        require(type(row) is dict and set(row) == {'kind', 'identity'} and row['kind'] == 'binary_payload', 'separate typed binary identity')
        identity_shape(path, row['identity'])
        require(path not in all_inputs and path not in observed, 'binary payload may never become JSON metadata')
        guard()
        require(identity_only(path) == row['identity'], 'binary stat identity changed')
        stat_observed[path] = copy.deepcopy(row['identity'])
    for row in budget.get('source_files', {}).values():
        ident = row['identity']
        identity_shape(ident['realpath'], ident)
        guard()
        require(identity_only(ident['realpath']) == ident, 'source model stat identity changed')
        stat_observed[ident['realpath']] = copy.deepcopy(ident)

    ordinary = {p: row for p, row in observed.items() if p not in preparation_only}
    merge_inputs(all_inputs, ordinary)
    new_overlay_inputs = copy.deepcopy(overlay['input_extension'])
    merge_inputs(new_overlay_inputs, {p: row for p, row in seed['input_extension'].items() if p not in base['inputs']})
    merge_inputs(new_overlay_inputs, {p: row for p, row in ordinary.items() if p not in base['inputs']})
    spec_inputs = copy.deepcopy(seed['input_extension'])
    merge_inputs(spec_inputs, {p: row for p, row in new_overlay_inputs.items() if p not in base['inputs']})
    extension = {p: sha for p, sha in pins.items() if p not in recovered}
    new_overlay = dict(overlay, current_serial_catalog=request['current_catalog'], source_extension=extension,
                       input_extension=new_overlay_inputs, output_root=future['overlay_output_root'])
    overlay_bytes = encode(new_overlay, ORDINARY_CAP)
    overlay_ref = dict(path=future['overlay_path'], sha256=hashlib.sha256(overlay_bytes).hexdigest())
    spec = dict(schema='qwen35_package_serial_spec_v1', run_name=future['run_name'], output_root=future['output_root'],
        history_overlay=overlay_ref, current_serial_catalog=request['current_catalog'], source_sha256=pins,
        input_extension=spec_inputs, frozen_science_delivery=copy.deepcopy(c.SCIENCE_DELIVERY),
        running_handoff_path=future['running_handoff_path'], limits=copy.deepcopy(c.LIMITS),
        lock_identity=copy.deepcopy(c.LOCK_IDENTITY), native_python=c.PYTHON, automatic_successor=False,
        lfm_r2_scope=seed['lfm_r2_scope'], case=request['case'], binary_inputs=request['binary_inputs'],
        resource_budget=request['resource_budget'], build_file_limit_bytes=budget['build_file_limit_bytes'])
    spec_bytes = encode(spec)
    spec_ref = dict(path=future['spec_path'], sha256=hashlib.sha256(spec_bytes).hexdigest())
    review = dict(schema='qwen35_package_serial_source_review_v1', reviewer='/root', passed=False,
        blockers=['Root independent exact binding review, full History and physical/native admission remain pending.'],
        spec=spec_ref, source_sha256=copy.deepcopy(pins), launch_authorized=False)
    # Pure validation only. The temporary true object is neither returned nor
    # published, and cannot itself authorize execution without root publication.
    validation_review = dict(review, passed=True, blockers=[], launch_authorized=True)
    c.validate_spec(spec, validation_review, spec_ref)
    del validation_review
    validate_budget_metadata(spec, budget, originals)
    review_bytes = encode(review)
    review_ref = dict(path=future['review_candidate_path'], sha256=hashlib.sha256(review_bytes).hexdigest())
    for path in set(observed) & set(source_observed):
        require(observed[path] == source_observed[path], 'metadata/source observations must agree exactly')
    for path, row in {**observed, **source_observed}.items():
        guard()
        require(identity_only(path) == row['identity'], 'observed preparation input changed before return: '+path)
    for path, expected in stat_observed.items():
        guard()
        require(identity_only(path) == expected, 'stat-only payload identity changed before return')
    guard()
    return dict(schema='qwen35_package_serial_candidate_v1', passed=False, launch_authorized=False,
        source_only=True, history_resolved=False, physical_checks_passed=False, native_executed=False,
        overlay=new_overlay, overlay_bytes=overlay_bytes, overlay_ref=overlay_ref,
        spec=spec, spec_bytes=spec_bytes, spec_ref=spec_ref,
        review=review, review_bytes=review_bytes, review_ref=review_ref,
        catalog_source_extension=extension,
        seed_references={k: request[k] for k in ('seed_spec', 'seed_review', 'seed_actual_exit')},
        preparation_only_references={p: observed[p] for p in sorted(preparation_only)},
        observed_metadata=observed, observed_sources=source_observed,
        preservation=dict(base_inputs=len(base['inputs']), seed_inputs=len(seed['input_extension']),
            overlay_inputs=len(overlay['input_extension']), recovered_sources=len(recovered),
            recovery_only_sources=len(set(recovered)-set(base['source_sha256'])),
            seed_sources=len(old_sources), complete_sources=len(pins),
            old_catalog_rows=len(old_catalog['runs']), current_catalog_rows=len(catalog['runs'])))


def validate_budget_metadata(spec, value, originals):
    """Match runtime budget contract without opening real model or source files.

    Runtime repeats this against its authenticated source_files.json and then
    performs fresh parent/device/path/resource checks before the original lock.
    """
    keys = {'schema', 'reviewer', 'passed', 'blockers', 'case', 'output_root', 'control_root',
            'classes', 'headroom_bytes', 'parent_identities', 'source_files', 'build_file_limit_bytes'}
    require(type(value) is dict and set(value) == keys and value['schema'] == 'qwen35_package_reviewed_budget_v1'
            and value['reviewer'] == '/root' and value['passed'] is True and value['blockers'] == [], 'root-reviewed exact resource budget required')
    require(value['case'] == spec['case'] and value['output_root'] == spec['output_root']
            and value['control_root'] == str(c.control_root(spec)), 'reviewed budget must bind all six roots')
    classes = {'view', 'artifact', 'compiler_cache', 'compiler_workdir', 'build_auxiliary'}
    require(type(value['classes']) is dict and set(value['classes']) == classes, 'all budget classes required')
    for row in value['classes'].values():
        require(type(row) is dict and set(row) == {'allocated_bytes', 'single_file_bytes', 'entries'}
                and all(type(n) is int and 0 < n < 2**63 for n in row.values())
                and row['single_file_bytes'] <= row['allocated_bytes'], 'finite per-class capacity required')
    require(value['classes']['build_auxiliary']['allocated_bytes'] >= 2*c.LIMITS['leaf_log_bytes']
            and value['build_file_limit_bytes'] == spec['build_file_limit_bytes']
            == max(row['single_file_bytes'] for row in value['classes'].values()), 'unchanged build auxiliary and file limit policy')
    require(type(value['headroom_bytes']) is dict and set(value['headroom_bytes']) == {'root', 'nvme'}
            and all(type(n) is int and 0 < n < 2**63 for n in value['headroom_bytes'].values()), 'finite reviewed headroom required')
    roots = [Path(spec['output_root']), c.control_root(spec)] + [Path(spec['case'][k]) for k in ('model_path', 'artifact', 'compiler_cache', 'compiler_workdir')]
    require(len(set(roots)) == 6 and not any(a in b.parents for a in roots for b in roots if a != b), 'six disjoint nonnested roots')
    require(set(value['parent_identities']) == {str(p.parent) for p in roots}, 'all parent identities required')
    for path, row in value['parent_identities'].items():
        require(type(row) is dict and set(row) == {'realpath', 'device', 'inode'} and row['realpath'] == path
                and all(type(row[k]) is int and row[k] >= 0 for k in ('device', 'inode')), 'exact parent stat identity')
    require(set(value['source_files']) == set(originals), 'all original model SOURCE_FILES identities required')
    for name, row in value['source_files'].items():
        require(type(row) is dict and set(row) == {'pin', 'identity'} and row['pin'] == originals[name]
                and row['identity']['realpath'] == str(Path('/mnt/nvme/models/Qwen3.5-2B')/name)
                and row['identity']['size'] == originals[name]['size'], 'exact source pin and stat-only identity')
    return value
