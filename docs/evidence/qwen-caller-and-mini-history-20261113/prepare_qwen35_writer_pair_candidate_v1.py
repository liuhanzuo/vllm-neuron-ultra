"""Explicit source-only writer-pair candidate; never runtime admission.

The complete inherited package v7 metadata/source/identity contract is retained.
All IO is injected. Real geometry and full package25 evidence, a reviewed
simultaneous two-build budget, and actual current serial inputs are mandatory.
Import performs no History, lock, payload read, native import or publication.
"""
import copy
import hashlib
from pathlib import Path
import re

from qwen35_writer_pair_serial_runtime_v2 import contract as c
from prepare_qwen35_package_serial_candidate_v7 import (
    require, encode, reference, short_ref, identity_shape, file_identity,
    StableReaders, finite_json_numbers, retained_reference_paths, merge_sources,
    merge_inputs, catalog_shape, manifest_pins, sdk_asset_identities,
    ORDINARY_CAP, ENVELOPE_CAP, TEXT_SUFFIXES, PWP_MANIFEST_SHA256)

SELF = str(Path(__file__).resolve())
REQUEST_KEYS = {'seed_spec', 'seed_review', 'seed_actual_exit', 'current_catalog',
    'idle_handoff', 'runtime_directory', 'runtime_delivery', 'runtime_review',
    'resource_budget', 'writer_cases', 'geometry_qualification', 'package_completion',
    'binary_inputs', 'preparation_sources', 'required_helpers', 'extra_metadata', 'future'}
FUTURE_KEYS = {'run_name', 'output_root', 'overlay_output_root', 'running_handoff_path',
    'overlay_path', 'spec_path', 'review_candidate_path'}
RUNTIME_DIRECTORY = c.Q / 'qwen35_writer_pair_serial_runtime_v2'

def _future(future, seed, overlay, forbidden):
    require(type(future) is dict and set(future) == FUTURE_KEYS, 'explicit future output bindings required')
    c.units(future['run_name'])
    require(future['run_name'] != seed['run_name'] and future['output_root'] != seed['output_root']
            and future['output_root'] == '/mnt/nvme/neuron-work/qww-'+future['run_name'], 'fresh exact writer output namespace')
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


def completed_prerequisites(request, catalog, read):
    """Authenticate executed geometry and package25 metadata without native IO.

    The geometry checker is pure and examines all saved calls. Package stage25
    is a separately executed independent audit, never inferred from an exit0
    alone. Its storage result is retained as input; this function does not guess
    a new dual-build budget or promote full-model/performance qualification.
    """
    from qwen35_writer_pair_validation_v1.geometry import check_complete_geometry, LAYOUTS
    from qwen35_package_serial_runtime_v8.wiring import STAGES as PACKAGE_STAGES
    qualification = read(request['geometry_qualification'])
    require(set(qualification) == {'schema', 'reviewer', 'passed', 'blockers', 'reports',
            'actual_exit', 'source_sha256', 'full_model_qualified'}
            and qualification['schema'] == 'qwen35_writer_geometry_independent_qualification_v1'
            and qualification['reviewer'] == '/root' and qualification['passed'] is True
            and qualification['blockers'] == [] and qualification['full_model_qualified'] is False
            and set(qualification['reports']) == set(LAYOUTS), 'complete independently qualified geometry required')
    actual = read(qualification['actual_exit'])
    require(actual['schema'] == 'qwen35_writer_geometry_root_actual_exit_v1'
            and actual['reviewer'] == '/root' and actual['actual_tool_exit_observed'] is True
            and actual['actual_tool']['exit_code'] == 0 and actual['geometry_qualified'] is True
            and actual['owners_gone'] is True and actual['physically_drained'] is True
            and actual['reports'] == qualification['reports']
            and actual['source_sha256'] == qualification['source_sha256']
            and actual['full_model_qualified'] is False, 'actual successful independently drained geometry required')
    rows = catalog_shape(catalog)
    row = rows.get(qualification['actual_exit']['path'])
    require(row is not None and row['terminal'] == qualification['actual_exit']
            and row['tool'] == actual['actual_tool'] and row['owners'] == actual['owners']
            and row['units'] == actual['units'], 'exact actual geometry row must already be retained')
    reports = {name: read(ref) for name, ref in qualification['reports'].items()}
    check_complete_geometry(reports)
    require(all(report['source_sha256'] == qualification['source_sha256'] for report in reports.values()),
            'executed geometry source identities must agree')
    package = read(request['package_completion'])
    require(package['schema'] == 'qwen35_package_root_actual_exit_v1' and package['reviewer'] == '/root'
            and package['actual_tool_exit_observed'] is True and package['actual_tool']['exit_code'] == 0
            and package['owners_gone'] is True and package['physically_drained'] is True
            and package['completed_stages'] == 24 and len(package['completed_stage_prefix']) == 24
            and package['outcome']['all25_stages_audited'] is True
            and package['stage25'] == package['outcome']['stage25']
            and package['outcome']['stage25_error'] is None, 'actual package completion must include full independent stage25')
    row = rows.get(request['package_completion']['path'])
    require(row is not None and row['terminal'] == request['package_completion']
            and row['tool'] == package['actual_tool'] and row['owners'] == package['owners']
            and row['units'] == package['all_declared_units'], 'exact completed package row must already be retained')
    package_spec = read(package['spec'], ENVELOPE_CAP)
    stage25 = read(package['stage25']); audit = read(stage25['result'])
    require(stage25['index'] == 25 and (stage25['arm'], stage25['stage']) == PACKAGE_STAGES[-1]
            and stage25['case'] == package_spec['case'] and stage25['completed'] is True
            and audit['schema'] == 'qwen35_package_independent_final_audit_v1'
            and audit['reviewer'] == '/root' and audit['all25_stages_audited'] is True
            and audit['spec'] == package['spec'] and audit['admission'] == package['admission']
            and audit['external_tool_exit'] == package['external_tool_exit']
            and audit['outer_terminal'] == package['outer_terminal']
            and audit['stage_receipts'] == package['completed_stage_prefix']
            and type(audit['storage']) is dict and audit['storage'],
            'actual package stage25 must bind exact full run and retained build/storage facts')
    for index, reference_ in enumerate(package['completed_stage_prefix'], 1):
        receipt = read(reference_)
        require(receipt['index'] == index and (receipt['arm'], receipt['stage']) == PACKAGE_STAGES[index-1]
                and receipt['completed'] is True and receipt['case'] == package_spec['case'],
                'all original package stages must remain an exact completed prefix')
        read(receipt['result'])
    return merge_sources(qualification['source_sha256'], package_spec['source_sha256'])


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
    retained_paths = retained_reference_paths(request)
    def read(ref, cap=ORDINARY_CAP):
        reference(ref)
        require(Path(ref['path']).suffix == '.json', 'metadata inputs must be JSON')
        guard()
        saved = read_metadata(copy.deepcopy(ref), cap)
        require(type(saved) is dict and set(saved) == {'value', 'identity'}, 'exact stable metadata reader receipt')
        identity_shape(ref['path'], saved['identity'])
        require(saved['identity']['size'] <= cap, 'original metadata cap exceeded')
        finite_json_numbers(saved['value'])
        retained_paths.update(retained_reference_paths(saved['value']))
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
            and runtime_directory == RUNTIME_DIRECTORY, 'exact accepted writer runtime v2 directory')
    runtime_review = read(request['runtime_review'])
    require(runtime_review.get('schema') == 'qwen35_writer_pair_runtime_v2_root_source_review_v1'
            and runtime_review.get('reviewer') == '/root' and runtime_review.get('source_accepted') is True
            and runtime_review.get('blockers') == [] and runtime_review.get('runtime_admitted') is False,
            'independent exact writer runtime source review required')
    runtime_ref = short_ref(runtime_review['source_delivery'])
    require(runtime_ref == request['runtime_delivery']
            and runtime_ref['path'] == str(runtime_directory/'source_delivery.json'),
            'exact root-selected writer runtime source delivery')
    runtime = read(runtime_ref)
    require(runtime.get('schema') == 'qwen35_writer_pair_serial_runtime_source_delivery_v2'
            and runtime.get('source_only') is True and runtime.get('runtime_admitted') is False,
            'accepted writer runtime remains source-only')
    runtime_pins = manifest_pins(runtime['files'])
    require(all(str(runtime_directory/name) in runtime_pins for name in c.RUNTIME_FILES),
            'complete writer runtime file closure')
    for path, sha in runtime_pins.items():
        source(dict(path=path, sha256=sha))
    science = read(c.SCIENCE_DELIVERY)
    science_pins = manifest_pins(science['files'])
    for name in ('input_sources.json', 'capture_sources.json', 'geometry_sources.json'):
        path = str(c.Q/'qwen35_writer_pair_validation_v1'/name)
        require(path in science_pins, 'frozen writer science manifest missing')
        science_pins = merge_sources(science_pins, manifest_pins(read(dict(path=path, sha256=science_pins[path]))))
    prerequisite_pins = completed_prerequisites(request, catalog, read)

    prep = request['preparation_sources']
    require(type(prep) is dict and SELF in prep and type(request['required_helpers']) is list
            and request['required_helpers'] and len(set(request['required_helpers'])) == len(request['required_helpers'])
            and all(p in prep for p in request['required_helpers']), 'preparer and every explicit root helper must be source pinned')
    for path, sha in prep.items():
        source(dict(path=path, sha256=sha))
    pins = merge_sources(old_sources, *[r['source_sha256'] for r in catalog['runs']], runtime_pins,
        {runtime_ref['path']: runtime_ref['sha256']}, science_pins, prerequisite_pins,
        {c.SCIENCE_DELIVERY['path']: c.SCIENCE_DELIVERY['sha256']}, prep)
    import_sources = (str(c.HERE/'__init__.py'), str(c.HERE/'contract.py'),
        str(c.Q/'qwen35_sampler_pair_protocol_v1.py'),
        str(c.Q/'qwen35_writer_pair_validation_v1/__init__.py'),
        str(c.Q/'qwen35_writer_pair_validation_v1/runtime.py'),
        str(c.Q/'prepare_qwen35_package_serial_candidate_v7.py'))
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
    _future(future, seed, overlay, set(observed) | set(all_inputs) | set(pins) | retained_paths)
    require(set(request['binary_inputs']) == set(c.BINARY_PATHS), 'all six exact binary paths required')
    stat_observed = {}
    if True:  # Exact accepted v2 retains the package v8 SDK manifest.
        assets_path = str(runtime_directory/'compiler_assets.json')
        require(runtime_pins[assets_path] == PWP_MANIFEST_SHA256, 'exact frozen SDK asset manifest pin required')
        assets = read(dict(path=assets_path, sha256=PWP_MANIFEST_SHA256))
        sdk_asset_identities(assets, identity_only, guard, stat_observed)
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
    spec = dict(schema='qwen35_writer_pair_serial_spec_v1', run_name=future['run_name'], output_root=future['output_root'],
        history_overlay=overlay_ref, current_serial_catalog=request['current_catalog'], source_sha256=pins,
        input_extension=spec_inputs, frozen_science_delivery=copy.deepcopy(c.SCIENCE_DELIVERY),
        running_handoff_path=future['running_handoff_path'], limits=copy.deepcopy(c.LIMITS),
        lock_identity=copy.deepcopy(c.LOCK_IDENTITY), native_python=c.PYTHON, automatic_successor=False,
        lfm_r2_scope=seed['lfm_r2_scope'], case=request['writer_cases']['baseline'],
        writer_cases=request['writer_cases'], geometry_qualification=request['geometry_qualification'],
        binary_inputs=request['binary_inputs'],
        resource_budget=request['resource_budget'], build_file_limit_bytes=budget['build_file_limit_bytes'])
    spec_bytes = encode(spec)
    spec_ref = dict(path=future['spec_path'], sha256=hashlib.sha256(spec_bytes).hexdigest())
    review = dict(schema='qwen35_writer_pair_serial_source_review_v1', reviewer='/root', passed=False,
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
    return dict(schema='qwen35_writer_pair_serial_candidate_v1', passed=False, launch_authorized=False,
        source_only=True, history_resolved=False, physical_checks_passed=False, native_executed=False,
        overlay=new_overlay, overlay_bytes=overlay_bytes, overlay_ref=overlay_ref,
        spec=spec, spec_bytes=spec_bytes, spec_ref=spec_ref,
        review=review, review_bytes=review_bytes, review_ref=review_ref,
        catalog_source_extension=extension,
        completed_prerequisites={key: request[key] for key in ('geometry_qualification', 'package_completion')},
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
    keys = {'schema', 'reviewer', 'passed', 'blockers', 'writer_cases', 'output_root', 'control_root',
            'classes', 'headroom_bytes', 'parent_identities', 'source_files', 'build_file_limit_bytes'}
    require(type(value) is dict and set(value) == keys and value['schema'] == 'qwen35_writer_pair_reviewed_budget_v1'
            and value['reviewer'] == '/root' and value['passed'] is True and value['blockers'] == [], 'root-reviewed exact resource budget required')
    require(value['writer_cases'] == spec['writer_cases'] and value['output_root'] == spec['output_root']
            and value['control_root'] == str(c.control_root(spec)), 'reviewed budget must bind all ten roots')
    classes = {arm+'_'+kind for arm in ('baseline', 'candidate')
               for kind in ('view', 'artifact', 'compiler_cache', 'compiler_workdir')} | {'build_auxiliary'}
    require(type(value['classes']) is dict and set(value['classes']) == classes, 'all budget classes required')
    for row in value['classes'].values():
        require(type(row) is dict and set(row) == {'allocated_bytes', 'single_file_bytes', 'entries'}
                and all(type(n) is int and 0 < n < 2**63 for n in row.values())
                and row['single_file_bytes'] <= row['allocated_bytes'], 'finite per-class capacity required')
    require(value['classes']['build_auxiliary']['allocated_bytes'] >= 4*c.LIMITS['leaf_log_bytes']
            and value['build_file_limit_bytes'] == spec['build_file_limit_bytes']
            == max(row['single_file_bytes'] for row in value['classes'].values()), 'unchanged build auxiliary and file limit policy')
    require(type(value['headroom_bytes']) is dict and set(value['headroom_bytes']) == {'root', 'nvme'}
            and all(type(n) is int and 0 < n < 2**63 for n in value['headroom_bytes'].values()), 'finite reviewed headroom required')
    roots = [Path(spec['output_root']), c.control_root(spec)] + [Path(spec['writer_cases'][arm][k]) for arm in ('baseline', 'candidate')
        for k in ('model_path', 'artifact', 'compiler_cache', 'compiler_workdir')]
    require(len(set(roots)) == 10 and not any(a in b.parents for a in roots for b in roots if a != b), 'ten disjoint nonnested roots')
    require(set(value['parent_identities']) == {str(p.parent) for p in roots}, 'all parent identities required')
    for path, row in value['parent_identities'].items():
        require(type(row) is dict and set(row) == {'realpath', 'device', 'inode'} and row['realpath'] == path
                and all(type(row[k]) is int and row[k] >= 0 for k in ('device', 'inode')), 'exact parent stat identity')
    require(set(value['source_files']) == set(originals), 'all original model SOURCE_FILES identities required')
    for name, row in value['source_files'].items():
        require(type(row) is dict and set(row) == {'pin', 'identity'} and row['pin'] == originals[name]
                and row['identity']['realpath'] == str(Path('/mnt/nvme/models/Qwen3.5-2B')/name)
                and row['identity']['size'] == originals[name]['size'], 'exact source pin and stat-only identity')
    for row in value['source_files'].values():
        identity_shape(row['identity']['realpath'], row['identity'])
    return value


def candidate_envelope(prepared, request):
    """Small publication candidate; caller writes exact returned bytes once.

    This carries no approved review and cannot launch a model. The root must
    publish overlay/spec/review-draft exact bytes and independently accept the
    complete binding with accept_qwen35_writer_pair_root_v1.py afterward.
    """
    require(prepared['schema'] == 'qwen35_writer_pair_serial_candidate_v1'
            and prepared['passed'] is False and prepared['launch_authorized'] is False,
            'only non-authorizing writer preparation may be enveloped')
    for kind in ('overlay', 'spec', 'review'):
        ref = prepared[kind+'_ref']; reference(ref)
        require(hashlib.sha256(prepared[kind+'_bytes']).hexdigest() == ref['sha256'],
                'exact prepared publication bytes required')
    require(prepared['spec']['current_serial_catalog'] == request['current_catalog']
            and prepared['completed_prerequisites'] == {key: request[key]
                for key in ('geometry_qualification', 'package_completion')},
            'publication envelope cannot select different completed inputs')
    value = dict(schema='qwen35_writer_pair_root_candidate_envelope_v1', passed=False,
        launch_authorized=False, spec_ref=prepared['spec_ref'], overlay_ref=prepared['overlay_ref'],
        review_draft_ref=prepared['review_ref'], catalog=request['current_catalog'],
        idle_handoff=request['idle_handoff'], runtime_source_review=request['runtime_review'],
        completed_prerequisites=prepared['completed_prerequisites'],
        preparation_only_references=prepared['preparation_only_references'],
        preservation=prepared['preservation'])
    return copy.deepcopy(value), encode(value, ORDINARY_CAP)
