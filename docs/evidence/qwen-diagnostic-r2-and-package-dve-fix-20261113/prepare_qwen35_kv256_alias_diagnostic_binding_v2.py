"""Explicit future diagnostic binding preparation; inert on import and never launch.

Mechanical successor of the integer-reference root preparer. The pure builder
returns candidate objects. The CLI only writes four fresh metadata files after
source/resource checks; it never resolves History, takes a lock or starts work.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys

Q = Path(__file__).resolve().parent
ROOT = Q.parent
SELF = Q / 'prepare_qwen35_kv256_alias_diagnostic_binding_v2.py'
DIAGNOSTIC_REVIEW_PATH = str(Q / 'qwen35-kv256-alias-diagnostic-serial-root-source-review-20261113-v1.json')
MIB = 2**20
IDENTITY_KEYS = {'realpath', 'device', 'inode', 'size', 'mtime_ns', 'ctime_ns'}
REF_KEYS = ('seed_spec', 'seed_review', 'seed_binding', 'catalog', 'idle_handoff',
            'prerequisite_exit', 'diagnostic_source_review', 'preparer_source_review',
            'preparer_delivery')
PATH_KEYS = ('overlay', 'spec', 'review_draft', 'review', 'candidate', 'output',
             'overlay_output', 'running')


# One unchanged inherited plan; the frozen runtime already has its typed reader.
PREDECESSOR_PREPARER = dict(path=str(Q/'prepare_qwen35_kv256_alias_diagnostic_binding_v1.py'),
    sha256='d903881e0a66e0d9e5baec131c7974d08d16935ebdf69439ac2576032f6c5c79')
INHERITED_PLAN = dict(path=str(Q/'minicpm5-layer1-operator-sdk-r2-20261111-plan.json'),
    sha256='2e5441f4b2c2ad2efd76192ae92c068c35d3ae9fc51c532062d4284692910632',
    size_bytes=12803326, schema='minicpm5_layer0_runtime_inputs_v1', original_typed_cap_bytes=32*MIB)


def require(value, message):
    if not value:
        raise ValueError(message)



def inherited_plan_shape(value):
    """Exact plan branch of the existing operator SDK envelope shape checks."""
    require(type(value)is dict and value.get('schema')==INHERITED_PLAN['schema'], 'Original typed plan schema')
    pins=value['source_sha256']
    require(type(pins)is dict and len(pins)<=100000, 'Original bounded complete plan source map')
    for name,digest in pins.items():
        require(type(name)is str and Path(name).is_absolute()and str(Path(name))==name and '..'not in Path(name).parts
            and type(digest)is str and re.fullmatch('[0-9a-f]{64}',digest), 'Original literal absolute source pin')
    reference=value['source_contract']
    require(type(reference)is dict and set(reference)in({'path','sha256'},{'path','sha256','bytes'})
        and type(reference['path'])is str and Path(reference['path']).is_absolute()
        and type(reference['sha256'])is str and re.fullmatch('[a-f0-9]{64}',reference['sha256'])
        and ('bytes'not in reference or type(reference['bytes'])is int and reference['bytes']>=0),
        'Original exact source-contract reference')
    require(type(value['latest_completion'])is dict and type(value['output_root'])is str
        and type(value['inputs'])is dict and len(value['inputs'])<=100000, 'Original complete typed plan fields')


def retained_input_caps(inputs,seed_inputs,observed,base_ref,base):
    """Ordinary8MiB plus exactly one already-inherited, unchanged typed input."""
    require(all(inputs.get(path)==row for path,row in seed_inputs.items()), 'Every complete seed input retained unchanged')
    retained={}
    for path,row in inputs.items():
        size=row['identity']['size']
        require(type(size)is int and size>=0, 'Finite original input size')
        if path!=INHERITED_PLAN['path']:
            require(size<=8*MIB, 'Ordinary runtime metadata retains original8MiB cap')
            continue
        require(base_ref=={k:INHERITED_PLAN[k]for k in('path','sha256')}
            and row['sha256']==INHERITED_PLAN['sha256']and size==INHERITED_PLAN['size_bytes']
            and size<=INHERITED_PLAN['original_typed_cap_bytes'], 'Only the exact original typed plan exception')
        require(seed_inputs.get(path)==row and observed.get(path)==row,
            'Same full inherited and freshly authenticated plan identity; never refresh or introduce an input')
        ident=row['identity']
        require(set(ident)==IDENTITY_KEYS and ident['realpath']==path
            and all(type(ident[k])is int and ident[k]>=0 for k in IDENTITY_KEYS-{'realpath'}),
            'Exact complete original plan identity')
        inherited_plan_shape(base);retained[path]=copy.deepcopy(row)
    return retained


def verify_executed_preparer(helper):
    """CLI source check before any candidate preparation or output publication."""
    require(helper['schema']=='qwen35_kv256_alias_diagnostic_binding_source_delivery_v2'
        and helper['source_sha256'].get(str(SELF))==hashlib.sha256(SELF.read_bytes()).hexdigest()
        and helper['source_sha256'].get(PREDECESSOR_PREPARER['path'])==PREDECESSOR_PREPARER['sha256'],
        'Exact executed v2 preparer and retained frozen v1 helper must be pre-enrolled')


def api():
    # These are accepted source-only imports. In particular do not import the
    # integer root script, whose historical top-level body performs real IO.
    for directory in (Q, ROOT):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    from qwen35_kv256_alias_diagnostic_serial_v1 import contract
    from prepare_qwen35_sampler_v9_candidate_v3 import encode, object_ref, merge_inputs, catalog_shape
    from qwen35_sampler_runtime_contract_v12 import merge_sources, reference
    return contract, encode, object_ref, merge_inputs, catalog_shape, merge_sources, reference


def prepare(future, read_metadata, fresh_path, guard):
    """Return four metadata objects; injected IO has no runtime/launch authority.

    read_metadata(ref) must return {value, identity} from a stable authenticated
    JSON read. fresh_path(path) must prove a canonical absent path and ordinary
    existing ancestors. Mocks must label their own result as synthetic.
    """
    c, encode, object_ref, merge_inputs, catalog_shape, merge_sources, reference = api()
    require(callable(read_metadata) and callable(fresh_path) and callable(guard), 'Explicit metadata IO required')
    require(type(future) is dict and set(future) == {
        'schema', *REF_KEYS, 'prerequisite_tool', 'run_name', 'paths'}, 'Exact explicit future request')
    require(future['schema'] == 'qwen35_kv256_alias_diagnostic_future_binding_v1', 'Future binding schema')
    for key in REF_KEYS:
        reference(future[key])
    require(future['diagnostic_source_review']['path'] == DIAGNOSTIC_REVIEW_PATH, 'Explicit future root diagnostic review; actual digest supplied by caller')
    paths = future['paths']
    require(type(paths) is dict and set(paths) == set(PATH_KEYS), 'All future paths explicit')
    require(len(set(paths.values())) == len(paths), 'Distinct fresh paths')
    require(type(future['run_name']) is str and re.fullmatch('[a-z0-9][a-z0-9-]{0,39}', future['run_name']), 'Bounded run name')
    for key in ('overlay', 'spec', 'review_draft', 'review', 'candidate', 'running'):
        p = Path(paths[key])
        require(p.parent == Q and str(p) == paths[key] and re.fullmatch('[a-z0-9][a-z0-9-]*\.json', p.name), 'Research JSON path')
    require(Path(paths['running']).name.startswith('continuation-handoff-'), 'Running handoff namespace')
    require(paths['output'] == '/mnt/nvme/neuron-work/qwen35-kv256-alias-diagnostic-' + future['run_name'], 'Original diagnostic output namespace')
    p = Path(paths['overlay_output'])
    require(p.parent == Path('/mnt/nvme') and str(p) == paths['overlay_output']
            and re.fullmatch('minicpm5-layer1-operator-native-[a-z0-9-]{1,64}', p.name), 'Explicit virtual-history output namespace')
    observed = {}
    large = {future['seed_spec']['path'], future['seed_review']['path']}

    def read(ref):
        reference(ref)
        require(Path(ref['path']).suffix == '.json', 'JSON metadata only')
        guard(); saved = read_metadata(copy.deepcopy(ref))
        require(type(saved) is dict and set(saved) == {'value', 'identity'}, 'Authenticated metadata envelope')
        ident = saved['identity']
        require(type(ident) is dict and set(ident) == IDENTITY_KEYS and ident['realpath'] == ref['path'], 'Exact metadata identity')
        require(all(type(ident[k]) is int and ident[k] >= 0 for k in IDENTITY_KEYS - {'realpath'}), 'Finite metadata identity')
        require(ident['size'] <= (64*MIB if ref['path'] in large else 8*MIB), 'Original metadata class cap')
        merge_inputs(observed, {ref['path']: dict(sha256=ref['sha256'], identity=ident)})
        guard(); return copy.deepcopy(saved['value'])

    seed = read(future['seed_spec']); review = read(future['seed_review']); binding = read(future['seed_binding'])
    require(binding['reviewer'] == '/root' and binding['passed'] is True
            and binding['full_source_history_resolution_passed'] is True
            and binding['original_failed_statuses_preserved'] is True
            and type(binding['lfm_children']) is int and binding['lfm_children'] >= 874
            and binding['spec'] == future['seed_spec'] and binding['review'] == future['seed_review'], 'Root accepted complete seed History and original 874 children')
    require(review['reviewer'] == '/root' and review['passed'] is True and review['launch_authorized'] is True
            and review['blockers'] == [] and review['spec'] == future['seed_spec']
            and review['source_sha256'] == seed['source_sha256'], 'Exact accepted seed source review')
    require(seed['lock_identity'] == c.LOCK_IDENTITY and seed['automatic_successor'] is False, 'Original lock and no automatic successor')
    overlay = read(seed['history_overlay']); old_catalog = read(seed['current_serial_catalog'])
    reference(overlay['base_plan']); large.add(overlay['base_plan']['path']); base = read(overlay['base_plan'])
    require(binding['catalog'] == seed['current_serial_catalog'] == overlay['current_serial_catalog']
            and binding['history_overlay'] == seed['history_overlay'], 'Exact seed catalog/overlay lineage')
    catalog = read(future['catalog']); catalog_shape(old_catalog); catalog_shape(catalog)
    require(catalog['runs'][:len(old_catalog['runs'])] == old_catalog['runs']
            and len(catalog['runs']) >= len(old_catalog['runs'])
            and catalog['completed_sdk'] == old_catalog['completed_sdk'], 'All historical rows retained in order')
    require(catalog['current_handoff'] == future['idle_handoff'], 'Explicit current catalog/handoff pair')
    handoff = read(future['idle_handoff'])
    require(handoff['serial_execution'] is True and handoff['running'] is False and handoff['active_serial'] is None
            and handoff['live_tool_sessions'] == handoff['active_heavy_units'] == [], 'Saved idle handoff required; physical freshness still pending')
    terminal = read(future['prerequisite_exit']); row = catalog['runs'][-1]
    require(row['terminal'] == future['prerequisite_exit'] and row['tool'] == future['prerequisite_tool']
            and terminal['actual_tool'] == row['tool'], 'Actual final prerequisite tool/exit exactly bound')
    tool = row['tool']
    require(set(tool) == {'kind', 'session_id', 'terminal_chunk', 'exit_code'} and tool['kind'] == 'tool_session'
            and type(tool['session_id']) is int and tool['session_id'] > 0
            and type(tool['terminal_chunk']) is str and re.fullmatch('[a-zA-Z0-9_-]{1,128}', tool['terminal_chunk'])
            and type(tool['exit_code']) is int, 'Actual tool terminal, including failed outcomes')
    require(terminal['owners_gone'] is True and terminal['physically_drained'] is True
            and row['owners'] == terminal['owners'], 'Saved complete predecessor drain required')
    # Both forms are existing audited terminal schemas, never inferred owners.
    units = terminal.get('units', terminal.get('all_declared_units'))
    lock = terminal.get('lock_identity', terminal.get('audit_lock_identity'))
    require(row['units'] == units and lock == c.LOCK_IDENTITY, 'Complete declared predecessor units and original lock')
    accepted = read(future['diagnostic_source_review']); delivery_ref = accepted['source_delivery']; delivery = read(delivery_ref)
    require(accepted['reviewer'] == '/root' and accepted['source_accepted'] is True
            and accepted['blockers'] == [] and accepted['runtime_admitted'] is False, 'Accepted diagnostic source only')
    require(delivery['schema'] == 'qwen35_kv256_alias_diagnostic_serial_source_delivery_v1'
            and delivery['source_only'] is True and delivery['primitive_manifest'] == c.MANIFEST
            and delivery['workload'] == c.protocol.workload() and delivery['limits'] == c.LIMITS, 'All diagnostic science and original limits preserved')
    helper_review = read(future['preparer_source_review']); helper = read(future['preparer_delivery'])
    require(helper_review['reviewer'] == '/root' and helper_review['source_accepted'] is True
            and helper_review['blockers'] == [] and helper_review['source_delivery'] == future['preparer_delivery'], 'Explicit independent helper source acceptance')
    require(helper['schema'] == 'qwen35_kv256_alias_diagnostic_binding_source_delivery_v2'
            and helper['predecessor_preparer'] == PREDECESSOR_PREPARER
            and helper['retained_input_exception'] == INHERITED_PLAN
            and helper['source_sha256'].get(PREDECESSOR_PREPARER['path']) == PREDECESSOR_PREPARER['sha256']
            and helper['source_only'] is True and str(SELF) in helper['source_sha256'], 'Complete reviewed helper source map')
    read(seed['lfm_r2_scope']); read({k: c.MANIFEST[k] for k in ('path', 'sha256')})
    for r in c.protocol.REFERENCE_EVIDENCE:
        read(r)
    recovered = {p: s for p, s in seed['source_sha256'].items() if p not in overlay['source_extension']}
    require(merge_sources(recovered, overlay['source_extension']) == seed['source_sha256'], 'Exact original recovered/source partition')
    ordinary = {p: v for p, v in observed.items() if p not in large}
    preparation_only = {p: v for p, v in observed.items() if p in large}
    pins = merge_sources(seed['source_sha256'],
        *(r['source_sha256'] for r in catalog['runs'] if r['source_sha256']),
        delivery['static_local_source_sha256'], delivery['sdk_source_sha256'], helper['source_sha256'],
        {p: v['sha256'] for p, v in ordinary.items()})
    forbidden = set(observed) | set(pins) | {seed['output_root'], seed['running_handoff_path'], overlay['output_root']}
    require(not set(paths.values()) & forbidden and future['run_name'] != seed['run_name'], 'Cannot reuse historical paths or run identity')
    require(all(fresh_path(p) is True for p in paths.values()), 'Every prospective path must be fresh')
    overlay = copy.deepcopy(overlay)
    overlay.update(current_serial_catalog=future['catalog'], output_root=paths['overlay_output'],
                   source_extension={p: s for p, s in pins.items() if p not in recovered})
    require(not set(overlay['input_extension']) & set(base['inputs']), 'Original overlay cannot replace base inputs')
    for p, v in ordinary.items():
        require(p not in base['inputs'] or base['inputs'][p] == v, 'Observed original input identity unchanged')
    merge_inputs(overlay['input_extension'], {p: v for p, v in ordinary.items() if p not in base['inputs']})
    overlay_ref = object_ref(paths['overlay'], overlay, ordinary=True)
    inputs = copy.deepcopy(seed['input_extension']); merge_inputs(inputs, ordinary)
    retained_large_inputs = retained_input_caps(inputs, seed['input_extension'], observed, overlay['base_plan'], base)
    for path in retained_large_inputs: preparation_only.pop(path, None)
    spec = dict(schema='qwen35_kv256_alias_diagnostic_serial_spec_v1', run_name=future['run_name'],
        output_root=paths['output'], history_overlay=overlay_ref, current_serial_catalog=future['catalog'],
        source_sha256=pins, input_extension=inputs, running_handoff_path=paths['running'],
        lfm_r2_scope=copy.deepcopy(seed['lfm_r2_scope']), source_delivery=delivery_ref,
        primitive_manifest=copy.deepcopy(c.MANIFEST), workload=c.protocol.workload(), limits=copy.deepcopy(c.LIMITS),
        lock_identity=copy.deepcopy(c.LOCK_IDENTITY), native_python=c.PYTHON, automatic_successor=False)
    spec_ref = object_ref(paths['spec'], spec)
    draft = dict(schema='qwen35_sampler_runtime_source_review_v1', reviewer='/root', passed=False,
        blockers=['Fresh complete History, source authentication, resources and root admission pending'],
        spec=spec_ref, source_sha256=pins, launch_authorized=False)
    # Shape validation only, exactly as in the integer preparer. Never publish
    # the temporary passing envelope or give it to a runtime/History method.
    c.validate(spec, dict(draft, passed=True, blockers=[], launch_authorized=True), spec_ref)
    candidate = dict(schema='qwen35_kv256_alias_diagnostic_binding_candidate_v1', paths=copy.deepcopy(paths),
        spec_ref=spec_ref, overlay_ref=overlay_ref, review_draft_ref=object_ref(paths['review_draft'], draft),
        source_review=future['diagnostic_source_review'], preparer_source_review=future['preparer_source_review'],
        seed_binding=future['seed_binding'], catalog=future['catalog'], idle_handoff=future['idle_handoff'],
        prerequisite_exit=future['prerequisite_exit'], prerequisite_tool=copy.deepcopy(tool),
        preparation_only_inputs=preparation_only, catalog_rows=len(catalog['runs']), source_files=len(pins),
        preservation=dict(original_catalog_rows=len(old_catalog['runs']), all_prior_rows_in_order=True,
            original_recovered_sources=len(recovered), original_base_inputs=len(base['inputs']),
            lfm_scope_preserved=seed['lfm_r2_scope'], seed_lfm_children=binding['lfm_children'],
            full_history_children_recheck_pending=True, original_resource_limits_preserved=True, exact_diagnostic_workload=True, full_geometry_qualified=False,
            exact_retained_typed_inputs=retained_large_inputs, ordinary_runtime_metadata_cap_bytes=8*MIB),
        root_review_pending=True, full_history_precheck_pending=True, physical_freshness_pending=True,
        actual_resource_check_pending=True, source_authentication_pending=True, passed=False,
        launch_authorized=False, physical_pair_performed=False, lock_acquired=False,
        native_executed=False, compiler_executed=False, runtime_launched=False,
        full_geometry_qualified=False, full_model_qualified=False, new_speedup_claim=False)
    values = dict(overlay=overlay, spec=spec, review_draft=draft, candidate=candidate)
    for value in values.values():
        encode(value)
    guard()
    return dict(values=values, observed=observed)


def fresh_path(path):
    p = Path(path)
    if not p.is_absolute() or str(p) != path or '..' in p.parts or os.path.lexists(p):
        return False
    return p.parent.is_dir() and all(not parent.is_symlink() for parent in p.parents)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True); parser.add_argument('--request-sha256', required=True)
    args = parser.parse_args(argv)
    require(sys.dont_write_bytecode and __debug__, 'python -B without optimization required')
    c, encode, _, _, _, _, _ = api()
    from prepare_qwen35_sampler_r6_candidate_v1 import JsonMetadata
    from qwen35_sampler_runtime_io_v12 import identity, ref
    reader = JsonMetadata(lambda: None)
    request_ref = dict(path=args.request, sha256=args.request_sha256)
    future = reader.read(request_ref)['value']
    require(reader.observed[args.request]['identity']['size'] <= MIB, 'Bounded explicit request')
    verify_executed_preparer(reader.read(future['preparer_delivery'])['value'])
    result = prepare(future, reader.read, fresh_path, lambda: None)
    # Still only source/text and resource observations; never History or lock.
    c.verify_delivery(result['values']['spec']); c.resources(result['values']['spec'], reserve=True)
    require(all(identity(p) == v['identity'] for p, v in reader.observed.items()), 'No metadata identity refreshed')
    require(all(fresh_path(p) for p in future['paths'].values()), 'Every future path still fresh')
    for key, value in result['values'].items():
        with Path(future['paths'][key]).open('xb') as stream:
            stream.write(encode(value)); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(dict(candidate=ref(future['paths']['candidate'], 64*MIB),
        spec=ref(future['paths']['spec'], 64*MIB), runtime_launched=False, full_history_precheck_pending=True)))


if __name__ == '__main__':
    main()
