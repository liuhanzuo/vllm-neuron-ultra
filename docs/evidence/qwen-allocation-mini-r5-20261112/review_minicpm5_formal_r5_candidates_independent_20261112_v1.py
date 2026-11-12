"""Read exact saved r5 candidates and source only; no runtime/history execution."""
import ast
import copy
import datetime
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

Q = Path('/home/ubuntu/experiments/trn1-latest-models-20261031/research')
PREFIX = 'minicpm5-formal-import-r5-20261112'
PREPARATION = dict(path=str(Q / (PREFIX + '-preparation-receipt.json')),
    sha256='44f676005505b633aa1a1e68ba3418c9875277aed871390350db1a1eb355074d')
FINALIZER = dict(path=str(Q / 'finalize_minicpm5_formal_r5_root_20261112_v2.py'),
    sha256='333e4445429d8b2ee6c054b23b3c79b78d46c6ae941d54960c0a3736156f6ca5')
REVIEW = Q / 'minicpm5-formal-import-r5-independent-content-review-20261112-v1.json'


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False) + '\n').encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def ref(path):
    path = Path(path)
    return dict(path=str(path), sha256=digest(path.read_bytes()))


def read(reference):
    raw = Path(reference['path']).read_bytes()
    assert digest(raw) == reference['sha256'], reference['path']
    if 'bytes' in reference:
        assert len(raw) == reference['bytes']
    return json.loads(raw)


def load(name):
    return json.loads((Q / name).read_bytes())


def source(path, expected):
    path = Path(path)
    assert path.suffix in ('.py', '.json'), str(path)
    raw = path.read_bytes()
    assert digest(raw) == expected, str(path)
    return raw


def main():
    prep = read(PREPARATION)
    assert prep['schema'] == 'minicpm5_formal_import_root_preparation_receipt_v2'
    assert prep['source_metadata_only'] is True and prep['reviews_still_pending'] is True
    assert prep['runtime_admitted'] is False and prep['launch_authorized'] is False
    assert prep['original_nine_native_checks_pending'] == 9
    assert prep['original_scientific_rows_still_pending'] == 1284
    names = {'complete-source-contract', 'complete-source-map', 'diagnostic-review-draft',
        'diagnostic-spec', 'native-policy-inputs', 'original-preparation', 'original-request',
        'outer-entry-plan', 'outer-native-policy-draft', 'outer-review-draft', 'outer-spec',
        'pv-review-draft', 'pv-spec', 'request', 'sdk-entry-template', 'sdk-native-policy-draft'}
    assert set(prep['artifacts']) == names and len(names) == 16
    values = {}
    for name, row in prep['artifacts'].items():
        assert row['reference']['path'] == str(Q / (PREFIX + '-' + name + '.json'))
        raw = Path(row['reference']['path']).read_bytes()
        value = read(row['reference'])
        assert len(raw) == row['bytes'] and encoded(value) == raw
        values[name] = value
    refs = {name: row['reference'] for name, row in prep['artifacts'].items()}
    pins = values['complete-source-map']
    assert len(pins) == prep['complete_source_files'] == 48223
    assert digest(encoded(pins).rstrip(b'\n')) == prep['complete_source_map_sha256']
    inherited = load('minicpm5-formal-import-r2-20261112-complete-source-map.json')
    assert len(inherited) == 48057 and all(pins.get(p) == h for p, h in inherited.items())
    assert values['complete-source-contract'] == load('minicpm5-formal-import-r2-20261112-complete-source-contract.json')
    read(values['complete-source-contract'])
    added = {p: h for p, h in pins.items() if p not in inherited}
    for path, sha in added.items():
        source(path, sha)
    catalog = read(prep['catalog'])
    assert len(catalog['runs']) == prep['catalog_rows'] == 155
    assert catalog['reviewer'] == '/root' and catalog['passed'] is True and catalog['blockers'] == []
    older = load('minicpm5-operator-post-r2-serial-workloads-root-catalog-20261112-v104.json')
    assert len(older['runs']) == 153 and catalog['runs'][:153] == older['runs']
    assert all(pins.get(p) == h for row in catalog['runs'] for p, h in row['source_sha256'].items())
    assert catalog['runs'][-1]['terminal'] == prep['predecessor']
    predecessor = read(prep['predecessor'])
    handoff = read(prep['handoff'])
    request = values['request']
    assert request['current_handoff'] == prep['handoff']
    assert handoff[request['handoff_catalog_field']] == prep['catalog']['path']
    assert request['expected_predecessor_exit'] == prep['predecessor']
    assert request['original_request'] == refs['original-request']
    assert values['original-request']['current_serial_catalog'] == prep['catalog']
    assert prep['request'] == refs['request']
    old = values['original-preparation']
    pv, diag, spec = [values[k] for k in ('pv-spec', 'diagnostic-spec', 'outer-spec')]
    outer, sdk = values['outer-entry-plan'], values['sdk-entry-template']
    assert old['pv_spec_candidate'] == pv and old['catalog_rows'] == 155
    assert old['complete_source_files'] == len(pins)
    assert old['complete_source_map_sha256'] == prep['complete_source_map_sha256']
    assert old['extension_source_files'] == len(pv['source_extension']) == 384
    assert old['inherited_source_files'] + old['extension_source_files'] == len(pins)
    packet = dict(source_contract=values['complete-source-contract'], source_sha256=pins)
    assert len(encoded(packet)) == old['complete_source_release_bytes'] == 8431863
    assert old['original_pipe_reservation_bytes'] == 65536
    assert len(encoded(packet)) + 65536 <= 16 * 2**20
    assert old['all_current_catalog_sources_retained'] and old['all_original_catalog_rows_retained']
    assert old['original_runtime_reconstruction_routes_agree']
    assert old['original_runtime_full_authentication_still_required']
    assert not old['runtime_admitted'] and not old['full_model_qualified'] and not old['new_speedup_claim']
    assert all(pins.get(p) == h for p, h in pv['source_extension'].items())
    limits = dict(address_space_bytes=2147483648, cpu=6, cpu_seconds=600,
                  hard_cpu_seconds=605, nice=19, output_file_bytes=2097152, wall_seconds=900)
    caps = dict(allocated_bytes=33554432, metadata_bytes=8388608, one_file_bytes=2097152,
                raw_bytes=12288000, raw_files=4392)
    assert pv['limits'] == diag['limits'] == spec['limits'] == limits
    assert pv['artifact_limits'] == diag['artifact_limits'] == spec['artifact_limits'] == caps
    assert pv['current_serial_catalog'] == prep['catalog']
    assert pv['worker'] == diag['worker'] == str(Q / 'minicpm5_pv_sdk_import_diagnostic_worker_v1.py')
    assert pv['output_root'] == values['original-request']['output_root']
    for role in ('pv', 'diagnostic', 'outer'):
        review, bound = values[role + '-review-draft'], values[role + '-spec']
        assert review['reviewer'] is None and review['passed'] is False
        assert review['blockers'] == ['Independent root review of these exact bytes is pending']
        assert review['spec'] == refs[role + '-spec']
        assert review['source_extension'] == bound['source_extension']
    assert diag['pv_spec'] == refs['pv-spec'] and diag['pv_review'] == refs['pv-review-draft']
    assert diag['source_extension'] == pv['source_extension']
    assert spec['diagnostic_spec'] == refs['diagnostic-spec'] and spec['diagnostic_review'] == refs['diagnostic-review-draft']
    assert spec['complete_source_map'] == refs['complete-source-map']
    assert spec['sdk_entry_template'] == refs['sdk-entry-template']
    assert spec['attempt_id'] == request['attempt_id'] and spec['unit'] == request['unit']
    assert spec['sdk_import_paths'] == ['/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages']
    assert json.loads(outer['successor_input']) == dict(spec=refs['outer-spec'], review=refs['outer-review-draft'])
    assert sdk['successor_input'] == 'admission:self'
    assert outer['entry_module'] == 'research.minicpm5_sdk_mapping_outer_source_entry_v9'
    assert sdk['entry_module'] == 'research.minicpm5_sdk_mapping_worker_source_entry_v6'
    assert outer['entry_path'] == str(Q / 'minicpm5_sdk_mapping_outer_source_entry_v9.py')
    assert sdk['entry_path'] == str(Q / 'minicpm5_sdk_mapping_worker_source_entry_v6.py')
    bootstrap = outer['bootstrap_sources']
    assert bootstrap == sdk['bootstrap_sources'] == spec['source_extension'] and len(bootstrap) == 172
    for path, sha in bootstrap.items():
        assert pins.get(path) == sha
        source(path, sha)
    binder_tree = ast.parse((Q / 'prepare_minicpm5_catalog_formal_binding_v3.py').read_text())
    runtime = next(ast.literal_eval(n.value) for n in binder_tree.body if isinstance(n, ast.Assign)
                   and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'RUNTIME')
    pending = {str(Q / name) for name in runtime}
    visited = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        raw = source(path, bootstrap[path])
        for node in ast.walk(ast.parse(raw)):
            if isinstance(node, ast.ImportFrom) and node.module == 'research':
                pending.update(str(Q / (item.name + '.py')) for item in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('research.'):
                pending.add(str(Q.parent / (node.module.replace('.', '/') + '.py')))
    assert len(visited) == 25
    for role, authority_role in (('outer', 'outer312'), ('sdk', 'sdk310')):
        policy = values[role + '-native-policy-draft']
        prior = load('minicpm5-formal-import-r1-20261112-' + role + '-native-policy-draft.json')
        assert policy['source_map_sha256'] == prep['complete_source_map_sha256']
        assert {k: v for k, v in policy.items() if k != 'source_map_sha256'} == {
            k: v for k, v in prior.items() if k != 'source_map_sha256'}
        assert policy['accepted'] is False and policy['reviewer'] is None
        assert policy == values['native-policy-inputs']['policies'][authority_role]
        plan = values[role + '-entry-template'] if role == 'sdk' else outer
        assert {k: plan['native_authority'][k] for k in ('path', 'sha256')} == refs[role + '-native-policy-draft']
    # Only pure, inspected argv-building functions execute; no project imports.
    launch_namespace = {}
    exec(compile((Q / 'minicpm5_sdk_mapping_nocache_launch_v1.py').read_text(), '<pure-launch>', 'exec'), launch_namespace)
    launch_namespace['separate_authorities'](outer, sdk)
    literal = source(prep['entry_source']['path'], prep['entry_source']['sha256'])
    assert len(literal) == 37178 and digest(literal) == 'bbf5affece28e3d6356ce27d48ccdb7f4d707fbf6717808a69ca1f61222763d0'
    assert spec['sdk_entry_source'] == prep['entry_source'] and prep['outer_entry_plan'] == refs['outer-entry-plan']
    argv = launch_namespace['launch'](literal, prep['entry_source'], outer, refs['outer-entry-plan'])
    assert argv[7] == literal.decode() and len(argv) == 15
    prior_outer = (Q / 'minicpm5_sdk_mapping_outer_source_entry_v8.py').read_text()
    current_outer = source(outer['entry_path'], bootstrap[outer['entry_path']]).decode()
    assert prior_outer.replace('minicpm5_fixed_input_pv_catalog_auth_v1', 'minicpm5_fixed_input_pv_catalog_auth_v2') == current_outer
    worker = source(sdk['entry_path'], bootstrap[sdk['entry_path']]).decode()
    assert "len(source_receipt['native_sha256'])==9" in worker
    assert "plan=original.restore_descriptor_plan(witness,metadata.read)" in worker
    assert "allowed=original.core.saved.validate_plan(plan)" in worker
    assert "require(not io._read" in worker
    assert "'complete_source_map_sha256'" in worker and "'complete_source_files'" in worker
    adapter = source(Q / 'minicpm5_layer1_fixed_input_pv_adapter_v3.py', pins[str(Q / 'minicpm5_layer1_fixed_input_pv_adapter_v3.py')]).decode()
    assert "len(witness['comparison_reports']) == 6" in adapter and 'Whole original1284 descriptor plan reconstructed' in adapter
    assert old['descriptor_plan_sha256'] == '1838866c752ef628bda60cb63d2c9938a2009d0fb09fa77e5539802e36206294'
    metadata_exit_ref = ref(Q / (PREFIX + '-metadata-root-actual-exit.json'))
    ended = read(metadata_exit_ref)
    assert ended['preparation'] == PREPARATION
    assert ended['actual_tool'] == dict(exit_code=0, kind='tool_session', session_id=1631, terminal_chunk='6906f6')
    assert ended['actual_tool_exit_observed'] and ended['owners_gone'] and ended['original_lock_reacquired_and_released']
    assert ended['physically_drained'] and ended['no_children_or_units_created'] and not ended['native_executed']
    assert len(ended['two_observations']) == 2 and all(x == dict(active_heavy_absent=True, owner_gone=True) for x in ended['two_observations'])
    read(ended['terminal'])
    # Isolated reader mock: execute this one nested pure helper, never finalizer main.
    finalizer_raw = source(FINALIZER['path'], FINALIZER['sha256'])
    before = (Q / 'finalize_minicpm5_formal_r5_root_20261112_v1.py').read_text()
    after = finalizer_raw.decode()
    old_reader = "  fake=SimpleNamespace(role='outer312',history_owner=None,plan=plan,read_source=lambda path,sha:(pathlib.Path(path).read_bytes(),None))"
    new_reader = "  def read_source(path,sha):\n   raw=pending[path] if path in pending else pathlib.Path(path).read_bytes()\n   assert hashlib.sha256(raw).hexdigest()==sha\n   return raw,None\n  fake=SimpleNamespace(role='outer312',history_owner=None,plan=plan,read_source=read_source)"
    assert before.replace(old_reader, new_reader) == after
    tree = ast.parse(after)
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'read_source')
    virtual_path, virtual_raw = '/never-created/minicpm5-review-mock.json', b'{"mock":true}\n'
    import pathlib
    namespace = dict(pending={virtual_path: virtual_raw}, pathlib=pathlib, hashlib=hashlib)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<source-reader-mock>', 'exec'), namespace)
    reader = namespace['read_source']
    assert reader(virtual_path, digest(virtual_raw)) == (virtual_raw, None)
    assert reader(PREPARATION['path'], PREPARATION['sha256'])[0] == Path(PREPARATION['path']).read_bytes()
    for path in (virtual_path, PREPARATION['path']):
        try:
            reader(path, '0' * 64)
        except AssertionError:
            pass
        else:
            raise AssertionError('Both reader branches must reject an incorrect digest')
    result = dict(schema='minicpm5_formal_import_independent_content_review_v1',
        reviewer='/root/mini_formal_native_review_20261112', observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        preparation=PREPARATION, source_only_accepted=True, passed=True, blockers=[],
        review_source=ref(__file__), finalizer_source=FINALIZER, finalizer_source_accepted=True,
        preparation_actual_exit=metadata_exit_ref, canonical_artifacts_checked=16,
        catalog=prep['catalog'], catalog_rows=155, full_v104_prefix_rows_retained=153,
        complete_source_files=48223, inherited_r2_pins_retained=48057,
        added_source_files_hashed=len(added), bootstrap_files_hashed=172, recursive_bootstrap_import_files=25,
        pv_extension_sources=384, complete_source_release_bytes=8431863, original_pipe_reservation_bytes=65536,
        full_literal_bytes=37178, literal_sha256=prep['entry_source']['sha256'],
        original_nine_native_checks_pending=9, original_scientific_rows_still_pending=1284,
        descriptor_plan_sha256=old['descriptor_plan_sha256'],
        preserved_gates=['cold cache', 'long decode', 'request slot reuse', 'quality', 'HTTP'],
        checks=['All 16 exact canonical draft artifacts and their internal reference edges verified',
            'Full saved catalog source pins retained; original r2 source pins and source contract preserved',
            'Policy bodies equal r1 drafts except complete source map SHA; root approval still pending',
            'Source-only successor keeps accepted outer8 tuple/list boundary; outer9 changes catalog auth import only',
            'Worker6 retains real nine-native check, whole original descriptor reconstruction and zero tensor reads',
            'Full literal argv, separate interpreter authorities and 172-file closure retained',
            'Finalizer v2 changes only pending-aware reader; isolated mock checks both branches and SHA rejection',
            'Saved metadata root exit checked; no current freshness or physical claim independently made'],
        runtime_launched=False, runtime_binding_created=False, current_proc_read=False,
        full_history_reconstruction_executed=False, native_files_rehashed=False, payloads_read=False,
        finalizer_main_executed=False, native_ABI_qualified=False, full_model_qualified=False, new_speedup_claim=False)
    with REVIEW.open('xb') as stream:
        stream.write(encoded(result))
    print(json.dumps(dict(review=ref(REVIEW), bytes=REVIEW.stat().st_size, passed=True), sort_keys=True))


if __name__ == '__main__':
    main()
