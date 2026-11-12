"""Latest saved v98 shape and real producer/validator methods; no native run.

Original functions loaded as AST only in these stdlib mocks. No runtime module
uses AST/code extraction. Saved history is not re-labelled as new authentication.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

BLOCKED = {'ctypes', '_ctypes', 'torch', 'numpy', 'neuronxcc', 'torch_neuronx',
           'neuronx_distributed', 'neuronx_distributed_inference'}
class NoNative:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in BLOCKED or fullname in {
            'research.prepare_minicpm5_pv_sdk_import_diagnostic_v1',
            'research.minicpm5_layer1_fixed_input_pv_prerequisites_v2',
            'research.minicpm5_pv_sdk_import_diagnostic_v1'}:
            raise AssertionError('No ordinary historical/native import in source tests: ' + fullname)
assert not BLOCKED.intersection(sys.modules)
sys.meta_path.insert(0, NoNative())
from research import minicpm5_completed_qwen_package_text_v1 as completed
from research import minicpm5_fixed_input_pv_catalog_auth_v1 as runtime
from research import prepare_minicpm5_pv_sdk_import_catalog_v2 as preparation
from research import prepare_minicpm5_catalog_formal_binding_v1 as wiring
from research import minicpm5_sdk_mapping_outer_source_entry_v7 as outer
from research import minicpm5_sdk_mapping_observer_logic_v1 as logic

Q = Path(__file__).resolve().parent
ROOT = Q.parent
MEASUREMENTS = []
CACHE = {}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def ref(path):
    path = Path(path)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def read(reference):
    completed.codec.reference(reference)
    path = Path(reference['path'])
    assert path.is_absolute() and path.suffix == '.json' and path.resolve() == path
    key = (str(path), reference['sha256'])
    if key not in CACHE:
        raw = path.read_bytes()
        assert 0 < len(raw) <= 16 * 2**20 and hashlib.sha256(raw).hexdigest() == reference['sha256']
        if 'bytes' in reference: assert len(raw) == reference['bytes']
        CACHE[key] = json.loads(raw)
    return copy.deepcopy(CACHE[key])


def text(path, sha=None):
    path = Path(path)
    assert path.is_absolute() and path.suffix in ('.py', '.json', '.md', '.txt', '.sh') and path.resolve() == path
    raw = path.read_bytes()
    assert sha is None or hashlib.sha256(raw).hexdigest() == sha
    return raw


def saved(name):
    return read(ref(Q / ('minicpm5-formal-import-r2-20261112-' + name + '.json')))


def source_function(filename, name, namespace):
    path = Q / filename
    tree = ast.parse(path.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def original_lfm():
    tree = ast.parse((Q / 'audit_minicpm5_layer1_saved_operators_v1.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'COMPLETED_LFM_TEXT' for t in n.targets))
    return ast.literal_eval(node.value)


def fixture_original():
    old = saved('original-preparation')
    oldpins = saved('complete-source-map')
    inherited = {p: s for p, s in oldpins.items() if p not in old['pv_spec_candidate']['source_extension']}
    request = read(ref(Q / 'minicpm5-formal-import-r3-20261112-original-request.json'))
    catalog = read(completed.HISTORICAL_CATALOG)
    ended = read(request['independent_audit_exit'])
    audit_spec = read(ended['audit_spec'])
    ns = dict(copy=copy, require=completed.require)
    merge = source_function('prepare_minicpm5_pv_sdk_import_diagnostic_v1.py', 'merge', ns)
    mandatory = tuple(old['mandatory_sources'])
    disjoint = source_function('prepare_minicpm5_pv_sdk_import_diagnostic_v1.py', 'disjoint_extension',
                               dict(ns, MANDATORY=mandatory, merge=merge))
    merged = source_function('audit_minicpm5_layer1_operator_native_result_v4.py', 'merged_plan', dict(ns))
    merge_extensions = source_function('minicpm5_layer1_fixed_input_pv_prerequisites_v2.py', 'merge_extensions', dict(ns))
    catalog_rows = source_function('compare_minicpm5_layer1_saved_operators_v1.py', 'catalog_rows', dict(ns))
    def pin_shape(pins):
        assert type(pins) is dict and len(pins) <= 100000
        for p, s in pins.items(): completed.codec.reference(dict(path=p, sha256=s))
    serial_catalog = source_function('minicpm5_layer1_operator_sdk_runtime_completion_v4.py', 'serial_catalog',
        dict(ns, sources=NS(reference=completed.codec.reference, metadata=NS(pin_shape=pin_shape)),
             sdk=NS(BINDING=catalog['completed_sdk']), re=re, MAX_SERIAL_RUNS=256,
             physical_evidence=NS(unit_name=lambda unit: unit)))
    r = NS(digest=digest, life=NS(ref=completed.codec.reference), p=NS(gone=Mock()),
           completion=NS(latest=NS(serial_catalog=serial_catalog)))
    serial = NS(encoded=completed.codec.encoded, CAPS=logic.CAPS, exact=lambda a, b: a == b,
                __file__=str(Q / 'minicpm5_layer1_fixed_input_pv_serial_v1.py'),
                leaf=NS(__file__=str(Q / 'minicpm5_layer1_fixed_input_pv_leaf_v1.py')),
                core=NS(__file__=str(Q / 'minicpm5_layer1_fixed_input_pv_core_v1.py')))
    previous = NS(catalog_rows=catalog_rows, preparation=NS(executed_extension=Mock(return_value={})))
    audited = NS(merged_plan=merged)
    prior = NS(COMPLETED_LFM_TEXT=original_lfm(), LIMITS=logic.LIMITS, previous=previous, audited=audited)
    auth = NS(__file__=runtime.ORIGINAL_PV, r=r, serial=serial, prior=prior, previous=previous, audited=audited,
              ROOT=ROOT, merge_extensions=merge_extensions, audit_terminal=Mock(return_value=ended['actual_tool']))
    original = NS(__file__=str(Q / 'minicpm5_layer1_fixed_input_pv_adapter_v3.py'), PIPE_CAP=16 * 2**20)
    runtime_sources = tuple(str(Q / name) for name in ('minicpm5_pv_sdk_import_diagnostic_v1.py',
        'run_minicpm5_pv_sdk_import_diagnostic_v1.py', 'minicpm5_pv_sdk_import_diagnostic_worker_v1.py'))
    diagnostic = NS(auth=auth, serial=serial, prior=prior, r=r, original=original, RUNTIME_SOURCES=runtime_sources)
    contract = saved('complete-source-contract')
    base = dict(plan=dict(source_sha256=inherited, source_contract=contract),
        comparison_plan=dict(source_sha256={p:s for p,s in inherited.items() if p not in audit_spec['source_extension']}, source_contract=contract),
        audit_spec=audit_spec, resolved={'plan': {'latest_completion': {'explicit_mock_only': True}}},
        old_catalog={'runs': catalog['runs'][:100]}, ended=ended, tool=ended['actual_tool'], witness_cache={},
        source_binding=old['source_binding'], descriptor_plan_sha256=old['descriptor_plan_sha256'])
    preparer = NS(__file__=preparation.ORIGINAL, diagnostic=diagnostic, merge=merge, disjoint_extension=disjoint,
        MANDATORY=mandatory, ORIGINAL_ADAPTER_SHA='50f7bedf915c3fbf83ce5e623f84fd391b8d270dc29135bd4b6c357ecbdbbba0',
        REQUEST_SCHEMA='minicpm5_sdk_import_future_binding_request_v1', RESULT_SCHEMA=old['schema'],
        reconstruct_audit_base=Mock(return_value=base), audited=audited)
    preparer.validated_review_sources = source_function('prepare_minicpm5_pv_sdk_import_diagnostic_v1.py', 'validated_review_sources',
        dict(ns, read=read, text=text, r=r, diagnostic=diagnostic, hashlib=hashlib,
             SOURCE_REVIEW_SCHEMA='minicpm5_pv_sdk_import_diagnostic_root_source_review_v1'))
    return preparer, request, base


def produced():
    original, request, base = fixture_original()
    request['additional_source_refs'] = [ref(Q / name) for name in wiring.SOURCES]
    value = preparation.prepare(original, request, read=read, text=text, guard=Mock())
    full = original.audited.merged_plan(base['plan'], value['pv_spec_candidate']['source_extension'])
    return original, request, base, value, full


class Checks(unittest.TestCase):
    def test_exact_root_cause_seventy_repo_pins_and_actual_v98_provenance(self):
        old = saved('original-preparation'); oldpins = saved('complete-source-map')
        inherited = {p:s for p,s in oldpins.items() if p not in old['pv_spec_candidate']['source_extension']}
        catalog = read(completed.HISTORICAL_CATALOG)
        bad = {}; sdk_retained = set()
        for row in catalog['runs']:
            for path, sha in row['source_sha256'].items():
                if not path.startswith(str(ROOT) + '/'):
                    if path not in inherited and original_lfm().get(path) != sha: bad[path] = sha
                    elif '/venvs/vllm_neuron/' in path: sdk_retained.add(path)
        self.assertEqual(bad, completed.QWEN_PACKAGE_TEXT)
        self.assertEqual(len(bad), 70)
        self.assertEqual(completed.authenticate_completed_catalog(completed.HISTORICAL_CATALOG, read), bad)
        numpy = '/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/numpy/testing/_private/utils.py'
        self.assertIn(numpy, inherited)
        self.assertIn(numpy, catalog['runs'][146]['source_sha256'])
        for path, sha in bad.items(): self.assertEqual(hashlib.sha256(text(path, sha)).hexdigest(), sha)

    def test_full_frozen_spec_review_terminal_bundle_edges_and_ordinary_runtime_reader_cap(self):
        big_refs = []
        for proof in completed.ENDED:
            ended, spec, review, bundle = (read(proof[key]) for key in ('terminal', 'spec', 'review', 'bundle'))
            self.assertEqual(ended['spec'], proof['spec']); self.assertEqual(ended['review'], proof['review'])
            self.assertEqual(spec['bundle'], proof['bundle']); self.assertEqual(review['spec'], proof['spec'])
            self.assertEqual(review['reviewer'], '/root'); self.assertTrue(review['passed'])
            self.assertEqual(review['blockers'], []); self.assertTrue(review['launch_authorized'])
            self.assertEqual(review['source_sha256'], spec['source_sha256'])
            self.assertTrue(all(spec['source_sha256'].get(p) == s for p,s in completed.QWEN_PACKAGE_TEXT.items()))
            self.assertEqual({bundle['checkout']+'/'+p: s for p,s in bundle['package_sources'].items()}, completed.QWEN_PACKAGE_TEXT)
            big_refs.extend([proof['spec'], proof['review']])
        calls = []
        def capped_read(reference):
            self.assertNotIn(reference, big_refs)
            self.assertLessEqual(Path(reference['path']).stat().st_size, 8 * 2**20)
            calls.append(reference)
            return read(reference)
        self.assertEqual(completed.authenticate_completed_catalog(completed.HISTORICAL_CATALOG, capped_read), completed.QWEN_PACKAGE_TEXT)
        self.assertEqual(len(calls), 5)

    def test_qwen_v5_eighteen_helper_python_pins_already_inherited(self):
        old = saved('original-preparation'); oldpins = saved('complete-source-map')
        inherited = {p:s for p,s in oldpins.items() if p not in old['pv_spec_candidate']['source_extension']}
        bundle_ref = ref(Q / 'qwen35-package-native-preflight-source-bundle-20261112-v5.json')
        bundle = read(bundle_ref)
        helpers = [row for row in bundle['pjrt_helper_sources'] if row['path'].endswith('.py')]
        self.assertEqual(len(helpers), 18)
        self.assertTrue(all(inherited.get(row['path']) == row['sha256'] for row in helpers))
        extensionless = [row for row in bundle['pjrt_helper_sources'] if not row['path'].endswith('.py')]
        self.assertEqual(len(extensionless), 1)
        self.assertNotIn(extensionless[0]['path'], inherited)
        self.assertNotIn(extensionless[0]['path'], completed.QWEN_PACKAGE_TEXT)
        with self.assertRaises(ValueError):
            completed.extension_shape(dict(completed.QWEN_PACKAGE_TEXT, **{extensionless[0]['path']: extensionless[0]['sha256']}),
                completed_lfm_text=original_lfm(), catalog_ref=completed.HISTORICAL_CATALOG, read=read)
        MEASUREMENTS.append(dict(scope='read-only v5 source bundle; no later current catalog or runtime binding',
            bundle=bundle_ref, helper_python_count=18, all_helper_python_pins_already_inherited=True,
            extensionless_helper_is_not_admitted_by_new_text_domain=True))

    def test_shared_shape_rejects_unknown_changed_partial_or_missing_terminal_evidence(self):
        for change in ('hash', 'unknown', 'partial', 'missing_terminal', 'not_ended', 'wrong_bundle', 'missing_row'):
            pins = dict(completed.QWEN_PACKAGE_TEXT)
            def changed_read(reference):
                if change == 'missing_terminal' and reference == completed.ENDED[0]['terminal']:
                    raise FileNotFoundError('mock absent actual terminal')
                value = read(reference)
                if change == 'not_ended' and reference == completed.ENDED[0]['terminal']: value['actual_tool_exit_observed'] = False
                if change == 'wrong_bundle' and reference == completed.ENDED[1]['bundle']: value['package_sources'].pop(next(iter(value['package_sources'])))
                if change == 'missing_row' and reference == completed.HISTORICAL_CATALOG: value['runs'].pop(145)
                return value
            if change == 'hash': pins[next(iter(pins))] = '0' * 64
            elif change == 'unknown': pins['/home/ubuntu/src/unknown/foreign.py'] = 'f' * 64
            elif change == 'partial': pins.pop(next(iter(pins)))
            with self.subTest(change=change), self.assertRaises((ValueError, FileNotFoundError)):
                completed.extension_shape(pins, completed_lfm_text=original_lfm(),
                    catalog_ref=completed.HISTORICAL_CATALOG, read=changed_read)

    def test_real_preparation_body_both_original_merges_and_all147_catalog_rows(self):
        original, request, base, result, full = produced()
        self.assertEqual(result['catalog_rows'], 147)
        self.assertTrue(result['all_original_catalog_rows_retained'])
        self.assertTrue(result['all_current_catalog_sources_retained'])
        self.assertFalse(result['runtime_admitted'])
        self.assertEqual(result['descriptor_plan_sha256'], saved('original-preparation')['descriptor_plan_sha256'])
        self.assertTrue(all(full['source_sha256'].get(p) == s for p,s in base['plan']['source_sha256'].items()))
        self.assertTrue(all(full['source_sha256'].get(p) == s for row in read(completed.HISTORICAL_CATALOG)['runs'] for p,s in row['source_sha256'].items()))
        extension = result['pv_spec_candidate']['source_extension']
        self.assertFalse(set(extension) & set(base['plan']['source_sha256']))
        for path in completed.QWEN_PACKAGE_TEXT: self.assertEqual(extension[path], completed.QWEN_PACKAGE_TEXT[path])
        old_shape = source_function('prepare_minicpm5_pv_sdk_import_diagnostic_v1.py', 'extension_shape',
            dict(require=completed.require, ROOT=ROOT, Path=Path, re=re, prior=original.diagnostic.prior))
        with self.assertRaises(ValueError): old_shape(extension)
        with self.assertRaises(ValueError): original.audited.merged_plan(base['plan'], {next(iter(base['plan']['source_sha256'])): 'f' * 64})
        MEASUREMENTS.append(dict(scope='actual preserved preparation body with saved v98 catalog and explicitly mocked original audit reconstruction',
            catalog_rows=result['catalog_rows'], inherited_source_files=result['inherited_source_files'],
            extension_source_files=result['extension_source_files'], complete_source_files=result['complete_source_files'],
            complete_source_release_bytes=result['complete_source_release_bytes'], external_new_exact_pins=70))

    def test_actual_runtime_validate_and_authenticate_before_original_reconstruct(self):
        original, request, base, result, full = produced()
        pv = result['pv_spec_candidate']; pv_ref = {'path': str(Q / 'source-test-pv.json'), 'sha256': digest(pv)}
        review_ref = {'path': str(Q / 'source-test-review.json'), 'sha256': 'b' * 64}
        review = dict(schema='minicpm5_fixed_input_pv_serial_source_review_v1', reviewer='/root', passed=True,
            blockers=[], spec=pv_ref, source_extension=pv['source_extension'])
        auth = original.diagnostic.auth
        old_validate = source_function('minicpm5_layer1_fixed_input_pv_prerequisites_v2.py', 'validate',
            dict(require=completed.require, r=auth.r, serial=auth.serial, prior=auth.prior, ROOT=ROOT,
                 Path=Path, re=re, __file__=runtime.ORIGINAL_PV))
        with self.assertRaises(ValueError): old_validate(pv_ref, review_ref, pv, review)
        self.assertEqual(runtime.validate(auth, pv_ref, review_ref, pv, review, read=read, guard=Mock()), Path(pv['output_root']))
        def metadata_read(reference):
            if reference == pv_ref: return copy.deepcopy(pv)
            if reference == review_ref: return copy.deepcopy(review)
            return read(reference)
        class ReachedOriginalReconstruct(Exception): pass
        def reconstruct(*args):
            self.assertTrue(all(args[2]['source_extension'][p] == sha for p,sha in completed.QWEN_PACKAGE_TEXT.items()))
            raise ReachedOriginalReconstruct()
        auth.prior.reconstruct = reconstruct
        with self.assertRaises(ReachedOriginalReconstruct):
            runtime.authenticate(auth, pv_ref, review_ref, lease=NS(), metadata=NS(read=metadata_read), guard=Mock())
        auth.audit_terminal.assert_called_once()
        # Same actual complete authenticate method is reached from the actual
        # gate successor, after its original state/descriptor/shared-metadata checks.
        diagnostic_ref = {'path': str(Q / 'source-test-diagnostic.json'), 'sha256': 'c'*64}
        diagnostic_review = {'path': str(Q / 'source-test-diagnostic-review.json'), 'sha256': 'd'*64}
        diagnostic_spec = dict(pv_spec=pv_ref, pv_review=review_ref)
        def combined_read(reference):
            if reference == diagnostic_ref: return diagnostic_spec
            if reference == diagnostic_review: return {'explicit_mock_only': True}
            return metadata_read(reference)
        metadata = NS(read=combined_read)
        lease = NS(admission={'exact': 'binding'}, metadata=metadata)
        guard = Mock(); guard.lease = lease
        gate = NS(state='history_loaded', bootstrap={'binding': lease.admission, 'source_sha256': full['source_sha256']},
            guard=Mock(), original=NS(auth=auth, validate_spec=Mock(), bind_authenticated=Mock(), r=auth.r))
        with self.assertRaises(ReachedOriginalReconstruct):
            runtime.authenticate_original(gate, diagnostic_ref, diagnostic_review, lease=lease, metadata=metadata, guard=guard)
        self.assertEqual(gate.state, 'failed')

    def test_actual_runtime_rejects_unknown_hash_or_absent_terminal_before_authentication(self):
        original, request, base, result, full = produced()
        auth = original.diagnostic.auth
        for change in ('unknown', 'hash', 'missing_terminal'):
            pv = copy.deepcopy(result['pv_spec_candidate'])
            if change == 'unknown': pv['source_extension']['/home/ubuntu/src/other.py'] = 'f'*64
            if change == 'hash': pv['source_extension'][next(iter(completed.QWEN_PACKAGE_TEXT))] = '0'*64
            spec_ref = dict(path=str(Q / 'source-test-pv.json'), sha256=digest(pv))
            review_ref = dict(path=str(Q / 'source-test-review.json'), sha256='a'*64)
            review = dict(schema='minicpm5_fixed_input_pv_serial_source_review_v1', reviewer='/root', passed=True,
                blockers=[], spec=spec_ref, source_extension=pv['source_extension'])
            def current_read(reference):
                if reference == spec_ref: return pv
                if reference == review_ref: return review
                if change == 'missing_terminal' and reference == completed.ENDED[0]['terminal']:
                    raise FileNotFoundError('actual pinned terminal unavailable fixture')
                return read(reference)
            with self.subTest(change=change), self.assertRaises((ValueError, FileNotFoundError)):
                runtime.authenticate(auth, spec_ref, review_ref, lease=NS(), metadata=NS(read=current_read), guard=Mock())
            auth.audit_terminal.assert_not_called()

    def test_current_v98_real_binding_orchestration_and_every_new_bootstrap_import(self):
        original, request, base = fixture_original()
        formal_request = read(ref(Q / 'minicpm5-formal-import-r3-20261112-request.json'))
        literal_manifest = ref(Q / 'minicpm5-formal-literal-source-manifest-20261112-v1.json')
        literal_preparer = ref(Q / 'prepare_minicpm5_literal_formal_binding_v1.py')
        additions = [ref(Q / name) for name in wiring.SOURCES]
        result = wiring.prepare(formal_request, literal_source_manifest=literal_manifest,
            preparation_source=literal_preparer, catalog_source_refs=additions,
            read=read, text=text, guard=Mock(), original=original)
        artifacts = result['artifacts']
        sdk, outer_plan = artifacts['sdk-entry-template']['value'], artifacts['outer-entry-plan']['value']
        bootstrap = outer_plan['bootstrap_sources']
        self.assertEqual(sdk['bootstrap_sources'], bootstrap)
        self.assertEqual(artifacts['outer-spec']['value']['source_extension'], bootstrap)
        self.assertEqual(outer_plan['entry_module'], 'research.minicpm5_sdk_mapping_outer_source_entry_v7')
        self.assertEqual(sdk['entry_module'], 'research.minicpm5_sdk_mapping_worker_source_entry_v6')
        pins = artifacts['complete-source-map']['value']
        for row in additions: self.assertEqual(pins[row['path']], row['sha256'])
        for name in wiring.RUNTIME: self.assertEqual(bootstrap[str(Q / name)], pins[str(Q / name)])
        visited, pending = set(), {str(Q / name) for name in wiring.RUNTIME}
        while pending:
            path = pending.pop()
            if path in visited: continue
            visited.add(path); self.assertIn(path, bootstrap)
            self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(), bootstrap[path])
            for node in ast.walk(ast.parse(Path(path).read_text())):
                if isinstance(node, ast.ImportFrom) and node.module == 'research':
                    pending.update(str(Q / (item.name + '.py')) for item in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('research.'):
                    pending.add(str(ROOT / (node.module.replace('.', '/') + '.py')))
        for name,row in artifacts.items():
            raw = wiring.launch.encoded(row['value'])
            self.assertEqual(len(raw), row['bytes']); self.assertEqual(hashlib.sha256(raw).hexdigest(), row['reference']['sha256'])
        self.assertEqual(result['catalog_rows'], 147)
        self.assertFalse(result['runtime_admitted']); self.assertFalse(result['launch_authorized'])
        self.assertEqual(result['original_nine_native_checks_pending'], 9)
        self.assertEqual(result['original_scientific_rows_still_pending'], 1284)
        literal_bytes = text(result['entry_source']['path'], result['entry_source']['sha256'])
        self.assertEqual(wiring.launch.launch(literal_bytes, result['entry_source'], outer_plan, result['outer_entry_plan'])[7], literal_bytes.decode())
        MEASUREMENTS.append(dict(scope='v98 actual source successor orchestration; original audit reconstruction explicitly mocked',
            complete_source_files=len(pins), bootstrap_sources=len(bootstrap), traversed_local_imports=len(visited),
            complete_release_bytes=result['original_preparation']['complete_source_release_bytes'],
            sdk_template_bytes=artifacts['sdk-entry-template']['bytes'], outer_entry_plan_bytes=artifacts['outer-entry-plan']['bytes']))

    def test_real_outer_gate_calls_successor_and_retains_full_map(self):
        original, request, base, result, full = produced()
        a = original.diagnostic.auth
        pv = result['pv_spec_candidate']
        diagnostic = NS(auth=a, r=a.r, validate_spec=Mock(), bind_authenticated=Mock())
        spec = dict(pv_spec={'path': '/source/pv.json', 'sha256': 'a'*64}, pv_review={'path': '/source/review.json', 'sha256': 'b'*64})
        spec_ref = {'path': '/source/diagnostic.json', 'sha256': 'c'*64}; review_ref = {'path': '/source/diagnostic-review.json', 'sha256': 'd'*64}
        metadata = NS(read=lambda ref: spec if ref == spec_ref else {'fixture': True})
        lease = NS(admission={'exact': 'binding'}, metadata=metadata)
        guard = Mock(); guard.lease = lease
        gate = NS(state='history_loaded', original=diagnostic, bootstrap=dict(binding=lease.admission, source_sha256=full['source_sha256']), guard=Mock())
        authenticated = dict(context={'pins_plan': full}, witness={'complete_source_map_sha256': digest(full['source_sha256'])})
        with patch.object(runtime, 'authenticate', return_value=authenticated) as actual:
            self.assertIs(runtime.authenticate_original(gate, spec_ref, review_ref, lease=lease, metadata=metadata, guard=guard), authenticated)
        actual.assert_called_once_with(a, spec['pv_spec'], spec['pv_review'], lease=lease, metadata=metadata, guard=guard)
        self.assertEqual(gate.state, 'complete_original_authentication')
        source = (Q / 'minicpm5_sdk_mapping_outer_source_entry_v7.py').read_text()
        self.assertIn('catalog_auth.authenticate_original(gate,', source)
        self.assertNotIn('authenticated=gate.authenticate_original(', source)

    def test_old_bodies_and_frozen_files_remain_unchanged(self):
        def fn(filename, name):
            tree = ast.parse((Q / filename).read_text())
            return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
        dump = lambda node: ast.dump(node, include_attributes=False)
        # Strip only explicit dependency aliases before the original request
        # check; then normalize the one shared-domain check.
        old_prepare = fn('prepare_minicpm5_pv_sdk_import_diagnostic_v1.py', 'prepare')
        new_prepare = fn('prepare_minicpm5_pv_sdk_import_catalog_v2.py', 'prepare')
        start = next(i for i,n in enumerate(new_prepare.body) if dump(n) == dump(old_prepare.body[1]))
        new_body = new_prepare.body[start:]
        old_body = old_prepare.body[1:]
        for i, node in enumerate(new_body):
            if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                    and isinstance(node.value.func.value, ast.Name) and node.value.func.value.id == 'completed'):
                new_body[i] = ast.parse('extension_shape(extension)').body[0]
        self.assertEqual([dump(n) for n in old_body], [dump(n) for n in new_body])
        old_auth = fn('minicpm5_layer1_fixed_input_pv_prerequisites_v2.py', 'authenticate')
        new_auth = fn('minicpm5_fixed_input_pv_catalog_auth_v1.py', 'authenticate')
        start = next(i for i,n in enumerate(new_auth.body) if dump(n) == dump(old_auth.body[1]))
        new_body = new_auth.body[start:]
        for node in ast.walk(ast.Module(body=new_body, type_ignores=[])):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'validate':
                node.args = node.args[1:]; node.keywords = []
        self.assertEqual([dump(n) for n in old_auth.body[1:]], [dump(n) for n in new_body])
        old_outer = fn('minicpm5_sdk_mapping_outer_source_entry_v6.py', 'source_entry')
        new_outer = fn('minicpm5_sdk_mapping_outer_source_entry_v7.py', 'source_entry')
        for node in ast.walk(new_outer):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == 'catalog_auth' and node.func.attr == 'authenticate_original'):
                node.func.value.id = 'gate'; node.args = node.args[1:]
        self.assertEqual(dump(old_outer), dump(new_outer))
        old_pins = saved('complete-source-map')
        for name in ('prepare_minicpm5_pv_sdk_import_diagnostic_v1.py',
                     'minicpm5_layer1_fixed_input_pv_prerequisites_v2.py',
                     'audit_minicpm5_layer1_operator_native_result_v4.py',
                     'audit_minicpm5_layer1_saved_operators_v1.py',
                     'compare_minicpm5_layer1_saved_operators_v1.py',
                     'minicpm5_layer1_operator_sdk_runtime_completion_v4.py'):
            path = Q / name
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), old_pins[str(path)])
        delivery = read(ref(Q / 'minicpm5-formal-literal-source-manifest-20261112-v1.json'))
        for row in delivery['source_files'] + delivery['preserved_original_files']:
            self.assertEqual(hashlib.sha256(Path(row['path']).read_bytes()).hexdigest(), row['sha256'])
        wiring_delivery = read(ref(Q / 'minicpm5-literal-formal-wiring-source-delivery-20261112-v1.json'))
        for row in wiring_delivery['source_files'] + [wiring_delivery['original_preparer']]:
            self.assertEqual(hashlib.sha256(Path(row['path']).read_bytes()).hexdigest(), row['sha256'])
        self.assertFalse(BLOCKED.intersection(sys.modules))


if __name__ == '__main__':
    unittest.main()
