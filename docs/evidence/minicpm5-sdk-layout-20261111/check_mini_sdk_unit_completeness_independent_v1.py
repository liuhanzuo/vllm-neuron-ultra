"""Independent bounded source/JSON review; no model, native or live-state work."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research import minicpm5_sdk_physical_evidence_v3 as physical
from research import minicpm5_sdk_post_exit_source_supplement_v1 as supplement


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def absent(name):
    return dict(group=physical.GROUP_PREFIX + name + '.service', absent=True)


def properties(name, loaded=False):
    return dict(Id=name + '.service', LoadState='loaded' if loaded else 'not-found',
        ActiveState='inactive', SubState='dead', MainPID='0', ControlPID='0',
        ExecMainPID='73' if loaded else '0', InvocationID='a'*32 if loaded else '',
        ExecMainCode='0', ExecMainStatus='0', Result='success', Job='')


class PhysicalReview(unittest.TestCase):
    def setUp(self):
        self.original = ['trn1-review-%04d' % n for n in range(1308)]
        self.extra = 'trn1-review-owner-only'
        self.tagged = 'trn1-review-tagged-owner'
        self.owners = [dict(pid=73, start_ticks='8123', ppid=9, argv=['python3', 'review.py'],
            cgroup='0::' + physical.GROUP_PREFIX + self.extra + '.service'),
            dict(pid=75, start_ticks='8125', ppid=9, argv=['python3', 'tagged.py'],
            cgroup='0::' + physical.GROUP_PREFIX + self.tagged + '.service',
            unit=self.tagged, invocation_id='b'*32)]
        self.tree = dict(complete=True, physical_checks_performed=True,
            all_current_and_prior_services_preserved=True,
            all_additional_owners_checked=copy.deepcopy(self.owners), lock_identity=dict(inode=19),
            nemotron=dict(complete=True, fresh_physical_drain_verified=True, services=1308,
                groups=[absent(name) for name in self.original]))
        self.done = dict(nemotron=dict(history=dict(rows=[dict(unit=n) for n in self.original])),
            owners=copy.deepcopy(self.owners), lock_identity=dict(inode=19))
        self.known = set(self.original) | {self.extra, self.tagged}

    def collect(self, tree_callback=None, props_callback=None):
        return physical.collect_pair(self.done, lambda: None,
            tree_callback or (lambda *args: copy.deepcopy(self.tree)),
            props_callback or properties, lambda path: absent(physical.group_unit(path)),
            known_units=self.known)

    def test_actual_saved_scope_independently_extracted_without_legacy_function(self):
        path = Path('/mnt/nvme/minicpm5-layer1-sdk-batch-layout-r2-20261111/predecessor-drain.json')
        raw = path.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),
            '871ac1d3767e561c4f806dafb677c78af74abef703e56f58f71447fb31a4e431')
        tree = json.loads(raw)['latest_physical']; before = canonical(tree)
        independent = set()
        def visit(value):
            if isinstance(value, list):
                for child in value: visit(child)
            elif isinstance(value, dict):
                if isinstance(value.get('unit'), str):
                    independent.add(value['unit'].removesuffix('.service'))
                for field in ('Id', 'group', 'cgroup'):
                    item = value.get(field)
                    if not isinstance(item, str): continue
                    if field == 'Id' and item.startswith('trn1-'):
                        independent.add(item.removesuffix('.service'))
                    else:
                        item = item.removeprefix('0::')
                        if item.startswith(physical.GROUP_PREFIX):
                            independent.add(item[len(physical.GROUP_PREFIX):].removesuffix('.service'))
                for child in value.values(): visit(child)
        visit(tree)
        scan = physical.inspect_tree(tree)
        self.assertEqual(len(independent), 1316)
        self.assertEqual(scan['units'], independent)
        self.assertEqual(scan['needs_supplement'], independent)
        self.assertEqual(len(scan['owner_only']), 5)
        self.assertEqual(len(tree['all_additional_owners_checked']), 1642)
        self.assertEqual(canonical(tree), before)

    def test_callbacks_really_execute_two_full_ordered_rounds(self):
        log = []
        def tree(done, guard):
            log.append(('physical', len(log)))
            self.assertEqual(done, self.done)
            return copy.deepcopy(self.tree)
        def props(name): log.append(('properties', name)); return properties(name)
        def group(path): log.append(('group', path)); return absent(physical.group_unit(path))
        before = canonical(self.tree)
        value = physical.collect_pair(self.done, lambda: None, tree, props, group, known_units=self.known)
        positions = [n for n, item in enumerate(log) if item[0] == 'physical']
        self.assertEqual(positions, [0, 1 + 2*len(self.known)])
        for pos in positions:
            actual = log[pos+1:pos+1+2*len(self.known)]
            expected = [(kind, name if kind == 'properties' else physical.GROUP_PREFIX+name+'.service')
                for name in sorted(self.known) for kind in ('properties', 'group')]
            self.assertEqual(actual, expected)
        self.assertEqual(physical.validate_pair(value, self.original, self.owners,
            self.done['lock_identity'], known_units=self.known), self.known)
        self.assertIsNot(value['first']['tree'], value['second']['tree'])
        self.assertEqual(canonical(self.tree), before)

    def test_second_physical_callback_difference_is_not_hidden_by_copy(self):
        calls = []
        def second_change(*args):
            calls.append(1); out = copy.deepcopy(self.tree)
            if len(calls) == 2: out['second_pass_sentinel'] = 'changed'
            return out
        with self.assertRaisesRegex(RuntimeError, 'stable and equal'):
            self.collect(tree_callback=second_change)
        self.assertEqual(len(calls), 2)

    def test_second_property_callback_difference_is_observed(self):
        calls = []
        def second_change(name):
            calls.append(name); out = properties(name)
            if len(calls) > len(self.known) and name == self.extra: out['Result'] = 'exit-code'
            return out
        with self.assertRaisesRegex(RuntimeError, 'stable and equal'):
            self.collect(props_callback=second_change)
        self.assertEqual(calls, sorted(self.known)*2)

    def test_missing_known_unit_rejected_before_any_properties(self):
        calls = []
        self.known.add('trn1-absent-authenticated-service')
        with self.assertRaisesRegex(RuntimeError, 'complete authenticated service set'):
            self.collect(props_callback=lambda name: calls.append(name))
        self.assertEqual(calls, [])

    def test_owner_boolean_nul_replacement_and_cross_group_rejected(self):
        for key, bad in [('pid', True), ('start_ticks', False), ('ppid', True),
                         ('argv', ['python3', 'bad\x00arg']), ('argv', 'python3'),
                         ('cgroup', '0::'+physical.GROUP_PREFIX+'../outside.service')]:
            tree = copy.deepcopy(self.tree); tree['all_additional_owners_checked'][0][key] = bad
            with self.subTest(key=key), self.assertRaises(RuntimeError): physical.inspect_tree(tree)
        tree = copy.deepcopy(self.tree)
        duplicate = copy.deepcopy(tree['all_additional_owners_checked'][0]); duplicate['ppid'] += 1
        tree['all_additional_owners_checked'].append(duplicate)
        with self.assertRaisesRegex(RuntimeError, 'repeated coordinator/rank'): physical.inspect_tree(tree)

    def test_sparse_legacy_owners_preserved_verbatim(self):
        sparse = dict(pid=77, start_ticks='9000')
        self.tree['all_additional_owners_checked'].append(sparse)
        self.done['owners'].append(copy.deepcopy(sparse))
        value = self.collect()
        for branch in ('first', 'second'):
            self.assertEqual(value[branch]['tree']['all_additional_owners_checked'], self.done['owners'])
        corrupted = copy.deepcopy(value)
        for branch in ('first', 'second'): corrupted[branch]['tree']['all_additional_owners_checked'].pop()
        with self.assertRaisesRegex(RuntimeError, 'additional owners retained'):
            physical.validate_pair(corrupted, self.original, self.done['owners'], self.done['lock_identity'])

    def test_missing_duplicate_extra_supplements_rejected(self):
        good = self.collect()
        for change in ('missing', 'duplicate', 'extra'):
            value = copy.deepcopy(good)
            for branch in ('first', 'second'):
                rows = value[branch]['supplemental_observations']
                if change == 'missing': rows.pop()
                elif change == 'duplicate': rows[-1] = copy.deepcopy(rows[0])
                else: rows.append(dict(unit='trn1-extra', properties=properties('trn1-extra'), group=absent('trn1-extra')))
            with self.subTest(change=change), self.assertRaises(RuntimeError): physical.validate_pair(value)

    def test_unanchored_loaded_service_unknown_history_still_needs_stability(self):
        def loaded(name): return properties(name, loaded=(name == self.extra))
        good = self.collect(props_callback=loaded)
        physical.validate_pair(good, self.original, self.owners, self.done['lock_identity'])
        for key, bad in [('MainPID', '73'), ('ControlPID', '74'), ('Job', '99')]:
            value = copy.deepcopy(good)
            for branch in ('first', 'second'):
                next(row for row in value[branch]['supplemental_observations']
                    if row['unit'] == self.extra)['properties'][key] = bad
            with self.subTest(key=key), self.assertRaises(RuntimeError): physical.validate_pair(value)
        value = copy.deepcopy(good)
        next(row for row in value['second']['supplemental_observations']
            if row['unit'] == self.extra)['properties']['InvocationID'] = 'c'*32
        with self.assertRaises(RuntimeError): physical.validate_pair(value)

    def test_tagged_loaded_service_requires_exact_historical_coordinator(self):
        def loaded(name):
            out = properties(name)
            if name == self.tagged: out.update(LoadState='loaded', InvocationID='b'*32, ExecMainPID='75')
            return out
        good = self.collect(props_callback=loaded)
        for key, bad in [('InvocationID', 'c'*32), ('ExecMainPID', '76')]:
            value = copy.deepcopy(good)
            for branch in ('first', 'second'):
                next(row for row in value[branch]['supplemental_observations']
                    if row['unit'] == self.tagged)['properties'][key] = bad
            with self.subTest(key=key), self.assertRaises(RuntimeError): physical.validate_pair(value)


class SourceReview(unittest.TestCase):
    def setUp(self):
        self.plan = dict(source_sha256={str(ROOT/'frozen-review-fixture.py'): hashlib.sha256(b'original').hexdigest()},
            scientific_original={'sentinel': [1, 2, 3]})
        self.pins = {path: hashlib.sha256(b'supplement').hexdigest() for path in supplement.SOURCE_PATHS}
        self.review = dict(schema=supplement.SCHEMA, reviewer='/root', passed=True, blockers=[],
            model_exit=copy.deepcopy(supplement.MODEL_EXIT), model_source_map_sha256=supplement.digest(self.plan['source_sha256']),
            source_sha256=self.pins, original_model_sources_preserved=True, frozen_model_closure_unchanged=True)
        self.ref = dict(path=str(ROOT/'research/minicpm5-sdk-post-exit-source-supplement-independent-fixture.json'), sha256='d'*64)

    def original_loaded_verifier(self, extra_unknown=False):
        source = ast.parse((ROOT/'minicpm5_layer1_sdk_batch_layout_cpu_runner_v4.py').read_text())
        function = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'verify_loaded_sources')
        module = ast.Module(body=[function], type_ignores=[])
        all_paths = set(self.plan['source_sha256']) | set(self.pins)
        if extra_unknown: all_paths.add(str(ROOT/'unreviewed-local-fixture.py'))
        def read(path, wanted): return b'original' if path in self.plan['source_sha256'] else b'supplement'
        namespace = dict(sys=SimpleNamespace(modules={str(n): SimpleNamespace(__file__=path) for n, path in enumerate(sorted(all_paths))}),
            ROOT=ROOT, Path=Path, hashlib=hashlib, p=SimpleNamespace(require=physical.require, clear_window=lambda: None),
            source_auth=SimpleNamespace(text=read), same=lambda a, b, message: physical.require(a == b, message))
        exec(compile(ast.fix_missing_locations(module), '<original-source-only-verifier>', 'exec'), namespace)
        return namespace['verify_loaded_sources']

    def execute(self, runner):
        def read(path, sha): return b'supplement'
        with patch.object(supplement.sources, 'load', return_value=self.review), \
             patch.object(supplement.sources, 'text', side_effect=read), \
             patch.object(supplement, 'identity', return_value={'stable': True}):
            return supplement.verify_loaded_sources(runner, self.plan, lambda: None, self.ref, supplement.MODEL_EXIT)

    def test_original_loaded_verifier_accepts_exact_union_and_preserves_plan(self):
        before = canonical(self.plan)
        output = self.execute(SimpleNamespace(verify_loaded_sources=self.original_loaded_verifier()))
        self.assertEqual(set(output['source_sha256']), set(self.plan['source_sha256']) | set(self.pins))
        self.assertEqual(canonical(self.plan), before)
        self.assertTrue(output['frozen_model_closure_unchanged'])

    def test_original_loaded_verifier_rejects_unreviewed_fifth_source(self):
        with self.assertRaisesRegex(RuntimeError, 'Every loaded local helper'):
            self.execute(SimpleNamespace(verify_loaded_sources=self.original_loaded_verifier(extra_unknown=True)))

    def test_mutated_original_scientific_plan_rejected(self):
        def mutate(view, guard):
            view['scientific_original']['sentinel'].append(4)
            return dict(complete=True)
        with self.assertRaisesRegex(RuntimeError, 'Frozen scientific plan unchanged'):
            self.execute(SimpleNamespace(verify_loaded_sources=mutate))

    def test_wrong_model_original_map_and_overlap_rejected(self):
        for model in (dict(supplement.MODEL_EXIT, sha256='e'*64), dict(supplement.MODEL_EXIT, path='/wrong/model.json')):
            with self.assertRaises(RuntimeError): supplement.validate_review(self.review, model, self.plan)
        changed = copy.deepcopy(self.plan); changed['source_sha256'][str(ROOT/'injected.py')] = 'f'*64
        with self.assertRaises(RuntimeError): supplement.validate_review(self.review, supplement.MODEL_EXIT, changed)
        changed = copy.deepcopy(self.plan); changed['source_sha256'].update(self.pins)
        review = dict(self.review, model_source_map_sha256=supplement.digest(changed['source_sha256']))
        with self.assertRaises(RuntimeError): supplement.validate_review(review, supplement.MODEL_EXIT, changed)

    def test_no_native_imports(self):
        self.assertFalse({'torch', 'numpy', 'neuronxcc', 'torch_neuronx'} & set(sys.modules))


if __name__ == '__main__': unittest.main(verbosity=2)
