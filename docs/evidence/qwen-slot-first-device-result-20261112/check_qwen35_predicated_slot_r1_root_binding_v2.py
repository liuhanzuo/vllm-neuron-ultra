"""Source-only in-memory candidate checks; never invoke or publish the recipe."""
import copy
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys
import unittest

Q = Path(__file__).resolve().parent
sys.path[:0] = [str(Q), str(Q.parent)]
BLOCKED = {'ctypes', '_ctypes', 'torch', 'torch_xla', 'torch_neuronx', 'neuronxcc',
           'numpy', 'nki', 'libneuronxla', 'transformers', 'vllm'}
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in BLOCKED:
            raise AssertionError('No native or compiler import: ' + fullname)
sys.meta_path.insert(0, NoNative())
import prepare_qwen35_predicated_slot_r1_root_binding_v2 as recipe


def real(reference):
    raw = Path(reference['path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == reference['sha256']
    return json.loads(raw)


def fixture():
    seed = real(recipe.FIXED['seed_spec'])
    old = real(seed['current_serial_catalog'])
    catalog = copy.deepcopy(old)
    future_catalog = dict(path=str(Q / 'source-only-future-slot-catalog.json'), sha256='a' * 64)
    future_handoff = dict(path=str(Q / 'source-only-future-slot-handoff.json'), sha256='b' * 64)
    future_terminal = dict(path=str(Q / 'source-only-future-slot-terminal.json'), sha256='c' * 64)
    terminal = real(dict(path=str(Q / 'qwen35-package-preflight-r4-root-actual-exit-20261112-v1.json'),
                         sha256='9d8e3c7ff9d28d16402ee5a739c555a643585f8a2887df441389452634b9da1a'))
    actual_catalog = real(dict(path=str(Q / 'minicpm5-operator-post-r2-serial-workloads-root-catalog-20261112-v101.json'),
                              sha256='3d6135fdf71318be12feb57dcfe402041bf27c7f3b921c68e095e291f6e57518'))
    catalog['current_handoff'] = future_handoff
    catalog['runs'].append(dict(terminal=future_terminal, tool=terminal['actual_tool'], owners=terminal['owners'],
                               units=terminal['units'], source_sha256=actual_catalog['runs'][-1]['source_sha256']))
    handoff = real(old['current_handoff'])
    mocked = {future_catalog['path']: catalog, future_handoff['path']: handoff, future_terminal['path']: terminal}
    observed = {}
    def read(reference):
        path = reference['path']
        if path in mocked:
            observed[path] = dict(sha256=reference['sha256'], identity=dict(realpath=path, device=1, inode=2,
                size=len(json.dumps(mocked[path]).encode()), mtime_ns=1, ctime_ns=1))
            return copy.deepcopy(mocked[path])
        value = real(reference)
        observed[path] = dict(sha256=reference['sha256'], identity=recipe.identity(path))
        return value
    def build():
        return recipe.candidate(future_catalog, future_handoff, future_terminal, 'source-only-r1-v3',
            str(Q / 'continuation-handoff-source-only-slot-running.json'), read=read, observed=observed,
            preparer_ref=recipe.ref(str(Q / 'prepare_qwen35_predicated_slot_r1_root_binding_v2.py')))
    return build, mocked, (future_catalog, future_handoff, future_terminal), observed


class SourceChecks(unittest.TestCase):
    def test_real_seed_and_contract_with_explicit_synthetic_future_in_memory(self):
        build, _, _, _ = fixture(); value = build()
        seed = real(recipe.FIXED['seed_spec']); spec = value['spec']
        self.assertEqual(value['catalog_rows'], 150)
        self.assertTrue(all(spec['source_sha256'][path] == sha for path, sha in seed['source_sha256'].items()))
        self.assertEqual(spec['limits'], recipe.slot.LIMITS)
        self.assertEqual(spec['limits']['full_workload_initial_free_bytes'], 40 * 2**30)
        self.assertEqual(spec['workload']['counts']['total']['graphs'], 32)
        self.assertEqual(spec['workload']['counts']['total']['total_runtime_calls'], 46368)
        self.assertEqual(len(spec['workload']['layouts']), 8)
        self.assertFalse(value['review_draft']['passed']); self.assertFalse(value['review_draft']['launch_authorized'])
        self.assertTrue(value['full_history_precheck_pending'])
        self.assertLess(len(recipe.objects.encode(value['overlay'])), 8 * 2**20)
        self.assertTrue(all(row['identity']['size'] <= 8 * 2**20 for row in spec['input_extension'].values()))
        for key in ('seed_spec', 'seed_review'):
            self.assertNotIn(recipe.FIXED[key]['path'], spec['input_extension'])
            self.assertIn(recipe.FIXED[key]['path'], value['preparation_only_inputs'])
        recipe.slot.verify_delivery(spec)
        for path in value['paths'].values(): self.assertFalse(Path(path).exists())
        self.assertFalse(BLOCKED.intersection(sys.modules))

    def test_catalog_replacement_omission_reorder_extra_row_and_live_handoff_rejected(self):
        for fault in ('missing', 'reordered', 'changed', 'extra', 'handoff_ref', 'live', 'heavy', 'running', 'active_serial'):
            build, values, refs, _ = fixture(); catalog, handoff = values[refs[0]['path']], values[refs[1]['path']]
            if fault == 'missing': catalog['runs'].pop(10)
            elif fault == 'reordered': catalog['runs'][1], catalog['runs'][2] = catalog['runs'][2], catalog['runs'][1]
            elif fault == 'changed': catalog['runs'][10]['tool']['exit_code'] = 253
            elif fault == 'extra': catalog['runs'].append(copy.deepcopy(catalog['runs'][-1]))
            elif fault == 'handoff_ref': catalog['current_handoff'] = dict(refs[1], sha256='f' * 64)
            elif fault == 'live': handoff['live_tool_sessions'] = [recipe.R4_SESSION]
            elif fault == 'heavy': handoff['active_heavy_units'] = ['running']
            elif fault == 'running': handoff['running'] = True
            else: handoff['active_serial'] = {'session': recipe.R4_SESSION}
            with self.subTest(fault=fault), self.assertRaises(ValueError): build()

    def test_actual_terminal_identity_drain_and_failed_status_preserved(self):
        for field, value in (('actual_tool_exit_observed', False), ('actual_outer_exit_observed', False),
                ('physically_drained', False), ('owners_gone', False), ('original_lock_reacquired_and_released', False),
                ('full_model_qualified', True), ('new_speedup_claim', True), ('owners', []), ('units', []),
                ('spec', dict(recipe.FIXED['seed_spec'], sha256='f' * 64))):
            build, values, refs, _ = fixture(); values[refs[2]['path']][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): build()
        build, values, refs, _ = fixture()
        values[refs[2]['path']]['actual_tool'] = dict(values[refs[2]['path']]['actual_tool'], session_id=1)
        with self.assertRaises(ValueError): build()
        build, values, refs, _ = fixture()
        values[refs[2]['path']]['actual_tool']['exit_code'] = 1
        values[refs[2]['path']]['native_preflight_passed'] = False
        value = build()
        self.assertEqual(values[refs[2]['path']]['actual_tool']['exit_code'], 1)
        self.assertFalse(values[refs[2]['path']]['native_preflight_passed'])
        self.assertFalse(value['spec']['automatic_successor'])

    def test_changed_source_pin_and_reduced_r4_map_rejected(self):
        for fault in ('changed', 'missing'):
            build, values, refs, _ = fixture(); pins = values[refs[0]['path']]['runs'][-1]['source_sha256']
            path = next(iter(pins))
            if fault == 'missing': pins.pop(path)
            else: pins[path] = 'f' * 64
            with self.subTest(fault=fault), self.assertRaises(ValueError): build()

    def test_real_contract_refuses_unapproved_draft(self):
        build, _, _, _ = fixture(); value = build()
        with self.assertRaisesRegex(ValueError, 'Original exact root review envelope'):
            recipe.slot.validate(value['spec'], value['review_draft'], value['spec_ref'])

    def test_original_input_identity_conflict_cannot_be_refreshed(self):
        build, _, _, observed = fixture()
        path, row = next(iter(real(recipe.FIXED['seed_spec'])['input_extension'].items()))
        observed[path] = copy.deepcopy(row); observed[path]['identity']['mtime_ns'] += 1
        with self.assertRaisesRegex(ValueError, 'Old input identity'): build()

    def test_frozen_delivery_source_hashes_unchanged(self):
        delivery = real(recipe.FIXED['delivery'])
        pins = dict(delivery['static_local_source_sha256'], **delivery['sdk_source_sha256'])
        for path, expected in pins.items():
            self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(), expected, path)
        self.assertFalse(BLOCKED.intersection(sys.modules))

    def test_actual_v101_candidate_and_v1_rejection_without_publication(self):
        refs = [dict(path=str(Q / name), sha256=sha) for name, sha in (
            ('minicpm5-operator-post-r2-serial-workloads-root-catalog-20261112-v101.json', '3d6135fdf71318be12feb57dcfe402041bf27c7f3b921c68e095e291f6e57518'),
            ('continuation-handoff-20261112-v567-qwen-package-r4-ended.json', '3e4f8b0d89014c56bb636fb984e7f9beea1646086824461a263ff89acf367f99'),
            ('qwen35-package-preflight-r4-root-actual-exit-20261112-v1.json', '9d8e3c7ff9d28d16402ee5a739c555a643585f8a2887df441389452634b9da1a'))]
        reader = recipe.JsonMetadata(lambda: None)
        def read(reference): return reader.read(reference)['value']
        kwargs = dict(read=read, observed=reader.observed,
                      preparer_ref=recipe.ref(str(Q / 'prepare_qwen35_predicated_slot_r1_root_binding_v2.py')))
        args = (*refs, 'source-only-r1-v3', str(Q / 'continuation-handoff-source-only-slot-running.json'))
        import prepare_qwen35_predicated_slot_r1_root_binding_v1 as old
        with self.assertRaisesRegex(ValueError, 'All actual package r4 sources'):
            old.candidate(*args, **kwargs)
        value = recipe.candidate(*args, **kwargs)
        seed = real(recipe.FIXED['seed_spec']); catalog = real(refs[0]); terminal = real(refs[2])
        self.assertEqual(len(catalog['runs'][-1]['source_sha256']), 729)
        self.assertEqual(len(seed['source_sha256']), 48194)
        self.assertTrue(all(value['spec']['source_sha256'][path] == sha for path, sha in seed['source_sha256'].items()))
        for row in catalog['runs']:
            self.assertTrue(all(value['spec']['source_sha256'][path] == sha for path, sha in row['source_sha256'].items()))
        auditor = terminal['auditor_source']
        self.assertEqual(value['spec']['source_sha256'][auditor['path']], auditor['sha256'])
        self.assertTrue(value['full_history_precheck_pending']); self.assertFalse(value['review_draft']['passed'])
        self.assertEqual(value['spec']['limits']['full_workload_initial_free_bytes'], 40 * 2**30)
        for path in value['paths'].values(): self.assertFalse(Path(path).exists())
        self.assertFalse(BLOCKED.intersection(sys.modules))

    def test_actual_auditor_pin_and_new_source_scope_bound_to_terminal(self):
        for fault in ('auditor_hash', 'extra_source', 'shared_source'):
            build, values, refs, _ = fixture(); terminal = values[refs[2]['path']]
            pins = values[refs[0]['path']]['runs'][-1]['source_sha256']
            if fault == 'auditor_hash': terminal['auditor_source']['sha256'] = 'f' * 64
            elif fault == 'extra_source': pins[str(Q / 'unreviewed-new-auditor.py')] = 'f' * 64
            else:
                path = next(path for path in pins if path != terminal['auditor_source']['path'])
                pins[path] = 'f' * 64
            with self.subTest(fault=fault), self.assertRaisesRegex(ValueError, 'executed-source increment'): build()


if __name__ == '__main__': unittest.main(verbosity=2)
