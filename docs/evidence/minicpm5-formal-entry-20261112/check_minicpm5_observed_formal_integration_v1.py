"""Saved JSON plus source/mock checks; no real authority/native/process action."""
import sys


class NoNative:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in {'torch', 'numpy', 'ctypes', '_ctypes', 'transformers',
                'vllm', 'neuron', 'neuronxcc', 'neuronx_distributed', 'neuronx_distributed_inference'}:
            raise AssertionError('No native imports in source/saved-JSON verification: ' + name)


sys.meta_path.insert(0, NoNative())
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

Q = Path(__file__).resolve().parent
ROOT = Q.parent
sys.path[:0] = [str(ROOT), str(Q)]
from research import prepare_minicpm5_observed_native_policies_v1 as prep
from research import minicpm5_sdk310_composite_origins_v2 as origins
from research import minicpm5_sdk_mapping_nocache_entry_v6 as entry
from research import minicpm5_bootstrap_types_source_v1 as types_source


class JSONOnly:
    def __init__(self):
        self.overrides = {}
        self.inputs = {}
    def json(self, reference):
        path = Path(reference['path'])
        if str(path) in self.overrides:
            return copy.deepcopy(self.overrides[str(path)])
        assert path.suffix == '.json' and path.resolve() == path and path.stat().st_size <= 16*2**20
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == reference['sha256']
        assert reference.get('bytes', len(data)) == len(data)
        self.inputs[str(path)] = reference
        return json.loads(data)
    def read(self, *args, **kwargs):
        raise AssertionError('No binary/archive/index payload reads in this suite')


def fixture():
    reader = JSONOnly()
    ended = reader.json(prep.ACTUAL_EXIT)
    plan = reader.json(ended['plan'])
    retained = copy.deepcopy(plan['source_sha256'])
    new = {str(Q/name): hashlib.sha256((Q/name).read_bytes()).hexdigest() for name in prep.NEW_SOURCES}
    manifest = dict(schema='minicpm5_observed_formal_entry_source_manifest_v1',
                    retained_source_sha256=retained, new_runtime_source_sha256=new)
    reference = dict(path='/synthetic/manifest.json', sha256='f'*64)
    reader.overrides[reference['path']] = manifest
    # Synthetic prospective map is intentionally not presented as reconstructed
    # original runtime closure. Production still runs full original equality.
    pins = {**retained, **new}
    table = prep.discovery.candidates(reader.json)
    # Original PV extension is local text only. Signed nonlocal stdlib rows
    # remain in each authenticator's explicit original origin policy.
    return reader, pins, reference, ended, table


class Checks(unittest.TestCase):
    def test_actual_saved_r2_prepares_distinct_unaccepted_policies(self):
        reader, pins, reference, ended, table = fixture()
        result = prep.build(reader.json, pins, reference)
        for role, policy in result['policies'].items():
            self.assertIsNone(policy['reviewer'])
            self.assertIs(policy['accepted'], False)
            self.assertEqual(policy['source_map_sha256'], prep.history.digest(pins))
            observed = reader.json(policy['observed_bootstrap_result'])
            self.assertEqual(set(policy['native_files']),
                set(observed['map_summaries']['post_ctypes']['known_selected_closure']['paths']))
        self.assertEqual(len(result['policies']['sdk310']['native_files']), 13)
        self.assertEqual(len(result['policies']['outer312']['native_files']), 8)
        self.assertFalse(result['complete_original_runtime_source_map_authenticated_here'])
        self.assertFalse(result['current_native_bytes_read_here'])
        self.assertFalse(result['accepted_native_authority_created'])
        self.assertTrue(result['nonlocal_sources_authenticated_by_original_origin_policy'])
        self.assertNotIn(types_source.PIN['path'], pins)

    def test_failed_actual_exit_and_unreplayed_maps_rejected(self):
        for field, value in [('actual_tool_exit_observed', False), ('physically_drained', False),
                             ('full_six_raw_maps_replayed', False), ('source_execution_gaps_recomputed', False)]:
            reader, pins, reference, ended, _ = fixture()
            ended[field] = value
            reader.overrides[prep.ACTUAL_EXIT['path']] = ended
            with self.assertRaises(ValueError): prep.build(reader.json, pins, reference)

    def test_observed_success_cannot_promote_qualification(self):
        reader, pins, reference, ended, _ = fixture()
        result = reader.json(ended['result']); result['native_ABI_qualified'] = True
        reader.overrides[ended['result']['path']] = result
        with self.assertRaisesRegex(ValueError, 'no native ABI'): prep.build(reader.json, pins, reference)

    def test_missing_source_and_changed_original_pin_rejected(self):
        for which in ('new', 'original', 'types'):
            reader, pins, reference, ended, _ = fixture()
            key = (str(Q/prep.NEW_SOURCES[0]) if which == 'new' else
                   next(iter(reader.json(ended['plan'])['source_sha256'])) if which == 'original' else
                   types_source.PIN['path'])
            pins[key] = '0'*64
            with self.assertRaises(ValueError): prep.build(reader.json, pins, reference)

    def test_root_role_audit_disagreement_and_source_gap_rejected(self):
        for change in ('gap', 'audit'):
            reader, pins, reference, ended, _ = fixture()
            result = reader.json(ended['result']); row = result['children'][1]
            value = reader.json(row['result'])
            if change == 'gap': value['source_execution_provenance_gaps'] = ['unproved']
            else: value['selected_stdlib_providers']['_ctypes']['origin'] = 'foreign'
            reader.overrides[row['result']['path']] = value
            with self.assertRaises(ValueError): prep.build(reader.json, pins, reference)

    def test_actual_supplemental_members_keep_old_full_verifier(self):
        reader, pins, reference, _, _ = fixture()
        policy = prep.build(reader.json, pins, reference)['policies']['sdk310']
        expected = {**policy['native_files'], **policy['pure_sources']}
        old_expected = {p:r for p,r in expected.items() if p not in origins.EXTRA_PATHS}
        calls = []
        def original(r, d, p):
            self.assertIs(r, reader); self.assertIs(p, pins)
            self.assertEqual(d['schema'], 'minicpm5_sdk310_native_source_review_v2')
            self.assertEqual(set(d['package_origins']), {'schema', 'old_libc6', 'new_eight'})
            self.assertEqual({**d['native_files'], **d['pure_sources']}, old_expected)
            calls.append(True)
            return copy.deepcopy(old_expected)
        with patch.object(origins.previous, 'replay_package_origins', original):
            self.assertEqual(origins.replay_package_origins(reader, policy, pins), expected)
        self.assertEqual(calls, [True])

    def test_supplemental_wrong_member_missing_member_and_origin_ref_rejected(self):
        for change in ('hash', 'missing', 'reference'):
            reader, pins, reference, _, _ = fixture()
            policy = prep.build(reader.json, pins, reference)['policies']['sdk310']
            expected = {p:r for p,r in {**policy['native_files'], **policy['pure_sources']}.items()
                        if p not in origins.EXTRA_PATHS}
            key = sorted(origins.EXTRA_PATHS)[0]
            if change == 'hash': policy['native_files'][key]['sha256'] = '0'*64
            elif change == 'missing': del policy['native_files'][key]
            else: policy['package_origins']['sdk310_supplemental_stdlib']['actual_exit']['sha256'] = '0'*64
            with patch.object(origins.previous, 'replay_package_origins', return_value=copy.deepcopy(expected)):
                with self.assertRaises((ValueError, AssertionError)):
                    origins.replay_package_origins(reader, policy, pins)

    def test_original_full_origin_failure_not_bypassed_by_supplement(self):
        reader, pins, reference, _, _ = fixture()
        policy = prep.build(reader.json, pins, reference)['policies']['sdk310']
        with patch.object(origins.previous, 'replay_package_origins', side_effect=ValueError('full original rejected')):
            with self.assertRaisesRegex(ValueError, 'full original rejected'):
                origins.replay_package_origins(reader, policy, pins)

    def test_literal_tracker_and_scientific_worker_bodies_match_original(self):
        def nodes(path):
            return {n.name:ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text()).body
                    if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        current = nodes(Path(entry.__file__))
        original = nodes(Path(types_source.__file__))
        for name in ('provider_snapshot', 'TypesSourceObservation'):
            self.assertEqual(current[name], original[name])
        self.assertEqual(nodes(Q/'minicpm5_sdk_mapping_worker_source_entry_v5.py')['source_entry'],
                         nodes(Q/'minicpm5_sdk_mapping_worker_source_entry_v4.py')['source_entry'])
        self.assertEqual(nodes(Q/'minicpm5_sdk_mapping_native_authenticator_v3.py'),
                         nodes(Q/'minicpm5_sdk_mapping_native_authenticator_v2.py'))

    def test_complete_literal_order_and_trusted_q_import_route(self):
        source = Path(entry.__file__).read_text()
        self.assertLess(len(source.encode()), 65536)
        main = source[source.index('\ndef main():'):]
        order = ['entry.startup(source)', 'TypesSourceObservation(sys)', 'entry.install(external)',
                 'install_before_json(entry, external, SOURCE_TOTAL_CAP)', 'import json',
                 'after_json()', 'entry.load_plan(', "sys.path[:0]=[ROOT,ROOT+'/research']", '__import__(']
        positions = [main.index(part) for part in order]
        self.assertEqual(positions, sorted(positions))
        # Direct source-route selection, no module execution or current native
        # load: the same explicit Q addition resolves this otherwise missing import.
        import importlib.machinery
        name = 'minicpm5_sdk310_extra_members_contract_v1'
        self.assertIsNone(importlib.machinery.PathFinder.find_spec(name, [str(ROOT)]))
        spec = importlib.machinery.PathFinder.find_spec(name, [str(ROOT), str(Q)])
        self.assertEqual(spec.origin, str(Q/(name+'.py')))


if __name__ == '__main__':
    unittest.main()
