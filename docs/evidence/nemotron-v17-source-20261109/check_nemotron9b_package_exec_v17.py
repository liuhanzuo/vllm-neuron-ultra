"""Source integration and actual metadata tests; no model/compiler execution.

Generated file bodies are synthetic. Installed resources use the original
RECORD-pinned policy. Storage availability is simulated only in these tests.
"""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
import nemotron9b_package_protocol_v16 as old
import nemotron9b_package_protocol_v17 as p
import nemotron9b_package_evidence_v17 as evidence
import nemotron9b_attention_completion_v1 as completion


class SourceClosure(unittest.TestCase):
    def test_all_frozen_v16_sources_preserved(self):
        spec = old.metadata.load(dict(path=str(R/'research/nemotron9b-package-runtime-v16-r8-20261109-spec.json'),
             sha256='9cfa93c91e0760a2a994375578476ca65899638923d8e562ab44bb79a2a2f27e'))
        paths = list(R.glob('nemotron9b_package_*v16.py'))
        self.assertEqual(len(paths), 16)
        for path in paths:
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), spec['source_sha256'][str(path)])

    def test_fourteen_execution_modules_are_only_version_renames(self):
        paths = [path for path in R.glob('nemotron9b_package_*v16.py')
                 if path.stem not in ('nemotron9b_package_protocol_v16', 'nemotron9b_package_metadata_v16')]
        self.assertEqual(len(paths), 14)
        for path in paths:
            self.assertEqual(path.read_text().replace('_v16', '_v17'),
                             path.with_name(path.name.replace('_v16', '_v17')).read_text())

    def test_all_original_science_and_resource_bounds_retained(self):
        for key in ('STAGES', 'SCIENTIFIC_STAGES', 'ROOT_FLOOR', 'NVME_FLOOR', 'ROOT_CAP',
                    'NVME_CAP', 'CAPS', 'BUCKETS', 'SPLIT_FLAGS', 'ALIAS_FLAGS', 'PACKAGE', 'COMMIT'):
            self.assertEqual(getattr(p, key), getattr(old, key))
        self.assertEqual(len(p.STAGES), 52)
        for stage in (*p.STAGES, 'controller'):
            self.assertEqual(p.resources(stage), old.resources(stage))

    def test_protocol_only_adds_explicit_completion_gate(self):
        def functions(source):
            return {n.name: n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
        left = functions(Path(old.__file__).read_text().replace('_v16', '_v17'))
        right = functions(Path(p.__file__).read_text())
        self.assertEqual(set(left), set(right))
        for name in left:
            if name == 'required_sources':
                continue
            if name == 'bundle':
                additions = [n for n in right[name].body if isinstance(n, ast.Expr)
                             and isinstance(n.value, ast.Call)
                             and ast.unparse(n.value.func) == 'attention_completion.validate_spec']
                self.assertEqual(len(additions), 1)
                self.assertEqual(ast.unparse(additions[0]), 'attention_completion.validate_spec(s, catalog)')
                right[name].body.remove(additions[0])
                prior_checks = [n for n in left[name].body if isinstance(n, ast.Expr)
                                and isinstance(n.value, ast.Call)
                                and ast.unparse(n.value.func) == 'completed_history.validate']
                self.assertEqual(len(prior_checks), 1)
                left[name].body.remove(prior_checks[0])
            self.assertEqual(ast.dump(left[name]), ast.dump(right[name]), name)

    def test_complete_source_and_evidence_paths_share_correct_routing(self):
        self.assertIs(p.compiler_links, p.output_inventory.compiler_links)
        self.assertIs(evidence.p, p)
        self.assertEqual(p.compiler_links.__name__, 'nemotron9b_compiler_asset_links_v4')
        self.assertTrue(set(old.required_sources()) < set(p.required_sources()))
        current = list(R.glob('nemotron9b_package_*v17.py'))
        self.assertEqual(len(current), 16)
        self.assertTrue({str(path) for path in current} <= set(p.required_sources()))
        self.assertTrue(all(Path(path).is_file() for path in p.required_sources()))
        self.assertIn(str(R/'nemotron9b_attention_completion_v1.py'), p.required_sources())

    def test_old_metadata_schemas_and_bounds_preserved(self):
        for schema, shape in old.metadata.ENVELOPE_SCHEMAS.items():
            self.assertEqual(p.metadata.ENVELOPE_SCHEMAS[schema], shape)
        self.assertEqual(p.metadata.LIMITS, old.metadata.LIMITS)
        self.assertEqual(set(p.metadata.ENVELOPE_SCHEMAS) - set(old.metadata.ENVELOPE_SCHEMAS),
                         {'nemotron9b_package_runtime_v17', 'nemotron9b_package_runtime_source_review_v17'})

    def test_no_native_library_imports(self):
        self.assertFalse(any(n.split('.')[0] in {'torch', 'numpy', 'vllm', 'neuronxcc', 'torch_neuronx'}
                             for n in sys.modules))


class Completion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = old.completed_history.load(old.completed_history.CATALOG)

    def test_actual_1096_services_290_owners_and_33_directories(self):
        history = completion.validate_base(self.base)
        self.assertEqual((len(history['rows']), len(history['processes']),
                          len(history['known_admitted_directories'])), (1096, 290, 33))
        self.assertFalse(history['current_physical_release_verified'])
        self.assertFalse(history['r8_outcome']['producer_qualified'])
        self.assertIn('additional_missing_metadata_preflight_owner', history)
        self.assertIn('attention_metadata_preflight_owner_missing', history)

    def test_all_required_metadata_are_bounded_and_in_spec(self):
        _, pins = completion.completed()
        spec = dict(required_attention_completion=completion.EXIT, source_sha256=pins)
        self.assertEqual(len(completion.validate_spec(spec, self.base)['processes']), 290)
        for path, digest in pins.items():
            self.assertEqual(p.metadata.sha(path, digest), digest)

    def test_omitted_completion_gate_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'Latest actual'):
            completion.validate_spec(dict(source_sha256={}), self.base)

    def test_omitted_new_metadata_pin_rejected(self):
        _, pins = completion.completed()
        del pins[completion.EXIT['path']]
        with self.assertRaisesRegex(RuntimeError, 'Every attention'):
            completion.validate_spec(dict(required_attention_completion=completion.EXIT,
                                          source_sha256=pins), self.base)

    def test_changed_base_processes_rejected(self):
        value = copy.deepcopy(self.base)
        value['processes'][0]['pid'] += 1
        with self.assertRaisesRegex(RuntimeError, 'immutable completed base'):
            completion.validate_base(value)

    def test_completed_snapshot_does_not_rediscover_old_directory_set(self):
        with patch.object(completion.base_history, 'validate',
                          side_effect=RuntimeError('Old discovery is not a current snapshot')):
            self.assertEqual(len(completion.validate_base(self.base)['processes']), 290)

    def test_changed_receipt_bytes_rejected(self):
        original = completion.stable_bytes
        def altered(path, cap):
            raw = original(path, cap)
            return raw + b' ' if str(path) == completion.EXIT['path'] else raw
        with patch.object(completion, 'stable_bytes', side_effect=altered):
            with self.assertRaisesRegex(RuntimeError, 'Exact completed'):
                completion.completed()

    def test_newer_r8_controller_cannot_be_exempted_as_current(self):
        history = completion.validate_base(self.base)
        name = next(row['unit'] for row in history['rows'][1043:]
                    if row['unit'].endswith('-controller'))
        with self.assertRaisesRegex(RuntimeError, 'No historical service exempted'):
            p.predecessor_drain(self.base, [name])


class FullPaths(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='nemotron-v17-source-', dir='/mnt/nvme')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.work = self.root/'work'
        self.compiler = self.work/'compile-work'
        self.spec = dict(work_root=str(self.work), control_root=str(self.root/'control'),
                         evidence_root=str(self.root/'evidence'), compiler_asset_policy=p.compiler_links.ASSET_REF)
        for key in ('work_root', 'control_root', 'evidence_root'):
            Path(self.spec[key]).mkdir()
        self.policy = p.compiler_links.load_policy()
        findings = json.loads((R/'research/nemotron9b-r8-sglnk-routing-findings-20261109-v1.json').read_text())
        self.links = []
        old_root = Path('/mnt/nvme/nemotron9b-package-r8-20261109/compile-work')
        for entry in findings['links']:
            row = entry['record']
            path = self.compiler/Path(row['path']).relative_to(old_root)
            path.parent.mkdir(parents=True, exist_ok=True)
            if row.get('compiler_generated'):
                target = self.compiler/Path(row['symlink']).relative_to(old_root)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b'SYNTHETIC ONLY '+str(target.relative_to(self.compiler)).encode())
            else:
                target = Path(row['symlink'])
            path.symlink_to(target)
            self.links.append(path)
        self.assertEqual(len(self.links), 44)

    def test_original_failure_is_reproduced_and_new_complete_paths_accept(self):
        with self.assertRaisesRegex(RuntimeError, 'Reviewed generated numeric basename'):
            old.usage(self.work, compiler_root=self.compiler, compiler_assets=self.policy)
        actual = p.usage(self.work, compiler_root=self.compiler, compiler_assets=self.policy)
        expected = sum(max(f.lstat().st_size, f.lstat().st_blocks*512)
                       for f in self.work.rglob('*') if not f.is_dir())
        self.assertEqual(actual, expected)
        # Synthetic capacity is explicit; this does not claim the machine has space.
        with patch.object(p.os, 'statvfs', return_value=types.SimpleNamespace(f_bavail=128*2**30, f_frsize=1)):
            self.assertEqual(p.storage(self.spec, reserve=True)['nvme_used'], expected)
        rows = evidence.tree(self.work, lambda: None, spec=self.spec)
        kinds = {}
        for row in rows.values():
            if row.get('compiler_asset'):
                kind = row['compiler_asset_kind']
                kinds[kind] = kinds.get(kind, 0)+1
                if row.get('compiler_generated'):
                    self.assertTrue(row['target']['content_hashed'])
        self.assertEqual(kinds, {'generated_debug': 17, 'installed_sglnk': 17, 'generated_numeric': 10})

    def test_unpinned_same_basename_is_rejected_by_both_paths(self):
        path = next(f for f in self.links if str(f.readlink()) in self.policy['assets'])
        target = path.parent/'untrusted.bin'
        target.write_bytes(b'SYNTHETIC untrusted')
        path.unlink()
        path.symlink_to(target)
        with self.assertRaisesRegex(RuntimeError, 'RECORD-pinned'):
            p.usage(self.work, compiler_root=self.compiler, compiler_assets=self.policy)
        with self.assertRaisesRegex(RuntimeError, 'RECORD-pinned'):
            evidence.tree(self.work, lambda: None, spec=self.spec)

    def test_storage_floor_is_not_relaxed(self):
        def insufficient(path):
            floor = p.ROOT_FLOOR+p.ROOT_CAP-1 if Path(path) == p.ROOT else 128*2**30
            return types.SimpleNamespace(f_bavail=floor, f_frsize=1)
        with patch.object(p.os, 'statvfs', side_effect=insufficient):
            with self.assertRaisesRegex(RuntimeError, 'Unreduced filesystem'):
                p.storage(self.spec, reserve=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
