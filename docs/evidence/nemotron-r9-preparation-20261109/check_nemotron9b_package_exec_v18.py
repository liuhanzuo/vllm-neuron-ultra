"""Post-maintenance source integration; synthetic filesystem, no native launch."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

R = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(R), str(R/'research')]
import nemotron9b_package_protocol_v17 as old
import nemotron9b_package_protocol_v18 as p
import nemotron9b_package_evidence_v18 as evidence
import nemotron9b_after_journal_compaction_v1 as completed
import check_nemotron9b_package_exec_v17 as previous


class Source(unittest.TestCase):
    def test_frozen_v17_sources_unchanged(self):
        receipt = json.loads((R/'research/nemotron9b-v17-source-integration-20261109-v1.json').read_text())
        for path in R.glob('nemotron9b_package_*v17.py'):
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), receipt['source_sha256'][str(path)])

    def test_fourteen_execution_modules_only_change_version(self):
        count = 0
        for path in R.glob('nemotron9b_package_*v17.py'):
            if path.stem in ('nemotron9b_package_protocol_v17', 'nemotron9b_package_metadata_v17'):
                continue
            self.assertEqual(path.read_text().replace('_v17', '_v18'),
                             path.with_name(path.name.replace('_v17', '_v18')).read_text())
            count += 1
        self.assertEqual(count, 14)

    def test_protocol_only_changes_dependencies_and_source_list(self):
        def functions(text):
            return {n.name:ast.dump(n) for n in ast.parse(text).body
                    if isinstance(n, ast.FunctionDef) and n.name != 'required_sources'}
        self.assertEqual(functions(Path(old.__file__).read_text().replace('_v17', '_v18')),
                         functions(Path(p.__file__).read_text()))

    def test_all_original_science_stages_and_resources_preserved(self):
        for key in ('STAGES','SCIENTIFIC_STAGES','ROOT_FLOOR','NVME_FLOOR','ROOT_CAP','NVME_CAP',
                    'CAPS','BUCKETS','SPLIT_FLAGS','ALIAS_FLAGS','PACKAGE','COMMIT'):
            self.assertEqual(getattr(p,key),getattr(old,key))
        self.assertEqual(len(p.STAGES),52)
        for stage in (*p.STAGES,'controller'):
            self.assertEqual(p.resources(stage),old.resources(stage))

    def test_complete_new_and_old_sources_are_bound(self):
        required = set(p.required_sources())
        self.assertTrue(set(old.required_sources()) < required)
        self.assertTrue({str(path) for path in R.glob('nemotron9b_package_*v18.py')} <= required)
        self.assertTrue(all(Path(path).is_file() for path in required))
        self.assertIs(evidence.p,p)
        self.assertIs(p.compiler_links,p.output_inventory.compiler_links)

    def test_metadata_limits_and_old_schemas_preserved(self):
        self.assertEqual(p.metadata.LIMITS,old.metadata.LIMITS)
        for schema,shape in old.metadata.ENVELOPE_SCHEMAS.items():
            self.assertEqual(p.metadata.ENVELOPE_SCHEMAS[schema],shape)
        self.assertEqual(set(p.metadata.ENVELOPE_SCHEMAS)-set(old.metadata.ENVELOPE_SCHEMAS),
                         {'nemotron9b_package_runtime_v18','nemotron9b_package_runtime_source_review_v18'})

    def test_no_native_imports(self):
        self.assertFalse(any(name.split('.')[0] in {'torch','numpy','vllm','neuronxcc'} for name in sys.modules))


class History(unittest.TestCase):
    def test_actual_maintenance_appended_to_all_prior_owners(self):
        before,_ = completed.attention.completed()
        after,pins = completed.completed()
        self.assertEqual(after['rows'],before['rows'])
        self.assertEqual(after['processes'][:-1],before['processes'])
        self.assertEqual((len(after['rows']),len(after['processes'])),(1096,291))
        self.assertFalse(after['current_physical_release_verified'])
        self.assertIn(completed.EXIT['path'],pins)

    def test_missing_maintenance_requirement_rejected(self):
        _,pins = completed.completed()
        base = old.completed_history.load(old.completed_history.CATALOG)
        with self.assertRaisesRegex(RuntimeError,'Latest lossless'):
            completed.validate_spec(dict(required_attention_completion=completed.attention.EXIT,
                                         source_sha256=pins),base)

    def test_missing_maintenance_pin_rejected(self):
        _,pins = completed.completed()
        del pins[completed.EXIT['path']]
        base = old.completed_history.load(old.completed_history.CATALOG)
        with self.assertRaisesRegex(RuntimeError,'Complete maintenance'):
            completed.validate_spec(dict(required_attention_completion=completed.attention.EXIT,
                                         required_journal_compaction=completed.EXIT,source_sha256=pins),base)

    def test_corrupted_actual_exit_bytes_rejected(self):
        original = completed.stable_bytes
        def corrupt(path,cap):
            raw = original(path,cap)
            return raw+b' ' if str(path)==completed.EXIT['path'] else raw
        with patch.object(completed,'stable_bytes',side_effect=corrupt):
            with self.assertRaisesRegex(RuntimeError,'Exact maintenance'):
                completed.completed()


class Paths(previous.FullPaths):
    def setUp(self):
        for name,value in [('p',p),('evidence',evidence)]:
            change = patch.object(previous,name,value)
            change.start()
            self.addCleanup(change.stop)
        super().setUp()


if __name__=='__main__':
    unittest.main(verbosity=2)
