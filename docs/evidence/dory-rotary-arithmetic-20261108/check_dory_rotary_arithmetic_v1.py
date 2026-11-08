"""Stdlib-only source/extraction checks; never execute torch or RoPE."""
import ast
import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

Q = Path('/home/ubuntu/experiments/trn1-latest-models-20261031/research')
SOURCE = Q/'dory_rotary_arithmetic_v1.py'
spec = importlib.util.spec_from_file_location('dory_rotary_preparation', SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class NeverTorch:
    def __getattr__(self, name):
        raise AssertionError('Native access prohibited in source test: ' + name)

class Checks(unittest.TestCase):
    def test_import_without_native(self):
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('vllm', sys.modules)
    def test_all_pinned_sources(self):
        texts, nodes, proof = m.prepare_sources()
        self.assertEqual(set(texts), {'base', 'common', 'candidate', 'config'})
        self.assertEqual(len(nodes), 7)
        self.assertEqual(len(proof['extracted_functions']), 7)
    def test_bodies_unchanged(self):
        texts, nodes, _ = m.prepare_sources()
        for key, cls, name in m.SELECTION:
            tree = ast.parse(texts[key]); parent = tree
            if cls is not None:
                parent = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
            original = copy.deepcopy(next(n for n in parent.body if isinstance(n, ast.FunctionDef) and n.name == name))
            original.decorator_list = []
            ident = '.'.join(x for x in (key, cls, name) if x)
            self.assertEqual(ast.dump(original, include_attributes=True), ast.dump(nodes[ident], include_attributes=True))
    def test_bind_without_native(self):
        _, nodes, _ = m.prepare_sources()
        funcs = m.bind_functions(nodes, NeverTorch())
        self.assertEqual(len(funcs), 7)
        self.assertTrue(all(callable(f) for f in funcs.values()))
    def test_original_candidate_not_imported(self):
        m.prepare_sources()
        self.assertNotIn('dory_reference_candidate', sys.modules)
    def test_complete_matrix(self):
        plan = m.plan()
        self.assertEqual(plan['expected_position_comparisons'], 96)
        self.assertEqual(plan['expected_position_comparisons'], len(plan['theta'])*len(plan['dtypes'])*len(plan['fixtures'])*2*len(plan['positions']))
    def test_original_boundaries(self):
        self.assertEqual(m.plan()['positions'], [0, 1, 1023, 1024, 2048, 262143])
    def test_full_cache_retained(self):
        plan = m.plan()
        self.assertEqual(plan['original_cache_positions'], 262144)
        self.assertEqual(plan['native_cache_fp32_bytes'], 268435456)
    def test_no_premature_qualification(self):
        plan = m.plan()
        for key in ('runtime_admitted', 'native_executed', 'numerical_parity_proven', 'native_reference_qualified', 'new_speedup_claim'):
            self.assertIs(plan[key], False)
    def test_guard_required_before_native(self):
        with self.assertRaisesRegex(ValueError, 'guard required'):
            m.run_comparison(torch=NeverTorch(), guard=None)
    def test_denied_guard_before_native(self):
        def deny(): raise RuntimeError('runtime not admitted')
        with self.assertRaisesRegex(RuntimeError, 'runtime not admitted'):
            m.run_comparison(torch=NeverTorch(), guard=deny)
    def test_missing_class(self):
        with self.assertRaisesRegex(ValueError, 'Unique original class'):
            m.select_function('def f(): pass', 'C', 'f')
    def test_duplicate_class(self):
        with self.assertRaisesRegex(ValueError, 'Unique original class'):
            m.select_function('class C: pass\nclass C: pass', 'C', 'f')
    def test_missing_function(self):
        with self.assertRaisesRegex(ValueError, 'Unique original function'):
            m.select_function('class C: pass', 'C', 'f')
    def test_duplicate_function(self):
        with self.assertRaisesRegex(ValueError, 'Unique original function'):
            m.select_function('def f(): pass\ndef f(): pass', None, 'f')
    def test_dynamic_decorator_rejected(self):
        with self.assertRaisesRegex(ValueError, 'staticmethod'):
            m.select_function('@execute()\ndef f(): pass', None, 'f')
    def test_multiple_decorators_rejected(self):
        with self.assertRaisesRegex(ValueError, 'staticmethod'):
            m.select_function('@staticmethod\n@staticmethod\ndef f(): pass', None, 'f')
    def test_static_binding_only(self):
        node = m.select_function('@staticmethod\ndef f(): return 42', None, 'f')
        self.assertEqual(node.decorator_list, [])
        self.assertEqual(node.body[0].value.value, 42)
    def test_metadata_digest_rejected(self):
        pins = dict(m.PINS); path, _ = pins['config']; pins['config'] = (path, '0'*64)
        with patch.object(m, 'PINS', pins), self.assertRaisesRegex(ValueError, 'Exact pinned source digest'):
            m.prepare_sources()
    def test_source_digest_rejected(self):
        pins = dict(m.PINS); path, _ = pins['candidate']; pins['candidate'] = (path, '0'*64)
        with patch.object(m, 'PINS', pins), self.assertRaisesRegex(ValueError, 'Exact pinned source digest'):
            m.prepare_sources()
    def test_source_size_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'too_large.py'; raw = b'x'*65537; path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, 'Bounded pinned source'):
                m.read_pinned(path, hashlib.sha256(raw).hexdigest())
    def test_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'original'; path.write_bytes(b'pass'); alias = Path(directory)/'alias'; alias.symlink_to(path)
            with self.assertRaisesRegex(ValueError, 'Canonical pinned source'):
                m.read_pinned(alias, hashlib.sha256(b'pass').hexdigest())
    def test_relative_path_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Canonical pinned source'):
            m.read_pinned(Path('relative'), '0'*64)
    def test_no_main_entrypoint(self):
        tree = ast.parse(SOURCE.read_text())
        imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any(isinstance(n, ast.Import) and any(a.name in ('torch', 'vllm') for a in n.names) for n in imports))
        self.assertFalse(any(isinstance(n, ast.If) for n in tree.body))

if __name__ == '__main__':
    unittest.main(verbosity=2)
