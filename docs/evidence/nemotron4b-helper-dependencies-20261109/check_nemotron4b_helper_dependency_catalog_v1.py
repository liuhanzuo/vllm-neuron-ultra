"""Generated source/metadata fixtures; no installed package execution."""
import base64
import hashlib
from pathlib import Path
import tempfile
import unittest

import nemotron4b_helper_dependency_catalog_v1 as c


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='nano4b-dependency-fixture-')
        self.addCleanup(self.tmp.cleanup)
        self.site = Path(self.tmp.name)
        self.raw = b'value = 1\n'
        self.file = self.site/'sample.py'; self.file.write_bytes(self.raw)
        self.digest = hashlib.sha256(self.raw).hexdigest()
        self.record = {'sample.py': ('sha256='+base64.urlsafe_b64encode(bytes.fromhex(self.digest)).decode().rstrip('='), str(len(self.raw)))}

    def test_record_verified_complete_source(self):
        raw, ref = c.verified_source(self.file, self.site, self.record, self.digest)
        self.assertEqual(raw, self.raw); self.assertTrue(ref['installed_RECORD_verified'])
        self.assertEqual(ref['sha256'], self.digest)

    def test_wrong_record_digest(self):
        self.file.write_bytes(b'value = 2\n')
        with self.assertRaisesRegex(ValueError, 'pin changed'):
            c.verified_source(self.file, self.site, self.record)

    def test_wrong_record_size(self):
        self.record['sample.py'] = (self.record['sample.py'][0], '999')
        with self.assertRaisesRegex(ValueError, 'size'):
            c.verified_source(self.file, self.site, self.record)

    def test_previous_pin_cannot_be_replaced_by_record(self):
        with self.assertRaisesRegex(ValueError, 'Prior source'):
            c.verified_source(self.file, self.site, self.record, '0'*64)

    def test_unrecorded_source(self):
        with self.assertRaisesRegex(ValueError, 'absent'):
            c.verified_source(self.file, self.site, {})

    def test_duplicate_record_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unique'):
            c.parse_record(b'a.py,,\na.py,,\n')

    def test_script_relative_record_retained_but_not_selected(self):
        rows = c.parse_record(b'../../../bin/helper,,\n')
        self.assertIn('../../../bin/helper', rows)
        with self.assertRaisesRegex(ValueError, 'under installed site'):
            c.record_digest(rows, '../../../bin/helper')

    def test_digest_must_be_canonical(self):
        for digest in ['sha256=@@@', 'sha256=YQ', 'md5=abc', self.record['sample.py'][0]+'=']:
            with self.subTest(digest=digest), self.assertRaises(ValueError):
                c.record_digest({'sample.py': (digest, '10')}, 'sample.py')

    def test_symlink_rejected(self):
        alias = self.site/'alias.py'; alias.symlink_to(self.file)
        with self.assertRaisesRegex(ValueError, 'Canonical'):
            c.source(alias)

    def test_native_payload_and_bytecode_rejected_before_read(self):
        for name in ['extension.so', 'weights.bin', 'sample.pyc']:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'allowlist'):
                c.source(self.site/name)

    def test_source_limit_before_read(self):
        with self.file.open('wb') as stream: stream.truncate(8*2**20+1)
        with self.assertRaisesRegex(ValueError, 'Bounded'):
            c.source(self.file)

    def test_module_paths(self):
        self.assertEqual(c.module_name('alpha/beta/__init__.py'), ('alpha.beta', True))
        self.assertEqual(c.module_name('six.py'), ('six', False))
        with self.assertRaises(ValueError): c.module_name('../escape.py')

    def test_cross_package_parents_and_namespace(self):
        modules = {'alpha': dict(source='from beta.parts import leaf\n', is_package=True),
                   'beta': dict(source='', is_package=True),
                   'beta.parts.leaf': dict(source='import os\n', is_package=False)}
        result = c.dependency_graph('import alpha', modules, {'beta.parts'}, {})
        self.assertEqual(result['reachable_python_modules'], sorted(modules))
        self.assertEqual(result['reachable_namespace_packages'], ['beta.parts'])
        self.assertEqual([r['target'] for r in result['unresolved_import_boundaries']], ['os'])
        self.assertFalse(result['complete_runtime_dependency_closure_proven'])

    def test_relative_import_and_native_boundary(self):
        modules = {'alpha': dict(source='from . import native\n', is_package=True)}
        result = c.dependency_graph('import alpha', modules, set(), {'alpha.native': {}})
        self.assertEqual(result['reachable_native_boundaries'], ['alpha.native'])

    def test_known_attribute_not_invented_as_module(self):
        modules = {'alpha': dict(source='answer = 42\n', is_package=True)}
        result = c.dependency_graph('from alpha import answer', modules, set(), {})
        self.assertEqual(result['unresolved_import_boundaries'], [])

    def test_both_branches_visited_but_functions_deferred(self):
        modules = {'alpha': dict(source='if flag:\n import left\nelse:\n import right\ndef later():\n import deferred\n', is_package=True)}
        result = c.dependency_graph('import alpha', modules, set(), {})
        self.assertEqual({r['target'] for r in result['unresolved_import_boundaries']}, {'left', 'right'})
        self.assertEqual(result['module_scans']['alpha']['function_body_imports'][0]['names'], ['deferred'])

    def test_dynamic_execution_remains_explicit(self):
        modules = {'alpha': dict(source="import importlib\nimportlib.import_module('other')\n", is_package=True)}
        result = c.dependency_graph('import alpha', modules, set(), {})
        self.assertEqual(len(result['module_scans']['alpha']['dynamic_import_or_execution_calls']), 1)
        self.assertNotIn('other', result['reachable_python_modules'])

    def test_missing_implicit_parent_rejected(self):
        modules = {'alpha.child': dict(source='', is_package=False)}
        with self.assertRaisesRegex(ValueError, 'implicit-parent'):
            c.dependency_graph('import alpha.child', modules, set(), {})


if __name__ == '__main__': unittest.main(verbosity=2)
