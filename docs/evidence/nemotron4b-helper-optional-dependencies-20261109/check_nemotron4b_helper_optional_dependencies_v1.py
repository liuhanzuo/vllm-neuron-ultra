"""Synthetic packaging controls; never import an inspected installed package."""
import base64
import csv
import hashlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_nemotron4b_helper_optional_dependencies_v1 as analyze


class OptionalCatalog(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='nano4b-optional-catalog-')
        self.addCleanup(self.temp.cleanup)
        self.site = Path(self.temp.name)
        self.name = 'example-1.0'
        self.folder = self.site/(self.name+'.dist-info')
        self.folder.mkdir()
        (self.site/'example').mkdir()
        self.marker = self.site/'must-not-execute'
        self.files = {
            'example/__init__.py': ('from pathlib import Path\nPath('+repr(str(self.marker))+').touch()\n').encode(),
            'example/feature.py': b'VALUE = 9\n',
            '_example_backend.cpython-310-x86_64-linux-gnu.so': b'SYNTHETIC-NOT-A-LIBRARY',
            self.name+'.dist-info/METADATA': b'Name: example\nVersion: 1.0\nRequires-Dist: extra\n',
        }
        for name, raw in self.files.items():
            (self.site/name).write_bytes(raw)
        self.record()
        self.roots = ['example', '_example_backend.cpython-310-x86_64-linux-gnu.so']

    def record(self, omit=(), sizes=None):
        sizes = sizes or {}
        out = io.StringIO()
        writer = csv.writer(out)
        for name, raw in self.files.items():
            if name not in omit:
                digest = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip('=')
                writer.writerow([name, 'sha256='+digest, sizes.get(name, len(raw))])
        writer.writerow([self.name+'.dist-info/RECORD', '', ''])
        (self.folder/'RECORD').write_text(out.getvalue())

    def test_top_level_native_is_stat_only_and_python_not_executed(self):
        original = os.open
        def only_source(path, *args, **kwargs):
            self.assertFalse(str(path).endswith('.so'), 'Native payload must not be opened')
            return original(path, *args, **kwargs)
        with patch.object(analyze.c.os, 'open', side_effect=only_source):
            distribution, sources, modules, native = analyze.inventory_distribution(
                self.site, self.name, self.roots)
        self.assertEqual(set(sources), {'example', 'example.feature'})
        self.assertEqual(set(native), {'_example_backend'})
        self.assertEqual(distribution['metadata_requirements'], ['Requires-Dist: extra'])
        self.assertFalse(native['_example_backend']['payload_digest_recomputed'])
        self.assertFalse(self.marker.exists())

    def test_missing_native_record_not_silently_dropped(self):
        self.record(omit=[self.roots[1]])
        with self.assertRaisesRegex(ValueError, 'native root'):
            analyze.inventory_distribution(self.site, self.name, self.roots)

    def test_native_record_size_mismatch_rejected_without_read(self):
        self.record(sizes={self.roots[1]: 1})
        with self.assertRaisesRegex(ValueError, 'Native RECORD size'):
            analyze.inventory_distribution(self.site, self.name, self.roots)

    def test_python_must_match_original_record(self):
        (self.site/'example/feature.py').write_bytes(b'VALUE = 8\n')
        with self.assertRaisesRegex(ValueError, 'pin changed'):
            analyze.inventory_distribution(self.site, self.name, self.roots)
        self.assertFalse(self.marker.exists())

    def test_native_root_symlink_rejected(self):
        native = self.site/self.roots[1]
        actual = self.site/'elsewhere'
        native.rename(actual)
        native.symlink_to(actual)
        with self.assertRaisesRegex(ValueError, 'Canonical ordinary path'):
            analyze.inventory_distribution(self.site, self.name, self.roots)

    def test_actual_namespace_parent_supported(self):
        (self.site/'space').mkdir()
        (self.site/'space/leaf.py').write_text('VALUE=1\n')
        namespaces = analyze.namespaces_for(self.site, {'space.leaf': {}}, {})
        self.assertEqual(namespaces, {'space':dict(path=str(self.site/'space'), init_py_absent=True)})

    def test_missing_package_source_not_reclassified_as_namespace(self):
        with self.assertRaisesRegex(ValueError, 'Actual namespace'):
            analyze.namespaces_for(self.site, {'example.feature': {}}, {})


if __name__ == '__main__':
    unittest.main(verbosity=2)
