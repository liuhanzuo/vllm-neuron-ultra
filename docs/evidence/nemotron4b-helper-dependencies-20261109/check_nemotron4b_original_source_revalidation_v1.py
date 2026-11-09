"""Original installed hardlinks are retained; original hashes stay mandatory."""
import hashlib
import os
from pathlib import Path
import tempfile
import unittest

from analyze_nemotron4b_helper_dependencies_v2 import original_source


class OriginalSources(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='nano4b-original-source-')
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/'source.py'
        self.path.write_bytes(b'original = True\n')
        self.digest = hashlib.sha256(self.path.read_bytes()).hexdigest()

    def test_original_hardlink_accepted_and_link_count_recorded(self):
        os.link(self.path, self.path.with_name('cache-copy.py'))
        result = original_source(self.path, self.digest)
        self.assertEqual(result['sha256'], self.digest)
        self.assertEqual(result['original_nlink'], 2)

    def test_change_through_hardlink_still_fails_original_pin(self):
        other = self.path.with_name('cache-copy.py'); os.link(self.path, other)
        other.write_bytes(b'original = False\n')
        with self.assertRaisesRegex(ValueError, 'digest unchanged'):
            original_source(self.path, self.digest)

    def test_symlink_not_treated_as_installed_hardlink(self):
        other = self.path.with_name('alias.py'); other.symlink_to(self.path)
        with self.assertRaisesRegex(ValueError, 'canonical'):
            original_source(other, self.digest)


if __name__ == '__main__': unittest.main(verbosity=2)
