"""Exact new compiler filename and unchanged path/byte boundaries."""
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest

Q = Path(__file__).resolve().parent
sys.path[:0] = [str(Q), str(Q.parent)]
import nemotron9b_compiler_dynamic_links_v2 as candidate
import nemotron9b_compiler_dynamic_links_v1 as original


class GeneratedLinks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/'compile-work'
        self.unit = self.root/'context_encoding_model/_tp0_bk2/neuronxcc-fixture'
        self.name = 'value_sg0001_tiled_pf_transposetiled_pf_transpose_sg0001_13912_identity_11_CRSM.npy'
        self.path = self.unit/'sgLnk/sg00'/self.name
        self.target = self.unit/'sg01'/self.name
        self.path.parent.mkdir(parents=True)
        self.target.parent.mkdir(parents=True)
        self.target.write_bytes(b'complete synthetic numeric bytes')
        self.path.symlink_to(self.target)

    def test_old_parser_reproduces_rejection(self):
        with self.assertRaisesRegex(RuntimeError, 'Reviewed generated numeric basename'):
            original.link_record(self.path, self.root)

    def test_new_name_full_target_hash(self):
        row = candidate.link_record(self.path, self.root)
        self.assertEqual(row['target']['sha256'], hashlib.sha256(self.target.read_bytes()).hexdigest())
        self.assertEqual(row['compiler_asset_kind'], 'generated_tiled_transpose_numeric')

    def test_metadata_scan_does_not_hash_target(self):
        row = candidate.link_record(self.path, self.root, hash_content=False)
        self.assertFalse(row['target']['content_hashed'])
        self.assertNotIn('sha256', row['target'])

    def test_filename_subgraph_mismatch_rejected(self):
        name = self.name.replace('transpose_sg0001', 'transpose_sg0002')
        with self.assertRaises(RuntimeError):
            candidate.target_for(self.path.with_name(name), self.root)

    def test_other_kernel_name_rejected(self):
        name = self.name.replace('tiled_pf_transposetiled_pf_transpose', 'unknown_kernel')
        with self.assertRaises(RuntimeError):
            candidate.target_for(self.path.with_name(name), self.root)

    def test_unknown_bucket_rejected(self):
        path = Path(str(self.path).replace('_tp0_bk2', '_tp0_bk8'))
        with self.assertRaises(RuntimeError):
            candidate.target_for(path, self.root)

    def test_target_subgraph_outside_two_digit_scope(self):
        with self.assertRaises(RuntimeError):
            candidate.target_for(self.path.with_name(self.name.replace('sg0001', 'sg0100')), self.root)

    def test_wrong_unit_target_rejected(self):
        self.path.unlink()
        self.path.symlink_to(str(self.target).replace('neuronxcc-fixture', 'neuronxcc-other'))
        with self.assertRaises(RuntimeError):
            candidate.link_record(self.path, self.root)

    def test_target_symlink_rejected(self):
        real = self.target.with_suffix('.data')
        self.target.rename(real)
        self.target.symlink_to(real)
        with self.assertRaises(RuntimeError):
            candidate.link_record(self.path, self.root)

    def test_target_hardlink_rejected(self):
        os.link(self.target, self.target.with_suffix('.second'))
        with self.assertRaises(RuntimeError):
            candidate.link_record(self.path, self.root)

    def test_parent_symlink_rejected(self):
        old = self.unit/'sg01'
        moved = self.unit/'moved'
        old.rename(moved)
        old.symlink_to(moved, target_is_directory=True)
        with self.assertRaises(OSError):
            candidate.link_record(self.path, self.root)

    def test_original_parser_not_mutated(self):
        candidate.link_record(self.path, self.root)
        with self.assertRaises(RuntimeError):
            original.target_for(self.path, self.root)

    def test_original_numeric_name_still_accepted(self):
        name = 'value_sg0001_identity_11_CRSM.npy'
        self.assertEqual(candidate.target_for(self.path.with_name(name), self.root),
                         (self.unit/'sg01'/name, 'generated_numeric'))

    def test_original_debug_name_still_accepted(self):
        name = 'debug_info_stack_frame_index.dbg'
        self.assertEqual(candidate.target_for(self.unit/'sgLnk'/name, self.root),
                         (self.unit/name, 'generated_debug'))


if __name__ == '__main__':unittest.main(verbosity=2)
