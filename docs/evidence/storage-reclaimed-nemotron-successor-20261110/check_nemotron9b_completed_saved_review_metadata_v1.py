"""Read-only tests for the exact completed Dory review envelope."""
import hashlib
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock,patch
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'research')]
import nemotron9b_package_metadata_v24 as metadata
from research import nemotron9b_completed_saved_review_metadata_v1 as c

class Envelope(unittest.TestCase):
    def test_actual_complete_saved_review_authenticated(self):
        raw=metadata.text(c.SAVED['path'],c.SAVED['sha256'])
        self.assertEqual(len(raw),16594498)
        self.assertEqual(hashlib.sha256(raw).hexdigest(),c.SAVED['sha256'])

    def test_wrong_path_rejected_before_read(self):
        reader=Mock()
        with self.assertRaises(RuntimeError):c.text(c.SAVED['path']+'.other',c.SAVED['sha256'],reader)
        reader.assert_not_called()

    def test_wrong_hash_rejected_before_read(self):
        reader=Mock()
        with self.assertRaises(RuntimeError):c.text(c.SAVED['path'],'0'*64,reader)
        reader.assert_not_called()

    def test_corrupt_external_exit_cannot_open_saved_review(self):
        reader=Mock(return_value=b'{}')
        with self.assertRaisesRegex(RuntimeError,'external completion'):c.text(c.SAVED['path'],c.SAVED['sha256'],reader)
        reader.assert_called_once_with(c.EXIT['path'],8*2**20)

    def test_corrupt_saved_review_rejected(self):
        original=metadata.stable_bytes(c.EXIT['path'],8*2**20)
        reader=Mock(side_effect=[original,b'{}'])
        with self.assertRaisesRegex(RuntimeError,'complete saved review bytes'):c.text(c.SAVED['path'],c.SAVED['sha256'],reader)
        self.assertEqual(reader.call_args.args,(c.SAVED['path'],32*2**20))

    def test_original_scope_check_is_mandatory(self):
        with patch.object(c.residual.old,'saved_scope',side_effect=RuntimeError('incomplete scientific scope'))as check:
            with self.assertRaisesRegex(RuntimeError,'scientific scope'):
                c.text(c.SAVED['path'],c.SAVED['sha256'],metadata.stable_bytes)
        check.assert_called_once()

    def test_ordinary_limits_unchanged(self):
        self.assertEqual(metadata.LIMIT,8*2**20)
        self.assertEqual(metadata.ENVELOPE_LIMIT,32*2**20)
        self.assertFalse(c.accepts('/tmp/saved-residual-review.json'))
        self.assertFalse(c.accepts(c.EXIT['path']))

if __name__=='__main__':unittest.main(verbosity=2)
