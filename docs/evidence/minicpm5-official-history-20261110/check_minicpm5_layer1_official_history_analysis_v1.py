"""Synthetic distance/target controls; no real model or observed payload reads."""
import hashlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import minicpm5_layer1_official_history_analysis_v1 as analysis


class AnalysisChecks(unittest.TestCase):
    def fixture(self):
        target = b'\x80\x3f'*8192
        raw = {}; rows = []; native = {}
        for arm in analysis.boundary.ARMS:
            for ci in analysis.boundary.DECODE_CALLS:
                frames = [dict(index=ci*4+i, fixture=i, slot=i, step=ci) for i in range(4)]
                for f in frames: native[f['fixture'], f['step']] = dict(targets={analysis.boundary.ATTENTION: target[:4096]})
                for mode in analysis.contract.MODES:
                    value = {'own': b'\x00\x40', 'restore': b'\x00\x40', 'official_key': b'\x80\x3f',
                             'official_value': b'\x40\x40', 'official_both': b'\x00\x3f'}[mode]*8192
                    path = '/synthetic/%s/%s/%s' % (arm, ci, mode); raw[path] = value
                    rows.append(dict(arm=arm, call=ci, mode=mode, path=path, sha256=hashlib.sha256(value).hexdigest(),
                        official_target_sha256=hashlib.sha256(target).hexdigest(), physical_frame_order=frames))
        proof = dict(complete=True, decode_calls=180, all_original_and_restored_outputs_exact=True,
                     rank_proofs=[dict(outputs=rows)])
        return proof, native, raw

    def test_all_frames_relative_distances_and_improvement(self):
        proof, native, raw = self.fixture()
        with patch.object(analysis.official, 'completed', return_value={}), patch.object(analysis.inputs, 'load', return_value=native), \
             patch.object(Path, 'read_bytes', lambda p: raw[str(p)]):
            result = analysis.analyze(proof, {'layer1_official_history': {}}, lambda r: None, lambda: None)
        self.assertEqual(len(result['records']), 360)
        for s in result['summary']:
            expected = {'own': (1.0, 0, 0), 'restore': (1.0, 0, 0), 'official_key': (0.0, 24, 0),
                        'official_value': (2.0, 0, 24), 'official_both': (0.5, 24, 0)}[s['mode']]
            self.assertEqual((s['aggregate_relative_l2'], s['improved_frames'], s['worsened_frames']), expected)
            self.assertEqual(s['frames'], 24)
        self.assertFalse(result['full_model_qualified']); self.assertFalse(result['new_speedup_claim'])

    def test_rehashed_different_target_binding_rejected(self):
        proof, native, raw = self.fixture(); proof['rank_proofs'][0]['outputs'][0]['official_target_sha256'] = '0'*64
        with patch.object(analysis.official, 'completed', return_value={}), patch.object(analysis.inputs, 'load', return_value=native), \
             patch.object(Path, 'read_bytes', lambda p: raw[str(p)]), self.assertRaises(ValueError):
            analysis.analyze(proof, {'layer1_official_history': {}}, lambda r: None, lambda: None)


if __name__ == '__main__': unittest.main(verbosity=2)
