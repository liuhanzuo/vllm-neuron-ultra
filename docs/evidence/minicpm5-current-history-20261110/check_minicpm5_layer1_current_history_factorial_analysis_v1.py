"""Synthetic distance/target controls; no real model or observed payload reads."""
import hashlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import minicpm5_layer1_current_history_factorial_analysis_v1 as analysis


class AnalysisChecks(unittest.TestCase):
    def fixture(self):
        target = b'\x80\x3f'*8192
        raw = {}; rows = []; native = {}
        for arm in analysis.boundary.ARMS:
            for ci in analysis.boundary.DECODE_CALLS:
                frames = [dict(index=ci*4+i, fixture=i, slot=i, step=ci) for i in range(4)]
                for f in frames: native[f['fixture'], f['step']] = dict(targets={analysis.boundary.ATTENTION: target[:4096]})
                for mode in analysis.contract.MODES:
                    value = {'own': b'\x00\x40', 'restore': b'\x00\x40', 'official_current': b'\x80\x3f',
                             'official_history': b'\x40\x40', 'official_both': b'\x00\x3f'}[mode]*8192
                    path = '/synthetic/%s/%s/%s' % (arm, ci, mode); raw[path] = value
                    rows.append(dict(arm=arm, call=ci, mode=mode, path=path, sha256=hashlib.sha256(value).hexdigest(),
                        official_target_sha256=hashlib.sha256(target).hexdigest(), physical_frame_order=frames))
        proof = dict(complete=True, decode_calls=180, all_original_and_restored_outputs_exact=True,
                     previous_real_controls_full_bytes_exact=True, previous_control_calls=108,
                     official_current_intervention_performed=True,
                     rank_proofs=[dict(outputs=rows)])
        return proof, native, raw

    def test_all_frames_relative_distances_and_improvement(self):
        proof, native, raw = self.fixture()
        with patch.object(analysis.official, 'completed', return_value={}), patch.object(analysis.inputs, 'load', return_value=native), \
             patch.object(Path, 'read_bytes', lambda p: raw[str(p)]):
            result = analysis.analyze(proof, {'layer1_official_history': {}}, lambda r: None, lambda: None)
        self.assertEqual(len(result['records']), 360)
        for s in result['summary']:
            expected = {'own': (1.0, 0, 0), 'restore': (1.0, 0, 0), 'official_current': (0.0, 24, 0),
                        'official_history': (2.0, 0, 24), 'official_both': (0.5, 24, 0)}[s['mode']]
            self.assertEqual((s['aggregate_relative_l2'], s['improved_frames'], s['worsened_frames']), expected)
            self.assertEqual(s['frames'], 24)
        self.assertFalse(result['full_model_qualified']); self.assertFalse(result['new_speedup_claim'])
        self.assertEqual(len(result['interactions']),72)
        self.assertTrue(result['current_qkv_change_together'])
        for row in result['interaction_summary']:
            self.assertEqual(row['frames'],24); self.assertEqual(row['nonzero_interaction_frames'],24)
            self.assertEqual(row['aggregate_interaction_relative_l2'],1.5)

    def test_rehashed_different_target_binding_rejected(self):
        proof, native, raw = self.fixture(); proof['rank_proofs'][0]['outputs'][0]['official_target_sha256'] = '0'*64
        with patch.object(analysis.official, 'completed', return_value={}), patch.object(analysis.inputs, 'load', return_value=native), \
             patch.object(Path, 'read_bytes', lambda p: raw[str(p)]), self.assertRaises(ValueError):
            analysis.analyze(proof, {'layer1_official_history': {}}, lambda r: None, lambda: None)


    def test_additive_and_interacting_vectors_have_distinct_diagnostics(self):
        import struct
        def raw(*values):
            return b''.join(struct.pack('<H',struct.unpack('<I',struct.pack('<f',v))[0]>>16)for v in values)
        cells=dict(own=raw(1,2),official_current=raw(2,4),official_history=raw(4,6),
            official_both=raw(5,8),restore=raw(1,2))
        additive=analysis.interaction(cells,raw(1,1))
        self.assertEqual(additive['interaction_l2_squared'],0)
        cells['official_both']=raw(6,6)
        nonlinear=analysis.interaction(cells,raw(1,1))
        self.assertEqual(nonlinear['interaction_l2_squared'],5)
        self.assertEqual(nonlinear['interaction_max_abs'],2)
        self.assertEqual(nonlinear['current_effect_l2_squared'],5)
        self.assertEqual(nonlinear['history_effect_l2_squared'],25)
        self.assertEqual(nonlinear['combined_effect_l2_squared'],41)
        self.assertAlmostEqual(nonlinear['interaction_relative_l2'],(5/2)**0.5)

    def test_missing_or_corrupt_restore_and_nonfinite_cells_rejected(self):
        base={mode:b'\x80\x3f' for mode in analysis.contract.MODES}
        bad=dict(base);bad.pop('official_history')
        with self.assertRaises(ValueError):analysis.interaction(bad,b'\x80\x3f')
        bad=dict(base,restore=b'\x00\x40')
        with self.assertRaises(ValueError):analysis.interaction(bad,b'\x80\x3f')
        bad=dict(base,official_both=b'\x80\x7f')
        with self.assertRaises(ValueError):analysis.interaction(bad,b'\x80\x3f')
        with self.assertRaises(ValueError):analysis.interaction(base,b'')

    def test_zero_target_norm_does_not_invent_relative_ratio(self):
        base={mode:b'\x80\x3f' for mode in analysis.contract.MODES}
        result=analysis.interaction(base,b'\x00\x00')
        self.assertEqual(result['interaction_l2_squared'],0)
        self.assertIsNone(result['interaction_relative_l2'])

    def test_analysis_has_no_native_imports(self):
        self.assertFalse({'torch','numpy','neuronxcc','torch_neuronx'}&set(sys.modules))


if __name__ == '__main__': unittest.main(verbosity=2)
