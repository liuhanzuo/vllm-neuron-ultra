"""Stdlib tests of pinned plan, admission boundary and original-shard IO rules.

No Torch/SDK/native import, tensor operation, checkpoint or process/lock work.
Temporary tiny byte fixtures test the reader; these are not model weights.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import spark26_native_port_diagnostic_contract_v1 as c
import spark26_native_port_diagnostic_v2 as core


class DiagnosticSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = c.plan()

    def authority(self, plan=None):
        return {'schema': 'spark26_native_port_diagnostic_authority_v1',
                'model': c.original.MODEL_ID, 'revision': c.original.REVISION,
                'plan_sha256': c.digest(plan or self.plan), 'checkpoint_root': '/future/original/spark',
                'native_python': c.PYTHON, 'lock_identity': dict(c.LOCK_IDENTITY),
                'native_sources_authenticated': True, 'native_libraries_authenticated': True,
                'actual_owned_serial_worker': True, 'source_sha256': dict(c.NATIVE_SOURCES)}

    def test_pinned_original_checkpoint_and_tied_key(self):
        checkpoint = self.plan['checkpoint']
        self.assertEqual(len(checkpoint['weight_map']), 290)
        self.assertEqual(set(checkpoint['weight_map']), set(c.original.state_schema()) - {'lm_head.weight'})
        self.assertEqual(sum(x['bytes'] for x in checkpoint['files'].values()), 8224192408)
        self.assertEqual(checkpoint['tied_head_source'], 'model.embedding.weight')
        self.assertEqual([r['name'] for r in self.plan['qkv_weights']],
                         [f'model.layers.{i}.self_attn.q_k_v_proj.weight' for i in range(36)])

    def test_original_logit_thresholds_retained_separately(self):
        self.assertEqual(self.plan['original_logit_criteria'],
            {'cosine_min': .995, 'kl_max_nats': .02, 'max_abs_error_max': 2.,
             'argmax_match_fraction_min': .9, 'all_reference_margins_gt_0.5_must_match': True})
        self.assertIn('not a new model gate', self.plan['projection_numerics'])
        self.assertFalse(self.plan['full_model_qualified'])
        self.assertFalse(self.plan['new_speedup_claim'])

    def test_complete_trace_has_history_and_reuse(self):
        trace = self.plan['cache_trace']; seen = {}; epochs = {}; longest = 0; chain = 0
        self.assertEqual(len(trace), 534)
        self.assertEqual([r['slots'] for r in trace[:4]], [[3], [1], [0], [2]])
        reused = set(); at_boundary = set(); partial = set(); unsorted = False
        for row in trace:
            if row['op'] == 'prefill':
                slot, length, epoch = row['slots'][0], row['lengths'][0], row['epochs'][0]
                if slot in seen: reused.add(slot)
                seen[slot], epochs[slot] = length, epoch; chain = 0
            else:
                for slot, pos, epoch in zip(row['slots'], row['positions'], row['epochs']):
                    self.assertEqual((pos, epoch), (seen[slot], epochs[slot]))
                    if pos in (510, 511, 512, 513, 4095): at_boundary.add(pos)
                    seen[slot] += 1
                partial.add(len(row['slots']))
                unsorted |= row['slots'] != sorted(row['slots'])
                chain = chain + 1 if row['slots'] == [3] else 0
                longest = max(longest, chain)
        self.assertEqual(reused, {0, 1, 2})
        self.assertEqual(at_boundary, {510, 511, 512, 513, 4095})
        self.assertEqual(partial, {1, 2, 3, 4})
        self.assertTrue(unsorted); self.assertGreaterEqual(longest, 520)

    def test_smaller_or_relaxed_plan_refused(self):
        mutations = [lambda x: x['qkv_weights'].pop(), lambda x: x['cache_trace'].pop(),
                     lambda x: x['cache_layers'].pop(), lambda x: x['tp_ranks'].pop(),
                     lambda x: x['original_logit_criteria'].update(kl_max_nats=.03),
                     lambda x: x['checkpoint']['files'].pop(next(iter(x['checkpoint']['files'])))]
        for mutate in mutations:
            value = copy.deepcopy(self.plan); mutate(value)
            with self.assertRaises(ValueError): c.validate_plan(value)

    def test_bad_authority_never_reaches_native_factory(self):
        changes = [{'revision': 'wrong'}, {'model': 'other'}, {'native_python': '/usr/bin/python3'},
                   {'lock_identity': {'device': 1, 'inode': 2}}, {'plan_sha256': '0' * 64},
                   {'checkpoint_root': '/other'}, {'actual_owned_serial_worker': False},
                   {'native_libraries_authenticated': False}, {'native_sources_authenticated': False},
                   {'source_sha256': {}}]
        for change in changes:
            authority = self.authority(); authority.update(change); called = []
            with self.assertRaises(ValueError):
                core.run(self.plan, '/future/original/spark', lambda: authority,
                         lambda: called.append('native'), lambda *args: None)
            self.assertEqual(called, [])

    def test_changed_guard_never_reaches_native_factory(self):
        initial = self.authority(); calls = []
        def guard():
            calls.append('guard')
            return initial if len(calls) == 1 else dict(initial, changed=True)
        native = []
        with self.assertRaises(ValueError):
            core.run(self.plan, '/future/original/spark', guard,
                     lambda: native.append('native'), lambda *args: None)
        self.assertEqual(native, [])

    def test_incomplete_native_provider_rejected(self):
        with self.assertRaisesRegex(ValueError, 'provider set'):
            core.run(self.plan, '/future/original/spark', lambda: self.authority(),
                     lambda: {}, lambda *args: None)

    def tiny_reader(self, directory, data=b'fixture-only'):
        path = Path(directory) / 'tiny.safetensors'
        path.write_bytes(data)
        plan = {'checkpoint': {'files': {path.name: {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}}},
                'revision': c.original.REVISION, 'qkv_weights': []}
        reader = core.OriginalWeights(directory, plan, None, None, lambda: None)
        return path, reader

    def test_tiny_fixture_full_hash_and_length(self):
        with tempfile.TemporaryDirectory() as directory:
            path, reader = self.tiny_reader(directory)
            result = reader.authenticate_all()
            self.assertTrue(result['all_original_shard_hashes_match'])
            self.assertEqual(result['files'], 1)
            self.assertEqual(reader.verified[path.name], core._identity(path))

    def test_same_size_wrong_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, reader = self.tiny_reader(directory)
            path.write_bytes(b'x' * path.stat().st_size)
            with self.assertRaisesRegex(ValueError, 'LFS hash'): reader.authenticate_all()

    def test_wrong_size_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, reader = self.tiny_reader(directory)
            path.write_bytes(b'oversized-fixture')
            with self.assertRaisesRegex(ValueError, 'length'): reader.authenticate_all()
        with tempfile.TemporaryDirectory() as directory:
            path, reader = self.tiny_reader(directory)
            target = path.with_name('target'); path.rename(target); path.symlink_to(target)
            with self.assertRaisesRegex(ValueError, 'nonsymlink'): reader.authenticate_all()

    def test_runtime_guard_failure_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path, reader = self.tiny_reader(directory)
            error = RuntimeError('lost real serial ownership')
            def fail(): raise error
            reader.guard = fail
            with self.assertRaises(RuntimeError) as caught: reader.authenticate_all()
            self.assertIs(caught.exception, error); self.assertEqual(reader.verified, {})

    def test_raw_shard_read_before_native_tensor(self):
        with tempfile.TemporaryDirectory() as directory:
            path, reader = self.tiny_reader(directory)
            case = {'name': 'original.qkv'}; reader.plan['qkv_weights'] = [case]
            reader.plan['checkpoint']['weight_map'] = {case['name']: path.name}
            called = []; reader.safe_open = lambda *a, **kw: called.append(a)
            with self.assertRaisesRegex(ValueError, 'authentication precedes'): reader.get(case)
            self.assertEqual(called, [])

    def test_import_is_stdlib_only(self):
        self.assertFalse({'torch', 'numpy', 'safetensors', 'transformers', 'neuronx_distributed'} & set(sys.modules))


if __name__ == '__main__':
    unittest.main(verbosity=2)
