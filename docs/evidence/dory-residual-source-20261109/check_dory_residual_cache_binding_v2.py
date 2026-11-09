"""Reproduce real cache-callback mismatch before first forward; no model load."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import dory_residual_replay_runtime_v1 as old
import dory_residual_replay_runtime_v2 as new
from dory_residual_request_banks_v1 import CACHE_IDS

Q = Path(__file__).resolve().parent


class ReadyToCapture(RuntimeError):
    pass


class CacheBinding(unittest.TestCase):
    def env(self):
        return dict(torch=SimpleNamespace(bfloat16='BF16',float32='F32'), official=object(),
            spec=dict(schema='dory_residual_complete_replay_diagnostic_v1',diagnostic_only=True,
                runtime_admitted=True,selection_plan={},expected_original_raw_bytes=5867691008,
                budget=dict(auxiliary_raw_bytes=712343552,nvme_new_bytes=8*2**30)),
            guard=lambda: None,raw=lambda x: b'',save=lambda *a: None,out=Path('/synthetic'),
            control_active=[False],caches=[SimpleNamespace(prefix=f'backbone.layers.{i}.mixer',keys=None,values=None)
                                         for i in CACHE_IDS],
            extract_layer_index=lambda name: int(name.split('.')[2]))

    def test_original_mismatch_reproduced(self):
        with patch.object(old,'OriginalEventStream'), patch.object(old,'CaptureResidualVisits') as capture:
            with self.assertRaisesRegex(AttributeError,'split'):
                old.run(self.env())
            capture.assert_not_called()

    def test_all56_module_prefixes_reach_original_parser(self):
        env=self.env();seen=[]
        def extract(name):
            self.assertIsInstance(name,str)
            seen.append(name)
            return int(name.split('.')[2])
        env['extract_layer_index']=extract
        with patch.object(new,'OriginalEventStream'), patch.object(new,'CaptureResidualVisits',side_effect=ReadyToCapture):
            with self.assertRaises(ReadyToCapture):new.run(env)
        self.assertEqual(len(seen),56)
        self.assertEqual({int(x.split('.')[2]) for x in seen},set(CACHE_IDS))

    def test_malformed_prefix_still_rejected(self):
        env=self.env();env['caches'][0].prefix='bad'
        with patch.object(new,'OriginalEventStream'), patch.object(new,'CaptureResidualVisits') as capture:
            with self.assertRaises(IndexError):new.run(env)
            capture.assert_not_called()

    def test_duplicate_parsed_cache_id_rejected(self):
        env=self.env();env['caches'][0].prefix=env['caches'][1].prefix
        with patch.object(new,'OriginalEventStream'), patch.object(new,'CaptureResidualVisits') as capture:
            with self.assertRaises(RuntimeError):new.run(env)
            capture.assert_not_called()

    def test_only_callback_changed_in_runtime(self):
        prior=ast.parse((Q/'dory_residual_replay_runtime_v1.py').read_text())
        current=ast.parse((Q/'dory_residual_replay_runtime_v2.py').read_text())
        for node in ast.walk(prior):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='ResidualRequestBanks':
                node.args[1]=ast.parse('lambda cache: env["extract_layer_index"](cache.prefix)',mode='eval').body
        self.assertEqual(ast.dump(prior,include_attributes=False),ast.dump(current,include_attributes=False))

    def test_full_original_setup_unchanged_except_replay_import(self):
        prior=ast.parse((Q/'dory_residual_worker_preparation_v1.py').read_text())
        current=ast.parse((Q/'dory_residual_worker_preparation_v2.py').read_text())
        for node in ast.walk(prior):
            if isinstance(node,ast.ImportFrom) and node.module=='dory_residual_replay_runtime_v1':
                node.module='dory_residual_replay_runtime_v2'
        self.assertEqual(ast.dump(prior,include_attributes=False),ast.dump(current,include_attributes=False))


if __name__ == '__main__':unittest.main(verbosity=2)
