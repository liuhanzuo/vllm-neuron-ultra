"""Builder checks: actual metadata with future completion mocked in memory only."""
import copy
import hashlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

R = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(R), str(R/'research')]
import prepare_nemotron9b_package_runtime_v20 as b


class Builder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent = b.p.metadata.load(b.PARENT)
        cls.catalog = b.p.metadata.load(b.CATALOG)
        reader = b.completion.residual.Reader()
        cls.anchor = b.completion.residual.anchor(reader)
        cls.sources = b.merge(cls.anchor['spec']['source_sha256'], reader.pins)
        cls.pins = {name:b.p.metadata.sha(name) for name in b.p.required_sources()}
        cls.pins.update({ref['path']:ref['sha256'] for ref in b.TEST_REFS})
        cls.exit = dict(path=str(b.Q/'synthetic-never-written-builder-residual-exit.json'), sha256='a'*64)
        cls.observer = dict(path=str(b.Q/'nemotron9b-package-runtime-v20-fixture-observer.json'), sha256='b'*64)

    def setUp(self):
        h = copy.deepcopy(self.anchor['history'])
        h['processes'].extend([dict(pid=990001+i, start_ticks=str(99999999+i)) for i in range(3)])
        h['known_admitted_directories'] += ['residual-replay-r2', 'residual-replay-review-r2']
        h['completed_residual_review'] = self.exit
        self.proof = dict(history=h, source_sha256=b.merge(self.sources, {self.exit['path']:self.exit['sha256']}),
            historical_binary_sha256=b.partition.exact_dory_payloads(self.anchor),
            runtime_admitted=False, fresh_physical_drain_verified=False, full_model_qualified=False, new_speedup_claim=False)

    def compose(self, **changes):
        args = dict(parent=self.parent, catalog=self.catalog, run_id='r10-builder-fixture',
                    observer_ref=self.observer, pins=self.pins, review_exit_ref=self.exit)
        args.update(changes)
        with patch.object(b.completion.residual, 'completed', return_value=self.proof):
            return b.compose(**args)

    def test_scientific_fields_and_all20_historical_binary_pins_preserved(self):
        before = copy.deepcopy(self.parent)
        spec = self.compose()
        self.assertEqual(self.parent, before)
        self.assertEqual(set(spec)-set(self.parent), b.ADDED)
        for key, expected in self.parent.items():
            if key not in b.CHANGED:self.assertEqual(spec[key], expected, key)
        parts = b.p.metadata.classify_sources(spec, self.catalog)
        self.assertEqual(len(parts['historical_payload_sha256']), 20)
        self.assertEqual(len(spec['historical_payload_sha256']), 14)
        self.assertFalse(spec['runtime_readiness']['ready'])
        self.assertIsNone(spec['speedup_denominator'])
        self.assertEqual(len(spec['stages']), 52)
        raw = b.encoded(b.Q/'nemotron9b-package-runtime-v20-fixture-spec.json', spec)
        self.assertLessEqual(len(raw), 32*2**20)

    def test_modified_science_or_input_parent_is_rejected(self):
        for key in ('package_commit', 'stages', 'resources', 'disk_budget', 'payloads', 'references'):
            parent = copy.deepcopy(self.parent)
            parent[key] = None
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, 'Exact actual admitted r9'):
                self.compose(parent=parent)

    def test_source_rebinding_or_missing_new_worker_is_rejected(self):
        pins = dict(self.pins)
        pins[next(iter(self.parent['source_sha256']))] = '0'*64
        with self.assertRaisesRegex(RuntimeError, 'Never rebind'):self.compose(pins=pins)
        pins = dict(self.pins)
        pins.pop(str(R/'nemotron9b_package_worker_v20.py'))
        with self.assertRaisesRegex(RuntimeError, 'Complete new source'):self.compose(pins=pins)

    def test_dory_source_or_binary_omission_is_rejected(self):
        key = next(iter(self.proof['historical_binary_sha256']))
        self.proof['source_sha256'].pop(key)
        self.proof['historical_binary_sha256'].pop(key)
        with self.assertRaises(RuntimeError):self.compose()

    def test_old_failed_service_namespace_cannot_be_reused(self):
        with self.assertRaisesRegex(RuntimeError, 'Fresh explicit'):self.compose(run_id=self.parent['run_id'])
        old_unit = self.proof['history']['rows'][-1]['unit']
        self.proof['history']['rows'][-1]['unit'] = 'trn1-latest-verify-nemotron9b-package-r10-builder-fixture-controller'
        with self.assertRaises(RuntimeError):self.compose()
        self.proof['history']['rows'][-1]['unit'] = old_unit

    def test_incomplete_future_history_or_failed_r9_promotion_is_rejected(self):
        for key in ('rows', 'processes', 'known_admitted_directories', 'r9_outcome'):
            self.setUp()
            if key == 'r9_outcome':self.proof['history'][key]['producer_qualified'] = True
            else:self.proof['history'][key].pop()
            with self.subTest(key=key), self.assertRaises(RuntimeError):self.compose()

    def test_missing_completion_is_rejected_before_authentication_or_writes(self):
        with patch.object(b.completion.residual, 'completed') as reader, patch.object(b.p, 'once') as write:
            with self.assertRaisesRegex(RuntimeError, 'Actual completed'):
                b.compose(self.parent, self.catalog, 'r10-fixture', self.observer, self.pins, None)
            reader.assert_not_called()
            write.assert_not_called()

    def test_actual_failed_native_receipt_is_not_a_successful_review_exit(self):
        path = b.Q/'dory-residual-native-r1-failed-actual-exit-20261109-v1.json'
        ref = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        with patch.object(b.p, 'once') as write:
            with self.assertRaisesRegex(RuntimeError, 'Actual successful saved-review exit'):
                b.build('r10-fixture', ref)
            write.assert_not_called()

    def test_missing_future_exit_rejected_without_output(self):
        self.assertFalse(Path(self.exit['path']).exists())
        before = set(b.Q.glob('nemotron9b-package-runtime-v20-*'))
        with patch.object(b.p, 'once') as write:
            with self.assertRaises(FileNotFoundError):b.build('r10-fixture', self.exit)
            write.assert_not_called()
        self.assertEqual(set(b.Q.glob('nemotron9b-package-runtime-v20-*')), before)

    def test_observer_policy_retains_root_only_contract(self):
        old = b.p.metadata.load(self.parent['external_observer_contract'])
        new = b.observer_policy(self.parent, old, 'f'*64)
        self.assertEqual(new['participants'], ['/root'])
        self.assertEqual({k:v for k,v in new.items() if k != 'source_sha256'},
                         {k:v for k,v in old.items() if k != 'source_sha256'})

    def test_no_runtime_spec_native_import_or_automatic_launch(self):
        self.assertFalse(list(b.Q.glob('nemotron9b-package-runtime-v20-*-spec.json')))
        self.assertFalse(any(x.split('.')[0] in {'torch','numpy','vllm','neuronxcc'} for x in sys.modules))
        spec = self.compose()
        self.assertFalse(spec['automatic_successor_launch'])
        self.assertEqual(spec['retirement_policy'], 'none')


if __name__ == '__main__':
    unittest.main(verbosity=2)
