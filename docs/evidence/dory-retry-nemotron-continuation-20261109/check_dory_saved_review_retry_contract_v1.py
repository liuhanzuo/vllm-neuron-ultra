"""Synthetic terminal-receipt mutations using authentic immutable metadata.

No fabricated external receipt is saved. Process disappearance and the future
failed tool event are simulated only inside tests. No model/raw tensor reads.
"""
import copy
import json
from pathlib import Path
import unittest

import dory_saved_review_retry_contract_v1 as c


class Retry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = {}
        cls.reader = c.original.Reader()
        admission = cls.reader(dict(path=str(c.OUT/'admission.json'), sha256=c.ANCHORS['admission.json']))
        cls.fixture_ref = dict(path='/synthetic-only/dory-failed-tool-exit.json', sha256='1'*64)
        cls.receipt = dict(schema='dory_residual_saved_review_r2_failed_actual_exit_v1',
            reviewer='/root', actual_external_exit_observed=True, session_id=86437,
            exit_code=1, tool_chunk='abcd01', owner=admission['owner'], spec=c.PARENT_SPEC,
            source_review=c.PARENT_REVIEW,
            output_files={name:dict(path=str(c.OUT/name),sha256=digest) for name,digest in c.ANCHORS.items()},
            full_model_qualified=False,new_speedup_claim=False,saved_review_completed=False)
        cls.directories = {p.name for p in c.BASE.iterdir() if p.is_dir() and (p/'admission.json').is_file()}
        def read(ref):
            if ref == cls.fixture_ref:
                return copy.deepcopy(cls.receipt)
            value = cls.reader(ref)
            cls.data[ref['path']] = value
            return value
        cls.positive = c.failed_history(cls.fixture_ref, read, gone=lambda owner: None,
            lookup=lambda pid: None, output_names=c.ANCHORS, directories=cls.directories)

    def setUp(self):
        self.receipt = copy.deepcopy(type(self).receipt)
        self.mutations = {}
        self.extra = {}
        self.names = set(c.ANCHORS)
        self.gone = []

    def read(self, ref):
        if ref == self.fixture_ref:
            return copy.deepcopy(self.receipt)
        value = copy.deepcopy(self.extra.get(ref['path'], self.data.get(ref['path'])))
        self.assertIsNotNone(value, ref['path'])
        if ref['path'] in self.mutations:
            self.mutations[ref['path']](value)
        return value

    def run_fixture(self, **kwargs):
        return c.failed_history(self.fixture_ref, self.read, gone=kwargs.pop('gone', self.gone.append),
            lookup=lambda pid: None, output_names=self.names, directories=self.directories, **kwargs)

    def test_full_failure_extension_retains_all_historical_counts(self):
        result = self.run_fixture()
        h = result['history']
        self.assertEqual((len(h['rows']),len(h['processes']),len(h['known_admitted_directories'])), (1149,348,37))
        self.assertEqual(h['failed_residual_saved_review'], self.fixture_ref)
        self.assertFalse(result['failed_review_complete'])
        self.assertEqual(result['raw_tensor_bytes_read'], 0)

    def test_successful_exit_is_not_a_retry_trigger(self):
        self.receipt['exit_code'] = 0
        with self.assertRaisesRegex(RuntimeError, 'nonzero'):
            self.run_fixture()

    def test_boolean_exit_code_is_rejected(self):
        self.receipt['exit_code'] = True
        with self.assertRaises(RuntimeError):
            self.run_fixture()

    def test_poll_is_not_external_exit(self):
        self.receipt['actual_external_exit_observed'] = False
        with self.assertRaises(RuntimeError):
            self.run_fixture()

    def test_wrong_session_is_rejected(self):
        self.receipt['session_id'] = 29930
        with self.assertRaises(RuntimeError):
            self.run_fixture()

    def test_live_original_owner_prevents_any_successor(self):
        def live(owner):
            raise RuntimeError('Exact original process still live')
        with self.assertRaisesRegex(RuntimeError, 'still live'):
            self.run_fixture(gone=live)

    def test_changed_recorded_owner_rejected(self):
        self.receipt['owner']['start_ticks'] = '403068362'
        with self.assertRaises(RuntimeError):
            self.run_fixture()

    def test_changed_argv_rejected(self):
        self.receipt['owner']['argv'][-1] = '0'*64
        with self.assertRaises(RuntimeError):
            self.run_fixture()

    def test_omitted_output_rejected(self):
        del self.receipt['output_files']['post-exit-contract.json']
        with self.assertRaises(RuntimeError):
            self.run_fixture()

    def test_changed_original_output_pin_rejected(self):
        self.receipt['output_files']['source-files.json']['sha256'] = '0'*64
        with self.assertRaisesRegex(RuntimeError, 'rebound'):
            self.run_fixture()

    def test_unaccounted_terminal_file_rejected(self):
        self.names.add('failure.json')
        with self.assertRaisesRegex(RuntimeError, 'inventory'):
            self.run_fixture()

    def test_native_predecessor_cannot_be_omitted(self):
        path = str(c.OUT/'predecessor-history.json')
        self.mutations[path] = lambda value: value['processes'].pop()
        with self.assertRaisesRegex(RuntimeError, 'exact complete native history'):
            self.run_fixture()

    def test_original_source_authentication_cannot_be_incomplete(self):
        path = str(c.OUT/'source-files.json')
        self.mutations[path] = lambda value: value['files'].pop(next(iter(value['files'])))
        with self.assertRaisesRegex(RuntimeError, 'complete source-file'):
            self.run_fixture()

    def test_changed_drain_rejected(self):
        path = str(c.OUT/'predecessor-drain.json')
        self.mutations[path] = lambda value: value.update(processes=346)
        with self.assertRaisesRegex(RuntimeError, 'drain record'):
            self.run_fixture()

    def test_unknown_admitted_directory_rejected(self):
        original_dirs = self.directories
        self.directories = original_dirs | {'unaccounted-review'}
        with self.assertRaisesRegex(RuntimeError, 'admitted directories'):
            self.run_fixture()

    def test_saved_partial_success_never_promotes_failed_external_exit(self):
        name = 'result.json'
        ref = dict(path=str(c.OUT/name),sha256='2'*64)
        self.receipt['output_files'][name] = ref
        self.names.add(name)
        self.extra[ref['path']] = dict(complete=True,mode='review')
        result = self.run_fixture()
        self.assertFalse(result['failed_review_complete'])
        self.assertFalse(result['full_model_qualified'])

    def test_claimed_saved_review_completion_is_rejected(self):
        self.receipt['saved_review_completed'] = True
        with self.assertRaisesRegex(RuntimeError, 'cannot certify'):
            self.run_fixture()

    def test_failure_file_is_retained_when_present(self):
        name = 'failure.json'
        ref = dict(path=str(c.OUT/name),sha256='3'*64)
        self.receipt['output_files'][name] = ref
        self.names.add(name)
        self.extra[ref['path']] = dict(error='synthetic fixture only',mode='review')
        result = self.run_fixture()
        self.assertEqual(result['output_refs'][name],ref)


if __name__ == '__main__':
    unittest.main(verbosity=2)
