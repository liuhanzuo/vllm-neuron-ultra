"""Metadata partition tests; payload and native binary opens are guarded."""
import builtins
import copy
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import dory_saved_retry_source_partition_v1 as c
import check_nemotron9b_residual_source_partition_v1 as original_fixtures


class Partition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        original_fixtures.Partition.setUpClass()
        cls.old = original_fixtures.Partition
        cls.observation = c.retry.original.load(c.OBSERVATION)
        cls.base = copy.deepcopy(cls.old.spec)
        cls.base['required_saved_retry_query_observation'] = c.OBSERVATION
        cls.base['source_sha256'].update({p:r['sha256'] for p,r in cls.observation['source_and_binary_pins'].items()})
        cls.base['source_sha256'][c.OBSERVATION['path']] = c.OBSERVATION['sha256']

    def setUp(self):
        self.spec = copy.deepcopy(self.base)

    def classify(self):
        return c.classify_maps(self.spec,self.old.catalog,self.old.parent,self.old.start,
                               self.old.parts,self.observation)

    def test_exact20_historical_and9_query_native_categories_are_disjoint(self):
        parts=self.classify()
        self.assertEqual(len(parts['historical_payload_sha256']),20)
        self.assertEqual(len(parts['query_native_binary_sha256']),9)
        self.assertEqual(len(parts['python_bytes_sha256']),3)
        joined={}
        for values in parts.values():
            self.assertFalse(set(joined)&set(values))
            joined.update(values)
        self.assertEqual(joined,self.spec['source_sha256'])

    def test_query_native_pin_cannot_be_changed(self):
        path=next(iter(self.observation['new_native_file_mappings']))
        self.spec['source_sha256'][path]='0'*64
        with self.assertRaisesRegex(RuntimeError,'All41'):self.classify()

    def test_query_native_pin_cannot_be_omitted(self):
        path=next(iter(self.observation['new_native_file_mappings']))
        self.spec['source_sha256'].pop(path)
        with self.assertRaises(RuntimeError):self.classify()

    def test_unknown_native_library_is_not_accepted_as_text(self):
        for path in ['/unopened/unknown.so','/unopened/libunknown.so.1.2.3']:
            with self.subTest(path=path):
                self.setUp();self.spec['source_sha256'][path]='0'*64
                with self.assertRaises(RuntimeError):self.classify()

    def test_original_historical_binary_pins_cannot_be_changed(self):
        for key in ['historical_payload_sha256','dory_historical_payload_sha256']:
            with self.subTest(key=key):
                self.setUp();path=next(iter(self.spec[key]))
                self.spec['source_sha256'][path]='0'*64
                with self.assertRaises(RuntimeError):self.classify()

    def test_non_utf8_fixture_category_stays_exact(self):
        self.spec['runtime_python_non_utf8_sha256'].popitem()
        with self.assertRaises(RuntimeError):self.classify()

    def test_original_metadata_limits_remain_unchanged(self):
        self.spec['metadata_limits']['ordinary_file_bytes']*=2
        with self.assertRaises(RuntimeError):self.classify()

    def test_observation_anchor_cannot_be_rebound(self):
        self.spec['required_saved_retry_query_observation']=dict(c.OBSERVATION,sha256='0'*64)
        with self.assertRaises(RuntimeError):self.classify()

    def test_only_exact_native_set_can_be_authenticated(self):
        pins=self.classify()['query_native_binary_sha256']
        pins['/unopened/libextra.so']='0'*64
        with patch.object(c,'authenticated_bytes')as opened:
            with self.assertRaisesRegex(RuntimeError,'exact nine'):
                c.authenticate_native_sources(pins,self.observation,lambda:None)
            opened.assert_not_called()

    def test_classification_preserves_every_scientific_field(self):
        before=copy.deepcopy(self.spec)
        self.classify()
        self.assertEqual(self.spec,before)

    def test_public_classifier_opens_only_metadata_never_payload_or_native_files(self):
        native=set(self.observation['new_native_file_mappings'])
        def check(path):
            if isinstance(path,(str,bytes,os.PathLike)):
                p=Path(os.fsdecode(path))
                self.assertNotIn(str(p),native)
                self.assertNotIn(p.suffix,c.previous.original.history.FORBIDDEN)
        old_os_open,old_open,old_path_open=os.open,builtins.open,Path.open
        def os_open(path,*a,**kw):check(path);return old_os_open(path,*a,**kw)
        def op(path,*a,**kw):check(path);return old_open(path,*a,**kw)
        def path_open(path,*a,**kw):check(path);return old_path_open(path,*a,**kw)
        with patch.object(os,'open',side_effect=os_open),patch.object(builtins,'open',side_effect=op),patch.object(Path,'open',path_open):
            actual=c.classify_sources(self.spec,self.old.catalog)
        self.assertEqual(actual,self.classify())


if __name__=='__main__':
    unittest.main(verbosity=2)
