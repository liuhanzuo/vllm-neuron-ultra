"""Actual metadata classification and bounded tokenizer reads; no tensor IO."""
import ast
import builtins
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

Q=Path(__file__).resolve().parent
R=Q.parent
sys.path[:0]=[str(R),str(Q)]
from research import nemotron9b_residual_source_partition_v1 as c
import nemotron9b_package_metadata_v20 as new
import nemotron9b_package_protocol_v18 as parent_protocol
old=c.original


class Partition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent=old.load(c.PARENT)
        cls.catalog=old.load(cls.parent['final_predecessor_catalog'])
        cls.parts=old.history.partition(cls.catalog)
        reader=c.residual.Reader()
        cls.start=c.residual.anchor(reader)
        cls.spec=copy.deepcopy(cls.parent)
        pins=cls.spec['source_sha256']
        c.residual.merge(pins,cls.start['spec']['source_sha256'])
        c.residual.merge(pins,reader.pins)
        c.residual.merge(pins,{c.PARENT['path']:c.PARENT['sha256']})
        cls.spec.update(required_residual_source_anchor=c.residual.PRODUCER,
                        dory_historical_payload_sha256=c.exact_dory_payloads(cls.start))

    def setUp(self):
        self.candidate=copy.deepcopy(self.spec)

    def classify(self):
        return c.classify_maps(self.candidate,self.catalog,self.parent,self.start,self.parts)

    def test_real_14_and6_are_preserved_disjoint_and_not_text(self):
        parts=self.classify()
        self.assertEqual(len(parts['historical_payload_sha256']),20)
        self.assertEqual(len(self.candidate['historical_payload_sha256']),14)
        self.assertEqual(len(self.candidate['dory_historical_payload_sha256']),6)
        self.assertFalse(set(parts['text_sha256'])&set(parts['historical_payload_sha256']))
        self.assertEqual(dict(parts['text_sha256'],**parts['historical_payload_sha256'],**parts['python_bytes_sha256']),
                         self.candidate['source_sha256'])
        self.assertEqual(self.spec['source_sha256'],self.candidate['source_sha256'])

    def test_original_classifier_reproduces_exact14_rejection(self):
        with self.assertRaisesRegex(RuntimeError,'exact14'):
            old.classify_sources(self.candidate,self.catalog)

    def test_original_source_or_binary_pin_cannot_be_removed_or_changed(self):
        keys=[next(iter(self.parent['source_sha256'])),next(iter(self.parent['historical_payload_sha256']))]
        for key in keys:
            for remove in (False,True):
                with self.subTest(key=key,remove=remove):
                    self.setUp()
                    if remove:del self.candidate['source_sha256'][key]
                    else:self.candidate['source_sha256'][key]='0'*64
                    with self.assertRaises(RuntimeError):self.classify()

    def test_dory_binary_pin_cannot_be_removed_or_changed(self):
        key=next(iter(self.candidate['dory_historical_payload_sha256']))
        for field in ('source_sha256','dory_historical_payload_sha256'):
            for remove in (False,True):
                with self.subTest(field=field,remove=remove):
                    self.setUp()
                    if remove:del self.candidate[field][key]
                    else:self.candidate[field][key]='0'*64
                    with self.assertRaises(RuntimeError):self.classify()

    def test_unknown_binary_suffixes_refused(self):
        for suffix in old.history.FORBIDDEN:
            with self.subTest(suffix=suffix):
                self.setUp();self.candidate['source_sha256']['/unopened/unknown'+suffix]='0'*64
                with self.assertRaisesRegex(RuntimeError,'All20'):self.classify()

    def test_fixed_source_anchor_cannot_be_changed(self):
        self.candidate['required_residual_source_anchor']=dict(c.residual.PRODUCER,sha256='0'*64)
        with self.assertRaises(RuntimeError):self.classify()

    def test_actual_admission_and_source_authentication_pins_required(self):
        for ref in(c.PARENT,c.residual.PRODUCER,c.residual.ADMISSION,c.residual.HISTORY,c.residual.CHILD,c.residual.SOURCE_FILES):
            with self.subTest(path=ref['path']):
                self.setUp();self.candidate['source_sha256'].pop(ref['path'])
                with self.assertRaises(RuntimeError):self.classify()

    def test_binary_count_or_admitted_size_ledger_cannot_change(self):
        start=copy.deepcopy(self.start)
        key=next(iter(self.candidate['dory_historical_payload_sha256']))
        start['sources']['files'][key]['bytes']-=1
        with self.assertRaises(RuntimeError):c.exact_dory_payloads(start)

    def test_all3_original_non_utf8_sources_retained(self):
        self.assertEqual(self.classify()['python_bytes_sha256'],old.NON_UTF8)
        for field in('runtime_python_non_utf8_sha256','runtime_python_source_sha256'):
            with self.subTest(field=field):
                self.setUp();self.candidate[field].pop(next(iter(old.NON_UTF8)))
                with self.assertRaises(RuntimeError):self.classify()

    def test_original_metadata_limits_are_not_increased(self):
        self.candidate['metadata_limits']['ordinary_file_bytes']*=2
        with self.assertRaisesRegex(RuntimeError,'8/32'):self.classify()

    def test_original_parent_scientific_fields_are_unchanged_by_classification(self):
        before=copy.deepcopy(self.candidate)
        self.classify()
        self.assertEqual(self.candidate,before)
        for name in('stages','resources','disk_budget','payloads','references','package_commit','automatic_successor_launch'):
            self.assertEqual(self.candidate[name],self.parent[name])

    def test_real_public_classifier_never_opens_binary_payloads(self):
        original_open=os.open
        original_read=Path.read_bytes
        original_builtin=builtins.open
        original_path_open=Path.open
        def check(path):
            if isinstance(path,(str,bytes,os.PathLike)):
                suffix=Path(os.fsdecode(path)).suffix
                self.assertNotIn(suffix,old.history.FORBIDDEN,'Unexpected binary payload read')
        def os_open(path,*a,**kw):check(path);return original_open(path,*a,**kw)
        def read(path,*a,**kw):check(path);return original_read(path,*a,**kw)
        def op(path,*a,**kw):check(path);return original_builtin(path,*a,**kw)
        def path_open(path,*a,**kw):check(path);return original_path_open(path,*a,**kw)
        with patch.object(os,'open',side_effect=os_open),patch.object(Path,'read_bytes',read),\
             patch.object(builtins,'open',side_effect=op),patch.object(Path,'open',path_open):
            actual=new.classify_sources(self.candidate,self.catalog)
        self.assertEqual(actual,self.classify())


class Tokenizer(unittest.TestCase):
    def test_old_ordinary_limit_rejects_exact_large_tokenizer(self):
        with self.assertRaisesRegex(RuntimeError,'bounded metadata'):
            old.text(c.TOKENIZER['path'],c.TOKENIZER['sha256'])

    def test_actual_exact_tokenizer_accepted_and_parsed(self):
        raw=new.text(c.TOKENIZER['path'],c.TOKENIZER['sha256'])
        self.assertEqual(len(raw),17077484)
        self.assertEqual(hashlib.sha256(raw).hexdigest(),c.TOKENIZER['sha256'])
        self.assertEqual(json.loads(raw)['model']['type'],'BPE')

    def test_unauthenticated_or_wrong_tokenizer_digest_rejected_before_open(self):
        for digest in(None,'0'*64):
            with self.subTest(digest=digest),patch.object(old,'stable_bytes')as read:
                with self.assertRaisesRegex(RuntimeError,'fixed Dory'):new.text(c.TOKENIZER['path'],digest)
                read.assert_not_called()

    def test_same_basename_elsewhere_has_no_exception(self):
        self.assertFalse(c.accepts('/untrusted/tokenizer.json'))
        with patch.object(old,'stable_bytes')as read:
            with self.assertRaisesRegex(RuntimeError,'fixed Dory'):
                c.text('/untrusted/tokenizer.json',c.TOKENIZER['sha256'])
            read.assert_not_called()

    def test_corrupted_exact_tokenizer_bytes_rejected(self):
        with patch.object(old,'stable_bytes',return_value=b'corrupt'):
            with self.assertRaisesRegex(RuntimeError,'Exact original'):c.text(c.TOKENIZER['path'],c.TOKENIZER['sha256'])

    def test_every_other_text_reader_function_and_limits_unchanged(self):
        self.assertEqual(new.LIMITS,old.LIMITS)
        self.assertEqual(new.TEXT_SUFFIXES,old.TEXT_SUFFIXES)
        for schema,shape in old.ENVELOPE_SCHEMAS.items():self.assertEqual(new.ENVELOPE_SCHEMAS[schema],shape)
        self.assertEqual(set(new.ENVELOPE_SCHEMAS)-set(old.ENVELOPE_SCHEMAS),
            {'nemotron9b_package_runtime_v20','nemotron9b_package_runtime_source_review_v20'})
        def functions(text):
            return{node.name:ast.dump(node)for node in ast.parse(text).body
                   if isinstance(node,ast.FunctionDef)and node.name not in('text','classify_sources')}
        self.assertEqual(functions(Path(old.__file__).read_text()),functions(Path(new.__file__).read_text()))

    def test_no_native_library_import_or_runtime_spec(self):
        self.assertFalse(any(n.split('.')[0]in{'torch','numpy','vllm','neuronxcc','torch_neuronx'}for n in sys.modules))
        self.assertFalse(list(Q.glob('nemotron9b-package-runtime-v20-*-spec.json')))


if __name__=='__main__':unittest.main(verbosity=2)
