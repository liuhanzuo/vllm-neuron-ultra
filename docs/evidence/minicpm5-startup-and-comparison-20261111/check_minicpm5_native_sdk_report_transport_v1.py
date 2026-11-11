"""Once-only full synthetic comparison transport plus finite codec corruption controls."""
import ast
import base64
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest
import zlib
ROOT=Path(__file__).resolve().parents[1];Q=ROOT/'research';sys.path[:0]=[str(Q),str(ROOT)]
from research import minicpm5_native_sdk_report_transport_v1 as t
from research.check_compare_minicpm5_layer1_native_sdk_v3 import fixture,c


class Transport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        *args,mem=fixture();cls.report=c.compare(*args,mem.read,lambda:None)
        cls.encoded=t.encode_report(cls.report);cls.envelope=json.loads(cls.encoded)
        cls.reads=len(mem.reads)
    def test_original_pure_codec_function_definitions_are_ast_identical(self):
        candidate=ast.parse(Path(t.__file__).read_text())
        for function,file in [('packed_physical','audit_minicpm5_layer1_operator_sdk_result_v3.py'),
                              ('unpack','record_minicpm5_layer1_sdk_batch_layout_audit_exit_v5.py')]:
            old=ast.parse((Q/file).read_text())
            a=next(x for x in old.body if isinstance(x,ast.FunctionDef)and x.name==function)
            b=next(x for x in candidate.body if isinstance(x,ast.FunctionDef)and x.name==function)
            self.assertEqual(ast.dump(a,include_attributes=False),ast.dump(b,include_attributes=False))
        self.assertNotIn('research.audit_minicpm5_layer1_operator_sdk_result_v3',sys.modules)
        self.assertNotIn('research.record_minicpm5_layer1_sdk_batch_layout_audit_exit_v5',sys.modules)
    def test_full_report_roundtrip_all1632_sources_boundaries_unknowns_and_complete_hash(self):
        restored=t.decode_report(self.encoded);self.assertEqual(restored,self.report)
        raw=t.canonical(self.report);self.assertEqual(hashlib.sha256(raw).hexdigest(),self.envelope['sha256'])
        self.assertEqual(len(raw),self.envelope['uncompressed_bytes']);self.assertEqual(self.reads,1632)
        self.assertEqual(restored['source_files'],self.report['source_files']);self.assertEqual(len(restored['source_files']),1632)
        for original,decoded in zip(self.report['comparisons'],restored['comparisons']):
            for key in ('boundaries','unknown_boundaries','sdk_padding','comparison_coverage',
                        'authenticated_sdk_leaves_not_individually_cross_compared','observed_scale_operations','alignment'):
                self.assertEqual(original[key],decoded[key])
        self.assertLess(len(self.encoded),2*2**20);self.assertFalse(restored['new_speedup_claim'])
        print(json.dumps(dict(synthetic_only=True,full_comparison_generations=1,complete_reads=self.reads,
            uncompressed_canonical_bytes=len(raw),uncompressed_sha256=self.envelope['sha256'],
            compressed_bytes=self.envelope['compressed_bytes'],base64_bytes=len(self.envelope['data']),
            stored_indent2_bytes_including_newline=len(self.encoded),original_file_cap_bytes=2*2**20,
            file_cap_remaining_bytes=2*2**20-len(self.encoded),uncompressed_cap_remaining_bytes=32*2**20-len(raw),
            encoded_sha256=hashlib.sha256(self.encoded).hexdigest(),full_decoded_json_equal=True)),flush=True)
        self.assertFalse({'torch','numpy','transformers','torch_neuronx','neuronxcc'}&set(sys.modules))
    def test_truncated_extra_zlib_stream_and_trailing_bytes_rejected(self):
        # Exercise unchanged generic decoder on small objects; full generation is never repeated.
        envelope=t.packed_physical({'complete':'small synthetic'});data=base64.b64decode(envelope['data'])
        for altered in (data[:-1],data+zlib.compress(b'{}'),data+b'junk'):
            bad=dict(envelope,data=base64.b64encode(altered).decode(),compressed_bytes=len(altered))
            with self.subTest(size=len(altered)),self.assertRaises(RuntimeError):t.unpack(bad)
    def test_declared_lengths_sha_base64_and_boolean_lengths_rejected(self):
        e=t.packed_physical({'a':[1,2,3]})
        for key,value in [('uncompressed_bytes',e['uncompressed_bytes']-1),('uncompressed_bytes',e['uncompressed_bytes']+1),
            ('compressed_bytes',e['compressed_bytes']-1),('sha256','0'*64),('data',e['data']+'!'),
            ('uncompressed_bytes',True),('compressed_bytes',True),('all_samples_retained',False)]:
            with self.subTest(key=key,value=value),self.assertRaises((RuntimeError,ValueError)):
                t.unpack(dict(e,**{key:value}))
    def test_original_uncompressed_encoded_and_complete_file_limits(self):
        e=t.packed_physical({'a':1})
        for changes in ({'uncompressed_bytes':32*2**20+1},{'compressed_bytes':2*2**20},{'data':'a'*(2*2**20)}):
            with self.subTest(changes=list(changes)),self.assertRaises(RuntimeError):t.unpack(dict(e,**changes))
        with self.assertRaises(RuntimeError):t.decode_report(b' '*(2*2**20))
    def test_missing_source_unknown_or_coverage_cannot_be_packed(self):
        for fault in ('source','unknown','coverage'):
            report=dict(self.report)
            if fault=='source':report['source_files']=report['source_files'][:-1]
            else:
                report['comparisons']=list(report['comparisons']);row=dict(report['comparisons'][0]);report['comparisons'][0]=row
                row['unknown_boundaries'if fault=='unknown'else'comparison_coverage']=[]
            with self.subTest(fault=fault),self.assertRaises(RuntimeError):t.pack_report(report)
    def test_duplicate_envelope_and_noncanonical_uncompressed_report_rejected(self):
        with self.assertRaises(RuntimeError):t.decode_report(b'{"encoding":"x","encoding":"y"}')
        # Same report JSON, deliberately different canonical bytes: a wrapper cannot silently normalize it.
        raw=b' '+t.canonical(self.report);compressed=zlib.compress(raw,9)
        bad=dict(self.envelope,uncompressed_bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
            compressed_bytes=len(compressed),data=base64.b64encode(compressed).decode())
        with self.assertRaisesRegex(RuntimeError,'canonical'):t.unpack_report(bad)


if __name__=='__main__':unittest.main(verbosity=2)
