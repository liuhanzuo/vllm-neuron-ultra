"""Stdlib source checks; synthetic BF16 only, no model execution or real captures."""
import copy
import hashlib
import math
from pathlib import Path
import random
import struct
import tempfile
import unittest
from unittest.mock import patch
import review_minicpm5_frame1_raw_boundaries_v1 as r


def packed(words):
    return struct.pack('<' + str(len(words)) + 'H', *words)


class Stream:
    def __init__(self, rows):
        self.rows = rows
        self.length, self.width = len(rows), len(rows[0])
        self.index = 0
    def next_row(self):
        value = self.rows[self.index]
        self.index += 1
        return tuple(value)


class Arithmetic(unittest.TestCase):
    def test_all_bf16_values_against_ieee_float32_decoder(self):
        for word in range(65536):
            f = struct.unpack('<f', struct.pack('<I', word << 16))[0]
            actual = r.VALUES[word]
            if math.isfinite(f):
                self.assertEqual(math.ldexp(float(actual), -133), f)
            else:
                self.assertIsNone(actual)

    def test_known_distances(self):
        m = r.Metric()
        m.add([0x3f80, 0x4000, 0xc000], [0x4000, 0x4000, 0x0000])
        result = m.result()
        self.assertEqual(result['different_elements'], 2)
        self.assertEqual(result['compared_elements'], 3)
        self.assertEqual(result['max_abs'], 2)
        self.assertEqual(result['mean_abs'], 1)
        self.assertAlmostEqual(result['cosine'], 1 / math.sqrt(2))

    def test_signed_zero_is_bit_difference(self):
        m = r.Metric(); m.add([0], [0x8000]); v = m.result()
        self.assertFalse(v['raw_exact'])
        self.assertEqual(v['different_elements'], 1)
        self.assertEqual(v['max_abs'], 0)
        self.assertEqual(v['mean_abs'], 0)
        self.assertIsNone(v['cosine'])

    def test_smallest_subnormal(self):
        m = r.Metric(); m.add([1], [0])
        self.assertEqual(m.result()['max_abs'], 2**-133)

    def test_extreme_finite_values_do_not_overflow(self):
        m = r.Metric(); m.add([0x7f7f, 0xff7f], [0xff7f, 0x7f7f])
        v = m.result()
        self.assertTrue(math.isfinite(v['max_abs']))
        self.assertAlmostEqual(v['cosine'], -1, places=14)

    def test_exact_integer_sum_agrees_with_fsum(self):
        rng = random.Random(51)
        a = [rng.randrange(0x3c00, 0x4100) for _ in range(701)]
        b = [rng.randrange(0x3c00, 0x4100) | 0x8000 for _ in range(701)]
        decode = lambda x: struct.unpack('<f', struct.pack('<I', x << 16))[0]
        af, bf = list(map(decode, a)), list(map(decode, b))
        m = r.Metric(); m.add(a[:301], b[:301]); m.add(a[301:], b[301:])
        v = m.result()
        self.assertEqual(v['mean_abs'], math.fsum(abs(x-y) for x,y in zip(af,bf))/701)
        expected = math.fsum(x*y for x,y in zip(af,bf)) / (
            math.sqrt(math.fsum(x*x for x in af))*math.sqrt(math.fsum(y*y for y in bf)))
        self.assertAlmostEqual(v['cosine'], expected, places=14)

    def test_nan_and_infinity_rejected(self):
        for word in (0x7f80, 0xff80, 0x7fc0):
            with self.subTest(word=word), self.assertRaises(ValueError):
                r.Metric().add([word], [0])

    def test_width_mismatch_rejected(self):
        with self.assertRaises(ValueError):r.Metric().add([0, 0], [0])

    def test_empty_metric_rejected(self):
        with self.assertRaises(ValueError):r.Metric().result()

    def test_raw_counts_have_no_tolerance(self):
        m=r.Metric();m.add([0x3f80],[0x4000]);v=m.result()
        for key,value in (('different_elements',0),('compared_elements',2),('raw_exact',True),
                          ('nonfinite_elements',1),('max_abs',1.000000000001)):
            bad=dict(v);bad[key]=value
            with self.subTest(key=key), self.assertRaises(ValueError):r.compare_metric(v,bad)

    def test_reduction_tolerance_is_explicit_and_bounded(self):
        m=r.Metric();m.add([0x3f80],[0x4000]);v=m.result()
        r.compare_metric(v,dict(v,mean_abs=1+1e-12,cosine=1-1e-12))
        for bad in (dict(v,mean_abs=1.001),dict(v,cosine=.99),dict(v,cosine=None),
                    dict(v,mean_abs=float('nan'))):
            with self.assertRaises(ValueError):r.compare_metric(v,bad)


class Boundaries(unittest.TestCase):
    def test_rank_assembly_is_per_token(self):
        native=Stream([[1,2,3,4],[5,6,7,8]])
        a=Stream([[1,2],[5,6],[11,12]])
        b=Stream([[3,4],[7,8],[13,14]])
        v=r.boundary([native,a,b],False)
        self.assertTrue(v['real_tokens']['raw_exact'])
        self.assertEqual(v['real_tokens']['compared_elements'],8)
        self.assertEqual(v['prediction']['compared_elements'],4)
        self.assertEqual([native.index,a.index,b.index],[2,3,3])

    def test_rank_swap_is_detected_in_metrics(self):
        v=r.boundary([Stream([[1,2,3,4]]),Stream([[3,4]]),Stream([[1,2]])],False)
        self.assertEqual(v['real_tokens']['different_elements'],4)

    def test_prediction_scope_is_last_real_token(self):
        v=r.boundary([Stream([[1,2],[3,4]]),Stream([[9],[3],[8]]),
                      Stream([[2],[4],[7]])],False)
        self.assertFalse(v['real_tokens']['raw_exact'])
        self.assertTrue(v['prediction']['raw_exact'])

    def test_replicated_padding_checked(self):
        with self.assertRaises(ValueError):
            r.boundary([Stream([[1,2]]),Stream([[1,2],[3,4]]),Stream([[1,2],[3,5]])],True)

    def test_replicated_boundaries_use_single_copy(self):
        v=r.boundary([Stream([[1,2]]),Stream([[1,2],[3,4]]),Stream([[1,2],[3,4]])],True)
        self.assertTrue(v['real_tokens']['raw_exact'])
        self.assertEqual(v['real_tokens']['compared_elements'],2)

    def test_geometry_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            r.boundary([Stream([[1,2]]),Stream([[1]]),Stream([[1]])],True)

class Files(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.directory=Path(self.tmp.name)
        self.path=self.directory/'input.bf16'
        self.path.write_bytes(packed([1,2,3,4,5,6]))
        digest=hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.row=dict(name='input',path=str(self.path),encoding='packed_bf16_le',
                      observed_not_recomputed=True,bytes=12,sha256=digest,
                      identity=r.identity(self.path,self.path.stat()),
                      metadata=dict(shape=[1,3,2],stride=[6,2,1],offset=0,dtype='torch.bfloat16',sha256=digest))
    def tearDown(self):
        self.tmp.cleanup()
    def read_all(self,row=None):
        with r.Payload(row or self.row,self.directory) as p:
            result=[p.next_row() for _ in range(3)]
            return result,p.finish()

    def test_full_capture_hash_and_rows(self):
        rows,proof=self.read_all()
        self.assertEqual(rows,[(1,2),(3,4),(5,6)])
        self.assertEqual(proof['bytes'],12)

    def test_feature_major_layout_still_has_logically_packed_rows(self):
        self.row['metadata']['stride']=[6,1,3]
        self.assertEqual(self.read_all()[0],[(1,2),(3,4),(5,6)])

    def test_digest_tampering_rejected(self):
        self.row['sha256']=self.row['metadata']['sha256']='0'*64
        with self.assertRaises(ValueError):self.read_all()

    def test_nonfinite_in_padding_rejected(self):
        self.path.write_bytes(packed([1,2,3,4,0x7f80,6]))
        self.row['identity']=r.identity(self.path,self.path.stat())
        self.row['sha256']=self.row['metadata']['sha256']=hashlib.sha256(self.path.read_bytes()).hexdigest()
        with self.assertRaises(ValueError):self.read_all()

    def test_original_identity_required(self):
        self.row['identity']['inode']+=1
        with self.assertRaises(ValueError):self.read_all()

    def test_changed_file_after_open_rejected(self):
        with r.Payload(self.row,self.directory) as p:
            for _ in range(3):p.next_row()
            self.path.write_bytes(packed([6,5,4,3,2,1]))
            with self.assertRaises(ValueError):p.finish()

    def test_truncation_after_open_rejected(self):
        with r.Payload(self.row,self.directory) as p:
            self.path.write_bytes(b'')
            with self.assertRaises(ValueError):p.next_row()

    def test_unread_tail_rejected(self):
        with r.Payload(self.row,self.directory) as p:
            p.next_row()
            with self.assertRaises(ValueError):p.finish()

    def test_wrong_field_path_rejected(self):
        self.row['name']='output'
        with self.assertRaises(ValueError):self.read_all()

    def test_symlink_rejected(self):
        target=self.directory/'other.bf16'
        self.path.rename(target);self.path.symlink_to(target)
        with self.assertRaises(ValueError):self.read_all()

    def test_false_encoding_rejected(self):
        self.row['encoding']='physical_storage'
        with self.assertRaises(ValueError):self.read_all()

    def test_overlapping_layout_rejected(self):
        self.row['metadata']['stride']=[6,1,1]
        with self.assertRaises(ValueError):self.read_all()

    def test_oversized_declared_payload_rejected_before_open(self):
        self.row['metadata']['shape']=[1,2048,6144]
        self.row['metadata']['stride']=[12582912,6144,1]
        self.row['bytes']=self.row['identity']['size']=25165824
        with patch.object(r.os,'open',side_effect=AssertionError('must not open')):
            with self.assertRaises(ValueError):self.read_all()



class CommandBindings(unittest.TestCase):
    def execute(self, change=None):
        import io
        import json
        import types
        from contextlib import redirect_stdout
        root=r.meta.R/'minicpm5-frame1-mlp-r999-20261108-v99'
        def ref(path):return dict(path=str(path),sha256='a'*64)
        spec_ref=ref(r.meta.R/'research/synthetic-raw-binding-spec.json')
        result_ref=ref(root/'result.json')
        replay_ref=ref(root/'replay.json')
        review_ref=ref(r.meta.R/'research/synthetic-raw-binding-review.json')
        spec=dict(schema='minicpm5_frame1_mlp_spec_v1',base_plan=r.meta.BASE,output_root=str(root))
        replay=dict(captures={role:ref(root/role/'capture.json') for role in r.ROLES})
        result=dict(report=replay_ref)
        checks=dict(metadata_consistent=True)
        docs={spec_ref['path']:spec,result_ref['path']:result,replay_ref['path']:replay,r.meta.BASE['path']:{}}
        refs=[spec_ref,result_ref,replay_ref,r.meta.BASE]
        for role,reference in replay['captures'].items():docs[reference['path']]={};refs.append(reference)
        review=dict(schema='minicpm5_frame1_independent_metadata_review_v1',reviewer='/root',
                    source_sha256=r.METADATA_SHA,spec=spec_ref,result=result_ref,report=replay_ref,
                    checks=checks,metadata_references={x['path']:dict(x,bytes=7) for x in refs})
        docs[review_ref['path']]=review
        if change:change(review,spec,result,replay)
        class Reader:
            def __init__(self):self.references={}
            def load(self,reference):
                self.references[reference['path']]=dict(reference,bytes=7)
                return copy.deepcopy(docs[reference['path']])
        dest=r.meta.R/'research/synthetic-raw-binding-output.json'
        output=io.StringIO()
        class Sink:
            def __enter__(self):return output
            def __exit__(self,*unused):pass
        original_open=Path.open
        def open_file(path,mode='r',*args,**kwargs):
            if path==dest:
                self.assertEqual(mode,'x');return Sink()
            return original_open(path,mode,*args,**kwargs)
        args=types.SimpleNamespace(metadata_review=review_ref['path'],metadata_review_sha256=review_ref['sha256'],output=str(dest))
        raw_checks=dict(synthetic_only=True,actual_exit_verified=False,physical_drain_verified=False,
                        full_model_qualified=False,new_speedup_claim=False)
        with patch.object(r.meta,'Reader',Reader),patch.object(r.meta,'audit',return_value=checks),\
             patch.object(r,'audit_raw',return_value=raw_checks) as raw,patch.object(Path,'open',open_file),redirect_stdout(io.StringIO()):
            r.main(args)
        return json.loads(output.getvalue()),raw.call_count

    def test_cli_bindings_synthetic_only(self):
        result,count=self.execute()
        self.assertEqual(count,1)
        self.assertTrue(result['checks']['synthetic_only'])
        self.assertFalse(result['checks']['full_model_qualified'])
    def test_reject_review_source_change(self):
        with self.assertRaises(ValueError):self.execute(lambda v,s,o,p:v.update(source_sha256='b'*64))
    def test_reject_plan_change(self):
        with self.assertRaises(ValueError):self.execute(lambda v,s,o,p:s.update(base_plan=dict(r.meta.BASE,sha256='b'*64)))
    def test_reject_result_path_change(self):
        with self.assertRaises(ValueError):self.execute(lambda v,s,o,p:v['result'].update(path=str(r.meta.R/'other.json')))
    def test_reject_replay_link_change(self):
        with self.assertRaises(ValueError):self.execute(lambda v,s,o,p:o.update(report=dict(v['report'],sha256='b'*64)))
    def test_reject_capture_role_path_change(self):
        with self.assertRaises(ValueError):self.execute(lambda v,s,o,p:p['captures']['native'].update(path=str(r.meta.R/'other.json')))
    def test_reject_unbound_metadata_digest(self):
        def change(v,s,o,p):v['metadata_references'][v['spec']['path']]['sha256']='b'*64
        with self.assertRaises(ValueError):self.execute(change)
    def test_reject_forged_metadata_conclusion(self):
        with self.assertRaises(ValueError):self.execute(lambda v,s,o,p:v.update(checks=dict(metadata_consistent=False)))

if __name__=='__main__':
    unittest.main(verbosity=2)
