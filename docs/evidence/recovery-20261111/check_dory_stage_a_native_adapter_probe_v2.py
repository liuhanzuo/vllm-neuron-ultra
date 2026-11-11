"""Source-only checks for future native ABI probe; FakeTorch is not native proof."""
import ast
import hashlib
import struct
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import dory_stage_a_native_adapter_probe_v2 as probe
from test_dory_stage_a_tensor_adapter_v1 import FakeTorch, FakeTensor, WIDTH


class ProbeAPI(FakeTorch):
    __version__='2.14.0+cpu'
    def frombuffer(self,buffer,*,dtype):
        return FakeTensor(self,[len(buffer)//WIDTH[dtype]],dtype,bytes(buffer))
    def as_strided(self,tensor,*,size,stride,storage_offset):
        return FakeTensor(self,size,tensor.dtype,storage=tensor.storage,stride=stride,offset=storage_offset)


class ProbeChecks(unittest.TestCase):
    def test_geometry_covers_singletons_broadcast_and_full_vocab(self):
        rows=probe.cases();self.assertEqual(len(rows),23)
        names={x['name']for x in rows}
        for name in ('singleton_leading','singleton_trailing','singleton_middle','all_singletons',
                     'broadcast_rows','chunk_boundary','recursive_columns','selected_rows'):
            self.assertTrue({name+'_BF16',name+'_F32'}<=names)
        self.assertIn('full_vocab_f32',names)

    def test_every_oracle_byte_matches_exact_storage_indices(self):
        case=dict(shape=[2,3],stride=[7,2],offset=1,rows=None,dtype='BF16')
        raw=bytearray(struct.pack('<13H',*range(13)))
        self.assertEqual(probe.expected_bytes(case,raw),struct.pack('<6H',1,3,5,8,10,12))
        case['rows']=[1]
        self.assertEqual(probe.expected_bytes(case,raw),struct.pack('<3H',8,10,12))

    def test_exact_nonfinite_and_signed_zero_patterns(self):
        case=dict(shape=[8],stride=[1],offset=0,rows=None,dtype='BF16')
        self.assertEqual(struct.unpack('<8H',probe.backing(case)),
                         (0,0x8000,1,0x3f80,0xbf80,0x7f7f,0x7f80,0x7fc1))
        case['dtype']='F32'
        self.assertEqual(struct.unpack('<8I',probe.backing(case)),
                         (0,0x80000000,1,0x3f800000,0xbf800000,0x7f7fffff,0x7f800000,0x7fc12345))

    def test_bound_rejection_precedes_buffer_allocation(self):
        with self.assertRaises(RuntimeError):probe.backing(dict(shape=[2**30],stride=[1],offset=0,rows=None,dtype='F32'))
        with self.assertRaises(RuntimeError):probe.backing(dict(shape=[1],stride=[1],offset=2**30,rows=None,dtype='F32'))

    def test_all_probe_cases_and_auxiliaries_with_fake_api(self):
        result=probe.run_probe(ProbeAPI(),guard=lambda:None)
        self.assertTrue(result['complete']);self.assertTrue(result['passed'])
        self.assertEqual(result['case_count'],23);self.assertEqual(len(result['auxiliary_cases']),3)
        self.assertTrue(all(x['full_bytes_compared']for x in result['serialization_cases']))
        self.assertTrue(all(x['largest_chunk_bytes']<=65536 for x in result['serialization_cases']))
        for name in ('native_source_provenance_authenticated_here','actual_process_exit_observed',
                     'runtime_admitted_here','production_singleton_layout_qualified','full_model_qualified','new_speedup_claim'):
            self.assertIs(result[name],False)

    def test_singleton_preserved_stride_is_observed_and_fails(self):
        clone=FakeTensor.clone
        def preserve(tensor,memory_format):
            result=clone(tensor,memory_format)
            if tensor.shape==(1,7) and tensor.stride()==(0,1):result._stride=(0,1)
            return result
        with patch.object(FakeTensor,'clone',preserve):
            result=probe.run_probe(ProbeAPI(),guard=lambda:None)
        self.assertFalse(result['passed'])
        failed=[x for x in result['serialization_cases']if x['case']['name'].startswith('singleton_leading')]
        self.assertEqual(len(failed),2)
        for row in failed:
            self.assertFalse(row['passed']);self.assertEqual(row['actual_small_clone_layout']['stride'],[0,1])
            self.assertIn('Independent exact native chunk',row['error'])
        self.assertEqual(result['case_count'],23)

    def test_bad_byte_output_recorded_without_promotion(self):
        api=ProbeAPI();api.byte_list_mutator=lambda x:[256]+x[1:]
        result=probe.run_probe(api,guard=lambda:None)
        self.assertFalse(result['passed'])
        self.assertTrue(all(not x['passed']for x in result['serialization_cases']))

    def test_guard_failure_aborts_instead_of_becoming_case_failure(self):
        calls=[0]
        def guard():
            calls[0]+=1
            if calls[0]==9:raise RuntimeError('serial guard expired')
        with self.assertRaises(BaseException)as raised:probe.run_probe(ProbeAPI(),guard=guard)
        self.assertEqual(calls[0],9);self.assertEqual(str(raised.exception.__cause__),'serial guard expired')

    def test_wrong_version_rejected_before_allocations(self):
        api=ProbeAPI();api.__version__='2.9.1'
        with self.assertRaises(RuntimeError):probe.run_probe(api,guard=lambda:None)
        self.assertEqual(api.operations,[])

    def test_padded_selected_rows_oracle_is_not_full_backing(self):
        case=next(x for x in probe.cases()if x['name']=='selected_rows_F32')
        raw=probe.backing(case);expected=probe.expected_bytes(case,raw)
        self.assertEqual(len(expected),3*9*4);self.assertLess(len(expected),len(raw))

    def test_repeated_first_chunk_cannot_pass_full_vocab(self):
        original=probe.NativeTensorAdapter.iter_bytes
        def corrupted(adapter,tensor,rows=None):
            if list(tensor.shape)==[1,131072]:
                first=next(original(adapter,tensor,rows))
                for _ in range(8):yield first
            else:yield from original(adapter,tensor,rows)
        with patch.object(probe.NativeTensorAdapter,'iter_bytes',corrupted):
            result=probe.run_probe(ProbeAPI(),guard=lambda:None)
        self.assertFalse(result['passed'])
        row=next(x for x in result['serialization_cases']if x['case']['name']=='full_vocab_f32')
        self.assertFalse(row['passed']);self.assertIn('independent storage oracle',row['error'])

    def test_selection_mutating_shared_backing_rejected(self):
        class SharedAPI(ProbeAPI):
            def frombuffer(self,buffer,*,dtype):
                tensor=FakeTensor(self,[len(buffer)//WIDTH[dtype]],dtype,bytes(buffer))
                tensor.storage=(buffer,tensor.storage[1]);return tensor
            def index_select(self,tensor,dim,index):
                tensor.storage[0][:]=bytes(len(tensor.storage[0]))
                return super().index_select(tensor,dim,index)
        result=probe.run_probe(SharedAPI(),guard=lambda:None)
        self.assertFalse(result['passed'])
        row=next(x for x in result['auxiliary_cases']if x['name']=='full_row_select')
        self.assertFalse(row['passed'])

    def test_long_blocks_are_position_dependent(self):
        for dtype in ('BF16','F32'):
            case=dict(shape=[65536],stride=[1],offset=0,rows=None,dtype=dtype)
            raw=probe.backing(case)
            self.assertNotEqual(raw[:65536],raw[65536:131072])

    def test_probe_source_has_no_native_import_file_io_or_launch(self):
        tree=ast.parse(Path(probe.__file__).read_text())
        names=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):names.extend(x.name for x in node.names)
            if isinstance(node,ast.ImportFrom):names.append(node.module)
            if isinstance(node,ast.Call)and isinstance(node.func,ast.Name):
                self.assertNotIn(node.func.id,('open','exec','eval','__import__'))
        self.assertFalse(any(x.split('.')[0]in ('torch','numpy','subprocess','ctypes','vllm')for x in names))
        self.assertNotIn('torch',sys.modules);self.assertNotIn('numpy',sys.modules)


if __name__=='__main__':unittest.main(verbosity=2)
