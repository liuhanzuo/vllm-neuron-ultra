"""Stdlib-only interface tests; no native tensors, model imports or old payloads.

The FakeTorch below models byte storage, logical indexing and strided views to
exercise adapter contracts. It does not prove actual Torch ABI, oneDNN behavior,
model correctness, performance, native NaN payload preservation or admission.
"""
import ast
from copy import deepcopy
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dory_stage_a_tensor_adapter_v1 import NativeTensorAdapter
from dory_stage_a_archive_core_v1 import ArchiveWriter
from dory_stage_a_saved_core_v1 import verify_archive


WIDTH = dict(bfloat16=2, float32=4, int64=8, uint8=1)


def strides(shape):
    result, n = [], 1
    for size in reversed(shape):
        result.insert(0, n)
        n *= size
    return tuple(result)


class FakeTensor:
    next_pointer = 65536

    def __init__(self, api, shape, dtype, raw=None, *, storage=None, stride=None, offset=0, flags=None):
        self.api, self.shape, self.dtype = api, tuple(shape), dtype
        self._stride = strides(shape) if stride is None else tuple(stride)
        self.offset = offset
        self.device, self.layout, self.is_quantized = SimpleNamespace(type='cpu'), 'strided', False
        self.conjugate = self.negative = False
        self.flags = {} if flags is None else flags
        if storage is None:
            FakeTensor.next_pointer += 10000000
            self.storage = (bytes(raw), FakeTensor.next_pointer)
        else:
            self.storage = storage

    def stride(self):
        return self._stride

    def storage_offset(self):
        return self.offset

    def element_size(self):
        return WIDTH[self.dtype]

    def numel(self):
        return math.prod(self.shape)

    def data_ptr(self):
        return self.storage[1]+self.offset*self.element_size()

    def is_conj(self):
        return self.conjugate

    def is_neg(self):
        return self.negative

    def _offsets(self):
        for coords in itertools.product(*(range(s) for s in self.shape)):
            yield self.offset+sum(i*s for i, s in zip(coords, self._stride))

    def raw_logical(self):
        word = self.element_size()
        return b''.join(self.storage[0][i*word:(i+1)*word] for i in self._offsets())

    def narrow(self, dim, start, length):
        self.api.operations.append(('narrow', tuple(self.shape), dim, start, length))
        if not (0 <= dim < len(self.shape) and 0 <= start and length > 0 and start+length <= self.shape[dim]):
            raise ValueError('invalid synthetic narrow')
        shape = list(self.shape)
        shape[dim] = length
        result = FakeTensor(self.api, shape, self.dtype, storage=self.storage, stride=self._stride,
                            offset=self.offset+start*self._stride[dim], flags=self.flags)
        if self.api.bad_narrow:
            result.shape = tuple([max(1, shape[0]-1), *shape[1:]])
        return result

    def select(self, dim, index):
        self.api.operations.append(('view_select', tuple(self.shape), dim, index))
        if not (0 <= index < self.shape[dim]):
            raise ValueError('invalid synthetic view select')
        return FakeTensor(self.api, self.shape[:dim]+self.shape[dim+1:], self.dtype, storage=self.storage,
                          stride=self._stride[:dim]+self._stride[dim+1:],
                          offset=self.offset+index*self._stride[dim], flags=self.flags)

    def detach(self):
        self.api.operations.append(('detach', tuple(self.shape)))
        return self

    def clone(self, memory_format):
        assert memory_format == 'contiguous_format'
        count = self.numel()*self.element_size()
        self.api.copy_sizes.append(count)
        self.api.operations.append(('clone', tuple(self.shape), count))
        if self.api.fail_clone:
            raise ValueError('synthetic native clone failure')
        if self.api.alias_clone:
            return self
        out = FakeTensor(self.api, self.shape, self.dtype, self.raw_logical(), flags=self.flags)
        if self.api.change_clone_dtype:
            out.dtype = 'float32'
        return out

    def contiguous(self):
        self.api.operations.append(('contiguous', tuple(self.shape)))
        return self if self._stride == strides(self.shape) else self.clone('contiguous_format')

    def view(self, dtype):
        assert dtype == 'uint8' and self._stride == strides(self.shape)
        self.api.operations.append(('view_dtype', self.dtype, dtype, tuple(self.shape)))
        shape = (*self.shape[:-1], self.shape[-1]*self.element_size())
        return FakeTensor(self.api, shape, dtype, storage=self.storage, stride=strides(shape),
                          offset=self.offset*self.element_size(), flags=self.flags)

    def reshape(self, size):
        assert size == -1 and self._stride == strides(self.shape)
        self.api.operations.append(('reshape', tuple(self.shape), size))
        return FakeTensor(self.api, [self.numel()], self.dtype, storage=self.storage,
                          stride=(1,), offset=self.offset, flags=self.flags)

    def tolist(self):
        self.api.operations.append(('tolist', self.dtype, tuple(self.shape)))
        # Fail if adapter accidentally converts BF16/F32 values through Python float.
        if self.dtype not in ('uint8', 'int64') or len(self.shape) != 1:
            raise AssertionError('floating value conversion or nonvector tolist forbidden')
        if self.dtype == 'uint8':
            result = list(self.raw_logical())
            if self.api.byte_list_mutator:
                result = self.api.byte_list_mutator(result)
            return result
        return list(struct.unpack('<'+'q'*self.shape[0], self.raw_logical()))


class FakeTorch:
    Tensor = FakeTensor
    bfloat16, float32, int64, uint8 = 'bfloat16', 'float32', 'int64', 'uint8'
    strided, contiguous_format = 'strided', 'contiguous_format'

    def __init__(self):
        self.operations, self.copy_sizes = [], []
        self.bad_narrow = self.fail_clone = self.alias_clone = self.change_clone_dtype = False
        self.byte_list_mutator = None
        self.bad_selected_width = False

    def tensor(self, values, *, dtype, device):
        assert dtype == 'int64' and device == 'cpu' and isinstance(values, list)
        self.operations.append(('allocate_indices', tuple(values)))
        return FakeTensor(self, [len(values)], dtype, struct.pack('<'+'q'*len(values), *values))

    def index_select(self, tensor, dim, index):
        assert dim == 0 and len(tensor.shape) == 2 and index.dtype == 'int64'
        chosen = index.tolist()
        self.operations.append(('index_select', tuple(tensor.shape), tuple(chosen)))
        data = b''.join(tensor.select(0, i).raw_logical() for i in chosen)
        width = tensor.shape[1] - (1 if self.bad_selected_width else 0)
        return FakeTensor(self, [len(chosen), width], tensor.dtype, data)

    def argmax(self, tensor, *, dim):
        assert dim == -1 and tensor.dtype == 'float32' and len(tensor.shape) == 2
        self.operations.append(('argmax', tuple(tensor.shape), dim))
        results = []
        for i in range(tensor.shape[0]):
            values = struct.unpack('<'+'f'*tensor.shape[1], tensor.select(0, i).raw_logical())
            results.append(max(range(len(values)), key=lambda j: values[j]))
        return self.tensor(results, dtype='int64', device='cpu')


def patterned(api, shape, dtype='bfloat16', *, stride=None, offset=0):
    stride = strides(shape) if stride is None else stride
    extent = offset+sum((n-1)*s for n, s in zip(shape, stride))+1
    # Distinct finite words where possible; expected oracle walks physical indices independently.
    if dtype == 'bfloat16':
        raw = struct.pack('<'+'H'*extent, *(0x1000+i%0x5000 for i in range(extent)))
    elif dtype == 'float32':
        raw = struct.pack('<'+'I'*extent, *(0x3f000000+i%1000000 for i in range(extent)))
    else:
        raw = struct.pack('<'+'q'*extent, *(i*1000000000001-99999 for i in range(extent)))
    return FakeTensor(api, shape, dtype, raw, stride=stride, offset=offset)


def independent_expected(tensor, rows=None):
    shape, word = tensor.shape, WIDTH[tensor.dtype]
    indices = list(itertools.product(*(range(n) for n in shape)))
    if rows is not None:
        indices = [coords for row in rows for coords in indices if coords[0] == row]
    result = bytearray()
    for coords in indices:
        physical = tensor.offset+sum(x*y for x, y in zip(coords, tensor._stride))
        result.extend(tensor.storage[0][physical*word:(physical+1)*word])
    return bytes(result)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.api = FakeTorch()
        self.ops = NativeTensorAdapter(self.api)

    def bytes_equal(self, tensor, rows=None):
        before = (deepcopy(tensor.shape), tensor.stride(), tensor.storage_offset(), tensor.storage, tensor.dtype)
        chunks = list(self.ops.iter_bytes(tensor, rows))
        self.assertTrue(chunks)
        self.assertTrue(all(type(c) is bytes and 0 < len(c) <= 65536 and len(c)%tensor.element_size() == 0 for c in chunks))
        self.assertEqual(b''.join(chunks), independent_expected(tensor, rows))
        self.assertEqual(before, (tensor.shape, tensor.stride(), tensor.storage_offset(), tensor.storage, tensor.dtype))
        self.assertTrue(all(n <= 65536 for n in self.api.copy_sizes))
        self.assertEqual(sum(map(len, chunks)), (tensor.numel() if rows is None else len(rows)*tensor.shape[1])*tensor.element_size())
        return chunks

    def test_describe_actual_layout_and_nonzero_offset_without_value_copy(self):
        tensor = patterned(self.api, [3, 5], stride=[19, 3], offset=7)
        self.assertEqual(self.ops.describe(tensor), dict(shape=[3, 5], dtype='BF16', stride=[19, 3], storage_offset=7))
        self.assertEqual(self.api.operations, [])

    def test_all_supported_dtypes_and_byte_widths(self):
        for dtype, name in [('bfloat16', 'BF16'), ('float32', 'F32'), ('int64', 'I64')]:
            with self.subTest(dtype=dtype):
                self.assertEqual(self.ops.describe(patterned(self.api, [3], dtype))['dtype'], name)

    def test_bf16_exact_bits_including_negative_zero_and_nan(self):
        raw = struct.pack('<8H', 0x0000, 0x8000, 0x3f80, 0xbf80, 0x0001, 0x7f80, 0xff80, 0x7fc1)
        tensor = FakeTensor(self.api, [2, 4], 'bfloat16', raw)
        self.assertEqual(b''.join(self.bytes_equal(tensor)), raw)
        self.assertTrue(all(op[1] == 'uint8' for op in self.api.operations if op[0] == 'tolist'))

    def test_fp32_exact_bits_including_signed_zero_subnormal_and_nan(self):
        raw = struct.pack('<8I', 0, 0x80000000, 0x3f800000, 0xbf800000, 1, 0x7f800000, 0xff800000, 0x7fc00011)
        self.assertEqual(b''.join(self.bytes_equal(FakeTensor(self.api, [2, 4], 'float32', raw))), raw)

    def test_transposed_2d_full_logical_order(self):
        self.bytes_equal(patterned(self.api, [5, 7], stride=[1, 5], offset=9))

    def test_column_strided_padding_and_offset_excluded(self):
        t = patterned(self.api, [5, 11], stride=[43, 3], offset=13)
        self.assertLess(t.numel()*2, len(t.storage[0]))
        self.bytes_equal(t)

    def test_permuted_3d_full_logical_order(self):
        self.bytes_equal(patterned(self.api, [3, 4, 5], stride=[5, 15, 1], offset=17))

    def test_broadcast_stride_zero_preserves_logical_repetition(self):
        self.bytes_equal(patterned(self.api, [4, 5, 7], stride=[0, 8, 1], offset=3))

    def test_noncontiguous_integer_vector_exact_large_values(self):
        t = patterned(self.api, [5], 'int64', stride=[3], offset=4)
        expect = list(struct.unpack('<5q', independent_expected(t)))
        self.assertEqual(self.ops.integers(t), expect)
        self.assertTrue(any(abs(x) > 2**40 for x in expect))
        self.assertEqual(self.api.copy_sizes, [])

    def test_int64_min_and_max_no_float_rounding(self):
        values = [-(2**63), 2**63-1, 2**53+1, -(2**53)-1]
        t = FakeTensor(self.api, [4], 'int64', struct.pack('<4q', *values))
        self.assertEqual(self.ops.integers(t), values)

    def test_select_noncontiguous_full_width_preserves_dtype_and_column_order(self):
        t = patterned(self.api, [7, 19], stride=[43, 2], offset=3)
        out = self.ops.select(t, [0, 2, 6])
        self.assertEqual(out.shape, (3, 19))
        self.assertEqual(out.dtype, t.dtype)
        self.assertEqual(out.raw_logical(), independent_expected(t, [0, 2, 6]))
        self.assertNotEqual(out.data_ptr(), t.data_ptr())

    def test_select_fp32_preserves_raw_words(self):
        t = patterned(self.api, [6, 7], 'float32', stride=[21, 2], offset=5)
        self.assertEqual(self.ops.select(t, [1, 5]).raw_logical(), independent_expected(t, [1, 5]))

    def test_selected_serializer_uses_real_full_rows_only(self):
        t = patterned(self.api, [9, 33], stride=[109, 3], offset=13)
        self.bytes_equal(t, [1, 5, 8])
        self.assertFalse(any(op[0] == 'index_select' for op in self.api.operations))
        self.assertEqual([op[3] for op in self.api.operations if op[0] == 'narrow'], [1, 5, 8])

    def test_selected_wide_rows_split_without_column_truncation(self):
        t = patterned(self.api, [3, 17003], 'float32', stride=[34009, 2], offset=3)
        chunks = self.bytes_equal(t, [0, 2])
        self.assertEqual([len(x) for x in chunks], [65536, 2476, 65536, 2476])

    def test_exact_64k_bf16_boundary(self):
        chunks = self.bytes_equal(patterned(self.api, [32768]))
        self.assertEqual([len(x) for x in chunks], [65536])

    def test_64k_plus_one_bf16_word_boundary(self):
        chunks = self.bytes_equal(patterned(self.api, [32769]))
        self.assertEqual([len(x) for x in chunks], [65536, 2])

    def test_fp32_vocabulary131072_all_columns(self):
        chunks = self.bytes_equal(patterned(self.api, [1, 131072], 'float32'))
        self.assertEqual([len(x) for x in chunks], [65536]*8)
        self.assertEqual(max(self.api.copy_sizes), 65536)

    def test_multiblock_hidden_width2560_noncontiguous(self):
        t = patterned(self.api, [27, 2560], stride=[7703, 3], offset=11)
        chunks = self.bytes_equal(t)
        self.assertEqual([len(x) for x in chunks], [61440, 61440, 15360])

    def test_cache_four_heads_complete_extent_and_order(self):
        t = patterned(self.api, [4, 257, 256], stride=[1, 1027, 4], offset=19)
        chunks = self.bytes_equal(t)
        self.assertEqual(len(chunks), 12)
        self.assertEqual([len(x) for x in chunks], [65536, 65536, 512]*4)

    def test_singleton_dimensions_and_offset(self):
        self.bytes_equal(patterned(self.api, [1, 2, 1], stride=[77, 5, 99], offset=23))

    def test_top1_full_columns_last_position_and_first_tie(self):
        data = struct.pack('<12f', -5, -4, -3, -2, -1, 11, 2, 9, 9, 1, 0, -1)
        t = FakeTensor(self.api, [2, 6], 'float32', data)
        self.assertEqual(self.ops.top1(t), [5, 1])
        self.assertIn(('argmax', (2, 6), -1), self.api.operations)

    def test_top1_noncontiguous_layout(self):
        # Logical rows [1, 9, 2] and [8, 3, 0], in transposed physical layout.
        t = FakeTensor(self.api, [2, 3], 'float32', struct.pack('<6f', 1, 8, 9, 3, 2, 0), stride=[1, 2])
        self.assertEqual(self.ops.top1(t), [1, 0])

    def test_adapter_and_existing_writer_verifier_roundtrip(self):
        t = patterned(self.api, [5, 11], stride=[37, 3], offset=7)
        layout = dict(x=dict(shape=[5, 11], dtype='BF16'))
        stream = io.BytesIO()
        writer = ArchiveWriter(stream, layout, byte_limit=110)
        writer.write('x', self.ops.iter_bytes(t))
        archive = writer.finish()
        result = verify_archive(io.BytesIO(stream.getvalue()), archive, layout, max_bytes=110,
            selected_row_checks={'x': dict(rows=[0, 4], sha256=hashlib.sha256(independent_expected(t, [0, 4])).hexdigest())})
        self.assertEqual(result['bytes_verified'], 110)
        self.assertEqual(result['selected_target_rows_verified'], 2)

    def test_adapter_preserves_nan_for_writer_to_reject(self):
        t = FakeTensor(self.api, [1], 'bfloat16', b'\xc1\x7f')
        writer = ArchiveWriter(io.BytesIO(), dict(x=dict(shape=[1], dtype='BF16')), byte_limit=2)
        with self.assertRaises(RuntimeError):
            writer.write('x', self.ops.iter_bytes(t))
        self.assertTrue(writer.poisoned)

    def test_reject_unsupported_device_layout_quantized_and_lazy_flags(self):
        for mutation in ['device', 'layout', 'quantized', 'conj', 'neg']:
            t = patterned(self.api, [2, 3])
            if mutation == 'device':
                t.device.type = 'cuda'
            elif mutation == 'layout':
                t.layout = 'sparse_coo'
            elif mutation == 'quantized':
                t.is_quantized = True
            elif mutation == 'conj':
                t.conjugate = True
            else:
                t.negative = True
            with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                list(self.ops.iter_bytes(t))

    def test_reject_wrong_dtype_rank_zero_or_zero_extent(self):
        for shape, dtype in [([], 'bfloat16'), ([0], 'bfloat16'), ([1, 1, 1, 1], 'bfloat16'), ([3], 'uint8')]:
            t = FakeTensor(self.api, shape, dtype, b'\0'*16)
            with self.subTest(shape=shape, dtype=dtype), self.assertRaises(RuntimeError):
                self.ops.describe(t)

    def test_reject_negative_stride_or_offset(self):
        for stride, offset in [([-1], 2), ([1], -1)]:
            t = FakeTensor(self.api, [3], 'bfloat16', b'\0'*12, stride=stride, offset=offset)
            with self.subTest(stride=stride, offset=offset), self.assertRaises(RuntimeError):
                self.ops.describe(t)

    def test_reject_selected_duplicates_order_bool_oob_or_empty(self):
        t = patterned(self.api, [5, 7])
        for rows in [[], [1, 1], [3, 2], [True], [-1], [5], (1, 3)]:
            with self.subTest(rows=rows), self.assertRaises(RuntimeError):
                list(self.ops.iter_bytes(t, rows))
            with self.subTest(select=rows), self.assertRaises(RuntimeError):
                self.ops.select(t, rows)

    def test_reject_selection_on_rank1_or_rank3(self):
        for shape in [[5], [2, 3, 4]]:
            t = patterned(self.api, shape)
            with self.subTest(shape=shape), self.assertRaises(RuntimeError):
                list(self.ops.iter_bytes(t, [0]))

    def test_reject_int64_payload_and_floating_integer_input(self):
        with self.assertRaises(RuntimeError):
            list(self.ops.iter_bytes(patterned(self.api, [3], 'int64')))
        with self.assertRaises(RuntimeError):
            self.ops.integers(patterned(self.api, [3], 'float32'))

    def test_reject_unbounded_integer_vector_before_tolist(self):
        t = patterned(self.api, [2061], 'int64')
        with self.assertRaises(RuntimeError):
            self.ops.integers(t)
        self.assertEqual(self.api.operations, [])

    def test_reject_integer_vector_noninteger_or_wrong_count(self):
        t = patterned(self.api, [3], 'int64')
        for values in [[1.0, 2, 3], [True, 2, 3], [1, 2], [2**63, 0, 0]]:
            t.tolist = lambda values=values: values
            with self.subTest(values=values), self.assertRaises(RuntimeError):
                self.ops.integers(t)

    def test_reject_wrong_byte_element_size_or_numel(self):
        t = patterned(self.api, [3])
        t.element_size = lambda: 4
        with self.assertRaises(RuntimeError):
            self.ops.describe(t)
        t = patterned(self.api, [3])
        t.numel = lambda: 2
        with self.assertRaises(RuntimeError):
            self.ops.describe(t)

    def test_clone_failure_propagates(self):
        self.api.fail_clone = True
        with self.assertRaisesRegex(ValueError, 'clone failure'):
            list(self.ops.iter_bytes(patterned(self.api, [3])))

    def test_clone_alias_and_dtype_substitution_rejected(self):
        for flag in ['alias_clone', 'change_clone_dtype']:
            setattr(self.api, flag, True)
            with self.subTest(flag=flag), self.assertRaises(RuntimeError):
                list(self.ops.iter_bytes(patterned(self.api, [3])))
            setattr(self.api, flag, False)

    def test_byte_list_bad_length_values_or_types_rejected(self):
        for fn in [lambda x: x[:-1], lambda x: [256]+x[1:], lambda x: [-1]+x[1:],
                   lambda x: [1.0]+x[1:], lambda x: [True]+x[1:]]:
            self.api.byte_list_mutator = fn
            with self.subTest(fn=fn), self.assertRaises(RuntimeError):
                list(self.ops.iter_bytes(patterned(self.api, [3])))

    def test_geometry_changed_between_chunks_rejected(self):
        t = patterned(self.api, [32769])
        iterator = self.ops.iter_bytes(t)
        self.assertEqual(len(next(iterator)), 65536)
        t.offset += 1
        with self.assertRaises(RuntimeError):
            next(iterator)

    def test_backing_pointer_changed_between_chunks_rejected(self):
        t = patterned(self.api, [32769])
        iterator = self.ops.iter_bytes(t)
        next(iterator)
        t.storage = (t.storage[0], t.storage[1]+10)
        with self.assertRaises(RuntimeError):
            next(iterator)

    def test_geometry_changed_after_last_yield_rejected_on_completion(self):
        t = patterned(self.api, [3])
        iterator = self.ops.iter_bytes(t)
        next(iterator)
        t._stride = (0,)
        with self.assertRaises(RuntimeError):
            next(iterator)

    def test_bounded_stream_is_lazy_and_no_full_copy_before_first_yield(self):
        t = patterned(self.api, [100, 2560], stride=[7681, 3], offset=7)
        iterator = self.ops.iter_bytes(t)
        self.assertEqual(self.api.operations, [])
        first = next(iterator)
        self.assertEqual(len(first), 61440)
        self.assertEqual(self.api.copy_sizes, [61440])
        iterator.close()
        self.assertEqual(self.api.copy_sizes, [61440])

    def test_no_full_contiguous_conversion_of_source(self):
        t = patterned(self.api, [3, 17003], 'float32', stride=[34009, 2], offset=3)
        self.bytes_equal(t)
        self.assertFalse(any(op[0] == 'contiguous' for op in self.api.operations))

    def test_wrong_narrow_extent_and_selected_width_rejected(self):
        self.api.bad_narrow = True
        with self.assertRaises(RuntimeError):
            list(self.ops.iter_bytes(patterned(self.api, [32769])))
        self.api.bad_narrow = False
        self.api.bad_selected_width = True
        with self.assertRaises(RuntimeError):
            self.ops.select(patterned(self.api, [3, 7]), [0, 2])

    def test_big_endian_rejected_without_native_operations(self):
        with patch('dory_stage_a_tensor_adapter_v1.sys.byteorder', 'big'), self.assertRaises(RuntimeError):
            list(self.ops.iter_bytes(patterned(self.api, [3])))
        self.assertEqual(self.api.operations, [])

    def test_top1_requires_full_fp32_2d_rows(self):
        for shape, dtype in [([3], 'float32'), ([2, 3], 'bfloat16'), ([2061, 1], 'float32')]:
            with self.subTest(shape=shape, dtype=dtype), self.assertRaises(RuntimeError):
                self.ops.top1(patterned(self.api, shape, dtype))

    def test_top1_invalid_native_result_rejected(self):
        t = patterned(self.api, [2, 3], 'float32')
        for values in [[0], [0, 3], [-1, 1]]:
            self.api.argmax = lambda tensor, dim, values=values: self.api.tensor(values, dtype='int64', device='cpu')
            with self.subTest(values=values), self.assertRaises(RuntimeError):
                self.ops.top1(t)

    def test_guard_failure_propagates_without_native_copy(self):
        def guard():
            raise ValueError('synthetic resource expiry')
        ops = NativeTensorAdapter(self.api, guard=guard)
        with self.assertRaisesRegex(ValueError, 'resource expiry'):
            list(ops.iter_bytes(patterned(self.api, [3])))
        self.assertEqual(self.api.copy_sizes, [])

    def test_no_native_import_or_filesystem_or_input_constructor_in_serializer(self):
        source = Path(__file__).with_name('dory_stage_a_tensor_adapter_v1.py').read_text()
        tree = ast.parse(source)
        imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertTrue(all(isinstance(n, ast.Import) and all(x.name in ('itertools', 'math', 'sys') for x in n.names)
                            for n in imports))
        self.assertFalse(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                             and n.func.id in ('open', 'eval', 'exec', '__import__') for n in ast.walk(tree)))
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('numpy', sys.modules)
        # Neither serializer nor describe/integers/top1 contains an allocation
        # from arbitrary data. The only torch.tensor call creates row indices.
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'iter_bytes')
        calls = [n.func.attr for n in ast.walk(method) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        self.assertFalse(any(name in calls for name in ('tensor', 'zeros', 'empty', 'full', 'pad', 'numpy', 'float', 'to')))
        self.assertNotIn('zeros', source)


if __name__ == '__main__':
    unittest.main(verbosity=2)
