"""Dory Stage A native tensor adapter, injected API, no native imports.

This file is source only: importing it does not import Torch/NumPy, open files,
construct tensors or load models. The future separately admitted worker passes
the original CPU torch module. No dtype conversion or numerical replacement
of attention, projections, residual normalization or output logits occurs.

iter_bytes reads every requested real logical element in row-major order.
Bounded views select portions for serialization only; the original forward
input never passes through this serializer or gets replaced by those views.
An emitted chunk is <=64 KiB; the clone and uint8 list cover only that chunk.
Python list/object overhead is additional and must be included in admission.
"""
import itertools
import math
import sys


def require(value, message):
    if not value:
        raise RuntimeError(message)


class NativeTensorAdapter:
    MAX_CHUNK_BYTES = 65536
    MAX_INTEGER_ELEMENTS = 2060

    def __init__(self, torch_api, *, guard=lambda: None):
        # Explicit injection only. No lazy import, subprocess, device initialization,
        # file access, tensor allocation or actual native operation occurs here.
        for field in ('Tensor', 'bfloat16', 'float32', 'int64', 'uint8', 'strided',
                      'contiguous_format', 'tensor', 'index_select', 'argmax'):
            require(hasattr(torch_api, field), 'Complete injected original torch interface')
        require(callable(guard), 'Explicit resource/lifetime guard')
        self.torch, self.guard = torch_api, guard

    def describe(self, tensor):
        """Read geometry without copying values or changing source layout."""
        self.guard()
        t = self.torch
        require(isinstance(tensor, t.Tensor), 'Actual injected torch Tensor instance')
        require(tensor.device.type == 'cpu' and tensor.layout == t.strided,
                'Original CPU strided tensor only; no device transfers')
        require(not tensor.is_quantized and not tensor.is_conj() and not tensor.is_neg(),
                'No quantization or lazy value-transform flags')
        shape, stride = list(tensor.shape), list(tensor.stride())
        require(1 <= len(shape) <= 3 and all(type(x) is int and x > 0 for x in shape)
                and len(stride) == len(shape)
                and all(type(x) is int and x >= 0 for x in stride),
                'Positive complete rank1/2/3 shape and actual nonnegative strides')
        offset = tensor.storage_offset()
        require(type(offset) is int and offset >= 0, 'Actual nonnegative storage offset')
        formats = ((t.bfloat16, 'BF16', 2), (t.float32, 'F32', 4), (t.int64, 'I64', 8))
        match = [(name, word) for dtype, name, word in formats if tensor.dtype == dtype]
        require(len(match) == 1, 'Original BF16/F32/I64 dtype only')
        name, word = match[0]
        require(tensor.element_size() == word and tensor.numel() == math.prod(shape),
                'Exact dtype word width and complete logical element count')
        return dict(shape=shape, dtype=name, stride=stride, storage_offset=offset)

    def _stamp(self, tensor):
        description = self.describe(tensor)
        pointer = tensor.data_ptr()
        require(type(pointer) is int and pointer > 0, 'Real nonempty CPU tensor storage')
        return description, pointer

    def _unchanged(self, tensor, stamp):
        require(self._stamp(tensor) == stamp, 'Source geometry/storage unchanged during adapter operation')

    def _rows(self, shape, rows):
        require(len(shape) == 2 and isinstance(rows, list) and bool(rows)
                and all(type(x) is int and 0 <= x < shape[0] for x in rows)
                and rows == sorted(set(rows)), 'Real sorted unique full-width row selection')
        return tuple(rows)

    def integers(self, tensor):
        stamp = self._stamp(tensor)
        desc = stamp[0]
        require(desc['dtype'] == 'I64' and len(desc['shape']) == 1
                and desc['shape'][0] <= self.MAX_INTEGER_ELEMENTS,
                'Bounded original int64 token/position vector')
        values = tensor.tolist()
        require(isinstance(values, list) and len(values) == desc['shape'][0]
                and all(type(x) is int and -(2**63) <= x < 2**63 for x in values),
                'Exact complete integer values without float conversion')
        self._unchanged(tensor, stamp)
        return values

    def select(self, tensor, rows):
        """Select real full-width output-head rows; never construct padded inputs."""
        stamp = self._stamp(tensor)
        desc = stamp[0]
        require(desc['dtype'] in ('BF16', 'F32'), 'Original floating output-head dtype')
        chosen = self._rows(desc['shape'], rows)
        require(len(chosen) <= self.MAX_INTEGER_ELEMENTS, 'Bounded original output-head selector')
        self.guard()
        index = self.torch.tensor(list(chosen), dtype=self.torch.int64, device='cpu')
        require(self.integers(index) == list(chosen), 'Exact int64 row index allocation')
        output = self.torch.index_select(tensor, 0, index).contiguous()
        result = self.describe(output)
        require(result['shape'] == [len(chosen), desc['shape'][1]]
                and result['dtype'] == desc['dtype']
                and result['stride'] == [desc['shape'][1], 1]
                and output.data_ptr() != tensor.data_ptr(),
                'Independent full-width contiguous selected rows with unchanged dtype')
        self._unchanged(tensor, stamp)
        return output

    def top1(self, tensor):
        """Original full-width argmax, including native first-index tie behavior.

        Stage A emits/finite-checks logits before this call. This adapter adds no
        softmax, sampling, cast or shortened vocabulary to the native operation.
        """
        stamp = self._stamp(tensor)
        desc = stamp[0]
        require(desc['dtype'] == 'F32' and len(desc['shape']) == 2
                and desc['shape'][0] <= self.MAX_INTEGER_ELEMENTS, 'Complete bounded FP32 logits rows')
        self.guard()
        result = self.torch.argmax(tensor, dim=-1)
        values = self.integers(result)
        require(len(values) == desc['shape'][0]
                and all(0 <= x < desc['shape'][1] for x in values),
                'One full-vocabulary top1 for each original row')
        self._unchanged(tensor, stamp)
        return values

    def iter_bytes(self, tensor, rows=None):
        """Yield every real BF16/F32 byte in logical row-major order, <=64 KiB.

        CPU little-endian only. No conversion to Python float or NumPy occurs.
        Each bounded native clone is reinterpreted as uint8; .tolist() therefore
        produces byte integers, preserving BF16 bits, signed zero and NaN bits.
        Actual finite acceptance remains the independent writer/reviewer's job.

        Arbitrary positive rank<=3 strided layouts (including stride0 broadcast)
        are serialized by logical indices, not their padded backing storage.
        Full rows=None serializes the complete tensor. Selected rows are allowed
        only for rank2 and every column of every selected row is retained.
        Concurrent value mutation cannot be proved absent by geometry checks;
        the future worker must own immutable observations during consumption.
        """
        require(sys.byteorder == 'little', 'Original little-endian CPU representation')
        stamp = self._stamp(tensor)
        desc = stamp[0]
        require(desc['dtype'] in ('BF16', 'F32'), 'Only actual BF16/F32 payload serialization')
        word = 2 if desc['dtype'] == 'BF16' else 4
        chosen = None if rows is None else self._rows(desc['shape'], rows)
        expected = word * (math.prod(desc['shape']) if chosen is None
                           else len(chosen)*desc['shape'][1])
        limit = self.MAX_CHUNK_BYTES // word
        emitted = 0

        def encode(view, shape):
            self.guard()
            self._unchanged(tensor, stamp)
            view_desc = self.describe(view)
            require(view_desc['shape'] == list(shape) and view_desc['dtype'] == desc['dtype'],
                    'Real bounded view retains every requested logical element and dtype')
            elements = math.prod(shape)
            require(0 < elements <= limit, 'Bounded native contiguous copy before allocation')
            # clone(contiguous_format), unlike contiguous(), forces a fresh exact
            # storage extent even if the view was already contiguous with offset.
            contiguous = view.detach().clone(memory_format=self.torch.contiguous_format)
            actual = self.describe(contiguous)
            canonical, count = [], 1
            for dimension in reversed(shape):
                canonical.insert(0, count)
                count *= dimension
            require(actual['shape'] == list(shape) and actual['dtype'] == desc['dtype']
                    and actual['stride'] == canonical and actual['storage_offset'] == 0
                    and contiguous.data_ptr() != view.data_ptr(), 'Independent exact native chunk')
            byte_view = contiguous.view(self.torch.uint8).reshape(-1)
            require(byte_view.device.type == 'cpu' and byte_view.dtype == self.torch.uint8
                    and list(byte_view.shape) == [elements*word]
                    and byte_view.numel() == elements*word and byte_view.element_size() == 1,
                    'Byte reinterpretation, no floating-point cast')
            values = byte_view.tolist()
            require(isinstance(values, list) and len(values) == elements*word
                    and all(type(x) is int and 0 <= x < 256 for x in values),
                    'Complete exact uint8 byte list')
            payload = bytes(values)
            require(0 < len(payload) <= self.MAX_CHUNK_BYTES and len(payload) % word == 0,
                    'Aligned nonempty <=64KiB stream chunk')
            self._unchanged(tensor, stamp)
            return payload

        def walk(view, shape):
            self.guard()
            if math.prod(shape) <= limit:
                yield encode(view, shape)
                return
            trailing = math.prod(shape[1:])
            if trailing <= limit:
                per_block = max(1, limit//trailing)
                for begin in range(0, shape[0], per_block):
                    length = min(per_block, shape[0]-begin)
                    yield encode(view.narrow(0, begin, length), (length, *shape[1:]))
            else:
                # Recurse by real first-dimension views. This never flattens or
                # copies the full noncontiguous input to obtain a bounded slice.
                for first in range(shape[0]):
                    yield from walk(view.select(0, first), shape[1:])

        views = ((tensor, tuple(desc['shape'])),) if chosen is None else (
            (tensor.narrow(0, row, 1), (1, desc['shape'][1])) for row in chosen)
        for view, shape in views:
            for payload in walk(view, shape):
                emitted += len(payload)
                require(emitted <= expected, 'No extra, repeated or padded payload bytes')
                yield payload
        self._unchanged(tensor, stamp)
        require(emitted == expected, 'Every original requested byte emitted exactly once')
