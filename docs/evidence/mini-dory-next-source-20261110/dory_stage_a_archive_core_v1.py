"""Small streaming archive core; no paths, native libraries or launch helpers.

The future admitted caller owns exclusive fresh file creation, quotas, flush /
fsync / close, metadata persistence and actual exit. This core writes only to
an injected stream and cannot itself establish durable or independently read
archives. Synthetic tests use a few in-memory bytes only.
"""
import array
from copy import deepcopy
import hashlib
import math
import sys


def require(value, message):
    if not value:
        raise RuntimeError(message)


class ArchiveWriter:
    def __init__(self, stream, layout, *, byte_limit, guard=lambda: None):
        require(isinstance(layout, dict) and bool(layout) and type(byte_limit) is int and byte_limit > 0,
                'Explicit nonempty layout and byte bound')
        self.layout = deepcopy(layout)
        self.total = 0
        for label, item in self.layout.items():
            require(isinstance(label, str) and bool(label) and set(item) == {'shape', 'dtype'}
                    and isinstance(item['shape'], list) and bool(item['shape'])
                    and all(type(x) is int and x > 0 for x in item['shape'])
                    and item['dtype'] in ('BF16', 'F32'), 'Positive explicit BF16/F32 layout')
            self.total += math.prod(item['shape']) * (2 if item['dtype'] == 'BF16' else 4)
        require(self.total <= byte_limit, 'Whole known archive fits before any write')
        self.stream, self.guard, self.limit = stream, guard, byte_limit
        self.written, self.records, self.whole = 0, {}, hashlib.sha256()
        self.poisoned, self.finished = False, False

    def write(self, label, chunks):
        require(not self.poisoned and not self.finished, 'Healthy unfinished archive')
        expected = list(self.layout)[len(self.records)] if len(self.records) < len(self.layout) else None
        require(label == expected, 'Exact archive tensor order without omissions or duplicates')
        item = self.layout[label]
        word = 2 if item['dtype'] == 'BF16' else 4
        size = math.prod(item['shape']) * word
        offset, count, part = self.written, 0, hashlib.sha256()
        try:
            for chunk in chunks:
                self.guard()
                require(type(chunk) is bytes and 0 < len(chunk) <= 65536 and len(chunk) % word == 0,
                        'Nonempty bounded aligned byte chunks')
                require(count+len(chunk) <= size and self.written+len(chunk) <= self.limit,
                        'Exact tensor extent and archive byte bound')
                require(sys.byteorder == 'little', 'Recorded little-endian tensor encoding')
                words = array.array('H' if word == 2 else 'I')
                words.frombytes(chunk)
                mask = 0x7f80 if word == 2 else 0x7f800000
                require(all(x & mask != mask for x in words), 'Finite actual BF16/F32 words')
                done = self.stream.write(chunk)
                require(type(done) is int and done == len(chunk), 'Complete chunk write')
                count += len(chunk)
                self.written += len(chunk)
                part.update(chunk)
                self.whole.update(chunk)
            require(count == size, 'No truncated tensor payload')
            result = dict(shape=list(item['shape']), dtype=item['dtype'], finite=True,
                          offset=offset, bytes=size, sha256=part.hexdigest())
            self.records[label] = result
            return deepcopy(result)
        except BaseException:
            self.poisoned = True
            raise

    def finish(self):
        require(not self.poisoned and not self.finished
                and list(self.records) == list(self.layout) and self.written == self.total,
                'Every planned tensor completely emitted')
        self.finished = True
        return dict(complete=True, bytes=self.written, sha256=self.whole.hexdigest(),
                    tensors=deepcopy(self.records))


class TensorEmitter:
    """Adapter over three independently bounded streams; no tensor math here.

    iter_bytes(t, rows) must yield native little-endian bytes in logical row-major
    order, preserving dtype and every column. rows=None serializes the complete
    tensor; selected rows must be real original rows, never synthesized padding.
    Adapters, source pinning and actual native finite behavior need native tests.
    """
    def __init__(self, writers, iter_bytes):
        require(set(writers) == {'target', 'selected', 'original'}, 'All three capture/control archives')
        self.writers, self.iter_bytes = dict(writers), iter_bytes

    def __call__(self, scope, label, tensor, rows):
        return self.writers[scope].write(label, self.iter_bytes(tensor, rows))

    def finish(self):
        return {scope: writer.finish() for scope, writer in self.writers.items()}
