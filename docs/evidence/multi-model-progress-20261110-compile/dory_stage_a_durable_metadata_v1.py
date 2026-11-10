"""Bounded exclusive durable JSON; no actual I/O until explicitly called.

No replace/unlink/mkdir. This records producer metadata, never actual process
exit or independent review. Caller owns current serial admission and root.
"""
import hashlib
import json
import math
import os
from pathlib import Path

from dory_stage_a_artifact_transaction_v2 import ArchiveFileStore
from dory_stage_a_saved_files_v1 import identity, ordinary
from dory_stage_a_archive_core_v1 import require

MAX_METADATA_BYTES = 16 * 1024**2
MAX_NODES = 1000000
MAX_DEPTH = 32
MAX_STRING = 65536


def encode_bounded(value, max_bytes, guard):
    require(type(max_bytes) is int and 0 < max_bytes <= MAX_METADATA_BYTES and callable(guard),
            'Explicit bounded JSON size and live guard')
    count, active = [0], set()
    def inspect(item, depth):
        count[0] += 1
        require(count[0] <= MAX_NODES and depth <= MAX_DEPTH, 'Bounded metadata tree')
        if count[0] % 1024 == 1:
            guard()
        if type(item) is dict:
            require(id(item) not in active and len(item) <= MAX_NODES, 'Acyclic bounded metadata object')
            active.add(id(item))
            try:
                for key, child in item.items():
                    require(type(key) is str and len(key) <= MAX_STRING, 'Bounded exact JSON key')
                    inspect(child, depth+1)
            finally:
                active.remove(id(item))
        elif type(item) in (list, tuple):
            require(id(item) not in active and len(item) <= MAX_NODES, 'Acyclic bounded metadata sequence')
            active.add(id(item))
            try:
                for child in item:
                    inspect(child, depth+1)
            finally:
                active.remove(id(item))
        elif type(item) is str:
            require(len(item) <= MAX_STRING, 'Bounded JSON string')
        elif type(item) is float:
            require(math.isfinite(item), 'Finite metadata scalar')
        else:
            require(item is None or type(item) is bool or type(item) is int and -(2**63) <= item < 2**63,
                    'Ordinary bounded JSON scalar')
    inspect(value, 0)
    output = bytearray()
    # Preserve insertion order: archive tensors are ordered physical extents.
    encoder = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    for part in encoder.iterencode(value):
        block = part.encode('utf-8')
        require(len(output)+len(block)+1 <= max_bytes, 'Metadata byte bound before opening output')
        output.extend(block)
    output.extend(b'\n')
    guard()
    return bytes(output)


def write_json_exclusive(root, name, value, *, max_bytes, guard):
    require(isinstance(name, str) and Path(name).name == name and name.endswith('.json'),
            'One ordinary JSON filename')
    raw = encode_bounded(value, max_bytes, guard)
    store = ArchiveFileStore(root, guard=guard)
    try:
        stream = store.open_exclusive(name)
        for offset in range(0, len(raw), 65536):
            guard()
            block = raw[offset:offset+65536]
            written = stream.write(block)
            require(type(written) is int and written == len(block), 'Complete bounded metadata write')
        stream.flush()
        os.fsync(stream.fileno())
        before = identity(os.fstat(stream.fileno()))
        ordinary(os.fstat(stream.fileno()), before, len(raw))
        ordinary(os.stat(name, dir_fd=store.directory, follow_symlinks=False), before, len(raw))
        stream.close()
        require(stream.closed, 'Metadata file closed before completion')
        os.fsync(store.directory)
        store._stable()
        ordinary(os.stat(name, dir_fd=store.directory, follow_symlinks=False), before, len(raw))
        store.close()
        return dict(complete=True, name=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                    producer_identity=before, durable_json_complete=True,
                    actual_process_exit_verified_here=False, independent_review_complete=False,
                    runtime_admitted_here=False, full_model_qualified=False, new_speedup_claim=False)
    except BaseException:
        store.abort()
        raise
