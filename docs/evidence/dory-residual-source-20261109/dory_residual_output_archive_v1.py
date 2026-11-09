"""Bounded raw observation archives for prepared Dory replay, no native imports.

The future admitted worker supplies raw/description callbacks pinned to original
tensor semantics. Partial files and completed descriptors are never deleted.
"""
import hashlib
import math
from pathlib import Path
import re

from dory_residual_visit_capture_v1 import require


class ObservationArchive:
    def __init__(self, root, raw, describe, guard, byte_limit):
        self.root = Path(root)
        require(self.root.is_dir() and self.root.resolve() == self.root,
                "Existing canonical admitted output directory")
        require(type(byte_limit) is int and byte_limit > 0, "Explicit auxiliary archive budget")
        self.raw, self.describe, self.guard = raw, describe, guard
        self.byte_limit = byte_limit
        self.bytes_written = 0
        self.completed = []
        self.pending = None
        self.failed = False

    def write(self, name, tensors):
        require(not self.failed and self.pending is None,
                "Failed or active observation archive cannot be resumed")
        require(isinstance(name, str) and re.fullmatch(r"[a-z0-9-]+\.bin", name),
                "Fresh observation archive basename")
        require(isinstance(tensors, dict), "Named observation tensors")
        self.guard()
        path = self.root / name
        info = dict(file=name, bytes=0, tensors={}, complete=False)
        digest = hashlib.sha256()
        # Open first; an existing file remains unchanged and no pending archive
        # or byte accounting is invented for a rejected duplicate filename.
        stream = path.open("xb")
        self.pending = info
        try:
            with stream:
                for label, tensor in tensors.items():
                    self.guard()
                    require(isinstance(label, str) and bool(label), "Nonempty tensor label")
                    description = self.describe(tensor)
                    require(set(description) == {"shape", "dtype", "finite"},
                            "Explicit original tensor description")
                    shape, dtype = description["shape"], description["dtype"]
                    require(isinstance(shape, list)
                            and all(type(n) is int and n >= 0 for n in shape)
                            and dtype in ("BF16", "F32")
                            and type(description["finite"]) is bool,
                            "Preserve supported raw tensor dtype and finite status")
                    size = math.prod(shape) * (2 if dtype == "BF16" else 4)
                    require(self.bytes_written + size <= self.byte_limit,
                            "Auxiliary raw archive budget exhausted")
                    payload = self.raw(tensor)
                    require(type(payload) is bytes and len(payload) == size,
                            "Complete raw tensor extent")
                    written = stream.write(payload)
                    require(written == len(payload), "Complete observation payload write")
                    stream.flush()
                    digest.update(payload)
                    info["tensors"][label] = dict(
                        description, offset=info["bytes"], bytes=len(payload),
                        sha256=hashlib.sha256(payload).hexdigest())
                    info["bytes"] += len(payload)
                    self.bytes_written += len(payload)
            # A successful close is part of completion. Failed flush/close may
            # leave extra file bytes beyond the completed tensor prefix below.
            info.update(complete=True, sha256=digest.hexdigest())
            self.completed.append(info)
            self.pending = None
            return info
        except BaseException:
            self.failed = True
            info["completed_tensor_prefix_sha256"] = digest.hexdigest()
            raise


def original_archive_comparison(actual, expected):
    """Metadata comparison after actual payload bytes have been archived.

    The result alone is not independent raw verification. The saved-byte
    reviewer must reread payloads against their source-bound hashes.
    """
    header_keys = (set(actual) | set(expected)) - {"tensors"}
    actual_tensors = actual.get("tensors", {})
    expected_tensors = expected.get("tensors", {})
    return dict(
        exact=actual == expected,
        header_differences=sorted(key for key in header_keys if actual.get(key) != expected.get(key)),
        missing_tensors=sorted(set(expected_tensors) - set(actual_tensors)),
        extra_tensors=sorted(set(actual_tensors) - set(expected_tensors)),
        changed_tensors=sorted(key for key in set(actual_tensors) & set(expected_tensors)
                               if actual_tensors[key] != expected_tensors[key]),
        tensor_count=len(actual_tensors),
        independent_raw_review_complete=False)
