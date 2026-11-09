"""Independent complete-output byte reconstruction for prepared Dory replay.

Source component only. The caller must authenticate every original action and
producer archive descriptor and admit this reader serially after producer exit.
No model, writer or runtime imports. Lifecycle markers here are bookkeeping:
their native failure/rejection payloads require a separate review.
"""
import hashlib

from dory_generation_cache_replay_v1 import OrdinaryFiles
from review_dory_long_matrix_saved_v3 import finite, head_hash


CACHE_IDS = tuple(sorted(tuple(range(0, 16, 2)) + tuple(range(16, 56, 2))
                         + tuple(range(88, 128, 2)) + tuple(range(56, 72, 2))))
LABELS = ("hidden", "logits") + tuple(
    f"cache-{i}-{kind}" for i in CACHE_IDS for kind in ("key", "value"))
HISTORY_BYTES_PER_TOKEN = 112 * 4 * 256 * 2


def require(value, message):
    if not value:
        raise RuntimeError(message)


def digest(value):
    require(isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value), "Exact SHA256")
    return value


class CompleteArchiveReplay:
    def __init__(self, root, guard, *, max_state_bytes, max_raw_bytes):
        require(type(max_state_bytes) is int and max_state_bytes > 0
                and type(max_raw_bytes) is int and max_raw_bytes > 0,
                "Explicit complete saved-byte review bounds")
        self.files, self.guard = OrdinaryFiles(root, guard), guard
        self.max_state_bytes, self.max_raw_bytes = max_state_bytes, max_raw_bytes
        self.banks, self.epochs = {}, {}
        self.events = self.calls = self.bytes_read = 0
        self.starts = self.releases = self.rejections = self.injections = 0
        self.poisoned = False

    def _owned(self, action):
        bank = self.banks.get(action["slot"])
        require(bank is not None and bank["alive"]
                and (bank["epoch"], bank["request"]) == (action["epoch"], action["request"]),
                "Exact live original saved request")
        return bank

    def accept(self, action, archive=None):
        require(not self.poisoned, "Interrupted archive replay cannot resume")
        try:
            return self._accept(action, archive)
        except BaseException:
            self.poisoned = True
            raise

    def _accept(self, action, archive):
        self.guard()
        require(type(action["event"]) is int and action["event"] == self.events,
                "Every original event in order")
        slot, epoch, request = (action[k] for k in ("slot", "epoch", "request"))
        require(type(slot) is int and slot in (0, 1, 2, 3, 10, 11) and type(epoch) is int and epoch > 0
                and isinstance(request, str) and bool(request), "Original bounded owner")
        kind = action["kind"]
        if kind == "start":
            require(archive is None and epoch == self.epochs.get(slot, 0) + 1,
                    "Fresh next request epoch without fabricated archive")
            self.banks[slot] = dict(epoch=epoch, request=request, alive=True, length=0, heads={})
            self.epochs[slot] = epoch
            self.starts += 1
        elif kind == "release":
            require(archive is None, "Release has no successful forward archive")
            self._owned(action)
            del self.banks[slot]
            self.releases += 1
        elif kind == "rejected_handle":
            require(archive is None, "Rejected handle has no successful forward archive")
            bank = self.banks.get(slot)
            require(bank is None or not bank["alive"]
                    or (bank["epoch"], bank["request"]) != (epoch, request),
                    "Rejected original handle is actually stale")
            self.rejections += 1
        elif kind == "injected_failure":
            require(archive is None, "Injection is not a successful forward archive")
            bank = self._owned(action)
            bank["alive"], bank["heads"] = False, {}
            self.injections += 1
        else:
            require(kind == "call" and action["native_call_required"] is True,
                    "Only original successful calls have archives")
            self._forward(action, archive)
            self.calls += 1
        self.events += 1

    def _forward(self, action, archive):
        bank = self._owned(action)
        start, end = action["start"], action["end"]
        require(type(start) is type(end) is int and start == bank["length"]
                and 0 <= start < end <= 2060, "Original contiguous full interval")
        require(archive == action["original_archive"],
                "New archive metadata equals authenticated original")
        require(set(archive) == {"file", "slot", "epoch", "path", "start", "end",
                                 "tensors", "bytes", "sha256"}
                and archive["file"] == f'{action["event"]:06d}-official.bin'
                and (archive["slot"], archive["epoch"], archive["path"],
                     archive["start"], archive["end"])
                    == (action["slot"], action["epoch"], "official", start, end),
                "Exact original archive identity")
        tensors = archive["tensors"]
        require(list(tensors) == list(LABELS), "Every original114 tensor in order")
        rows = 33 if action["phase"] == "full_causal" else 1
        require(len(action["original_top1"]) == rows, "Original complete output-head rows")
        expected_offset = 0
        for label, tensor in tensors.items():
            cache = label.startswith("cache-")
            shape, dtype = ([4, end, 256], "BF16") if cache else (
                ([end-start, 2560], "BF16") if label == "hidden" else ([rows, 131072], "F32"))
            size = 2048*(end-start) if cache else (
                (end-start)*5120 if label == "hidden" else rows*131072*4)
            keys = {"shape", "dtype", "full_sha256", "mode", "offset", "bytes", "sha256", "finite"}
            if cache:
                keys |= {"prefix_sha256", "expected_previous", "prefix_valid"}
            require(set(tensor) == keys and tensor["shape"] == shape and tensor["dtype"] == dtype
                    and tensor["mode"] == ("append" if cache else "full")
                    and type(tensor["bytes"]) is int and tensor["bytes"] == size
                    and type(tensor["offset"]) is int and tensor["offset"] == expected_offset
                    and tensor["finite"] is True, "Complete original tensor geometry and finite claim")
            digest(tensor["sha256"]); digest(tensor["full_sha256"])
            if cache:
                digest(tensor["prefix_sha256"])
                require(tensor["prefix_valid"] is True, "Successful original append only")
            expected_offset += size
        require(type(archive["bytes"]) is int and archive["bytes"] == expected_offset,
                "Complete original archive extent")
        digest(archive["sha256"])
        require(self.bytes_read + expected_offset <= self.max_raw_bytes,
                "Complete raw read budget before opening payload")
        state_tokens = end + sum(b["length"] for other, b in self.banks.items()
                                 if other != action["slot"] and b["alive"])
        require(state_tokens * HISTORY_BYTES_PER_TOKEN <= self.max_state_bytes,
                "Every active request cache fits explicit reconstruction budget")
        file_hash = hashlib.sha256()
        offset = 0
        with self.files.open(archive["file"], archive["bytes"]) as stream:
            for label, tensor in tensors.items():
                self.guard()
                part_hash = hashlib.sha256()
                actual_finite = True

                def consume(size):
                    nonlocal actual_finite, offset
                    remaining = size
                    while remaining:
                        self.guard()
                        block = stream.read(min(65536, remaining))
                        element_bytes = 4 if tensor["dtype"] == "F32" else 2
                        require(bool(block) and len(block) % element_bytes == 0,
                                "Complete aligned saved payload")
                        remaining -= len(block)
                        self.bytes_read += len(block)
                        offset += len(block)
                        file_hash.update(block); part_hash.update(block)
                        actual_finite &= finite(block, tensor["dtype"])
                        yield block

                if label.startswith("cache-"):
                    heads = bank["heads"].get(label)
                    prior = None if heads is None else dict(
                        length=start, sha256=head_hash(heads), finite=True)
                    require(tensor["expected_previous"] == prior,
                            "Recorded predecessor equals actual reconstructed history")
                    prefix_sha = hashlib.sha256(b"").hexdigest() if heads is None else prior["sha256"]
                    require(tensor["prefix_sha256"] == prefix_sha
                            and ((heads is None and start == 0) or heads is not None),
                            "Every original cache prefix independently reproduced")
                    if heads is None:
                        heads = [bytearray() for _ in range(4)]
                    require(len(heads) == 4, "Four original independent head streams")
                    for head in heads:
                        require(len(head) == start*512, "Contiguous per-head history")
                        for block in consume((end-start)*512):
                            head.extend(block)
                    full_sha = head_hash(heads)
                    bank["heads"][label] = heads
                else:
                    for _ in consume(tensor["bytes"]):
                        pass
                    full_sha = part_hash.hexdigest()
                require(actual_finite is True and part_hash.hexdigest() == tensor["sha256"]
                        and full_sha == tensor["full_sha256"],
                        "Actual saved payload, finite status and complete tensor hash")
            require(offset == archive["bytes"] and stream.read(1) == b""
                    and file_hash.hexdigest() == archive["sha256"],
                    "Entire original saved archive authenticated")
        bank["length"] = end

    def finish(self):
        require(not self.poisoned, "Failed archive reader cannot finalize")
        require((self.events, self.calls, self.starts, self.releases, self.rejections, self.injections)
                == (6200, 6185, 9, 2, 3, 1), "Only complete original history can finalize")
        require(self.bytes_read == self.max_raw_bytes, "Exact full raw read total")
        return dict(complete=True, original_events=self.events, original_calls=self.calls,
                    raw_bytes_verified=self.bytes_read, tensor_labels=114, cache_histories=56,
                    full_cache_hashes_reconstructed=True, hidden_and_logits_bytes_verified=True,
                    lifecycle_failure_payloads_verified=False, original_top1_recomputed=False,
                    source_checkpoint_exit_provenance_verified_here=False,
                    native_framework_imported=False, model_executed=False,
                    full_model_qualified=False, new_speedup_claim=False)
