"""Prepared complete Dory event metadata validation; no native imports.

The indexed producer JSON must be authenticated by the caller before accept().
This validates metadata, not tensor bytes, native execution or model accuracy.
All 112 cache tensors plus hidden/logits are covered. The original first-five
plan supplies marked-event identities only; it does not limit cache coverage.
"""
import copy
import hashlib

from dory_residual_visit_capture_v1 import require
from dory_residual_request_banks_v1 import CACHE_IDS, Handle

CACHE_LABELS = tuple(f"cache-{key}-{kind}" for key in sorted(CACHE_IDS)
                     for kind in ("key", "value"))
TENSOR_LABELS = ("hidden", "logits", *CACHE_LABELS)
EMPTY_SHA = hashlib.sha256(b"").hexdigest()
PHYSICAL_CALLS = {str(i): 2 if 16 <= i < 56 else 1 for i in range(72)}
INJECTION_ADVANCED = frozenset((*range(0, 56, 2), 88, 90, 92, 94))


def check_sha(value):
    require(isinstance(value, str) and len(value) == 64
            and set(value) <= set("0123456789abcdef"), "Original SHA256 format")
    return value


class ResidualReplayHistory:
    def __init__(self, selected, reference_positions):
        require(len(reference_positions) == 33 and len(set(reference_positions)) == 33
                and all(type(p) is int and 0 <= p < 2060 for p in reference_positions),
                "Original 33 reference output positions")
        self.selected = copy.deepcopy(selected)
        self.reference_positions = tuple(reference_positions)
        self.banks, self.epochs = {}, {}
        self.events = self.calls = self.starts = self.releases = 0
        self.rejections = self.injections = self.raw_archive_bytes = 0
        self.selected_seen = []
        self.full_reference_calls = 0
        self.marked_calls = 0

    def inventory(self, value, exclude=None):
        wanted = {str(slot): bank for slot, bank in self.banks.items() if slot != exclude}
        require(set(value) == set(wanted), "Every original request slot accounted for")
        for slot, bank in wanted.items():
            actual = value[slot]["official"]
            require(all(actual[k] == bank[k] for k in ("epoch", "request", "alive", "length")),
                    "Exact original request cursor and owner")
            expected = {label.removeprefix("cache-").replace("-key", "-0").replace("-value", "-1"): digest
                        for label, digest in bank["hashes"].items()}
            require(actual["histories"] == expected, "All 112 original history hashes match")

    def _archive(self, row, bank):
        start, end = row["start"], row["end"]
        head_rows = 33 if row["phase"] == "full_causal" else 1
        if row["phase"] == "full_causal":
            require(start == 0 and end == 2060, "Original complete full-causal shape")
        require(row["head_input_shape"] == [head_rows, 2560],
                "Original selected output-head call shape")
        require(row["physical_layer_invocations"] == PHYSICAL_CALLS,
                "All 112 original physical/loop visits")
        archive = row["official"]
        require(all(archive[k] == row[k] for k in ("slot", "epoch", "start", "end"))
                and archive["path"] == "official"
                and archive["file"] == f'{row["event"]:06d}-official.bin',
                "Original archive belongs to exact call")
        require(tuple(archive["tensors"]) == TENSOR_LABELS
                and set(row["exact"]) == set(TENSOR_LABELS)
                and all(v is True for v in row["exact"].values()),
                "Every original hidden/logit/cache tensor and pairing")
        check_sha(archive["sha256"])
        offset = 0
        for label in TENSOR_LABELS:
            value = archive["tensors"][label]
            require(type(value["offset"]) is int and value["offset"] == offset
                    and type(value["bytes"]) is int and value["bytes"] > 0
                    and value["finite"] is True, "Dense original finite tensor extents")
            check_sha(value["sha256"])
            check_sha(value["full_sha256"])
            if label.startswith("cache-"):
                previous = bank["hashes"].get(label)
                require(value["shape"] == [4, end, 256] and value["dtype"] == "BF16"
                        and value["mode"] == "append" and value["prefix_valid"] is True
                        and value["bytes"] == (end-start)*2048,
                        "All original per-loop BF16 cache append extents")
                require(value["expected_previous"] == (
                    dict(length=start, sha256=previous, finite=True) if previous is not None else None),
                    "Exact prior original full-history descriptor")
                require(value["prefix_sha256"] == (previous if previous is not None else EMPTY_SHA),
                        "Original full-history prefix SHA chain")
                if start == 0:
                    require(value["sha256"] == value["full_sha256"],
                            "Fresh cache append is the complete original cache")
            else:
                shape, dtype, size = (([end-start, 2560], "BF16", (end-start)*2560*2)
                                     if label == "hidden"
                                     else ([head_rows, 131072], "F32", head_rows*131072*4))
                require(value["shape"] == shape and value["dtype"] == dtype
                        and value["mode"] == "full" and value["bytes"] == size
                        and value["sha256"] == value["full_sha256"],
                        "Complete original hidden/logit raw tensor")
            offset += value["bytes"]
        require(archive["bytes"] == offset, "Whole original archive extent")
        return archive

    def _accept(self, row):
        require(type(row["event"]) is int and row["event"] == self.events,
                "Exact original chronological event")
        handle = Handle(row["slot"], row["epoch"], row["request"])
        slot, epoch, request = handle.slot, handle.epoch, handle.request
        kind = row["kind"]
        action = dict(event=self.events, kind=kind, slot=slot, epoch=epoch, request=request,
                      native_call_required=False, native_rejection_required=False,
                      native_injection_required=False, native_execution=False)
        if kind == "start":
            old = self.banks.get(slot)
            require(epoch == self.epochs.get(slot, 0)+1, "Exact next original request epoch")
            require(row["reset_without_release"] is (old is not None)
                    and row["old_epoch"] == (old["epoch"] if old else None),
                    "Original release/reset distinction")
            require(row["other_slots_before"] == row["other_slots_after"],
                    "Original start preserves other requests")
            self.inventory(row["other_slots_before"], slot)
            self.epochs[slot] = epoch
            self.banks[slot] = dict(epoch=epoch, request=request, alive=True,
                                    length=0, hashes={}, fixture=None)
            self.starts += 1
        elif kind == "release":
            bank = self.banks[slot]
            require(bank["alive"] and (epoch, request) == (bank["epoch"], bank["request"]),
                    "Release exact live original owner")
            del self.banks[slot]
            self.releases += 1
        elif kind == "rejected_handle":
            bank = self.banks[slot]
            require(not bank["alive"] or (epoch, request) != (bank["epoch"], bank["request"]),
                    "Rejected original handle must actually be invalid")
            require(row["projection_calls"] == 0 and row["before"] == row["after"]
                    and set(row["errors"]) == {"official", "control"},
                    "Original rejection has no projection or state mutation")
            self.inventory(row["before"])
            self.rejections += 1
            action.update(native_rejection_required=True, label=row["label"],
                          original_errors=copy.deepcopy(row["errors"]), start=bank["length"],
                          end=bank["length"]+1, fixture=1)
        elif kind == "injected_failure":
            bank = self.banks[slot]
            require(bank["alive"] and (epoch, request) == (bank["epoch"], bank["request"]),
                    "Injected failure belongs to live original owner")
            require(row["layer"] == 22 and row["loop"] == 1
                    and row["failed_histories_cleared"] is True
                    and row["bound_caches_cleared"] is True
                    and row["other_slots_before"] == row["other_slots_after"],
                    "Original after-output layer22 loop1 failure and cleanup")
            self.inventory(row["other_slots_before"], slot)
            require(bank["length"] == 33 and bank["fixture"] == 1,
                    "Original injection after reset fixture at length33")
            partial = {str(key): 34 if key in INJECTION_ADVANCED else 33 for key in CACHE_IDS}
            observed = row["observed"]
            require(observed["official_partial_lengths"] == partial
                    and observed["control_partial_lengths"] == partial
                    and observed["official_projection_calls"] == 158
                    and observed["control_projection_calls"] == 253,
                    "Complete original partial-cache map and projection counts")
            bank["alive"] = False
            bank["hashes"] = {}
            self.injections += 1
            action.update(native_injection_required=True, after_output=True,
                          layer=22, loop=1, fixture=1, start=33, end=34,
                          original_observed=copy.deepcopy(observed))
        else:
            require(kind == "call", "Known original event kind")
            bank = self.banks[slot]
            require(bank["alive"] and (epoch, request) == (bank["epoch"], bank["request"]),
                    "Original call has live matching owner")
            start, end = row["start"], row["end"]
            require(type(start) is int and type(end) is int and start == bank["length"]
                    and 0 <= start < end <= 2060 and type(row["fixture"]) is int
                    and row["fixture"] in (0, 1),
                    "Original contiguous fixture positions")
            require(row["finite_and_prefixes_valid"] is True
                    and row["official_onednn_calls"] == 281 and row["control_onednn_calls"] == 449,
                    "Complete original successful producer call")
            require(bank["fixture"] is None or bank["fixture"] == row["fixture"],
                    "Original request keeps its fixture until reset")
            head_rows = 33 if row["phase"] == "full_causal" else 1
            require(isinstance(row["official_top1"], list)
                    and len(row["official_top1"]) == head_rows
                    and all(type(token) is int and 0 <= token < 131072 for token in row["official_top1"])
                    and row["official_top1"] == row["control_top1"],
                    "Original paired top1 metadata and head rows")
            archive = self._archive(row, bank)
            selected = self.selected.get(row["event"])
            needed = row["phase"] == "full_causal" or row.get("marked") is True
            require((selected is not None) == needed, "Every original marked/reference call selected")
            if selected is not None:
                require(all(row[k] == selected[k] for k in ("fixture", "slot", "epoch", "start", "end", "phase")),
                        "Original selected event identity and shape")
                expected_rows = list(self.reference_positions) if row["phase"] == "full_causal" else [end-start-1]
                require(selected["selected_local_rows"] == expected_rows,
                        "Original selected residual rows without shrinking forward")
                prior_subset = selected["expected_original_cache"]
                require(set(prior_subset) == {f"cache-{i}-{kind}" for i in (0, 2, 4) for kind in ("key", "value")}
                        and all(archive["tensors"][k] == value for k, value in prior_subset.items()),
                        "Original first-five annotations match full archive")
                if row["phase"] != "full_causal":
                    require(row["full_causal_vs_incremental"] == selected["original_FP32_metrics"],
                            "Original FP32 metrics retained without reclassification")
                self.selected_seen.append(row["event"])
            self.calls += 1
            self.raw_archive_bytes += archive["bytes"]
            self.full_reference_calls += row["phase"] == "full_causal"
            self.marked_calls += row.get("marked") is True
            bank.update(length=end, fixture=row["fixture"],
                        hashes={label: archive["tensors"][label]["full_sha256"] for label in CACHE_LABELS})
            action.update(native_call_required=True, fixture=row["fixture"], start=start, end=end,
                          phase=row["phase"], original_archive=copy.deepcopy(archive),
                          selected_local_rows=selected["selected_local_rows"] if selected else [],
                          original_FP32_metrics=selected["original_FP32_metrics"] if selected else None,
                          original_pointwise_failed=selected["original_pointwise_failed"] if selected else None,
                          original_top1=list(row["official_top1"]),
                          official_projection_calls_required=281)
        self.events += 1
        return action

    def accept(self, row):
        saved = copy.deepcopy(self.__dict__)
        try:
            return self._accept(row)
        except BaseException:
            self.__dict__.clear()
            self.__dict__.update(saved)
            raise

    def finish(self):
        require((self.events, self.calls, self.starts, self.releases,
                 self.rejections, self.injections, self.full_reference_calls, self.marked_calls)
                == (6200, 6185, 9, 2, 3, 1, 2, 87),
                "Every original event, native call, rejection and injection required")
        require(len(self.selected_seen) == 89 and set(self.selected_seen) == set(self.selected)
                and sum(row["original_pointwise_failed"] is True for row in self.selected.values()) == 7,
                "All original selected rows and seven failures retained")
        return dict(events=6200, native_successful_calls_required=6185,
                    native_rejections_required=3, native_injections_required=1,
                    selected_calls=89, original_pointwise_failures=7,
                    cache_tensor_labels=len(CACHE_LABELS), tensor_labels=len(TENSOR_LABELS),
                    original_raw_archive_bytes=self.raw_archive_bytes,
                    metadata_only=True, native_execution=False,
                    raw_tensor_bytes_verified=False, full_model_qualified=False,
                    new_speedup_claim=False)
