"""Prepared complete residual producer provenance and saved-byte review.

No launch entrypoint or automatic admission. A future exclusive reviewer must
receive an actual producer-spec pin and externally recorded process exit.
Frozen output metadata is reauthenticated before checkpoint/payload reads.
"""
import hashlib
import json
import os
from pathlib import Path
import stat

from dory_generation_execution_provenance_v1 import (
    inspect_receipts, current_process, review_source_pins)
from dory_generation_journal_replay_v1 import SavedMetadata
from dory_residual_event_stream_v1 import OriginalEventStream
from dory_residual_saved_payloads_v2 import review_payloads
from dory_residual_saved_observations_v1 import INJECTION_BYTES
import dory_generation_weight_provenance_v1 as weights


Q = Path(__file__).resolve().parent
SCHEMA = "dory_residual_complete_replay_diagnostic_v1"
CONTRACT_SCHEMA = "dory_residual_post_exit_contract_v1"
RESIDUAL_BYTES = 175472640
BINARY_FILES = 6185+89+1
JSON_FILES = 6200+1+1+4


def require(value, message):
    if not value:
        raise RuntimeError(message)


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "No duplicate frozen metadata keys")
        result[key] = value
    return result


def invalid(value):
    raise ValueError("Nonfinite metadata: "+value)


def parse(raw):
    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)


def identity(value):
    return {key: getattr(value, key) for key in
            ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")}


class MetadataPins:
    def __init__(self, guard, *, total_limit=1536*2**20, single_limit=64*2**20):
        require(type(total_limit) is int and type(single_limit) is int
                and 0 < single_limit <= 64*2**20 and single_limit <= total_limit <= 1536*2**20,
                "Explicit bounded metadata pin reservation")
        self.guard, self.total_limit, self.single_limit = guard, total_limit, single_limit
        self.pins, self.bytes = {}, 0
        self.poisoned = False

    def read(self, path, expected=None):
        require(not self.poisoned, "Interrupted metadata freeze cannot resume")
        try:
            return self._read(Path(path), expected)
        except BaseException:
            self.poisoned = True
            raise

    def _read(self, path, expected):
        self.guard()
        require(path.is_absolute() and path.resolve() == path and not path.is_symlink()
                and path.suffix in (".json", ".jsonl"), "Canonical ordinary metadata path")
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
                    and 0 < before.st_size <= self.single_limit
                    and (str(path) in self.pins or self.bytes+before.st_size <= self.total_limit),
                    "Ordinary metadata within complete byte reservation")
            data = stream.read(before.st_size+1)
            after = os.fstat(stream.fileno())
        require(len(data) == before.st_size
                and identity(before) == identity(after) == identity(path.lstat()),
                "Frozen metadata identity remains stable")
        digest = hashlib.sha256(data).hexdigest()
        require(expected is None or digest == expected, "Exact external metadata pin")
        pin = dict(bytes=len(data), sha256=digest)
        require(str(path) not in self.pins or self.pins[str(path)] == pin,
                "Frozen metadata cannot change")
        if path.suffix == ".jsonl":
            require(data.endswith(b"\n"), "Complete frozen journal termination")
            value = [parse(line) for line in data.splitlines()]
        else:
            value = parse(data)
        if str(path) not in self.pins:
            self.pins[str(path)] = pin
            self.bytes += len(data)
        return value

    def recheck(self):
        for name, pin in list(self.pins.items()):
            self.read(name, pin["sha256"])


def validate_producer(spec):
    require(spec["schema"] == SCHEMA and spec["diagnostic_only"] is True
            and spec["runtime_admitted"] is True and spec["review_only_spec"] is False,
            "Exact admitted complete residual producer")
    root = Path(spec["output"])
    require(root.is_absolute() and root.resolve() == root, "Canonical complete producer output")
    raw = spec["expected_original_raw_bytes"]
    budget = spec["payload_review_budget"]
    require(type(raw) is int and raw > 0 and budget["expected_original_raw_bytes"] == raw
            and budget["auxiliary_raw_bytes"] >= RESIDUAL_BYTES+INJECTION_BYTES
            and type(budget["combined_cache_bytes"]) is int
            and budget["combined_cache_bytes"] >= 2*4*2060*112*4*256*2
            and budget["selected_reference_bytes"] >= 2*33*224*2560*2,
            "Full original/new cache and selected-reference reservation")
    require(spec["budget"]["nvme_new_bytes"] >= raw + spec["budget"]["auxiliary_raw_bytes"] + 2**30
            and spec["budget"]["auxiliary_raw_bytes"] >= RESIDUAL_BYTES+512*2**20,
            "Complete producer output plus failure and metadata storage")
    plan = spec["selection_plan"]
    require(spec["source_sha256"][plan["path"]] == plan["sha256"],
            "Original full selection plan is source-pinned")
    required = [str(Q/name) for name in (
        "dory_residual_worker_preparation_v1.py", "dory_residual_replay_runtime_v1.py",
        "dory_residual_visit_capture_v1.py", "dory_residual_request_banks_v1.py",
        "dory_residual_replay_history_v1.py", "dory_residual_event_stream_v1.py",
        "dory_residual_output_archive_v1.py", "dory_long_delta_writer_v1.py",
        "dory_residual_saved_archives_v2.py", "dory_residual_saved_observations_v1.py",
        "dory_residual_saved_payloads_v2.py", "dory_residual_saved_complete_v1.py",
        "dory_checkpoint_loader_v1.py", "dory_generation_weight_provenance_v1.py",
        "dory_generation_execution_provenance_v1.py", "dory_generation_journal_replay_v1.py",
        "dory_generation_cache_replay_v1.py", "review_dory_long_matrix_saved_v3.py",
        "explain_dory_long_cache_differences_v1.py")]
    required += [spec["worker"], spec["reviewer"], spec["executor"], plan["path"]]
    require(set(required) <= set(spec["essential_sources"]) <= set(spec["source_sha256"]),
            "All full producer and independent reviewer dependencies pinned")
    return spec


def terminal_review(spec_ref, spec, read):
    root = Path(spec["output"])
    names = ("admission.json", "child.json", "child-owner.json", "child-exit.json", "result.json")
    records = [read(root/name) for name in names]
    command = [str(Path(spec["environment"])/"bin/python"), "-B", spec["worker"],
               "--spec", spec_ref["path"], "--admission", str(root/"admission.json")]
    require(records[0]["command"] == command and records[0]["budget"] == spec["budget"],
            "Actual worker command and complete budget equal pinned producer spec")
    return records


def freeze(spec_ref, actual_exit_ref, guard):
    guard()
    pins = MetadataPins(guard)
    spec = validate_producer(pins.read(spec_ref["path"], spec_ref["sha256"]))
    root, diag = Path(spec["output"]), Path(spec["output"])/"diagnostic"
    require(not any((root/n).exists() for n in ("failure.json", "cleanup-pending.json"))
            and not (diag/"residual-replay-failure.json").exists(),
            "Failed producer cannot be frozen as complete")
    external = pins.read(actual_exit_ref["path"], actual_exit_ref["sha256"])
    records = terminal_review(spec_ref, spec, pins.read)
    terminal = inspect_receipts(spec_ref["sha256"], *records, external, process_lookup=current_process)
    plan = pins.read(spec["selection_plan"]["path"], spec["selection_plan"]["sha256"])
    require(plan["original_event_count"] == 6200 and plan["original_saved_call_count"] == 6185,
            "Complete original selection scope")
    binary, json_count, journal_count = {}, 0, 0
    directory_before = diag.stat(follow_symlinks=False)
    require(stat.S_ISDIR(directory_before.st_mode) and diag.resolve() == diag,
            "Original ordinary diagnostic directory")
    for path in sorted(diag.iterdir()):
        guard()
        entry = path.lstat()
        require(stat.S_ISREG(entry.st_mode) and entry.st_nlink == 1,
                "Only ordinary original output files")
        if path.suffix in (".json", ".jsonl"):
            pins.read(path)
            if path.suffix == ".json":
                json_count += 1
            else:
                require(path.name == "residual-events.jsonl", "Only complete residual journal")
                journal_count += 1
        else:
            require(path.suffix == ".bin" and 0 < entry.st_size <= spec["budget"]["single_file_bytes"],
                    "Bounded complete producer binary file")
            binary[path.name] = identity(entry)
    require((json_count, journal_count, len(binary)) == (JSON_FILES, 1, BINARY_FILES),
            "Every original replay JSON, journal and raw file")
    raw_bytes = sum(row["st_size"] for row in binary.values())
    require(raw_bytes == spec["expected_original_raw_bytes"]+RESIDUAL_BYTES+INJECTION_BYTES,
            "Exact original-successful, selected-residual and failure byte totals")
    pins.recheck()
    require(identity(directory_before) == identity(diag.stat(follow_symlinks=False)),
            "Output directory unchanged while freezing")
    return dict(schema=CONTRACT_SCHEMA, producer_spec=spec_ref, actual_exit=actual_exit_ref,
        metadata_pins=pins.pins, metadata_bytes=pins.bytes, binary_identities=binary,
        output_directory_identity=identity(directory_before), raw_file_count=len(binary), raw_bytes=raw_bytes,
        terminal_review=terminal, raw_payloads_authenticated=False,
        original_pointwise_failures=7, full_model_qualified=False, new_speedup_claim=False)


def review_complete(contract, guard):
    guard()
    require(contract["schema"] == CONTRACT_SCHEMA and contract["raw_payloads_authenticated"] is False,
            "Frozen complete residual post-exit contract awaiting byte review")
    pins = contract["metadata_pins"]
    def read(path, maximum=64*2**20):
        path = Path(path)
        require(str(path) in pins, "Every metadata input frozen after actual producer exit")
        return SavedMetadata(path.parent, {path.name: pins[str(path)]}, guard).json(path.name, maximum)
    spec_ref, external_ref = contract["producer_spec"], contract["actual_exit"]
    require(pins[spec_ref["path"]]["sha256"] == spec_ref["sha256"]
            and pins[external_ref["path"]]["sha256"] == external_ref["sha256"],
            "Externally pinned producer specification and actual exit")
    spec = validate_producer(read(spec_ref["path"]))
    root, diag = Path(spec["output"]), Path(spec["output"])/"diagnostic"
    require(not any((root/n).exists() for n in ("failure.json", "cleanup-pending.json"))
            and not (diag/"residual-replay-failure.json").exists(), "No failed producer promotion")
    records = terminal_review(spec_ref, spec, read)
    terminal = inspect_receipts(spec_ref["sha256"], *records, read(external_ref["path"]),
                                process_lookup=current_process)
    sources = review_source_pins(spec["source_sha256"], spec["essential_sources"],
                                 max_total_bytes=spec["source_read_cap_bytes"], guard=guard)
    checkpoint = weights.review_saved(diag,
        {name: pins[str(diag/name)] for name in ("loader-proof.json", "loaded-parameters.json", "model.json")},
        spec["checkpoint"], sources["files"][str(Q/"dory_checkpoint_loader_v1.py")]["sha256"], guard=guard)
    require(checkpoint["checkpoint_bytes_independently_authenticated"] is True
            and checkpoint["checkpoint_bytes_read"] == 7950515472
            and checkpoint["original_tensors"] == 507 and checkpoint["mapped_physical_parameters"] == 399
            and checkpoint["complete_packed_mapping_verified"] is True,
            "Every original BF16 checkpoint tensor and complete physical parameter mapping")
    plan_ref = spec["selection_plan"]
    require(pins[plan_ref["path"]]["sha256"] == plan_ref["sha256"], "Same frozen original selection plan")
    read(plan_ref["path"])
    output_pins = {Path(name).name: pin for name, pin in pins.items() if Path(name).parent == diag}
    def binary_identity_check():
        require(identity(diag.stat(follow_symlinks=False)) == contract["output_directory_identity"],
                "Complete output directory remains frozen")
        actual = {}
        for path in diag.iterdir():
            guard()
            info = path.lstat()
            require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "Ordinary frozen output files")
            if path.suffix == ".bin":
                actual[path.name] = identity(info)
        require(actual == contract["binary_identities"], "All original frozen binary identities unchanged")
    binary_identity_check()
    stream = OriginalEventStream(plan_ref, guard)
    result = review_payloads(stream, diag, output_pins, spec["payload_review_budget"], guard)
    require(result["complete"] is True and result["original_events"] == 6200
            and result["original_successful_calls"] == 6185 and result["original_pointwise_failures"] == 7
            and result["both_original_and_new_successful_payloads_verified"] is True
            and result["all_producer_output_names_accounted"] is True
            and all(result[k] is False for k in (
                "original_top1_recomputed", "original_FP32_metrics_recomputed",
                "production_lifecycle_qualified", "root_cause_established", "full_model_qualified",
                "new_speedup_claim")), "Full saved payload scope and original limited verdicts")
    binary_identity_check()
    result.update(schema="dory_residual_combined_saved_review_v1",
        source_file_review=sources, checkpoint_review=checkpoint, terminal_review=terminal,
        recorded_source_checkpoint_exit_bound=True,
        actual_loaded_libraries_JIT_verified=False, complete_history_drain_verified_here=False)
    return result
