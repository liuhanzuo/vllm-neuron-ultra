"""Small metadata/source I/O with mocked weight/payload readers and old PIDs.

These checks exercise evidence association and rejection ordering, not real
checkpoint bytes, real process history, a full output freeze or native replay.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import dory_residual_saved_complete_v1 as combined


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2)+"\n")


class Pins(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.path = self.root/"value.json"
        dump(self.path, {"a": 1})
        self.reader = combined.MetadataPins(lambda: None, total_limit=128, single_limit=64)

    def test_real_json_and_jsonl_pins_are_retained_and_rechecked(self):
        self.assertEqual(self.reader.read(self.path, sha(self.path)), {"a": 1})
        journal = self.root/"journal.jsonl"
        journal.write_text('{"event": 0}\n{"event": 1}\n')
        self.assertEqual(self.reader.read(journal), [{"event": 0}, {"event": 1}])
        self.reader.recheck()
        self.assertEqual(self.reader.bytes, self.path.stat().st_size+journal.stat().st_size)

    def test_same_length_change_after_freeze_is_rejected(self):
        self.reader.read(self.path)
        self.path.write_text(self.path.read_text().replace("1", "2"))
        with self.assertRaisesRegex(RuntimeError, "cannot change"):
            self.reader.read(self.path)

    def test_external_hash_mismatch_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "external metadata pin"):
            self.reader.read(self.path, "0"*64)

    def test_duplicate_json_keys_are_rejected(self):
        self.path.write_text('{"a":1,"a":2}')
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            self.reader.read(self.path)

    def test_nonfinite_metadata_is_rejected(self):
        self.path.write_text('{"a":NaN}')
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            self.reader.read(self.path)

    def test_unterminated_journal_is_rejected(self):
        p = self.root/"events.jsonl"; p.write_text('{"event":0}')
        with self.assertRaisesRegex(RuntimeError, "termination"):
            self.reader.read(p)

    def test_symlink_is_rejected(self):
        p = self.root/"link.json"; p.symlink_to(self.path)
        with self.assertRaisesRegex(RuntimeError, "Canonical"):
            self.reader.read(p)

    def test_single_file_limit_is_enforced_before_read(self):
        self.path.write_text(" "*65)
        with self.assertRaisesRegex(RuntimeError, "reservation"):
            self.reader.read(self.path)

    def test_total_budget_is_enforced(self):
        reader = combined.MetadataPins(lambda: None, total_limit=32, single_limit=32)
        self.path.write_text('{"value":"abcdefghij"}')
        reader.read(self.path)
        p = self.root/"other.json"; p.write_bytes(self.path.read_bytes())
        with self.assertRaisesRegex(RuntimeError, "reservation"):
            reader.read(p)

    def test_failed_pin_reader_cannot_resume(self):
        with self.assertRaises(RuntimeError):
            self.reader.read(self.path, "0"*64)
        with self.assertRaisesRegex(RuntimeError, "cannot resume"):
            self.reader.read(self.path, sha(self.path))


class Complete(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.output = self.root/"native"; self.diag = self.output/"diagnostic"
        self.diag.mkdir(parents=True)
        self.worker = self.root/"worker.py"; self.worker.write_text("# synthetic never executed\n")
        plan = self.root/"plan.json"
        dump(plan, {"original_event_count": 6200, "original_saved_call_count": 6185})
        required_names = [
            "dory_residual_worker_preparation_v1.py", "dory_residual_replay_runtime_v1.py",
            "dory_residual_visit_capture_v1.py", "dory_residual_request_banks_v1.py",
            "dory_residual_replay_history_v1.py", "dory_residual_event_stream_v1.py",
            "dory_residual_output_archive_v1.py", "dory_long_delta_writer_v1.py",
            "dory_residual_saved_archives_v2.py", "dory_residual_saved_observations_v1.py",
            "dory_residual_saved_payloads_v2.py", "dory_residual_saved_complete_v1.py",
            "dory_checkpoint_loader_v1.py", "dory_generation_weight_provenance_v1.py",
            "dory_generation_execution_provenance_v1.py", "dory_generation_journal_replay_v1.py",
            "dory_generation_cache_replay_v1.py", "review_dory_long_matrix_saved_v3.py",
            "explain_dory_long_cache_differences_v1.py"]
        required = [str(combined.Q/name) for name in required_names]+[str(self.worker), str(plan)]
        self.spec = dict(schema=combined.SCHEMA, diagnostic_only=True, runtime_admitted=True,
            review_only_spec=False, output=str(self.output), environment=str(self.root/"venv"),
            worker=str(self.worker), reviewer=str(combined.Q/"dory_residual_saved_complete_v1.py"),
            executor=str(self.worker), expected_original_raw_bytes=12345,
            selection_plan=dict(path=str(plan), sha256=sha(plan)),
            source_sha256={name:sha(Path(name)) for name in required},
            essential_sources=required, source_read_cap_bytes=8*2**20,
            payload_review_budget=dict(expected_original_raw_bytes=12345,
                auxiliary_raw_bytes=combined.RESIDUAL_BYTES+combined.INJECTION_BYTES,
                combined_cache_bytes=2*4*2060*112*4*256*2,
                selected_reference_bytes=2*33*224*2560*2),
            budget=dict(nvme_new_bytes=2**31, auxiliary_raw_bytes=combined.RESIDUAL_BYTES+512*2**20,
                        single_file_bytes=2**31), checkpoint={"path": "/not-read/synthetic-weights"})
        self.sp = self.root/"producer-spec.json"; dump(self.sp, self.spec)
        self.parent = dict(pid=100, start_ticks="1000", ppid=1, argv=["synthetic-parent"])
        self.child_owner = dict(pid=200, start_ticks="1001", ppid=100)
        command = [self.spec["environment"]+"/bin/python", "-B", str(self.worker), "--spec", str(self.sp),
                   "--admission", str(self.output/"admission.json")]
        self.child_owner["argv"] = command
        records = {
            "admission.json": dict(spec_sha256=sha(self.sp), review=False, performance_measurement=False,
                owner=self.parent, command=command, budget=self.spec["budget"]),
            "child.json": dict(command=command, pid=200),
            "child-owner.json": self.child_owner,
            "child-exit.json": dict(returncode=0, child_pid=200),
            "result.json": dict(complete=True, review=False, child_exit=0, actual_self_exit_claimed=False,
                full_model_qualified=False, new_speedup_claim=False)}
        for name, value in records.items():
            dump(self.output/name, value)
        self.external = self.root/"actual-exit.json"
        dump(self.external, dict(session_id=123, exit_code=0, tool_chunk="abc123",
            actual_external_exit_observed=True, owner=self.parent, child_owner=self.child_owner))
        for name in ("loader-proof.json", "loaded-parameters.json", "model.json", "tokenizer.json"):
            dump(self.diag/name, {})
        paths = [self.sp, self.external, plan]+list(self.output.glob("*.json"))+list(self.diag.glob("*.json"))
        self.contract = dict(schema=combined.CONTRACT_SCHEMA, raw_payloads_authenticated=False,
            producer_spec=dict(path=str(self.sp), sha256=sha(self.sp)),
            actual_exit=dict(path=str(self.external), sha256=sha(self.external)),
            metadata_pins={str(p):dict(bytes=p.stat().st_size, sha256=sha(p)) for p in paths},
            binary_identities={}, output_directory_identity=combined.identity(self.diag.stat()))
        self.calls = []
        self.checkpoint_result = dict(checkpoint_bytes_independently_authenticated=True,
            checkpoint_bytes_read=7950515472, original_tensors=507,
            mapped_physical_parameters=399, complete_packed_mapping_verified=True)
        self.payload_result = dict(complete=True, original_events=6200, original_successful_calls=6185,
            original_pointwise_failures=7, both_original_and_new_successful_payloads_verified=True,
            all_producer_output_names_accounted=True, original_top1_recomputed=False,
            original_FP32_metrics_recomputed=False, production_lifecycle_qualified=False,
            root_cause_established=False, full_model_qualified=False, new_speedup_claim=False)

    def repin(self, path):
        self.contract["metadata_pins"][str(path)] = dict(bytes=path.stat().st_size, sha256=sha(path))

    def run_review(self, process=None):
        def checkpoint(root, pins, checkpoint_identity, loader_sha, *, guard):
            guard(); self.calls.append("checkpoint")
            self.assertEqual(root, self.diag)
            self.assertEqual(checkpoint_identity, self.spec["checkpoint"])
            self.assertEqual(loader_sha, sha(combined.Q/"dory_checkpoint_loader_v1.py"))
            return self.checkpoint_result
        def payload(stream, source, pins, budget, guard):
            guard(); self.calls.append("payload")
            self.assertEqual(source, self.diag)
            self.assertEqual(budget, self.spec["payload_review_budget"])
            return copy.deepcopy(self.payload_result)
        with patch.object(combined, "current_process", process or (lambda pid: None)), \
             patch.object(combined.weights, "review_saved", checkpoint), \
             patch.object(combined, "OriginalEventStream", lambda pin, guard: {"synthetic": pin}), \
             patch.object(combined, "review_payloads", payload):
            return combined.review_complete(self.contract, lambda: None)

    def test_real_metadata_source_and_receipt_checks_precede_stub_payload(self):
        result = self.run_review()
        self.assertEqual(self.calls, ["checkpoint", "payload"])
        self.assertTrue(result["recorded_source_checkpoint_exit_bound"])
        self.assertFalse(result["actual_loaded_libraries_JIT_verified"])
        self.assertFalse(result["full_model_qualified"])

    def test_changed_receipt_bytes_block_weight_and_payload_reads(self):
        p = self.output/"child-exit.json"; p.write_text(p.read_text().replace("0", "1", 1))
        with self.assertRaises(ValueError):
            self.run_review()
        self.assertEqual(self.calls, [])

    def test_changed_worker_source_blocks_weight_and_payload_reads(self):
        self.worker.write_text("# modified\n")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.run_review()
        self.assertEqual(self.calls, [])

    def test_still_live_exact_owner_blocks_review(self):
        def live(pid):
            return dict(pid=pid, start_ticks="1000" if pid == 100 else "1001", state="S")
        with self.assertRaisesRegex(ValueError, "still live"):
            self.run_review(live)
        self.assertEqual(self.calls, [])

    def test_actual_external_exit_failure_blocks_review(self):
        record = json.loads(self.external.read_text()); record["exit_code"] = 1
        dump(self.external, record); self.repin(self.external)
        self.contract["actual_exit"]["sha256"] = sha(self.external)
        with self.assertRaisesRegex(ValueError, "actual tool-session exit"):
            self.run_review()
        self.assertEqual(self.calls, [])

    def test_checkpoint_incomplete_mapping_blocks_payload(self):
        self.checkpoint_result["complete_packed_mapping_verified"] = False
        with self.assertRaisesRegex(RuntimeError, "checkpoint tensor"):
            self.run_review()
        self.assertEqual(self.calls, ["checkpoint"])

    def test_wrong_payload_scope_cannot_be_promoted(self):
        self.payload_result["original_successful_calls"] = 6184
        with self.assertRaisesRegex(RuntimeError, "Full saved payload scope"):
            self.run_review()

    def test_payload_may_not_claim_model_qualification(self):
        self.payload_result["full_model_qualified"] = True
        with self.assertRaisesRegex(RuntimeError, "limited verdicts"):
            self.run_review()

    def test_new_output_file_after_freeze_is_rejected(self):
        (self.diag/"extra.bin").write_bytes(b"ab")
        with self.assertRaisesRegex(RuntimeError, "remains frozen"):
            self.run_review()
        self.assertNotIn("payload", self.calls)

    def test_failure_receipt_blocks_review(self):
        dump(self.output/"failure.json", {})
        with self.assertRaisesRegex(RuntimeError, "failed producer"):
            self.run_review()
        self.assertEqual(self.calls, [])

    def test_missing_external_exit_pin_is_rejected(self):
        self.contract["metadata_pins"].pop(str(self.external))
        with self.assertRaises(KeyError):
            self.run_review()
        self.assertEqual(self.calls, [])

    def test_missing_full_runtime_source_pin_is_rejected(self):
        spec = copy.deepcopy(self.spec)
        spec["essential_sources"].remove(str(combined.Q/"dory_residual_replay_runtime_v1.py"))
        with self.assertRaisesRegex(RuntimeError, "dependencies pinned"):
            combined.validate_producer(spec)

    def test_underbudget_two_full_reconstructed_states_is_rejected(self):
        spec = copy.deepcopy(self.spec); spec["payload_review_budget"]["combined_cache_bytes"] //= 2
        with self.assertRaisesRegex(RuntimeError, "cache and selected-reference"):
            combined.validate_producer(spec)

    def test_old_first_five_schema_is_rejected(self):
        spec = copy.deepcopy(self.spec); spec["schema"] = "dory_first5_propagation_diagnostic_v1"
        with self.assertRaisesRegex(RuntimeError, "complete residual producer"):
            combined.validate_producer(spec)

    def test_guard_rejects_before_freezing_any_metadata(self):
        def deny():
            raise RuntimeError("no serial ownership")
        with self.assertRaisesRegex(RuntimeError, "serial ownership"):
            combined.freeze(None, None, deny)

    def test_incomplete_output_directory_cannot_be_frozen(self):
        with patch.object(combined, "current_process", lambda pid: None):
            with self.assertRaisesRegex(RuntimeError, "Every original replay JSON"):
                combined.freeze(self.contract["producer_spec"], self.contract["actual_exit"], lambda: None)

    def test_no_native_framework_imported(self):
        self.assertFalse({"torch", "numpy", "vllm", "transformers", "tokenizers"}.intersection(
            name.split(".")[0] for name in sys.modules))


if __name__ == "__main__":
    unittest.main(verbosity=2)
