# SPDX-License-Identifier: Apache-2.0
"""Stdlib-only admission tests with fake tiny artifacts; no native execution."""
import builtins
import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import nullcontext
from vllm_neuron.models.lfm3 import register, artifacts, identity
from vllm_neuron.models.lfm3.identity import ARTIFACT_MANIFEST, AdmissionError, selected_variant
from vllm_neuron.models.lfm3.provenance import SDK_SOURCE_FILES, NKI_BINARY_MODULES


def changed_saved_body(configuration, mutation):
    value = copy.deepcopy(configuration)
    if mutation == "truncation":
        value["neuron_config"]["allow_input_truncation"] = True
    elif mutation == "new_top_level_field":
        value["unknown_future_option"] = True
    elif mutation == "missing_top_level_field":
        del value["batch_size"]
    elif mutation == "missing_nested_field":
        del value["neuron_config"]["allow_input_truncation"]
    elif mutation == "json_type_change":
        value["neuron_config"]["allow_input_truncation"] = 0
    else:
        raise AssertionError(mutation)
    return value


BODY_MUTATIONS = ("truncation", "new_top_level_field", "missing_top_level_field",
                  "missing_nested_field", "json_type_change")


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "compiled"
        self.identity = {"variant": "reference", "scope": {"context": 2048},
                         "sources": {"model.py": "source-a"}, "model": {"revision": "model-a"},
                         "software": {"compiler": "compiler-a"},
                         "configuration": {"metadata": None, "batch_size": 4,
                                           "neuron_config": {"allow_input_truncation": False,
                                                             "quantize_clamp_bound": float("inf")}}}
        self.calls = []
        calls = self.calls
        identity = self.identity
        class Model:
            def __init__(self, *args):
                calls.append("construct")
            def compile(self, path):
                calls.append("compile")
                (Path(path) / "model.pt").write_text("fake model")
                configuration = copy.deepcopy(identity["configuration"])
                configuration["metadata"] = {"lfm3_ultra": identity}
                (Path(path) / "neuron_config.json").write_text(json.dumps(configuration))
            def load(self, path):
                calls.append("load-new")
        self.model = Model
        self.owner = SimpleNamespace(_load_compiled_model=lambda *args: calls.append("load-existing"))
        self.config = SimpleNamespace(to_json_string=lambda: '{"metadata": null}')
        self.enter(patch.dict(os.environ, {"NEURON_COMPILED_ARTIFACTS": str(self.path),
                                          "NEURON_LFM3_ALLOW_COMPILE": "1"}))
        self.enter(patch.object(artifacts, "build_identity", side_effect=lambda *args: copy.deepcopy(self.identity)))
        self.enter(patch.object(artifacts, "serial_compiler", side_effect=nullcontext))

    def enter(self, context):
        value = context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        return value

    def invoke(self):
        return artifacts.load_or_compile(self.owner, "unused-view", self.model, self.config, {})

    def complete(self):
        self.assertTrue(self.invoke()[0])
        self.assertEqual(self.calls, ["construct", "compile", "load-new"])
        self.calls.clear()

    def snapshot(self):
        return {str(p.relative_to(self.path)): p.read_bytes() for p in self.path.rglob("*") if p.is_file()}

    def test_complete_load_does_not_compile_or_mutate(self):
        self.complete()
        before = self.snapshot()
        self.invoke()
        self.assertEqual(self.calls, ["load-existing"])
        self.assertEqual(before, self.snapshot())

    def test_identity_components_reject_before_load(self):
        self.complete()
        before = self.snapshot()
        for field in self.identity:
            with self.subTest(field=field):
                changed = copy.deepcopy(self.identity)
                changed[field] = {"unexpected": "identity"}
                with patch.object(artifacts, "build_identity", return_value=changed):
                    with self.assertRaises(AdmissionError): self.invoke()
                self.assertEqual(self.calls, [])
                self.assertEqual(before, self.snapshot())

    def test_corrupted_artifact_rejects_without_rebuild(self):
        self.complete()
        (self.path / "model.pt").write_text("changed")
        before = self.snapshot()
        with self.assertRaises(AdmissionError): self.invoke()
        self.assertEqual(self.calls, [])
        self.assertEqual(before, self.snapshot())

    def test_missing_invalid_incomplete_manifest_retains_directory(self):
        self.path.mkdir()
        (self.path / "keep.txt").write_text("unrelated existing file")
        for contents in (None, "invalid-json", '{"complete":false}', "[]"):
            with self.subTest(contents=contents):
                if contents is not None: (self.path / ARTIFACT_MANIFEST).write_text(contents)
                before = self.snapshot()
                with self.assertRaises(AdmissionError): self.invoke()
                self.assertEqual(self.calls, [])
                self.assertEqual(before, self.snapshot())

    def test_opt_in_precedes_directory_creation(self):
        with patch.dict(os.environ, {"NEURON_LFM3_ALLOW_COMPILE": "0"}):
            with self.assertRaises(AdmissionError): self.invoke()
        self.assertFalse(self.path.exists())
        self.assertEqual(self.calls, [])

    def test_compile_failure_retains_unadmitted_partial_directory(self):
        def fail(model, path):
            (Path(path) / "partial.txt").write_text("retain for diagnosis")
            raise ValueError("fake compile failure")
        with patch.object(self.model, "compile", fail):
            with self.assertRaises(ValueError): self.invoke()
        self.assertFalse((self.path / ARTIFACT_MANIFEST).exists())
        self.calls.clear()
        before = self.snapshot()
        with self.assertRaises(AdmissionError): self.invoke()
        self.assertEqual(self.calls, [])
        self.assertEqual(before, self.snapshot())

    def test_load_failure_does_not_publish_completion(self):
        with patch.object(self.model, "load", side_effect=RuntimeError("fake load failure")):
            with self.assertRaises(RuntimeError): self.invoke()
        self.assertFalse((self.path / ARTIFACT_MANIFEST).exists())

    def test_existing_load_valueerror_never_recompiles(self):
        self.complete()
        before = self.snapshot()
        def fail(*args): raise ValueError("fake load failure")
        self.owner._load_compiled_model = fail
        with self.assertRaises(ValueError): self.invoke()
        self.assertEqual(self.calls, [])
        self.assertEqual(before, self.snapshot())

    def test_symlink_directory_is_rejected(self):
        target = self.path.parent / "elsewhere"
        target.mkdir()
        self.path.symlink_to(target, target_is_directory=True)
        with self.assertRaises(AdmissionError): self.invoke()
        self.assertEqual(self.calls, [])
        self.assertEqual(list(target.iterdir()), [])

    def test_changed_embedded_identity_rejects_even_with_refreshed_file_list(self):
        self.complete()
        config = self.path / "neuron_config.json"
        config.write_text('{"metadata":{"lfm3_ultra":{"variant":"wrong"}}}')
        manifest_path = self.path / ARTIFACT_MANIFEST
        manifest = json.loads(manifest_path.read_text())
        manifest["files"] = artifacts.artifact_files(self.path)
        manifest_path.write_text(json.dumps(manifest))
        before = self.snapshot()
        with self.assertRaises(AdmissionError): self.invoke()
        self.assertEqual(self.calls, [])
        self.assertEqual(before, self.snapshot())

    def test_saved_body_rejects_with_valid_metadata_and_refreshed_file_list(self):
        self.complete()
        config_path = self.path / "neuron_config.json"
        original = json.loads(config_path.read_text())
        for mutation in BODY_MUTATIONS:
            with self.subTest(mutation=mutation):
                config_path.write_text(json.dumps(changed_saved_body(original, mutation)))
                manifest_path = self.path / ARTIFACT_MANIFEST
                manifest = json.loads(manifest_path.read_text())
                manifest["files"] = artifacts.artifact_files(self.path)
                manifest_path.write_text(json.dumps(manifest))
                before = self.snapshot()
                with self.assertRaisesRegex(AdmissionError, "configuration body differs"):
                    self.invoke()
                self.assertEqual(self.calls, [])
                self.assertEqual(before, self.snapshot())

    def test_new_build_body_changes_reject_before_model_load_or_manifest(self):
        original_compile = self.model.compile
        base = self.path
        for mutation in BODY_MUTATIONS:
            with self.subTest(mutation=mutation):
                self.path = base.with_name("new-" + mutation)
                self.calls.clear()
                def wrong_compile(model, path):
                    original_compile(model, path)
                    config_path = Path(path) / "neuron_config.json"
                    changed = changed_saved_body(json.loads(config_path.read_text()), mutation)
                    config_path.write_text(json.dumps(changed))
                with patch.dict(os.environ, {"NEURON_COMPILED_ARTIFACTS": str(self.path)}):
                    with patch.object(self.model, "compile", wrong_compile):
                        with self.assertRaisesRegex(AdmissionError, "configuration body differs"):
                            self.invoke()
                self.assertEqual(self.calls, ["construct", "compile"])
                self.assertFalse((self.path / ARTIFACT_MANIFEST).exists())
                self.assertTrue((self.path / "neuron_config.json").exists())


class ScopeAndSdkIdentityTests(unittest.TestCase):
    def fixture(self):
        model = copy.deepcopy(identity.EXPECTED_MODEL)
        neuron = dict(tp_degree=2, batch_size=4, max_batch_size=4,
                      tkg_batch_size=4, ctx_batch_size=1, seq_len=2048,
                      max_length=2048, max_context_length=2048,
                      padding_side="right", apply_seq_ids_mask=True,
                      is_continuous_batching=True, enable_bucketing=True,
                      save_sharded_checkpoint=False, skip_sharding=False,
                      skip_warmup=False, logical_nc_config=1, start_rank_id=0,
                      local_ranks_size=2, ep_degree=1, world_size=2,
                      torch_dtype="bfloat16", buckets=[2048])
        return model, neuron

    def test_scope_rejects_each_changed_or_missing_required_field(self):
        model, neuron = self.fixture()
        identity.validate_scope(model, neuron)
        for key in model:
            for changed in ({**model, key: None}, {k: v for k, v in model.items() if k != key}):
                with self.subTest(model_field=key), self.assertRaises(AdmissionError):
                    identity.validate_scope(changed, neuron)
        for key in neuron:
            for changed in ({**neuron, key: None}, {k: v for k, v in neuron.items() if k != key}):
                with self.subTest(neuron_field=key), self.assertRaises(AdmissionError):
                    identity.validate_scope(model, changed)
        for key, value in (("tp_degree", 1), ("apply_seq_ids_mask", 1), ("on_device_sampling_config", {}),
                           ("is_prefix_caching", True), ("is_chunked_prefill", True), ("torch_dtype", "float32")):
            with self.subTest(option=key), self.assertRaises(AdmissionError):
                identity.validate_scope(model, {**neuron, key: value})

    def test_expert_and_world_parallelism_are_pinned(self):
        model, original = self.fixture()
        identity.validate_scope(model, original)
        for changes in ({"ep_degree": 2}, {"world_size": 4},
                        {"ep_degree": 2, "world_size": 4, "local_ranks_size": 2}):
            with self.subTest(changes=changes):
                with self.assertRaises(AdmissionError):
                    identity.validate_scope(model, {**original, **changes})
        for missing in ("ep_degree", "world_size"):
            changed = dict(original)
            del changed[missing]
            with self.subTest(missing=missing):
                with self.assertRaises(AdmissionError):
                    identity.validate_scope(model, changed)

    def test_same_version_bucket_or_tp_source_change_changes_sdk_identity(self):
        required = [
            ("neuronx-distributed-inference", "neuronx_distributed_inference/modules/autobucketing.py"),
            ("neuronx-distributed", "neuronx_distributed/parallel_layers/mappings.py"),
            ("neuronx-distributed", "neuronx_distributed/parallel_layers/utils.py"),
        ]
        self.assertTrue(set(required) <= set(SDK_SOURCE_FILES))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for distribution, relative in SDK_SOURCE_FILES:
                path = root / distribution / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# fake SDK source\n")
            binary_files = []
            for module in NKI_BINARY_MODULES:
                relative = module + ".cpython-310-test.so"
                binary_files.append(Path(relative))
                path = root / "neuronx-cc" / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fake native module identity, never imported")
            def distribution(name):
                return SimpleNamespace(files=binary_files if name == "neuronx-cc" else [],
                                       locate_file=lambda relative: root / name / relative)
            with patch.object(identity.importlib.metadata, "version", side_effect=identity.SOFTWARE.__getitem__):
                with patch.object(identity.importlib.metadata, "distribution", side_effect=distribution):
                    original = identity.software_identity()
                    for name, relative in required:
                        path = root / name / relative
                        path.write_text("# same-version changed SDK source\n")
                        changed = identity.software_identity()
                        self.assertEqual(changed["versions"], original["versions"])
                        self.assertNotEqual(changed["source_sha256"], original["source_sha256"])
                        key = name + "/" + relative
                        self.assertNotEqual(changed["source_sha256"][key], original["source_sha256"][key])
                        path.write_text("# fake SDK source\n")


class CanonicalConfigurationTests(unittest.TestCase):
    def fixture(self):
        return {"metadata": None, "neuron_config": {
            "quantize_clamp_bound": float("inf"), "quantized": False}}

    def test_documented_positive_infinity_survives_sdk_style_json_roundtrip(self):
        expected = self.fixture()
        # SDK InferenceConfig.to_json_string uses ordinary json.dumps, which
        # intentionally preserves the documented positive-Infinity default.
        saved = json.loads(json.dumps(expected))
        self.assertEqual(identity.canonical_configuration(expected),
                         identity.canonical_configuration(saved))

    def test_sentinel_cannot_collide_with_any_user_container_or_scalar(self):
        expected = identity.canonical_configuration(self.fixture())
        for replacement in ("sdk-positive-infinity", ["sdk-positive-infinity"],
                            {"type": "sdk-positive-infinity"}, 1, True, 1.0,
                            None, "Infinity", 1e30):
            with self.subTest(replacement=replacement):
                changed = self.fixture()
                changed["neuron_config"]["quantize_clamp_bound"] = replacement
                self.assertNotEqual(identity.canonical_configuration(changed), expected)

    def test_other_nonfinites_rejected_in_known_unknown_and_nested_fields(self):
        for value in (float("nan"), float("-inf")):
            with self.subTest(known_field=repr(value)):
                changed = self.fixture()
                changed["neuron_config"]["quantize_clamp_bound"] = value
                with self.assertRaises(AdmissionError):
                    identity.canonical_configuration(changed)
        for value in (float("inf"), float("-inf"), float("nan")):
            for location in ("other_field", "top_level", "nested_list"):
                with self.subTest(value=repr(value), location=location):
                    changed = self.fixture()
                    if location == "other_field":
                        changed["neuron_config"]["other"] = value
                    elif location == "top_level":
                        changed["quantize_clamp_bound"] = value
                    else:
                        changed["neuron_config"]["quantize_clamp_bound"] = [value]
                    with self.assertRaises(AdmissionError):
                        identity.canonical_configuration(changed)

    def test_json_scalar_and_container_types_remain_distinct(self):
        values = [None, False, 0, 0.0, -0.0, "0", [], {}]
        encodings = [identity.canonical_configuration({"value": value}) for value in values]
        self.assertEqual(len(set(encodings)), len(values))

    def test_invalid_config_rejects_before_model_payload_hash_or_software_probe(self):
        for field, value in (("quantize_clamp_bound", float("nan")),
                             ("quantize_clamp_bound", float("-inf")),
                             ("unexpected", float("inf"))):
            with self.subTest(field=field, value=repr(value)):
                changed = self.fixture()
                changed["neuron_config"][field] = value
                with patch.object(identity, "verify_view") as model_hash:
                    with patch.object(identity, "software_identity") as software:
                        with self.assertRaises(AdmissionError):
                            identity.build_identity("never-open-this-model", changed)
                        model_hash.assert_not_called()
                        software.assert_not_called()


class DisabledPluginTests(unittest.TestCase):
    def test_disabled_path_imports_no_model_library(self):
        original = builtins.__import__
        def restricted(name, *args, **kwargs):
            if name.startswith(("torch", "transformers", "neuron")) or name == "registration":
                raise AssertionError("Unexpected model import: " + name)
            return original(name, *args, **kwargs)
        with patch.dict(os.environ, {"NEURON_LFM3": "0"}):
            with patch.object(builtins, "__import__", side_effect=restricted): register()

    def test_variant_default_and_invalid_value(self):
        with patch.dict(os.environ, {"NEURON_LFM3": "1", "NEURON_CPU_SAMPLER_FAST": "1"}):
            with patch.dict(os.environ):
                os.environ.pop("NEURON_LFM3_NKI_WRITER", None)
                self.assertEqual(selected_variant(), "reference")
            with patch.dict(os.environ, {"NEURON_LFM3_NKI_WRITER": "invalid"}):
                with self.assertRaises(AdmissionError): selected_variant()

    def test_compiler_pool_context_restores_only_builder_namespace(self):
        # A fake import target verifies ownership and restoration without
        # importing the actual SDK. No compiler function is called.
        import concurrent.futures
        original = SimpleNamespace(futures=concurrent.futures)
        builder = SimpleNamespace(concurrent=original)
        trace = SimpleNamespace(model_builder=builder)
        package = SimpleNamespace(trace=trace)
        modules = {"neuronx_distributed": package,
                   "neuronx_distributed.trace": trace,
                   "neuronx_distributed.trace.model_builder": builder}
        with patch.dict(sys.modules, modules):
            with self.assertRaisesRegex(RuntimeError, "fake build failure"):
                with artifacts.serial_compiler():
                    self.assertIsNot(builder.concurrent, original)
                    with builder.concurrent.futures.ThreadPoolExecutor(8) as executor:
                        self.assertEqual(executor._max_workers, 1)
                    raise RuntimeError("fake build failure")
            self.assertIs(builder.concurrent, original)


if __name__ == "__main__":
    unittest.main()
