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
from vllm_neuron.models.nemotron9b import register, artifacts, identity
from vllm_neuron.models.nemotron9b.identity import ARTIFACT_MANIFEST, AdmissionError, selected_variant
from vllm_neuron.models.nemotron9b.provenance import SDK_SOURCE_FILES, NKI_BINARY_MODULES


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
        self.identity = {"variant": "split_nki_v5", "scope": {"context": 4096},
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
                configuration["metadata"] = {"nemotron9b_ultra": identity}
                (Path(path) / "neuron_config.json").write_text(json.dumps(configuration))
            def load(self, path):
                calls.append("load-new")
        self.model = Model
        self.owner = SimpleNamespace(_load_compiled_model=lambda *args: calls.append("load-existing"))
        self.config = SimpleNamespace(to_json_string=lambda: '{"metadata": null}')
        self.enter(patch.dict(os.environ, {"NEURON_COMPILED_ARTIFACTS": str(self.path),
                                          "NEURON_NEMOTRON9B_ALLOW_COMPILE": "1"}))
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
        with patch.dict(os.environ, {"NEURON_NEMOTRON9B_ALLOW_COMPILE": "0"}):
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
        config.write_text('{"metadata":{"nemotron9b_ultra":{"variant":"wrong"}}}')
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








if __name__ == "__main__":
    unittest.main()
