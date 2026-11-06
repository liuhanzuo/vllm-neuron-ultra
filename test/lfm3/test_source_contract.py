# SPDX-License-Identifier: Apache-2.0
"""Standard-library source, tokenizer and scope controls; no real model imports."""
import ast
import builtins
import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace, ModuleType
import sys
import tempfile
import unittest
from unittest.mock import patch

from vllm_neuron.models.lfm3 import identity, prepare_view
from vllm_neuron.models.lfm3.identity import AdmissionError

PACKAGE = Path(identity.__file__).parent


class TokenizerViewTests(unittest.TestCase):
    def test_transform_only_compatibility_fields(self):
        original = {"tokenizer_class": "TokenizersBackend", "extra_special_tokens": [],
                    "bos_token": "<bos>", "image_token": "<image>", "vocab_size": 128000}
        result = prepare_view.compatible_tokenizer_config(original)
        self.assertEqual(result, {**original, "tokenizer_class": "PreTrainedTokenizerFast",
                                 "extra_special_tokens": {"image_token": "<image>"}})
        self.assertEqual(original["tokenizer_class"], "TokenizersBackend")
        for key, value in (("tokenizer_class", "foreign"), ("extra_special_tokens", ["new"])):
            with self.assertRaises(AdmissionError):
                prepare_view.compatible_tokenizer_config({**original, key: value})

    def test_all_102_original_inputs_and_special_tokens(self):
        fixtures = identity.read_json(PACKAGE / "tokenizer_fixtures.json")
        rows = fixtures["hf_prompts"] + fixtures["quality_inputs"]
        inputs = {row["text"]: row["input_ids"] for row in rows}
        calls = []
        def apply(messages, **kwargs):
            self.assertEqual(kwargs, {"tokenize": True, "add_generation_prompt": True, "enable_thinking": False})
            calls.append(messages[0]["content"])
            return inputs[messages[0]["content"]]
        fake = SimpleNamespace(apply_chat_template=apply, all_special_ids=fixtures["all_special_ids"],
            decode=lambda ids, skip_special_tokens: next(row["text"] for row in fixtures["decode_checks"]
                if row["skip_special_tokens"] == skip_special_tokens))
        self.assertEqual(prepare_view.validate_tokenizer(fake, fixtures), {"hf_prompts": 6, "quality_inputs": 96})
        self.assertEqual(len(calls), 102)
        for row in rows:
            original = inputs[row["text"]]
            inputs[row["text"]] = original + [0]
            with self.subTest(id=row["id"]), self.assertRaises(AdmissionError):
                prepare_view.validate_tokenizer(fake, fixtures)
            inputs[row["text"]] = original
        fake.all_special_ids = [0]
        with self.assertRaises(AdmissionError): prepare_view.validate_tokenizer(fake, fixtures)

    def test_existing_destination_rejected_before_payload_or_tokenizer(self):
        with tempfile.TemporaryDirectory() as root:
            with patch.object(prepare_view, "verify_files") as verify, patch.object(prepare_view, "load_tokenizer") as load:
                with self.assertRaises(AdmissionError): prepare_view.prepare(root, root)
                verify.assert_not_called(); load.assert_not_called()

    def test_prepare_publishes_only_after_token_validation(self):
        for fails in (False, True):
            with self.subTest(fails=fails), tempfile.TemporaryDirectory() as temporary:
                source = Path(temporary) / "source"; source.mkdir()
                destination = Path(temporary) / "view"
                config = {"tokenizer_class": "TokenizersBackend", "extra_special_tokens": [], "image_token": "<image>"}
                (source / "tokenizer_config.json").write_text(json.dumps(config))
                (source / "fake-checkpoint.txt").write_text("tiny fake model data")
                descriptors = {path.name: {"algorithm": "sha256", "digest": hashlib.sha256(path.read_bytes()).hexdigest(), "size": path.stat().st_size} for path in source.iterdir()}
                view_bytes = json.dumps(prepare_view.compatible_tokenizer_config(config), ensure_ascii=False, indent=2).encode()
                def validate(*args):
                    self.assertFalse((destination / identity.VIEW_MANIFEST).exists())
                    if fails: raise AdmissionError("token mismatch")
                    return {"hf_prompts": 6, "quality_inputs": 96}
                with patch.object(prepare_view, "SOURCE_FILES", descriptors), patch.object(prepare_view, "TOKENIZER_VIEW_SHA256", hashlib.sha256(view_bytes).hexdigest()), patch.object(prepare_view, "load_tokenizer", return_value=object()), patch.object(prepare_view, "validate_tokenizer", side_effect=validate):
                    if fails:
                        with self.assertRaises(AdmissionError): prepare_view.prepare(source, destination)
                    else: prepare_view.prepare(source, destination)
                self.assertTrue(destination.is_dir())
                self.assertEqual((destination / identity.VIEW_MANIFEST).exists(), not fails)
                self.assertEqual(json.loads((source / "tokenizer_config.json").read_text()), config)

    def test_git_blob_and_lfs_identities_use_exact_size_and_bytes(self):
        raw = b"small fake source\n"
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "fixture.txt"; path.write_bytes(raw)
            for algorithm in ("git-blob-sha1", "sha256"):
                digest = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() if algorithm == "git-blob-sha1" else hashlib.sha256(raw).hexdigest()
                descriptor = {"algorithm": algorithm, "digest": digest, "size": len(raw)}
                identity.verify_files(root, {path.name: descriptor})
                for key, value in (("digest", "0" * len(digest)), ("size", len(raw) + 1)):
                    with self.assertRaises(AdmissionError): identity.verify_files(root, {path.name: {**descriptor, key: value}})


class DispatchAndRegistrationTests(unittest.TestCase):
    def test_environment_cannot_relabel_an_installed_writer(self):
        fake = SimpleNamespace(_INSTALLED=True, _VARIANT="reference")
        with patch.dict(sys.modules, {"vllm_neuron.models.lfm3.registration": fake}):
            with patch.dict(os.environ, {"NEURON_LFM3": "1", "NEURON_CPU_SAMPLER_FAST": "1", "NEURON_LFM3_NKI_WRITER": "1"}):
                with self.assertRaises(AdmissionError): identity.selected_variant()

    def test_lfm_keeps_original_decode_routing(self):
        tree = ast.parse((PACKAGE / "modeling.py").read_text())
        model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NeuronLfm3ForCausalLM")
        self.assertNotIn("load", [node.name for node in model.body if isinstance(node, ast.FunctionDef)])
        self.assertNotIn("get_target_bucket", (PACKAGE / "modeling.py").read_text())
        from vllm_neuron.models.lfm3.provenance import HISTORICAL_ROOT_CONFIG
        self.assertEqual(HISTORICAL_ROOT_CONFIG["buckets"], [2048])
        self.assertEqual(HISTORICAL_ROOT_CONFIG["decode_routes_observed"], [128, 256, 512, 1024, 2048])

    def test_one_row_prefill_preserves_original_writer(self):
        tree = ast.parse((PACKAGE / "cache_writer.py").read_text())
        fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "store_slots_nki_predicated_decode")
        calls = []
        ns = {"original_store_slots": lambda *args: calls.append("original"),
              "store_slots_nki_predicated": lambda *args: calls.append("nki")}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "<writer-dispatch>", "exec"), ns)
        for rows, expected in ((1, "original"), (2, "nki"), (4, "nki")):
            ns[fn.name](None, SimpleNamespace(shape=(rows,)), None)
            self.assertEqual(calls[-1], expected)

    def registration(self, collision=None):
        tree = ast.parse((PACKAGE / "registration.py").read_text())
        fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "install")
        calls = []; base = object(); target = object(); config = object(); wrapper = object()
        types = {} if collision != "nxdi" else {"lfm3": {"causal-lm": object()}}
        constants = ModuleType("neuronx_distributed_inference.utils.constants"); constants.MODEL_TYPES = types
        text = ModuleType("vllm_neuron.models.lfm3.text_core"); text.store_slots = base
        helpers = ModuleType("vllm_neuron.models.lfm3.helpers"); helpers.store_slots = base
        writer = ModuleType("vllm_neuron.models.lfm3.cache_writer"); writer.install = lambda: calls.append("writer")
        namespace = {"__package__": "vllm_neuron.models.lfm3", "_INSTALLED": False, "_VARIANT": None,
            "selected_variant": lambda: "nki_predicated_decode_v4", "require": identity.require,
            "CONFIG_MAPPING": {"lfm3": object()} if collision == "transformers" else {},
            "ModelRegistry": SimpleNamespace(get_supported_archs=lambda: ["Lfm3ForCausalLM"] if collision == "vllm" else [],
                register_model=lambda *args: calls.append("vllm")),
            "AutoConfig": SimpleNamespace(register=lambda *args: calls.append("config")),
            "NeuronLfm3ForCausalLM": target, "Lfm3NeuronConfig": config, "Lfm3ForCausalLM": wrapper,
            "install_nxdi": lambda: calls.append("nxdi")}
        modules = {constants.__name__: constants, text.__name__: text, helpers.__name__: helpers, writer.__name__: writer}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "<registration>", "exec"), namespace)
        return namespace, modules, calls

    def test_all_registration_collisions_reject_before_mutation(self):
        for collision in ("transformers", "vllm", "nxdi"):
            ns, modules, calls = self.registration(collision)
            with patch.dict(sys.modules, modules), self.assertRaises(AdmissionError): ns["install"]()
            self.assertEqual(calls, [])

    def test_install_order_idempotence_and_variant_lock(self):
        ns, modules, calls = self.registration()
        with patch.dict(sys.modules, modules):
            ns["install"](); ns["install"]()
            self.assertEqual(calls, ["writer", "nxdi", "config", "vllm"])
            ns["selected_variant"] = lambda: "reference"
            with self.assertRaises(AdmissionError): ns["install"]()
        fresh, modules, calls = self.registration()
        with patch.dict(sys.modules, modules): fresh["install"]()
        self.assertEqual(calls, ["writer", "nxdi", "config", "vllm"])


if __name__ == "__main__":
    unittest.main()
