# SPDX-License-Identifier: Apache-2.0
"""Merged plugin wiring with inert objects; no native libraries or model data."""
import ast
import builtins
import importlib
import os
from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[2]
MODELS = {
    "lfm3": ("lfm3", "lfm3"),
    "neuron_smollm3": ("smollm3", "smollm3_neuron_text"),
    "nemotron9b": ("nemotron9b", "nemotron9b_neuron_text"),
}


def setup_metadata():
    fake = ModuleType("setuptools")
    fake.setup = Mock()
    fake.find_packages = lambda **kwargs: []
    with patch.dict(sys.modules, {"setuptools": fake}):
        runpy.run_path(str(ROOT / "setup.py"))
    return fake.setup.call_args.kwargs


def load_method():
    path = ROOT / "vllm_neuron/worker/neuronx_distributed_model_loader.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "NeuronModelBase")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                  and n.name == "_load_weights_common")
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
         namespace)
    return namespace[method.name]


class MultiModelIntegrationTests(unittest.TestCase):
    def test_all_installed_entry_points_are_inert_when_disabled(self):
        entries = setup_metadata()["entry_points"]["vllm.general_plugins"]
        self.assertEqual(len(entries), len(MODELS))
        seen = set()
        original = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in {
                "torch", "torch_xla", "torch_neuronx", "transformers", "vllm",
                "neuronxcc", "neuronx_distributed", "neuronx_distributed_inference",
            } or name.endswith("registration"):
                raise AssertionError("Disabled plugin imported " + name)
            return original(name, *args, **kwargs)

        with patch.dict(os.environ, {}, clear=True), \
                patch.object(builtins, "__import__", side_effect=guarded):
            for entry in entries:
                name, target = (part.strip() for part in entry.split("=", 1))
                package, function = target.split(":", 1)
                self.assertNotIn(name, seen)
                seen.add(name)
                self.assertEqual(package, "vllm_neuron.models." + MODELS[name][0])
                self.assertIsNone(getattr(importlib.import_module(package), function)())
        self.assertEqual(seen, set(MODELS))

    def test_required_package_resources_survive_merge(self):
        data = setup_metadata()["package_data"]
        for model, required in {
            "lfm3": {"tokenizer_fixtures.json", "licenses/GSM8K-MIT.txt"},
            "nemotron9b": {"tokenizer_fixtures.json", "core_contract.json",
                           "neuron_config_contract.json", "licenses/GSM8K-MIT.txt"},
        }.items():
            root = ROOT / "vllm_neuron/models" / model
            files = {str(path.relative_to(root))
                     for pattern in data["vllm_neuron.models." + model]
                     for path in root.glob(pattern) if path.is_file()}
            self.assertTrue(required <= files, required - files)

    def route(self, rejected):
        method = load_method()
        for package, model_type in MODELS.values():
            with self.subTest(model=package, rejected=rejected):
                modules = {}
                for other, _ in MODELS.values():
                    name = "vllm_neuron.models." + other + ".artifacts"
                    module = ModuleType(name)
                    module.load_or_compile = Mock(
                        side_effect=AssertionError("wrong adapter selected"))
                    modules[name] = module
                selected = modules["vllm_neuron.models." + package + ".artifacts"]
                result = object()
                failure = ValueError("artifact identity mismatch")
                selected.load_or_compile = Mock(
                    side_effect=failure if rejected else None, return_value=result)
                owner = SimpleNamespace()
                config = SimpleNamespace(model_type=model_type)
                model = SimpleNamespace(get_neuron_config_cls=lambda: SimpleNamespace)
                kwargs = {"neuron_config": {}, "config": config}
                with patch.dict(sys.modules, modules):
                    if rejected:
                        with self.assertRaises(ValueError) as caught:
                            method(owner, "inert-model-path", model, **kwargs)
                        self.assertIs(caught.exception, failure)
                    else:
                        self.assertIs(method(owner, "inert-model-path", model,
                                             **kwargs), result)
                selected.load_or_compile.assert_called_once_with(
                    owner, "inert-model-path", model, config, kwargs)
                for name, module in modules.items():
                    if module is not selected:
                        module.load_or_compile.assert_not_called()

    def test_each_model_uses_its_own_admission_route(self):
        self.route(rejected=False)

    def test_admission_failure_propagates_without_generic_recompile(self):
        self.route(rejected=True)


if __name__ == "__main__":
    unittest.main()
