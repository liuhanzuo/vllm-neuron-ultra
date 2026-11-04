# SPDX-License-Identifier: Apache-2.0
"""Exercise the shipped CPU sampling method without importing the Neuron SDK."""

import ast
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def cpu_sample():
    # Compile the actual method body, avoiding model-runner SDK import effects.
    path = (Path(__file__).resolve().parents[2] / "vllm_neuron" /
            "worker" / "neuronx_distributed_model_runner.py")
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "NeuronxDistributedModelRunner")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_cpu_sample")
    logger = logging.getLogger("test.cpu_sampling_debug")
    namespace = {"torch": torch, "logging": logging, "logger": logger,
                 "ModelInputForNeuron": object, "SamplerOutput": object}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
         namespace)
    return namespace["_cpu_sample"], logger


def runner():
    output = SimpleNamespace(sampled_token_ids=torch.tensor([[1]]))
    return SimpleNamespace(
        model_config=SimpleNamespace(get_vocab_size=lambda: 256),
        input_batch=SimpleNamespace(sampling_metadata=object()),
        cpu_sampler=Mock(return_value=output))


def test_info_logging_skips_debug_reductions(cpu_sample):
    method, logger = cpu_sample
    logger.setLevel(logging.INFO)
    logits = Mock()
    logits.dim.return_value = 2
    logits.shape = (1, 256)
    model = runner()
    result = method(model, logits, SimpleNamespace(request_ids=["request"]))
    assert result is model.cpu_sampler.return_value
    for name in ("min", "max", "mean", "std"):
        getattr(logits, name).assert_not_called()
    model.cpu_sampler.assert_called_once_with(
        logits, model.input_batch.sampling_metadata)


def test_debug_logging_retains_statistics(cpu_sample, caplog):
    method, logger = cpu_sample
    model = runner()
    with caplog.at_level(logging.DEBUG, logger=logger.name):
        method(model, torch.linspace(-3, 3, 256)[None],
               SimpleNamespace(request_ids=["request"]))
    assert "Logits statistics - min:" in caplog.text
    assert "Logits statistics - mean:" in caplog.text
    assert "First sequence top-5 logits:" in caplog.text


@pytest.mark.parametrize("case", ["shape", "vocab", "metadata", "output", "tokens"])
def test_validation_is_retained_with_debug_disabled(cpu_sample, case):
    method, logger = cpu_sample
    logger.setLevel(logging.INFO)
    model = runner()
    logits = torch.zeros(1, 256)
    if case == "shape":
        logits = logits[:, :, None]
    elif case == "vocab":
        logits = logits[:, :128]
    elif case == "metadata":
        model.input_batch.sampling_metadata = None
    elif case == "output":
        model.cpu_sampler.return_value = None
    else:
        model.cpu_sampler.return_value.sampled_token_ids = None
    with pytest.raises(RuntimeError, match="CPU sampling failed"):
        method(model, logits, SimpleNamespace(request_ids=["request"]))
