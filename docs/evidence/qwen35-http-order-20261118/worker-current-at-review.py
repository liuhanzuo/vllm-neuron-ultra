"""Load-only worker for comparing two sampler modules on the frozen Qwen3.5 adapter."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

from qwen35_sampler_instance_v1 import worker_class
from vllm_neuron.worker.neuron_worker import NeuronWorker
from vllm_neuron.worker.neuronx_distributed_model_loader import NeuronModelBase


def refuse_compile(*args, **kwargs):
    raise RuntimeError("Sampler benchmark requires the previously validated Qwen3.5 graph")


NeuronModelBase._compile_and_load_model = refuse_compile


def build_sampler(reference):
    path = Path(os.environ["SAMPLER_SOURCE"])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == os.environ["SAMPLER_SHA256"]
    assert type(reference).__name__ == "TopKTopPSampler"
    spec = importlib.util.spec_from_file_location("measured_sampler", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BoundedTopKTopPSampler(reference)


def record(worker, runner, attachment):
    assert attachment["operator_replaced"] and not attachment["observation_enabled"]
    operator = runner.cpu_sampler.topk_topp_sampler
    assert type(operator).__module__ == "measured_sampler"
    receipt = dict(worker_pid=os.getpid(), worker_start_ticks=int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]),
                   source=os.environ["SAMPLER_SOURCE"], sha256=os.environ["SAMPLER_SHA256"],
                   operator=type(operator).__name__, debug_behavior=attachment["debug_behavior"])
    with Path(os.environ["SAMPLER_RECEIPT"]).open("x") as stream:
        json.dump(receipt, stream)


class Worker(worker_class(NeuronWorker, "candidate", "measured", operation_factory=build_sampler, on_attach=record)):
    pass
