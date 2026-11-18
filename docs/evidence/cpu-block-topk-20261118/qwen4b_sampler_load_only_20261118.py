"""Start the package server from a cached graph, refusing any compilation."""
import os
from pathlib import Path
import runpy

import vllm
from vllm_neuron.worker import cpu_sampler
from vllm_neuron.worker.neuronx_distributed_model_loader import NeuronModelBase

assert Path(cpu_sampler.__file__).resolve().is_relative_to(Path(os.environ["PYTHONPATH"]))
print("Measured sampler:", cpu_sampler.__file__, flush=True)


def forbid_compile(*args, **kwargs):
    raise RuntimeError("Existing BF16 graph required for sampler comparison")


NeuronModelBase._compile_and_load_model = forbid_compile

if __name__ == "__main__":
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
