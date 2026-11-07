# SPDX-License-Identifier: Apache-2.0
"""Explicit serving entry; registration also runs in multiprocessing children.

Example: python -m vllm_neuron.models.nemotron9b.serve --model PREPARED_VIEW ...
Set the opt-in/environment documented in docs/nemotron9b.md before running.
"""
import os
import runpy
from .identity import require, selected_variant
selected_variant()
require(os.environ.get("VLLM_WORKER_MULTIPROC_METHOD", "spawn") == "spawn", "Spawn workers required")
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
from . import register
register()

if __name__=="__main__":
    runpy.run_module("vllm.entrypoints.openai.api_server",run_name="__main__",alter_sys=False)
