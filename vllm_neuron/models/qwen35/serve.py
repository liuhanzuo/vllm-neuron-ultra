# SPDX-License-Identifier: Apache-2.0
"""Explicit serving entry; spawn children repeat opt-in architecture registration.

python -m vllm_neuron.models.qwen35.serve --model PREPARED_VIEW ...
Experimental serving: fresh packaged graph/native/HTTP qualification is pending.
Artifact integrity is checked before loading; this is not a numerical quality claim.
"""
import os
import runpy
from .identity import require, selected_variant
selected_variant()
require(os.environ.get("VLLM_WORKER_MULTIPROC_METHOD", "spawn") == "spawn", "Spawn workers required")
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
from . import register
register()

if __name__ == "__main__":
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__", alter_sys=False)
