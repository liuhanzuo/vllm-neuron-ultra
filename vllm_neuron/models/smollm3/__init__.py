# SPDX-License-Identifier: Apache-2.0
"""Opt-in SmolLM3 support; importing this package has no native dependencies."""


def register():
    """vLLM general-plugin entry; disabled before importing model libraries."""
    import os
    if os.environ.get("NEURON_SMOLLM3", "0") != "1":
        return
    from .registration import install
    install()
