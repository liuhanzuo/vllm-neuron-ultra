# SPDX-License-Identifier: Apache-2.0
"""Opt-in LFM3 SDK 0.5.3 candidate integration; package import has no native dependencies."""


def register():
    """vLLM general plugin; disable before importing any model dependencies."""
    import os
    if os.environ.get("NEURON_LFM3", "0") != "1":
        return
    from .identity import selected_variant
    selected_variant()
    from .registration import install
    install()
