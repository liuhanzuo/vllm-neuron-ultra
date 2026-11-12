# SPDX-License-Identifier: Apache-2.0
"""Explicit opt-in Qwen3.5 text integration; package import is native-free."""


def register():
    import os
    if os.environ.get("NEURON_QWEN35", "0") != "1":
        return
    from .identity import selected_variant
    selected_variant()
    from .registration import install
    install()
