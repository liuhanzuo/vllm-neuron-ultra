# SPDX-License-Identifier: Apache-2.0
"""Explicit opt-in; disabled import performs no native work."""
def register():
    import os
    if os.environ.get("NEURON_NEMOTRON9B", "0") != "1":
        return
    from .identity import selected_variant
    selected_variant()
    from .registration import install
    install()
