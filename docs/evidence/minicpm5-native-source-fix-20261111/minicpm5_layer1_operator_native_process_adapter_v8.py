"""Diagnostic successor; unchanged v7 lifecycle and loaded-source gate scope.

The native runner calls this exact verifier before/after native import and after
scientific capture. The inherited resolve_admitted still executes v7's original
early local-source gate; no already executed source is patched or monkeypatched.
"""
from research.minicpm5_layer1_operator_native_process_adapter_v7 import *
from research import minicpm5_native_python_sources_v1 as native_python_sources


def verify_loaded(plan, guard):
    return native_python_sources.verify_loaded(plan, guard, sources.text)
