# SPDX-License-Identifier: Apache-2.0
"""Process-local compile/load scope; external device scheduling stays with owner.

No environment flag alone produces a writer capability. The artifact loader
opens this context after complete model/software/source/config admission.
"""
import os
from contextlib import contextmanager
from .identity import SCHEMA, require, selected_scope, selected_variant

_ACTIVE = None


def admitted():
    return (_ACTIVE is not None and _ACTIVE["schema"] == SCHEMA and
            _ACTIVE["variant"] == selected_variant() and
            _ACTIVE["scope"] == selected_scope() and
            os.environ.get("NEURON_LFM3_EXPERIMENTAL_RUNTIME") == "1" and
            os.environ.get("NEURON_PLATFORM_TARGET_OVERRIDE") == "trn1" and
            os.environ.get("NEURON_LOGICAL_NC_CONFIG") == "1" and
            os.environ.get("NKI_SIMULATOR", "0") == "0")


@contextmanager
def admission(identity):
    global _ACTIVE
    require(_ACTIVE is None, "Nested/concurrent LFM3 compile/load admission unsupported")
    require(identity["schema"] == SCHEMA and identity["variant"] == selected_variant()
            and identity["scope"] == selected_scope(), "LFM3 runtime identity differs")
    _ACTIVE = identity
    try:
        require(admitted(), "Unqualified LFM3 runtime needs explicit experimental opt-in and Trn1/LNC1")
        yield
    finally:
        _ACTIVE = None
