# SPDX-License-Identifier: Apache-2.0
"""Lazy NKI 0.5 typed writer; one-row calls retain the original operation."""
from .identity import require, selected_scope


def install():
    from . import text_core
    from .helpers import store_slots
    from .runtime import admitted
    from .lfm3_nki05_typed_dispatch_v1 import make_writer
    require(text_core.store_slots is store_slots, "LFM3 writer already replaced")
    capacity = selected_scope()["decode_batch"]
    typed = None

    def typed_call(cache, updates, slots):
        nonlocal typed
        require(admitted(), "Typed writer requires loader-owned compile/load admission")
        if typed is None:
            from .slot_cache_nki05_typed_jit_v1 import make_device_kernel
            from .slot_cache_nki05_typed_torch_v1 import make_typed_writer
            typed = make_typed_writer(make_device_kernel(admitted), admitted)
        return typed(cache, updates, slots)

    dispatch = make_writer(capacity, store_slots, typed_call)

    def writer(cache, updates, slots):
        require(admitted(), "LFM3 writer called outside admitted compile/load scope")
        return dispatch(cache, updates, slots)

    text_core.store_slots = writer
