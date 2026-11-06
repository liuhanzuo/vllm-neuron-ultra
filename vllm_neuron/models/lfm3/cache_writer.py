# SPDX-License-Identifier: Apache-2.0
"""Process-local LFM decode writer; preserve the original one-row prefill path."""
from .helpers import store_slots as original_store_slots
from .nki_write_torch import store_slots_nki_predicated


def store_slots_nki_predicated_decode(cache, updates, slots):
    if updates.shape[0] == 1:
        return original_store_slots(cache, updates, slots)
    return store_slots_nki_predicated(cache, updates, slots)


def install():
    from . import text_core as lfm3_text_port
    if lfm3_text_port.store_slots is store_slots_nki_predicated_decode:
        return
    if lfm3_text_port.store_slots is not original_store_slots:
        raise RuntimeError('Another LFM writer is installed')
    lfm3_text_port.store_slots = store_slots_nki_predicated_decode
