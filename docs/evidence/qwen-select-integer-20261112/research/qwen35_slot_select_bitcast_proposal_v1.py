"""Three source-only discriminating probes; no imports/native work on import.

This is an independent diagnostic proposal, not a replacement reference or a
qualified primitive. An existing owned successor runner may instantiate these
modules only at its original native-import boundary. No launch path is provided.
"""

SHAPE = (4, 3072, 4)
ARMS = ('one_bf16_select', 'internal_s16_select', 'external_s16_select')
PROFILES = ('original_raw', 'negative_nan_raw')
MAPPINGS = (('idle', (-1, -1, -1, -1)), ('one_active', (3, -1, -1, -1)))


def schedule():
    return [dict(arm=arm, profile=profile, mapping=name, slots=list(slots))
            for arm in ARMS for profile in PROFILES for name, slots in MAPPINGS]


def _lower_as_s16(value):
    return value.bitcast('s16')


def _lower_as_bf16(value):
    return value.bitcast('bf16')


def make_modules():
    """Future owned trace only. Does not compile, call a device, or select flags.

    The pinned SDK implements Op.bitcast via BitcastConvert, separately from
    Op.cast/Convert. This source fact does not establish Trn1 support or output
    fidelity; inspect each actual emitted HLO and perform all twelve raw checks.
    """
    import torch
    from torch_neuronx.xla_impl.base import xla_call
    from qwen35_text_port import store_slots

    as_s16 = xla_call(_lower_as_s16, name='qwen35_select_probe_as_s16_v1')
    as_bf16 = xla_call(_lower_as_bf16, name='qwen35_select_probe_as_bf16_v1')

    def exact_inputs(cache, updates, slots, dtype):
        assert tuple(cache.shape) == tuple(updates.shape) == SHAPE
        assert cache.dtype == updates.dtype == dtype
        assert tuple(slots.shape) == (4,) and slots.dtype == torch.int64

    class OneBF16Select(torch.nn.Module):
        def forward(self, cache, updates, slots):
            exact_inputs(cache, updates, slots, torch.bfloat16)
            indices = torch.arange(cache.shape[0], device=cache.device)
            selected = indices == slots[0]
            return torch.where(selected.reshape((4, 1, 1)), updates[0:1], cache)

    class InternalS16Select(torch.nn.Module):
        def forward(self, cache, updates, slots):
            exact_inputs(cache, updates, slots, torch.bfloat16)
            # Reinterpret the entire payload on device; no .to(int16) conversion.
            cache_words = as_s16(cache)
            update_words = as_s16(updates)
            selected_words = store_slots(cache_words, update_words, slots)
            return as_bf16(selected_words)

    class ExternalS16Select(torch.nn.Module):
        def forward(self, cache, updates, slots):
            exact_inputs(cache, updates, slots, torch.int16)
            return store_slots(cache, updates, slots)

    return dict(zip(ARMS, (OneBF16Select().eval(), InternalS16Select().eval(),
                           ExternalS16Select().eval())))


def byte_oracle(cache, updates, slots):
    """Whole raw copy oracle restricted to the two declared probe mappings."""
    assert type(cache) is bytes and type(updates) is bytes
    assert len(cache) == len(updates) == 98304
    assert tuple(slots) in [mapping for _, mapping in MAPPINGS]
    if tuple(slots) == MAPPINGS[0][1]:
        return cache
    return cache[:3 * 24576] + updates[:24576]
