"""Future trace-only integer reference. Importing this file is stdlib-only.

This is a new comparison reference, not a claim that the original BF16 path
passed or that NKI improved the original model baseline. No numeric casts.
"""
from qwen35_predicated_slot_protocol_v1 import LAYOUTS


def _as_s16(value):
    return value.bitcast('s16')


def _as_bf16(value):
    return value.bitcast('bf16')


def _as_s32(value):
    return value.bitcast('s32')


def _as_f32(value):
    return value.bitcast('f32')


def make_integer_writer(dtype_name, shape):
    """Instantiate only inside the original leaf's native import boundary."""
    if (dtype_name, tuple(shape)) not in [(dtype, tuple(dims)) for _, dtype, dims in LAYOUTS]:
        raise ValueError('One original complete layout required')
    import torch
    from torch_neuronx.xla_impl.base import xla_call
    from qwen35_text_port import store_slots
    if dtype_name == 'bfloat16':
        dtype = torch.bfloat16
        to_integer = xla_call(_as_s16, name='qwen35_integer_slot_reference_to_s16_v1')
        to_float = xla_call(_as_bf16, name='qwen35_integer_slot_reference_to_bf16_v1')
    else:
        dtype = torch.float32
        to_integer = xla_call(_as_s32, name='qwen35_integer_slot_reference_to_s32_v1')
        to_float = xla_call(_as_f32, name='qwen35_integer_slot_reference_to_f32_v1')
    shape = tuple(shape)

    def writer(cache, updates, slots):
        if tuple(cache.shape) != shape or tuple(updates.shape) != shape or cache.dtype != dtype or updates.dtype != dtype:
            raise ValueError('Exact original floating storage dtype and shape required')
        if tuple(slots.shape) != (4,) or slots.dtype != torch.int64:
            raise ValueError('Exact original four signed int64 slot values required')
        # Both complete inputs and the complete result are bit-reinterpreted.
        # Original store_slots ordering retains last-writer-wins duplicates.
        return to_float(store_slots(to_integer(cache), to_integer(updates), slots))

    return writer
