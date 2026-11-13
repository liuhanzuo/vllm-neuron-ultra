"""Additional KV256 control; import performs no IO or native imports.

Only a future separately admitted caller may invoke make_assembler/control.
This is not registered in any current run, model, writer or geometry matrix.
It preserves BF16 graph inputs/state/outputs and changes assembly internally.
"""

PREFIX_SHAPE = (4, 1, 256, 256)
TAIL_SHAPE = (4, 1, 3840, 256)
STATE_SHAPE = (4, 1, 4096, 256)
OP_NAME = "qwen35_kv256_s16_concat_control_v1"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _tensor(value, shape):
    _require(tuple(value.shape) == shape and str(value.dtype) == "torch.bfloat16",
             "Exact original BF16 geometry required")


def _build_assembler(xla_call, xb):
    """Dependency-injected construction for pure mocks; production uses SDK API.

    Both operands enter and leave the registered computation as BF16. S16 is
    equal-width bit reinterpretation, never numeric Convert or tensor.to().
    """
    _require(callable(xla_call), "Explicit XLA registration API required")
    _require(xb.Type.S16 == "s16" and xb.Type.BF16 == "bf16",
             "Pinned equal-width XLA element type spellings required")

    def lower(prefix, tail):
        for operand, sizes in ((prefix, PREFIX_SHAPE), (tail, TAIL_SHAPE)):
            shape = operand.shape()
            _require(shape.dtype == xb.Type.BF16 and tuple(shape.sizes) == sizes,
                     "Exact BF16 lowering operands required")
        prefix_words = prefix.bitcast(xb.Type.S16)
        tail_words = tail.bitcast(xb.Type.S16)
        joined_words = xb.Op.concat_in_dim((prefix_words, tail_words), dimension=2)
        result = joined_words.bitcast(xb.Type.BF16)
        _require(result.shape().dtype == xb.Type.BF16
                 and tuple(result.shape().sizes) == STATE_SHAPE,
                 "Full original BF16 result geometry required")
        return result

    operation = xla_call(lower, name=OP_NAME)
    _require(callable(operation), "Registered callable required")

    def assemble(prefix, tail):
        _tensor(prefix, PREFIX_SHAPE)
        _tensor(tail, TAIL_SHAPE)
        _require(prefix.device.type == tail.device.type == "xla"
                 and str(prefix.device) == str(tail.device),
                 "Two operands on the same tracing XLA device required")
        result = operation(prefix, tail)
        _tensor(result, STATE_SHAPE)
        return result

    return assemble


def make_assembler():
    """Future admitted native factory only; no compilation is initiated here."""
    from torch_neuronx.xla_impl.base import xla_call
    from torch_xla.core import xla_builder
    return _build_assembler(xla_call, xla_builder)


def _build_control(torch_api, assembler, initial_cache, prefix_writer):
    """Pure-mockable module construction; future runtime must bind writer source."""
    _tensor(initial_cache, STATE_SHAPE)
    _require(callable(assembler) and callable(prefix_writer),
             "Explicit original writer and assembly callable required")

    class Control(torch_api.nn.Module):
        def __init__(self):
            super().__init__()
            self.cache = torch_api.nn.Parameter(initial_cache.clone(), requires_grad=False)

        def forward(self, updates, slots):
            _tensor(updates, PREFIX_SHAPE)
            _require(tuple(slots.shape) == (4,) and str(slots.dtype) == "torch.int32",
                     "Original four int32 slots required")
            old = self.cache[slots.clamp_min(0)]
            prefix = prefix_writer(self.cache[:, :, :256, :], updates, slots)
            tail = self.cache[:, :, 256:, :]
            new = assembler(prefix, tail)
            return old, new

    return Control().eval()


def make_control(initial_cache, prefix_writer):
    """Build, but do not trace, the original old/new output-1-alias control.

    Future caller supplies the pinned original NKI prefix writer and finite
    BF16 initial state, then traces with input_output_aliases=alias_map(control).
    This factory neither selects a source checkout nor opens any saved inputs.
    """
    import torch
    return _build_control(torch, make_assembler(), initial_cache, prefix_writer)


def alias_map(control):
    """Keep new-state at physical output 1; no state dtype or ABI reordering."""
    _tensor(control.cache, STATE_SHAPE)
    return {control.cache: 1}
