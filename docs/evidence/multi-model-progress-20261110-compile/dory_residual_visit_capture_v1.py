"""Prepared Dory residual observations; no model imports or execution entrypoint.

Observe the unchanged complete backbone with (physical layer, recurrent loop)
identities. The caller owns original weights, request histories, serial admission,
selected-row copies and raw-byte review. These hooks establish no qualification.
"""
from types import MappingProxyType


def require(value, message):
    if not value:
        raise RuntimeError(message)


def visits(config):
    fields = ("num_hidden_layers", "n_input_layers", "n_recurrent_layers",
              "n_output_layers", "n_recurrent_loops", "hidden_size")
    actual = tuple(getattr(config, field, None) for field in fields)
    require(all(type(value) is int for value in actual)
            and actual == (72, 16, 40, 16, 2, 2560),
            "Pinned original Dory architecture required")
    return (tuple((i, 0) for i in range(16))
            + tuple((i, loop) for loop in range(2) for i in range(16, 56))
            + tuple((i, 0) for i in range(56, 72)))


def selected_raw_bytes(selected_rows):
    require(type(selected_rows) is int and selected_rows >= 0,
            "Nonnegative selected row count")
    return selected_rows * 112 * 2 * 2560 * 2


class CaptureResidualVisits:
    """Call original forward once and copy only chosen input/output rows.

    Hooks return None; original arguments and outputs are never replaced.
    On error, partial_values and pending_visit remain available to the caller
    for a failure archive. A failed call is never converted to a success.
    """

    def __init__(self, official, select_copy, guard):
        self.official = official
        self.select_copy, self.guard = select_copy, guard
        self.expected = visits(official.backbone.config)
        self.handles = []
        self.active = False
        self.values = {}
        self.order = []
        self.completed = 0
        self.pending_visit = None
        self.selectors = None
        self.last_call_succeeded = False

    @property
    def partial_values(self):
        return MappingProxyType(self.values)

    def _identity(self, layer, args, kwargs):
        require(len(args) in (2, 3) and set(kwargs) <= {"loop_idx"},
                "Original decoder positional arguments and loop keyword")
        require(not (len(args) == 3 and "loop_idx" in kwargs),
                "Loop index must not be supplied twice")
        loop = args[2] if len(args) == 3 else kwargs.get("loop_idx", 0)
        require(type(loop) is int, "Integer original loop identity")
        return layer, loop

    def _record(self, visit, label, tensor):
        require(self.active and getattr(tensor, "ndim", None) == 2
                and tensor.shape[1] == 2560, "Original residual boundary shape")
        key = (*visit, label)
        require(key not in self.values, "Each visit boundary exactly once")
        self.guard()
        copied = self.select_copy(tensor, self.selectors)
        require(getattr(copied, "ndim", None) == 2
                and tuple(copied.shape) == (len(self.selectors), 2560),
                "Copy every selected residual row")
        self.values[key] = copied
        self.order.append(key)

    def install(self):
        layers = self.official.backbone.layers
        require(not self.handles and not self.active
                and len(layers) == 72 and len({id(x) for x in layers}) == 72
                and self.official.backbone.num_loops == 2
                and visits(self.official.backbone.config) == self.expected,
                "Distinct original physical layers and unchanged architecture")

        def before(layer):
            def hook(module, args, kwargs):
                require(self.active and self.pending_visit is None
                        and self.completed < len(self.expected),
                        "No nested, extra or unowned decoder call")
                visit = self._identity(layer, args, kwargs)
                require(visit == self.expected[self.completed],
                        "Exact physical layer and recurrent-loop execution order")
                self.pending_visit = visit
                self._record(visit, "input", args[1])
            return hook

        def after(layer):
            def hook(module, args, kwargs, output):
                visit = self._identity(layer, args, kwargs)
                require(self.active and visit == self.pending_visit,
                        "Output belongs to active original decoder call")
                self._record(visit, "residual_output", output)
                self.completed += 1
                self.pending_visit = None
            return hook

        try:
            for layer, block in enumerate(layers):
                self.handles.append(block.register_forward_pre_hook(
                    before(layer), with_kwargs=True))
                self.handles.append(block.register_forward_hook(
                    after(layer), with_kwargs=True))
        except BaseException:
            self.close()
            raise

    def run(self, tokens, positions, selectors):
        require(len(self.handles) == 144 and not self.active,
                "Installed inactive complete-backbone observer")
        require(visits(self.official.backbone.config) == self.expected,
                "Architecture unchanged before forward")
        self.active = True
        self.values = {}
        self.order = []
        self.completed = 0
        self.pending_visit = None
        self.selectors = selectors
        self.last_call_succeeded = False
        try:
            hidden = self.official(tokens, positions)
            expected_order = [(*visit, label) for visit in self.expected
                              for label in ("input", "residual_output")]
            require(self.completed == 112 and self.pending_visit is None
                    and self.order == expected_order,
                    "All 112 original visits and 224 boundaries completed")
            require(getattr(hidden, "ndim", None) == 2
                    and hidden.shape[1] == 2560,
                    "Original final hidden result")
            self.last_call_succeeded = True
            return hidden, dict(self.values)
        finally:
            self.active = False
            self.selectors = None

    def close(self):
        require(not self.active, "Close hooks only outside an active forward")
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
