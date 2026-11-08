"""Stdlib contracts for an UNVALIDATED Dory TP1 reference candidate.

Derived from vLLM 75fc05a53a366068f534a61ab3ee6b50485419f2; see provenance.json.
This module never opens checkpoint files or imports a numerical framework.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class DoryContract:
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    initializer_range: float
    ngpt_alpha_init: float
    attention_output_gate_logit_scale_init: float | None
    n_input_layers: int
    n_recurrent_layers: int
    n_output_layers: int
    n_recurrent_loops: int
    sliding_window: int
    layer_types: tuple[str, ...]
    rope_thetas: tuple[tuple[str, float], ...]

    @classmethod
    def from_mapping(cls, raw, *, tensor_parallel_size=1):
        """Explicit saved config only: no implicit architectural defaults/overrides."""
        if not isinstance(raw, dict):
            raise ValueError("Config must be an explicit dictionary")
        if type(tensor_parallel_size) is not int or tensor_parallel_size != 1:
            raise ValueError("This candidate supports TP1 only")
        metadata = {"architectures", "model_type", "dtype", "hidden_act",
                    "tie_word_embeddings", "mlp_bias", "attention_bias",
                    "recurrent_kv_cache_mode", "rope_parameters"}
        if metadata - raw.keys():
            raise ValueError(f"Missing metadata fields: {sorted(metadata - raw.keys())}")
        fields = set(cls.__dataclass_fields__) - {"rope_thetas"}
        if set(raw) - fields - metadata:
            raise ValueError(f"Unknown config fields: {sorted(set(raw) - fields - metadata)}")
        if raw.get("model_type") != "dory" or raw.get("architectures") != ["DoryForCausalLM"]:
            raise ValueError("Expected the DoryForCausalLM entrypoint")
        if raw.get("dtype") != "bfloat16":
            raise ValueError("Original reference requires the BF16 checkpoint")
        if raw.get("hidden_act") != "silu" or any(
            raw.get(k) is not False for k in ("tie_word_embeddings", "mlp_bias", "attention_bias")
        ):
            raise ValueError("Require SiLU, untied embeddings and bias-free projections")
        if raw.get("recurrent_kv_cache_mode") != "per_loop" or raw.get("n_recurrent_loops") != 2:
            raise ValueError("Pinned baseline requires 2 loops and per_loop cache")
        if fields - raw.keys():
            raise ValueError(f"Missing explicit config fields: {sorted(fields - raw.keys())}")
        nonnegative = ("n_input_layers", "n_recurrent_layers", "n_output_layers")
        positive = ("vocab_size", "hidden_size", "num_hidden_layers", "intermediate_size",
                    "num_attention_heads", "num_key_value_heads", "head_dim",
                    "max_position_embeddings", "n_recurrent_loops", "sliding_window")
        for k in nonnegative + positive:
            if type(raw[k]) is not int or raw[k] < (0 if k in nonnegative else 1):
                raise ValueError(f"Invalid integer field: {k}")
        if sum(raw[k] for k in nonnegative) != raw["num_hidden_layers"]:
            raise ValueError("Input/recurrent/output partitions must cover physical layers")
        if raw["num_attention_heads"] % raw["num_key_value_heads"] or raw["head_dim"] % 2:
            raise ValueError("Require divisible GQA heads and an even head_dim")
        for k in ("initializer_range", "ngpt_alpha_init"):
            if type(raw[k]) not in (int, float) or not math.isfinite(raw[k]) or raw[k] <= 0:
                raise ValueError(f"Invalid scale: {k}")
        gate = raw["attention_output_gate_logit_scale_init"]
        if gate is not None and (type(gate) not in (int, float) or not math.isfinite(gate) or gate <= 0):
            raise ValueError("Invalid attention gate scale")
        types = raw["layer_types"]
        if not isinstance(types, list) or len(types) != raw["num_hidden_layers"] or any(
            t not in ("sliding_attention", "full_attention", "mlp") for t in types
        ):
            raise ValueError("Invalid layer_types")
        rope = raw["rope_parameters"]
        attention_types = set(types) - {"mlp"}
        if not isinstance(rope, dict) or set(rope) != attention_types:
            raise ValueError("Require RoPE configuration for each attention type")
        for settings in rope.values():
            if not isinstance(settings, dict) or set(settings) != {
                "rope_type", "rope_theta", "partial_rotary_factor"
            } or settings["rope_type"] != "default" or settings["partial_rotary_factor"] != 1.0:
                raise ValueError("Pinned candidate requires explicit default full-dimensional RoPE")
            theta = settings["rope_theta"]
            if type(theta) not in (int, float) or not math.isfinite(theta) or theta <= 0:
                raise ValueError("Invalid rope_theta")
        values = {k: raw[k] for k in fields}
        values["layer_types"] = tuple(types)
        values["rope_thetas"] = tuple(sorted((k, float(v["rope_theta"])) for k, v in rope.items()))
        return cls(**values)

    def visits(self):
        """(physical weight layer, loop index): weights never acquire virtual names."""
        end = self.n_input_layers + self.n_recurrent_layers
        return (tuple((i, 0) for i in range(self.n_input_layers))
                + tuple((i, loop) for loop in range(self.n_recurrent_loops)
                        for i in range(self.n_input_layers, end))
                + tuple((i, 0) for i in range(end, self.num_hidden_layers)))

    def cache_plan(self):
        """Upstream virtual layer = loop * num_hidden_layers + physical layer."""
        return tuple({"cache_id": loop * self.num_hidden_layers + i,
                      "physical_layer": i, "loop": loop, "kind": self.layer_types[i],
                      "window": self.sliding_window if self.layer_types[i] == "sliding_attention" else None}
                     for i, loop in self.visits() if self.layer_types[i] != "mlp")

    def weight_shapes(self):
        """Expected UNPACKED BF16 artifact names, inferred from the official loader.

        This is not a checkpoint inventory: no checkpoint has been inspected.
        """
        h, v, f = self.hidden_size, self.vocab_size, self.intermediate_size
        q, kv = self.num_attention_heads * self.head_dim, self.num_key_value_heads * self.head_dim
        shapes = {"backbone.embeddings.weight": (v, h), "lm_head.weight": (v, h),
                  "lm_head.logit_scale": (v,)}
        for i, kind in enumerate(self.layer_types):
            layer = f"backbone.layers.{i}"
            if kind == "mlp":
                entries = {"gate_proj.weight": (f, h), "up_proj.weight": (f, h),
                           "down_proj.weight": (h, f), "suv_gate": (f,), "suv_up": (f,)}
                alpha = "mlp_alpha"
            else:
                entries = {"q_proj.weight": (q, h), "k_proj.weight": (kv, h),
                           "v_proj.weight": (kv, h), "g_proj.weight": (q, h),
                           "o_proj.weight": (h, q), "sqk": (q,), "s_gate": (q,)}
                alpha = "attn_alpha"
            shapes.update({f"{layer}.mixer.{name}": shape for name, shape in entries.items()})
            shapes[f"{layer}.{alpha}"] = (h,)
        return shapes

    def validate_weight_shapes(self, actual):
        expected = self.weight_shapes()
        if set(actual) != set(expected):
            raise ValueError(f"Checkpoint names differ: missing={sorted(set(expected)-set(actual))}; "
                             f"unexpected={sorted(set(actual)-set(expected))}")
        for name, shape in expected.items():
            if tuple(actual[name]) != shape:
                raise ValueError(f"Shape mismatch for {name}: expected {shape}; got {actual[name]}")


class RequestCursor:
    """Append-only single-request transaction; contains no tensors or file access.

    Tensor owners must clear every history on abort and on starting a new epoch.
    A forward commits only after every expected cache was updated once, in order.
    """
    def __init__(self, config, epoch, request_id):
        if type(epoch) is not int or epoch < 1 or not isinstance(request_id, str) or not request_id:
            raise ValueError("Require positive epoch and nonempty request identity")
        self.epoch, self.request_id = epoch, request_id
        self.maximum = config.max_position_embeddings
        self.expected = tuple(row["cache_id"] for row in config.cache_plan())
        self.length, self.pending_end, self.seen, self.alive = 0, None, [], True

    def check_owner(self, epoch, request_id):
        if type(epoch) is not int or not isinstance(request_id, str) or not self.alive or (epoch, request_id) != (self.epoch, self.request_id):
            raise ValueError("Stale, invalidated, or foreign request handle")

    def begin(self, epoch, request_id, positions):
        self.check_owner(epoch, request_id)
        positions = tuple(positions)
        if self.pending_end is not None:
            raise ValueError("A forward transaction is already pending")
        if not positions or any(type(p) is not int for p in positions):
            raise ValueError("Require nonempty integer positions")
        end = self.length + len(positions)
        if positions != tuple(range(self.length, end)) or end > self.maximum:
            raise ValueError("Positions must append contiguously inside context limit")
        self.pending_end, self.seen = end, []

    def record(self, cache_id):
        if self.pending_end is None or len(self.seen) >= len(self.expected):
            raise ValueError("No cache write is expected")
        if cache_id != self.expected[len(self.seen)]:
            raise ValueError("Missing, duplicated, or misordered loop cache write")
        self.seen.append(cache_id)

    def commit(self):
        if self.pending_end is None or tuple(self.seen) != self.expected:
            raise ValueError("Cannot commit an incomplete forward")
        self.length, self.pending_end, self.seen = self.pending_end, None, []

    def abort(self):
        self.alive, self.pending_end, self.seen = False, None, []


def allowed_key_positions(query_position, available_length, window=None):
    """Mask oracle: key <= query and, for SWA, key > query - window."""
    return tuple(k for k in range(available_length)
                 if k <= query_position and (window is None or k > query_position - window))
