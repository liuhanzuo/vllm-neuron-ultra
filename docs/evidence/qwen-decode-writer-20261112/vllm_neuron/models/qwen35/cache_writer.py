# SPDX-License-Identifier: Apache-2.0
"""Explicit trace-time cache writer selection for the pinned Qwen3.5 TP2 model.

Import is stdlib-only. NKI is imported only by the selected decode callable.
CPU text references and every prefill graph keep the original torch writer.
This candidate still requires actual int32, sliced-KV and full-model validation.
"""
import os
from pathlib import Path

from .identity import BUCKETS, SCHEMA, digest_file, require, same_identity
from .provenance import EXPECTED_MODEL

WRITER_ENV = "NEURON_QWEN35_CACHE_WRITER"
ORIGINAL = "torch_where"
CANDIDATE = "nki_predicated_decode_v1"


def selected_cache_writer():
    mode = os.environ.get(WRITER_ENV, ORIGINAL)
    require(mode in (ORIGINAL, CANDIDATE), "Unknown Qwen3.5 cache writer")
    return mode


def cache_writer_identity():
    """Bind both arms to the same sources and exact decode-only scope."""
    mode = selected_cache_writer()
    package = Path(__file__).resolve().parent
    return dict(schema="qwen35-cache-writer-v1", mode=mode,
        prefill_writer=ORIGINAL, decode_writer=mode, version=1,
        slots_dtype="torch.int32", capacity=4, decode_rows=4, tp_degree=2,
        state_buffers_per_rank=48, state_buffers_total=96,
        payloads={"conv": {"shape": [4, 3072, 4], "dtype": "torch.bfloat16"},
                  "recurrent": {"shape": [4, 8, 128, 128], "dtype": "torch.float32"},
                  "kv": {"shape": [4, 1, "bucket", 256], "dtype": "torch.bfloat16"}},
        buckets=list(BUCKETS), graphs_per_artifact={"prefill": 6, "decode": 6},
        kv_contract="prefix view in, untouched maximum-cache tail concatenated by model",
        duplicate_slots="last write wins", inactive_slot=-1,
        source_sha256={name: digest_file(package / name)
                       for name in ("cache_writer.py", "slot_cache_nki_predicated.py")})


def resolve_cache_writer(config, prefill, original_writer):
    """Check parent/trace-worker agreement once, before binding attention layers."""
    require(type(prefill) is bool and callable(original_writer), "Explicit writer phase and original callable required")
    actual = cache_writer_identity()
    metadata = getattr(config, "metadata", None)
    require(type(metadata) is dict and set(metadata) == {"qwen35_ultra"},
            "Qwen3.5 writer requires the parent artifact identity")
    identity = metadata["qwen35_ultra"]
    require(type(identity) is dict and identity.get("schema") == SCHEMA
            and same_identity(identity.get("cache_writer"), actual),
            "Qwen3.5 parent/trace-worker writer mode or source differs")
    nc = config.neuron_config
    expected = {"tp_degree": 2, "max_batch_size": 4, "ctx_batch_size": 1,
                "tkg_batch_size": 4, "max_length": 4096,
                "batch_size": 1 if prefill else 4, "is_prefill_stage": prefill}
    require(same_identity({key: getattr(nc, key, None) for key in expected}, expected),
            "Writer requires the original TP2/B4/prefill-B1/context4096 scope")
    require(str(nc.torch_dtype) == "torch.bfloat16", "Original BF16 model dtype required")
    keys = ("layer_types", "linear_num_key_heads", "linear_num_value_heads",
            "linear_key_head_dim", "linear_value_head_dim", "linear_conv_kernel_dim",
            "num_key_value_heads", "head_dim")
    require(same_identity({key: getattr(config, key, None) for key in keys},
                          {key: EXPECTED_MODEL[key] for key in keys}),
            "Writer shape/state layout differs from the pinned Qwen3.5 model")
    require(len(config.layer_types) == 24
            and config.layer_types.count("linear_attention") == 18
            and config.layer_types.count("full_attention") == 6,
            "Original 48 state buffers per rank required")
    if prefill or actual["mode"] == ORIGINAL:
        return original_writer
    return store_slots_nki_decode


def store_slots_nki_decode(cache, updates, slots):
    """Functional conditional copies for exactly the supported four-row decode.

    Do not force contiguity or change payload dtype: KV prefix reshape/alias
    behavior must be verified in the integrated graphs before qualification.
    """
    shape = tuple(cache.shape)
    dtype = str(cache.dtype)
    allowed = ((shape == (4, 3072, 4) and dtype == "torch.bfloat16")
               or (shape == (4, 8, 128, 128) and dtype == "torch.float32")
               or (len(shape) == 4 and shape[:2] == (4, 1)
                   and shape[2] in BUCKETS and shape[3] == 256
                   and dtype == "torch.bfloat16"))
    require(allowed and tuple(updates.shape) == shape and updates.dtype == cache.dtype,
            "Unsupported Qwen3.5 decode cache/update geometry or precision")
    require(tuple(slots.shape) == (4,) and str(slots.dtype) == "torch.int32",
            "Qwen3.5 decode writer requires actual four-row int32 sequence slots")
    from .slot_cache_nki_predicated import slot_cache_gather_predicated
    result = slot_cache_gather_predicated(cache.reshape(4, -1), updates.reshape(4, -1), slots)
    return result.reshape(shape)
