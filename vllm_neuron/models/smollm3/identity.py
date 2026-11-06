# SPDX-License-Identifier: Apache-2.0
"""Portable, fail-closed SmolLM3 identity checks; standard library only.

Hashing model or compiled files is deliberately a startup/build operation.
Never invoke these functions from a timed request or a source-only review.
"""
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path

MODEL_ID = "HuggingFaceTB/SmolLM3-3B"
REVISION = "a07cc9a04f16550a088caea529712d1d335b0ac1"
SCHEMA = "vllm-neuron-ultra-smollm3-v1"
VIEW_MANIFEST = "smollm3-ultra-view.json"
ARTIFACT_MANIFEST = "smollm3-ultra-artifact.json"
SOFTWARE = {
    "vllm": "0.11.0",
    "torch": "2.8.0",
    "torch-neuronx": "2.8.0.2.12.22436+0f1dac25",
    "neuronx-distributed": "0.17.26814+4b18de63",
    "neuronx-distributed-inference": "0.8.16251+f3ca5575",
    "neuronx-cc": "2.23.6484.0+3b612583",
    "transformers": "4.57.6",
}
SOURCE_FILES = {
    "config.json": "c72b1031274ff4626e434d0019e88e95a767460135db9ee492eb80652b786af1",
    "tokenizer_config.json": "59a64b39f05ee0acf7479bbccf99a44d9e481c1a4a494eb03838ae3c35eb29be",
    "tokenizer.json": "7b6a500b662a34eb3f0374db856ba4ad7de4c81040571d78dc0d357238930005",
    "special_tokens_map.json": "12403efb970f069ba62ddf3b66166a27254be4a198166b61ad1ee604bd3a4515",
    "generation_config.json": "3458d62b3f9654f4bbe285223b0f4e5a1e67cae61c92bb6aeaadf04468bf9e01",
    "chat_template.jinja": "3e0e275a527f249670056e0078e474eadbdd007c134b4fdfae5a7b3089702814",
    "model.safetensors.index.json": "e3a7254c086c4a78f95ad5864f435d165ffed1ed8998326ce5107b4423c15b76",
    "model-00001-of-00002.safetensors": "e5eed4113a925264c33e3cdf76cd9f547c06a207ef2270914a7274a2bf685ffd",
    "model-00002-of-00002.safetensors": "6c3ad90646457295723e4da5ee8afac47006e20407b122df5174693dd8d68a43",
}
VIEW_FILES = {
    **SOURCE_FILES,
    "config.json": "49573147d99313bea6f2006dc5a372b6bc23003cb7f9cc2f28fb73af7c2a805f",
    "tokenizer_config.json": "54f508dcb41c5c2d5b4f8e778b93fcafc8d7479bfaa006373a48c5b5d11c57b3",
}
SCOPE = {
    "hardware": "trn1",
    "dtype": "bfloat16",
    "tp": 2,
    "decode_batch": 4,
    "prefill_batch": 1,
    "context": 2048,
    "minimum_decode_bucket": 2048,
    "apply_seq_ids_mask": True,
    "cpu_sampler": "standard",
}
CACHE_VARIANTS = ("reference", "nki_read", "nki_write", "nki_read_write")


class AdmissionError(RuntimeError):
    """Must not be caught by the generic loader's ValueError compile fallback."""


def require(condition, message):
    if not condition:
        raise AdmissionError(message)


def canonical_configuration(configuration):
    """Type-tag all JSON values, allowing one documented SDK +Infinity.

    NxDI's non-quantized default quantize_clamp_bound is float('inf'). Its
    serialization preserves that value. No other nonfinite value is admitted.
    Tags wrap every user value, so strings/lists/dicts cannot impersonate the
    internal SDK sentinel or conflate booleans with integers.
    """
    require(type(configuration) is dict, "SmolLM3 configuration must be an object")

    def visit(value, path):
        if value is None:
            return ("null",)
        if type(value) is bool:
            return ("bool", value)
        if type(value) is int:
            return ("int", str(value))
        if type(value) is float:
            if math.isfinite(value):
                return ("float", value.hex())
            if (path == ("neuron_config", "quantize_clamp_bound")
                    and value == math.inf):
                return ("sdk-positive-infinity",)
            raise AdmissionError("Unsupported nonfinite SmolLM3 configuration value at "
                                 + repr(path))
        if type(value) is str:
            return ("str", value)
        if type(value) is list:
            return ("list", tuple(visit(item, path + (index,))
                                  for index, item in enumerate(value)))
        if type(value) is dict:
            require(all(type(key) is str for key in value),
                    "SmolLM3 configuration object keys must be strings")
            return ("dict", tuple((key, visit(value[key], path + (key,)))
                                  for key in sorted(value)))
        raise AdmissionError("Unsupported SmolLM3 configuration type at " + repr(path))

    return visit(configuration, ())


def read_json(path, limit=1024 * 1024):
    try:
        path = Path(path)
        require(path.is_file() and path.stat().st_size <= limit,
                f"Missing or oversized identity metadata: {path}")
        value = json.loads(path.read_text())
        require(isinstance(value, dict), f"Expected JSON object: {path}")
        return value
    except (OSError, ValueError) as exc:
        raise AdmissionError(f"Cannot read identity metadata: {path}") from exc


def digest_file(path):
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
    except OSError as exc:
        raise AdmissionError(f"Cannot hash admitted file: {path}") from exc
    return digest.hexdigest()


def verify_files(root, expected):
    root = Path(root)
    for name, digest in expected.items():
        require(digest_file(root / name) == digest,
                f"SmolLM3 file identity mismatch: {root / name}")


def verify_view(path):
    path = Path(path)
    expected = {"schema": SCHEMA, "model": MODEL_ID,
                "revision": REVISION, "files": VIEW_FILES}
    require(read_json(path / VIEW_MANIFEST) == expected,
            "Use the pinned SmolLM3 prepare_view command to create a new view")
    verify_files(path, VIEW_FILES)
    return expected


def selected_variant():
    require(os.environ.get("NEURON_SMOLLM3") == "1",
            "SmolLM3 support requires NEURON_SMOLLM3=1")
    explicit = os.environ.get("NEURON_SMOLLM3_CACHE_VARIANT")
    if explicit is not None:
        require(explicit in CACHE_VARIANTS, "Unsupported SmolLM3 cache variant")
        require("NEURON_SMOLLM3_NKI_CACHE" not in os.environ,
                "Set only one SmolLM3 cache selector")
    value = os.environ.get("NEURON_SMOLLM3_NKI_CACHE", "0")
    require(value in ("0", "1"), "NEURON_SMOLLM3_NKI_CACHE must be 0 or 1")
    require(os.environ.get("NEURON_CPU_SAMPLER_FAST", "0") == "0",
            "Initial SmolLM3 qualification requires the standard CPU sampler")
    return explicit if explicit is not None else ("nki_read_write" if value == "1" else "reference")


def validate_scope(model, neuron):
    expected_model = {
        "model_type": "smollm3_neuron_text", "hidden_size": 2048,
        "intermediate_size": 11008, "num_hidden_layers": 36,
        "num_attention_heads": 16, "num_key_value_heads": 4,
        "head_dim": 128, "vocab_size": 128256, "pad_token_id": 128004,
        "tie_word_embeddings": True, "hidden_act": "silu",
        "rope_theta": 5000000.0, "rms_norm_eps": 1e-6,
        "no_rope_layers": [1, 1, 1, 0] * 9,
        "layer_types": ["full_attention"] * 36,
    }
    for key, value in expected_model.items():
        require(model.get(key) == value, f"Unsupported SmolLM3 model field: {key}")
    expected_neuron = {
        "tp_degree": 2, "batch_size": 4, "max_batch_size": 4,
        "tkg_batch_size": 4, "ctx_batch_size": 1,
        "seq_len": 2048, "max_length": 2048, "max_context_length": 2048,
        "padding_side": "right", "apply_seq_ids_mask": True,
        "is_continuous_batching": True, "enable_bucketing": True,
        "save_sharded_checkpoint": False, "skip_sharding": False,
        "skip_warmup": False, "logical_nc_config": 1,
        "start_rank_id": 0, "local_ranks_size": 2,
        "ep_degree": 1, "world_size": 2,
    }
    for key, value in expected_neuron.items():
        require(neuron.get(key) == value, f"Unsupported SmolLM3 Neuron field: {key}")
    require(str(neuron.get("torch_dtype")) in ("bfloat16", "torch.bfloat16"),
            "Only original BF16 SmolLM3 is supported")
    for key in ("quantized", "sequence_parallel_enabled", "flash_decoding_enabled",
                "is_prefix_caching", "is_block_kv_layout", "is_chunked_prefill",
                "enable_fused_speculation", "enable_eagle_speculation",
                "is_eagle_draft", "output_logits", "async_mode",
                "speculation_length", "medusa_speculation_length"):
        require(not neuron.get(key), f"Unsupported SmolLM3 route: {key}")
    for key in ("on_device_sampling_config", "lora_config", "chunked_prefill_config",
                "token_generation_batches", "tensor_capture_config",
                "context_encoding_buckets", "token_generation_buckets"):
        require(neuron.get(key) is None, f"Unsupported SmolLM3 option: {key}")
    for key in ("cp_degree", "attention_dp_degree", "pp_degree"):
        require(neuron.get(key, 1) == 1, f"Unsupported SmolLM3 parallel mode: {key}")
    require(neuron.get("buckets") == [2048],
            "Use the original SmolLM3 bucket configuration and host floor2048")


def software_identity():
    try:
        found = {name: importlib.metadata.version(name) for name in SOFTWARE}
    except importlib.metadata.PackageNotFoundError as exc:
        raise AdmissionError("The pinned SmolLM3 SDK environment is missing") from exc
    require(found == SOFTWARE, f"Unsupported SmolLM3 software versions: {found}")
    from .provenance import SDK_SOURCE_FILES
    sources = {}
    for distribution, relative in SDK_SOURCE_FILES:
        path = importlib.metadata.distribution(distribution).locate_file(relative)
        sources[f"{distribution}/{relative}"] = digest_file(path)
    return {"versions": found, "source_sha256": sources}


def source_identity():
    package = Path(__file__).resolve().parent
    plugin = package.parents[1]
    paths = list(package.glob("*.py")) + [
        plugin / "__init__.py", plugin / "models/__init__.py",
        plugin / "worker/neuronx_distributed_model_loader.py",
        plugin / "worker/neuronx_distributed_model_runner.py",
        plugin / "worker/cpu_sampler.py",
    ]
    return {str(path.relative_to(plugin)): digest_file(path)
            for path in sorted(paths)}


def hardware_identity():
    devices = sorted(Path("/sys/class/neuron_device").glob("neuron[0-9]*"))
    require(bool(devices), "SmolLM3 requires the inspected Trn1 Neuron device")
    for device in devices:
        try:
            architecture = device / "info/architecture"
            require((architecture / "instance_type").read_text().strip() == "Trn1"
                    and (architecture / "device_name").read_text().strip() == "Trainium1",
                    "Only Trn1/Trainium1 hardware is admitted")
        except OSError as exc:
            raise AdmissionError("Cannot identify the Neuron device architecture") from exc
    # Do not bind a portable artifact to a machine serial number.
    return {"instance_type": "Trn1", "device_name": "Trainium1"}


def build_identity(model_path, configuration):
    """Bind source, model payload, configuration and toolchain before loading."""
    # Reject malformed/unexpected nonfinite config before model payload hashing,
    # software inspection, artifact directory creation or native compilation.
    canonical_configuration(configuration)
    require("metadata" in configuration and configuration["metadata"] is None,
            "Caller-supplied SmolLM3 build metadata is unsupported")
    variant = selected_variant()
    model = verify_view(model_path)
    software = software_identity()
    return {
        "schema": SCHEMA, "variant": variant, "scope": SCOPE,
        "model": model, "configuration": configuration,
        "software": software, "hardware": hardware_identity(),
        "sources": source_identity(),
        "environment": {key: os.environ.get(key) for key in (
            "NEURON_CC_FLAGS", "NEURON_RT_STOCHASTIC_ROUNDING_EN",
            "NEURON_LOGICAL_NC_CONFIG", "XLA_USE_BF16", "XLA_DOWNCAST_BF16")},
    }
