# SPDX-License-Identifier: Apache-2.0
"""Source-bound Qwen3.5 admission. Native/model I/O is explicit, never on import."""
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import stat
import sys
from .provenance import (MODEL_ID, REVISION, SOURCE_FILES, EXPECTED_MODEL, CONFIG_VIEW_SHA256, SOFTWARE, SDK_SOURCE_FILES, SDK_EXPECTED_SOURCE_SHA256)

SCHEMA = "vllm-neuron-ultra-qwen35-split-v1"
VIEW_MANIFEST = "qwen35-ultra-view.json"
ARTIFACT_MANIFEST = "qwen35-ultra-artifact.json"
BUCKETS = [128, 256, 512, 1024, 2048, 4096]
VARIANT = "corrected_tensor_split_original_delta"

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
    require(type(configuration) is dict, "Qwen3.5 configuration must be an object")

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
            raise AdmissionError("Unsupported nonfinite Qwen3.5 configuration value at "
                                 + repr(path))
        if type(value) is str:
            return ("str", value)
        if type(value) is list:
            return ("list", tuple(visit(item, path + (index,))
                                  for index, item in enumerate(value)))
        if type(value) is dict:
            require(all(type(key) is str for key in value),
                    "Qwen3.5 configuration object keys must be strings")
            return ("dict", tuple((key, visit(value[key], path + (key,)))
                                  for key in sorted(value)))
        raise AdmissionError("Unsupported Qwen3.5 configuration type at " + repr(path))

    return visit(configuration, ())


def same_identity(actual, expected):
    """Compare identities with the SDK Infinity rule scoped to configuration."""
    if type(actual) is not dict or type(expected) is not dict or actual.keys() != expected.keys():
        return False
    left, right = dict(actual), dict(expected)
    if "configuration" in left:
        if canonical_configuration(left.pop("configuration")) != canonical_configuration(right.pop("configuration")):
            return False
    return canonical_configuration(left) == canonical_configuration(right)


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
    """Verify the pinned Hub Git-blob IDs or LFS SHA256s with bounded reads."""
    root = Path(root)
    for name, identity in expected.items():
        path = root / name
        require(Path(name).name == name and path.is_file(), "Missing pinned Qwen3.5 file: " + name)
        algorithm = identity["algorithm"]
        require(algorithm in ("git-blob-sha1", "sha256"), "Unknown identity algorithm")
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            require(stat.S_ISREG(before.st_mode) and before.st_size == identity["size"],
                    "Qwen3.5 file size/type differs: " + name)
            digest = hashlib.sha256() if algorithm == "sha256" else hashlib.sha1()
            if algorithm == "git-blob-sha1":
                digest.update(("blob " + str(before.st_size) + "\0").encode())
            while block := stream.read(1024 * 1024):
                digest.update(block)
            after = os.fstat(stream.fileno())
        fields = lambda value: (value.st_dev, value.st_ino, value.st_size,
                                value.st_mtime_ns, value.st_ctime_ns)
        require(fields(before) == fields(after) and digest.hexdigest() == identity["digest"],
                "Qwen3.5 pinned file changed: " + name)

def hardware_identity():
    devices = sorted(Path("/sys/class/neuron_device").glob("neuron[0-9]*"))
    require(bool(devices), "Qwen3.5 requires the inspected Trn1 Neuron device")
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


def selected_variant():
    require(os.environ.get("NEURON_QWEN35") == "1", "Qwen3.5 requires NEURON_QWEN35=1")
    require(os.environ.get("NEURON_ON_DEVICE_SAMPLING_DISABLED") == "1", "Qwen3.5 requires CPU sampling")
    require(os.environ.get("NEURON_CPU_SAMPLER_FAST", "0") in ("0", "1"), "Invalid CPU sampler selection")
    require(os.environ.get("XLA_DISABLE_FUNCTIONALIZATION") == "1", "Pinned split frontend requires XLA_DISABLE_FUNCTIONALIZATION=1")
    require(not any(k.startswith("TRN1_QWEN35") or k == "TRN1_CPU_SAMPLER_FAST" for k in os.environ),
            "Historical experiment switches cannot select packaged Qwen3.5")
    return VARIANT


def validate_scope(model, neuron):
    for key, value in EXPECTED_MODEL.items():
        # Pinned hf_adapter.load_pretrained_config moves both aliases into NeuronConfig.
        # Raw view config remains byte-pinned; normalized runtime BF16 is checked below.
        if key in ("dtype", "torch_dtype"):
            continue
        require(canonical_configuration({"v": model.get(key)}) == canonical_configuration({"v": value}),
                "Unsupported Qwen3.5 model field: " + key)
    expected = {"tp_degree": 2, "batch_size": 4, "max_batch_size": 4, "tkg_batch_size": 4,
                "ctx_batch_size": 1, "seq_len": 4096, "max_length": 4096, "max_context_length": 4096,
                "padding_side": "right", "apply_seq_ids_mask": True, "is_continuous_batching": True,
                "enable_bucketing": True, "buckets": [4096], "context_encoding_buckets": BUCKETS,
                "save_sharded_checkpoint": False, "skip_sharding": False, "skip_warmup": False,
                "logical_nc_config": 1, "start_rank_id": 0, "local_ranks_size": 2,
                "ep_degree": 1, "world_size": 2}
    for key, value in expected.items():
        require(canonical_configuration({"v": neuron.get(key)}) == canonical_configuration({"v": value}),
                "Unsupported Qwen3.5 Neuron field: " + key)
    require(neuron.get("torch_dtype") in ("bfloat16", "torch.bfloat16"), "Original BF16 weights required")
    require(neuron.get("token_generation_buckets") in (None, BUCKETS), "All six decode buckets required")
    for key in ("quantized", "sequence_parallel_enabled", "flash_decoding_enabled", "is_prefix_caching",
                "is_block_kv_layout", "is_chunked_prefill", "enable_fused_speculation", "enable_eagle_speculation",
                "is_eagle_draft", "output_logits", "async_mode", "speculation_length", "medusa_speculation_length"):
        require(not neuron.get(key), "Unsupported Qwen3.5 route: " + key)
    for key in ("on_device_sampling_config", "lora_config", "chunked_prefill_config", "token_generation_batches",
                "tensor_capture_config"):
        require(neuron.get(key) is None, "Unsupported Qwen3.5 option: " + key)
    for key in ("cp_degree", "attention_dp_degree", "pp_degree"):
        require(neuron.get(key, 1) == 1, "Unsupported Qwen3.5 parallel mode: " + key)


def view_manifest():
    return {"schema": SCHEMA, "model": MODEL_ID, "revision": REVISION,
            "source_files": SOURCE_FILES, "config_sha256": CONFIG_VIEW_SHA256,
            "tokenizer": "unchanged pinned files", "weights": "unchanged pinned files", "text_only": True}


def verify_view(path):
    path = Path(path)
    require(path.is_dir() and not path.is_symlink(), "Use an ordinary prepared Qwen3.5 view directory")
    require(read_json(path / VIEW_MANIFEST) == view_manifest(), "Prepare a fresh packaged Qwen3.5 text view")
    verify_files(path, {name: item for name, item in SOURCE_FILES.items() if name != "config.json"})
    require(digest_file(path / "config.json") == CONFIG_VIEW_SHA256, "Text view config changed")
    return view_manifest()


def software_identity():
    require(sys.version_info[:2] == (3, 10), "Pinned Qwen3.5 SDK requires Python 3.10")
    try:
        found = {name: importlib.metadata.version(name) for name in SOFTWARE}
    except importlib.metadata.PackageNotFoundError as exc:
        raise AdmissionError("Pinned Qwen3.5 runtime is missing") from exc
    require(found == SOFTWARE, "Qwen3.5 runtime versions differ from pinned baseline")
    sources = {f"{distribution}/{relative}": digest_file(importlib.metadata.distribution(distribution).locate_file(relative))
               for distribution, relative in SDK_SOURCE_FILES}
    require(sources == SDK_EXPECTED_SOURCE_SHA256, "Pinned SDK implementation changed; requalify separately")
    return {"python": "3.10", "versions": found, "source_sha256": sources}


def source_identity():
    package = Path(__file__).resolve().parent
    plugin = package.parents[1]
    paths = list(package.glob("*.py")) + [plugin / "__init__.py", plugin / "models/__init__.py",
        plugin / "worker/neuronx_distributed_model_loader.py",
        plugin / "worker/neuronx_distributed_model_runner.py", plugin / "worker/cpu_sampler.py"]
    return {str(path.relative_to(plugin)): digest_file(path) for path in sorted(paths)}


def build_identity(model_path, configuration):
    canonical_configuration(configuration)
    require(configuration.get("metadata", "missing") is None, "Caller build metadata must be null")
    variant = selected_variant()
    validate_scope(configuration, configuration.get("neuron_config", {}))
    return {"schema": SCHEMA, "variant": variant, "model": verify_view(model_path),
            "configuration": configuration, "software": software_identity(), "hardware": hardware_identity(),
            "sources": source_identity(),
            "environment": {k: os.environ.get(k) for k in ("NEURON_CC_FLAGS", "NEURON_RT_STOCHASTIC_ROUNDING_EN",
                "NEURON_LOGICAL_NC_CONFIG", "XLA_USE_BF16", "XLA_DOWNCAST_BF16", "XLA_DISABLE_FUNCTIONALIZATION")}}
