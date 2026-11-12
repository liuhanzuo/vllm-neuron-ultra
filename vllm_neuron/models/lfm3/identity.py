# SPDX-License-Identifier: Apache-2.0
"""LFM3 identity/admission; standard library only, no work on import.

Model and compiled-file hashing occurs only during an explicit preparation or
runtime admission. Source review must not call those operations on real data.
"""
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import stat
import sys

MODEL_ID = "LiquidAI/LFM3-1.7B"
REVISION = "ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7"
VIEW_SCHEMA = "vllm-neuron-ultra-lfm3-v1"
SCHEMA = "vllm-neuron-ultra-lfm3-sdk053-v1"
VIEW_MANIFEST = "lfm3-ultra-view.json"
ARTIFACT_MANIFEST = "lfm3-sdk053-artifact.json"
TOKENIZER_VIEW_SHA256 = "52e838a235f05e61a180b742c4ad67378a07594ff314934363ba9fc451d92e3a"
SOFTWARE = {
    "vllm": "0.16.0+empty", "vllm-neuron": "0.5.3",
    "torch": "2.9.1+cpu", "torch-neuronx": "2.9.0.2.15.32035+de43f57c",
    "neuronx-distributed": "0.19.28492+435aae2b",
    "neuronx-distributed-inference": "0.10.18399+ed62453e",
    "neuronx-cc": "2.26.6360.0+6f180f47",
    "nki": "0.5.0+28631259367.ga768afa6", "transformers": "4.57.6",
}
SOURCE_FILES = {'LICENSE': {'algorithm': 'git-blob-sha1',
             'digest': '25e731c697c4fe4c52ee654cedf7106a827748d2',
             'size': 10574},
 'chat_template.jinja': {'algorithm': 'git-blob-sha1',
                         'digest': 'd0747b8b5fc99eaecfe02a04eefcff75eb181334',
                         'size': 12129},
 'config.json': {'algorithm': 'git-blob-sha1',
                 'digest': '99c878848ecd01f7b599148315c84634b5c5c656',
                 'size': 1454},
 'generation_config.json': {'algorithm': 'git-blob-sha1',
                            'digest': '828e9a1ac7789d8b38af0dfe573ab6362598745b',
                            'size': 328},
 'model.safetensors': {'algorithm': 'sha256',
                       'digest': '135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330',
                       'size': 3457635448},
 'special_tokens_map.json': {'algorithm': 'git-blob-sha1',
                             'digest': '5d871daaacbf26aaa9cb47c2935f30e797ee35a7',
                             'size': 92},
 'tokenizer.json': {'algorithm': 'sha256',
                    'digest': '7e226e34bface23ccf6e495d157bf77186c1c30a7659880e1a8994db528c3701',
                    'size': 17908635},
 'tokenizer_config.json': {'algorithm': 'git-blob-sha1',
                           'digest': '562b1ed0762f36c964646b8737081ced6af40754',
                           'size': 1545}}
EXPECTED_MODEL = {'architectures': ['Lfm3ForCausalLM'],
 'model_type': 'lfm3',
 'hidden_size': 2048,
 'intermediate_size': 7680,
 'num_hidden_layers': 24,
 'num_attention_heads': 16,
 'num_key_value_heads': 4,
 'head_dim': 128,
 'vocab_size': 128000,
 'pad_token_id': 124893,
 'bos_token_id': 124894,
 'eos_token_id': 124900,
 'tie_word_embeddings': True,
 'conv_L_cache': 7,
 'conv_bias': False,
 'use_sandwich_norm': True,
 'norm_eps': 1e-05,
 'layer_types': ['conv',
                 'conv',
                 'full_attention',
                 'conv',
                 'sliding_attention',
                 'conv',
                 'full_attention',
                 'conv',
                 'sliding_attention',
                 'conv',
                 'full_attention',
                 'conv',
                 'sliding_attention',
                 'conv',
                 'full_attention',
                 'conv',
                 'sliding_attention',
                 'conv',
                 'full_attention',
                 'conv',
                 'sliding_attention',
                 'conv',
                 'full_attention',
                 'conv'],
 'rope_parameters': {'full_attention': {'partial_rotary_factor': 0.5,
                                        'rope_theta': 1000000.0,
                                        'rope_type': 'default'},
                     'sliding_attention': {'rope_theta': 10000, 'rope_type': 'default'}},
 'sliding_window': 256,
 'max_position_embeddings': 262144}


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
    require(type(configuration) is dict, "LFM3 configuration must be an object")

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
            raise AdmissionError("Unsupported nonfinite LFM3 configuration value at "
                                 + repr(path))
        if type(value) is str:
            return ("str", value)
        if type(value) is list:
            return ("list", tuple(visit(item, path + (index,))
                                  for index, item in enumerate(value)))
        if type(value) is dict:
            require(all(type(key) is str for key in value),
                    "LFM3 configuration object keys must be strings")
            return ("dict", tuple((key, visit(value[key], path + (key,)))
                                  for key in sorted(value)))
        raise AdmissionError("Unsupported LFM3 configuration type at " + repr(path))

    return visit(configuration, ())


def read_json(path, limit=32 * 1024 * 1024):
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
        require(Path(name).name == name and path.is_file(), "Missing pinned LFM3 file: " + name)
        algorithm = identity["algorithm"]
        require(algorithm in ("git-blob-sha1", "sha256"), "Unknown identity algorithm")
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            require(stat.S_ISREG(before.st_mode) and before.st_size == identity["size"],
                    "LFM3 file size/type differs: " + name)
            digest = hashlib.sha256() if algorithm == "sha256" else hashlib.sha1()
            if algorithm == "git-blob-sha1":
                digest.update(("blob " + str(before.st_size) + "\0").encode())
            while block := stream.read(1024 * 1024):
                digest.update(block)
            after = os.fstat(stream.fileno())
        fields = lambda value: (value.st_dev, value.st_ino, value.st_size,
                                value.st_mtime_ns, value.st_ctime_ns)
        require(fields(before) == fields(after) and digest.hexdigest() == identity["digest"],
                "LFM3 pinned file changed: " + name)


def tokenizer_fixture_identity():
    path = Path(__file__).with_name("tokenizer_fixtures.json")
    return digest_file(path)


def verify_view(path):
    path = Path(path)
    require(path.is_dir() and not path.is_symlink(), "Use an ordinary LFM3 view directory")
    manifest = read_json(path / VIEW_MANIFEST)
    expected = {"schema": VIEW_SCHEMA, "model": MODEL_ID, "revision": REVISION,
                "source_files": SOURCE_FILES, "tokenizer_config_sha256": TOKENIZER_VIEW_SHA256,
                "tokenizer_fixtures_sha256": tokenizer_fixture_identity(),
                "tokenizer_checks": {"hf_prompts": 6, "quality_inputs": 96}}
    require(manifest == expected, "Use prepare_view to build and validate a fresh LFM3 tokenizer view")
    verify_files(path, {name: row for name, row in SOURCE_FILES.items() if name != "tokenizer_config.json"})
    require(digest_file(path / "tokenizer_config.json") == TOKENIZER_VIEW_SHA256,
            "LFM3 tokenizer compatibility fields differ")
    return expected


def selected_scope():
    try:
        capacity = int(os.environ.get("NEURON_LFM3_CAPACITY", "4"))
        context = int(os.environ.get("NEURON_LFM3_CONTEXT", "2048"))
    except ValueError as exc:
        raise AdmissionError("LFM3 capacity/context must be integers") from exc
    require(capacity in (4, 8, 16) and context in (2048, 4096),
            "Only candidate B4/B8/B16 and context 2048/4096 are supported")
    return {"hardware": "trn1", "dtype": "bfloat16", "tp": 2,
            "decode_batch": capacity, "prefill_batch": 1, "context": context,
            "apply_seq_ids_mask": True, "invalid_slot": -1,
            "cpu_sampler": "NEURON_CPU_SAMPLER_FAST=1",
            "qualification": "unqualified-sdk053-candidate"}


def selected_variant():
    require(os.environ.get("NEURON_LFM3") == "1", "LFM3 requires NEURON_LFM3=1")
    value = os.environ.get("NEURON_LFM3_NKI_WRITER", "0")
    require(value in ("0", "1"), "NEURON_LFM3_NKI_WRITER must be 0 or 1")
    require(os.environ.get("NEURON_CPU_SAMPLER_FAST") == "1",
            "Both LFM3 variants require the existing fast CPU sampler")
    require(not any(key.startswith("TRN1_LFM3") or key == "TRN1_CPU_SAMPLER_FAST"
                    for key in os.environ), "Use package switches, not historical experiment switches")
    variant = "nki05_typed_decode_v1" if value == "1" else "reference_sdk053_v1"
    registration = sys.modules.get(__package__ + ".registration")
    if registration is not None and getattr(registration, "_INSTALLED", False):
        require(registration._VARIANT == variant and registration._SCOPE == selected_scope(),
                "Cannot switch installed LFM3 writer/capacity/context within a process")
    return variant


def validate_scope(model, neuron):
    for key, value in EXPECTED_MODEL.items():
        require(canonical_configuration({"v": model.get(key)}) == canonical_configuration({"v": value}),
                "Unsupported LFM3 model field: " + key)
    scope = selected_scope()
    capacity, context = scope["decode_batch"], scope["context"]
    expected = {"tp_degree": 2, "batch_size": capacity, "max_batch_size": capacity,
                "tkg_batch_size": capacity, "ctx_batch_size": 1,
                "seq_len": context, "max_length": context, "max_context_length": context,
                "padding_side": "right", "apply_seq_ids_mask": True,
                "is_continuous_batching": True, "enable_bucketing": True,
                "save_sharded_checkpoint": False, "skip_sharding": False, "skip_warmup": False,
                "logical_nc_config": 1, "start_rank_id": 0, "local_ranks_size": 2,
                "ep_degree": 1, "world_size": 2, "buckets": [context],
                "allow_input_truncation": False}
    for key, value in expected.items():
        require(canonical_configuration({"v": neuron.get(key)}) == canonical_configuration({"v": value}),
                "Unsupported LFM3 Neuron field: " + key)
    require(str(neuron.get("torch_dtype")) in ("bfloat16", "torch.bfloat16"), "Only original BF16 is supported")
    for key in ("quantized", "sequence_parallel_enabled", "flash_decoding_enabled", "is_prefix_caching",
                "is_block_kv_layout", "is_chunked_prefill", "enable_fused_speculation", "enable_eagle_speculation",
                "is_eagle_draft", "output_logits", "async_mode", "speculation_length", "medusa_speculation_length"):
        require(not neuron.get(key), "Unsupported LFM3 route: " + key)
    for key in ("on_device_sampling_config", "lora_config", "chunked_prefill_config", "token_generation_batches",
                "tensor_capture_config", "context_encoding_buckets", "token_generation_buckets"):
        require(neuron.get(key) is None, "Unsupported LFM3 option: " + key)
    for key in ("cp_degree", "attention_dp_degree", "pp_degree"):
        require(neuron.get(key, 1) == 1, "Unsupported LFM3 parallel mode: " + key)


def software_identity():
    """Bind real installed sources and binaries, including editable vLLM source.

    Called only on explicit runtime admission, never on plugin import. The
    +empty distribution label is metadata; actual vLLM Python source is hashed.
    """
    require(sys.version_info[:2] == (3, 12), "Pinned LFM3 runtime requires Python 3.12")
    try:
        found = {name: importlib.metadata.version(name) for name in SOFTWARE}
    except importlib.metadata.PackageNotFoundError as exc:
        raise AdmissionError("The pinned LFM3 SDK environment is missing") from exc
    require(found == SOFTWARE, f"Unsupported LFM3 software versions: {found}")
    sources = {}
    for name in SOFTWARE:
        distribution = importlib.metadata.distribution(name)
        entries = list(distribution.files or ())
        require(entries, "Missing installed source manifest: " + name)
        for entry in entries:
            relative = str(entry)
            if (relative.endswith((".py", ".so")) or ".so." in Path(relative).name or
                    Path(relative).name in ("METADATA", "RECORD")):
                sources[name + "/" + relative] = digest_file(distribution.locate_file(entry))
    # Editable finder RECORD does not enumerate its target's source tree.
    for name in ("vllm", "vllm_neuron"):
        module = sys.modules.get(name)
        require(module is not None and getattr(module, "__file__", None),
                "Cannot resolve loaded editable source: " + name)
        root = Path(module.__file__).resolve().parent
        files = sorted(root.rglob("*.py"))
        require(files, "Empty editable source tree: " + name)
        for path in files:
            sources["editable/" + name + "/" + str(path.relative_to(root))] = digest_file(path)
    return {"python": "3.12", "versions": found, "source_sha256": sources}


def source_identity():
    package = Path(__file__).resolve().parent
    plugin = package.parents[1]
    paths = list(package.rglob("*.py")) + [package / "tokenizer_fixtures.json",
        plugin / "__init__.py", plugin / "models/__init__.py",
        plugin / "worker/neuronx_distributed_model_loader.py",
        plugin / "worker/neuronx_distributed_model_runner.py",
        plugin / "worker/cpu_sampler.py",
        plugin / "worker/neuron_worker.py", plugin / "platform.py",
        plugin.parent / "pyproject.toml",
    ]
    return {os.path.relpath(path, plugin): digest_file(path)
            for path in sorted(paths)}


def hardware_identity():
    devices = sorted(Path("/sys/class/neuron_device").glob("neuron[0-9]*"))
    require(bool(devices), "LFM3 requires the inspected Trn1 Neuron device")
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
            "Caller-supplied LFM3 build metadata is unsupported")
    variant = selected_variant()
    validate_scope(configuration, configuration.get("neuron_config", {}))
    model = verify_view(model_path)
    software = software_identity()
    return {
        "schema": SCHEMA, "variant": variant, "scope": selected_scope(),
        "model": model, "configuration": configuration,
        "software": software, "hardware": hardware_identity(),
        "sources": source_identity(),
        "environment": {key: os.environ.get(key) for key in (
            "NEURON_CC_FLAGS", "NEURON_RT_STOCHASTIC_ROUNDING_EN",
            "NEURON_LOGICAL_NC_CONFIG", "XLA_USE_BF16", "XLA_DOWNCAST_BF16",
            "NEURON_PLATFORM_TARGET_OVERRIDE", "NKI_SIMULATOR",
            "BASE_COMPILE_WORK_DIR", "NEURON_COMPILE_CACHE_URL")},
    }
