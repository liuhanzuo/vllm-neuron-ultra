# SPDX-License-Identifier: Apache-2.0
"""Nemotron9B identity/admission; standard library only, no work on import.

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

CONFIG_VIEW_SHA256 = '1b55f0c58f738c97e767a20e8c8f0d90f905b8dbdfe33ecd2d01c8f0e9b0413e'
MODEL_ID = 'nvidia/NVIDIA-Nemotron-Nano-9B-v2'
REVISION = '6533e8de2c68e4536bf7c411d7a3ce5734111476'
SCHEMA = "vllm-neuron-ultra-nemotron9b-v1"
VIEW_MANIFEST = "nemotron9b-ultra-view.json"
ARTIFACT_MANIFEST = "nemotron9b-ultra-artifact.json"
TOKENIZER_VIEW_SHA256 = '635172bc2434acb305fff156813c1953de25773b02226cc8a13d95f6abb9ab2b'
SOFTWARE = {'vllm': '0.11.0',
 'torch': '2.8.0',
 'torch-xla': '2.8.1',
 'torch-neuronx': '2.8.0.2.12.22436+0f1dac25',
 'neuronx-distributed': '0.17.26814+4b18de63',
 'neuronx-distributed-inference': '0.8.16251+f3ca5575',
 'neuronx-cc': '2.23.6484.0+3b612583',
 'transformers': '4.57.6'}
SOURCE_FILES = {'config.json': {'algorithm': 'sha256',
                 'digest': '642dc554a725ea216be55c3018d0a6f3dc9963b24779c3cc97f5a0ea21e9dcc3',
                 'size': 1507},
 'generation_config.json': {'algorithm': 'sha256',
                            'digest': 'dc3497ad1a44de1dd3956dc21b5fa527888f4e99d36f8dcebcdeb8b83f5969d9',
                            'size': 158},
 'special_tokens_map.json': {'algorithm': 'sha256',
                             'digest': '2a4d2e7403546286e5d75f5b6b3c197490be67fb1e2118e5c60ad5c26e6668b1',
                             'size': 422},
 'tokenizer_config.json': {'algorithm': 'sha256',
                           'digest': 'a3e4d48a0f8b4ce6fd746464199299cbdefe6ee202e34048940da2d9838a9aa6',
                           'size': 181326},
 'model.safetensors.index.json': {'algorithm': 'sha256',
                                  'digest': 'd9ef253cc311f57c1029f1551a64a2c034efb0d9d468e0bcc780846a4ee26e6e',
                                  'size': 26843},
 'tokenizer.json': {'algorithm': 'sha256',
                    'digest': '3277c00fe5fb3963b3cb7c07b7f183722d2af4d775a4aea7cfb3684d7cccbc2f',
                    'size': 17078330},
 'model-00001-of-00004.safetensors': {'algorithm': 'sha256',
                                      'digest': 'fca4d01f61c81b279e8adfac2f947fe9518b9ed0b5a41a72199aa68b61666a23',
                                      'size': 4924823528},
 'model-00002-of-00004.safetensors': {'algorithm': 'sha256',
                                      'digest': '57cd13bbc8b1d3795ab9c1609c55b25ead2f88bbd9c2e9fd978e60a3a73f7981',
                                      'size': 4937507160},
 'model-00003-of-00004.safetensors': {'algorithm': 'sha256',
                                      'digest': 'bb2ead0334c209dde90e8d0ce9ce5dac8a693626bf6d37614832cb678acb88db',
                                      'size': 4871563216},
 'model-00004-of-00004.safetensors': {'algorithm': 'sha256',
                                      'digest': '85d28ad854fdaa954b57c69779ca1c125347d34d1d2259aaa5790ca1ebd9fd95',
                                      'size': 3042598608}}
EXPECTED_MODEL = {'architectures': ['Nemotron9BNeuronTextForCausalLM'],
 'attention_bias': False,
 'attention_dropout': 0.0,
 'bos_token_id': 1,
 'chunk_size': 128,
 'conv_kernel': 4,
 'eos_token_id': 12,
 'head_dim': 128,
 'hidden_dropout': 0.0,
 'hidden_size': 4480,
 'hybrid_override_pattern': 'M-M-M-MM-M-M-M*-M-M-M*-M-M-M-M*-M-M-M-M*-M-MM-M-M-M-M-M-',
 'initializer_range': 0.02,
 'intermediate_size': 15680,
 'layer_norm_epsilon': 1e-05,
 'mamba_head_dim': 80,
 'mamba_hidden_act': 'silu',
 'mamba_num_groups': 8,
 'mamba_num_heads': 128,
 'mamba_proj_bias': False,
 'mamba_state_dim': 128,
 'max_position_embeddings': 131072,
 'mlp_bias': False,
 'mlp_hidden_act': 'relu2',
 'model_type': 'nemotron9b_neuron_text',
 'n_groups': 8,
 'num_attention_heads': 40,
 'num_hidden_layers': 56,
 'num_key_value_heads': 8,
 'num_logits_to_keep': 1,
 'num_query_groups': 8,
 'pad_token_id': 0,
 'rescale_prenorm_residual': True,
 'residual_in_fp32': False,
 'rms_norm_eps': 1e-05,
 'sliding_window': None,
 'ssm_state_size': 128,
 'tie_word_embeddings': False,
 'time_step_floor': 0.0001,
 'time_step_max': 0.1,
 'time_step_min': 0.001,
 'time_step_rank': 256,
 'torch_dtype': 'bfloat16',
 'transformers_version': '4.51.3',
 'use_bias': False,
 'use_cache': True,
 'use_conv_bias': True,
 'use_mamba_kernels': True,
 'vocab_size': 131072,
 'time_step_limit': [0.0, float('inf')]}
SCOPE = {'hardware': 'trn1',
 'dtype': 'bfloat16',
 'tp': 2,
 'decode_batch': 4,
 'prefill_batch': 1,
 'context': 4096,
 'apply_seq_ids_mask': True,
 'cpu_sampler': 'standard',
 'mamba_state': 'float32',
 'variant': 'split_nki_v5'}


class AdmissionError(RuntimeError):
    """Must not be caught by the generic loader's ValueError compile fallback."""


def require(condition, message):
    if not condition:
        raise AdmissionError(message)


def canonical_configuration(configuration):
    """Type-tag all JSON values, allowing the two pinned +Infinity fields.

    NxDI's non-quantized default quantize_clamp_bound is float('inf'). Its
    serialization preserves that value. The pinned native CPU contract also
    uses +Infinity at time_step_limit[1]. No other nonfinite value is admitted.
    Tags wrap every user value, so strings/lists/dicts cannot impersonate the
    internal SDK sentinel or conflate booleans with integers.
    """
    require(type(configuration) is dict, "Nemotron9B configuration must be an object")

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
            if (path in (("neuron_config", "quantize_clamp_bound"), ("time_step_limit", 1))
                    and value == math.inf):
                return ("sdk-positive-infinity",)
            raise AdmissionError("Unsupported nonfinite Nemotron9B configuration value at "
                                 + repr(path))
        if type(value) is str:
            return ("str", value)
        if type(value) is list:
            return ("list", tuple(visit(item, path + (index,))
                                  for index, item in enumerate(value)))
        if type(value) is dict:
            require(all(type(key) is str for key in value),
                    "Nemotron9B configuration object keys must be strings")
            return ("dict", tuple((key, visit(value[key], path + (key,)))
                                  for key in sorted(value)))
        raise AdmissionError("Unsupported Nemotron9B configuration type at " + repr(path))

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
    """Verify the pinned Hub Git-blob IDs or LFS SHA256s with bounded reads."""
    root = Path(root)
    for name, identity in expected.items():
        path = root / name
        require(Path(name).name == name and path.is_file(), "Missing pinned Nemotron9B file: " + name)
        algorithm = identity["algorithm"]
        require(algorithm in ("git-blob-sha1", "sha256"), "Unknown identity algorithm")
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            require(stat.S_ISREG(before.st_mode) and before.st_size == identity["size"],
                    "Nemotron9B file size/type differs: " + name)
            digest = hashlib.sha256() if algorithm == "sha256" else hashlib.sha1()
            if algorithm == "git-blob-sha1":
                digest.update(("blob " + str(before.st_size) + "\0").encode())
            while block := stream.read(1024 * 1024):
                digest.update(block)
            after = os.fstat(stream.fileno())
        fields = lambda value: (value.st_dev, value.st_ino, value.st_size,
                                value.st_mtime_ns, value.st_ctime_ns)
        require(fields(before) == fields(after) and digest.hexdigest() == identity["digest"],
                "Nemotron9B pinned file changed: " + name)


def tokenizer_fixture_identity():
    path = Path(__file__).with_name("tokenizer_fixtures.json")
    return digest_file(path)


def verify_view(path):
    path = Path(path)
    require(path.is_dir() and not path.is_symlink(), "Use an ordinary prepared model view")
    manifest = read_json(path / VIEW_MANIFEST)
    expected = view_manifest()
    require(manifest == expected, "Prepared view identity mismatch")
    verify_files(path, {name: row for name, row in SOURCE_FILES.items() if name not in ("config.json", "tokenizer_config.json")})
    require(digest_file(path / "config.json") == CONFIG_VIEW_SHA256, "Model view transform changed")
    require(digest_file(path / "tokenizer_config.json") == TOKENIZER_VIEW_SHA256, "Tokenizer view transform changed")
    return expected


def view_manifest():
    return dict(schema=SCHEMA,model=MODEL_ID,revision=REVISION,source_files=SOURCE_FILES,
                config_sha256=CONFIG_VIEW_SHA256,tokenizer_config_sha256=TOKENIZER_VIEW_SHA256,
                tokenizer_fixtures_sha256=tokenizer_fixture_identity(),
                tokenizer_checks={"quality_inputs":96,"raw_checks":22,"thinking_controls":2})


def selected_variant():
    require(os.environ.get("NEURON_NEMOTRON9B") == "1", "Explicit Nemotron opt-in required")
    require(os.environ.get("NEURON_CPU_SAMPLER_FAST", "0") == "0", "Historical route requires standard CPU sampler")
    require(os.environ.get("XLA_DISABLE_FUNCTIONALIZATION") == "1", "Historical XLA functionalization setting required")
    require(not any(key.startswith("TRN1_") for key in os.environ), "Historical experiment switches are unsupported")
    return "split_nki_v5"


def validate_scope(model, neuron):
    for key, value in EXPECTED_MODEL.items():
        if key in ("torch_dtype", "transformers_version"):
            continue
        require(canonical_configuration({key: model.get(key)}) == canonical_configuration({key: value}),
                "Unsupported Nemotron9B model field: " + key)
    expected = {"tp_degree": 2, "batch_size": 4, "max_batch_size": 4,
                "tkg_batch_size": 4, "ctx_batch_size": 1,
                "seq_len": 4096, "max_length": 4096, "max_context_length": 4096,
                "padding_side": "right", "apply_seq_ids_mask": True,
                "is_continuous_batching": True, "enable_bucketing": True,
                "save_sharded_checkpoint": False, "skip_sharding": False, "skip_warmup": False,
                "logical_nc_config": 1, "start_rank_id": 0, "local_ranks_size": 2,
                "ep_degree": 1, "world_size": 2, "buckets": [4096],
                "context_encoding_buckets": [128,256,512,1024,2048,4096]}
    for key, value in expected.items():
        require(canonical_configuration({"v": neuron.get(key)}) == canonical_configuration({"v": value}),
                "Unsupported Nemotron9B Neuron field: " + key)
    require(str(neuron.get("torch_dtype")) in ("bfloat16", "torch.bfloat16"), "Only original BF16 is supported")
    for key in ("quantized", "sequence_parallel_enabled", "flash_decoding_enabled", "is_prefix_caching",
                "is_block_kv_layout", "is_chunked_prefill", "enable_fused_speculation", "enable_eagle_speculation",
                "is_eagle_draft", "output_logits", "async_mode", "speculation_length", "medusa_speculation_length"):
        require(not neuron.get(key), "Unsupported Nemotron9B route: " + key)
    for key in ("on_device_sampling_config", "lora_config", "chunked_prefill_config", "token_generation_batches",
                "tensor_capture_config", "token_generation_buckets"):
        require(neuron.get(key) is None, "Unsupported Nemotron9B option: " + key)
    for key in ("cp_degree", "attention_dp_degree", "pp_degree"):
        require(neuron.get(key, 1) == 1, "Unsupported Nemotron9B parallel mode: " + key)
    # v1 admits only this original full SDK configuration, including disabled
    # secondary kernels, layouts and numeric switches. New flags need review.
    contract = read_json(Path(__file__).with_name("neuron_config_contract.json"))
    require(canonical_configuration({"neuron_config": neuron}) ==
            canonical_configuration({"neuron_config": contract}),
            "Full Nemotron9B SDK configuration differs from the pinned route")


def software_identity():
    require(sys.version_info[:2] == (3, 10), "Pinned Nemotron9B runtime requires Python 3.10")
    try:
        found = {name: importlib.metadata.version(name) for name in SOFTWARE}
    except importlib.metadata.PackageNotFoundError as exc:
        raise AdmissionError("The pinned Nemotron9B SDK environment is missing") from exc
    require(found == SOFTWARE, f"Unsupported Nemotron9B software versions: {found}")
    from .provenance import SDK_SOURCE_FILES, SDK_EXPECTED_SOURCE_SHA256
    sources = {}
    for distribution, relative in SDK_SOURCE_FILES:
        path = importlib.metadata.distribution(distribution).locate_file(relative)
        sources[f"{distribution}/{relative}"] = digest_file(path)
    require(sources == SDK_EXPECTED_SOURCE_SHA256, "Pinned SDK Python sources changed")
    from .provenance import NKI_BINARY_MODULES
    compiler = importlib.metadata.distribution("neuronx-cc")
    entries = list(compiler.files or ())
    for module in NKI_BINARY_MODULES:
        matches = [entry for entry in entries if str(entry).startswith(module + ".")
                   and str(entry).endswith(".so") and "/" not in str(entry)[len(module):]]
        require(len(matches) == 1, "Missing/ambiguous installed NKI module: " + module)
        sources[str(matches[0])] = digest_file(compiler.locate_file(matches[0]))
    return {"python": "3.10", "versions": found, "source_sha256": sources}


def source_identity():
    package = Path(__file__).resolve().parent
    plugin = package.parents[1]
    paths = list(package.glob("*.py")) + list(package.glob("licenses/*.txt")) + [
        package / "tokenizer_fixtures.json", package / "core_contract.json",
        package / "neuron_config_contract.json",
        plugin / "__init__.py", plugin / "models/__init__.py",
        plugin / "worker/neuronx_distributed_model_loader.py",
        plugin / "worker/neuronx_distributed_model_runner.py",
        plugin / "worker/cpu_sampler.py",
    ]
    return {str(path.relative_to(plugin.parent)): digest_file(path)
            for path in sorted(paths)}


def hardware_identity():
    devices = sorted(Path("/sys/class/neuron_device").glob("neuron[0-9]*"))
    require(bool(devices), "Nemotron9B requires the inspected Trn1 Neuron device")
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
            "Caller-supplied Nemotron9B build metadata is unsupported")
    variant = selected_variant()
    validate_scope(configuration, configuration.get("neuron_config", {}))
    model = verify_view(model_path)
    software = software_identity()
    return {
        "schema": SCHEMA, "variant": variant, "scope": SCOPE,
        "model": model, "configuration": configuration,
        "software": software, "hardware": hardware_identity(),
        "sources": source_identity(),
        "environment": {key: os.environ.get(key) for key in (
            "NEURON_CC_FLAGS", "NEURON_RT_STOCHASTIC_ROUNDING_EN",
            "NEURON_LOGICAL_NC_CONFIG", "XLA_USE_BF16", "XLA_DOWNCAST_BF16",
            "XLA_DISABLE_FUNCTIONALIZATION", "NEURON_CPU_SAMPLER_FAST")},
    }
