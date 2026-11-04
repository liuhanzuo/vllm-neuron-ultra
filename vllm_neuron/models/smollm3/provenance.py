# SPDX-License-Identifier: Apache-2.0
"""Origin of the experimental SmolLM3 integration.

Derived HF architecture code retains its upstream copyright/license header.
Upstream: huggingface/transformers, commit
66f51e76efd8542232b8bdd40ade7d1edbaac46f,
src/transformers/models/smollm3/{modeling,configuration}_smollm3.py.
Copyright 2025 The HuggingFace Inc. team. Apache License 2.0.

Modified local port: explicit KV state, TP2 adapter, functional cache slots,
Trn1 NKI read/write kernels and fixed minimum decode routing. Those original
experiment files are pinned below. This package removes experiment launchers,
absolute workspace paths and the cross-model inheritance/import dependencies.

Evidence: smollm3-samefloor-full-v1 (2026-11-04), original numeric32/cold84/
long84/quality96 and matched5x3 serving checks. Experimental throughput belongs
to the frozen experiment, not this new integration. No model weights or SDK
binaries are bundled. See docs/smollm3.md for the unqualified branch status.
"""

UPSTREAM_SOURCES = {'modeling_smollm3.py': {'url': 'https://raw.githubusercontent.com/huggingface/transformers/66f51e76efd8542232b8bdd40ade7d1edbaac46f/src/transformers/models/smollm3/modeling_smollm3.py', 'sha256': 'fb40411ac11343de0e7b7676bbe23a5de4afa14cb5cf3b6e26b16d925360db24'}, 'configuration_smollm3.py': {'url': 'https://raw.githubusercontent.com/huggingface/transformers/66f51e76efd8542232b8bdd40ade7d1edbaac46f/src/transformers/models/smollm3/configuration_smollm3.py', 'sha256': 'ea4f3df9b4fa1d7bbc25c5bcc5f92c57d5cdefd30db7b8ec1981b58f563017e2'}}

EXPERIMENT_SOURCES = {'smollm3_text_port.py': '7f78bdc123ad679df8a04c0174250557cd604d084e57d73cecad9fec4dc69f9b', 'smollm3_neuron_adapter.py': '91477a55a5d94a9eb71f27c1732e694d93e0ab035f95e8cf7a1a78e32efe221e', 'smollm3_nki_kv_cache.py': '8d4f531a30a48aa882b5083b4e059b64d929b5be76483e3263ffb1ebdb76a0bb', 'smollm3_vllm_registration.py': 'f2c72e5749551b721b27ae5acd5670e240b869bdbac1613c91cdeda5b98669a7', 'smollm3_decode_bucket_routing.py': 'caeceefc7e57dcef27b94f80b6db197fdb80b81dc3bb44e89ff170541f869969', 'slot_cache_nki_read.py': 'ab53af7a2c4e8e4739fc751cf6a38d0f94af18e337125060fe46b720a4bbe141', 'slot_cache_nki_read_torch.py': '970ad4e7b12b97d2b49ff270991c9bd0f7aa8de40df936922c19ae410cfae771', 'slot_cache_nki_predicated.py': 'b2f0ef3a727409c9816dc9c5e77cfd13df3a1777f4cdc31c8ce3e8cd373eca7e', 'slot_cache_nki_predicated_torch.py': '33f4ffdd55d0a3eb3a1b7c9f3ddf2f7505161a7809a2fabb22956988475b7f67', 'qwen35_neuron_adapter.py': 'de89afc16b64e06235d19246db486b465f5bb45ab4f4b1d6a67199439db89fe2', 'qwen35_text_port.py': '3b560a3dcbb925a4c010a679ebe8c5532178672d77edbe768d490e307cdd2e34'}

# Relevant SDK call-chain sources additionally bind each artifact.
SDK_SOURCE_FILES = [
    ('neuronx-distributed-inference', 'neuronx_distributed_inference/models/config.py'),
    ('neuronx-distributed-inference', 'neuronx_distributed_inference/models/model_base.py'),
    ('neuronx-distributed-inference', 'neuronx_distributed_inference/models/model_wrapper.py'),
    ('neuronx-distributed-inference', 'neuronx_distributed_inference/models/application_base.py'),
    ('neuronx-distributed-inference', 'neuronx_distributed_inference/utils/constants.py'),
    ('neuronx-distributed-inference', 'neuronx_distributed_inference/utils/hf_adapter.py'),
    ('neuronx-distributed', 'neuronx_distributed/trace/model_builder.py'),
    ('neuronx-distributed-inference', 'neuronx_distributed_inference/modules/autobucketing.py'),
    ('neuronx-distributed-inference', 'neuronx_distributed_inference/modules/padding.py'),
    ('neuronx-distributed-inference', 'neuronx_distributed_inference/utils/compile_env.py'),
    ('neuronx-distributed-inference', 'neuronx_distributed_inference/utils/runtime_env.py'),
    ('neuronx-distributed', 'neuronx_distributed/parallel_layers/layers.py'),
    ('neuronx-distributed', 'neuronx_distributed/parallel_layers/mappings.py'),
    ('neuronx-distributed', 'neuronx_distributed/parallel_layers/parallel_state.py'),
    ('neuronx-distributed', 'neuronx_distributed/parallel_layers/utils.py'),
    ('neuronx-distributed', 'neuronx_distributed/parallel_layers/layers_utils.py'),
    ('neuronx-distributed', 'neuronx_distributed/parallel_layers/random.py'),
]
