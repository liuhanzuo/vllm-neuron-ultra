# SPDX-License-Identifier: Apache-2.0
"""LFM3 historical v4 source provenance.

Architecture derived from the pinned LiquidAI repository's Transformers code:
Copyright 2026 The HuggingFace Team. Apache License 2.0. Modified into explicit
hybrid cache state and an NxDI TP adapter; original source headers retained.
The model/checkpoint's separate LFM Open License 1.0 still applies to its data.
No model weights, native binaries or historical artifacts are distributed here.

The v4 experiment passed the recorded gates on 2026-10-31. These pins establish
origin only: repository integration and the packaged CPU sampler need a new
matched qualification. Later typed-writer pilots are excluded.
"""
UPSTREAM_SOURCES = {'modeling_lfm3.py': {'url': 'https://huggingface.co/LiquidAI/LFM3-1.7B/resolve/ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7/modeling_lfm3.py',
                      'sha256': 'e8cd5254ea99ce6bd731b57b10908c2e8b5010ce8315341f895a38484885eeab'},
 'configuration_lfm3.py': {'url': 'https://huggingface.co/LiquidAI/LFM3-1.7B/resolve/ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7/configuration_lfm3.py',
                           'sha256': '176cced35657aedfea7d7002221360582d1c05fcaca882c21508bc3a73b282e5'},
 'lfm3_conv.py': {'url': 'https://huggingface.co/LiquidAI/LFM3-1.7B/resolve/ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7/lfm3_conv.py',
                  'sha256': '80d4a985382a22774212759248a6b4c737bfbf9c2a47b3dd7f797dda748e4bb4'}}

EXPERIMENT_SOURCES = {'lfm3_text_port.py': '54c13e7c220d2c00740ca27bf0608b831137368bc7da0636a8c8c84e0be481b9',
 'lfm3_neuron_adapter.py': 'e376364bfd9da851cec02c8efce107e589a9066a967a014122381cd36a361be5',
 'lfm3_vllm_registration.py': 'ddadc08b60d11333d1ff6ae3969d1584eabe54e325f15d211d6a34bc7ab8f64d',
 'lfm3_nki_predicated_slot_update.py': '5dc4fd5483083589d402de104ad208fa2d6cc5cb43b52a0e3bd5479ee7846394',
 'slot_cache_nki_predicated.py': 'b2f0ef3a727409c9816dc9c5e77cfd13df3a1777f4cdc31c8ce3e8cd373eca7e',
 'slot_cache_nki_predicated_torch.py': '33f4ffdd55d0a3eb3a1b7c9f3ddf2f7505161a7809a2fabb22956988475b7f67',
 'lfm3_nki_predicated_slot_runtime.py': '44d29fc3561eb9239e98e594946e9bf002abb126449d2072ee96d877cfc491ab',
 'server_with_lfm3_nki_predicated_slot_update.py': 'ec779755ed4004f879605589c4fd8a1f8f7ff52b39f0d922e82dcee74cbb99b5',
 'qwen35_neuron_adapter.py': 'de89afc16b64e06235d19246db486b465f5bb45ab4f4b1d6a67199439db89fe2',
 'qwen35_text_port.py': '3b560a3dcbb925a4c010a679ebe8c5532178672d77edbe768d490e307cdd2e34'}

SDK_SOURCE_FILES = [('neuronx-distributed-inference', 'neuronx_distributed_inference/models/config.py'),
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
 ('neuronx-distributed', 'neuronx_distributed/parallel_layers/random.py')]

NKI_BINARY_MODULES = ['neuronxcc/nki/__init__',
 'neuronxcc/nki/compile',
 'neuronxcc/nki/language/memory_ops',
 'neuronxcc/nki/language/math_ops',
 'neuronxcc/nki/isa/__init__',
 'neuronxcc/nki/isa/neuron_isa',
 'neuronxcc/nki/isa/constants']

# These are small metadata origins, not a claim of current binary qualification.
SOURCE_METADATA_SHA256 = {'research/LiquidAI__LFM3-1.7B.metadata.json': 'b831b2d9cf7def22261ae86d5c50621a4fc2f7846159649bbe310bfdaae6524a',
 'research/LiquidAI__LFM3-1.7B.config.json': '3d0d90df0d85cdb9aa861bebe67118b0c7de1fdc6679fca9900959cb6d75287e',
 'research/lfm3-reference-package-manifest.json': '210e751a7e5f88d6d9a773f753c637962bbff8899cd14851d75e9c5715d13d0f',
 'lfm3-tokenizer-view-manifest.json': '8319d5f2ee57b3e930cb250e03ded7750e25b2f3f7675e52ac4f16aad66cbf4e',
 'lfm3-nki-predicated-slot-resume/plan.json': '947b8db498e1bf698a252d33ea30ab298409678c42b77306e335f18235fc7166'}
SOFTWARE_VERSION_METADATA = {'torch-neuronx': {'version': '2.8.0.2.12.22436+0f1dac25',
                   'metadata_sha256': 'f80a6048df29078b6353b95d3e7fe4cf52b98d01b1eb7ad9326e9932f5556b37'},
 'torch': {'version': '2.8.0',
           'metadata_sha256': '57ede492698857503164f9ab76185bc0d70aafd1e0078e3ea4f137a56320433c'},
 'vllm': {'version': '0.11.0',
          'metadata_sha256': '7faa4f3a4a2588709bc8e7bf3bb0fd80c4f0a22ef559ada8e34793f4ada36e8f'},
 'neuronx-distributed-inference': {'version': '0.8.16251+f3ca5575',
                                   'metadata_sha256': '739871500b36569a49af69ccd62eea8dd26d5d3ce6b1308a7b45d835e82b00d7'},
 'neuronx-cc': {'version': '2.23.6484.0+3b612583',
                'metadata_sha256': 'cb8d5ae74efe0a7403198ec68e91f920b612ba2339605de2bd1dc8f2f802fac2'},
 'neuronx-distributed': {'version': '0.17.26814+4b18de63',
                         'metadata_sha256': '3ac3d0d896495380b734e3414682458c5ac1a3f9d21ce13551b276c4e721bdfc'},
 'transformers': {'version': '4.57.6',
                  'metadata_sha256': '91601044a74fe02571a825356754dbbaa9e1d3e3dd3c2237254c15e903e421e3'}}
HISTORICAL_ROOT_CONFIG = {'relative_artifact': 'latest-20261031-lfm3-17b-bf16-tp2-b4-nki-predicated-slot-v4/neuron_config.json',
 'sha256': '243df3eba317cf196efcde6c5eee47da39c1edaf13adb4ccab9d022a3751a0ba',
 'buckets': [2048],
 'context_encoding_buckets': None,
 'token_generation_buckets': None,
 'decode_routes_observed': [128, 256, 512, 1024, 2048],
 'note': 'Root config matches the pinned v4 artifact; no Smol minimum-decode-floor override is '
         'installed.'}
