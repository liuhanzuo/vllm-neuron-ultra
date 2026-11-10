"""Unchanged Nano4B native/meta scientific functions, in a successor namespace.

Copied exactly from authenticated v2 source; function ASTs are checked against
that original. No native imports execute at module import time. Only the work
namespace and outer execution/lifecycle are successors; scope is unchanged.
"""
from collections import Counter
from datetime import timedelta
import hashlib
import importlib.metadata
import json
from pathlib import Path
import os
import stat
import sys

from nemotron4b_process_handshake_v1 import require

ROOT = Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
WORK = ROOT / 'nemotron4b-native-prerequisite-v3'
ORIGINAL_WORKER = ROOT / 'nemotron4b_native_prerequisite_v2.py'
ORIGINAL_WORKER_SHA256 = '21955045d187fec0744cd1e9a93230e4241a0a24858a61479983137b36a0ba6d'
ORIGINAL_PLAN = ROOT / 'research/nemotron4b-native-prerequisite-plan-20261106-v2.json'
ORIGINAL_PLAN_SHA256 = '7e5f16c07ad07c56b5bd3e6f5122ac6dd13b5e09d387da18bf116e035d81a2fd'


def read(path):
    p=Path(path)
    require(p.is_absolute() and p.resolve()==p and not p.is_symlink(), 'Exact original metadata/source path')
    fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        before=os.fstat(fd);require(stat.S_ISREG(before.st_mode)and before.st_size<8*2**20,
                                  'Small original regular metadata/source')
        parts=[];size=0
        while True:
            part=os.read(fd,min(65536,8*2**20+1-size))
            if not part:break
            size+=len(part);require(size<=8*2**20,'Bounded complete original source read');parts.append(part)
        after=os.fstat(fd)
        fields=('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')
        require(size==before.st_size and all(getattr(before,k)==getattr(after,k)==getattr(p.lstat(),k) for k in fields),
                'Stable complete original source and identity')
        return b''.join(parts)
    finally:os.close(fd)


def sha(path):return hashlib.sha256(read(path)).hexdigest()
def metadata(path):return json.loads(read(path))


def inventory_from_header(plan):
    from nemotron4b_baseline_contract_v1 import original_inventory
    header = metadata(plan['actual_header'])
    inventory = {name: row['shape'] for name, row in header.items() if name != '__metadata__'}
    require(all(row['dtype'] == 'BF16' for name, row in header.items() if name != '__metadata__'), 'Actual header dtype differs')
    require(inventory == original_inventory(), 'Source inventory differs from verified actual checkpoint header')
    return inventory


def verify_native_shapes(state, inventory, expected_prefix='model.'):
    expected = {k.replace('backbone.', expected_prefix, 1): shape for k, shape in inventory.items()}
    require(set(state) == set(expected), 'Native keys differ from actual checkpoint header')
    for name, tensor in state.items():
        require(list(tensor.shape) == expected[name] and tensor.device.type == 'meta'
                and str(tensor.dtype) == 'torch.bfloat16', 'Native meta shape/device/dtype: ' + name)
    require(len(state) == 263, 'Expected original263 tensors')


def verify_rank_shapes(state, specs):
    require(set(state) == set(specs) and len(state) == 432, 'Expected431 learned entries plus rank utility')
    for name, tensor in state.items():
        spec = specs[name]
        require(list(tensor.shape) == spec['shape'], 'Rank shape mismatch: ' + name)
        require(str(tensor.dtype) == ('torch.int32' if spec['dtype'] == 'I32' else 'torch.bfloat16'), 'Rank dtype mismatch: ' + name)
        if name != 'rank_util.rank':
            require(tensor.device.type == 'meta', 'Learned tensor must remain meta: ' + name)


def native_meta(plan):
    import torch
    from transformers.models.nemotron_h import modeling_nemotron_h as hf
    from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
    from nemotron4b_baseline_contract_v1 import native_config_values, original_inventory, PATTERN
    from nemotron4b_native_reference_contract_v2 import apply_native_runtime_contract
    require(sha(hf.__file__) == plan['native_model_sha256'], 'Wrong native model source imported')
    torch.set_num_threads(1)
    raw = metadata(plan['config'])
    config = NemotronHConfig.from_dict(native_config_values(raw))
    config._attn_implementation = 'eager'
    require(config.num_hidden_layers == 42 and config.hybrid_override_pattern == PATTERN, 'Actual derived native config differs')
    require(config.mamba_ssm_cache_dtype == 'float32', 'Native recurrence cache must remain FP32')
    serialized = config.to_dict()
    require('num_hidden_layers' not in serialized, 'Pinned derived serialization behavior unexpectedly changed')
    with torch.device('meta'):
        model = hf.NemotronHForCausalLM(config).to(torch.bfloat16).eval()
    overrides = apply_native_runtime_contract(model)
    state = model.state_dict()
    verify_native_shapes(state, inventory_from_header(plan))
    require(all(x.device.type == 'meta' for x in model.parameters()), 'Native unexpectedly allocated parameter storage')
    require(sum(x.numel() for x in model.parameters()) == 3973556832, 'Native parameter count differs')
    counts = Counter(x.block_type for x in model.model.layers)
    require(counts == {'linear_attention': 21, 'mlp': 17, 'full_attention': 4}, 'Actual layer pattern differs')
    require(len(overrides) == 21, 'All Mamba dt limits must be explicitly covered')
    return dict(native_meta_passed=True, original_tensor_shapes=263, parameters=3973556832,
        layer_counts=dict(counts), derived_native_num_hidden_layers=config.num_hidden_layers,
        serialized_num_hidden_layers_missing=True, runtime_dt_overrides=overrides,
        native_model_source=str(Path(hf.__file__).resolve()), native_model_sha256=sha(hf.__file__),
        native_parameter_values_verified=False, original_weights_loaded=False,
        native_cache_forward_executed=False, dtype='BF16', reference_helper='nemotron4b_native_reference_contract_v2')


def runtime_receipt(plan, phase):
    expected = plan['environments'][phase]
    versions = {name: importlib.metadata.version(name) for name in expected['distributions']}
    require(versions == expected['distributions'], 'Native package versions changed')
    require(list(sys.version_info[:2]) == expected['python_version'], 'Python major/minor changed')
    origins = {}
    for name, path in expected['loaded_source_origins'].items():
        require(name in sys.modules, 'Required module was not actually imported: ' + name)
        actual = str(Path(sys.modules[name].__file__).resolve())
        require(actual == path and sha(actual) == plan['source_sha256'][path], 'Wrong imported module source: ' + name)
        origins[name] = dict(path=actual, sha256=sha(actual))
    return dict(executable=sys.executable, python=sys.version, distributions=versions, loaded_source_origins=origins)


def nxd_meta(plan):
    import torch
    from neuronx_distributed.parallel_layers import parallel_state
    from neuronx_distributed.trace.trace import get_sharded_checkpoint
    from neuronx_distributed.utils.model_utils import init_on_device
    from neuronx_distributed_inference.models.config import NeuronConfig
    from neuronx_distributed_inference.models.model_wrapper import DecoderModelInstance
    from neuronx_distributed_inference.utils.constants import MODEL_TYPES
    from transformers import AutoConfig
    from vllm.model_executor.models.registry import ModelRegistry
    from nemotron4b_baseline_contract_v1 import (original_inventory, sharded_entries, state_contract,
        view_config_values, ARCHITECTURE, MODEL_TYPE, NXDI_KEY)
    from nemotron4b_neuron_adapter_v1 import (Nemotron4BV1InferenceConfig, NeuronNemotron4BV1Model,
        NeuronNemotron4BV1ForCausalLM)
    from nemotron4b_vllm_registration_v1 import install, Nemotron4BNeuronTextV1ForCausalLM
    torch.set_num_threads(1)
    rank = int(os.environ['RANK']); world = int(os.environ['WORLD_SIZE'])
    require(world == 2 and rank in (0, 1), 'Actual TP2 rank environment required')
    require(ARCHITECTURE not in ModelRegistry.get_supported_archs() and NXDI_KEY not in MODEL_TYPES,
            'Registration namespace already occupied')
    before_arches = set(ModelRegistry.get_supported_archs()); before_nxd = dict(MODEL_TYPES)
    install(); install()
    require(set(ModelRegistry.get_supported_archs()) == before_arches | {ARCHITECTURE}, 'Registration changed unrelated architectures')
    require(all(MODEL_TYPES[k] is v for k, v in before_nxd.items()), 'Registration replaced existing NxDI model')
    require(MODEL_TYPES[NXDI_KEY]['causal-lm'] is NeuronNemotron4BV1ForCausalLM, 'NxDI registration mismatch')
    registered = ModelRegistry.models[ARCHITECTURE]
    require(registered.load_model_cls() is Nemotron4BNeuronTextV1ForCausalLM, 'vLLM registration mismatch')
    info = registered.inspect_model_cls()
    require(info.is_text_generation_model and not info.is_hybrid and not info.has_inner_state
            and not info.supports_multimodal, 'Generic Neuron serving wrapper contract differs')
    raw = metadata(plan['config']); view = view_config_values(raw)
    hf_config = AutoConfig.for_model(MODEL_TYPE, **{k: v for k, v in view.items() if k != 'model_type'})
    require(hf_config.num_hidden_layers == 42 and hf_config.eos_token_id == 2 and hf_config.head_dim == 128,
            'Actual registered config geometry changed')
    rendezvous = WORK / 'gloo-rendezvous'
    torch.distributed.init_process_group('gloo', init_method='file://' + str(rendezvous),
        rank=rank, world_size=world, timeout=timedelta(seconds=90))
    try:
        parallel_state.initialize_model_parallel(2, skip_collective_init=True)
        rows = []
        for prefill in (True, False):
            nc = NeuronConfig(tp_degree=2, batch_size=4, max_batch_size=4, seq_len=4096,
                max_length=4096, torch_dtype=torch.bfloat16, on_device_sampling_config=None,
                apply_seq_ids_mask=True)
            nc.is_prefill_stage = prefill
            config = Nemotron4BV1InferenceConfig(nc, **view)
            # Both guards keep constructor temporaries and registered buffers on meta.
            with torch.device('meta'), init_on_device(torch.device('meta'), include_buffers=True, force_custom_init_on_device=True):
                instance = DecoderModelInstance(NeuronNemotron4BV1Model, config)
                instance.load_module()
            model = instance.module
            require(all(p.device.type == 'meta' for p in model.parameters()), 'NxDI parameter unexpectedly materialized')
            original = {name: torch.empty(shape, dtype=torch.bfloat16, device='meta')
                        for name, shape in inventory_from_header(plan).items()}
            state = NeuronNemotron4BV1ForCausalLM.convert_hf_to_neuron_state_dict(original, config)
            get_sharded_checkpoint(state, model, rank, 2)
            specs = sharded_entries(rank)
            verify_rank_shapes(state, specs)
            require(state['rank_util.rank'].device.type == 'cpu' and int(state['rank_util.rank'].item()) == rank,
                    'Actual NxDI rank utility slice differs')
            actual = {k: v for k, v in model.state_dict().items() if not k.startswith('kv_mgr.')}
            verify_rank_shapes(actual, specs)
            contract = state_contract(); caches = model.kv_mgr.past_key_values
            require(list(model.kv_mgr.offsets) == contract['layer_offsets'], 'Layer/cache offsets differ')
            require(len(caches) == 50, 'Expected50 buffers per rank')
            for tensor, expected in zip(caches, contract['per_rank']):
                require(list(tensor.shape) == expected['shape'] and tensor.device.type == 'meta'
                    and str(tensor.dtype) == ('torch.float32' if expected['dtype'] == 'F32' else 'torch.bfloat16'),
                    'Actual NxDI cache shape/dtype differs')
            _, aliases = instance.get(0)
            require(len(aliases) == 50 and sorted(aliases.values()) == list(range(1, 51)), 'Alias outputs differ')
            require(all(aliases[tensor] == i + 1 for i, tensor in enumerate(caches)), 'Alias does not bind original cache parameter')
            model.to(torch.bfloat16)
            require(all(str(model.kv_mgr.past_key_values[x['index']].dtype) == 'torch.float32'
                        for x in contract['per_rank'] if x['kind'] == 'ssm'), 'Blanket cast lost FP32 recurrent state')
            rows.append(dict(prefill=prefill, learned_entries=431, rank_utility_entries=1,
                cache_buffers=50, alias_outputs=list(range(1, 51)), original_tensors=263,
                metadata_sharder_executed=True, full_cache_bytes_per_rank=contract['bytes_per_rank']))
            del model, instance, original, state, actual, caches
        torch.distributed.barrier()
        return dict(nxd_meta_passed=True, rank=rank, world_size=2, stages=rows,
            process_local_registration_passed=True, registration_idempotent=True,
            architecture=ARCHITECTURE, model_info=vars(info), tokenizer_executed=False,
            engine_config_executed=False, original_weight_values_verified=False,
            state_value_or_alias_execution_verified=False, device_executed=False)
    finally:
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
