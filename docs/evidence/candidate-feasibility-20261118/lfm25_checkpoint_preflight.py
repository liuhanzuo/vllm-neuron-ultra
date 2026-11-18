"""Inspect a pinned official checkpoint header and its full meta-model schema.

No weight values, compiled graph or device runtime are loaded by this probe.
"""
import ast
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import urllib.request

ROOT = Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
OUT = ROOT / 'research/candidate-feasibility-20261118'
SOURCE = OUT / 'LiquidAI--LFM2.5-230M-Base'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

with (ROOT / 'serial-enrollment.lock').open() as lock:
    assert (os.fstat(lock.fileno()).st_dev, os.fstat(lock.fileno()).st_ino) == (66306, 8294421)
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    api = json.loads((SOURCE / 'api.json').read_text())
    commit = api['sha']
    file = next(row for row in api['siblings'] if row['rfilename'] == 'model.safetensors')
    url = 'https://huggingface.co/LiquidAI/LFM2.5-230M-Base/resolve/' + commit + '/model.safetensors'
    request = urllib.request.Request(url, headers={'Range': 'bytes=0-262143', 'User-Agent': 'Trn1-model-research/1.0'})
    with urllib.request.urlopen(request, timeout=30) as response:
        assert response.status == 206, 'Range response required; refusing a full download'
        content_range = response.headers['Content-Range']
        assert content_range.startswith('bytes 0-') and content_range.endswith('/' + str(file['size']))
        data = response.read(262145)
    assert len(data) <= 262144
    header_size = struct.unpack('<Q', data[:8])[0]
    assert header_size < len(data) - 8
    header_bytes = data[8:8 + header_size]
    (SOURCE / 'safetensors-header.json').write_bytes(header_bytes)
    header = json.loads(header_bytes)
    weights = {name: entry for name, entry in header.items() if name != '__metadata__'}
    assert all(entry['dtype'] == 'BF16' for entry in weights.values())
    total = sum(math.prod(entry['shape']) for entry in weights.values())
    assert total == api['safetensors']['total'] == 229693184
    offset = 0
    for entry in sorted(weights.values(), key=lambda row: row['data_offsets'][0]):
        assert entry['data_offsets'][0] == offset
        assert entry['data_offsets'][1] - offset == math.prod(entry['shape']) * 2
        offset = entry['data_offsets'][1]
    assert offset + 8 + header_size == file['size']

    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.set_num_threads(1)
    config = AutoConfig.from_pretrained(SOURCE, local_files_only=True)
    config._attn_implementation = 'eager'
    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16, attn_implementation='eager')
    state = model.state_dict()
    shape_mismatches = {name: {'header': entry['shape'], 'model': list(state[name].shape) if name in state else None}
                        for name, entry in weights.items() if name not in state or list(state[name].shape) != entry['shape']}
    assert not shape_mismatches, shape_mismatches
    extras = sorted(set(state) - set(weights))
    assert extras == ['lm_head.weight'] and model.lm_head.weight is model.model.embed_tokens.weight
    assert all(tensor.device.type == 'meta' and tensor.dtype == torch.bfloat16 for tensor in state.values())
    assert sum(parameter.numel() for parameter in model.parameters()) == total
    assert config.rope_theta == 1000000.0 == json.loads((SOURCE / 'config.json').read_text())['rope_parameters']['rope_theta']
    constants = Path('/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/neuronx_distributed_inference/utils/constants.py')
    module = ast.parse(constants.read_text())
    registry = next(node.value for node in module.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'MODEL_TYPES' for target in node.targets))
    types = [ast.literal_eval(key) for key in registry.keys]
    assert 'lfm2' not in types
    source_module = Path(transformers.__file__).parent / 'models/lfm2/modeling_lfm2.py'
    report = {
        'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'model': api['id'], 'revision': commit,
        'official_weight_bytes': file['size'], 'official_weight_sha256_from_lfs_metadata': file['lfs']['sha256'],
        'full_weight_hash_verified': False,
        'safetensors_header_bytes': header_size, 'tensor_count': len(weights), 'total_parameters': total,
        'all_checkpoint_tensor_dtypes': ['BF16'], 'meta_model_parameter_count': total,
        'checkpoint_names_and_shapes_match_meta_model': True, 'extra_tied_state_dict_keys': extras,
        'rope_theta_matches_official_config': True,
        'transformers': transformers.__version__, 'torch': torch.__version__,
        'native_nxdi_lfm2_registered': False,
        'architecture': config.architectures, 'layer_count': config.num_hidden_layers,
        'conv_layers': config.layer_types.count('conv'), 'attention_layers': config.layer_types.count('full_attention'),
        'conv_width': config.conv_L_cache, 'attention_head_dim': config.hidden_size // config.num_attention_heads,
        'scope': 'Configuration and checkpoint header schema against the installed full BF16 meta-model only. No weight-value, forward, compiler, device, performance or quality qualification.',
        'next_step': 'Implement an isolated LFM2 baseline adapter preserving width-3 convolution, pre-norm-only residual blocks, ungated GQA and native weight names; validate against the official BF16 CPU reference before device compilation.',
        'source_sha256': {'config.json': sha(SOURCE / 'config.json'), 'safetensors-header.json': sha(SOURCE / 'safetensors-header.json'),
                          'installed_modeling_lfm2.py': sha(source_module), 'nxdi_constants.py': sha(constants),
                          'probe.py': sha(Path(__file__))}
    }
    (OUT / 'lfm25-checkpoint-preflight.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
