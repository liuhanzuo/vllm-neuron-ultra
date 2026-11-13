"""Fresh observer-off native42 loader; no CLI or import-time input IO.

The owning serial worker authenticates the complete original source, weight,
config and input closure, reserves resources and supplies a live raising guard.
This leaf loads the original native model and calls the frozen complete control
reference leaf. It does not authorize execution or establish raw reproduction.
"""
import hashlib
import json
from pathlib import Path
import re
import sys

MODEL_PATH = Path('/mnt/nvme/models/DeepSeek-R1-0528-Qwen3-8B')
PARAMETERS = 8190735360
NATIVE_SOURCE_SHA256 = '64fec98db8cd0edaa740baf026be6adb1d5011e38bfa050ccdf6693b1da9f95d'
NATIVE_MODULES = ('torch', 'torch_neuronx', 'neuronxcc', 'neuronx_distributed',
                  'neuronx_distributed_inference', 'numpy', 'transformers', 'vllm', 'vllm_neuron')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def paths(destination, reference_directory, data_root):
    destination, reference_directory, data_root = map(Path, (destination, reference_directory, data_root))
    require(reference_directory == destination/'cold-reference', 'Exact fresh native reference directory required')
    for path in (destination, reference_directory, data_root):
        require(path.is_absolute() and path.resolve() == path
                and not any(p.is_symlink() for p in (path, *path.parents))
                and 'deepseek-cold-kv-diagnostic-v2' not in path.parts,
                'Ordinary independent native output paths required')
    require(destination.is_dir() and data_root.is_dir(), 'Owning runtime must prepare native output roots')
    roots = (destination, data_root, MODEL_PATH)
    require(len(set(roots)) == 3 and all(a not in b.parents and b not in a.parents
            for i, a in enumerate(roots) for b in roots[i+1:]), 'Disjoint native outputs and original model required')
    targets = (reference_directory, destination/'cache-captures/native', data_root/'native')
    require(all(not p.exists() and not p.is_symlink() and p.resolve() == p for p in targets),
            'Fresh complete observer-off native outputs required')
    return destination, reference_directory, data_root


def _native_dependencies():
    import torch
    from qwen3_controls_reference import hf, canonical_config
    return torch, hf, canonical_config


def run(destination, reference_directory, data_root, execution_guard):
    """Load the original BF16/eager model and execute the entire frozen42 loop.

    execution_guard must raise if current full source/weight/config/input,
    resource, process ownership or serial admission no longer holds. A truthy
    return is not an admission. All real model reads follow this outer guard.
    Native/off, native/on, device/off and device/on require distinct processes.
    """
    require(callable(execution_guard), 'Owning authenticated serial execution guard required')
    execution_guard()
    require(__debug__ and sys.gettrace() is None
            and not any(name.split('.')[0] in NATIVE_MODULES for name in sys.modules),
            'Fresh original native process without native imports or trace required')
    selected = paths(destination, reference_directory, data_root)
    destination, reference_directory, data_root = selected
    from deepseek_cold_control_entry_v1 import reference as original_reference
    from deepseek_lifecycle_full_protocol_v2 import build
    import deepseek_cold_kv_contract_v2 as contract
    import validate_deepseek_cold_control_v1 as driver
    execution_guard()
    torch, hf, canonical_config = _native_dependencies()
    torch.set_num_threads(2)
    protocol = build()
    native_source_sha256 = hashlib.sha256(Path(hf.__file__).read_bytes()).hexdigest()
    require(native_source_sha256 == protocol['native_source_sha256'] == NATIVE_SOURCE_SHA256,
            'Original pinned native source required')
    execution_guard()
    config = canonical_config(json.loads((MODEL_PATH/'config.json').read_text()))
    execution_guard()
    model, loading = hf.Qwen3ForCausalLM.from_pretrained(
        str(MODEL_PATH), config=config, dtype=torch.bfloat16, attn_implementation='eager',
        local_files_only=True, output_loading_info=True)
    require(type(model) is hf.Qwen3ForCausalLM and type(loading) is dict
            and not any(loading.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')),
            'Complete original native class and checkpoint load required')
    require(sum(p.numel() for p in model.parameters()) == PARAMETERS, 'Original parameter count required')
    require(all(p.device.type == 'cpu' for p in model.parameters()), 'Original native CPU parameter placement required')
    require(model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr(), 'Original untied head required')
    model.eval()
    require(not model.training and str(model.dtype) == 'torch.bfloat16', 'Original BF16 eval model required')
    execution_guard()
    paths(*selected)

    # No model loop, capture hook or result transformation is introduced here.
    # Preserve an inherited forward descriptor as well as an existing override.
    own = 'forward' in vars(model)
    stored = vars(model).get('forward')
    original_forward = model.forward
    calls = 0
    def guarded_forward(*args, **kwargs):
        nonlocal calls
        execution_guard()
        result = original_forward(*args, **kwargs)
        execution_guard()
        calls += 1
        return result
    try:
        model.forward = guarded_forward
        execution_guard()
        result = original_reference(model, destination, data_root, execution_guard)
        execution_guard()
    finally:
        if own:
            model.forward = stored
        elif 'forward' in vars(model):
            delattr(model, 'forward')
    require(('forward' in vars(model)) == own and (not own or vars(model)['forward'] is stored),
            'Original forward attribute ownership restored')
    require(sys.gettrace() is None, 'No residual native observer trace permitted')
    require(result == dict(native_frames=42, control_generated=True, original_raw_reproduction_pending=True,
            full_model_qualified=False, new_speedup_claim=False) and calls == 42,
            'Entire frozen42 native control leaf required')
    execution_guard()
    manifest = contract.old.read(destination/'cache-captures/native/manifest.json')
    contract.check_manifest(manifest, 'native')
    report = contract.old.read(reference_directory/'cold-reference.json')
    require(type(report) is dict and 'error' not in report and report['complete'] is True
            and report['all_finite'] is True and report['frames'] == 42
            and report['prefixes'] == list(contract.PREFIXES)
            and report['revision'] == protocol['revision'] and report['criteria'] == protocol['criteria']
            and report['native_source_sha256'] == NATIVE_SOURCE_SHA256
            and report['source_sha256'] == hashlib.sha256(Path(driver.__file__).read_bytes()).hexdigest()
            and type(report['tensor_sha256']) is str and re.fullmatch('[a-f0-9]{64}', report['tensor_sha256']),
            'Complete original native reference metadata required')
    execution_guard()
    return dict(observer='off', native_frames=42, original_forward_calls=calls, full_native_KV_tensors=3024,
        reference_directory=str(reference_directory), destination=str(destination), data_root=str(data_root),
        original_precision='BF16', model_parameters=PARAMETERS, native_source_sha256=native_source_sha256,
        forward_attribute_restored=True, original_raw_reproduction_pending=True, device_execution_performed=False,
        neutrality_proven=False, full_model_qualified=False, new_speedup_claim=False)
