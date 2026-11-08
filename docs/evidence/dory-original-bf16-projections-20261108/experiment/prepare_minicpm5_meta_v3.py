"""Build a meta plan only when root supplies actual queue/predecessor identities.

No systemd launch, download, native import or enrollment occurs in this module.
Do not run against a prepared namespace. Root owns controller implementation.
"""
from pathlib import Path
import minicpm5_meta_protocol_v3 as p


def source_paths():
    sdk = Path('/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages')
    native = p.ROOT / 'research/transformers-reference/src'
    paths = [Path(__file__), p.SCRIPT, p.ROOT / 'minicpm5_meta_protocol_v3.py',
        p.ROOT / 'minicpm5_meta_alias_contract_v3.py', p.ROOT / 'minicpm5_meta_padding_v3.py',
        sdk / 'torch/nn/modules/module.py', sdk / 'torch/utils/_device.py']
    paths.extend(p.INPUT / name for name in ('config.json', 'generation_config.json', 'tokenizer_config.json',
        'special_tokens_map.json', 'chat_template.jinja', 'inventory.json', 'safetensors-header.json',
        'official-metadata.json', 'model.safetensors.index.json', 'fetch-provenance.json'))
    for base in (sdk, native):
        paths.extend(base / rel for rel in ('transformers/models/llama/modeling_llama.py',
            'transformers/models/llama/configuration_llama.py', 'transformers/modeling_rope_utils.py', 'transformers/configuration_utils.py'))
    paths.extend(sdk / rel for rel in ('neuronx_distributed_inference/models/llama/modeling_llama.py',
        'neuronx_distributed_inference/models/application_base.py', 'neuronx_distributed_inference/models/model_base.py',
        'neuronx_distributed_inference/models/model_wrapper.py', 'neuronx_distributed_inference/models/config.py',
        'neuronx_distributed_inference/utils/constants.py', 'neuronx_distributed_inference/modules/attention/attention_base.py',
        'neuronx_distributed_inference/modules/attention/gqa.py', 'neuronx_distributed_inference/modules/kvcache/kv_cache_manager.py',
        'neuronx_distributed_inference/modules/kvcache/utils.py',
        'neuronx_distributed/trace/trace.py', 'neuronx_distributed/parallel_layers/layers.py',
        'neuronx_distributed/utils/safetensors_utils.py', 'neuronx_distributed/utils/model_utils.py'))
    paths.append(Path('/home/ubuntu/src/vllm-neuron/vllm_neuron/worker/neuronx_distributed_model_loader.py'))
    paths.append(Path('/home/ubuntu/venvs/model_reference_2026/lib/python3.12/site-packages/__editable__.transformers-5.23.0.dev0.pth'))
    return paths


def build_plan(queue_unit, queue_script, predecessors):
    """Return an unpersisted plan; root freezes/launches after reviewing its controller."""
    queue_script = Path(queue_script).resolve()
    assert queue_script.is_file() and queue_script.suffix == '.py'
    assert queue_unit.startswith('trn1-') and queue_unit.endswith('-queue')
    assert isinstance(predecessors, list)
    seen = set()
    for row in predecessors:
        assert set(row) >= {'pid', 'start_ticks', 'script', 'unit'}
        assert isinstance(row['pid'], int) and row['pid'] > 0 and str(row['start_ticks']).isdigit()
        assert (row['pid'], str(row['start_ticks'])) not in seen
        seen.add((row['pid'], str(row['start_ticks'])))
    paths = source_paths() + [queue_script]
    return dict(schema='minicpm5-meta-v3', model=p.MODEL, checkpoint_revision=p.REVISION,
        phases=list(p.PHASES), buckets=p.BUCKETS, thinking=False, weight_payload_bytes_allowed=0,
        queue_unit=queue_unit, queue_script=str(queue_script), predecessors=predecessors,
        source_sha256={str(path): p.sha(path) for path in paths},
        scope='HF full-shape meta plus installed Llama conversion/sharder with only GQA-helper meta allocation context adapted by the harness. '
            'No weights, model forward, device, collective or compilation. Root must retain full baseline requirements.')


if __name__ == '__main__':
    raise SystemExit('Library only: root must supply reviewed queue identity and actual predecessor chain; no automatic launch.')
