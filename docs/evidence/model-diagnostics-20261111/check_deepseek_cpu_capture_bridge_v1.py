"""Source/stdlib checks only; never import the adapter, Torch, NumPy or NxDI."""
import ast
from copy import deepcopy
import hashlib
import json
import logging
from pathlib import Path
from types import MethodType, SimpleNamespace as N

import deepseek_cpu_capture_bridge_v1 as b

ROOT = Path(__file__).resolve().parent
SDK = Path('/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages')
MODELS = SDK / 'neuronx_distributed_inference/models'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def method(path, class_name, name):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)


def compile_nodes(nodes, namespace=None):
    namespace = {} if namespace is None else namespace
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<source-only AST fixture>', 'exec'), namespace)
    return namespace


def reject(function):
    try:
        function()
    except (ValueError, IndexError):
        return
    raise AssertionError('negative control unexpectedly accepted')


class Index:
    def __init__(self, values):
        self.values = tuple(values); self.shape = (len(values),)
    def reshape(self, shape):
        self.reshape_shape = tuple(shape); return self
    def expand(self, shape):
        self.expanded_shape = tuple(shape); return self


class Tensor:
    def __init__(self, shape, label, dtype='torch.bfloat16', selection=None):
        self.shape = tuple(shape); self.label = label; self.dtype = dtype; self.selection = selection
    def gather(self, axis, index):
        assert len(index.values) == self.shape[0] and all(0 <= i < self.shape[axis] for i in index.values)
        shape = list(self.shape); shape[axis] = 1
        assert index.expanded_shape == tuple(shape)
        return Tensor(shape, self.label, self.dtype, (axis, index.values))
    def unsqueeze(self, axis):
        assert axis == -1
        return Tensor((*self.shape, 1), self.label, self.dtype, self.selection)


def main():
    source_paths = [MODELS / 'config.py', MODELS / 'model_base.py', MODELS / 'model_wrapper.py',
                    MODELS / 'qwen3/modeling_qwen3.py',
                    SDK / 'neuronx_distributed_inference/modules/attention/attention_base.py',
                    SDK / 'neuronx_distributed_inference/modules/attention/utils.py',
                    ROOT / 'research/transformers-reference/src/transformers/models/qwen3/modeling_qwen3.py',
                    ROOT / 'validate_deepseek_cold_kv_v2.py', ROOT / 'deepseek_qwen3_neuron_adapter.py']
    old_plan = json.loads((ROOT / 'deepseek-cold-kv-diagnostic-v2/plan.json').read_text())
    for path in source_paths:
        if str(path) in old_plan['source_sha256']:
            assert sha(path) == old_plan['source_sha256'][str(path)]
    pins = {str(path): sha(path) for path in source_paths}
    checks = []

    # Reproduce all three original failures using actual installed method ASTs.
    tree = ast.parse((MODELS / 'config.py').read_text())
    gate = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                and ast.unparse(n.test) == 'self.tensor_capture_config and (not self.output_logits)')
    subject = N(tensor_capture_config=object(), output_logits=False, on_device_sampling_config=None)
    class IncompatibleConfigError(ValueError):
        pass
    try:
        compile_nodes([gate], dict(self=subject, logging=logging, IncompatibleConfigError=IncompatibleConfigError))
    except IncompatibleConfigError as error:
        original_config = dict(error=str(error), output_logits_mutated=subject.output_logits)
    else:
        raise AssertionError('stock CPU sampler unexpectedly accepts capture')
    assert original_config['output_logits_mutated'] is True

    original_methods = compile_nodes([method(MODELS / 'model_base.py', 'NeuronBaseForCausalLM', n)
                                      for n in ('_get_constructed_outputs', '_copy_past_key_values')])
    cfg = N(output_logits=False, on_device_sampling_config=None, enable_fused_speculation=False, is_medusa=False)
    subject = N(on_device_sampling=False, neuron_config=cfg, _get_captured_tensors_offset=lambda: 2,
                _construct_output=lambda value: N(logits=value, captured_tensors=None))
    actual = original_methods['_get_constructed_outputs'](subject, ['logits', 'capture0', 'capture1'], True)
    assert actual.logits == ['logits', 'capture0', 'capture1'] and actual.captured_tensors is None
    cfg.output_logits = True
    cells = [N(data=None) for _ in range(72)]
    subject.token_generation_model = N(model=N(kv_mgr=N(past_key_values=cells)))
    subject.context_encoding_model = N(model=N(kv_mgr=N(past_key_values=[N(data=None) for _ in range(72)])))
    reject(lambda: original_methods['_copy_past_key_values'](subject, ['logits', 'duplicate_logits', 'capture0', 'capture1', *range(72)]))
    assert cells[0].data == 'capture1' and cells[1].data == 0
    checks.append('three_original_SDK_capture_CPU_sampler_failures_reproduced_from_actual_AST')

    # Instantiate successor class ASTs against inert bases; no native import.
    tensor_config_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TensorCaptureConfig')
    TensorCaptureConfig = compile_nodes([tensor_config_class])['TensorCaptureConfig']
    class QwenConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            assert kwargs['tensor_capture_config'] is None
    class App:
        def _construct_output(self, logits):
            return N(logits=logits, captured_tensors=None)
        def _get_model_outputs(self, input_ids, *args, **kwargs):
            return input_ids
    class Core:
        pass
    adapter = ROOT / 'deepseek_qwen3_capture_adapter_v1.py'
    classes = [n for n in ast.parse(adapter.read_text()).body if isinstance(n, ast.ClassDef)]
    def gather(tensor, process_group):
        assert tensor.selection is not None and tensor.shape[-1] == 1
        assert process_group.size() == 2
        return Tensor((*tensor.shape[:-1], 2), tensor.label, tensor.dtype, tensor.selection)
    ns = compile_nodes(classes, dict(bridge=b, MethodType=MethodType, Qwen3NeuronConfig=QwenConfig,
        Qwen3InferenceConfig=object, TensorCaptureConfig=TensorCaptureConfig,
        NeuronDeepSeekQwen3Model=Core, NeuronDeepSeekQwen3ForCausalLM=App,
        gather_from_tensor_model_parallel_region=gather,
        get_tensor_model_parallel_group=lambda: N(size=lambda: 2)))
    original_case = json.loads((ROOT / 'deepseek-cold-kv-diagnostic-v2/case.json').read_text())
    config_path = Path(original_case['artifact']) / 'neuron_config.json'
    original = json.loads(config_path.read_text()); pins[str(config_path)] = sha(config_path)
    prepared = b.instrument_config_dict(original)
    assert original['neuron_config']['tensor_capture_config'] is None
    changed = {k for k, v in prepared['neuron_config'].items() if v != original['neuron_config'].get(k)}
    assert changed == {'tensor_capture_config', 'deepseek_cpu_capture_version'}
    cfg = ns['DeepSeekCaptureNeuronConfig'](**prepared['neuron_config'])
    assert cfg.output_logits is False and cfg.on_device_sampling_config is None
    roundtrip = dict(vars(cfg)); roundtrip['tensor_capture_config'] = vars(cfg.tensor_capture_config)
    again = ns['DeepSeekCaptureNeuronConfig'](**roundtrip)
    assert again.tensor_capture_config.get_offset() == b.CAPTURE_COUNT
    reject(lambda: ns['DeepSeekCaptureNeuronConfig'](**dict(prepared['neuron_config'], output_logits=True)))
    reject(lambda: ns['DeepSeekCaptureNeuronConfig'](**dict(prepared['neuron_config'], on_device_sampling_config={})))
    checks.append('config_roundtrip_preserves_CPU_sampler_BF16_TP2_and_rejects_second_logits_head')

    # Verify exact SDK token padding and core position argument order.
    pad = method(MODELS / 'model_wrapper.py', 'ModelWrapper', 'pad_inputs')
    assignments = [n for n in ast.walk(pad) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'tensor_pad_vals' for t in n.targets)]
    assert len(assignments) == 1 and ast.literal_eval(assignments[0].value.elts[2]) == 1
    forward = method(MODELS / 'model_base.py', 'NeuronBaseModel', 'forward')
    assert [x.arg for x in forward.args.args[:4]] == ['self', 'input_ids', 'attention_mask', 'position_ids']
    attention = SDK / 'neuronx_distributed_inference/modules/attention/attention_base.py'
    prep = method(attention, 'NeuronAttentionBase', 'prep_qkv_tensors')
    returns = [n for n in ast.walk(prep) if isinstance(n, ast.Return)]
    assert len(returns) == 1 and [x.id for x in returns[0].value.elts] == ['Q', 'K', 'V', 'cos_cache', 'sin_cache', 'residual']
    for prefix in b.PREFIXES:
        bucket = next(x for x in (128, 256, 512, 1024, 2048, 4096) if x > prefix)
        padded = list(range(prefix)) + [1] * (bucket-prefix)
        assert max(range(len(padded)), key=padded.__getitem__) == prefix-1
    checks.append('all_seven_CTE_prediction_indices_use_actual_padding_value_1_and_TKG_column_0')

    # Directly test BSHD versus BHSD slicing and all-gather ordering.
    core = ns['DeepSeekCaptureModel'].__new__(ns['DeepSeekCaptureModel'])
    captures = {}
    for name in b.NAMES:
        shape = (4, 16, 257, 128) if b.token_axis(name) == 2 else ((4, 257, 16, 128) if name.endswith(('q_normalized', 'k_normalized')) else (4, 257, 4096))
        indices = Index([31, 127, 255, 256])
        selected = b.select_prediction(Tensor(shape, name), name, indices)
        assert selected.shape[b.token_axis(name)] == 1
        captures[name] = selected
    core._boundary_values = captures
    gathered = core._get_captured_tensors(None)
    assert [x.label for x in gathered] == list(b.NAMES)
    assert all(t.shape[-1] == 2 and t.selection[0] == b.token_axis(n) for n, t in zip(b.NAMES, gathered))
    assert all(len({gathered[b.NAMES.index(f'layers.{i}.{n}')].label for n in ('query', 'key', 'value')}) == 3 for i in range(36))
    checks.append('all_468_named_boundaries_keep_every_feature_and_Q_K_V_then_gather_both_TP_ranks')

    logits = Tensor((4, 1, 151936), 'logits', 'torch.float32')
    kv = [Tensor(b.KV_SHAPE, f'kv{i}') for i in range(72)]
    device = [logits, *gathered]; cpu = [*device, *kv]
    app = ns['DeepSeekCaptureForCausalLM'](); app.neuron_config = cfg
    result = app._get_constructed_outputs(device, True)
    assert result.logits is logits and result.captured_tensors == tuple(gathered) and result.capture_names == b.NAMES
    app.token_generation_model = N(model=N(kv_mgr=N(past_key_values=[N(data=None) for _ in range(72)])))
    app.context_encoding_model = N(model=N(kv_mgr=N(past_key_values=[N(data=None) for _ in range(72)])))
    app._copy_past_key_values(cpu)
    assert [v.data for v in app.token_generation_model.model.kv_mgr.past_key_values] == kv
    assert [v.data for v in app.context_encoding_model.model.kv_mgr.past_key_values] == kv
    reject(lambda: app._copy_past_key_values([logits, logits, *gathered, *kv]))
    reject(lambda: app._get_constructed_outputs([logits, *gathered[:-1]], True))
    reject(lambda: app._get_constructed_outputs([logits, *gathered, *kv], True))
    reject(lambda: b.check_alias_indices(range(b.KV_OFFSET+1, b.KV_OFFSET+73)))
    b.check_alias_indices(range(b.KV_OFFSET, b.KV_OFFSET+72))
    checks.append('successor_constructed_outputs_and_all72_CPU_cache_destinations_use_exact_one_head_offset')

    # Actual SDK alias arithmetic (not a copied formula), with inert model.
    get = method(MODELS / 'model_wrapper.py', 'DecoderModelInstance', 'get')
    aliases_assign = next(n for n in ast.walk(get) if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'num_output_from_trace' for t in n.targets))
    capture_increment = next(n for n in ast.walk(get) if isinstance(n, ast.If)
                             and ast.unparse(n.test) == 'self.neuron_config.tensor_capture_config')
    env = compile_nodes([aliases_assign, capture_increment], {'self': N(neuron_config=cfg)})
    assert env['num_output_from_trace'] == b.KV_OFFSET
    checks.append('actual_frozen_DecoderModelInstance_alias_header_matches_successor_ABI')

    # Ensure successor retains every original schedule/recording loop verbatim.
    old = ast.parse((ROOT / 'validate_deepseek_cold_kv_v2.py').read_text())
    new_path = ROOT / 'validate_deepseek_cold_boundary_v1.py'
    new = ast.parse(new_path.read_text())
    for function in ('main', 'reference'):
        left = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == function)
        right = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == function)
        a = [ast.dump(n, include_attributes=False) for n in ast.walk(left) if isinstance(n, ast.For)]
        z = [ast.dump(n, include_attributes=False) for n in ast.walk(right) if isinstance(n, ast.For)]
        assert a == z, function
    assert len(b.frames('native')) == 42 and len(b.frames('device')) == 84
    data = json.loads((ROOT / 'deepseek-cold-kv-diagnostic-v2/cold-buckets.json').read_text())
    neutrality = b.check_neutrality_metadata(data, deepcopy(data))
    changed = deepcopy(data); changed['experiments'][0]['rows'][0]['metrics']['cosine'] = 0.5
    reject(lambda: b.check_neutrality_metadata(data, changed))
    checks.append('original_cold_loops_AST_equal_native42_device84_all14_failure_indices_retained')

    # Reject loading any native dependency during this preparation check.
    import sys
    assert not any(n == 'torch' or n.startswith(('torch.', 'numpy', 'neuronx_distributed')) for n in sys.modules)
    outputs = [ROOT / n for n in ('deepseek_cpu_capture_bridge_v1.py', 'deepseek_qwen3_capture_adapter_v1.py',
               'deepseek_cold_boundary_capture_v1.py', 'validate_deepseek_cold_boundary_v1.py', Path(__file__).name)]
    for path in outputs:
        compile(path.read_text(), str(path), 'exec')
    result = dict(passed=True, kind='actual-source AST plus inert stdlib fixtures; no numerical or native execution',
                  checks=checks, original_failure_reproductions=dict(config=original_config,
                      capture_list_misparsed_as_logits=True, forced_second_logits_head_misaligns_KV=True),
                  source_sha256=pins, successor_sha256={str(p): sha(p) for p in outputs},
                  metadata_neutrality_control=neutrality, boundary_byte_budget=b.boundary_byte_budget(),
                  capture_graph_compiled=False, native42_executed=False, device84_executed=False,
                  raw_logits_KV_neutrality_executed=False, full_model_qualified=False, new_speedup_claim=False)
    destination = ROOT / 'research/deepseek-cpu-capture-bridge-source-checks-20261111-v1.json'
    assert not destination.exists(), 'preserve prior check evidence'
    destination.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
