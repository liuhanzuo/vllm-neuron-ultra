"""Explicit SDK host options; no configuration, tensor or graph mutation."""
import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

SDK_BASE = Path('/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/neuronx_distributed_inference/models/model_base.py')


def forward_options():
    return dict(output_attentions=False, output_hidden_states=False, return_dict=True)


def call_with_options(function, *args, **kwargs):
    bound = inspect.signature(function).bind(*args, **kwargs)
    for name, value in forward_options().items():
        if bound.arguments.get(name) is None:
            bound.arguments[name] = value
    return function(*bound.args, **bound.kwargs)


def check():
    # Execute only this exact, stdlib-only SDK method. Importing the SDK here
    # would load torch before serial admission.
    tree = ast.parse(SDK_BASE.read_text())
    methods = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
               and n.name == '_setup_func_config']
    assert len(methods) == 1
    method = methods[0]
    assert [a.arg for a in method.args.args] == ['self', 'output_attentions', 'output_hidden_states', 'return_dict']
    ns = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(SDK_BASE), 'exec'), ns)
    setup = ns['_setup_func_config']
    sparse = SimpleNamespace(text_config=SimpleNamespace(), config=SimpleNamespace())
    try:
        setup(sparse, None, None, None)
    except AttributeError as error:
        assert 'output_attentions' in str(error)
    else:
        raise AssertionError('The original sparse-config failure was not reproduced')
    assert setup(sparse, **forward_options()) == (False, False, True)
    assert not vars(sparse.text_config) and not vars(sparse.config)
    defaults = SimpleNamespace(text_config=SimpleNamespace(output_attentions=False, output_hidden_states=False),
                               config=SimpleNamespace(use_return_dict=True))
    assert setup(defaults, None, None, None) == setup(sparse, **forward_options())
    assert setup(defaults, True, True, False) == (True, True, False)
    assert call_with_options(setup, sparse, None, None, None) == (False, False, True)
    assert call_with_options(setup, sparse, True, True, False) == (True, True, False)
    assert call_with_options(setup, sparse, output_attentions=None,
                             output_hidden_states=None, return_dict=None) == (False, False, True)
    return dict(passed=True, original_sparse_config_failure_reproduced=True,
                explicit_false_flags_equal_sdk_default_false_flags=True,
                explicit_caller_values_preserved=True, configuration_mutated=False,
                sdk_imported=False, model_executed=False)
