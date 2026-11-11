"""Early native dependency gate around the unchanged v4 scientific driver.

No native import on module import. A future separately admitted worker checks
the actual native import set before model allocation, then checks the loading
dependencies before the unchanged24-checkpoint/72-arm scientific capture.
"""
import importlib
from pathlib import Path
from research import minicpm5_layer1_operator_native_driver_v4 as scientific
from research import minicpm5_native_python_sources_v1 as native_sources

require = scientific.require


def run(context, guard, native_bundle):
    guard()
    context.guard('native')
    require(context._sources_checked, 'Original complete source authentication before native import')
    # All package Python contents were authenticated by the unchanged full source
    # phase. Recheck the finite filename set before executing its lazy imports.
    native_sources.require_package_coverage(context.plan, guard)
    import minicpm5_layer0_native_adapter_v2 as loader
    from minicpm5_layer0_python_trace_source_v1 import source_text
    with context.activate():
        native = importlib.import_module('transformers.models.llama.modeling_llama')
        require(Path(native.__file__).resolve() == native_sources.PACKAGE / 'models/llama/modeling_llama.py',
                'Original editable native Llama implementation before model allocation')
        context.verify_loaded_sources()
        torch, model, reference, records = loader.load(context)
        module = importlib.import_module(type(model.model.layers[1].self_attn).__module__)
        require(module is native, 'Original imported native module retained after official load')
        source = source_text(context, module, 'native')
        context.verify_loaded_sources()
        del reference, records
        return scientific.run_loaded(torch, model, module, context, guard, native_bundle, source)
