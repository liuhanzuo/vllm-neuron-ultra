"""Exact Torch namespace provenance for the local-source verifier; inert import.

torch.ops/classes are ModuleType instances with deliberately relative synthetic
__file__ labels. Authenticate their installed constructors and identity instead
of resolving those labels against the experiment cwd. No other relative path
is exempted. This helper imports no Torch/NKI/NumPy and performs no IO itself.
"""
import hashlib
from pathlib import Path
from types import CodeType, FunctionType, ModuleType

PREFIX = '/home/ubuntu/venvs/vllm_neuron_053_py312/lib/python3.12/site-packages'
SOURCE_SHA256 = {
    PREFIX + '/torch/__init__.py': '3caf7f40140ede2465bde40b9003af10cbbc8f7bcf436fa1de026471daa1b288',
    PREFIX + '/torch/_ops.py': '3b58079d356d15cc73082c20b1279468ee9cfe7f5e1340d329a93b6e877a30d5',
    PREFIX + '/torch/_classes.py': '0db3dc302e28fd085065ea2883ce33fa4af406b9fd1439f2a431f053e9a764b9',
    PREFIX + '/nki/framework/_torch_convert.py': '2c510075ca5943ce655f3edd21553398ae6d83d05194ddaca21d8da3ff8ea729',
}
NAMES = {
    'torch.ops': ('ops', 'torch._ops', '_Ops', '_ops.py'),
    'torch.classes': ('classes', 'torch._classes', '_Classes', '_classes.py'),
}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def source_class_code(raw, path, name):
    """Compile text only, never execute the installed package or class."""
    require(type(raw) is bytes and len(raw) <= 2**20, 'Bounded exact installed Python source text')
    module = compile(raw, path, 'exec', dont_inherit=True, optimize=0)
    matches = [code for code in module.co_consts if isinstance(code, CodeType) and code.co_name == name]
    require(len(matches) == 1, 'Unique original top-level Torch namespace class')
    return matches[0]


def known_namespace(name, module, modules, pins, read_source):
    """Only a fully authenticated exact torch.ops/classes singleton returns True.

    Unknown names return False and remain subject to the caller's original
    file/path/hash check. A known name with changed provenance fails closed.
    The callback reads source text only; it must not import a native package.
    """
    if name not in NAMES:
        return False
    require(callable(read_source), 'Explicit source-text reader')
    attr, backend_name, class_name, label = NAMES[name]
    torch = modules.get('torch')
    backend = modules.get(backend_name)
    require(type(torch) is ModuleType and type(backend) is ModuleType,
            'Exact real Torch package and namespace implementation modules')
    parent_data, backend_data = vars(torch), vars(backend)
    for loaded, package_name, source in (
            (parent_data, 'torch', PREFIX + '/torch/__init__.py'),
            (backend_data, backend_name, PREFIX + '/torch/' + label)):
        spec = loaded.get('__spec__')
        require(loaded.get('__name__') == package_name and loaded.get('__file__') == source
                and Path(source).is_absolute() and getattr(spec, 'origin', None) == source,
                'Original installed namespace defining source, never a foreign package or cwd alias')
    cls = backend_data.get(class_name)
    require(type(cls) is type and cls.__bases__ == (ModuleType,)
            and cls.__module__ == backend_name and cls.__name__ == cls.__qualname__ == class_name
            and type(module) is cls and modules.get(name) is module
            and parent_data.get(attr) is backend_data.get(attr) is module,
            'Original namespace singleton/type across sys.modules, Torch and defining module')
    data = vars(module)
    require(data.get('__name__') == name and '__file__' not in data
            and vars(cls).get('__file__') == label,
            'Exact original synthetic label without instance override')
    # All four explicit pins are required even though only the defining class
    # is compiled here. The caller already verifies the entire source map.
    require(all(pins.get(path) == sha for path, sha in SOURCE_SHA256.items()),
            'Pinned namespace sources and the actual NKI lazy-Torch conversion route')
    path = PREFIX + '/torch/' + label
    raw = read_source(path)
    require(type(raw) is bytes and hashlib.sha256(raw).hexdigest() == SOURCE_SHA256[path],
            'Original installed namespace constructor source content')
    class_code = source_class_code(raw, path, class_name)
    for method_name in ('__init__', '__getattr__'):
        candidates = [code for code in class_code.co_consts
                      if isinstance(code, CodeType) and code.co_name == method_name]
        method = vars(cls).get(method_name)
        require(len(candidates) == 1 and type(method) is FunctionType
                and method.__globals__ is backend_data and method.__code__ == candidates[0],
                'Original namespace constructor/lookup code, not a replaced class or callable')
    return True
