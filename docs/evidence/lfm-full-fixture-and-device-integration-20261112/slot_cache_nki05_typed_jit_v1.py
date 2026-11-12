"""NKI0.5 device factory; no native import or compilation on module import."""
import os
from importlib import metadata

DEVICE_SOFTWARE = {
    'nki': '0.5.0+28631259367.ga768afa6',
    'torch': '2.9.1+cpu',
    'torch-neuronx': '2.9.0.2.15.32035+de43f57c',
    'neuronx-cc': '2.26.6360.0+6f180f47',
}


def require(value,message):
    if not value:raise RuntimeError(message)


def wrap_device_kernel(function,nki_api,environment):
    """Public NKI jit; this does not invoke the kernel or simulate it."""
    require(callable(function),'Exact observer-free function supplied')
    require(environment.get('NEURON_PLATFORM_TARGET_OVERRIDE')=='trn1', 'Explicit Trn1 target required')
    require(environment.get('NEURON_LOGICAL_NC_CONFIG')=='1', 'Original LNC1 required')
    require(environment.get('NKI_SIMULATOR','0')=='0','Device path cannot use simulator')
    return nki_api.jit(function)


def make_device_kernel(admit):
    require(callable(admit) and admit() is True,'Future owned device admission required before native import')
    require({name:metadata.version(name) for name in DEVICE_SOFTWARE}==DEVICE_SOFTWARE,
            'Distinct exact NKI0.5 device software identity; old package identity is not changed')
    import nki
    from slot_cache_nki05_typed_selector_v1 import make_kernel
    kernel=wrap_device_kernel(make_kernel(),nki,os.environ)
    require(admit() is True,'Device admission survives JIT construction')
    return kernel
