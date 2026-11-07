"""Process-local compatibility fix for unpacked quantized Q/K/V bias checkpoints.

Keep this installation at module top level so multiprocessing spawn children
install the same narrow loader fix. Installed SDK source is never changed.
"""
import os
import runpy
import concurrent.futures
from types import SimpleNamespace

import torch
from neuronx_distributed.quantization.quantization_layers import QuantizedParallelLinearLayerStateDictAdaptor


def set_unpacked_bias(prefix,tensor,state_dict):
    key=prefix+'bias'
    if key not in state_dict or not isinstance(tensor,torch.Tensor):
        raise NotImplementedError('Local bias adapter supports existing unpacked tensor bias entries only: '+key)
    state_dict[key]=tensor.clone()


QuantizedParallelLinearLayerStateDictAdaptor.set_bias_to_state_dict=staticmethod(set_unpacked_bias)
print(f'Local quantized bias state-dict adapter installed in PID {os.getpid()}; SDK files unchanged',flush=True)

# Bound only ModelBuilder's compiler submission pool. The global executor and
# the serving/sampling thread pools retain their original implementations.
if os.environ.get('TRN1_COMPILE_WORKERS'):
    workers=int(os.environ['TRN1_COMPILE_WORKERS'])
    if not 1<=workers<=8:raise ValueError('TRN1_COMPILE_WORKERS must be between1and8')
    import neuronx_distributed.trace.model_builder as builder
    original_futures=concurrent.futures
    class LimitedCompilerPool(original_futures.ThreadPoolExecutor):
        def __init__(self,max_workers=None,*args,**kwargs):
            super().__init__(workers if max_workers is None else min(workers,max_workers),*args,**kwargs)
    class FuturesProxy:
        ThreadPoolExecutor=LimitedCompilerPool
        def __getattr__(self,name):return getattr(original_futures,name)
    builder.concurrent=SimpleNamespace(futures=FuturesProxy())
    print(f'Local ModelBuilder compiler pool limited to {workers} tasks; inference executors unchanged',flush=True)

if os.environ.get('TRN1_SAMPLER_DIAGNOSTIC_DIR'):
    if os.environ.get('TRN1_CPU_SAMPLER_FAST'):raise ValueError('Profile the reference sampler separately from the optimized sampler')
    from sampler_diagnostics import install
    install(os.environ['TRN1_SAMPLER_DIAGNOSTIC_DIR'])

if os.environ.get('TRN1_CPU_SAMPLER_FAST')=='1':
    from cpu_bounded_sampler import install
    install(os.environ.get('TRN1_CPU_SAMPLER_STATS_DIR'))
    from cpu_debug_guard import install as install_debug_guard
    install_debug_guard()


if __name__=='__main__':
    os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
    runpy.run_module('vllm.entrypoints.openai.api_server',run_name='__main__',alter_sys=False)
