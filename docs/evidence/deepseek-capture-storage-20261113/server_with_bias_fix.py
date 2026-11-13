"""Process-local compatibility fix for unpacked quantized Q/K/V bias checkpoints.

Keep this installation at module top level so multiprocessing spawn children
install the same narrow loader fix. Installed SDK source is never changed.
"""
import os
import runpy
import concurrent.futures
from types import SimpleNamespace

import torch
from granite_adapter import install as install_granite
install_granite()
if os.environ.get('TRN1_QWEN35_TEXT_ADAPTER')=='1':
    from qwen35_vllm_registration import install as install_qwen35
    install_qwen35()
if os.environ.get('TRN1_LFM3_ADAPTER')=='1':
    from lfm3_vllm_registration import install as install_lfm3
    install_lfm3()
if os.environ.get('TRN1_GEMMA4_TEXT_ADAPTER')=='1':
    from gemma4_vllm_registration import install as install_gemma4
    install_gemma4()
if os.environ.get('TRN1_MINISTRAL3_TEXT_ADAPTER')=='1':
    from ministral3_vllm_registration import install as install_ministral3
    install_ministral3()
if os.environ.get('TRN1_SMOLLM3_TEXT_ADAPTER')=='1':
    from smollm3_vllm_registration import install as install_smollm3
    install_smollm3()
if os.environ.get('TRN1_SMOLLM3_MIN_DECODE_BUCKET'):
    import hashlib
    from pathlib import Path
    import smollm3_decode_bucket_routing as smollm3_routing
    assert os.environ.get('TRN1_SMOLLM3_TEXT_ADAPTER')=='1'
    routing_hash=hashlib.sha256(Path(smollm3_routing.__file__).read_bytes()).hexdigest()
    assert routing_hash==os.environ['TRN1_SMOLLM3_ROUTING_SHA256']
    minimum=int(os.environ['TRN1_SMOLLM3_MIN_DECODE_BUCKET'])
    smollm3_routing.install(minimum)
    print(f'Experimental SmolLM3 minimum decode bucket={minimum}; routing SHA256={routing_hash}',flush=True)
if os.environ.get('TRN1_PHI4_TEXT_ADAPTER')=='1':
    from phi4_vllm_registration import install as install_phi4
    install_phi4()
if os.environ.get('TRN1_OLMO3_TEXT_ADAPTER')=='1':
    from olmo3_vllm_registration import install as install_olmo3
    install_olmo3()
if os.environ.get('TRN1_NANBEIGE42_TEXT_ADAPTER')=='1':
    from nanbeige42_vllm_registration import install as install_nanbeige42
    install_nanbeige42()
if os.environ.get('TRN1_RNJ15_TEXT_ADAPTER')=='1':
    from rnj15_vllm_registration import install as install_rnj15
    install_rnj15()
if os.environ.get('TRN1_NEMOTRON9B_TEXT_ADAPTER')=='1':
    from nemotron9b_vllm_registration import install as install_nemotron9b
    install_nemotron9b()
if os.environ.get('TRN1_DEEPSEEK_QWEN3_YARN')=='1':
    from deepseek_qwen3_vllm_registration import install as install_deepseek_qwen3
    install_deepseek_qwen3()
if os.environ.get('TRN1_GRANITE_PREFILL_FLASH')=='1':
    from granite_flash_adapter import install as install_granite_flash
    install_granite_flash()
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
