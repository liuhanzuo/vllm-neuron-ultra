"""Isolated worker-factory integration. Importing this module is stdlib only.

A later, admitted entry module can bind the class returned by worker_class.
This module does not register workers, alter environment, start servers, patch
classes globally, install the DEBUG guard, or edit graph/cache identities.
"""
import os

from qwen35_sampler_pair_protocol_v1 import ARMS, PHASES, require


def check_environment(environ):
    for name in ("TRN1_CPU_SAMPLER_FAST", "NEURON_CPU_SAMPLER_FAST", "TRN1_SAMPLER_DIAGNOSTIC_DIR"):
        require(name not in environ, "mixed sampler activation forbidden: " + name)


def native_operation(reference):
    # This lazy native import is for a future admitted worker, never source tests.
    from qwen35_sampler_bounded_operation_v1 import BoundedTopKTopPSampler
    require(type(reference).__module__ == "vllm.v1.sample.ops.topk_topp_sampler"
            and type(reference).__name__ == "TopKTopPSampler", "unexpected reference sampler")
    return BoundedTopKTopPSampler(reference)


def attach(runner, arm, phase, *, operation_factory=None, observer_factory=None, environ=None):
    require(arm in ARMS and phase in PHASES, "unknown arm/phase")
    check_environment(os.environ if environ is None else environ)
    require((observer_factory is not None) == (phase == "correctness_capture"),
            "capture must be separate from measured execution")
    sampler = runner.cpu_sampler
    original = sampler.topk_topp_sampler
    require(not getattr(original, "_qwen35_pair_bound", False), "duplicate sampler attachment")
    require(not getattr(sampler, "_qwen35_pair_observed", False), "duplicate capture attachment")
    if arm == "candidate":
        operation = (operation_factory or native_operation)(original)
        require(operation is not original and operation.reference is original, "reference fallback lost")
        operation._qwen35_pair_bound = True
        sampler.topk_topp_sampler = operation
    else:
        require(operation_factory is None, "baseline must keep its original operation")
    observer = None
    if observer_factory is not None:
        observer = observer_factory(runner, sampler, arm)
        observer.install()
    return {"arm": arm, "phase": phase, "observation_enabled": observer is not None,
            "debug_behavior": "unchanged", "operator_replaced": arm == "candidate",
            "observer": observer}


def worker_class(base_worker, arm, phase, *, operation_factory=None, observer_factory=None, on_attach=None, environ=None):
    """Override only the existing factory, after the original runner is built."""
    require(arm in ARMS and phase in PHASES, "unknown arm/phase")

    class SamplerPairWorker(base_worker):
        def get_neuronx_distributed_model_runner(self, vllm_config, device):
            runner = super().get_neuronx_distributed_model_runner(vllm_config, device)
            self.qwen35_sampler_host_attachment = attach(
                runner, arm, phase, operation_factory=operation_factory,
                observer_factory=observer_factory, environ=environ)
            if on_attach is not None:
                on_attach(self, runner, self.qwen35_sampler_host_attachment)
            return runner

    return SamplerPairWorker
