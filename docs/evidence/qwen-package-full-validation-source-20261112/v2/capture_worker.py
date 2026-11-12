"""Spawn-importable diagnostic workers observing the package's installed sampler.

Only capture uses these workers. No sampler is replaced here: the unchanged
package runner already applies NEURON_CPU_SAMPLER_FAST to its own instance.
"""
import hashlib
import json
import os
from pathlib import Path

from .runtime import read, require, validate_case
from .capture_writer import CaptureFileWriter, nvme_destination

EXPORTS = {'BaselineCaptureWorker': 'baseline', 'CandidateCaptureWorker': 'candidate'}
REFERENCE = 'vllm.v1.sample.ops.topk_topp_sampler.TopKTopPSampler'
CANDIDATE = 'vllm_neuron.worker.cpu_sampler.BoundedTopKTopPSampler'
ENV_PREFIX = 'QWEN35_PACKAGE_CAPTURE_'


def worker_name(arm):
    require(arm in EXPORTS.values(), 'Unknown capture arm')
    return __name__ + '.' + next(k for k, v in EXPORTS.items() if v == arm)


def class_name(value):
    return type(value).__module__ + '.' + type(value).__name__


def capture_identity(case, arm, artifact_manifest_sha256):
    validate_case(case)
    require(arm in EXPORTS.values() and type(artifact_manifest_sha256) is str
            and len(artifact_manifest_sha256) == 64, 'Exact package capture identity required')
    sources = read(Path(__file__).with_name('capture_sources.json'))
    value = dict(schema='qwen35_package_sampler_capture_v2', arm=arm, phase='correctness_capture',
        case=case, artifact_manifest_sha256=artifact_manifest_sha256,
        source_sha256=sources, worker_class=worker_name(arm),
        operation_class=REFERENCE if arm == 'baseline' else CANDIDATE,
        reference_class=REFERENCE, observation_enabled=True, timing_eligible=False)
    value['sha256'] = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return value


def check_environment(environ, arm):
    require(arm in EXPORTS.values(), 'Unknown capture arm')
    require(environ.get('NEURON_CPU_SAMPLER_FAST') == ('0' if arm == 'baseline' else '1'),
            'Capture must observe the actual package sampler arm')
    require(environ.get('NEURON_QWEN35') == environ.get('NEURON_ON_DEVICE_SAMPLING_DISABLED') == '1'
            and environ.get('NEURON_QWEN35_ALLOW_COMPILE') == '0', 'Capture requires package load-only CPU sampling')
    require(not any(k.startswith(('TRN1_QWEN35', 'QWEN35_SAMPLER_'))
                    or k in ('TRN1_CPU_SAMPLER_FAST', 'TRN1_SAMPLER_DIAGNOSTIC_DIR') for k in environ),
            'Research sampler activation/capture is forbidden')


def attach(runner, arm, observer_factory, environ=None):
    check_environment(os.environ if environ is None else environ, arm)
    require(callable(observer_factory), 'Capture observer required')
    sampler = runner.cpu_sampler
    operation = sampler.topk_topp_sampler
    require(not getattr(sampler, '_qwen35_pair_observed', False), 'Duplicate capture attachment')
    require(class_name(operation) == (REFERENCE if arm == 'baseline' else CANDIDATE),
            'Runner did not install the actual package operation')
    reference = getattr(operation, 'reference', operation)
    require(class_name(reference) == REFERENCE, 'Single original reference fallback required')
    observer = observer_factory(runner, sampler, arm)
    observer.install()
    require(sampler.topk_topp_sampler is operation, 'Observation replaced the package sampler')
    return dict(arm=arm, phase='correctness_capture', observation_enabled=True,
        debug_behavior='unchanged', operator_replaced=arm == 'candidate',
        operator_installed_by_package=True, observer_only_attachment=True,
        operation_class=class_name(operation), reference_class=class_name(reference), observer=observer)


def make_observer(runner, sampler, arm):
    from qwen35_sampler_native_capture_v1 import CaptureObserver, NativeSnapshotter
    return CaptureObserver(runner, sampler, arm,
        CaptureFileWriter(os.environ[ENV_PREFIX + 'ROOT'], arm), NativeSnapshotter())


def record_attachment(worker, runner, attachment):
    identity = json.loads(os.environ[ENV_PREFIX + 'IDENTITY'])
    expected = capture_identity(identity['case'], attachment['arm'], identity['artifact_manifest_sha256'])
    require(identity == expected, 'Capture identity/source set changed')
    for name, digest in identity['source_sha256'].items():
        source = Path(name)
        require(source.is_absolute() and source.stat().st_size < 4 * 1024**2, 'Invalid capture source')
        require(hashlib.sha256(source.read_bytes()).hexdigest() == digest, 'Capture source changed: ' + name)
    case = identity['case']
    require(os.environ.get('NEURON_COMPILED_ARTIFACTS') == case['artifact'], 'Capture artifact environment mismatch')
    from vllm_neuron.models.qwen35.identity import ARTIFACT_MANIFEST
    require(hashlib.sha256((Path(case['artifact']) / ARTIFACT_MANIFEST).read_bytes()).hexdigest()
            == identity['artifact_manifest_sha256'], 'Capture is not bound to the new package artifact')
    path = nvme_destination(os.environ[ENV_PREFIX + 'RECEIPT'])
    packets = nvme_destination(os.environ[ENV_PREFIX + 'ROOT'])
    require(path == packets.parent / 'worker-attachment.json' and packets.name == 'packets',
            'Exact fresh capture output binding required')
    value = {k: v for k, v in attachment.items() if k != 'observer'}
    value.update(pid=os.getpid(), worker_class=type(worker).__module__ + '.' + type(worker).__qualname__,
        host_identity_sha256=identity['sha256'], case=case,
        artifact_manifest_sha256=identity['artifact_manifest_sha256'], startup_only=True,
        full_model_qualified=False, timing_eligible=False)
    require(value['worker_class'] == identity['worker_class'], 'Actual capture worker differs')
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def worker_class(base_worker, arm, observer_factory=make_observer, on_attach=record_attachment):
    class PackageCaptureWorker(base_worker):
        def get_neuronx_distributed_model_runner(self, vllm_config, device):
            runner = super().get_neuronx_distributed_model_runner(vllm_config, device)
            self.qwen35_package_capture_attachment = attach(runner, arm, observer_factory)
            on_attach(self, runner, self.qwen35_package_capture_attachment)
            return runner
    return PackageCaptureWorker


def __getattr__(name):
    if name not in EXPORTS:
        raise AttributeError(name)
    from vllm_neuron.worker.neuron_worker import NeuronWorker
    cls = worker_class(NeuronWorker, EXPORTS[name])
    cls.__name__, cls.__qualname__, cls.__module__ = name, name, __name__
    globals()[name] = cls
    return cls
