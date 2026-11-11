"""Stable importable workers for vLLM --worker-cls, including spawn children.

The module itself is stdlib only. Resolving one of the four exported worker
names lazily imports NeuronWorker in the future admitted native process. The
class is cached under a stable module/qualname, not sent as a local closure.
The unchanged server_with_qwen35_split_v4.py remains the server entrypoint.
"""
import hashlib
import json
import os
from pathlib import Path

from qwen35_sampler_instance_v1 import worker_class
from qwen35_sampler_pair_protocol_v1 import MAX_CAPTURE_BYTES_PER_ARM, MAX_PACKET_BYTES, host_identity, require

EXPORTS = {
    "BaselineMeasuredWorker": ("baseline", "measured"),
    "CandidateMeasuredWorker": ("candidate", "measured"),
    "BaselineCaptureWorker": ("baseline", "correctness_capture"),
    "CandidateCaptureWorker": ("candidate", "correctness_capture"),
}


def worker_name(arm, phase):
    names = [name for name, pair in EXPORTS.items() if pair == (arm, phase)]
    require(len(names) == 1, "unknown worker arm/phase")
    return __name__ + "." + names[0]


def nvme_destination(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path, "absolute nonsymlink output required")
    require(Path("/mnt/nvme/neuron-work") in path.parents, "sampler outputs must use NVMe neuron-work")
    return path


def atomic_metadata(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class CaptureFileWriter:
    """Only a supplied fresh owned directory; never picks or admits a run."""
    def __init__(self, directory, arm):
        self.directory = nvme_destination(directory)
        require(self.directory.is_dir() and not list(self.directory.iterdir()), "capture directory must already exist and be fresh")
        self.arm, self.rows, self.total = arm, [], 0
        with (self.directory / "writer-owner.json").open("x") as stream:
            json.dump({"pid": os.getpid(), "arm": arm, "phase": "correctness_capture"}, stream)

    def __call__(self, packet):
        from qwen35_sampler_native_capture_v1 import parse_packet
        require(type(packet) is bytes and len(packet) <= MAX_PACKET_BYTES and len(self.rows) < 64, "capture writer cap exceeded")
        require(self.total + len(packet) <= MAX_CAPTURE_BYTES_PER_ARM, "capture writer total cap exceeded")
        header, _ = parse_packet(packet)
        require(header["arm"] == self.arm, "capture writer arm changed")
        shape = header["processed_logits"]["shape"]
        require(len(shape) == 2 and shape[0] in range(1, 5) and shape[1] == 248320, "capture geometry changed")
        require(sum(row["rows"] == shape[0] for row in self.rows) < 16, "row-count capture cap exceeded")
        name = "capture-%04d.bin" % len(self.rows)
        with (self.directory / name).open("xb") as stream:
            written = stream.write(packet)
            require(written == len(packet), "short packet write")
            stream.flush()
            os.fsync(stream.fileno())
        digest = hashlib.sha256(packet).hexdigest()
        self.total += len(packet)
        self.rows.append({"file": name, "rows": shape[0], "bytes": len(packet), "sha256": digest,
                          "call_id": header["call_id"]})
        counts = {str(n): sum(row["rows"] == n for row in self.rows) for n in range(1, 5)}
        captured_rows = sum(row["rows"] for row in self.rows)
        atomic_metadata(self.directory / "capture-receipt.json", {
            "arm": self.arm, "phase": "correctness_capture", "packets": len(self.rows),
            "captured_calls_by_rows": counts, "captured_rows": captured_rows,
            "processed_logit_bytes": captured_rows * 248320 * 4, "total_packet_bytes": self.total,
            "files": self.rows, "timing_eligible": False, "full_model_qualified": False})
        return {"bytes": len(packet), "sha256": digest}


def make_observer(runner, sampler, arm):
    from qwen35_sampler_native_capture_v1 import CaptureObserver, NativeSnapshotter
    path = os.environ["QWEN35_SAMPLER_CAPTURE_ROOT"]
    return CaptureObserver(runner, sampler, arm, CaptureFileWriter(path, arm), NativeSnapshotter())


def record_attachment(worker, runner, attachment):
    """Startup-only evidence from the actual worker; same write on both arms."""
    path = nvme_destination(os.environ["QWEN35_SAMPLER_HOST_RECEIPT"])
    identity = json.loads(os.environ["QWEN35_SAMPLER_HOST_IDENTITY"])
    require(identity == host_identity(attachment["arm"], attachment["phase"], identity["source_sha256"]),
            "host identity does not match actual worker")
    for name, expected in identity["source_sha256"].items():
        source = Path(name)
        require(source.is_absolute() and source.stat().st_size < 4 * 1024**2, "invalid host source pin")
        require(hashlib.sha256(source.read_bytes()).hexdigest() == expected, "host source changed: " + name)
    operation = runner.cpu_sampler.topk_topp_sampler
    reference = getattr(operation, "reference", operation)
    value = {key: value for key, value in attachment.items() if key != "observer"}
    value.update(pid=os.getpid(), worker_class=type(worker).__module__ + "." + type(worker).__qualname__,
                 operation_class=type(operation).__module__ + "." + type(operation).__qualname__,
                 reference_class=type(reference).__module__ + "." + type(reference).__qualname__,
                 host_identity_sha256=identity["sha256"], startup_only=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def __getattr__(name):
    if name not in EXPORTS:
        raise AttributeError(name)
    arm, phase = EXPORTS[name]
    from vllm_neuron.worker.neuron_worker import NeuronWorker
    cls = worker_class(NeuronWorker, arm, phase,
                       observer_factory=make_observer if phase == "correctness_capture" else None,
                       on_attach=record_attachment)
    cls.__name__, cls.__qualname__, cls.__module__ = name, name, __name__
    globals()[name] = cls
    return cls
