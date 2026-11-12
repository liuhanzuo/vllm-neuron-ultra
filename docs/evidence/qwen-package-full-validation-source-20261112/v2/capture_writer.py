"""Original bounded capture writer; no native imports or files on import."""
import hashlib
import json
import os
from pathlib import Path
from qwen35_sampler_pair_protocol_v1 import MAX_CAPTURE_BYTES_PER_ARM, MAX_PACKET_BYTES, require


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
