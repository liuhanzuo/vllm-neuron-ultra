"""Compare bounded CPU sampler versions on frozen device logits."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import time

import torch
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler

from vllm_neuron.worker import cpu_sampler


def measure(sampler, batches, k, p):
    start = time.perf_counter_ns()
    for logits in batches:
        sampler(logits, {}, k, p)
    return (time.perf_counter_ns() - start) / (len(batches) * 1000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("captures", nargs="+", type=Path)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("prior_sampler", args.previous)
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    reference = TopKTopPSampler("raw_logprobs")
    reference.forward = reference.forward_native
    samplers = [module.BoundedTopKTopPSampler(reference)
                for module in (previous, cpu_sampler)]
    assert len(os.sched_getaffinity(0)) == 1
    result = {"cpu": sorted(os.sched_getaffinity(0)), "threads": 1,
              "torch": str(torch.__version__), "top_k": 40, "top_p": 0.9,
              "temperature": 0.7, "rounds": 8, "rows": []}
    for path in args.captures:
        frames = torch.load(path, map_location="cpu", weights_only=True)
        logits = torch.stack([frame["logits"].flatten() for frame in frames]).float() / 0.7
        for batch in (1, 4):
            assert len(frames) % batch == 0
            batches = logits.split(batch)
            k = torch.full((batch,), 40)
            p = torch.full((batch,), 0.9)
            for sampler in samplers:
                measure(sampler, batches, k, p)
            times = [[], []]
            for rnd in range(result["rounds"]):
                for i in (rnd % 2, 1 - rnd % 2):
                    times[i].append(measure(samplers[i], batches, k, p))
            medians = [statistics.median(samples) for samples in times]
            result["rows"].append({
                "capture": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "frames": len(frames), "batch": batch, "width": logits.shape[1],
                "prior_us": times[0], "candidate_us": times[1],
                "prior_median_us": medians[0], "candidate_median_us": medians[1],
                "speedup": medians[0] / medians[1],
            })
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    with torch.inference_mode():
        main()
