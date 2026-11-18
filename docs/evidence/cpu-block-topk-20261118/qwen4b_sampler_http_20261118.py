"""Paired HTTP diagnostic on the existing Trn1 BF16 TP2 graph."""
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import statistics
import subprocess
import sys
import time
import urllib.request

R = Path(__file__).resolve().parents[1]
REPO = Path("/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107")
MODEL = Path("/mnt/nvme/models/Qwen3-4B-Instruct-2507")
ART = Path("/mnt/nvme/neuron-artifacts/latest-20261031-qwen3-4b-2507-bf16-cpu-tp2-b4-ctx4096")
PYTHON = "/home/ubuntu/venvs/vllm_neuron/bin/python"
PORT = 18272
BASE = f"http://127.0.0.1:{PORT}"
PROMPTS = [[151643] + [1000 + ((i + 7 * j) % 97) for i in range(127)] for j in range(4)]
VERSIONS = [("prior", "e96918402600ff68be17e663429405116c38df67"),
            ("block", "04f9cde24e321b5ed55142fd26f133440d686986")]


def http(path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=150) as res:
        return json.load(res)


def batch(n, greedy):
    def request(ids):
        params = dict(model="qwen4b-diag", prompt=ids, max_tokens=64,
                      temperature=0 if greedy else .7, ignore_eos=True)
        if not greedy:
            params.update(top_k=40, top_p=.9)
        result = http("/v1/completions", params)
        assert result["usage"]["completion_tokens"] == 64 and result["usage"]["prompt_tokens"] == 128
        return result["choices"][0]["text"]
    begin = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
        texts = list(pool.map(request, PROMPTS[:n]))
    elapsed = time.monotonic() - begin
    return dict(texts=texts, seconds=elapsed, output_tokens=n * 64, tokens_s=n * 64 / elapsed)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_arm(path, name, commit, graph):
    code = path / name
    code.mkdir()
    with subprocess.Popen(["git", "-C", REPO, "archive", commit, "vllm_neuron"],
                          stdout=subprocess.PIPE) as archive:
        subprocess.run(["tar", "-xf", "-", "-C", code], stdin=archive.stdout, check=True)
        assert archive.wait() == 0
    log = path / f"{name}.log"
    unit = f"trn1-q4-sampler-{name}-{path.name}"
    env = dict(PYTHONPATH=str(code), NEURON_CPU_SAMPLER_FAST="1", HF_HUB_OFFLINE="1",
               NEURON_COMPILED_ARTIFACTS=str(graph), NEURON_RT_VISIBLE_CORES="0-1",
               BASE_COMPILE_WORK_DIR=str(path / f"{name}-compiler"), HF_HOME="/mnt/nvme/hf",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
               PATH="/home/ubuntu/venvs/vllm_neuron/bin:/opt/aws/neuron/bin:/usr/local/bin:/usr/bin:/bin")
    # Imported by the parent and the spawned worker. Abort a cache miss before
    # it can replace an artifact or run a compiler during this comparison.
    entry = R / "research/qwen4b_sampler_load_only_20261118.py"
    args = ["systemd-run", "--user", "--quiet", "--collect", "--unit", unit,
            f"--working-directory={code}", "-p", "MemoryMax=26G", "-p", "MemoryHigh=25G",
            "-p", "MemorySwapMax=4G", "-p", "RuntimeMaxSec=600", "-p", "TimeoutStopSec=15s",
            "-p", f"StandardOutput=append:{log}", "-p", "StandardError=inherit",
            *[f"--setenv={k}={v}" for k,v in env.items()], PYTHON, "-B", str(entry),
            "--model", str(MODEL), "--served-model-name", "qwen4b-diag", "--tensor-parallel-size", "2",
            "--max-model-len", "4096", "--max-num-seqs", "4", "--no-enable-prefix-caching",
            "--host", "127.0.0.1", "--port", str(PORT), "--additional-config",
            '{"override_neuron_config":{"on_device_sampling_config":null,"apply_seq_ids_mask":true}}']
    subprocess.run(args, check=True)
    try:
        deadline = time.monotonic() + 250
        while time.monotonic() < deadline:
            if subprocess.run(["systemctl", "--user", "is-active", "--quiet", unit]).returncode:
                raise RuntimeError(f"{name} exited before readiness: {log}")
            try:
                if http("/v1/models")["data"][0]["id"] == "qwen4b-diag":
                    break
            except OSError:
                time.sleep(2)
        else:
            raise TimeoutError("readiness")
        content = log.read_text(errors="replace")
        assert "Successfully loaded pre-compiled model artifacts" in content
        assert str(code / "vllm_neuron/worker/cpu_sampler.py") in content
        greedy = batch(4, True)
        metrics = {}
        for n in (1, 4):
            batch(n, False)
            metrics[str(n)] = [batch(n, False) for _ in range(3)]
        return dict(commit=commit, sampler_sha256=sha(code / "vllm_neuron/worker/cpu_sampler.py"),
                    greedy=greedy, metrics=metrics)
    finally:
        subprocess.run(["systemctl", "--user", "stop", unit], stdout=subprocess.DEVNULL)
        assert subprocess.run(["systemctl", "--user", "is-active", "--quiet", unit]).returncode != 0


def main():
    output = R / sys.argv[1]
    output.mkdir()
    with (R / "serial-enrollment.lock").open("r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not socket.socket().connect_ex(("127.0.0.1", PORT)) == 0
        graph = output / "graph"
        graph.mkdir()
        pins = {}
        for fname in ("model.pt","neuron_config.json"):
            pins[fname] = sha(ART / fname)
            shutil.copyfile(ART / fname, graph / fname)
            assert sha(graph / fname) == pins[fname]
        arms = {}
        try:
            for name, commit in VERSIONS:
                print(f"starting {name} {commit}", flush=True)
                arms[name] = run_arm(output, name, commit, graph)
                (output / f"{name}.json").write_text(json.dumps(arms[name], indent=2))
                assert all(sha(graph / k) == v for k,v in pins.items())
            assert arms["prior"]["greedy"]["texts"] == arms["block"]["greedy"]["texts"]
            speeds = {name:{n:statistics.median(s["tokens_s"] for s in rows)
                            for n,rows in arm["metrics"].items()} for name,arm in arms.items()}
            summary = dict(greedy_4_exact=True, graph_sha256=pins, random_medians=speeds,
                           ratios={n:speeds["block"][n]/speeds["prior"][n] for n in ("1","4")},
                           baseline_full_numerical_passed=False, model_qualified=False)
            (output/"summary.json").write_text(json.dumps(summary, indent=2))
            print(json.dumps(summary),flush=True)
        finally:
            # Drop only exact temporary copies. Keep both full source versions,
            # request results, logs and any unexpected files for inspection.
            for k, v in pins.items():
                if (graph / k).is_file() and sha(graph / k) == sha(ART / k) == v:
                    (graph / k).unlink()


if __name__ == "__main__":
    main()
