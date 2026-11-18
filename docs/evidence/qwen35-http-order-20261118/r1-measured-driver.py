"""Short HTTP sampler diagnostic on the frozen r10 Qwen3.5 research adapter."""
import argparse
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

ROOT = Path("/home/ubuntu/experiments/trn1-latest-models-20261031")
sys.path.insert(0, str(ROOT))
from bench_http import run_round

WORK = Path("/mnt/nvme/neuron-work")
CONTRACT = WORK / "qwen35-sampler-pair-r10-v12-20261112/contract.json"
PROOF = WORK / "qw-r10-v12-20261112/candidate-http/startup-config-proof.json"
REPO = Path("/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107")
PORT = 18275
COMMITS = {"prior": "e96918402600ff68be17e663429405116c38df67",
           "block": "04f9cde24e321b5ed55142fd26f133440d686986"}


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def measure(model, n, greedy=False):
    prompts = [[1000 + (i + 7*j) % 97 for i in range(128)] for j in range(n)]
    options = {} if greedy else dict(temperature=.7, top_k=40, top_p=.9)
    result = run_round(PORT, model, prompts, 64, False, **options)
    for row in result["requests"]:
        assert row["usage"]["completion_tokens"] == 64 and row["usage"]["prompt_tokens"] == 128
    return result


def run_arm(output, arm, case, proof, graph):
    folder = output / arm
    folder.mkdir()
    sampler = folder / "sampler.py"
    sampler.write_bytes(subprocess.check_output(["git", "-C", REPO, "show", COMMITS[arm] + ":vllm_neuron/worker/cpu_sampler.py"]))
    environment = dict(proof["actual_api_initial_environment"])
    for key in ["QWEN35_SAMPLER_HOST_IDENTITY", "QWEN35_SAMPLER_HOST_RECEIPT"]:
        del environment[key]
    environment.update(NEURON_COMPILED_ARTIFACTS=str(graph), TMPDIR=str(folder), VLLM_RPC_BASE_PATH=str(folder),
                       HF_HUB_OFFLINE="1", SAMPLER_SHA256=sha(sampler), SAMPLER_SOURCE=str(sampler),
                       SAMPLER_RECEIPT=str(folder/"worker.json"))
    argv = list(proof["actual_api_argv"])
    argv[argv.index("--worker-cls")+1] = "q35_http_worker_nov18.Worker"
    argv[argv.index("--port")+1] = str(PORT)
    log = folder / "server.log"
    unit = f"trn1-{output.name}-{arm}"
    subprocess.run(["systemd-run", "--user", "--quiet", "--unit", unit, "--collect", "--service-type=exec",
                    "--property=MemoryMax=26G", "--property=MemoryHigh=25G", "--property=MemorySwapMax=4G",
                    "--property=RuntimeMaxSec=600", "--property=TimeoutStopSec=15", "-p", "StandardOutput=append:"+str(log),
                    "-p", "StandardError=inherit", "--working-directory="+str(ROOT), "/usr/bin/env", "-i",
                    *[k+"="+v for k,v in sorted(environment.items())], *argv], check=True)
    try:
        props = subprocess.check_output(["systemctl", "--user", "show", unit, "-p", "MainPID,InvocationID,ControlGroup,Id"], text=True)
        owner = dict(s.split("=", 1) for s in props.splitlines())
        pid = int(owner["MainPID"])
        owner["start_ticks"] = int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])
        write(folder/"owner.json", owner)
        for _ in range(150):
            if subprocess.run(["systemctl", "--user", "is-active", "--quiet", unit]).returncode:
                raise RuntimeError(f"Early server exit, see {log}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=2) as response:
                    if json.load(response)["data"][0]["id"] == case["served_name"]:
                        break
            except OSError:
                pass
            time.sleep(2)
        else:
            raise TimeoutError(str(log))
        assert "Successfully loaded pre-compiled model artifacts from "+str(graph) in log.read_text()
        assert read(folder/"worker.json")["sha256"] == sha(sampler)
        greedy = measure(case["served_name"], 4, greedy=True)
        write(folder/"greedy.json", greedy)
        metrics = {}
        for n in [1,4]:
            measure(case["served_name"], n)
            rounds = [measure(case["served_name"], n) for _ in range(3)]
            write(folder/f"rounds-{n}.json", rounds)
            metrics[str(n)] = [row["throughput_tok_s"] for row in rounds]
        return [r["text"] for r in greedy["requests"]], metrics
    finally:
        subprocess.run(["systemctl", "--user", "stop", unit], check=False)
        assert subprocess.run(["systemctl", "--user", "is-active", "--quiet", unit]).returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("attempt", type=int, choices=range(1,100))
    args = parser.parse_args()
    output = WORK / f"qs35-n18-r{args.attempt}"
    with (ROOT/"serial-enrollment.lock").open("r+") as lock:
        assert (os.fstat(lock.fileno()).st_dev, os.fstat(lock.fileno()).st_ino) == (66306, 8294421)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with socket.socket() as port_check:
            port_check.bind(("127.0.0.1", PORT))
        contract = read(CONTRACT)
        case = contract["case"]
        identity = contract["graph_reuse"]["identity"]
        art = Path(case["artifact"])
        assert read(art/"adapter-provenance.json") == identity["graph_provenance"]
        for name, pin in identity["graph_provenance"]["sources"].items():
            assert sha(Path(name)) == pin, name
        assert sha(art/"neuron_config.json") == identity["config_file_sha256"]
        for name, old in identity["large_file_identities"].items():
            path = Path(name)
            actual = path.stat()
            assert str(path.resolve()) == old["realpath"] and (actual.st_ino,actual.st_dev,actual.st_size,actual.st_mtime_ns,actual.st_ctime_ns) == (
                old["inode"],old["device"],old["size"],old["mtime_ns"],old["ctime_ns"]), name
        output.mkdir()
        graph = output/"graph"
        graph.mkdir()
        pins = {}
        for name in ("model.pt","neuron_config.json","adapter-provenance.json"):
            pins[name] = sha(art/name)
            shutil.copyfile(art/name, graph/name)
            assert sha(graph/name) == pins[name]
        try:
            results = {}
            for arm in COMMITS:
                print(f"Starting {arm}", flush=True)
                results[arm] = run_arm(output, arm, case, read(PROOF), graph)
                assert all(sha(graph/name) == pin for name,pin in pins.items())
            assert results["prior"][0] == results["block"][0]
            rounds = {arm:data[1] for arm,data in results.items()}
            med = {arm:{n:statistics.median(rows) for n,rows in data.items()} for arm,data in rounds.items()}
            summary = dict(model="Qwen3.5-2B text", commits=COMMITS, graph_sha256=pins, input_tokens=128, output_tokens=64,
                           rounds_tok_s=rounds, medians_tok_s=med, greedy_texts_matched=4, model_qualified_by_this_run=False,
                           ratios={n:med["block"][n]/med["prior"][n] for n in ["1","4"]})
            write(output/"summary.json",summary)
            print(json.dumps(summary),flush=True)
        finally:
            for name,pin in pins.items():
                if (graph/name).is_file() and sha(graph/name) == sha(art/name) == pin:
                    (graph/name).unlink()


if __name__ == "__main__":
    main()
