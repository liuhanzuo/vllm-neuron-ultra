"""HTTP throughput, streaming latency and small correctness probes; keep raw runs."""
import argparse
import concurrent.futures
import json
import statistics
import threading
import time
import urllib.request
from pathlib import Path


def request(port, model, prompt, output_tokens=120, stream=False, force=True, **sampling):
    body = {"model": model, "prompt": prompt, "max_tokens": output_tokens, "temperature": 0.0, **sampling}
    if force:
        body.update(min_tokens=output_tokens, ignore_eos=True)
    if stream:
        body.update(stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    first = None
    pieces = []
    usage = {}
    events = []
    with urllib.request.urlopen(req, timeout=240) as response:
        if stream:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                obj = json.loads(line[5:])
                for choice in obj.get("choices", []):
                    content = choice.get("text", "")
                    if content:
                        stamp = time.perf_counter()
                        first = first if first is not None else stamp
                        events.append(stamp - start)
                        pieces.append(content)
                if obj.get("usage"):
                    usage = obj["usage"]
        else:
            obj = json.load(response)
            pieces = [obj["choices"][0]["text"]]
            usage = obj["usage"]
    end = time.perf_counter()
    generated = usage.get("completion_tokens")
    return {"text": "".join(pieces), "elapsed_s": end-start, "ttft_s": None if first is None else first-start,
            "mean_token_interval_s": ((end-first)/(generated-1)) if first is not None and generated and generated > 1 else None,
            "usage": usage, "stream_events_s": events}


def run_round(port, model, prompts, output_tokens, stream, **sampling):
    barrier = threading.Barrier(len(prompts))
    def job(prompt):
        barrier.wait()
        return request(port, model, prompt, output_tokens, stream, **sampling)
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(len(prompts)) as executor:
        rows = list(executor.map(job, prompts))
    elapsed = time.perf_counter()-start
    total = sum(r["usage"]["completion_tokens"] for r in rows)
    return {"elapsed_s": elapsed, "output_tokens": total, "throughput_tok_s": total/elapsed, "requests": rows}


def benchmark(port, model, prompts, batch, output_tokens=120, rounds=3, sampling=None):
    sampling=sampling or {}
    request(port, model, prompts[0], 8, **sampling)
    output = {"model": model, "batch": batch, "output_tokens_per_request": output_tokens, "sampling":sampling,"prompt_inputs":prompts,"rounds": []}
    for i in range(rounds):
        single = run_round(port, model, prompts[:1], output_tokens, False, **sampling)
        concurrent = run_round(port, model, (prompts*batch)[:batch], output_tokens, False, **sampling)
        output["rounds"].append({"single": single, "concurrent": concurrent})
        print(json.dumps({"round": i+1, "single": single["throughput_tok_s"], "concurrent": concurrent["throughput_tok_s"]}), flush=True)
    output["streaming"] = run_round(port, model, (prompts*batch)[:batch], output_tokens, True, **sampling)
    output["summary"] = {"single_best_tok_s": max(r["single"]["throughput_tok_s"] for r in output["rounds"]),
                         "single_median_tok_s": statistics.median(r["single"]["throughput_tok_s"] for r in output["rounds"]),
                         "concurrent_best_tok_s": max(r["concurrent"]["throughput_tok_s"] for r in output["rounds"]),
                         "concurrent_median_tok_s": statistics.median(r["concurrent"]["throughput_tok_s"] for r in output["rounds"]),
                         "streaming_ttft_ms": [1000*r["ttft_s"] if r["ttft_s"] is not None else None for r in output["streaming"]["requests"]],
                         "streaming_mean_token_interval_ms": [1000*r["mean_token_interval_s"] if r["mean_token_interval_s"] is not None else None for r in output["streaming"]["requests"]]}
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--model", default="Qwen3-8B")
    ap.add_argument("--model-path", default="/mnt/nvme/models/Qwen3-8B")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    prompt = tokenizer.apply_chat_template([{"role":"user", "content":"请详细介绍一下法国的首都。"}], tokenize=True, add_generation_prompt=True, enable_thinking=False)
    out = benchmark(args.port,args.model,[prompt],args.batch)
    out["prompt_ids"] = prompt
    out["prompt_tokens"] = len(prompt)
    Path(args.output).write_text(json.dumps(out,ensure_ascii=False,indent=2))
    print(json.dumps(out["summary"],ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
