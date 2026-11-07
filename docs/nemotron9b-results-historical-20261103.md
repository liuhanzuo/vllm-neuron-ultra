# Nemotron Nano 9B v2 — historical experiment, 2026-11-03

**These are absolute rates from the earlier experiment. They are not new
package results or an optimization speedup.** The original cache baseline
failed qualification, so there is no qualified denominator. The package on
`nemotron9b-experimental-v1` still requires all [fresh runtime gates](nemotron9b.md).

The frozen experiment used original BF16 weights at revision
`6533e8de2c68e4536bf7c411d7a3ce5734111476`, FP32 Mamba SSM state, TP2,
physical decode batch 4, prefill batch 1 and context 4096. It used the
split-corrected NKI cache path and standard CPU sampler, preserving the
original thinking template and the disclosed native CPU timestep limit
`[0, +Infinity]`. CUDA numerical equivalence was not established.

| Historical workload | Single median token/s | Four-request aggregate median token/s |
|---|---:|---:|
| Short decode | 13.1823 | 51.0603 |
| Medium prompt | 12.7829 | 45.6749 |
| Long prompt | 11.9357 | 38.5035 |
| Short stochastic | 11.4097 | 32.1444 |

The mixed-length workload measured 26.8639 aggregate token/s. These were
five workloads with three rounds; the terminal audit recomputed rates and
budgets for 124 HTTP requests / 30,848 completion tokens. Streaming records
and full-precision values are retained in the [compact evidence JSON](nemotron9b-historical-evidence-20261103.json).

The historical configuration passed 32 numerical frames, cold-prefix
84 frames / 868 raw-zero checks, and 2,041 long-decode steps / 93 checkpoints,
covering request-slot reuse and all 12 routes. Its second workflow reused
the earlier two-pattern 248 fill / 248 zero state-copy proof after a clock
failure; it did **not** execute a fresh copy proof. Package qualification
must execute that proof again.

Quality was **55/96 with 41 output-limit truncations**, not a broad model
quality certification. GSM8K was 55/64 with nine capped outputs; all 32 short
arithmetic/JSON/retrieval/list cases reached their output limits. The original
template ignores `enable_thinking=False`; no `/no_think` substitution was
made. Prompts, budgets and scores must remain matched in any new comparison.

The historical editable serving repository was at
`ad82965376f1eab5dcfecb6b1bae99ae8e262a8d`. The package base is
`5e5ac4c9c9abb74bbf6c1cc3afa3e2d1b487f994`, which also removes costly
DEBUG logits-statistics work even with the fast sampler off. That runner
difference prevents inheriting historical speed claims. Original failures
remain evidence; the model family is not marked fully optimized.

The evidence JSON records source paths and SHA256 values for the producer
summary, qualified record, terminal audit, exact plan, tokenizer identities,
primitive proof and packaging inventory. During packaging only source and
JSON metadata were read; weights, native libraries, tensor references, HLO,
NEFF and compiled model payloads were not read or hashed.
