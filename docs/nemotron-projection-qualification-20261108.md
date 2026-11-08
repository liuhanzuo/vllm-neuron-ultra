# Nemotron projection split: eight device cases — 2026-11-08

The `tensor_split` candidate passed every completed projection case in the
original package qualification: six prefill lengths and two decode shapes.
All 16 original/candidate workers have completed with retained original exit 0.
The original path fails for multiple logical rows but passes for one row in
these tests. This narrows the defect to the tested shape-dependent split
consumer and compiler path; it is not a general claim about all split operations.

Input shape is `[batch, length, 11328]`, with split sizes `[5120, 6144, 64]`.
All CPU checks and compilations passed. The table reports compiled graph address
checks alongside **complete device-output byte comparisons** for two patterns,
three outputs per pattern. A passing original negative-control worker means
that the expected mismatch was reproduced; it does not make that output correct.

| Batch | Tokens | Original wrong/sample addresses | Candidate wrong/sample addresses | Full original / candidate output |
|---:|---:|---:|---:|---|
| 1 | 128 | 1,143/1,152 | 0/1,152 | Different / Exact |
| 1 | 256 | 2,295/2,304 | 0/2,304 | Different / Exact |
| 1 | 512 | 4,599/4,608 | 0/4,608 | Different / Exact |
| 1 | 1024 | 9,207/9,216 | 0/9,216 | Different / Exact |
| 1 | 2048 | 18,423/18,432 | 0/18,432 | Different / Exact |
| 1 | 4096 | 36,855/36,864 | 0/36,864 | Different / Exact |
| 4 | 1 | 27/36 | 0/36 | Different / Exact |
| 1 | 1 | 0/9 | 0/9 | Exact / Exact |

At 2026-11-08T02:31:44.014314+00:00, the run had advanced to
`split-nemotron-conv-b1-t128-tensor-split`. All nine earlier alias stages also passed
70 device checks with normal retained original exits. The serial run keeps
package `60ab7f5` and all 52 planned stages; it does not substitute the unified
implementation.

Convolution split tests, whole-model compilation, state copying, numeric checks,
cold cache, long decoding, quality, slot reuse, paired workloads and raw result
verification remain mandatory. **There is no new model speedup ratio.** An
incorrect original output is not an eligible performance baseline.

The [retained evidence](evidence/nemotron-projection-20261108/progress.json)
contains complete comparison hashes, sampled-address diagnostics and references
to the original primitive report, worker report and retained exit. Its
[manifest](evidence/nemotron-projection-20261108/manifest.json) binds the published
snapshot. The [storage race report](nemotron-storage-race-20261108.md) explains the
prior failed run; historical performance remains in [the model table](multi-model.md).
