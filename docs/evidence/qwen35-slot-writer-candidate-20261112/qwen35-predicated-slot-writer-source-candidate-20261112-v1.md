# One next candidate: Qwen3.5 decode slot-write movement

Reuse `store_slots_nki_predicated` for Qwen3.5's compiled B4 decode writes. It resolves the last writer for each destination in integer control, then emits each output tile once. Keep the original one-row prefill path, all cache reads/scatters, attention/GDN arithmetic, BF16 activations and FP32 recurrent state. No new kernel is needed for this candidate. This is a source-derived optimization hypothesis, not a measured Qwen bottleneck or speedup.

## Exact source target and suitability

- `qwen35_text_port.py:47` implements `store_slots`: for each of four update rows, line52 selects into a complete capacity-four cache. LinearAttention calls it for conv/state at line102; FullAttention calls it for K/V at line149.
- `slot_cache_nki_predicated_torch.py:6` already flattens each slot to width and restores the original shape. `slot_cache_nki_predicated.py:20` walks destinations; lines22-25 select the LAST matching update row. Lines33-36 load original/update tiles, apply `tensor_copy_predicated`, and store the full output once. A destination without a matching row keeps its original payload. No floating payload arithmetic or BF16/FP32 cast is introduced.
- The kernel supports1..4 rows/capacity and widths divisible by128. Qwen's exact flattened widths are conv12288 (96 groups, one masked128-group tile), recurrent131072 (1024 groups), and KV256×L (256..8192 groups for L128..4096). Its dimension-agnostic flattened copy supports head_dim256 without changing attention geometry.
- `qwen35_neuron_adapter.py:83` retains conv BF16 and recurrent FP32; line89 defines maximum KV allocation. Lines186-191 slice KV to the decode bucket and concatenate the untouched tail. Keep these aliases/layouts/tail concatenations unchanged in this candidate.
- Preserve duplicate slots and inactive-1 exactly: sequential original writes imply last-row-wins, not summation and not first-row-wins. The existing kernel's integer selection implements that rule. Keep the original validated slot domain and int32 conversion contract; do not silently widen it.

The current adapter/text/primitives/split/reset source hashes all match the qualified reset baseline's frozen plan. Do not start from any failed GDN variant.

## Exact state sizes and logical byte opportunity

Per TP rank there are18 conv buffers `[4,3072,4]` BF16 (96KiB each),18 recurrent buffers `[4,8,128,128]` FP32 (2MiB each), and12 KV buffers `[4,1,4096,256]` BF16 (8MiB each). The full persistent pool is133.6875MiB/rank,267.375MiB across TP2. The actual writer receives only the current KV prefix; conv and recurrent sizes are fixed.

| Decode bucket L | One complete writer result per rank | Four original select results across TP2 | One NKI result across TP2 | Logical result-volume reduction |
|---|---:|---:|---:|---:|
|128|40.6875MiB|325.5MiB|81.375MiB|244.125MiB|
|1024|61.6875MiB|493.5MiB|123.375MiB|370.125MiB|
|4096|133.6875MiB|1069.5MiB|267.375MiB|802.125MiB|

Formula per rank: `S(L)=18×4×3072×4×2 + 18×4×8×128×128×4 + 12×4×L×256×2` bytes. Across TP2, four functional selects represent8S result bytes versus2S for the proposed writer, a6S logical difference.

These are source-level tensor-result volumes, NOT measured HBM traffic. The compiler may fuse selects or alias buffers already. The NKI kernel still loads both old and replacement tiles and stores the ENTIRE output, including inactive slots; prefix/tail concatenation and gathers remain. Therefore this candidate collapses repeated full-capacity select materializations, but is neither zero-copy nor a token-only in-place write. No time or throughput ratio can be inferred from these byte counts.

## What existing measurements do and do not establish

The actual LFM3 isolated multi-row writer comparison passed its original gates and measured short-decode2.058× C1/1.813× C4 with the same sampler on both sides (`lfm3-nki-predicated-slot-full/paired-comparison.json`). That validates this movement approach on LFM3, not a Qwen speedup. Its later qualified-device profile reports core0 bucket2048 total20.726ms and DMA-active11.814ms; overlapping activities and the independent completion audit explicitly prevent attribution to a specific cache kernel.

SmolLM3 uses the SAME predicated writer at `smollm3_nki_kv_cache.py:13` and separately replaces gathers with an NKI reader at line8. Its measured5.58×–12.51× workload ratios also include replacing a min2048 bucket workaround with natural buckets, so they cannot estimate Qwen's isolated writer benefit. Do not add Smol's read replacement to this first Qwen candidate.

Qwen's qualified reset baseline measured10.365tok/s C1 and39.578tok/s C4 short decode, with32 numerical,84 cold,1541-step/87-frame long,96 quality and five×three HTTP gates complete. No Qwen cache-operator timing was found in the bounded source/small-metadata inspection; r10's unfinished sampler run is not evidence for this writer.

Existing primitive evidence covers9 real-shape graphs/43 byte checks, including FP32 `[4,64,80,128]` SSM and signed-zero/Inf/NaN inputs, plus12 persistent-alias graphs/60 checks. The latter exercise BF16 KV, not Qwen's exact FP32 recurrent alias. Exact Qwen shapes and persistent FP32 state remain unqualified.

## Small next implementation and validation

First add only a later, separately scheduled primitive comparison at Qwen's full conv, FP32 recurrent and all six KV-prefix shapes: original sequential `store_slots` versus the UNCHANGED existing predicated NKI writer. Use exact byte oracles, all permutations/partial/all-inactive mappings, duplicate writers, dirty destination buffers and consecutive calls. Add actual MUST_ALIAS state checks for both BF16 and FP32, with attention/state reads in the same graph, so outputs and persistent inputs both match the sequential snapshot semantics. Include signed zeros, infinities and NaN payloads; report arithmetic results separately from movement bytes.

After correctness, measure paired warmed primitive calls at those exact shapes/slots with input creation and CPU comparisons outside the timer. Record both compiled paths, actual completion, elapsed distributions and output/state bytes. This determines whether the compiler already collapses the original selects; retain slower or neutral outcomes. No compiler, benchmark or payload read occurred in this review.

Only if that comparison supports it, add a small Qwen-specific decode dispatcher using the existing wrapper. Current prefill is one-row and stays original; assert the frozen trace shape/domain, or use an explicit prefill branch rather than assuming every future B1 call is prefill. Critical install detail: `frontend_split_correction_v2.clone_forward` line60 COPIES function globals. Installing `port.store_slots` AFTER the current split correction would miss cloned LinearAttention. Bind the dispatcher before that original split clone (parent AND spawned engine), or pass it explicitly in the clone's reviewed overrides; verify both exact runtime providers. FullAttention must bind the same dispatcher. Keep the original tensor_split correction and all arithmetic bodies unchanged.

Full-model acceptance still requires the original32 numerical frames,84 cold frames/672 zero checks/12 routes,1541 long steps/87 frames including slot reuse/inactive resume,96 quality cases and five workloads×three rounds. Use the qualified split/reset baseline with the SAME sampler on both sides; any successful sampler variant is a separate controlled dimension. There is no reuse of failed GDN qualification and no proposed change to weights, BF16/FP32 policy, context, capacity, slots or thresholds.
