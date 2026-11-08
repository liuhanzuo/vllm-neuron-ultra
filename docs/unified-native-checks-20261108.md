# Unified package native checks — 2026-11-08

The exact wheel built from implementation `e0a01d3bd9e1fec9faf2b6b0f37ecbe2d8bf92dc`
passed all **9 serial installation, plugin, subprocess and configuration phases**
on this Trn1 machine after correcting a configuration probe bug. The model
package was unchanged. This checks real Python/Neuron/vLLM integration; it does
not qualify device inference, HTTP serving or a new speedup.

| Phase | Result | Context / variant |
|---|---|---|
| Fresh wheel installation | Passed | Original wheel, isolated target, no dependency/global installation |
| All model plugins disabled | Passed | Parent and actual spawn child |
| LFM3 reference | Passed | 2048 / reference |
| LFM3 cache writer | Passed | 2048 / nki_predicated_decode_v4 |
| SmolLM3 reference | Passed | 2048 / reference |
| SmolLM3 read | Passed | 2048 / nki_read |
| SmolLM3 write | Passed | 2048 / nki_write |
| SmolLM3 combined | Passed | 2048 / nki_read_write |
| Nemotron Nano 9B v2 | Passed | 4096 / split_nki_v5 |

Each non-install phase used a fresh interpreter and an actual `spawn` child.
For the seven model-selected modes, both processes constructed the real SDK
configuration, preserved the complete serialized body, round-tripped the saved
root and its embedded metadata, rejected all five invalid scope mutations and
all seven invalid saved-body mutations. Across both processes this supplies 14
configuration executions, 70 scope rejections and 98 saved-body rejections.
Plugin discovery, registration, import origins and variant selection were checked.
Smol attention is selected during model construction, so this preflight does not
prove that its attention/kernel path actually executed.

The outer process and all nine original services exited successfully. Eight
spawn children exited successfully; all observed processes and service groups
were drained. The preceding failed run remains recorded.

## Probe bug found and corrected

The first run passed the first eight phases, then failed Nemotron before its
spawn child. The probe passed `vars(config)` / `vars(config.neuron_config)` to
Nemotron's strict configuration admission. Those dictionaries contain a live
`torch.dtype`, while the production artifact loader passes the complete JSON
body returned by `config.to_json_string()`. Strict admission correctly rejected
that non-JSON object at `('neuron_config', 'torch_dtype')`.

Both positive and negative probe calls now consume the complete SDK-serialized
JSON body, as the production loader does. No field was removed and no package
validation was weakened. Eleven focused source tests, an independent review,
and the complete fresh nine-phase rerun passed. The negative probe was fixed
too, so an unrelated dtype rejection cannot masquerade as rejection of a bad
parallelism or context setting.

## Inherited platform warnings and limits

The logs show an early `vllm.config` circular import and a missing
`vllm_neuron.patches` module. The affected platform files are byte-identical to
the imported release base `ad82965376f1eab5dcfecb6b1bae99ae8e262a8d`;
the patch package is absent in both that base and this branch.

The early configuration override has a later retry path, but this preflight
does not assert its final flag or method identity. The missing module skips
the vLLM 0.11.0 Harmony chat-stream override and leaves the original generator.
There is a potential multi-token streaming-delta concern on that path; no
served-model text loss was observed or tested here. These findings do not
establish a failure in the three model adapters. The checks also do not assert
OpenAI request validation or HTTP streaming behavior.

Before promoting the merged branch, still run fresh compile/load and artifact
rejection, numerical comparisons, cold cache, long decode, request-slot reuse,
quality and paired serving measurements. Existing [speedup tables](multi-model.md)
retain their original implementation, reference and workload boundaries.

## Evidence

[Summary and actual exit records](evidence/unified-native-20261108/summary.json)
include the exact wheel hash, per-phase outcomes and the full local root-audit
reference. The [manifest](evidence/unified-native-20261108/manifest.json) pins
raw phase results, original service exits, spawn acknowledgments, the failed
Nemotron log, both probe sources and independent source checks. No weights,
model tensors or compiled binaries are included.
