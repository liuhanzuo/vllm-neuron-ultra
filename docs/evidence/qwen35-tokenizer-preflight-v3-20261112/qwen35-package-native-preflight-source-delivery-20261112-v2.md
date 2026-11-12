Qwen3.5 packaged native preflight v2 source delivery, 2026-11-12

This immutable successor adds complete known-input tokenizer comparisons to the packaged native preflight. It is source delivery only. No native tokenizer, model, compiler or device execution occurred, and no original tokenizer/model payload was read during preparation or tests. No runtime input, plan, admission/review receipt or publication was created. The active Qwen run, original serial owner, locks, history, cgroups, services, catalog and frozen package checkout were not changed. V1 remains intact.

All new files are in `/home/ubuntu/experiments/trn1-latest-models-20261031/research/`:

- `qwen35_package_native_preflight_v2.py`: worker; native work still requires explicit invocation and a root-supplied binding.
- `qwen35_package_tokenizer_checks_v2.py`: lazy native tokenizer helper; exact original input functions and bounded source projections.
- `prepare_qwen35_package_native_preflight_v2.py`: source/metadata-only bundle preparation, exclusive-create destination.
- `check_qwen35_package_native_preflight_v2.py`: 28 passing stdlib/source/mock tests, including the original 18 tests.
- `qwen35-package-native-preflight-source-bundle-20261112-v2.json`: 19,612 bytes, SHA256 `1337e32ad65a8aae802a487a155c51d2738e41da88cdcd23adea8d6f7eba203a`.

The new bundle pins these four sources, 70 checkout Python sources, 20 SDK sources, the frozen package-v2 delivery, and 11 original protocol/source/small JSON references. Package source commit remains `e40fd753ba080a9a4640688f4d6beda9985b7099`; package delivery SHA256 remains `e73b5d59d28be857f16cc9349d042f3e0687956bdf61f569c6c1e3ea4a3ff835`. Preparation extracts tokenizer file descriptors from the pinned package provenance without opening those payloads.

The existing native boundaries and obligations from `qwen35-package-native-preflight-source-delivery-20261112-v1.md` remain: the original Python3.10 interpreter and checkout, actual parent plus multiprocessing spawn, native dependency/source/origin checks, repeated registration, tiny BF16 CPU split, whole SDK config roundtrip and package scope/admission negatives, per-instance standard/fast sampler checks, root-owned current complete history and actual original lock/cgroup binding, and root-owned final descendant drain/reconciliation. No weights, model constructor, compile, accelerator graph or generated outputs are part of this preflight. Cgroup ceilings remain 12 GiB, zero swap, 256 tasks and at most 8 CPUs; worker ceilings remain 600 s wall, 480 s CPU and 16 MiB per output file. Native runtime results are pending.

Future tokenizer payload scope is explicit: the helper verifies only `config.json`, `tokenizer_config.json`, `tokenizer.json`, `vocab.json`, `merges.txt`, and `chat_template.jinja` at `/mnt/nvme/models/Qwen3.5-2B`, using the original pinned Git-blob/SHA256 descriptors and bounded streamed reads. Those files total about 22.9 MB. It verifies the historical extracted text-view config against the package-generated config and confirms all five historical tokenizer symlinks still target their exact original files. It creates equivalent symlinks only in the fresh worker's shared config directory. Original and historical tokenizer files are read-only. The Python runtime audit guard restricts opens under both original model directories to that exact tokenizer/config inventory and retains the model/archive/device/network/subprocess rejections. Native library internals are not claimed to be sandboxed by Python audit hooks; the enclosing bounded root service remains responsible for isolation and cleanup.

Both the actual parent and actual spawn child will load the unchanged original tokenizer and package text-view tokenizer through `AutoTokenizer.from_pretrained(..., local_files_only=True)`. Neither a Mistral regex fix nor warning suppression is passed. The generic installed Transformers warning is preserved. Equality must establish behavior; the warning text does not establish a tokenizer failure or justify rewriting tokenizer bytes/config.

Known-input coverage, compared as complete integer arrays rather than lengths:

| Input protocol | Current original versus packaged view | Saved reference also required |
| --- | --- | --- |
| All 96 unique pinned quality-suite cases | Every complete input ID array, ordered case IDs and exact chat messages/kwargs | All 96 historical actual `quality96.json` request arrays |
| Actual HTTP workload inputs 32, 128, 512 and 1536 | Original complete workload construction and exact trimmed arrays, including the preserved trailing 16 tokens | All four historical `workloads.json` arrays |
| Initial HTTP prompt | Exact `NativeClients.initial` prompt expression and full IDs | No saved original HF initial reference is pinned |
| Diagnostic HTTP prompt | Exact `NativeClients.diagnostic` expression, equal to workload 128 | Historical workload 128 IDs via the workload check |
| Four unique original smoke prompts | Original literal checklist through original `prompt_ids` | First smoke prompt also matches one of the saved six HF fixtures |
| Six original HF fixtures | All six texts through original `prompt_ids` and all returned IDs | All six saved original HF `input_ids` arrays |
| Two original long sequences | Original pre-tensor tokenizer statements, complete 2060-ID arrays | Both saved original long HF arrays |

The quality, workload and `prompt_ids` checks execute the complete unchanged FunctionDef bodies extracted from their pinned source files. Their tokenizer dependency is injected; quality receives its original module-root namespace. The whole modules are not imported because they contain HTTP/native dependencies. Initial/diagnostic prompt assignments, smoke checklist and long pre-tensor statements are explicitly source projections. They do not constitute complete execution of the HTTP clients or long HF oracle. The long projection stops before the first tensor/model operation and rejects model/torch names in the executed statement slice. Full HTTP workloads reuse the four known workload inputs; no new benchmark prompts were invented.

Every original chat call requires exactly `tokenize=True`, `add_generation_prompt=True`, `enable_thinking=False`. Long body encoding requires `add_special_tokens=False`. The two current tokenizers must also produce equal recorded messages, kwargs and complete template results. Historical quality/workload arrays are served-view evidence, not an independent saved original HF tokenizer reference. The receipt states this limitation and identifies initial/other-smoke reference gaps. Current independent original-versus-view full-ID equality remains mandatory across all listed inputs. The parent and child must agree on their complete tokenizer evidence.

Validation performed:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 research/check_qwen35_package_native_preflight_v2.py
PYTHONDONTWRITEBYTECODE=1 python3 research/prepare_qwen35_package_native_preflight_v2.py \
  research/qwen35-package-native-preflight-source-bundle-20261112-v2.json
```

28/28 tests passed. New checks exercise the actual pinned functions with a deterministic fake tokenizer, assert all 96 cases and known protocol coverage, reject a same-length single-token change in every input family, reject saved-reference mismatch/missing cases/template-argument changes, exercise the exact long source slice, reject changed source hashes, verify temporary small-file digest/lineage behavior and the exact read-only audit allowlist, prohibit real model-directory reads during preparation, and authenticate immutable v1 source bytes plus unchanged v1 worker function bodies. Fake IDs only test checking mechanics; they provide no native tokenizer equivalence evidence. Synthetic nested evidence fits the existing reader/output ceilings; real execution still enforces those ceilings.

Root input fields remain `{schema,bundle,work,binding}` with the same exact owner, original lock, complete current history and cgroup structures as v1. The schema is now `qwen35-package-native-preflight-v2`, and the bundle reference is the concrete path/size/hash above. No actual runtime binding or runnable input has been issued here. Only root can supply its then-current binding and launch the future worker in its existing serial lifecycle.

Passing this future preflight would add native tokenizer-input equivalence to import/config/CPU integration evidence. It would not establish model numerical correctness, state behavior, generated quality, HTTP performance, artifact admission or any inherited r10 performance claim.
