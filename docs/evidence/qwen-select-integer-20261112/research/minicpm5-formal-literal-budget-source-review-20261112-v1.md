# Minimum first step after formal r5 SIGXCPU: same-call budget reuse

This SOURCE/MOCK successor changes one pre-spawn pure calculation. It combines
the original runtime and stdout reservations so that they share a fully checked
admission decode, one conservative prospective terminal, and its immutable exit
wire. It adds no persistent cache, runtime framework, authority, current binding,
new phase or limit. It does not claim to resolve the observed CPU exhaustion.

## Concrete source difference

There are two small new runtime files and three copied entry files with one
import replacement each:

```text
outer_source_entry_v10          (v9 body unchanged; import source_adapter_v4)
  source_adapter_v4             (v3 body unchanged; import observed_adapter_v6)
    observed_adapter_v6         (v5 body unchanged; import observer_backend_v5)
      observer_backend_v5      (inherits frozen v4 native backend and Store)
        formal_literal_budget_v1.combined_budget
```

The backend overrides only `reserve_runtime`. Its AST is identical to the
original method after replacing these two statements:

```python
budget = codec.future_runtime_budget(raw, argv, self.literal)
budget.update(literal_result.future_stdout_budget(raw, argv, self.literal, str(self.root)))
```

with:

```python
budget = prepared.combined_budget(raw, argv, self.literal, str(self.root))
```

The combined function still invokes the frozen codec's complete prospective
terminal validation and uses its exact owner, exit and fallback encodings. It
retains both runtime/stdout schemas, all fields, conservative maximum scalar
widths, complete runtime/fallback/stdout caps, six-file domain and allocation
margin. It reuses the already computed exit bytes only inside this one call.
No shared state survives return; changed valid inputs are recomputed, and invalid
input changes are rejected by the original codec checks.

`minicpm5-formal-literal-budget-source-diff-20261112-v1.patch` contains the small
review diff and the two new implementations. It is review material; all files
already exist separately, and frozen files must not be patched in place.

## Preserved verification and execution boundaries

The original Store's `prospective_admission`, `before_spawn`, `write`, `load`,
`inventory` and `validate_observer_budget` methods are inherited unchanged.
Every actual pre-spawn durable readback still runs. Actual wire hashes, file
identities and original guard callbacks remain at their original boundaries.
The two old pure budget functions supplied no live guard callback; sharing their
internal immutable results therefore removes no runtime guard checkpoint.

The native factory and NativePorts are the exact frozen objects from backend v4.
The source adapter, complete outer body and observed adapter body are unchanged.
Thus source/native authentication, all fresh identity scans, the sole complete
historical reconstruction, ownership/ACK/gate/ptrace/wait/drain, SDK worker v6,
nine-native checks and the 1,284-capture scientific gate retain their paths.
CPU soft600/hard605, wall900, affinity, memory and artifact limits are unchanged.
The earlier source-only review's plan-digest and broader source-pass proposals
are not implemented in this deliberately smaller first step.

## Verification evidence

Pinned Python command:

```text
/home/ubuntu/venvs/vllm_neuron/bin/python -I -B /home/ubuntu/experiments/trn1-latest-models-20261031/research/check_minicpm5_formal_literal_budget_v1.py
```

Eight tests passed: tool chunk `7ec22b`, actual exit 0. Tests block ctypes, torch,
NumPy, Neuron and related native package imports. No native history module is
imported or executed. They use only saved r5 admission/owner metadata, frozen
source text and explicitly fake in-memory Store boundaries.

For the actual saved r5 admission/owner values:

| Calculation | Pure SHA calls | Total bytes submitted to SHA |
| --- | ---: | ---: |
| Original two reservations | 18 | 911,953 |
| Combined reservation | 12 | 552,023 |
| Removed duplicate computation | 6 | 359,930 |

Every budget field and its canonical complete JSON are equal. Full Store mock
preparation, admission write and two successive `before_spawn` checks produce
the same guard event sequence and the same fresh read/reference/write events.
Both successive spawn checks still perform complete readback. The counting SHA
oracle is checked against the real frozen pure SHA implementation for digest
value and every guard callback, including padding/chunk boundaries and the
37,178-byte literal.

Negative controls reject altered wire bytes, expanded admission fields with
unchanged hash, literal, argv literal/role/flags/SHA, reservation reference,
repeated reservation, input changes after preparation, changed durable bytes,
changed identity and mutation between reference/read. Both old and new budget
paths reject exceeded runtime/summary caps. Valid changed inputs yield freshly
computed results identical to the old path. Repeated calls repeat computation;
there is no cross-call or cross-run cache.

AST/text checks prove the three one-import differences and the one replacement
expression. Inherited Store methods and native implementations retain the same
code objects. All eight pinned frozen entry/codec/result sources remain intact.

This proves a finite reduction in duplicate pure work and preservation of the
tested boundaries. It is not a runtime profile, SDK import result, CPU-peak fix,
or demonstration that a complete run will finish within 600 CPU seconds.

## Root review and eventual integration

The source delivery inventories all six new source files, their exact hashes,
the review diff and this document. It is an additive source delta, not a selected
current 48,223-row map and not a runtime admission.

For a separately reviewed later run, root would retain every original source
pin and add all five new runtime files to the complete source map and the outer
bootstrap closure. The outer role plan's entry module/path becomes
`research.minicpm5_sdk_mapping_outer_source_entry_v10`; the exact literal remains
frozen `nocache_entry_v6`. The SDK worker/source entry remains frozen v6. Root
must independently bind/review the then-current full history and source/native
authority, including the failed r5 terminal. No actual spec, authority, catalog,
output directory, lock or service is created here. No new run is authorized or
started by this delivery.
