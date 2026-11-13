# MiniCPM5 completed Qwen writer r2+r3 domain renderer

This source-only successor prepares the still-unpublished seven Mini source files from the accepted r2 renderer. It adds exactly four externally located r3 root helpers. It does not choose actual completion references, write rendered files, read model payloads, run History, acquire the serial lock, or launch a process. No model qualification or speedup follows from source tests.

## Actual input contract

Only after real r2 and r3 finalization, call `render(predecessors, inputs)` in `render_minicpm5_completed_writer_domain_sources_v2.py`. The six `predecessors` byte strings and their fixed SHA-256 values are unchanged from v1. All production output names must still be fresh; the renderer itself performs no filesystem check or write.

`inputs` has exactly ten keys:

- `base`, `catalog`, `ended`, `finalization`: the same authentic completed r2 envelopes required by v1. They preserve all 163 baseline rows, the failed outer tool 72363 / 129c0c / 1, one completed layout, 7,865 calls, seven graphs, and auditor 9791's actual successful terminal.
- `r3_catalog`, `r3_ended`, `r3_finalization`: authentic completed r3 envelopes. The full 164-row r2 prefix and SDK history must remain byte-equivalent as parsed objects; the appended five-field row is checked in full.
- `r3_spec`, `r3_overlay`: authentic actual executed source-spec and overlay envelopes, as bound by the r3 audit. Both must reference the actual completed r2 catalog. The geometry schema, exact r3 run name, overlay schema and spec-to-overlay reference are checked.
- `r3_qualification`: the actual separate qualification envelope on successful geometry, and explicit `None` on failed geometry. Failure remains a valid completed historical row. Root must independently verify filesystem absence of a qualification on failure; a pure function cannot establish absence.

Every non-`None` envelope has exactly `reference: {path, sha256}` and `payload: bytes`. References must be canonical research JSON paths. Full bytes must match their digest. Ordinary catalog/exit/finalization/overlay/qualification inputs retain the 8 MiB cap. Only the bound geometry `r3_spec` uses geometry's existing 64 MiB source-spec cap (`qwen35_writer_geometry_serial_v3/run.py`, `load_spec`); the frozen r2 spec is already 9,414,011 bytes. This is a pure source-renderer input distinction. It does not change Mini metadata/runtime readers, admission, or the accepted exact-r10 transport.

Actual authenticity remains root's duty. Synthetic test bytes with plausible fields cannot establish that an external tool ran. The returned flags `actual_inputs_independently_verified`, `source_accepted`, `launch_authorized`, and `failed_qualification_absence_independently_verified` remain false.

## Exactly preserved source history

The renderer retains the original r2 validation and all 38 r2 exceptional source pairs. It independently reconstructs r3's entire source delta from the supplied full spec, overlay, all 164 old rows, and the four external helpers. Source conflicts, missing historical pins, changed deltas, wrong counts and altered prefixes are rejected. The original 44,532 recovered pairs are additionally bound to their canonical sorted compact-JSON mapping digest `4a740e86760e46bbd19bc9874c8b1e765a10fe2122380262b27197f9c2a2d07a`, derived from the frozen r2 spec and overlay; equal counts alone cannot authorize changed recovered sources.

The four external pairs are fixed in `R3_HELPERS`: accepted request v2, acceptance v2, ACK v1, and finalizer v3. Their complete exact set must appear in real `finalization.external_sources` and in the new catalog source delta. Draft finalizers, neighbors, subsets, different digests and unknown new external paths remain rejected.

The accepted r3 runtime/auditor delivery adds nine canonical research `.py` files relative to frozen r2. Their exact pairs are separately fixed in `R3_RUNTIME_PINS` and must be retained in the spec and appended delta. They already pass the inherited ordinary source-domain rule. No wildcard, directory or suffix permission is added for them. All other source-text readers and the complete real source reconstruction still apply at Mini preparation and acceptance.

The generated shared domain reads the exact authenticated r2 and r3 metadata references, checks the full historical row prefix, and checks selected r3 scientific, ownership, source and audit status. It does not copy `full_history_pair` or layout payloads into Python source. Full source bytes remain below the original 4 MiB text cap; a regression with a 5 MiB synthetic saved-history payload checks that this data is retained through the full metadata hash and reader rather than duplicated into the generated module. The small root finalization and separate qualification retain their complete expected values.

A later current catalog must preserve the complete 165-row historical prefix and SDK history. This grants only the fixed 42 exceptional source pairs. A later external root helper still fails the original domain checker, regardless of any additional catalog row. This renderer is not arbitrary future-history compatibility.

## Unchanged generated wiring and execution requirements

The seven proposed output filenames remain the v1 names, which were still absent when source preparation began. Only the generated `minicpm5_completed_qwen_package_text_v4.py` gains r3 source. The other six output byte strings remain exactly equal to v1's reversible mechanical transformations, retaining every frozen predecessor byte through the edit ledgers.

The full wiring remains 31 catalog source references, nine root helper references and 194 bootstrap sources. Worker v8 and saved-operator reconstruction v2 remain unchanged, including all nine native checks and 1,284 capture requirements. Original 600/605 CPU limits, 600-second metadata wall limit, 900-second runtime wall limit, original weights and precision, numerical gates, cold cache, long decode, request-slot reuse, quality and HTTP requirements remain unchanged. This does not prove SIGXCPU solved.

After actual finalizations, root must independently review real returned source bytes, publish only fresh paths, and complete genuine source reconstruction, budgets, full History, preparation and independent acceptance before native execution.

## Source-test record

The initial 12-test run 42490 / 195890 / exit 1 exposed the incorrect 8 MiB cap applied to geometry source specs. A corrected intermediate run 94857 / 643234 / exit 0 passed 12 tests. Strengthened tests then relinked downstream references after semantic mutations; run 29078 / f2944a / exit 1 exposed the recovered-source mutation gap. The exact frozen recovered-source digest fixes that gap. The final delivery records the final 14-test terminal separately.

Tests use only the real frozen v116 metadata and frozen r2 spec/overlay as existing historical inputs. All prospective r2/r3 completion envelopes and all generated source outputs remain in memory. Native imports, IO, locks and process creation are blocked during renderer calls. Tests cover successful and failed r3, whole-prefix preservation, full source composition, exact helper/runtime pins, negative semantic mutations with valid dependent references, immutable mechanical wiring, large-history source size, metadata caps/digests and future external-domain rejection.
