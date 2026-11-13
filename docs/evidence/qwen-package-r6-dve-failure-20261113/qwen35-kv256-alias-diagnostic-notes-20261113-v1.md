Qwen KV256 persistent writer failure: source-only diagnostic delivery
==================================================================

The actual geometry r3 run remains failed. This delivery supplies a separate,
fixed diagnostic; it does not repair the writer, rerun geometry, qualify a model,
or establish any speedup. No native import, compilation, device call, actual
artifact/tensor read, History resolution, lock acquisition, runtime preparation,
runtime admission or experiment cleanup was performed for this source delivery.
Tests read Python/JSON metadata and use in-memory synthetic bytes. A few storage
tests create and automatically remove tiny explicitly synthetic temporary files.

Observed failure
----------------

The saved r3 reports show conv, recurrent and KV128 completing 5,000 calls each.
KV256 has 3,751 calls: standalone torch_where, standalone NKI and persistent
torch_where each completed 1,250. The first persistent NKI call failed at raw,
mapping index 0, int32 slots `[-1, -1, -1, -1]`. The old gathered output and host
inputs were correct. Full new output and the actual persistent state shared the
wrong hash, including corruption in the untouched tail. There are no retained
raw failing bytes from that original run, so the original exact mismatch
positions and cause cannot be recovered from its hashes alone.

| Evidence | SHA256 |
| --- | --- |
| Actual full new/state | ed6bc91123d8ffc49e028b2685f894ed7f6ed131fab83fefe75f6f9a70ac059d |
| Expected full state | 585fbf90d9c212afe9be4e1620de1c21d0aaaee82186b012b926a80d34da71c4 |
| Actual tail | a368a20c104d9fcbb16472a0773fed04943697b646d8a324e2f46859e06b8586 |
| Expected tail | f8ccf07f1fd1ce156c8fd983f93a411179a19f78ee4998ec63bcdc8e062340b2 |

The original source passes a noncontiguous prefix to the NKI writer, concatenates
the original tail, then aliases the full returned new state back to the persistent
parameter. Alias, concatenation and layout interactions are hypotheses. The
root's independently saved analysis reconstructed the expected hashes using only
integer bytes. Common NaN quieting/canonicalization and subnormal transforms did
not reproduce the actual hashes; this does not establish an alternative cause.

The original raw pattern repeats every 65,536 BF16 words. Both a maximum row and
a KV256 prefix contain whole periods, so rows have identical raw bytes. The
diagnostic preserves raw and finite inputs, and adds a separately named
`row_tagged_raw` profile to expose row/address mixups. It never replaces raw with
a numerically easier input or treats NaNs as equal by value.

Fixed diagnostic
----------------

Every graph gets exactly one all-inactive call for each of raw, finite and
row_tagged_raw: eight graphs, three profiles, 24 calls, one live graph at a time.
State is BF16 `[4, 1, 4096, 256]`; slots are int32. A persistent graph is reset to
the complete exact input bytes before each call, and the actual state binding is
checked. The saved artifacts are authenticated at load and after all calls; they
are not copied into the new output tree.

| Graph | Origin | Control |
| --- | --- | --- |
| saved256_nki_alias1 | r3 exact saved graph | Original failing graph |
| saved256_nki_standalone | r3 exact saved graph | Original graph without persistent alias |
| saved256_where_alias1 | r3 exact saved graph | Original torch_where persistent control |
| saved128_nki_alias1 | r3 exact saved graph | Smaller active prefix, same maximum state |
| fresh256_cat_only_alias1 | Fresh trace | Prefix/tail concatenation without the writer |
| fresh256_nki_alias0 | Fresh trace | Physical `(new, old)`, new state aliases output 0 |
| fresh256_nki_tail_copy_alias1 | Fresh trace | Writer plus explicit NKI fresh-HBM tail copy |
| fresh256_full_copy_alias1 | Fresh trace | Explicit NKI fresh-HBM full-state identity copy |

The alias0 control keeps persistent semantics identical: `new` remains the state
write. It returns physical `(new, old)`, normalizes outputs back to logical
`(old, new)`, and expects persistent state equal to the original cache. It does
not alias the gathered old output. The synthetic row-tagged check distinguishes
these state oracles and tests both the return-order source and normalization.

The cat-only graph may be optimized to an identity; a source-level cat control
does not prove a physical compiled concatenate operation. The explicit NKI copy
controls also change graph/layout choices. Outcomes can narrow hypotheses but
cannot alone prove an alias implementation bug. No control is assumed to compile
or fit its resources until an actual admitted run establishes that fact.

Evidence and limits
-------------------

The leaf retains full actual old/new/state raw bytes, exact input/oracle bytes,
complete LSB-first bitmaps with one bit for every byte position, row/prefix/tail
hashes and mismatch counts, and the first 64 differing words with coordinates,
hex words and BF16 class transitions. The 64-word preview is bounded, while all
difference positions remain in the bitmap. Changed host inputs are retained too.
There are 69 complete output/state comparisons for 24 calls.

A completed mismatch remains diagnostic evidence and the remaining fixed
controls run. Native, reset, resource, source or ownership errors abort. A
completed diagnostic can therefore contain mismatches; completion never means
geometry correctness. The outer and independent external observer each re-read
all saved comparisons and reconstruct every expected pattern and bitmap before
accepting diagnostic completion. Interrupted reports remain failed prefixes;
they do not receive complete diagnostic byte-audit claims.

The original interpreter, flags, SDKEnvironment context restoration, visible
core 0, CPU affinity [6, 7], nice 19, memory 6 GiB/high 5 GiB/no swap, 900-second
leaf, finite outer bounds, source closure, full History, actual external session
ACK, original serial lock identity and whole-cgroup drain are retained. The
runtime has one explicit layout named `kv256_alias_diag`; it cannot masquerade
as conv or pass the original eight-layout validator.

The entire owned NVMe output tree, including root History snapshots, all
directories, compiler intermediates, caches, logs, temporary reports and final
reports, must fit 1 GiB. Both logical and allocated bytes are scanned using the
original strict singleton-file/publication-race checks. All objects fall into
one of three caps:

| Class | Cap |
| --- | ---: |
| All inputs, raw captures, changed host inputs and bitmaps | 708 MiB |
| Four newly saved model.pt files, 32 MiB each | 128 MiB |
| Every other owned output, including directories | 188 MiB |

The worst-case complete raw calculation is 706.5 MiB + 400 bytes. The remaining
class caps are enforced, not predicted compiler consumption. Crossing any cap
fails the diagnostic; there is no automatic cleanup, reduced matrix, compression
fallback, larger cap or retry. Original disk floors/reservations also remain.
Root research JSON receipts retain the original separate 8 MiB publication cap;
they are not a location for compiler/cache/raw spill.

Concrete future caller chain
----------------------------

`derive_qwen35_kv256_alias_diagnostic_serial_v1.py --check` reconstructs the exact
caller files from authenticated frozen source predecessors and checks them
without writing. It derives the serial caller/contract/storage, preparer,
binding tests and external audit observer. The inherited owner, subprocess wait,
SDK, ACK and drain functions are additionally compared by AST in tests.

The source-only preparer accepts explicit future seed/catalog/idle/prerequisite
references. It retains all old rows and sources, source conflicts, complete seed
History, original LFM scope and input identities, and creates only an unadmitted
candidate and failing review draft. It requires an actual root source review at
`qwen35-kv256-alias-diagnostic-serial-root-source-review-20261113-v1.json`; its
digest must be supplied by the future caller. No fake accepted digest is built
into the source. A separate root helper/preparer source review is also required.

`qwen35_kv256_alias_diagnostic_root_helpers_v1.py` has explicit `accept`, `ack`
and `finalize` modes, each taking a caller-supplied JSON request and digest.
`accept` retains the original 2 GiB address-space limit, CPU 600/605-second limit
and alarm 600, full History precheck and original lock. `ack` requires the actual
externally observed session, live owner identities and descriptor, exact idle
state and a complete handoff identity recheck before publishing running.

The independent `qwen35_kv256_alias_diagnostic_audit_v1.py` requires the actual
outer tool session, terminal chunk and exit code. It preserves v5 freshness
guard semantics, exact running ACK, two complete current service-drain checks,
and the original full two-pass History collector. It writes a diagnostic actual
exit and, only after full completion, a diagnostic completion receipt. It does
not write a geometry qualification.

`finalize` requires separately observed outer and successful auditor tool exits,
checks that the actual auditor process and original owners are gone, repeats
both current service drains, retains the entire old catalog prefix, and appends
exactly one true source delta row. All helpers, including accept/ACK/auditor/
finalizer and its adapters, must already be enrolled in the spec before future
preparation. `source_delta(..., external={})` rejects the need for source retrofit.

The original pre-admission/no-admission failure handling remains a root
responsibility if a future outer preflight fails before admission/owner receipts
exist. The diagnostic external observer is intentionally for admitted runs and
does not invent absent owners or an actual session. No current candidate,
request, review, launch command with current hashes, catalog or handoff was
created by this source task.
