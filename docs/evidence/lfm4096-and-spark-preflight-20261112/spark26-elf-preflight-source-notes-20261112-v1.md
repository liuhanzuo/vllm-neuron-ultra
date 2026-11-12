# Spark metadata-only ELF preflight successor

This is an unexecuted source proposal. It does not admit the existing candidate, remove extras, install packages, repair OpenCV, inspect actual ELF bytes, import Torch/ctypes/history, operate a real lock/service/process, or depend on original model weights during this source task. Parent must independently review the actual fresh binding before any runtime action.

## Extras source finding

The separate spark26-generator-extras-source-analysis-20261112-v1.json pins 15 source excerpts plus both METADATA files. The frozen provider has 16 required package roots; no extras were specified there or in the scientific contract. ROOT_EXTRAS was added by the manifest generator. Transformers imports external accelerate only behind its availability check; internal integrations/accelerate.py explicitly contains copied native fallback helpers. NxD catches ImportError and implements its own init_on_device fallback. The diagnostic constructs projection modules directly and calls DynamicCache / QKV split / host reorder methods; it does not call from_pretrained, device_map dispatch, or offloading. This is evidence that transformers[torch] is not required by the inspected direct diagnostic path, not proof of every dynamic import.

safetensors.torch is actually imported through Transformers modeling_utils. It needs packaging and Torch, and its installer extra also requests NumPy. All these concrete dependencies already exist in the original required/transitive manifest. Extras are installation metadata, not runtime switches. Counterfactual metadata graphs with either or both generator-added extras absent retain the exact same178 installed distributions; only the missing accelerate node disappears. Existing extras, candidate files,16 roots, and all scientific gates remain frozen.

## Concrete preflight implementation

spark26_elf_preflight_core_v1.py consumes the exact existing manifest/preparation references. It reconstructs all41,173 payloads using the unchanged accepted provider, retains every1,271 shared-object candidate and all15 conflicting ownership rows, and rejects a changed scope. It has no generic execution callback and can be invoked only by the concrete successor WorkerContext.

The future owned worker would hash every declared application/SDK payload completely, then parse the native candidates on the same still-open fd with bounded pread. Device/inode/size/mtime/ctime are checked before/open/after, with final identity rechecks across the complete scan. A hash mismatch is recorded honestly; no native override is authorized. All1271 candidate slots and15 conflicts remain represented even when per-object metadata is unsupported or unreadable. Size/canonical-path failures remain explicit unresolved rows. Lease, global budget or output-cap failures abort the attempt; no partial attempt is relabelled complete.

The parser supports bounded ELF64 little-endian x86-64 ET_DYN/ET_EXEC program headers, the dynamic table, needed/SONAME/RPATH/RUNPATH/interpreter and auxiliary/filter/audit directives. It uses no readelf, ldd, subprocess, ctypes, dlopen, or library initialization. Unknown/unsupported formats, range errors, excessive strings, and overlarge output become explicit metadata failures. Static metadata never proves actual loader selection or dependency closure. A later step must review dependency candidates and any runtime imports separately; this preflight does not do that work implicitly.

worker_v1 and outer_v1 are new successors derived from the accepted scientific worker/outer. Both preserve original Python3.10 -S -B, resource limits, original lock inode, full historical drain, independent actual env-to-Python ACK, observed success/failure/unknown exit distinctions, two stable recursive drains, and retained-lock cleanup. The worker proves the original outer descriptor before the deferred historical ctypes import; application packages remain blocked throughout. Historical ctypes is the existing explicit host trust boundary, not application authentication. There is no scientific authority method, native provider import, model checkpoint key/path/check, or automatic successor.

The future admission schema explicitly requires metadata_preflight_admitted=true AND runtime_admitted=false. Both root reviews must bind the exact successor source map/spec/outer binding. This proposal writes no such reviews or admission and selects no live catalog/handoff/service name. The candidate's missing optional-extra dependency,15 conflicts and dynamic-closure blockers remain present even if collection later succeeds.

## Original budgets and finite I/O

MemoryMax16GiB / MemoryHigh14GiB / no swap / address space24GiB / CPU6-7 / nice19 /10800seconds, root28GiB free, NVMe30GiB free plus Qwen2GiB reservation, MemAvailable2GiB,48MiB total output,8MiB/file,16MiB events,16MiB receipts,8MiB logs,4MiB temp/directories and64 entries are unchanged. Outer retains its original separate2GiB address-space limit and no kill-on-deadline cleanup shortcut.

Declared full hash bytes are13,247,146,338. The preflight adds a128MiB aggregate ELF metadata sub-cap,1MiB per object,4096-byte strings and a256KiB dynamic table cap. Hash requests plus ELF requests together must remain <= the original16GiB authentication ceiling, including EOF probe requests. This is stricter than the original one-pass authentication allowance and does not claim to bound all text metadata/history/OS I/O. No unknown linked library is auto-added, downloaded or opened.

Events are segmented every4096 rows with the original3072-byte row limit. Complete output requires41,173 hash rows,1,271 ELF rows and15 conflict rows;13 event files total. Outer independently rereads every event, checks all index sequences/counts/file hashes and reconciles failures. Complete metadata collection can still report authentication/ELF failures and always leaves runtime admission, ownership resolution and dynamic closure false.

## Validation evidence

-24 inherited/adapted lifecycle tests passed: tool7927e0, exit0 (mocked process/service/lock observations only).
-20 parser/authentication/scope/output tests passed: toolb72437, exit0. ELF bytes are in-memory synthetic fixtures; fd operations are mocked. A full42,459-row temporary JSON event fixture was independently reread and tampering rejected. Actual disk reads for candidate reconstruction were source/JSON/RECORD only.
-The initial19-test core suite had one test-helper assertion error (it incorrectly required more than3 guard calls for immediate unsupported-ELF failure); the v2 test successor fixes that expectation and adds honest metadata-summary failure/admission checks. Original tested files remain intact. Worker source emits a harmless Python regex escape deprecation warning under the host test interpreter; its match semantics and source remain unchanged.

No current native payload has been authenticated by these tests. No preflight was started. Original4,844 scientific events and cold/long/slot/quality96/matched HTTP/logits requirements are untouched.
