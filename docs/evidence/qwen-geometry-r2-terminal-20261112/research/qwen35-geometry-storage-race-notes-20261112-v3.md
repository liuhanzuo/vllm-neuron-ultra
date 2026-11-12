# Geometry storage scanner successor v3

This is a source-only successor to geometry runtime v2, preserving its History v3 compatibility. It does not authorize or start a new experiment. No current or future catalog is selected.

## Actual failure and limits of diagnosis

Root reported actual geometry r2 outer session 72363, terminal 129c0c, exit 1. The error was `Singleton regular output only`; conv completed 5000 calls and recurrent retained 2865 of 5000. The triggering path and stat metadata were not recorded. Root's post-exit metadata inventory has 97 entries and no non-singleton regular or special files. The actual triggering path and actual cause remain unknown. This delivery does not promote those partial calculations to qualification.

A bounded isolated root-filesystem reproduction used unmodified `update_json` and `observed_stat`. Actual tool session 24795, terminal 628a08, exit 0: 1249 tiny JSON replacements, 2882692 lstat calls, 965 observations of regular old inodes with nlink 0, with eight complete examples retained. In each example fresh same-path metadata named a different singleton inode. Runtime was 10.0035 seconds, and there were no errors. Reproduction device 66306 differs from the real output NVMe device 66304. This proves the metadata-race mechanism can occur; it does not prove the r2 failure had that cause. No native imports, actual History, locks, model data or runtime payloads were involved.

## Scanner change

The original publisher writes an exclusive `.tmp`, fsyncs, calls `os.replace`, then fsyncs the directory. Its update function remains unchanged. Old directory lookup metadata may briefly report the replaced inode with nlink 0.

The scanner never counts that zero-link object as acceptable output. A zero-link regular object can restart the whole scan only within the exact declared mutable names (root `current-layout.json` and its `.tmp`, layout `report.json` and its `.tmp`) or the existing layout `tmp`, `cache`, and `artifacts` subtrees. Its original metadata must satisfy the original same-volume and file-cap rules. A fresh path observation must name a different inode that is a regular singleton on the same volume within the same cap. A missing path retries only for a declared mutable temporary or existing compiler-volatile namespace. A disappeared final report, stable zero-link inode, unchanged inode, hardlink, special file, symlink, wrong device, oversized object, or zero-link observation at an immutable path still fails.

The original inherited hardlink publication validator remains authoritative. The new wrapper preserves its retries and failures with path/stat diagnostics. No broad zero-link acceptance, dropped entry, partial-tree accounting, or generic retry on validation failure is added. A successful scan counts the complete tree using only valid singleton ordinary files.

Exactly eight attempts and seven 10 ms sleeps remain. Each retry diagnostic is complete and limited to 8 KiB, at most eight observations are retained, and the final unsettled-tree diagnostic is limited to 128 KiB. Exceeding a diagnostic limit fails closed without truncation. Successful retry observations are returned in the existing storage result; failures carry diagnostic text. There is no new persistent log writer. Original output and log caps remain in force.

## Complete wiring and preserved contracts

`qwen35_writer_geometry_serial_v3.contract` imports the original contract and overrides only `scan_once`, `scan_output`, and `resources`. Runtime v3 is the v2 source with one contract-import substitution; auditor v3 is the v2 source with one stage-import substitution. All runtime and auditor function ASTs are identical. The resources function AST is identical to the original, but resolves the new local scanner. The geometry implementation and its 8 layouts, 40000 calls, 32 graphs, int32 oracle, state checks, callbacks, failure handling, full History and retained-lock lifecycle are unchanged. The leaf still invokes the frozen original geometry module.

Original budgets remain: 8 GiB total output, 1 GiB per layout, 128 MiB `model.pt`, 16 MiB report, 8 MiB log, 65536 entries; root 28 GiB/NVMe 30 GiB free floors, plus 2 GiB headroom and remaining 8 GiB output reserve (40 GiB initial NVMe requirement); 6 GiB memory maximum, 5 GiB high, no swap; 900 seconds per layout, 7200 service seconds, 9000 outer seconds. No science, precision, model, reference or acceptance threshold changes.

The delivery extends the previously accepted helper-compatible source map without changing old pins. The unchanged metadata binding helper can consume it after explicit root source acceptance. The original geometry source delivery remains the scientific delivery named by future concrete specs. A future concrete selection must explicitly use runtime v3 and auditor v3 with the unchanged History v3 and full then-current serial lineage. This source delivery supplies no such selection and no launch permission.

## Validation

48 source/mock tests passed: 18 focused scanner tests plus all 30 original geometry controls, with native/ctypes imports blocked. Actual tool session 98377, terminal e39719, exit 0. Initial session 77250, terminal d419b8, exit 1 exposed a test-only directory-order assumption: strict rejection reported the other hardlink peer. The corrected assertion accepts either exact peer and still requires nlink 2. Production rejection was unchanged.

The suite includes real small-file atomic replacement with retained old-inode metadata, whole-tree recount, missing final and temporary paths, immutable-path rejection, hardlink/symlink/FIFO/device/cap negatives, exact eight-attempt timing, diagnostic size failure, inherited publication checks, sparse file/layout caps, full runtime/audit/resource AST preservation, and inherited lifecycle/science controls. It does not establish actual Neuron correctness or performance and contains no new speedup claim.

Independent root source review, actual current terminal/catalog/idle checks, fresh complete binding, original History admission, and actual experiment/audit remain future work. Frozen v1/v2 and prior helper/History sources are unchanged.
