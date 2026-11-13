# KV256 proposal v4: delivery correction and alias-0 evidence boundary

This fresh addendum leaves the delivered v2 and v3 files untouched.

## V2 was edited and restored after delivery

After v2 had been delivered, a root message requested updating it with the actual
partial terminal. The author edited v2, then received root's time-crossed message
requiring a fresh version instead. The author restored the original v2 contents
and put the terminal information into the new v3. This edit-and-restore happened;
the restored file must not be represented as having unchanged inode, modification
time or change time.

The observed restored v2 is 7,597 bytes with SHA256:

```text
005c20694338f781e985f63622097e0da62f4bc0e1cadc962f8d19b5bbe5c04b
```

Observed filesystem metadata after restoration was device 66306, inode 8348050,
mtime_ns 1794541670878334653 and ctime_ns 1794541670878334653. No pre-edit metadata
identity is claimed. Root confirms that v2 was never enrolled in the current
running spec; this restoration does not change any bound experiment source.
V2 will not be edited again. Future changes will use fresh successor files.

V3's delivered bytes are 4,083 bytes, SHA256:

```text
621e35a84a74ac893cd00e329b2c587d7a784726847283f4859a05fff8b68966
```

## Alias-0 has no complete saved call record

The 15 complete saved call records do not prove that the process never began
call 16 before interruption. The precise observation is: six graph records exist,
including alias-0, but **alias-0 has no complete saved call record**. Whether an
additional call began or ran partly is not determined by this saved prefix.
No result is inferred for alias-0 or the subsequent tail-copy/full-copy controls.

V3 already gives the exact source order: comparison proofs are computed first,
then `expected` is assigned for the next call, then `require(passed)` uses the
already-computed proofs for candidate and finite cases. That ordering preserves
the immediate failure check. Original where/raw continuation explains its later
tail equality only; it is not grounds to dismiss passing candidate/finite checks.

The root-supplied outer failure, NVMe reservation cause and independent audit
status remain as in v3. No new runtime observation or independent raw audit was
performed for this correction, and no compiler pass is claimed as proven.
