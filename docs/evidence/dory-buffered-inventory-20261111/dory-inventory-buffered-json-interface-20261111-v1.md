# Dory bounded buffered JSON successor

Source-only successor dory_native_inventory_finish_v4.py. The original v3 and
all historical failure evidence remain unchanged. No actual raw,output,proc,
process,lock,native or serial experiment operation was performed here. A source
successor and synthetic mock tests do not establish why the historical r4 CPU
limit was reached and do not supply a model or wall-clock speedup claim.

## Narrow change

The original writer guards/budgets/writes every JSONEncoder.iterencode fragment,
including punctuation and tiny scalar pieces. The new json_blocks(value) gathers
the exact same compact stream into blocks no larger than65,536 bytes before
budget.add and stream.write. JSONEncoder separators remain(',',':'),allow_nan=False
and default ensure_ascii=True. Its ASCII text is sliced before encoding, so the
writer's pending byte buffer never exceeds64 KiB, even when the encoder returns
a large escaped string fragment. The encoder's own string allocation behavior
is unchanged; this is not a claim of bounded memory for arbitrary input objects.

A new budget.check runs before buffering the first bytes. Every emitted chunk
still checks the original per-file byte limit, then original aggregate output
budget/actual guard, before writing. A final short chunk preserves the exact
stream without whitespace,newline,truncation or sorting changes. The only other
source change is source_successor metadata naming v4.

Original Budget,LIMITS,PINS,file_bytes and all other functions remain unchanged.
CPU600s,wall900s,output64 MiB,per-file32 MiB,metadata8 MiB and every other limit
retain their existing values. No worker,launcher,plan or catalog is rewired.

Exclusive creation, complete write count, file flush/fsync,directory fsync,
full postwrite hash/size/EOF/identity validation through unchanged file_bytes and
returned result_ref remain. Errors retain already written prefixes and the
created file, and propagate the original failure. With buffering, an error can
leave a shorter partial prefix than the old fragment writer; no partially
written file is treated as a completed result and no evidence is deleted.

## Pure mocked controls

12 tests passed (tool chunk142199,exit0). All file/descriptor/path operations
are injected memory mocks, including unchanged file_bytes readback. Test cases
compare actual old/new writer bytes for nested collections,Unicode escapes,
surrogates,empty values,large strings and long lists. Boundary lengths around
64 KiB and128 KiB preserve exact bytes and never emit an empty/oversized block.

Tests retain original file+directory fsync order and complete readback; reject
readback hash/identity changes; enforce per-file/aggregate caps before writes;
retain partial files after NaN/Infinity,guard,short-write and sync failures; and
retain exclusive-create behavior. AST comparison proves only json_blocks,
write_json and the source_successor label changed. No native imports occur.

A separate in-memory example (chunk7867b6,exit0) serializes3,000 synthetic file
records to345,781 identical bytes. Old budget.add/write calls:42,001. New:6,
with lengths[65536,65536,65536,65536,65536,18101]. Total guard calls including
postwrite full readback fall from42,008 to14. This is a deterministic source
operation-count observation, not timing and not proof of the historical failure's
causal attribution. Root must independently review/rebind concrete source and
all original runtime constraints before any further real inventory execution.
