"""Future parent-admitted metadata collector API; no CLI or built-in transport.

Explicit ports must be supplied by the reviewed original serial wrapper.
Imports cannot download, read saved files, acquire locks or start processes.
"""
import dory_glibc_source_metadata_v1 as core


def run(read_saved, fetch_metadata, save_metadata, guard):
    core.need(all(callable(x) for x in (read_saved,fetch_metadata,save_metadata,guard)),
              'Explicit reviewed IO and original retained-lease guard required')
    raw = {}
    for key, reference in core.INPUTS.items():
        guard()
        raw[key] = read_saved(dict(reference), reference['bytes'])
        core.match(raw[key], reference, core.LIMITS['saved_metadata_bytes'])
    prepared = core.prepare(raw, guard)
    del raw
    result = core.collect(prepared, fetch_metadata, save_metadata, guard)
    guard()
    return result


if __name__ == '__main__':
    raise RuntimeError('Source-only API: fresh original serial wrapper admission and bounded IO ports required')
