"""Explicit read-only parity check; creates no service and launches no model.

This checks the currently observed service state, not an induced transition.
It does not admit this transport for the running or a future saved reviewer.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

Q = Path(__file__).resolve().parent
sys.path[:0] = [str(Q), str(Q.parent)]
CACHE = Q / 'dory-service-query-preparation-no-bytecode-cache-v1'
assert not CACHE.exists()
assert sys.dont_write_bytecode and 'dbus' not in sys.modules
sys.pycache_prefix = str(CACHE)

import minicpm5_layer0_execution_protocol_v4 as original
import dory_saved_review_service_query_v1 as candidate


def pin(path):
    path = Path(path)
    before = path.stat()
    assert before.st_size < 8 * 2**20
    raw = path.read_bytes()
    after = path.stat()
    fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    assert all(getattr(before, k) == getattr(after, k) for k in fields)
    return dict(path=str(path), resolved_path=str(path.resolve()), bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest())


def main():
    original.user_bus()
    query = candidate.connect_fixed_user_bus()
    rows = []
    try:
        for i in range(20):
            before = original.active_heavy()
            actual = query.check()
            after = original.active_heavy()
            assert before == actual == after
            rows.append(dict(index=i, original_before=before, persistent=actual,
                             original_after=after))
        assert query.requests == 20 and not query.poisoned
    finally:
        query.close()
    assert query.closed and not CACHE.exists()
    import dbus
    package = Path(dbus.__file__).resolve().parent
    dependencies = [pin(p) for p in sorted(package.rglob('*.py'))]
    dependencies.append(pin(sys.modules['_dbus_bindings'].__file__))
    mappings = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith('/') and 'libdbus-' in fields[5]:
            mappings.add(fields[5])
    assert mappings
    dependencies += [pin(p) for p in sorted(mappings)]
    result = dict(schema='dory_saved_review_service_query_live_parity_v1',
        observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        complete=True, read_only=True, original_checks=40, persistent_checks=20,
        rows=rows, original_states=list(candidate.STATES),
        connection_closed=True, service_created=False, service_stopped=False,
        actual_positive_transition_tested=False, latency_benchmark=False,
        new_raw_tensor_review=False, running_reviewer_modified=False,
        runtime_admitted=False, full_model_qualified=False, new_speedup_claim=False,
        bytecode_cache_prefix=str(CACHE), bytecode_cache_prefix_remains_absent=True,
        dependencies_observed=dependencies,
        full_runtime_environment_review_complete=False,
        source_files=[pin(Q / name) for name in (
            'dory_saved_review_service_query_v1.py',
            'check_dory_saved_review_service_query_v1.py',
            'check_dory_saved_review_service_query_live_v1.py')])
    path = Q / 'dory-saved-review-service-query-live-parity-20261109-v1.json'
    raw = (json.dumps(result, indent=2) + '\n').encode()
    with path.open('xb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(dict(result=pin(path), checks=len(rows),
                         dependency_files=len(dependencies))))


if __name__ == '__main__':
    main()
