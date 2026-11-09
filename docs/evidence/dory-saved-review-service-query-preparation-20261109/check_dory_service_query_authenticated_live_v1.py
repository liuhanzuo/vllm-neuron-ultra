"""Read-only authenticated-query parity; never admits or alters a reviewer."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

Q = Path(__file__).resolve().parent
sys.path[:0] = [str(Q), str(Q.parent)]
import minicpm5_layer0_execution_protocol_v4 as original
import dory_service_query_auth_v1 as auth

OBSERVATION = Q / 'dory-service-query-dependency-observation-20261109-v1.json'
OBSERVATION_SHA = '11de174a895bf018dc209814be7d349b7c7cf88413038fd29d2537eeae8e41c4'


def main():
    original.user_bus()
    observed = auth.read_observation(OBSERVATION, OBSERVATION_SHA)
    initial_finders = list(sys.meta_path)
    startup_checks = []
    def before_check():
        startup_checks.append(original.active_heavy())
    query = auth.AuthenticatedServiceQuery(observed, before_check=before_check)
    rows = []
    try:
        for i in range(20):
            before = original.active_heavy()
            actual = query.check()
            after = original.active_heavy()
            assert before == actual == after
            rows.append(dict(index=i, original_before=before, authenticated=actual, original_after=after))
        assert query.requests == 20 and not query.poisoned
    finally:
        query.close()
    assert query.closed and query.restored and sys.meta_path == initial_finders
    assert query.startup_file_count == len(startup_checks) == 41
    assert query.startup_bytes == 4326816 and not query.poisoned
    result = dict(schema='dory_service_query_authenticated_live_parity_v1',
        observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        complete=True, read_only=True, dependency_observation=dict(path=str(OBSERVATION),sha256=OBSERVATION_SHA),
        authenticated_files_before_import=query.startup_file_count,
        authenticated_bytes_before_import=query.startup_bytes,
        startup_original_service_checks=len(startup_checks), original_parity_checks=40,
        persistent_query_checks=20, parity_rows=rows, final_audit=query.final_audit,
        private_connection_closed=query.closed, original_import_policy_restored=query.restored,
        file_pins_rechecked_at_close=True, actual_positive_transition_tested=False,
        service_mutated=False, running_reviewer_modified=False, raw_tensor_bytes_read=0,
        full_runtime_environment_review_complete=False, successor_runtime_admitted=False,
        latency_benchmark=False, model_qualified=False, new_speedup_claim=False)
    path = Q / 'dory-service-query-authenticated-live-parity-20261109-v1.json'
    raw = (json.dumps(result, indent=2) + '\n').encode()
    with path.open('xb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())
    print(json.dumps(dict(result=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest()),
        authenticated_files=query.startup_file_count, authenticated_bytes=query.startup_bytes,
        parity_cases=len(rows), authenticated_imports=len(query.final_audit['authenticated_imports']),
        new_native_files=query.final_audit['new_native_files'])))


if __name__ == '__main__':
    main()
