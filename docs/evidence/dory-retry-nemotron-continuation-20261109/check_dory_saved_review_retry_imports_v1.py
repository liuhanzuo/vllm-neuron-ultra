"""Read-only query compatibility in the exact prepared retry import context.

No lock acquisition, historical drain, full reviewer, model or raw tensor read.
The bytecode prefix remains absent. Initial executor imports are observed and
checked after import; this does not certify the full future runtime environment.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

Q = Path(__file__).resolve().parent
PREFIX = Q/'dory-saved-review-retry-imports-no-bytecode-cache-v1'
assert sys.dont_write_bytecode and sys.flags.optimize == 0 and not PREFIX.exists()
sys.pycache_prefix = str(PREFIX)
sys.path[:0] = [str(Q), str(Q.parent)]

import run_dory_saved_review_retry_v1 as executor
# This literal import occurs during physical_drain before the runtime query
# is constructed. Preload it here without invoking the drain or its services.
from research.finalize_smollm3_sio14_root_20261108_v1 import life
from dory_service_query_auth_v1 import AuthenticatedServiceQuery, read_observation


def main():
    assert not any(n in sys.modules for n in ('torch','numpy','transformers','torch_neuronx','neuronx_distributed'))
    parent = executor.contract.original.load(executor.contract.PARENT_SPEC)
    sources = {}
    for name, module in list(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if filename is None:
            continue
        path = Path(filename)
        if not path.is_relative_to(Q.parent):
            continue
        assert path.suffix == '.py' and path.resolve() == path
        cached = getattr(module, '__cached__', None)
        if cached:
            assert Path(cached).is_relative_to(PREFIX) and not Path(cached).exists()
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if str(path) in parent['source_sha256']:
            assert parent['source_sha256'][str(path)] == digest
        sources[name] = dict(path=str(path),sha256=digest,bytes=len(raw),
                            original_source_pin=str(path) in parent['source_sha256'])
    observation_ref = executor.contract.QUERY_OBSERVATION
    observation = read_observation(observation_ref['path'], observation_ref['sha256'])
    original_finders = list(sys.meta_path)
    query = AuthenticatedServiceQuery(observation, before_check=executor.p.low.active_heavy)
    rows = []
    try:
        for index in range(20):
            before = executor.p.low.active_heavy()
            actual = query.check()
            after = executor.p.low.active_heavy()
            assert before == actual == after
            rows.append(dict(index=index,original_before=before,persistent=actual,original_after=after))
        assert query.requests == 20 and not query.poisoned
    finally:
        query.close()
    assert sys.meta_path == original_finders and query.restored and not query.poisoned
    assert not PREFIX.exists()
    result = dict(schema='dory_saved_review_retry_import_compatibility_v1',
        observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        complete=True,read_only=True,prepared_executor=str(Path(executor.__file__)),
        observed_local_modules=sources,observed_local_module_count=len(sources),
        original_source_pins_matched=sum(row['original_source_pin'] for row in sources.values()),
        executor_imports_authenticated_before_execution=False,
        query_dependencies_authenticated_before_import=True,
        query_authenticated_files=query.startup_file_count,query_authenticated_bytes=query.startup_bytes,
        query_parity_cases=20,original_parity_queries=40,parity_rows=rows,
        query_final_audit=query.final_audit,query_closed=query.closed,import_policy_restored=query.restored,
        bytecode_prefix=str(PREFIX),bytecode_prefix_absent=True,
        lock_acquired=False,physical_drain_executed=False,full_saved_review_executed=False,
        tensor_or_model_bytes_read=0,model_imported=False,service_mutated=False,
        current_reviewer_modified=False,actual_positive_service_transition_tested=False,
        full_runtime_environment_review_complete=False,runtime_admitted=False,new_speedup_claim=False)
    path = Q/'dory-saved-review-retry-import-compatibility-20261109-v1.json'
    raw = (json.dumps(result,indent=2)+'\n').encode()
    with path.open('xb') as f:
        f.write(raw);f.flush();os.fsync(f.fileno())
    print(json.dumps(dict(result=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest()),
        observed_local_modules=len(sources),original_source_pins_matched=result['original_source_pins_matched'],
        authenticated_query_files=query.startup_file_count,query_parity_cases=20,
        authenticated_query_imports=len(query.final_audit['authenticated_imports']))))


if __name__ == '__main__':
    main()
