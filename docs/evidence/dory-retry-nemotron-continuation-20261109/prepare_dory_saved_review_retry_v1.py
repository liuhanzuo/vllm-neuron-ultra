"""Source-only retry assembler; cannot run before the actual failed r2 exit.

Returns a concrete candidate for separate root review; writes no specification,
does not launch, and does not read model/tensor/source binary payloads.
"""
import copy
from pathlib import Path

import dory_saved_review_retry_contract_v1 as contract
from prepare_dory_residual_spec_v3 import local_imports, merge

Q = Path(__file__).resolve().parent
ENTRIES = tuple(Q/name for name in (
    'prepare_dory_saved_review_retry_v1.py', 'dory_saved_review_retry_contract_v1.py',
    'run_dory_saved_review_retry_v1.py', 'dory_service_query_auth_v1.py',
    'dory_saved_review_service_query_v1.py',
    'dory_saved_review_guard_v1.py', 'check_dory_saved_review_retry_contract_v1.py',
    'check_dory_saved_review_retry_executor_v1.py', 'check_dory_saved_review_retry_imports_v1.py',
    'check_dory_saved_review_retry_builder_v1.py'))
require = contract.require


def assemble(failed_exit_ref):
    reader = contract.original.Reader()
    previous = contract.failed_history(failed_exit_ref, reader)
    require(not contract.NEXT_OUT.exists(), 'Retry cannot reuse an existing output directory')
    parent = previous['parent']
    observation = reader(contract.QUERY_OBSERVATION)
    require(observation['schema'] == 'dory_service_query_dependency_observation_v1'
            and observation['complete_observation'] is True
            and len(observation['source_and_binary_pins']) == 41
            and observation['source_and_binary_bytes_hashed'] == 4326816,
            'Exact complete query dependency observation')
    graph = local_imports(entries=ENTRIES)
    pins = {}
    merge(pins, parent['source_sha256'])
    merge(pins, graph['source_sha256'])
    merge(pins, {p: row['sha256'] for p, row in observation['source_and_binary_pins'].items()})
    merge(pins, reader.pins)
    sizes = {p: Path(p).stat().st_size for p in pins}
    spec = dict(schema=contract.SCHEMA, parent_review_spec=contract.PARENT_SPEC,
        output=str(contract.NEXT_OUT), executor=str(contract.EXECUTOR), reviewer=parent['reviewer'],
        producer_spec=parent['producer_spec'], native_exit=parent['native_exit'],
        failed_review_exit=copy.deepcopy(failed_exit_ref), budget=copy.deepcopy(parent['budget']),
        query_dependency_observation=contract.QUERY_OBSERVATION,
        source_sha256=pins, essential_sources=sorted(pins),
        source_read_cap_bytes=sum(sizes.values())+4*2**20,
        diagnostic_only=True, runtime_admitted=True, automatic_launch=False,
        full_model_qualified=False, new_speedup_claim=False)
    contract.validate_spec(spec, reader)
    require(all(spec['source_sha256'].get(p) == digest for p, digest in reader.pins.items()),
            'Every consumed terminal metadata input belongs to the concrete candidate closure')
    return spec, dict(schema='dory_saved_review_retry_spec_preparation_v1',
        actual_failed_review_exit=copy.deepcopy(failed_exit_ref),
        actual_terminal_metadata_pins=reader.pins, local_import_graph=graph,
        actual_predecessor_services=1149, actual_predecessor_owners=348,
        actual_admitted_directories=37, source_file_sizes=sizes,
        source_bytes_planned=sum(sizes.values()), source_binary_bytes_authenticated_here=0,
        query_dependency_files=41, full_runtime_environment_review_complete=False,
        source_review_required=True, downstream_completion_validators_required=True,
        candidate_specification_only=True, specification_written=False,
        serial_lock_acquired=False, raw_tensor_bytes_read=0, native_started=False,
        automatic_launch=False, full_model_qualified=False, new_speedup_claim=False)


if __name__ == '__main__':
    raise SystemExit('Root must call assemble after a real failed saved-review exit and review its result')
