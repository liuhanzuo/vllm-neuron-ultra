"""Assemble a future concrete candidate from actual completed predecessor data.

Writes nothing, launches nothing and never reads tensor/native payloads.
Call only after the selected v22/v23/v24 Nemotron run really completes and exits.
The candidate still requires final root source review and fresh serial checks.
"""
import copy
from pathlib import Path
import os

import minicpm5_saved_tap_runtime_contract_v4 as c
from prepare_dory_residual_spec_v3 import local_imports

Q,ROOT=c.Q,c.ROOT
ENTRIES=tuple(Q/name for name in (
    'minicpm5_after_nemotron_package_v2.py','check_minicpm5_after_nemotron_package_v2.py',
    'minicpm5_saved_tap_runtime_contract_v4.py','run_minicpm5_saved_tap_serial_v4.py',
    'prepare_minicpm5_saved_tap_runtime_v4.py','check_minicpm5_saved_tap_runtime_contract_v4.py',
    'check_run_minicpm5_saved_tap_serial_v4.py','check_prepare_minicpm5_saved_tap_runtime_v4.py',
    'check_minicpm5_saved_tap_runtime_metadata_v2.py',
    'check_minicpm5_saved_tap_v24_integration_v1.py',
    'dory_saved_review_service_query_v1.py'))+tuple(ROOT/name for name in (
    'nemotron9b_package_protocol_v22.py','nemotron9b_package_protocol_v23.py','nemotron9b_package_protocol_v24.py',
    'nemotron9b_completed_storage_completion_v1.py','nemotron9b_completed_storage_completion_after_retry_v1.py',
    'nemotron9b_completed_storage_completion_after_retry_v2.py'))
ENTRIES += tuple(Q/name for name in ('minicpm5_after_nemotron_package_v1.py', 'check_minicpm5_after_nemotron_package_v1.py', 'minicpm5_saved_tap_runtime_contract_v1.py', 'run_minicpm5_saved_tap_serial_v1.py', 'prepare_minicpm5_saved_tap_runtime_v1.py', 'check_minicpm5_saved_tap_runtime_contract_v1.py', 'check_run_minicpm5_saved_tap_serial_v1.py', 'check_prepare_minicpm5_saved_tap_runtime_v1.py', 'check_minicpm5_saved_tap_runtime_metadata_v1.py'))


ENTRIES += tuple(Q/name for name in ('minicpm5_saved_tap_runtime_contract_v2.py', 'run_minicpm5_saved_tap_serial_v2.py', 'prepare_minicpm5_saved_tap_runtime_v2.py', 'check_minicpm5_saved_tap_runtime_contract_v2.py', 'check_prepare_minicpm5_saved_tap_runtime_v2.py'))


ENTRIES += (Q/'minicpm5_after_nemotron_package_v3.py',)


def assemble(profile,actual_exit_ref,failed_release_exit_ref=None):
    completed=c.history.completed(profile,actual_exit_ref,failed_release_exit_ref)
    c.require(not c.OUT.exists(),'Fresh saved-tap output namespace; no partial-output reuse')
    reader=c.history.Reader(lambda:None)
    io_review=reader(c.IO_REVIEW)
    diagnosis,manifest=reader(c.inputs.DIAGNOSIS_REF),reader(c.MANIFEST)
    c.inputs.validate_manifest(diagnosis,manifest,os.getuid())
    observation=reader(c.QUERY_OBSERVATION)
    c.require(len(observation['source_and_binary_pins'])==41
              and observation['source_and_binary_bytes_hashed']==4326816,'Original complete query dependencies')
    graph=local_imports(entries=ENTRIES)
    pins={}
    for addition in (completed['source_sha256'],io_review['source_sha256'],graph['source_sha256'],reader.pins,
        {p:r['sha256'] for p,r in observation['source_and_binary_pins'].items()}):c.history.merge(pins,addition)
    sizes={p:Path(p).stat().st_size for p in pins}
    spec=dict(schema=c.SCHEMA,output=str(c.OUT),executor=str(c.EXECUTOR),budget=copy.deepcopy(c.BUDGET),
        profile=profile,nemotron_actual_exit=copy.deepcopy(actual_exit_ref),nemotron_failed_release_exit=copy.deepcopy(failed_release_exit_ref),
        diagnosis=copy.deepcopy(c.inputs.DIAGNOSIS_REF),
        input_manifest=copy.deepcopy(c.MANIFEST),io_preparation_review=copy.deepcopy(c.IO_REVIEW),
        query_dependency_observation=copy.deepcopy(c.QUERY_OBSERVATION),source_sha256=pins,
        essential_sources=sorted(pins),source_read_cap_bytes=sum(sizes.values())+4*2**20,
        diagnostic_only=True,runtime_admitted=False,automatic_launch=False,full_model_qualified=False,new_speedup_claim=False)
    c.validate_spec(spec)
    return spec,dict(schema='minicpm5_saved_tap_runtime_preparation_v4',profile=profile,
        completed_predecessor_exit=actual_exit_ref,actual_failed_release_exit=failed_release_exit_ref,predecessor_services=1202,
        predecessor_owners=len(completed['history']['processes']),local_import_graph=graph,
        source_file_sizes=sizes,source_bytes_planned=sum(sizes.values()),source_review_required=True,
        runtime_specification_written=False,serial_lock_acquired=False,raw_tensor_bytes_read=0,
        native_started=False,full_model_qualified=False,new_speedup_claim=False)


if __name__=='__main__':raise SystemExit('Call assemble only after actual completed Nemotron; separate final review is mandatory')
