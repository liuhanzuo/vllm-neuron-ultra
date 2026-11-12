"""Exact seven-input glibc metadata task inside the original inventory policy."""
import copy
from pathlib import Path

import dory_serial_stage_contract_v2 as c
import dory_descriptor_inventory_runtime_v2 as runtime
import dory_glibc_source_metadata_v1 as core
import dory_glibc_source_metadata_runtime_v1 as adapter
from dory_ubuntu_metadata_input_binding_v5 import storage_usage, save_final_result

WORK_KIND = 'dory_glibc_exact_source_metadata_only_v1'
INPUTS = copy.deepcopy(core.INPUTS)
NETWORK_POLICY = dict(initial_requests=core.request_table(), maximum_requests=3,
    maximum_transfer_bytes=core.LIMITS['transfer_bytes'], socket_timeout_seconds=5,
    request_wall_check_seconds=60, complete_EOF_required=True, unique_exact_Content_Length_required=True,
    DSC_selected_only_from_authenticated_Sources=True, redirects=False, retries=False,
    proxies=False, mirror_fallback=False, source_archive_download=False,
    historical_signed_snapshot_only=True, new_signature_verification=False)
RUNTIME_FILES = ('run_dory_glibc_source_metadata_root_v1.py', 'dory_glibc_source_metadata_worker_v1.py',
                 'dory_glibc_source_metadata_binding_v1.py', 'dory_glibc_source_metadata_runtime_v1.py')
SOURCE_PINS = {
    str(c.Q/'dory_glibc_source_metadata_v1.py'): '6d0e4b2f79f4b2958429265210344afaa8c061b1540e6fd000e01fa74005c437',
    str(c.Q/'collect_dory_glibc_source_metadata_v1.py'): '838ee05f1d3b5aa83e0af28928cd5f06592ae18a586d6deb61c80b05fc4df8d9',
    str(c.Q/'dory_ubuntu_package_sources_v4.py'): '69c8a9c9d34df8a126fc67ec7fe621a9db79bfb4a5ca3352a684dc746e46fc43',
    str(c.Q/'dory_ubuntu_metadata_adapter_v6.py'): 'de6253582e1077c0fb315ce0ebd10b9e22b1cf7484c02624c5bb7248e4201515',
    str(c.Q/'dory_ubuntu_gpgv_status_compliance_v1.py'): 'ed4b3d6cb5108c5a09bb997d65df155057d8ef5fad085a9f76aaaf3e05719ed5',
    str(c.Q/'dory_ubuntu_metadata_adapter_v2.py'): '7048014d89d8c725e5cdc7b9f1ebc1f7a15aacd769c5a7951b7640b6dac891eb',
    str(c.Q/'dory_ubuntu_package_sources_v1.py'): '7a9f1fe5754fb7b29dedad6d2532779f7af2252b2da0053683652b7569c608bd',
    str(c.Q/'dory_ubuntu_metadata_input_binding_v5.py'): 'c92ed8069bd08e6bee5c4ba9610b4f1d37990f3f20999dac3382d39772eb8d25',
    str(c.Q/'dory_serial_stage_contract_v2.py'): 'bb3ed6e61632574f32a5a98b7cee8183394b416225193f9e8f080054f98ecc34',
    str(c.Q/'dory_descriptor_inventory_runtime_v2.py'): '758b075dc7d1dfa10b12e5b4dee445b11691dcfebf7996853fb4b6d4479df556',
    str(c.Q/'run_dory_serial_stage_root_v5.py'): '3589c286b78e7196105fa9d76c45a14c6a47535fb48b6a9604ffdb11c2c3d2f7',
    str(c.Q/'dory_serial_stage_journal_v2.py'): '4003a0f43bf95245f5b8ffc79bbd60c649680a3aa305ec6d33514157e39f4550',
}


def validate(plan, review, plan_ref):
    runtime.validate(plan, review, plan_ref)
    c.require(plan.get('work_kind') == review.get('work_kind') == WORK_KIND
              and plan.get('metadata_inputs') == review.get('metadata_inputs') == INPUTS
              and plan.get('network_policy') == review.get('network_policy') == NETWORK_POLICY,
              'Exact glibc metadata task, seven inputs and finite network policy')
    c.require(all(plan.get(key) is None for key in ('external_authorities', 'native_authority', 'ubuntu_trust')),
              'No native, package payload or new signing authority admission')
    for path, digest in SOURCE_PINS.items():
        c.require(plan['source_sha256'].get(path) == digest, 'Frozen approved metadata API and lifecycle sources')
    for name in RUNTIME_FILES:
        c.require(str(c.Q/name) in plan['source_sha256'], 'Every new runtime source explicitly reviewed')
    return plan


def authenticate(guard):
    raw = {key: adapter.read_saved(dict(reference), reference['bytes'], guard) for key, reference in INPUTS.items()}
    prepared = core.prepare(raw, guard)
    guard()
    return dict(schema='dory_glibc_authenticated_metadata_inputs_v1', work_kind=WORK_KIND,
        inputs=copy.deepcopy(INPUTS), network_policy=copy.deepcopy(NETWORK_POLICY),
        preparation_sha256=adapter.preparation_digest(prepared), source=copy.deepcopy(core.SOURCE),
        binary=copy.deepcopy(core.TARGET), historical_signed_snapshot_only=True,
        new_signature_verification=False, source_archive_contents_authenticated=False,
        exact_loader_source_semantics_established=False, unresolved_loader_edges=15,
        native_ABI_qualified=False, full_model_qualified=False, new_speedup_claim=False)


def validate_admission(value, authenticated):
    c.require(value.get('work_kind') == WORK_KIND and value.get('metadata_input_binding') == authenticated,
              'Original parent admission binds exact glibc metadata inputs and network policy')


def verify_collected(out, value, authenticated, guard):
    """Replay exact saved metadata through the frozen API; no network or writes."""
    root = Path(out)/'glibc-source-metadata'
    c.require(value['result']['path'] == str(root/'source-result.json'), 'Owned glibc metadata result')
    result = c.load(value['result'], core.LIMITS['result_bytes'])
    c.require(result['schema'] == 'dory_glibc_exact_source_metadata_result_v1'
              and adapter.preparation_digest(result['preparation']) == authenticated['preparation_sha256'],
              'Exact authenticated saved preparation')
    saved = result['saved_metadata']
    names = [x['output_name'] for x in core.request_table()]
    if result['status'] == 'exact_source_metadata_ready': names.append('glibc-exact.dsc')
    c.require(result['status'] in ('exact_source_metadata_ready', 'exact_source_version_missing')
              and len(saved) == len(names), 'Complete finite source metadata output set')
    by_name = {}
    for name, reference in zip(names, saved):
        c.require(set(reference) == {'path','bytes','sha256'} and reference['path'] == str(root/name),
                  'Only exact owned metadata outputs can be replayed')
        by_name[name] = reference
    cursor = [0]
    def fetch(url, cap):
        guard(); index = cursor[0]; cursor[0] += 1
        c.require(index < len(saved) and url == result['metadata_requests'][index]
                  and cap == saved[index]['bytes'], 'Exact recorded metadata request replay')
        reference = saved[index]
        body = adapter.stable_read(reference['path'], cap, reference['sha256'], guard, cap)
        return dict(body=body, final_url=url)
    def save(name, body):
        reference = by_name[name]
        c.require(len(body) == reference['bytes'] and core.sha(body) == reference['sha256'], 'Exact saved full metadata bytes')
        return dict(reference)
    replayed = core.collect(result['preparation'], fetch, save, guard)
    c.require(replayed == {k:v for k,v in result.items() if k != 'runtime_transport'}, 'Complete metadata-only result reproduced')
    transport = result['runtime_transport']
    expected_transport = dict(requests=[dict(url=url, bytes=reference['bytes'], sha256=reference['sha256'], status=200,
        content_length=reference['bytes'], complete_body_EOF_observed=True, redirected=False, retried=False)
        for url, reference in zip(replayed['metadata_requests'], saved)],
        socket_timeout_seconds=5, request_wall_check_seconds=60,
        blocking_DNS_and_IO_also_bounded_by_original_parent_lifecycle=True,
        automatic_retry=False, mirror_fallback=False, redirects_followed=False)
    c.require(transport == expected_transport and value['status'] == result['status']
              and value['metadata_requests'] == cursor[0] == len(names)
              and value['metadata_transfer_bytes'] == replayed['metadata_transfer_bytes'],
              'Exact complete observed transport receipts and summary')
    guard()
    return result


def save_collected(out, value, authenticated, plan_ref, admission_ref, guard):
    result = verify_collected(out, value, authenticated, guard)
    receipt = dict(schema='dory_glibc_source_metadata_receipt_v1', work_kind=WORK_KIND,
        source_result=value['result'], status=result['status'], metadata_requests=value['metadata_requests'],
        metadata_transfer_bytes=value['metadata_transfer_bytes'], metadata_input_binding=authenticated,
        plan=plan_ref, admission=admission_ref, complete_saved_metadata_replay=True,
        source_archive_contents_authenticated=False, actual_source_archive_download=False,
        exact_loader_source_semantics_established=False, unresolved_loader_edges=15,
        native_ABI_qualified=False, full_model_qualified=False, new_speedup_claim=False)
    guard()
    reference = c.save(out/'metadata-result.json', receipt, 8*c.MIB)
    c.require(storage_usage(out) <= c.INVENTORY['output_bytes'], 'Complete original allocated metadata stage budget')
    guard()
    return reference
