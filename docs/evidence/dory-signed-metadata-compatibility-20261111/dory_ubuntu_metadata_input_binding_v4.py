"""Fixed signed-metadata work binding; original serial contract is unchanged."""
import os
from pathlib import Path
import stat
import dory_serial_stage_contract_v2 as c
import dory_descriptor_inventory_runtime_v2 as runtime
import dory_ubuntu_package_sources_v3 as discovery

WORK_KIND = 'dory_ubuntu_signed_metadata_only_v1'
AUTHORITY = dict(path=str(c.Q/'dory-ubuntu-archive-key-authority-20261111-v1.json'),
                 sha256='b729344d76014db061a946e2568e3f00219e759c4fa2cf7d535111e01742fad0')
TRUST = dict(keyring_sha256='1a4dd63e5c76728960a2edddae22e2e0fc53df8e8b87806deb971030ac704eb0',
             primary_fingerprints=['F6ECB3762474EDA9D21B7022871920D1991BC93C',
                                   '790BC7277767219C42C86F933B4FE6ACC0B21F32'], authority_ref=AUTHORITY)
INPUTS = {key: dict(path=str(c.Q/name), sha256=pin) for key, (name, pin) in discovery.INPUT_PINS.items()}
INPUTS['key_authority'] = AUTHORITY
SOURCE_PINS = {
    str(c.Q/'dory_ubuntu_package_sources_v1.py'): '7a9f1fe5754fb7b29dedad6d2532779f7af2252b2da0053683652b7569c608bd',
    str(c.Q/'dory_ubuntu_metadata_adapter_v2.py'): '7048014d89d8c725e5cdc7b9f1ebc1f7a15aacd769c5a7951b7640b6dac891eb',
    str(c.Q/'dory_ubuntu_metadata_adapter_v3.py'): '858464636f7b454e9d21f5bd3f28b9893a75574b0ff60bfff03b7ebb5d1f9abb',
    str(c.Q/'dory_ubuntu_gpgv_status_compliance_v1.py'): 'ed4b3d6cb5108c5a09bb997d65df155057d8ef5fad085a9f76aaaf3e05719ed5',
    str(c.Q/'dory_ubuntu_package_sources_v2.py'): 'eb3bdcd4a532bec4905c12195813a95514a27ede39145e7e944be349a152688d',
    str(c.Q/'dory_ubuntu_metadata_adapter_v4.py'): 'de949c4cceb8df8872ac68dddcc864e1adb9c1d9c86f8144fec3547f3001fd89',
    str(c.Q/'dory_ubuntu_package_sources_v3.py'): '7f343eb3e355a1f6cf889e5bfe2fb34811cf8dd687bd6debf2a89bf254ab89f7',
    str(c.Q/'dory_ubuntu_metadata_adapter_v5.py'): '23689da504c05ed243a989af3cea9cf756e44ca9c8e5704b431bf2457bbbf59d',
}


def validate(plan, review, plan_ref):
    runtime.validate(plan, review, plan_ref)
    c.require(plan.get('work_kind') == review.get('work_kind') == WORK_KIND
              and plan.get('metadata_inputs') == review.get('metadata_inputs') == INPUTS
              and plan.get('ubuntu_trust') == review.get('ubuntu_trust') == TRUST,
              'Exact signed Ubuntu metadata task, input refs and key authority')
    c.require(plan.get('external_authorities') is None, 'No package payload or native authority admission')
    for name, pin in SOURCE_PINS.items():
        c.require(plan['source_sha256'].get(name) == pin, 'Frozen actual metadata source pins')
    return plan


def authenticate(guard):
    raw_inputs = {}
    for key, ref in INPUTS.items():
        guard()
        observed, raw = c.reference(ref['path'], c.MIB)
        c.require(observed['sha256'] == ref['sha256'], 'Complete metadata input binding')
        if key != 'key_authority':
            raw_inputs[key] = raw
    discovery.bind_inputs(raw_inputs)
    authority = c.load(AUTHORITY, c.MIB)
    c.require(authority['schema'] == 'dory_ubuntu_archive_key_authority_v1'
              and authority['independent_authority_verified'] is True
              and authority['keyring']['sha256'] == TRUST['keyring_sha256']
              and authority['primary_fingerprints'] == TRUST['primary_fingerprints'], 'Exact saved authority')
    guard()
    return dict(schema='dory_ubuntu_authenticated_metadata_inputs_v1', work_kind=WORK_KIND,
                inputs=INPUTS, ubuntu_trust=TRUST, fixed_packages=6, fixed_targets=12,
                original_payload_refs=11, maximum_metadata_requests=12, actual_keyring_reobserved=False,
                package_content_authenticated=False, loader_selection_established=False, native_ABI_qualified=False)


def validate_admission(value, authenticated):
    c.require(value.get('work_kind') == WORK_KIND and value.get('metadata_input_binding') == authenticated,
              'Parent admission binds exact signed metadata scope')


def storage_usage(out):
    """Count all entries and directories, including unexpected verifier outputs."""
    root = Path(out)
    pending, total, count = [root], 0, 0
    while pending:
        path = pending.pop()
        info = path.lstat()
        count += 1
        c.require(count <= 2000 and not stat.S_ISLNK(info.st_mode), 'Bounded complete output tree without links')
        c.require(stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode), 'Only regular output and directories')
        total += max(info.st_size, info.st_blocks * 512)
        if stat.S_ISDIR(info.st_mode):
            pending.extend(path.iterdir())
    return total


def save_final_result(out, result, budget):
    ref = c.save(out/'result.json', result)
    c.require(storage_usage(out) <= budget['output_bytes'], 'Final result included in complete allocated budget')
    return ref


def save_collected(out, value, authenticated, plan_ref, admission_ref, guard):
    c.require(value['result']['path'] == str(out/'signed-metadata'/'sources-result.json'), 'Owned sources result')
    result = c.load(value['result'], 8*c.MIB)
    c.require(result['schema'] == 'dory_ubuntu_exact_package_sources_v1'
              and result['status'] == value['status'] and result['status'] in ('exact_addresses_ready', 'exact_versions_missing')
              and result['trust_anchor'] == TRUST and len(result['packages']) == value['packages']
              and result['adapter']['requests'] == 12 and result['adapter']['signature_invocations'] == 3
              and result['package_content_authenticated'] is False and result['actual_archive_download'] is False
              and result['unresolved_closed'] == 0 and result['native_ABI_qualified'] is False,
              'Complete metadata-only source receipt, never package bytes')
    guard()
    receipt = dict(schema='dory_ubuntu_signed_metadata_receipt_v1', work_kind=WORK_KIND,
                   sources_result=value['result'], status=value['status'], packages=value['packages'],
                   metadata_input_binding=authenticated, plan=plan_ref, admission=admission_ref,
                   package_content_authenticated=False, loader_selection_established=False,
                   unresolved_closed=0, native_ABI_qualified=False, full_model_qualified=False, new_speedup_claim=False)
    ref = c.save(out/'metadata-result.json', receipt, 8*c.MIB)
    c.require(storage_usage(out) <= c.INVENTORY['output_bytes'], 'Complete allocated metadata stage output')
    guard()
    return ref
