"""Exact supplemental scope for the existing descriptor-owned inventory stage.

Import and metadata preparation do not inspect archives or current binaries.
The runtime injects its original guard into collect; all prior results stay fixed.
"""
import copy
import os
from pathlib import Path
import stat
import minicpm5_sdk310_extra_members_contract_v1 as c
import minicpm5_sdk310_stdlib_extra_members_v1 as helper

WORK_KIND = 'minicpm5_sdk310_two_supplemental_stdlib_members_v1'
BASE = dict(path=str(c.Q/'minicpm5-sdk310-deb-replay-r1-20261112-plan.json'),
            sha256='d70f5134dbf391d5d6eab53883ba850ea657a6641d18fc87f191eab12c87a049')
HELPER_REVIEW = dict(path=str(c.Q/'minicpm5-sdk310-stdlib-extra-members-root-source-review-20261112-v1.json'),
                     sha256='40f681c7cf3375b0f721b51806b5ea478faf12fe8881b5095dfee40ecbfdf3e9')
NEW_NAMES = ('minicpm5_sdk310_stdlib_extra_members_v1.py',
             'minicpm5_sdk310_extra_members_contract_v1.py',
             'minicpm5_sdk310_extra_members_binding_v1.py',
             'run_minicpm5_sdk310_extra_members_v1.py',
             'prepare_minicpm5_sdk310_deb_plan_v4.py',
             'prepare_minicpm5_sdk310_extra_members_plan_v1.py')
MAX_EVENTS = 4
FALSE_FLAGS = ('download_performed', 'installed_anything', 'actual_process_exit_claimed',
               'accepted_native_authority_created', 'complete_native_dependency_closure_established',
               'native_ABI_qualified', 'full_model_qualified', 'new_speedup_claim')


def validate_sources_manifest(manifest, read=c.load):
    old = read(BASE)['source_sha256']; pins = manifest['source_sha256']
    c.require(manifest['schema'] == 'minicpm5_sdk310_extra_members_source_manifest_v1'
              and manifest['base_plan'] == BASE and manifest['helper_review'] == HELPER_REVIEW,
              'Exact frozen predecessor plan and accepted helper')
    c.require(len(old) == 106 and len(pins) == 112
              and set(pins) == set(old) | {str(c.Q/name) for name in NEW_NAMES}
              and all(pins.get(path) == sha for path,sha in old.items()),
              'All 106 prior source hashes retained, exactly six successor sources')
    accepted = read(HELPER_REVIEW)
    c.require(accepted['reviewer'] == '/root' and accepted['passed'] is True and accepted['blockers'] == []
              and all(pins.get(path) == sha for path,sha in helper.SOURCE_PINS.items())
              and pins[str(c.Q/'minicpm5_sdk310_stdlib_extra_members_v1.py')]
                  == accepted['source_sha256'][str(c.Q/'minicpm5_sdk310_stdlib_extra_members_v1.py')],
              'The actual accepted helper and original parser/transport are unchanged')
    c.require(manifest['budget'] == c.INVENTORY and manifest['mini_scientific_constraints'] == helper.CONSTRAINTS,
              'Original inventory and separate model diagnostic budgets')
    return manifest


def validate(plan, review, plan_ref):
    import dory_descriptor_inventory_runtime_v2 as runtime
    import minicpm5_sdk310_retired_lfm_r2_v2 as retirement
    runtime.validate(plan, review, plan_ref)
    retirement.validate(plan, review)
    c.require(plan.get('work_kind') == review.get('work_kind') == WORK_KIND
              and plan.get('extra_member_metadata') == review.get('extra_member_metadata') == helper.REFS,
              'Only the two fixed supplemental members and their complete prior metadata')
    c.require(plan.get('external_authorities') is None
              and plan.get('mini_scientific_constraints') == review.get('mini_scientific_constraints') == helper.CONSTRAINTS,
              'No native/model authority; preserve original scientific envelope')
    c.require(Path(plan['output_root']).name.startswith('dory-serial-inventory-minicpm5-sdk310-extra-members-'),
              'Fresh supplemental-only inventory output namespace')
    c.require(plan.get('supplemental_source_manifest') == review.get('supplemental_source_manifest')
              and plan.get('supplemental_source_acceptance') == review.get('supplemental_source_acceptance')
              and plan.get('metadata_source_count_limit') == review.get('metadata_source_count_limit') == 112,
              'Explicitly reviewed source manifest, acceptance and exact source count')
    manifest = validate_sources_manifest(c.load(plan['supplemental_source_manifest']))
    accepted = c.load(plan['supplemental_source_acceptance'])
    c.require(manifest['source_sha256'] == plan['source_sha256']
              and accepted['reviewer'] == '/root' and accepted['passed'] is True and accepted['blockers'] == []
              and accepted['manifest'] == plan['supplemental_source_manifest']
              and accepted['source_sha256'] == manifest['source_sha256'], 'Separate source acceptance binds exact plan sources')
    c.require(all(key not in plan for key in ('archive_inputs','archive_mode','replay_from')),
              'Supplemental result cannot be relabelled as original eight-package run')
    return plan


def authenticated(plan, guard):
    guard(); c.require(plan['extra_member_metadata'] == helper.REFS, 'Fixed supplemental metadata')
    request = helper.prepare(); guard()
    return request


def authenticate(plan, guard):
    request = authenticated(plan, guard)
    return dict(schema='minicpm5_sdk310_extra_members_inputs_authenticated_v1', work_kind=WORK_KIND,
                metadata_refs=copy.deepcopy(request['metadata_refs']), retained_archive=copy.deepcopy(request['archive']),
                prior_package_count=8, prior_target_count=12, supplemental_package_count=1, supplemental_target_count=2,
                mini_scientific_constraints=helper.CONSTRAINTS.copy(), package_content_authenticated=False,
                native_ABI_qualified=False, full_model_qualified=False, new_speedup_claim=False)


def validate_admission(value, authenticated_inputs):
    c.require(value.get('work_kind') == WORK_KIND and value.get('metadata_input_binding') == authenticated_inputs,
              'Actual original parent admission binds supplemental scope')


def validate_result(value, request):
    c.require(value['schema'] == 'minicpm5_sdk310_stdlib_extra_members_result_v1'
              and value['complete'] is True and value['metadata_refs'] == helper.REFS,
              'Complete independent supplemental result')
    c.require(value['prior_package_count'] == 8 and value['prior_target_count'] == 12
              and value['supplemental_package_count'] == 1 and value['supplemental_target_count'] == 2
              and value['original_match_count'] == value['extracted_files'] == 0
              and value['current_match_count'] == 2 and value['package_content_authenticated'] is True,
              'Original eight/twelve result remains distinct from two new comparisons')
    c.require(value['mini_scientific_constraints'] == helper.CONSTRAINTS and value['archive_stage_budget'] == c.INVENTORY
              and all(value[key] is False for key in FALSE_FLAGS), 'No resource or qualification promotion')
    helper.validate_inspection(value['inspection'], request)
    c.require(type(value['comparisons']) is list and len(value['comparisons']) == 2, 'Two ordered new comparisons only')
    for saved, target in zip(value['comparisons'], request['targets']):
        path=target['observed_path']; member=value['inspection']['tar']['targets'][path]; identity=saved['current_identity']
        wanted=dict(bytes=member['size'],sha256=member['sha256'],terminal=member['terminal'])
        c.require(saved['path'] == path and saved['module'] == target['module']
                  and saved['package'] == helper.PACKAGE and saved['version'] == helper.VERSION
                  and saved['member'] == wanted and saved['current'] == dict(path=path,bytes=member['size'],sha256=member['sha256'])
                  and saved['observation_source'] == target['observation_source'], 'Exact member/current/map provenance association')
        c.require(set(identity) == {'dev','ino','mode','nlink','size','mtime_ns','ctime_ns'}
                  and all(type(v) is int for v in identity.values())
                  and stat.S_ISREG(identity['mode']) and identity['nlink'] == 1 and identity['size'] == member['size'],
                  'Complete current regular single-link file identity')
        for rows in target['saved_mappings'].values():
            c.require(all(row['inode'] == identity['ino'] and row['device'] == [os.major(identity['dev']), os.minor(identity['dev'])]
                          for row in rows), 'Every saved map belongs to the current object')
        c.require(saved['current_matches'] is saved['saved_mapping_device_inode_matches'] is saved['new_observation_only'] is True
                  and saved['historical_full_file_content_proven'] is False and saved['original_matches'] is None,
                  'Current authenticated bytes do not invent historical complete content')
    return value


def validate_success(events, request, value, result_ref):
    inspection=value['inspection']; archive=inspection['archive']; decoder=inspection['decompressor']
    expected=[dict(event='supplemental_archive_replay_start',package=helper.PACKAGE,archive=request['archive']),
              dict(event='dpkg_deb_start',argv=decoder['argv']),
              dict(event='dpkg_deb_complete',package=helper.PACKAGE,receipt=decoder),
              dict(event='minicpm5_extra_members_complete',result=result_ref,prior_packages=8,prior_targets=12,
                   supplemental_packages=1,supplemental_targets=2,current_matches=2)]
    c.require(events == expected and len(events) == MAX_EVENTS, 'Exactly four complete ordered supplemental events')
    c.require(archive == request['archive'] and decoder['returncode'] == 0
              and decoder['argv'] == ['/usr/bin/dpkg-deb','--fsys-tarfile',archive['path']]
              and decoder['stdout_bytes'] == inspection['tar']['stream_bytes'], 'Exact owned successful decompressor and full EOF')
    return dict(events=4, supplemental_targets=2, ordered=True)


def run(plan, guard, emit, out):
    request=authenticated(plan, guard)
    emit(dict(event='supplemental_archive_replay_start',package=helper.PACKAGE,archive=request['archive']))
    value=helper.collect(guard, emit)
    validate_result(value, request); guard()
    ref=c.save(Path(out)/'extra-members-result.json',value,helper.METADATA_CAP)
    guard();emit(dict(event='minicpm5_extra_members_complete',result=ref,prior_packages=8,prior_targets=12,
                      supplemental_packages=1,supplemental_targets=2,current_matches=2))
    return dict(result=ref)


def storage_usage(out):
    from minicpm5_sdk310_deb_input_binding_v1 import storage_usage as existing
    return existing(out)


def save_collected(out,value,authenticated_inputs,plan_ref,admission_ref,guard,*,plan):
    c.require(value['result']['path'] == str(out/'extra-members-result.json'), 'Owned independent supplemental result')
    request=authenticated(plan,guard)
    validate_result(c.load(value['result'],helper.METADATA_CAP),request)
    guard();ref=c.save(out/'metadata-result.json',dict(schema='minicpm5_sdk310_extra_members_receipt_v1',work_kind=WORK_KIND,
         sources_result=value['result'],metadata_input_binding=authenticated_inputs,plan=plan_ref,admission=admission_ref,
         package_content_authenticated=True,native_ABI_qualified=False,full_model_qualified=False,new_speedup_claim=False))
    c.require(storage_usage(out) <= c.INVENTORY['output_bytes'], 'Original total output cap')
    return ref
