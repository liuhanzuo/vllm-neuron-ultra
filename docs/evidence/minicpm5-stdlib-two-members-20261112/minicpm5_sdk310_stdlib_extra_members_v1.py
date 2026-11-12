"""Supplement exactly two SDK310 members from one already completed signed deb.

Import/prepare perform source/JSON work only. collect() is a future admitted
root-stage leaf, not a launcher: the caller must retain the existing descriptor,
wall/resource/output guards, source closure and actual child/outer exit audit.
No download, install, extraction, native loading, or history framework is added.
"""
import copy
import os
from pathlib import Path
import stat
import dory_serial_stage_contract_v2 as c

REFS = {'mapping': {'path': '/home/ubuntu/experiments/trn1-latest-models-20261031/research/minicpm5-sdk310-bootstrap-two-member-next-inputs-20261112-v1.json',
             'bytes': 15840,
             'sha256': 'fb62e0b9149a441b5d244cff21487276995ad64d763aaddfe8f3dc41175f0839'},
 'bootstrap_exit': {'path': '/home/ubuntu/experiments/trn1-latest-models-20261031/research/minicpm5-bootstrap-observation-r1-root-actual-exit-20261112-v1.json',
                    'bytes': 125709,
                    'sha256': '19e460fa40b3aaf8e73040c39464ec67419e85841441464e9f6f263b05808496'},
 'observation': {'path': '/mnt/nvme/dory-serial-inventory-minicpm5-sdk310-bootstrap-r1-20261112/sdk310/observation.json',
                 'bytes': 59729,
                 'sha256': 'cc013649c7a61f22f6dbab59c746be59a375e9d2507aadcd786168daab27dc4d'},
 'completed': {'path': '/home/ubuntu/experiments/trn1-latest-models-20261031/research/minicpm5-sdk310-eight-archive-completed-root-summary-20261112-v1.json',
               'bytes': 1641,
               'sha256': '6395f1f392dcd2e253fbbc9c91b2cc42f974ac1a4b3b1f2a5bfaa6946af3b8a3'},
 'download_exit': {'path': '/home/ubuntu/experiments/trn1-latest-models-20261031/research/minicpm5-sdk310-deb-download-r4-actual-exit-20261112-v1.json',
                   'bytes': 23208,
                   'sha256': '6dfd0585eca06b423f99b7773af69909634a25445068ed47344d8988b30f99ae'},
 'replay_exit': {'path': '/home/ubuntu/experiments/trn1-latest-models-20261031/research/minicpm5-sdk310-deb-replay-r1-actual-exit-20261112-v1.json',
                 'bytes': 21360,
                 'sha256': '9182e315ddad211c2c4a995aceb8da570f4b0be3e09a02d07e2af0df6ea52787'},
 'recorder_exit': {'path': '/home/ubuntu/experiments/trn1-latest-models-20261031/research/minicpm5-sdk310-deb-replay-r1-recorder-actual-exit-20261112-v1.json',
                   'bytes': 1942,
                   'sha256': '425867257bd0fd084117430f67c81de61554a5cef5711e303b7d375e328c5a8f'},
 'download': {'path': '/mnt/nvme/dory-serial-inventory-minicpm5-sdk310-download-r4-20261112/packages/sources-result.json',
              'bytes': 37467,
              'sha256': '729a494ce859051191712f00a0ce3fd119745cb404564c29ca0a96a8793f6bf9'},
 'replay': {'path': '/mnt/nvme/dory-serial-inventory-minicpm5-sdk310-replay-r1-20261112/packages/sources-result.json',
            'bytes': 37645,
            'sha256': '2e8793f5ad3ff8dae91ccadbde05ce65db53ef02b17583661e558b283b38e6a3'},
 'prior_inputs': {'path': '/home/ubuntu/experiments/trn1-latest-models-20261031/research/minicpm5-sdk310-archive-member-next-inputs-20261112-v1.json',
                  'bytes': 55789,
                  'sha256': 'e91bca576461d9d1d4b206e7f8b508c20a103a3231bef9b29535e97d3e8da266'},
 'tool': {'path': '/home/ubuntu/experiments/trn1-latest-models-20261031/research/dory-dpkg-deb-tool-authority-20261111-v1.json',
          'bytes': 1750,
          'sha256': 'a806f318a23e8a8d1bb5f9f408dbc24eaad3e0625be38ab3e8ebfdccc434b174'},
 'archive_authority': {'path': '/home/ubuntu/experiments/trn1-latest-models-20261031/research/dory-ubuntu-archive-key-authority-20261111-v1.json',
                       'bytes': 3842,
                       'sha256': 'b729344d76014db061a946e2568e3f00219e759c4fa2cf7d535111e01742fad0'}}
SOURCE_PINS = {'/home/ubuntu/experiments/trn1-latest-models-20261031/research/dory_deb_transport_v3.py': '3a9d382507bcdb167fb5989be70e9edaba16e070f9894a0975b27edd18bc617c',
 '/home/ubuntu/experiments/trn1-latest-models-20261031/research/dory_deb_tar_members_v2.py': 'd43b3938ee61e13239e09d687e7b6299383da5baa527c54192d25b80d78225c6',
 '/home/ubuntu/experiments/trn1-latest-models-20261031/research/dory_ubuntu_metadata_adapter_v6.py': 'de6253582e1077c0fb315ce0ebd10b9e22b1cf7484c02624c5bb7248e4201515',
 '/home/ubuntu/experiments/trn1-latest-models-20261031/research/dory_serial_stage_contract_v2.py': 'bb3ed6e61632574f32a5a98b7cee8183394b416225193f9e8f080054f98ecc34'}
PATHS = tuple('/usr/lib/python3.10/lib-dynload/' + name + '.cpython-310-x86_64-linux-gnu.so'
              for name in ('_contextvars', 'resource'))
PACKAGE = 'libpython3.10-stdlib'
VERSION = '3.10.12-1~22.04.18'
CONSTRAINTS = dict(native_checks=9, scientific_descriptor_rows=1284, absolute_wall_seconds=900,
                   address_space_bytes=2**31, single_file_bytes=2**21)
METADATA_CAP = 256 * 2**10


def same(a, b):
    return all(a[key] == b[key] for key in ('path', 'sha256')) and a.get('bytes', b.get('bytes')) == b.get('bytes', a.get('bytes'))


def prepare(load=c.load):
    """Return metadata request only; each default input is a fixed whole JSON pin."""
    data = {key: load(ref, METADATA_CAP) for key, ref in REFS.items()}
    mapping, n, prior = data['mapping'], data['prior_inputs'], data['replay']
    completed, boot = data['completed'], data['bootstrap_exit']
    c.require(completed['passed'] is True and completed['blockers'] == []
              and completed['full_saved_package_results_identical'] is True
              and completed['packages'] == 8 and completed['selected_members'] == completed['current_matches'] == 12,
              'Accepted complete eight-package/twelve-target provenance')
    for key in ('download', 'replay'):
        ended = data[key + '_exit']
        c.require(ended['actual_tool']['exit_code'] == 0 and ended['stage_success_observed'] is True
                  and ended['owners_gone'] is ended['child_group_empty'] is True
                  and same(ended['result'], REFS[key]) and same(completed[key + '_exit'], REFS[key + '_exit'])
                  and same(completed[key + '_result'], REFS[key]), 'Prior producer and replay really completed')
    recorder = data['recorder_exit']
    c.require(recorder['actual_tool']['exit_code'] == 0 and recorder['owner_gone'] is True
              and same(recorder['result'], REFS['replay_exit'])
              and same(completed['replay_recorder_exit'], REFS['recorder_exit']), 'Completed replay recorder')
    c.require(boot['actual_tool']['exit_code'] == boot['actual_exit_code'] == 0
              and boot['actual_tool_exit_observed'] is boot['stage_success_observed'] is True
              and boot['physically_drained'] is boot['owners_gone'] is boot['child_group_empty'] is True
              and boot['full_six_raw_maps_replayed'] is True, 'Pinned actual bootstrap completion, no live operations here')
    c.require(mapping['schema'] == 'minicpm5_sdk310_bootstrap_two_member_next_inputs_v1'
              and mapping['scientific_constraints'] == n['original_mini_diagnostic_constraints'] == CONSTRAINTS
              and mapping['runtime_admitted'] is False and mapping['no_actual_plan'] is True
              and mapping['preserved_prior_package_count'] == 8 and mapping['preserved_prior_target_count'] == 12
              and same(mapping['prior_complete_replay'], REFS['replay'])
              and same(mapping['prior_signed_next_inputs'], REFS['prior_inputs']), 'Exact supplemental metadata scope')
    c.require(prior['complete'] is True and prior['mode'] == 'replay'
              and prior['target_count'] == prior['current_match_count'] == 12 and len(prior['packages']) == 8
              and prior['packages'] == data['download']['packages'] and prior['original_match_count'] == 0
              and same(prior['replay_from'], REFS['download']), 'Original eight/twelve records preserved exactly')
    rows = [row for row in n['packages'] if row['package'] == PACKAGE]
    packages = [row for row in prior['packages'] if row['package'] == PACKAGE]
    c.require(len(rows) == len(packages) == 1, 'One exact stdlib provider')
    row, old = copy.deepcopy(rows[0]), packages[0]
    archive = old['inspection']['archive']; proposed = mapping['proposed_archive_reuse']
    c.require(row['version'] == old['version'] == proposed['version'] == VERSION and row['architecture'] == 'amd64'
              and all(archive[key] == row[key] for key in ('bytes', 'sha256'))
              and proposed['archive'] == archive and proposed['signed_sources'] == row['sources']
              and proposed['prior_tar_stream_bytes'] == old['inspection']['tar']['stream_bytes']
              and proposed['prior_tar_stream_sha256'] == old['inspection']['tar']['stream_sha256'],
              'Original exact signed archive tuple and complete tar provenance')
    c.require(same(mapping['tool_authority'], REFS['tool']) and same(mapping['trust_authority'], REFS['archive_authority'])
              and same(n['dpkg_deb_tool_authority'], REFS['tool']) and same(n['trust_authority'], REFS['archive_authority'])
              and data['tool']['reviewer'] == '/root' and data['tool']['tool']['path'] == '/usr/bin/dpkg-deb'
              and data['archive_authority']['independent_authority_verified'] is True,
              'Retain completed package trust and decompressor authorities')
    observation = data['observation']
    c.require(observation['complete'] is True and observation['role'] == 'sdk310'
              and observation['map_summaries'] == boot['selected_providers_and_gaps']['sdk310']['map_summaries'],
              'Two new members derive from exact independently audited SDK observation')
    c.require([t['observed_path'] for t in mapping['targets']] == list(PATHS), 'Only two exact ordered targets')
    for target, path in zip(mapping['targets'], PATHS):
        c.require(target['proposed_package'] == PACKAGE and target['proposed_archive_member'] == path.lstrip('/')
                  and same(target['observation_source'], REFS['observation'])
                  and target['archive_member_bytes'] is target['archive_member_sha256'] is target['current_full_file_identity'] is None
                  and target['member_presence_authenticated'] is target['current_content_read'] is False,
                  'No invented new member bytes or full current identity')
        for phase, maps in target['saved_mappings'].items():
            expected = [gap['mapping'] for gap in observation['map_summaries'][phase]['gaps']
                        if gap['reason'] == 'missing_saved_content_origin' and gap['mapping']['path'] == path]
            c.require(maps == expected and len(maps) == 5, 'Exact saved mapping segments, not full-file hashes')
        c.require(set(target['saved_mappings']) == {'pre_ctypes', 'post_ctypes'}, 'Both original snapshots retained')
    row['targets'] += [dict(observed_path=path, archive_member=path.lstrip('/')) for path in PATHS]
    row['saved_package_names'] += list(PATHS)
    c.require(len(row['targets']) == 6 and len(row['saved_package_names']) == 6,
              'Original four members plus exactly two supplemental members')
    return dict(schema='minicpm5_sdk310_stdlib_extra_members_request_v1', metadata_refs=copy.deepcopy(REFS),
                package=row, old_package=copy.deepcopy(old), archive=copy.deepcopy(archive),
                tool=copy.deepcopy(data['tool']['tool']), targets=copy.deepcopy(mapping['targets']),
                archive_stage_budget=copy.deepcopy(c.INVENTORY), mini_scientific_constraints=CONSTRAINTS.copy(),
                source_sha256=SOURCE_PINS.copy(), runtime_admitted=False)


def validate_inspection(value, request):
    old = request['old_package']['inspection']; tar = value['tar']
    c.require(value['archive'] == request['archive'] and value['archive_identity'] == old['archive_identity']
              and value['decompressor'] == old['decompressor'], 'Same retained complete archive identity and decompressor receipt')
    c.require(tar['complete'] is True and {k:v for k,v in tar.items() if k not in ('targets', 'aliases')}
              == {k:v for k,v in old['tar'].items() if k not in ('targets', 'aliases')},
              'Same entire authenticated tar stream, headers, sizes, counts and nonpromotion flags')
    c.require(set(tar['targets']) == set(old['tar']['targets']) | set(PATHS)
              and set(tar['aliases']) == set(old['tar']['aliases']) | {p.lstrip('/') for p in PATHS},
              'Exactly four retained and two supplemental members/aliases')
    for key in ('targets', 'aliases'):
        c.require(all(tar[key][path] == member for path,member in old['tar'][key].items()),
                  'All four prior member/alias results replay identically')
    for path in PATHS:
        member = tar['targets'][path]; name = path.lstrip('/')
        c.require(member['requested'] == member['terminal'] == name and member['link_hops'] == 0
                  and 0 < member['size'] <= 64*c.MIB and type(member['sha256']) is str
                  and len(member['sha256']) == 64 and set(member['sha256']) <= set('0123456789abcdef')
                  and member['chain'] == [dict(name=name, type='file', size=member['size'], sha256=member['sha256'])]
                  and tar['aliases'][name] == dict(member, target=path), 'Exact regular file supplemental member and full hash')
    return value


def collect(guard, emit):
    """Future owned serial leaf only; no download/install or self-exit claim.

    guard must be the existing admitted lifecycle guard, not a caller success
    flag. Root must bind this helper and inherited dependencies in its fresh
    plan/review after the current Qwen task has actually exited and drained.
    This function returns metadata; the owner saves it inside its existing cap.
    """
    guard(); c.verify_sources(SOURCE_PINS, guard)
    request = prepare()
    import dory_deb_transport_v3 as io
    guard(); inspection = io.inspect(request['package'], Path(request['archive']['path']), request['tool'], guard, emit)
    validate_inspection(inspection, request)
    comparisons = []
    for target in request['targets']:
        path = target['observed_path']; member = inspection['tar']['targets'][path]
        current, identity = io.file_ref(path, guard, dict(bytes=member['size'], sha256=member['sha256']))
        c.require(stat.S_ISREG(identity['mode']) and identity['nlink'] == 1 and current['path'] == path
                  and identity['size'] == current['bytes'] == member['size'] and current['sha256'] == member['sha256'],
                  'Complete current file equals exact authenticated member')
        for rows in target['saved_mappings'].values():
            c.require(all(row['inode'] == identity['ino'] and row['device'] == [os.major(identity['dev']), os.minor(identity['dev'])]
                          for row in rows), 'Current device/inode equals prior mapped object; historical bytes remain unproved')
        comparisons.append(dict(path=path, module=target['module'], package=PACKAGE, version=VERSION,
            member=dict(bytes=member['size'], sha256=member['sha256'], terminal=member['terminal']),
            current=current, current_identity=identity, current_matches=True,
            saved_mapping_device_inode_matches=True, observation_source=target['observation_source'],
            historical_full_file_content_proven=False, original_matches=None, new_observation_only=True))
    guard()
    return dict(schema='minicpm5_sdk310_stdlib_extra_members_result_v1', complete=True, metadata_refs=request['metadata_refs'],
        prior_package_count=8, prior_target_count=12, supplemental_package_count=1, supplemental_target_count=2,
        mini_scientific_constraints=CONSTRAINTS.copy(), archive_stage_budget=copy.deepcopy(c.INVENTORY),
        inspection=inspection, comparisons=comparisons, package_content_authenticated=True,
        original_match_count=0, current_match_count=2, download_performed=False, installed_anything=False,
        extracted_files=0, actual_process_exit_claimed=False, accepted_native_authority_created=False,
        complete_native_dependency_closure_established=False, native_ABI_qualified=False,
        full_model_qualified=False, new_speedup_claim=False)
