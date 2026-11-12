"""Fixed Mini SDK310 eight-package preparation; no IO at import.

This authenticates package origins only. It does not establish the SDK native
closure, load libraries, or replace its nine checks / 1,284 scientific rows.
"""
import datetime as dt
import hashlib
import os
from pathlib import Path
import stat
import dory_serial_stage_contract_v2 as c
import dory_descriptor_inventory_runtime_v2 as runtime
import dory_ubuntu_package_sources_v4 as metadata
from dory_ubuntu_metadata_input_binding_v5 import storage_usage, save_final_result

WORK_KIND = 'minicpm5_sdk310_eight_deb_member_verification_v1'
NEXT = dict(path=str(c.Q/'minicpm5-sdk310-archive-member-next-inputs-20261112-v1.json'), sha256='e91bca576461d9d1d4b206e7f8b508c20a103a3231bef9b29535e97d3e8da266')
TOOL = dict(path=str(c.Q/'dory-dpkg-deb-tool-authority-20261111-v1.json'), sha256='a806f318a23e8a8d1bb5f9f408dbc24eaad3e0625be38ab3e8ebfdccc434b174')
TRUST = dict(next_inputs=NEXT, tool_authority=TOOL)
RESULT_SCHEMA = 'minicpm5_sdk310_deb_members_result_v1'
CONSTRAINTS = dict(native_checks=9, scientific_descriptor_rows=1284, absolute_wall_seconds=900, address_space_bytes=2147483648, single_file_bytes=2097152)
NEW_NAMES = ('minicpm5_sdk310_deb_input_binding_v1.py', 'minicpm5_sdk310_deb_members_adapter_v1.py', 'minicpm5_sdk310_deb_events_v1.py', 'run_minicpm5_sdk310_deb_members_root_v1.py', 'minicpm5_sdk310_deb_members_worker_v1.py', 'record_minicpm5_sdk310_deb_stage_exit_v1.py')
SOURCE_PINS = {'/home/ubuntu/experiments/trn1-latest-models-20261031/research/dory_deb_transport_v3.py': '3a9d382507bcdb167fb5989be70e9edaba16e070f9894a0975b27edd18bc617c', '/home/ubuntu/experiments/trn1-latest-models-20261031/research/dory_deb_tar_members_v2.py': 'd43b3938ee61e13239e09d687e7b6299383da5baa527c54192d25b80d78225c6', '/home/ubuntu/experiments/trn1-latest-models-20261031/research/dory_deb_input_binding_v3.py': '1e65cb5273d1e702b5b57dcf4a5be6c33e4d3525fae2d496c6c37f1055482f75', '/home/ubuntu/experiments/trn1-latest-models-20261031/research/dory_deb_members_adapter_v3.py': 'ab12722776eb8a77fcd05fec84aa333b4969be6c34885b64e573f1622fa31448', '/home/ubuntu/experiments/trn1-latest-models-20261031/research/run_dory_deb_members_root_v3.py': '8f705c24e0064970cb67619235c1a8f89e9d95df1d16436cd3e28ec3f095f9d8', '/home/ubuntu/experiments/trn1-latest-models-20261031/research/dory_deb_members_worker_v3.py': '1e2ad24d79b6d14351acef80c3c28ae8f094a4f2b9c1aabed219078182fa1dc2', '/home/ubuntu/experiments/trn1-latest-models-20261031/research/record_dory_deb_stage_exit_v3.py': '239da644b42e5ff991834f9d77c2c4134297912932352ef7bf2c30e9ed350425', '/home/ubuntu/experiments/trn1-latest-models-20261031/research/dory_ubuntu_package_sources_v4.py': '69c8a9c9d34df8a126fc67ec7fe621a9db79bfb4a5ca3352a684dc746e46fc43', '/home/ubuntu/experiments/trn1-latest-models-20261031/research/dory_ubuntu_metadata_adapter_v6.py': 'de6253582e1077c0fb315ce0ebd10b9e22b1cf7484c02624c5bb7248e4201515', '/home/ubuntu/experiments/trn1-latest-models-20261031/research/dory_descriptor_inventory_runtime_v2.py': '758b075dc7d1dfa10b12e5b4dee445b11691dcfebf7996853fb4b6d4479df556', '/home/ubuntu/experiments/trn1-latest-models-20261031/research/dory_serial_stage_contract_v2.py': 'bb3ed6e61632574f32a5a98b7cee8183394b416225193f9e8f080054f98ecc34', '/home/ubuntu/experiments/trn1-latest-models-20261031/research/run_dory_serial_stage_root_v5.py': '3589c286b78e7196105fa9d76c45a14c6a47535fb48b6a9604ffdb11c2c3d2f7', '/home/ubuntu/experiments/trn1-latest-models-20261031/research/dory_serial_stage_journal_v2.py': '4003a0f43bf95245f5b8ffc79bbd60c649680a3aa305ec6d33514157e39f4550'}
PACKAGE_VERSIONS = {'python3.10-minimal': '3.10.12-1~22.04.18', 'libpython3.10-stdlib': '3.10.12-1~22.04.18', 'libffi8': '3.4.2-4', 'libpython3.10-minimal': '3.10.12-1~22.04.18', 'libssl3': '3.0.2-0ubuntu1.30', 'libexpat1': '2.4.7-1ubuntu0.9', 'zlib1g': '1:1.2.11.dfsg-2ubuntu9.2', 'libcrypt1': '1:4.4.27-1'}
TARGET_SELECTION = {'python3.10-minimal': [('/usr/bin/python3.10', 'usr/bin/python3.10', 'required')], 'libpython3.10-stdlib': [('/usr/lib/python3.10/lib-dynload/_ctypes.cpython-310-x86_64-linux-gnu.so', 'usr/lib/python3.10/lib-dynload/_ctypes.cpython-310-x86_64-linux-gnu.so', 'required'), ('/usr/lib/python3.10/ctypes/__init__.py', 'usr/lib/python3.10/ctypes/__init__.py', 'required'), ('/usr/lib/python3.10/ctypes/_endian.py', 'usr/lib/python3.10/ctypes/_endian.py', 'required'), ('/usr/lib/python3.10/lib-dynload/_json.cpython-310-x86_64-linux-gnu.so', 'usr/lib/python3.10/lib-dynload/_json.cpython-310-x86_64-linux-gnu.so', 'candidate')], 'libffi8': [('/usr/lib/x86_64-linux-gnu/libffi.so.8.1.0', 'usr/lib/x86_64-linux-gnu/libffi.so.8.1.0', 'required')], 'libpython3.10-minimal': [('/usr/lib/python3.10/lib-dynload/_hashlib.cpython-310-x86_64-linux-gnu.so', 'usr/lib/python3.10/lib-dynload/_hashlib.cpython-310-x86_64-linux-gnu.so', 'candidate')], 'libssl3': [('/usr/lib/x86_64-linux-gnu/libcrypto.so.3', 'usr/lib/x86_64-linux-gnu/libcrypto.so.3', 'candidate'), ('/usr/lib/x86_64-linux-gnu/libssl.so.3', 'usr/lib/x86_64-linux-gnu/libssl.so.3', 'candidate')], 'libexpat1': [('/usr/lib/x86_64-linux-gnu/libexpat.so.1.8.7', 'lib/x86_64-linux-gnu/libexpat.so.1.8.7', 'candidate')], 'zlib1g': [('/usr/lib/x86_64-linux-gnu/libz.so.1.2.11', 'lib/x86_64-linux-gnu/libz.so.1.2.11', 'candidate')], 'libcrypt1': [('/usr/lib/x86_64-linux-gnu/libcrypt.so.1.1.0', 'lib/x86_64-linux-gnu/libcrypt.so.1.1.0', 'candidate')]}


def validate(plan, review, plan_ref):
    runtime.validate(plan, review, plan_ref)
    c.require(plan.get('work_kind') == review.get('work_kind') == WORK_KIND and plan.get('archive_inputs') == review.get('archive_inputs') == TRUST, 'Exact Mini eight-package task/input binding')
    c.require(plan.get('archive_mode') == review.get('archive_mode') and plan['archive_mode'] in ('download', 'replay') and plan.get('replay_from') == review.get('replay_from') and (plan.get('replay_from') is None) == (plan['archive_mode'] == 'download'), 'Exact reviewed download or independent saved replay')
    c.require(plan.get('external_authorities') is None, 'No native/model authority')
    c.require(plan.get('mini_scientific_constraints') == review.get('mini_scientific_constraints') == CONSTRAINTS, 'Original Mini scientific envelope retained separately')
    c.require(Path(plan['output_root']).name.startswith('dory-serial-inventory-minicpm5-sdk310-'), 'Mini outputs in retained inventory namespace')
    for name, pin in SOURCE_PINS.items():
        c.require(plan['source_sha256'].get(name) == pin, 'Exact reused source pin')
    for name in NEW_NAMES:
        c.require(str(c.Q/name) in plan['source_sha256'], 'Every Mini successor source independently reviewed')
    return plan


def validate_manifest(n):
    c.require(n['schema'] == 'minicpm5_sdk310_archive_member_preparation_next_inputs_v1' and n['offline_only'] is True and n['original_mini_diagnostic_constraints'] == CONSTRAINTS, 'Frozen offline input and scientific constraints')
    rows = n['packages']
    c.require([r['package'] for r in rows] == list(PACKAGE_VERSIONS), 'Eight unique ordered packages')
    c.require(n['counts'] == dict(packages=8, required_packages=3, candidate_packages=5, selected_targets=12, required_targets=5, candidate_targets=7, original_full_payload_hashes=0, all_eight_download_events=41, all_eight_replay_events=33), 'Exact eight-package counts')
    c.require(sum(r['bytes'] for r in rows) == n['archive_total_bytes'] == 7110864 and sum(r['bytes'] for r in rows[:3]) == n['required_package_archive_bytes'] == 4150014 and n['candidate_package_archive_bytes'] == 2960850, 'Exact aggregate signed sizes')
    for index, row in enumerate(rows):
        c.require(row['version'] == PACKAGE_VERSIONS[row['package']] and row['architecture'] == 'amd64' and row['necessity'] == ('required' if index < 3 else 'candidate'), 'Exact package version/architecture/role')
        c.require(row['actual_loaded'] is None and row['actual_loader_mapping_observed_here'] is False and row['package_content_authenticated'] is False, 'No inferred actual package mapping')
        c.require([(t['observed_path'], t['archive_member'], t['necessity']) for t in row['targets']] == TARGET_SELECTION[row['package']], 'Exact selected member identities and roles')
        c.require(row['saved_package_names'] == ['/'+t['archive_member'] for t in row['targets']], 'Canonical target members only; no alias promotion')
        for target in row['targets']:
            c.require(target['original_observed_reference'] is None and target['original_full_payload_hash_present'] is False and target['actual_loaded'] is None and target['actual_loader_mapping_observed_here'] is False, 'Zero original hashes and unknown member mappings')
        urls = sorted({s['url'] for s in row['sources']})
        c.require(row['url'] == urls[0] == 'https://archive.ubuntu.com/ubuntu/'+row['filename'] and row['authenticated_alternates_not_automatic_retries'] == urls[1:], 'Exact selected signed pool URL; no automatic alternates')
    return n


def read_text(ref, guard, allowed, cap=128*c.MIB):
    """Only exact pinned plaintext metadata; never follows native/archive paths."""
    path = Path(ref['path'])
    c.require(str(path) in allowed and allowed[str(path)] == ref and path.is_absolute() and path.resolve() == path, 'Exact finite plaintext input')
    guard(); fd = os.open(path, os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        c.require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and 0 < before.st_size == ref['bytes'] <= cap, 'Bounded whole plaintext metadata')
        chunks = []; size = 0; digest = hashlib.sha256()
        while size < before.st_size:
            guard(); part = os.read(fd, min(65536, before.st_size-size))
            c.require(part, 'Complete plaintext read'); chunks.append(part); size += len(part); digest.update(part)
        c.require(os.read(fd, 1) == b'' and c.identity(before) == c.identity(os.fstat(fd)) == c.identity(path.lstat()) and path.resolve() == path, 'Stable exact plaintext EOF')
        c.require(digest.hexdigest() == ref['sha256'], 'Whole plaintext signed hash')
        return b''.join(chunks)
    finally:
        os.close(fd)


def verify_ancestry(n, signed, authority, read, guard, now):
    """Replay saved signed Release -> complete raw index -> exact package tuple."""
    trust = signed['trust_anchor']
    c.require(authority['schema'] == 'dory_ubuntu_archive_key_authority_v1' and authority['independent_authority_verified'] is True and authority['keyring']['sha256'] == trust['keyring_sha256'] and authority['primary_fingerprints'] == trust['primary_fingerprints'] and all(trust['authority_ref'][k] == n['trust_authority'][k] for k in ('path','sha256')), 'Accepted exact archive trust anchor')
    selected = {r['package']: r for r in n['packages']}; occurrences = 0
    for chain in n['signed_index_reuse']:
        guard(); matches = [r for r in signed['releases'] if r['release_sha256'] == chain['release']['sha256']]
        c.require(len(matches) == 1, 'Unique signed Release ancestry'); release = matches[0]; proof = release['signature_proof']
        c.require(proof == chain['saved_signature_proof'] and proof['verified'] is True and proof['process_returncode'] == 0 and proof['primary_fingerprint'] in trust['primary_fingerprints'] and proof['keyring_sha256'] == trust['keyring_sha256'] and proof['status_reference'] == chain['status_text'], 'Saved successful signature binds this exact Release')
        read(chain['status_text'])
        indexes, _ = metadata.release_indexes(read(chain['release']), chain['suite'], now)
        idx = chain['current_uncompressed_index']; expected = dict(bytes=idx['bytes'], sha256=idx['sha256'])
        c.require(indexes['main/binary-amd64/Packages'] == expected, 'Signed raw Packages size/hash')
        source_index = [x for x in release['indexes'] if x['url'] == chain['index_url']]
        c.require(len(source_index) == 1 and source_index[0]['uncompressed'] == expected, 'Same audited signed index')
        raw = read(idx); found = {}
        for ordinal, stanza in enumerate(metadata.stanzas(raw, 128*c.MIB)):
            if ordinal % 128 == 0: guard()
            name = stanza.get('package'); row = selected.get(name)
            if row is None or stanza.get('version') != row['version'] or stanza.get('architecture') != row['architecture']: continue
            c.require(name not in found, 'No duplicate exact package stanza')
            found[name] = stanza
        for row in n['packages']:
            sources = [x for x in row['sources'] if x['index']['path'] == idx['path']]
            c.require(len(sources) == (1 if row['package'] in found else 0), 'No omitted, duplicate or cross-package source')
            if not sources: continue
            source = sources[0]; stanza = found[row['package']]; saved = source['matched_stanza']
            c.require(source['index'] == idx and source['index_url'] == chain['index_url'] and source['release_url'] == release['release_url'] and source['release_sha256'] == release['release_sha256'] and source['url'] == release['release_url'].split('dists/')[0]+row['filename'], 'Exact package/index/Release/URL association')
            expected_fields = dict(package=row['package'], version=row['version'], architecture=row['architecture'], filename=row['filename'], size=str(row['bytes']), sha256=row['sha256'])
            c.require(all(stanza[k] == v for k,v in expected_fields.items()) and saved['fields'] == expected_fields and saved['index'] == idx, 'Signed exact archive tuple')
            start, size = saved['byte_offset'], saved['bytes']; c.require(type(start) is int and type(size) is int and start >= 0 and 0 < size <= 256*1024 and start+size <= len(raw), 'Finite full stanza slice')
            block = raw[start:start+size]
            c.require((start == 0 or raw[start-2:start] == b'\n\n') and raw[start+size:start+size+2] in (b'\n\n',b'') and hashlib.sha256(block).hexdigest() == saved['sha256'], 'Complete pinned stanza boundaries/hash')
            exact = list(metadata.stanzas(block, 256*1024)); c.require(len(exact) == 1 and exact[0] == stanza, 'Pinned stanza is matching full index stanza')
            occurrences += 1
    c.require(occurrences == sum(len(r['sources']) for r in n['packages']) == 14, 'All fourteen signed package occurrences')
    return occurrences


def authenticated(guard):
    guard(); n = validate_manifest(c.load(NEXT)); signed = c.load(n['signed_metadata']); ended = c.load(n['producer_actual_exit']); recorder = c.load(n['recorder_actual_exit']); authority = c.load(n['trust_authority']); gap = c.load(n['source_gap_report']); tool = c.load(TOOL); tool_exit = c.load(n['dpkg_deb_tool_actual_exit'])
    c.require(signed['status'] == 'exact_addresses_ready' and signed['missing'] == [] and ended['stage_success_observed'] and ended['actual_tool']['exit_code'] == 0 and ended['complete_events_audit']['recomputed']['entire_discovery_reproduced'] and ended['evidence']['signed-metadata/sources-result.json'] == n['signed_metadata'], 'Complete saved signed metadata producer audit')
    c.require(recorder['actual_tool']['exit_code'] == 0 and recorder['owner_gone'] and recorder['result'] == n['producer_actual_exit'], 'Actual signed metadata recorder completed')
    c.require(tool['schema'] == 'dory_dpkg_deb_tool_authority_v1' and tool['reviewer'] == '/root' and tool['tool']['path'] == '/usr/bin/dpkg-deb' and all(n['dpkg_deb_tool_authority'][k] == TOOL[k] for k in TOOL), 'Accepted exact installed decompressor trust base')
    c.require(tool_exit['actual_tool']['exit_code'] == 0 and tool_exit['owner_gone'] and tool_exit['result'] == TOOL, 'Actual decompressor-authority exit')
    status_refs = [x for x in gap['metadata_inputs'] if x['path'] == '/var/lib/dpkg/status']; c.require(len(status_refs) == 1, 'One exact dpkg status input')
    refs = status_refs + [r['dpkg_list'] for r in n['packages']]
    for chain in n['signed_index_reuse']: refs += [chain['status_text'], chain['release'], chain['current_uncompressed_index']]
    allowed = {r['path']:r for r in refs}; c.require(len(allowed) == len(refs), 'Finite unique plaintext input set')
    read = lambda r: read_text(r, guard, allowed)
    verify_ancestry(n, signed, authority, read, guard, dt.datetime.now(dt.timezone.utc))
    installed = {}
    for row in metadata.stanzas(read(status_refs[0]), 8*c.MIB):
        if row.get('package') in PACKAGE_VERSIONS:
            c.require(row['package'] not in installed, 'Unique installed package'); installed[row['package']] = row
    for row in n['packages']:
        guard(); status = installed[row['package']]
        c.require(status['status'] == 'install ok installed' and status['version'] == row['version'] and status['architecture'] == 'amd64', 'Same exact installed version')
        names = read(row['dpkg_list']).decode('utf8').splitlines()
        c.require(all('/'+t['archive_member'] in names for t in row['targets']), 'Declared package member names still present')
    guard(); return n, tool


def authenticate(guard):
    authenticated(guard)
    return dict(schema='minicpm5_sdk310_deb_inputs_authenticated_v1', work_kind=WORK_KIND, inputs=TRUST, packages=8, required_packages=3, candidate_packages=5, targets=12, required_targets=5, candidate_targets=7, original_full_hashes=0, package_content_authenticated=False, actual_loader_mapping_observed=False, complete_native_dependency_closure_established=False, mini_scientific_constraints=CONSTRAINTS, native_ABI_qualified=False)


def validate_admission(value, authenticated):
    c.require(value.get('work_kind') == WORK_KIND and value.get('metadata_input_binding') == authenticated, 'Original parent admission binds exact Mini archive scope')


def validate_result(result, n, mode, prior_ref):
    c.require(result['schema'] == RESULT_SCHEMA and result['complete'] is True and result['inputs'] == TRUST and result['mode'] == mode and result['replay_from'] == prior_ref and result['archive_total_bytes'] == 7110864 and result['package_content_authenticated'] is True, 'Complete exact Mini archive receipt')
    c.require(len(result['packages']) == 8 and result['target_count'] == result['current_match_count'] == 12 and result['original_match_count'] == 0 and result['required_package_count'] == 3 and result['candidate_package_count'] == 5, 'Complete eight/12/zero-old comparisons')
    for key in ('loader_selection_established','native_ABI_qualified','full_model_qualified','new_speedup_claim','complete_native_dependency_closure_established'): c.require(result[key] is False, 'No stronger claim from package bytes')
    c.require(result['decompressed_total_bytes'] == sum(p['inspection']['tar']['stream_bytes'] for p in result['packages']) <= 512*c.MIB, 'Exact complete bounded stream total')
    for row, saved in zip(n['packages'], result['packages']):
        c.require(saved['package'] == row['package'] and saved['version'] == row['version'] and saved['necessity'] == row['necessity'] and saved['actual_loaded'] is None and len(saved['comparisons']) == len(row['targets']), 'Ordered complete package scope')
        c.require(saved['inspection']['archive']['bytes'] == row['bytes'] and saved['inspection']['archive']['sha256'] == row['sha256'], 'Exact signed archive result')
        c.require(set(saved['inspection']['tar']['targets']) == {t['observed_path'] for t in row['targets']}, 'Exact inspected target set')
        for target, comp in zip(row['targets'], saved['comparisons']):
            member = saved['inspection']['tar']['targets'][target['observed_path']]
            c.require(comp['member'] == dict(bytes=member['size'],sha256=member['sha256'],terminal=member['terminal']) and comp['current_identity'] == {k:target['current_lstat']['st_'+k] for k in ('dev','ino','mode','nlink','size','mtime_ns','ctime_ns')}, 'Member/current comparison binds inspected content and exact identity')
            c.require(comp['target'] == target and comp['original_matches'] is None and comp['new_observation_only'] is True and comp['current_matches'] is True and comp['current']['path'] == target['observed_path'] and all(comp['current'][k] == comp['member'][k] for k in ('bytes','sha256')), 'Every exact current target matches new member; no old hash inferred')
    return result


def save_collected(out, value, authenticated_inputs, plan_ref, admission_ref, guard):
    c.require(value['result']['path'] == str(out/'packages'/'sources-result.json'), 'Owned Mini package result')
    n, _ = authenticated(guard)
    result = c.load(value['result'], 8*c.MIB); validate_result(result, n, value['mode'], value['replay_from'])
    guard(); ref = c.save(out/'metadata-result.json', dict(schema='minicpm5_sdk310_deb_member_receipt_v1', work_kind=WORK_KIND, sources_result=value['result'], metadata_input_binding=authenticated_inputs, plan=plan_ref, admission=admission_ref, package_content_authenticated=True, loader_selection_established=False, complete_native_dependency_closure_established=False, native_ABI_qualified=False, full_model_qualified=False, new_speedup_claim=False))
    c.require(storage_usage(out) <= c.INVENTORY['output_bytes'], 'Original total output budget'); return ref
