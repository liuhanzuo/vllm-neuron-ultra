"""Explicit three-index immutable read binding; no IO or patches on import.

Original input n, signature chain and fourteen package occurrences are retained.
The three index reads resolve to authenticated saved plaintext. Dpkg status is
explicitly observed as old and current full bytes; only the fixed two-package
delta is accepted, and current installed state is never inferred from old bytes.
"""
import copy
import datetime as dt
import hashlib
import os
from pathlib import Path
import re
import stat
from minicpm5_sdk310_deb_input_binding_v3 import *
import minicpm5_sdk310_deb_input_binding_v3 as previous
import minicpm5_sdk310_deb_input_binding_v1 as original
import minicpm5_sdk310_snapshot_contract_v2 as c
import dory_deb_transport_v3 as io

OLD_MANIFEST = dict(path=str(c.Q/'minicpm5-sdk310-deb-source-manifest-20261112-v3.json'),
                    sha256='2fe3541566bcd56932df09b808f2f5fcee6ee2d8a28235c308f92b2f02a184de')
STAGE_MANIFEST = dict(path=str(c.Q/'minicpm5-sdk310-signed-index-snapshot-source-manifest-20261112-v1.json'),
                      sha256='291891b5fc69c5decf263f7436066898e1796a0d868445bc42e55b3f4b12e7c7')
STAGE_NAMES = ('minicpm5_sdk310_signed_index_snapshot_v1.py',
               'run_minicpm5_sdk310_signed_index_snapshot_v1.py')
READER_NAMES = ('minicpm5_sdk310_snapshot_contract_v1.py',
                'minicpm5_sdk310_snapshot_read_binding_v1.py',
                'run_minicpm5_sdk310_snapshot_archive_v1.py')
READER_V1_MANIFEST = dict(path=str(c.Q/'minicpm5-sdk310-snapshot-archive-source-manifest-20261112-v1.json'),
                          sha256='2e011a5d1f1fe364a67493405b7b506d1c998613cdee2fa74c3ae930071673f9')
STATUS_NAMES = ('minicpm5_sdk310_snapshot_contract_v2.py',
                'minicpm5_sdk310_snapshot_read_binding_v2.py',
                'run_minicpm5_sdk310_snapshot_archive_v2.py')
STATUS_KEYS = {'old_snapshot','current_snapshot','current_ref','delta_evidence'}
STATUS_DIRECTORY = c.Q/'minicpm5-sdk310-dpkg-status-observation-20261112-v1'
OLD_STATUS = dict(path='/var/lib/dpkg/status',bytes=1535655,
                  sha256='a2e433584eecaf16541653fa498631824f365af688fd214ab45bb5363840e317')
CURRENT_STATUS = dict(path='/var/lib/dpkg/status',bytes=1535655,
                      sha256='70f92d75443e9750a54c8546214a8ef48e35b87b5c6081885e342cb88d505b2a')
STATUS_CHANGES = {
    'gstreamer1.0-plugins-base': {'Version': {'old':'1.20.1-1ubuntu0.7','new':'1.20.1-1ubuntu0.8'},
                                 'Installed-Size': {'old':'2400','new':'2401'}},
    'libgstreamer-plugins-base1.0-0': {'Version': {'old':'1.20.1-1ubuntu0.7','new':'1.20.1-1ubuntu0.8'},
                                     'Installed-Size': {'old':'2740','new':'2748'}},
}
ASSESSMENT = dict(path=str(c.Q/'minicpm5-sdk310-signed-index-snapshot-source-assessment-20261112-v1.json'), sha256='c1b98284c6486b8e870a9446412bcf42782bfe605747a79699cc94a7e2b1d7d0')
SNAPSHOT_SCHEMA = 'minicpm5_sdk310_signed_index_snapshot_v1'
BINDING_KEYS = {'manifest', 'stage_exit', 'recorder_exit'}
SUITES = ('jammy', 'jammy-updates', 'jammy-security')


def same_ref(left, right):
    return (type(left) is dict and type(right) is dict and
            left.get('path') == right.get('path') and left.get('sha256') == right.get('sha256') and
            ('bytes' not in left or 'bytes' not in right or left['bytes'] == right['bytes']))


def metadata_ref(value):
    c.require(type(value) is dict and {'path','sha256'} <= set(value) <= {'path','sha256','bytes'}
              and Path(value['path']).is_absolute() and Path(value['path']).suffix == '.json'
              and type(value['sha256']) is str and re.fullmatch('[0-9a-f]{64}',value['sha256']),
              'Explicit complete snapshot metadata reference')
    if 'bytes' in value:c.require(type(value['bytes']) is int and 0 < value['bytes'] <= 32*c.MIB,
                                 'Bounded snapshot metadata size')
    return value


def actual_zero(value):
    c.require(type(value) is dict and value.get('kind') in ('tool_call','tool_session')
              and type(value.get('exit_code')) is int and value['exit_code'] == 0
              and type(value.get('terminal_chunk')) is str
              and re.fullmatch('[0-9a-f]{3,64}',value['terminal_chunk']),
              'Actual successful completed tool observation, never a manifest self-claim')
    keys={'kind','terminal_chunk','exit_code'}
    if value['kind'] == 'tool_session':
        c.require(set(value) == keys | {'session_id'} and type(value.get('session_id')) is int
                  and value['session_id'] > 0, 'Completed session has its real positive session id')
    else:
        c.require(set(value) == keys, 'Immediate completed tool call has no fabricated session id')


def snapshot_chain(plan, read, n):
    """Authenticate saved completion and explicit namespace without raw reads."""
    bound = plan.get('signed_index_snapshot')
    c.require(type(bound) is dict and set(bound) == BINDING_KEYS, 'All three snapshot completion references required')
    for ref in bound.values():metadata_ref(ref)
    manifest, ended, recorded = (read(bound[key]) for key in ('manifest','stage_exit','recorder_exit'))
    c.require(manifest['schema'] == SNAPSHOT_SCHEMA and manifest['complete'] is True
              and manifest['actual_process_exit_claimed'] is False,
              'Complete immutable snapshot manifest without actual-exit self-claim')
    for key in ('system_APT_files_changed','native_ABI_qualified','full_model_qualified','new_speedup_claim'):
        c.require(manifest[key] is False, 'Snapshot does not qualify native/model/performance')
    c.require(same_ref(manifest['original_inputs'],NEXT) and manifest['signed_metadata'] == n['signed_metadata']
              and manifest['producer_actual_exit'] == n['producer_actual_exit']
              and manifest['recorder_actual_exit'] == n['recorder_actual_exit'],
              'Original input n and signed metadata producer/recorder unchanged')
    c.require(ended['schema'] == 'minicpm5_sdk310_snapshot_stage_actual_exit_v1'
              and ended['stage_success_observed'] is True and ended['owners_gone'] is True
              and ended['child_group_empty'] is True and ended['recorder_self_exit_claimed'] is False
              and same_ref(ended['result'],bound['manifest']), 'Actual successful drained snapshot stage')
    actual_zero(ended['actual_tool'])
    c.require(recorded['schema'] == 'minicpm5_sdk310_snapshot_recorder_actual_exit_v1'
              and recorded['actual_tool_exit_observed'] is True and recorded['owner_gone'] is True
              and recorded['owner'] == ended['recorder'] and same_ref(recorded['result'],bound['stage_exit']),
              'Snapshot external recorder itself really exited; exact recorder owner and report')
    actual_zero(recorded['actual_tool'])
    stage_plan, stage_review = read(ended['plan']), read(ended['review'])
    c.require(stage_review['reviewer'] == '/root' and stage_review['passed'] is True
              and stage_review['blockers'] == [] and same_ref(stage_review['plan'],ended['plan'])
              and stage_review['source_sha256'] == stage_plan['source_sha256'],
              'Exact independently reviewed snapshot stage source closure')
    c.require(stage_plan['stage'] == 'inventory' and stage_plan['budget'] == c.INVENTORY
              and stage_plan['automatic_successor_launch'] is False
              and stage_plan['output_root'] == ended['output_root'], 'Original bounded snapshot inventory stage')
    root = Path(stage_plan['output_root'])
    c.require(root.parent == Path('/mnt/nvme') and root.name.startswith('dory-serial-inventory-minicpm5-sdk310-')
              and Path(bound['manifest']['path']) == root/'snapshot-manifest.json', 'Exact fresh snapshot namespace')
    old = read(OLD_MANIFEST)['source_sha256']
    stage_pins, pins = stage_plan['source_sha256'], plan['source_sha256']
    retained = read(READER_V1_MANIFEST)['source_sha256']
    c.require(stage_pins == read(STAGE_MANIFEST)['source_sha256'], 'Exact independently frozen 100-source snapshot stage')
    c.require(len(old) == 98 and len(stage_pins) == 100 and len(retained) == 103 and len(pins) == 106
              and set(stage_pins) == set(old) | {str(c.Q/name) for name in STAGE_NAMES}
              and set(retained) == set(stage_pins) | {str(c.Q/name) for name in READER_NAMES}
              and set(pins) == set(retained) | {str(c.Q/name) for name in STATUS_NAMES}
              and all(pins.get(path) == sha for path,sha in retained.items())
              and all(stage_pins.get(path) == sha for path,sha in old.items())
              and all(pins.get(path) == sha for path,sha in stage_pins.items()),
              'All 103 frozen reader/stage/history pins retained; exactly three explicit status sources added')
    catalog = read(plan['complete_current_serial_catalog'])
    for key,value in (('stage_exit',ended),('recorder_exit',recorded)):
        rows=[row for row in catalog['runs'] if same_ref(row['terminal'],bound[key])]
        c.require(len(rows) == 1 and rows[0]['tool'] == value['actual_tool']
                  and all(rows[0]['source_sha256'].get(path) == sha for path,sha in stage_pins.items()),
                  'Both actual snapshot and recorder exits retained in current serial history')
    for owner in (ended['owner'],ended['child_owner'],ended['recorder']):
        c.require(owner in plan['predecessor_owners'], 'All actual snapshot stage/recorder owners retained for fresh drain checks')
    c.require(same_ref(manifest['assessment'],ASSESSMENT), 'Fixed source assessment retained')
    assessment=read(ASSESSMENT)
    rows=manifest['rows'];chains=n['signed_index_reuse']
    c.require(len(rows) == len(chains) == 3 and [row['suite'] for row in rows] == list(SUITES)
              and [chain['suite'] for chain in chains] == list(SUITES), 'Exactly three original signed main indexes')
    logical_paths=set();physical_paths=set()
    c.require(len(assessment['rows']) == 3, 'All original compressed/raw reference pairs')
    for row,chain,assessed in zip(rows,chains,assessment['rows']):
        logical,physical=row['logical_ref'],row['snapshot_ref']
        expected=dict(suite=assessed['suite'],logical_ref=assessed['original_logical_index_ref'],
            snapshot_ref=dict(path=str(root/'indexes'/assessed['proposed_snapshot_basename']),**assessed['expected_uncompressed']),
            compressed_ref=assessed['compressed'],release=assessed['release'],status_text=assessed['status_text'],
            saved_signature_proof=assessed['saved_signature_proof'])
        c.require(row == expected, 'Exact original compressed/logical references and entire snapshot row')
        c.require(logical == chain['current_uncompressed_index'] and row['release'] == chain['release']
                  and row['status_text'] == chain['status_text']
                  and row['saved_signature_proof'] == chain['saved_signature_proof'],
                  'Unchanged logical index, signed Release and signature proof')
        c.require(set(physical) == {'path','bytes','sha256'} and
                  physical['bytes'] == logical['bytes'] and physical['sha256'] == logical['sha256']
                  and Path(physical['path']) == root/'indexes'/(row['suite']+'-main-Packages'),
                  'Only exact full original plaintext bytes in the explicit snapshot namespace')
        c.require(logical['path'] not in logical_paths and physical['path'] not in physical_paths,
                  'No duplicate logical or physical index binding')
        logical_paths.add(logical['path']);physical_paths.add(physical['path'])
        relative='indexes/'+row['suite']+'-main-Packages'
        c.require(ended['evidence'][relative] == physical, 'Independent stage recorder hashed every complete raw index')
    c.require(ended['evidence']['snapshot-manifest.json'] == ended['result']
              and manifest['raw_total_bytes'] == sum(row['logical_ref']['bytes'] for row in rows) == 50319218,
              'Whole manifest and all original raw bytes independently recorded')
    return manifest


class SnapshotReadResolver:
    def __init__(self, plan, guard, n, *, read_metadata=None):
        self.guard=guard
        read_metadata = c.load if read_metadata is None else read_metadata
        def read(ref):
            guard();value=read_metadata(ref);guard();return value
        manifest=snapshot_chain(plan,read,n)
        self.rows={row['logical_ref']['path']:copy.deepcopy(row) for row in manifest['rows']}

    def read(self, ref, allowed, cap=128*c.MIB):
        row=self.rows.get(ref['path'])
        if row is None:
            return original.read_text(ref,self.guard,allowed,cap)
        c.require(allowed.get(ref['path']) == ref == row['logical_ref'],
                  'Exact unchanged allowed logical index reference; no generic path redirect')
        physical=row['snapshot_ref'];path=Path(physical['path'])
        self.guard()
        before=path.lstat()
        c.require(stat.S_ISREG(before.st_mode) and stat.S_IMODE(before.st_mode) == 0o400
                  and before.st_nlink == 1 and path.resolve() == path
                  and not any(parent.is_symlink() for parent in path.parents),
                  'Snapshot must remain a single-link immutable ordinary file in ordinary parents')
        raw=original.read_text(physical,self.guard,{physical['path']:physical},cap)
        after=path.lstat()
        c.require(c.identity(before) == c.identity(after) and before.st_mode == after.st_mode
                  and after.st_nlink == 1, 'Snapshot identity/permissions stable across full authenticated read')
        return raw


def validate(plan, review, plan_ref):
    previous.validate(plan,review,plan_ref)
    c.require(plan.get('signed_index_snapshot') == review.get('signed_index_snapshot')
              and type(plan.get('signed_index_snapshot')) is dict
              and set(plan['signed_index_snapshot']) == BINDING_KEYS,
              'Snapshot read binding explicitly accepted in this exact plan and review')
    c.require(plan.get('metadata_source_count_limit') == review.get('metadata_source_count_limit') == 106,
              'Only the new 106-source metadata contract is admitted')
    c.require(plan.get('dpkg_status_observation') == review.get('dpkg_status_observation')
              and type(plan.get('dpkg_status_observation')) is dict
              and set(plan['dpkg_status_observation']) == STATUS_KEYS, 'Exact explicitly reviewed old/current status observation')
    for name in READER_NAMES+STATUS_NAMES:c.require(str(c.Q/name) in plan['source_sha256'], 'All three reader runtime sources reviewed')
    return plan


def status_delta(old_raw, current_raw):
    """Prove the exact four changed lines; preserve every other original byte."""
    def rows(raw):
        c.require(type(raw) is bytes and raw.endswith(b'\n\n'), 'Complete dpkg paragraph boundaries')
        separated=raw.split(b'\n\n')
        c.require(separated[-1]==b'', 'No unparsed bytes after the final status separator')
        pieces=separated[:-1]
        parsed=list(metadata.stanzas(raw,8*c.MIB))
        c.require(len(pieces)==len(parsed)==1521 and all(pieces), 'Same complete 1521-package status')
        keyed=[(r['package'],r.get('architecture')) for r in parsed]
        c.require(len(set(keyed))==1521, 'Unique package/architecture paragraphs')
        return pieces,parsed,keyed
    old_parts,old_rows,old_keys=rows(old_raw);new_parts,new_rows,new_keys=rows(current_raw)
    c.require(old_keys==new_keys, 'No package added, removed or reordered')
    changed=[];selected=[]
    for before,after,old,new in zip(old_parts,new_parts,old_rows,new_rows):
        package=old['package']
        if package in PACKAGE_VERSIONS:
            c.require(before==after and old==new and old['architecture']=='amd64'
                      and old['status']=='install ok installed' and old['version']==PACKAGE_VERSIONS[package],
                      'Every complete selected package stanza remains unchanged')
            selected.append(package)
        if before==after:continue
        c.require(package in STATUS_CHANGES and old['architecture']==new['architecture']=='amd64'
                  and old['status']==new['status']=='install ok installed', 'Only two fixed installed unrelated packages may change')
        expected=STATUS_CHANGES[package];rebuilt=before.split(b'\n')
        for field,values in expected.items():
            c.require(old[field.lower()]==values['old'] and new[field.lower()]==values['new'], 'Exact observed old/new field values')
            line=(field+': '+values['old']).encode();replacement=(field+': '+values['new']).encode()
            c.require(rebuilt.count(line)==1, 'One exact changed status line')
            rebuilt[rebuilt.index(line)]=replacement
        c.require(b'\n'.join(rebuilt)==after, 'Every other byte of both changed stanzas remains identical')
        changed.append(dict(package=package,fields=copy.deepcopy(expected),
                            old_stanza_sha256=hashlib.sha256(before).hexdigest(),
                            new_stanza_sha256=hashlib.sha256(after).hexdigest()))
    c.require(sorted(r['package'] for r in changed)==sorted(STATUS_CHANGES)
              and len(selected)==8 and set(selected)==set(PACKAGE_VERSIONS), 'Exactly two changed and all eight unchanged selected packages')
    return sorted(changed,key=lambda r:r['package'])


def status_metadata(plan, read, n, gap, old_ref):
    bound=plan.get('dpkg_status_observation')
    c.require(type(bound) is dict and set(bound)==STATUS_KEYS, 'Four explicit status observation references')
    metadata_ref(bound['delta_evidence']);evidence=read(bound['delta_evidence'])
    expected_old=dict(OLD_STATUS,path=str(STATUS_DIRECTORY/'original-status.txt'))
    expected_current=dict(CURRENT_STATUS,path=str(STATUS_DIRECTORY/'observed-status.txt'))
    c.require(old_ref==OLD_STATUS and bound['old_snapshot']==expected_old
              and bound['current_snapshot']==expected_current and bound['current_ref']==CURRENT_STATUS,
              'Original status hash retained separately; exact new live full hash and immutable snapshots')
    c.require(evidence['schema']=='minicpm5_sdk310_dpkg_status_observation_v1'
              and evidence['reviewer']=='/root' and evidence['passed'] is True and evidence['blockers']==[]
              and evidence['original_inputs']==NEXT and evidence['source_gap_report']==n['source_gap_report']
              and evidence['original_status_ref']==old_ref
              and all(evidence[k]==bound[k] for k in ('old_snapshot','current_snapshot','current_ref')),
              'Root evidence explicitly binds original input/gap and old/current status identities')
    c.require(evidence['package_count']==1521 and evidence['unchanged_selected_packages']==list(PACKAGE_VERSIONS)
              and evidence['unchanged_package_lists']==[r['dpkg_list'] for r in n['packages']]
              and evidence['metadata_only'] is True and evidence['new_speedup_claim'] is False,
              'Only metadata observation, with all eight selected packages retained')
    c.require([r['package'] for r in evidence['changed_packages']]==sorted(STATUS_CHANGES)
              and all(r['fields']==STATUS_CHANGES[r['package']] for r in evidence['changed_packages']),
              'Exactly the observed four field changes in saved evidence')
    c.require([r for r in gap['metadata_inputs'] if r['path']==OLD_STATUS['path']]==[old_ref],
              'No old source-gap or n rewrite')
    return evidence


def immutable_status(ref, guard):
    path=Path(ref['path']);guard();before=path.lstat()
    c.require(stat.S_ISREG(before.st_mode) and stat.S_IMODE(before.st_mode)==0o400
              and before.st_nlink==1 and path.resolve()==path
              and not any(parent.is_symlink() for parent in path.parents), 'Immutable single-link status snapshot and ordinary parents')
    raw=original.read_text(ref,guard,{ref['path']:ref},8*c.MIB)
    after=path.lstat()
    c.require(c.identity(before)==c.identity(after) and before.st_mode==after.st_mode and after.st_nlink==1,
              'Stable complete immutable status snapshot')
    return raw


def authenticate_status(plan, guard, n, gap, old_ref):
    guard();evidence=status_metadata(plan,c.load,n,gap,old_ref);guard()
    bound=plan['dpkg_status_observation']
    old_raw=immutable_status(bound['old_snapshot'],guard)
    saved_current=immutable_status(bound['current_snapshot'],guard)
    # This is an explicit new live observation, never a resolution of old_ref.
    current_ref=bound['current_ref']
    current_raw=original.read_text(current_ref,guard,{current_ref['path']:current_ref},8*c.MIB)
    c.require(current_raw==saved_current, 'Current live status still equals the exact saved current observation')
    c.require(status_delta(old_raw,current_raw)==evidence['changed_packages'], 'Full exact delta independently recomputed on every authentication')
    guard();return current_raw


def authenticated(plan, guard):
    guard(); n = validate_manifest(c.load(NEXT)); signed = c.load(n['signed_metadata']); ended = c.load(n['producer_actual_exit']); recorder = c.load(n['recorder_actual_exit']); authority = c.load(n['trust_authority']); gap = c.load(n['source_gap_report']); tool = c.load(TOOL); tool_exit = c.load(n['dpkg_deb_tool_actual_exit'])
    c.require(signed['status'] == 'exact_addresses_ready' and signed['missing'] == [] and ended['stage_success_observed'] and ended['actual_tool']['exit_code'] == 0 and ended['complete_events_audit']['recomputed']['entire_discovery_reproduced'] and ended['evidence']['signed-metadata/sources-result.json'] == n['signed_metadata'], 'Complete saved signed metadata producer audit')
    c.require(recorder['actual_tool']['exit_code'] == 0 and recorder['owner_gone'] and recorder['result'] == n['producer_actual_exit'], 'Actual signed metadata recorder completed')
    c.require(tool['schema'] == 'dory_dpkg_deb_tool_authority_v1' and tool['reviewer'] == '/root' and tool['tool']['path'] == '/usr/bin/dpkg-deb' and all(n['dpkg_deb_tool_authority'][k] == TOOL[k] for k in TOOL), 'Accepted exact installed decompressor trust base')
    c.require(tool_exit['actual_tool']['exit_code'] == 0 and tool_exit['owner_gone'] and tool_exit['result'] == TOOL, 'Actual decompressor-authority exit')
    status_refs = [x for x in gap['metadata_inputs'] if x['path'] == '/var/lib/dpkg/status']; c.require(len(status_refs) == 1, 'One exact dpkg status input')
    current_status = authenticate_status(plan,guard,n,gap,status_refs[0])
    refs = [r['dpkg_list'] for r in n['packages']]
    for chain in n['signed_index_reuse']: refs += [chain['status_text'], chain['release'], chain['current_uncompressed_index']]
    allowed = {r['path']:r for r in refs}; c.require(len(allowed) == len(refs), 'Finite unique plaintext input set')
    resolver = SnapshotReadResolver(plan,guard,n)
    read = lambda r: resolver.read(r,allowed)
    verify_ancestry(n, signed, authority, read, guard, dt.datetime.now(dt.timezone.utc))
    installed = {}
    for row in metadata.stanzas(current_status, 8*c.MIB):
        if row.get('package') in PACKAGE_VERSIONS:
            c.require(row['package'] not in installed, 'Unique installed package'); installed[row['package']] = row
    for row in n['packages']:
        guard(); status = installed[row['package']]
        c.require(status['status'] == 'install ok installed' and status['version'] == row['version'] and status['architecture'] == 'amd64', 'Same exact installed version')
        names = read(row['dpkg_list']).decode('utf8').splitlines()
        c.require(all('/'+t['archive_member'] in names for t in row['targets']), 'Declared package member names still present')
    guard(); return n, tool


def authenticate(plan, guard):
    authenticated(plan,guard)
    return dict(schema='minicpm5_sdk310_deb_inputs_authenticated_snapshot_status_v2', dpkg_status_observation=copy.deepcopy(plan['dpkg_status_observation']), signed_index_snapshot=copy.deepcopy(plan['signed_index_snapshot']), work_kind=WORK_KIND, inputs=TRUST, packages=8, required_packages=3, candidate_packages=5, targets=12, required_targets=5, candidate_targets=7, original_full_hashes=0, package_content_authenticated=False, actual_loader_mapping_observed=False, complete_native_dependency_closure_established=False, mini_scientific_constraints=CONSTRAINTS, native_ABI_qualified=False)


def save_collected(out, value, authenticated_inputs, plan_ref, admission_ref, guard, *, plan):
    c.require(value['result']['path'] == str(out/'packages'/'sources-result.json'), 'Owned Mini package result')
    c.require(authenticated_inputs.get('signed_index_snapshot') == plan['signed_index_snapshot'], 'Collected receipt retains exact snapshot read binding')
    c.require(authenticated_inputs.get('dpkg_status_observation') == plan['dpkg_status_observation'], 'Collected receipt retains explicit current status observation')
    n, _ = authenticated(plan,guard)
    result = c.load(value['result'], 8*c.MIB); validate_result(result, n, value['mode'], value['replay_from'])
    guard(); ref = c.save(out/'metadata-result.json', dict(schema='minicpm5_sdk310_deb_member_receipt_v1', work_kind=WORK_KIND, sources_result=value['result'], metadata_input_binding=authenticated_inputs, plan=plan_ref, admission=admission_ref, package_content_authenticated=True, loader_selection_established=False, complete_native_dependency_closure_established=False, native_ABI_qualified=False, full_model_qualified=False, new_speedup_claim=False))
    c.require(storage_usage(out) <= c.INVENTORY['output_bytes'], 'Original total output budget'); return ref


def run(plan, guard, emit, trust, out, mode, prior_ref=None):
    c.require(trust == TRUST and mode in ('download','replay') and (prior_ref is None) == (mode == 'download'), 'Exact reviewed Mini archive mode')
    n, authority = authenticated(plan,guard); tool = authority['tool']; guard(); Path(out).mkdir(mode=0o700)
    prior = c.load(prior_ref, 8*c.MIB) if mode == 'replay' else None
    if prior is not None:
        validate_result(prior, n, 'download', None)
    packages = []; stream_total = current = 0
    for index, row in enumerate(n['packages']):
        guard()
        if mode == 'download':
            archive = Path(out)/(row['package']+'.deb'); io.download(row, archive, guard, emit)
        else:
            previous = prior['packages'][index]
            archive = Path(previous['inspection']['archive']['path']); io.file_ref(archive, guard, row)
            emit(dict(event='archive_replay_start', package=row['package'], archive=previous['inspection']['archive']))
        inspection = io.inspect(row, archive, tool, guard, emit)
        stream_total += inspection['tar']['stream_bytes']; c.require(stream_total <= 512*c.MIB, 'Original aggregate decompressed stream cap')
        comparisons = []
        for target in row['targets']:
            guard(); member = inspection['tar']['targets'][target['observed_path']]; expected = dict(bytes=member['size'],sha256=member['sha256'])
            c.require(target['original_observed_reference'] is None, 'No invented original SDK member hash')
            observed, identity = io.file_ref(target['observed_path'], guard, expected, expected_nlink=1)
            saved_identity = target['current_lstat']
            c.require(identity == {k:saved_identity['st_'+k] for k in ('dev','ino','mode','nlink','size','mtime_ns','ctime_ns')}, 'Exact prior lstat identity plus new complete content hash')
            current += 1
            comparisons.append(dict(target=target, member=dict(bytes=member['size'],sha256=member['sha256'],terminal=member['terminal']), original_matches=None, current=observed, current_identity=identity, current_matches=True, new_observation_only=True))
        package = dict(package=row['package'], version=row['version'], necessity=row['necessity'], actual_loaded=None, inspection=inspection, comparisons=comparisons)
        if prior is not None: c.require(package == prior['packages'][index], 'Independent complete archive/member/current replay equals producer')
        packages.append(package)
        emit(dict(event='package_members_verified', package=row['package'], targets=len(comparisons), tar_sha256=inspection['tar']['stream_sha256']))
    c.require(current == 12 and len(packages) == 8, 'All twelve selected targets across eight packages')
    value = dict(schema=RESULT_SCHEMA, complete=True, mode=mode, replay_from=prior_ref, inputs=trust, packages=packages, archive_total_bytes=n['archive_total_bytes'], decompressed_total_bytes=stream_total, target_count=12, original_match_count=0, current_match_count=current, required_package_count=3, candidate_package_count=5, package_content_authenticated=True, loader_selection_established=False, complete_native_dependency_closure_established=False, native_ABI_qualified=False, full_model_qualified=False, new_speedup_claim=False)
    validate_result(value, n, mode, prior_ref)
    result = c.save(Path(out)/'sources-result.json', value, 8*c.MIB); guard()
    emit(dict(event='minicpm5_eight_packages_complete', mode=mode, result=result, packages=8, targets=12, original_matches=0, current_matches=12, required_packages=3, candidate_packages=5))
    return dict(result=result, packages=8, targets=12, mode=mode, replay_from=prior_ref)
