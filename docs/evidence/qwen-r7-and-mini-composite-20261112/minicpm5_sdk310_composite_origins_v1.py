"""Saved old-libc6 plus new-eight package origins, with no IO on import.

Only explicit reader operations read saved metadata and complete signed plaintext.
No archive/native read, current-status substitution, process or native operation
is performed here. The caller still authenticates current ELF/maps and original9.
"""
import hashlib
from pathlib import PurePosixPath as Path
import re

Q='/home/ubuntu/experiments/trn1-latest-models-20261031/research/'
NEXT=dict(path=Q+'minicpm5-sdk310-archive-member-next-inputs-20261112-v1.json',
          sha256='e91bca576461d9d1d4b206e7f8b508c20a103a3231bef9b29535e97d3e8da266')
OLD_REPLAY=dict(path='/mnt/nvme/dory-serial-inventory-deb-replay-r1-20261111/packages/sources-result.json',
                sha256='594f9c3fa05579fa1da9af0a486ca252d988ff16e05041bd05b1b18237574a14')
CONSTRAINTS=dict(native_checks=9,scientific_descriptor_rows=1284,absolute_wall_seconds=900,
                 address_space_bytes=2147483648,single_file_bytes=2097152)
BUNDLE_KEYS={'producer_result','producer_exit','producer_recorder_exit',
             'replay_result','replay_exit','replay_recorder_exit'}
FALSE_CLAIMS=('loader_selection_established','native_ABI_qualified','full_model_qualified','new_speedup_claim')
RAW_TOTAL=50319218


def require(value,message):
    if not value:raise ValueError(message)


def reference(ref):
    require(type(ref) is dict and {'path','sha256'}<=set(ref)<={'path','sha256','bytes'}
        and type(ref['path']) is str and Path(ref['path']).is_absolute()
        and '..' not in Path(ref['path']).parts and type(ref['sha256']) is str
        and re.fullmatch('[0-9a-f]{64}',ref['sha256']),'Exact bounded origin reference')
    if 'bytes' in ref:require(type(ref['bytes']) is int and 0<ref['bytes']<=64*2**20,'Bounded origin bytes')
    return ref


def same(left,right):
    reference(left);reference(right)
    return left['path']==right['path'] and left['sha256']==right['sha256'] and (
        'bytes' not in left or 'bytes' not in right or left['bytes']==right['bytes'])


def actual_zero(tool):
    require(type(tool) is dict and tool.get('kind') in ('tool_call','tool_session')
        and type(tool.get('exit_code')) is int and tool['exit_code']==0
        and type(tool.get('terminal_chunk')) is str
        and re.fullmatch('[0-9a-f]{3,64}',tool['terminal_chunk']),'Real successful terminal observation')
    keys={'kind','exit_code','terminal_chunk'}
    if tool['kind']=='tool_session':
        require(set(tool)==keys|{'session_id'} and type(tool['session_id']) is int
            and tool['session_id']>0,'Actual session id')
    else:require(set(tool)==keys,'Immediate tool call never invents a session id')


def metadata(reader,ref):
    reference(ref);require(Path(ref['path']).suffix=='.json','JSON metadata only')
    return reader.json(ref)


def stage(reader,result_ref,exit_ref,recorder_ref,prefix,pins,mode=None):
    """Bind saved producer and separately observed recorder, never self-exit."""
    ended=metadata(reader,exit_ref);recorded=metadata(reader,recorder_ref)
    require(ended['schema']==prefix+'_stage_actual_exit_v1'
        and ended['stage_success_observed'] is True and ended['owners_gone'] is True
        and ended['child_group_empty'] is True and ended['recorder_self_exit_claimed'] is False
        and same(ended['result'],result_ref),'Successful drained exact package/snapshot stage')
    actual_zero(ended['actual_tool']);actual_zero(recorded['actual_tool'])
    require(recorded['schema']==prefix+'_recorder_actual_exit_v1' and recorded['reviewer']=='/root'
        and recorded['actual_tool_exit_observed'] is True and recorded['owner_gone'] is True
        and recorded['owner']==ended['recorder'] and same(recorded['result'],exit_ref),
        'Separate real recorder completion binds the exact stage')
    for value in (ended,recorded):
        require(value['full_model_qualified'] is False and value['new_speedup_claim'] is False,
                'Origin stage makes no model or speed claim')
    plan=metadata(reader,ended['plan']);review=metadata(reader,ended['review'])
    require(review['reviewer']=='/root' and review['passed'] is True and review['blockers']==[]
        and same(review['plan'],ended['plan']) and review['source_sha256']==plan['source_sha256']
        and plan['source_sha256'] and all(pins.get(p)==h for p,h in plan['source_sha256'].items()),
        'Exact accepted stage sources retained in complete SDK source map')
    require(plan['stage']=='inventory' and plan['automatic_successor_launch'] is False
        and plan['output_root']==ended['output_root'],'Bounded completed inventory, no successor launch')
    root=Path(ended['output_root']);require(root.parent==Path('/mnt/nvme'),'Saved inventory namespace')
    relative=str(Path(result_ref['path']).relative_to(root))
    require(same(ended['evidence'][relative],result_ref),'Result fully hashed by separate stage recorder')
    if mode is not None:
        require(ended['mode']==plan['archive_mode']==review['archive_mode']==mode
            and plan['archive_inputs']==review['archive_inputs']
            and plan['replay_from']==review['replay_from'],'Exact accepted archive mode and input binding')
    if prefix=='minicpm5_sdk310_deb':
        require(recorded['actual_self_exit_claimed'] is False
            and recorded['original_lock_nonblocking_acquired_and_released'] is True
            and same(recorded['plan'],ended['plan']) and same(recorded['review'],ended['review'])
            and recorded['lock_identity']==ended['lock_identity'],'Exact external recorder lock and plan evidence')
    return plan,ended


def bundle(reader,bound,prefix,pins):
    require(type(bound) is dict and set(bound)==BUNDLE_KEYS,'Explicit producer/replay and both recorder refs')
    require(bound['producer_exit']['path']!=bound['replay_exit']['path']
        and bound['producer_recorder_exit']['path']!=bound['replay_recorder_exit']['path']
        and bound['producer_result']['path']!=bound['replay_result']['path'],'Independent distinct completion records')
    produced=metadata(reader,bound['producer_result']);replayed=metadata(reader,bound['replay_result'])
    plans=[]
    for name,mode,result,prior in [('producer','download',produced,None),
                                  ('replay','replay',replayed,bound['producer_result'])]:
        plan,_=stage(reader,bound[name+'_result'],bound[name+'_exit'],bound[name+'_recorder_exit'],prefix,pins,mode)
        require(result['complete'] is True and result['package_content_authenticated'] is True
            and result['mode']==mode and result['replay_from']==plan['replay_from']
            and (result['replay_from'] is None if prior is None else same(result['replay_from'],prior))
            and plan['archive_inputs']==result['inputs'],
            'Complete actual result matches its reviewed stage and exact prior')
        require(all(result[k] is False for k in FALSE_CLAIMS),'Package provenance is not loader authority')
        plans.append(plan)
    require(produced['inputs']==replayed['inputs'] and produced['packages']==replayed['packages'],
            'Independent replay reproduces every complete package/member/comparison record')
    return replayed,plans


def fields(block):
    """Strict Debian control stanza; continuation lines belong to prior field."""
    result={};key=None
    for line in block.decode('utf8').splitlines():
        require(line!='','Single complete stanza')
        if line[0] in ' \t':
            require(key is not None,'Continuation has a field');result[key]+='\n'+line
        else:
            require(':' in line,'Debian field delimiter');key,value=line.split(':',1);key=key.lower()
            require(re.fullmatch('[a-z0-9-]+',key) and key not in result,'Unique Debian control fields')
            result[key]=value.lstrip(' \t')
    return result


def signed_metadata(reader,n):
    signed=metadata(reader,n['signed_metadata']);ended=metadata(reader,n['producer_actual_exit'])
    recorded=metadata(reader,n['recorder_actual_exit']);authority=metadata(reader,n['trust_authority'])
    require(signed['status']=='exact_addresses_ready' and signed['missing']==[]
        and ended['schema']=='dory_ubuntu_metadata_actual_exit_v1'
        and ended['stage_success_observed'] is True
        and ended['parent_owner_gone'] is True and ended['child_owner_gone'] is True
        and ended['child_group_empty'] is True
        and ended['complete_events_audit']['recomputed']['entire_discovery_reproduced'] is True
        and same(ended['evidence']['signed-metadata/sources-result.json'],n['signed_metadata']),
        'Original actual signed-index producer audit')
    actual_zero(ended['actual_tool']);actual_zero(recorded['actual_tool'])
    require(recorded['schema']=='dory_ubuntu_metadata_recorder_actual_exit_v1'
        and recorded['reviewer']=='/root' and recorded['actual_tool_exit_observed'] is True
        and recorded['owner_gone'] is True and recorded['owner']==ended['recorder']
        and recorded['self_exit_claimed'] is False and same(recorded['result'],n['producer_actual_exit']),
            'Original signed-index recorder actually completed')
    trust=signed['trust_anchor']
    require(authority['schema']=='dory_ubuntu_archive_key_authority_v1'
        and authority['independent_authority_verified'] is True
        and authority['keyring']['sha256']==trust['keyring_sha256']
        and authority['primary_fingerprints']==trust['primary_fingerprints']
        and same(trust['authority_ref'],n['trust_authority']),'Original independent Ubuntu signing authority')
    return signed


def snapshot_ancestry(reader,n,signed,plans,pins):
    """Read each entire immutable index, not just selected stanza slices."""
    binding=plans[0]['signed_index_snapshot']
    require(binding==plans[1]['signed_index_snapshot'] and set(binding)=={'manifest','stage_exit','recorder_exit'},
            'Both new stages bind the same real complete index snapshot')
    manifest=metadata(reader,binding['manifest'])
    _,ended=stage(reader,binding['manifest'],binding['stage_exit'],binding['recorder_exit'],
                  'minicpm5_sdk310_snapshot',pins)
    require(manifest['schema']=='minicpm5_sdk310_signed_index_snapshot_v1' and manifest['complete'] is True
        and same(manifest['original_inputs'],NEXT) and manifest['signed_metadata']==n['signed_metadata']
        and manifest['producer_actual_exit']==n['producer_actual_exit']
        and manifest['recorder_actual_exit']==n['recorder_actual_exit'], 'Snapshot retains original n/signed ancestry')
    require(all(manifest[k] is False for k in ('actual_process_exit_claimed','system_APT_files_changed',
        'native_ABI_qualified','full_model_qualified','new_speedup_claim')),'No snapshot success promotion')
    assessment=metadata(reader,manifest['assessment'])
    require(len(manifest['rows'])==len(n['signed_index_reuse'])==len(assessment['rows'])==3,
            'Exactly all three original complete index sources')
    selected={r['package']:r for r in n['packages']};occurrences=0;total=0
    for row,chain,assessed in zip(manifest['rows'],n['signed_index_reuse'],assessment['rows']):
        logical=chain['current_uncompressed_index'];physical=row['snapshot_ref']
        expected=dict(suite=assessed['suite'],logical_ref=assessed['original_logical_index_ref'],
            snapshot_ref=dict(path=str(Path(ended['output_root'])/'indexes'/assessed['proposed_snapshot_basename']),
                              **assessed['expected_uncompressed']),compressed_ref=assessed['compressed'],
            release=assessed['release'],status_text=assessed['status_text'],saved_signature_proof=assessed['saved_signature_proof'])
        require(row==expected and row['suite']==chain['suite'] and row['logical_ref']==logical
            and row['release']==chain['release'] and row['status_text']==chain['status_text']
            and row['saved_signature_proof']==chain['saved_signature_proof']
            and {k:physical[k] for k in ('bytes','sha256')}=={k:logical[k] for k in ('bytes','sha256')}
            and same(ended['evidence']['indexes/'+row['suite']+'-main-Packages'],physical),
            'Exact whole assessed snapshot row, logical bytes and independent recorder hash')
        matches=[r for r in signed['releases'] if r['release_sha256']==chain['release']['sha256']]
        require(len(matches)==1,'Unique original signed Release');release=matches[0];proof=release['signature_proof']
        require(proof==chain['saved_signature_proof'] and proof['verified'] is True
            and proof['process_returncode']==0 and proof['primary_fingerprint'] in signed['trust_anchor']['primary_fingerprints']
            and proof['keyring_sha256']==signed['trust_anchor']['keyring_sha256']
            and proof['status_reference']==chain['status_text'],'Actual saved successful signature for this Release')
        reader.read(chain['status_text'],2**20)
        release_raw,_=reader.read(chain['release'],2**20);release_fields=fields(release_raw.rstrip(b'\n'))
        entries=[line.split() for line in release_fields['sha256'].splitlines() if line.strip()]
        matches=[e for e in entries if len(e)==3 and e[2]=='main/binary-amd64/Packages']
        require(matches==[[logical['sha256'],str(logical['bytes']),'main/binary-amd64/Packages']],
                'Full signed Release explicitly binds whole uncompressed Packages')
        indexes=[i for i in release['indexes'] if i['url']==chain['index_url']]
        require(len(indexes)==1 and indexes[0]['uncompressed']=={k:logical[k] for k in ('sha256','bytes')}
            and indexes[0]['compressed']=={k:row['compressed_ref'][k] for k in ('sha256','bytes')},
            'Original compressed and complete plaintext index provenance')
        raw,observed=reader.read(physical,64*2**20);identity=observed['identity']
        require(identity['st_mode']&0o7777==0o400 and identity['st_nlink']==1,
                'Whole authenticated immutable single-link snapshot')
        total+=len(raw);found={};offset=0
        for ordinal,block in enumerate(raw.split(b'\n\n')):
            if ordinal%128==0:reader.guard()
            start=offset;offset+=len(block)+2
            if not block.strip():continue
            stanza=fields(block.rstrip(b'\n'));candidate=selected.get(stanza.get('package'))
            if candidate is None or stanza.get('version')!=candidate['version'] or stanza.get('architecture')!=candidate['architecture']:continue
            require(candidate['package'] not in found,'Unique exact package stanza in full index')
            found[candidate['package']]=(start,block,stanza)
        for package in n['packages']:
            sources=[s for s in package['sources'] if s['index']['path']==logical['path']]
            require(len(sources)==(1 if package['package'] in found else 0),'All signed package occurrences retained')
            if not sources:continue
            source=sources[0];start,block,stanza=found[package['package']];saved=source['matched_stanza']
            expected_fields=dict(package=package['package'],version=package['version'],architecture=package['architecture'],
                filename=package['filename'],size=str(package['bytes']),sha256=package['sha256'])
            require(source['index']==logical and source['index_url']==chain['index_url']
                and source['release_url']==release['release_url'] and source['release_sha256']==release['release_sha256']
                and source['url']==release['release_url'].split('dists/')[0]+package['filename']
                and saved['index']==logical and saved['fields']==expected_fields
                and all(stanza[k]==v for k,v in expected_fields.items())
                and saved['byte_offset']==start and saved['bytes']==len(block)
                and saved['sha256']==hashlib.sha256(block).hexdigest(),
                'Entire exact stanza at original offset binds signed archive/version/member source')
            occurrences+=1
    require(total==manifest['raw_total_bytes']==RAW_TOTAL
        and occurrences==sum(len(r['sources']) for r in n['packages'])==14,
        'All complete index bytes and fourteen original package occurrences')


def new_packages(result,n):
    require(result['schema']=='minicpm5_sdk310_deb_members_result_v1'
        and same(result['inputs']['next_inputs'],NEXT) and result['archive_total_bytes']==n['archive_total_bytes']==7110864
        and len(result['packages'])==8 and result['target_count']==result['current_match_count']==12
        and result['original_match_count']==0 and result['required_package_count']==3 and result['candidate_package_count']==5
        and result['complete_native_dependency_closure_established'] is False,'Exact eight/12/zero-original result')
    require(result['decompressed_total_bytes']==sum(p['inspection']['tar']['stream_bytes'] for p in result['packages'])
        <=512*2**20,'Complete bounded decompressed streams')
    for row,saved in zip(n['packages'],result['packages']):
        inspection=saved['inspection'];tar=inspection['tar']
        require(saved['package']==row['package'] and saved['version']==row['version']
            and saved['necessity']==row['necessity'] and saved['actual_loaded'] is None
            and tar['complete'] is True and inspection['decompressor']['returncode']==0
            and all(inspection['archive'][k]==row[k] for k in ('sha256','bytes'))
            and set(tar['targets'])=={t['observed_path'] for t in row['targets']}
            and len(saved['comparisons'])==len(row['targets']),'Whole exact ordered package/member replay')
        for target,comparison in zip(row['targets'],saved['comparisons']):
            member=tar['targets'][target['observed_path']]
            require(target['original_observed_reference'] is None and target['original_full_payload_hash_present'] is False
                and comparison['target']==target and comparison['original_matches'] is None
                and comparison['current_matches'] is True and comparison['new_observation_only'] is True
                and comparison['member']==dict(bytes=member['size'],sha256=member['sha256'],terminal=member['terminal'])
                and comparison['current']['path']==target['observed_path']
                and all(comparison['current'][k]==comparison['member'][k] for k in ('sha256','bytes'))
                and comparison['current_identity']=={k:target['current_lstat']['st_'+k]
                    for k in ('dev','ino','mode','nlink','size','mtime_ns','ctime_ns')},
                'Saved new observation is complete; never invents an old or present-time hash')


def replay_package_origins(reader,decision,pins):
    require(decision['schema']=='minicpm5_sdk310_native_source_review_v2','Explicit composite authority successor')
    bound=decision['package_origins']
    require(type(bound) is dict and set(bound)=={'schema','old_libc6','new_eight'}
        and bound['schema']=='minicpm5_sdk310_composite_package_origins_v1','Two distinct real origin chains')
    require(same(bound['old_libc6']['replay_result'],OLD_REPLAY),'Reuse exact already completed libc6 replay')
    old,_=bundle(reader,bound['old_libc6'],'dory_deb',pins)
    new,plans=bundle(reader,bound['new_eight'],'minicpm5_sdk310_deb',pins)
    n=metadata(reader,NEXT);signed=signed_metadata(reader,n)
    require(n['original_mini_diagnostic_constraints']==CONSTRAINTS and n['counts']['packages']==8
        and n['counts']['selected_targets']==12 and n['counts']['original_full_payload_hashes']==0,
        'Original nine/1284/900s/2GiB scientific constraints and zero invented old hashes')
    old_n=metadata(reader,old['inputs']['next_inputs'])
    require(same(old_n['signed_metadata'],n['signed_metadata']) and old['schema']=='dory_deb_members_result_v1'
        and len(old['packages'])==6,'Independent old libc6 signed chain remains unchanged')
    for plan in plans:
        require(plan['mini_scientific_constraints']==CONSTRAINTS
            and plan['archive_inputs']==new['inputs'],'New reviewed stage retains exact scientific/input binding')
    require(same(new['inputs']['tool_authority'],n['dpkg_deb_tool_authority']), 'Same reviewed decompressor authority')
    snapshot_ancestry(reader,n,signed,plans,pins);new_packages(new,n)
    natives=decision['native_files'];pure=decision['pure_sources'];expected={**natives,**pure}
    require(len(expected)==len(natives)+len(pure),'Native and pure origin domains are disjoint')
    official_old=[r for r in signed['packages'] if r['package']=='libc6' and r['version']=='2.35-0ubuntu3.15']
    actual_old=[r for r in old['packages'] if r['package']=='libc6' and r['version']=='2.35-0ubuntu3.15']
    require(len(official_old)==len(actual_old)==1,'One exact independently replayed old libc6')
    official=official_old[0];inspection=actual_old[0]['inspection']
    require(inspection['tar']['complete'] is True and inspection['decompressor']['returncode']==0
        and all(inspection['archive'][k]==official[k] for k in ('sha256','bytes')) and official['sources'],
        'Complete original libc6 archive matches signed package')
    for source in official['sources']:
        releases=[r for r in signed['releases'] if r['release_url']==source['release_url']]
        require(len(releases)==1,'Unique libc6 signed release');release=releases[0];proof=release['signature_proof']
        require(release['release_sha256']==source['release_sha256'] and proof['verified'] is True
            and proof['process_returncode']==0 and proof['primary_fingerprint'] in signed['trust_anchor']['primary_fingerprints'],
            'Original signed libc6 source retained')
    packages={(r['package'],r['version']):r for r in new['packages']}
    packages[('libc6','2.35-0ubuntu3.15')]=actual_old[0]
    require(len(packages)==9,'Unique disjoint new-eight and old-libc6 package providers')
    for path,row in expected.items():
        require(row['path']==path and set(row)=={'path','sha256','bytes','package','version'},'Exact selected source member')
        key=(row['package'],row['version']);require(key in packages,'No unverified provider outside the two origin chains')
        member=packages[key]['inspection']['tar']['targets'][path]
        require(row['sha256']==member['sha256'] and row['bytes']==member['size'],'Exact selected signed member origin')
    require(any(row['package']=='python3.10-minimal' for row in expected.values())
        and any(row['package']=='libc6' for row in expected.values()),'Both distinct origin chains actually required')
    return expected
