"""Finite discovery helpers, not formal native authority; inert on import."""
import copy
import hashlib
import json
import os
from pathlib import Path
import re

ROOT=Path('/home/ubuntu/experiments/trn1-latest-models-20261031');Q=ROOT/'research'
CANDIDATES=dict(path=str(Q/'minicpm5-dual-python-bootstrap-observation-next-inputs-20261112-v2.json'),
 sha256='a867df704bab8e8ebbd08300cb3eeba999066810800fe03a39bd84417d65a82a')
RETAINED_MANIFEST=dict(path=str(Q/'minicpm5-bootstrap-observation-source-manifest-20261112-v2.json'),
 sha256='05bbd33755eb64e37c16cb5b7c66ec0d890fea0217fe8f1af9287c93b65f7466')
ROLES=('sdk310','outer312')
ENTRIES=dict(sdk310='research.minicpm5_sdk_mapping_worker_source_entry_v4',
 outer312='research.minicpm5_sdk_mapping_outer_source_entry_v4')
LIMITS=dict(address_space_bytes=2147483648,cpu_seconds=600,hard_cpu_seconds=605,
 wall_seconds=900,output_file_bytes=2097152,cpu=6,nice=19)
CAPS=dict(allocated_bytes=33554432,metadata_bytes=8388608,one_file_bytes=2097152,
 raw_bytes=12288000,raw_files=4392)
BUDGET=dict(address_space_bytes=2147483648,cpu_seconds=600,wall_seconds=900,
 single_file_bytes=2097152,output_bytes=33554432,log_bytes=1048576,cpus=[6],nice=19,
 root_floor_bytes=28*2**30,nvme_floor_bytes=30*2**30,memory_available_floor_bytes=2**30)
MAP_CAP=1048576
SOURCE_CAP=256
SOURCE_BYTES=8*2**20
HELPER='minicpm5_sdk_mapping_nocache_entry_v5.py'
CHILD='minicpm5_bootstrap_observation_child_v2.py'
TYPES='minicpm5_bootstrap_types_source_v1.py'
RUNNER='run_minicpm5_bootstrap_observation_v2.py'


def require(ok,message):
    if not ok:raise ValueError(message)


def encoded(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def ref2(ref):return {k:ref[k] for k in ('path','sha256')}


def validate(plan,review,plan_ref):
    require(plan['schema']=='minicpm5_bootstrap_observation_plan_v2'
      and plan['stage']=='inventory' and plan['runtime_bound'] is True
      and plan['automatic_successor_launch'] is False and plan['role_order']==list(ROLES)
      and plan['candidate_inputs']==CANDIDATES and plan['limits']==LIMITS
      and plan['artifact_limits']==CAPS and plan['budget']==BUDGET,'Exact discovery scope and original limits')
    require(plan['source_component_reviews']==COMPONENT_REVIEWS and plan['supplemental_origin_receipts']==SUPPLEMENTAL_RECEIPTS,
       'Exact accepted source components and independently completed supplemental origins')
    require(plan['observation_only'] is True and plan['accepted_native_authority_created'] is False,
       'Plan itself cannot promote native authority')
    require(review['schema']=='minicpm5_bootstrap_observation_review_v2'
      and review['reviewer']=='/root' and review['passed'] is True and review['blockers']==[]
      and ref2(review['plan'])==ref2(plan_ref),'Exact independent root source review')
    for key in ('source_sha256','candidate_inputs','limits','artifact_limits','budget',
       'role_order','complete_current_serial_catalog','root_handoff','predecessor_exit','lock_policy',
       'source_component_reviews','supplemental_origin_receipts'):
        require(review[key]==plan[key],'Exact reviewed '+key)
    require(plan['mini_scientific_constraints']==dict(native_checks=9,scientific_descriptor_rows=1284,
       absolute_wall_seconds=900,address_space_bytes=2147483648,single_file_bytes=2097152),
       'Original9/1284/900s/2GiB remain pending and unchanged')
    require(review['observation_only'] is True and review['accepted_native_authority_created'] is False,
       'Discovery review is not native authority')
    pins=plan['source_sha256'];require(type(pins) is dict and 1<=len(pins)<=SOURCE_CAP,'Bounded whole local source map')
    for name in (RUNNER,CHILD,HELPER,TYPES,'minicpm5_bootstrap_observation_v2.py',
       'minicpm5_sdk_mapping_native_closure_v2.py',
       'minicpm5_sdk_mapping_outer_source_entry_v4.py','minicpm5_sdk_mapping_worker_source_entry_v4.py'):
        require(str(Q/name) in pins,'Every discovery/formal-entry local source pinned')
    out=Path(plan['output_root'])
    require(out.parent==Path('/mnt/nvme') and out.name.startswith('dory-serial-inventory-minicpm5-sdk310-bootstrap-'),
       'Fresh explicit discovery namespace')
    return plan


def verify_sources(pins,read,guard=lambda:None):
    require(1<=len(pins)<=SOURCE_CAP,'Finite source closure');total=0
    for path,sha in pins.items():
        guard();require(Path(path).is_relative_to(ROOT) and path.endswith('.py'),'Local source only')
        observed,raw=read(path,2*2**20);total+=len(raw)
        require(observed['sha256']==sha and total<=SOURCE_BYTES,'Exact complete source bytes, unchanged pins')
    return total


COMPONENT_REVIEWS=dict(
    elf=dict(path=str(Q/'minicpm5-sdk-mapping-native-closure-v2-root-source-review-20261112-v1.json'),
             sha256='b84b1aa939a903f4a5d29cc91ec43de3939d941ba4f45ad83c9f66759d181959'),
    types=dict(path=str(Q/'minicpm5-bootstrap-types-root-source-review-20261112-v1.json'),
               sha256='01ec35166cc24b4e4c2c3f8ce2bd6a26282cd68d761ecd392eca7a621e41b5cf'))
COMPONENT_SOURCES=dict(
    elf=dict(path=str(Q/'minicpm5_sdk_mapping_native_closure_v2.py'),
             sha256='445a06eeaf4707844e24c06a589950d2eaedb9cd3abd333e93b1b4a41ce7e626'),
    types=dict(path=str(Q/TYPES),sha256='c1c95da4aa85b71518c9dd5dd3e077a1433cd483786b86b5213186a7091529b3'))
SUPPLEMENTAL_RECEIPTS=dict(
    actual_exit=dict(path=str(Q/'minicpm5-sdk310-extra-members-r1-actual-exit-20261112-v1.json'),
                     sha256='803840bb8334cc4980396ef2b88f3510fb1dbe3d9659a5162fd885dc67bea6f9'),
    recorder_actual_exit=dict(path=str(Q/'minicpm5-sdk310-extra-members-r1-recorder-root-actual-exit-20261112-v1.json'),
                              sha256='67a4d3147994310cc84061b3cb5be78f38dec513c122f76c3838288d8391c689'))


def component_reviews(read,pins):
    for key,ref in COMPONENT_REVIEWS.items():
        reviewed=read(ref);source=COMPONENT_SOURCES[key]
        require(reviewed['reviewer']=='/root' and reviewed['passed'] is True and reviewed['blockers']==[]
                and ref2(reviewed['source' if key=='elf' else 'component'])==source
                and pins.get(source['path'])==source['sha256'],'Exact previously accepted ELF/types sources')


def supplemental(read):
    from research import minicpm5_sdk310_extra_members_binding_v1 as extra
    # Metadata-only use of the existing independently checked result validator.
    ended=read(SUPPLEMENTAL_RECEIPTS['actual_exit']);recorded=read(SUPPLEMENTAL_RECEIPTS['recorder_actual_exit'])
    require(ended['schema']=='minicpm5_sdk310_extra_members_stage_actual_exit_v1'
            and ended['actual_tool']==dict(kind='tool_session',session_id=89995,terminal_chunk='cc05c8',exit_code=0)
            and ended['stage_success_observed'] is ended['owners_gone'] is ended['child_group_empty'] is True
            and ended['events']==4 and ended['mode']=='supplemental_existing_archive_replay',
            'Actual completed supplemental archive/member run, never source-only acceptance')
    require(recorded['schema']=='minicpm5_sdk310_extra_members_r1_recorder_root_actual_exit_v1'
            and recorded['reviewer']=='/root' and recorded['actual_tool']==dict(kind='tool_call',terminal_chunk='4d4770',exit_code=0)
            and ref2(recorded['result'])==SUPPLEMENTAL_RECEIPTS['actual_exit'] and recorded['owner']==ended['recorder']
            and recorded['owner_gone'] is recorded['all_three_owners_gone'] is True
            and recorded['original_lock_nonblocking_acquired_and_released'] is True
            and recorded['complete_four_events_revalidated'] is recorded['full_saved_supplemental_result_revalidated'] is True
            and recorded['original_eight_packages_twelve_targets_unchanged'] is True,
            'Recorder actual exit and independent full output/result/journal/drain audit')
    require(ended['result']==ended['evidence']['extra-members-result.json'], 'Complete supplemental result receipt binding')
    result=read(ended['result']);request=extra.helper.prepare(load=lambda ref,cap:read(ref))
    extra.validate_result(result,request)
    require(recorded['new_members']==[dict(path=row['path'],**row['member']) for row in result['comparisons']],
            'The two exact member hashes and sizes independently revalidated')
    for value in (ended,recorded):
        require(all(value[key] is False for key in ('native_ABI_qualified','full_model_qualified','new_speedup_claim')),
                'No performance or native qualification inherited from package verification')
    return ended,result


def candidates(read):
    from research import minicpm5_bootstrap_observation_v1 as previous
    old=previous.candidates(read);ended,result=supplemental(read)
    expected=copy.deepcopy(old)
    expected['schema']='minicpm5_dual_python_bootstrap_observation_candidates_v2'
    table=read(CANDIDATES)
    expected['input_refs']['previous_candidates']=table['input_refs']['previous_candidates']
    require(ref2(expected['input_refs']['previous_candidates'])==previous.CANDIDATES
            and read(expected['input_refs']['previous_candidates'])==old,'Exact old candidate table and full reference')
    for name,key in (('supplemental_member_actual_exit','actual_exit'),('supplemental_member_recorder_actual_exit','recorder_actual_exit')):
        expected['input_refs'][name]=table['input_refs'][name]
        require(ref2(expected['input_refs'][name])==SUPPLEMENTAL_RECEIPTS[key]
                and read(expected['input_refs'][name])==read(SUPPLEMENTAL_RECEIPTS[key]),
                'Exact complete supplemental terminal provenance including optional full size')
    expected['input_refs']['supplemental_member_result']=ended['result']
    expected['package_origins']['sdk310_supplemental_stdlib']=dict(result=ended['result'],
        actual_exit=expected['input_refs']['supplemental_member_actual_exit'],
        recorder_actual_exit=expected['input_refs']['supplemental_member_recorder_actual_exit'])
    for row in result['comparisons']:
        path=row['path'];require(path not in expected['sdk310']['native_candidates'],'Only genuinely additive origins')
        expected['sdk310']['native_candidates'][path]=dict(path=path,bytes=row['member']['bytes'],sha256=row['member']['sha256'],
            package=row['package'],version=row['version'],archive_member=row['member']['terminal'],
            archive_reference=result['inspection']['archive'],saved_origin_result=ended['result'],
            actual_loaded=None,current_bytes_observed_here=False)
    expected['distinct_native_candidates']+=2
    expected['distinct_native_candidate_bytes']+=sum(row['member']['bytes'] for row in result['comparisons'])
    require(table==expected and table['distinct_native_candidates']==20
            and table['distinct_native_candidate_bytes']==46398112
            and len(table['sdk310']['native_candidates'])==19, 'Original table byte values preserved; exact two accepted origins added')
    return table


def collect_content(table,reader,closure,guard):
    """Only finite known candidate payloads; unknown mapped files are never opened."""
    refs={}
    for role in ROLES:
        for path,row in {**table[role]['native_candidates'],**table[role]['pure_source_candidates']}.items():
            ref={k:row[k] for k in ('path','sha256','bytes')}
            require(path not in refs or refs[path]==ref,'Shared candidates have identical independent origins');refs[path]=ref
    files={};ffi_paths={table[role]['ffi_candidate'] for role in ROLES}
    for path,ref in refs.items():
        guard();raw,observed=reader.read(ref,64*2**20)
        row=dict(reference=ref,identity=observed['identity'])
        if path.endswith('.py'):row['pure_source']=True
        else:
            for key,fn in [('elf',closure.parse_elf64)]+([('ffi',closure.ffi_definitions)] if path in ffi_paths else []):
                try:row[key]=fn(raw)
                except (ValueError,KeyError,UnicodeError) as error:row[key+'_failure']=dict(type=type(error).__name__,message=str(error))
        files[path]=row
    guard()
    return dict(schema='minicpm5_bootstrap_candidate_content_v1',files=files,
       current_candidate_bytes_authenticated=True,actual_loader_selection_observed=False,
       accepted_native_authority_created=False,full_model_qualified=False,new_speedup_claim=False)


def summarize_maps(text,role,table,content,closure,resolve):
    """Observe unknown paths/dev/inode without reading their content or promoting them."""
    rows=closure.parse_maps(text);known=table[role]['native_candidates'];selected={};gaps=[]
    for row in rows:
        path=row['path'];native=path.startswith('/') and ('x' in row['permissions'] or re.search(r'\.so(?:\.|$)',Path(path.removesuffix(' (deleted)')).name))
        if native:
            if path not in known:gaps.append(dict(reason='missing_saved_content_origin',mapping=row));continue
            file=content['files'][path];identity=file['identity']
            if row['device']!=[os.major(identity['st_dev']),os.minor(identity['st_dev'])] or row['inode']!=identity['st_ino']:
                gaps.append(dict(reason='known_candidate_mapping_identity_mismatch',mapping=row));continue
            if 'elf' not in file:gaps.append(dict(reason='known_candidate_ELF_unproved',mapping=row));continue
            selected[path]=file
        elif 'x' in row['permissions'] and path not in ('[vdso]','[vsyscall]'):
            gaps.append(dict(reason='uncovered_executable_mapping',mapping=row))
    evidence=None
    if not gaps:
        try:
            evidence=closure.mapped_closure({p:f['elf'] for p,f in selected.items()},
                {p:dict(device=[os.major(f['identity']['st_dev']),os.minor(f['identity']['st_dev'])],inode=f['identity']['st_ino']) for p,f in selected.items()},
                text,table[role]['executable'],'/usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2',resolve)
        except (ValueError,KeyError) as error:gaps.append(dict(reason='actual_dependency_closure_unproved',message=str(error)))
    return dict(selected_known_paths=sorted(selected),gaps=gaps,known_selected_closure=evidence,
       accepted_native_authority_created=False)


def source_execution_gaps(role,observation,sources):
    require(role in ROLES,'Exact observed role')
    if role=='sdk310':
        require(observation is None,'No outer types tracker imposed on SDK');return []
    require(type(observation) is dict and observation['schema']=='minicpm5_outer_types_source_observation_v1'
            and observation['current_source_bytes_verified'] is observation['loaded_provider_metadata_matches'] is True
            and observation['preexisting_module_reloaded'] is False
            and observation['interpreter_startup_cause_established'] is False,
            'Honest current source/module metadata and explicit earliest timing observation')
    pin=dict(path='/home/ubuntu/.local/share/uv/python/cpython-3.12.16-linux-x86_64-gnu/lib/python3.12/types.py',
             bytes=10993,sha256='345474ef027a1273f353da9bdc1f7c18f65335e72e681bcc0376774cc51f2405')
    require(observation['current_source_reference']==pin
            and all(observation[key] is False for key in ('accepted_native_authority_created','full_model_qualified','new_speedup_claim')),
            'Only the exact previously reviewed types source; no authority promotion')
    path=pin['path'];executed=observation['executed_through_pinned_loader']
    require(type(executed) is bool and observation['original_execution_provenance_gap'] is (not executed),
            'Source execution gap is independent of mapped native content gaps')
    if executed:
        require(observation['first_checkpoint']['present'] is False and observation['initial_load_timing']=='after_first_checkpoint'
                and observation['execution_receipt'] is not None and sources.get(path)==observation['execution_receipt']
                and all(observation['execution_receipt'][key]==pin[key] for key in ('bytes','sha256')),
                'Full successful pinned-loader execution receipt must actually exist')
        return []
    require(observation['first_checkpoint']['present'] is True and observation['initial_load_timing']=='before_first_checkpoint'
            and observation['execution_receipt'] is None and path not in sources,
            'Already-loaded module cannot be promoted by current-file correlation')
    return [dict(path=path,reason='preexisting_module_execution_source_unproved')]


def literal(helper_source,child_source,types_source):
    require(all(type(value) is str for value in (helper_source,child_source,types_source)),'Exact source strings')
    text="helpers={'__name__':'mini_discovery_helpers','__file__':"+repr(str(Q/HELPER))+"}\nexec(compile("+repr(helper_source)+","+repr(str(Q/HELPER))+",'exec'),helpers)\n"
    text+="types_helpers={'__name__':'mini_types_source_helpers','__file__':"+repr(str(Q/TYPES))+"}\nexec(compile("+repr(types_source)+","+repr(str(Q/TYPES))+",'exec'),types_helpers)\n"
    text+="child_scope={'__name__':'mini_discovery_child','__file__':"+repr(str(Q/CHILD))+",'helpers':helpers,'types_helpers':types_helpers}\nexec(compile("+repr(child_source)+","+repr(str(Q/CHILD))+",'exec'),child_scope)\nchild_scope['main']()\n"
    require(len(text.encode())<65536,'One bounded exact literal argv, no shell')
    return text


def argv_for(role,table,source,plan_ref,admission_ref,ready_fd,deadline):
    require(role in ROLES and type(ready_fd) is int and ready_fd>=0
      and type(deadline) in (int,float) and 0<deadline<float('inf'),'Exact role/fd/inherited absolute deadline')
    return [table[role]['python'],'-I','-S','-B','-X','pycache_prefix=/dev/null','-c',source,
       '--mini-bootstrap-observe',role,plan_ref['path'],plan_ref['sha256'],admission_ref['path'],
       admission_ref['sha256'],str(ready_fd),repr(deadline),hashlib.sha256(source.encode()).hexdigest()]
