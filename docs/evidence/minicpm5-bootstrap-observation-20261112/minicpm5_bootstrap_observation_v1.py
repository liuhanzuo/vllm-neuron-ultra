"""Finite discovery helpers, not formal native authority; inert on import."""
import copy
import hashlib
import json
import os
from pathlib import Path
import re

ROOT=Path('/home/ubuntu/experiments/trn1-latest-models-20261031');Q=ROOT/'research'
CANDIDATES=dict(path=str(Q/'minicpm5-dual-python-bootstrap-observation-next-inputs-20261112-v1.json'),
 sha256='b1fbabd668bfc165679c23de8e1f29a8a86512c52e50e75727607072a7eb0088')
RETAINED_MANIFEST=dict(path=str(Q/'minicpm5-sdk310-snapshot-archive-source-manifest-20261112-v2.json'),
 sha256='7df3762d70e81e8d568f9bda131b97edffb8186921fcc8e3f3cf3b8120cd053e')
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
CHILD='minicpm5_bootstrap_observation_child_v1.py'
RUNNER='run_minicpm5_bootstrap_observation_v1.py'


def require(ok,message):
    if not ok:raise ValueError(message)


def encoded(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def ref2(ref):return {k:ref[k] for k in ('path','sha256')}


def validate(plan,review,plan_ref):
    require(plan['schema']=='minicpm5_bootstrap_observation_plan_v1'
      and plan['stage']=='inventory' and plan['runtime_bound'] is True
      and plan['automatic_successor_launch'] is False and plan['role_order']==list(ROLES)
      and plan['candidate_inputs']==CANDIDATES and plan['limits']==LIMITS
      and plan['artifact_limits']==CAPS and plan['budget']==BUDGET,'Exact discovery scope and original limits')
    require(plan['observation_only'] is True and plan['accepted_native_authority_created'] is False,
       'Plan itself cannot promote native authority')
    require(review['schema']=='minicpm5_bootstrap_observation_review_v1'
      and review['reviewer']=='/root' and review['passed'] is True and review['blockers']==[]
      and ref2(review['plan'])==ref2(plan_ref),'Exact independent root source review')
    for key in ('source_sha256','candidate_inputs','limits','artifact_limits','budget',
       'role_order','complete_current_serial_catalog','root_handoff','predecessor_exit','lock_policy'):
        require(review[key]==plan[key],'Exact reviewed '+key)
    require(plan['mini_scientific_constraints']==dict(native_checks=9,scientific_descriptor_rows=1284,
       absolute_wall_seconds=900,address_space_bytes=2147483648,single_file_bytes=2097152),
       'Original9/1284/900s/2GiB remain pending and unchanged')
    require(review['observation_only'] is True and review['accepted_native_authority_created'] is False,
       'Discovery review is not native authority')
    pins=plan['source_sha256'];require(type(pins) is dict and 1<=len(pins)<=SOURCE_CAP,'Bounded whole local source map')
    for name in (RUNNER,CHILD,HELPER,'minicpm5_bootstrap_observation_v1.py',
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


def candidates(read):
    table=read(CANDIDATES)
    require(table['source_metadata_only'] is True and table['runtime_admitted'] is False
       and table['distinct_native_candidates']==18 and table['distinct_native_candidate_bytes']==46364152,
       'Exact finite saved origin candidate table, never complete mapping authority')
    saved=read(table['input_refs']['composite_completed_metadata'])
    require(table['package_origins']['old_libc6']==saved['old_libc6']
       and table['package_origins']['new_eight']==saved['new_eight'],'Exact completed package chains retained')
    package_results={key:read(saved[key]['replay_result']) for key in ('old_libc6','new_eight')}
    for value in package_results.values():require(value['complete'] is True and value['package_content_authenticated'] is True,'Completed saved member replay')
    for row in list(table['sdk310']['native_candidates'].values())+list(table['sdk310']['pure_source_candidates'].values()):
        key='old_libc6' if row['package']=='libc6' else 'new_eight'
        require(row['saved_origin_result']==saved[key]['replay_result'],'Member bound to exact replay')
        providers=[p for p in package_results[key]['packages'] if (p['package'],p['version'])==(row['package'],row['version'])]
        require(len(providers)==1,'Unique saved package provider');inspection=providers[0]['inspection']
        member=inspection['tar']['targets'][row['path']]
        require(inspection['tar']['complete'] is True and row['archive_reference']==inspection['archive']
           and row['sha256']==member['sha256'] and row['bytes']==member['size']
           and row['archive_member']==member['terminal'],'Every candidate equals actual saved signed member')
    outer=read(table['input_refs']['outer_saved_origin'])
    require(table['outer312']['native_candidates']==outer['saved_native_candidates'],'Exact separately reviewed outer archive candidates')
    for path,row in table['outer312']['pure_source_candidates'].items():
        original=outer['exact_original_executable_and_pure_source_rows'][path]
        require(row['sha256']==original['sha256'] and row['bytes']==original['bytes'],'Exact outer pure-source archive member')
    require(len(table['sdk310']['native_candidates'])==17 and len(table['outer312']['native_candidates'])==8,
       'Original finite candidate counts')
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


def literal(helper_source,child_source):
    require(type(helper_source) is str and type(child_source) is str,'Exact source strings')
    text="helpers={'__name__':'mini_discovery_helpers','__file__':"+repr(str(Q/HELPER))+"}\nexec(compile("+repr(helper_source)+","+repr(str(Q/HELPER))+",'exec'),helpers)\n"
    text+="child_scope={'__name__':'mini_discovery_child','__file__':"+repr(str(Q/CHILD))+",'helpers':helpers}\nexec(compile("+repr(child_source)+","+repr(str(Q/CHILD))+",'exec'),child_scope)\nchild_scope['main']()\n"
    require(len(text.encode())<65536,'One bounded exact literal argv, no shell')
    return text


def argv_for(role,table,source,plan_ref,admission_ref,ready_fd,deadline):
    require(role in ROLES and type(ready_fd) is int and ready_fd>=0
      and type(deadline) in (int,float) and 0<deadline<float('inf'),'Exact role/fd/inherited absolute deadline')
    return [table[role]['python'],'-I','-S','-B','-X','pycache_prefix=/dev/null','-c',source,
       '--mini-bootstrap-observe',role,plan_ref['path'],plan_ref['sha256'],admission_ref['path'],
       admission_ref['sha256'],str(ready_fd),repr(deadline),hashlib.sha256(source.encode()).hexdigest()]
