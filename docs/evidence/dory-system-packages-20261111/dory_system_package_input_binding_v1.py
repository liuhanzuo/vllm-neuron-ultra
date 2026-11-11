"""Authenticate two fixed saved JSON inputs before finite metadata queries."""
from pathlib import PurePosixPath
import dory_serial_stage_contract_v2 as c
import dory_descriptor_inventory_runtime_v2 as runtime
import dory_system_package_metadata_v1 as collector

WORK_KIND='dory_system_package_metadata_only_v1'
INPUTS=dict(external=dict(collector.EXTERNAL),gap_classification=dict(
    path=str(c.Q/'dory-inventory-r5-gap-classification-20261111-v1.json'),bytes=22719,
    sha256='e902d47f1e0e580025480a0e32da7e75db3b603548e1915167722e113e9b1d03'))
SOURCE_PINS={str(c.Q/'dory_system_package_metadata_v1.py'):'ddcf8273ca59b04a4b08e3afed564f9f78ec28cd206f0d60ae0c7dc1c9e4c889',
    str(c.Q/'dory_native_inventory_finish_v4.py'):'0d69e2bd6cb648653e10df3e1d969b00555edb43010c85ee0fb9abc318b5e07e'}


def validate(plan,review,plan_ref):
    runtime.validate(plan,review,plan_ref)
    c.require(plan.get('work_kind')==review.get('work_kind')==WORK_KIND
        and plan.get('metadata_inputs')==review.get('metadata_inputs')==INPUTS,'Exact independently reviewed metadata-only task and inputs')
    c.require(plan.get('external_authorities') is None,'No package authority acceptance or inventory payload work')
    for name,digest in SOURCE_PINS.items():c.require(plan['source_sha256'].get(name)==digest,'Exact accepted collector and command-budget helper')
    return plan


def derive(external,gaps):
    """Only saved strings/records; no library stat, resolve, open or query."""
    counts=dict(system_package_files=11,vllm_missing_loader_edges=13,torch_ambiguous_libgomp_edges=2,cpython_timestamp_caches=4)
    c.require(gaps['schema']=='dory_inventory_r5_gap_classification_v1' and gaps['counts']==counts
        and gaps['total_unresolved']==30 and gaps['unresolved_closed_here']==0
        and gaps['inputs']['external']==INPUTS['external'],'Exact saved gap classification and external evidence binding')
    c.require(all(gaps[k] is False for k in ('actual_loaded_selection_established','complete_authenticated_inventory','native_ABI_qualified','full_model_qualified','new_speedup_claim')),'No qualification promoted by metadata stage')
    c.require(len(external['unresolved'])==30 and external['actual_loaded_maps_pending'] is True,'Original unresolved external context')
    files={r['reference']['path']:r for r in external['files']}
    c.require(len(files)==len(external['files'])==24,'Unique original external file rows')
    system={r['path'] for r in external['unresolved'] if r['reason']=='independent_system_package_bytes_not_yet_bound'}
    declared={r['path']:r for r in gaps['system_package_files']}
    c.require(len(system)==len(declared)==len(gaps['system_package_files'])==11 and system==set(declared),'All eleven exact system paths')
    aliases={path:{path} for path in system}
    for path in system:
        row=files[path];c.require(row['authority']=='observed_hash_only' and row['supplied_authority'] is None
            and row['reference']==declared[path]['observed_reference'],'Original observed library refs, not independent package authority')
    candidates=[r for r in external['unresolved'] if r['reason']=='ambiguous_or_unresolved_loader_search' and r.get('needed')=='libgomp.so.1']
    c.require(len(candidates)==len(gaps['torch_ambiguous_libgomp_edges'])==2,'Both saved libgomp ambiguity records')
    extra=set(collector.TARGETS)-system;c.require(len(extra)==1,'Exactly one additional saved system libgomp candidate')
    gomp=next(iter(extra));aliases[gomp]={gomp}
    for issue,declared_issue in zip(candidates,gaps['torch_ambiguous_libgomp_edges']):
        c.require(issue['path']==declared_issue['path'] and issue['needed']==declared_issue['needed']
            and issue['candidates']==declared_issue['saved_candidates'] and gomp in issue['candidates']
            and len(issue['candidates'])==2 and declared_issue['actual_selected_path'] is None,'Unselected saved system candidate, not an inferred load choice')
    for row in external['files']:
        link=row['link']
        if link['resolved'] in aliases:aliases[link['resolved']].add(link['requested'])
        for dep in row['dependencies']:
            resolved=dep['resolved_candidates']
            if len(resolved)==1 and resolved[0] in aliases:aliases[resolved[0]].update(dep['candidates'])
            if dep['name']=='libgomp.so.1' and gomp in resolved:
                # Both names were saved, but the correspondence is only a
                # proposed metadata query: live link observations must prove it.
                aliases[gomp].update(p for p in dep['candidates'] if p.startswith('/lib/'))
    derived={path:tuple(sorted(names)) for path,names in aliases.items()}
    c.require(derived==collector.TARGETS and tuple(sorted({p for names in derived.values() for p in names}))==collector.PATHS,'Exact twelve-target/twenty-five-name provenance')
    c.require(len(collector.PATHS)==25 and len(collector.PACKAGES)==6 and len(collector.COMMANDS)==32,'Exact fixed finite collector')
    return dict(schema='dory_system_package_authenticated_inputs_v1',work_kind=WORK_KIND,inputs=INPUTS,
        exact_saved_targets=12,exact_saved_query_names=25,fixed_package_candidates=6,fixed_commands=32,
        query_name_provenance_verified=True,actual_alias_mapping_verified=False,
        package_content_authenticated=False,loader_selection_established=False,native_ABI_qualified=False)


def authenticate(guard):
    guard();external=c.load(INPUTS['external'],c.MIB)
    guard();gaps=c.load(INPUTS['gap_classification'],c.MIB)
    result=derive(external,gaps);guard();return result


def validate_admission(value,authenticated):
    c.require(value.get('work_kind')==WORK_KIND and value.get('metadata_input_binding')==authenticated,'Actual parent admission binds authenticated metadata scope')


def save_collected(out,value,authenticated,plan_ref,admission_ref,guard):
    c.require(value['schema']=='dory_system_package_metadata_v1' and value['saved_external_evidence']==INPUTS['external']
        and value['library_payload_bytes_read']==0,'Finite metadata collector result only')
    for key in ('package_archive_authenticated','loader_selection_established','native_ABI_qualified','full_model_qualified','new_speedup_claim','actual_process_exit_claimed'):
        c.require(value[key] is False,'No content/model/exit qualification')
    guard()
    result=c.save(out/'metadata-result.json',dict(schema='dory_system_package_metadata_receipt_v1',
        work_kind=WORK_KIND,input_binding=authenticated,plan=plan_ref,admission=admission_ref,metadata=value,
        native_ABI_qualified=False,full_model_qualified=False,new_speedup_claim=False,actual_process_exit_claimed=False),8*c.MIB)
    guard();return result
