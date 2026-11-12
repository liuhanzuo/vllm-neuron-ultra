"""Complete unapproved formal-entry candidate assembly, with no IO or launch.

prepare() reuses the frozen original metadata reconstruction through an
explicitly supplied original module. Root must import that historical module
only in its separately reviewed serial preparation context; this module never
imports it, acquires a lock, reads native/archive bytes, or approves a review.
assemble() is also callable with already reconstructed original metadata.
All returned artifact bytes are canonical and references are fully bound.
"""
import copy
import hashlib
import json
from pathlib import Path
import re

from research import prepare_minicpm5_observed_native_policies_v1 as policies
from research import minicpm5_sdk_mapping_nocache_launch_v2 as launch
from research import minicpm5_sdk_mapping_observer_logic_v1 as logic

ROOT,Q,require=policies.ROOT,policies.Q,policies.require
SELF=str(Q/'prepare_minicpm5_observed_formal_binding_v1.py')
ORIGINAL=str(Q/'prepare_minicpm5_pv_sdk_import_diagnostic_v1.py')
REQUEST_SCHEMA='minicpm5_observed_formal_binding_request_v1'
SOURCE_CAP=16*2**20
PENDING=['Independent root review of these exact bytes is pending']


def merge(first,second):
    result=copy.deepcopy(first)
    for path,sha in second.items():
        require(path not in result or result[path]==sha,'An original source pin cannot be replaced')
        result[path]=sha
    return result


def validate_request(request,read):
    require(type(request) is dict and set(request)=={'schema','original_request','current_handoff',
        'handoff_catalog_field','expected_predecessor_exit','source_manifest','artifact_prefix','attempt_id','unit'}
        and request['schema']==REQUEST_SCHEMA,'Explicit complete original history and candidate destinations')
    for name in ('original_request','current_handoff','expected_predecessor_exit','source_manifest'):
        launch.reference(request[name])
    old=read(request['original_request']);handoff=read(request['current_handoff'])
    require(old['schema']=='minicpm5_sdk_import_future_binding_request_v1','Unchanged original request schema')
    field=request['handoff_catalog_field']
    require(field in ('complete_current_serial_catalog','complete_current_serial_catalog_next'),
        'Explicit saved handoff catalog field; no discovery of a newer input')
    bound=handoff[field]
    require(bound==old['current_serial_catalog'] or
        (field.endswith('_next') and bound==old['current_serial_catalog']['path']),
        'Exact caller-selected saved handoff catalog')
    catalog=read(old['current_serial_catalog'])
    require(type(catalog['runs']) is list and catalog['runs'] and
        catalog['runs'][-1]['terminal']==request['expected_predecessor_exit'],
        'Explicit last ended catalog predecessor, no implicit queue advancement')
    # This is saved metadata only. Original runtime still checks every actual
    # owner, service, supplemental child, physical artifact and lock identity.
    read(request['expected_predecessor_exit'])
    prefix=Path(request['artifact_prefix'])
    require(prefix.parent==Q and str(prefix)==request['artifact_prefix'] and
        re.fullmatch('minicpm5-formal-import-[a-z0-9][a-z0-9-]{0,39}',prefix.name),
        'Fresh bounded explicit candidate artifact prefix')
    require(type(request['attempt_id']) is str and re.fullmatch('[a-f0-9]{32}',request['attempt_id'])
        and type(request['unit']) is str and re.fullmatch('trn1-mini-[a-z0-9-]{1,150}',request['unit']),
        'Explicit original descriptor attempt and unit')
    manifest=read(request['source_manifest'])
    require(manifest['schema']=='minicpm5_observed_formal_entry_source_manifest_v1'
        and set(manifest['new_runtime_source_sha256'])=={str(Q/name) for name in policies.NEW_SOURCES}
        and set(manifest['preparation_source_sha256'])=={SELF},'Exact reviewed integration and preparation sources')
    additions=merge(manifest['retained_source_sha256'],manifest['new_runtime_source_sha256'])
    additions=merge(additions,manifest['preparation_source_sha256'])
    require(all(path.startswith(str(ROOT)+'/') and Path(path).suffix=='.py' for path in additions),
        'Local Python bootstrap additions preserve original PV extension domain')
    return old,manifest,additions,catalog


def prepare(request,*,read,text,guard,original):
    """Root-supplied ordinary historical module; no import/admission here.

The canonical original prepare/reconstruct/merged_plan functions are used
unchanged. The caller owns any necessary serial/native import context. No AST
extraction, replacement historical module, fake lease, or authenticate call.
"""
    old,manifest,additions,_=validate_request(request,read)
    require(str(Path(original.__file__).resolve())==ORIGINAL,
        'Only the ordinary original preparation module supplied by root')
    expected=manifest['retained_source_sha256'].get(ORIGINAL)
    require(expected is not None and hashlib.sha256(text(ORIGINAL,expected)).hexdigest()==expected,
        'Immutable original preparation source retained')
    old=copy.deepcopy(old)
    extra={row['path']:row['sha256'] for row in old['additional_source_refs']}
    require(len(extra)==len(old['additional_source_refs']),'No duplicate caller additions')
    extra=merge(extra,additions)
    old['additional_source_refs']=[dict(path=path,sha256=sha) for path,sha in sorted(extra.items())]
    candidate=original.prepare(old,read=read,text=text,guard=guard)
    base=original.reconstruct_audit_base(old['independent_audit_exit'],old['recovery_binding'],read,text,guard)
    full=original.audited.merged_plan(base['plan'],candidate['pv_spec_candidate']['source_extension'])
    require(policies.history.digest(full['source_sha256'])==candidate['complete_source_map_sha256']
        and len(full['source_sha256'])==candidate['complete_source_files'],
        'Whole original reconstruction agrees with already checked preparation')
    guard()
    result=assemble(request,candidate,full,read=read,text=text)
    guard()
    return result


def assemble(request,candidate,full,*,read,text):
    """Pure complete candidate graph. Accepted reviews are never generated."""
    old,manifest,additions,catalog=validate_request(request,read)
    require(candidate['schema']=='minicpm5_sdk_import_future_binding_candidate_v1'
        and candidate['source_metadata_only'] is True
        and candidate['original_runtime_reconstruction_routes_agree'] is True
        and candidate['all_original_catalog_rows_retained'] is True
        and candidate['all_current_catalog_sources_retained'] is True
        and candidate['actual_freshness_or_physical_checks_performed'] is False
        and candidate['runtime_admitted'] is candidate['launch_authorized'] is False,
        'Original complete source-only reconstruction with unchanged physical obligations')
    require(type(full) is dict and {'source_sha256','source_contract'}<=set(full),
        'Complete original source plan and source contract are both retained')
    pins=full['source_sha256'];pv=copy.deepcopy(candidate['pv_spec_candidate'])
    require(policies.history.digest(pins)==candidate['complete_source_map_sha256']
        and len(pins)==candidate['complete_source_files']
        and pv['current_serial_catalog']==old['current_serial_catalog']
        and pv['independent_audit_exit']==old['independent_audit_exit']
        and pv['output_root']==old['output_root'] and pv['limits']==logic.LIMITS
        and pv['artifact_limits']==logic.CAPS
        and pv['worker']==str(Q/'minicpm5_pv_sdk_import_diagnostic_worker_v1.py'),
        'Original source map/catalog/audit/output/resource/scientific worker binding')
    require(all(pins.get(path)==sha for path,sha in additions.items())
        and all(pins.get(path)==sha for path,sha in pv['source_extension'].items()),
        'Complete original map retains every prospective source addition')
    for path,sha in additions.items():
        require(hashlib.sha256(text(path,sha)).hexdigest()==sha,'Current complete selected local source text')
    source_subset={key:full[key] for key in ('source_contract','source_sha256')}
    require(len(launch.encoded(source_subset))==candidate['complete_source_release_bytes']
        and len(launch.encoded(source_subset))+65536<=SOURCE_CAP,
        'Unchanged original16MiB release pipe and65536-byte reservation')
    policy_inputs=policies.build(read,pins,request['source_manifest'])
    artifacts={};prefix=request['artifact_prefix']

    def add(name,value,cap=2**20):
        raw=launch.encoded(value)
        require(0<len(raw)<=cap,'Bounded candidate metadata: '+name)
        path=prefix+'-'+name+'.json'
        reference=dict(path=path,sha256=hashlib.sha256(raw).hexdigest())
        artifacts[name]=dict(reference=reference,bytes=len(raw),value=copy.deepcopy(value))
        return copy.deepcopy(reference)

    def review(schema,spec,extension):
        return dict(schema=schema,reviewer=None,passed=False,blockers=list(PENDING),
            spec=copy.deepcopy(spec),source_extension=copy.deepcopy(extension))

    map_ref=add('complete-source-map',pins,SOURCE_CAP)
    contract_ref=add('complete-source-contract',full['source_contract'],SOURCE_CAP)
    policy_ref=add('native-policy-inputs',policy_inputs)
    outer_native=add('outer-native-policy-draft',policy_inputs['policies']['outer312'])
    sdk_native=add('sdk-native-policy-draft',policy_inputs['policies']['sdk310'])
    pv_ref=add('pv-spec',pv)
    pv_review=add('pv-review-draft',review('minicpm5_fixed_input_pv_serial_source_review_v1',pv_ref,pv['source_extension']))
    diagnostic=dict(schema='minicpm5_fixed_input_pv_sdk_import_diagnostic_spec_v1',
        pv_spec=pv_ref,pv_review=pv_review,worker=pv['worker'],limits=copy.deepcopy(pv['limits']),
        artifact_limits=copy.deepcopy(pv['artifact_limits']),source_extension=copy.deepcopy(pv['source_extension']))
    diagnostic_ref=add('diagnostic-spec',diagnostic)
    diagnostic_review=add('diagnostic-review-draft',review('minicpm5_fixed_input_pv_sdk_import_diagnostic_review_v1',
        diagnostic_ref,pv['source_extension']))
    entry_path=str(Q/'minicpm5_sdk_mapping_nocache_entry_v6.py')
    entry_ref=dict(path=entry_path,sha256=pins[entry_path])
    entry_bytes=text(entry_path,entry_ref['sha256'])
    require(0<len(entry_bytes)<65536,'Original exact literal entry size bound')
    bootstrap={path:sha for path,sha in additions.items() if path.endswith('.py')}
    require(0<len(bootstrap)<=2048,'Original bounded local bootstrap source set')

    def entry_plan(role,authority,module,successor):
        python=launch.previous.OUTER if role=='outer312' else launch.previous.SDK
        exe=('/home/ubuntu/.local/share/uv/python/cpython-3.12.16-linux-x86_64-gnu/bin/python3.12'
            if role=='outer312' else '/usr/bin/python3.10')
        return dict(schema='minicpm5_source_entry_plan_v1',role=role,python=python,executable=exe,
            native_authority=dict(role=role,python=python,**authority),bootstrap_sources=copy.deepcopy(bootstrap),
            entry_sha256=entry_ref['sha256'],entry_module='research.'+module,
            entry_path=str(Q/(module+'.py')),successor_input=successor)

    sdk=entry_plan('sdk310',sdk_native,'minicpm5_sdk_mapping_worker_source_entry_v5','admission:self')
    sdk_ref=add('sdk-entry-template',sdk)
    outer_spec=dict(schema='minicpm5_mapping_source_outer_spec_v1',attempt_id=request['attempt_id'],unit=request['unit'],
        diagnostic_spec=diagnostic_ref,diagnostic_review=diagnostic_review,complete_source_map=map_ref,
        sdk_entry_template=sdk_ref,sdk_entry_source=entry_ref,limits=copy.deepcopy(logic.LIMITS),
        artifact_limits=copy.deepcopy(logic.CAPS),source_extension=copy.deepcopy(bootstrap),
        sdk_import_paths=['/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages'])
    outer_ref=add('outer-spec',outer_spec)
    outer_review=add('outer-review-draft',review('minicpm5_mapping_source_outer_review_v1',outer_ref,bootstrap))
    outer=entry_plan('outer312',outer_native,'minicpm5_sdk_mapping_outer_source_entry_v5',
        json.dumps(dict(spec=outer_ref,review=outer_review),sort_keys=True,separators=(',',':')))
    plan_ref=add('outer-entry-plan',outer)
    launch.separate_authorities(outer,sdk)
    # Validates canonical literal/plan and exact role metadata; no process call.
    launch.launch(entry_bytes,entry_ref,outer,plan_ref)
    return dict(schema='minicpm5_observed_formal_binding_candidate_v1',request=copy.deepcopy(request),
        artifacts=artifacts,original_preparation=copy.deepcopy(candidate),
        complete_source_map=map_ref,complete_source_contract=contract_ref,native_policy_inputs=policy_ref,
        outer_entry_plan=plan_ref,entry_source=entry_ref,catalog_rows=len(catalog['runs']),
        full_original_source_contract_retained=True,full_original_history_retained=True,
        nonlocal_pure_sources_remain_in_separate_signed_policies=True,
        review_artifacts=['pv-review-draft','diagnostic-review-draft','outer-review-draft',
            'outer-native-policy-draft','sdk-native-policy-draft'],
        root_must_rebind_hashes_after_independent_review=True,
        source_metadata_only=True,source_only_assembly_is_not_original_runtime_authentication=True,
        current_native_bytes_read_here=False,current_physical_checks_performed=False,
        accepted_native_authority_created=False,runtime_admitted=False,launch_authorized=False,
        native_ABI_qualified=False,full_model_qualified=False,new_speedup_claim=False,
        original_nine_native_checks_pending=9,original_scientific_rows_still_pending=1284)
