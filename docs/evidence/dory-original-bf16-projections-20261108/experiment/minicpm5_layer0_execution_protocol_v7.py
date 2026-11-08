"""Original88 successor admission contracts; source import never runs admission.

Explicit reuse of generic old transport and exact modern source-identity helpers.
Never invokes protocol4.bundle/validate_spec or native160 lifecycle admission as
an original88 substitute. No actual spec/catalog/output is created here.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import minicpm5_layer0_execution_protocol_v4 as low
import minicpm5_python_trace_lifecycle_v4 as identities
import historical_b3_payload_partition_v1 as history
from research import prepare_minicpm5_original88_python_trace_20261107_v3 as plan_source
ROOT=low.ROOT;GIB=low.GIB;MIB=low.MIB;STAGES=low.STAGES;ARMS=low.ARMS;RESOURCE=low.RESOURCE
LOCK=low.LOCK;GROUP_PREFIX=low.GROUP_PREFIX
require=low.require;digest=low.digest;clear_window=low.clear_window;read=low.read;sha=low.sha;text=low.text
command=low.command;properties=low.properties;proc=low.proc;gone=low.gone;empty_group=low.empty_group
terminal=low.terminal;invocation=low.invocation;lock_proof=low.lock_proof;active_heavy=low.active_heavy;user_bus=low.user_bus
stage_argv=low.stage_argv;rank_argv=low.rank_argv;environment=low.environment;elastic_run_id=low.elastic_run_id
service_command=low.service_command;service_contract=low.service_contract;service_identity=low.service_identity;phase_drain=low.phase_drain
controller_argv=low.controller_argv;controller_command=low.controller_command;controller_identity=low.controller_identity
once=identities.once
MINIMUM_CATALOG=dict(path=str(ROOT/'research/root-post-storage-2c33-complete-predecessors-20261107-v1.json'),sha256='19bb8130c581f08220b32a5dfc76a9fa4186cf99f58bdea9a6bd914a0926770b')
SMOL_SPEC=dict(path=str(ROOT/'research/smollm3-sio6-execution-spec-20261107-v1.json'),sha256='0bc182c3eb644a93c062295507055cc633ec81a8dacb51dde7b579a4a11cbf63')
SMOL_LIVE=dict(path=str(ROOT/'research/smollm3-observer-external-root-live-sio6-20261107-v1.json'),sha256='aeef358260a4e8bc97e7e37a605aad5f4b9e5633a0e665711be196e4f6adb36f')
FINAL_CATALOG=dict(path=str(ROOT/'research/root-post-smol-sio6-complete-predecessors-20261107-v5.json'),sha256='5c3221f3ad5462171e443a21d9b97cbb275301fe9db63ee24508ff078dd5af63')
FINAL_CATALOG_REVIEW=dict(path=str(ROOT/'research/minicpm5-original88-predecessor-catalog-root-review-20261107-v1.json'),sha256='ca4f21efd1c84c1a1c48cf0bb5caf557f906ca7541ed74c4f610717b507bae2b')
SMOL_FINALIZED=dict(path=str(ROOT/'research/smollm3-sio6-root-finalization-execution-20261107-v1.json'),sha256='246a674d2aa75ccd455b5155f75fd8ac070c4e877f6e65b2c3a0901ba0e9f07d')
SMOL_TERMINAL=dict(path=str(ROOT/'research/smollm3-observer-sio6-root-terminal-audit-20261107-v5.json'),sha256='621aa77d5371c047faa537e225db0c9f02c027001268ba02a31d1dd1972bd835')
SMOL_EXIT=dict(path=str(ROOT/'research/smollm3-observer-external-exit-root-sio6-20261107-v1.json'),sha256='ee8201fe9e6c70697e07e470150c0a90f48457f59f1b50b17c2e34027651ef8b')
DESIGN_ACCEPT=dict(path=str(ROOT/'research/minicpm5-original88-python-trace-runtime-root-design-review-20261107-v1.json'),sha256='9544201b0f86c2365918144034b38da78e4de6ae77c1768ad68755b80c9fb9c7')
NATIVE160=dict(
 audit=dict(path=str(ROOT/'research/minicpm5-python-trace-post-resume-20261107-v3-audit.json'),sha256='162ed1803846ceb39898f6d23e4008e0fe4ef9efb4bf654c6897d7ad664c8825'),
 finalization=dict(path=str(ROOT/'research/minicpm5-python-trace-resume-root-finalization-20261107-v3.json'),sha256='f880c7d3863292b4005473a5c08ea1b9dee5f8e437a23a1bbace3c49fa3a624d'),
 spec=dict(path=str(ROOT/'research/minicpm5-python-trace-resume-spec-20261107-v3.json'),sha256='6ea3e3cef5966c190cb869646a98ea24d93f9295ca7a736f5d0be303afd2a1e7'),
 review=dict(path=str(ROOT/'research/minicpm5-python-trace-resume-root-execution-review-20261107-v3.json'),sha256='cbb7b672ab8ad09c3b07f728024c0e6a8d4c9e468e34794e23576a0ed80bd349'),
 release=dict(path=str(ROOT/'minicpm5-python-trace-control-v3/root-release.json'),sha256='daeb73dc22163fd849f65944a5a93cb4c51da9b4e0323e57c8249868fffa9575'))
SUCCESSORS=('minicpm5_layer0_execution_protocol_v7.py','minicpm5_layer0_worker_v7.py','minicpm5_layer0_external_guard_v7.py',
 'minicpm5_original88_python_trace_result_parser_v3.py','research/run_minicpm5_layer0_root_v7.py',
 'minicpm5_layer0_external_supervisor_v7.py','research/prepare_minicpm5_original88_python_trace_20261107_v3.py',
 'research/audit_minicpm5_original88_python_trace_root_20261107_v3.py')

def read_ref(reference):return read(reference['path'],reference['sha256'])

def source_key(spec,path):return identities.source_key(spec,path)
def source_sha(spec,path):return identities.source_sha(spec,path)

def verify_sources(spec):
    for path,wanted in spec['source_sha256'].items():
        clear_window();before=source_key(spec,path)
        require(source_sha(spec,path)==wanted and source_key(spec,path)==before,'Frozen execution source changed: '+path)
    return True

def validate_spec(spec):
    require(spec['schema']=='minicpm5_original88_python_trace_execution_spec_v1'and spec['root_finalized']is True,'New root-finalized original88 execution schema')
    require(spec['stages']==list(STAGES)and spec['resources']==RESOURCE,'Original three phases/resources')
    require(spec['output_cap_bytes']==2*GIB and spec['control_cap_bytes']==32*MIB and spec['root_floor_bytes']==28*GIB and
            spec['admission_required_free_bytes']==30*GIB+32*MIB,'Exact original30GiB+32MiB admission/28GiB floor')
    require(spec['automatic_successor_launch']is False,'No automatic successor')
    for field,wanted in [('worker',ROOT/SUCCESSORS[1]),('supervisor',ROOT/SUCCESSORS[4]),('external_supervisor',ROOT/SUCCESSORS[5])]:
        require(spec[field]==str(wanted),'Exact successor executable '+field)
    require(spec['native_python']=='/home/ubuntu/venvs/model_reference_2026/bin/python'and
            spec['cpu_python']=='/home/ubuntu/venvs/vllm_neuron/bin/python'and spec['controller_python']=='/usr/bin/python3','Original interpreters')
    require(spec['science_handoff']==plan_source.SCIENCE and spec['scientific_review']==plan_source.ACCEPT and
            spec['science_independent_review']==plan_source.INDEPENDENT and spec['selected_validator']==plan_source.VALIDATOR and
            spec['interpreter_contracts']==plan_source.CONTRACTS and spec['runtime_design_acceptance']==DESIGN_ACCEPT,'Accepted science2/validator2/contracts/runtime design')
    require(spec['minimum_predecessor_catalog']==MINIMUM_CATALOG and spec['native160']==NATIVE160,'Fixed minimum history/native160 prerequisite')
    require(type(spec['master_port'])is int and 20000<=spec['master_port']<=60000,'Fixed rendezvous port')
    require(set(spec['units'])==set(STAGES)and len(set(spec['units'].values()))==3,'Three unique phase units')
    prefix='trn1-latest-verify-minicpm5-original88-'
    require(re.fullmatch(prefix+'[a-z0-9-]+-controller',spec['controller_unit']),'Fresh controller namespace')
    for stage,unit in spec['units'].items():require(re.fullmatch(prefix+'[a-z0-9-]+-'+stage.replace('_','-'),unit),'Fresh phase namespace')
    require(len(set([spec['controller_unit'],*spec['units'].values()]))==4,'Controller/phase namespace distinct')
    for field in('control_root','output_root'):
        path=Path(spec[field]);require(str(path)==spec[field]and path.parent==ROOT and
            path.name.startswith('minicpm5-original88-python-trace-')and '..'not in path.parts,'Fresh root namespace')
    require(spec['control_root']!=spec['output_root'],'Distinct output/control roots')
    require(len(spec['predecessor_units'])==len(set(spec['predecessor_units']))>=440,'Full history through Smol25planned units')
    require(not set(spec['units'].values())&set(spec['predecessor_units'])and spec['controller_unit']not in spec['predecessor_units'],'No historical service reuse')
    history.pin_shape(spec['source_sha256'])
    require(all(Path(path).suffix not in history.FORBIDDEN for path in spec['source_sha256']),'Strict source-only execution closure')
    require(all(str(ROOT/name)in spec['source_sha256']for name in SUCCESSORS),'All eight successor sources pinned')
    require(set(spec['canonical_sha256'])=={'continuation-state.json','roadmap.json','best-known-configurations.json'},'Complete canonical pins')
    return spec

def final_catalog(spec):
    require(spec['predecessor_catalog']==FINAL_CATALOG and spec['predecessor_catalog_review']==FINAL_CATALOG_REVIEW,'Exact root final440/44 Smol history')
    catalog=read_ref(spec['predecessor_catalog']);base=read_ref(MINIMUM_CATALOG)
    require(len(base['rows'])==415 and len(base['processes'])==43,'Exact415/43 minimum history')
    require(catalog['reviewer']=='/root'and catalog['root_finalized']is catalog['all_planned_launches_accounted_for']is True and
            catalog['complete_count']==len(catalog['rows'])>=440 and len(catalog['processes'])>=44,'Complete actual successor history')
    require(catalog['rows'][:415]==base['rows']and catalog['processes'][:43]==base['processes'],'Lossless original415/43 prefix')
    dynamic={'schema','rows','processes','complete_count','source_sha256','root_finalization_observed_at','appended_from','appended_terminal_audit'}
    require(all(catalog.get(k)==v for k,v in base.items()if k not in dynamic),'Every prior failure/extra field preserved')
    require(all(catalog['source_sha256'].get(k)==v for k,v in base['source_sha256'].items()),'Every historical source/fingerprint preserved')
    from smollm3_token_capture_predecessors_v7 import validate_rows
    validate_rows(catalog['rows'],catalog['processes'])
    smol=read_ref(SMOL_SPEC);live=read_ref(SMOL_LIVE)
    require(set(smol['planned_units'])<=set(row['unit']for row in catalog['rows'])and
            any(all(owner.get(k)==v for k,v in live['owner'].items())for owner in catalog['processes']),
            'Actual Smol60684 external owner and every planned unit accounted')
    review=read_ref(spec['predecessor_catalog_review'])
    require(review['schema']=='minicpm5_original88_predecessor_catalog_root_review_v1'and review['actual_state_observation']==SMOL_FINALIZED and review['reviewer']=='/root'and review['passed']is review['all_planned_launches_accounted_for']is review['actual_state_checked_by_root']is True,'Root actual final-catalog review')
    require(review['catalog']==spec['predecessor_catalog']and review['expected_units']==spec['predecessor_units']==[x['unit']for x in catalog['rows']]and
            review['expected_processes']==catalog['processes']and review['catalog_content_sha256']==digest(dict(rows=catalog['rows'],processes=catalog['processes'])),'Exact final catalog order/owners')
    terminal=review['smol60684_terminal']
    require(terminal['actual_external_exit_observed']is True and type(terminal['actual_external_exit_code'])is int and
            terminal['owner']==live['owner']and terminal['all_planned_launches_accounted_for']is terminal['all_groups_drained']is True,
            'Actual Smol terminal, never infer success from absence')
    reference=terminal['root_terminal_evidence']
    require(reference==SMOL_TERMINAL and catalog['schema']=='root_post_smollm3_sio6_complete_predecessors_v5'and
            catalog['appended_terminal_audit']==reference and catalog['appended_from']==MINIMUM_CATALOG,
            'Exact accepted Smol finalizer5 catalog/audit chain')
    proof=read_ref(reference)
    require(proof['schema']=='smollm3_sio6_root_terminal_audit_v5'and proof['reviewer']=='/root'and
            proof['passed']is proof['all_planned_launches_accounted_for']is proof['all_groups_drained']is True and
            proof['canonical_unchanged']is True and proof['complete_rows']==len(catalog['rows'])==440 and
            proof['complete_processes']==len(catalog['processes'])==44,'Root final440/44 physically drained proof')
    refs=proof['references']
    require(set(refs)=={'execution_spec','execution_review','previous_catalog','base','source_template','callable_handoff',
            'candidate','terminal','release','controller','supervisor','supervisor_owner','supervisor_launch','supervisor_binding',
            'identity','source','payload','live','exited'}and refs['execution_spec']==SMOL_SPEC and refs['previous_catalog']==MINIMUM_CATALOG and
            refs['live']==SMOL_LIVE and refs['exited']==SMOL_EXIT,'Exact complete Smol actual references')
    for bound in(reference,SMOL_EXIT,SMOL_LIVE,SMOL_SPEC,FINAL_CATALOG_REVIEW,SMOL_FINALIZED,proof['source'],*refs.values()):
        require(type(bound)is dict and set(bound)=={'path','sha256'}and
                spec['source_sha256'].get(bound['path'])==bound['sha256'],'Every actual terminal reference in complete source closure')
    require(terminal['actual_exit']==SMOL_EXIT and terminal['execution_spec']==SMOL_SPEC and terminal['live']==SMOL_LIVE and
            terminal['source_review']==refs['execution_review'],'Review binds same live/spec/review/actual exit')
    finalized=read_ref(SMOL_FINALIZED);final_handle=finalized['actual_execution_handle']
    require(finalized['schema']=='smollm3_sio6_root_finalization_execution_v1'and finalized['reviewer']=='/root'and
            finalized['audit']==reference and finalized['catalog']==FINAL_CATALOG and finalized['finalizer']==proof['source']and
            final_handle['kind']=='tool_session'and final_handle['session_id']==95421 and final_handle['actual_completion_observed']is True and
            type(final_handle['exit_code'])is int and final_handle['exit_code']==0 and finalized['cleanup_finalized']is True and
            finalized['experiment_exit_code']==1 and finalized['experiment_qualified']is finalized['new_speedup_claim']is False and
            finalized['completed_request_captures']==0,'Actual root95421 finalizer exit0 with experiment failure retained')
    exited=read_ref(refs['exited']);release=read_ref(refs['release']);external=read_ref(refs['terminal']);controller=read_ref(refs['controller'])
    require(live['schema']=='smollm3_observer_external_root_live_v1'and live['reviewer']=='/root'and live['actual_identity_verified']is True and
            live['execution_spec']==SMOL_SPEC and live['source_review']==refs['execution_review'],'Actual live launch/spec/review')
    require(exited['schema']=='smollm3_observer_external_root_exit_v1'and exited['reviewer']=='/root'and
            exited['execution_spec']==SMOL_SPEC and exited['source_review']==refs['execution_review']and
            exited['owner']==live['owner']==terminal['owner']and exited['release']==refs['release']and
            exited['terminal']is exited['actual_exit_observed_by_root']is exited['gone']is True,'Actual root-observed same external exit')
    handle=exited['execution_handle'];live_handle=live['actual_execution_handle']
    require(handle['kind']==live_handle['kind']=='tool_session'and handle['session_id']==live_handle['session_id']==60684 and
            handle['actual_completion_observed']is True and type(exited['exit_code'])is int and
            handle['exit_code']==exited['exit_code']==proof['actual_external_exit_code']==terminal['actual_external_exit_code']==1,
            'Real60684 exit1 consistently retained')
    require(release['released']is release['failed']is release['all_planned_groups_drained']is True and
            release['external_owner']==external['external_owner']==live['owner']and
            external['schema']=='smollm3_observer_invariance_external_terminal_v3'and
            external['all_planned_groups_drained']is external['external_self_exit_not_claimed']is True,'Failed external release physically drained')
    require(proof['experiment_qualified']is external['qualified']is external['lifecycle_qualified']is False and
            proof['protocol_completed']is controller['protocol_completed']is external['protocol_completed']is False and
            proof['observer_api_invariance_passed']is controller['api_invariance_passed']is external['observer_api_invariance_passed']is None and
            proof['full_model_qualified']is proof['new_speedup_claim']is False,'Physical release never promotes failed Smol science')
    require(proof['original_controller_error']==controller['original_error']==external['original_controller_error']and
            proof['original_controller_cleanup_errors']==controller['cleanup_errors']==external['original_controller_cleanup_errors']and
            len(controller['cleanup_errors'])==1 and proof['external_cleanup_errors']==external['cleanup_errors']==[]and
            proof['controller_cleanup_failure_recovered_by_external']is True,'Original controller failures preserved after outer recovery')
    require(proof['completed_captures']==[]and proof['launched_units']==2 and proof['never_launched_units']==23 and
            [row['unit']for row in proof['original_row_evidence']]==smol['planned_units']and
            [row['unit']for row in proof['physical_observations']]==[row['unit']for row in catalog['rows']],
            'All440 physical observations and25 actual planned records')

    return catalog

def predecessor_drain(catalog):
    result=low.predecessor_drain(catalog)
    for row in catalog['rows']:
        for path in row['recorded_cgroups']:
            require(path=='/sys/fs/cgroup'+GROUP_PREFIX+row['unit']+'.service','Exact recorded cgroup')
            empty_group(path.removeprefix('/sys/fs/cgroup'))
    return result

def historical_closure(spec,catalog):
    parts=history.partition(catalog,read_ref=read_ref)
    require(spec['historical_payload_sha256']==parts['historical_payload_sha256']and
            spec['historical_payload_provenance_sha256']==parts['provenance_sha256'],'Exact14 fingerprints/provenance partition')
    pins=spec['source_sha256'];combined=dict(pins,**spec['historical_payload_sha256'])
    require(not set(pins)&set(spec['historical_payload_sha256'])and
            all(pins.get(k)==v for k,v in parts['source_sha256'].items())and
            all(pins.get(k)==v for k,v in parts['provenance_sha256'].items())and
            all(combined.get(k)==v for k,v in catalog['source_sha256'].items()),'Lossless disjoint complete closure')
    identities.alias_shape(spec['metadata_source_alias_identities'],pins)
    identities.shared_shape(spec['shared_source_identities'],pins)
    return combined

def contracts(spec,plan):
    from minicpm5_original88_python_trace_validation_v2 import combine_contract_bundles
    return combine_contract_bundles(read_ref(spec['interpreter_contracts']['native']),read_ref(spec['interpreter_contracts']['cpu']),plan)

def bundle(spec_ref,review_ref):
    clear_window();spec=validate_spec(read_ref(spec_ref));review=read_ref(review_ref)
    require(review['reviewer']=='/root'and review['passed']is review['independent_review']is review['source_only']is True and
            review['blockers']==[] and review['execution_spec_sha256']==spec_ref['sha256'],'Exact independently accepted execution spec')
    for key in('source_sha256','historical_payload_sha256','historical_payload_provenance_sha256','metadata_source_alias_identities','shared_source_identities'):
        require(review[key]==spec[key],'Root reviewed '+key)
    science=read_ref(plan_source.SCIENCE);accepted=read_ref(plan_source.ACCEPT)
    original=read_ref(plan_source.ORIGINAL);plan=read_ref(spec['input_plan'])
    require(digest(plan)==spec['input_plan']['canonical_sha256']and plan['output_root']==spec['output_root'],'Exact derived plan')
    plan_source.validate_namespace_equivalence(original,plan,science,accepted)
    catalog=final_catalog(spec);combined=historical_closure(spec,catalog)
    for source in(plan['source_sha256'],science['source_sha256'],science['original_evidence_sha256']):
        require(all(spec['source_sha256'].get(k)==v for k,v in source.items()),'Complete scientific/worker closure')
    references=[plan_source.SCIENCE,plan_source.ACCEPT,plan_source.INDEPENDENT,plan_source.ORIGINAL,DESIGN_ACCEPT,MINIMUM_CATALOG,SMOL_SPEC,SMOL_LIVE,
        spec['input_plan'],spec['predecessor_catalog'],spec['predecessor_catalog_review'],*NATIVE160.values(),*plan_source.CONTRACTS.values(),
        identities.HISTORICAL_DESIGN,identities.HISTORICAL_HELPER,identities.SOURCE_ALIAS_DESIGN]
    for reference in references:require(spec['source_sha256'].get(reference['path'])==reference['sha256'],'Pinned prerequisite metadata')
    for name in('minicpm5_layer0_execution_protocol_v4.py','minicpm5_python_trace_lifecycle_v4.py',
                'historical_b3_payload_partition_v1.py','smollm3_token_capture_predecessors_v7.py',
                'smollm3_token_capture_protocol_v7.py',
                'minicpm5_python_trace_result_parser_v4.py','minicpm5_python_trace_factory_v3.py',
                'minicpm5_layer0_runtime_auth_v1.py','minicpm5_layer0_writer_factory_v1.py',
                'minicpm5_layer0_tree_monitor_v1.py'):
        require(str(ROOT/name)in spec['source_sha256'],'Explicit helper source pin '+name)
    verify_sources(spec);contracts(spec,plan)
    return spec,plan,catalog

def control_usage(spec):
    # Original64files/32MiB bounds include atomic staging and final files.
    root=Path(spec['control_root']);total=0;count=0
    for attempt in range(4):
        total=0;count=0;retry=False
        for path in root.iterdir():
            try:info=path.lstat()
            except FileNotFoundError:
                match=re.fullmatch(r'\.pending-([A-Za-z0-9_-]+)-[1-9][0-9]*\.json',path.name)
                require(match is not None,'Only an atomic staging file may disappear')
                final=root/(match.group(1)+'.json');published=final.lstat()
                require(stat.S_ISREG(published.st_mode)and published.st_uid==1000 and published.st_nlink==1 and
                        published.st_size<16*MIB and not final.is_symlink(),'Atomic rename requires exact owned final file')
                retry=True;break
            require(stat.S_ISREG(info.st_mode)and info.st_uid==1000 and info.st_nlink==1 and not path.is_symlink(),'Control regular owned file')
            require(path.suffix in('.json','.log')and info.st_size<16*MIB,'Control cap/type')
            count+=1;total+=max(info.st_size,info.st_blocks*512)
        if not retry:break
    require(not retry,'Control publication did not stabilize in four scans')
    require(count<=64 and total<=spec['control_cap_bytes'],'Original complete control cap')
    fs=os.statvfs(ROOT);require(fs.f_bavail*fs.f_frsize>=spec['root_floor_bytes'],'Root28GiB runtime floor')
    return total

def validate_native160_receipt(receipt,admission):
    require(receipt['schema']=='minicpm5_original88_native160_prerequisite_v1'and receipt['passed']is True and
            receipt['execution_spec']==admission['execution_spec']and receipt['source_review']==admission['source_review']and
            receipt['controller']==admission['controller']and receipt['lock_identity']==admission['lock_identity']and
            receipt['attempt_id']==admission['attempt_id']and receipt['predecessor_catalog']==admission['predecessor_catalog']and
            receipt['native160']==NATIVE160,'Current spec/review/catalog/owner/nonce native160 receipt')
    result=receipt['completion']
    require(result['diagnostic_complete']is result['all_raw_exact']is result['all_arm_pairs_equal']is result['raw_audit_complete']is True and
            result['frames']==160 and result['raw_files']==1552 and result['logical_raw_bytes']==314961320 and
            len(result['paired_comparisons'])==10 and all(x['frames']==x['equal_frames']==32 for x in result['paired_comparisons']),
            'Fresh complete160/1552raw prerequisite')
    from itertools import combinations
    require(result['original_hard_gate_preserved']is result['original88_scope_preserved']is True and
            result['scale_observation_schema']=='minicpm5_python_ast_scale_observation_v1'and
            result['arms']=={arm:32 for arm in identities.ARMS}and
            [(row['left'],row['right'])for row in result['paired_comparisons']]==list(combinations(identities.ARMS,2))and
            all(row['differing_frames']==[]for row in result['paired_comparisons']), 'Exact160 five-arm science and all ten pair identities')
    require(result['full_model_qualified']is result['new_speedup_claim']is result['causal_operator_proven']is False and
            receipt['historical_payloads_rehashed']is False,'Prerequisite diagnostic only,14historical fingerprints remain historical')
    inventory=receipt['raw_inventory']
    require(type(inventory)is list and len(inventory)==len({row['path']for row in inventory})==1552 and
            sum(row['bytes']for row in inventory)==314961320 and digest(inventory)==receipt['raw_inventory_sha256'],
            'Complete original raw identity/hash inventory bound by digest')
    return True
