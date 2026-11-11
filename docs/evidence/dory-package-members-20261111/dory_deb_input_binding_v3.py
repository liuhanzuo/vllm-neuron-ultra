"""Concrete six signed archives; original serial lifetime/resource contract."""
from pathlib import Path
import dory_serial_stage_contract_v2 as c
import dory_descriptor_inventory_runtime_v2 as runtime
import dory_ubuntu_metadata_input_binding_v5 as metadata
import dory_ubuntu_package_sources_v4 as discovery
from dory_ubuntu_metadata_input_binding_v5 import storage_usage,save_final_result
WORK_KIND='dory_six_deb_member_verification_v1'
NEXT=dict(path=str(c.Q/'dory-six-deb-member-verification-next-inputs-20261111-v1.json'),sha256='ea62d3a13088728d477d041bb3f3676324d3c7c0ec082e2140ad2000d249c98c')
TOOL=dict(path=str(c.Q/'dory-dpkg-deb-tool-authority-20261111-v1.json'),sha256='a806f318a23e8a8d1bb5f9f408dbc24eaad3e0625be38ab3e8ebfdccc434b174')
TRUST=dict(next_inputs=NEXT,tool_authority=TOOL)
SOURCE_PINS={'/home/ubuntu/experiments/trn1-latest-models-20261031/research/dory_deb_transport_v1.py': 'eb7919d9cc4e3d3e0be315db0e398cda4239e157add8a477b6a1c2f9ffff2d06', '/home/ubuntu/experiments/trn1-latest-models-20261031/research/dory_deb_members_adapter_v1.py': '81dab76a14422480b8b290ac6d346c8a8fcc2479245b976e3837d621f3f6cb30', '/home/ubuntu/experiments/trn1-latest-models-20261031/research/dory_deb_tar_members_v1.py': 'd9ee5cd18dc6e8dfb2f9575cdefa3afe26511d743f7054aa3dd9fb0c60dfafa5', '/home/ubuntu/experiments/trn1-latest-models-20261031/research/dory_deb_transport_v2.py': '23b767f142c192d830464068022f626fc6d6db5010ef5200e59313f9eb6ae533', '/home/ubuntu/experiments/trn1-latest-models-20261031/research/dory_deb_members_adapter_v2.py': '8bf8b8ceafd963de5911d0abb93d4212b2c2649ff5a04be1775ec405d8914b3d', '/home/ubuntu/experiments/trn1-latest-models-20261031/research/dory_deb_tar_members_v2.py': 'd43b3938ee61e13239e09d687e7b6299383da5baa527c54192d25b80d78225c6', '/home/ubuntu/experiments/trn1-latest-models-20261031/research/dory_deb_transport_v3.py': '3a9d382507bcdb167fb5989be70e9edaba16e070f9894a0975b27edd18bc617c', '/home/ubuntu/experiments/trn1-latest-models-20261031/research/dory_deb_members_adapter_v3.py': 'ab12722776eb8a77fcd05fec84aa333b4969be6c34885b64e573f1622fa31448'}

def validate(plan,review,plan_ref):
    runtime.validate(plan,review,plan_ref)
    c.require(plan.get('work_kind')==review.get('work_kind')==WORK_KIND and plan.get('archive_inputs')==review.get('archive_inputs')==TRUST,'Exact six-archive task/input binding')
    c.require(plan.get('archive_mode')==review.get('archive_mode') and plan['archive_mode']in ('download','replay')and plan.get('replay_from')==review.get('replay_from')and (plan.get('replay_from')is None)==(plan['archive_mode']=='download'),'Exact reviewed download or saved replay mode')
    c.require(plan.get('external_authorities') is None,'No native/model authority')
    for name,pin in SOURCE_PINS.items():c.require(plan['source_sha256'].get(name)==pin,'Exact new source pins')
    return plan

def authenticated(guard):
    guard();n=c.load(NEXT);s=c.load(n['signed_metadata']);ended=c.load(n['producer_actual_exit']);rec=c.load(n['recorder_actual_exit']);tool=c.load(TOOL)
    c.require(n['schema']=='dory_six_deb_member_verification_next_inputs_v1'and s['status']=='exact_addresses_ready'and s['missing']==[] and len(s['packages'])==len(n['packages'])==6,'Exact complete signed versions')
    c.require(ended['stage_success_observed'] and ended['actual_tool']['exit_code']==0 and ended['complete_events_audit']['recomputed']['entire_discovery_reproduced'] and ended['evidence']['signed-metadata/sources-result.json']==n['signed_metadata'],'Successful full signed metadata audit')
    c.require(rec['actual_tool']['exit_code']==0 and rec['owner_gone']and rec['result']==n['producer_actual_exit'],'Completed actual metadata recorder')
    for row,old in zip(n['packages'],s['packages']):
        c.require(all(row[k]==old[k]for k in ('package','version','architecture','filename','bytes','sha256','targets','saved_package_names')),'Exact signed package/target provenance')
        c.require(row['version']==discovery.PACKAGE_VERSIONS[row['package']] and row['architecture']=='amd64'and row['url']==sorted({x['url']for x in old['sources']})[0],'Fixed selected signed URL/version')
    c.require(sum(x['bytes']for x in n['packages'])==n['archive_total_bytes']==4234134 and sum(len(x['targets'])for x in n['packages'])==12,'Exact six sizes/twelve targets')
    c.require(tool['schema']=='dory_dpkg_deb_tool_authority_v1'and tool['reviewer']=='/root'and tool['tool']['path']=='/usr/bin/dpkg-deb','Explicit installed OS tool authority')
    guard();return n,tool

def authenticate(guard):
    authenticated(guard)
    return dict(schema='dory_deb_inputs_authenticated_v1',work_kind=WORK_KIND,inputs=TRUST,packages=6,targets=12,original_full_hashes=11,package_content_authenticated=False,native_ABI_qualified=False)

def validate_admission(value,authenticated):
    c.require(value.get('work_kind')==WORK_KIND and value.get('metadata_input_binding')==authenticated,'Original parent admission binds archive scope')

def save_collected(out,value,authenticated,plan_ref,admission_ref,guard):
    c.require(value['result']['path']==str(out/'packages'/'sources-result.json'),'Owned package result')
    result=c.load(value['result'],8*c.MIB)
    c.require(result['schema']=='dory_deb_members_result_v1'and result['complete']and len(result['packages'])==6 and result['target_count']==12 and result['original_match_count']==11 and result['current_match_count']==12,'Complete fixed targets and original/current comparisons')
    guard();ref=c.save(out/'metadata-result.json',dict(schema='dory_deb_member_receipt_v1',work_kind=WORK_KIND,sources_result=value['result'],metadata_input_binding=authenticated,plan=plan_ref,admission=admission_ref,package_content_authenticated=True,loader_selection_established=False,native_ABI_qualified=False,full_model_qualified=False,new_speedup_claim=False))
    c.require(storage_usage(out)<=c.INVENTORY['output_bytes'],'Original total output budget');return ref
