"""Prepare a fresh v12 spec only from actual completed Smol+maintenance history.

No framework imports, payload reads, model launches, cleanup, source review
approval or execution admission. The independent root review and runtime
coordinator retain all fresh physical/storage/payload checks.
"""
import argparse,copy,hashlib,json,re,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];Q=R/'research';sys.path.insert(0,str(R))
import nemotron9b_package_protocol_v12 as p
import nemotron9b_post_smol_maintenance_v1 as maintenance
import nemotron9b_smol_terminal_chain_v1 as smol
PARENT=dict(path=str(Q/'nemotron9b-package-runtime-v10-r4-20261108-spec-v2.json'),sha256='a782f6a824bbf2877fa95997f58a1e0dbc441fcaf30a98b0087a4c2a8713c8e5')
PREPARATION=dict(path=str(Q/'nemotron9b-v12-maintenance-runtime-source-preparation-20261108-v1.json'),sha256='dd61764e97b51dad03204222d8cfb4c584a8ebbc48fef8afa92fde9636cf53c0')
CURRENT=Q/'nemotron9b-package-post-smol-maintenance-r2-20261108-catalog-v1.json'
CHANGED={'schema','run_id','controller','worker','units','control_root','evidence_root','work_root','final_predecessor_catalog','required_predecessor_releases','source_sha256','runtime_readiness','external_observer_contract','required_smol_predecessor_chain'}
ADDED={'required_post_smol_maintenance','maintenance_runtime_successor'}

def merge(*groups):
 result={}
 for pins in groups:
  for name,digest in pins.items():
   p.require(name not in result or result[name]==digest,'Never rebind original source: '+name);result[name]=digest
 return result

def encoded(path,value):return p.metadata.encode(path,value)
def value_ref(path,value):return dict(path=str(path),sha256=hashlib.sha256(encoded(path,value)).hexdigest())

def validate_parent(parent):
 p.require(parent['schema']=='nemotron9b_package_runtime_v10'and parent['root_finalized']is True,'Exact original admitted parent')
 p.require(parent['package_commit']==p.COMMIT and parent['package_root']==str(p.PACKAGE),'Original model implementation')
 p.require(parent['stages']==list(p.STAGES)and len(parent['stages'])==52,'Every original scientific and verification stage')
 p.require(parent['resources']=={x:p.resources(x)for x in (*p.STAGES,'controller')},'All original resource bounds')
 p.require(parent['disk_budget']==dict(root_floor=p.ROOT_FLOOR,nvme_floor=p.NVME_FLOOR,root_cap=p.ROOT_CAP,nvme_cap=p.NVME_CAP,caps=p.CAPS),'Unreduced storage policy')
 p.require(parent['retirement_policy']=='none'and parent['automatic_successor_launch']is False and parent['speedup_denominator']is None,'No automatic restart/cleanup or failed denominator')


def selected_chain(current_ref,current,load):
 result=[];seen=set()
 while smol.catalog_number(current_ref['path'])is not None:
  p.require(len(result)<16 and current_ref['path']not in seen,'Bounded acyclic original Smol history');seen.add(current_ref['path'])
  audit_ref=current['appended_terminal_audit'];audit=load(audit_ref)
  result.append(dict(catalog=copy.deepcopy(current_ref),terminal_audit=copy.deepcopy(audit_ref),**{key:copy.deepcopy(audit['references'][key])for key in smol.FIELDS}))
  current_ref=current['appended_from'];current=load(current_ref)
 p.require(bool(result),'At least actual completed sio14 producer')
 return result


def observer_policy(parent,original,window_hash):
 p.require(original['schema']=='nemotron_external_observer_contract_v3'and original['gate']==parent['observer_gate'],'Original complete observer contract')
 for key in ['root_reviewed','all_external_operations_use_gate','unregistered_observers_paused','shared_gate_covers_entire_operation','no_detached_observer_work']:
  p.require(original[key]is True,'Retained observer isolation: '+key)
 p.require(original['participants']==['/root'],'Root-only serial observer policy')
 result=copy.deepcopy(original);result['source_sha256']=window_hash;return result


def compose(parent,catalog,catalog_ref,run_id,observer_ref,chain,pins):
 """Pure transformation; caller separately authenticates every input and outcome."""
 validate_parent(parent);p.require(re.fullmatch('[a-z0-9-]{1,40}',run_id)and run_id!=parent['run_id'],'Fresh explicit run identifier')
 p.require(catalog_ref['path']==str(CURRENT)and catalog['schema']==maintenance.SCHEMA,'Exact completed maintenance namespace')
 p.require(catalog['root_finalized']is True and catalog['all_planned_launches_accounted_for']is True,'Actual completed predecessor history')
 p.require(bool(chain)and chain[0]['catalog']==catalog['maintenance_extension']['prior_catalog'],'Selected original Smol anchor')
 sources=merge(parent['source_sha256'],catalog['source_sha256'],pins,{catalog_ref['path']:catalog_ref['sha256'],PARENT['path']:PARENT['sha256'],PREPARATION['path']:PREPARATION['sha256'],observer_ref['path']:observer_ref['sha256']})
 result=copy.deepcopy(parent)
 result.update(schema='nemotron9b_package_runtime_v12',run_id=run_id,controller=str(R/'nemotron9b_package_controller_v12.py'),worker=str(R/'nemotron9b_package_worker_v12.py'),
  units={stage:'trn1-latest-verify-nemotron9b-package-'+run_id+'-'+stage.replace('_','-')for stage in (*p.STAGES,'controller')},
  control_root=str(R/('nemotron9b-package-'+run_id+'-control')),evidence_root=str(R/('nemotron9b-package-'+run_id+'-evidence')),work_root='/mnt/nvme/nemotron9b-package-'+run_id,
  final_predecessor_catalog=copy.deepcopy(catalog_ref),required_predecessor_releases=copy.deepcopy(catalog['root_release_refs']),source_sha256=sources,
  external_observer_contract=copy.deepcopy(observer_ref),required_smol_predecessor_chain=copy.deepcopy(chain),required_post_smol_maintenance=copy.deepcopy(catalog['maintenance_extension']),
  runtime_readiness=dict(ready=False,source_compatible=True,blockers=['Separate root review of exact generated specification','Fresh complete physical drain, observer, storage and payload checks','All52 original stages remain unexecuted in this new run']),
  maintenance_runtime_successor=dict(parent_spec=PARENT,source_preparation=PREPARATION,original_failures_retained=True,model_code_unchanged=True,metadata_preparation_is_not_runtime_admission=True))
 p.require(set(result)==set(parent)|ADDED,'Only reviewed successor fields added')
 p.require(all(result[k]==value for k,value in parent.items()if k not in CHANGED),'Every other original scientific/input/provenance field unchanged')
 p.require(set(p.required_sources())<=set(sources),'Complete new source closure')
 p.require(len(set(result['units'].values()))==53 and not set(result['units'].values())&{row['unit']for row in catalog['rows']},'Unique never-reused service names')
 p.require(all(re.fullmatch('trn1-latest-verify-nemotron9b-package-[a-z0-9-]+',name)for name in result['units'].values()),'Normalized service names, original stage names retained')
 p.alias_metadata.validate_spec(result)
 return result


def current_source_checks(spec,catalog,pending):
 parts=p.metadata.classify_sources(spec,catalog)
 for name,digest in parts['text_sha256'].items():
  if name in pending:
   p.require(value_ref(Path(name),pending[name])['sha256']==digest,'Exact unpublished metadata bytes');continue
  actual=(p.alias_metadata.source_sha(name,spec['metadata_aliases'],p.metadata.text)if name in spec['metadata_aliases']else p.metadata.sha(name,digest))
  p.require(actual==digest,'Source fingerprint changed: '+name)
 for name,digest in parts['python_bytes_sha256'].items():p.metadata.python_source_bytes(name,digest)
 for name,row in spec['payloads'].items():p.require(p.file_stat(name)==row['stat'],'Original payload identity changed; no new payload claim')
 for name,identity in spec['runtime_binary_stat'].items():p.require(p.file_stat(name)==identity,'Original runtime binary identity changed')
 return dict(source_entries_rehashed=len(parts['text_sha256'])+len(parts['python_bytes_sha256']),historical_raw_reads=0,payload_bytes_read=0,payload_stats_checked=len(spec['payloads']),runtime_binary_stats_checked=len(spec['runtime_binary_stat']))


def build(catalog_ref,run_id):
 p.require(catalog_ref['path']==str(CURRENT),'Only real finalized maintenance history accepted')
 parent=p.metadata.load(PARENT);prep=p.metadata.load(PREPARATION);catalog=p.metadata.load(catalog_ref)
 p.require(len(catalog['rows'])==catalog['complete_count']==863 and len(catalog['processes'])==166,'All actual preceding services/processes retained')
 pins=dict(prep['source_sha256'])
 for name,digest in pins.items():p.metadata.sha(name,digest)
 for name in [*p.required_sources(),str(Path(__file__).resolve()),str(Q/'check_prepare_nemotron9b_package_runtime_v12.py')]:pins=merge(pins,{name:p.metadata.sha(name)})
 original=p.metadata.load(parent['external_observer_contract'])
 observer=observer_policy(parent,original,pins[str(R/'nemotron9b_package_window_v12.py')])
 prefix=Q/('nemotron9b-package-runtime-v12-'+run_id);observer_path=Path(str(prefix)+'-observer-contract.json');spec_path=Path(str(prefix)+'-spec.json');checks_path=Path(str(prefix)+'-preparation-checks.json')
 p.require(not any(path.exists()for path in [observer_path,spec_path,checks_path]),'New immutable preparation files')
 observer_ref=value_ref(observer_path,observer);anchor_ref=catalog['maintenance_extension']['prior_catalog'];anchor=p.metadata.load(anchor_ref)
 chain=selected_chain(anchor_ref,anchor,p.metadata.load)
 spec=compose(parent,catalog,catalog_ref,run_id,observer_ref,chain,pins)
 proven=maintenance.validate(spec,catalog,p.metadata.load)
 smol_proof=smol.validate(dict(spec,final_predecessor_catalog=proven['catalog_reference']),proven['catalog'],p.metadata.load)
 p.require(proven['fresh_physical_drain_proven']is smol_proof['fresh_physical_drain_proven']is False,'Source evidence grants no fresh runtime drain')
 for field in ['control_root','evidence_root','work_root']:p.require(not Path(spec[field]).exists(),'Never reuse experiment output roots')
 stats=current_source_checks(spec,catalog,{str(observer_path):observer})
 from nemotron9b_package_evidence_v12 import validate_payload_closure
 validate_payload_closure(spec)
 # No review is generated. Root must independently inspect the exact spec and
 # publish a distinct review, then call the unchanged coordinator admission.
 encoded(observer_path,observer);encoded(spec_path,spec)
 p.once(observer_path,observer);p.once(spec_path,spec)
 output=dict(schema='nemotron9b_v12_spec_preparation_checks_v1',reviewer='/root',specification=p.ref(spec_path),observer_contract=p.ref(observer_path),catalog=catalog_ref,parent=PARENT,source_preparation=PREPARATION,
  source= p.ref(Path(__file__).resolve()),stats=stats,complete_services=863,complete_processes=166,original_stage_count=52,all_other_parent_fields_preserved=True,
  original_payload_and_runtime_stats_unchanged=True,smol_predecessor_proof=smol_proof,maintenance_proof=proven['maintenance'],
  native_libraries_imported=False,execution_review_created=False,runtime_admitted=False,native_started=False,new_speedup_claim=False)
 p.once(checks_path,output)
 return dict(specification=p.ref(spec_path),observer_contract=p.ref(observer_path),checks=p.ref(checks_path),runtime_admitted=False)
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--catalog',required=True);parser.add_argument('--catalog-sha256',required=True);parser.add_argument('--run-id',required=True);args=parser.parse_args()
 p.require(sys.dont_write_bytecode and __debug__,'Bytecode disabled, assertions enabled')
 print(json.dumps(build(dict(path=args.catalog,sha256=args.catalog_sha256),args.run_id)))
