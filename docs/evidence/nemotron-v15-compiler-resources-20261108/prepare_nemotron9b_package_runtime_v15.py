"""Prepare v15 from actual failed r6 and completed Mini upstream 981/196 history.
No source approval, execution admission, native imports or cleanup.
"""
import argparse,copy,hashlib,json,re,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];Q=R/'research';sys.path.insert(0,str(R))
import nemotron9b_package_protocol_v15 as p
import nemotron9b_post_upstream_history_v1 as history
from research.prepare_nemotron9b_package_runtime_v13 import merge,observer_policy
PARENT=dict(path=str(Q/'nemotron9b-package-runtime-v14-r6-20261108-spec.json'),sha256='cd779a014effdca2e4fd9a8595b9764ba5e0b51d37df05279292ed266de2c4ff')
CATALOG=dict(history.CATALOG)
FINALIZER=dict(history.EXIT)
TESTS_PATH=Q/'nemotron9b-v15-integration-tests-20261108-v1.json'
CHANGED={'schema','run_id','controller','worker','units','control_root','evidence_root','work_root','final_predecessor_catalog','required_predecessor_releases','source_sha256','runtime_readiness','external_observer_contract'}
CHANGED |= {'required_post_mini_maintenance','compiler_asset_policy','completed_history_finalizer_exit'}
ADDED={'compiler_resource_runtime_successor','required_prelaunch_maintenance'}

def encoded(path,value):return p.metadata.encode(path,value)
def value_ref(path,value):return dict(path=str(path),sha256=hashlib.sha256(encoded(path,value)).hexdigest())

def validate_parent(parent):
 p.require(parent['schema']=='nemotron9b_package_runtime_v14'and parent['root_finalized']is True,'Exact original admitted r6 parent')
 p.require(parent['package_commit']==p.COMMIT and parent['package_root']==str(p.PACKAGE),'Original model implementation')
 p.require(parent['stages']==list(p.STAGES)and len(parent['stages'])==52,'Every original scientific and verification stage')
 p.require(parent['resources']=={x:p.resources(x)for x in (*p.STAGES,'controller')},'All original resource bounds')
 p.require(parent['disk_budget']==dict(root_floor=p.ROOT_FLOOR,nvme_floor=p.NVME_FLOOR,root_cap=p.ROOT_CAP,nvme_cap=p.NVME_CAP,caps=p.CAPS),'Unreduced storage policy')
 p.require(parent['retirement_policy']=='none'and parent['automatic_successor_launch']is False and parent['speedup_denominator']is None,'No automatic restart/cleanup or failed denominator')

def compose(parent,catalog,run_id,observer_ref,pins):
 validate_parent(parent);p.require(re.fullmatch('[a-z0-9-]{1,40}',run_id)and run_id!=parent['run_id'],'Fresh explicit run identifier')
 p.require(catalog['schema']==history.SCHEMA and catalog['root_finalized']is True and catalog['all_planned_launches_accounted_for']is True,'Exact actual completed history')
 p.require(catalog['complete_count']==len(catalog['rows'])==981 and len(catalog['processes'])==196,'All actual preceding services/processes')
 sources=merge(parent['source_sha256'],catalog['source_sha256'],history.MAINTENANCE_PINS,pins,{ref['path']:ref['sha256']for ref in [PARENT,CATALOG,FINALIZER,p.compiler_links.ASSET_REF,observer_ref]})
 result=copy.deepcopy(parent)
 result.update(schema='nemotron9b_package_runtime_v15',run_id=run_id,controller=str(R/'nemotron9b_package_controller_v15.py'),worker=str(R/'nemotron9b_package_worker_v15.py'),
  units={stage:'trn1-latest-verify-nemotron9b-package-'+run_id+'-'+stage.replace('_','-')for stage in (*p.STAGES,'controller')},
  control_root=str(R/('nemotron9b-package-'+run_id+'-control')),evidence_root=str(R/('nemotron9b-package-'+run_id+'-evidence')),work_root='/mnt/nvme/nemotron9b-package-'+run_id,
  final_predecessor_catalog=copy.deepcopy(CATALOG),required_predecessor_releases=copy.deepcopy(catalog['root_release_refs']),source_sha256=sources,external_observer_contract=copy.deepcopy(observer_ref),
  required_prelaunch_maintenance=copy.deepcopy(history.MAINTENANCE),required_post_mini_maintenance=copy.deepcopy(catalog['post_mini_maintenance']),compiler_asset_policy=copy.deepcopy(p.compiler_links.ASSET_REF),completed_history_finalizer_exit=copy.deepcopy(FINALIZER),
  runtime_readiness=dict(ready=False,source_compatible=True,blockers=['Separate root review of exact generated specification','Fresh complete physical drain, observer, storage and payload checks','All52 original stages remain unexecuted in this new run']),
  compiler_resource_runtime_successor=dict(parent_spec=PARENT,integration_tests=dict(path=str(TESTS_PATH),sha256=pins[str(TESTS_PATH)]),original_failures_retained=True,model_code_unchanged=True,metadata_preparation_is_not_runtime_admission=True))
 p.require(set(result)==set(parent)|ADDED,'Only reviewed successor fields added')
 p.require(all(result[k]==value for k,value in parent.items()if k not in CHANGED),'Every other original scientific/input/provenance field unchanged')
 p.require(set(p.required_sources())<=set(sources),'Complete new source closure')
 p.require(len(set(result['units'].values()))==53 and not set(result['units'].values())&{row['unit']for row in catalog['rows']},'Unique never-reused service names')
 p.alias_metadata.validate_spec(result);return result

def validate_history(spec,catalog):
 history.validate(catalog)
 finish=p.metadata.load(FINALIZER)
 history.finalizer(catalog,finish);history.validate_maintenance(spec,p.metadata.load)
 import nemotron9b_post_smol_maintenance_v2 as maintenance
 import nemotron9b_smol_terminal_chain_v1 as smol
 old_path=str(Q/'nemotron9b-package-post-smol-maintenance-r4-20261108-catalog-v2.json');old_ref=dict(path=old_path,sha256=catalog['source_sha256'][old_path]);old=p.metadata.load(old_ref)
 proof=maintenance.validate(dict(spec,final_predecessor_catalog=old_ref,required_predecessor_releases=old['root_release_refs']),old,p.metadata.load)
 chain=smol.validate(dict(spec,final_predecessor_catalog=proof['catalog_reference'],required_predecessor_releases=proof['catalog']['root_release_refs']),proof['catalog'],p.metadata.load)
 p.require(proof['fresh_physical_drain_proven']is chain['fresh_physical_drain_proven']is False,'History grants no runtime drain')
 return dict(complete_services=981,complete_processes=196,original_maintenance_verified=True,original_smol_chain_verified=True,fresh_physical_drain_proven=False)

def current_source_checks(spec,catalog,pending):
 parts=p.metadata.classify_sources(spec,catalog)
 for name,digest in parts['text_sha256'].items():
  if name in pending:
   p.require(value_ref(Path(name),pending[name])['sha256']==digest,'Exact unpublished metadata bytes');continue
  actual=(p.alias_metadata.source_sha(name,spec['metadata_aliases'],p.metadata.text)if name in spec['metadata_aliases']else p.metadata.sha(name,digest))
  p.require(actual==digest,'Source fingerprint changed: '+name)
 for name,digest in parts['python_bytes_sha256'].items():p.metadata.python_source_bytes(name,digest)
 for name,row in spec['payloads'].items():p.require(p.file_stat(name)==row['stat'],'Original payload identity changed')
 for name,identity in spec['runtime_binary_stat'].items():p.require(p.file_stat(name)==identity,'Original runtime binary identity changed')
 return dict(source_entries_rehashed=len(parts['text_sha256'])+len(parts['python_bytes_sha256']),historical_raw_reads=0,payload_bytes_read=0,payload_stats_checked=len(spec['payloads']),runtime_binary_stats_checked=len(spec['runtime_binary_stat']))

def build(run_id):
 parent=p.metadata.load(PARENT);catalog=p.metadata.load(CATALOG);tests=p.read(TESTS_PATH);test_ref=p.ref(TESTS_PATH)
 p.require(tests['all_passed']is True and tests['tests']>=90 and tests['actual_exit']['exit_code']==0 and tests['actual_tool_exit_observed']is True,'Actual v15 integration tests')
 pins=merge(tests['source_sha256'],{test_ref['path']:test_ref['sha256']})
 for name,digest in pins.items():p.metadata.sha(name,digest)
 for name in [*p.required_sources(),str(Path(__file__).resolve()),str(Q/'check_prepare_nemotron9b_package_runtime_v15.py')]:pins=merge(pins,{name:p.metadata.sha(name)})
 observer=observer_policy(parent,p.metadata.load(parent['external_observer_contract']),pins[str(R/'nemotron9b_package_window_v15.py')])
 prefix=Q/('nemotron9b-package-runtime-v15-'+run_id);observer_path=Path(str(prefix)+'-observer-contract.json');spec_path=Path(str(prefix)+'-spec.json');checks_path=Path(str(prefix)+'-preparation-checks.json')
 p.require(not any(path.exists()for path in [observer_path,spec_path,checks_path]),'New immutable preparation files')
 spec=compose(parent,catalog,run_id,value_ref(observer_path,observer),pins);proof=validate_history(spec,catalog)
 for field in ('control_root','evidence_root','work_root'):p.require(not Path(spec[field]).exists(),'Never reuse experiment output roots')
 p.storage(spec,reserve=True);p.compiler_links.verify_assets(p.metadata.load(p.compiler_links.ASSET_REF),hash_content=True)
 stats=current_source_checks(spec,catalog,{str(observer_path):observer})
 from nemotron9b_package_evidence_v15 import validate_payload_closure
 validate_payload_closure(spec)
 encoded(observer_path,observer);encoded(spec_path,spec)
 p.once(observer_path,observer);p.once(spec_path,spec)
 output=dict(schema='nemotron9b_v15_spec_preparation_checks_v1',reviewer='/root',specification=p.ref(spec_path),observer_contract=p.ref(observer_path),catalog=CATALOG,parent=PARENT,tests=test_ref,source=p.ref(Path(__file__).resolve()),stats=stats,history=proof,
  original_stage_count=52,all_other_parent_fields_preserved=True,original_payload_and_runtime_stats_unchanged=True,native_libraries_imported=False,execution_review_created=False,runtime_admitted=False,native_started=False,new_speedup_claim=False)
 p.once(checks_path,output);return dict(specification=p.ref(spec_path),observer_contract=p.ref(observer_path),checks=p.ref(checks_path),runtime_admitted=False)
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-id',required=True);args=parser.parse_args()
 p.require(sys.dont_write_bytecode and __debug__,'Bytecode disabled, assertions enabled');print(json.dumps(build(args.run_id)))
