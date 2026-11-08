"""Independent finalizer for failed gzip + successful XZ + actual retirement.
Only run after the actual retirement tool exit. No restart or model execution.
"""
import argparse,copy,datetime,hashlib,json,re,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];Q=R/'research';sys.path.insert(0,str(R))
import nemotron9b_package_protocol_v13 as p
import nemotron9b_post_smol_maintenance_v2 as bridge
import minicpm5_python_trace_execute_v4 as life
from research.finalize_smollm3_sio9_root_20261107_v1 import physical_drain
from research import retire_completed_cache_xz_20261108_v1 as retirement
EXIT=Q/'completed-cache-xz-retirement-root-exit-20261108-v1.json'
AUDIT=Q/'completed-cache-xz-maintenance-root-terminal-20261108-v1.json'

def raw(value):
 data=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode();assert len(data)<=16*2**20;return data

def value_ref(path,value):return dict(path=str(path),sha256=hashlib.sha256(raw(value)).hexdigest())
def once(path,value):
 data=raw(value)
 with path.open('xb') as f:f.write(data);f.flush();retirement.os.fsync(f.fileno())
 p.metadata.sync_parent(path);return value_ref(path,value)
def ref(path):return retirement.ref(path)

def build(prior,failed,probe_exit,probe_exit_ref,start,start_ref,result,result_ref,session,chunk,drains,verification,sources):
 assert type(session)is int and session>0 and re.fullmatch('[0-9a-f]{6}',chunk)
 assert len(drains)==len(prior['rows']) and {x['unit'] for x in drains}=={x['unit'] for x in prior['rows']}
 now=datetime.datetime.now(datetime.timezone.utc).isoformat()
 exited=dict(schema='completed_cache_xz_retirement_root_exit_v1',reviewer='/root',observed_at=now,owner=copy.deepcopy(start['owner']),actual_exit_observed_by_root=True,gone=True,terminal=True,exit_code=0,started=start_ref,result=result_ref,execution_handle=dict(kind='tool_session',session_id=session,actual_completion_observed=True,exit_code=0,tool_chunk=chunk))
 ext=dict(prior_catalog=bridge.PRIOR,failed_exit=bridge.FAILED,probe_exit=probe_exit_ref,retirement_exit=value_ref(EXIT,exited),candidate=bridge.CANDIDATE)
 owners=[failed['owner'],probe_exit['owner'],start['owner']]
 audit=dict(schema='completed_cache_xz_maintenance_root_terminal_v1',reviewer='/root',observed_at=now,passed=True,references=copy.deepcopy(ext),owners=copy.deepcopy(owners),all_current_groups_recursively_drained=True,owner_absence_observed=True,archive_bytes_rehashed=True,archive=result['archive'],manifest=result['manifest'],canonical_sha256=result['canonical_sha256'],full_model_qualified=False,new_speedup_claim=False,complete_services=len(prior['rows']),complete_processes=len(prior['processes'])+3,physical_observations=drains,archive_verification=verification,source=sources[0])
 ext['root_audit']=value_ref(AUDIT,audit)
 catalog=copy.deepcopy(prior);catalog.update(schema=bridge.SCHEMA,maintenance_extension=ext,processes=copy.deepcopy(prior['processes'])+[dict(o,role=role) for o,role in zip(owners,bridge.ROLES)])
 if 'process_predecessors'in catalog:catalog['process_predecessors']=copy.deepcopy(catalog['processes'])
 all_refs=[*ext.values(),failed['started'],failed['failure'],failed['manifest'],probe_exit['started'],probe_exit['result'],start_ref,result_ref,*sources,bridge.old.SOURCE,bridge.PROBE_SOURCE,bridge.RETIRE_SOURCE]
 for reference in all_refs:
  name,digest=reference['path'],reference['sha256'];assert name not in catalog['source_sha256'] or catalog['source_sha256'][name]==digest;catalog['source_sha256'][name]=digest
 catalog_ref=value_ref(bridge.CURRENT,catalog)
 spec=dict(final_predecessor_catalog=catalog_ref,source_sha256=dict(catalog['source_sha256'],**{catalog_ref['path']:catalog_ref['sha256']}),required_predecessor_releases=prior['root_release_refs'],required_post_smol_maintenance=ext)
 return exited,audit,catalog,spec

def main(args):
 assert sys.dont_write_bytecode and __debug__ and args.exit_code==0
 assert not any(f.exists() for f in [EXIT,AUDIT,bridge.CURRENT])
 prior=p.metadata.load(bridge.PRIOR);failed=p.metadata.load(bridge.FAILED)
 probe_exit_ref=dict(path=str(retirement.PROBE_EXIT),sha256=args.probe_exit_sha256);probe_exit=p.metadata.load(probe_exit_ref);proof=p.metadata.load(probe_exit['result'])
 start_ref=dict(path=str(retirement.DEST/'start.json'),sha256=args.start_sha256);result_ref=dict(path=str(retirement.DEST/'result.json'),sha256=args.result_sha256)
 start=p.metadata.load(start_ref);result=p.metadata.load(result_ref);manifest=p.metadata.load(result['manifest'])
 assert not (retirement.DEST/'failure.json').exists()
 fd,owner,lock=life.acquire()
 try:
  def guard():
   p.low.lock_proof(owner,lock);assert p.canonical()==result['canonical_sha256'];assert retirement.free()>=p.NVME_FLOOR
  for old in [failed['owner'],probe_exit['owner'],start['owner']]:p.low.gone(old)
  physical_drain(prior);p.low.active_heavy();p.accelerator_idle();guard()
  verification=retirement.verified_archive(Path(result['archive']['path']),manifest,proof['uncompressed_tar_sha256'],proof['uncompressed_tar_bytes'],guard)
  assert verification['archive']==result['archive']
  assert all(not Path(name).exists() for name in manifest['directories']) and not Path(proof['original_archive']['path']).exists()
  assert retirement.free()>=p.NVME_FLOOR+p.NVME_CAP
  drains=physical_drain(prior)
  for old in [failed['owner'],probe_exit['owner'],start['owner']]:p.low.gone(old)
  p.low.active_heavy();p.accelerator_idle();guard()
  sources=[ref(Path(__file__)),ref(Path(bridge.__file__)),ref(Q/'check_nemotron9b_post_smol_maintenance_v2.py'),ref(Q/'check_finalize_completed_cache_xz_maintenance_20261108_v1.py'),ref(Q/'completed-cache-xz-retirement-root-source-review-20261108-v1.json'),ref(Q/'completed-cache-xz-retirement-source-controls-20261108-v1.json'),ref(Q/'check_retire_completed_cache_xz_20261108_v1.py')]
  exited,audit,catalog,spec=build(prior,failed,probe_exit,probe_exit_ref,start,start_ref,result,result_ref,args.session,args.tool_chunk,drains,verification,sources)
  pending={str(EXIT):exited,str(AUDIT):audit}
  def load(reference):
   if reference['path'] in pending:
    value=pending[reference['path']];assert value_ref(Path(reference['path']),value)==reference;return value
   return p.metadata.load(reference)
  proof=bridge.validate(spec,catalog,load);assert proof['catalog']==prior and proof['fresh_physical_drain_proven'] is False
  assert once(EXIT,exited)==spec['required_post_smol_maintenance']['retirement_exit']
  assert once(AUDIT,audit)==spec['required_post_smol_maintenance']['root_audit']
  assert once(bridge.CURRENT,catalog)==spec['final_predecessor_catalog']
  assert p.metadata.load(spec['final_predecessor_catalog'])==catalog
  print(json.dumps(dict(catalog=spec['final_predecessor_catalog'],actual_exit=ref(EXIT),audit=ref(AUDIT),complete_services=len(catalog['rows']),complete_processes=len(catalog['processes']))),flush=True)
 finally:life.release(fd)
if __name__=='__main__':
 parser=argparse.ArgumentParser()
 for name in ['probe-exit-sha256','start-sha256','result-sha256','tool-chunk']:parser.add_argument('--'+name,required=True)
 parser.add_argument('--session',type=int,required=True);parser.add_argument('--exit-code',type=int,required=True);main(parser.parse_args())
