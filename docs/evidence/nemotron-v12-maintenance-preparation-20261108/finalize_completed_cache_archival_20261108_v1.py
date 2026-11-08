"""Root finalizer, runnable only after actual sio14 and archive-tool completion."""
import argparse,copy,datetime,hashlib,json,os,re,stat,sys,tarfile
from pathlib import Path
R=Path(__file__).resolve().parents[1];Q=R/'research';sys.path.insert(0,str(R))
import nemotron9b_package_protocol_v12 as p
import nemotron9b_post_smol_maintenance_v1 as bridge
import minicpm5_python_trace_execute_v4 as lifecycle
from research.finalize_smollm3_sio9_root_20261107_v1 import physical_drain
from research import archive_completed_neuron_cache_20261108_v2 as archival
EXIT=Q/'completed-cache-archival-root-exit-20261108-v2.json'
AUDIT=Q/'completed-cache-archival-root-terminal-20261108-v2.json'
CATALOG=Q/'nemotron9b-package-post-smol-maintenance-r2-20261108-catalog-v1.json'

def digest_stream(stream):
 h=hashlib.sha256()
 while data:=stream.read(2**20):h.update(data)
 return h.hexdigest()

def ref(path,expected=None):
 return dict(path=str(path),sha256=p.metadata.sha(path,expected))

def bytes_for(value,cap=16*2**20):
 data=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode();assert len(data)<=cap;return data

def value_ref(path,value):
 return dict(path=str(path),sha256=hashlib.sha256(bytes_for(value)).hexdigest())

def record(path,value):
 data=bytes_for(value)
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'wb')as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
 p.metadata.sync_parent(path)
 return dict(path=str(path),sha256=hashlib.sha256(data).hexdigest())

def verify_archive(result,manifest,candidate):
 path=Path(result['archive']['path']);assert path==bridge.ARCHIVE_ROOT/'cache.tar.gz'and path.resolve()==path
 before=path.lstat();assert stat.S_ISREG(before.st_mode)and before.st_nlink==1
 with path.open('rb')as stream:assert digest_stream(stream)==result['archive']['sha256']
 files=manifest['files'];directories=manifest['directories'];seen=set();seen_dirs=set();root=Path(candidate['cache_root'])
 with tarfile.open(path,'r:gz')as archive:
  for member in archive:
   if member.isdir():
    key=str(root/member.name);assert key in directories and key not in seen_dirs;seen_dirs.add(key);row=directories[key]
   else:
    assert member.isfile()and member.name in files and member.name not in seen;seen.add(member.name);row=files[member.name]
    assert member.size==row['size']
    with archive.extractfile(member)as stream:assert digest_stream(stream)==row['sha256']
   assert member.mode==stat.S_IMODE(row['mode'])and member.uid==row['uid']and member.gid==row['gid']
 assert seen==set(files)and seen_dirs==set(directories)
 after=path.lstat();keys=['st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns','st_nlink']
 assert all(getattr(before,k)==getattr(after,k)for k in keys)
 assert all(not Path(name).exists()for name in directories)
 return dict(files_rehashed=len(seen),directories_verified=len(seen_dirs),archive=result['archive'],manifest=result['manifest'],archive_identity=p.file_stat(path))

def build(prior,prior_ref,start,start_ref,result,result_ref,session,chunk,observed,drains,archive_check,source_refs):
 assert type(session)is int and session>0 and re.fullmatch('[0-9a-f]{6}',chunk)
 assert archive_check['files_rehashed']==1039 and archive_check['directories_verified']==253
 assert len(drains)==len(prior['rows'])and {row['unit']for row in drains}=={row['unit']for row in prior['rows']}
 owner=start['owner']
 exited=dict(schema='completed_cache_archival_root_exit_v1',reviewer='/root',observed_at=observed,owner=copy.deepcopy(owner),actual_exit_observed_by_root=True,gone=True,terminal=True,exit_code=0,result=result_ref,
  execution_handle=dict(kind='tool_session',session_id=session,actual_completion_observed=True,exit_code=0,tool_chunk=chunk))
 ext=dict(prior_catalog=prior_ref,started=start_ref,result=result_ref,actual_exit=value_ref(EXIT,exited),source=bridge.SOURCE,candidate=bridge.CANDIDATE)
 audit=dict(schema='completed_cache_archival_root_terminal_v1',reviewer='/root',observed_at=observed,passed=True,references=copy.deepcopy(ext),owner=copy.deepcopy(owner),all_current_groups_recursively_drained=True,owner_absence_observed=True,archive_bytes_rehashed=True,
  archive=result['archive'],manifest=result['manifest'],canonical_sha256=result['canonical_sha256'],full_model_qualified=False,new_speedup_claim=False,complete_services=len(prior['rows']),complete_processes=len(prior['processes'])+1,
  physical_observations=drains,archive_verification=archive_check,source=source_refs[0])
 ext['root_audit']=value_ref(AUDIT,audit)
 catalog=copy.deepcopy(prior);catalog.update(schema=bridge.SCHEMA,maintenance_extension=ext,processes=copy.deepcopy(prior['processes'])+[dict(owner,role='completed_cache_archival_maintenance')])
 if 'process_predecessors'in catalog:catalog['process_predecessors']=copy.deepcopy(catalog['processes'])
 for reference in [*ext.values(),result['manifest'],*source_refs]:
  name,digest=reference['path'],reference['sha256'];assert name not in catalog['source_sha256']or catalog['source_sha256'][name]==digest;catalog['source_sha256'][name]=digest
 reference=value_ref(CATALOG,catalog)
 spec=dict(final_predecessor_catalog=reference,source_sha256=dict(catalog['source_sha256'],**{reference['path']:reference['sha256']}),required_predecessor_releases=prior['root_release_refs'],required_post_smol_maintenance=ext)
 return exited,audit,catalog,spec

def main(args):
 assert sys.dont_write_bytecode and __debug__
 assert type(args.session)is int and args.session>0 and re.fullmatch('[0-9a-f]{6}',args.tool_chunk)
 assert args.exit_code==0,'Root must observe successful original archive-tool completion'
 assert not any(path.exists()for path in [EXIT,AUDIT,CATALOG])
 assert ref(Path(bridge.SOURCE['path']))==bridge.SOURCE and ref(Path(bridge.CANDIDATE['path']))==bridge.CANDIDATE
 prior_ref,prior=archival.completed_catalog(args)
 start_ref=dict(path=str(bridge.ARCHIVE_ROOT/'start.json'),sha256=args.start_sha256)
 result_ref=dict(path=str(Q/'storage-nvme-completed-cache-retirement-20261108-v2.json'),sha256=args.result_sha256)
 start=p.metadata.load(start_ref);result=p.metadata.load(result_ref);manifest=p.metadata.load(result['manifest']);candidate=p.metadata.load(bridge.CANDIDATE)
 assert not (bridge.ARCHIVE_ROOT/'failure.json').exists()
 fd,owner,lock=lifecycle.acquire()
 try:
  p.low.gone(start['owner']);physical_drain(prior);p.low.active_heavy();p.accelerator_idle()
  assert p.canonical()==result['canonical_sha256']==start['canonical_sha256']
  verification=verify_archive(result,manifest,candidate)
  drains=physical_drain(prior);p.low.gone(start['owner']);p.low.lock_proof(owner,lock);p.low.active_heavy();p.accelerator_idle()
  for reference in [prior_ref,start_ref,result_ref,result['manifest'],bridge.SOURCE,bridge.CANDIDATE]:assert ref(Path(reference['path']),reference['sha256'])==reference
  assert p.canonical()==result['canonical_sha256']
  source_refs=[ref(Path(__file__).resolve()),ref(Q/'check_finalize_completed_cache_archival_20261108_v1.py'),ref(Path(bridge.__file__))]
  exited,audit,catalog,spec=build(prior,prior_ref,start,start_ref,result,result_ref,args.session,args.tool_chunk,datetime.datetime.now(datetime.timezone.utc).isoformat(),drains,verification,source_refs)
  pending={str(EXIT):exited,str(AUDIT):audit}
  def loaded(reference):
   if reference['path']in pending:
    value=pending[reference['path']];assert value_ref(Path(reference['path']),value)==reference;return value
   return p.metadata.load(reference)
  anchor=bridge.validate(spec,catalog,loaded);assert anchor['catalog']==prior and anchor['fresh_physical_drain_proven']is False
  # All independent validation precedes publication. Exclusive files preserve prior history.
  assert record(EXIT,exited)==spec['required_post_smol_maintenance']['actual_exit']
  assert record(AUDIT,audit)==spec['required_post_smol_maintenance']['root_audit']
  assert record(CATALOG,catalog)==spec['final_predecessor_catalog']
  assert p.metadata.load(spec['final_predecessor_catalog'])==catalog
  print(json.dumps(dict(catalog=spec['final_predecessor_catalog'],actual_exit=ref(EXIT),audit=ref(AUDIT),complete_services=len(catalog['rows']),complete_processes=len(catalog['processes']),new_speedup_claim=False)))
 finally:lifecycle.release(fd)
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ['catalog','catalog-sha256','start-sha256','result-sha256','tool-chunk']:parser.add_argument('--'+name,required=True)
 parser.add_argument('--session',type=int,required=True);parser.add_argument('--exit-code',type=int,required=True)
 main(parser.parse_args())
