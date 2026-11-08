"""Retire nine exact completed compiler intermediates; preserve original experiments."""
import hashlib,json,os,stat,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];Q=R/'research';sys.path.insert(0,str(R))
import minicpm5_upstream_completed_history_v1 as h
import nemotron9b_package_protocol_v15 as p
PLAN=dict(path=str(Q/'nemotron9b-r7-space-retirement-plan-20261108-v1.json'),sha256='b482eae9246668dbbe137ebf175e8e9031751588be09e09f76486b30c966abd6')
PREFIX='nemotron9b-r7-space-retirement-20261108-v1'
KEYS=('st_dev','st_ino','st_mode','st_uid','st_gid','st_nlink','st_size','st_mtime_ns','st_ctime_ns','st_blocks')
def identity(path):return {k:getattr(Path(path).lstat(),k)for k in KEYS}
def protected(plan):
 rows={}
 for root in plan['protected_roots']:
  for base,dirs,files in os.walk(root,followlinks=False):
   for name in files:
    path=Path(base)/name;rows[str(path)]=identity(path)
 return dict(files=len(rows),sha256=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest())
def scan():
 result=subprocess.run(['sudo','-n','/usr/bin/python3','-I','-B',str(Q/'inspect_nemotron9b_r7_retirement_processes_v1.py')],capture_output=True,text=True,timeout=30)
 p.require(result.returncode==0,result.stdout+result.stderr);return json.loads(result.stdout)
def free():return {str(x):os.statvfs(x).f_bavail*os.statvfs(x).f_frsize for x in (R,Path('/mnt/nvme'))}
def main():
 plan=p.metadata.load(PLAN);catalog=h.load(plan['catalog']);fd,owner,lock=h.m.old.post.external.acquire_lock();removed=[];started=time.monotonic()
 try:
  p.low.active_heavy();h.physical_drain(catalog);p.completed_history.finalizer(catalog,h.small(p.completed_history.EXIT))
  p.require(len(plan['files'])==9,'Exact nine completed compiler intermediates')
  names={row['path']for row in plan['files']};p.require(not names&set(catalog['source_sha256']),'No frozen source removal')
  refs=[p.metadata.load(x)for x in plan['retained_records']]
  p.require(refs[0]['complete']and not refs[0]['passed']and refs[1]['experiments_complete']and refs[3]['passed']and refs[3]['full_declared_workflow_complete'],'Original completed failures and qualified Qwen retained')
  parent=p.metadata.load(dict(path=str(Q/'nemotron9b-package-runtime-v14-r6-20261108-spec.json'),sha256='cd779a014effdca2e4fd9a8595b9764ba5e0b51d37df05279292ed266de2c4ff'))
  protected_before=protected(plan);canonical=p.canonical();before=free();process_scan=scan()
  p.once(Q/(PREFIX+'-intent.json'),dict(owner=owner,lock_identity=lock,plan=PLAN,protected=protected_before,canonical=canonical,free_before=before,scan=process_scan,source=p.ref(__file__)))
  hashes={}
  for row in plan['files']:
   path=Path(row['path']);s=path.lstat()
   p.require(identity(path)==row['identity']and stat.S_ISREG(s.st_mode)and s.st_nlink==1 and s.st_uid==os.getuid()and path.resolve()==path,'Exact owned regular compiler intermediate')
   hashes[str(path)]=p.stream_sha(path)
   p.require(identity(path)==row['identity'],'Compiler intermediate stable')
  p.once(Q/(PREFIX+'-hashes.json'),dict(plan=PLAN,sha256=hashes))
  scan()
  with (Q/(PREFIX+'-actions.jsonl')).open('x')as stream:
   for row in plan['files']:
    path=Path(row['path']);p.require(identity(path)==row['identity'],'Delete exact previously hashed intermediate');path.unlink();removed.append(str(path))
    stream.write(json.dumps(dict(deleted=str(path)))+'\n');stream.flush();os.fsync(stream.fileno())
  p.require(all(not Path(name).exists()for name in names),'All selected intermediates absent')
  p.require(protected(plan)==protected_before and p.canonical()==canonical,'Serving artifacts and canonical files unchanged')
  for name,row in parent['payloads'].items():p.require(p.file_stat(name)==row['stat'],'Original model/oracle payload stat retained')
  for name,identity_ in parent['runtime_binary_stat'].items():p.require(p.file_stat(name)==identity_,'Original runtime binary stat retained')
  p.compiler_links.verify_assets(p.compiler_links.load_policy(),hash_content=True)
  result=dict(schema='nemotron9b_r7_completed_intermediates_retirement_result_v1',complete=True,owner=owner,lock_identity=lock,plan=PLAN,catalog=plan['catalog'],source=p.ref(__file__),intent=p.ref(Q/(PREFIX+'-intent.json')),hashes=p.ref(Q/(PREFIX+'-hashes.json')),actions=p.ref(Q/(PREFIX+'-actions.jsonl')),removed=removed,free_before=before,free_after=free(),protected_unchanged=True,canonical_unchanged=True,original_payload_and_runtime_stats_unchanged=True,compiler_assets_unchanged=True,actual_self_exit_claimed=False,model_or_compiler_started=False,new_speedup_claim=False,elapsed_seconds=time.monotonic()-started)
  p.once(Q/(PREFIX+'-result.json'),result);print(json.dumps(result),flush=True)
 except BaseException as error:
  p.once(Q/(PREFIX+'-failure.json'),dict(owner=owner,plan=PLAN,removed=removed,error=repr(error)));raise
 finally:h.m.old.post.external.release_lock(fd)
if __name__=='__main__':main()
