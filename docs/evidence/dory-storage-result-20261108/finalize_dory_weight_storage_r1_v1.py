"""Separate post-exit byte review of actual Dory storage probe; no native imports."""
import argparse,hashlib,json,os,resource,signal,sys,time
from pathlib import Path
Q=Path(__file__).resolve().parent;R=Q.parent;sys.path[:0]=[str(Q),str(R)]
import run_dory_weight_storage_root_v2 as d
SELF=Path(__file__).resolve();BUDGET=dict(cpus=[6],address_space_bytes=2*2**30,cpu_seconds=90,wall_seconds=120)
def validate(exited,live,t,request):
 d.require(exited['schema']=='dory_weight_storage_actual_external_exit_v1'and exited['reviewer']=='/root','Actual root-observed storage exit')
 d.actual_exit(exited['actual_exit']);d.require(exited['actual_tool_exit_observed']is exited['exact_owner_gone']is True and exited['new_speedup_claim']is False,'Actual external completion')
 d.require(live['schema']=='dory_weight_storage_actual_live_v1'and live['actual_identity_verified']is True and live['actual_exit_observed']is False,'Original actual native live identity')
 d.m.same(exited['actual_exit']['session_id'],live['actual_tool_session'],'Original actual tool session')
 for k in ('owner','request','review'):d.m.same(exited[k],live[k],'Actual original '+k)
 d.require(t['schema']=='dory_weight_storage_actual_terminal_v1'and t['complete']is t['passed']is True,'Actual native storage result complete')
 d.m.same((t['owner'],t['request']),(exited['owner'],exited['request']),'Original native result owner and request')
 d.m.same((t['saved_tensors'],t['saved_raw_bytes']),(18,786864),'Complete unchanged storage fixture')
 d.require(t['native_storage_lifetime_observed']is t['all_saved_words_exact']is True,'Native lifetime and storage checks passed')
 for k in ('actual_self_exit_claimed','original_weights_loaded','model_constructed','full_model_memory_fit_proven','full_model_qualified','new_speedup_claim'):d.require(t[k]is False,'Storage probe cannot promote model result')
 d.m.same(request['output_root'],str(R/'dory-weight-storage-r1-20261108-v1'),'Exact actual storage output namespace')
 return t

def finalize(exit_ref):
 e=d.load(exit_ref);live=d.load(e['live']);request=d.validate_request(d.load(e['request']));root=Path(request['output_root']);tr=d.ref(root/'terminal.json');t=validate(e,live,d.load(tr),request)
 approval=d.load(e['review']);d.require(approval['passed']is approval['source_only']is approval['independent_review']is True and approval['blockers']==[],'Original native source review');d.m.same(approval['request'],e['request'],'Original approved request');d.m.same(approval['source_sha256'],request['source_sha256'],'Actual approved native source closure')
 admission=d.load(t['admission']);d.m.same((admission['request'],admission['review'],admission['owner'],admission['lock_identity']),(e['request'],e['review'],e['owner'],request['lock_identity']),'Actual native admission')
 dest=Q/'dory-weight-storage-r1-20261108-v1-post-exit-review.json';target=Q/'dory-weight-storage-r1-20261108-v1-root-terminal.json';d.require(not dest.exists()and not target.exists(),'Fresh post-exit review outputs')
 fd,owner,lock=d.m.old.post.external.acquire_lock();start=time.monotonic()
 try:
  d.m.same(lock,admission['lock_identity'],'Same serial lock');d.m.p.clear_window();d.m.p.active_heavy()
  for _ in range(2):d.m.p.gone(e['owner']);d.m.p.gone(request['preparation_owner']);d.m.p.lock_proof(owner,lock)
  os.sched_setaffinity(0,set(BUDGET['cpus']));os.nice(19);resource.setrlimit(resource.RLIMIT_AS,(BUDGET['address_space_bytes'],)*2);resource.setrlimit(resource.RLIMIT_CPU,(BUDGET['cpu_seconds'],BUDGET['cpu_seconds']+5))
  def timeout(signum,frame):raise TimeoutError('Storage post-exit review wall limit')
  signal.signal(signal.SIGALRM,timeout);signal.alarm(BUDGET['wall_seconds'])
  def guard():
   d.require(time.monotonic()-start<BUDGET['wall_seconds'],'Finite independent review');d.m.p.lock_proof(owner,lock);d.m.p.clear_window();d.runtime.usage(root);return True
  before={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in root.rglob('*')if p.is_file()}
  result=d.capture.review(t['result'],guard);d.m.same(result,d.load(t['independent_review']),'Entire saved-byte review independently reproduced after native exit');review_ref=d.once(dest,result)
  after={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in root.rglob('*')if p.is_file()};d.m.same(before,after,'Every original native output unchanged')
  d.require(not{'torch','numpy','transformers','torch_neuronx'}&set(sys.modules),'Post-exit review does not load native frameworks');usage=resource.getrusage(resource.RUSAGE_SELF)
  d.m.same({n:d.m.p.sha(R/n)for n in d.m.CANONICAL},d.m.CANONICAL,'Canonical unchanged')
  value=dict(schema='dory_weight_storage_root_terminal_v1',reviewer='/root',complete=True,passed=True,actual_external_exit=exit_ref,native_terminal=tr,request=e['request'],source_review=e['review'],native_owner=e['owner'],preparation_owner=request['preparation_owner'],verifier=owner,lock_identity=lock,verifier_source=d.ref(SELF),verifier_source_sha256={str(Path(x.__file__).resolve()):d.m.p.sha(x.__file__)for x in tuple(sys.modules.values())if getattr(x,'__file__',None)and Path(x.__file__).is_relative_to(R)},post_exit_review=review_ref,all18_saved_tensors_reverified=True,saved_raw_bytes=786864,actual_native_exit_observed=True,native_owner_gone=True,all_original_outputs_unchanged=True,original_files_sha256=before,native_lifetime_independently_reexecuted=False,measured_review_resources=dict(elapsed_seconds=time.monotonic()-start,peak_rss_bytes=usage.ru_maxrss*1024),actual_self_exit_claimed=False,original_weights_loaded=False,full_model_qualified=False,new_speedup_claim=False)
  out=d.once(target,value);print(json.dumps(dict(terminal=out,review=review_ref,verifier=owner,actual_self_exit_claimed=False)),flush=True)
 finally:signal.alarm(0);d.m.old.post.external.release_lock(fd)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--actual-exit',required=True);parser.add_argument('--actual-exit-sha256',required=True);args=parser.parse_args();finalize(dict(path=args.actual_exit,sha256=args.actual_exit_sha256))
