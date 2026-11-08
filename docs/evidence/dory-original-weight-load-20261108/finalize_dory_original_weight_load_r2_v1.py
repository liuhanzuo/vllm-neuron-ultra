"""Independent post-exit complete checkpoint/tensor rehash; no model imports."""
from pathlib import Path
import hashlib,json,os,resource,signal,sys,time,struct
Q=Path(__file__).resolve().parent;R=Q.parent;sys.path[:0]=[str(Q),str(R)]
import run_dory_original_weight_load_v2 as a
d=a.d
EXIT=dict(path=str(Q/'dory-original-weight-load-r2-20261108-v1-actual-external-exit.json'),sha256='5649633884d8ab8d80b9ea7cb41f13b69075446989f404b0628ba2cc1bbf1e75')
BUDGET=dict(cpus=[6],address_space_bytes=2*2**30,cpu_seconds=180,wall_seconds=300)
def main():
 e=d.load(EXIT);live=d.load(e['live']);result=d.load(e['result']);proof=d.load(result['loader_proof']);admission=d.load(result['admission']);review=d.load(result['source_review']);contract=a.bridge.loader.source_contract()
 d.m.same(e['actual_exit'],dict(session_id=14230,exit_code=0,tool_chunk='569ec5'),'Actual successful original weight load tool exit');d.require(e['actual_tool_exit_observed']is e['exact_owner_gone']is live['actual_identity_verified']is True,'Actual live and exit observed');d.m.same(live['actual_tool_session'],14230,'Original actual session')
 for item in (e,live,result,admission):d.m.same(item['owner'],e['owner'],'Original actual loader owner')
 d.m.same((e['source_review'],live['source_review'],admission['source_review']),(result['source_review'],)*3,'Original runtime review')
 d.require(review['passed']is review['source_only']is review['independent_review']is True and review['blockers']==[],'Original exact source review');d.m.same(review['source_sha256'],a.sources(),'All execution sources retained')
 d.require(result['complete']is result['passed']is result['original_weights_loaded']is result['full_reference_constructed']is result['all_detached_weights_share_original_storage']is True,'Actual complete original weight load')
 d.m.same((result['physical_tensors'],result['parameters'],result['weight_bytes'],result['n_recurrent_loops'],result['cache_histories']),(507,3975229440,7950458880,2,56),'All original tensors/configuration')
 for key in ('model_forward_executed','native_reference_qualified','neuron_baseline_qualified','full_model_inference_memory_fit_proven','new_speedup_claim','actual_self_exit_claimed'):d.require(result[key]is False,'No forward or qualification promotion')
 storage=proof['storage'];load=proof['loader'];d.require(storage['complete']is storage['all_detached_weights_share_original_storage']is storage['all_physical_weights_have_distinct_buffers']is storage['learned_scales_included']is True and storage['tensor_payload_copied_by_bridge']is False,'Actual recorded storage adoption')
 d.m.same((load['file_bytes_read'],load['weight_bytes'],load['full_sha256']),(contract['file_bytes'],contract['data_bytes'],contract['full_sha256']),'Original full loader proof');d.require(load['full_file_hash_verified']is True,'Actual loader complete hash verified')
 hashes={x['name']:x for x in load['tensors']};shared={x['name']:x for x in storage['tensors']};d.m.same(set(hashes),set(shared),'Complete tensor hash/adoption mapping');d.m.same(len(hashes),507,'Complete507 unique tensor hashes')
 checkpoint=Path(result['checkpoint']['path']);d.m.same(a.bridge.loader.identity(checkpoint),result['checkpoint'],'Original unchanged checkpoint identity');fd,owner,lock=d.m.old.post.external.acquire_lock();start=time.monotonic()
 try:
  d.m.same(lock,admission['lock_identity'],'Same serial lock');d.m.p.clear_window();d.m.p.active_heavy();a.check_failed_history(owner,lock)
  for _ in range(2):d.m.p.gone(e['owner']);d.m.p.lock_proof(owner,lock)
  os.sched_setaffinity(0,{6});os.nice(19);resource.setrlimit(resource.RLIMIT_AS,(BUDGET['address_space_bytes'],)*2);resource.setrlimit(resource.RLIMIT_CPU,(BUDGET['cpu_seconds'],BUDGET['cpu_seconds']+5))
  def timeout(signum,frame):raise TimeoutError('Independent original weight audit wall budget')
  signal.signal(signal.SIGALRM,timeout);signal.alarm(BUDGET['wall_seconds']);allhash=hashlib.sha256();rows=[]
  with checkpoint.open('rb')as source:
   head=source.read(8+contract['header_bytes'])if 'header_bytes'in contract else source.read(8+a.bridge.loader.HEADER_BYTES)
   d.require(head==struct.pack('<Q',a.bridge.loader.HEADER_BYTES)+contract['header'],'Whole original header');allhash.update(head)
   for row in contract['rows']:
    digest=hashlib.sha256();remaining=row['end']-row['start'];size=remaining
    while remaining:
     raw=source.read(min(8*2**20,remaining));d.require(bool(raw),'Full checkpoint tensor without truncation');digest.update(raw);allhash.update(raw);remaining-=len(raw)
    want=dict(name=row['name'],shape=row['shape'],dtype='BF16',bytes=size,sha256=digest.hexdigest());d.m.same(hashes[row['name']],want,'Independent full tensor hash matches actual loader');d.m.same(shared[row['name']],dict(name=row['name'],shape=row['shape'],bytes=size,loader_sha256=want['sha256'],same_storage=True),'Original actual storage record and rehashed bytes');rows.append(want);d.m.p.lock_proof(owner,lock);d.m.p.clear_window()
   d.require(not source.read(1),'No trailing checkpoint bytes')
  d.m.same(allhash.hexdigest(),contract['full_sha256'],'Independent full official file hash');d.m.same(a.bridge.loader.identity(checkpoint),result['checkpoint'],'Checkpoint unchanged throughout audit');d.require(not{'torch','numpy','transformers','torch_neuronx'}&set(sys.modules),'Independent stdlib-only byte review')
  d.m.same({n:d.m.p.sha(R/n)for n in d.m.CANONICAL},d.m.CANONICAL,'Canonical unchanged')
  out=d.once(Q/'dory-original-weight-load-r2-20261108-v1-root-terminal.json',dict(schema='dory_original_weight_load_root_terminal_v1',reviewer='/root',complete=True,passed=True,actual_external_exit=EXIT,original_result=e['result'],loader_proof=result['loader_proof'],source_review=result['source_review'],native_owner=e['owner'],verifier=owner,verifier_source=d.ref(Path(__file__)),lock_identity=lock,checkpoint=result['checkpoint'],all507_tensor_hashes_independently_recomputed=True,whole_checkpoint_sha256=allhash.hexdigest(),tensors=rows,actual_native_exit_observed=True,native_owner_gone=True,original_weight_load_succeeded=True,full_reference_constructed=True,storage_aliasing_report_authenticated=True,storage_aliasing_independently_reexecuted=False,finite_values_independently_rechecked=False,model_forward_executed=False,native_reference_qualified=False,neuron_baseline_qualified=False,new_speedup_claim=False,actual_self_exit_claimed=False,measured_resources=dict(elapsed_seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)));print(json.dumps(dict(terminal=out,verifier=owner,actual_self_exit_claimed=False)),flush=True)
 finally:signal.alarm(0);d.m.old.post.external.release_lock(fd)
if __name__=='__main__':main()
