"""Load all original Dory BF16 weights and construct unchanged reference; no forward."""
import argparse,hashlib,json,os,resource,signal,sys,time
from pathlib import Path
Q=Path(__file__).resolve().parent;R=Q.parent;sys.path[:0]=[str(Q),str(R)]
import run_dory_weight_storage_root_v2 as d
import dory_reference_loader_bridge_v1 as bridge
SELF=Path(__file__).resolve();ROOT=R/'dory-original-weight-load-r2-20261108-v1'
BUDGET=dict(cpus=[6],address_space_bytes=16*2**30,cpu_seconds=600,wall_seconds=900,file_bytes=16*2**20,output_bytes=16*2**20,root_floor_bytes=28*2**30,available_memory_floor_bytes=8*2**30)
DOWNLOAD=dict(path=str(Q/'dory-checkpoint-download-r1-20261108-v1-result.json'),sha256='16b62956e1b2b908f59c0fe75539425514ad8448d63173a58a92e0d28bdb3792')
DOWNLOAD_EXIT=dict(path=str(Q/'dory-checkpoint-download-r1-20261108-v1-external-exit.json'),sha256='42aae901d1f6b8c0430025d765c5dd346ed64b0371cb22bec55b9bccc595125b')
STORAGE=dict(path=str(Q/'dory-weight-storage-r1-20261108-v1-root-terminal.json'),sha256='ac648635e066c3dd0b2904b456c1a775d9953a5ce3fe1db36d03345badddacf1')
STORAGE_EXIT=dict(path=str(Q/'dory-weight-storage-r1-20261108-v1-finalizer-external-exit.json'),sha256='2d69489aa0ab984d4b332b085cc0dc1833e281544f4cba3caff47c216b24ea16')
FAILED=dict(path=str(Q/'dory-original-weight-load-r1-20261108-v1-failed-terminal.json'),sha256='2e5ca76f97a76ca4a5de5baa89e521addbe0af498919e6aa2c540ae1cfa171b9')
FAILED_EXIT={'path': '/home/ubuntu/experiments/trn1-latest-models-20261031/research/dory-original-weight-load-r1-20261108-v1-failed-verifier-exit.json', 'sha256': '0e64a122b6ba0a475209621e8e55b3cc90d33843b195769ffe11c6106518f2cf'}

def check_failed_history(owner,lock):
 t=d.load(FAILED);e=d.load(FAILED_EXIT);d.m.same((e['terminal'],e['owner']),(FAILED,t['verifier']),'Actual prior failure verification');d.require(e['actual_tool_exit_observed']is e['exact_owner_gone']is True,'Actual prior verifier exit');d.m.same(e['actual_exit'],dict(session_id=None,exit_code=0,tool_chunk='ba5538'),'Actual prior verifier tool exit');d.m.same(t['actual_exit'],dict(session_id=4859,exit_code=1,tool_chunk='b9bd88'),'Original failed runtime retained');d.m.same(lock,t['lock_identity'],'Same original failed lock')
 for proc in (t['outer_owner'],t['verifier']):d.m.p.gone(proc)
 for row in t['partial_files']:d.m.same(hashlib.sha256(Path(row['path']).read_bytes()).hexdigest(),row['sha256'],'Original failed files preserved')
 return dict(terminal=FAILED,verifier_exit=FAILED_EXIT)

def source_guard_allows_service_query(barrier):return barrier is None

def sources():
 pins=d.sources()
 for p in (SELF,Path(bridge.__file__),Path(bridge.loader.__file__),bridge.CANDIDATE,bridge.loader.CONTRACT_PATH):pins[str(p)]=d.ref(p)['sha256']
 return pins
def available_memory():
 row=next(x for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:'));return int(row.split()[1])*1024
def predecessors():
 result=d.load(DOWNLOAD);ex=d.load(DOWNLOAD_EXIT);storage=d.load(STORAGE);se=d.load(STORAGE_EXIT)
 d.m.same((ex['result'],ex['owner']),(DOWNLOAD,result['owner']),'Actual checkpoint result and external owner');d.actual_exit(ex['actual_exit']);d.require(ex['actual_tool_exit_observed']is ex['exact_owner_gone']is ex['full_checkpoint_downloaded_and_hash_verified']is True,'Actual successful full checkpoint download')
 d.m.same((result['bytes'],result['sha256']),(bridge.loader.FILE_BYTES,bridge.loader.FULL_SHA),'Exact official checkpoint extent/hash');d.require(result['network_hash_verified']is result['whole_disk_hash_reverified']is result['original507_tensor_header_exact']is True,'Actual network and complete saved checkpoint verification')
 d.m.same(bridge.loader.identity(Path(result['checkpoint']['path'])),result['checkpoint'],'Original downloaded checkpoint identity')
 d.m.same((se['terminal'],se['owner']),(STORAGE,storage['verifier']),'Actual storage verifier result and owner');d.actual_exit(se['actual_exit']);d.require(se['actual_tool_exit_observed']is se['exact_owner_gone']is storage['all18_saved_tensors_reverified']is storage['native_owner_gone']is True,'Actual storage probe and independent review completion')
 d.m.same(storage['saved_raw_bytes'],786864,'Original complete storage fixture')
 return result,storage,[result['owner'],storage['verifier'],storage['native_owner'],storage['preparation_owner']]
def run(review_ref):
 review=d.load(review_ref);d.require(review['schema']=='dory_original_weight_load_source_review_v1'and review['reviewer']=='/root'and review['passed']is review['source_only']is review['independent_review']is True and review['blockers']==[],'Exact independently reviewed original weight loading')
 d.m.same((review['download'],review['download_exit'],review['storage'],review['storage_exit']),(DOWNLOAD,DOWNLOAD_EXIT,STORAGE,STORAGE_EXIT),'Exact actual predecessor references');d.m.same(review['budgets'],BUDGET,'Exact finite original loading budgets');d.m.same(review['prior_failure'],dict(terminal=FAILED,verifier_exit=FAILED_EXIT),'Original failed predecessor retained');d.m.same(review['source_sha256'],sources(),'Actual complete unchanged sources plus new loader executor')
 d.require(sys.executable==str(d.runtime.PYTHON)and sys.flags.no_site and sys.flags.isolated and sys.dont_write_bytecode and __debug__,'Original isolated native interpreter')
 d.require({k:os.environ.get(k)for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONNOUSERSITE')}==dict(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONNOUSERSITE='1'),'Original explicit single-thread native environment')
 d.require(not{'torch','numpy','transformers','torch_neuronx'}&set(sys.modules),'Authenticate before framework import')
 download,storage,owners=predecessors();request=d.validate_request(d.load(storage['request']));d.require(not ROOT.exists(),'Fresh original weight load output');d.m.p.user_bus();fd,owner,lock=d.m.old.post.external.acquire_lock();start=time.monotonic();barrier=None;created=False;last=[0.];model=None;memory_samples=[]
 try:
  d.m.same(lock,storage['lock_identity'],'Same serial queue');d.m.p.clear_window();d.m.p.active_heavy();check_failed_history(owner,lock)
  for p in owners:d.m.p.gone(p)
  d.require(available_memory()>12*2**30,'Actual memory headroom for full original weight loading')
  os.sched_setaffinity(0,{6});os.nice(19);resource.setrlimit(resource.RLIMIT_AS,(BUDGET['address_space_bytes'],)*2);resource.setrlimit(resource.RLIMIT_CPU,(BUDGET['cpu_seconds'],BUDGET['cpu_seconds']+5));resource.setrlimit(resource.RLIMIT_FSIZE,(BUDGET['file_bytes'],)*2)
  def timeout(signum,frame):raise TimeoutError('Dory original weight loading wall budget')
  signal.signal(signal.SIGALRM,timeout);signal.alarm(BUDGET['wall_seconds']);ROOT.mkdir(mode=0o700);created=True
  def guard():
   now=time.monotonic();d.require(now-start<BUDGET['wall_seconds'],'Finite original weight loading duration');d.require(os.sched_getaffinity(0)=={6},'Original CPU affinity')
   if now-last[0]>=.5:
    d.m.p.lock_proof(owner,lock);d.m.p.clear_window()
    if source_guard_allows_service_query(barrier):d.m.p.active_heavy()
    d.runtime.usage(ROOT);mem=available_memory();d.require(mem>=BUDGET['available_memory_floor_bytes'],'Retain available host memory');memory_samples.append(mem);last[0]=now
    if barrier is not None:barrier.verify()
  admission=d.once(ROOT/'admission.json',dict(owner=owner,lock_identity=lock,source_review=review_ref,storage=STORAGE,storage_exit=STORAGE_EXIT,download=DOWNLOAD,download_exit=DOWNLOAD_EXIT,checkpoint=download['checkpoint'],budgets=BUDGET,model_forward_permitted=False,prior_failure=dict(terminal=FAILED,verifier_exit=FAILED_EXIT),actual_self_exit_claimed=False))
  guard();runtime_files=d.runtime.validate_runtime_identity(request['runtime_identity'],True,guard);native_files=d.runtime.verify_torch_install(guard);d.once(ROOT/'runtime-files.json',dict(files=native_files,runtime_files=runtime_files,records=d.runtime.RECORDS))
  barrier=d.runtime.NoNativeChildren();barrier.verify();sys.addaudithook(barrier.audit);sys.path.append(str(d.runtime.BASE))
  import torch
  d.require(str(torch.__version__)=='2.9.1+cpu'and Path(torch.__file__).resolve()==d.runtime.BASE/'torch/__init__.py','Actual original PyTorch installation');torch.set_num_threads(1);torch.set_num_interop_threads(1)
  with torch.inference_mode():model,proof=bridge.build_reference(Path(download['checkpoint']['path']),download['checkpoint'],torch=torch,guard=guard)
  d.require(not model.slots and not model.epochs,'No inference requests or forward');d.m.same((len(model.weights),model.config.n_recurrent_loops,len(model.config.cache_plan())),(507,2,56),'All original physical weights and recurrent history configuration')
  proof_ref=d.once(ROOT/'loader-proof.json',proof);guard();usage=resource.getrusage(resource.RUSAGE_SELF)
  for row in native_files:d.require(d.runtime.file_identity(row['path'])==row['identity'],'Actual native dependencies unchanged')
  d.m.same(bridge.loader.identity(Path(download['checkpoint']['path'])),download['checkpoint'],'Checkpoint unchanged after full loading');d.m.same(review['source_sha256'],sources(),'All execution sources retained');d.m.same({n:d.m.p.sha(R/n)for n in d.m.CANONICAL},d.m.CANONICAL,'Canonical retained')
  result=d.once(ROOT/'result.json',dict(schema='dory_original_weight_load_result_v1',complete=True,passed=True,owner=owner,admission=admission,source_review=review_ref,checkpoint=download['checkpoint'],loader_proof=proof_ref,physical_tensors=507,parameters=3975229440,weight_bytes=7950458880,original_weights_loaded=True,full_reference_constructed=True,all_detached_weights_share_original_storage=proof['storage']['all_detached_weights_share_original_storage'],n_recurrent_loops=2,cache_histories=56,model_forward_executed=False,native_reference_qualified=False,neuron_baseline_qualified=False,full_model_inference_memory_fit_proven=False,new_speedup_claim=False,measured_resources=dict(elapsed_seconds=time.monotonic()-start,peak_rss_bytes=usage.ru_maxrss*1024,user_seconds=usage.ru_utime,system_seconds=usage.ru_stime,minimum_sampled_available_memory_bytes=min(memory_samples)),actual_self_exit_claimed=False))
  print(json.dumps(dict(result=result,owner=owner,actual_self_exit_claimed=False)),flush=True)
 except BaseException as error:
  if created and not(ROOT/'failure.json').exists():d.once(ROOT/'failure.json',dict(owner=owner,error_type=type(error).__name__,error=str(error),original_weight_load_qualified=False,actual_self_exit_claimed=False))
  raise
 finally:
  model=None;signal.alarm(0)
  if barrier is not None:barrier.active=False
  d.runtime.drain_children();d.m.old.post.external.release_lock(fd)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--review',required=True);parser.add_argument('--review-sha256',required=True);args=parser.parse_args();run(dict(path=args.review,sha256=args.review_sha256))
