"""Independently review completed captures after real replay failure; no model rerun."""
import argparse,hashlib,json,os,resource,signal,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];Q=R/'research';sys.path[:0]=[str(R),str(Q)]
import minicpm5_frame1_attention_runner_v4 as m
import review_minicpm5_frame1_attention_complete_v3 as independent
import minicpm5_attention_failed_r2_history_v1 as prior
ROOT=R/'minicpm5-frame1-attention-r3-20261108-v1'
TERMINAL=dict(path=str(Q/(ROOT.name+'-failed-terminal.json')),sha256='ebbabf495394f8614fd835f52028da199ca5d40b3651b7e85a3dc8acb1b00505')
EXIT=dict(path=str(Q/(ROOT.name+'-failed-verifier-external-exit.json')),sha256='e15ce89f5f664516c0aa492c1b8634facfc6b6c12421cb56d28f8c144d5c0a47')
BUDGET=dict(cpus=[6],address_space_bytes=2*2**30,cpu_seconds=600,wall_seconds=900,file_bytes=16*2**20)
SELF=Path(__file__).resolve()
def loaded_sources():return {str(Path(x.__file__).resolve()):m.p.sha(x.__file__)for x in tuple(sys.modules.values())if getattr(x,'__file__',None)and Path(x.__file__).is_relative_to(R)}
def validate(t,e):
 m.p.require(t['schema']=='minicpm5_attention_r3_failed_replay_terminal_v1'and t['reviewer']=='/root','Genuine r3 failure terminal')
 m.p.require(t['all_three_planned_groups_drained']is t['native_and_cpu_complete']is t['replay_failed']is True,'Original failure and successful capture phases retained')
 m.p.require(t['independent_saved_byte_review_complete']is t['full_model_qualified']is t['new_speedup_claim']is False,'No prior completion promotion')
 m.same((t['raw_files'],t['raw_bytes']),(96,353284922),'All original raw captures')
 m.same(e['terminal'],TERMINAL,'Actual verifier exit bound');m.same(e['owner'],t['verifier'],'Actual verifier owner');m.same(e['actual_exit'],dict(session_id=24839,exit_code=0,tool_chunk='cb23cd'),'Real verifier tool exit')
 m.p.require(e['actual_tool_exit_observed']is e['exact_owner_gone']is True,'Actual verifier completion')
 actual=m.p.read_ref(t['actual_external_exit']);m.same(actual['actual_exit'],dict(session_id=33412,exit_code=1,tool_chunk='25546f'),'Original failed model runner not relabeled')
 return t

def run(approval_ref):
 approval=m.p.read_ref(approval_ref);t=validate(m.p.read_ref(TERMINAL),m.p.read_ref(EXIT));spec=m.p.read_ref(t['spec']);admission=m.p.read_ref(t['admission'])
 m.p.require(approval['schema']=='minicpm5_attention_failed_replay_saved_review_approval_v1'and approval['reviewer']=='/root'and approval['passed']is approval['source_only']is approval['independent_review']is True and approval['blockers']==[],'Exact separately reviewed recovery source')
 m.same(approval['terminal'],TERMINAL,'Exact failure binding');m.same(approval['verifier_exit'],EXIT,'Actual external verification binding');m.same(approval['budgets'],BUDGET,'Original independent review budget')
 for p,h in approval['source_sha256'].items():m.same(m.p.sha(p),h,'Reviewed recovery source retained')
 for p,h in loaded_sources().items():m.same(approval['source_sha256'].get(p),h,'All loaded local recovery helpers pinned')
 m.same(approval['source_sha256'].get(str(SELF)),m.p.sha(SELF),'Recovery entrypoint pinned')
 for group in (m.p.read_ref(m.BASE_PLAN)['source_sha256'],spec['new_sources']):
  for p,h in group.items():m.same(m.p.sha(p),h,'Original successful capture source retained')
 paths={k:Q/(ROOT.name+'-recovered-review-'+k+'.json')for k in ('admission','result','run','failure')};m.p.require(not any(p.exists()for p in paths.values()),'Fresh recovery output namespace')
 fd,owner,lock=m.old.post.external.acquire_lock();started=time.monotonic()
 try:
  m.same(lock,t['lock_identity'],'Same original serial lock');m.p.clear_window();m.p.active_heavy();prior.physical(spec,owner,lock)
  for proc in [t['outer_owner'],t['verifier'],*t['phase_owners'].values(),*t['rank_owners']]:m.p.gone(proc)
  samples=[]
  for phase in m.PHASES:
   value=m.p.properties(spec['units'][phase]);m.service_identity(spec,t['admission'],admission['attempt_id'],phase,value,t['phase_owners'][phase]);m.p.require(m.p.terminal(value),'Original phase terminal');m.same((value['Result'],value['ExecMainCode'],value['ExecMainStatus']),('exit-code','1','1')if phase=='replay'else('success','1','0'),'Actual phase outcome retained');samples.append(dict(phase=phase,group=m.p.empty_group(m.p.GROUP_PREFIX+spec['units'][phase]+'.service')))
  os.sched_setaffinity(0,set(BUDGET['cpus']));os.nice(19);resource.setrlimit(resource.RLIMIT_AS,(BUDGET['address_space_bytes'],)*2);resource.setrlimit(resource.RLIMIT_CPU,(BUDGET['cpu_seconds'],BUDGET['cpu_seconds']+5));resource.setrlimit(resource.RLIMIT_FSIZE,(BUDGET['file_bytes'],)*2)
  def timeout(signum,frame):raise TimeoutError('Bounded saved-byte recovery wall budget')
  signal.signal(signal.SIGALRM,timeout);signal.alarm(BUDGET['wall_seconds'])
  def guard():
   m.p.require(time.monotonic()-started<BUDGET['wall_seconds'],'Finite independent review');m.p.lock_proof(owner,lock);m.p.clear_window();m.usage(ROOT,spec['root_floor_bytes']);return True
  ar=m.worker.once(paths['admission'],dict(terminal=TERMINAL,verifier_exit=EXIT,source_review=approval_ref,owner=owner,lock_identity=lock,budgets=BUDGET,original_phase_outcomes_preserved=True,groups=samples,actual_self_exit_claimed=False))
  def original_files():
   m.same({str(p)for p in ROOT.rglob('*')if p.is_file()},{x['path']for x in t['partial_files']},'Original complete retained file inventory')
   for row in t['partial_files']:
    guard();p=Path(row['path']);m.p.require(not p.is_symlink()and p.is_relative_to(ROOT)and p.stat().st_size==row['bytes'],'Original retained file identity/size');m.same(hashlib.sha256(p.read_bytes()).hexdigest(),row['sha256'],'Original complete saved file bytes unchanged')
  original_files();result=independent.review(root=ROOT,capture_refs=t['captures'],guard=guard)
  m.p.require(result['complete']is result['all96_full_payloads_authenticated']is result['all21_original_upstream_payloads_exact']is True,'Complete independent saved-byte review');m.same((result['files'],result['raw_bytes']),(96,353284922),'All original payloads reread')
  m.p.require(type(result['passed'])is bool and result['passed']is result['all21_own_input_controls_exact'],'Preserve actual computed scientific verdict')
  for k in ('full_model_qualified','new_speedup_claim','process_lifecycle_verified','unique_internal_or_final_KL_cause_proven'):m.p.require(result[k]is False,'No unrelated qualification claim')
  original_files();rr=m.worker.once(paths['result'],result);usage=resource.getrusage(resource.RUSAGE_SELF)
  m.same({n:m.p.sha(R/n)for n in m.CANONICAL},m.CANONICAL,'Canonical unchanged');m.p.require(not{'torch','numpy','transformers','torch_neuronx'}&set(sys.modules),'No native model or framework execution')
  run=m.worker.once(paths['run'],dict(schema='minicpm5_attention_failed_replay_recovered_review_run_v1',complete=True,owner=owner,terminal=TERMINAL,verifier_exit=EXIT,source_review=approval_ref,admission=ar,result=rr,original_runner_actual_exit=dict(session_id=33412,exit_code=1,tool_chunk='25546f'),original_failed_replay_preserved=True,model_rerun=False,all_original_files_unchanged=True,all21_own_input_controls_exact=result['all21_own_input_controls_exact'],measured_resources=dict(elapsed_seconds=time.monotonic()-started,user_seconds=usage.ru_utime,system_seconds=usage.ru_stime,peak_rss_bytes=usage.ru_maxrss*1024),actual_self_exit_claimed=False,full_model_qualified=False,new_speedup_claim=False))
  print(json.dumps(dict(run=run,result=rr,owner=owner,all21_own_input_controls_exact=result['all21_own_input_controls_exact'],actual_self_exit_claimed=False)),flush=True)
 except BaseException as error:
  m.old.guarded_receipt(paths['failure'],dict(owner=owner,terminal=TERMINAL,error_type=type(error).__name__,error=str(error),actual_self_exit_claimed=False),error);raise
 finally:signal.alarm(0);m.old.post.external.release_lock(fd)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--review',required=True);parser.add_argument('--review-sha256',required=True);args=parser.parse_args();run(dict(path=args.review,sha256=args.review_sha256))
