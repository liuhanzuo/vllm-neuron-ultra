"""Bounded post-exit Mini attention saved-byte review. No model imports or launch."""
import argparse,datetime,hashlib,json,os,resource,signal,sys,time,re
from pathlib import Path
R=Path(__file__).resolve().parents[1];Q=R/'research';sys.path[:0]=[str(R),str(Q)]
import minicpm5_frame1_attention_runner_v2 as m
CALLBACK=Q/'review_minicpm5_frame1_attention_complete_v1.py'
CALLBACK_SHA='64343e8d37b48fb2c180a2cffe62d90eebda2a35746224057235e2bc661cf77b'
BUDGET=dict(cpus=[6],address_space_bytes=2*2**30,cpu_seconds=600,wall_seconds=900,file_bytes=16*2**20)
SELF=Path(__file__).resolve()

def validate_terminal(t):
 m.p.require(t['schema']=='minicpm5_frame1_attention_root_terminal_v1'and t['reviewer']=='/root','Actual Mini attention terminal')
 m.p.require(t['actual_tool_exit_observed']is t['exact_outer_owner_gone']is t['all_three_groups_physically_drained']is True,'External exit and complete physical drain')
 actual=t['actual_exit'];m.p.require(type(actual['session_id'])is int and actual['session_id']>0 and type(actual['exit_code'])is int and actual['exit_code']==0,'Actual successful process exit')
 m.p.require(t['all_three_phases_succeeded']is True and set(t['phase_owners'])==set(m.PHASES)and len(t['rank_owners'])==2,'All original phases and ranks accounted')
 m.p.require(t['full_model_qualified']is t['new_speedup_claim']is False,'No local diagnostic promotion')
 m.p.require(re.fullmatch('[0-9a-f]{6}',actual['tool_chunk'])is not None,'Actual external tool chunk')
 m.p.require(type(t['all_own_input_controls_exact'])is bool,'Scientific outcome recorded independently of process success')
 m.same((t['captured_boundary_files'],t['norm_weight_files'],t['internal_attention_files'],t['computed_control_files'],t['raw_bytes']),(21,6,48,21,353284922),'Complete original capture scope')
 return t

def compare_review(result,recorded):
 m.p.require(result['schema']=='minicpm5_frame1_attention_complete_review_v1'and result['complete']is True,'Complete independent saved-byte result')
 m.p.require(type(result['passed'])is bool and result['passed']is result['all21_own_input_controls_exact'],'Derived own-input outcome')
 m.p.require(result['all96_full_payloads_authenticated']is result['all21_original_upstream_payloads_exact']is True,'Complete original raw evidence')
 m.same((result['files'],result['raw_bytes']),(96,353284922),'Exact raw count and budget')
 for name in ('full_model_qualified','new_speedup_claim','process_lifecycle_verified','unique_internal_or_final_KL_cause_proven'):
  m.p.require(result[name]is False,'Saved-byte review cannot promote unrelated claims')
 m.same(result,recorded,'All rederived bytes, metrics and controls match actual phase report')
 return result['passed']

def review_run(terminal_ref,source_review_ref):
 t=validate_terminal(m.p.read_ref(terminal_ref));approval=m.p.read_ref(source_review_ref)
 m.p.require(approval['schema']=='minicpm5_attention_post_exit_source_review_v1'and approval['reviewer']=='/root'and approval['passed']is approval['source_only']is approval['independent_review']is True and approval['blockers']==[],'Separate post-exit source review')
 m.same(approval['terminal'],terminal_ref,'Exact actual terminal');m.same(approval['budgets'],BUDGET,'Original bounded independent review')
 for path,wanted in approval['source_sha256'].items():m.same(m.p.sha(path),wanted,'Exact separate reviewed source')
 m.p.require(approval['source_sha256'].get(str(SELF))==m.p.sha(SELF)and approval['source_sha256'].get(str(CALLBACK))==CALLBACK_SHA,'Runner and callback authenticated')
 spec=m.p.read_ref(t['spec']);root=Path(spec['output_root']);m.same(str(root),t['output_root'],'Actual output namespace')
 m.p.require(root.parent==R and re.fullmatch('minicpm5-frame1-attention-r[0-9]+-20261108-v[0-9]+',root.name),'Finite attention output')
 m.same(spec['schema'],'minicpm5_frame1_attention_spec_v1','Exact attention runtime specification')
 m.same(spec['runner'],m.SELF,'Original admitted attention runner')
 m.same(spec['independent_reviewer'],str(CALLBACK),'Actual full saved-byte callback')
 admission=m.p.read_ref(t['references']['admission.json']);m.same(admission['spec'],t['spec'],'Actual original spec')
 refs={role:m.p.read(root/(role+'-result.json'))['capture']for role in ('native','cpu-rank0','cpu-rank1')}
 replay=m.old.reference(root/'attention-review.json');recorded=m.p.read_ref(replay)
 m.same(m.p.read(root/'result.json')['report'],replay,'Exact completed phase review')
 m.same(t['references']['attention-review.json'],replay,'Actual terminal bound review')
 m.same(recorded['captures'],refs,'Actual completed captures bound by original review')
 m.same(recorded['all21_own_input_controls_exact'],t['all_own_input_controls_exact'],'Terminal preserves failed scientific controls')
 prefix=Q/(root.name+'-independent-review')
 outputs={name:Path(str(prefix)+'-'+name+'.json')for name in ('admission','result','run','failure')}
 m.p.require(not any(p.exists()for p in outputs.values()),'Never overwrite prior review')
 fd,owner,lock=m.old.post.external.acquire_lock();started=time.monotonic()
 try:
  m.same(lock,t['lock_identity'],'Same serial lock inode');m.p.active_heavy()
  for prior in [t['outer_owner'],t['terminal_observer'],*t['phase_owners'].values(),*t['rank_owners']]:m.p.gone(prior)
  for phase,unit in spec['units'].items():
   value=m.p.properties(unit);m.service_identity(spec,t['references']['admission.json'],admission['attempt_id'],phase,value,t['phase_owners'][phase])
   m.p.require(m.p.terminal(value)and value['Result']=='success'and value['ExecMainStatus']=='0','Original successful service still terminal')
   m.p.empty_group(m.p.GROUP_PREFIX+unit+'.service')
  os.sched_setaffinity(0,set(BUDGET['cpus']));os.nice(19)
  resource.setrlimit(resource.RLIMIT_AS,(BUDGET['address_space_bytes'],)*2)
  resource.setrlimit(resource.RLIMIT_CPU,(BUDGET['cpu_seconds'],BUDGET['cpu_seconds']+5))
  resource.setrlimit(resource.RLIMIT_FSIZE,(BUDGET['file_bytes'],)*2)
  def timeout(signum,frame):raise TimeoutError('Mini attention saved-byte review wall budget')
  signal.signal(signal.SIGALRM,timeout);signal.alarm(BUDGET['wall_seconds'])
  def guard():
   m.p.require(time.monotonic()-started<BUDGET['wall_seconds'],'Bounded actual independent review')
   m.p.lock_proof(owner,lock);m.p.clear_window();m.usage(root,spec['root_floor_bytes'])
   return True
  guard();ar=m.worker.once(outputs['admission'],dict(terminal=terminal_ref,source_review=source_review_ref,owner=owner,lock_identity=lock,budgets=BUDGET,actual_self_exit_claimed=False))
  import review_minicpm5_frame1_attention_complete_v1 as independent
  result=independent.review(root=root,capture_refs=refs,guard=guard)
  compare_review(result,recorded)
  result_ref=m.worker.once(outputs['result'],result);usage=resource.getrusage(resource.RUSAGE_SELF)
  m.p.require(not {'torch','numpy','transformers','torch_neuronx'}&set(sys.modules),'Independent stdlib-only review')
  value=dict(schema='minicpm5_attention_independent_review_run_v1',complete=True,owner=owner,terminal=terminal_ref,source_review=source_review_ref,admission=ar,result=result_ref,actual_model_exit=t['actual_exit'],model_exit_verified=True,physical_drain_verified=True,original_phase_report=replay,complete_report_reproduced=True,all21_own_input_controls_exact=result['all21_own_input_controls_exact'],measured_resources=dict(elapsed_seconds=time.monotonic()-started,user_seconds=usage.ru_utime,system_seconds=usage.ru_stime,peak_rss_bytes=usage.ru_maxrss*1024),actual_self_exit_claimed=False,full_model_qualified=False,new_speedup_claim=False)
  m.same({name:m.p.sha(R/name)for name in m.CANONICAL},m.CANONICAL,'Canonical files unchanged');run_ref=m.worker.once(outputs['run'],value)
  print(json.dumps(dict(run=run_ref,result=result_ref,owner=owner,all21_own_input_controls_exact=result['all21_own_input_controls_exact'],actual_self_exit_claimed=False)),flush=True)
 except BaseException as error:
  m.old.guarded_receipt(outputs['failure'],dict(owner=owner,terminal=terminal_ref,error_type=type(error).__name__,error=str(error),actual_self_exit_claimed=False,full_model_qualified=False),error)
  raise
 finally:signal.alarm(0);m.old.post.external.release_lock(fd)

if __name__=='__main__':
 parser=argparse.ArgumentParser()
 for name in ('terminal','terminal-sha256','review','review-sha256'):parser.add_argument('--'+name,required=True)
 args=parser.parse_args();m.p.require(sys.dont_write_bytecode and __debug__,'No bytecode/assertions active')
 review_run(dict(path=args.terminal,sha256=args.terminal_sha256),dict(path=args.review,sha256=args.review_sha256))
