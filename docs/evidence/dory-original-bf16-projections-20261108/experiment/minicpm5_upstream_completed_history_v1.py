"""Complete actual Mini upstream history; preserve Dory/r6 and every old outcome."""
import argparse,copy,datetime,hashlib,json,re,sys
from pathlib import Path
import minicpm5_frame1_upstream_runner_v1 as m
import minicpm5_post_dory_history_v1 as previous
R,Q=previous.R,previous.Q
PRIOR=previous.CATALOG
TERMINAL=dict(path=str(Q/'minicpm5-frame1-upstream-r1-20261108-v1-root-terminal.json'),sha256='60ca5ca1f89c05c1b8fac76854cd671dd725275d0d3eeacdf77f5cac7eb80bdb')
ROOT=R/'minicpm5-frame1-upstream-r1-20261108-v1'
RUN=Q/(ROOT.name+'-independent-review-run.json')
CURRENT=Q/(ROOT.name+'-complete-catalog-v1.json')
AUDIT=Q/(ROOT.name+'-final-root-audit-v1.json')
SCHEMA='root_post_minicpm5_upstream_complete_history_v1'
CHANGED=previous.dory.CHANGED|{'rows','complete_count'}
require=m.p.require
same=m.same

def ref(path):return previous.dory.ref(path)
def small(reference):return previous.dory.load_small(reference)
def once(path,value):return previous.dory.write_once(path,value)

def exit_check(exited,run_ref,run):
 require(exited['schema']=='minicpm5_upstream_review_actual_external_exit_v1'and exited['reviewer']=='/root','Actual review external receipt')
 require(exited['actual_tool_exit_observed']is exited['exact_owner_gone']is True,'Review actual external exit')
 same(exited['run'],run_ref,'Exact completed review report');same(exited['owner'],run['owner'],'Actual review process')
 actual=exited['actual_exit'];require(type(actual['session_id'])is int and actual['session_id']>0 and type(actual['exit_code'])is int and actual['exit_code']==0 and re.fullmatch('[0-9a-f]{6}',actual['tool_chunk']),'Observed successful review session/code/chunk')
 require(run['schema']=='minicpm5_upstream_independent_review_run_v1'and run['complete']is True and run['actual_self_exit_claimed']is False,'Complete review does not self-certify exit')
 require(run['model_exit_verified']is run['physical_drain_verified']is True and run['full_model_qualified']is run['new_speedup_claim']is False,'Review scope remains local')
 same(run['terminal'],TERMINAL,'Actual upstream terminal');same(run['actual_model_exit'],dict(session_id=10266,exit_code=0,tool_chunk='d0cecb'),'Actual model tool exit0')
 return run

def added_rows(t):
 rows=[]
 for sample in t['physical_drain_samples'][0]:
  phase=sample['phase'];owner=t['phase_owners'][phase];unit=owner['unit'];argv=re.findall(r'argv\[\]=(.*?)\s*;',sample['properties']['ExecStart'])
  require(len(argv)==1,'One actual service argv')
  rows.append(dict(unit=unit,owner=dict(pid=owner['pid'],start_ticks=owner['start_ticks'],unit=unit,script=str(R/'minicpm5_frame1_upstream_runner_v1.py')),captured_process_owner=owner,installed_argv=argv[0],recorded_cgroups=['/sys/fs/cgroup'+owner['cgroup'].split('::',1)[1]],historical_invocation_unknown=False,invocation_id=owner['invocation_id'],never_launched=False,absent_records=[]))
 require(len(rows)==3,'Exactly three actual phases');return rows

def new_processes(t,run,owner):
 known=[t['outer_owner'],t['terminal_observer'],*t['rank_owners'],run['owner'],owner]
 roles=['outer','terminal_observer','cpu_rank0','cpu_rank1','independent_review','history_finalizer']
 return [dict(p,role='minicpm5_upstream_'+role,**(dict(next_admission_must_prove_gone=True)if role=='history_finalizer'else{}))for p,role in zip(known,roles)]

def compose(old,rows,processes,pins,ext):
 value=copy.deepcopy(old);value['rows']+=rows;value['processes']+=processes
 if 'process_predecessors'in value:value['process_predecessors']=copy.deepcopy(value['processes'])
 for path,sha in pins.items():require(path not in value['source_sha256']or value['source_sha256'][path]==sha,'No historical source rebound');value['source_sha256'][path]=sha
 value.update(schema=SCHEMA,complete_count=981,appended_from=PRIOR,appended_terminal_audit=ext['audit'],latest_producer='minicpm5_upstream_arithmetic',latest_producer_qualified=False,latest_producer_exit=ext['review_exit'],latest_producer_terminal_audit=ext['audit'],minicpm5_upstream_append=ext)
 extension(old,value);return value

def extension(old,value):
 require(set(value)==set(old)|{'minicpm5_upstream_append'},'Only exact actual upstream extension')
 for key,wanted in old.items():
  if key not in CHANGED:same(value[key],wanted,'Original model outcomes retained: '+key)
 require(len(old['rows'])==978 and len(old['processes'])==190,'Actual post-Dory predecessor')
 same(value['rows'][:978],old['rows'],'All original services unchanged');same(value['processes'][:190],old['processes'],'All original known processes unchanged')
 require(len(value['rows'])==value['complete_count']==981 and len(value['processes'])==196,'All actual new services/processes')
 require(len({x['unit']for x in value['rows']})==981 and len({(x['pid'],x['start_ticks'])for x in value['processes']})==196,'No reused identities')
 require(all(value['source_sha256'].get(k)==v for k,v in old['source_sha256'].items()),'No old source omitted')
 if 'process_predecessors'in old:same(value['process_predecessors'],value['processes'],'Complete process alias')
 ext=value['minicpm5_upstream_append'];same(ext['prior'],PRIOR,'Exact prior');same(value['appended_from'],PRIOR,'Exact append');same(value['appended_terminal_audit'],ext['audit'],'Exact audit')
 same(value['latest_producer_exit'],ext['review_exit'],'Exact final review exit');same(value['latest_producer_terminal_audit'],ext['audit'],'Exact latest audit')
 require(value['schema']==SCHEMA and value['latest_producer']=='minicpm5_upstream_arithmetic'and value['latest_producer_qualified']is False,'Diagnostic never model qualification')
 require(value['processes'][-1]['next_admission_must_prove_gone']is True,'Finalizer external exit still required')
 for key in ('prior','terminal','review_run','review_exit','audit'):
  reference=ext[key];same(value['source_sha256'].get(reference['path']),reference['sha256'],'All completion evidence pinned')
 return value

def validate(value):
 old=previous.load(PRIOR);extension(old,value);ext=value['minicpm5_upstream_append'];same(ext['terminal'],TERMINAL,'Original actual Mini terminal')
 t=small(TERMINAL);run=small(ext['review_run']);exit_check(small(ext['review_exit']),ext['review_run'],run);audit=small(ext['audit']);result=small(run['result'])
 require(t['all_three_phases_succeeded']is t['all_three_groups_physically_drained']is t['actual_tool_exit_observed']is True,'Actual native/CPU/replay completed')
 same(t['actual_exit'],run['actual_model_exit'],'Actual model exit retained');same(value['rows'][978:],added_rows(t),'Three rows derive from actual lifecycle')
 same(value['processes'][190:],new_processes(t,run,audit['finalizer_owner']),'All actual new owners')
 require(result['complete']is result['all21_original_payloads_authenticated']is result['all6_norm_weight_files_authenticated']is result['all_observed_metrics_independently_recomputed']is result['residual_addition_independently_reexecuted']is True,'Complete independent saved-byte evidence')
 same(result['replay_controls_passed'],run['replay_controls_passed'],'Actual local replay outcome');same(audit['replay_controls_passed'],run['replay_controls_passed'],'Audit retains local outcome')
 require(audit['finalizer_self_exit_claimed']is False and audit['all_groups_physically_drained']is True and audit['full_model_qualified']is audit['new_speedup_claim']is False,'No self-exit/model promotion')
 same(audit['terminal'],TERMINAL,'Audit actual terminal');same(audit['review_exit'],ext['review_exit'],'Actual review external receipt');same(audit['review_run'],ext['review_run'],'Actual review run')
 for reference in [run['result'],run['admission'],run['source_review'],t['spec'],t['review'],t['live'],*t['references'].values()]:
  same(ref(reference['path']),reference,'Actual immutable saved evidence');same(value['source_sha256'].get(reference['path']),reference['sha256'],'Every actual evidence pin retained')
 return value

def text(path,expected=None):
 if Path(path)!=CURRENT:return previous.text(path,expected)
 raw=previous.dory.p.metadata.stable_bytes(CURRENT,32*2**20);same(hashlib.sha256(raw).hexdigest(),expected,'Exact current completed history hash');validate(json.loads(raw));return raw

def load(reference):return json.loads(text(reference['path'],reference['sha256']))

def drain_mini(t,run=None):
 spec=small(t['spec']);a_ref=t['references']['admission.json'];a=small(a_ref)
 for p in [t['outer_owner'],t['terminal_observer'],*t['rank_owners'],*t['phase_owners'].values(),*([run['owner']]if run else[])]:m.p.gone(p)
 for phase,unit in spec['units'].items():
  props=m.p.properties(unit);m.service_identity(spec,a_ref,a['attempt_id'],phase,props,t['phase_owners'][phase])
  require(m.p.terminal(props)and props['Result']=='success'and props['ExecMainStatus']=='0','Original phase remains successful/terminal');m.p.empty_group(m.p.GROUP_PREFIX+unit+'.service')
 return True

def physical_drain(value):
 validate(value);previous.physical_drain(previous.load(PRIOR));ext=value['minicpm5_upstream_append'];drain_mini(small(TERMINAL),small(ext['review_run']));m.p.gone(small(ext['audit'])['finalizer_owner']);return dict(services=981,processes=196)

def finalize(exit_ref,approval_ref):
 old=previous.load(PRIOR);t=small(TERMINAL);run_ref=ref(RUN);run=small(run_ref);exit_check(small(exit_ref),run_ref,run)
 approval=small(approval_ref);require(approval['schema']=='minicpm5_upstream_finalizer_source_review_v1'and approval['reviewer']=='/root'and approval['passed']is approval['independent_review']is approval['source_only']is True and approval['blockers']==[],'Separate finalizer source review')
 same(approval['review_exit'],exit_ref,'Exact actual review exit');same(approval['terminal'],TERMINAL,'Exact actual model terminal')
 for path,sha in approval['source_sha256'].items():same(ref(path)['sha256'],sha,'Exact reviewed finalizer source')
 require(approval['source_sha256'].get(str(Path(__file__).resolve()))==ref(__file__)['sha256'],'Exact finalizer implementation');require(not CURRENT.exists()and not AUDIT.exists(),'Never overwrite completion history')
 fd,owner,lock=m.old.post.external.acquire_lock()
 try:
  m.p.active_heavy();same(lock,t['lock_identity'],'Same serial lock');previous.physical_drain(old);drain_mini(t,run)
  same({n:m.p.sha(R/n)for n in m.CANONICAL},m.CANONICAL,'Canonical state retained')
  import os
  fs=os.statvfs(R);require(fs.f_bavail*fs.f_frsize>=small(t['spec'])['root_floor_bytes']+32*2**20,'Original root floor plus bounded completion output')
  audit=dict(schema='minicpm5_upstream_final_root_audit_v1',reviewer='/root',observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),terminal=TERMINAL,review_run=run_ref,review_exit=exit_ref,finalizer_source_review=approval_ref,finalizer_owner=owner,lock_identity=lock,all_groups_physically_drained=True,replay_controls_passed=run['replay_controls_passed'],finalizer_self_exit_claimed=False,full_model_qualified=False,new_speedup_claim=False)
  aref=once(AUDIT,audit);ext=dict(prior=PRIOR,terminal=TERMINAL,review_run=run_ref,review_exit=exit_ref,audit=aref)
  pins=dict(small(t['spec'])['new_sources']);pins.update(approval['source_sha256'])
  refs=[PRIOR,TERMINAL,run_ref,exit_ref,aref,approval_ref,run['result'],run['admission'],run['source_review'],t['spec'],t['review'],t['live'],*t['references'].values()]
  for reference in refs:require(reference['path']not in pins or pins[reference['path']]==reference['sha256'],'No conflicting completion pin');pins[reference['path']]=reference['sha256']
  value=compose(old,added_rows(t),new_processes(t,run,owner),pins,ext);validate(value);cref=once(CURRENT,value)
  print(json.dumps(dict(catalog=cref,audit=aref,owner=owner,services=981,processes=196,actual_self_exit_claimed=False)),flush=True)
 finally:m.old.post.external.release_lock(fd)

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for name in ('review-exit','review-exit-sha256','source-review','source-review-sha256'):p.add_argument('--'+name,required=True)
 a=p.parse_args();require(sys.dont_write_bytecode and __debug__,'Bytecode disabled/assertions active')
 finalize(dict(path=a.review_exit,sha256=a.review_exit_sha256),dict(path=a.source_review,sha256=a.source_review_sha256))
