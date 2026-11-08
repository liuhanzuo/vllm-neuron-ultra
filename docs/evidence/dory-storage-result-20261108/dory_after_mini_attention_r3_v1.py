"""Authenticate actual r3 capture/recovery history before Dory; never relabel failure."""
from pathlib import Path
import hashlib,json,re
import recover_minicpm5_attention_r3_saved_review_v1 as recovery
import observe_minicpm5_attention_terminal_v4 as terminal
m=recovery.m;R,Q=recovery.R,recovery.Q;require=m.p.require
RUN=dict(path=str(Q/'minicpm5-frame1-attention-r3-20261108-v1-recovered-review-run.json'),sha256='77917c97fceec9243c5a59da44df13aebb82e1204d20c88da4f5ff169f652f04')
EXIT=dict(path=str(Q/'minicpm5-attention-r3-recovered-review-external-exit-v1.json'),sha256='1d988633e36d69beee968853a8fc56dc33944925c9c4de16b8ff4e88ba85105d')
def reference(path):return dict(path=str(path),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
def bind(run,ended,report,t):
 require(run['schema']=='minicpm5_attention_failed_replay_recovered_review_run_v1'and run['complete']is True,'Actual complete recovered review')
 m.same(run['terminal'],recovery.TERMINAL,'Original failed replay terminal');m.same(run['verifier_exit'],recovery.EXIT,'Actual original failure verifier exit')
 require(run['original_failed_replay_preserved']is run['all_original_files_unchanged']is True and run['model_rerun']is False,'Failed replay retained without model rerun')
 m.same(run['original_runner_actual_exit'],dict(session_id=33412,exit_code=1,tool_chunk='25546f'),'Original runner actually failed')
 require(run['actual_self_exit_claimed']is run['full_model_qualified']is run['new_speedup_claim']is False,'No self-exit or full-model promotion')
 require(ended['schema']=='minicpm5_attention_recovered_review_actual_external_exit_v1'and ended['reviewer']=='/root','Actual root-observed recovered review exit')
 m.same((ended['run'],ended['terminal'],ended['owner']),(RUN,recovery.TERMINAL,run['owner']),'Actual recovered review result and owner')
 require(ended['actual_tool_exit_observed']is ended['exact_owner_gone']is ended['original_model_runner_failed']is True and ended['new_speedup_claim']is False,'Actual failure preserved and review owner gone')
 actual=ended['actual_exit'];require(type(actual['session_id'])is int and actual['session_id']>0 and type(actual['exit_code'])is int and actual['exit_code']==0 and re.fullmatch('[0-9a-f]{6}',actual['tool_chunk']),'Actual successful external recovered review exit')
 require(report['complete']is report['all96_full_payloads_authenticated']is report['all21_original_upstream_payloads_exact']is True,'All original saved payloads reviewed')
 m.same((report['files'],report['raw_bytes']),(96,353284922),'Whole original output extent');m.same(report['captures'],t['captures'],'Exact original three captures')
 require(type(report['passed'])is bool and report['passed']is report['all21_own_input_controls_exact']is run['all21_own_input_controls_exact'],'Actual scientific verdict including false retained')
 for k in ('full_model_qualified','new_speedup_claim','process_lifecycle_verified','unique_internal_or_final_KL_cause_proven'):require(report[k]is False,'Local review does not promote other claims')
 terminal.process(run['owner']);return run['owner']
def predecessor(terminal_ref,review_run_ref,review_exit_ref):
 m.same((terminal_ref,review_run_ref,review_exit_ref),(recovery.TERMINAL,RUN,EXIT),'Exact actual r3 recovery predecessor references')
 t=recovery.validate(m.p.read_ref(terminal_ref),m.p.read_ref(recovery.EXIT));run=m.p.read_ref(RUN);ended=m.p.read_ref(EXIT);report=m.p.read_ref(run['result']);review_owner=bind(run,ended,report,t)
 spec=m.p.read_ref(t['spec']);approval=m.p.read_ref(t['review']);base=m.p.read_ref(m.BASE_PLAN);a=m.p.read_ref(t['admission']);root=Path(spec['output_root'])
 m.same((spec['runner'],root),(m.SELF,recovery.ROOT),'Actual original r3 runner and output');m.same(approval['spec'],t['spec'],'Actual capture source approval')
 require(approval['passed']is approval['source_only']is approval['independent_review']is True and approval['blockers']==[],'Original approved capture sources')
 for group in (base['source_sha256'],spec['new_sources']):
  for p,h in group.items():m.same(m.p.sha(p),h,'Original capture source unchanged')
 m.same((a['spec'],a['review'],a['outer_owner']),(t['spec'],t['review'],t['outer_owner']),'Actual original admission')
 for phase in m.PHASES:
  ex=m.p.read(root/(phase+'-exit.json'));m.same(ex['owner'],t['phase_owners'][phase],'Actual original phase exit');require(ex['successful']is(phase!='replay'),'Original phase failure retained');m.same(ex['observed_by_outer'],t['outer_owner'],'Original coordinator observed exit')
 cpu=m.p.read(root/'cpu-result.json');m.same(cpu['actual_rank_exits'],[0,0],'Actual successful CPU ranks');m.same(cpu['ranks'],t['rank_owners'],'Actual rank identities')
 reports={role:m.p.read_ref(ref)for role,ref in t['captures'].items()}
 metadata=recovery.independent.meta.validate(root=root,reports=reports,plan=base);m.same(metadata,report['metadata'],'Independently rederived metadata matches complete byte review')
 post=m.p.read_ref(run['source_review']);m.same((post['terminal'],post['verifier_exit'],post['budgets']),(terminal_ref,recovery.EXIT,recovery.BUDGET),'Original independent review source/budget binding')
 require(post['passed']is post['source_only']is post['independent_review']is True and post['blockers']==[],'Separate reviewed saved-byte recovery')
 for p,h in post['source_sha256'].items():m.same(m.p.sha(p),h,'Original independent recovery source retained')
 m.same(post['source_sha256'].get(str(recovery.SELF)),m.p.sha(recovery.SELF),'Original recovery entrypoint pinned')
 reviewed=m.p.read_ref(run['admission']);m.same((reviewed['terminal'],reviewed['verifier_exit'],reviewed['source_review'],reviewed['owner'],reviewed['lock_identity']),(terminal_ref,recovery.EXIT,run['source_review'],review_owner,t['lock_identity']),'Actual recovery admission')
 m.load_history_helpers();catalog=m.current_history(spec);recovery.prior.validate(spec)
 owners=[t['outer_owner'],t['verifier'],*t['phase_owners'].values(),*t['rank_owners'],review_owner];require(len({(o['pid'],o['start_ticks'])for o in owners})==8,'All eight actual r3 capture/failure/review owners')
 return dict(terminal=t,spec=spec,catalog=catalog,owners=owners,admission=a,review_run=run)
def physical_drain(checked):
 m.load_history_helpers();m.history_metadata.physical_drain(checked['catalog'],checked['spec'])
 for owner in checked['owners']:m.p.gone(owner)
 t,spec=checked['terminal'],checked['spec'];samples=[]
 # Callers hold this same lock. Recheck both earlier failed Mini histories.
 fd_owner=m.p.proc(__import__('os').getpid());recovery.prior.physical(spec,fd_owner,t['lock_identity'])
 for phase in m.PHASES:
  value=m.p.properties(spec['units'][phase]);m.service_identity(spec,t['admission'],checked['admission']['attempt_id'],phase,value,t['phase_owners'][phase]);require(m.p.terminal(value),'Original r3 phase terminal');m.same((value['Result'],value['ExecMainCode'],value['ExecMainStatus']),('exit-code','1','1')if phase=='replay'else('success','1','0'),'Original failed replay not relabeled');samples.append(dict(phase=phase,properties=value,group=m.p.empty_group(m.p.GROUP_PREFIX+spec['units'][phase]+'.service')))
 root=Path(spec['output_root']);m.same({str(p)for p in root.rglob('*')if p.is_file()},{x['path']for x in t['partial_files']},'All original r3 files retained')
 for row in t['partial_files']:
  p=Path(row['path']);require(p.is_relative_to(root)and not p.is_symlink()and p.stat().st_size==row['bytes'],'Exact original retained file extent');m.same(hashlib.sha256(p.read_bytes()).hexdigest(),row['sha256'],'Original r3 full capture and failure bytes retained')
 return samples
