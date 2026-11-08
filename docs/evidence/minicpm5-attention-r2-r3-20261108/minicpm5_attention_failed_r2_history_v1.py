"""Exact r2 failure plus preserved r1 before a new attention run; no launch."""
from pathlib import Path
import hashlib,stat
import minicpm5_attention_failed_r1_history_v1 as earlier
import minicpm5_frame1_attention_runner_v3 as m
R,Q=m.ROOT,m.ROOT/'research'
TERMINAL={'path': '/home/ubuntu/experiments/trn1-latest-models-20261031/research/minicpm5-frame1-attention-r2-20261108-v1-failed-terminal.json', 'sha256': '8173f65643399fac862cd62299e7378012980acdd3929933e18a627dfb1b983e'}
EXIT={'path': '/home/ubuntu/experiments/trn1-latest-models-20261031/research/minicpm5-frame1-attention-r2-20261108-v1-failed-verifier-external-exit.json', 'sha256': '2802ec658e35f8891f0c0f59369dbc3e46b440e0fd5ecb04c8d8e2407e1c7ce4'}

def validate(spec):
 earlier.validate(dict(prior_attention_failure=dict(terminal=earlier.TERMINAL,verifier_exit=earlier.EXIT)))
 m.same(spec['prior_attention_failure'],dict(terminal=TERMINAL,verifier_exit=EXIT),'Exact real r2 predecessor')
 t=m.p.read_ref(TERMINAL);e=m.p.read_ref(EXIT);actual=m.p.read_ref(t['actual_external_exit']);live=m.p.read_ref(t['live'])
 m.same(actual['actual_exit'],dict(session_id=70802,exit_code=1,tool_chunk='b90116'),'Actual r2 failed CPU execution');m.same(actual['outer_owner'],t['outer_owner'],'Original r2 owner');m.same(live['outer_owner'],t['outer_owner'],'Original live owner')
 m.p.require(actual['actual_tool_exit_observed']is actual['exact_owner_gone']is True and t['all_three_planned_groups_drained']is t['replay_never_launched']is True,'Actual failed r2 drained')
 m.same(e['terminal'],TERMINAL,'Actual verifier terminal binding');m.same(e['owner'],t['verifier'],'Original verifier identity');m.same(e['actual_exit'],dict(session_id=None,exit_code=0,tool_chunk='e70f9b'),'Actual verifier tool return')
 m.p.require(e['actual_tool_exit_observed']is e['exact_owner_gone']is True and t['complete_capture']is t['full_model_qualified']is t['new_speedup_claim']is False,'No failed completion promotion')
 return t

def physical(spec,owner,lock):
 prior=earlier.physical(dict(prior_attention_failure=dict(terminal=earlier.TERMINAL,verifier_exit=earlier.EXIT)),owner,lock)
 t=validate(spec);m.same(lock,t['lock_identity'],'Exact original lock');m.p.lock_proof(owner,lock);m.p.active_heavy();s=m.p.read_ref(t['spec']);admission=m.p.read_ref(t['admission'])
 for proc in [t['outer_owner'],t['verifier'],*t['phase_owners'].values(),*t['rank_owners']]:m.p.gone(proc)
 for phase in m.PHASES:
  value=m.p.properties(s['units'][phase]);m.p.require(m.p.terminal(value),'Original r2 terminal service');m.p.empty_group(m.p.GROUP_PREFIX+s['units'][phase]+'.service')
  if phase in t['phase_owners']:
   m.service_identity(s,t['admission'],admission['attempt_id'],phase,value,t['phase_owners'][phase]);m.same((value['Result'],value['ExecMainCode'],value['ExecMainStatus']),('success','1','0')if phase=='native'else('exit-code','1','1'),'Original actual phase outcome')
  else:m.p.require(value['LoadState']=='not-found'and not value['InvocationID'],'Original replay never launched')
 root=Path(s['output_root']);m.p.require(root.resolve()==root,'Physical original root');m.p.require({str(p)for p in root.rglob('*')if p.is_file()}=={row['path']for row in t['partial_files']},'All54 original r2 files retained')
 for row in t['partial_files']:
  p=Path(row['path']);st=p.lstat();m.p.require(p.is_relative_to(root)and not p.is_symlink()and stat.S_ISREG(st.st_mode)and st.st_uid==1000 and st.st_nlink==1 and st.st_size==row['bytes']<=16*2**20,'Exact bounded original partial file');raw=p.read_bytes();after=p.lstat();m.same((st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns),(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'Stable actual r2 bytes');m.same(hashlib.sha256(raw).hexdigest(),row['sha256'],'Complete retained r2 content')
 return dict(terminal=TERMINAL,verifier_exit=EXIT,prior_r1=prior,all3_r2_groups_drained=True,all6_r2_known_owners_gone=True,all54_partial_files_unchanged=True,original_failed_r2_retained=True)
