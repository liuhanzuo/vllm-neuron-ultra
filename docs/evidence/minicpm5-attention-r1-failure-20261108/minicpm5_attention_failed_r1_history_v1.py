"""Admit only the actual failed attention r1 predecessor, preserving all output."""
from pathlib import Path
import hashlib,json,stat
import minicpm5_frame1_attention_runner_v2 as prior
m=prior
R,Q=m.ROOT,m.ROOT/'research'
TERMINAL=dict(path=str(Q/'minicpm5-frame1-attention-r1-20261108-v1-failed-terminal.json'),sha256='2b53f1989ab7aae7f827f6feb82edd47860bdf7b55d7a42d758346954f36a261')
EXIT={'path': '/home/ubuntu/experiments/trn1-latest-models-20261031/research/minicpm5-frame1-attention-r1-20261108-v1-failed-verifier-external-exit.json', 'sha256': 'c35909f0935bc159781225fc6d2b9c1c9d6dcf976af3641becae95156a5f21fe'}

def validate(spec):
 m.same(spec['prior_attention_failure'],dict(terminal=TERMINAL,verifier_exit=EXIT),'Exact real failed attention predecessor')
 t=m.p.read_ref(TERMINAL);e=m.p.read_ref(EXIT);live=m.p.read_ref(t['live']);exited=m.p.read_ref(t['actual_external_exit'])
 m.same(exited['actual_exit'],dict(session_id=93091,exit_code=1,tool_chunk='b9fabe'),'Original actual failed execution')
 m.same(exited['outer_owner'],live['outer_owner'],'Original failed run owner');m.same(t['outer_owner'],live['outer_owner'],'Terminal same owner')
 m.p.require(exited['actual_tool_exit_observed']is exited['exact_owner_gone']is True,'Original external exit verified')
 m.p.require(t['all_three_planned_groups_drained']is t['cpu_and_replay_never_launched']is True and t['complete_capture']is False,'Original stopped failed attempt')
 m.same(e['terminal'],TERMINAL,'Actual verifier completion binds terminal');m.same(e['owner'],t['verifier'],'Exact actual verifier')
 m.p.require(e['actual_tool_exit_observed']is e['exact_owner_gone']is True and e['actual_exit']==dict(session_id=None,exit_code=0,tool_chunk='115b4d'),'Actual completed verifier tool result')
 m.p.require(t['full_model_qualified']is t['new_speedup_claim']is False,'No failed model promotion')
 return t

def physical(spec,owner,lock):
 t=validate(spec);m.same(lock,t['lock_identity'],'Original serial lock');m.p.lock_proof(owner,lock);m.p.active_heavy()
 oldspec=m.p.read_ref(t['spec']);admission=m.p.read_ref(t['admission']);exited=m.p.read_ref(t['native_exit'])
 m.same(exited['owner'],t['native_owner'],'Original retained native exit');m.p.require(exited['successful']is False,'Original native failure retained')
 for proc in [t['outer_owner'],t['native_owner'],t['verifier'],t['prior_inspector_attempt']['owner']]:m.p.gone(proc)
 for phase in m.PHASES:
  value=m.p.properties(oldspec['units'][phase]);m.p.require(m.p.terminal(value),'Every prior phase terminal');m.p.empty_group(m.p.GROUP_PREFIX+oldspec['units'][phase]+'.service')
  if phase=='native':
   m.service_identity(oldspec,t['admission'],admission['attempt_id'],phase,value,t['native_owner']);m.p.require((value['Result'],value['ExecMainCode'],value['ExecMainStatus'])==('exit-code','1','1'),'Exact failed original native state')
  else:m.p.require(value['LoadState']=='not-found'and not value['InvocationID'],'Other original phases never launched')
 root=Path(oldspec['output_root']);m.p.require(root.resolve()==root,'Original physical output root');actual={str(p)for p in root.rglob('*')if p.is_file()}
 m.p.require(actual=={row['path']for row in t['partial_files']},'Every original partial file retained')
 for row in t['partial_files']:
  path=Path(row['path']);before=path.lstat();m.p.require(path.is_relative_to(root)and not path.is_symlink()and stat.S_ISREG(before.st_mode)and before.st_uid==1000 and before.st_nlink==1 and before.st_size==row['bytes']<=16*2**20,'Owned exact bounded original file')
  raw=path.read_bytes();after=path.lstat();m.same((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns),(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'Stable original failure bytes');m.same(hashlib.sha256(raw).hexdigest(),row['sha256'],'Original complete partial content')
 return dict(terminal=TERMINAL,verifier_exit=EXIT,all3_planned_groups_drained=True,all4_known_owners_gone=True,all23_partial_files_unchanged=True,original_failure_retained=True)
