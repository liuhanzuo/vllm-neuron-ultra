"""Preserve genuine r3 replay failure and all completed captures; no model launch."""
from pathlib import Path
import datetime,hashlib,json,stat,sys
R=Path(__file__).resolve().parents[1];Q=R/'research';sys.path[:0]=[str(R),str(Q)]
import minicpm5_frame1_attention_runner_v4 as m
import minicpm5_attention_failed_r2_history_v1 as prior
ROOT=R/'minicpm5-frame1-attention-r3-20261108-v1'
EXIT=dict(path=str(Q/(ROOT.name+'-actual-external-exit.json')),sha256='5d76baf2715c1aa1244365166784380bd804f91d8fc96d064bddccac039895bc')
def reference(path):return dict(path=str(path),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
def main():
 e=m.p.read_ref(EXIT);live=m.p.read_ref(e['live']);spec=m.p.read_ref(e['spec']);review=m.p.read_ref(e['review'])
 m.same(e['actual_exit'],dict(session_id=33412,exit_code=1,tool_chunk='25546f'),'Actual failed original r3 tool exit')
 for k in ('spec','review','outer_owner'):m.same(e[k],live[k],'Actual original r3 binding')
 m.p.require(e['actual_tool_exit_observed']is e['exact_owner_gone']is True,'Actual observed external failure')
 m.same(review['spec'],e['spec'],'Original r3 approval');m.p.require(review['passed']is review['source_only']is review['independent_review']is True and review['blockers']==[],'Original approved r3 source')
 for p,h in review['source_sha256'].items():m.same(m.p.sha(p),h,'Original r3 exact source')
 ar=reference(ROOT/'admission.json');a=m.p.read_ref(ar)
 for k in ('spec','review','outer_owner'):m.same(a[k],e[k],'Actual original admission')
 m.same(m.p.read(ROOT/'failure.json'),dict(phase='replay',error_type='RuntimeError',error='Actual phase failure: replay',owner=e['outer_owner']),'Original actual failure')
 m.p.require(not(ROOT/'result.json').exists()and not(ROOT/'replay-result.json').exists()and not(ROOT/'attention-review.json').exists(),'No fabricated successful replay')
 phases={p:m.p.read(ROOT/(p+'-owner.json'))for p in m.PHASES};ranks=[m.p.read(ROOT/('rank'+str(i)+'-owner.json'))for i in (0,1)]
 cpu=m.p.read(ROOT/'cpu-result.json');m.same(cpu['actual_rank_exits'],[0,0],'Two actual successful CPU ranks');m.same(cpu['ranks'],ranks,'Two original actual CPU rank owners')
 for p in m.PHASES:
  ex=m.p.read(ROOT/(p+'-exit.json'));m.same(ex['owner'],phases[p],'Actual original phase owner');m.same(ex['observed_by_outer'],e['outer_owner'],'Actual original coordinator');m.p.require(ex['successful']is(p!='replay'),'Actual phase outcome retained')
 for rank,owner in enumerate(ranks):
  m.same(owner['argv'],m.arguments('rank',ar,'cpu'),'Original rank command');m.same((owner['ppid'],owner['cgroup']),(phases['cpu']['pid'],phases['cpu']['cgroup']),'Original rank parent/group')
  rr=m.p.read(ROOT/('cpu-rank'+str(rank)+'-result.json'));m.p.require(rr['complete']is True,'Complete actual CPU capture');m.same(rr['process'],owner,'Original rank result owner')
 captures={}
 for role in ('native','cpu-rank0','cpu-rank1'):
  cap=reference(ROOT/role/'capture.json');rec=m.p.read_ref(cap);m.p.require(rec['complete']is True,'Complete actual capture');m.same(m.p.read(ROOT/(role+'-result.json'))['capture'],cap,'Original exact capture result');captures[role]=cap
 m.p.user_bus();fd,owner,lock=m.old.post.external.acquire_lock()
 try:
  m.same(lock,a['lock_identity'],'Original serial lock');previous=prior.physical(spec,owner,lock);samples=[]
  for _ in range(2):
   m.p.lock_proof(owner,lock);m.p.clear_window();m.p.active_heavy()
   for proc in [e['outer_owner'],*phases.values(),*ranks]:m.p.gone(proc)
   rows=[]
   for p in m.PHASES:
    value=m.p.properties(spec['units'][p]);m.service_identity(spec,ar,a['attempt_id'],p,value,phases[p]);m.p.require(m.p.terminal(value),'Original phase terminal');m.same((value['Result'],value['ExecMainCode'],value['ExecMainStatus']),('exit-code','1','1')if p=='replay'else('success','1','0'),'Original actual phase outcome');rows.append(dict(phase=p,properties=value,group=m.p.empty_group(m.p.GROUP_PREFIX+spec['units'][p]+'.service')))
   samples.append(rows)
  files=[]
  for p in sorted(ROOT.rglob('*')):
   m.p.require(not p.is_symlink(),'No unexpected failed output symlink')
   if not p.is_file():continue
   st=p.lstat();m.p.require(stat.S_ISREG(st.st_mode)and st.st_nlink==1 and st.st_size<=16*2**20,'Bounded retained file');raw=p.read_bytes();m.same((st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns),tuple(getattr(p.lstat(),k)for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')),'Stable retained file');files.append(dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
  raw=[x for x in files if x['path'].endswith(('.raw','.bf16'))];m.same((len(raw),sum(x['bytes']for x in raw)),(96,353284922),'All complete original raw captures present')
  m.same({n:m.p.sha(R/n)for n in m.CANONICAL},m.CANONICAL,'Canonical files unchanged')
  terminal=dict(schema='minicpm5_attention_r3_failed_replay_terminal_v1',reviewer='/root',observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),spec=e['spec'],review=e['review'],live=e['live'],actual_external_exit=EXIT,outer_owner=e['outer_owner'],verifier=owner,lock_identity=lock,admission=ar,phase_owners=phases,rank_owners=ranks,captures=captures,prior_failures=previous,physical_drain_samples=samples,all_three_planned_groups_drained=True,native_and_cpu_complete=True,replay_failed=True,partial_files=files,raw_files=len(raw),raw_bytes=sum(x['bytes']for x in raw),independent_saved_byte_review_complete=False,full_model_qualified=False,new_speedup_claim=False)
  target=Q/(ROOT.name+'-failed-terminal.json');out=m.worker.once(target,terminal);print(json.dumps(dict(terminal=out,verifier=owner,files=len(files),bytes=sum(x['bytes']for x in files),actual_self_exit_claimed=False)),flush=True)
 finally:m.old.post.external.release_lock(fd)
if __name__=='__main__':main()
