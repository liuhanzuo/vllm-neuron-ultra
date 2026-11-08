from pathlib import Path
import json,hashlib,sys
Q=Path(__file__).resolve().parent;R=Q.parent;sys.path[:0]=[str(Q),str(R)]
import run_dory_original_weight_load_v1 as old
r=old.ROOT;a=old.d.load(old.d.ref(r/'admission.json'));failure=old.d.load(old.d.ref(r/'failure.json'));old.d.m.same(a['owner'],failure['owner'],'Original actual failed owner');old.d.m.same(failure['error'],'Dory eager arithmetic may not spawn or replace processes','Actual original runtime guard failure');old.d.require(not(r/'loader-proof.json').exists()and not(r/'result.json').exists(),'No successful original weight load')
fd,owner,lock=old.d.m.old.post.external.acquire_lock()
try:
 old.d.m.same(lock,a['lock_identity'],'Same original serial lock')
 for _ in range(2):old.d.m.p.gone(a['owner']);old.d.m.p.lock_proof(owner,lock);old.d.m.p.clear_window();old.d.m.p.active_heavy()
 files=[dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())for p in sorted(r.iterdir())if p.is_file()]
 t=old.d.once(Q/'dory-original-weight-load-r1-20261108-v1-failed-terminal.json',dict(schema='dory_original_weight_load_failed_terminal_v1',reviewer='/root',outer_owner=a['owner'],verifier=owner,lock_identity=lock,admission=old.d.ref(r/'admission.json'),failure=old.d.ref(r/'failure.json'),actual_exit=dict(session_id=4859,exit_code=1,tool_chunk='b9bd88'),actual_tool_exit_observed=True,exact_owner_gone=True,live_identity_not_captured_before_exit=True,partial_files=files,original_weights_loaded=False,full_model_qualified=False,new_speedup_claim=False));print(json.dumps(dict(terminal=t,verifier=owner)),flush=True)
finally:old.d.m.old.post.external.release_lock(fd)
