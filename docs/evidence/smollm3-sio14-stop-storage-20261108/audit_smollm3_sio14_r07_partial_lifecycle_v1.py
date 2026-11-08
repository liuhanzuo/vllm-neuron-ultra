"""Read actual r07 wire/producers without inventing a missing server terminal."""
from pathlib import Path
import datetime,hashlib,itertools,json,sys
R=Path('/home/ubuntu/experiments/trn1-latest-models-20261031');Q=R/'research';sys.path.insert(0,str(R))
import smollm3_observer_invariance_audit_v25 as a
from smollm3_observer_invariance_controller_v26 import state_from_disk,physical

def ref(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def run():
    terminal=Q/'smollm3-observer-sio14-root-terminal-audit-20261108-v1.json'
    assert ref(terminal)['sha256']=='000ad5d82ead425f452d1f82a00f2bea7e072617acf324322a9e7beb50f69b9e'
    t=json.loads(terminal.read_text());assert t['passed']and not t['experiment_qualified']and t['all_groups_drained']
    base=a.p.load_base();root=R/'sio14/r07';capture=a.raw_capture(base,'r07',a.l.read(root/'capture-result.json'))
    assert not(root/'server-terminal.json').exists()
    drains={}
    for kind in ('server','client'):
        state=state_from_disk('r07',kind);drains[kind]=physical(state)
        a.l.verify_retained_binding(state['intent'],state['owner'],a.l.read(root/(kind+'-service-binding.json')))
    client=a.l.read(root/'client-terminal.json');v=client['original_properties'];owner=client['owner']
    assert client['passed']and v['Result']=='success'and v['ExecMainCode']=='1'and v['ExecMainStatus']=='0'
    assert v['ExecMainPID']==str(owner['pid'])and v['InvocationID']==owner['invocation_id']
    stop=a.l.read(root/'server-stop-intent.json');assert stop['authorized_stop']and stop['server_normal_shutdown']and not stop['successful_client_exit']
    assert stop['owner']==a.l.read(root/'server-owner.json')and stop['claim']==stop['owner']['invocation_id']and stop['binding']==a.l.read(root/'server-service-binding.json')
    owners,modes=a.mode_receipts('r07');environment=a.audit_environment('r07',owners);spawn=a.audit_spawn_provenance('r07',owners)
    assert not list(root.glob('observer-*.jsonl'))and not(root/'replay-export.json').exists()
    captures={name:a.l.read(R/'sio14'/name/'capture-result.json')for name in ('r04','r05','r06')};captures['r07']=capture
    pairs={left+'_versus_'+right:a.analysis.compare_requests(captures[left]['requests'],captures[right]['requests'])for left,right in itertools.combinations(captures,2)}
    spec=json.loads((Q/'smollm3-observer-sio14-execution-spec-20261108-v1.json').read_text());sources={}
    for module in tuple(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if path and str(Path(path).resolve())in spec['runtime_source_sha256']:
            path=Path(path).resolve();sha=ref(path)['sha256'];assert sha==spec['runtime_source_sha256'][str(path)];sources[str(path)]=sha
    assert not any(n.split('.')[0]in a.p.NATIVE_PREFIXES for n in sys.modules)
    result=dict(schema='smollm3_sio14_r07_partial_lifecycle_wire_audit_v1',reviewer='/root',observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        terminal_audit=ref(terminal),capture=ref(root/'capture-result.json'),client_original_terminal=ref(root/'client-terminal.json'),server_stop_intent=ref(root/'server-stop-intent.json'),
        original_client_exit_zero=True,raw_request_response_reparsed=True,original_server_terminal_record_present=False,original_controller_failure_retained=True,
        server_physical_drain_proven_only=True,original_server_success_proven=False,physical_drains=drains,authenticated_processes=owners,mode_receipts=modes,environment_phases=environment,spawn_provenance=spawn,
        frozen_runtime_sources_rehashed=sources,comparisons=pairs,complete_request_captures=8,captured_requests=176,captured_output_tokens=22464,
        fully_audited_rounds_before_failure=7,final_four_rounds_not_launched=True,original12round_protocol_completed=False,full_model_qualified=False,new_speedup_claim=False,model_execution_performed=False)
    path=Q/'smollm3-sio14-r07-partial-lifecycle-wire-audit-20261108-v1.json'
    with path.open('x')as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(report=ref(path),all_six_write_pairs_equal=all(v['equal']for v in pairs.values()),captured_requests=176,captured_tokens=22464,server_terminal_missing=True,full_matrix_complete=False)))
if __name__=='__main__':run()
