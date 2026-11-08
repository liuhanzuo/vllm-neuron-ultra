"""Read-only original producer/wire audit of completed write/on repeat r06."""
from pathlib import Path
import datetime
import hashlib
import json
import sys
ROOT=Path('/home/ubuntu/experiments/trn1-latest-models-20261031');Q=ROOT/'research'
sys.path.insert(0,str(ROOT))
import smollm3_observer_invariance_audit_v25 as audit

def ref(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def once(p,value):
    data=(json.dumps(value,indent=2,ensure_ascii=False)+'\n').encode()
    assert len(data)<8*2**20
    with p.open('xb')as f:f.write(data)
    return ref(p)

def run():
    previous=Q/'smollm3-sio14-first-write-pair-root-audit-20261108-v1.json'
    assert ref(previous)['sha256']=='550c536c8ce5a7c233c7ed8504eea61b0d8a52ef0ea6211258d5831188a549ea'
    prior=json.loads(previous.read_text())
    spec_path=Q/'smollm3-observer-sio14-execution-spec-20261108-v1.json'
    assert ref(spec_path)['sha256']=='6ae6357c45f3d6bbd0c2fc38d5a77fe24d2c537073b67ca5589554da05c14337'
    spec=json.loads(spec_path.read_text());base=audit.p.load_base()
    capture,proof,export=audit.audit_run(base,'r06',write_export=False)
    proof_ref=once(Q/'smollm3-sio14-r06-root-independent-mode-audit-20261108-v1.json',proof)
    comparisons={}
    for name in ('r04','r05'):
        p=ROOT/'sio14'/name/'capture-result.json'
        assert ref(p)==prior['runs'][name]['capture']
        comparisons[name+'_versus_r06']=audit.analysis.compare_requests(audit.l.read(p)['requests'],capture['requests'])
    source_pins={}
    for module in tuple(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if path and str(Path(path).resolve())in spec['runtime_source_sha256']:
            path=str(Path(path).resolve());sha=ref(Path(path))['sha256']
            assert sha==spec['runtime_source_sha256'][path];source_pins[path]=sha
    assert export is not None
    samples=sum(len(row['samples'])for row in export['requests']);assert samples==2808
    assert not any(name.split('.')[0]in audit.p.NATIVE_PREFIXES for name in sys.modules)
    report=dict(schema='smollm3_sio14_write_on_repeat_root_audit_v1',reviewer='/root',observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        execution_spec=ref(spec_path),prior_pair_audit=ref(previous),frozen_runtime_sources_rehashed=source_pins,
        capture=ref(ROOT/'sio14/r06/capture-result.json'),mode_proof=proof_ref,export=ref(ROOT/'sio14/r06/replay-export.json'),
        raw_wire_reparsed=True,original_client_exit0=True,original_owners_gone_and_groups_empty=True,
        observer_export_independently_recomputed=True,observer_token_samples=samples,
        original_server_success_proven=False,original_server_physical_only=True,
        comparisons=comparisons,completed_rounds=7,completed_requests=154,completed_output_tokens=19656,
        full12round_matrix_complete=False,write_all_four_rounds_complete=False,
        cross_configuration_difference_cause_proven=False,full_graph_state_equivalence_proven=False,
        full_model_qualified=False,new_speedup_claim=False,model_framework_imported=False,experiment_files_modified=False)
    out=Q/'smollm3-sio14-write-on-repeat-root-audit-20261108-v1.json';out_ref=once(out,report)
    print(json.dumps(dict(report=out_ref,comparisons={k:dict(equal=v['equal'],requests=v['requests'],tokens=v['tokens'])for k,v in comparisons.items()},
        observer_samples=samples,frozen_sources_rehashed=len(source_pins),completed_rounds=7)))
if __name__=='__main__':run()
