"""Read-only full raw/producer audit of completed r04/r05, no experiment writes."""
from pathlib import Path
import datetime
import hashlib
import json
import sys
ROOT=Path('/home/ubuntu/experiments/trn1-latest-models-20261031');Q=ROOT/'research'
sys.path.insert(0,str(ROOT))
import smollm3_observer_invariance_audit_v25 as audit


def reference(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def once(p,value):
    data=(json.dumps(value,indent=2,ensure_ascii=False)+'\n').encode()
    assert len(data)<8*2**20
    with p.open('xb')as f:f.write(data)
    return reference(p)

def run():
    spec_path=Q/'smollm3-observer-sio14-execution-spec-20261108-v1.json'
    assert reference(spec_path)['sha256']=='6ae6357c45f3d6bbd0c2fc38d5a77fe24d2c537073b67ca5589554da05c14337'
    spec=json.loads(spec_path.read_text());base=audit.p.load_base();results={};captures={}
    for name in ('r04','r05'):
        capture,proof,export=audit.audit_run(base,name,write_export=False)
        captures[name]=capture
        proof_path=Q/('smollm3-sio14-'+name+'-root-independent-mode-audit-20261108-v1.json')
        proof_ref=once(proof_path,proof)
        results[name]=dict(capture=reference(ROOT/'sio14'/name/'capture-result.json'),mode_proof=proof_ref,
                          original_client_exit0=True,original_owners_gone_and_groups_empty=True,
                          raw_wire_reparsed=True,observer_export_independently_recomputed=export is not None,
                          export=reference(ROOT/'sio14'/name/'replay-export.json')if export is not None else None,
                          observer_token_samples=sum(len(row['samples'])for row in export['requests'])if export else 0,
                          original_server_success_proven=False,original_server_physical_only=True)
    source_pins={}
    for module in tuple(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if path and str(Path(path).resolve())in spec['runtime_source_sha256']:
            path=str(Path(path).resolve());sha=hashlib.sha256(Path(path).read_bytes()).hexdigest()
            assert sha==spec['runtime_source_sha256'][path];source_pins[path]=sha
    pair=audit.analysis.compare_requests(captures['r04']['requests'],captures['r05']['requests'])
    ref_capture=audit.l.read(ROOT/'sio14/r00/capture-result.json')
    contrast=audit.analysis.compare_requests(ref_capture['requests'],captures['r05']['requests'])
    assert not any(name.split('.')[0]in audit.p.NATIVE_PREFIXES for name in sys.modules)
    report=dict(schema='smollm3_sio14_first_write_pair_root_audit_v1',reviewer='/root',observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                execution_spec=reference(spec_path),frozen_runtime_sources_rehashed=source_pins,runs=results,
                write_off_on_pair=pair,write_on_vs_reference_off=contrast,completed_rounds=6,completed_requests=132,completed_output_tokens=16848,
                full12round_matrix_complete=False,write_same_mode_repeats_complete=False,
                cross_configuration_difference_cause_proven=False,full_graph_state_equivalence_proven=False,
                full_model_qualified=False,new_speedup_claim=False,model_framework_imported=False,experiment_files_modified=False)
    out=Q/'smollm3-sio14-first-write-pair-root-audit-20261108-v1.json';ref=once(out,report)
    print(json.dumps(dict(report=ref,off_on_equal=pair['equal'],requests=pair['requests'],tokens=pair['tokens'],
                         reference_equal_requests=sum(row['equal']for row in contrast['rows']),observer_samples=results['r05']['observer_token_samples'],
                         frozen_sources_rehashed=len(source_pins),completed_rounds=6)))

if __name__=='__main__':run()
