"""Actual pinned science/source reads; future runtime-history call is mocked, never admitted."""
import datetime
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
ROOT=Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
sys.path.insert(0,str(ROOT))
import minicpm5_frame1_mlp_runner_v1 as r


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run():
    base=r.p.read_ref(r.BASE_PLAN);extra={}
    paths={str(Path(m.__file__).resolve()) for m in tuple(sys.modules.values()) if getattr(m,'__file__',None) and Path(m.__file__).is_relative_to(ROOT)}
    paths|={r.SELF,str(ROOT/'minicpm5_frame1_mlp_replay_v2.py'),*r.worker.ACTIVATIONS.values()}
    for path in sorted(paths):
        if path not in base['source_sha256']:extra[path]=sha(path)
    added={ref['path']:dict(identity=r.identity(ref['path']),sha256=ref['sha256'])for ref in (*r.worker.CPU_ANCHORS,*r.worker.FULL_CPU_ANCHORS)}
    # These two literals are test identities in memory, never runtime receipts.
    sref=dict(path='/test/memory-only-spec.json',sha256='1'*64);rref=dict(path='/test/memory-only-review.json',sha256='2'*64)
    spec=dict(schema='minicpm5_frame1_mlp_spec_v1',base_plan=r.BASE_PLAN,base_spec=r.BASE_SPEC,runner=r.SELF,
              worker=str(ROOT/'minicpm5_frame1_mlp_worker_v2.py'),replay=str(ROOT/'minicpm5_frame1_mlp_replay_v2.py'),
              output_root=str(ROOT/'minicpm5-frame1-mlp-r999-20261108-v1'),output_cap_bytes=r.CAP,root_floor_bytes=28*2**30,
              units={p:'trn1-mini-f1-mlp-loader-fixture-'+p for p in r.PHASES},master_port=29571,
              resources={p:r.props(p)for p in r.PHASES},canonical_sha256=r.CANONICAL,model_qualification_claim=False,
              automatic_full_gates_launch=False,new_sources=extra,added_json=added,science=r.worker.science())
    review=dict(reviewer='/root',passed=True,source_only=True,independent_review=True,blockers=[],spec=sref)
    original_read=r.p.read_ref
    def read(ref):
        if ref==sref:return spec
        if ref==rref:return review
        return original_read(ref)
    with patch.object(r.p,'read_ref',side_effect=read),patch.object(r,'current_history',return_value={})as future:
        actual,expanded=r.load_spec(sref,rref)
        assert actual==spec;future.assert_called_once_with(spec)
    assert not {'torch','torch_neuronx','transformers','neuronx_distributed'}&set(sys.modules)
    assert not Path(spec['output_root']).exists()
    report=dict(schema='minicpm5_frame1_actual_source_loader_probe_v2',observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                source_only=True,source_loader_passed=True,original_plan=r.BASE_PLAN,original_source_count=len(base['source_sha256']),
                added_source_count=len(extra),expanded_source_count=len(expanded['source_sha256']),source_additions=extra,
                added_actual_input_metadata=added,original_input_count=len(base['inputs']),expanded_input_count=len(expanded['inputs']),
                actual_base_and_added_sources_rehashed=True,all_four_actual_strict_baselines_validated=True,
                synthetic_spec_and_review_only_in_memory=True,actual_current_history_mocked_not_proven=True,
                actual_model_execution=False,model_framework_imported=False,runtime_spec_created=False,runtime_review_created=False,
                runtime_admitted=False,full_model_qualified=False,new_speedup_claim=False)
    out=ROOT/'research/minicpm5-frame1-actual-source-loader-probe-20261108-v2.json'
    with out.open('x')as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(dict(path=str(out),sha256=sha(out),original_sources=len(base['source_sha256']),added_sources=len(extra),
                         expanded_sources=len(expanded['source_sha256']),actual_input_anchors=4,runtime_admitted=False)))


if __name__=='__main__':run()
