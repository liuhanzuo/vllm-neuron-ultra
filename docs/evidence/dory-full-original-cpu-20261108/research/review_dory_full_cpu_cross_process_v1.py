"""Check that the causal intervention did not change official reference bytes.

This verifies the observed short trajectories in two fresh processes. It is not
the project's full cold-cache, long-decode or production-slot acceptance gate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

Q=Path(__file__).resolve().parent
sys.path[:0]=[str(Q),str(Q.parent)]
import run_dory_weight_storage_root_v2 as d
BASE=Path('/mnt/nvme/dory-official-cpu-20261108-v1')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source-sha256',required=True)
    args=ap.parse_args()
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==args.source_sha256
    fd,owner,lock=d.m.old.post.external.acquire_lock()
    started=time.monotonic()
    try:
        d.m.p.clear_window()
        d.m.p.active_heavy()
        os.sched_setaffinity(0,{4})
        runs=['full-original-r1','full-original-projection-control-r1']
        reviews=['full-original-review-r1','full-original-projection-control-review-r1']
        results=[]
        inputs=[]
        for run,review_run in zip(runs,reviews):
            for name in [run,review_run]:
                admission=json.loads((BASE/name/'admission.json').read_text())
                d.m.p.gone(admission['owner'])
                assert json.loads((BASE/name/'result.json').read_text())['complete']
            for path in [BASE/run/'diagnostic/result.json',BASE/review_run/'saved-review.json']:
                inputs.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            review=json.loads((BASE/review_run/'saved-review.json').read_text())
            assert review['passed'] and review['saved_archives']==54
            results.append(json.loads((BASE/run/'diagnostic/result.json').read_text()))
        a,b=results
        assert a['tokens']==b['tokens'] and a['schedule']==b['schedule']
        pairs=[(l['official'],r['official'])for l,r in zip(a['steps'],b['steps'])]
        pairs.append((a['reuse']['official'],b['reuse']['official']))
        assert len(pairs)==27
        checked=[]
        for left,right in pairs:
            d.m.p.lock_proof(owner,lock)
            assert time.monotonic()-started<120
            assert left==right
            h=hashlib.sha256()
            total=0
            with (BASE/runs[0]/'diagnostic'/left['file']).open('rb')as lf, (BASE/runs[1]/'diagnostic'/right['file']).open('rb')as rf:
                while True:
                    lb,rb=lf.read(4*2**20),rf.read(4*2**20)
                    assert lb==rb
                    if not lb:
                        break
                    total+=len(lb)
                    h.update(lb)
            assert total==left['bytes'] and h.hexdigest()==left['sha256']
            checked.append(dict(file=left['file'],bytes_each=total,sha256=h.hexdigest(),exact_bytes=True))
        assert not {'torch','numpy','transformers'}&set(sys.modules)
        result=dict(passed=True,complete=True,owner=owner,source_sha256=args.source_sha256,inputs=inputs,
            official_archive_pairs=27,official_tensor_pairs=27*114,
            bytes_read=sum(r['bytes_each']*2 for r in checked),all_official_reference_bytes_exact=True,
            original_process_short_trajectory_reproduced=True,
            common_backend_control_did_not_change_official_saved_outputs=True,
            model_or_GEMM_reexecuted=False,full_cold_cache_gate_passed=False,
            full_model_qualified=False,new_speedup_claim=False,files=checked,
            elapsed_seconds=time.monotonic()-started)
        path=BASE/'full-original-cross-process-review-v1.json'
        with path.open('x')as f:
            json.dump(result,f,indent=2)
            f.write('\n')
        print(json.dumps({k:v for k,v in result.items()if k not in ('files','owner','inputs')}),flush=True)
    finally:
        d.m.old.post.external.release_lock(fd)


if __name__=='__main__':
    main()
