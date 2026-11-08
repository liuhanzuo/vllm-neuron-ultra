"""Post-hoc batch-prefix analysis of saved original-size BF16 projections.

No model/GEMM execution. Comparing identical input prefixes characterizes batch
dependence separately for oneDNN and F.linear; it does not pick a new baseline.
"""
import argparse
from array import array
import hashlib
import json
import os
from pathlib import Path
import resource
import struct
import sys
import time

Q = Path(__file__).resolve().parent
sys.path[:0] = [str(Q), str(Q.parent)]
import run_dory_weight_storage_root_v2 as d

BASE = Path('/mnt/nvme/dory-official-cpu-20261108-v1')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-sha256', required=True)
    args = ap.parse_args()
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == args.source_sha256
    fd, owner, lock = d.m.old.post.external.acquire_lock()
    started = time.monotonic()
    try:
        d.m.p.clear_window()
        d.m.p.active_heavy()
        os.sched_setaffinity(0, {4})
        resource.setrlimit(resource.RLIMIT_AS, (2*2**30,)*2)
        resource.setrlimit(resource.RLIMIT_CPU, (300,305))
        root = BASE/'projection-sweep-r2/diagnostic'
        result = json.loads((root/'result.json').read_text())
        review_path = BASE/'projection-sweep-review-r2/saved-review.json'
        review = json.loads(review_path.read_text())
        assert review['passed'] and review['saved_comparisons_reproduced'] == 975
        records = {}
        for row in result['cases']:
            records.setdefault(row['group']['id'], {})[row['batch']] = row
        assert len(records) == 181
        checked_files = {}

        def load(row):
            data = (root/row['file']).read_bytes()
            assert len(data) == row['bytes']
            digest = hashlib.sha256(data).hexdigest()
            assert digest == row['sha256'] == review['files'][row['file']]['sha256']
            checked_files[row['file']] = digest
            return data

        def compare(left, right):
            assert len(left) == len(right) and len(left) % 2 == 0
            a,b = array('H'),array('H')
            a.frombytes(left)
            b.frombytes(right)
            unequal, maximum = 0,0.
            for x,y in zip(a,b):
                if x != y:
                    xx = struct.unpack('<f', struct.pack('<I', x<<16))[0]
                    yy = struct.unpack('<f', struct.pack('<I', y<<16))[0]
                    unequal += xx != yy
                    maximum = max(maximum, abs(xx-yy))
            return dict(exact_bytes=left==right, unequal_values=unequal, max_abs=maximum)

        cases,summary = [],{}
        for group, batches in records.items():
            d.m.p.lock_proof(owner,lock)
            assert time.monotonic()-started < 300
            assert set(batches) == {1,7,32}
            inputs = {m:load(batches[m]['input']) for m in batches}
            for small,large in [(1,7),(1,32),(7,32)]:
                assert inputs[small] == inputs[large][:len(inputs[small])]
            for backend in batches[1]['outputs']:
                outputs = {m:load(batches[m]['outputs'][backend]) for m in batches}
                for small,large in [(1,7),(1,32),(7,32)]:
                    c = compare(outputs[small], outputs[large][:len(outputs[small])])
                    cases.append(dict(group=group, kind=batches[1]['group']['kind'], backend=backend,
                                      smaller_batch=small, larger_batch=large, comparison=c))
                    key = f'{backend}/{small}-vs-{large}'
                    aggregate = summary.setdefault(key, dict(cases=0, exact=0, unequal_values=0, max_abs=0.))
                    aggregate['cases'] += 1
                    aggregate['exact'] += c['exact_bytes']
                    aggregate['unequal_values'] += c['unequal_values']
                    aggregate['max_abs'] = max(aggregate['max_abs'], c['max_abs'])
        assert len(cases) == 1302 and len(checked_files) == 1311
        assert not {'torch','numpy','transformers'} & set(sys.modules)
        value = dict(complete=True, post_hoc_saved_byte_analysis=True, owner=owner,
            source_sha256=args.source_sha256,
            input_result_sha256=hashlib.sha256((root/'result.json').read_bytes()).hexdigest(),
            independent_review_sha256=hashlib.sha256(review_path.read_bytes()).hexdigest(),
            identical_input_prefixes_verified=True, saved_files_rehashed=len(checked_files),
            comparisons=len(cases), summary=summary, cases=cases,
            matrix_multiplication_reexecuted=False, full_model_qualified=False, new_speedup_claim=False,
            elapsed_seconds=time.monotonic()-started)
        destination = BASE/'projection-sweep-batch-prefix-v1.json'
        with destination.open('x') as f:
            json.dump(value,f,indent=2)
            f.write('\n')
        print(json.dumps(dict(destination=str(destination), comparisons=len(cases), summary=summary)),flush=True)
    finally:
        d.m.old.post.external.release_lock(fd)


if __name__ == '__main__':
    main()
