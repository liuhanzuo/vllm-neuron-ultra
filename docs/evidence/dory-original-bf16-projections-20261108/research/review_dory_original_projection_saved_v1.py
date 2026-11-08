"""Independent stdlib verification of complete saved BF16 projection outputs.

No framework import, model inference or GEMM replay. Original checkpoint bytes
are reread; synthetic inputs are authenticated but not numerically regenerated.
"""
import argparse
from array import array
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import struct
import sys
import time

Q = Path(__file__).resolve().parent
sys.path[:0] = [str(Q), str(Q.parent)]
import run_dory_weight_storage_root_v2 as d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', required=True)
    ap.add_argument('--admission', required=True)
    args = ap.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    admission = json.loads(Path(args.admission).read_text())
    assert hashlib.sha256(Path(args.spec).read_bytes()).hexdigest() == admission['spec_sha256']
    for name, sha in spec['source_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == sha
    budget = spec['budget']
    resource.setrlimit(resource.RLIMIT_AS, (budget['address_space_bytes'],)*2)
    resource.setrlimit(resource.RLIMIT_CPU, (budget['cpu_seconds'], budget['cpu_seconds']+5))
    started, last = time.monotonic(), [0.]

    def guard():
        now = time.monotonic()
        assert now-started < budget['wall_seconds']
        if now-last[0] > .5:
            d.m.p.lock_proof(admission['owner'], admission['lock'])
            d.m.p.clear_window()
            last[0] = now

    out = Path(spec['output'])/'diagnostic'
    result = json.loads((out/'result.json').read_text())
    proof = json.loads((out/'loader-proof.json').read_text())
    assert result['complete'] and result['compared_cases'] == 543
    assert len(result['cases']) == 543
    assert sys.byteorder == 'little'
    checkpoint = Path(spec['checkpoint']['path'])
    stat = checkpoint.stat()
    assert dict(path=str(checkpoint), device=stat.st_dev, inode=stat.st_ino, size=stat.st_size,
                mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns) == spec['checkpoint']
    tensors = {r['name']:r for r in proof['tensors']}
    with checkpoint.open('rb') as f:
        prefix = f.read(8)
        header_bytes = f.read(struct.unpack('<Q', prefix)[0])
        header = json.loads(header_bytes)
        base = 8+len(header_bytes)
        assert set(header) == set(tensors) and len(header) == 507
        full_hash = hashlib.sha256(prefix+header_bytes)
        for name, row in sorted(header.items(), key=lambda x:x[1]['data_offsets'][0]):
            begin, end = row['data_offsets']
            assert f.tell() == base+begin and row['dtype'] == 'BF16'
            h = hashlib.sha256()
            remaining = end-begin
            while remaining:
                guard()
                data = f.read(min(remaining, 4*2**20))
                assert data
                remaining -= len(data)
                h.update(data)
                full_hash.update(data)
            assert h.hexdigest() == tensors[name]['sha256']
            assert row['shape'] == tensors[name]['shape']
            assert end-begin == tensors[name]['bytes']
        assert f.tell() == spec['checkpoint']['size'] and not f.read(1)
        assert full_hash.hexdigest() == spec['checkpoint_sha256'] == proof['full_sha256']
        expected_projection_weights = {n for n, r in header.items()
            if len(r['shape']) == 2 and n != 'backbone.embeddings.weight'}
        assert expected_projection_weights == {n for g in spec['groups'] for n in g['weights']}
        assert len(expected_projection_weights) == 289
        packed_hashes = {}
        for group in spec['groups']:
            h = hashlib.sha256()
            for name in group['weights']:
                begin, end = header[name]['data_offsets']
                f.seek(base+begin)
                remaining = end-begin
                while remaining:
                    guard()
                    data = f.read(min(remaining, 4*2**20))
                    assert data
                    remaining -= len(data)
                    h.update(data)
            packed_hashes[group['id']] = h.hexdigest()
        embedding_bytes = bytearray()
        begin = header['backbone.embeddings.weight']['data_offsets'][0]
        for token in spec['tokens']:
            f.seek(base+begin+token*2560*2)
            embedding_bytes.extend(f.read(2560*2))
    files = {}

    def load(row):
        guard()
        name = row['file']
        assert Path(name).name == name and row['dtype'] == 'BF16'
        data = (out/name).read_bytes()
        assert len(data) == row['bytes'] == 2*math.prod(row['shape'])
        assert hashlib.sha256(data).hexdigest() == row['sha256']
        files[name] = dict(bytes=len(data), sha256=row['sha256'])
        values = array('H')
        values.frombytes(data)
        assert all((v & 0x7f80) != 0x7f80 for v in values), 'Finite BF16'
        return data, values

    inputs = json.loads((out/'inputs.json').read_text())
    assert set(inputs) == {f'{k}-{m}' for k in (2560,4096,7168) for m in (1,7,32)}
    for key, row in inputs.items():
        data, _ = load(row)
        k, m = map(int, key.split('-'))
        assert row['shape'] == [m, k]
        if k == 2560:
            assert data == embedding_bytes[:m*k*2]
        else:
            assert data == (out/inputs[f'{k}-32']['file']).read_bytes()[:m*k*2]

    decoded = {}

    def number(word):
        if word not in decoded:
            decoded[word] = struct.unpack('<f', struct.pack('<I', word << 16))[0]
        return decoded[word]

    def compare(a, b):
        assert len(a[1]) == len(b[1])
        unequal, maximum = 0, 0.
        for x, y in zip(a[1], b[1]):
            if x != y:
                xv, yv = number(x), number(y)
                unequal += xv != yv
                maximum = max(maximum, abs(xv-yv))
        return dict(exact_bytes=a[0] == b[0], unequal_values=unequal, max_abs=maximum, finite=True)

    expected_cases = {(g['id'], m) for g in spec['groups'] for m in spec['batches']}
    assert len(expected_cases) == 543
    seen, counts, summary = set(), [0, 0], {}
    for row in result['cases']:
        group, m = row['group'], row['batch']
        key = (group['id'], m)
        assert key in expected_cases and key not in seen
        seen.add(key)
        assert group == next(g for g in spec['groups'] if g['id'] == group['id'])
        assert json.loads((out/(row['id']+'.json')).read_text()) == row
        assert row['actual_onednn_calls'] == 1 and 'onednn_mm' in row['backend_names']
        assert row['loaded']['original_concat_sha256'] == row['loaded']['official_loaded_sha256'] == packed_hashes[group['id']]
        assert row['loaded']['component_sha256'] == {n:tensors[n]['sha256'] for n in group['weights']}
        n = sum(header[w]['shape'][0] for w in group['weights'])
        k = header[group['weights'][0]]['shape'][1]
        assert row['loaded']['shape'] == [n, k] and row['input'] == inputs[f'{k}-{m}']
        names = {'official','packed','split'} if len(group['weights']) > 1 else {'official','packed'}
        assert set(row['outputs']) == names
        values = {name:load(r) for name,r in row['outputs'].items()}
        assert all(r['shape'] == [m,n] for r in row['outputs'].values())
        pairs = [('official','packed')]+([('official','split'),('packed','split')] if 'split' in names else [])
        assert set(row['comparisons']) == {a+'_vs_'+b for a,b in pairs}
        for a,b in pairs:
            comparison = compare(values[a], values[b])
            name = a+'_vs_'+b
            assert comparison == row['comparisons'][name], (row['id'], name)
            counts[0] += 1
            counts[1] += comparison['exact_bytes']
            key = group['kind']+f'/M={m}/'+name
            aggregate = summary.setdefault(key, dict(cases=0, exact=0, unequal_values=0, max_abs=0.))
            aggregate['cases'] += 1
            aggregate['exact'] += comparison['exact_bytes']
            aggregate['unequal_values'] += comparison['unequal_values']
            aggregate['max_abs'] = max(aggregate['max_abs'], comparison['max_abs'])
    assert seen == expected_cases and counts[0] == 975
    assert result['comparisons'] == counts[0] and result['exact_comparisons'] == counts[1]
    assert result['all_exact'] == (counts[0] == counts[1])
    assert result['max_abs'] == max(r['max_abs'] for r in summary.values())
    assert len(files) == 1311
    assert not {'torch','numpy','transformers'} & set(sys.modules)
    guard()
    record = dict(passed=True, saved_cases_checked=543, saved_comparisons_reproduced=975,
        exact_comparisons=counts[1], saved_raw_files=len(files),
        saved_raw_bytes=sum(r['bytes'] for r in files.values()), all_inputs_outputs_finite=True,
        original_checkpoint_sha256=full_hash.hexdigest(), original_tensors_rehashed=507,
        packed_weight_hashes_independently_reconstructed=181,
        embedding_inputs_independently_reconstructed=True,
        synthetic_inputs_numerically_regenerated=False, matrix_multiplication_reexecuted=False,
        summaries=summary, files=files, elapsed_seconds=time.monotonic()-started,
        full_model_qualified=False, new_speedup_claim=False)
    destination = Path(spec['review_output'])/'saved-review.json'
    with destination.open('x') as f:
        json.dump(record, f, indent=2)
        f.write('\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ('files','summaries')}), flush=True)


if __name__ == '__main__':
    main()
