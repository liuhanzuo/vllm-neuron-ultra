# SPDX-License-Identifier: Apache-2.0
"""Pure diagnostic contracts for future external runtime qualification.

These functions neither execute a model nor certify the package for serving.
"""
from collections import Counter
import math
from .identity import require

MARKS = [0, 22, 30, 31, 32, 33, 63, 64, 65, 126, 127, 128, 129, 255, 256, 257, 510, 511, 512, 513, 1023, 1024, 1025, 1542, 1543, 1668, 1669, 2047, 2048, 2049, 2051, 2052, 2053]

def strict_rows(rows, count):
    assert len(rows) == count
    for r in rows:
        assert all(math.isfinite(r[k]) for k in ('cosine', 'kl_reference_to_neuron', 'max_abs_error', 'reference_margin'))
        assert r['cosine'] >= .995 and r['kl_reference_to_neuron'] <= .02 and r['max_abs_error'] <= 2
        assert r['reference_margin'] <= .5 or r['argmax_match']
    assert sum(r['argmax_match'] for r in rows) / len(rows) >= .9

def expected_long_frames():
    rows = [dict(fixture=r, position=n-1, slot=s, phase="prefill")
            for r, n, s in zip([0, 1, 0, 1], [23, 127, 511, 1], [3, 1, 0, 2])]
    for step in range(1541):
        rows += [dict(fixture=r, position=n+step, slot=s, phase="long_decode")
                 for r, n, s in zip([0, 1, 0, 1], [23, 127, 511, 1], [3, 1, 0, 2]) if n+step in MARKS]
    rows.append(dict(fixture=1, position=30, slot=3, phase="slot_reuse_prefill"))
    rows += [dict(fixture=r, position=p, slot=s, phase="decode_after_slot_reuse")
             for r, p, s in zip([1, 1, 0, 1], [31, 1668, 2052, 1542], [3, 1, 0, 2])]
    rows += [dict(fixture=r, position=p, slot=s, phase="partial_batch")
             for r, p, s in zip([1, 0, 1], [1669, 2053, 1543], [1, 0, 2])]
    rows.append(dict(fixture=1, position=32, slot=3, phase="resume_inactive"))
    return rows

def validate_states(rows, zero=False):
    require(len(rows) == len({(r["rank"], r["name"]) for r in rows}) == 96, "All 96 state buffers required")
    expected = {((4, 3072, 4), "torch.bfloat16"): 18,
                ((4, 8, 128, 128), "torch.float32"): 18,
                ((4, 1, 4096, 256), "torch.bfloat16"): 12}
    for rank in (0, 1):
        require(Counter((tuple(r["shape"]), r["dtype"]) for r in rows if r["rank"] == rank) == expected,
                "State dtype/shape/rank changed")
    require(all(r["nonzero_bytes"] == 0 for r in rows) if zero else
            all(r["nan_elements"] == r["inf_elements"] == 0 for r in rows), "State finite/raw-zero gate failed")
