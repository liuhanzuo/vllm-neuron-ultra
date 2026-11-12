"""Additional observations inside the exclusive full-model cold worker."""
import ast

from vllm_neuron.models.qwen35.identity import BUCKETS
from vllm_neuron.models.qwen35.validation import validate_states


def raw_zero_states(model):
    import torch
    rows = []
    for rank, tensors in enumerate(model.traced_model.nxd_model.state):
        for name, tensor in sorted(tensors.items()):
            cpu = tensor.detach().cpu().contiguous()
            rows.append(dict(rank=rank, name=name, shape=list(cpu.shape), dtype=str(cpu.dtype),
                             nonzero_bytes=int(torch.count_nonzero(cpu.view(torch.uint8)))))
            del cpu
    validate_states(rows, zero=True)
    return rows


def graph_routes(model):
    rows = []
    mapping = model.traced_model.nxd_model.input_shape_map
    for signature, chosen in mapping.items():
        shapes = ast.literal_eval(signature)
        tag, index = str(chosen[0]), int(chosen[1])
        assert tag in ('context_encoding_model', 'token_generation_model')
        batch = 1 if tag == 'context_encoding_model' else 4
        assert shapes[0][0] == shapes[1][0] == batch
        bucket = shapes[0][1] if tag == 'context_encoding_model' else shapes[1][1]
        if tag == 'token_generation_model':
            assert shapes[0][1] == 1
        rows.append(dict(tag=tag, index=index, bucket=bucket, input_shapes=shapes))
    assert len(rows) == 12
    for tag in ('context_encoding_model', 'token_generation_model'):
        assert sorted((r['index'], r['bucket']) for r in rows if r['tag'] == tag) == list(enumerate(BUCKETS))
    return dict(complete=True, buckets=list(BUCKETS), graphs=rows)


import json
from pathlib import Path
ROOT = Path("/home/ubuntu/experiments/trn1-latest-models-20261031")

def verify_cold_reproduction(captured, report):
    import torch
    prior_dir = ROOT / 'qwen35-split-corrected-full-v4'
    prior = torch.load(prior_dir / 'cold-buckets.pt', map_location='cpu', weights_only=True)
    original = json.loads((prior_dir / 'cold-buckets.json').read_text())
    assert len(prior) == 72 and len(captured) == 84 and not original['complete']
    keys = ('prefix', 'phase', 'step', 'slot', 'fixture')
    for index, (old, new) in enumerate(zip(prior, captured)):
        assert {key: old[key] for key in keys} == {key: new[key] for key in keys}, index
        a, b = old['logits'], new['logits']
        assert a.dtype == b.dtype == torch.float32 and a.shape == b.shape == (248320,)
        assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)), index
    receipts = report['state_reset_receipts']
    assert len(receipts) == 7 and all(row['complete'] and row['states_written'] == 96 and row['bindings_unchanged'] for row in receipts)
    assert all(row['bindings'] == receipts[0]['bindings'] for row in receipts)
    assert all(row['states_nonzero_before_clear'] > 0 for row in receipts[1:])
    return dict(passed=True, original_partial_vectors_reproduced_exactly=72,
        complete_candidate_vectors=84, clearing_operations=7, state_writes=672,
        same96_tensor_bindings_across_all_seven_clears=True,
        all_six_subsequent_clears_observed_dirty_states=True,
        original_reinitialization_allocation_failure_preserved=True)
