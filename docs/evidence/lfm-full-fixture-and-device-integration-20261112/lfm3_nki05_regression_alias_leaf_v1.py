"""Test cache writes with real MUST_ALIAS state and simultaneous attention reads.

This is an isolated compiler/runtime diagnostic. Run only in a serialized slot.
It does not replace the failed original-weight numerical gate.
"""
import gc
import hashlib
import json
import time
from pathlib import Path

from lfm3_nki05_device_context_v1 import require_context


def run(context):
    phase='regression_alias'
    ROOT,DEST=require_context(context,phase)
    proof=context.prerequisite('regression_device')
    assert proof['complete'] and proof['passed'] and len(proof['graphs'])==9
    assert sum(len(row['checks']) for row in proof['graphs'])==43
    import torch
    import torch_neuronx
    store_slots_nki_predicated=context.writer(phase)
    from lfm3_nki05_device_oracle_v1 import store_slots
    torch.set_num_threads(1)
    torch.manual_seed(20261031)
    output = DEST / 'validation.json'
    assert not output.exists(), 'Preserve previous probe results'
    report = {'complete': False, 'graphs': [],
              'scope': 'Real stateful input-output aliasing, attention reads and writes in the same '
                       'compiled graph, then consecutive calls. Cache bytes and output finiteness '
                       'are checked; attention rounding is recorded separately. No fullmodel '
                       'qualification or throughput claim.'}

    def save():
        context.save(output,report,phase)

    def same(a, b):
        return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))

    class CacheAttention(torch.nn.Module):
        def __init__(self, kc, vc, writer):
            super().__init__()
            self.kc = torch.nn.Parameter(kc.clone(), requires_grad=False)
            self.vc = torch.nn.Parameter(vc.clone(), requires_grad=False)
            self.writer = writer

        def forward(self, q, k, v, positions, slots):
            index = positions[:, None, :, None].expand(4, self.kc.shape[1], 1, 128)
            sk = torch.scatter(self.kc[slots.clamp_min(0)], 2, index, k)
            sv = torch.scatter(self.vc[slots.clamp_min(0)], 2, index, v)
            scores = (q @ sk.repeat_interleave(4, dim=1).transpose(-1, -2)) * (128 ** -.5)
            allowed = torch.arange(self.kc.shape[2], device=positions.device)[None, None, None] <= positions[:, None, :, None]
            scores = scores + torch.where(allowed, torch.zeros_like(scores),
                                         torch.full_like(scores, torch.finfo(scores.dtype).min))
            attended = scores.softmax(-1, dtype=torch.float32).to(q.dtype) @ sv.repeat_interleave(4, dim=1)
            return attended, self.writer(self.kc, sk, slots), self.writer(self.vc, sv, slots)

    mappings = [[0, 1, 2, 3], [3, 1, 0, 2], [3, -1, -1, -1], [-1, -1, -1, -1], [2, 2, 0, 3]]
    try:
        with torch.inference_mode():
            for length, heads in [(128, 2), (256, 2), (512, 2), (1024, 2), (2048, 2), (4096, 4)]:
                initial_k = torch.randn(4, heads, length, 128, dtype=torch.bfloat16) * .1
                initial_v = torch.randn_like(initial_k) * .1
                samples = []
                for step, slots in enumerate(mappings):
                    samples.append((torch.randn(4, 4 * heads, 1, 128, dtype=torch.bfloat16) * .1,
                        torch.randn(4, heads, 1, 128, dtype=torch.bfloat16) * .1,
                        torch.randn(4, heads, 1, 128, dtype=torch.bfloat16) * .1,
                        torch.tensor([[11 + step], [39 + step], [length - 7 + step], [min(119, length - 7) + step]]),
                        torch.tensor(slots)))
                for label, writer in [('reference', store_slots), ('candidate', store_slots_nki_predicated)]:
                    eager = CacheAttention(initial_k, initial_v, store_slots).eval()
                    source = CacheAttention(initial_k, initial_v, writer).eval()
                    directory = DEST / (label + '-' + str(length))
                    assert not directory.exists()
                    directory.mkdir()
                    start = time.monotonic()
                    compiled = torch_neuronx.trace(source, samples[0],
                        input_output_aliases={source.kc: 1, source.vc: 2},
                        compiler_workdir=str(directory), compiler_args=['--target=trn1', '--auto-cast=none',
                            '--model-type=transformer', '-O2', '--lnc=1',
                            "--tensorizer-options=--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=1 --vectorize-strided-dma"])
                    compiled.save(str(directory / 'model.pt'))
                    artifact_identity=context.artifact(directory/'model.pt',phase,dict(variant=label,length=length,kv_heads=heads,shape=[4,heads,length,128],rows=4,dtype='bfloat16'))
                    states = dict(compiled.states._parameters.items())
                    k_names = [name for name, value in states.items() if same(value, initial_k)]
                    v_names = [name for name, value in states.items() if same(value, initial_v)]
                    assert len(k_names) == len(v_names) == 1 and k_names != v_names
                    del states
                    torch_neuronx.move_trace_to_device(compiled, 0)
                    assert all(value.device.type == "privateuseone" for value in dict(compiled.states._parameters.items()).values())
                    row = {'variant': label, 'length': length, 'kv_heads': heads, 'artifact_identity':artifact_identity, 'compile_seconds': time.monotonic() - start, 'checks': []}
                    report['graphs'].append(row)
                    for step, sample in enumerate(samples):
                        expected = eager(*sample)
                        actual = tuple(value.detach().cpu().clone() for value in compiled(*sample))
                        states = dict(compiled.states._parameters.items())
                        finite = bool(torch.isfinite(actual[0]).all())
                        check = {'step': step, 'slots': sample[-1].tolist(),
                                 'attention_finite': finite,
                                 'cache_k_bitwise': same(actual[1], expected[1]),
                                 'cache_v_bitwise': same(actual[2], expected[2]),
                                 'input_k_bitwise': same(states[k_names[0]].detach().cpu(), expected[1]),
                                 'input_v_bitwise': same(states[v_names[0]].detach().cpu(), expected[2]),
                                 'attention_max_abs_error': float((actual[0].float() - expected[0].float()).abs().max()) if finite else None}
                        row['checks'].append(check)
                        eager.kc.copy_(expected[1])
                        eager.vc.copy_(expected[2])
                        save()
                    row['passed'] = all(c['attention_finite'] and c['cache_k_bitwise'] and c['cache_v_bitwise'] and c['input_k_bitwise'] and c['input_v_bitwise'] for c in row['checks'])
                    print(json.dumps(row), flush=True)
                    del states, compiled
                    gc.collect()
            report['complete'] = True
            report['passed'] = len(report['graphs']) == 12 and all(row['passed'] for row in report['graphs'])
            assert report['passed'], 'Persistent alias correctness failed'
    except Exception as exc:
        report['error'] = repr(exc)
        raise
    finally:
        report['source_sha256'] = context.sources()
        save()
