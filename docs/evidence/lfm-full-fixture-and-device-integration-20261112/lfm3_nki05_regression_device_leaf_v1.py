"""Compile actual KV/conv/SSM shapes only after the complete CPU simulator gate."""
import hashlib
import json
import time
from pathlib import Path

from lfm3_nki05_device_context_v1 import require_context


def run(context):
    phase='regression_device'
    ROOT,DEST=require_context(context,phase)
    proof=context.prerequisite('raw_device')
    assert proof['complete'] and proof['passed'] and len(proof['graphs'])==4 and len(proof['checks'])==36
    import torch
    import torch_neuronx
    store_slots_nki_predicated=context.writer(phase)
    from lfm3_nki05_device_oracle_v1 import store_slots
    torch.set_num_threads(1)
    torch.manual_seed(20261031)
    output = DEST / 'validation.json'
    assert not output.exists()
    report = {'complete': False, 'passed': False, 'graphs': [],
              'source_sha256':context.sources(),
              'scope': 'Standalone actual Trn1 NKI primitive graphs. LFM five KV lengths, convolution and one-row '
                       'prefill; RNJ4096 KV and Nemotron FP32 SSM widths. Raw bytes against sequential Torch '
                       'oracle, including signed zero/Inf/NaN and unchanged input buffers. '
                       'No alias-state, full-model, quality or throughput claim.'}

    def save():
        context.save(output,report,phase)

    def same(a, b):
        return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))

    class Write(torch.nn.Module):
        def forward(self, cache, updates, slots):
            return store_slots_nki_predicated(cache, updates, slots)

    shapes = [(f'kv{length}_b4', (4, 2, length, 128), 4, torch.bfloat16)
              for length in (128, 256, 512, 1024, 2048)]
    shapes += [('conv_b4', (4, 1024, 7), 4, torch.bfloat16),
               ('kv2048_b1', (4, 2, 2048, 128), 1, torch.bfloat16),
               ('rnj_kv4096_b4', (4, 4, 4096, 128), 4, torch.bfloat16),
               ('nemotron_ssm_b4', (4, 64, 80, 128), 4, torch.float32)]
    mappings = [[0, 1, 2, 3], [3, 1, 0, 2], [3, -1, -1, -1], [-1, -1, -1, -1], [2, 2, 0, 3]]
    try:
        with torch.inference_mode():
            for name, shape, rows, dtype in shapes:
                cache = torch.randn(shape, dtype=dtype)
                updates = torch.randn((rows,) + shape[1:], dtype=dtype)
                special = torch.tensor([-0., float('inf'), float('-inf'), float('nan')], dtype=dtype)
                cache.reshape(4, -1)[:, :4] = special
                updates.reshape(rows, -1)[:, :4] = special.flip(0)
                before_cache, before_updates = cache.clone(), updates.clone()
                directory = DEST / name
                assert not directory.exists()
                directory.mkdir()
                started = time.monotonic()
                compiled = torch_neuronx.trace(Write().eval(), (cache, updates, torch.arange(rows)),
                    compiler_workdir=str(directory), compiler_args=['--target=trn1', '--auto-cast=none', '-O2'])
                artifact = directory / 'model.pt'
                compiled.save(str(artifact))
                artifact_identity=context.artifact(artifact,phase,dict(name=name,shape=list(shape),rows=rows,dtype=str(dtype),variant='candidate'))
                row = {'name': name, 'shape': list(shape), 'dtype': str(dtype),
                       'compile_seconds': time.monotonic() - started,
                       'artifact_sha256': context.hash_artifact(artifact,phase), 'artifact_identity':artifact_identity, 'checks': []}
                report['graphs'].append(row)
                for mapping in mappings if rows == 4 else [[0], [3], [-1]]:
                    slots = torch.tensor(mapping)
                    expected = store_slots(cache, updates, slots)
                    actual = compiled(cache, updates, slots)
                    check = {'slots': mapping, 'bitwise_equal': same(actual, expected),
                             'cache_input_unchanged': same(cache, before_cache),
                             'updates_input_unchanged': same(updates, before_updates)}
                    row['checks'].append(check)
                    save()
                    assert all(check[key] for key in ('bitwise_equal', 'cache_input_unchanged', 'updates_input_unchanged')), (name, mapping)
                del compiled
                print(json.dumps(row), flush=True)
            assert len(report['graphs']) == 9 and sum(len(row['checks']) for row in report['graphs']) == 43
            report.update(complete=True, passed=True)
    except Exception as exc:
        report['error'] = repr(exc)
        raise
    finally:
        save()
