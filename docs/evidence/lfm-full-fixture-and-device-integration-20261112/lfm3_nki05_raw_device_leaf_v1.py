"""Complete original raw device matrix with explicit NKI0.5 context; no CLI."""
import json
import sys
from lfm3_nki05_device_context_v1 import require_context

MAPPINGS = [[0, 1, 2, 3], [3, 1, 0, 2], [3, -1, -1, -1],
            [-1, -1, -1, -1], [2, 2, 0, 3], [4, 0, -1, 2],
            [0, 9, 0, 0], [-9, -9, -9, -9], [4, 5, 6, 7]]


def run(context):
    phase='raw_device'
    ROOT,DEST=require_context(context,phase)
    mode='device'
    plan={'source_sha256':context.sources()}
    import numpy as np
    import ml_dtypes
    import torch
    import torch_neuronx
    store_slots_nki_predicated=context.writer(phase)
    torch.set_num_threads(1)

    class Write(torch.nn.Module):
        def forward(self,cache,updates,slots):
            return store_slots_nki_predicated(cache,updates,slots)

    report = {'complete': False, 'passed': False, 'mode': mode, 'checks': [], 'graphs': [],
              'source_sha256': plan['source_sha256'],
              'scope': 'Original four actual device graphs and 36 strict raw-bit mappings. Every BF16 word; no simulator writeback in this device path. Complete 3716 CPU gates are separate. No alias/model/speed claim.'}
    output = DEST / (mode + '.json')
    assert not output.exists()

    def save():
        context.save(output,report,phase)

    try:
        for dtype_name in ('bfloat16', 'float32'):
            dtype = ml_dtypes.bfloat16 if dtype_name == 'bfloat16' else np.float32
            for width in (7168, 65536):
                itemsize = np.dtype(dtype).itemsize
                words = np.arange(4 * width * (itemsize // 2), dtype=np.uint32).astype(np.uint16).reshape(4, -1)
                words = np.stack([np.roll(words[row], row * 101) for row in range(4)])
                cache = words.copy().view(dtype)
                updates = np.bitwise_xor(words, np.uint16(0x5555)).view(dtype)
                specials = [0x8000, 0, 0x7f80, 0xff80, 0x7fc1, 0x7fa1, 1, 0x8001] if itemsize == 2 else \
                           [0x80000000, 0, 0x7f800000, 0xff800000, 0x7fc00001, 0x7fa00001, 1, 0x80000001]
                raw_dtype = np.uint16 if itemsize == 2 else np.uint32
                cache.view(raw_dtype)[:, :8] = np.asarray(specials, dtype=raw_dtype)
                updates.view(raw_dtype)[:, :8] = np.asarray(specials[::-1], dtype=raw_dtype)
                if width == 65536:
                    assert np.unique(cache.view(np.uint16)).size == 65536
                    assert np.unique(updates.view(np.uint16)).size == 65536
                before_cache, before_updates = cache.tobytes(), updates.tobytes()
                tdtype = torch.bfloat16 if itemsize == 2 else torch.float32
                tc = torch.from_numpy(cache.view(np.uint16).copy()).view(tdtype)
                tu = torch.from_numpy(updates.view(np.uint16).copy()).view(tdtype)
                directory = DEST / f'{dtype_name}-{width}'
                assert not directory.exists()
                directory.mkdir()
                compiled = torch_neuronx.trace(Write().eval(), (tc, tu, torch.arange(4)),
                    compiler_workdir=str(directory),
                    compiler_args=['--target=trn1', '--auto-cast=none', '-O2'])
                artifact = directory / 'model.pt'
                compiled.save(str(artifact))
                artifact_identity=context.artifact(artifact,phase,dict(dtype=dtype_name,shape=[4,width],rows=4,variant='candidate'))
                report['graphs'].append({'dtype': dtype_name, 'width': width,
                    'artifact_sha256': context.hash_artifact(artifact,phase), 'artifact_identity':artifact_identity})
                for mapping in MAPPINGS:
                    expected = cache.copy()
                    for row, slot in enumerate(mapping):
                        if 0 <= slot < 4:
                            expected[slot] = updates[row]
                    actual = compiled(tc, tu, torch.tensor(mapping))
                    assert actual.shape == tc.shape and actual.dtype == tc.dtype
                    actual_bytes = actual.contiguous().view(torch.uint8).numpy().tobytes()
                    unchanged = tc.view(torch.uint8).numpy().tobytes() == before_cache and tu.view(torch.uint8).numpy().tobytes() == before_updates
                    row = {'dtype': dtype_name, 'width': width, 'slots': mapping,
                           'bitwise_equal': actual_bytes == expected.tobytes(), 'inputs_unchanged': unchanged}
                    report['checks'].append(row)
                    save()
                    assert row['bitwise_equal'] and unchanged, row
                del compiled
        assert len(report['graphs'])==4 and len(report['checks'])==36
        report.update(complete=True, passed=True, torch_imported='torch' in sys.modules)
    except Exception as exc:
        report['error'] = repr(exc)
        raise
    finally:
        save()
        print(json.dumps({k: v for k, v in report.items() if k not in ('checks', 'source_sha256')}), flush=True)
