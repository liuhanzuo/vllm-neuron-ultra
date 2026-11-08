"""Original BF16 projection diagnostics, not a full-model qualification.

The unchanged official CPU modules/oneDNN path are the reference. Standalone
packed and split F.linear calls are diagnostics. All mismatches are retained.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

Q = Path(__file__).resolve().parent
sys.path[:0] = [str(Q), str(Q.parent)]
import dory_checkpoint_loader_v1 as loader
import run_dory_weight_storage_root_v2 as d


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def groups(config):
    rows = []
    for i, kind in enumerate(config['layer_types']):
        prefix = f'backbone.layers.{i}.mixer.'
        if kind in ('sliding_attention', 'full_attention'):
            items = [('qkv', ['q_proj', 'k_proj', 'v_proj']),
                     ('gate', ['g_proj']), ('output', ['o_proj'])]
        else:
            assert kind == 'mlp'
            items = [('gate_up', ['gate_proj', 'up_proj']), ('down', ['down_proj'])]
        for kind, names in items:
            rows.append(dict(id=f'layer-{i:02d}-{kind}', layer=i, kind=kind,
                             weights=[prefix+n+'.weight' for n in names]))
    rows.append(dict(id='lm-head', layer=None, kind='lm_head', weights=['lm_head.weight']))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', required=True)
    ap.add_argument('--admission', required=True)
    args = ap.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    admission = json.loads(Path(args.admission).read_text())
    assert admission['spec_sha256'] == hashlib.sha256(Path(args.spec).read_bytes()).hexdigest()
    for name, digest in spec['source_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
    budget = spec['budget']
    resource.setrlimit(resource.RLIMIT_AS, (budget['address_space_bytes'],)*2)
    resource.setrlimit(resource.RLIMIT_CPU, (budget['cpu_seconds'], budget['cpu_seconds']+5))
    resource.setrlimit(resource.RLIMIT_FSIZE, (budget['single_file_bytes'],)*2)
    started = time.monotonic()
    last_guard = [0.]
    out = Path(spec['output'])/'diagnostic'
    out.mkdir()

    def guard():
        now = time.monotonic()
        assert now-started < budget['wall_seconds']
        if now-last_guard[0] > .5:
            d.m.p.lock_proof(admission['owner'], admission['lock'])
            d.m.p.clear_window()
            for path, floor in [('/', budget['root_floor']), ('/mnt/nvme', budget['nvme_floor'])]:
                fs = os.statvfs(path)
                assert fs.f_bavail*fs.f_frsize >= floor
            last_guard[0] = now

    guard()
    import torch
    assert str(torch.__version__) == '2.14.0+cpu'
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    from vllm.config import VllmConfig, DeviceConfig, CompilationConfig, set_current_vllm_config
    from vllm.distributed import parallel_state
    from vllm.model_executor.layers.linear import (
        QKVParallelLinear, MergedColumnParallelLinear, ColumnParallelLinear, RowParallelLinear)
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
    from vllm.platforms import current_platform
    import vllm._custom_ops as ops
    assert current_platform.is_cpu() and ops._supports_onednn
    calls = [0]
    original_mm = ops.onednn_mm

    def counted_mm(*a, **kw):
        calls[0] += 1
        return original_mm(*a, **kw)

    ops.onednn_mm = counted_mm
    parallel_state._TP = SimpleNamespace(world_size=1, rank_in_group=0)
    vc = VllmConfig(device_config=DeviceConfig(device='cpu'),
                    compilation_config=CompilationConfig(mode=0, custom_ops=['none']))

    def raw(t):
        assert t.dtype == torch.bfloat16 and t.device.type == 'cpu'
        return t.detach().contiguous().view(torch.uint8).numpy().tobytes()

    def digest(t):
        return hashlib.sha256(raw(t)).hexdigest()

    def archive(name, t):
        guard()
        data = raw(t)
        path = out/(name+'.bf16')
        with path.open('xb') as f:
            f.write(data)
        return dict(file=path.name, shape=list(t.shape), dtype='BF16', bytes=len(data),
                    sha256=hashlib.sha256(data).hexdigest())

    def compare(a, b):
        assert a.shape == b.shape and a.dtype == b.dtype == torch.bfloat16
        return dict(exact_bytes=digest(a) == digest(b), unequal_values=int((a != b).sum()),
                    max_abs=float((a.double()-b.double()).abs().max()),
                    finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()))

    with torch.inference_mode(), set_current_vllm_config(vc):
        checkpoint = Path(spec['checkpoint']['path'])
        weights, proof = loader.load_checkpoint(checkpoint, spec['checkpoint'], torch=torch, guard=guard)
        save(out/'loader-proof.json', proof)
        expected_proof = json.loads(Path(spec['previous_loader_proof']).read_text())
        assert proof['tensors'] == expected_proof['loader']['tensors']
        plan = groups(loader.source_contract()['config'])
        assert plan == spec['groups'] and len(plan) == 181
        assert len({n for g in plan for n in g['weights']}) == 289
        token_source = json.loads(Path(spec['token_captures']).read_text())
        tokens = next(r['input_ids'] for r in token_source['cases'] if r['name'] == spec['token_case'])
        assert tokens == spec['tokens'] and len(tokens) == 32
        inputs = {2560: weights['backbone.embeddings.weight'][torch.tensor(tokens)]}
        # Deterministic normalized synthetic vectors. These are not captured
        # downstream activations, even though their K dimensions are original.
        for k in (4096, 7168):
            x = (torch.arange(32*k, dtype=torch.int64).reshape(32, k)*17+11).remainder(251).float()-125
            inputs[k] = torch.nn.functional.normalize(x, p=2, dim=-1).to(torch.bfloat16)
        input_records = {}
        for k, x in inputs.items():
            for m in spec['batches']:
                input_records[f'{k}-{m}'] = archive(f'input-{k}-{m}', x[:m])
        save(out/'inputs.json', input_records)
        rows = []
        for group in plan:
            guard()
            matrices = [weights[n] for n in group['weights']]
            packed = torch.cat(matrices, dim=0) if len(matrices) > 1 else matrices[0]
            n, k = packed.shape
            kw = dict(bias=False, params_dtype=torch.bfloat16, prefix=group['id'])
            kind = group['kind']
            if kind == 'qkv':
                layer = QKVParallelLinear(2560, 256, 16, 4, **kw)
                for shard, weight in zip(('q', 'k', 'v'), matrices):
                    layer.weight.weight_loader(layer.weight, weight, shard)
            elif kind == 'gate_up':
                layer = MergedColumnParallelLinear(2560, [7168, 7168], **kw)
                for shard, weight in enumerate(matrices):
                    layer.weight.weight_loader(layer.weight, weight, shard)
            else:
                cls = ParallelLMHead if kind == 'lm_head' else ColumnParallelLinear if kind == 'gate' else RowParallelLinear
                layer = cls(n, k, **kw) if kind == 'lm_head' else cls(k, n, **kw)
                layer.weight.weight_loader(layer.weight, matrices[0])
            assert layer.weight.shape == packed.shape
            expected_sha, loaded_sha = digest(packed), digest(layer.weight)
            assert expected_sha == loaded_sha
            packed_load = dict(shape=[n, k], original_concat_sha256=expected_sha,
                               official_loaded_sha256=loaded_sha,
                               component_sha256={name:digest(weights[name]) for name in group['weights']})
            layer.quant_method.process_weights_after_loading(layer)
            backend_names = list(layer.cpu_linear.__code__.co_names)
            assert 'onednn_mm' in backend_names, backend_names
            for m in spec['batches']:
                guard()
                x = inputs[k][:m]
                before = calls[0]
                official = layer.quant_method.apply(layer, x, bias=None) if kind == 'lm_head' else layer(x)[0]
                assert calls[0]-before == 1
                diagnostic = torch.nn.functional.linear(x, packed)
                values = dict(official=official, packed=diagnostic)
                checks = dict(official_vs_packed=compare(official, diagnostic))
                if len(matrices) > 1:
                    split = torch.cat([torch.nn.functional.linear(x, w) for w in matrices], dim=-1)
                    values['split'] = split
                    checks['official_vs_split'] = compare(official, split)
                    checks['packed_vs_split'] = compare(diagnostic, split)
                case_id = group['id']+f'-m{m}'
                case = dict(id=case_id, group=group, batch=m, input=input_records[f'{k}-{m}'],
                            loaded=packed_load, backend_names=backend_names,
                            outputs={name:archive(case_id+'-'+name, value) for name,value in values.items()},
                            comparisons=checks, actual_onednn_calls=1)
                save(out/(case_id+'.json'), case)
                rows.append(case)
                del values, official, diagnostic
                if len(matrices) > 1:
                    del split
            del layer, packed, matrices
            gc.collect()
            if len(rows) % 30 == 0 or group['id'] == 'lm-head':
                print(json.dumps(dict(completed_cases=len(rows), total_cases=543,
                    mismatching_comparisons=sum(not c['exact_bytes'] for r in rows for c in r['comparisons'].values()),
                    elapsed_seconds=time.monotonic()-started)), flush=True)
        comparisons = [c for row in rows for c in row['comparisons'].values()]
        assert len(rows) == 543 and len(comparisons) == 975 and calls[0] == 543
        assert all(c['finite'] for c in comparisons)
        guard()
        save(out/'result.json', dict(complete=True, cases=rows, groups=181, physical_matrices=289,
             compared_cases=len(rows), comparisons=len(comparisons),
             exact_comparisons=sum(c['exact_bytes'] for c in comparisons),
             all_exact=all(c['exact_bytes'] for c in comparisons), actual_onednn_calls=calls[0],
             max_abs=max(c['max_abs'] for c in comparisons),
             full_model_forward_executed=False, production_slot_reuse_executed=False,
             full_model_qualified=False, new_speedup_claim=False,
             elapsed_seconds=time.monotonic()-started,
             peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
    print(json.dumps(dict(result=str(out/'result.json'), comparisons=len(comparisons),
                         exact=sum(c['exact_bytes'] for c in comparisons))), flush=True)


if __name__ == '__main__':
    main()
