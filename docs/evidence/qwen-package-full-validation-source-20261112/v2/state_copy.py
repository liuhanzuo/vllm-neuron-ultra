"""Test real privateuseone copy and clearing for all96 original state buffers."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from .runtime import save, read_build_inventory
from vllm_neuron.models.qwen35.state import binding_metadata, clear_states


def digest(value):
    import torch
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main(directory, build_inventory):
    DEST = Path(directory)
    output = DEST / 'state-copy.json'; assert not output.exists()
    report = dict(complete=False, passed=False, patterns=[], source_sha256={str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        full_model_qualified=False, performance_measured=False)
    try:
        import torch
        import torch_neuronx
        torch.set_num_threads(1)
        layout = read_build_inventory(build_inventory)['state_layout']
        states = [{}, {}]
        for row in layout:
            dtype = {'torch.bfloat16': torch.bfloat16, 'torch.float32': torch.float32}[row['dtype']]
            states[row['rank']][row['name']] = torch.zeros(row['shape'], dtype=dtype).to('privateuseone:' + str(row['rank']))
        model = SimpleNamespace(state=states); bindings = binding_metadata(model)
        for pattern in range(2):
            fills = []
            for rank, tensors in enumerate(states):
                for index, (name, target) in enumerate(sorted(tensors.items())):
                    words = torch.arange(target.numel(), dtype=torch.int64)
                    salt = 17 + pattern * 65537 + rank * 131 + index * 31
                    if target.dtype == torch.bfloat16:
                        source = ((words * 637 + salt) % 65536).to(torch.int16).view(torch.bfloat16).reshape(target.shape)
                    else:
                        source = (words * 2654435761 + salt).to(torch.int32).view(torch.float32).reshape(target.shape)
                    target.copy_(source)
                    actual = target.detach().cpu().contiguous()
                    assert torch.equal(actual.view(torch.uint8), source.contiguous().view(torch.uint8)), (pattern, rank, name)
                    fills.append(dict(rank=rank, name=name, shape=list(source.shape), dtype=str(source.dtype),
                        bytes_compared=source.numel() * source.element_size(), expected_sha256=digest(source),
                        actual_sha256=digest(actual), passed=True))
                    del words, source, actual
            assert binding_metadata(model) == bindings
            receipt = clear_states(model, bindings)
            assert receipt['states_nonzero_before_clear'] == 96
            assert [(x['rank'], x['name'], x['before_sha256']) for x in receipt['rows']] == [
                (x['rank'], x['name'], x['expected_sha256']) for x in fills]
            zero_checks = []
            for rank, tensors in enumerate(states):
                for name, target in sorted(tensors.items()):
                    actual = target.detach().cpu().contiguous()
                    nonzero = int(torch.count_nonzero(actual.view(torch.uint8)))
                    assert nonzero == 0, (pattern, rank, name)
                    zero_checks.append(dict(rank=rank, name=name, shape=list(actual.shape), dtype=str(actual.dtype),
                        bytes_checked=actual.numel() * actual.element_size(), nonzero_bytes=nonzero, passed=True))
                    del actual
            assert binding_metadata(model) == bindings
            report['patterns'].append(dict(pattern=pattern, fills=fills, reset=receipt, zero_checks=zero_checks))
            save(output, report)
        report.update(complete=True, passed=True, initial_bindings=bindings, patterns_tested=2,
            state_buffers=96, fill_roundtrip_comparisons=192, raw_zero_comparisons=192,
            same_tensor_bindings_preserved=True, graph_models_loaded=0)
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        save(output, report)


if __name__ == '__main__': raise SystemExit('Use entry.run_model_stage with the root execution guard')
