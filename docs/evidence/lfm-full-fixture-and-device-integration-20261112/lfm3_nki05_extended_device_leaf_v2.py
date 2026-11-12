"""Full immutable successors of original extended device and alias scientific bodies."""
import gc
import hashlib
import time
from lfm3_nki05_device_context_v1 import require_context
from lfm3_nki05_device_matrix_v1 import BoundMatrix


def torch_same(a, b):
    import torch
    return torch.equal(a.detach().cpu().contiguous().view(torch.uint8),
                       b.detach().cpu().contiguous().view(torch.uint8))


def tensor_sha(value):
    import torch
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def device(context,index):
    phase='extended_device'
    require_context(context,phase)
    proof=context.prerequisite('regression_alias')
    assert proof['complete'] and proof['passed']
    p=BoundMatrix(context,phase)
    assert type(index) is int and 0<=index<len(p.device_specs())
    import torch
    import torch_neuronx
    store_slots_nki_predicated=context.writer(phase)
    from lfm3_nki05_device_oracle_v1 import store_slots
    torch.set_num_threads(1)
    torch.manual_seed(20261103 + index)
    spec = p.device_specs()[index]
    capacity, rows = spec['capacity'], spec['rows']
    directory = p.DEST / spec['name']
    directory.mkdir()
    cache = torch.randn(spec['shape'], dtype=torch.bfloat16)
    updates = torch.randn((rows, *spec['shape'][1:]), dtype=torch.bfloat16)
    special = torch.tensor([0, 0x8000, 0x7f80, 0xff80, 0x7fc1, 0x7fff, 0xffc3, 1,
                            0x7f, 0x80, 0x3f80, 0xbf80], dtype=torch.uint16).view(torch.bfloat16)
    for i, row in enumerate(cache.reshape(capacity, -1)):
        row[:len(special)] = special.roll(i)
    for i, row in enumerate(updates.reshape(rows, -1)):
        row[:len(special)] = special.flip(0).roll(i)
    before_cache, before_updates = cache.clone(), updates.clone()
    class Write(torch.nn.Module):
        def forward(self, cache, updates, slots):
            return store_slots_nki_predicated(cache, updates, slots)
    result = dict(complete=False, passed=False, spec=spec, checks=[])
    began = time.monotonic()
    try:
        with torch.inference_mode():
            compiled = torch_neuronx.trace(Write().eval(), (cache, updates, torch.arange(rows)),
                compiler_workdir=str(directory / 'compiler'),
                compiler_args=['--target=trn1', '--auto-cast=none', '-O2'])
            artifact = directory / 'model.pt'
            compiled.save(str(artifact))
            result['artifact_identity']=context.artifact(artifact,phase,dict(spec,variant='candidate'))
            result.update(compile_seconds=time.monotonic()-began, artifact_sha256=p.sha(artifact))
            for mapping in p.mappings(capacity, rows):
                slots = torch.tensor(mapping)
                expected = store_slots(cache, updates, slots)
                actual = compiled(cache, updates, slots)
                check = dict(slots=mapping, bitwise_equal=torch_same(actual, expected),
                    cache_input_unchanged=torch_same(cache, before_cache),
                    updates_input_unchanged=torch_same(updates, before_updates),
                    slots_input_unchanged=slots.tolist()==mapping,
                    actual_sha256=tensor_sha(actual), expected_sha256=tensor_sha(expected))
                result['checks'].append(check)
                p.save(directory / 'result.json', result)
                assert all(check[x] for x in ('bitwise_equal', 'cache_input_unchanged',
                    'updates_input_unchanged', 'slots_input_unchanged')), (spec, mapping)
            result.update(complete=True, passed=True)
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        result['seconds'] = time.monotonic()-began
        p.save(directory / 'result.json', result)


def alias(context,index):
    phase='extended_alias'
    require_context(context,phase)
    proof=context.prerequisite('extended_device')
    assert proof['complete'] and proof['passed']
    p=BoundMatrix(context,phase)
    assert type(index) is int and 0<=index<len(p.alias_specs())
    import torch
    import torch_neuronx
    store_slots_nki_predicated=context.writer(phase)
    from lfm3_nki05_device_oracle_v1 import store_slots
    torch.set_num_threads(1)
    spec = p.alias_specs()[index]
    # A reference/candidate pair receives the same initial state and call tensors.
    torch.manual_seed(20261103 + index // 2)
    capacity, length = spec['capacity'], spec['length']
    writer = store_slots if spec['variant'] == 'reference' else store_slots_nki_predicated
    conv = spec['kind'] == 'conv'
    directory = p.DEST / spec['name']
    directory.mkdir()

    class Attention(torch.nn.Module):
        def __init__(self, k, v, write):
            super().__init__()
            self.kc = torch.nn.Parameter(k.clone(), requires_grad=False)
            self.vc = torch.nn.Parameter(v.clone(), requires_grad=False)
            self.write = write
        def forward(self, q, k, v, positions, slots):
            ix = positions[:, None, :, None].expand(capacity, 2, 1, 128)
            sk = torch.scatter(self.kc[slots.clamp_min(0)], 2, ix, k)
            sv = torch.scatter(self.vc[slots.clamp_min(0)], 2, ix, v)
            scores = (q @ sk.repeat_interleave(4, dim=1).transpose(-1,-2)) * (128 ** -.5)
            allowed = torch.arange(length, device=positions.device)[None,None,None] <= positions[:,None,:,None]
            scores = scores + torch.where(allowed, torch.zeros_like(scores),
                torch.full_like(scores, torch.finfo(scores.dtype).min))
            attended = scores.softmax(-1, dtype=torch.float32).to(q.dtype) @ sv.repeat_interleave(4, dim=1)
            return attended, self.write(self.kc, sk, slots), self.write(self.vc, sv, slots)

    class ConvolutionState(torch.nn.Module):
        def __init__(self, state, write):
            super().__init__()
            self.cache = torch.nn.Parameter(state.clone(), requires_grad=False)
            self.write = write
        def forward(self, value, slots):
            shifted = torch.cat((self.cache[slots.clamp_min(0)][:,:,1:], value), dim=2)
            return shifted, self.write(self.cache, shifted, slots)

    mappings = p.alias_mappings(capacity)
    if conv:
        initial = [torch.randn(capacity,1024,7,dtype=torch.bfloat16)*.1]
        source = ConvolutionState(initial[0], writer).eval()
        eager = ConvolutionState(initial[0], store_slots).eval()
        samples = [(torch.randn(capacity,1024,1,dtype=torch.bfloat16)*.1, torch.tensor(slots)) for slots in mappings]
        parameters = [source.cache]
    else:
        initial = [torch.randn(capacity,2,length,128,dtype=torch.bfloat16)*.1 for _ in range(2)]
        source = Attention(*initial, writer).eval()
        eager = Attention(*initial, store_slots).eval()
        samples = [(torch.randn(capacity,8,1,128,dtype=torch.bfloat16)*.1,
                    torch.randn(capacity,2,1,128,dtype=torch.bfloat16)*.1,
                    torch.randn(capacity,2,1,128,dtype=torch.bfloat16)*.1,
                    torch.tensor([[(11 + 7*i + step) % length] for i in range(capacity)]),
                    torch.tensor(slots)) for step, slots in enumerate(mappings)]
        parameters = [source.kc, source.vc]
    result = dict(complete=False, passed=False, spec=spec, checks=[],
        initial_sha256=[tensor_sha(x) for x in initial],
        inputs_sha256=[[tensor_sha(x) for x in sample] for sample in samples],
        attention_rounding_is_diagnostic_only=True)
    began = time.monotonic()
    try:
        with torch.inference_mode():
            compiled = torch_neuronx.trace(source, samples[0],
                input_output_aliases={value:i+1 for i,value in enumerate(parameters)},
                compiler_workdir=str(directory/'compiler'),
                compiler_args=['--target=trn1','--auto-cast=none','--model-type=transformer',
                    '-O2','--lnc=1',
                    '--tensorizer-options=--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=1 --vectorize-strided-dma'])
            artifact = directory/'model.pt'
            compiled.save(str(artifact))
            result['artifact_identity']=context.artifact(artifact,phase,dict(spec,dtype='bfloat16',rows=capacity,shapes=[list(value.shape) for value in initial]))
            result.update(compile_seconds=time.monotonic()-began, artifact_sha256=p.sha(artifact))
            states = dict(compiled.states._parameters)
            names = [[name for name,value in states.items() if torch_same(value,expected)] for expected in initial]
            assert all(len(x)==1 for x in names) and len({x[0] for x in names})==len(initial)
            names = [x[0] for x in names]
            del states
            torch_neuronx.move_trace_to_device(compiled,0)
            assert all(value.device.type=='privateuseone' for value in compiled.states._parameters.values())
            bindings = {name:id(value) for name,value in compiled.states._parameters.items()}
            result['state_names']=names
            for step,sample in enumerate(samples):
                expected = eager(*sample)
                actual = tuple(x.detach().cpu().clone() for x in compiled(*sample))
                states = dict(compiled.states._parameters)
                check = dict(step=step,slots=mappings[step],output_finite=bool(torch.isfinite(actual[0]).all()),
                    cache_outputs_bitwise=all(torch_same(a,b) for a,b in zip(actual[1:],expected[1:])),
                    persistent_states_bitwise=all(torch_same(states[name],value) for name,value in zip(names,expected[1:])),
                    bindings_unchanged={name:id(value) for name,value in states.items()}==bindings,
                    inputs_unchanged=[tensor_sha(x) for x in sample]==result['inputs_sha256'][step],
                    output_bitwise=torch_same(actual[0],expected[0]),
                    output_max_abs_error=float((actual[0].float()-expected[0].float()).abs().max()),
                    output_sha256=tensor_sha(actual[0]), state_sha256=[tensor_sha(x) for x in actual[1:]])
                result['checks'].append(check)
                p.save(directory/'result.json',result)
                assert all(check[x] for x in ('output_finite','cache_outputs_bitwise',
                    'persistent_states_bitwise','bindings_unchanged','inputs_unchanged'))
                if conv:assert check['output_bitwise']
                for param,value in zip([eager.cache] if conv else [eager.kc,eager.vc],expected[1:]):
                    param.copy_(value)
            result.update(complete=True,passed=True)
            del states,compiled
            gc.collect()
    except BaseException as error:
        result['error']=repr(error)
        raise
    finally:
        result['seconds']=time.monotonic()-began
        p.save(directory/'result.json',result)

