"""Additional real-device gates; no launch context or native work on import.

FP32 persistent alias is separate from the old FP32 standalone SSM test.
Wrapper controls exercise huge signed int64 selectors and production one-row
original dispatch. These cases supplement, never replace, original matrices.
"""
import gc
import time
from lfm3_nki05_device_context_v1 import require_context
from lfm3_nki05_typed_dispatch_v1 import make_writer
from lfm3_nki05_device_oracle_v1 import store_slots
from lfm3_nki05_extended_device_leaf_v2 import torch_same,tensor_sha
from lfm3_typed_slot_writer_contract_v1 import bit_patterns


def control_mappings(capacity,rows):
    assert capacity in (4,8,16) and 1<=rows<=capacity
    choices=[list(range(rows)),list(reversed(range(capacity)))[:rows],
        [capacity-1]*rows,[-1]*rows]
    for invalid in (-9,capacity,capacity+9,-(2**31),2**31-1,
                    -(2**32),2**32,2**32+capacity-1,-(2**63),2**63-1):
        choices.append([invalid]*rows)
        if rows>1:
            choices.extend([[invalid]+[capacity-1]*(rows-1),[0]*(rows-1)+[invalid]])
    return [row for i,row in enumerate(choices) if row not in choices[:i]]


def wrapper_specs():
    return [dict(capacity=b,rows=rows,dtype=dtype,shape=[b,256],
                 name=f'b{b}-r{rows}-{dtype}',variant='production_dispatch')
            for b in (4,8,16) for rows in sorted({1,4,b})
            for dtype in ('bfloat16','float32')]


def fp32_alias_specs():
    return [dict(capacity=b,rows=b,dtype='float32',shape=[b,64,80,128],
                 name=f'b{b}-fp32-ssm-{variant}',variant=variant)
            for b in (4,8,16) for variant in ('reference','candidate')]


def raw_fixture(shape,dtype_name,reverse=False):
    import torch
    width=1
    for size in shape[1:]:width*=size
    words=list(bit_patterns(dtype_name))
    if reverse:words=words[::-1]
    raw=torch.tensor(words,dtype=torch.uint16 if dtype_name=='bfloat16' else torch.uint32)
    raw=raw.repeat((width+len(words)-1)//len(words))[:width]
    # Distinct row patterns are essential: permutations and duplicate winners
    # would be invisible with identical payloads in every slot.
    raw=torch.stack([raw.roll(row) for row in range(shape[0])])
    return raw.view(torch.bfloat16 if dtype_name=='bfloat16' else torch.float32).reshape(shape)


def wrapper_controls(context,index):
    phase='wrapper_controls'
    ROOT,DEST=require_context(context,phase)
    proof=context.prerequisite('fp32_alias')
    assert proof['complete'] and proof['passed']
    specs=wrapper_specs()
    assert type(index) is int and 0<=index<len(specs)
    spec=specs[index]
    import torch
    import torch_neuronx
    torch.set_num_threads(1)
    b,rows=spec['capacity'],spec['rows']
    typed=context.writer(phase)
    branches=dict(original=0,typed=0)
    def original(cache,updates,slots):
        branches['original']+=1
        return store_slots(cache,updates,slots)
    def candidate(cache,updates,slots):
        branches['typed']+=1
        return typed(cache,updates,slots)
    write=make_writer(b,original,candidate)
    class Write(torch.nn.Module):
        def forward(self,cache,updates,slots):return write(cache,updates,slots)
    cache=raw_fixture(spec['shape'],spec['dtype'])
    updates=raw_fixture([rows,256],spec['dtype'],True)
    initial=[tensor_sha(x) for x in (cache,updates)]
    directory=DEST/spec['name'];directory.mkdir()
    result=dict(complete=False,passed=False,spec=spec,checks=[],
                full_model_qualified=False,new_speedup_claim=False)
    started=time.monotonic()
    try:
        with torch.inference_mode():
            compiled=torch_neuronx.trace(Write().eval(),(cache,updates,torch.arange(rows,dtype=torch.int64)),
                compiler_workdir=str(directory/'compiler'),
                compiler_args=['--target=trn1','--auto-cast=none','-O2'])
            expected_branch='original' if rows==1 else 'typed'
            assert branches[expected_branch]>0 and branches['typed' if rows==1 else 'original']==0
            artifact=directory/'model.pt';compiled.save(str(artifact))
            result['artifact_identity']=context.artifact(artifact,phase,spec)
            result['trace_dispatch_counts']=dict(branches)
            for mapping in control_mappings(b,rows):
                slots=torch.tensor(mapping,dtype=torch.int64)
                expected=store_slots(cache,updates,slots)
                actual=compiled(cache,updates,slots)
                check=dict(slots=mapping,bitwise_equal=torch_same(actual,expected),
                    dtype_shape_preserved=actual.dtype==cache.dtype and actual.shape==cache.shape,
                    inputs_unchanged=initial==[tensor_sha(x) for x in (cache,updates)] and slots.tolist()==mapping)
                result['checks'].append(check);context.save(directory/'result.json',result,phase)
                assert all(check[k] for k in ('bitwise_equal','dtype_shape_preserved','inputs_unchanged'))
            result.update(complete=True,passed=True)
            del compiled;gc.collect()
    except BaseException as error:
        result['error']=repr(error);raise
    finally:
        result['seconds']=time.monotonic()-started
        context.save(directory/'result.json',result,phase)


def fp32_alias(context,index):
    phase='fp32_alias'
    ROOT,DEST=require_context(context,phase)
    proof=context.prerequisite('extended_alias')
    assert proof['complete'] and proof['passed']
    specs=fp32_alias_specs()
    assert type(index) is int and 0<=index<len(specs)
    spec=specs[index]
    import torch
    import torch_neuronx
    torch.set_num_threads(1)
    typed=context.writer(phase)
    writer=store_slots if spec['variant']=='reference' else typed
    class State(torch.nn.Module):
        def __init__(self,state,write):
            super().__init__()
            self.cache=torch.nn.Parameter(state.clone(),requires_grad=False)
            self.write=write
        def forward(self,updates,slots):
            return self.cache.clone(),self.write(self.cache,updates,slots)
    b=spec['capacity']
    initial=raw_fixture(spec['shape'],'float32')
    updates=raw_fixture(spec['shape'],'float32',True)
    source=State(initial,writer).eval();eager=State(initial,store_slots).eval()
    # Full refill, permutation, inactive, duplicate last-writer, out-of-range,
    # huge int64 normalization and reuse; actual state persists between calls.
    mappings=control_mappings(b,b)+[list(range(b)),[-1]*b,list(reversed(range(b)))]
    directory=DEST/spec['name'];directory.mkdir()
    result=dict(complete=False,passed=False,spec=spec,checks=[],
                full_model_qualified=False,new_speedup_claim=False)
    started=time.monotonic()
    try:
        with torch.inference_mode():
            compiled=torch_neuronx.trace(source,(updates,torch.arange(b,dtype=torch.int64)),
                input_output_aliases={source.cache:1},compiler_workdir=str(directory/'compiler'),
                compiler_args=['--target=trn1','--auto-cast=none','--model-type=transformer',
                    '-O2','--lnc=1',
                    '--tensorizer-options=--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=1 --vectorize-strided-dma'])
            artifact=directory/'model.pt';compiled.save(str(artifact))
            result['artifact_identity']=context.artifact(artifact,phase,spec)
            names=[name for name,value in compiled.states._parameters.items() if torch_same(value,initial)]
            assert len(names)==1
            torch_neuronx.move_trace_to_device(compiled,0)
            assert all(value.device.type=='privateuseone' for value in compiled.states._parameters.values())
            bindings={name:id(value) for name,value in compiled.states._parameters.items()}
            for step,mapping in enumerate(mappings):
                # Vary writes after refill to make lost state and reuse observable.
                value=updates if step%2==0 else initial
                slots=torch.tensor(mapping,dtype=torch.int64)
                before=[tensor_sha(x) for x in (value,slots)]
                expected=eager(value,slots)
                actual=tuple(x.detach().cpu().clone() for x in compiled(value,slots))
                states=dict(compiled.states._parameters)
                check=dict(step=step,slots=mapping,prior_state_bitwise=torch_same(actual[0],expected[0]),
                    cache_output_bitwise=torch_same(actual[1],expected[1]),
                    persistent_state_bitwise=torch_same(states[names[0]],expected[1]),
                    bindings_unchanged={name:id(x) for name,x in states.items()}==bindings,
                    inputs_unchanged=before==[tensor_sha(x) for x in (value,slots)])
                result['checks'].append(check);context.save(directory/'result.json',result,phase)
                assert all(check[k] for k in ('prior_state_bitwise','cache_output_bitwise',
                    'persistent_state_bitwise','bindings_unchanged','inputs_unchanged'))
                eager.cache.copy_(expected[1])
            result.update(complete=True,passed=True)
            del states,compiled;gc.collect()
    except BaseException as error:
        result['error']=repr(error);raise
    finally:
        result['seconds']=time.monotonic()-started
        context.save(directory/'result.json',result,phase)
