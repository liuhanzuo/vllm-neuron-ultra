"""Original independent bitwise primitive oracles; lazily imported native libraries."""
import hashlib
import math
import json
from pathlib import Path
MAPPINGS=[[0,1,2,3],[3,2,1,0],[3,-1,-1,-1],[-1,-1,-1,-1],[2,2,0,3],[-2147483648,3,-9,0],[0,0,0,0],[3,3,3,3]]
from nemotron9b_package_protocol_v18 import stream_sha

def mappings(rows):
    return MAPPINGS if rows == 4 else [[slot] for slot in [0, 1, 2, 3, -1, -9, -2147483648]]

def array_for(dtype_name, shape, raw, salt=0):
    import numpy as np
    import ml_dtypes
    dtype = ml_dtypes.bfloat16 if dtype_name == 'bfloat16' else np.float32
    count = math.prod(shape)
    if not raw:
        result = ((np.arange(count, dtype=np.int32) + salt) % 257 - 128).astype(dtype).reshape(shape)
        storage = result.view(np.uint16 if dtype_name == 'bfloat16' else np.uint32).reshape(-1)
        storage[:2] = [0, 0x8000 if dtype_name == 'bfloat16' else 0x80000000]
        return result
    if dtype_name == 'bfloat16':
        indices = np.arange(count, dtype=np.uint32).reshape(shape[0], -1)
        # Distinguish slots even when each row contains a full uint16 cycle.
        indices = indices + np.arange(shape[0], dtype=np.uint32)[:, None] * 101
        words = ((indices + salt * 37) % 65536).astype(np.uint16).reshape(-1)
        specials = [0, 0x8000, 0x7f80, 0xff80, 0x7fc1, 0x7fa1, 1, 0x8001]
    else:
        words = np.arange(count, dtype=np.uint32) * np.uint32(2654435761) + np.uint32(salt * 37)
        specials = [0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc00001, 0x7fa00001, 1, 0x80000001]
    # Keep each BF16 slot's full 65536-word cycle intact; specials are already
    # present there. Short shapes and FP32 get explicit exceptional values.
    if dtype_name == 'float32' or math.prod(shape[1:]) < 65536:
        words.reshape(shape[0], -1)[:, :8] = specials
    result = words.view(dtype).reshape(shape)
    if dtype_name == 'bfloat16' and math.prod(shape[1:]) >= 65536:
        assert all(np.unique(row.view(np.uint16)).size == 65536 for row in result)
    return result

def select(cache, slots):
    import numpy as np
    assert all(-2147483648 <= slot < 4 for slot in slots)
    return np.stack([cache[max(slot, 0)].copy() for slot in slots])

def compare(actual, expected):
    # Inputs are numpy arrays; integer views avoid NaN comparisons/conversions.
    import numpy as np
    same_shape = actual.shape == expected.shape
    same_dtype = actual.dtype == expected.dtype
    a, b = actual.tobytes(), expected.tobytes()
    row = {'passed': same_shape and same_dtype and a == b, 'shape_equal': same_shape,
           'dtype_equal': same_dtype, 'bitwise_equal': a == b,
           'actual_sha256': hashlib.sha256(a).hexdigest(),
           'expected_sha256': hashlib.sha256(b).hexdigest(), 'bytes_compared': len(b)}
    if a != b and len(a) == len(b):
        differences = np.flatnonzero(np.frombuffer(a, dtype=np.uint8) != np.frombuffer(b, dtype=np.uint8))
        row.update(different_bytes=int(differences.size), first_different_bytes=differences[:16].tolist())
    return row

def classify(expected,actual):
    """Classify raw storage words; never convert NaNs through floating point."""
    import numpy as np
    assert expected.shape==actual.shape and expected.dtype==actual.dtype
    bits=expected.dtype.itemsize*8;assert bits in (16,32)
    dtype=np.uint16 if bits==16 else np.uint32
    e=expected.view(dtype).reshape(-1);a=actual.view(dtype).reshape(-1)
    different=e!=a;indices=np.flatnonzero(different)
    sign=1<<(bits-1);exponent=0x7f80 if bits==16 else 0x7f800000;mantissa=0x7f if bits==16 else 0x7fffff
    labels=['zero','subnormal','normal','infinity','nan']
    def kinds(values):
        exp=values&exponent;frac=values&mantissa
        return np.where(exp==0,np.where(frac==0,0,1),np.where(exp==exponent,np.where(frac==0,3,4),2))
    ek,ak=kinds(e[indices]),kinds(a[indices])
    codes,counts=np.unique(ek*5+ak,return_counts=True)
    transitions={labels[int(code)//5]+'->'+labels[int(code)%5]:int(count) for code,count in zip(codes,counts)}
    return dict(different_words=int(indices.size),transitions=transitions,
                negative_zero_to_positive=int(np.count_nonzero((e==sign)&(a==0))),
                nan_sign_changes=int(np.count_nonzero((ek==4)&(ak==4)&(((e[indices]^a[indices])&sign)!=0))),
                samples=[dict(flat_index=int(i),expected_hex=f'{int(e[i]):0{bits//4}x}',actual_hex=f'{int(a[i]):0{bits//4}x}')
                         for i in indices[:24]])

def classifier_controls():
    import numpy as np
    import ml_dtypes
    result=[]
    for word,dtype,values in [(np.uint16,ml_dtypes.bfloat16,[0x8000,0x7fc1,0x3f80,1]),
                              (np.uint32,np.float32,[0x80000000,0x7fc00001,0x3f800000,1])]:
        expected=np.asarray(values,dtype=word).view(dtype)
        actual=np.asarray([0,values[1]-1,values[2]+1,0],dtype=word).view(dtype)
        info=classify(expected,actual)
        assert info['different_words']==4 and info['negative_zero_to_positive']==1
        assert info['transitions']=={'zero->zero':1,'subnormal->zero':1,'normal->normal':1,'nan->nan':1}
        assert classify(expected,expected)['different_words']==0
        result.append(dict(bits=expected.dtype.itemsize*8,passed=True))
    return result

def output_shapes(case):
    b, t = case['batch'], case['length']
    if case['kind'] == 'nemotron_projection':
        return [[b, t, 5120], [b, 6144, t], [b, t, 64]]
    if case['kind'] == 'nemotron_conv':
        return [[b, t, 64, 80], [b, t, 4, 128], [b, t, 4, 128]]
    return [[b, t, 8, 128] for _ in range(3)]

def address_samples(case):
    offset = 0
    for output, width in enumerate(case['sizes']):
        for batch in range(case['batch']):
            for token in range(case['length']):
                for channel in sorted({0, width // 2, width - 1}):
                    if case['kind'] == 'nemotron_projection':
                        coordinates = [batch, channel, token] if output == 1 else [batch, token, channel]
                        expected = (batch * case['length'] + token) * sum(case['sizes']) + offset + channel
                    else:
                        head_dim = 80 if case['kind'] == 'nemotron_conv' and output == 0 else 128
                        coordinates = [batch, token, channel // head_dim, channel % head_dim]
                        expected = (batch * sum(case['sizes']) + offset + channel) * case['length'] + token
                    yield output, coordinates, expected
        offset += width

def ravel(indices, dimensions):
    result = 0
    assert len(indices) == len(dimensions)
    for index, dimension in zip(indices, dimensions):
        assert 0 <= index < dimension
        result = result * dimension + index
    return result

def unravel(index, dimensions):
    result = []
    for dimension in reversed(dimensions):
        result.append(index % dimension)
        index //= dimension
    assert index == 0
    return list(reversed(result))

def inspect_hlo(path, case):
    from torch_neuronx.pyhlo import hlo_pb2
    module = hlo_pb2.HloModuleProto()
    module.ParseFromString(Path(path).read_bytes())
    entry = next(c for c in module.computations if c.id == module.entry_computation_id)
    nodes = {i.id: i for i in entry.instructions}
    root = nodes[entry.root_id]
    assert root.opcode == 'tuple' and len(root.operand_ids) == 3
    outputs = [nodes[n] for n in root.operand_ids]
    assert [list(n.shape.dimensions) for n in outputs] == output_shapes(case)
    parameters = [n for n in nodes.values() if n.opcode == 'parameter']
    assert len(parameters) == 1 and list(parameters[0].shape.dimensions) == case['input_shape']

    def source_index(node, coordinates):
        if node.opcode == 'parameter':
            assert node.id == parameters[0].id
            return ravel(coordinates, node.shape.dimensions)
        assert len(node.operand_ids) == 1, (node.name, node.opcode)
        child = nodes[node.operand_ids[0]]
        if node.opcode in ('convert', 'copy'):
            assert list(node.shape.dimensions) == list(child.shape.dimensions)
            previous = coordinates
        elif node.opcode == 'reshape':
            previous = unravel(ravel(coordinates, node.shape.dimensions), child.shape.dimensions)
        elif node.opcode == 'transpose':
            previous = [0] * len(coordinates)
            for output_axis, input_axis in enumerate(node.dimensions):
                previous[input_axis] = coordinates[output_axis]
        elif node.opcode == 'slice':
            assert len(node.slice_dimensions) == len(coordinates)
            previous = [s.start + value * s.stride for s, value in zip(node.slice_dimensions, coordinates)]
        else:
            raise AssertionError(('Unrecognized address operation; do not infer correctness', node.name, node.opcode))
        return source_index(child, previous)

    count, mismatches, examples = 0, 0, []
    for output, coordinates, expected in address_samples(case):
        actual = source_index(outputs[output], coordinates)
        count += 1
        mismatches += actual != expected
        if actual != expected and len(examples) < 12:
            examples.append(dict(output=output, coordinates=coordinates, actual=actual, expected=expected))
    return dict(hlo=str(path), sha256=stream_sha(path), sampled_addresses=count,
                mismatching_sampled_addresses=mismatches, first_mismatches=examples,
                sampled_mapping_passed=mismatches == 0,
                scope='Every logical batch/token row; first, middle and last channel per split. '
                      'This sampled structural check is paired with complete output byte comparisons.')
