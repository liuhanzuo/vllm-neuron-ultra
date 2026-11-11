"""Bounded injected-API probe for the frozen Dory tensor adapter.

Source preparation only: no native import, disk payload, model, admission or
launch at import/CLI. Future caller must authenticate the actual CPU Torch API,
own original serial admission and supply a resource/identity guard. Tiny
constructed inputs diagnose ABI/layout serialization only, never model quality.
"""
import hashlib
import itertools
import math
import struct

from dory_stage_a_tensor_adapter_v1 import NativeTensorAdapter, require

MAX_BACKING_BYTES = 2 * 1024**2
MAX_EXPECTED_BYTES = 1024**2
SCHEMA = 'dory_stage_a_native_adapter_probe_v2'


def cases():
    # In particular, size-one dimensions can retain noncanonical strides while
    # being logically contiguous; do not assume clone normalizes them.
    rows = [
        ('vector_offset', [13], [1], 3, None),
        ('padded_rows', [3, 7], [13, 1], 2, None),
        ('transposed', [5, 3], [1, 7], 1, None),
        ('singleton_leading', [1, 7], [0, 1], 3, None),
        ('singleton_trailing', [5, 1], [2, 17], 2, None),
        ('singleton_middle', [3, 1, 5], [17, 0, 2], 1, None),
        ('all_singletons', [1, 1, 1], [0, 9, 4], 2, None),
        ('broadcast_rows', [3, 7], [0, 1], 2, None),
        ('selected_rows', [7, 9], [21, 2], 3, [0, 3, 6]),
        ('chunk_boundary', [32769], [1], 1, None),
        ('recursive_columns', [2, 17003], [34009, 2], 3, None),
    ]
    result = [dict(name=name+'_'+dtype, shape=shape, stride=stride, offset=offset,
                   rows=selected, dtype=dtype)
              for name,shape,stride,offset,selected in rows for dtype in ('BF16','F32')]
    result += [dict(name='full_vocab_f32',shape=[1,131072],stride=[131072,1],offset=1,rows=None,dtype='F32')]
    return result


def backing(case):
    shape,stride,offset = case['shape'],case['stride'],case['offset']
    require(1 <= len(shape) <= 3 and len(shape)==len(stride)
            and all(type(v)is int and v>0 for v in shape)
            and all(type(v)is int and v>=0 for v in stride)
            and type(offset)is int and offset>=0, 'Exact bounded probe geometry')
    width = {'BF16':2,'F32':4}[case['dtype']]
    extent = offset+sum((n-1)*step for n,step in zip(shape,stride))+1
    require(extent*width<=MAX_BACKING_BYTES and math.prod(shape)*width<=MAX_EXPECTED_BYTES,
            'Bound before allocating any probe buffer')
    # Exact word patterns include signed zero, subnormal, finite extremes,
    # infinities and NaN payloads. Serialization must not perform FP arithmetic.
    words = (0,0x8000,1,0x3f80,0xbf80,0x7f7f,0x7f80,0x7fc1) if width==2 else (
             0,0x80000000,1,0x3f800000,0xbf800000,0x7f7fffff,0x7f800000,0x7fc12345)
    out=bytearray(extent*width)
    for i in range(extent):
        value=words[i] if i<len(words) else ((i*40503+0x1234)&0xffff if width==2 else (i*2654435761+0x12345678)&0xffffffff)
        struct.pack_into('<H' if width==2 else '<I',out,i*width,value)
    return out


def expected_bytes(case, raw):
    width = {'BF16':2,'F32':4}[case['dtype']]
    shape=case['shape'];first=range(shape[0]) if case['rows'] is None else case['rows']
    result=bytearray()
    for coord in itertools.product(first,*(range(n)for n in shape[1:])):
        index=case['offset']+sum(i*s for i,s in zip(coord,case['stride']))
        result.extend(raw[index*width:(index+1)*width])
    require(len(result)<=MAX_EXPECTED_BYTES,'Bounded independently indexed oracle')
    return bytes(result)


def make_tensor(torch, case, raw):
    dtype=torch.bfloat16 if case['dtype']=='BF16'else torch.float32
    flat=torch.frombuffer(raw,dtype=dtype)
    return torch.as_strided(flat,size=tuple(case['shape']),stride=tuple(case['stride']),
                            storage_offset=case['offset'])


def run_probe(torch, *, guard):
    require(callable(guard), 'Explicit actual serial/resource guard')
    original_guard = guard
    class GuardAbort(BaseException):
        pass
    def guard():
        try:
            original_guard()
        except BaseException as error:
            raise GuardAbort('Actual guard rejected probe') from error
    guard()
    require(str(torch.__version__)=='2.14.0+cpu', 'Original Dory CPU Torch version')
    require(callable(torch.frombuffer)and callable(torch.as_strided), 'Actual explicit view creation API')
    adapter=NativeTensorAdapter(torch,guard=guard)
    records=[]
    for case in cases():
        guard();raw=backing(case);original=bytes(raw);expected=expected_bytes(case,original)
        row=dict(case=case,backing_bytes=len(raw),expected_bytes=len(expected),passed=False)
        try:
            tensor=make_tensor(torch,case,raw);description=adapter.describe(tensor)
            require(description==dict(shape=case['shape'],dtype=case['dtype'],
                    stride=case['stride'],storage_offset=case['offset']), 'Exact actual created view')
            row['source_layout']=description
            if math.prod(case['shape'])*(2 if case['dtype']=='BF16'else 4)<=adapter.MAX_CHUNK_BYTES:
                clone=tensor.detach().clone(memory_format=torch.contiguous_format)
                row['actual_small_clone_layout']=adapter.describe(clone)
                row['small_clone_distinct_storage']=clone.data_ptr()!=tensor.data_ptr()
            parts=[];consumed=0
            for chunk in adapter.iter_bytes(tensor,case['rows']):
                guard();require(type(chunk)is bytes and 0<len(chunk)<=adapter.MAX_CHUNK_BYTES,
                                'Real bounded adapter bytes')
                consumed+=len(chunk)
                require(consumed<=len(expected), 'No oversized or repeated probe output')
                parts.append(chunk)
            actual=b''.join(parts)
            require(actual==expected, 'Every logical BF16/F32 byte equals independent storage oracle')
            require(bytes(raw)==original, 'Original probe backing remains bitwise unchanged')
            row.update(passed=True,chunk_count=len(parts),largest_chunk_bytes=max(map(len,parts)),
                       sha256=hashlib.sha256(actual).hexdigest(),full_bytes_compared=True)
        except Exception as error:
            row.update(error_type=type(error).__name__,error=str(error)[:512],
                       original_backing_unchanged=bytes(raw)==original)
        records.append(row);guard()
    auxiliary=[]
    for label,operation in [('integer_tokens',lambda: adapter.integers(torch.tensor(
            [0,1,131071,-1,2**53+1],dtype=torch.int64,device='cpu'))==[0,1,131071,-1,2**53+1]),
            ('full_row_select',lambda:_selection(torch,adapter)),
            ('first_index_ties',lambda:_argmax(torch,adapter))]:
        guard()
        try:
            require(operation(),'Exact auxiliary ABI result');auxiliary.append(dict(name=label,passed=True))
        except Exception as error:
            auxiliary.append(dict(name=label,passed=False,error_type=type(error).__name__,error=str(error)[:512]))
    guard()
    return dict(schema=SCHEMA,complete=True,passed=all(x['passed']for x in records+auxiliary),
        serialization_cases=records,auxiliary_cases=auxiliary,case_count=len(records),
        max_backing_bytes=MAX_BACKING_BYTES,max_expected_bytes=MAX_EXPECTED_BYTES,
        caller_supplied_torch_version=str(torch.__version__),
        native_source_provenance_authenticated_here=False,actual_process_exit_observed=False,
        runtime_admitted_here=False,model_or_original_weight_or_raw_payload_read=False,
        production_singleton_layout_qualified=False,full_model_qualified=False,new_speedup_claim=False)


def _selection(torch,adapter):
    case=dict(shape=[5,7],stride=[17,2],offset=3,rows=[0,2,4],dtype='BF16')
    raw=backing(case);original=bytes(raw);expected=expected_bytes(case,original)
    tensor=make_tensor(torch,case,raw);selected=adapter.select(tensor,case['rows'])
    return b''.join(adapter.iter_bytes(selected))==expected and bytes(raw)==original


def _argmax(torch,adapter):
    # Finite exact FP32 integers, ties must choose the first index.
    raw=bytearray(struct.pack('<12f',1,3,3,-1,-2,-2,-3,-4,0,0,0,0))
    original=bytes(raw)
    flat=torch.frombuffer(raw,dtype=torch.float32)
    value=torch.as_strided(flat,size=(3,4),stride=(4,1),storage_offset=0)
    return adapter.top1(value)==[1,0,0] and bytes(raw)==original


if __name__=='__main__':
    raise RuntimeError('Source only; authenticated native API, actual serial admission and parent exit/audit required')
