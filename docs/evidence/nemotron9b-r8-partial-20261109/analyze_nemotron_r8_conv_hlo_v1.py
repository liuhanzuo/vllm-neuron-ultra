"""Read saved small HLO protobufs; independently trace index mappings, no SDK execution."""
import ast
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys

R=Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
Q=R/'research'
CONTROL=R/'nemotron9b-package-r8-20261109-control'
JOBS=Path('/mnt/nvme/nemotron9b-package-r8-20261109/primitives/jobs')
SITE=Path('/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages')
DESCRIPTORS={
 'torch_neuronx/pyhlo/xla_data_pb2.py':'642ae1b5dc64a3188401d5f40dac475e0e4e20878b6473dda8e0e3306d8ba1f3',
 'torch_neuronx/pyhlo/service/hlo_pb2.py':'8d332b4180bc80e2b376a3018de0013920e6878b3eb260c52540ef07bc5b0922'}


def small(path,maximum=2**20):
    path=Path(path);before=path.stat();assert before.st_size<=maximum
    raw=path.read_bytes();after=path.stat()
    assert (before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns)
    return raw,dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def flatten(index,shape):
    assert len(index)==len(shape)
    value=0
    for i,n in zip(index,shape):assert 0<=i<n;value=value*n+i
    return value


def unflatten(value,shape):
    assert 0<=value<math.prod(shape)
    result=[]
    for size in reversed(shape):result.append(value%size);value//=size
    assert value==0
    return list(reversed(result))


def source_offset(nodes,node_id,index):
    node=nodes[node_id];shape=list(node.shape.dimensions);flatten(index,shape)
    if node.opcode=='parameter':assert node.parameter_number==0;return flatten(index,shape)
    assert len(node.operand_ids)==1
    parent=nodes[node.operand_ids[0]];parent_shape=list(parent.shape.dimensions)
    if node.opcode=='convert':new=index
    elif node.opcode=='reshape':new=unflatten(flatten(index,shape),parent_shape)
    elif node.opcode=='slice':new=[s.start+i*s.stride for s,i in zip(node.slice_dimensions,index)]
    elif node.opcode=='transpose':
        new=[None]*len(index)
        for output_axis,input_axis in enumerate(node.dimensions):new[input_axis]=index[output_axis]
    else:raise ValueError('Unsupported address operation: '+node.opcode)
    return source_offset(nodes,parent.id,new)


def main():
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python';sys.path.insert(0,str(SITE))
    from google.protobuf import descriptor_pb2,descriptor_pool,message_factory
    from google.protobuf.internal import api_implementation
    assert api_implementation.Type()=='python'
    pool=descriptor_pool.DescriptorPool();descriptor_refs=[]
    for name,expected in DESCRIPTORS.items():
        raw,ref=small(SITE/name);assert ref['sha256']==expected;descriptor_refs.append(ref)
        tree=ast.parse(raw);values=[ast.literal_eval(n.args[0])for n in ast.walk(tree)
            if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='AddSerializedFile']
        assert len(values)==1
        d=descriptor_pb2.FileDescriptorProto();d.ParseFromString(values[0]);pool.AddSerializedFile(values[0])
    message=message_factory.GetMessageClass(pool.FindMessageTypeByName(d.package+'.HloModuleProto'))
    rows=[]
    for length in (128,256,512):
        for variant in ('original','tensor-split'):
            name=f'split-nemotron-conv-b1-t{length}-{variant}'
            raw,result_ref=small(CONTROL/(name+'-worker-result.json'),8*2**20);outer=json.loads(raw);result=outer['result']
            assert outer['complete']is True and outer['passed']is True and result['compile_passed']is True
            assert result['job']['case']['input_shape']==[1,6144,length]
            assert result['device_exact']is (variant=='tensor-split')
            _,exit_ref=small(CONTROL/(name+'-retained-exit.json'))
            _,drain_ref=small(CONTROL/(name+'-retained-drain.json'))
            raw,hlo_ref=small(JOBS/name/'compiler-work/model/graph.hlo',16384)
            assert hlo_ref['sha256']==result['hlo']['sha256']
            module=message();module.ParseFromString(raw)
            entry=next(c for c in module.computations if c.id==module.entry_computation_id)
            nodes={n.id:n for n in entry.instructions};root=nodes[entry.root_id]
            assert root.opcode=='tuple'and len(root.operand_ids)==3
            sampled=0;mismatches=0;first=[];offset=0;examples=[]
            for output,(width,heads,head_dim) in enumerate(((5120,64,80),(512,4,128),(512,4,128))):
                node_id=root.operand_ids[output];assert list(nodes[node_id].shape.dimensions)==[1,length,heads,head_dim]
                for token in range(length):
                    for channel in (0,width//2,width-1):
                        index=[0,token,channel//head_dim,channel%head_dim]
                        actual=source_offset(nodes,node_id,index);expected=(offset+channel)*length+token
                        sampled+=1
                        if actual!=expected:
                            mismatches+=1
                            if len(first)<6:first.append(dict(output=output,index=index,actual_input_flat_offset=actual,expected_input_flat_offset=expected))
                index=[0,1,0,0]
                examples.append(dict(output=output,index=index,actual_input_flat_offset=source_offset(nodes,node_id,index),
                    expected_input_flat_offset=offset*length+1))
                offset+=width
            assert sampled==result['hlo']['sampled_addresses']and mismatches==result['hlo']['mismatching_sampled_addresses']
            assert (mismatches==0)is (variant=='tensor-split')
            rows.append(dict(length=length,variant=variant,hlo=hlo_ref,result=result_ref,retained_exit=exit_ref,retained_drain=drain_ref,
                entry_root=entry.root_id,nodes=[dict(id=n.id,opcode=n.opcode,dimensions=list(n.shape.dimensions),operands=list(n.operand_ids),
                    slices=[dict(start=s.start,limit=s.limit,stride=s.stride)for s in n.slice_dimensions],transpose_axes=list(n.dimensions))for n in entry.instructions],
                sampled_addresses=sampled,mismatching_sampled_addresses=mismatches,first_mismatches=first,second_token_examples=examples,
                producer_device_exact=result['device_exact'],original_failure_control_confirmed=result.get('control_outcome_confirmed'),
                full_output_bytes_independently_reread=False))
    assert not any(name.split('.')[0] in {'torch','torch_neuronx','vllm','numpy','neuronxcc','transformers'} for name in sys.modules)
    _,source=small(Path(__file__))
    report=dict(schema='nemotron_r8_conv_hlo_structure_v1',complete=True,observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        analyzer=source,descriptor_sources=descriptor_refs,descriptor_loaded_by_AST_literal_without_executing_sdk=True,
        protobuf_implementation='python',rows=rows,
        finding='For the saved B1T128/256/512 consumer graphs, original split flattens transposed [B,T,6144] then slices contiguous spans; tensor_split slices its channel dimension before reshaping to heads.',
        scope='Saved graph address semantics and producer metadata only. No new native run, complete raw device-byte review, full-model qualification, universal backend claim or new speedup.',
        full_model_qualified=False,new_speedup_claim=False)
    out=Q/'nemotron9b-r8-conv128-512-hlo-structure-20261109-v1.json'
    with out.open('x')as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    _,ref=small(out)
    print(json.dumps(dict(result=ref,rows=[{k:r[k]for k in ('length','variant','sampled_addresses','mismatching_sampled_addresses','second_token_examples')}for r in rows])))


if __name__=='__main__':main()
