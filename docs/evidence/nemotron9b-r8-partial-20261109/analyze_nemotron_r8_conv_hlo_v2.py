"""All16 saved convolution split HLO cases; exact logical coordinate replay only."""
import ast
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

from analyze_nemotron_r8_conv_hlo_v1 import small,source_offset,R,Q,CONTROL,JOBS,SITE,DESCRIPTORS


def main():
    prior=json.loads((Q/'nemotron9b-r8-conv128-512-hlo-structure-20261109-v1.json').read_text())
    source=Path(prior['analyzer']['path']);assert hashlib.sha256(source.read_bytes()).hexdigest()==prior['analyzer']['sha256']
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
    for batch,length in [(1,x)for x in (128,256,512,1024,2048,4096)]+[(4,1),(1,1)]:
        for variant in ('original','tensor-split'):
            name=f'split-nemotron-conv-b{batch}-t{length}-{variant}'
            raw,result_ref=small(CONTROL/(name+'-worker-result.json'),8*2**20);outer=json.loads(raw);result=outer['result']
            assert outer['complete']is True and outer['passed']is True and result['compile_passed']is True
            assert result['job']['case']['input_shape']==[batch,6144,length]
            expected_pass=variant=='tensor-split'or(batch==length==1)
            assert result['device_exact']is expected_pass
            _,exit_ref=small(CONTROL/(name+'-retained-exit.json'));_,drain_ref=small(CONTROL/(name+'-retained-drain.json'))
            raw,hlo_ref=small(JOBS/name/'compiler-work/model/graph.hlo',16384);assert hlo_ref['sha256']==result['hlo']['sha256']
            module=message();module.ParseFromString(raw);entry=next(c for c in module.computations if c.id==module.entry_computation_id)
            nodes={n.id:n for n in entry.instructions};root=nodes[entry.root_id];assert root.opcode=='tuple'and len(root.operand_ids)==3
            sampled=0;mismatches=0;first=[];offset=0
            for output,(width,heads,head_dim)in enumerate(((5120,64,80),(512,4,128),(512,4,128))):
                node_id=root.operand_ids[output];assert list(nodes[node_id].shape.dimensions)==[batch,length,heads,head_dim]
                for b in range(batch):
                    for token in range(length):
                        for channel in (0,width//2,width-1):
                            index=[b,token,channel//head_dim,channel%head_dim]
                            actual=source_offset(nodes,node_id,index);expected=(b*6144+offset+channel)*length+token
                            sampled+=1
                            if actual!=expected:
                                mismatches+=1
                                if len(first)<6:first.append(dict(output=output,index=index,actual_input_flat_offset=actual,expected_input_flat_offset=expected))
                offset+=width
            assert sampled==result['hlo']['sampled_addresses']and mismatches==result['hlo']['mismatching_sampled_addresses']
            assert (mismatches==0)is expected_pass
            byte_comparisons=[v for check in result['device_checks']for v in check['comparisons']]
            assert len(byte_comparisons)==6 and all(v['passed']is expected_pass for v in byte_comparisons)
            rows.append(dict(batch=batch,length=length,variant=variant,hlo=hlo_ref,result=result_ref,retained_exit=exit_ref,retained_drain=drain_ref,
                sampled_addresses=sampled,mismatching_sampled_addresses=mismatches,first_mismatches=first,
                producer_cpu_passed=result['cpu_passed'],producer_device_exact=result['device_exact'],original_failure_control_confirmed=result.get('control_outcome_confirmed'),
                device_comparisons=byte_comparisons,full_raw_output_bytes_independently_reread=False,
                nodes=[dict(id=n.id,opcode=n.opcode,dimensions=list(n.shape.dimensions),operands=list(n.operand_ids),
                    slices=[dict(start=s.start,limit=s.limit,stride=s.stride)for s in n.slice_dimensions],transpose_axes=list(n.dimensions))for n in entry.instructions]))
    assert len(rows)==16
    assert not any(name.split('.')[0]in {'torch','torch_neuronx','vllm','numpy','neuronxcc','transformers'}for name in sys.modules)
    _,self_ref=small(Path(__file__))
    report=dict(schema='nemotron_r8_all_conv_hlo_structure_v2',complete=True,observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        analyzer=self_ref,helper=prior['analyzer'],descriptor_sources=descriptor_refs,protobuf_implementation='python',
        rows=rows,all_original_conv_shapes_reparsed=True,original_long_and_B4T1_mismatches_preserved=True,original_B1T1_expected_exact=True,
        native_or_compiler_execution=False,full_raw_device_bytes_independently_reread=False,full_model_qualified=False,new_speedup_claim=False)
    out=Q/'nemotron9b-r8-all-conv-hlo-structure-20261109-v2.json'
    with out.open('x')as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    _,ref=small(out)
    print(json.dumps(dict(result=ref,rows=[{k:r[k]for k in ('batch','length','variant','sampled_addresses','mismatching_sampled_addresses')}for r in rows])))


if __name__=='__main__':main()
