"""Check saved weight/state keys, shard geometry, and serialized user-input routing.

This reads tensor headers, serialized runtime descriptors and small script code.
It never loads tensor payloads, executes model arithmetic, or imports Torch.
"""
from collections import Counter
from datetime import datetime,timezone
import ast
import hashlib
import json
import math
from pathlib import Path
import resource
import struct
import sys
import zipfile
from audit_qwen35_serialized_runtime import RuntimeReader,Payload,fields,tensor,digest

ROOT=Path(__file__).resolve().parent
DEST=ROOT/'qwen35-serving-failure'


def shard_spec(name,shape):
    shape=list(shape);axis=None;stride=1;dtype=14
    if name=='embed_tokens.weight':axis=1
    elif name=='lm_head.weight':axis=0
    elif name.endswith(('.mlp.gate_proj.weight','.mlp.up_proj.weight')):axis=0
    elif name.endswith('.mlp.down_proj.weight'):axis=1
    elif '.linear_attn.' in name:
        leaf=name.split('.linear_attn.',1)[1]
        if leaf in ['A_log','dt_bias']:axis=0;dtype=1
        elif leaf in ['in_proj_qkv.weight','conv1d.weight']:axis=0;stride=3
        elif leaf in ['in_proj_z.weight','in_proj_b.weight','in_proj_a.weight']:axis=0
        elif leaf=='out_proj.weight':axis=1
        else:assert leaf=='norm.weight'
    elif '.self_attn.' in name:
        leaf=name.split('.self_attn.',1)[1]
        if leaf in ['q_proj.weight','k_proj.weight','v_proj.weight']:axis=0
        elif leaf=='o_proj.weight':axis=1
        else:assert leaf in ['q_norm.weight','k_norm.weight']
    else:assert name=='norm.weight' or name.endswith(('.input_layernorm.weight','.post_attention_layernorm.weight'))
    if axis is not None:
        assert shape[axis]%(2*stride)==0
        shape[axis]//=2
    return dict(checkpoint_key=name,shape=shape,data_type=dtype,partition_axis=axis,partition_stride=stride)


def main():
    resource.setrlimit(resource.RLIMIT_AS,(256<<20,256<<20))
    resource.setrlimit(resource.RLIMIT_CPU,(30,30))
    output=DEST/'weight-state-binding-audit.json';assert not output.exists()
    interface_path=DEST/'serialized-runtime-interface-audit.json'
    interfaces=json.loads(interface_path.read_text());assert interfaces['complete']
    case_path=ROOT/'qwen35-2b-baseline-skip256.json';case=json.loads(case_path.read_text())
    config_path=Path(case['model_path'])/'config.json';config=json.loads(config_path.read_text())
    checkpoint=Path(case['model_path'])/'model.safetensors-00001-of-00001.safetensors'
    with checkpoint.open('rb') as stream:
        header_size=struct.unpack('<Q',stream.read(8))[0];assert header_size<1<<20
        header_raw=stream.read(header_size);header=json.loads(header_raw)
    original={};ignored=Counter()
    for key,value in header.items():
        if key=='__metadata__':continue
        if key.startswith(('model.visual.','mtp.')):
            ignored['visual' if key.startswith('model.visual.') else 'mtp']+=1
            continue
        assert key.startswith('model.language_model.'),key
        name=key.removeprefix('model.language_model.');assert name not in original
        assert value['dtype'] in ['BF16','F32'];original[name]=value
    assert len(original)==320 and 'lm_head.weight' not in original and config['tie_word_embeddings']
    original['lm_head.weight']=original['embed_tokens.weight']
    expected={name:shard_spec(name,row['shape']) for name,row in original.items()}
    runtime_weights={t['checkpoint_key']:t for t in interfaces['graphs'][0]['interface']['inputs'] if t['type']==3}
    assert set(runtime_weights)==set(expected) and len(expected)==321
    for name,spec in expected.items():
        assert runtime_weights[name]['shape']==spec['shape'],name
        assert runtime_weights[name]['data_type']==spec['data_type'],name

    artifact=Path(case['artifact'])/'model.pt';stat=artifact.stat()
    assert dict(realpath=str(artifact.resolve()),device=stat.st_dev,inode=stat.st_ino,size=stat.st_size,
                mtime_ns=stat.st_mtime_ns,ctime_ns=stat.st_ctime_ns)==interfaces['artifact_identity']
    with zipfile.ZipFile(artifact) as archive:
        with archive.open('model/data.pkl') as stream:state=RuntimeReader(stream).load().state['nxd_model'].state
        loader=state['weight_loader'];assert loader.origin=='__torch__.torch.classes.neuron.LayoutTransformation'
        neff,encoded,tag,ranks=loader.state
        assert isinstance(neff,Payload) and ranks==2
        raw=encoded.encode('utf8','surrogateescape');assert len(raw)<1<<20
        wlt={kind:[] for kind in ['inputs','outputs']}
        for number,wire,value in fields(raw):
            assert number in [1,2] and wire==2
            wlt['inputs' if number==1 else 'outputs'].append(tensor(value))
        assert len(wlt['inputs'])==len(wlt['outputs'])==321
        for kind in wlt:
            assert {t['checkpoint_key'] for t in wlt[kind]}==set(expected)
            for i,t in enumerate(wlt[kind]):
                name=t['checkpoint_key'];assert t['shape']==expected[name]['shape']
                assert t['data_type']==expected[name]['data_type']
                assert t['name']==('input' if kind=='inputs' else 'output')+str(i)
        assert [t['checkpoint_key'] for t in wlt['inputs']]==[t['checkpoint_key'] for t in wlt['outputs']]

        code_hashes={};routes=[]
        flattener=state['flattener_map'].state
        for graph in interfaces['graphs']:
            key=graph['kind']+'_'+str(graph['index'])
            origin=flattener[key].origin
            code_path='model/code/'+origin.rsplit('.',1)[0].replace('.','/')+'.py'
            code=archive.read(code_path);assert len(code)<4096
            code_hashes[code_path]=hashlib.sha256(code).hexdigest()
            tree=ast.parse(code.decode())
            method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='forward')
            assert len(method.body)==2 and isinstance(method.body[0],ast.Assign) and isinstance(method.body[1],ast.Return)
            assign=method.body[0];assert isinstance(assign.value,ast.Name) and assign.value.id=='inputs'
            names=[n.id for n in assign.targets[0].elts];assert len(names)==7
            indices=[names.index(n.id) for n in method.body[1].value.elts]
            expected_indices=[0,1,2,3] if graph['kind']=='context_encoding_model' else [0,2,3]
            assert indices==expected_indices
            user=[t for t in graph['interface']['inputs'] if t['type']==1]
            assert [t['shape'] for t in user]==[graph['route_shapes'][i] for i in indices]
            assert all(t['data_type']==2 for t in user)
            aliases=graph['interface']['aliases']
            for out,inp in aliases.items():
                assert graph['interface']['inputs'][inp]['checkpoint_key']=='kv_mgr.past_key_values.'+str(int(out)-1)
            routes.append(dict(kind=graph['kind'],index=graph['index'],bucket=graph['bucket'],
                original_user_argument_indices=indices,cache_outputs_reach_same_layer_state=True,script=code_path))
        spmd_path='model/code/__torch__/neuronx_distributed/trace/spmd.py';code=archive.read(spmd_path)
        code_hashes[spmd_path]=hashlib.sha256(code).hexdigest();text=code.decode()
        assert 'checkpoint: List[Dict[str, Tensor]]' in text and 'weights: List[Dict[str, Tensor]]' in text
        assert '(model).initialize(states, weights, start_rank_id, )' in text
        assert '(weight_loader).forward(checkpoint, False, )' in text
        assert 'self.state = (state_initializer).forward()' in text

    init=interfaces['state_initializer'];nc=json.loads(artifact.with_name('neuron_config.json').read_text())['neuron_config']
    assert config['num_hidden_layers']==24 and config['layer_types'].count('linear_attention')==18
    cache_rows=[]
    for i,kind in enumerate(config['layer_types']):
        if kind=='linear_attention':
            heads=config['linear_num_value_heads']//2;dim=config['linear_value_head_dim']
            shapes=[[4,3*heads*dim,config['linear_conv_kernel_dim']],[4,heads,config['linear_key_head_dim'],dim]]
            dtypes=[15,6];sizes=[2,4]
        else:
            shapes=[[4,config['num_key_value_heads']//2,nc['max_length'],config['head_dim']]]*2
            dtypes=[15,15];sizes=[2,2]
        for offset,(shape,dtype,size) in enumerate(zip(shapes,dtypes,sizes)):
            name=f'kv_mgr.past_key_values.{2*i+offset}'
            assert init['shapes'][name]==shape and init['dtypes'][name]==dtype
            cache_rows.append(dict(name=name,layer=i,kind=kind,shape=shape,torch_dtype_id=dtype,
                                   declared_bytes_per_rank=math.prod(shape)*size))
    weights_bytes=sum(math.prod(v['shape'])*(4 if v['data_type']==1 else 2) for v in expected.values())
    sources=[Path(__file__),ROOT/'audit_qwen35_serialized_runtime.py',interface_path,case_path,config_path,
        ROOT/'qwen35_neuron_adapter.py',ROOT/'qwen35_text_port.py',
        ROOT/'qwen35-baseline/meta-original-sharding-validation.json',
        artifact.with_name('neuron_config.json')]
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    result=dict(complete=True,observed_at=datetime.now(timezone.utc).isoformat(),
        original_checkpoint_header=dict(path=str(checkpoint),bytes=header_size,sha256=hashlib.sha256(header_raw).hexdigest(),
                                       unique_text_tensors=320,ignored_nontext_tensor_counts=dict(ignored),weight_payloads_read=False),
        expected_runtime_weights=expected,runtime_weight_inputs=321,
        all_weight_keys_shapes_types_match_original_checkpoint_shard_rules=True,
        tied_lm_head='Originalembedding clonedbyexistingadapter, thenembedding shardedonhiddenaxis andhead onvocabularyaxis. Metadataagreementdoesnotproveactualpayload equality.',
        layout_transformation=dict(neff_bytes=neff.size,neff_sha256=neff.digest,metaneff_bytes=len(raw),
            metaneff_sha256=hashlib.sha256(raw).hexdigest(),serialization_tag=tag,local_ranks=ranks,
            input_count=321,output_count=321,all_named_shapes_types_match=True,
            original_input_output_key_order_equal=True,interfaces=wlt),
        graph_routes=routes,cache_states=cache_rows,all_528_aliases_preserve_exact_layer_index=True,
        declared_weight_bytes_per_rank=weights_bytes,
        declared_cache_bytes_per_rank=sum(t['declared_bytes_per_rank'] for t in cache_rows),
        serialized_script_sha256=code_hashes,source_sha256={str(p):digest(p) for p in sources},
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        scope='Structuralchecks only: originaltensorheaders->TP2shape/dtype rules->layoutloaderinterfaces->all11runtimegraphs; '
              'all528cachealiases preserve layerindex,11actualserializedflattenersselectexpecteduserarguments. '
              'Differentweightparameterordersareexplicitlyretained; actualserializedinitializationpassesnameddictionaries. '
              'No rawcheckpointweight/loadedweight/statepayloadcomparison,HLOoperatorvalidation ormodelresult. '
              'Declaredweights/cachebytes excludecompilerlayouts,scratch,duplicatedbuffers andhoststaging;notpeakHBM/RAM.')
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['complete','runtime_weight_inputs',
        'all_weight_keys_shapes_types_match_original_checkpoint_shard_rules','all_528_aliases_preserve_exact_layer_index',
        'declared_weight_bytes_per_rank','declared_cache_bytes_per_rank','peak_rss_bytes']}))


if __name__=='__main__':main()
