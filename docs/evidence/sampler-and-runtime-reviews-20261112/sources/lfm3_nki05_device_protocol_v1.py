"""Complete fixed device task schedule and scientific report validators."""
from pathlib import Path
from lfm3_nki05_device_matrix_v1 import device_specs,alias_specs,mappings,alias_mappings
from lfm3_nki05_device_supplements_v2 import wrapper_specs,fp32_alias_specs,control_mappings
from lfm3_nki05_raw_device_leaf_v1 import MAPPINGS
from lfm3_nki05_device_context_v1 import ARTIFACT_SCHEMA,VARIANT,FULL_MODEL_GATES
from slot_cache_nki05_typed_jit_v1 import DEVICE_SOFTWARE

PHASES=('raw_device','regression_device','regression_alias','extended_device','extended_alias','fp32_alias','wrapper_controls')
COUNTS=dict(raw_device=(4,36),regression_device=(9,43),regression_alias=(12,60),extended_device=(42,2520),extended_alias=(24,192),fp32_alias=(6,222),wrapper_controls=(16,418))


def require(value,message):
    if not value:raise RuntimeError(message)


def tasks():
    result=[]
    for phase in PHASES:
        count={'extended_device':42,'extended_alias':24,'fp32_alias':6,'wrapper_controls':16}.get(phase)
        for index in range(count) if count is not None else (-1,):result.append(dict(phase=phase,index=index))
    require(len(result)==91,'Complete 91 original/supplemental child tasks')
    return result


def key(task):
    require(task in tasks(),'Fixed complete task matrix')
    return task['phase']+('-all' if task['index']==-1 else '-'+str(task['index']).zfill(3))


def manifest():
    return dict(tasks=tasks(),graphs=113,checks=3491,phase_counts={p:list(v) for p,v in COUNTS.items()},
        original_cpu_cases=3716,full_cpu_passes=3,first_actual_graph=dict(dtype='bfloat16',shape=[4,7168]),
        original_model_gates=FULL_MODEL_GATES,context4096_target_retained=True,context4096_qualified=False,
        full_model_qualified=False,new_speedup_claim=False,automatic_successor_launch=False)


def graph_spec(task):
    phase=task['phase'];index=task['index']
    return {'extended_device':device_specs,'extended_alias':alias_specs,'fp32_alias':fp32_alias_specs,'wrapper_controls':wrapper_specs}[phase]()[index]


def scientific_path(root,task):
    phase=task['phase'];base=Path(root)/phase
    if phase=='raw_device':return base/'device.json'
    if phase in ('regression_device','regression_alias'):return base/'validation.json'
    return base/graph_spec(task)['name']/'result.json'


def receipt(value,phase,sources):
    require(value['schema']==ARTIFACT_SCHEMA and value['variant']==VARIANT and value['software']==DEVICE_SOFTWARE
            and value['source_sha256']==sources and value['phase']==phase,'Complete new device artifact identity')
    require(type(value['sha256']) is str and len(value['sha256'])==64 and set(value['sha256'])<=set('0123456789abcdef'),'Complete artifact digest')
    return value


def strict(rows,expected_slots,fields):
    require(len(rows)==len(expected_slots) and [row['slots'] for row in rows]==expected_slots,'Every original mapping in exact order')
    require(all(all(row[k] is True for k in fields) for row in rows),'Every original byte/state/input/shape gate')


def complete_digest(value):
    return type(value) is str and len(value)==64 and set(value)<=set('0123456789abcdef')


def validate_task(task,value,sources):
    require(task in tasks() and value['complete'] is value['passed'] is True,'Actual complete scientific task')
    phase=task['phase'];five=[[0,1,2,3],[3,1,0,2],[3,-1,-1,-1],[-1]*4,[2,2,0,3]]
    if phase=='raw_device':
        specs=[(dtype,width) for dtype in ('bfloat16','float32') for width in (7168,65536)]
        require([(g['dtype'],g['width']) for g in value['graphs']]==specs,'All four raw graphs in original order')
        require(len(value['checks'])==36,'Full raw check count')
        for i,(dtype,width) in enumerate(specs):
            graph=value['graphs'][i];identity=receipt(graph['artifact_identity'],phase,sources)
            require(identity['case']==dict(dtype=dtype,shape=[4,width],rows=4,variant='candidate') and identity['sha256']==graph['artifact_sha256'],'Actual raw artifact case')
            rows=value['checks'][i*9:(i+1)*9];strict(rows,MAPPINGS,('bitwise_equal','inputs_unchanged'))
            require(all((row['dtype'],row['width'])==(dtype,width) for row in rows),'Every raw row dtype and shape')
        return value
    if phase=='regression_device':
        specs=[(f'kv{n}_b4',[4,2,n,128],4,'torch.bfloat16') for n in (128,256,512,1024,2048)]
        specs += [('conv_b4',[4,1024,7],4,'torch.bfloat16'),('kv2048_b1',[4,2,2048,128],1,'torch.bfloat16'),('rnj_kv4096_b4',[4,4,4096,128],4,'torch.bfloat16'),('nemotron_ssm_b4',[4,64,80,128],4,'torch.float32')]
        require(len(value['graphs'])==9,'Original nine standalone graphs')
        for graph,(name,shape,rows,dtype) in zip(value['graphs'],specs):
            require((graph['name'],graph['shape'],graph['dtype'])==(name,shape,dtype),'Original standalone shape/dtype/order')
            identity=receipt(graph['artifact_identity'],phase,sources)
            require(identity['case']==dict(name=name,shape=shape,rows=rows,dtype=dtype,variant='candidate') and identity['sha256']==graph['artifact_sha256'],'Exact standalone artifact')
            strict(graph['checks'],five if rows==4 else [[0],[3],[-1]],('bitwise_equal','cache_input_unchanged','updates_input_unchanged'))
        return value
    if phase=='regression_alias':
        specs=[(length,heads,label) for length,heads in [(128,2),(256,2),(512,2),(1024,2),(2048,2),(4096,4)] for label in ('reference','candidate')]
        require(len(value['graphs'])==12,'Original twelve alias graphs')
        for graph,(length,heads,label) in zip(value['graphs'],specs):
            require((graph['length'],graph['kv_heads'],graph['variant'])==(length,heads,label) and graph['passed'] is True,'Original complete alias graph order')
            identity=receipt(graph['artifact_identity'],phase,sources)
            require(identity['case']==dict(variant=label,length=length,kv_heads=heads,shape=[4,heads,length,128],rows=4,dtype='bfloat16'),'Exact alias artifact shape')
            strict(graph['checks'],five,('attention_finite','cache_k_bitwise','cache_v_bitwise','input_k_bitwise','input_v_bitwise'))
        return value
    spec=graph_spec(task);require(value['spec']==spec,'Exact immutable indexed spec')
    identity=receipt(value['artifact_identity'],phase,sources)
    if phase=='extended_device':
        require(identity['case']==dict(spec,variant='candidate') and identity['sha256']==value['artifact_sha256'],'Actual extended device artifact')
        strict(value['checks'],mappings(spec['capacity'],spec['rows']),('bitwise_equal','cache_input_unchanged','updates_input_unchanged','slots_input_unchanged'))
        require(all(complete_digest(row['actual_sha256']) and row['actual_sha256']==row['expected_sha256'] for row in value['checks']),'Complete original output bytes match oracle')
    elif phase=='extended_alias':
        b,n=spec['capacity'],spec['length'];shapes=[[b,1024,7]] if spec['kind']=='conv' else [[b,2,n,128]]*2
        require(identity['case']==dict(spec,dtype='bfloat16',rows=b,shapes=shapes) and identity['sha256']==value['artifact_sha256'],'Actual extended alias shapes and dtype')
        fields=('output_finite','cache_outputs_bitwise','persistent_states_bitwise','bindings_unchanged','inputs_unchanged')
        strict(value['checks'],alias_mappings(b),fields+(('output_bitwise',) if spec['kind']=='conv' else ()))
        require(value['attention_rounding_is_diagnostic_only'] is True
                and len(value['initial_sha256'])==len(shapes) and all(complete_digest(x) for x in value['initial_sha256'])
                and len(value['inputs_sha256'])==len(value['checks'])
                and all(len(row)==(2 if spec['kind']=='conv' else 5) and all(complete_digest(x) for x in row) for row in value['inputs_sha256']), 'Every complete paired alias input and initial-state hash')
        require([row['step'] for row in value['checks']]==list(range(len(value['checks'])))
                and all(complete_digest(row['output_sha256']) and len(row['state_sha256'])==len(shapes)
                        and all(complete_digest(x) for x in row['state_sha256']) for row in value['checks']), 'All sequential alias output/state hashes')
    elif phase=='fp32_alias':
        require(identity['case']==spec,'Actual FP32 companion case')
        b=spec['capacity'];sequence=control_mappings(b,b)+[list(range(b)),[-1]*b,list(reversed(range(b)))]
        strict(value['checks'],sequence,('prior_state_bitwise','cache_output_bitwise','persistent_state_bitwise','bindings_unchanged','inputs_unchanged'))
        require([row['step'] for row in value['checks']]==list(range(len(sequence))),'Every FP32 persistent sequential reuse step')
    else:
        require(identity['case']==spec,'Actual production dispatch wrapper case')
        strict(value['checks'],control_mappings(spec['capacity'],spec['rows']),('bitwise_equal','dtype_shape_preserved','inputs_unchanged'))
        branch='original' if spec['rows']==1 else 'typed';other='typed' if spec['rows']==1 else 'original'
        require(value['trace_dispatch_counts'][branch]>0 and value['trace_dispatch_counts'][other]==0,'Actual original-single-row trace dispatch')
    return value
