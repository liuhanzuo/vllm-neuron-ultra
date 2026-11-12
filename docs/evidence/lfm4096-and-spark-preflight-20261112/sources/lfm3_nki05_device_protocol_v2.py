"""Additive LFM KV4096 successor of the accepted full 91-task protocol."""
from pathlib import Path
import lfm3_nki05_device_protocol_v1 as original
from lfm3_nki05_context4096_matrix_v1 import device_specs,alias_specs,mappings,alias_mappings,alias_positions

require=original.require
PHASES=original.PHASES+('context4096_device','context4096_alias')
COUNTS=dict(original.COUNTS,
    context4096_device=(len(device_specs()),sum(len(mappings(x['capacity'],x['rows'])) for x in device_specs())),
    context4096_alias=(len(alias_specs()),sum(len(alias_mappings(x['capacity'])) for x in alias_specs())))


def tasks():
    return original.tasks()+[dict(phase=phase,index=i)
        for phase,specs in (('context4096_device',device_specs()),('context4096_alias',alias_specs()))
        for i in range(len(specs))]


def key(task):
    require(type(task) is dict and set(task)=={'phase','index'} and type(task['phase']) is str
            and type(task['index']) is int and task in tasks(),'Fixed full original plus additive task')
    return task['phase']+('-all' if task['index']==-1 else '-'+str(task['index']).zfill(3))


def manifest():
    prior=original.manifest()
    return dict(prior,tasks=tasks(),graphs=sum(x[0] for x in COUNTS.values()),
        checks=sum(x[1] for x in COUNTS.values()),phase_counts={p:list(v) for p,v in COUNTS.items()},
        retained_protocol=prior,context4096_primitive_only=True,
        context4096_supplement=dict(capacities=[4,8,16],kv_heads_per_rank=2,length=4096,
            standalone=device_specs(),persistent_alias=alias_specs(),signed_int64_controls=True,
            all_original_gates_retained=True,model_cold_long_reuse_serving_still_required=True))


def graph_spec(task):
    key(task)
    if task['phase'] in original.PHASES:return original.graph_spec(task)
    return (device_specs if task['phase']=='context4096_device' else alias_specs)()[task['index']]


def scientific_path(root,task):
    key(task)
    if task['phase'] in original.PHASES:return original.scientific_path(root,task)
    return Path(root)/task['phase']/graph_spec(task)['name']/'result.json'


def validate_task(task,value,sources):
    key(task)
    if task['phase'] in original.PHASES:return original.validate_task(task,value,sources)
    require(value['complete'] is value['passed'] is True,'Actual complete additive scientific task')
    phase=task['phase'];spec=graph_spec(task)
    require(value['spec']==spec,'Exact LFM 2-head4096 immutable spec')
    identity=original.receipt(value['artifact_identity'],phase,sources)
    if phase=='context4096_device':
        require(identity['case']==dict(spec,variant='candidate') and identity['sha256']==value['artifact_sha256'],
                'Actual LFM standalone shape/dtype/rows artifact')
        original.strict(value['checks'],mappings(spec['capacity'],spec['rows']),
            ('bitwise_equal','cache_input_unchanged','updates_input_unchanged','slots_input_unchanged'))
        require(all(original.complete_digest(row['actual_sha256']) and row['actual_sha256']==row['expected_sha256']
                    for row in value['checks']),'Complete LFM4096 output bytes equal oracle')
    else:
        b=spec['capacity'];sequence=alias_mappings(b);shapes=[[b,2,4096,128]]*2
        require(identity['case']==dict(spec,dtype='bfloat16',rows=b,shapes=shapes)
                and identity['sha256']==value['artifact_sha256'],'Actual persistent LFM K/V shape and dtype')
        original.strict(value['checks'],sequence,('output_finite','cache_outputs_bitwise',
            'persistent_states_bitwise','bindings_unchanged','inputs_unchanged'))
        require(value['attention_rounding_is_diagnostic_only'] is True
                and len(value['initial_sha256'])==2 and all(original.complete_digest(x) for x in value['initial_sha256'])
                and len(value['inputs_sha256'])==len(sequence)
                and all(len(row)==5 and all(original.complete_digest(x) for x in row) for row in value['inputs_sha256']),
                'Complete paired initial-state and input hashes')
        require(value['positions']==[alias_positions(b,i) for i in range(len(sequence))],
                'Exact 4096 boundary-position samples')
        require([row['step'] for row in value['checks']]==list(range(len(sequence)))
                and all(original.complete_digest(row['output_sha256']) and len(row['state_sha256'])==2
                        and all(original.complete_digest(x) for x in row['state_sha256']) for row in value['checks']),
                'All actual sequential persistent K/V hashes')
    return value
