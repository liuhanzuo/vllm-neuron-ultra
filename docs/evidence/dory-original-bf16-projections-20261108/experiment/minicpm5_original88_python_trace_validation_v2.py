"""Original88 metadata-only science validation. Never reads raw or native data."""
import copy
import hashlib
import struct
import types
import sys
import minicpm5_python_trace_observer_v2 as native
import minicpm5_layer0_sdk_python_trace_observer_v1 as sdk

require=native.require
trace=native.trace
ARMS=('HF_native','C_Hd_control','C_Hc_history_generation')


def contract_bundle(contracts,kind,plan):
    """Called compile-only under each original interpreter, never model runtime."""
    require(kind in('native','cpu')and sys.executable==plan['interpreters'][kind],'Actual original contract interpreter')
    routes=('hf_eager',)if kind=='native'else('sdk_prefill','sdk_decode')
    return dict(schema='minicpm5_original88_python_trace_contract_bundle_v1',kind=kind,python=sys.executable,
                version=list(sys.version_info[:3]),implementation=sys.implementation.name,cache_tag=sys.implementation.cache_tag,
                source_only=True,contracts={route:copy.deepcopy(contracts[route])for route in routes})


def combine_contract_bundles(native_bundle,cpu_bundle,plan):
    contracts={}
    for kind,bundle,routes in(('native',native_bundle,('hf_eager',)),('cpu',cpu_bundle,('sdk_prefill','sdk_decode'))):
        require(bundle['schema']=='minicpm5_original88_python_trace_contract_bundle_v1'and bundle['kind']==kind and
                bundle['python']==plan['interpreters'][kind]and bundle['implementation']=='cpython'and
                bundle['source_only']is True and set(bundle['contracts'])==set(routes),'Exact authenticated interpreter-specific route bundle')
        require(type(bundle['version'])is list and len(bundle['version'])==3 and all(type(x)is int for x in bundle['version'])and
                bundle['cache_tag']=='cpython-'+''.join(str(x)for x in bundle['version'][:2]),'Original CPython contract version')
        path=plan['native_provenance']['source']if kind=='native'else plan['sdk_attention_source']
        for route in routes:
            contract=bundle['contracts'][route]
            require(contract['source']==dict(path=path,sha256=plan['source_sha256'][path]),'Contract original source digest')
            if route!='hf_eager':require(contract['route']==route,'SDK contract exact route')
            contracts[route]=copy.deepcopy(contract)
    return contracts


def descriptors(native_source,sdk_source):
    """Compile whole authenticated source; create functions but never execute."""
    result={}
    for route,source,name in (('hf_eager',native_source,'eager_attention_forward'),
            ('sdk_prefill',sdk_source,'scaled_qk'),('sdk_decode',sdk_source,'compute_for_token_gen')):
        code=compile(source['text'],source['path'],'exec',dont_inherit=True)
        found=[]
        def visit(item):
            for child in item.co_consts:
                if type(child)is types.CodeType:
                    if child.co_name==name:found.append(child)
                    visit(child)
        visit(code);require(len(found)==1,'Unique original source function')
        function=types.FunctionType(found[0],{})
        binding=trace.contract(function,source['text'],source['path'],source['sha256'],route,object())
        result[route]=native.descriptor(binding)if route=='hf_eager'else sdk.sdk_descriptor(binding)
    return result


def owner_key(value):
    return value['arm'],value['rank'],value['call'],tuple(value['frame_indices'])


def expected_keys(plan):
    expected=[]
    for fixture in plan['fixtures']:
        for step in range(4):
            frame=next(x for x in plan['frames']if x['fixture']==fixture['id']and x['step']==step)
            call=next(x['index']for x in plan['calls']if frame in x['frames'])
            expected.append(('HF_native',-1,call,(frame['index'],)))
    for arm in ARMS[1:]:
        for rank in (0,1):
            for call in plan['calls']:
                frames=call['frames']if call['phase']=='prefill'else sorted(call['frames'],key=lambda frame:frame['slot'])
                expected.append((arm,rank,call['index'],tuple(frame['index']for frame in frames)))
    require(len(expected)==len(set(expected))==88,'Exact original88 owner set')
    require({(x['arm'],x['rank'],x['call'],tuple(sorted(x['frame_indices'])))for x in plan['writer_contexts']}==
            {(arm,rank,call,tuple(sorted(frames)))for arm,rank,call,frames in expected},'Unchanged writer context matrix')
    return expected


def exact_frames(actual,expected):
    """Detached complete frame values, including integer types and order."""
    require(type(actual)is list and len(actual)==len(expected),'Exact frame list size')
    for frame,wanted in zip(actual,expected):
        require(type(frame)is dict and set(frame)==set(wanted) and
                all(type(frame[key])is type(value) for key,value in wanted.items()) and frame==wanted,
                'Complete frozen frame identity, not index only')
    return True


def expected_cpu_geometry(plan,call):
    """Independently derive the original driver seq_ids and two row orders.

    The actual driver verifies its tensor seq_ids before using them. Here the
    same fixed values are authenticated against frozen int32 graph-input JSON
    hashes without reading/decompressing any payload or calling endpoint helpers.
    """
    require(call['phase']in('prefill','decode'),'Frozen CPU call phase')
    frames=call['frames']
    require(len(frames)==(1 if call['phase']=='prefill'else 4),'Original CPU call rows')
    seq_ids=[frames[0]['slot']]if call['phase']=='prefill'else[0,1,2,3]
    require(all(type(slot)is int and 0<=slot<4 for slot in seq_ids),'Original int32 slot IDs')
    source_bytes=struct.pack('<'+'i'*len(seq_ids),*seq_ids)
    wanted=dict(shape=[len(seq_ids)],stride=[1],offset=0,dtype='torch.int32',
                sha256=hashlib.sha256(source_bytes).hexdigest())
    graph=plan['graph_calls'][call['index']];entry=graph['compressed_inputs'][3]
    require(graph['call']==call and graph['input_metadata'][3]==wanted and
            entry['name']=='seq_ids'and entry['metadata']==wanted and
            entry['raw_bytes']==len(source_bytes)and entry['raw_sha256']==wanted['sha256'],
            'Derived seq_ids match frozen actual graph-input metadata')
    rows=[dict(frame=copy.deepcopy(frame),physical_row=seq_ids.index(frame['slot']))for frame in frames]
    require(len({row['physical_row']for row in rows})==len(frames),'Original unique physical rows')
    # The map retains original call order; observer frames use physical order.
    physical_frames=[row['frame']for row in sorted(rows,key=lambda row:row['physical_row'])]
    return physical_frames,rows


def validate_context(value,plan,contracts):
    require(value['complete']is True,'Complete original context')
    permit=value['permit'];observer=value['observer'];endpoint=value['endpoint'];key=owner_key(permit)
    require(key in expected_keys(plan),'Exact original88 context owner')
    arm,rank,call_index,frame_indices=key;call=plan['calls'][call_index]
    require(observer['arm']==arm and observer['rank']==rank and observer['call']==call_index,
            'Observer exact writer owner')
    require(call['index']==call_index and call['phase']in('prefill','decode'),'Exact frozen call/phase')
    frozen_frames=[next(frame for frame in plan['frames']if frame['index']==index)for index in frame_indices]
    require(all(frame in call['frames']for frame in frozen_frames),'Every original frame belongs to frozen call')
    exact_frames(observer['frames'],frozen_frames)
    require(observer['phase']==call['phase'],'Observer exact frozen native/CPU phase')
    require(endpoint['passed']is True and endpoint['raw_exact']is True and endpoint['original_scope_preserved']is True and
            endpoint['full_model_qualified']is False,'Original unchanged endpoint gate')
    if arm=='HF_native':
        native.validate_capture_summary(observer,contracts['hf_eager'])
        frame=next(x for x in plan['frames']if x['index']==frame_indices[0])
        require(observer['phase']==('prefill'if frame['step']==0 else'decode'),
                'Native phase matches original frame step')
        require(len(frame_indices)==1 and endpoint['schema']=='minicpm5_layer0_native_endpoint_v1'and
                endpoint['frame']==frame and endpoint['full_vocab']==130560 and
                endpoint['original_reference_payload_sha256']==plan['hf_reference']['sha256'],'Native original frame and full vocabulary')
        return 1
    physical_frames,rowmap=expected_cpu_geometry(plan,call)
    exact_frames(observer['frames'],physical_frames)
    actual=observer['physical_row_map']
    require(type(actual)is list and len(actual)==len(rowmap),'Complete CPU physical row map')
    for row,wanted in zip(actual,rowmap):
        require(type(row)is dict and set(row)=={'frame','physical_row'}and
                type(row['physical_row'])is int and row['physical_row']==wanted['physical_row'],
                'Exact physical row in original call order')
        exact_frames([row['frame']],[wanted['frame']])
    route='sdk_prefill'if call['phase']=='prefill'else'sdk_decode'
    sdk.validate_sdk_capture_summary(observer,contracts[route])
    require(endpoint['schema']=='minicpm5_layer0_cpu_endpoint_v1'and endpoint['arm']==arm and endpoint['rank']==rank and
            endpoint['call']==call and endpoint['input_state_unchanged']is True and
            endpoint['original_poststate_reproduced']is True and len(endpoint['all213_tap_sha256'])==213 and
            len(endpoint['all84_input_hashes'])==len(endpoint['all84_post_hashes'])==84,'All original CPU endpoint observations')
    return 1 if route=='sdk_prefill'else 2


def validate_partition(values,plan,contracts,arm,rank):
    require(arm in ARMS and(rank==-1 if arm=='HF_native'else rank in(0,1)),'Original arm/rank')
    expected=[key for key in expected_keys(plan)if key[:2]==(arm,rank)]
    require([owner_key(value['permit'])for value in values]==expected,'Complete ordered native32 or CPU14 context partition')
    events=sum(validate_context(value,plan,contracts)for value in values)
    require(events==(32 if arm=='HF_native'else 20),'Exact native32 or CPU20 scaling events')
    return dict(arm=arm,rank=rank,contexts=len(values),scale_events=events,diagnostic_complete=True,
                full_model_qualified=False,new_speedup_claim=False)


def validate_all(values,plan,contracts):
    require([owner_key(value['permit'])for value in values]==expected_keys(plan),'All88 contexts in exact arm/rank/call/frame order')
    reports=[]
    for arm,rank in [('HF_native',-1)]+[(arm,rank)for arm in ARMS[1:]for rank in(0,1)]:
        reports.append(validate_partition([value for value in values if owner_key(value['permit'])[:2]==(arm,rank)],plan,contracts,arm,rank))
    require(sum(row['contexts']for row in reports)==88 and sum(row['scale_events']for row in reports)==112,'Original88/112 complete')
    return dict(diagnostic_complete=True,contexts=88,scale_events=112,partitions=reports,
                cpu_history_origin='actual_previous_CPU_post_across_all14calls',full_model_qualified=False,new_speedup_claim=False)


def validate_endpoint_reference(value,plan,read_json):
    """Rebind saved endpoint digests to the original authenticated JSON records."""
    permit=value['permit'];endpoint=value['endpoint'];arm=permit['arm'];rank=permit['rank'];call=permit['call']
    if arm=='HF_native':
        frame=permit['frame_indices'][0];reference=read_json(plan['hf_frame_records'][frame])
        require(reference['frame']==endpoint['frame']and reference['original_logits_raw_exact']is True and
                reference['payload_sha256']==endpoint['original_v2_frame_payload_sha256'],'Exact frozen native frame reference')
        # Original FP32 logits are separately compared with the frozen payload
        # by unchanged native_endpoint; this JSON's retained tap is BF16.
        require(type(endpoint['logits_sha256'])is str and len(endpoint['logits_sha256'])==64,'Retained full-vocabulary actual hash')
        return True
    records=[read_json(ref)for ref in plan['cpu_records']]
    matches=[row for row in records if row['arm']==arm and row['rank']==rank and row['call']['index']==call]
    require(len(matches)==1,'Unique original CPU endpoint reference');reference=matches[0]
    require(reference['complete']is True and reference['call']==endpoint['call']and
            endpoint['output_sha256']==reference['output_metadata']['sha256']and
            endpoint['all213_tap_sha256']==reference['stage_raw_sha256'],'All original CPU output and213 tap hashes')
    def states(ref):
        manifest=read_json(ref);require(manifest['complete']is True,'Original state manifest complete')
        rows=[row for row in manifest['rows']if row['rank']==rank]
        require([(row['rank'],row['name'])for row in rows]==[(rank,'kv_mgr.past_key_values.%d'%i)for i in range(84)],'Original84 ordered states')
        return [row['raw_sha256']for row in rows]
    require(endpoint['all84_input_hashes']==states(reference['imported_manifest'])and
            endpoint['all84_post_hashes']==states(reference['after']),'All84 before and produced post hashes rebound')
    origin=dict(kind='cpu_history_previous_post',arm=arm,call=call-1)if arm=='C_Hc_history_generation'and call>0 else dict(kind='frozen_v2_device_before',call=call)
    require(reference['history']['origin']==origin,'Original generated CPU history origin, including logical resets')
    return True


def validate_references(reports,plan,contracts,read_json,arm=None,rank=None):
    """For worker and independent audit; read_json authenticates bounded JSON."""
    values=[read_json(ref)for ref in reports]
    result=validate_all(values,plan,contracts)if arm is None else validate_partition(values,plan,contracts,arm,rank)
    for value in values:validate_endpoint_reference(value,plan,read_json)
    return result


def validate_worker_result(result,plan,contracts,read_json,arm,rank):
    """Future worker interface; does not authorize or launch that worker."""
    require(result['complete']is True and result['full_model_qualified']is False and
            result['scale_observation_schema']==native.SCHEMA,'Original88 worker trace schema')
    if arm=='HF_native':
        require(rank==-1 and result['frames']==32 and result['original_full_vocab_raw_exact']is True and
                result['scale_events']==32,'Complete original native32 worker')
    else:
        require(result['arm']==arm and result['rank']==rank and result['calls']==14 and result['frames']==32 and
                result['all_original_outputs_taps_input_poststate_raw_exact']is True and result['scale_events']==20 and
                result['generated_cpu_history_preserved']is(arm=='C_Hc_history_generation'),'Complete original CPU14 worker and history origin')
    return validate_references(result['reports'],plan,contracts,read_json,arm,rank)
