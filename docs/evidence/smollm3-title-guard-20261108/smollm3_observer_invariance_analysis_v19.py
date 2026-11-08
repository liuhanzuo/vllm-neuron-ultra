"""Pure JSON analysis; output differences are retained results, not exceptions.

Protocol violations reject qualification. No payload/service/native/file IO.
"""
import hashlib
import re
import smollm3_observer_invariance_protocol_v19 as p


def validate_capture(base,run_id,value):
    row=p.run_row(run_id);requests=value['requests'];groups=p.old.groups()
    expected=[(g,index) for g in groups for index in range(g['count'])]
    p.require(value['schema']=='smollm3_observer_invariance_capture_v2' and value['run']==row and
        value['passed'] is True and value['role']==row['role'] and
        value['diagnostic_complete'] is True and value['protocol_passed'] is True,'Complete exact run capture')
    p.require(value['request_scope']==len(requests)==len(expected)==22 and value['output_tokens']==2808,'Original complete22/2808')
    ids=set()
    for actual,(group,index) in zip(requests,expected):
        p.require(actual['group']==group['id'] and actual['index']==index and actual['prompt_token_ids']==base['prompts'][group['prompt']],'Exact original request order/prompt')
        p.require(type(actual['request_id'])is str and actual['request_id'] and actual['request_id'] not in ids,'Unique real API request ID');ids.add(actual['request_id'])
        p.require(actual['model']=='smollm3-3b' and actual['streaming'] is group['stream'] and actual['finish_reason']=='length','Original model/stream/finish')
        tokens=actual['token_ids'];p.require(type(tokens)is list and len(tokens)==group['tokens'] and all(type(x)is int and 0<=x<128256 for x in tokens),'Actual original engine token IDs/budget')
        p.require(type(actual['text'])is str and actual['response_status']==200 and actual['token_id_origin']=='fresh_engine_response_not_retokenized_historical_text','Exact wire result provenance')
        wanted=dict(prompt_tokens=len(actual['prompt_token_ids']),completion_tokens=len(tokens),total_tokens=len(actual['prompt_token_ids'])+len(tokens))
        p.require(all(type(actual['usage'].get(k))is int and actual['usage'][k]==v for k,v in wanted.items()),'Usage counts')
        p.require(actual['request_body_sha256']==hashlib.sha256(p.old.json.dumps(p.old.request_body(base,group)).encode()).hexdigest(),'Original request bytes')
    p.require(sum(len(x['token_ids']) for x in requests)==2808 and value['historical']==historical(base,row['role'],requests),'Complete original tokens and truthful historical result')
    p.require(value['full_model_qualified'] is value['new_speedup_claim'] is value['original76_requests_qualified'] is False,'Diagnostic only')
    return requests


def historical(base,role,requests):
    expected=base['historical'][role];initial=[x for x in requests if x['group'].startswith('initial-') and x['group']!='initial-warm8'];target=requests[-1]
    return dict(initial_count=len(initial),initial_text_matches=[dict(group=x['group'],index=x['index'],
        matches=hashlib.sha256(x['text'].encode()).hexdigest()==expected['initial_text_sha256']) for x in initial],
        target_text_matches=hashlib.sha256(target['text'].encode()).hexdigest()==expected['short_first_single_text_sha256'],
        target_usage_matches=target['usage']==expected['short_first_single_usage'],
        all_match=len(initial)==19 and all(hashlib.sha256(x['text'].encode()).hexdigest()==expected['initial_text_sha256'] for x in initial)
            and hashlib.sha256(target['text'].encode()).hexdigest()==expected['short_first_single_text_sha256'] and target['usage']==expected['short_first_single_usage'],
        is_admission_requirement=False,historical_token_ids_reconstructed=False)


def compare_requests(left,right):
    p.require(len(left)==len(right)==22,'Complete paired request arrays');rows=[]
    for a,b in zip(left,right):
        p.require((a['group'],a['index'],a['prompt_token_ids'])==(b['group'],b['index'],b['prompt_token_ids']),'Paired request identity')
        p.require(len(a['token_ids'])==len(b['token_ids']),'Paired actual budget')
        differences=[dict(index=i,left_token=x,right_token=y) for i,(x,y) in enumerate(zip(a['token_ids'],b['token_ids'])) if x!=y]
        equal_fields={key:a[key]==b[key] for key in ('text','usage','finish_reason','model','streaming')}
        rows.append(dict(group=a['group'],index=a['index'],tokens=len(a['token_ids']),equal=not differences and all(equal_fields.values()),
            token_ids_equal=not differences,first_token_difference=differences[0] if differences else None,
            all_token_differences=differences,**{key+'_equal':value for key,value in equal_fields.items()}))
    return dict(equal=all(x['equal'] for x in rows),requests=22,tokens=2808,rows=rows)


def mode_evidence(run_id,evidence):
    """Root external inventory/authentication is required, not file presence."""
    run=p.run_row(run_id);p.require(evidence['run_id']==run_id and evidence['mode']==run['mode'] and evidence['root_authenticated'] is True,'Independent mode evidence')
    for flag in ('actual_parent_and_engine_verified','all_unit_invocations_verified','all_successful_exits_retained_and_audited','all_groups_drained','output_inventory_complete'):
        p.require(evidence[flag] is True,'Actual mode/lifecycle proof: '+flag)
    records=evidence['mode_records'];p.require(2<=len(records)<=8 and sum(x['spawned'] is False for x in records)==1 and any(x['spawned'] is True for x in records),'Parent/spawn mode receipts')
    p.require(len({(x['pid'],x['start_ticks']) for x in records})==len(records),'Distinct actual mode owners')
    for item in records:
        p.require(item['run']==run and item['mode']==run['mode'] and item['observer_install_called'] is (run['mode']=='on')
            and item['original_methods_unchanged'] is (run['mode']=='off') and item['observer_module_imported'] is (run['mode']=='on'),'Actual process installation mode')
    if run['mode']=='off':
        p.require(evidence['observer_files']==[] and evidence['graph_slot_history_observed'] is False,'Off has no observer or slot observation')
    else:
        p.require(2<=len(evidence['observer_files'])<=8 and evidence['graph_slot_history_observed'] is True,'On complete bounded observer inventory')
    return evidence


def on_history_summary(export):
    p.require(export['passed'] is True and len(export['requests'])==22,'Successful complete on export')
    p.require(sum(len(x['token_ids']) for x in export['requests'])==2808,'Full on token trace')
    rows=[]
    for item in export['requests']:
        samples=item['samples'];p.require(len(samples)==len(item['token_ids']) and [x['token'] for x in samples]==item['token_ids'],'Export sample/token binding')
        rows.append(dict(group=item['group'],index=item['index'],samples=[dict(generated_index=i,engine_step=x['engine_step'],physical_slot=x['physical_slot'],frames=x['frames']) for i,x in enumerate(samples)]))
    return dict(observed=True,requests=rows,frames_count=export['frames_count'],engine_steps_count=export['engine_steps_count'],
        graph_request_logical_alignment_proven=False,storage_alias_equivalence_proven=False)


def bind_on_export(run,capture,evidence,export):
    """Check root-supplied post-drain file/producer proof and actual API IDs.

    The root lifecycle must authenticate the real service and file bytes before
    supplying this proof. This pure function cannot replace that physical audit.
    """
    proof=evidence['engine_observer_source'];producer=proof['producer']
    p.require(proof['run']==run and proof['capture_sha256']==p.digest(capture)
        and proof['file_verified_by_root'] is True and proof['producer_identity_verified_by_root'] is True,
        'Root authenticated exact capture and observer producer')
    p.require(type(producer['pid']) is int and producer['pid']>0 and type(producer['ppid']) is int and producer['ppid']>0
        and type(producer['start_ticks']) is str and producer['start_ticks'].isdigit()
        and int(producer['start_ticks'])>0 and re.fullmatch('[0-9a-f]{32}',producer['invocation_id'])
        and producer['invocation_id']!='0'*32,'Exact engine process identity')
    p.require(type(producer['argv']) is list and producer['argv'] and all(type(x)is str for x in producer['argv'])
        and producer['cgroup']=='0::'+p.old.GROUP_PREFIX+p.unit(run['run_id'],'server')+'.service',
        'Exact engine argv and run service')
    matching=[x for x in evidence['mode_records'] if x['spawned'] is True
        and (x['pid'],x['start_ticks'])==(producer['pid'],producer['start_ticks'])]
    p.require(len(matching)==1,'Observer producer bound to authenticated spawned engine mode')
    expected_path=str(p.directory(run['run_id'])/('observer-'+str(producer['pid'])+'.jsonl'))
    p.require(proof['observer_file']==expected_path and expected_path in evidence['observer_files']
        and re.fullmatch('[0-9a-f]{64}',proof['observer_sha256']),'Exact observer source within this run')
    p.require(export['run']==run and export['capture_sha256']==proof['capture_sha256']
        and export['observer_file']==proof['observer_file'] and export['observer_sha256']==proof['observer_sha256']
        and export['observer_producer']==producer and export['pid']==producer['pid'] and export['role']==run['role'],
        'Export bound to exact run capture and independently authenticated source')
    p.require(len(export['requests'])==len(capture['requests'])==22,'Complete export request identities')
    for actual,request in zip(export['requests'],capture['requests']):
        p.require(actual['api_request_id']==request['request_id'] and actual['engine_request_id']==request['request_id']+'-0',
            'Export actual API and engine request identity')
        p.require(all(actual[k]==request[k] for k in ('group','index','prompt_token_ids','token_ids')),
            'Exact API/export tokens and request history')
    return producer


def compare_matrix(base,captures,evidence,exports):
    expected={x['run_id'] for x in p.matrix()};p.require(set(captures)==set(evidence)==expected,'Complete twelve runs, no dropped failure/difference')
    on_ids={x['run_id'] for x in p.matrix() if x['mode']=='on'};p.require(set(exports)==on_ids,'Exactly six on histories; no fabricated off export')
    requests={};histories={};producers=set();all_requests=set();all_owners=set()
    for run in p.matrix():
        run_id=run['run_id'];requests[run_id]=validate_capture(base,run_id,captures[run_id]);mode_evidence(run_id,evidence[run_id])
        for request in requests[run_id]:
            p.require(request['request_id'] not in all_requests,'Fresh request IDs across all twelve runs')
            all_requests.add(request['request_id'])
        for record in evidence[run_id]['mode_records']:
            identity=(record['pid'],record['start_ticks'])
            p.require(identity not in all_owners,'Fresh parent and engine across all twelve runs');all_owners.add(identity)
        if run['mode']=='on':
            export=exports[run_id];producer=bind_on_export(run,captures[run_id],evidence[run_id],export)
            identity=(producer['pid'],producer['start_ticks'],producer['invocation_id'])
            p.require(identity not in producers,'Fresh engine for every on run');producers.add(identity)
            histories[run_id]=on_history_summary(export)
        else:histories[run_id]=dict(observed=False,reason='Observer completely absent; no slot or graph history claimed')
    pairs=[];repeats=[];schedule=[]
    for role in p.ROLES:
        runs=[x for x in p.matrix() if x['role']==role]
        for pair in (0,1):
            off=next(x['run_id'] for x in runs if x['pair']==pair and x['mode']=='off');on=next(x['run_id'] for x in runs if x['pair']==pair and x['mode']=='on')
            pairs.append(dict(role=role,pair=pair,off=off,on=on,comparison=compare_requests(requests[off],requests[on])))
        for mode in ('off','on'):
            a,b=[x['run_id'] for x in runs if x['mode']==mode];repeats.append(dict(role=role,mode=mode,left=a,right=b,comparison=compare_requests(requests[a],requests[b])))
        a,b=[x['run_id'] for x in runs if x['mode']=='on'];left=histories[a]['requests'];right=histories[b]['requests']
        differences=[]
        for x,y in zip(left,right):
            if x['samples']!=y['samples']:differences.append(dict(group=x['group'],index=x['index'],left=x['samples'],right=y['samples']))
        schedule.append(dict(role=role,left=a,right=b,complete_on_history_rows_equal=not differences,differences=differences,
            off_history_comparison_possible=False,equality_is_not_complete_graph_state_equivalence=True))
    equal=all(x['comparison']['equal'] for x in pairs+repeats)
    return dict(schema='smollm3_observer_invariance_matrix_result_v2',diagnostic_complete=True,protocol_passed=True,passed=equal,
        observer_api_invariance_passed=equal,scope='Only finite API outputs across two off/on pairs and repeated identical modes',
        requests=264,output_tokens=33696,pairs=pairs,repeats=repeats,history_observations=histories,on_schedule_comparisons=schedule,
        historical={key:value['historical'] for key,value in captures.items()},differences_retained=True,
        uninstrumented_graph_state_equivalence=False,causal_claim=False,full_model_qualified=False,new_speedup_claim=False,
        original76_requests_qualified=False,cold_long_reuse_gates_unchanged=True)

