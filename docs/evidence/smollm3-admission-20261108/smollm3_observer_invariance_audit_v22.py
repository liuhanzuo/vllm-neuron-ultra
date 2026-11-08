"""Future external launch/drain/audit owner; no effects unless --execute.

No native imports. This owner keeps the serial lock through independent terminal
audit, includes itself in the successor process catalog, and never claims its
own future exit has already happened.

Source-only v19 stop-race successor; a fresh root-reviewed namespace and full
transport/dependency migration are required before runtime. Frozen v18 is unchanged.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import signal
import socket
import sys
import time

import smollm3_observer_invariance_lifecycle_v21 as l
import smollm3_observer_invariance_protocol_v17 as p
import smollm3_observer_invariance_analysis_v17 as analysis
import smollm3_observer_invariance_worker_v17 as worker
import smollm3_token_capture_export_v3 as exporter
import smollm3_authorized_stop_transition_v1 as stop_transition


def read_observer(path, producer, role):
    before=l.ordinary(path);p.require(before.st_size<=64*l.MIB,'Observer64MiB cap')
    rows=[];digest=hashlib.sha256()
    with Path(path).open('rb')as stream:
        while True:
            raw=stream.readline(256*1024+1)
            if not raw:break
            p.require(len(raw)<=256*1024 and raw.endswith(b'\n'),'Complete bounded observer row')
            digest.update(raw);row=json.loads(raw)
            p.require(row['pid']==producer['pid']and row['role']==role,'Actual observer producer/role')
            rows.append(row);p.require(len(rows)<=50000,'Observer row count')
    after=l.ordinary(path)
    p.require((before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==
              (after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'Postdrain observer changed')
    p.require(rows and [x['event']for x in rows]==list(range(1,len(rows)+1)),'Complete observer event sequence')
    return rows,digest.hexdigest()


def audit_failure_parts(root, value):
    """Every cap+1 byte observed must be accounted, without reading any tail."""
    p.require(value['passed']is False and value['protocol_passed']is False,'Original false request failure')
    count=value['observed_response_bytes'];p.require(type(count)is int and 0<=count<=4*l.MIB+1,'Reader cap+1 bound')
    p.require(value['response_cap_exceeded']is(count>4*l.MIB),'Overflow truth')
    parts=value['retained_parts'];errors=value['raw_write_errors'];dig=hashlib.sha256();observed=0
    all_parts={}
    for row in parts:
        p.require(row['index']not in all_parts,'Duplicate successful part');all_parts[row['index']]=dict(row,success=True)
    for row in errors:
        p.require(row['index']not in all_parts,'Overlapping failed part');all_parts[row['index']]=dict(row,success=False)
    if value['response_previously_saved']:
        p.require(not parts and not errors,'Previously saved raw has no replacement parts')
        raw=l.read_bytes(root/(value['group']+'-request'+str(value['index'])+'-response.raw'),4*l.MIB)
        p.require(len(raw)==count and l.digest_bytes(raw)==value['raw_sha256'],'Previously retained observed bytes')
        p.require(value['all_observed_bytes_durable']is True,'Previously saved durability')
        return dict(audited=True,bytes=count,all_observed_bytes_durable=True)
    p.require(sorted(all_parts)==list(range(len(all_parts))),'Ordered complete part attempts')
    for index,row in sorted(all_parts.items()):
        p.require(row['offset']==observed and 0<=row['bytes']<=4*l.MIB,'Contiguous bounded part')
        p.require(re.fullmatch(r'[A-Za-z0-9_-]+\.raw',row['name']),'Safe raw part name')
        if row['success']:
            raw=l.read_bytes(root/row['name'],4*l.MIB)
            p.require(len(raw)==row['bytes']and l.digest_bytes(raw)==row['sha256'],'Retained part exact bytes')
            receipt=row['receipt'];p.require(receipt['path']==str(root/row['name'])and receipt['sha256']==row['sha256']and
                receipt['bytes']==len(raw)and receipt['durable']is receipt['exclusive']is True,'Actual part receipt binding')
            dig.update(raw)
        observed+=row['bytes']
    p.require(observed==count,'Every observed byte has success or explicit write error')
    durable=not errors
    p.require(value['all_observed_bytes_durable']is durable,'Truthful durability flag')
    if durable:p.require(dig.hexdigest()==value['raw_sha256'],'Ordered whole observed response digest')
    return dict(audited=True,bytes=count,all_observed_bytes_durable=durable,write_errors=errors)


def raw_capture(base,run_id,capture):
    analysis.validate_capture(base,run_id,capture)
    root=p.directory(run_id);parser=worker.old_worker.parser_module()
    expected=[(group,index)for group in p.old.groups()for index in range(group['count'])]
    for record,(group,index)in zip(capture['requests'],expected):
        stem=group['id']+'-request'+str(index);body=p.old.request_body(base,group)
        wire=l.read_bytes(root/(stem+'-request.raw'),4*l.MIB)
        p.require(wire==json.dumps(body).encode()and l.read(root/(stem+'-request.json'))==body,'Original exact request wire/body')
        p.require(l.digest_bytes(wire)==record['request_body_sha256'],'Request wire digest')
        raw=l.read_bytes(root/(stem+'-response.raw'),4*l.MIB)
        kw=dict(prompt_ids=body['prompt'],output_tokens=body['max_tokens'],model=body['model'])
        parsed=parser.parse_sse(raw.splitlines(keepends=True),**kw)if group['stream']else parser.parse_completion(raw,**kw)
        parsed.update(group=group['id'],index=index,request_body_sha256=l.digest_bytes(wire),response_status=200)
        p.require(parsed==record==l.read(root/(stem+'-parsed.json')),'Fresh full wire reparse equals captured record')
    for group in p.old.groups():
        report=l.read(root/(group['id']+'-group.json'))
        p.require(report==dict(complete=True,group=group,requests=[x for x in capture['requests']if x['group']==group['id']]),'Complete10 group receipts')
    return capture


def mode_receipts(run_id):
    root=p.directory(run_id);run=p.run_row(run_id);owners=[];records=[]
    original_parent=l.read(root/'server-owner.json');original_intent=l.read(root/'server-launch.json')
    for hello_path in sorted(root.glob('hello-server-*.json')):
        hello=l.read(hello_path);pid=hello['process']['pid']
        bundle=l.read(root/('permit-server-'+str(pid)+'.json'));producer=bundle['producer']
        p.require(bundle['hello_sha256']==p.digest(hello),'Postdrain hello ACK binding')
        p.require(hello['run']==run and hello['attempt_id']==producer['attempt_id']and hello['spawned']is producer['spawned'],'Hello run/nonce/spawn')
        p.require(all(hello['process'][key]==producer[key]for key in ('pid','start_ticks','ppid','argv','cgroup')),'Actual authenticated process provenance')
        p.require(producer['run']==run and producer['mode']==run['mode']and
            producer['cgroup']=='0::'+l.GROUP_PREFIX+p.unit(run_id,'server')+'.service','Producer exact run service')
        p.require(bundle['intent']==original_intent and producer['invocation_id']==original_parent['invocation_id']and
                  producer['attempt_id']==original_parent['attempt_id'],'Postdrain producer exact original service')
        if producer['spawned']:
            chain=producer['parent_chain'];p.require(chain and chain[0]==hello['process'],'Recorded actual spawn ancestry')
            for left,right in zip(chain,chain[1:]):p.require(left['ppid']==right['pid']and left['cgroup']==right['cgroup'],'Recorded ancestry continuity')
            p.require(chain[-1]['ppid']==original_parent['pid'],'Spawn ancestry reaches original service owner')
        else:p.require(all(producer[k]==original_parent[k]for k in('pid','start_ticks','ppid','argv','cgroup')),'Exact parent mode owner')
        permit=bundle['permit'];p.require(all(permit[key]==producer[key]for key in ('pid','start_ticks','ppid','argv','cgroup','unit','invocation_id','attempt_id')),'Permit complete producer identity')
        mode=l.read(root/('mode-'+str(pid)+'.json'))
        p.require(mode['pid']==pid and mode['start_ticks']==producer['start_ticks']and mode['spawned']is producer['spawned'],'Mode actual PID/start')
        admitted=l.read(root/('admitted-server-'+str(pid)+'.json'))
        p.require(admitted['actual_process']==hello['process']and admitted['permit_sha256']==p.digest(permit)and
                  admitted['hello_sha256']==p.digest(hello)and admitted['mode']==run['mode'],'Worker observed same root permit')
        l.gone(producer);owners.append(producer);records.append(mode)
    p.require(2<=len(owners)<=8 and len({(x['pid'],x['start_ticks'])for x in owners})==len(owners),'All parent and model spawn mode owners')
    p.require(set(x.name for x in root.glob('mode-*.json'))=={'mode-'+str(x['pid'])+'.json'for x in owners},'No unauthenticated mode receipt')
    observed=[l.read(path)for path in sorted(root.glob('process-seen-server-*.json'))]
    p.require(observed,'Retained actual service process inventory')
    for item in observed:
        l.gone(item['process'])
        p.require(item['invocation_id']==original_parent['invocation_id']and item['run']==run and item['mode']==run['mode'],'Observed process run/mode/invocation')
        if '--multiprocessing-fork'in item['process']['argv']:
            p.require(any(all(owner[k]==item['process'][k]for k in('pid','start_ticks','ppid','argv','cgroup'))for owner in owners),'Every observed spawned engine has actual mode receipt')
    return owners,records


def chunks(root,key,rows,write):
    refs=[]
    for start in range(0,len(rows),64):
        name=key+'-'+str(start//64)+'.json';value=dict(kind=key,start=start,rows=rows[start:start+64]);raw=l.json_bytes(value)
        path=root/name
        if write:l.write_once(path,raw)
        else:p.require(l.read_bytes(path)==raw,'Unchanged complete export chunk')
        refs.append(dict(path=str(path),sha256=l.digest_bytes(raw),start=start,count=len(value['rows'])))
    return refs


def audit_spawn_provenance(run_id,owners):
    """Postdrain all dispatched model spawns: none unpaired, ignored or failed."""
    root=p.directory(run_id);by_pid={x['pid']:x for x in owners};expected=set();proofs=[]
    for owner in owners:
        bundle=l.read(root/('permit-server-'+str(owner['pid'])+'.json'));permit=bundle['permit']
        p.require(permit['environment_lineage']==owner['environment_lineage'] and permit['parent_chain']==owner['parent_chain'],'Retained permit full ancestry')
        p.environment_policy.validate_lineage(permit['environment_lineage'],permit,permit['spawned'])
        expected.add('spawn-hook-'+str(owner['pid'])+'.json')
        hook=l.read(root/('spawn-hook-'+str(owner['pid'])+'.json'))
        p.require(hook==dict(schema='smollm3_spawn_hook_v3',pid=owner['pid'],start_ticks=owner['start_ticks'],mode=owner['mode'],
                  same_in_both_modes=True,original_transport_code_verified=True,title_identity_policy=l.title.POLICY,
                  source_sha256={x:permit['root_runtime_source_sha256'][x]for x in l.witness.TRANSPORT_FILES}),'Every actual identical hook/source receipt')
        ray=p.environment_policy.ray
        hook_name='ray-hook-'+str(owner['pid'])+'.json';expected.add(hook_name)
        p.require(l.read(root/hook_name)==dict(schema='smollm3_ray_hook_v1',process=p.environment_policy.identity(permit),
            permit_sha256=p.digest(permit),same_in_both_modes=True,source_sha256={str(ray.SITE/k):v for k,v in ray.PINS.items()}),'Exact same off/on Ray witness')
        local_before=root/('ray-before-'+str(owner['pid'])+'.json');local_after=root/('ray-after-'+str(owner['pid'])+'.json')
        if not permit['spawned'] or local_before.exists() or local_after.exists():
            local_state=ray.validate_receipts(l.read(local_before),l.read(local_after),permit)
            l.authenticate_ray_state(permit,local_state);expected.update((local_before.name,local_after.name))
        if permit['spawned']:
            p.require(permit['ray_clean_observation']==ray.ZERO and bundle['producer']['ray_clean_observation']==ray.ZERO,'Actual inherited child clean value')
        else:p.require(permit['ray_clean_observation']==ray.ABSENT,'Root clean Ray absent')
        edges=permit['environment_lineage']['edges']
        p.require(owner['parent_chain']==[x['child']for x in reversed(edges)],'Whole retained kernel ancestry')
        if permit['spawned']:
            edge=edges[-1];parent_pid=edge['parent']['pid'];sequence=edge['sequence']
            p.require(parent_pid in by_pid,'Immediate parent admitted in complete owner set')
            parent_bundle=l.read(root/('permit-server-'+str(parent_pid)+'.json'))
            before_name='spawn-before-'+str(parent_pid)+'-'+str(sequence)+'.json'
            after_name='spawn-after-'+str(parent_pid)+'-'+str(sequence)+'.json'
            p.require(after_name not in expected,'Dispatch consumed once');expected.update((before_name,after_name))
            before=l.read(root/before_name);after=l.read(root/after_name)
            p.require(l.witness.validate_dispatch(before,after,parent_bundle['permit'],owner)==edge,'Complete final dispatch binding')
            p.require(p.environment_policy.extend_lineage(parent_bundle['permit']['environment_lineage'],edge)==permit['environment_lineage'],'No invented prefix depth')
        for stage in ('before','after'):
            name='runner-'+stage+'-'+str(owner['pid'])+'.json';expected.add(name)
        before=l.read(root/('runner-before-'+str(owner['pid'])+'.json'));after=l.read(root/('runner-after-'+str(owner['pid'])+'.json'))
        p.require(before==dict(schema='smollm3_runner_import_before_v1',pid=owner['pid'],start_ticks=owner['start_ticks'],
                  permit_sha256=p.digest(permit),lineage_sha256=p.digest(permit['environment_lineage']),stage='native_ready',
                  ld_source_state=p.environment_policy.ld_state(permit['environment_lineage'],'native_ready'),cv2_loaded=False,exact_values_checked=True),'Exact runner-before receipt')
        p.require(after==dict(schema='smollm3_runner_import_after_v1',pid=owner['pid'],start_ticks=owner['start_ticks'],
                  before_sha256=p.digest(before),permit_sha256=p.digest(permit),
                  ld_source_state=p.environment_policy.ld_state(permit['environment_lineage'],'runner_ready'),exact_values_checked=True),'Exact single runner-after receipt')
    title_commits=0
    from smollm3_enginecore_title_hook_v3 import PINS as title_pins
    for owner in owners:
        bundle=l.read(root/('permit-server-'+str(owner['pid'])+'.json'));permit=bundle['permit']
        if permit['spawned']:
            hook_name='title-hook-'+str(owner['pid'])+'.json';expected.add(hook_name)
            p.require(l.read(root/hook_name)==dict(schema='smollm3_enginecore_title_hook_v3',
                spawn_identity=l.title.identity(permit),permit_sha256=l.title.digest(permit),source_sha256=title_pins,
                single_ray_trace_owner=True,same_in_both_modes=True,actual_engine_entry_observed=False),
                'Exact child title hook, separate from actual invocation proof')
        outcome=l.title_runtime.audit(permit)
        if outcome['changed']:
            p.require(permit['spawned'],'Root server cannot claim a title transition')
            title_commits+=1;expected.update(Path(item['path']).name for item in outcome['references'])
    p.require(title_commits>=1,'Successful serving includes actual committed EngineCore title invocation')
    actual={x.name for pattern in ('spawn-*.json','runner-*.json','ray-*.json','title-*.json')for x in root.glob(pattern)}
    p.require(actual==expected,'Every new witness accounted; failures/unpaired/unlaunched dispatches forbid qualification')
    for name in sorted(expected):proofs.append(dict(path=str(root/name),sha256=l.digest_bytes(l.read_bytes(root/name))))
    return proofs


def audit_environment(run_id,owners):
    """All exact server six/client four phases and source-derived LD states."""
    root=p.directory(run_id);run=p.run_row(run_id);expected_files=set();proofs=[]
    client=l.read(root/'client-owner.json')
    for owner,kind in [(x,'server')for x in owners]+[(client,'client')]:
        bundle=l.read(root/('permit-'+kind+'-'+str(owner['pid'])+'.json'));permit=bundle['permit']
        lineage=permit['environment_lineage'];contract_path=str(p.environment_policy.CONTRACT)
        contract_sha=permit['root_runtime_source_sha256'][contract_path]
        stages=p.environment_policy.stages(kind,permit['spawned'])
        ray=p.environment_policy.ray;ray_before=None;ray_after=None
        if kind=='server':
            before_path=root/('ray-before-'+str(owner['pid'])+'.json')
            after_path=root/('ray-after-'+str(owner['pid'])+'.json')
            if before_path.exists()or after_path.exists():
                ray_before=l.read(before_path);ray_after=l.read(after_path)
        for stage in stages:
            path=root/('environment-'+stage+'-'+str(owner['pid'])+'.json');expected_files.add(path.name)
            row=l.read(path);ray_state=row['ray_source_state']
            allowed=p.environment_policy.permitted(permit['environment'],kind,stage,permit['spawned'],lineage,ray_state)
            p.environment_policy.validate_ray_state(kind,stage,permit['spawned'],ray_state)
            expected_ray=None if kind=='client'else ray.expected_snapshot_state(permit,stage,ray_before,ray_after)
            p.require(ray_state==expected_ray,'Exact monotonic Ray snapshot state; no future local receipt or regression')
            if kind=='server':l.authenticate_ray_state(permit,ray_state)
            p.require(row['ray_observation']==(p.environment_policy.ray.ZERO if kind=='server'and ray_state['kind']!='absent'else p.environment_policy.ray.ABSENT),'Exact saved Ray phase observation')
            keys=row['dependency_keys']
            p.require((ray.KEY in keys)is(kind=='server'and ray_state['kind']!='absent'),
                      'Ray dependency key membership matches exact nonabsent state')
            p.require(row['stage']==stage and row['kind']==kind and row['spawned']is permit['spawned']and
                row['pid']==permit['pid']and row['start_ticks']==permit['start_ticks']and row['attempt_id']==permit['attempt_id']and
                row['run_id']==run_id and row['mode']==run['mode']and row['source_contract']==dict(path=contract_path,sha256=contract_sha),
                'Environment phase actual permit/source binding')
            p.require(row['exact_values_checked']is True and row['unknown_keys']==[]and row['values_recorded']is False
                and type(keys)is list and keys==sorted(set(keys))and set(keys)<=set(allowed),'Exact source-reviewed environment phase receipt')
            expected_state=p.environment_policy.ld_state(lineage,stage)if kind=='server'else None
            p.require(row['ld_source_state']==expected_state,'Exact non-sensitive source LD state')
            if kind=='server'and(stage in('native_ready','runner_ready','api_loading','ray_ready','serve')or permit['spawned']):
                p.require(set(p.environment_policy.constants())<=set(keys),'All required imported dependency settings')
            if permit['spawned']:p.require('VLLM_USE_V1'in keys,'Spawn inherits actual serving engine version')
            p.require(('LD_LIBRARY_PATH'in keys)is(kind=='server'and p.environment_policy.expected_ld(lineage,stage)is not None),'Exact LD absence/presence phase')
            proofs.append(dict(path=str(path),sha256=l.digest_bytes(l.read_bytes(path))))
    p.require({x.name for x in root.glob('environment-*.json')}==expected_files,'Complete exact phase receipt set')
    return proofs


def audit_run(base,run_id,write_export=False):
    """Postdrain raw bytes and authenticated actual observers, never fake flags."""
    from smollm3_observer_invariance_controller_v23 import state_from_disk,physical
    run=p.run_row(run_id);root=p.directory(run_id)
    for kind in ('server','client'):
        state=state_from_disk(run_id,kind);p.require(state is not None and state['owner']is not None,'Both actual service owners')
        physical(state)
        l.verify_retained_binding(state['intent'],state['owner'],l.read(root/(kind+'-service-binding.json')))
    client=l.read(root/'client-terminal.json');value=client['original_properties']
    p.require(client['passed']is True and value['Result']=='success'and value['ExecMainCode']=='1'and value['ExecMainStatus']=='0','Original successful client exit')
    p.require(value['ExecMainPID']==str(client['owner']['pid'])and value['InvocationID']==client['owner']['invocation_id'],'Retained exit belongs to actual client')
    server=l.read(root/'server-terminal.json');stop=l.read(root/'server-stop-intent.json')
    p.require(server['drained']is True and server['authorized_stop']is True and stop['authorized_stop']is True and
        stop['server_normal_shutdown']is True and stop['successful_client_exit']is False,'Server authorized stop distinct from client success')
    p.require(stop['owner']==l.read(root/'server-owner.json')and stop['claim']==stop['owner']['invocation_id']and
              stop['binding']==l.read(root/'server-service-binding.json'),'Authorized server stop identity')
    capture=raw_capture(base,run_id,l.read(root/'capture-result.json'))
    owners,modes=mode_receipts(run_id)
    environments=audit_environment(run_id,owners)
    spawn_provenance=audit_spawn_provenance(run_id,owners)
    source_files=sorted(root.glob('observer-*.jsonl'))
    proof=dict(run_id=run_id,mode=run['mode'],root_authenticated=True,actual_parent_and_engine_verified=True,
        all_unit_invocations_verified=True,all_successful_exits_retained_and_audited=True,all_groups_drained=True,
        output_inventory_complete=True,mode_records=modes,observer_files=[str(x)for x in source_files],
        graph_slot_history_observed=run['mode']=='on',authenticated_processes=owners,environment_phases=environments,spawn_provenance=spawn_provenance)
    exported=None
    if run['mode']=='off':
        p.require(not source_files and not list(root.glob('frames-*.json'))and not list(root.glob('engine_steps-*.json'))and
                  not(root/'replay-export.json').exists(),'Off has no observer/export evidence')
    else:
        by_pid={x['pid']:x for x in owners};logs=[];engine=[]
        p.require({x.name for x in source_files}=={'observer-'+str(pid)+'.jsonl'for pid in by_pid},'Every actual on parent/spawn has exactly one observer')
        for path in source_files:
            pid=int(path.stem.split('-')[1]);rows,digest=read_observer(path,by_pid[pid],run['role'])
            logs.append(dict(path=str(path),sha256=digest,rows=len(rows)))
            if any(x['kind']=='engine_begin'for x in rows):engine.append((path,rows,digest,by_pid[pid]))
            else:p.require(all(x['kind']in('installed','model_load_begin','model_loaded')or
                    (x['kind']in('graph_inputs','sample','sample_failure')and x.get('scope')=='startup')for x in rows),'Startup observer only before request engine')
        p.require(len(engine)==1 and engine[0][3]['spawned']is True,'Exactly one actual spawned request observer')
        path,rows,digest,producer=engine[0]
        exported=exporter.export(rows,worker.export_capture_view(base,run_id,capture),retain_decoded=False)
        source=dict(run=run,capture_sha256=p.digest(capture),producer={k:producer[k]for k in ('pid','start_ticks','ppid','argv','cgroup','invocation_id')},
                    observer_file=str(path),observer_sha256=digest,file_verified_by_root=True,producer_identity_verified_by_root=True)
        proof['engine_observer_source']=source
        exported.update(run=run,capture_sha256=p.digest(capture),observer_file=str(path),observer_sha256=digest,
                        observer_producer=source['producer'],all_observer_logs=logs)
        for key in ('frames','engine_steps'):
            values=exported.pop(key);exported[key+'_chunks']=chunks(root,key,values,write_export);exported[key+'_count']=len(values)
        analysis.bind_on_export(run,capture,proof,exported)
        if write_export:l.record(root/'replay-export.json',exported)
        else:p.require(l.read(root/'replay-export.json')==exported,'Exact independently recomputed exporter3 output')
    analysis.mode_evidence(run_id,proof)
    l.monitor(run_id,reserve=False)
    if write_export:l.record(root/'root-mode-lifecycle.json',proof)
    else:p.require(l.read(root/'root-mode-lifecycle.json')==proof,'Exact independent mode/lifecycle audit')
    return capture,proof,exported


def failed_evidence():
    """Retain protocol failures and report damaged evidence without replacing root error."""
    results=[];errors=[];witnesses=[]
    for run in p.matrix():
        root=p.directory(run['run_id'])
        if not root.exists():continue
        for path in sorted(root.glob('*-failure.json')):
            try:results.append(dict(path=str(path),audit=audit_failure_parts(root,l.read(path))))
            except BaseException as error:
                if isinstance(error,l.DeadlineExceeded):raise
                errors.append(dict(path=str(path),error=str(error)[:4096]))
        # Failed/unpaired dispatches are never dropped just because no client
        # or complete model-owner set exists. All bytes remain charged by the
        # unchanged full output inventory and included in successor pins.
        for pattern in ('spawn-*.json','runner-*.json','ray-*.json'):
            for path in sorted(root.glob(pattern)):
                try:
                    raw=l.read_bytes(path);value=l.decode_bytes(path,raw)
                    witnesses.append(dict(path=str(path),sha256=l.digest_bytes(raw),bytes=len(raw),schema=value.get('schema'),
                        failed=value.get('passed')is False,partial_evidence=True,qualification_proven=False))
                except BaseException as error:
                    if isinstance(error,l.DeadlineExceeded):raise
                    errors.append(dict(path=str(path),error_type=type(error).__name__))
    return dict(request_failures=results,spawn_and_runner_witnesses=witnesses,evidence_audit_errors=errors)


def predecessor_catalog(spec):
    value=l.reference(spec['final_predecessor_catalog'])
    l.complete_predecessor_extension(value,l.reference(l.PREDECESSOR_FLOOR))
    p.require(value['reviewer']=='/root'and value['root_finalized']is True and value['all_planned_launches_accounted_for']is True,'Final root catalog')
    rows=value['rows'];processes=value.get('processes',value.get('process_predecessors',[]))
    p.require(value['complete_count']==len(rows) and len({x['unit']for x in rows})==len(rows),'Actual catalog count without hardcoded history')
    # Reuse only the bounded row validator and known historical missing-start
    # exception, never the old builder/bound(plan) or its sc7 admission.
    import smollm3_token_capture_predecessors_v7 as old_catalog
    old_catalog.validate_rows(rows,processes)
    return value,rows,processes


def drain_predecessors(spec):
    value,rows,processes=predecessor_catalog(spec);observations=[]
    for owner in processes:l.gone(owner)
    for row in rows:
        name=row['unit'];samples=[]
        for index in range(2):
            current=l.props(name);p.require(l.terminal(current)and current['Id']==name+'.service','Final predecessor terminal')
            if row['owner']is None:
                p.require(row['never_launched']is True and current['LoadState']=='not-found'and not current['InvocationID']and
                          all(not Path(x).exists()for x in row['absent_records']),'Actual never-launched absence')
            else:
                l.gone(row['owner'])
                if current['LoadState']=='not-found':
                    p.require(not current['InvocationID']and current['ExecMainPID']=='0','Retired historical service identity')
                else:
                    p.require(current['InvocationID']==row['invocation_id']and current['ExecMainPID']==str(row['owner']['pid'])and
                              l.installed(current)==row['installed_argv'],'Exact historical invocation/command')
            paths=set(row['recorded_cgroups'])|{'/sys/fs/cgroup'+l.GROUP_PREFIX+name+'.service'}
            groups=[]
            for path in sorted(paths):
                group=path.removeprefix('/sys/fs/cgroup');p.require(group==l.GROUP_PREFIX+name+'.service','Historical group belongs to unit')
                groups.append(l.cgroup_inventory(group,require_empty=True))
            samples.append(dict(properties=current,groups=groups))
            if index==0:time.sleep(.02)
        p.require(samples[0]==samples[1],'Stable predecessor drain');observations.append(dict(unit=name,samples=samples))
    running=l.command(['systemctl','--user','list-units','--type=service','--state=running,activating,deactivating','--plain','--no-legend','--no-pager'])
    p.require(not[x for x in running.splitlines()if x.split()and x.split()[0].startswith('trn1-')],'No other heavy Trn1 job')
    return dict(catalog=spec['final_predecessor_catalog'],count=len(rows),observations=observations,processes=processes,
                all_final_predecessors_drained=True,physical_absence_is_not_success=True)


def payload_proof(base,deadline):
    verified={};stats={}
    for name,row in base['payloads'].items():
        l.check_deadline(deadline)
        before=l.file_stat(name);p.require(before==row['identity'],'Original payload metadata')
        digest=hashlib.sha256()
        with Path(name).open('rb')as stream:
            while True:
                block=stream.read(4*l.MIB)
                if not block:break
                digest.update(block);l.check_deadline(deadline)
        p.require(digest.hexdigest()==row['sha256']and l.file_stat(name)==before,'Full original payload bytes/stats')
        l.check_deadline(deadline)
        verified[name]=digest.hexdigest();stats[name]=before
    return dict(verified_sha256=verified,stats=stats)


def supervisor_launch(args):
    env=dict(PATH='/usr/bin:/bin',HOME='/home/ubuntu',LANG='C.UTF-8',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
             PYTHONNOUSERSITE='1',XDG_RUNTIME_DIR='/run/user/1000',DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1000/bus')
    argv=['/usr/bin/python3','-B','-u',str(l.CONTROLLER),'--spec',args.spec,'--spec-sha',args.spec_sha,
          '--review',args.review,'--review-sha',args.review_sha]
    execargv=['/usr/bin/env','-i',*[k+'='+v for k,v in env.items()],*argv]
    command=['systemd-run','--user','--quiet','--unit='+l.UNIT]
    for option in ['MemoryMax=536870912','MemoryHigh=469762048','MemorySwapMax=0','AllowedCPUs=7','CPUAffinity=7',
                   'Nice=19','TasksMax=128','RuntimeMaxSec=86400','TimeoutStopSec=60','KillMode=control-group','SendSIGKILL=yes',
                   'RemainAfterExit=yes','LimitFSIZE=16777216','UMask=0077','WorkingDirectory='+str(p.ROOT),
                   'StandardOutput=append:'+str(l.CONTROL/'controller.log'),'StandardError=inherit']:
        command+=['-p',option]
    return dict(unit=l.UNIT,argv=argv,execargv=execargv,command=command+execargv,environment=env)


def supervisor_loaded_sample(intent,value):
    """Original supervisor predicates applied to this exact observed sample."""
    p.require(value['LoadState']=='loaded'and value['Id']==intent['unit']+'.service'and
              intent['unit']==l.UNIT and l.installed(value)==' '.join(intent['execargv']),'Exact supervisor command')
    p.require(re.fullmatch('[0-9a-f]{32}',value['InvocationID'])and value['InvocationID']!='0'*32,'Supervisor invocation')
    p.require(value['ControlGroup']in('',l.GROUP_PREFIX+l.UNIT+'.service'),'Supervisor group')
    p.require(tuple(int(value[k])for k in('MemoryMax','MemoryHigh','MemorySwapMax'))==(536870912,469762048,0)and
        value['AllowedCPUs']==value['CPUAffinity']=='7'and value['Nice']=='19'and value['TasksMax']=='128'and
        value['KillMode']=='control-group'and value['RemainAfterExit']=='yes'and value['RuntimeMaxUSec']=='1d'and
        value['LimitFSIZE']=='16777216'and value['WorkingDirectory']==str(p.ROOT),'Supervisor exact resources')
    return value


def supervisor_stop_authorization(intent,owner,record):
    """A durable exact authorization is neither successful stop nor exit proof."""
    path=l.CONTROL/'supervisor-service-binding.json'
    p.require(path.exists()and type(record)is dict,'Durable supervisor binding and stop record required')
    binding=l.read(path)
    p.require(record.get('authorized_stop')is True and record.get('intent')==intent and
              record.get('owner')==owner and record.get('binding')==binding,'Exact supervisor stop owner/intent/binding/authorization')
    p.require(record.get('success_proven',False)is False and record.get('successful_exit_proven',False)is False,
              'Stop authorization cannot claim successful original exit')
    value=record.get('properties');p.require(type(value)is dict,'Retained supervisor stop properties required')
    supervisor_loaded_sample(intent,value);l.verify_binding(intent,value,binding)
    if owner is not None:
        p.require(owner['unit']==intent['unit']and owner['argv']==intent['argv']and
                  owner['cgroup']=='0::'+l.GROUP_PREFIX+intent['unit']+'.service','Retained supervisor owner command/group')
        l.verify_retained_binding(intent,owner,binding)
    key=str(path)
    p.require(key not in l._SERVICE_OBSERVATIONS or l._SERVICE_OBSERVATIONS[key]==binding,
              'Supervisor stop differs from first observed invocation')
    return dict(authorized_stop=True,authorization_only=True,binding=copy.deepcopy(binding),
                original_owner=copy.deepcopy(owner),successful_exit_proven=False)


def supervisor_stop_absence(intent,owner,value,origin):
    path=l.CONTROL/'supervisor-stop-intent.json'
    p.require(path.exists(),'Supervisor vanished without durable stop authorization')
    p.require(value['LoadState']=='not-found'and value['Id']==intent['unit']+'.service'and intent['unit']==l.UNIT,
              'Exact missing supervisor observation')
    p.require(l.terminal(value)and value['ActiveState']=='inactive'and value['SubState']=='dead'and
              value['InvocationID']==''and value['ExecMainPID']=='0'and value['ExecMainCode']=='0'and
              value['ExecMainStatus']=='0'and value['ControlGroup']=='','Complete supervisor not-found properties')
    authorization=supervisor_stop_authorization(intent,owner,l.read(path))
    return dict(schema='smollm3_supervisor_authorized_stop_absence_v1',origin=origin,properties=copy.deepcopy(value),
                authorization=authorization,physical_only=True,success_proven=False)


def supervisor_value(intent,owner=None,initial=None,allow_stop_absence=False):
    def classified(value,origin):
        if value['LoadState']=='not-found'and allow_stop_absence:
            raise stop_transition.AuthorizedStopAbsent(supervisor_stop_absence(intent,owner,value,origin))
        return supervisor_loaded_sample(intent,value)
    value=classified(l.props(l.UNIT)if initial is None else initial,'initial')
    binding=l.bind_service(l.CONTROL/'supervisor-service-binding.json',intent,value,owner)
    def refresh():
        current=classified(l.props(l.UNIT),'exit-transition-refresh')
        return l.verify_binding(intent,current,binding)
    if owner:
        p.require(value['InvocationID']==owner['invocation_id'],'Supervisor invocation changed')
        value=l.owned_live_or_retained_exit(owner,value,refresh)
    return l.verify_binding(intent,value,binding)


def terminal_qualification(original,scientific_error,supervisor_exit,controller_terminal,cleanup_errors):
    """A completed matrix and a qualified lifecycle are separate conclusions."""
    protocol_complete=scientific_error is None and controller_terminal.get('protocol_completed') is True
    value=supervisor_exit.get('properties',{})
    original_exit_ok=(supervisor_exit.get('success') is True and value.get('Result')=='success' and
                      value.get('ExecMainCode')=='1' and value.get('ExecMainStatus')=='0')
    lifecycle_ok=(original is None and original_exit_ok and controller_terminal.get('original_error') is None and
                  controller_terminal.get('cleanup_errors')==[] and cleanup_errors==[])
    return dict(protocol_completed=protocol_complete,lifecycle_qualified=lifecycle_ok,
                qualified=protocol_complete and lifecycle_ok,original_supervisor_exit_zero=original_exit_ok)


def supervisor_physical_stop(intent,owner,absence_observations):
    """Stable recursive physical evidence; no default absence implies success."""
    samples=[]
    for index in range(2):
        value=l.props(l.UNIT)
        if value['LoadState']=='not-found':
            absence_observations.append(supervisor_stop_absence(intent,owner,value,'physical-drain-sample-'+str(index)))
        else:
            supervisor_loaded_sample(intent,value)
            binding=l.read(l.CONTROL/'supervisor-service-binding.json');l.verify_binding(intent,value,binding)
            if owner is not None:l.verify_retained_binding(intent,owner,binding)
        p.require(value['Id']==l.UNIT+'.service'and l.terminal(value),'Stopped supervisor must be terminal')
        if owner is not None:l.gone(owner)
        group=l.cgroup_inventory(l.GROUP_PREFIX+l.UNIT+'.service',require_empty=True)
        samples.append(dict(properties=value,group=group))
        if index==0:time.sleep(.05)
    p.require(samples[0]==samples[1],'Stable recursive supervisor stop drain')
    return dict(schema='smollm3_supervisor_stop_drain_v1',drained=True,physical_only=True,success_proven=False,
                original_owner=copy.deepcopy(owner),samples=samples,stop_absence_observations=absence_observations)


def stop_supervisor(intent,owner):
    absence_observations=[]
    try:
        value=supervisor_value(intent,owner,initial=l.props(l.UNIT),allow_stop_absence=True)
        if not l.terminal(value)or l.cgroup_inventory(l.GROUP_PREFIX+l.UNIT+'.service')['pids']:
            path=l.CONTROL/'supervisor-stop-intent.json'
            binding=l.read(l.CONTROL/'supervisor-service-binding.json')
            if not path.exists():l.record(path,dict(owner=owner,intent=intent,properties=value,binding=binding,authorized_stop=True))
            else:supervisor_stop_authorization(intent,owner,l.read(path))
            supervisor_value(intent,owner,allow_stop_absence=True)
            l.command(['systemctl','--user','stop','--no-block',l.UNIT]);deadline=time.monotonic()+75;forced=False
            while True:
                # Each authenticated value is the classified sample, including
                # any bounded owner-exit refresh; no discarded/stale outer read.
                value=supervisor_value(intent,owner,allow_stop_absence=True)
                group=l.cgroup_inventory(l.GROUP_PREFIX+l.UNIT+'.service')
                if l.terminal(value)and not group['pids']:break
                if not forced and time.monotonic()>deadline-15:
                    value=supervisor_value(intent,owner,allow_stop_absence=True)
                    group=l.cgroup_inventory(l.GROUP_PREFIX+l.UNIT+'.service')
                    if l.terminal(value)and not group['pids']:break
                    l.command(['systemctl','--user','kill','--kill-who=all','--signal=KILL',l.UNIT]);forced=True
                p.require(time.monotonic()<deadline,'Supervisor bounded drain');time.sleep(.2)
    except stop_transition.AuthorizedStopAbsent as transition:
        # Only exact authorized disappearance reaches this physical-only path.
        # Identity/resource/record/command/drain failures remain errors.
        absence_observations.append(transition.observation)
    return supervisor_physical_stop(intent,owner,absence_observations)


def planned_physical_drain(spec):
    """Two stable recursive samples for all24 planned workers and supervisor."""
    observations=[]
    for name in spec['planned_units']:
        samples=[]
        for index in range(2):
            value=l.props(name)
            p.require(value['Id']==name+'.service'and l.terminal(value),'Actual planned service remains live or changed')
            group=l.cgroup_inventory(l.GROUP_PREFIX+name+'.service',require_empty=True)
            samples.append(dict(properties=value,group=group))
            if index==0:time.sleep(.02)
        p.require(samples[0]==samples[1],'Stable all-planned physical drain')
        observations.append(dict(unit=name,samples=samples,physical_only=True,successful_execution_proven=False))
    return observations


def catalog_owner(owner, script):
    """Keep historical row schema while retaining full captured identity separately."""
    if owner is None:return None
    return dict(pid=owner['pid'],start_ticks=owner['start_ticks'],unit=owner['unit'],script=str(script))


def append_catalog(spec,external,supervisor_owner,intent):
    """All planned names remain present, including never-launched failures."""
    catalog,rows,processes=predecessor_catalog(spec);rows=copy.deepcopy(rows);processes=copy.deepcopy(processes)
    for run in p.matrix():
        for kind in ('server','client'):
            root=p.directory(run['run_id']);name=p.unit(run['run_id'],kind);launch=root/(kind+'-launch.json');owner_path=root/(kind+'-owner.json')
            owner=l.read(owner_path)if owner_path.exists()else None
            row=dict(unit=name,recorded_cgroups=['/sys/fs/cgroup'+l.GROUP_PREFIX+name+'.service'],owner=catalog_owner(owner,l.WRAPPER),
                     captured_process_owner=owner,
                     never_launched=not launch.exists(),invocation_id=None if owner is None else owner['invocation_id'],
                     installed_argv=None if not launch.exists()else' '.join(l.read(launch)['execargv']),
                     historical_invocation_unknown=False,absent_records=[str(launch),str(owner_path)]if not launch.exists()else[],
                     runtime_evidence_root=str(root),successful_execution_proven=False)
            # A launch may fail before owner capture. Preserve the real gap; root
            # must audit/normalize it with actual invocation/exit evidence before
            # this successor catalog is accepted by another admission.
            if launch.exists()and owner is None:row['launch_without_authenticated_owner']=True
            rows.append(row)
    rows.append(dict(unit=l.UNIT,recorded_cgroups=['/sys/fs/cgroup'+l.GROUP_PREFIX+l.UNIT+'.service'],owner=catalog_owner(supervisor_owner,l.CONTROLLER),
        captured_process_owner=supervisor_owner,
        never_launched=intent is None,invocation_id=None if supervisor_owner is None else supervisor_owner['invocation_id'],
        installed_argv=None if intent is None else' '.join(intent['execargv']),historical_invocation_unknown=False,
        absent_records=[str(l.CONTROL/'supervisor-launch.json'),str(l.CONTROL/'supervisor-owner.json')]if intent is None else[],
        launch_without_authenticated_owner=intent is not None and supervisor_owner is None))
    processes.append(dict(external,role='external_launcher_and_final_auditor',self_exit_not_yet_observed=True,
                          next_admission_must_prove_gone=True))
    result=copy.deepcopy(catalog)
    p.require('historical_payload_sha256' not in result or result['historical_payload_sha256']==spec['historical_payload_sha256'], 'No conflicting historical payload provenance')
    result['historical_payload_sha256']=copy.deepcopy(spec['historical_payload_sha256'])
    pins=result.setdefault('source_sha256',{})
    additions=dict(spec['runtime_source_sha256'])
    ref=spec['final_predecessor_catalog'];additions[ref['path']]=ref['sha256']
    # Only small durable metadata is newly pinned; raw/model payloads retain
    # their separate complete byte proof and are never folded into source pins.
    for path in sorted(p.DEST.rglob('*.json')):
        # The catalog cannot include its own transport dependencies. They live
        # in a separate namespace and are authenticated by every catalog read;
        # root finalization must retain them with authenticated_chunks().
        if path.is_relative_to(l.CONTROL/'catalog-maps'):continue
        if path.name.startswith('.pending-')or path.name in('successor-catalog.json','external-terminal.json','external-release.json','external-recovery-required.json')or path.name.startswith('recovery-release-'):continue
        raw=l.read_bytes(path);additions[str(path)]=l.digest_bytes(raw)
    for name,digest in additions.items():
        p.require(name not in pins or pins[name]==digest,'Conflicting retained catalog source pin')
        pins[name]=digest
    result.update(schema='smollm3_observer_invariance_successor_catalog_v6',rows=rows,processes=processes,
        complete_count=len(rows),added_planned_units=25,previous_catalog=spec['final_predecessor_catalog'],
        all_planned_launches_accounted_for=True,root_finalized=False,requires_root_terminal_review=True,
        physical_absence_is_not_success=True,external_self_exit_claimed=False,new_source_and_evidence_pins=additions)
    if 'process_predecessors'in result:result['process_predecessors']=copy.deepcopy(processes)
    return result



def execute(args):
    p.performance_guard();p.require(sys.dont_write_bytecode and __debug__ and os.getuid()==1000,'Use -B/no-O uid1000')
    spec,plan,base,handoff=l.load_spec(args.spec,args.spec_sha)
    accepted=l.reference(dict(path=args.review,sha256=args.review_sha))
    p.require(accepted['reviewer']=='/root'and accepted['passed']is accepted['independent_review']is True and
        accepted['spec_sha256']==args.spec_sha and accepted['runtime_source_sha256']==spec['runtime_source_sha256'],'Future independent root runtime acceptance')
    callable_acceptance=l.reference(spec['callable_root_acceptance'])
    p.require(callable_acceptance['passed']is True and callable_acceptance['reviewer']=='/root'and
              callable_acceptance['source_handoff']==spec['callable_handoff']and callable_acceptance['source_only']is True,'Exact root callable acceptance')
    external=l.process(os.getpid());lock_fd=os.open(l.LOCK,os.O_RDWR|os.O_NOFOLLOW)
    created=False;intent=None;supervisor_owner=None;original=None;release=False;final=None
    def interrupted(signum,frame):raise RuntimeError('External owner signal '+str(signum))
    try:
        fcntl.flock(lock_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        lock_stat=os.fstat(lock_fd);lock=dict(path=str(l.LOCK),device=lock_stat.st_dev,inode=lock_stat.st_ino)
        l.lock_proof(external,lock)
        for sig in(signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,interrupted)
        p.require(not p.DEST.exists()and not p.DEST.is_symlink(),'Never reuse sio12 namespace')
        prior=drain_predecessors(spec)
        for path,floor in ((p.ROOT,30*l.GIB),(Path('/mnt/nvme'),5*l.GIB)):
            st=os.statvfs(path);p.require(st.f_bavail*st.f_frsize>=floor,'Admission storage')
        p.DEST.mkdir(mode=0o700);l.CONTROL.mkdir(mode=0o700);created=True
        l.metadata_store.provision(p.DEST)
        resource.setrlimit(resource.RLIMIT_AS,(512*l.MIB,512*l.MIB))
        resource.setrlimit(resource.RLIMIT_FSIZE,(16*l.MIB,16*l.MIB))
        os.sched_setaffinity(0,{7});os.setpriority(os.PRIO_PROCESS,0,19)
        logfd=os.open(l.CONTROL/'external.log',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        os.dup2(logfd,1);os.dup2(logfd,2);os.close(logfd)
        l.record(l.CONTROL/'external-owner.json',external)
        l.record(l.CONTROL/'predecessor-drain.json',prior)
        for name,wanted in spec['canonical_sha256'].items():p.require(l.digest_bytes(p.bounded_source(p.ROOT/name))==wanted,'Admission canonical snapshot')
        source_stats=l.source_closure(plan,spec['runtime_source_sha256'],handoff['original_metadata_symlink_resolutions'])
        p.old.serialization_review(base['serialization_root_review'])
        payload=payload_proof(base,time.monotonic()+1800)
        l.record(l.CONTROL/'source-admission.json',dict(source_stats=source_stats,source_pins_verified=True))
        l.record(l.CONTROL/'payload-admission.json',payload)
        started=time.monotonic()
        envelope=dict(external_owner=external,lock=lock,source_stats=source_stats,payload_stats=payload['stats'],
                      matrix_started_monotonic=started,spec_sha256=args.spec_sha,root_review=dict(path=args.review,sha256=args.review_sha),
                      proofs={key:True for key in ('independent_root_source_review','all_source_pins_verified','all_payloads_verified',
                        'all_final_predecessors_drained','serial_lock_proven','live_service_identity_verified','recursive_output_monitor_active',
                        'previous_run_fully_drained','fresh_run_namespace','actual_tiny_serialization_verified')})
        l.record(l.CONTROL/'external-admission.json',envelope)
        p.require(l.props(l.UNIT)['LoadState']=='not-found','Fresh supervisor unit')
        intent=supervisor_launch(args);l.record(l.CONTROL/'supervisor-launch.json',intent);l.command(intent['command'],timeout=30)
        deadline=time.monotonic()+45
        while supervisor_owner is None:
            value=supervisor_value(intent)
            if value['MainPID']!='0':
                try:actual=l.process(int(value['MainPID']))
                except FileNotFoundError:actual=None
                if actual and actual['argv']==intent['argv']:
                    p.require(actual['cgroup']=='0::'+l.GROUP_PREFIX+l.UNIT+'.service','Supervisor actual cgroup')
                    supervisor_owner=dict(actual,unit=l.UNIT,invocation_id=value['InvocationID'])
                    l.record(l.CONTROL/'supervisor-owner.json',supervisor_owner)
                    envelope['supervisor']=actual;l.record(l.CONTROL/'supervisor-owner-bundle.json',envelope)
            p.require(time.monotonic()<deadline and not l.terminal(value),'Supervisor owner ACK failed');time.sleep(.1)
        while True:
            value=supervisor_value(intent,supervisor_owner)
            run_id='r00'
            for run in p.matrix():
                if(p.directory(run['run_id'])/'run-admission.json').exists():run_id=run['run_id']
            l.monitor(run_id)
            l.lock_proof(external,lock)
            if l.terminal(value):
                l.record(l.CONTROL/'supervisor-original-exit.json',dict(owner=supervisor_owner,properties=value,
                    success=value['Result']=='success'and value['ExecMainCode']=='1'and value['ExecMainStatus']=='0'))
                break
            p.require(time.monotonic()<started+86400,'External matrix24h deadline')
            admission_path=p.directory(run_id)/'run-admission.json'
            if admission_path.exists():p.require(time.monotonic()<l.read(admission_path)['run_started_monotonic']+7200,'External shared run7200s deadline')
            time.sleep(.25)
    except BaseException as error:
        original=dict(type=type(error).__name__,message=str(error)[:4096])
    finally:
        if created:
            audit_deadline=time.monotonic()+1800
            for sig in(signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,signal.SIG_IGN)
            cleanup_errors=[];drains=[]
            from smollm3_observer_invariance_controller_v23 import cleanup_all
            try:
                with l.bounded_phase(audit_deadline):
                    # Preserve the original error first, then attempt every owned stop.
                    try:l.record(l.CONTROL/'external-original-result.json',dict(original_error=original,independent_audit_started=True))
                    except BaseException as error:
                        if isinstance(error,l.DeadlineExceeded):raise
                        cleanup_errors.append(dict(scope='error_receipt',error=str(error)))
                    if intent:
                        try:stop_supervisor(intent,supervisor_owner)
                        except BaseException as error:
                            if isinstance(error,l.DeadlineExceeded):raise
                            cleanup_errors.append(dict(scope='supervisor_stop',error=str(error)))
                    drains,errors=cleanup_all();cleanup_errors.extend(errors)
                    l.check_deadline(audit_deadline)
                    # Physical release is separate from scientific success. Final
                    # verification iterates all planned names, including unlaunched.
                    planned_drains=planned_physical_drain(spec)
                    if supervisor_owner:l.gone(supervisor_owner)
                    captures={};evidence={};exports={};scientific_error=None;terminal={};supervisor_exit={}
                    try:
                        supervisor_exit=l.read(l.CONTROL/'supervisor-original-exit.json')
                        p.require(supervisor_exit['owner']==supervisor_owner,'Retained supervisor original exit owner')
                        value=supervisor_exit['properties']
                        l.verify_retained_binding(intent,supervisor_owner,l.read(l.CONTROL/'supervisor-service-binding.json'))
                        p.require(value['ExecMainPID']==str(supervisor_owner['pid'])and value['InvocationID']==supervisor_owner['invocation_id']and
                            l.installed(value)==' '.join(intent['execargv']),'Original supervisor exit full identity')
                        terminal=l.read(l.CONTROL/'supervisor-terminal.json')
                        p.require(terminal['original_error']is None and terminal['protocol_completed']is True,'Original controller failure retained')
                        for run in p.matrix():
                            p.require(time.monotonic()<audit_deadline,'Independent export audit deadline')
                            c,e,x=audit_run(base,run['run_id'],write_export=False);captures[run['run_id']]=c;evidence[run['run_id']]=e
                            if x is not None:exports[run['run_id']]=x
                        result=analysis.compare_matrix(base,captures,evidence,exports)
                        p.require(result==l.read(l.CONTROL/'matrix-result.json'),'Independent complete matrix reanalysis')
                    except BaseException as error:
                        if isinstance(error,l.DeadlineExceeded):raise
                        scientific_error=dict(type=type(error).__name__,message=str(error)[:4096])
                    failures=failed_evidence()
                    p.require(time.monotonic()<audit_deadline,'Final source/payload audit deadline')
                    final_source_stats=l.source_closure(plan,spec['runtime_source_sha256'],handoff['original_metadata_symlink_resolutions'],deadline=audit_deadline)
                    final_payload=payload_proof(base,audit_deadline)
                    l.record(l.CONTROL/'final-identity-rehash.json',dict(source_stats=final_source_stats,
                        payload_sha256=final_payload['verified_sha256'],full_source_rehash=True,full_payload_rehash=True,
                        within_external_audit_seconds=1800,measurement_window=False))
                    flags=terminal_qualification(original,scientific_error,supervisor_exit,terminal,cleanup_errors)
                    if failures['evidence_audit_errors']:
                        flags['lifecycle_qualified']=False;flags['qualified']=False
                    catalog=append_catalog(spec,external,supervisor_owner,intent)
                    l.record(l.CONTROL/'successor-catalog.json',catalog)
                    final=dict(schema='smollm3_observer_invariance_external_terminal_v3',original_error=original,
                        original_controller_error=terminal.get('original_error'),original_controller_cleanup_errors=terminal.get('cleanup_errors'),
                        scientific_error=scientific_error,cleanup_errors=cleanup_errors,request_failure_audit=failures,
                        **flags,
                        observer_api_invariance_passed=result['observer_api_invariance_passed']if scientific_error is None else None,
                        all_planned_groups_drained=True,planned_physical_drains=planned_drains,external_owner=external,external_self_exit_not_claimed=True,
                        successor_catalog=str(l.CONTROL/'successor-catalog.json'),full_model_qualified=False,new_speedup_claim=False)
                    l.record(l.CONTROL/'external-terminal.json',final);l.monitor('r11',reserve=False)
                    for name,wanted in spec['canonical_sha256'].items():p.require(l.digest_bytes(p.bounded_source(p.ROOT/name))==wanted,'Final canonical unchanged')
                    l.record(l.CONTROL/'external-release.json',dict(released=True,failed=not final['qualified'],
                        all_planned_groups_drained=True,external_owner=external,next_admission_must_verify_external_exit=True))
                release=True
            except BaseException as error:
                try:l.record(l.CONTROL/'external-recovery-required.json',dict(original_error=original,audit_error=str(error)[:4096],
                    cleanup_errors=cleanup_errors,serial_lock_retained=True,automatic_successor_launch=False,external_owner=external))
                except BaseException:pass
                # No live group can be abandoned just because evidence writes
                # failed. Operator/root must resolve and request a new bounded
                # physical audit; no inference experiment is retried here.
                requested=[False]
                def retry(signum,frame):requested[0]=True
                for sig in(signal.SIGUSR1,signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,retry)
                while not release:
                    while not requested[0]:time.sleep(.2)
                    requested[0]=False
                    try:
                        with l.bounded_phase(time.monotonic()+1800):
                            if intent:stop_supervisor(intent,supervisor_owner)
                            cleanup_all()
                            planned_drains=planned_physical_drain(spec)
                            catalog=append_catalog(spec,external,supervisor_owner,intent)
                            catalog_path=l.CONTROL/'successor-catalog.json'
                            if not catalog_path.exists():l.record(catalog_path,catalog)
                            else:p.require(l.read(catalog_path)==catalog,'Recovery planned catalog must preserve original records')
                            index=len(list(l.CONTROL.glob('recovery-release-*.json')))
                            l.record(l.CONTROL/('recovery-release-'+str(index)+'.json'),dict(released=True,failed=True,original_error=original,
                                audit_error=str(error)[:4096],all_planned_groups_drained=True,planned_physical_drains=planned_drains,external_owner=external,
                                next_admission_must_verify_external_exit=True,scientific_success=False))
                        release=True
                        if final is not None:
                            final['qualified']=False;final['lifecycle_qualified']=False
                    except BaseException:pass
        os.close(lock_fd)
    return final


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true',required=True)
    for name in('spec','spec-sha','review','review-sha'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();result=execute(args)
    if result is None or not result['qualified']:raise SystemExit(1)


if __name__=='__main__':main()
