"""Exact Mini r2 failed-start history with explicitly uncaptured native owner.

The one missing-ACK row is never called never-launched. No PID is fabricated.
Original predecessor validation/drain remains unchanged for every other row.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys
import minicpm5_frame1_completed_history_v1 as previous
from minicpm5_layer0_runtime_auth_v1 import require

R=Path(__file__).resolve().parent;Q=R/'research'
PRIOR=dict(path=str(previous.PATH),sha256=previous.SHA)
TERMINAL=dict(path=str(Q/'minicpm5-frame1-r2-failed-startup-root-terminal-20261108-v1.json'),sha256='6bee2dff7853980515f1e55af56d541841e81c36c302870e9e71273759c47b04')
PATH=Q/'minicpm5-frame1-r2-complete-catalog-20261108-v1.json'
FINALIZER=Q/'finalize_minicpm5_frame1_r2_history_v1.py'
UNIT='trn1-mini-f1-mlp-r2-20261108-v1-native'
INVOCATION='b51e6e64bcc745029fa46edfea183802'
NON_UTF8=previous.NON_UTF8
python_source_bytes=previous.python_source_bytes
GROUP_PREFIX='/user.slice/user-1000.slice/user@1000.service/app.slice/'


def ref(path):
    raw=previous.original.stable_bytes(Path(path),previous.original.ENVELOPE_LIMIT)
    return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest())


def evidence():
    terminal=previous.load(TERMINAL)
    require(terminal['actual_exit']==dict(session_id=29335,exit_code=1,tool_chunk='b9d958')and
            terminal['actual_tool_exit_observed']is True and terminal['exact_outer_owner_gone']is True,
            'Actual failed outer exit')
    require(terminal['native_service_was_started']is True and terminal['native_service_owner_not_captured']is True and
            terminal['native_service_invocation_id']==INVOCATION and terminal['native_owner_ack_absent']is True,
            'Started native service with explicitly missing owner ACK')
    require(terminal['full_history_catalog_created']is False and terminal['all_three_groups_physically_drained']is True and
            terminal['full_model_qualified']is False and terminal['new_speedup_claim']is False,'Original incomplete failed history remains failed')
    admission=previous.load(terminal['admission']);failure=previous.load(terminal['failure']);spec=previous.load(terminal['spec'])
    require(admission['outer_owner']==failure['owner']==terminal['outer_owner']and admission['spec']==terminal['spec']and
            admission['review']==terminal['review']and admission['current_catalog']==PRIOR,'Original producer and admission chain')
    require(failure['phase']=='native'and failure['error']=='Only exact transient env exec may wait','Original primary failure')
    journal=previous.text(terminal['native_journal']['path'],terminal['native_journal']['sha256'])
    rows=[json.loads(line)for line in journal.decode().splitlines()]
    require(len(rows)==4 and [row['MESSAGE'].split(' ',1)[0]for row in rows]==['Starting','Started','Stopping','Stopped'],
            'Original complete four service journal events')
    require(all(row['USER_UNIT']==UNIT+'.service'and row['USER_INVOCATION_ID']==INVOCATION and
                row['_PID']=='1583' for row in rows),'One actual invocation; manager PID is not native owner')
    require(int(rows[-1]['__REALTIME_TIMESTAMP'])-int(rows[1]['__REALTIME_TIMESTAMP'])==16323,
            'Original journal interval, not a performance measurement')
    require(spec['units']=={phase:'trn1-mini-f1-mlp-r2-20261108-v1-'+phase for phase in ('native','cpu','replay')},'Three original planned units')
    samples=terminal['physical_drain_samples'];require(len(samples)==2 and samples[0]==samples[1],'Original two equal physical snapshots')
    for row in samples[0]:
        require(row['properties']['LoadState']=='not-found'and row['properties']['MainPID']==row['properties']['ControlPID']=='0'and
                not row['properties']['InvocationID'] and row['group']['absent']is True,'Original absent service/group')
    return terminal,admission,spec


def new_rows(terminal,spec):
    rows=[];root=Path(spec['output_root'])
    for phase in ('native','cpu','replay'):
        unit=spec['units'][phase];common=dict(unit=unit,recorded_cgroups=['/sys/fs/cgroup'+GROUP_PREFIX+unit+'.service'],
            owner=None,captured_process_owner=None,installed_argv=None)
        if phase=='native':
            rows.append(dict(common,historical_invocation_unknown=False,invocation_id=INVOCATION,never_launched=False,
                absent_records=[],owner_capture_state='started_before_owner_ack',owner_identity_unknown=True,
                journal_invocation_only=True,failed_startup_terminal=TERMINAL,native_journal=terminal['native_journal'],
                scientific_success=False))
        else:
            rows.append(dict(common,historical_invocation_unknown=False,invocation_id=None,never_launched=True,
                absent_records=[str(root/(phase+'-owner.json')),str(root/(phase+'.log'))]))
    return rows


def validate_catalog(value,prior,terminal,spec):
    require(value['schema']=='root_post_minicpm5_frame1_r2_complete_predecessors_v1'and value['reviewer']=='/root',
            'Explicit current failed startup catalog')
    require(value['rows'][:919]==prior['rows']and value['processes'][:170]==prior['processes'],'Every original service and process retained')
    require(value['rows'][919:]==new_rows(terminal,spec)and len(value['rows'])==value['complete_count']==922,'Exact three new stage rows')
    require(len(value['processes'])==173,'All actual known new owners, including finalizer')
    wanted=[(terminal['outer_owner'],'minicpm5_frame1_r2_outer'),(terminal['terminal_observer'],'minicpm5_frame1_r2_terminal_observer')]
    for row,(owner,role)in zip(value['processes'][170:172],wanted):require(row==dict(owner,role=role),'Original known process identity')
    last=value['processes'][-1]
    require(last['role']=='minicpm5_frame1_r2_history_finalizer'and last['argv'][1:]==['-B',str(FINALIZER)]and
            type(last['pid'])is int and last['pid']>0 and last['start_ticks'].isdigit(),'Actual finalizer process')
    require(value['root_finalized']is True and value['all_planned_launches_accounted_for']is True and
            value['latest_producer_qualified']is False and value['latest_producer']=='minicpm5_frame1_r2', 'Failed history cannot become success')
    require(value['appended_from']==PRIOR and value['appended_terminal_audit']==TERMINAL and value['latest_producer_exit']==TERMINAL and
            value['latest_producer_terminal_audit']==TERMINAL,'Actual retained failed references')
    require(value['root_release_refs']==prior['root_release_refs'],'No invented experiment release')
    require(all(value['source_sha256'].get(k)==v for k,v in prior['source_sha256'].items()),'All historical source pins retained')
    required_refs=[PRIOR,TERMINAL,terminal['spec'],terminal['review'],terminal['live'],terminal['admission'],
                   terminal['failure'],terminal['native_journal'],ref(FINALIZER),ref(Path(__file__))]
    require(all(value['source_sha256'].get(item['path'])==item['sha256']for item in required_refs),
            'Current evidence and exact finalizer/reader source retained')
    changed={'schema','rows','processes','process_predecessors','source_sha256','complete_count','appended_from','appended_terminal_audit',
             'latest_producer','latest_producer_qualified','latest_producer_exit','latest_producer_terminal_audit',
             'root_finalized','all_planned_launches_accounted_for'}
    added={'minicpm5_frame1_r2_append'}
    require(set(value)==set(prior)|added and all(value[k]==v for k,v in prior.items()if k not in changed),'Only current failure appended')
    if 'process_predecessors'in value:require(value['process_predecessors']==value['processes'],'Complete process alias')
    require(value['minicpm5_frame1_r2_append']==dict(services=value['rows'][919:],processes=value['processes'][170:],
            terminal=TERMINAL,owner_capture_gap_retained=True,full_model_qualified=False),'Explicit failed append')
    import smollm3_token_capture_predecessors_v7 as validator
    # The exact missing-ACK row is validated above; every other row uses the
    # unchanged legacy validator, including all historical unknown-start cases.
    validator.validate_rows([row for row in value['rows']if row['unit']!=UNIT],value['processes'])
    return True


def text(path,expected=None):
    path=Path(path)
    if path!=PATH:return previous.text(path,expected)
    previous.original.low.clear_window()
    require(type(expected)is str and len(expected)==64,'Caller pins actual complete r2 catalog')
    raw=previous.original.stable_bytes(path,previous.original.ENVELOPE_LIMIT);raw.decode('utf8')
    require(hashlib.sha256(raw).hexdigest()==expected,'Exact complete r2 history bytes')
    terminal,_,spec=evidence();validate_catalog(json.loads(raw),previous.load(PRIOR),terminal,spec)
    return raw


def sha(path,expected=None):return hashlib.sha256(text(path,expected)).hexdigest()


def load(reference):
    if Path(reference['path'])!=PATH:return previous.load(reference)
    require(set(reference)in ({'path','sha256'},{'path','sha256','bytes'}),'Exact current catalog reference')
    raw=text(reference['path'],reference['sha256'])
    require('bytes'not in reference or type(reference['bytes'])is int and reference['bytes']==len(raw),'Exact optional byte count')
    return json.loads(raw)


def physical_drain(candidate):
    terminal,_,spec=evidence();validate_catalog(candidate,previous.load(PRIOR),terminal,spec)
    research_path=str(Q)
    if research_path not in sys.path:sys.path.insert(0,research_path)
    from research.finalize_smollm3_sio14_root_20261108_v1 import physical_drain as legacy_drain,life
    # Neither exclusion from the legacy parser nor physical absence implies
    # never-launched. The exact missing-ACK invocation is separately drained.
    normal=[row for row in candidate['rows']if row['unit']!=UNIT]
    observed=legacy_drain(dict(rows=normal,processes=candidate['processes']))
    samples=[]
    for _ in range(2):
        props=life.props(UNIT)
        require(life.terminal(props)and props['Id']==UNIT+'.service'and props['LoadState']=='not-found'and
                not props['InvocationID']and props['MainPID']==props['ControlPID']==props['ExecMainPID']=='0',
                'Unacknowledged service must now be entirely absent, never a replacement invocation')
        group=life.cgroup_inventory(GROUP_PREFIX+UNIT+'.service',require_empty=True)
        samples.append(dict(properties=props,groups=[group]))
    require(samples[0]==samples[1],'Stable missing-ACK service physical absence')
    observed.append(dict(unit=UNIT,samples=samples,owner_identity_unknown=True,scientific_success=False))
    by_unit={row['unit']:row for row in observed};require(set(by_unit)=={row['unit']for row in candidate['rows']},'All922groups checked')
    return [by_unit[row['unit']]for row in candidate['rows']]
