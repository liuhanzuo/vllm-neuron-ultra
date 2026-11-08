"""Exact completed r3 history; retains the separately proved r2 missing ACK."""
import hashlib
import json
from pathlib import Path
import re
import sys
import minicpm5_frame1_completed_history_v2 as previous
require=previous.require;R=previous.R;Q=previous.Q
PATH=Q/'minicpm5-frame1-r3-complete-catalog-20261108-v1.json'
PRIOR=dict(path=str(previous.PATH),sha256='d3941454ea1c95f4b79d96887ec9203b4a8261f0f14503d5a3e5a6f1e3842993')
TERMINAL=dict(path=str(Q/'minicpm5-frame1-r3-root-terminal-20261108-v1.json'),sha256='6dd0904acb31baa784905bb3b6ec74dbf17e3e390c432e16f16e7447b341e5ef')
NON_UTF8=previous.NON_UTF8;python_source_bytes=previous.python_source_bytes

def added_rows(terminal):
    result=[]
    for sample in terminal['physical_drain_samples'][0]:
        phase=sample['phase'];owner=terminal['phase_owners'][phase];unit=owner['unit']
        argv=re.findall(r'argv\[\]=(.*?)\s*;',sample['properties']['ExecStart']);require(len(argv)==1,'Actual installed phase argv')
        result.append(dict(unit=unit,owner=dict(pid=owner['pid'],start_ticks=owner['start_ticks'],unit=unit,script=str(R/'minicpm5_frame1_mlp_runner_v5.py')),
            captured_process_owner=owner,installed_argv=argv[0],recorded_cgroups=['/sys/fs/cgroup'+owner['cgroup'].split('::',1)[1]],
            historical_invocation_unknown=False,invocation_id=owner['invocation_id'],never_launched=False,absent_records=[]))
    return result

def validate(value):
    prior=previous.load(PRIOR);terminal=previous.load(TERMINAL)
    require(value['schema']=='root_post_minicpm5_frame1_r3_complete_predecessors_v1'and value['reviewer']=='/root','Actual r3 complete history')
    require(value['rows'][:922]==prior['rows']and value['rows'][922:]==added_rows(terminal)and value['complete_count']==len(value['rows'])==925,
            'All original services plus exactly three completed phases')
    require(value['processes'][:173]==prior['processes']and len(value['processes'])==179,'All original processes plus six actual owners')
    audit=previous.load(value['appended_terminal_audit'])
    require(audit['terminal']==TERMINAL and audit['actual_model_exit']==terminal['actual_exit']==dict(session_id=15892,exit_code=0,tool_chunk='aa5edd')and
            audit['actual_review_exit']==dict(session_id=14042,exit_code=0,tool_chunk='c537bd'),'Actual observed exits')
    review=previous.load(audit['independent_review']);require(review['complete']is True,'Actual independent review complete')
    expected=[dict(terminal['outer_owner'],role='minicpm5_frame1_r3_outer'),dict(terminal['terminal_observer'],role='minicpm5_frame1_r3_terminal_observer')]
    expected += [dict(owner,role='minicpm5_frame1_r3_cpu_rank'+str(i))for i,owner in enumerate(terminal['rank_owners'])]
    expected += [dict(review['owner'],role='minicpm5_frame1_r3_independent_review'),dict(audit['finalizer_owner'],role='minicpm5_frame1_r3_history_finalizer')]
    require(value['processes'][173:]==expected,'Exactly all known new owners')
    require(value['appended_from']==PRIOR and value['root_release_refs']==prior['root_release_refs']and
            value['root_finalized']is True and value['all_planned_launches_accounted_for']is True and value['runtime_physical_drain_required']is True,
            'Actual complete append, fresh runtime drain still required')
    require(value['latest_producer']=='minicpm5_frame1_r3'and value['latest_producer_qualified']is False and
            value['latest_producer_exit']==value['latest_producer_terminal_audit']==value['appended_terminal_audit'],
            'Successful diagnostic cannot become model qualification')
    require(all(value['source_sha256'].get(k)==v for k,v in prior['source_sha256'].items()),'No historical source replacement')
    for ref in [PRIOR,TERMINAL,value['appended_terminal_audit'],audit['independent_review'],review['metadata_review'],review['raw_review']]:
        require(value['source_sha256'].get(ref['path'])==ref['sha256'],'Actual retained completion evidence')
    if 'process_predecessors'in value:require(value['process_predecessors']==value['processes'],'Exact process alias')
    changed={'schema','rows','processes','process_predecessors','source_sha256','complete_count','appended_from','appended_terminal_audit','latest_producer','latest_producer_qualified','latest_producer_exit','latest_producer_terminal_audit'}
    require(set(value)==set(prior)and all(value[k]==v for k,v in prior.items()if k not in changed),'Only latest completed history changed')
    import smollm3_token_capture_predecessors_v7 as validator
    validator.validate_rows([row for row in value['rows']if row['unit']!=previous.UNIT],value['processes'])
    return value

def text(path,expected=None):
    path=Path(path)
    if path!=PATH:return previous.text(path,expected)
    require(type(expected)is str and re.fullmatch('[0-9a-f]{64}',expected),'Caller pins exact completed r3 history')
    previous.previous.original.low.clear_window()
    raw=previous.previous.original.stable_bytes(path,previous.previous.original.ENVELOPE_LIMIT)
    require(hashlib.sha256(raw).hexdigest()==expected,'Exact r3 history bytes');validate(json.loads(raw));return raw

def sha(path,expected=None):return hashlib.sha256(text(path,expected)).hexdigest()

def load(reference):
    if Path(reference['path'])!=PATH:return previous.load(reference)
    require(set(reference)in ({'path','sha256'},{'path','sha256','bytes'}),'Exact catalog reference')
    raw=text(reference['path'],reference['sha256'])
    require('bytes'not in reference or reference['bytes']==len(raw),'Exact optional size');return json.loads(raw)

def physical_drain(value):
    validate(value);observed=previous.physical_drain(previous.load(PRIOR))
    if str(Q)not in sys.path:sys.path.insert(0,str(Q))
    from research.finalize_smollm3_sio14_root_20261108_v1 import physical_drain as drain
    return observed+drain(dict(rows=value['rows'][922:],processes=value['processes'][173:]))
