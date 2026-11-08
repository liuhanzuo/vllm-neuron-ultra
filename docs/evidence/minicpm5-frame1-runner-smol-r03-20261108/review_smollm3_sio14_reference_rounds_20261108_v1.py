"""Read-only actual completed reference-round audit; never grants runtime admission."""
from pathlib import Path
import datetime
import hashlib
import itertools
import json
import sys
ROOT=Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
sys.path.insert(0,str(ROOT))
from smollm3_observer_metadata_store_v4 import decode
BASE=ROOT/'sio14'


def reference(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(p):return decode(BASE,p,p.read_bytes())
def require(value,message):
    if not value:raise RuntimeError(message)
def gone(owner):
    stat=Path('/proc')/str(owner['pid'])/'stat'
    if stat.exists():require(stat.read_text().rsplit(')',1)[1].split()[19]!=owner['start_ticks'],'Original owner still alive')
    group=Path('/sys/fs/cgroup')/owner['cgroup'].split(':',2)[2].lstrip('/')
    if group.exists():
        for p in [group/'cgroup.procs',*group.rglob('cgroup.procs')]:
            if p.exists():require(not p.read_text().split(),'Original group still populated')
    return True


def run():
    rounds=[];requests=[]
    for index in range(4):
        name='r'+str(index).zfill(2);directory=BASE/name
        terminal=read(directory/'client-terminal.json');server=read(directory/'server-terminal.json')
        props=terminal['original_properties']
        require(terminal['passed']is True and props['Result']=='success' and str(props['ExecMainCode'])=='1' and str(props['ExecMainStatus'])=='0','Actual original client exit0')
        require(terminal['drain']['drained']is True and server['drained']is True and server['authorized_stop']is True,'Both original terminal drains')
        require(server['physical_only']is True and server['success_proven']is False,'Preserve physical-only server outcome')
        gone(terminal['owner']);gone(server['original_owner'])
        capture_path=directory/'capture-result.json';capture=read(capture_path)
        require(capture['passed']is True and capture['protocol_passed']is True and capture['diagnostic_complete']is True,'Original capture completed')
        require(capture['run']['run_id']==name and capture['role']=='reference','Original reference configuration')
        require(capture['run']['mode']==('off','on','on','off')[index],'Original off/on/on/off order')
        require(capture['request_scope']==len(capture['requests'])==22 and capture['output_tokens']==2808,'Original full round request count')
        rows=[]
        for request in capture['requests']:
            prefix=directory/(request['group']+'-request'+str(request['index']))
            parsed=Path(str(prefix)+'-parsed.json');req=Path(str(prefix)+'-request.raw');resp=Path(str(prefix)+'-response.raw')
            require(read(parsed)==request,'Actual parsed record equals captured record')
            require(reference(req)['sha256']==request['request_body_sha256'],'Original request bytes match')
            require(reference(resp)['sha256']==request['raw_sha256'],'Original response bytes match')
            require(len(request['token_ids'])==request['usage']['completion_tokens'],'Original actual token count')
            rows.append(dict(parsed=reference(parsed),request=reference(req),response=reference(resp),tokens=len(request['token_ids'])))
        require(sum(row['tokens']for row in rows)==2808,'Complete original output tokens')
        rounds.append(dict(run_id=name,capture=reference(capture_path),client_terminal=reference(directory/'client-terminal.json'),server_terminal=reference(directory/'server-terminal.json'),
            requests=22,output_tokens=2808,request_records=rows,raw_bytes_rehashed=True,parsed_records_match_capture=True,
            client_original_exit0=True,server_success_proven=False,server_physical_only=True,both_original_owners_gone=True,both_groups_empty=True))
        requests.append(capture['requests'])
    fields=('model','prompt_token_ids','token_ids','text','finish_reason','usage','streaming','token_id_origin','group','index','response_status')
    comparisons=[]
    for left,right in itertools.combinations(range(4),2):
        for a,b in zip(requests[left],requests[right]):
            require(all(a[k]==b[k]for k in fields),'All corresponding reference outputs identical')
        comparisons.append(dict(left='r'+str(left).zfill(2),right='r'+str(right).zfill(2),corresponding_requests=22,all_equal=True,fields=list(fields)))
    result=dict(schema='smollm3_sio14_reference_four_round_actual_progress_v1',reviewer='/root',observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        actual_session=57994,actual_external_exit_observed=False,completed_rounds=[r['run_id']for r in rounds],completed_requests=88,completed_output_tokens=11232,
        rounds=rounds,pairwise_comparisons=comparisons,complete_reference_mode_matrix=True,all_three_configuration_matrix_complete=False,
        scientific_scope=dict(rounds=12,requests=264,output_tokens=33696),full_model_qualified=False,new_speedup_claim=False)
    out=ROOT/'research/smollm3-sio14-reference-four-rounds-progress-20261108-v1.json'
    with out.open('x')as f:json.dump(result,f,indent=2,ensure_ascii=False);f.write('\n')
    print(json.dumps(dict(report=reference(out),completed_rounds=4,requests=88,output_tokens=11232,pairwise_comparisons=6)))


if __name__=='__main__':run()
