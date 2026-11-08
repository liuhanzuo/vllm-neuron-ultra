"""Actual r04 capture/terminal audit, without changing the running experiment."""
import datetime
import hashlib
import json
from pathlib import Path
import review_smollm3_sio14_reference_rounds_20261108_v1 as base
r=base.ROOT;q=r/'research';b=base.BASE

def run():
    prior=q/'smollm3-sio14-reference-four-rounds-progress-20261108-v1.json'
    base.require(base.reference(prior)['sha256']=='2233917982218b09f07a87835c829a3a7a5edc54c00c11b3ea81e7611e649ec7','Exact previous four-round audit')
    old=json.loads(prior.read_text());directory=b/'r04'
    terminal=base.read(directory/'client-terminal.json');server=base.read(directory/'server-terminal.json')
    props=terminal['original_properties']
    base.require(terminal['passed'] is True and props['Result']=='success' and props['ExecMainCode']=='1' and props['ExecMainStatus']=='0','Actual client exit0')
    base.require(terminal['drain']['drained'] is True and server['drained'] is True and server['authorized_stop'] is True and server['physical_only'] is True and server['success_proven'] is False,'Original physical-only server terminal')
    base.gone(terminal['owner']);base.gone(server['original_owner'])
    spec_path=q/'smollm3-observer-sio14-execution-spec-20261108-v1.json'
    base.require(base.reference(spec_path)['sha256']=='6ae6357c45f3d6bbd0c2fc38d5a77fe24d2c537073b67ca5589554da05c14337','Exact running execution spec')
    spec=json.loads(spec_path.read_text());capture=base.read(directory/'capture-result.json')
    base.require(capture['run']==spec['matrix'][4] and capture['role']=='nki_write','Actual first write-only variant round')
    base.require(capture['passed'] is True and capture['protocol_passed'] is True and capture['diagnostic_complete'] is True and capture['request_scope']==len(capture['requests'])==22 and capture['output_tokens']==2808,'Whole actual round')
    rows=[]
    for request in capture['requests']:
        prefix=directory/(request['group']+'-request'+str(request['index']))
        parsed=Path(str(prefix)+'-parsed.json');req=Path(str(prefix)+'-request.raw');resp=Path(str(prefix)+'-response.raw')
        base.require(base.read(parsed)==request,'Capture equals original parsed record')
        base.require(base.reference(req)['sha256']==request['request_body_sha256'] and base.reference(resp)['sha256']==request['raw_sha256'],'Original request and response hashes')
        base.require(len(request['token_ids'])==request['usage']['completion_tokens'],'Actual output token count')
        rows.append(dict(parsed=base.reference(parsed),request=base.reference(req),response=base.reference(resp),tokens=len(request['token_ids'])))
    base.require(sum(x['tokens']for x in rows)==2808,'All original tokens accounted')
    fields=('model','prompt_token_ids','token_ids','text','finish_reason','usage','streaming','token_id_origin','group','index','response_status')
    comparisons=[]
    for reference_round in old['rounds']:
        path=Path(reference_round['capture']['path']);base.require(base.reference(path)==reference_round['capture'],'Earlier audited capture unchanged')
        earlier=base.read(path)
        for left,right in zip(earlier['requests'],capture['requests']):base.require(all(left[k]==right[k]for k in fields),'Corresponding actual write/reference request output differs')
        comparisons.append(dict(reference=reference_round['run_id'],candidate='r04',requests=22,all_equal=True,fields=list(fields)))
    report=dict(schema='smollm3_sio14_write_first_round_actual_progress_v1',reviewer='/root',observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                previous_four_round_audit=base.reference(prior),execution_spec=base.reference(spec_path),capture=base.reference(directory/'capture-result.json'),
                client_terminal=base.reference(directory/'client-terminal.json'),server_terminal=base.reference(directory/'server-terminal.json'),requests=rows,
                actual_original_client_exit0=True,original_owners_gone_and_groups_empty=True,server_physical_only=True,server_success_proven=False,
                cross_configuration_comparisons=comparisons,completed_rounds=5,completed_requests=110,completed_output_tokens=14040,
                full12round_matrix_complete=False,full_model_qualified=False,new_speedup_claim=False)
    out=q/'smollm3-sio14-write-first-round-progress-20261108-v1.json'
    with out.open('x')as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(dict(report=base.reference(out),completed_rounds=5,requests=110,tokens=14040,new_write_reference_comparisons=4)))

if __name__=='__main__':run()
