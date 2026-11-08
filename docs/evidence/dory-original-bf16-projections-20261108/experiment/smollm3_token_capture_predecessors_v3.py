"""Source-only catalog/authentication, no services/locks/native work on import.

Frozen Phi catalog retains all 219 historical identities/provenance; exact 30
LFM planned units are resolved only at terminal runtime admission. Historical
missing invocation is explicit, and never reconstructed from current state.
"""
import json
from pathlib import Path
import re
import smollm3_token_capture_protocol_v3 as p
import phi4_saved_boundary_root_protocol_v3 as q
import lfm3_package_paired_protocol_v2 as lfm

FIXED=p.ROOT/'research/phi4-saved-boundary-cpu-v6-fixed-predecessors-20261107-v1.json'
LFM_SHA='8958693311a80cdf98d20f2aa6e7be28b989e050524d5be0eb7179269373f9da'


def invocation(value):
    return type(value)is str and re.fullmatch('[0-9a-f]{32}',value)is not None and value!='0'*32


def metadata(path):
    p.performance_guard();path=Path(path)
    p.require(path.suffix=='.json' and path.resolve()==path,'Fixed regular metadata path')
    return p.read(path)


def expected():
    return q.expected_predecessors(metadata(p.ROOT/'smollm3-cache-ablation-full-v8/plan.json'))


def frozen():
    value=metadata(FIXED)
    for name,digest in value['source_sha256'].items():
        p.performance_guard();path=Path(name)
        p.require(path.suffix in ('.py','.json') and path.stat().st_size<8*2**20,'Catalog source type/cap')
        p.require(p.sha(path)==digest,'Catalog evidence changed: '+name)
    rows=value['rows'];p.require(len(rows)==219 and [x['unit']for x in rows]==expected()[:219],'Frozen219 catalog')
    original=metadata(p.ROOT/'smollm3-cache-ablation-full-v8/plan.json')
    for old,row in zip(original['predecessors'],rows[:170]):
        p.require(q.owner_key(old)==row['owner'],'Historical PID/start/script lost')
    for i,row in enumerate(rows):
        owner=row['owner']
        if owner is None:
            p.require(i>=170 and row['never_launched']is True and row['absent_records'],'Unproved never-launched')
        else:
            p.require(q.owner_key(owner)==owner and owner['unit']==row['unit'] and not row['never_launched'],'Saved owner binding')
            if row['historical_invocation_unknown']:
                p.require(i<170 and row['invocation_id']is None,'Unknown newer invocation')
            else:p.require(invocation(row['invocation_id']) and row['installed_argv'],'Captured command/invocation absent')
        p.require(row['recorded_cgroups'],'No saved cgroups')
        for group in row['recorded_cgroups']:q.group_contract(row['unit'],group)
    return value


def absent_paths(directory,label):
    return [str(directory/(label+suffix))for suffix in ('-launch.json','-owner.json','-terminal.json','-drain.json',
        '-capture-failure-drain.json','-startup-failure-drain.json','-failure-drain.json')]


def saved_drain(proof,owner,command):
    value=proof['properties'];q.terminal_state(value)
    if value['LoadState']=='not-found':
        p.require(not value['InvocationID'] and value['ExecMainPID']=='0','Retired saved state fabricated')
    else:
        p.require(value['LoadState']=='loaded' and value['InvocationID']==owner['invocation_id'] and
            value['ExecMainPID']==str(owner['pid']),'Saved terminal invocation/PID')
        p.require(value['ExecStart'].split(' ; argv[]=',1)[1].split(' ; ignore_errors=',1)[0]==command,'Saved terminal command')
    groups=[]
    for name in ('first','final'):
        sample=proof[name];path=sample.get('path',sample.get('group'))
        path=path if path.startswith('/sys/fs/cgroup/') else '/sys/fs/cgroup'+path
        q.group_contract(owner['unit'],path)
        p.require(sample.get('absent')is True or sample.get('populated')is False or
            sample.get('empty')is True or (sample.get('recursive')is True and sample.get('processes')==[]),
            'Saved recursive drain lacks emptiness proof')
        groups.append(path)
    p.require(proof['first']==proof['final'],'Saved drain unstable')
    return sorted(set(groups+['/sys/fs/cgroup'+owner['cgroup']]))


def lfm_row(label,directory=None):
    directory=lfm.DEST if directory is None else Path(directory)
    p.require(label in q.lfm_pair_labels(),'Unplanned LFM label')
    unit=q.LFM_PREFIX+label;launch=directory/(label+'-launch.json');ownerpath=directory/(label+'-owner.json')
    group=p.GROUP_PREFIX+unit+'.service'
    if not launch.exists():
        absent=absent_paths(directory,label)
        p.require(all(not Path(x).exists()for x in absent),'Unlaunched LFM has evidence')
        return dict(unit=unit,owner=None,invocation_id=None,installed_argv=None,recorded_cgroups=['/sys/fs/cgroup'+group],
            historical_invocation_unknown=False,never_launched=True,absent_records=absent)
    p.require(ownerpath.exists(),'LFM launch without authenticated owner requires root release')
    intent=metadata(launch);owner=metadata(ownerpath);role='candidate' if label.startswith('candidate-')else'reference'
    server=label.endswith(('-compile','-quality-server'))or'-workloads-server-'in label
    if server:argv=lfm.server_argv();script=lfm.CLI
    else:
        phase=label.removeprefix(role+'-');trial=None
        if phase.startswith('workloads-'):phase,trial='workloads',int(phase.rsplit('-',1)[1])
        script=str(p.ROOT/'lfm3_package_paired_worker_v2.py')
        argv=[lfm.PYTHON,'-B',script,'--execute-native','--phase',phase,'--role',role,'--label',label,'--plan-sha256',LFM_SHA]
        if trial is not None:argv+=['--round',str(trial)]
    p.require(owner['argv']==intent['argv']==argv and owner['unit']==unit,'Exact LFM owner/launch argv')
    p.require(owner['cgroup']==group and invocation(owner['invocation_id']) and invocation(intent['nonce']),'LFM owner/nonce')
    env=lfm.environment(role,label.endswith('-compile'))
    env.update(LFM_PACKAGE_ATTEMPT=intent['nonce'],LFM_PACKAGE_PHASE='server'if server else phase,LFM_PACKAGE_ROLE=role)
    execargv=['/usr/bin/env','-i',*[k+'='+v for k,v in env.items()],*argv]
    p.require(intent['command'][intent['command'].index('/usr/bin/env'):]==execargv,'LFM full env command')
    p.require(intent['installed_argv']==owner['installed_argv']==' '.join(execargv),'LFM command owner binding')
    p.require(intent['server']==server and intent['compile_allowed']==label.endswith('-compile'),'LFM mode')
    terminal=directory/(label+('-drain.json'if server else'-terminal.json'))
    if not terminal.exists():
        choices=[directory/(label+suffix)for suffix in ('-startup-failure-drain.json','-failure-drain.json')]
        choices=[x for x in choices if x.exists()];p.require(len(choices)==1,'LFM saved drain required');terminal=choices[0]
    proof=metadata(terminal);groups=saved_drain(proof,owner,owner['installed_argv'])
    return dict(unit=unit,owner=q.owner_key(dict(owner,script=script)),captured_owner=owner,
        invocation_id=owner['invocation_id'],installed_argv=owner['installed_argv'],recorded_cgroups=groups,
        historical_invocation_unknown=False,never_launched=False,absent_records=[],
        saved_provenance={str(x):p.sha(x)for x in (launch,ownerpath,terminal)})


def lfm_rows(plan):
    p.require(p.sha(lfm.PLAN)==LFM_SHA,'LFM source plan')
    source=metadata(lfm.PLAN)
    for name,digest in source['source_sha256'].items():
        p.performance_guard();p.require(p.sha(name)==digest,'LFM source changed')
    status=metadata(lfm.DEST/'status.json');previous=plan['lfm_supervisor']
    p.require(status['stage']in('complete','failed') and status['performance_window']is False and
        status['pid']==previous['pid'],'LFM standalone controller not terminal')
    admission=metadata(lfm.DEST/'admission.json')
    p.require(admission['plan_sha256']==LFM_SHA and admission['driver_sha256']==
        plan['source_sha256'][str(p.ROOT/'run_lfm3_package_paired_full_v2.py')],'LFM launch plan/driver binding')
    return [lfm_row(label)for label in q.lfm_pair_labels()]


def canonical_tail(rows):
    tail=rows[170]['owner'];smol=metadata(p.ROOT/'smollm3-cache-ablation-full-v8/plan.json')
    for name in ('continuation-state.json','roadmap.json'):
        value=metadata(p.ROOT/name)
        p.require(value['serialized_queue_tail']['queue']==tail and value['current_authoritative_queue_tail']['queue']==tail,
            'Serial tail changed; new review required')
        barrier=value['prohibit_successor_enrollment_until_terminal']
        p.require(barrier['queue']==tail and barrier['plan_sha256']==p.sha(p.ROOT/'smollm3-cache-ablation-full-v8/plan.json'),
            'Canonical terminal barrier changed')
    return tail
