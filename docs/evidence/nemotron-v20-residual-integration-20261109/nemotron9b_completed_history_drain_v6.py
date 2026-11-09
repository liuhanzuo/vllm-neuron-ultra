"""Complete post-r9 and actual residual-review history drain, with only the actual current controller exempted from the global idle scan."""
from pathlib import Path
import re,sys,time
import nemotron9b_post_dory_history_v1 as history
import nemotron9b_residual_completion_v1 as attention
require=history.require

def physical_drain(candidate,allowed_units=(),*,spec):
    candidate=attention.validate_spec(spec,candidate)
    require(len(allowed_units)<=1 and all(re.fullmatch(r'trn1-latest-verify-nemotron9b-package-[a-z0-9-]+-controller',x)for x in allowed_units),'Only one exact new controller may be live')
    require(not set(allowed_units)&{row['unit']for row in candidate['rows']},'No historical service exempted')
    if str(history.Q)not in sys.path:sys.path.insert(0,str(history.Q))
    from research.finalize_smollm3_sio14_root_20261108_v1 import life
    for process in candidate['processes']:life.gone(process)
    for row in candidate['missing_child_start_identities']:
        require(not Path('/proc', str(row['child']['recorded_identity']['pid'])).exists(), 'Historical missing-start PID present')
    observations=[]
    for row in candidate['rows']:
        name=row['unit'];samples=[]
        for index in range(2):
            value=life.props(name)
            require(life.terminal(value)and value['Id']==name+'.service','Historical service terminal')
            if name==history.MISSING_ACK_UNIT:
                require(row['owner']is None and row['never_launched']is False and value['LoadState']=='not-found'and not value['InvocationID']and
                        value['MainPID']==value['ControlPID']==value['ExecMainPID']=='0','Exact recorded missing-ACK service entirely absent')
            elif row['owner']is None:
                require(row['never_launched']is True and value['LoadState']=='not-found'and not value['InvocationID']and
                        all(not Path(p).exists()for p in row['absent_records']),'Actual unlaunched historical service still absent')
            else:
                life.gone(row['owner'])
                if value['LoadState']=='not-found':require(not value['InvocationID']and value['ExecMainPID']=='0','Original historical service collected')
                else:require(value['InvocationID']==row['invocation_id']and value['ExecMainPID']==str(row['owner']['pid'])and life.installed(value)==row['installed_argv'],'Original historical service retained')
            paths=set(row['recorded_cgroups'])|{'/sys/fs/cgroup'+life.GROUP_PREFIX+name+'.service'};groups=[]
            for path in sorted(paths):
                group=path.removeprefix('/sys/fs/cgroup');require(group==life.GROUP_PREFIX+name+'.service','Exact historical cgroup')
                groups.append(life.cgroup_inventory(group,require_empty=True))
            samples.append(dict(properties=value,groups=groups))
            if index==0:time.sleep(.02)
        require(samples[0]==samples[1],'Stable double historical drain');observations.append(dict(unit=name,samples=samples))
    running=life.command(['systemctl','--user','list-units','--type=service','--state=running,activating,deactivating','--plain','--no-legend','--no-pager'])
    names={line.split()[0]for line in running.splitlines()if line.split()and line.split()[0].startswith('trn1-')}
    require(names<={x+'.service'for x in allowed_units},'Only explicitly acknowledged current controller may be live')
    return observations
