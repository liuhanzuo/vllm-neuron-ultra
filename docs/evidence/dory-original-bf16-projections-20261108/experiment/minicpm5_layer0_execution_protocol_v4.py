"""Strict serial Mini layer0 process/service contracts, no runtime on import.

The execution spec is a future root-frozen binding of input plan, reviews,
complete predecessor catalog, service names and this source closure.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import stat
import subprocess
import time
from minicpm5_layer0_runtime_auth_v1 import clear_window,require,digest
ROOT=Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
LOCK=ROOT/'serial-enrollment.lock'
GROUP_PREFIX='/user.slice/user-1000.slice/user@1000.service/app.slice/'
STAGES=('native','cpu_control','cpu_history')
ARMS=dict(native='HF_native',cpu_control='C_Hd_control',cpu_history='C_Hc_history_generation')
GIB=2**30;MIB=2**20
RESOURCE=dict(MemoryMax=str(26*GIB),MemoryHigh=str(24*GIB),MemorySwapMax='0',TasksMax='256',
    AllowedCPUs='6-7',CPUAffinity='6-7',Nice='19',KillMode='control-group',RemainAfterExit='yes',SendSIGKILL='yes',
    WorkingDirectory=str(ROOT),LimitFSIZE=str(16*MIB),RuntimeMaxUSec='12h',TimeoutStopUSec='30s')
KEYS='BindsTo,After,Id,LoadState,ActiveState,SubState,MainPID,ControlPID,Job,InvocationID,ControlGroup,Result,ExecMainPID,ExecMainCode,ExecMainStatus,ExecStart,Environment,ExecStopPost,'+','.join(RESOURCE)


def text(path,cap=8*MIB):
    clear_window();path=Path(path);require(path.is_absolute() and path.resolve()==path and not any(p.is_symlink() for p in (path,*path.parents)),'Fixed absolute metadata path')
    require(path.suffix not in ('.pt','.safetensors','.bin','.neff','.raw','.zlib','.whl','.so','.pb'),'Metadata only')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        before=os.fstat(fd);require(stat.S_ISREG(before.st_mode) and before.st_size<=cap,'Metadata type/cap')
        value=bytearray()
        while len(value)<=cap:
            block=os.read(fd,min(MIB,cap+1-len(value)))
            if not block:break
            value.extend(block)
        after=os.fstat(fd);require(len(value)<=cap and all(getattr(before,k)==getattr(after,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')),'Metadata changed')
        value.decode('utf8');return bytes(value)
    finally:os.close(fd)

def sha(path):return hashlib.sha256(text(path)).hexdigest()
def read(path,wanted=None):
    raw=text(path);require(wanted is None or hashlib.sha256(raw).hexdigest()==wanted,'Metadata pin');return json.loads(raw)
def once(path,value):
    clear_window()
    raw=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode();require(len(raw)<8*MIB,'Evidence cap')
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    directory=os.open(Path(path).parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:os.fsync(directory)
    finally:os.close(directory)

def command(argv,cap=65536,timeout=15):
    process=subprocess.Popen(argv,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);selector=selectors.DefaultSelector();selector.register(process.stdout,selectors.EVENT_READ)
    out=bytearray();deadline=time.monotonic()+timeout
    try:
        while selector.get_map():
            require(time.monotonic()<deadline,'Command deadline')
            for key,_ in selector.select(.1):
                block=os.read(key.fileobj.fileno(),min(8192,cap+1-len(out)))
                if not block:selector.unregister(key.fileobj);continue
                out.extend(block);require(len(out)<=cap,'Command output cap')
        require(process.wait(timeout=max(.1,deadline-time.monotonic()))==0,'Command failed: '+bytes(out[-1000:]).decode(errors='replace'));return out.decode()
    finally:
        selector.close()
        if process.poll() is None:process.kill()
        process.wait(timeout=5);process.stdout.close()

def properties(unit):
    require(re.fullmatch(r'trn1-[a-z0-9_-]{1,220}',unit),'Fixed unit syntax')
    raw=command(['systemctl','--user','show','--all',unit,'--property='+KEYS])
    return parse_properties(raw)


def parse_properties(raw):
    value=dict(line.split('=',1) for line in raw.splitlines() if '=' in line)
    # org.freedesktop.systemd1.Service.ExecStopPost is a(sasbttttuii).
    # This installed systemctl omits its zero-length array even with --all.
    # Normalize only that optional empty array; never invent other properties.
    # A present nonempty value is preserved and rejected by both contracts.
    value.setdefault('ExecStopPost','')
    return value

def proc(pid):
    def small(name,cap=65536):
        with Path('/proc',str(pid),name).open('rb') as f:b=f.read(cap+1)
        require(len(b)<=cap,'Bounded process metadata');return b
    try:
        before=small('stat').decode().rsplit(')',1)[1].split();argv=[x.decode() for x in small('cmdline').split(b'\0') if x];group=small('cgroup').decode().strip()
        after=small('stat').decode().rsplit(')',1)[1].split();require(before[19]==after[19],'PID reused during observation')
        return dict(pid=pid,start_ticks=before[19],ppid=int(before[1]),argv=argv,cgroup=group)
    except FileNotFoundError:return None

def gone(owner):
    actual=proc(owner['pid']);require(actual is None or actual['start_ticks']!=owner['start_ticks'],'Owned PID/start still live')

def empty_group(group):
    require(group.startswith(GROUP_PREFIX+'trn1-') and '..' not in Path(group).parts,'Fixed owned group')
    root=Path('/sys/fs/cgroup')/group.lstrip('/');require(not root.is_symlink(),'Group symlink')
    if not root.exists():return dict(group=group,absent=True)
    pending=[root];count=0
    while pending:
        path=pending.pop();count+=1;require(count<=256,'Group recursion cap')
        require(not path.is_symlink(),'Group symlink')
        with (path/'cgroup.procs').open() as f:require(not f.read(65537).strip(),'Live group descendants')
        for child in path.iterdir():
            require(not child.is_symlink(),'Child group symlink')
            if child.is_dir():pending.append(child)
    return dict(group=group,absent=False,cgroups=count)

def terminal(value):return value['MainPID']==value['ControlPID']=='0' and not value['Job'] and (value['ActiveState'] in ('inactive','failed') or (value['ActiveState'],value['SubState'])==('active','exited'))
def invocation(value):return type(value) is str and re.fullmatch('[0-9a-f]{32}',value) and value!='0'*32

def lock_proof(owner,lock):
    info=LOCK.lstat();require(stat.S_ISREG(info.st_mode) and info.st_nlink==1 and not LOCK.is_symlink(),'Serial lock file')
    require(dict(device=info.st_dev,inode=info.st_ino)==lock,'Serial lock identity changed')
    actual=proc(owner['pid']);require(actual==owner,'Controller actual process identity changed')
    with Path('/proc/locks').open() as f:raw=f.read(2*MIB+1)
    require(len(raw)<=2*MIB,'Lock listing cap');matches=[]
    for line in raw.splitlines():
        fields=line.split()
        if len(fields)==8 and fields[1:4]==['FLOCK','ADVISORY','WRITE'] and fields[4]==str(owner['pid']):
            major,minor,inode=fields[5].split(':')
            if os.makedev(int(major,16),int(minor,16))==info.st_dev and int(inode)==info.st_ino:matches.append(fields)
    require(len(matches)==1 and matches[0][6:]==['0','EOF'],'Controller does not hold whole serial flock')
    return dict(owner=actual,lock=lock,whole_file_write=True,actual_proc_locks_line=' '.join(matches[0]))

def validate_spec(spec):
    require(spec['schema']=='minicpm5_layer0_execution_spec_v4','Execution spec schema')
    require(spec['stages']==list(STAGES),'Exact native/control/history serial order')
    require(spec['resources']==RESOURCE and spec['output_cap_bytes']==2*GIB and spec['control_cap_bytes']==32*MIB and spec['root_floor_bytes']==28*GIB,'Fixed resource envelope')
    require(Path(spec['control_root']).parent==ROOT and Path(spec['control_root']).name.startswith('minicpm5-layer0-control-'),'Fixed fresh control root')
    require(Path(spec['worker'])==ROOT/'minicpm5_layer0_worker_v4.py' and Path(spec['supervisor'])==ROOT/'research/run_minicpm5_layer0_root_v4.py','Fixed executable source paths')
    require(set(spec['units'])==set(STAGES) and len(set(spec['units'].values()))==3,'Exactly three phase services')
    require(all(re.fullmatch('trn1-latest-verify-minicpm5-layer0-[a-z0-9-]+-'+stage.replace('_','-'),unit) for stage,unit in spec['units'].items()),'Phase unit namespace')
    require(len(spec['predecessor_units'])==len(set(spec['predecessor_units'])) and len(spec['predecessor_units'])==257,'Complete fixed predecessor catalog required')
    require(spec['automatic_successor_launch'] is False,'No automatic successor')
    require(re.fullmatch('trn1-latest-verify-minicpm5-layer0-[a-z0-9-]+-controller',spec['controller_unit']),'Fixed controller service')
    require(type(spec['master_port']) is int and 20000<=spec['master_port']<=60000,'Fixed local rendezvous port')
    require(spec['native_python']=='/home/ubuntu/venvs/model_reference_2026/bin/python' and spec['cpu_python']=='/home/ubuntu/venvs/vllm_neuron/bin/python','Exact interpreters')
    require(all(spec[k] in spec['source_sha256'] for k in ('worker','supervisor')),'Complete executable source pins')
    require(spec['controller_python']=='/usr/bin/python3','Controller interpreter')
    require(spec['external_supervisor']==str(ROOT/'minicpm5_layer0_external_supervisor_v4.py') and spec['external_supervisor'] in spec['source_sha256'],'External supervisor pinned')
    return spec

def bundle(spec_ref,review_ref):
    spec=validate_spec(read(spec_ref['path'],spec_ref['sha256']));review=read(review_ref['path'],review_ref['sha256'])
    require(review['passed'] is review['independent_review'] is review['source_only'] is True and review['reviewer']=='/root' and review['blockers']==[],'Independent root source review')
    require(review['source_sha256']==spec['source_sha256'] and review['execution_spec_sha256']==spec_ref['sha256'],'Exact root-reviewed closure/spec')
    plan=read(spec['input_plan']['path'],spec['input_plan']['sha256']);require(digest(plan)==spec['input_plan']['canonical_sha256'],'Canonical input plan')
    for path,wanted in spec['source_sha256'].items():require(sha(path)==wanted,'Frozen execution source pin: '+path)
    require(all(spec['source_sha256'].get(k)==v for k,v in plan['source_sha256'].items()),'All original input sources retained')
    catalog=read(spec['predecessor_catalog']['path'],spec['predecessor_catalog']['sha256'])
    require(catalog['schema']=='minicpm5_layer0_complete_predecessors_v2' and catalog['all_planned_launches_accounted_for'] is True,'Complete reviewed catalog')
    require(all(spec['source_sha256'].get(path)==wanted for path,wanted in catalog['source_sha256'].items()),'Full predecessor evidence source closure')
    require([row['unit'] for row in catalog['rows']]==spec['predecessor_units'],'Complete exact predecessor list')
    return spec,plan,catalog

def predecessor_owner_gone(row):
    owner=row['owner']
    if owner['start_ticks'] is None:
        # The one v3 launch exited before live-owner ACK. Do not invent start
        # ticks or treat PID reuse as evidence that this uncaptured owner died.
        require(row['unit']=='trn1-latest-verify-minicpm5-layer0-v3-controller' and owner['pid']==1432134 and
            owner.get('start_ticks_observed') is False and row.get('captured_live_owner') is False and
            row['never_launched'] is False and row['invocation_id']=='30de9e643d7a4af08445e7d96ed1d644',
            'Only exact audited v3 uncaptured controller may have unknown start')
        require(proc(owner['pid']) is None,'Uncaptured predecessor PID must be entirely absent')
    else:gone(owner)


def predecessor_drain(catalog):
    observed=[]
    for row in catalog['rows']:
        samples=[]
        for _ in range(2):
            value=properties(row['unit']);require(terminal(value),'Predecessor not terminal')
            owner=row['owner'];group=GROUP_PREFIX+row['unit']+'.service'
            if owner is None:
                require(row['never_launched'] is True and value['LoadState']=='not-found' and not value['InvocationID'],'Never-launched unit appeared')
                require(all(not Path(p).exists() for p in row['absent_records']),'Never-launched evidence appeared')
            else:
                predecessor_owner_gone(row)
                if value['LoadState']=='not-found':require(not value['InvocationID'],'Retired unit invocation')
                else:
                    require(not row.get('historical_invocation_unknown',False) and row['invocation_id'] is not None and value['LoadState']=='loaded' and value['InvocationID']==row['invocation_id'] and value['ExecMainPID']==str(owner['pid']),'Historical terminal owner changed')
                    require(re.findall(r'argv\[\]=(.*?)\s*;',value['ExecStart'])==[row['installed_argv']],'Historical terminal argv changed')
            require(value['ControlGroup'] in ('',group),'Predecessor cgroup changed')
            samples.append(dict(properties=value,group=empty_group(group)))
        require(samples[0]==samples[1],'Unstable predecessor drain');observed.append(dict(unit=row['unit'],owner=owner,samples=samples))
    for owner in catalog['processes']:gone(owner)
    return dict(complete=True,rows=observed,processes=catalog['processes'])

def stage_argv(spec,stage,admission_ref):
    python=spec['native_python'] if stage=='native' else spec['cpu_python']
    mode='native' if stage=='native' else 'coordinator'
    return [python,'-B',spec['worker'],'--mode',mode,'--stage',stage,'--admission',admission_ref['path'],'--admission-sha256',admission_ref['sha256']]

def rank_argv(spec,stage,admission_ref):
    return [spec['cpu_python'],'-B',spec['worker'],'--mode','rank','--stage',stage,'--admission',admission_ref['path'],'--admission-sha256',admission_ref['sha256']]

def environment(spec,stage,attempt):
    env=dict(PATH='/usr/bin:/bin',HOME='/home/ubuntu',LANG='C.UTF-8',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        XDG_RUNTIME_DIR='/run/user/1000',DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1000/bus',MINI_LAYER0_ATTEMPT=attempt,
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2' if stage=='native' else '1',MKL_NUM_THREADS='2' if stage=='native' else '1')
    if stage!='native':env.update(PATH='/home/ubuntu/venvs/vllm_neuron/bin:/usr/bin:/bin',NXD_CPU_MODE='1',NEURON_RT_VISIBLE_CORES='0-1',NEURON_COMPILE_CACHE_URL='file:///tmp/mini-layer0-no-compile',TOKENIZERS_PARALLELISM='false')
    require(all('\n' not in k+v and ' ' not in k+v for k,v in env.items()),'Single token environment')
    return env

def elastic_run_id(attempt,stage):
    require(re.fullmatch('[0-9a-f]{32}',attempt) and stage in ('cpu_control','cpu_history'),'Bound elastic attempt/stage')
    return 'mini-layer0-'+attempt+'-'+stage


def service_command(spec,stage,admission_ref,attempt):
    control=Path(spec['control_root']);props=dict(RESOURCE);props['RuntimeMaxSec']=props.pop('RuntimeMaxUSec');props['TimeoutStopSec']=props.pop('TimeoutStopUSec')
    env=environment(spec,stage,attempt);props.update(BindsTo=spec['controller_unit']+'.service',After=spec['controller_unit']+'.service')
    return ['systemd-run','--user','--quiet','--unit='+spec['units'][stage],'--service-type=exec',*[arg for k,v in props.items() for arg in ('-p',k+'='+v)],
        '-p','StandardOutput=append:'+str(control/(stage+'.log')),'-p','StandardError=inherit',
        '/usr/bin/env','-i',*[k+'='+v for k,v in env.items()],*stage_argv(spec,stage,admission_ref)]

def service_contract(spec,stage,admission_ref,attempt,value):
    group=GROUP_PREFIX+spec['units'][stage]+'.service'
    require(value['LoadState']=='loaded' and invocation(value['InvocationID']),'Loaded exact service required')
    for k,v in RESOURCE.items():require(duration_seconds(value[k])==duration_seconds(v) if k in ('RuntimeMaxUSec','TimeoutStopUSec') else value[k]==v,'Actual service resource: '+k)
    require(spec['controller_unit']+'.service' in value['BindsTo'].split() and spec['controller_unit']+'.service' in value['After'].split(),'Worker bound to live controller service')
    expected=service_command(spec,stage,admission_ref,attempt);installed=' '.join(expected[expected.index('/usr/bin/env'):])
    require(re.findall(r'argv\[\]=(.*?)\s*;',value['ExecStart'])==[installed] and value['ExecStopPost']=='','Actual service command/cleanup')
    require(value['ControlGroup'] in ('',group),'Actual service group')
    return value

def service_identity(spec,stage,admission_ref,attempt,owner=None,observation=None):
    # Authenticate the caller's one observation; never issue a hidden second show.
    value=service_contract(spec,stage,admission_ref,attempt,properties(spec['units'][stage]) if observation is None else observation);group=GROUP_PREFIX+spec['units'][stage]+'.service'
    if owner:
        require(value['InvocationID']==owner['invocation_id'] and value['ExecMainPID']==str(owner['pid']),'Actual terminal service owner')
    if value['MainPID']!='0':
        actual=proc(int(value['MainPID']));require(actual is not None and actual['argv']==stage_argv(spec,stage,admission_ref) and actual['cgroup']=='0::'+group,'Live service PID/argv/group')
        current=dict(**actual,unit=spec['units'][stage],invocation_id=value['InvocationID'])
        require(owner is None or current==owner,'Service owner identity changed');owner=current
    require(owner is not None,'No captured service owner')
    return owner,value

def phase_drain(spec,stage,admission_ref,attempt,owner):
    samples=[]
    for _ in range(2):
        seen,value=service_identity(spec,stage,admission_ref,attempt,owner);require(seen==owner and terminal(value),'Retained terminal service required')
        gone(owner);samples.append(dict(properties=value,group=empty_group(GROUP_PREFIX+owner['unit']+'.service')))
    require(samples[0]==samples[1],'Unstable phase terminal')
    v=samples[0]['properties'];return dict(owner=owner,samples=samples,physically_drained=True,identity_verified=True,successful=v['Result']=='success' and v['ExecMainCode']=='1' and v['ExecMainStatus']=='0')

def active_heavy(excluded=()):
    raw=command(['systemctl','--user','list-units','--type=service','--state=running,activating,deactivating','--plain','--no-legend','--no-pager'],cap=256*1024)
    units=[line.split()[0].removesuffix('.service') for line in raw.splitlines() if line.split() and line.split()[0].startswith('trn1-')]
    require(not [u for u in units if u not in excluded],'Competing active heavy service');return units

def user_bus():
    require(os.getuid()==os.geteuid()==1000,'UID1000 required')
    require(os.environ.get('XDG_RUNTIME_DIR')=='/run/user/1000' and os.environ.get('DBUS_SESSION_BUS_ADDRESS')=='unix:path=/run/user/1000/bus','Explicit user bus')
    path=Path('/run/user/1000');a=path.lstat();b=(path/'bus').lstat()
    require(stat.S_ISDIR(a.st_mode) and stat.S_IMODE(a.st_mode)==0o700 and a.st_uid==1000 and path.resolve()==path,'User runtime directory')
    require(stat.S_ISSOCK(b.st_mode) and b.st_uid==1000,'User bus socket')

def control_usage(spec):
    root=Path(spec['control_root']);total=0;count=0
    for path in root.iterdir():
        info=path.lstat();require(stat.S_ISREG(info.st_mode) and info.st_uid==1000 and info.st_nlink==1,'Control evidence regular owned file');count+=1
        require(path.suffix in ('.json','.log') and info.st_size<16*MIB,'Control evidence cap/type');total+=max(info.st_size,info.st_blocks*512)
    require(count<=64 and total<=spec['control_cap_bytes'],'Whole control tree cap')
    fs=os.statvfs(ROOT);require(fs.f_bavail*fs.f_frsize>=spec['root_floor_bytes'],'Root storage floor');return total


def controller_argv(spec,spec_ref,review_ref):
    return [spec['controller_python'],'-B',spec['supervisor'],'--execute','--spec',spec_ref['path'],'--spec-sha256',spec_ref['sha256'],'--review',review_ref['path'],'--review-sha256',review_ref['sha256']]

def controller_command(spec,spec_ref,review_ref):
    env=dict(PATH='/usr/bin:/bin',HOME='/home/ubuntu',LANG='C.UTF-8',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',XDG_RUNTIME_DIR='/run/user/1000',DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1000/bus')
    props=dict(MemoryMax=str(512*MIB),MemoryHigh=str(384*MIB),MemorySwapMax='0',TasksMax='64',AllowedCPUs='6-7',CPUAffinity='6-7',Nice='19',KillMode='control-group',RemainAfterExit='yes',SendSIGKILL='yes',WorkingDirectory=str(ROOT),LimitFSIZE=str(16*MIB),RuntimeMaxSec='40h',TimeoutStopSec='2min')
    return ['systemd-run','--user','--quiet','--unit='+spec['controller_unit'],'--service-type=exec',*[arg for k,v in props.items() for arg in ('-p',k+'='+v)],'/usr/bin/env','-i',*[k+'='+v for k,v in env.items()],*controller_argv(spec,spec_ref,review_ref)]

def duration_seconds(value):
    if value=='infinity':return float('inf')
    matches=re.findall(r'([0-9]+(?:\.[0-9]+)?)(us|ms|s|min|h|d)',value)
    require(matches and ''.join(a+b for a,b in matches)==value.replace(' ',''),'Duration representation')
    return sum(float(a)*dict(us=.000001,ms=.001,s=1,min=60,h=3600,d=86400)[b] for a,b in matches)

def controller_identity(spec,spec_ref,review_ref,owner=None,expected_invocation=None,observation=None):
    # A loaded->not-found GC between independent calls is a valid stop transition.
    value=properties(spec['controller_unit']) if observation is None else observation;require(value['LoadState']=='loaded' and invocation(value['InvocationID']),'Loaded controller invocation')
    expected=controller_command(spec,spec_ref,review_ref);installed=' '.join(expected[expected.index('/usr/bin/env'):])
    require(re.findall(r'argv\[\]=(.*?)\s*;',value['ExecStart'])==[installed],'Actual controller command')
    for k,wanted in dict(MemoryMax=str(512*MIB),MemoryHigh=str(384*MIB),MemorySwapMax='0',TasksMax='64',AllowedCPUs='6-7',CPUAffinity='6-7',Nice='19',KillMode='control-group',RemainAfterExit='yes',SendSIGKILL='yes',WorkingDirectory=str(ROOT),LimitFSIZE=str(16*MIB),RuntimeMaxUSec='40h',TimeoutStopUSec='2min').items():
        require(duration_seconds(value[k])==duration_seconds(wanted) if k in ('RuntimeMaxUSec','TimeoutStopUSec') else value[k]==wanted,'Controller resource '+k)
    require(value['ExecStopPost']=='','Unexpected controller cleanup command')
    if expected_invocation:require(value['InvocationID']==expected_invocation,'Controller invocation changed')
    group=GROUP_PREFIX+spec['controller_unit']+'.service';require(value['ControlGroup'] in ('',group),'Controller group')
    if value['MainPID']!='0':
        actual=proc(int(value['MainPID']));require(actual is not None and actual['argv']==controller_argv(spec,spec_ref,review_ref) and actual['cgroup']=='0::'+group,'Actual controller process')
        require(owner is None or actual==owner,'Controller PID/start changed');owner=actual
    elif owner is not None:require(value['ExecMainPID']==str(owner['pid']),'Terminal controller PID changed');gone(owner)
    return owner,value
