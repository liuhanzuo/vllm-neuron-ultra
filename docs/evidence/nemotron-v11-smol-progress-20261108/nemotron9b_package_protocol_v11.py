"""Source-only by default. Future root spec binds all runtime identities."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import statistics
import sys
import time
import minicpm5_layer0_execution_protocol_v4 as low
import nemotron9b_package_metadata_alias_v1 as alias_metadata
import nemotron9b_package_metadata_v11 as metadata
import nemotron9b_package_storage_inventory_candidate_v1 as output_inventory

ROOT = Path(__file__).resolve().parent
PACKAGE = Path('/home/ubuntu/src/vllm-neuron-ultra-nemotron9b-v1')
COMMIT = '60ab7f54a2c54ef0bc9b498621a45719ee7f0e84'
PYTHON = '/home/ubuntu/venvs/vllm_neuron/bin/python'
GIB = 2**30
MIB = 2**20
require, read, sha, once, command = low.require, metadata.read, metadata.sha, metadata.once, low.command
BUCKETS = (128,256,512,1024,2048,4096)
ROOT_FLOOR, NVME_FLOOR = 28*GIB,30*GIB
ROOT_CAP, NVME_CAP = GIB,18*GIB
CAPS = {'compile-work':10*GIB,'primitives':4*GIB,'artifact':GIB,
        'compile-cache':2*GIB,'temporary':GIB}
ALIAS_FLAGS = ['--target=trn1','--auto-cast=none','--model-type=transformer','-O2','--lnc=1',
 '--tensorizer-options=--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=1 --vectorize-strided-dma']
SPLIT_FLAGS = ['--target=trn1','--auto-cast=none','--model-type=transformer','-O1','--lnc=1',
 '--tensorizer-options=--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2 --vectorize-strided-dma']


def quiet():
    for name in ('smollm3-cache-ablation-full-v8','lfm3-package-paired-full-v2'):
        require(read(ROOT/name/'status.json')['performance_window'] is False,'Existing timed performance window')


def alias_jobs():
    shapes=[('nemotron-conv-r4','bfloat16',[4,6144,4],4),
            ('nemotron-ssm-r4','float32',[4,64,80,128],4)]
    shapes += [(f'ministral-kv{n}-r{b}','bfloat16',[4,4,n,128],b)
               for n in (256,4096)for b in (1,4)]
    jobs=[dict(name='alias-'+name,case=dict(name=name,dtype=dtype,shape=shape,rows=rows),
               mode='persistent',data='raw',variant='read_predicated')for name,dtype,shape,rows in shapes]
    base=jobs[0]['case']
    jobs += [dict(name='alias-control-'+mode+'-'+data,case=base,mode=mode,data=data,variant='read_predicated')
             for mode,data in [('functional','raw'),('functional','finite_nonzero'),('persistent','finite_nonzero')]]
    return jobs


def split_jobs():
    result=[]
    for kind,sizes in [('nemotron_projection',[5120,6144,64]),('nemotron_conv',[5120,512,512])]:
        for b,t in [(1,n)for n in BUCKETS]+[(4,1),(1,1)]:
            case=dict(name=f'{kind}-b{b}-t{t}',kind=kind,batch=b,length=t,sizes=sizes,
                input_shape=[b,t,sum(sizes)]if kind.endswith('projection')else[b,sum(sizes),t],
                dtype='bfloat16',single_row_control=b*t==1)
            for variant in ('original','tensor_split'):
                result.append(dict(name=('split-'+case['name']+'-'+variant).replace('_','-'),case=case,variant=variant))
    return result


SCIENTIFIC_STAGES = ('prepare',*[x['name']for x in alias_jobs()],*[x['name']for x in split_jobs()],
          'compile','state-copy','numeric','cold','long','quality','workloads-0','workloads-1','workloads-2')

STAGES=(*SCIENTIFIC_STAGES,'verify-raw')


def kind(stage):
    require(stage in (*STAGES,'controller'),'Known immutable phase')
    if stage.startswith('alias-'):return 'alias'
    if stage.startswith('split-'):return 'split'
    if stage.startswith('workloads-'):return 'workloads'
    return stage


def resources(stage):
    k=kind(stage)
    memory,high,swap,seconds=(512*MIB,384*MIB,0,172800)if k=='controller'else(
        (4*GIB,3*GIB,0,3600)if k=='prepare'else
        (6*GIB,5*GIB,2*GIB,900)if k=='alias'else
        (8*GIB,7*GIB,2*GIB,600)if k=='split'else
        (26*GIB,25*GIB,12*GIB,14400)if k=='compile'else(26*GIB,25*GIB,6*GIB,7200))
    return dict(MemoryMax=str(memory),MemoryHigh=str(high),MemorySwapMax=str(swap),TasksMax='2048',
        AllowedCPUs='0-7',CPUAffinity='0-7',Nice='10',KillMode='control-group',RemainAfterExit='yes',SendSIGKILL='yes',
        WorkingDirectory=str(ROOT),LimitFSIZE=str(16*GIB),RuntimeMaxUSec=str(seconds)+'s',TimeoutStopUSec='120s')


def ref(path):return dict(path=str(path),sha256=sha(path))


def stream_sha(path,guard=lambda:None):
    """Runtime-only payload hashing; source checks must not call on payloads."""
    value=hashlib.sha256()
    with Path(path).open('rb')as f:
        while data:=f.read(MIB):guard();value.update(data)
    return value.hexdigest()


def file_stat(path):
    p=Path(path);s=p.stat()
    return dict(realpath=str(p.resolve()),device=s.st_dev,inode=s.st_ino,size=s.st_size,
                mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)


def save(path,value):
    return metadata.save(path,value)


def sources(spec):
    quiet()
    catalog=metadata.load(spec['final_predecessor_catalog'])
    parts=metadata.classify_sources(spec,catalog)
    for name,digest in parts['text_sha256'].items():
        actual=(alias_metadata.source_sha(name,spec['metadata_aliases'],metadata.text)
                if name in spec['metadata_aliases']else metadata.sha(name,digest))
        require(actual==digest,'Source changed: '+name)
    for name,digest in parts['python_bytes_sha256'].items():metadata.python_source_bytes(name,digest)


def required_sources():
    files=list(ROOT.glob('nemotron9b_package_*v11.py'))
    files += list(ROOT.glob('nemotron9b_package_*v10.py'))
    files += [ROOT/'nemotron9b_package_storage_inventory_candidate_v1.py', ROOT/'research/check_nemotron9b_storage_inventory_candidate_20261108_v1.py', ROOT/'research/check_nemotron9b_package_exec_v11.py']
    files += list(ROOT.glob('nemotron9b_package_*v9.py'))
    files += [ROOT/'nemotron9b_smol_terminal_chain_v1.py', ROOT/'research/check_nemotron9b_smol_terminal_chain_v1.py', ROOT/'research/check_nemotron9b_package_exec_v10.py']
    files += list(ROOT.glob('nemotron9b_package_*v8.py'))
    files += [ROOT/'nemotron9b_completed_smol_metadata_v1.py',ROOT/'research/check_nemotron9b_completed_smol_metadata_v1.py',ROOT/'research/check_nemotron9b_package_exec_v8.py']
    files += list(ROOT.glob('nemotron9b_package_*v7.py'))
    files += [ROOT/'nemotron9b_exec_transition_v1.py',ROOT/'research/check_nemotron9b_exec_transition_v1.py',ROOT/'research/check_nemotron9b_package_exec_v10.py']
    files += list(ROOT.glob('nemotron9b_package_*v6.py'))
    files += list(ROOT.glob('nemotron9b_package_*v5.py'))
    files += list(ROOT.glob('nemotron9b_package_*v4.py'))
    files += list(ROOT.glob('nemotron9b_package_*v3.py'))
    files += [ROOT/'nemotron9b_package_metadata_alias_v1.py',ROOT/'research/prepare_nemotron9b_package_runtime_v1.py',ROOT/'research/check_nemotron9b_package_preparation_v1.py',ROOT/'research/nemotron9b-package-preparation-input-contract-20261107-v1.json']
    files += [ROOT/'historical_b3_payload_partition_v1.py',ROOT/'research/prepare_nemotron9b_package_runtime_v4.py',ROOT/'research/nemotron9b_preparation_history_v1.py',ROOT/'research/migrate_nemotron9b_package_runtime_v5.py',ROOT/'research/check_nemotron9b_package_runtime_v5.py']
    files += [ROOT/'minicpm5_layer0_execution_protocol_v4.py',ROOT/'minicpm5_layer0_runtime_auth_v1.py']
    files += [ROOT/'research/nemotron9b-package-loop-contract-20261107-v3.json',ROOT/'check_nemotron9b_package_qualification_v3.py',ROOT/'check_nemotron9b_package_server_lifecycle_v4.py',ROOT/'check_nemotron9b_package_raw_window_v3.py']
    files += list((PACKAGE/'vllm_neuron/models/nemotron9b').glob('*.py'))
    files += list((PACKAGE/'vllm_neuron/models/nemotron9b').glob('*.json'))
    files += [PACKAGE/'vllm_neuron/__init__.py',PACKAGE/'vllm_neuron/models/__init__.py',PACKAGE/'setup.py']
    files += [PACKAGE/'vllm_neuron/worker'/name for name in ('neuronx_distributed_model_loader.py','neuronx_distributed_model_runner.py','cpu_sampler.py','constants.py','__init__.py')]
    return sorted(str(p)for p in files)


def bundle(spec_ref,review_ref):
    quiet();s=read(spec_ref['path'],spec_ref['sha256']);review=read(review_ref['path'],review_ref['sha256'])
    alias_metadata.validate_spec(s)
    require(s['metadata_limits']==metadata.LIMITS,'Reviewed bounded metadata transport')
    require(s['schema']=='nemotron9b_package_runtime_v11' and s['root_finalized'] is True,'Future root-finalized spec required')
    require(s['package_commit']==COMMIT and s['package_root']==str(PACKAGE),'Exact reviewed package')
    require(re.fullmatch('[a-z0-9-]{1,40}',s['run_id']),'Unique run identifier')
    require(s['stages']==list(STAGES) and s['resources']=={x:resources(x)for x in (*STAGES,'controller')},'All phases/resources retained')
    require(review['schema']=='nemotron9b_package_runtime_source_review_v11','Exact v5 source review schema')
    require(review['passed'] is review['source_only'] is review['independent_review'] is True and review['reviewer']=='/root'
        and review['blockers']==[] and review['execution_spec_sha256']==spec_ref['sha256']
        and review['source_sha256']==s['source_sha256'],'Exact independent root source review')
    require(s['automatic_successor_launch'] is False and s['speedup_denominator'] is None,'No automatic successor or failed denominator')
    require(s['runtime_software_identity']['python']=='3.10'and s['runtime_software_identity']['versions']['vllm']=='0.11.0','Original runtime identity')
    require(s['retirement_policy']=='none','v3 cannot reclaim space or reduce reservations automatically')
    require(s['disk_budget']==dict(root_floor=ROOT_FLOOR,nvme_floor=NVME_FLOOR,root_cap=ROOT_CAP,nvme_cap=NVME_CAP,caps=CAPS),'Unreduced storage policy')
    require(s['controller']==str(ROOT/'nemotron9b_package_controller_v11.py')and s['worker']==str(ROOT/'nemotron9b_package_worker_v11.py'),'Executable closure')
    require(set(required_sources())<=set(s['source_sha256']),'Complete package/lifecycle/source closure')
    require(set(s['units'])==set((*STAGES,'controller'))and len(set(s['units'].values()))==len(STAGES)+1,'Unique retained units')
    for stage,unit in s['units'].items():require(re.fullmatch('trn1-latest-verify-nemotron9b-package-[a-z0-9-]+',unit),'Unit namespace')
    for field,parent in [('control_root',ROOT),('evidence_root',ROOT),('work_root',Path('/mnt/nvme'))]:
        p=Path(s[field]);require(p.parent==parent and p.name.startswith('nemotron9b-package-')and p.resolve()==p,'Fresh output namespace')
    require(len({s[k]for k in ('control_root','evidence_root','work_root')})==3,'Disjoint output roots')
    catalog=read(s['final_predecessor_catalog']['path'],s['final_predecessor_catalog']['sha256'])
    require(catalog['root_finalized'] is True and catalog['all_planned_launches_accounted_for']is True,'Final complete catalog, not preparation snapshot')
    require(catalog['root_release_refs']==s['required_predecessor_releases']and bool(s['required_predecessor_releases']),'All latest actual releases bound')
    for item in s['required_predecessor_releases']:
        release=read(item['path'],item['sha256']);require(release['released']is True and release['all_current_groups_recursively_drained']is True,'Final predecessor physical release')
    from nemotron9b_smol_terminal_chain_v1 import validate as validate_smol_terminal_chain
    validate_smol_terminal_chain(s,catalog,metadata.load)
    require(bool(catalog['rows'])and len({x['unit']for x in catalog['rows']})==len(catalog['rows']),'Nonempty unique final catalog')
    require(not set(s['units'].values())&{x['unit']for x in catalog['rows']},'New units cannot replace predecessors')
    require(all(s['source_sha256'].get(k)==v for k,v in catalog['source_sha256'].items()),'Final catalog source closure')
    require(set(s['payloads'])>=set(s['references'].values()),'Every reference payload bound')
    require(set(s['references'])=={'numeric','cold','long','historical_numeric','historical_cold','historical_long'},'All fresh and migration comparisons')
    for path,row in s['payloads'].items():require(file_stat(path)==row['stat'],'Payload stat changed; no source-time payload read')
    require(Path(s['original_model']).is_dir()and s['original_model']=='/mnt/nvme/models/NVIDIA-Nemotron-Nano-9B-v2','Pinned original model location')
    for name,digest in s['runtime_python_source_sha256'].items():require(s['source_sha256'].get(name)==digest,'Actual runtime Python closure')
    require(bool(s['runtime_python_source_sha256'])and bool(s['runtime_binary_stat']),'Actual runtime sources/binaries bound')
    for name,identity in s['runtime_binary_stat'].items():require(file_stat(name)==identity,'Runtime binary stat changed')
    for item in [s['quality_suite'],s['historical_quality'],*s['reference_metadata'].values()]:
        require(s['source_sha256'].get(item['path'])==item['sha256'],'Reference metadata in source closure')
    require(s['quality_suite']['path']==str(ROOT/'quality-suite.json'),'Original quality suite location')
    cold=read(s['reference_metadata']['cold']['path'],s['reference_metadata']['cold']['sha256'])
    require(cold['complete']is True and cold['frames']==42 and len(cold['jobs'])==14
        and cold['revision']=='6533e8de2c68e4536bf7c411d7a3ce5734111476'
        and cold['payload_sha256']==s['payloads'][s['references']['cold']]['sha256'],'Complete original native cold oracle')
    long=read(s['reference_metadata']['long']['path'],s['reference_metadata']['long']['sha256'])
    require(long['cached_trajectories_complete']is True and len(long['fixtures'])==5
        and long['revision']=='6533e8de2c68e4536bf7c411d7a3ce5734111476','Original five cached trajectories')
    from nemotron9b_package_evidence_v11 import validate_payload_closure
    from nemotron9b_package_window_v11 import contract
    validate_payload_closure(s);contract(s)
    sources(s);return s,catalog


def paths(spec):
    w=Path(spec['work_root']);e=Path(spec['evidence_root'])
    return dict(work=w,evidence=e,view=e/'view',artifact=w/'artifact',primitives=w/'primitives')


def case(spec):
    p=paths(spec)
    return dict(name='nemotron9b-package-'+spec['run_id'],model_path=str(p['view']),served_name='nemotron-nano-9b-v2',
        artifact=str(p['artifact']),batch=4,tp=2,max_model_len=4096,cpu=True,cpu_fast_sampler=False,adapter='packaged_nemotron9b')


def environment(spec,stage):
    w=Path(spec['work_root']);e=Path(spec['evidence_root'])
    return dict(PATH='/home/ubuntu/venvs/vllm_neuron/bin:/opt/aws/neuron/bin:/usr/bin:/bin',HOME='/home/ubuntu',LANG='C.UTF-8',
        XDG_RUNTIME_DIR='/run/user/1000',DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1000/bus',
        PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',PYTHONOPTIMIZE='0',
        PYTHONPATH=str(PACKAGE)+':'+str(ROOT),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME=str(e/'hf'),
        TMPDIR=str(w/'temporary/tmp'),XDG_CACHE_HOME=str(w/'temporary/xdg'),TORCHINDUCTOR_CACHE_DIR=str(w/'temporary/torch'),
        BASE_COMPILE_WORK_DIR=str(w/'compile-work'),NEURON_COMPILE_CACHE_URL=str(w/'compile-cache'),
        NEURON_COMPILED_ARTIFACTS=str(w/'artifact'),NEURON_NEMOTRON9B='1',NEURON_NEMOTRON9B_ALLOW_COMPILE='1'if stage=='compile'else'0',
        NEURON_CPU_SAMPLER_FAST='0',XLA_DISABLE_FUNCTIONALIZATION='1',NEURON_RT_VISIBLE_CORES='0-1',
        VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_PLUGINS='neuron',TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TORCHINDUCTOR_COMPILE_THREADS='1')


def argv(spec,stage,spec_ref,review_ref):
    common=['--spec',spec_ref['path'],'--spec-sha256',spec_ref['sha256'],'--review',review_ref['path'],'--review-sha256',review_ref['sha256']]
    return (['/usr/bin/python3','-B',spec['controller'],'--execute']if stage=='controller'else
            [PYTHON,'-B',spec['worker'],'--phase',stage])+common


def group(spec,stage):return low.GROUP_PREFIX+spec['units'][stage]+'.service'


def service_command(spec,stage,spec_ref,review_ref):
    props=resources(stage);props['RuntimeMaxSec']=props.pop('RuntimeMaxUSec');props['TimeoutStopSec']=props.pop('TimeoutStopUSec')
    if stage!='controller':props.update(BindsTo=spec['units']['controller']+'.service',After=spec['units']['controller']+'.service')
    props.update(StandardOutput='append:'+str(Path(spec['control_root'])/(stage+'.log')),StandardError='inherit')
    return ['systemd-run','--user','--quiet','--unit='+spec['units'][stage],'--service-type=exec',
        *[part for k,v in props.items()for part in ('-p',k+'='+v)],'/usr/bin/env','-i',
        *[k+'='+v for k,v in environment(spec,stage).items()],*argv(spec,stage,spec_ref,review_ref)]


def _service_binding(spec,stage,spec_ref,review_ref,owner,value):
    require(value['LoadState']=='loaded' and low.invocation(value['InvocationID']),'Loaded actual service')
    require(value['Id']==spec['units'][stage]+'.service' and value['ExecStopPost']=='','Exact owned service')
    expected=service_command(spec,stage,spec_ref,review_ref);installed=' '.join(expected[expected.index('/usr/bin/env'):])
    require(re.findall(r'argv\[\]=(.*?)\s*;',value['ExecStart'])==[installed],'Installed command changed')
    for key,wanted in resources(stage).items():
        require(low.duration_seconds(value[key])==low.duration_seconds(wanted)if key in ('RuntimeMaxUSec','TimeoutStopUSec')else value[key]==wanted,'Resource changed '+key)
    if stage!='controller':require(spec['units']['controller']+'.service'in value['BindsTo'].split(),'Worker bound to controller')
    require(value['ControlGroup']in ('',group(spec,stage)),'Owned group')
    if owner:require(value['InvocationID']==owner['invocation_id']and value['ExecMainPID']==str(owner['pid']),'Original invocation/owner')


def _original_exit_sample(actual,owner):
    """Only an absent process or the exact original zombie can trigger retry."""
    if actual is None:return True
    expected=proc_owner(owner)
    if actual != dict(expected,argv=[]):return False
    try:
        with Path('/proc',str(owner['pid']),'stat').open('rb')as stream:raw=stream.read(65537)
    except FileNotFoundError:return True
    require(len(raw)<=65536,'Bounded original exit stat')
    fields=raw.decode().rsplit(')',1)[1].split()
    return fields[0]in ('Z','X')and fields[19]==owner['start_ticks']


def service_identity(spec,stage,spec_ref,review_ref,owner=None,observation=None):
    value=low.properties(spec['units'][stage])if observation is None else observation
    deadline=None
    for attempt in range(21):
        _service_binding(spec,stage,spec_ref,review_ref,owner,value)
        if value['MainPID']=='0':
            if deadline is None:return owner,value
            if low.terminal(value):
                low.gone(proc_owner(owner))
                return owner,value
        else:
            actual=low.proc(int(value['MainPID']))
            valid=actual and actual['argv']==argv(spec,stage,spec_ref,review_ref)and actual['cgroup']=='0::'+group(spec,stage)
            if valid:
                found=dict(**actual,invocation_id=value['InvocationID'])
                require(owner is None or found==owner,'Owner replaced')
                require(deadline is None,'Original process became live after exit sample')
                return found,value
            require(owner is not None and value['MainPID']==str(owner['pid'])and _original_exit_sample(actual,owner),'Actual process argv/group')
        if deadline is None:deadline=time.monotonic()+1.0
        require(attempt<20 and time.monotonic()<deadline,'Original exit transition did not reach retained terminal state')
        time.sleep(.05)
        value=low.properties(spec['units'][stage])
    raise RuntimeError('Unreachable exit observation limit')


def proc_owner(owner):return {k:v for k,v in owner.items()if k!='invocation_id'}
def producer(owner):return dict(pid=owner['pid'],starttime=int(owner['start_ticks']),invocation_id=owner['invocation_id'])


def _walk_error(error):raise error


def _usage_once(root,allow_view,*,ipc_root=None):
    return output_inventory.inventory(root,allow_view,ipc_root=ipc_root)


def usage(root,allow_view=False,*,ipc_root=None):
    """Retry the entire inventory on rename races; never skip vanished entries."""
    try:before=root.lstat()
    except FileNotFoundError:return 0
    require(stat.S_ISDIR(before.st_mode)and not root.is_symlink(),'Ordinary inventory root')
    identity=(before.st_dev,before.st_ino)
    for attempt in range(8):
        try:
            total=_usage_once(root,allow_view,ipc_root=ipc_root)
            after=root.lstat()
            require((after.st_dev,after.st_ino)==identity,'Inventory root replaced')
            return total
        except FileNotFoundError:
            try:current=root.lstat()
            except FileNotFoundError:raise RuntimeError('Inventory root disappeared')
            require((current.st_dev,current.st_ino)==identity,'Inventory root replaced')
            require(attempt<7,'Output inventory did not stabilize after complete rescans')
            time.sleep(.01)
    raise RuntimeError('Unreachable inventory retry limit')


def file_identity(path):
    s=Path(path).lstat();return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns


def storage(spec,reserve=False):
    root_used=usage(Path(spec['control_root']))+usage(Path(spec['evidence_root']),True)
    work=Path(spec['work_root']);ipc_root=work/'temporary/tmp'
    nvme_used=usage(work,ipc_root=ipc_root)
    require(root_used<=ROOT_CAP and nvme_used<=NVME_CAP,'Combined output reservation exceeded')
    for name,cap in CAPS.items():
        require(usage(work/name,ipc_root=ipc_root if name=='temporary'else None)<=cap,'Output category cap '+name)
    values={}
    for name,parent,floor,cap,used in [('root',ROOT,ROOT_FLOOR,ROOT_CAP,root_used),('nvme',Path('/mnt/nvme'),NVME_FLOOR,NVME_CAP,nvme_used)]:
        fs=os.statvfs(parent);free=fs.f_bavail*fs.f_frsize;values[name]=free
        require(free>=floor+(cap-used if reserve else 0),'Unreduced filesystem floor/reservation '+name)
    return dict(free=values,root_used=root_used,nvme_used=nvme_used)


def canonical():return {name:sha(ROOT/name)for name in ('continuation-state.json','roadmap.json','best-known-configurations.json')}


def accelerator_idle():
    value=json.loads(command(['/opt/aws/neuron/bin/neuron-ls','--json-output']))
    require(isinstance(value,list)and not any(x.get('neuron_processes')for x in value),'No other accelerator process')
    return True


def healthy(spec,admission,stage=None,timed=False):
    quiet();low.lock_proof(proc_owner(admission['controller']),admission['lock_identity'])
    allowed=[spec['units']['controller']]+([spec['units'][stage]]if stage else[])
    low.active_heavy(allowed)
    require(not (Path(spec['control_root'])/'cancel.json').exists(),'Root cancelled run')
    if timed:
        for parent,floor in [(ROOT,ROOT_FLOOR),(Path('/mnt/nvme'),NVME_FLOOR)]:
            fs=os.statvfs(parent);require(fs.f_bavail*fs.f_frsize>=floor,'Timed storage floor')
    else:
        storage(spec,reserve=True);require(canonical()==admission['canonical_sha256'],'Canonical changed')
        for path,row in spec['payloads'].items():require(file_stat(path)==row['stat'],'Admitted payload stat changed')
        for path,identity in spec['runtime_binary_stat'].items():require(file_stat(path)==identity,'Runtime binary stat changed')


def server_argv(spec):
    return [PYTHON,'-B','-m','vllm_neuron.models.nemotron9b.serve','--model',case(spec)['model_path'],
        '--served-model-name','nemotron-nano-9b-v2','--tensor-parallel-size','2','--dtype','bfloat16',
        '--max-model-len','4096','--max-num-seqs','4','--no-enable-prefix-caching','--host','127.0.0.1','--port','8010',
        '--additional-config',json.dumps({'override_neuron_config':{'on_device_sampling_config':None,'apply_seq_ids_mask':True,
            'context_encoding_buckets':list(BUCKETS)}},separators=(',',':'))]


def snapshot(spec,stage,spec_ref,review_ref,owner,allow_live=False):
    value=low.properties(spec['units'][stage])
    if value['LoadState']=='not-found':
        require(low.terminal(value) and not value['InvocationID'],'Missing unit must be terminal')
        if owner:low.gone(proc_owner(owner))
        return dict(stage=stage,owner=owner,properties=value,successful=False,identity_verified=False)
    observed,value=service_identity(spec,stage,spec_ref,review_ref,owner,observation=value)
    require(allow_live or low.terminal(value),'Save terminal result before stopping')
    return dict(stage=stage,owner=owner,properties=value,identity_verified=owner is not None,
        successful=owner is not None and low.terminal(value) and value['Result']=='success' and value['ExecMainCode']=='1' and value['ExecMainStatus']=='0')

def retained_drain(spec,stage,spec_ref,review_ref,owner):
    """Successful unit stays retained until independent final external audit."""
    require(owner is not None,'Captured live owner required');samples=[]
    for _ in range(2):
        seen=snapshot(spec,stage,spec_ref,review_ref,owner)
        require(seen['identity_verified'] and seen['successful'],'Successful retained phase required')
        low.gone(proc_owner(owner));physical=low.empty_group(group(spec,stage))
        samples.append(dict(snapshot=seen,group=physical))
    require(samples[0]==samples[1],'Stable retained exit and physical drain')
    return dict(physically_drained=True,successful_exit=True,retained_for_external_audit=True,samples=samples)

def validate_after(spec,stage,spec_ref,review_ref,saved,value):
    before=saved['properties']
    if value['LoadState']=='loaded':
        require(before['LoadState']=='loaded' and value['InvocationID']==before['InvocationID'] and value['ExecMainPID']==before['ExecMainPID'],'Refuse replacement service during cleanup')
        service_identity(spec,stage,spec_ref,review_ref,saved['owner'],observation=value)
    else:require(value['LoadState']=='not-found' and not value['InvocationID'] and low.terminal(value),'Unknown GC state')
    return value

def stop_and_drain(spec,stage,spec_ref,review_ref,saved):
    """Caller must persist saved original status before invoking cleanup."""
    value=validate_after(spec,stage,spec_ref,review_ref,saved,low.properties(spec['units'][stage]))
    if value['LoadState']=='loaded':
        try:command(['systemctl','--user','stop','--no-block',spec['units'][stage]])
        except RuntimeError:
            now=validate_after(spec,stage,spec_ref,review_ref,saved,low.properties(spec['units'][stage]))
            require(now['LoadState']=='not-found','Stop failed without authenticated GC')
    deadline=time.monotonic()+150
    while time.monotonic()<deadline:
        samples=[];pending=False
        for _ in range(2):
            value=validate_after(spec,stage,spec_ref,review_ref,saved,low.properties(spec['units'][stage]))
            # Identity errors above are not ordinary stop progress.
            try:
                require(low.terminal(value) and value['ActiveState'] in ('inactive','failed'),'Stop still in progress')
                if saved['owner']:low.gone(proc_owner(saved['owner']))
                physical=low.empty_group(group(spec,stage))
            except RuntimeError:pending=True;break
            samples.append(dict(properties=value,group=physical))
        if not pending and samples[0]==samples[1]:return dict(physically_drained=True,samples=samples,successful_exit=saved['successful'])
        time.sleep(.2)
    raise RuntimeError('Authenticated physical drain incomplete; retain serial lock')


class AuditGuard:
    def __init__(self,callback):self.callback=callback;self.last=0.
    def force(self,terminal=False):self.callback();self.last=time.monotonic()
    def __call__(self):
        if time.monotonic()-self.last>=1:self.force()


def validate_result(spec,stage,result):
    """Independent metadata acceptance; counts are derived, not trusted claims."""
    from nemotron9b_package_science_v11 import passed,expected_long_frames
    k=kind(stage);e=Path(spec['evidence_root']);w=Path(spec['work_root'])
    require(result.get('complete')is True,'Complete phase result')
    if k=='prepare':
        require(result['passed']is True and result['quality_inputs']==96 and result['raw_tokenizer_checks']==22,'Prepared complete input suite')
        require(result['software']==spec['runtime_software_identity'],'Prepared software identity')
    elif k=='verify-raw':
        proof=read(result['report']['path'],result['report']['sha256'])
        require(proof['complete']is proof['passed']is True and proof['independent_process']is True
            and proof['inference_performed']is False and proof['models_loaded']==0,'Independent raw verifier')
        require(set(proof['phases'])==set(SCIENTIFIC_STAGES),'All original51 scientific phases replay-bound')
        for name,count in [('numeric',32),('cold',84),('long',93)]:
            row=proof['raw_replay'][name]
            require(row['complete']is row['passed']is True and row['frames']==len(row['rows'])==count,'All raw replay frames')
            require(passed([r['metrics']for r in row['rows']])and passed([r['migration']for r in row['rows']]),'Reapplied exact raw numerical thresholds')
    elif k=='alias':
        job=next(j for j in alias_jobs()if j['name']==stage)
        require(result['job']==job and result['passed']is True,'Exact alias case')
        require(len(result['checks'])==(8 if job['case']['rows']==4 else 7),'All alias mappings')
        require(all(x['passed']and x['input_unchanged']and all(v['passed']for v in x['comparisons'].values())for x in result['checks']),'All raw alias bytes')
    elif k=='split':
        job=next(j for j in split_jobs()if j['name']==stage)
        require(result['job']==job and result['cpu_passed']is True and len(result['cpu_checks'])==4,'Exact split case/CPU controls')
        require(result['primitive_passed']is True if job['variant']=='tensor_split'else result.get('control_outcome_confirmed')is True,'Unchanged split candidate/control gate')
    elif k=='compile':
        require(result['passed']is True and len(result['smoke'])==2,'Fresh serving smoke')
        m=read(result['artifact_manifest']['path'],result['artifact_manifest']['sha256'])
        require(m['complete']is True and m['identity']['model']['revision']=='6533e8de2c68e4536bf7c411d7a3ce5734111476','Fresh admitted artifact')
    elif k in ('state-copy','numeric','cold','long'):
        proof=read(result['report']['path'],result['report']['sha256']);require(proof['passed']is True,'Numerical/state gate')
        if k=='state-copy':
            require(proof['complete']and len(proof['patterns'])==2 and proof['state_buffers']==124,'Fresh state proof')
            require(proof['fill_roundtrip_comparisons']==proof['raw_zero_comparisons']==248,'All248 fills/zeros')
            for pattern in proof['patterns']:
                require(len(pattern['fills'])==len(pattern['zero_checks'])==124,'124 distinct buffers per pattern')
                require(all(x['passed']and x['actual_sha256']==x['expected_sha256']for x in pattern['fills']),'Raw fill equality')
                require(all(x['passed']and x['nonzero_bytes']==0 for x in pattern['zero_checks']),'Raw zero equality')
        elif k=='numeric':require(len(proof['rows'])==32 and passed(proof['rows']),'Original strict32')
        elif k=='cold':
            from nemotron9b_package_model_phases_v11 import configure,validate_cold
            configure(spec);validate_cold(proof)
        else:
            info=read(spec['reference_metadata']['long']['path'],spec['reference_metadata']['long']['sha256'])
            require(proof['teacher_forced_decode_steps']==2041 and len(proof['rows'])==93 and passed(proof['rows']),'Original2041/93 strict thresholds')
            keys=('trajectory','fixture','position','slot','phase')
            require([{k:r[k]for k in keys}for r in proof['rows']]==expected_long_frames(info),'Exact reuse/inactive/partial long history')
        if k!='state-copy':
            comparison=read(e/(k+'-migration.json'))
            require(comparison['complete']and comparison['passed']and comparison['frames']=={'numeric':32,'cold':84,'long':93}[k],'Historical candidate migration comparison')
    elif k=='quality':
        proof=read(result['quality']['path'],result['quality']['sha256']);comparison=read(result['comparison']['path'],result['comparison']['sha256'])
        require(proof['technical_complete']and proof['total']==len(proof['rows'])==96,'Complete quality96')
        require(comparison['total']==comparison['identical_texts']==96 and not comparison['lost_correct_cases'],'Quality96 identical outputs/scores')
    else:
        proof=read(result['report']['path'],result['report']['sha256']);workload_rates(proof)
    return True


def workload_rates(proof):
    require(proof['complete']is True and proof['passed']is True,'Complete performance round')
    expected={'short_decode':(128,512,{}),'medium_prompt':(512,128,{}),'long_prompt':(1536,128,{}),
        'short_stochastic':(128,512,dict(temperature=.7,top_p=.9,top_k=40))}
    require(set(proof['workloads'])==set(expected)|{'mixed_lengths'},'All five original workloads')
    rates={}
    def checked(group,n,m,count):
        rows=group['requests'];require(len(rows)==count,'Exact request count')
        require(all(r['usage']['prompt_tokens']==n and r['usage']['completion_tokens']==m for r in rows),'Exact token budgets')
        require(math.isfinite(group['elapsed_s'])and group['elapsed_s']>0,'Positive wall time')
        total=sum(r['usage']['completion_tokens']for r in rows);rate=total/group['elapsed_s']
        require(group['output_tokens']==total and math.isclose(group['throughput_tok_s'],rate,rel_tol=1e-12),'Recomputed throughput')
        return rate
    for name,(n,m,sampling)in expected.items():
        item=proof['workloads'][name];require(len(item['rounds'])==1 and item['sampling']==sampling and item['output_tokens_per_request']==m,'Exact workload protocol')
        require(item['prompt_inputs']==[proof['inputs'][str(n)]],'Original prompt tokens')
        rates[name]={k:checked(item['rounds'][0][k],n,m,1 if k=='single'else 4)for k in ('single','concurrent')}
        rates[name]['streaming']=checked(item['streaming'],n,m,4)
    item=proof['workloads']['mixed_lengths'];require(len(item['rounds'])==1 and item['max_client_inflight']==4 and item['requests_per_round']==16,'Mixed churn contract')
    trial=item['rounds'][0];require([(x['input_tokens'],x['output_tokens'])for x in trial['requests']]==[(32,32),(128,128),(512,256),(1536,128)]*4,'Exact mixed sequence')
    require(all(x['usage']['prompt_tokens']==x['input_tokens']and x['usage']['completion_tokens']==x['output_tokens']for x in trial['requests']),'Mixed budgets')
    require(math.isfinite(trial['elapsed_s'])and trial['elapsed_s']>0,'Mixed wall time')
    total=sum(x['output_tokens']for x in trial['requests']);rate=total/trial['elapsed_s']
    require(trial['output_tokens']==total and math.isclose(rate,trial['throughput_tok_s'],rel_tol=1e-12),'Mixed rate recomputed')
    rates['mixed_lengths']={'mixed':rate};return rates


def completion(spec,admission,guard):
    c=Path(spec['control_root']);e=Path(spec['evidence_root']);rows={}
    for stage in STAGES:
        guard();record=read(c/(stage+'-worker-result.json'));owner=read(c/(stage+'-owner.json'))
        require(record['passed']is True and record['complete']is True and record['phase']==stage and record['producer']==producer(owner)
            and record['lease']==ref(c/(stage+'-lease.json')),'Exact terminal worker result')
        validate_result(spec,stage,record['result']);rows[stage]=ref(c/(stage+'-worker-result.json'))
        if stage in SCIENTIFIC_STAGES:
            from nemotron9b_package_evidence_v11 import verify_phase
            verify_phase(spec,stage,record['result'],owner,record['lease'],guard,hash_files=False)
        if kind(stage)in ('compile','quality','workloads'):
            from nemotron9b_package_server_lifecycle_v11 import validate_chain
            validate_chain(spec,stage,owner,record['result'],admission)
        if kind(stage)=='workloads':
            from nemotron9b_package_window_v11 import validate_window
            validate_window(spec,stage,owner)
    rounds=[workload_rates(read(e/('workloads-'+str(n)+'.json')))for n in range(3)]
    rates={name:{mode:statistics.median(r[name][mode]for r in rounds)for mode in rounds[0][name]}for name in rounds[0]}
    return dict(complete=True,full_model_qualified=False,awaiting_external_raw_byte_audit=True,independent_raw_replay_complete=True,package_commit=COMMIT,stages=len(STAGES),scientific_stages=len(SCIENTIFIC_STAGES),phase_results=rows,
        primitive_alias_graphs=6,primitive_alias_checks=46,additional_candidate_controls=3,split_cases=32,
        fresh_routes=12,numeric_frames=32,state_fills=248,state_zeros=248,cold_frames=84,cold_zero_checks=868,
        long_steps=2041,long_checkpoints=93,quality_cases=96,workloads=5,rounds=3,absolute_median_tok_s=rates,
        original_baseline_qualified=False,speedup_denominator=None,new_speedup_claim=False,family_optimization_complete=False)
