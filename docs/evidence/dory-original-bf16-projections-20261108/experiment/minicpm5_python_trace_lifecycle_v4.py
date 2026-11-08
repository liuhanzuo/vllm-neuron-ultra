"""Mini five-arm lifecycle; stdlib-only until explicit admitted worker entry.

Import only bounded v4 transport, never its old fixed model/stage admission.
Root must provide a freshly finalized specification and independent review.
"""
import copy
import ctypes
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import stat
import struct
import sys
import time
import minicpm5_layer0_execution_protocol_v4 as low
import historical_b3_payload_partition_v1 as history

ROOT=Path(__file__).resolve().parent
ARMS=('unobserved','previous213_hooks','noop_wrappers','noop_trace','full_trace')
STAGES=ARMS
GIB=2**30;MIB=2**20
PREFIX='trn1-latest-verify-minicpm5-python-trace-v3-'
SCIENCE=ROOT/'research/minicpm5-python-trace-source-plan-20261107-v3.json'
SCIENCE_SHA='82070536dea4dffaacbc97b01ae6197cc686d4e092131dc55bd4ea1a6128a5a4'
BASE=ROOT/'research/minicpm5-layer0-complete-predecessors-20261107-v3.json'
BASE_SHA='b4d7182207d20001893f3d11e49cc7c3283189b214d767e3ba6ee81b2830a98f'
MINIMUM_CATALOG=dict(path=str(ROOT/'research/root-post-smol-sio5-failed-complete-predecessors-20261107-v1.json'),
    sha256='b4beec61b5f7b0ec34ca65e8a69172ca9023b2ca55c8e62e83b0a936f7395a3e')
TRANSPORT_ACCEPTANCE=dict(path=str(ROOT/'research/minicpm5-python-trace-transport-root-source-acceptance-20261107-v1.json'),
    sha256='667488243c0fae085ee946a3580c66ae9a93e2063e2b2f1974b9c5e2fdd97348')
ENTRY=ROOT/'minicpm5_python_trace_execute_v4.py'
WORKER=ROOT/'minicpm5_python_trace_worker_v3.py'
DRIVER=ROOT/'minicpm5_python_trace_driver_v3.py'
NATIVE_TORCH_INIT=Path('/home/ubuntu/venvs/model_reference_2026/lib/python3.12/site-packages/torch/__init__.py')
SHARED_SOURCE_PATHS={
    str(NATIVE_TORCH_INIT),
    '/home/ubuntu/venvs/model_reference_2026/lib/python3.12/site-packages/torch/version.py',
    '/home/ubuntu/venvs/vllm_neuron_053_py312/lib/python3.12/site-packages/setuptools-79.0.1.dist-info/METADATA',
    '/home/ubuntu/venvs/vllm_neuron_053_py312/lib/python3.12/site-packages/setuptools-79.0.1.dist-info/WHEEL',
    '/home/ubuntu/venvs/vllm_neuron_053_py312/lib/python3.12/site-packages/setuptools-79.0.1.dist-info/entry_points.txt',
    '/home/ubuntu/venvs/model_reference_2026/lib/python3.12/site-packages/__editable__.transformers-5.23.0.dev0.pth',
}
ORIGINAL_SCIENCE_SHA='3b0af34e9b8162a6cbcceb54697b49a6deb7fef232ed720e25ca23c9e5649e32'
ORIGINAL_SCIENCE=ROOT/'research/minicpm5-native-ablation-source-plan-20261107-v3.json'
require=low.require;read=low.read;sha=low.sha;command=low.command
HISTORICAL_DESIGN=dict(path=str(ROOT/'research/historical-b3-payload-partition-root-design-review-20261107-v1.json'),
    sha256='c579f915346b50d85a0e0843172c329ebd41876cad612c6845a496d594a7ce73')
HISTORICAL_HELPER=dict(path=str(ROOT/'historical_b3_payload_partition_v1.py'),
    sha256='8a3dab13e910f7657cda07e0ff9b1fe3c99d535c2a6a084c4f3e009857c5ab5c')
SUCCESSOR_SOURCES=(
    'minicpm5_python_trace_lifecycle_v4.py','minicpm5_python_trace_execute_v4.py',
    'minicpm5_python_trace_result_parser_v4.py',
    'research/build_minicpm5_python_trace_spec_metadata_20261107_v4.py',
    'research/run_minicpm5_python_trace_external_root_20261107_v4.py',
    'research/audit_minicpm5_python_trace_root_20261107_v5.py',
    'minicpm5_python_trace_plan_v3.py','minicpm5_python_trace_factory_v3.py',
    'minicpm5_python_trace_monitor_v3.py','minicpm5_python_trace_worker_v3.py',
    'minicpm5_python_trace_driver_v3.py','minicpm5_python_trace_adapter_v2.py',
    'minicpm5_python_trace_observer_v2.py','research/minicpm5_python_scale_trace_candidate_20261107_v3.py')
SOURCE_ALIAS_DESIGN=dict(path=str(ROOT/'research/minicpm5-historical-source-alias-design-20261107-v1.json'),
    sha256='603c09ed2b13e8f3a8e113ba37573e6692409d83445d5e7b54e9910565d5da66')
SOURCE_ALIASES={
    '/mnt/nvme/models/SmolLM3-3B-neuron-text-exp20261031/'+name:'/mnt/nvme/models/SmolLM3-3B/'+name
    for name in ('chat_template.jinja','generation_config.json','model.safetensors.index.json','special_tokens_map.json')}
GUARD_POLICY=dict(schema='minicpm5_python_trace_guard_v1',full_audit_seconds=1.0,max_guard_calls=64,
    source_hash_initial_terminal_or_identity_change=True,full_predecessors_initial_terminal=True)


def rename_noreplace(source,destination):
    """Linux atomic publication with no interval of a partial final pathname."""
    libc=ctypes.CDLL(None,use_errno=True)
    operation=libc.renameat2
    operation.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint]
    operation.restype=ctypes.c_int
    if operation(-100,os.fsencode(source),-100,os.fsencode(destination),1)!=0:
        code=ctypes.get_errno()
        raise OSError(code,os.strerror(code),str(destination))


def once(path,value):
    """Durable immutable JSON; failed staging is retained and budgeted."""
    low.clear_window();path=Path(path)
    require(path.is_absolute() and path.suffix=='.json','Absolute JSON receipt')
    raw=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
    require(len(raw)<8*MIB,'Evidence cap')
    stage=path.with_name('.pending-'+path.stem+'-'+str(os.getpid())+'.json')
    fd=os.open(stage,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())
    # A failed publish leaves an ordinary single-link JSON staging receipt.
    # storage() counts it; no final pathname becomes visible before full fsync.
    rename_noreplace(stage,path)
    directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:os.fsync(directory)
    finally:os.close(directory)


def ref(path):return dict(path=str(path),sha256=sha(path))


def final_catalog(spec):
    catalog=read(spec['final_predecessor_catalog']['path'],spec['final_predecessor_catalog']['sha256'])
    require(spec['minimum_predecessor_catalog']==MINIMUM_CATALOG,'Exact historical lower-bound catalog')
    base=read(MINIMUM_CATALOG['path'],MINIMUM_CATALOG['sha256'])
    require(catalog['reviewer']=='/root' and catalog['root_finalized'] is True and catalog['all_planned_launches_accounted_for'] is True
        and catalog['complete_count']==len(catalog['rows'])>=409 and len(catalog['processes'])>=39,'Root-finalized current catalog;409/39 are lower bounds only')
    require(len(base['rows'])==409 and len(base['processes'])==39 and catalog['rows'][:409]==base['rows']
        and catalog['processes'][:39]==base['processes'],'Exact complete original history prefix retained')
    dynamic={'schema','rows','processes','complete_count','source_sha256','root_finalization_observed_at','appended_from','appended_terminal_audit'}
    require(all(catalog.get(k)==v for k,v in base.items() if k not in dynamic),'Retain all prior failure and extra metadata')
    require(all(catalog['source_sha256'].get(k)==v for k,v in base['source_sha256'].items()),'Retain all historical source/evidence pins')
    import smollm3_token_capture_predecessors_v6 as old
    old.validate_rows(catalog['rows'],catalog['processes'])
    review=read(spec['predecessor_catalog_review']['path'],spec['predecessor_catalog_review']['sha256'])
    require(review['reviewer']=='/root' and review['passed'] is True and review['all_planned_launches_accounted_for'] is True
        and review['actual_state_checked_by_root'] is True,'Independent actual final catalog review')
    require(review['catalog']==spec['final_predecessor_catalog'] and review['expected_units']==[x['unit'] for x in catalog['rows']] and review['expected_processes']==catalog['processes']
        and review['catalog_content_sha256']==low.digest(dict(rows=catalog['rows'],processes=catalog['processes'])),'Exact final owners/order binding')
    return catalog


def predecessor_drain(catalog):
    result=low.predecessor_drain(catalog)
    # v4 handles installed/retired identities; also recheck every recorded path.
    for row in catalog['rows']:
        for path in row['recorded_cgroups']:
            expected='/sys/fs/cgroup'+low.GROUP_PREFIX+row['unit']+'.service'
            require(path==expected,'Exact recorded predecessor cgroup')
            low.empty_group(path.removeprefix('/sys/fs/cgroup'))
    return result


def source_stat(path):
    value=Path(path).lstat()
    return dict(device=value.st_dev,inode=value.st_ino,size=value.st_size,mtime_ns=value.st_mtime_ns,
                ctime_ns=value.st_ctime_ns,nlink=value.st_nlink,mode=value.st_mode,uid=value.st_uid)


def alias_shape(aliases,pins):
    require(type(aliases)is dict and set(aliases)==set(SOURCE_ALIASES),'Only four original exact metadata aliases')
    fields={'device','inode','size','mtime_ns','ctime_ns','nlink','mode','uid'}
    for path,row in aliases.items():
        require(type(row)is dict and set(row)=={'target','sha256','readlink','link_identity','target_identity'} and
                row['target']==SOURCE_ALIASES[path] and row['sha256']==pins.get(path)==pins.get(row['target']),
                'Exact alias/target paths and equal pinned source digests')
        require(type(row['readlink'])is str and row['readlink'] and
                (Path(path).parent/row['readlink']).as_posix()==row['target'], 'Exact absolute original readlink text')
        for key,is_link in (('link_identity',True),('target_identity',False)):
            value=row[key]
            require(type(value)is dict and set(value)==fields and all(type(v)is int for v in value.values()) and
                    value['uid']==1000 and value['device']>=0 and value['inode']>0 and 0<value['size']<8*MIB and
                    value['mtime_ns']>=0 and value['ctime_ns']>=0 and value['nlink']==1 and
                    (stat.S_ISLNK(value['mode']) if is_link else stat.S_ISREG(value['mode'])), 'Root-bound original alias/source identity')


def alias_identity(spec,name):
    row=spec['metadata_source_alias_identities'][name];path=Path(name);target=Path(SOURCE_ALIASES[name])
    require(row['target']==str(target) and path.is_absolute() and path.parent.resolve()==path.parent and
            not any(p.is_symlink() for p in path.parents),'Exact original source alias parent')
    link=source_stat(path)
    require(link==row['link_identity'] and stat.S_ISLNK(link['mode']) and path.is_symlink() and
            os.readlink(path)==row['readlink'] and path.resolve()==target,'Original alias link identity/readlink/target')
    require(target.resolve()==target and not any(p.is_symlink() for p in (target,*target.parents)), 'Canonical ordinary alias target')
    actual=source_stat(target)
    require(actual==row['target_identity'] and stat.S_ISREG(actual['mode']), 'Original exact source target identity')
    return dict(link_identity=link,target_identity=actual,readlink=row['readlink'],target=str(target))


def source_sha(spec,path):
    name=str(path)
    if name not in SOURCE_ALIASES:return sha(path)
    before=alias_identity(spec,name);value=sha(SOURCE_ALIASES[name])
    require(value==spec['source_sha256'][name] and before==alias_identity(spec,name), 'Stable original metadata alias source digest')
    return value


def historical_fields(spec):
    """Historical fingerprints are never asserted freshly rehashed/current."""
    pins=spec['source_sha256'];payload=spec['historical_payload_sha256']
    provenance=spec['historical_payload_provenance_sha256']
    for mapping in (pins,payload,provenance):history.pin_shape(mapping)
    require(payload=={row['path']:row['sha256'] for row in history.WHITELIST},'Exact14 historical payload fingerprints')
    require(provenance=={row['path']:row['sha256'] for row in (history.CATALOG,history.AUDIT)},'Exact historical metadata provenance')
    require(not set(pins)&set(payload) and all(Path(k).suffix not in history.FORBIDDEN for k in pins),
            'Disjoint source-only closure; no payload reader exemption')
    require(all(pins.get(k)==v for k,v in provenance.items()),'Historical provenance remains source-authenticated')
    for ref in (HISTORICAL_DESIGN,HISTORICAL_HELPER,SOURCE_ALIAS_DESIGN):
        require(pins.get(ref['path'])==ref['sha256'],'Exact approved historical partition source/design')
    require(all(str(ROOT/name) in pins for name in SUCCESSOR_SOURCES),'Complete reviewed partition successor source closure')
    alias_shape(spec['metadata_source_alias_identities'],pins)
    return dict(pins,**payload)


def historical_closure(spec,catalog):
    combined=historical_fields(spec)
    parts=history.partition(catalog,read_ref=lambda ref:read(ref['path'],ref['sha256']))
    require(spec['historical_payload_sha256']==parts['historical_payload_sha256'] and
            spec['historical_payload_provenance_sha256']==parts['provenance_sha256'],'Authenticated exact historical partition')
    require(all(spec['source_sha256'].get(k)==v for k,v in parts['source_sha256'].items()) and
            all(combined.get(k)==v for k,v in catalog['source_sha256'].items()),'Every original catalog pin retained without mutation')
    return combined


def bundle(spec_ref,review_ref):
    low.clear_window();spec=read(spec_ref['path'],spec_ref['sha256']);review=read(review_ref['path'],review_ref['sha256'])
    require(spec['schema']=='minicpm5_python_trace_runtime_spec_v4' and spec['root_finalized'] is True,'Root finalized runtime specification')
    require(review['passed'] is review['independent_review'] is review['source_only'] is True and review['reviewer']=='/root'
        and review['blockers']==[] and review['execution_spec_sha256']==spec_ref['sha256']
        and review['source_sha256']==spec['source_sha256']
        and review['historical_payload_sha256']==spec['historical_payload_sha256']
        and review['historical_payload_provenance_sha256']==spec['historical_payload_provenance_sha256'],
        'Exact root independent closure/spec/historical review')
    require(review['metadata_source_alias_identities']==spec['metadata_source_alias_identities'], 'Root reviewed exact original metadata alias identities')
    require(review['shared_source_identities']==spec['shared_source_identities'], 'Root reviewed exact six shared source identities')
    require(spec['arms']==list(ARMS) and spec['automatic_successor_launch'] is False,'Fixed five-arm order')
    require(spec['source_plan']==dict(path=str(SCIENCE),sha256=SCIENCE_SHA),'Corrected noop source successor required')
    require(spec['transport_source_acceptance']==TRANSPORT_ACCEPTANCE,'Exact independently accepted trace transport source')
    transport_acceptance=read(TRANSPORT_ACCEPTANCE['path'],TRANSPORT_ACCEPTANCE['sha256'])
    require(transport_acceptance['passed'] is True and transport_acceptance['source_only'] is True,'Accepted source only, never inferred runtime admission')
    require(spec['worker_module']=='minicpm5_python_trace_worker_v3' and spec['worker_source']==str(WORKER),'Corrected immutable callable')
    require(spec['python']=='/home/ubuntu/venvs/model_reference_2026/bin/python' and spec['controller_python']=='/usr/bin/python3','Exact interpreters')
    require(spec['controller_unit']==PREFIX+'controller' and spec['units']=={arm:PREFIX+arm for arm in ARMS},'Exact six service namespace')
    require(spec['control_root']==str(ROOT/'minicpm5-python-trace-control-v3')
        and spec['output_root']==str(ROOT/'minicpm5-native-python-trace-v3'),'Fresh fixed namespaces')
    require(spec['resources']=={s:resources(s) for s in ('controller',*ARMS)} and spec['guard_policy']==GUARD_POLICY,'Exact resource/guard policy')
    require(spec['output_cap_bytes']==2*GIB and spec['control_cap_bytes']==32*MIB and spec['root_floor_bytes']==28*GIB,'Exact disk caps')
    environment(spec)
    science=read(SCIENCE,SCIENCE_SHA);inputs=read(science['input_context']['path'],science['input_context']['sha256'])
    require(low.digest(inputs)==science['input_context']['canonical_sha256'],'Frozen input canonical digest')
    namespace_equivalence(science,inputs)
    from minicpm5_python_trace_factory_v3 import validate_partitions
    validate_partitions(science,inputs)
    accepted=read(spec['scientific_review']['path'],spec['scientific_review']['sha256'])
    require(accepted['reviewer']=='/root' and accepted['passed'] is accepted['independent_review'] is True
        and accepted['source_plan']==spec['source_plan'] and accepted['blockers']==[], 'Root accepted corrected scientific callable')
    tiny=read(spec['actual_tiny_root_review']['path'],spec['actual_tiny_root_review']['sha256'])
    require(tiny['reviewer']=='/root' and tiny['passed'] is True and tiny['actual_native_executed'] is True
        and tiny['source_plan_sha256']==ORIGINAL_SCIENCE_SHA and tiny['noop_original_objects_verified'] is True
        and tiny['model_calls_executed'] is False and tiny['model_free'] is True and tiny['checks']==29
        and tiny['python']==spec['python'] and tiny['all_groups_drained'] is True,'Actual root-reviewed tiny builtin wrapper prerequisite')
    tiny_result=read(tiny['raw_result']['path'],tiny['raw_result']['sha256'])
    require(tiny_result['passed'] is True,'Actual tiny raw result success')
    require(spec['source_sha256'].get(tiny['raw_result']['path'])==tiny['raw_result']['sha256'],'Actual tiny result source closure')
    catalog=final_catalog(spec)
    combined=historical_closure(spec,catalog)
    require(accepted['metadata_source_alias_identities']==spec['metadata_source_alias_identities'], 'Actual root state binds source aliases')
    pins=spec['source_sha256']
    shared_shape(spec['shared_source_identities'],pins)
    for path,expected in spec['shared_source_identities'].items():
        require(expected['sha256']==pins.get(path) and expected['identity']==shared_source_identity(path),
                'Root-bound shared source identity and content pin')
    for path in (ENTRY,Path(__file__).resolve(),WORKER,NATIVE_TORCH_INIT,
                 Path(inputs['native_provenance']['source']).parents[2]/'__init__.py'):
        require(str(path) in pins,'Executable/native backend source pin')
    # Root review itself binds the spec; all earlier prerequisite records must
    # additionally remain part of the repeatedly checked immutable closure.
    for reference in (spec['source_plan'],science['input_context'],spec['scientific_review'],
                      spec['actual_tiny_root_review'],spec['final_predecessor_catalog'],spec['predecessor_catalog_review'],MINIMUM_CATALOG,TRANSPORT_ACCEPTANCE):
        require(pins.get(reference['path'])==reference['sha256'],'Prerequisite receipt source closure')
    for module in ('minicpm5_layer0_execution_protocol_v4','smollm3_token_capture_predecessors_v6',
                   'minicpm5_python_trace_factory_v3','minicpm5_python_trace_monitor_v3',
                   'minicpm5_python_trace_plan_v3','minicpm5_python_trace_result_parser_v4','minicpm5_python_trace_analysis_v1',
                   'smollm3_token_capture_predecessors_v3','smollm3_token_capture_protocol_v3',
                   'smollm3_token_capture_protocol_v6','phi4_saved_boundary_root_protocol_v3','lfm3_package_paired_protocol_v2'):
        require(str(ROOT/(module+'.py')) in pins,'Lifecycle helper source closure')
    for original in (science,inputs,catalog,transport_acceptance):
        require(all(combined.get(k)==v for k,v in original['source_sha256'].items()),'Complete transitive source and historical closure')
    for path,wanted in pins.items():
        key=source_key(spec,path)
        require(source_sha(spec,path)==wanted and source_key(spec,path)==key,'Frozen source changed: '+path)
    require(spec['canonical_sha256'].keys()=={'continuation-state.json','roadmap.json','best-known-configurations.json'},'Complete canonical fingerprint')
    return spec,science,inputs,catalog


def canonical():return {name:sha(ROOT/name) for name in ('continuation-state.json','roadmap.json','best-known-configurations.json')}


def shared_source_identity(path):
    """Read-only installed source; output and lease rules remain single-link."""
    path=Path(path)
    require(str(path) in SHARED_SOURCE_PATHS and path.resolve()==path
            and not any(x.is_symlink() for x in (path,*path.parents)), 'Exact canonical shared source')
    value=path.lstat()
    require(stat.S_ISREG(value.st_mode) and value.st_uid==1000 and value.st_nlink>1 and 0<value.st_size<8*MIB,
            'Original shared regular source')
    return dict(device=value.st_dev,inode=value.st_ino,size=value.st_size,mtime_ns=value.st_mtime_ns,
                ctime_ns=value.st_ctime_ns,nlink=value.st_nlink,mode=value.st_mode,uid=value.st_uid)



def shared_shape(shared,pins):
    require(type(shared)is dict and set(shared)==SHARED_SOURCE_PATHS,'Exactly six pre-existing shared read-only sources')
    fields={'device','inode','size','mtime_ns','ctime_ns','nlink','mode','uid'}
    for path,row in shared.items():
        require(type(row)is dict and set(row)=={'identity','sha256'} and row['sha256']==pins.get(path),'Exact shared source hash binding')
        value=row['identity']
        require(type(value)is dict and set(value)==fields and all(type(v)is int for v in value.values()) and
            value['uid']==1000 and value['device']>=0 and value['inode']>0 and 0<value['size']<8*MIB and
            value['mtime_ns']>=0 and value['ctime_ns']>=0 and value['nlink']>1 and stat.S_ISREG(value['mode']),
            'Root-bound original shared source identity')


def source_key(spec,path):
    name=str(path)
    if name in SOURCE_ALIASES:return alias_identity(spec,name)
    if name in SHARED_SOURCE_PATHS:
        actual=shared_source_identity(path)
        require(actual==spec['shared_source_identities'][name]['identity'], 'Shared read-only source replaced or changed')
        return actual
    return RuntimeGuard.file_key(path)


def namespace_equivalence(science,inputs):
    """Reconstruct the exact accepted source-only successor, not legacy arms."""
    from minicpm5_python_trace_plan_v3 import build,validate_input
    validate_input(inputs)
    original=read(ROOT/'research/minicpm5-native-ablation-source-plan-20261107-v4.json',
        'b10f98118b16d9a23942dd88b065569f482f87bab90dbdffada2aed7f091ee18')
    old_inputs=read(original['input_context']['path'],original['input_context']['sha256'])
    added={k:v for k,v in science['source_sha256'].items() if k not in original['source_sha256']}
    expected=build(old_inputs,original,added)
    require(inputs==expected['input_context'] and science==expected['source_plan'],'Exact new schema/arms with unchanged original science')
    return True


def directory_identity(path):
    path=Path(path);s=path.lstat()
    require(path.resolve()==path and not any(x.is_symlink() for x in (path,*path.parents)) and stat.S_ISDIR(s.st_mode)
        and s.st_uid==1000 and stat.S_IMODE(s.st_mode)==0o700,'Owned private output directory')
    return dict(path=str(path),device=s.st_dev,inode=s.st_ino)


def storage(spec,reserve=False):
    fs=os.statvfs(ROOT);free=fs.f_bavail*fs.f_frsize
    require(free>=28*GIB+(2*GIB+32*MIB if reserve else 0),'Root floor/reservation')
    control=Path(spec['control_root']);total=0;count=0
    if control.exists():
        directory_identity(control)
        for attempt in range(4):
            total=0;count=0;retry=False
            for path in control.iterdir():
                try:s=path.lstat()
                except FileNotFoundError:
                    match=re.fullmatch(r'\.pending-([A-Za-z0-9_-]+)-[1-9][0-9]*\.json',path.name)
                    require(match is not None,
                            'Only an atomic-publication staging file may disappear')
                    final=control/(match.group(1)+'.json');published=final.lstat()
                    require(stat.S_ISREG(published.st_mode) and published.st_uid==1000 and published.st_nlink==1
                        and published.st_size<=16*MIB,'Disappearing staging file needs an owned published final')
                    retry=True;break
                count+=1
                require(stat.S_ISREG(s.st_mode) and s.st_uid==1000 and s.st_nlink==1 and path.suffix in ('.json','.log')
                    and not path.is_symlink() and s.st_size<=16*MIB,'Owned bounded control evidence')
                total+=max(s.st_size,s.st_blocks*512)
            if not retry:break
        require(not retry,'Control publication did not stabilize in four scans')
        require(count<=96 and total<=32*MIB,'Combined control cap')
    return dict(root_free_bytes=free,control_allocated_bytes=total,control_files=count)


def healthy(spec,admission,stage=None,monitor=None):
    low.clear_window();low.lock_proof(proc_owner(admission['controller']),admission['lock_identity']);storage(spec)
    require(canonical()==admission['canonical_sha256']==spec['canonical_sha256'],'Canonical changed')
    require(directory_identity(spec['output_root'])==admission['output_root_identity'],'Output root replaced')
    require(not (Path(spec['control_root'])/'cancel.json').exists(),'Root cancelled experiment')
    low.active_heavy([spec['controller_unit']]+([spec['units'][stage]] if stage else []))
    if monitor is not None:monitor.poll()


def arm_lease(spec,arm,spec_ref,review_ref,admission,owner):
    source=read(SCIENCE,SCIENCE_SHA);process=proc_owner(owner)
    common=dict(**process,uid=1000,python=spec['python'],environment=environment(spec),attempt_id=spec['attempt_id'],
        source_review_accepted=True,serial_exclusive=True,all_predecessors_drained=True)
    runtime=dict(common,schema='minicpm5_layer0_runtime_permit_v1',role='native',arm='HF_native',rank=-1,
        plan_sha256=source['input_context']['canonical_sha256'],output_root=spec['output_root'])
    permit=dict(common,schema='minicpm5_python_trace_arm_permit_v1',arm=arm,source_plan_sha256=low.digest(source),
        fresh_process=True,original_native32_gate_unchanged=True,previous_arm_processes_drained=True,
        source_plan_all_pins_verified_by_root=True,external_resource_limits_verified=True,
        output_root_identity=admission['output_root_identity'],runtime_permit=runtime)
    return dict(schema='minicpm5_python_trace_lease_v1',arm=arm,execution_spec=spec_ref,source_review=review_ref,
        source_plan=spec['source_plan'],owner=owner,controller=admission['controller'],admission=ref(Path(spec['control_root'])/'admission.json'),
        guard_policy=GUARD_POLICY,permit=permit)


def backend_observation():
    """Observe already imported native modules; never import or change flags."""
    torch=sys.modules.get('torch');transformers=sys.modules.get('transformers')
    native=sys.modules.get('transformers.models.llama.modeling_llama')
    if torch is None or transformers is None or native is None or torch.get_num_threads()!=2:return None
    def get(obj,name):
        value=getattr(obj,name,None)
        return dict(available=False) if value is None else dict(available=True,value=value() if callable(value) else value)
    result=dict(schema='minicpm5_native_backend_observation_v1',python=sys.executable,python_version=sys.version,
        module_files={name:str(module.__file__) for name,module in [('torch',torch),('transformers',transformers),('native_llama',native)]},
        versions=dict(torch=str(torch.__version__),transformers=str(transformers.__version__)),
        threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),environment=dict(os.environ),
        affinity=sorted(os.sched_getaffinity(0)),nice=os.getpriority(os.PRIO_PROCESS,0),kernel=list(os.uname()),
        mkldnn={name:get(torch.backends.mkldnn,name) for name in ('enabled','is_available','deterministic')},
        mkl=get(torch.backends.mkl,'is_available'),openmp=get(torch.backends.openmp,'is_available'),
        cpu_capability=get(torch.backends.cpu,'get_cpu_capability'),matmul_precision=get(torch,'get_float32_matmul_precision'),
        deterministic_algorithms=get(torch,'are_deterministic_algorithms_enabled'),
        build_config=str(torch.__config__.show()),parallel_info=str(torch.__config__.parallel_info()),settings_changed=False)
    require(len(json.dumps(result,allow_nan=False).encode())<=64*1024,'Bound backend record')
    return result


def first_driver_guard(caller):
    """Recognize only the two fixed call frames after original_load returned.

    No stack walk or native import. The frozen driver starts run with guard();
    ArmGuard calls this external guard directly from its own __call__.
    """
    worker=sys.modules.get('minicpm5_python_trace_worker_v3')
    driver=sys.modules.get('minicpm5_python_trace_driver_v3')
    if worker is None or driver is None or caller is None or caller.f_back is None:return False
    outer=caller.f_back;code=driver.run.__code__
    return (caller.f_code is worker.ArmGuard.__call__.__code__ and outer.f_code is code
        and outer.f_lineno==code.co_firstlineno+1 and code.co_filename==str(DRIVER))


class RuntimeGuard:
    """Cheap live identity/floors each callback; complete audit <=1s/64calls."""
    def __init__(self,spec,spec_ref,review_ref,admission,owner,arm,lease,lease_path,monitor):
        self.spec=spec;self.sr=spec_ref;self.rr=review_ref;self.admission=admission;self.owner=owner
        self.arm=arm;self.lease=lease;self.path=Path(lease_path);self.monitor=monitor;self.stats={};self.last=None;self.calls=0;self.backend=None
        self.lease_identity=self.file_key(self.path);self.roots={k:directory_identity(spec[k]) for k in ('control_root','output_root')}
        require(lease['guard_policy']==GUARD_POLICY,'Explicit layered guard lease')
    @staticmethod
    def file_key(path):
        s=Path(path).lstat();require(stat.S_ISREG(s.st_mode) and s.st_nlink==1 and not Path(path).is_symlink(),'Immutable regular source/lease')
        return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns
    def source_key(self,path):
        return source_key(self.spec,path)
    def cheap(self):
        low.clear_window();require(low.proc(self.owner['pid'])==proc_owner(self.owner),'Actual worker PID/start/argv/cgroup')
        require(low.proc(self.admission['controller']['pid'])==proc_owner(self.admission['controller']),'Actual controller remains alive')
        require(self.file_key(self.path)==self.lease_identity,'Immutable lease replaced')
        require(all(directory_identity(self.spec[k])==value for k,value in self.roots.items()),'Output directory replaced')
        fs=os.statvfs(ROOT);require(fs.f_bavail*fs.f_frsize>=28*GIB,'Worker root floor')
        require(not (Path(self.spec['control_root'])/'cancel.json').exists(),'Root cancelled worker')
    def force(self,terminal=False):
        historical_fields(self.spec)
        self.cheap();service_identity(self.spec,self.arm,self.sr,self.rr,self.owner)
        service_identity(self.spec,'controller',self.sr,self.rr,self.admission['controller'])
        healthy(self.spec,self.admission,self.arm,self.monitor)
        for path,wanted in self.spec['source_sha256'].items():
            key=self.source_key(path)
            if terminal or self.stats.get(path)!=key:
                require(source_sha(self.spec,path)==wanted and self.source_key(path)==key,'Source changed during worker')
                self.stats[path]=key
        self.last=time.monotonic();self.calls=0
    def __call__(self):
        self.cheap();self.calls+=1
        if self.last is None or time.monotonic()-self.last>=1 or self.calls>=64:self.force()
        if self.backend is None and first_driver_guard(sys._getframe(1)):
            observed=backend_observation()
            require(observed is not None,'Original loader completed with expected native backend before first forward')
            require(all(path in self.spec['source_sha256'] for path in observed['module_files'].values()),'Pinned actual native backend modules')
            self.backend=observed;once(Path(self.spec['control_root'])/(self.arm+'-backend-before.json'),observed)
        return copy.deepcopy(self.lease['permit'])


class AuditGuard:
    """Bound controller/external raw validation overhead without dropping floors."""
    def __init__(self,full_audit):self.full_audit=full_audit;self.last=None;self.calls=0
    def __call__(self,force=False):
        low.clear_window();fs=os.statvfs(ROOT);require(fs.f_bavail*fs.f_frsize>=28*GIB,'Audit root floor')
        self.calls+=1
        if force or self.last is None or self.calls>=64 or time.monotonic()-self.last>=1:
            self.full_audit();self.last=time.monotonic();self.calls=0


def raw_verify(ref,directory,guard):
    from minicpm5_python_trace_result_parser_v4 import raw_verify as parser
    return parser(ref,directory,guard)


def bf16_to_finite_fp32(raw):
    from minicpm5_python_trace_result_parser_v4 import bf16_to_finite_fp32 as parser
    return parser(raw)


def validate_arm_result(result,arm,owner,science):
    from minicpm5_python_trace_result_parser_v4 import validate_arm_result as parser
    return parser(result,arm,owner,science)


def completion(spec,science,inputs,admission,guard):
    from minicpm5_python_trace_result_parser_v4 import completion as parser
    return parser(spec,science,inputs,admission,guard)


def resources(stage):
    require(stage in (*ARMS,'controller'),'Exact arm/controller')
    controller=stage=='controller'
    return dict(MemoryMax=str(512*MIB if controller else 26*GIB),MemoryHigh=str(384*MIB if controller else 24*GIB),
        MemorySwapMax='0',TasksMax='64' if controller else '256',AllowedCPUs='6-7',CPUAffinity='6-7',Nice='19',
        KillMode='control-group',RemainAfterExit='yes',SendSIGKILL='yes',WorkingDirectory=str(ROOT),
        LimitFSIZE=str(16*MIB),RuntimeMaxUSec='60h' if controller else '12h',TimeoutStopUSec='2min' if controller else '30s')


def environment(spec):
    require(re.fullmatch('[0-9a-f]{32}',spec['attempt_id']) is not None,'Exact common attempt')
    return dict(PATH='/usr/bin:/bin',HOME='/home/ubuntu',LANG='C.UTF-8',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        XDG_RUNTIME_DIR='/run/user/1000',DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1000/bus',MINI_LAYER0_ATTEMPT=spec['attempt_id'],
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')


def argv(spec,stage,spec_ref,review_ref):
    args=[spec['controller_python'] if stage=='controller' else spec['python'],'-B',str(ENTRY),
        '--execute' if stage=='controller' else '--worker']
    if stage!='controller':args+=['--arm',stage]
    return args+['--spec',spec_ref['path'],'--spec-sha256',spec_ref['sha256'],'--review',review_ref['path'],'--review-sha256',review_ref['sha256']]


def service_command(spec,stage,spec_ref,review_ref):
    props=resources(stage);props['RuntimeMaxSec']=props.pop('RuntimeMaxUSec');props['TimeoutStopSec']=props.pop('TimeoutStopUSec')
    if stage!='controller':props.update(BindsTo=spec['controller_unit']+'.service',After=spec['controller_unit']+'.service')
    props.update(StandardOutput='append:'+str(Path(spec['control_root'])/(stage+'.log')),StandardError='inherit')
    return ['systemd-run','--user','--quiet','--unit='+unit_for(spec,stage),'--service-type=exec',*[part for k,v in props.items() for part in ('-p',k+'='+v)],
        '/usr/bin/env','-i',*[k+'='+v for k,v in environment(spec).items()],*argv(spec,stage,spec_ref,review_ref)]


def unit_for(spec,stage):return spec['controller_unit'] if stage=='controller' else spec['units'][stage]


def group_for(spec,stage):return low.GROUP_PREFIX+unit_for(spec,stage)+'.service'


def service_identity(spec,stage,spec_ref,review_ref,owner=None,observation=None):
    # One observed systemd snapshot per identity decision: no hidden second show.
    value=low.properties(unit_for(spec,stage)) if observation is None else observation
    require(value['LoadState']=='loaded' and low.invocation(value['InvocationID']),'Loaded actual Mini service')
    require(value['Id']==unit_for(spec,stage)+'.service' and value['ExecStopPost']=='','Exact unit/empty optional cleanup')
    expected=service_command(spec,stage,spec_ref,review_ref);installed=' '.join(expected[expected.index('/usr/bin/env'):])
    require(re.findall(r'argv\[\]=(.*?)\s*;',value['ExecStart'])==[installed],'Actual installed Mini argv')
    for key,wanted in resources(stage).items():
        require(low.duration_seconds(value[key])==low.duration_seconds(wanted) if key in ('RuntimeMaxUSec','TimeoutStopUSec') else value[key]==wanted,'Actual phase resource '+key)
    if stage!='controller':require(spec['controller_unit']+'.service' in value['BindsTo'].split() and spec['controller_unit']+'.service' in value['After'].split(),'Worker lifecycle bound to controller')
    require(value['ControlGroup'] in ('',group_for(spec,stage)),'Exact Mini cgroup')
    if owner is not None:require(value['InvocationID']==owner['invocation_id'] and value['ExecMainPID']==str(owner['pid']),'Saved owner invocation/PID changed')
    if value['MainPID']!='0':
        actual=low.proc(int(value['MainPID']));require(actual is not None and actual['argv']==argv(spec,stage,spec_ref,review_ref) and actual['cgroup']=='0::'+group_for(spec,stage),'Real process argv/cgroup')
        found=dict(**actual,invocation_id=value['InvocationID'])
        require(owner is None or found==owner,'Real owner PID/start replaced');owner=found
    return owner,value


def proc_owner(owner):return {k:v for k,v in owner.items() if k!='invocation_id'}


def snapshot(spec,stage,spec_ref,review_ref,owner,allow_live=False):
    value=low.properties(unit_for(spec,stage))
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
        low.gone(proc_owner(owner));group=low.empty_group(group_for(spec,stage))
        samples.append(dict(snapshot=seen,group=group))
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
    value=validate_after(spec,stage,spec_ref,review_ref,saved,low.properties(unit_for(spec,stage)))
    if value['LoadState']=='loaded':
        try:command(['systemctl','--user','stop','--no-block',unit_for(spec,stage)])
        except RuntimeError:
            now=validate_after(spec,stage,spec_ref,review_ref,saved,low.properties(unit_for(spec,stage)))
            require(now['LoadState']=='not-found','Stop failed without authenticated GC')
    deadline=time.monotonic()+150
    while time.monotonic()<deadline:
        samples=[];pending=False
        for _ in range(2):
            value=validate_after(spec,stage,spec_ref,review_ref,saved,low.properties(unit_for(spec,stage)))
            # Identity errors above are not ordinary stop progress.
            try:
                require(low.terminal(value) and value['ActiveState'] in ('inactive','failed'),'Stop still in progress')
                if saved['owner']:low.gone(proc_owner(saved['owner']))
                group=low.empty_group(group_for(spec,stage))
            except RuntimeError:pending=True;break
            samples.append(dict(properties=value,group=group))
        if not pending and samples[0]==samples[1]:return dict(physically_drained=True,samples=samples,successful_exit=saved['successful'])
        time.sleep(.2)
    raise RuntimeError('Authenticated physical drain incomplete; retain serial lock')


