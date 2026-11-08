"""Exact dependency/LD transitions. No environment mutation or native imports."""
import hashlib
import json
import re
from pathlib import Path
import smollm3_observer_invariance_environment_v2 as previous

ROOT=Path(__file__).resolve().parent
SITE=previous.SITE
CONTRACT=ROOT/'research/smollm3-observer-environment-source-contract-20261107-v3.json'
STAGES=('clean','startup','import','native_ready','runner_ready','serve')
CLIENT_STAGES=('clean','startup','import','serve')
SYSTEM_KEYS=previous.SYSTEM_KEYS
LD_PREFIX=str(SITE/'cv2')+'/../../lib64'
IDENTITY_KEYS=('pid','start_ticks','ppid','argv','cgroup')
constants=previous.constants
capture_system=previous.capture_system


def require(value,message):
    if not value:raise RuntimeError(message)


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def identity(value):return {k:value[k] for k in IDENTITY_KEYS}


def valid_identity(row):
    require(set(row)==set(IDENTITY_KEYS),'Exact lineage process fields')
    require(type(row['pid']) is int and row['pid']>0 and type(row['ppid']) is int and row['ppid']>0,'Lineage PID type')
    require(type(row['start_ticks']) is str and row['start_ticks'].isdigit() and int(row['start_ticks'])>0,'Lineage start ticks')
    require(type(row['argv']) is list and row['argv'] and all(type(x)is str for x in row['argv']),'Lineage argv')
    require(type(row['cgroup']) is str and row['cgroup'].startswith('0::/'),'Lineage cgroup')


def root_lineage(owner):
    row=dict(schema='smollm3_spawn_lineage_v1',root=identity(owner),edges=[])
    validate_lineage(row,owner,False);return row


def validate_lineage(value,owner=None,spawned=None):
    """Structural chain validation; lifecycle independently authenticates receipts.

    No public count/prefix-pattern input is accepted. The chain is retained in
    the immutable actual root permit, reauthenticated on every worker check.
    """
    require(type(value)is dict and set(value)=={'schema','root','edges'} and value['schema']=='smollm3_spawn_lineage_v1','Exact lineage schema')
    valid_identity(value['root']);edges=value['edges']
    require(type(edges)is list and len(edges)<8,'Original producer cap bounds lineage')
    current=value['root'];seen={(current['pid'],current['start_ticks'])};dispatches=set()
    for edge in edges:
        require(set(edge)=={'parent','child','before_sha256','after_sha256','parent_permit_sha256','sequence'},'Exact lineage edge fields')
        valid_identity(edge['parent']);valid_identity(edge['child'])
        require(edge['parent']==current and edge['child']['ppid']==current['pid'] and edge['child']['cgroup']==current['cgroup'],'Direct parent lineage continuity')
        for key in ('before_sha256','after_sha256','parent_permit_sha256'):
            require(type(edge[key])is str and re.fullmatch('[0-9a-f]{64}',edge[key]) and edge[key]!='0'*64,'Lineage receipt digest')
        require(type(edge['sequence'])is int and 1<=edge['sequence']<=7,'Bounded spawn sequence')
        token=(current['pid'],current['start_ticks'],edge['sequence'])
        child=(edge['child']['pid'],edge['child']['start_ticks'])
        require(token not in dispatches and child not in seen,'No replay/cycle in lineage')
        dispatches.add(token);seen.add(child);current=edge['child']
    if owner is not None:require(current==identity(owner),'Lineage exact current process')
    if spawned is not None:require(bool(edges)is spawned,'Lineage root/spawn binding')
    return value


def extend_lineage(parent_lineage,edge):
    value=json.loads(json.dumps(parent_lineage));value['edges'].append(edge)
    validate_lineage(value,edge['child'],True);return value


def expected_ld(lineage,stage):
    validate_lineage(lineage);require(stage in STAGES,'Known LD phase')
    # Each edge represents a separate actual authenticated exec, whose parent
    # completed exactly one successful runner import before spawning it.
    imports=len(lineage['edges'])+(stage in ('runner_ready','serve'))
    return None if imports==0 else (LD_PREFIX+':')*imports


def ld_state(lineage,stage):
    expected=expected_ld(lineage,stage)
    return dict(schema='smollm3_ld_source_state_v1',lineage_sha256=digest(lineage),
                runner_import_complete=stage in ('runner_ready','serve'),present=expected is not None,
                expected_value_sha256=None if expected is None else hashlib.sha256(expected.encode()).hexdigest())


def permitted(launch,kind,stage,spawned,lineage=None):
    require(stage in (CLIENT_STAGES if kind=='client' else STAGES),'Kind-specific phase')
    old_stage='serve' if stage in ('native_ready','runner_ready','serve') else stage
    result=previous.permitted(launch,kind,old_stage,spawned)
    if kind=='server':
        expected=expected_ld(lineage,stage)
        if expected is not None:result['LD_LIBRARY_PATH']=expected
    return result


def check(actual,launch,kind,stage,spawned,system,lineage=None):
    require(kind in ('server','client') and type(spawned)is bool and not(kind=='client'and spawned),'Known environment identity')
    require(set(system)<=SYSTEM_KEYS,'Unreviewed system environment keys')
    if kind=='server':validate_lineage(lineage,spawned=spawned)
    else:require(lineage is None,'Client has no server lineage')
    wanted=dict(launch,**system);changed=sorted(k for k,v in wanted.items() if actual.get(k)!=v)
    require(not changed,'Environment launch/system value changed; keys='+','.join(changed))
    extras=permitted(launch,kind,stage,spawned,lineage)
    unknown=sorted(set(actual)-set(wanted)-set(extras))
    require(not unknown,'Unreviewed environment keys='+','.join(unknown))
    invalid=sorted(k for k,v in extras.items() if k in actual and actual[k]!=v)
    require(not invalid,'Dependency environment value mismatch; keys='+','.join(invalid))
    required=set(constants()) if kind=='server'and(stage in ('native_ready','runner_ready','serve')or spawned) else set()
    if spawned:required.add('VLLM_USE_V1')
    if kind=='server'and expected_ld(lineage,stage)is not None:required.add('LD_LIBRARY_PATH')
    missing=sorted(required-set(actual));require(not missing,'Required dependency environment missing; keys='+','.join(missing))
    return dict(stage=stage,kind=kind,spawned=spawned,dependency_keys=sorted(set(actual)&set(extras)),
                exact_values_checked=True,unknown_keys=[],values_recorded=False,
                ld_source_state=None if kind=='client' else ld_state(lineage,stage))
