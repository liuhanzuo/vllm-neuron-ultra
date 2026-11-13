"""Finite lifecycle successor; the accepted raw/science and History stay frozen.

All rendering is pure. Runtime/publication paths and all executing sources are
bound before admission. No actual tool identity or completion is inferred.
"""
import copy
import json
from pathlib import PurePosixPath as Path
import minicpm5_retained_replay_contract_v1 as old

ROOT,Q,PYTHON,LOCK,PLAN,ORIGINAL_PLAN,ACCEPTED,FROZEN,LIMITS,PROPERTIES = (
    old.ROOT,old.Q,old.PYTHON,old.LOCK,old.PLAN,old.ORIGINAL_PLAN,old.ACCEPTED,
    old.FROZEN,old.LIMITS,old.PROPERTIES)
require,canonical,sha,reference,names,environment = (
    old.require,old.canonical,old.sha,old.reference,old.names,old.environment)
SUCCESSOR_FILES=tuple(str(Q/name)for name in (
    'minicpm5_retained_replay_contract_v2.py','minicpm5_retained_replay_caller_v2.py',
    'minicpm5_retained_replay_observer_v2.py','minicpm5_retained_replay_catalog_v1.py',
    'minicpm5_retained_replay_finalizer_v3.py'))
FROZEN_CALLER=dict(zip(old.FILES,(
    'd798f2366976956c5b9f48e09a57d7c4f297a6c1c4358cba554d1e2ecd3b68dd',
    '5d2afbd0ed29ac719181e4cfcee6420617c9130f5e9b64c2439ca16e4e564643',
    'a5a2ecb9f6817bd028108af6423fe1e467db30f5b5abd8f273767dfd51495381',
    'fbf09156cbe567485cd2a7b99c3f12f6654b6cebccf1d93e3721a821610f8f5a')))
FROZEN_CALLER[str(Q/'minicpm5_retained_replay_finalizer_v2.py')]='4e2b9327e6427064cad77a5034dac33afa43f0b7f8bf0560038ac4ef740f9e86'
# Preserve the positional caller/observer entries consumed by the old protocol.
FILES=(SUCCESSOR_FILES[0],old.FILES[1],SUCCESSOR_FILES[1],SUCCESSOR_FILES[2],
       SUCCESSOR_FILES[3],SUCCESSOR_FILES[4],*FROZEN_CALLER)


def publication_paths(run):
    names(run)
    root=ROOT/('minicpm5-retained-verification-'+run)
    return dict(outer_actual_exit=str(root/'outer-actual-exit.json'),
        observer_actual_exit=str(root/'observer-actual-exit.json'),
        successor_catalog=str(Q/('minicpm5-retained-'+run+'-serial-catalog.json')))


def render_binding(context,source_bytes,original_plan_bytes,successor_bytes):
    require(set(source_bytes)==set(old.FILES),'All frozen caller sources retained')
    require(set(successor_bytes)==set(SUCCESSOR_FILES),'Every explicit lifecycle successor source required')
    context=copy.deepcopy(context);pins=context['inherited_source_sha256']
    for path,h in FROZEN_CALLER.items():
        require(pins.get(path)==h,'Every frozen caller/finalizer pin retained')
    for path,data in source_bytes.items():require(sha(data)==FROZEN_CALLER[path],'Frozen caller bytes unchanged')
    for path,data in successor_bytes.items():
        require(type(data)is bytes and 0<len(data)<1<<20,'Finite complete successor source')
        h=sha(data);require(path not in pins or pins[path]==h,'No successor source replacement');pins[path]=h
    result=old.render_binding(context,source_bytes,original_plan_bytes)
    spec=result['spec'];spec.update(schema='minicpm5_retained_replay_caller_spec_v2',
        publication_paths=publication_paths(spec['run_name']))
    validate_spec(spec);result.update(spec_bytes=canonical(spec),spec_sha256=sha(canonical(spec)),
        mandatory_source_sha256={p:spec['source_sha256'][p]for p in FILES},
        final_publisher=dict(path=SUCCESSOR_FILES[4],sha256=pins[SUCCESSOR_FILES[4]]))
    return result


def validate_spec(spec):
    require(type(spec)is dict and spec.get('schema')=='minicpm5_retained_replay_caller_spec_v2',
        'Only reviewed lifecycle v2 caller bindings')
    base=copy.deepcopy(spec);paths=base.pop('publication_paths');base['schema']='minicpm5_retained_replay_caller_spec_v1'
    old.validate_spec(base)
    require(paths==publication_paths(spec['run_name']),'Exact prebound finite completion outputs')
    require(all(spec['source_sha256'].get(p)==h for p,h in FROZEN_CALLER.items())
        and set(FILES)<=set(spec['source_sha256']),'Complete frozen plus successor source closure')
    return spec


def phase_argv(phase,admission_ref):
    require(phase in ('pair','replay'),'Only two original phases');reference(admission_ref)
    return [PYTHON,'-B','-u',FILES[2],'--mode',phase,'--admission',admission_ref['path'],
        '--admission-sha256',admission_ref['sha256']]


def active_descriptor(execution):
    return dict(kind='minicpm5_retained_replay',phase=execution['phase'],tool=copy.deepcopy(execution['tool']),
        unit=execution['worker']['unit'],outer_owner=copy.deepcopy(execution['outer_owner']),
        worker_ref=copy.deepcopy(execution['worker_ref']),admission=copy.deepcopy(execution['admission']))


def running_value(previous,phase,tool,admission_ref,admission,worker,worker_ref,updated_at,previous_ref):
    require(phase in ('pair','replay')and set(tool)=={'kind','session_id'}and tool['kind']=='tool_session'
        and type(tool['session_id'])is int and tool['session_id']>0,'Externally observed actual outer tool session')
    require(previous['serial_execution']is True,'Complete original serial handoff')
    for ref in (admission_ref,worker_ref,previous_ref):reference(ref)
    require(admission['schema']=='minicpm5_retained_replay_outer_admission_v1'
        and admission['lock_identity']==LOCK and worker['unit']==names(admission['run_name'])[phase]
        and worker['nonce']==admission['attempts'][phase]
        and worker['process']['argv']==phase_argv(phase,admission_ref)
        and worker['owner']['pid']==worker['process']['pid']
        and str(worker['owner']['start_ticks'])==str(worker['process']['start_ticks']),'Exact actual admitted worker')
    if phase=='pair':
        require(previous['running']is False and previous['active_serial']is None
            and previous['live_tool_sessions']==previous['active_heavy_units']==[]
            and previous_ref==admission['previous'],'Actual complete idle predecessor required')
    else:
        prior=previous['minicpm5_retained_replay_current_execution']
        require(previous['schema']=='minicpm5_retained_replay_running_v2'and previous['running']is True
            and previous['active_serial']==active_descriptor(prior)
            and previous['live_tool_sessions']==[tool['session_id']]
            and previous['active_heavy_units']==[names(admission['run_name'])['pair']]
            and prior['phase']=='pair'and prior['tool']==tool and prior['admission']==admission_ref
            and previous_ref['path']==str(Q/('continuation-handoff-mini-retained-'+admission['run_name']+'-pair.json')),
            'Same complete actual pair-to-replay predecessor')
    execution=dict(phase=phase,tool=copy.deepcopy(tool),admission=copy.deepcopy(admission_ref),
        outer_owner=copy.deepcopy(admission['outer_owner']),worker=copy.deepcopy(worker),worker_ref=copy.deepcopy(worker_ref),
        lock_identity=copy.deepcopy(LOCK),current_serial_catalog=copy.deepcopy(admission['current_serial_catalog']),
        automatic_successor=False,deletion_authorized=False)
    result=copy.deepcopy(previous)
    result.update(schema='minicpm5_retained_replay_running_v2',updated_at=updated_at,running=True,
        active_serial=active_descriptor(execution),previous=copy.deepcopy(previous_ref),
        publication_only_since_bound_terminal=False,live_tool_sessions=[tool['session_id']],
        active_heavy_units=[worker['unit']],minicpm5_retained_replay_current_execution=execution)
    return result


def publication_bytes(value):
    data=(json.dumps(value,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    require(len(data)<=LIMITS['ordinary_metadata_bytes'],'Original 8 MiB publication cap')
    return data


def publication_ref(path,value):return dict(path=str(path),sha256=sha(publication_bytes(value)))


def metadata_reference(value):
    require(type(value)is dict and set(value)in ({'path','sha256'},{'path','sha256','bytes'}),
        'Exact metadata reference with optional complete-byte length')
    reference({k:value[k]for k in ('path','sha256')})
    require('bytes'not in value or type(value['bytes'])is int and 0<value['bytes']<=LIMITS['spec_bytes'],
        'Bounded optional complete metadata bytes')
