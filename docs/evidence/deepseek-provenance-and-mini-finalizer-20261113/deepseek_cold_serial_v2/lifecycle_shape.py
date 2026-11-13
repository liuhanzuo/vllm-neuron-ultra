"""Accepted Mini v2 complete running semantics, adapted to DeepSeek coordinator."""
import copy
from datetime import datetime
import re
from .contract import LOCK_IDENTITY,reference,require
from .service import argv
def active_descriptor(execution):
    return dict(kind='deepseek_cold_serial',phase=execution['phase'],tool=copy.deepcopy(execution['tool']),
        unit=execution['coordinator']['service_owner']['unit'],outer_owner=copy.deepcopy(execution['outer_owner']),
        coordinator_owner=copy.deepcopy(execution['coordinator_owner']),admission=copy.deepcopy(execution['admission']))

def expected_running(previous,updated_at,tool,admission_ref,admission,coordinator_ref,coordinator):
    phase='cold-diagnostic';previous_ref=admission['freshness']['latest_handoff']
    require(type(tool)is dict and set(tool)=={'kind','session_id'}and tool['kind']=='tool_session'
        and type(tool['session_id'])is int and tool['session_id']>0,'Externally observed actual outer tool session')
    require(type(updated_at)is str and len(updated_at)<48 and datetime.fromisoformat(updated_at).utcoffset()is not None,
        'Zoned external running observation')
    for value in(admission_ref,coordinator_ref,previous_ref):reference(value)
    process=coordinator['process'];owner=coordinator['service_owner']
    require(admission['schema']=='deepseek_cold_serial_admission_v2'and admission['lock_identity']==LOCK_IDENTITY
        and owner['unit']==admission['units']['coordinator']and process['argv']==argv('coordinator',admission_ref)
        and owner['pid']==process['pid']and str(owner['start_ticks'])==str(process['start_ticks'])
        and type(process['pid'])is int and process['pid']>0 and type(process['ppid'])is int and process['ppid']>0
        and re.fullmatch('[0-9a-f]{32}',owner['invocation_id'])
        and process['cgroup'].endswith('/'+owner['unit']+'.service'),'Exact admitted real coordinator identity')
    require(previous['serial_execution']is True and previous['running']is False and previous['active_serial']is None
        and previous['live_tool_sessions']==previous['active_heavy_units']==[],
        'Actual complete idle predecessor; no running-to-fake-idle transition')
    execution=dict(phase=phase,tool=copy.deepcopy(tool),admission=copy.deepcopy(admission_ref),
        outer_owner=copy.deepcopy(admission['outer_owner']),coordinator=copy.deepcopy(coordinator),coordinator_owner=copy.deepcopy(coordinator_ref),
        lock_identity=copy.deepcopy(LOCK_IDENTITY),current_serial_catalog=copy.deepcopy(admission['current_serial_catalog']),
        automatic_successor=False,deletion_authorized=False)
    execution.update(spec=copy.deepcopy(admission['spec']),review=copy.deepcopy(admission['review']),
        phase_units=copy.deepcopy(admission['units']),checkpoint_provenance=copy.deepcopy(admission['checkpoint_provenance']))
    result=copy.deepcopy(previous)
    result.update(schema='deepseek_cold_serial_running_v2',updated_at=updated_at,running=True,
        active_serial=active_descriptor(execution),previous=copy.deepcopy(previous_ref),
        publication_only_since_bound_terminal=False,live_tool_sessions=[tool['session_id']],
        active_heavy_units=[coordinator['service_owner']['unit']],deepseek_cold_serial_current_execution=execution)
    return result
