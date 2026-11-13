"""Authenticate DeepSeek's complete live publication; never publish idle here."""
import re
from .lifecycle_shape import active_descriptor,expected_running
from .contract import require
from .io import identity,read,ref

def check_running(bundle,admission_ref,admission,coordinator_ref,coordinator):
    from research import minicpm5_layer1_operator_sdk_runtime_completion_v4 as latest
    path=bundle.spec['running_handoff_path'];value_ref=ref(path);value=read(value_ref)
    previous=bundle.meta.read(admission['freshness']['latest_handoff'])
    expected=expected_running(previous,value['updated_at'],value['deepseek_cold_serial_current_execution']['tool'],
        admission_ref,admission,coordinator_ref,coordinator)
    require(value==expected and read(coordinator_ref)==coordinator,'Exact full original-history DeepSeek running binding')
    observation=dict(path=path,identity=identity(path))
    require(latest.latest_handoff()==observation,'Unknown later running handoff')
    return dict(reference=value_ref,observation=observation,value=value)

def require_external_exit(running,terminal,actual_tool,*,owner_gone,all_owned_drained):
    """Evidence gate only. No idle renderer exists in this running successor."""
    execution=running['deepseek_cold_serial_current_execution']
    require(running['schema']=='deepseek_cold_serial_running_v2'and running['running']is True
        and running['active_serial']==active_descriptor(execution),'Exact running state retained until independent exit')
    require(type(actual_tool)is dict and set(actual_tool)=={'kind','session_id','exit_code','terminal_chunk'}
        and actual_tool['kind']=='tool_session'and type(actual_tool['session_id'])is int and actual_tool['session_id']>0
        and type(actual_tool['exit_code'])is int and 0<=actual_tool['exit_code']<=255
        and type(actual_tool['terminal_chunk'])is str and re.fullmatch('[a-f0-9]{3,64}',actual_tool['terminal_chunk']),
        'External actual tool terminal required; child/outer self-report is insufficient')
    require({k:actual_tool[k]for k in('kind','session_id')}==execution['tool']
        and terminal['admission']==execution['admission']and terminal['outer_owner']==execution['outer_owner']
        and terminal['self_exit_claimed']is False and terminal['actual_external_tool_exit_pending']is True
        and terminal['whole_owned_scope_drained']is True and owner_gone is True and all_owned_drained is True,
        'Actual externally observed matching outer exit plus independent whole-owned-scope drain required')
    # Completing the current catalog/full historical finalizer remains an
    # external duty. Returning a checked binding does not publish an idle state.
    return dict(actual_outer_exit_observed=True,actual_tool=dict(actual_tool),idle_publication_authorized=False,
        full_model_qualified=False,new_speedup_claim=False)
