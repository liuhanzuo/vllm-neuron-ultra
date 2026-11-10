"""Recorded-child v2 receipts accepted by the original owned-group core.

The original v1 service ownership, cleanup, resource and terminal semantics
remain unchanged. This adds only the durable capture-before-grant evidence.
"""
import copy
from pathlib import Path
import re

from nemotron4b_owned_group_drain_v1 import OwnedPhase as OriginalOwnedPhase
from nemotron4b_owned_group_drain_v1 import supervisor_summary as original_summary
from nemotron4b_process_handshake_v1 import require


def supervisor_summary(result,nonce):
    if result.get('schema')!='nemotron4b_gated_child_supervision_v2':
        return original_summary(result,nonce)
    if result.get('successful')is True:
        proof=result['handshake'];ref=proof['capture_record'];grant=proof['grant']
        require(type(ref)is dict and set(ref)=={'path','sha256'}
                and type(ref['path'])is str and Path(ref['path']).is_absolute()
                and type(ref['sha256'])is str and re.fullmatch('[0-9a-f]{64}',ref['sha256']),
                'Original recorded child capture reference')
        recorded=proof['capture_recorded_ns']
        require(type(recorded)is int and grant['capture_ns']<=recorded
                <=grant['guard_complete_ns']<=grant['grant_ns'],
                'Actual capture persistence precedes guard and release')
    compatible=copy.deepcopy(result);compatible['schema']='nemotron4b_gated_child_supervision_v1'
    summary=original_summary(compatible,nonce)
    summary['supervisor_schema']=result['schema']
    summary['capture_persisted_before_grant']=result['successful']is True
    return summary


class OwnedPhase(OriginalOwnedPhase):
    def finish_supervisor(self,result,*,persist_result):
        try:
            require(callable(persist_result),'Durable original supervisor output writer')
            summary=supervisor_summary(result,self.nonce)
            reference=persist_result(result)
            require(type(reference)is dict and set(reference)=={'path','sha256'}
                    and type(reference['path'])is str and reference['path'].startswith('/')
                    and type(reference['sha256'])is str and re.fullmatch('[0-9a-f]{64}',reference['sha256']),
                    'Original supervisor receipt must be durably referenced')
            summary['original_supervisor_result']=copy.deepcopy(reference)
        except BaseException as error:
            self.finish(dict(successful=False,supervisor_adapter_failure=dict(
                type=type(error).__name__,message=str(error)[:2048])))
            raise
        return self.finish(summary)
