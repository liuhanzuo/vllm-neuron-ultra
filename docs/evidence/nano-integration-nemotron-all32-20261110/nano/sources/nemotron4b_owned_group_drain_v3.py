"""Accept v3 child output evidence without promoting incomplete capture.

Preserves v2 capture-before-grant and the original full owned-group cleanup.
Only the result adapter changes; no services are launched or admitted here.
"""
import copy
from pathlib import Path
import re

from nemotron4b_owned_group_drain_v2 import OwnedPhase as OriginalOwnedPhase
from nemotron4b_owned_group_drain_v2 import supervisor_summary as original_summary
from nemotron4b_process_handshake_v1 import require


def capture_summary(result):
    require(type(result)is dict and result.get('schema')=='nemotron4b_gated_child_supervision_v3',
            'Exact v3 child output evidence')
    value=result['output_capture']
    require(type(value)is dict and set(value)=={'eof','truncated','drain_timed_out','drain_seconds'},
            'Complete v3 bounded output capture fields')
    for key in('eof','truncated'):
        require(type(value[key])is dict and set(value[key])=={'stdout','stderr'}
                and all(type(x)is bool for x in value[key].values()),'Typed per-stream capture flags')
    require(type(value['drain_timed_out'])is bool and type(value['drain_seconds'])is float
            and value['drain_seconds']==.25,'Original bounded post-cleanup drain')
    require(all(not(value['eof'][name]and value['truncated'][name])for name in('stdout','stderr')),
            'Truncated pipe cannot also claim observed EOF')
    complete=all(value['eof'].values())and not any(value['truncated'].values())
    require(type(result['output_pipe_capture_complete'])is bool
            and result['output_pipe_capture_complete']==complete
            and(not value['drain_timed_out']or not complete),'Consistent pipe capture completion')
    partial=result['failure']is not None or not complete
    require(type(result['output_may_be_partial'])is bool
            and result['output_may_be_partial']==partial,'Explicit partial workload output status')
    require(type(result['successful'])is bool,'Typed actual child outcome')
    if result['successful']:
        require(complete and not partial and not value['drain_timed_out'],
                'Successful child requires complete untruncated output')
    return dict(output_pipe_capture_complete=complete,
                output_capture=copy.deepcopy(value),output_may_be_partial=partial)


def supervisor_summary(result,nonce):
    if result.get('schema')!='nemotron4b_gated_child_supervision_v3':
        return original_summary(result,nonce)
    captured=capture_summary(result)
    compatible=copy.deepcopy(result);compatible['schema']='nemotron4b_gated_child_supervision_v2'
    summary=original_summary(compatible,nonce)
    summary['supervisor_schema']=result['schema'];summary.update(captured)
    return summary


class OwnedPhase(OriginalOwnedPhase):
    def finish_supervisor(self,result,*,persist_result):
        try:
            require(callable(persist_result),'Durable original supervisor output writer')
            summary=supervisor_summary(result,self.nonce)
            reference=persist_result(result)
            require(type(reference)is dict and set(reference)=={'path','sha256'}
                    and type(reference['path'])is str and Path(reference['path']).is_absolute()
                    and type(reference['sha256'])is str and re.fullmatch('[0-9a-f]{64}',reference['sha256']),
                    'Original supervisor receipt must be durably referenced')
            summary['original_supervisor_result']=copy.deepcopy(reference)
        except BaseException as error:
            self.finish(dict(successful=False,supervisor_adapter_failure=dict(
                type=type(error).__name__,message=str(error)[:2048])))
            raise
        return self.finish(summary)
