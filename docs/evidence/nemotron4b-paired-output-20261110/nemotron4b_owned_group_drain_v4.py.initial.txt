"""Accept complete per-rank output evidence from the paired supervisor v2.

The existing owned-service cleanup and original result persistence remain.
No service launch, SDK import, native dependency read or runtime admission.
"""
import copy
from pathlib import Path
import re

from nemotron4b_owned_group_drain_v3 import OwnedPhase as OriginalOwnedPhase
from nemotron4b_owned_group_drain_v3 import supervisor_summary as original_summary
from nemotron4b_owned_group_drain_v3 import capture_summary
from nemotron4b_process_handshake_v1 import require


def supervisor_summary(result, nonce):
    if result.get('schema') != 'nemotron4b_gated_rank_pair_v2':
        return original_summary(result, nonce)
    require(set(result['ranks']) == {'rank-0', 'rank-1'}, 'Both original rank outcomes')
    captured = {}
    for role, row in result['ranks'].items():
        # Reuse the exact existing EOF/truncation/partial-result semantics.
        child_view = dict(row, schema='nemotron4b_gated_child_supervision_v3',
                          failure=result['failure'],
                          successful=result['direct_children_successful'])
        captured[role] = capture_summary(child_view)
    compatible = copy.deepcopy(result)
    compatible['schema'] = 'nemotron4b_gated_rank_pair_v1'
    summary = original_summary(compatible, nonce)
    summary.update(supervisor_schema=result['schema'], rank_output_capture=captured)
    return summary


class OwnedPhase(OriginalOwnedPhase):
    def finish_supervisor(self, result, *, persist_result):
        try:
            require(callable(persist_result), 'Durable original supervisor output writer')
            summary = supervisor_summary(result, self.nonce)
            reference = persist_result(result)
            require(type(reference) is dict and set(reference) == {'path', 'sha256'}
                    and type(reference['path']) is str and Path(reference['path']).is_absolute()
                    and type(reference['sha256']) is str
                    and re.fullmatch('[0-9a-f]{64}', reference['sha256']),
                    'Original supervisor receipt must be durably referenced')
            summary['original_supervisor_result'] = copy.deepcopy(reference)
        except BaseException as error:
            self.finish(dict(successful=False, supervisor_adapter_failure=dict(
                type=type(error).__name__, message=str(error)[:2048])))
            raise
        return self.finish(summary)
