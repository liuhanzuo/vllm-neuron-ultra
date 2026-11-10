"""Exercise existing generated worker/helper fixtures with the new pair adapter.

Scientific functions and startup origins remain explicit test doubles. Actual
gated subprocesses and durable receipt I/O run; no model or SDK executes.
"""
import unittest
from unittest.mock import patch

import check_nemotron4b_native_worker_execution_v4 as original
import nemotron4b_gated_rank_pair_v2 as pair
import nemotron4b_owned_group_drain_v4 as group


class WorkerPairIntegration(unittest.TestCase):
    invoke = original.WorkerFlow.invoke

    def setUp(self):
        for name, value in (('pair', pair), ('group', group)):
            replacement = patch.object(original, name, value)
            replacement.start()
            self.addCleanup(replacement.stop)

    test_two_actual_rank_workers_and_helpers_share_receipt_directory = (
        original.WorkerFlow.test_two_actual_rank_workers_and_helpers_share_receipt_directory)
    test_early_helper_failure_bytes_survive_in_durable_worker_supervision = (
        original.WorkerFlow.test_early_helper_failure_bytes_survive_in_durable_worker_supervision)
    test_successful_worker_helper_keeps_output_evidence_in_proof_and_group_summary = (
        original.WorkerFlow.test_successful_worker_helper_keeps_output_evidence_in_proof_and_group_summary)
    test_group_adapter_rejects_forged_completion_and_keeps_v2_compatibility = (
        original.WorkerFlow.test_group_adapter_rejects_forged_completion_and_keeps_v2_compatibility)


if __name__ == '__main__':
    unittest.main(verbosity=2)
