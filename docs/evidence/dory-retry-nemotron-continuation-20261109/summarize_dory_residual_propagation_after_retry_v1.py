"""Original residual analysis after a proven full saved-review retry success.

Uses the original analysis implementation without changing frames, thresholds,
metrics or conclusions. Requires both actual failed-r2 and successful-r3 exits.
"""
import argparse
import copy
import json
from pathlib import Path

import nemotron9b_after_dory_saved_retry_v1 as completion
import summarize_dory_residual_propagation_v1 as original
from completed_storage_files_v1 import once

Q = Path(__file__).resolve().parent
OUTPUT = Q/'dory-saved-residual-propagation-summary-after-retry-20261109-v1.json'


def from_actual_review(actual_exit_ref):
    done = completion.completed(actual_exit_ref)
    reader = completion.Reader()
    receipt = reader(actual_exit_ref)
    saved = reader(receipt['saved_review'])
    plan = reader(original.PLAN)
    result = original.analyze(saved, plan)
    result.update(actual_saved_review_exit=copy.deepcopy(actual_exit_ref), saved_review=receipt['saved_review'],
        selection_plan=original.PLAN, completed_history_owners=len(done['history']['processes']),
        failed_saved_review_exit=copy.deepcopy(done['history']['failed_residual_saved_review']),
        completion_protocol='nemotron9b_after_dory_saved_retry_v1',
        metadata_only=True, raw_tensor_bytes_read=0, weight_bytes_read=0)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--saved-review-exit',required=True)
    parser.add_argument('--saved-review-exit-sha256',required=True)
    args = parser.parse_args()
    result = from_actual_review(dict(path=args.saved_review_exit,sha256=args.saved_review_exit_sha256))
    print(json.dumps(once(OUTPUT,result)))
