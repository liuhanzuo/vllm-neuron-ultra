"""Regression checks using the original, complete, saved r9 HTTP responses."""
import copy
import json
from pathlib import Path
import sys

Q = Path(__file__).resolve().parent
sys.path[:0] = [str(Q), str(Q.parent)]
from qwen35_sampler_pair_driver_v1 import check_initial_pair as original
from qwen35_sampler_pair_driver_v2 import check_initial_pair as corrected

OUT = Path('/mnt/nvme/neuron-work/qwen35-sampler-pair-r9-v11-20261112')
left, right = [json.loads((OUT / arm / 'http/initial120.json').read_text())
               for arm in ('baseline', 'candidate')]


def reject(fn, a, b):
    try:
        fn(a, b)
    except (ValueError, AssertionError, KeyError, TypeError):
        return
    raise AssertionError('invalid response was accepted')


reject(original, left, right)
corrected(left, right)
checks = 2
for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
    for bad in (None, True, 0, '120', 120.0):
        a, b = copy.deepcopy(left), copy.deepcopy(right)
        for report in (a, b):
            report['rounds'][0]['single']['requests'][0]['usage'][key] = bad
        reject(corrected, a, b)
        checks += 1
    a, b = copy.deepcopy(left), copy.deepcopy(right)
    for report in (a, b):
        del report['rounds'][0]['single']['requests'][0]['usage'][key]
    reject(corrected, a, b)
    checks += 1
for key in ('text', 'usage'):
    b = copy.deepcopy(right)
    row = b['streaming']['requests'][0]
    if key == 'text':
        row[key] += ' changed'
    else:
        row[key]['prompt_tokens_details'] = {'cached_tokens': 1}
    reject(corrected, left, b)
    checks += 1
b = copy.deepcopy(right)
b['rounds'][0]['single']['throughput_tok_s'] *= 2
reject(corrected, left, b)
checks += 1
# Each integer field must remain an integer, including its correct float value.
for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
    a, b = copy.deepcopy(left), copy.deepcopy(right)
    for report in (a, b):
        usage = report['rounds'][0]['single']['requests'][0]['usage']
        usage[key] = float(usage[key])
    reject(corrected, a, b)
    checks += 1
print(json.dumps(dict(passed=True, checks=checks,
    original_saved_response_rejected=True, corrected_saved_response_accepted=True,
    model_or_native_execution=False)))
