"""Describe a root-audited five-arm Mini run using bounded JSON only.

No tensor reads, native imports, tolerance changes or model qualification.
Fresh-process differences are descriptive; they do not identify an operator.
"""
import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re

ROOT = Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
ARMS = ('unobserved', 'previous213_hooks', 'noop_wrappers', 'dispatch_only', 'full_observer')


def require(value, message):
    if not value:
        raise ValueError(message)


def quiet():
    for name in ('smollm3-cache-ablation-full-v8', 'lfm3-package-paired-full-v2'):
        require(json.loads((ROOT/name/'status.json').read_text())['performance_window'] is False,
                'No metadata work during timed performance')


def read_ref(reference):
    path = Path(reference['path'])
    require(path.is_absolute() and path.suffix == '.json' and path.is_relative_to(ROOT)
            and path.resolve() == path and not path.is_symlink(), 'Bounded local JSON reference')
    before = path.stat()
    require(before.st_size <= 8*2**20, 'JSON size cap')
    raw = path.read_bytes()
    after = path.stat()
    fields = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    require(fields(before) == fields(after) and len(raw) == before.st_size
            and hashlib.sha256(raw).hexdigest() == reference['sha256'], 'Immutable JSON reference')
    return json.loads(raw)


def gone(owner):
    path = Path('/proc')/str(owner['pid'])/'stat'
    try:
        current = path.read_text().rsplit(')', 1)[1].split()[19]
    except FileNotFoundError:
        return
    require(current != str(owner['start_ticks']), 'Root raw auditor must actually exit first')


def summarize(frames, arms, completion):
    require(len(frames) == 32 and {f['index'] for f in frames} == set(range(32)), 'Original complete32 frames')
    require(set(arms) == set(ARMS), 'Exactly five fresh arms')
    require(completion['diagnostic_complete'] is True and completion['frames'] == 160
            and completion['original_hard_gate_preserved'] is True
            and completion['full_model_qualified'] is completion['new_speedup_claim'] is completion['causal_operator_proven'] is False,
            'Complete diagnostic, original gate, no promotion')
    originals = {}; indexed = {}; arm_summaries = []
    for arm in ARMS:
        values = arms[arm]
        require(len(values) == 32 and {v['frame']['index'] for v in values} == set(range(32)), 'Every frame once in each arm')
        by_index = {v['frame']['index']: v for v in values}
        equal = []; unequal = []
        for frame in frames:
            value = by_index[frame['index']]
            require(value['frame'] == frame, 'Original fixture/step/phase identity')
            actual = value['actual_fp32_sha256']; expected = value['original_expected_fp32_sha256']
            require(all(type(x) is str and re.fullmatch('[0-9a-f]{64}', x) for x in (actual, expected)), 'Full vocabulary digest')
            originals.setdefault(frame['index'], expected)
            require(originals[frame['index']] == expected, 'All arms use the same frozen original reference')
            same = actual == expected
            require(value['passed'] is same and value['raw_exact'] is same, 'Unchanged exact-byte endpoint gate')
            (equal if same else unequal).append(frame)
        require(completion['arms'][arm] == len(equal), 'Independent raw audit count agrees')
        indexed[arm] = by_index
        arm_summaries.append(dict(arm=arm, frames=32, exact_original_frames=len(equal),
                                  differing_original_frames=unequal))
    require(set(completion['arms']) == set(ARMS)
            and completion['all_raw_exact'] is all(x['exact_original_frames'] == 32 for x in arm_summaries),
            'Truthful full exact diagnostic result')
    pairs = []
    for left, right in combinations(ARMS, 2):
        changed = []
        for frame in frames:
            a = indexed[left][frame['index']]; b = indexed[right][frame['index']]
            if a['actual_fp32_sha256'] != b['actual_fp32_sha256']:
                changed.append(dict(frame=frame, left_actual_sha256=a['actual_fp32_sha256'],
                                    right_actual_sha256=b['actual_fp32_sha256'],
                                    left_matches_original=a['passed'], right_matches_original=b['passed']))
        pairs.append(dict(left=left, right=right, compared_frames=32,
                          equal_frames=32-len(changed), differing_frames=changed))
    require(len(pairs) == 10, 'All ten arm pairs')
    return dict(frames=160, arm_summaries=arm_summaries, paired_comparisons=pairs,
                all_arms_match_original=completion['all_raw_exact'], all_arm_pairs_equal=all(x['equal_frames'] == 32 for x in pairs),
                full_model_qualified=False, new_speedup_claim=False, causal_operator_proven=False)


def analyze(audit_reference):
    quiet(); audit = read_ref(audit_reference)
    require(audit['reviewer'] == '/root' and audit['passed'] is audit['runtime_successful'] is audit['diagnostic_complete'] is True,
            'Root-audited complete runtime required')
    require(audit['all_groups_drained'] is audit['all_planned_launches_accounted_for'] is True,
            'All planned and actual work accounted for')
    gone(audit['post_auditor_owner'])
    pins = audit['source_evidence_sha256']
    def pinned(reference):
        require(pins.get(reference['path']) == reference['sha256'], 'Root audit binds exact metadata')
        return read_ref(reference)
    spec = pinned(audit['execution_spec']); pinned(audit['source_review'])
    require(spec['arms'] == list(ARMS), 'Original five-arm order')
    science = pinned(spec['source_plan']); inputs = pinned(science['input_context'])
    control = Path(spec['control_root']); frames = inputs['frames']; arms = {}; owners = set()
    for arm in ARMS:
        owner_path = str(control/(arm+'-owner.json'))
        owner = pinned(dict(path=owner_path, sha256=pins[owner_path]))
        identity = (owner['pid'], owner['start_ticks'])
        require(identity not in owners, 'Fresh process for each arm'); owners.add(identity)
        path = str(control/(arm+'-result.json')); result = pinned(dict(path=path, sha256=pins[path]))
        require(result['arm'] == arm and result['pid'] == owner['pid'] and result['start_ticks'] == owner['start_ticks']
                and result['diagnostic_complete'] is True, 'Actual arm producer result')
        inner = result['result']; require(inner['arm'] == arm and inner['observed_frames'] == 32, 'Original arm result')
        values = []
        for row in inner['frames']:
            witness = read_ref(row['witness'])
            require(witness['identity']['arm'] == arm and witness['identity']['frame'] == row['frame']
                    and witness['original_full_vocab'] == 130560 and witness['passed'] is row['passed']
                    and witness['full_model_qualified'] is witness['new_speedup_claim'] is False, 'Original full-vocabulary witness')
            values.append(dict(frame=row['frame'], passed=witness['passed'], raw_exact=witness['raw_exact'],
                               actual_fp32_sha256=witness['actual_fp32_sha256'],
                               original_expected_fp32_sha256=witness['original_expected_fp32_sha256']))
        require(inner['raw_exact_frames'] == sum(v['passed'] is True for v in values)
                and result['passed'] is inner['passed'] is all(v['passed'] for v in values), 'Truthful arm counts')
        arms[arm] = values
    result = summarize(frames, arms, audit['completion'])
    result.update(schema='minicpm5_native_ablation_metadata_analysis_v1', reviewer='/root', complete=True,
                  audit=audit_reference, execution_spec=audit['execution_spec'], tensor_payloads_read=False,
                  limitations=['Five fresh runs characterize these fixed32frames, not universal observer equivalence.',
                               'Identical/different endpoint hashes do not identify a causal operator or intermediate state.',
                               'All original exact gates retained; numeric differences do not invalidate a complete diagnostic.',
                               'Full model numerical/cold-cache/long-decode/slot-reuse/quality/performance gates remain required.'])
    quiet(); return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--audit', required=True); ap.add_argument('--audit-sha256', required=True)
    ap.add_argument('--output', required=True); args = ap.parse_args()
    result = analyze(dict(path=args.audit, sha256=args.audit_sha256))
    path = Path(args.output)
    require(path.parent == ROOT/'research' and path.suffix == '.json', 'New research JSON output')
    with path.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), model_qualified=False)))


if __name__ == '__main__':
    main()
