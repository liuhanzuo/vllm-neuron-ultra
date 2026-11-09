"""Exact post-residual maintenance contract; no implicit completion or launch."""
import copy
import hashlib
import json
from pathlib import Path
import re
import sys

Q = Path(__file__).resolve().parent
R = Q.parent
sys.path.insert(0, str(R))
import nemotron9b_package_protocol_v20 as p
from research import nemotron9b_after_dory_saved_retry_v1 as residual
import dory_saved_retry_source_partition_v1 as query_partition
from research.nemotron9b_residual_source_partition_v1 import PARENT
from completed_storage_files_v1 import require, ordinary, snapshot_tree, once

CANDIDATE = dict(path=str(Q/'qwen3-8b-completed-original-download-retirement-candidate-20261109-v1.json'),
    sha256='9eb7e33cc61caf8a837f3c0aa1033e1d870afd89bf2b893268e64ccfb79a7c3b')
AUDIT = dict(path=str(Q/'post-residual-storage-candidates-20261109-v1/result.json'),
    sha256='12059aa39d84c8ede1358e7299d3cf1360066af0fae9869e83fb060ace623387')
JOURNAL_ROOT = Path('/var/log/journal/ec2139d93c43b899a9a1defcd3a54e08')
MODEL_ROOT = Path('/mnt/nvme/models/Qwen3-8B')
OUT = R/'completed-qwen-storage-maintenance-20261109-r1'
SCHEMA = 'completed_qwen_storage_after_retry_plan_v1'
EXECUTOR = Q/'run_completed_storage_after_retry_v1.py'
PRIVILEGED = Q/'completed_storage_privileged_after_retry_v1.py'
NAMES = ('completed_storage_files_v1.py', 'check_completed_storage_files_v1.py',
         'completed_storage_contract_v1.py', 'completed_storage_privileged_v1.py',
         'run_completed_storage_serial_v1.py', 'check_completed_storage_contract_v1.py')
NAMES += ('completed_storage_contract_after_retry_v1.py', 'run_completed_storage_after_retry_v1.py',
          'completed_storage_privileged_after_retry_v1.py', 'nemotron9b_after_dory_saved_retry_v1.py',
          'dory_saved_retry_source_partition_v1.py', 'check_completed_storage_after_retry_v1.py')
BUDGET = dict(root_floor=28*2**30, nvme_floor=30*2**30, gzip_cap=256*2**20,
              metadata_cap=64*2**20, wall_seconds=1800, cpu_seconds=1500,
              address_space_bytes=2*2**30, cpus=[4])


def load(ref, *, plan=False):
    require(set(ref) == {'path', 'sha256'} and Path(ref['path']).is_absolute()
            and re.fullmatch('[0-9a-f]{64}', ref['sha256']), 'Exact metadata reference')
    raw = p.metadata.stable_bytes(ref['path'], (32 if plan else 8)*2**20)
    require(hashlib.sha256(raw).hexdigest() == ref['sha256'], 'Authenticated metadata reference')
    value = json.loads(raw)
    if plan: require(value['schema'] == SCHEMA, 'Only explicit maintenance plan uses32MiB envelope')
    return value


def fixed_targets():
    candidate, audit = load(CANDIDATE), load(AUDIT)
    weights, journals = candidate['weights'], audit['closed_journal_candidates']
    expected = {str(MODEL_ROOT/f'model-{i:05d}-of-00005.safetensors') for i in range(1,6)}
    require(len(weights) == 5 and {r['path'] for r in weights} == expected,
            'Only five exact completed Qwen3-8B original downloads')
    require(len(journals) == 8 and len({r['path'] for r in journals}) == 8,
            'Only eight exact previously observed closed journals')
    for row in journals:
        path = Path(row['path'])
        require(path.parent == JOURNAL_ROOT and re.fullmatch(r'user-1000@[0-9a-f-]+\.journal',path.name)
                and row['stat']['st_size'] == 128*2**20
                and row['stat']['st_mtime_ns'] < 1794009600*10**9,
                'Only closed user journals before November7, never active journal')
    require(candidate['official_repository'] == 'Qwen/Qwen3-8B'
            and candidate['official_revision'] == 'b968826d9c46dd6066d109eabc6255188de91218',
            'Exact restorable original model revision')
    return candidate, weights, journals


def evidence(candidate):
    refs = [CANDIDATE, AUDIT, candidate['completed_experiment_state'],
            candidate['completed_experiment_summary'], candidate['official_api_metadata'],
            candidate['current_scope'], *candidate['current_input_maps']]
    refs += [row['download_metadata'] for row in candidate['weights']]
    refs += [v for v in candidate['preserved_int8_startup'].values()
             if isinstance(v, dict) and set(v) == {'path','sha256'}]
    return refs


def fixed_evidence(candidate):
    state = load(candidate['completed_experiment_state'])
    require(state['completed'] is True and state['queue_completed'] is True,
            'Completed old experiment, not an active family')
    original = load(candidate['official_api_metadata'])
    require(original['id'] == candidate['official_repository']
            and original['sha'] == candidate['official_revision']
            and original['gated'] is False and original['private'] is False,
            'Public original repository and exact revision')
    shards = {row['rfilename']:row for row in original['siblings'] if 'lfs' in row}
    for row in candidate['weights']:
        remote = shards[Path(row['path']).name]
        require(remote['lfs']['sha256'] == row['sha256_from_official_metadata']
                and remote['lfs']['size'] == remote['size'] == row['stat']['st_size']
                and row['restore_url'] == 'https://huggingface.co/Qwen/Qwen3-8B/resolve/'
                    +candidate['official_revision']+'/'+Path(row['path']).name,
                'Exact official shard hash, length, restoration URL')
    for ref in evidence(candidate):
        # Evidence also contains Python and download metadata; authenticate bytes.
        if Path(ref['path']).suffix in ('.metadata','.sh'):
            raw = p.metadata.stable_bytes(ref['path'],8*2**20)
            raw.decode('utf8')
            require(hashlib.sha256(raw).hexdigest() == ref['sha256'], 'Preserved exact text dependency pin')
        else:
            require(p.metadata.sha(ref['path'],ref['sha256']) == ref['sha256'], 'Preserved dependency pin')


def validation(plan):
    require(plan['schema'] == SCHEMA and plan['output'] == str(OUT)
            and plan['executor'] == str(EXECUTOR) and plan['privileged_helper'] == str(PRIVILEGED),
            'Fixed new maintenance namespace and executables')
    require(plan['candidate'] == CANDIDATE and plan['audit'] == AUDIT
            and plan['protected_model_spec'] == PARENT and plan['budget'] == BUDGET,
            'Exact admitted candidate, model inputs, and resource bounds')
    candidate, weights, journals = fixed_targets()
    require(plan['weights'] == weights and plan['journals'] == journals,
            'No broadened or changed retirement targets')
    require(plan['scientific_constraints_changed'] is False and plan['runtime_admitted'] is False
            and plan['automatic_retry'] is False and plan['original_qwen_bf16_restore_required'] is True,
            'Storage maintenance never changes science or grants model admission')
    require(all(str(Q/name) in plan['source_sha256'] for name in NAMES), 'Complete new executable/test closure')
    require(set(p.required_sources()) <= set(plan['source_sha256']), 'Complete existing controller/protocol closure')
    return candidate


def combined_sources(plan, completed):
    parent = p.metadata.load(PARENT)
    effective = copy.deepcopy(parent)
    # Only these fixed new .metadata/.sh paths need a separately bounded text reader.
    candidate = load(CANDIDATE)
    exceptional = {ref['path']:ref['sha256'] for ref in evidence(candidate)
                   if Path(ref['path']).suffix in ('.metadata','.sh')}
    require(len(exceptional) == 6 and all(plan['source_sha256'].get(k) == v for k,v in exceptional.items()),
            'Only five exact download receipts and original INT8 launcher text exceptions')
    effective.update(source_sha256={k:v for k,v in plan['source_sha256'].items() if k not in exceptional},
        required_residual_source_anchor=residual.PRODUCER,
        dory_historical_payload_sha256=completed['historical_binary_sha256'],
        required_saved_retry_query_observation=query_partition.OBSERVATION)
    for pins in (parent['source_sha256'], completed['source_sha256']):
        require(all(plan['source_sha256'].get(k) == v for k,v in pins.items()),
                'Every original and completed residual source retained')
    query_partition.classify_sources(effective, p.completed_history.load(p.completed_history.CATALOG))
    targets = {row['path'] for row in plan['weights']+plan['journals']}
    protected = set(plan['source_sha256'])|set(parent['payloads'])|set(parent['runtime_binary_stat'])
    require(not targets & protected, 'No source, raw evidence, runtime binary or active model target')
    return parent, effective


def source_checks(plan, effective, guard=lambda: None):
    fixed_evidence(load(CANDIDATE))
    parts = query_partition.classify_sources(effective, p.completed_history.load(p.completed_history.CATALOG))
    native = parts['query_native_binary_sha256']
    projected = copy.deepcopy(effective)
    projected['source_sha256'] = {path:digest for path,digest in effective['source_sha256'].items() if path not in native}
    p.sources(projected)
    observation = residual.old.load(query_partition.OBSERVATION)
    query_partition.authenticate_native_sources(native, observation, guard)


def protected_trees(weights):
    targets = [row['path'] for row in weights]
    # Protect every installed model directory; only the five explicit files are excluded.
    models = Path('/mnt/nvme/models')
    roots = [str(path) for path in sorted(models.iterdir()) if path.is_dir() and not path.is_symlink()]
    roots += [str(residual.BASE), '/mnt/nvme/neuron-artifacts',
              '/home/ubuntu/experiments/trn1-model-bench-20261030']
    return {root:snapshot_tree(root, targets if root == str(MODEL_ROOT) else ()) for root in roots}


def build(actual_review_exit):
    # No real plan can be emitted while native or saved review is live/failed.
    done = residual.completed(actual_review_exit)
    require((len(done['history']['rows']),len(done['history']['processes']),
             len(done['history']['known_admitted_directories'])) == (1149,349,38),
            'All actual completed predecessor services/owners/directories')
    candidate, weights, journals = fixed_targets()
    fixed_evidence(candidate)
    require(not OUT.exists(), 'No existing maintenance output may be reused')
    pins = {}
    residual.merge(pins, p.metadata.load(PARENT)['source_sha256'])
    residual.merge(pins, done['source_sha256'])
    residual.merge(pins, {ref['path']:ref['sha256'] for ref in [PARENT,*evidence(candidate)]})
    residual.merge(pins, {str(Q/name):p.metadata.sha(Q/name) for name in NAMES})
    residual.merge(pins, {name:p.metadata.sha(name) for name in p.required_sources()})
    plan = dict(schema=SCHEMA, output=str(OUT), executor=str(EXECUTOR),privileged_helper=str(PRIVILEGED),
        candidate=CANDIDATE, audit=AUDIT, protected_model_spec=PARENT, budget=BUDGET,
        completed_residual_review=actual_review_exit, weights=weights,journals=journals,source_sha256=pins,
        scientific_constraints_changed=False,runtime_admitted=False,automatic_retry=False,
        original_qwen_bf16_restore_required=True)
    validation(plan)
    _, effective = combined_sources(plan, done)
    source_checks(plan, effective)
    for row in weights+journals: require(ordinary(row['path']) == row['stat'], 'Original target still unchanged')
    plan['protected_trees'] = protected_trees(weights)
    return plan


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--completed-review-exit', required=True)
    parser.add_argument('--completed-review-exit-sha256', required=True)
    args = parser.parse_args()
    plan = build(dict(path=args.completed_review_exit, sha256=args.completed_review_exit_sha256))
    print(json.dumps(once(Q/'completed-qwen-storage-after-retry-plan-20261109-v1.json', plan)))
