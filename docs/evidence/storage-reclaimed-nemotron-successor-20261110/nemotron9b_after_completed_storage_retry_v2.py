"""Authenticate actual post-residual storage completion for a future successor.

No default exit, launch, cleanup, byte-level archive review or physical release.
The producer's actual external success and complete retained history are required.
Synthetic metadata belongs only in unit-test memory, never in runtime receipts.
"""
import copy
import hashlib
import json
from pathlib import Path
import re

import completed_storage_contract_after_retry_v3 as storage
import completed_storage_files_v1 as files

residual = storage.residual
require = files.require
Q, OUT = storage.Q, storage.OUT
PLAN = Q/'completed-qwen-storage-after-retry-plan-20261110-v3.json'
SOURCE_REVIEW = Q/'completed-qwen-storage-after-retry-source-review-20261110-v3.json'
PREPARATION = dict(path=str(Q/'completed-storage-source-preparation-review-20261109-v1.json'),
    sha256='330b9f9684e0e66e546bab8212b378a80c059f6ccf584306b56178390bbaba75')
PREPARATION_V3 = dict(path=str(Q/'completed-storage-v3-preparation-checks-20261110-v1.json'),
    sha256='16fcbef3744c8574f83788f298d57ea505eb0e81453fb32824990d440e101961')
OUTPUTS = {'intent':'intent.json', 'admission':'admission.json',
           'predecessor_history':'predecessor-history.json',
           'predecessor_drain':'predecessor-drain.json',
           'protected_before':'protected-before.json',
           'verification':'verified-before-retirement.json',
           'actions':'actions.jsonl', 'result':'result.json'}


class Reader:
    """Exact bounded metadata only; never opens original weights or gzip files."""
    def __init__(self): self.pins, self.bytes = {}, 0

    def raw(self, ref):
        require(set(ref) == {'path','sha256'}, 'Explicit completion reference')
        residual.merge(self.pins,{ref['path']:ref['sha256']})
        path = Path(ref['path'])
        require(path.suffix == '.json' or path == OUT/'actions.jsonl', 'Only exact completion metadata')
        cap = 32*2**20 if path == PLAN else (2*2**20 if path.suffix == '.jsonl' else 8*2**20)
        raw = storage.p.metadata.stable_bytes(path,cap)
        require(hashlib.sha256(raw).hexdigest() == ref['sha256'], 'Exact completion metadata bytes')
        self.bytes += len(raw)
        require(self.bytes <= 128*2**20, 'Bounded complete maintenance metadata reads')
        return raw

    def __call__(self, ref):
        raw = self.raw(ref)
        if Path(ref['path']).suffix == '.jsonl': return [json.loads(line) for line in raw.splitlines()]
        value = json.loads(raw)
        if Path(ref['path']) == PLAN:
            require(value['schema'] == storage.SCHEMA, 'Only fixed maintenance schema uses32MiB')
        return value


def validate_verified(plan, verified):
    weights, journals = verified['weights'], verified['journals']
    require(verified['all_five_weights_authenticated'] is True
            and verified['all_eight_journals_decompressed'] is True
            and len(weights) == 5 and len(journals) == 8,
            'All thirteen targets verified before any retirement')
    for actual, original in zip(weights,plan['weights']):
        require(actual == dict(path=original['path'],stat=original['stat'],
            bytes=original['stat']['st_size'],sha256=original['sha256_from_official_metadata']),
            'All five exact full original LFS hashes, sizes and identities')
    for actual, original in zip(journals,plan['journals']):
        require(actual['source'] == original['path'] and actual['source_stat'] == original['stat']
                and actual['original_bytes'] == original['stat']['st_size']
                and actual['archive'] == str(OUT/(Path(original['path']).name+'.gz'))
                and actual['independently_decompressed'] is True
                and actual['archive_stat']['st_size'] == actual['gzip_bytes']
                and actual['archive_stat']['st_nlink'] == 1
                and 0 < actual['gzip_bytes'] <= storage.BUDGET['gzip_cap']
                and all(re.fullmatch('[0-9a-f]{64}',actual[key])
                        for key in ('original_sha256','gzip_sha256')),
                'Eight exact complete independently decompressed archives')
    require(len(verified['availability']) == 5, 'Every original shard had a fresh HEAD check')
    for actual, original in zip(verified['availability'],plan['weights']):
        require(actual == dict(path=original['path'],status=200,bytes=original['stat']['st_size'],body_bytes_read=0),
                'Original revision remained downloadable without replacement')


def validate_actions(plan, verified_ref, actions):
    targets = plan['weights']+plan['journals']
    require(len(actions) == 26, 'All thirteen before/after retirement records')
    for index, row in enumerate(targets):
        before, after = actions[2*index:2*index+2]
        require(before == dict(phase='before',path=row['path'],verification=verified_ref)
                and after['phase'] == 'after' and after['retired'] == row['path'],
                'Exact ordered retirement and durable pre-verification')
        if index < 5:
            require(after == dict(phase='after',retired=row['path'],restore_url=row['restore_url'],
                    sha256=row['sha256_from_official_metadata']), 'Exact original checkpoint restoration metadata')
        else:
            require(after['archive'] == str(OUT/(Path(row['path']).name+'.gz'))
                    and after['unused']['complete'] is True and not after['unused']['matches']
                    and not after['unused']['errors'], 'Archived closed journal and final privileged unused check')


def verify_layout(plan, verified):
    expected = set(OUTPUTS.values()) | {Path(row['archive']).name for row in verified['journals']}
    require({path.name for path in OUT.iterdir()} == expected, 'No failed, missing or unaccounted maintenance output')
    require({str(path) for path in storage.R.glob('completed-qwen-storage-maintenance-*')} == {str(OUT)},
            'No unaccounted intervening storage maintenance directory')
    for path in OUT.iterdir(): files.ordinary(path)
    for row in plan['weights']+plan['journals']:
        path = Path(row['path'])
        require(not path.exists() and not path.is_symlink(), 'Exact original target remains retired')
    for row in verified['journals']:
        require(files.ordinary(row['archive']) == row['archive_stat'], 'Retained archive identity unchanged')


def extend(done, receipt_ref, receipt, plan, read, *, gone=storage.p.low.gone, layout=verify_layout):
    """Extend a separately authenticated actual residual completion; injectable tests."""
    residual.external(receipt,'storage-maintenance')
    require(receipt['plan']['path'] == str(PLAN) and receipt['source_review']['path'] == str(SOURCE_REVIEW),
            'Exact separately reviewed maintenance plan namespace')
    require(plan['completed_residual_review'] == done['history']['completed_residual_review'],
            'Same original completed residual review')
    for key,name in OUTPUTS.items():
        require(receipt[key]['path'] == str(OUT/name), 'Exact maintenance record namespace: '+key)
    records = {key:read(receipt[key]) for key in OUTPUTS}
    source_review = read(receipt['source_review'])
    ad,intent,result = (records[key] for key in ('admission','intent','result'))
    owner = receipt['owner']
    expected = ['/usr/bin/python3','-B',str(storage.EXECUTOR),'--plan',receipt['plan']['path'],
                '--plan-sha256',receipt['plan']['sha256'],'--review',receipt['source_review']['path'],
                '--review-sha256',receipt['source_review']['sha256']]
    require(owner['argv'] == expected and ad['owner'] == intent['owner'] == result['owner'] == owner
            and ad['lock'] == intent['lock'], 'One exact directly invoked maintenance owner')
    require(intent['plan'] == ad['plan'] == result['plan'] == source_review['plan'] == receipt['plan']
            and intent['source_review'] == ad['source_review'] == receipt['source_review'],
            'Actual intent/admission/result/source-review bindings')
    require(source_review['schema'] == 'completed_qwen_storage_after_retry_source_review_v3'
            and source_review['reviewer'] == '/root' and source_review['passed'] is True
            and source_review['blockers'] == [] and source_review['source_sha256'] == plan['source_sha256'],
            'Final exact source review, not source-only preparation approval')
    require(intent['actual_self_exit_claimed'] is False and ad['fully_drained'] is True
            and ad['maintenance_only'] is True and ad['performance_measurement'] is False,
            'Serial maintenance cannot be a model/performance run')
    prior = done['history']
    require(records['predecessor_history'] == prior
            and ad['predecessor_history'] == receipt['predecessor_history']
            and ad['predecessor_drain'] == receipt['predecessor_drain']
            and len(prior['rows']) == 1149 and len(prior['processes']) == 349
            and len(prior['known_admitted_directories']) == 38
            and isinstance(prior.get('failed_residual_saved_review'),dict),
            'Every actual predecessor retained without a truncated history')
    drain = records['predecessor_drain']
    require(drain['complete'] is True and (drain['services'],drain['processes']) == (1149,349)
            and [row['unit'] for row in drain['observations']] == [row['unit'] for row in prior['rows']]
            and all(len(row['samples']) == 2 for row in drain['observations']),
            'Complete double physical drain at actual maintenance admission')
    require(ad['canonical_sha256'] == prior['canonical_sha256'], 'Original canonical inputs at admission')
    protected = records['protected_before']
    require(protected['trees'] == plan['protected_trees']
            and protected['unused']['complete'] is True and not protected['unused']['matches']
            and not protected['unused']['errors'], 'Original retained trees and unused targets at admission')
    verified = records['verification']
    require(verified['plan'] == receipt['plan'] and result['verification'] == receipt['verification'],
            'Exact complete pre-retirement verification')
    validate_verified(plan,verified)
    validate_actions(plan,receipt['verification'],records['actions'])
    require(result['complete'] is True and result['retired_weight_files'] == 5
            and result['archived_journal_files'] == 8
            and result['weight_bytes_verified'] == sum(row['stat']['st_size'] for row in plan['weights'])
            and result['journal_bytes_retained'] == sum(row['stat']['st_size'] for row in plan['journals'])
            and result['gzip_bytes'] == sum(row['gzip_bytes'] for row in verified['journals'])
            and result['gzip_bytes'] <= storage.BUDGET['gzip_cap'], 'All exact planned bytes accounted')
    for key in ('protected_trees_unchanged','original_payloads_and_runtime_unchanged',
                'canonical_unchanged','original_qwen_bf16_restore_required'):
        require(result[key] is True, 'Original model/evidence retained: '+key)
    for key in ('actual_self_exit_claimed','runtime_admitted','full_model_qualified','new_speedup_claim'):
        require(result[key] is False, 'No storage-outcome promotion: '+key)
    require(result['before_space'] == ad['before_space']
            and result['before_space']['root'] >= storage.BUDGET['root_floor']
                +storage.BUDGET['gzip_cap']+storage.BUDGET['metadata_cap']
            and result['before_space']['nvme'] >= storage.BUDGET['nvme_floor']
            and result['after_space']['root'] >= storage.BUDGET['root_floor']
            and result['after_space']['nvme'] >= storage.BUDGET['nvme_floor'],
            'Unreduced maintenance bounds; next model capacity still requires fresh checks')
    gone(owner)
    layout(plan,verified)
    history = copy.deepcopy(prior)
    residual.append_owner(history,owner)
    require(len(history['processes']) == 350, 'Exactly one actual completed maintenance owner appended')
    history.update(completed_storage_maintenance=copy.deepcopy(receipt_ref),current_physical_release_verified=False)
    pins = {}
    residual.merge(pins,done['source_sha256'])
    residual.merge(pins,plan['source_sha256'])
    return history,pins


def completed(actual_exit_ref):
    reader = Reader()
    receipt = reader(actual_exit_ref)
    residual.external(receipt,'storage-maintenance')
    require(receipt['plan']['path'] == str(PLAN), 'Exact fixed maintenance plan')
    plan = reader(receipt['plan'])
    storage.validation(plan)
    prep = reader(PREPARATION)
    require(prep['source_only'] is True and prep['preparation_review_passed'] is True
            and prep['runtime_admitted'] is False and prep['tests_passed'] == 30,
            'Source preparation remains distinct from actual runtime completion')
    require(all(plan['source_sha256'].get(k) == v for k,v in prep['source_sha256'].items()),
            'Exact reviewed storage implementation')
    prep_v3 = reader(PREPARATION_V3)
    require(prep_v3['plan'] == receipt['plan'] and prep_v3['tests_passed'] == 17
            and prep_v3['tests_actual_exit']['exit_code'] == prep_v3['builder_actual_exit']['exit_code'] == 0
            and prep_v3['worker_and_privileged_all_function_AST_identical'] is True
            and prep_v3['runtime_admitted'] is prep_v3['cleanup_executed'] is False,
            'Exact actual v3 preparation and unchanged retirement implementation')
    require(all(plan['source_sha256'].get(k) == v for k,v in prep_v3['source_sha256'].items()),
            'Exact final v3 implementation pins')
    done = residual.completed(plan['completed_residual_review'])
    parent,_ = storage.combined_sources(plan,done)
    h,pins = extend(done,actual_exit_ref,receipt,plan,reader)
    protected = reader(receipt['protected_before'])
    require(protected['payloads'] == {path:row['stat'] for path,row in parent['payloads'].items()}
            and protected['binaries'] == parent['runtime_binary_stat'], 'All original Nemotron payload/runtime identities')
    residual.merge(pins,reader.pins)
    return dict(history=h,source_sha256=pins,
        historical_binary_sha256=copy.deepcopy(done['historical_binary_sha256']),
        metadata_bytes_read=reader.bytes,archived_journal_payload_bytes_read=0,
        original_weight_payload_bytes_read=0,fresh_physical_drain_verified=False,
        runtime_admitted=False,full_model_qualified=False,new_speedup_claim=False)


if __name__ == '__main__':
    raise RuntimeError('No default exit or automatic launch; requires actual future maintenance success')
