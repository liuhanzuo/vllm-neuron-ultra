"""Future completed Dory residual-review history; metadata only, no launch.

The native r2 anchor is already admitted, not already completed. A successor
requires separate actual successful producer and saved-review exits. This
module must never write those receipts itself or classify the seven failures
as model success. It does not authenticate tensor payloads or freshly drain
all historical service groups; the future serial launcher must do that.
"""
import copy
import hashlib
import json
from pathlib import Path
import re

from dory_residual_serial_contract_v4 import validate, load, require, PROPOSAL
from dory_generation_execution_provenance_v1 import inspect_receipts, current_process, owner_key
from nemotron9b_package_metadata_v18 import stable_bytes
import minicpm5_layer0_execution_protocol_v4 as low

Q = Path(__file__).resolve().parent
BASE = Path('/mnt/nvme/dory-official-cpu-20261108-v1')
PRODUCER = dict(path=str(Q/'dory-residual-native-r2-runtime-spec-20261109-v1.json'),
    sha256='5e65071628109101f252eec373a07e2520e1bd46003989d4e9989716a8ca5c5d')
ADMISSION = dict(path=str(BASE/'residual-replay-r2/admission.json'),
    sha256='2b719248c4ca8f8ee701156ef430674a791d60cea926e99ff6e67726c9e9af88')
HISTORY = dict(path=str(BASE/'residual-replay-r2/predecessor-history.json'),
    sha256='2589a6b511135ef5669a2d0c6dbe3a388af8f03e02f4a0d30ecc9c114f010599')
CHILD = dict(path=str(BASE/'residual-replay-r2/child-owner.json'),
    sha256='44963fb76b753e23203975ab512e2f8ced893297d4fe9c8a225c2361d1c1be89')
SOURCE_FILES = dict(path=str(BASE/'residual-replay-r2/source-files.json'),
    sha256='a161a6690419d3f3daed3cfa55f3f317e1c1611890d1a64e54b4280d52419cc9')
RAW = 5867691008


def merge(pins, additions):
    for path, digest in additions.items():
        require(Path(path).is_absolute() and isinstance(digest,str)
                and re.fullmatch('[0-9a-f]{64}',digest), 'Absolute exact source pin')
        require(path not in pins or pins[path] == digest, 'Never rebind a historical source pin')
        pins[path] = digest


class Reader:
    """Bounded JSON authentication; binary/source files are never opened here."""
    def __init__(self):
        self.pins, self.sizes, self.total = {}, {}, 0

    def __call__(self, ref):
        require(set(ref)=={'path','sha256'} and Path(ref['path']).suffix=='.json',
                'Explicit JSON-only completion reference')
        merge(self.pins,{ref['path']:ref['sha256']})
        raw=stable_bytes(ref['path'],32*2**20)
        require(hashlib.sha256(raw).hexdigest()==ref['sha256'], 'Exact completion metadata bytes')
        if ref['path'] not in self.sizes:
            require(self.total+len(raw)<=128*2**20,'Bounded complete history metadata reads')
            self.total+=len(raw);self.sizes[ref['path']]=len(raw)
        return json.loads(raw)


def anchor(read):
    """Authenticate the real admitted snapshot, without asserting live r2 exit."""
    spec=validate(read(PRODUCER),'native')
    admission,history,child,sources=[read(ref)for ref in (ADMISSION,HISTORY,CHILD,SOURCE_FILES)]
    require(admission['spec_sha256']==PRODUCER['sha256']
            and admission['predecessor_history']==HISTORY
            and admission['mode']=='native' and admission['review'] is False
            and admission['performance_measurement'] is False
            and admission['budget']==spec['budget'], 'Exact actual admitted r2 anchor')
    require(history['schema']=='dory_post_nemotron_r9_history_v1'
            and history['complete_count']==len(history['rows'])==1149
            and len(history['processes'])==345 and len(history['known_admitted_directories'])==35
            and len(history['r9_stages'])==53
            and history['r9_actual_exit']==spec['r9_actual_exit']
            and history['r9_additional_actual_exits']==spec['r9_additional_actual_exits']
            and history['r9_recovery_operations']==spec['r9_recovery_operations']
            and history['residual_metadata_preparation_exit']==spec['metadata_preparation_exit']
            and history['residual_first_native_failure']==spec['failed_native_exit'],
            'All actual r9, metadata-preparation and failed-native history retained')
    require(history['r9_outcome']==dict(original_tool_exit_code=1,
            external_scientific_audit_passed=False,producer_qualified=False,
            failed_external_exit=True,failed_release=True,new_speedup_claim=False),
            'Original r9 failed outcome retained')
    require(admission['canonical_sha256']==history['canonical_sha256']
            and admission['predecessor_drain']['complete'] is True
            and (admission['predecessor_drain']['services'],admission['predecessor_drain']['processes'])==(1149,345),
            'Actual original admission drain, not a fresh successor drain')
    require(child['ppid']==admission['owner']['pid'] and child['argv']==admission['command']
            and owner_key(child)!=owner_key(admission['owner']), 'Actual admitted child identity')
    require(sources['source_and_binary_files_authenticated'] is True
            and {path:row['sha256']for path,row in sources['files'].items()}==spec['source_sha256'],
            'Actual complete producer source authentication ledger')
    require(all(spec['source_sha256'].get(k)==v for k,v in history['predecessor_source_sha256'].items()),
            'Original predecessor pins retained')
    return dict(spec=spec,admission=admission,history=history,child=child,sources=sources)


def external(receipt, mode):
    require(receipt['reviewer']=='/root' and receipt['actual_external_exit_observed'] is True
            and type(receipt['session_id']) is int and receipt['session_id']>0
            and type(receipt['exit_code']) is int and receipt['exit_code']==0
            and isinstance(receipt['tool_chunk'],str)
            and re.fullmatch('[0-9a-f]+',receipt['tool_chunk']), 'Actual successful '+mode+' exit')


def append_owner(history,owner):
    key=owner_key(owner)
    require(all(owner_key(old)!=key for old in history['processes']), 'Each completed owner appended once')
    history['processes'].append(copy.deepcopy(owner))


def saved_scope(result):
    require(result['schema']=='dory_residual_combined_saved_review_v1'
            and result['complete'] is True and result['diagnostic_only'] is True
            and (result['original_events'],result['original_successful_calls'],
                 result['original_pointwise_failures'])==(6200,6185,7)
            and result['both_original_and_new_successful_payloads_verified'] is True
            and result['all_producer_output_names_accounted'] is True
            and result['recorded_source_checkpoint_exit_bound'] is True,
            'Full completed saved review, original seven failures retained')
    for key in ('original_top1_recomputed','original_FP32_metrics_recomputed','production_lifecycle_qualified',
                'root_cause_established','full_model_qualified','new_speedup_claim',
                'actual_loaded_libraries_JIT_verified','complete_history_drain_verified_here'):
        require(result[key] is False,'No saved metadata outcome promotion: '+key)
    for key in ('original_archives','replay_archives'):
        row=result[key]
        require(row['complete'] is True and (row['original_events'],row['original_calls'],
                row['raw_bytes_verified'],row['tensor_labels'],row['cache_histories'])==(6200,6185,RAW,114,56)
                and row['full_cache_hashes_reconstructed'] is True
                and row['hidden_and_logits_bytes_verified'] is True, 'Both complete original/new saved archives')
    row=result['observations']
    require(row['complete'] is True and (row['selected_calls'],row['selected_rows'],
            row['residual_bytes_verified'],row['injected_bytes_verified'],row['original_pointwise_failures'])
            ==(89,153,175472640,8345600,7), 'All selected residual and injected bytes reviewed')
    weights=result['checkpoint_review']
    require(weights['checkpoint_bytes_independently_authenticated'] is True
            and weights['checkpoint_bytes_read']==7950515472 and weights['original_tensors']==507
            and weights['mapped_physical_parameters']==399 and weights['complete_packed_mapping_verified'] is True,
            'Original checkpoint and complete packed mapping reviewed')


def extend(start, review_exit_ref, read, *, lookup=current_process, gone=low.gone, directories=None):
    """Validate a future actual success chain; injectable readers are for tests."""
    producer,ad,prior,child=(start[k]for k in ('spec','admission','history','child'))
    receipt=read(review_exit_ref);external(receipt,'saved-review')
    review_ref=receipt['spec'];review=validate(read(review_ref),'review')
    require(review['producer_spec']==PRODUCER and review['output']==producer['output']
            and review['metadata_preparation_exit']==producer['metadata_preparation_exit']
            and review['failed_native_exit']==producer['failed_native_exit'],
            'Saved review belongs to exact original r2 producer')
    require(all(review['source_sha256'].get(k)==v for k,v in producer['source_sha256'].items()),
            'All original producer pins in review closure')
    require(all(review['source_sha256'].get(ref['path'])==ref['sha256']
                for ref in (ADMISSION,HISTORY,CHILD)), 'Fixed actual admission/history/child byte pins retained')
    native_ref=review['native_exit']
    require(review['source_sha256'].get(native_ref['path'])==native_ref['sha256'],
            'Native actual exit pinned in review sources')
    native=read(native_ref);external(native,'native')
    require(native['spec']==PRODUCER and native['owner']==ad['owner'] and native['child_owner']==child,
            'Native receipt identifies exact actual anchored parent and child')
    def producer_record(name):
        path=str(Path(producer['output'])/name)
        require(path in review['source_sha256'],'Every producer terminal record source-pinned')
        return read(dict(path=path,sha256=review['source_sha256'][path]))
    actual_ad=producer_record('admission.json')
    actual_child=producer_record('child-owner.json')
    require(actual_ad==ad and actual_child==child and producer_record('predecessor-history.json')==prior,
            'Producer terminal records retain fixed actual admission and history')
    records=[ad,producer_record('child.json'),child,producer_record('child-exit.json'),producer_record('result.json')]
    require(records[3]['child_owner']==child, 'Child exit binds the original full child identity')
    failed_paths=[Path(producer['output'])/name for name in ('failure.json','cleanup-pending.json','diagnostic/residual-replay-failure.json')]
    failed_paths += [Path(review['review_output'])/name for name in ('failure.json','cleanup-pending.json')]
    require(not any(path.exists()for path in failed_paths), 'Failed native or reviewer output cannot become completion')
    inspect_receipts(PRODUCER['sha256'],*records,native,process_lookup=lookup)
    require(records[-1]['owner']==ad['owner'] and records[-1]['mode']=='native',
            'Successful terminal result belongs to anchored owner')
    for owner in (ad['owner'],child):gone(owner)
    h=copy.deepcopy(prior)
    for owner in (ad['owner'],child):append_owner(h,owner)
    require(Path(producer['output']).name not in h['known_admitted_directories'],'Fresh native directory extension')
    h['known_admitted_directories'].append(Path(producer['output']).name)
    review_ad=read(receipt['admission']);review_result=read(receipt['result'])
    require(receipt['admission']['path']==str(Path(review['review_output'])/'admission.json')
            and receipt['result']['path']==str(Path(review['review_output'])/'result.json')
            and receipt['predecessor_history']==review_ad['predecessor_history']
            and read(receipt['predecessor_history'])==h, 'Exact saved reviewer post-native history')
    require(review_ad['spec_sha256']==review_ref['sha256'] and review_ad['mode']=='review'
            and review_ad['review'] is True and review_ad['performance_measurement'] is False
            and review_ad['budget']==review['budget']
            and review_ad['canonical_sha256']==prior['canonical_sha256']
            and review_ad['predecessor_drain']['complete'] is True
            and (review_ad['predecessor_drain']['services'],review_ad['predecessor_drain']['processes'])==(1149,347),
            'Actual admitted separate saved reviewer and its complete historical drain')
    review_owner=receipt['owner']
    require(review_ad['owner']==review_result['owner']==review_owner
            and review_result['complete'] is True and review_result['mode']=='review'
            and review_result['review'] is True and review_result['actual_self_exit_claimed'] is False
            and review_result['full_model_qualified'] is review_result['new_speedup_claim'] is False,
            'Actual successful saved-review result and external owner')
    source_review=read(receipt['source_review'])
    require(source_review['reviewer']=='/root' and source_review['passed'] is True
            and source_review['blockers']==[] and source_review['spec_sha256']==review_ref['sha256']
            and source_review['source_sha256']==review['source_sha256'],
            'Separate exact reviewer-source approval')
    expected=[str(Path('/usr/bin/python3')),'-B',review['executor'],'--mode','review','--spec',review_ref['path'],
        '--spec-sha256',review_ref['sha256'],'--review',receipt['source_review']['path'],
        '--review-sha256',receipt['source_review']['sha256']]
    require(review_owner['argv']==expected,'Actual saved-review invocation binds spec and source review')
    require(receipt['saved_review']['path']==str(Path(review['review_output'])/'saved-residual-review.json'),
            'Exact complete saved-review output namespace')
    saved=read(receipt['saved_review']);saved_scope(saved)
    require({path:row['sha256']for path,row in saved['source_file_review']['files'].items()}
            ==producer['source_sha256'], 'Saved reviewer authenticated every original producer source')
    gone(review_owner);append_owner(h,review_owner)
    require(Path(review['review_output']).name not in h['known_admitted_directories'],'Fresh review directory extension')
    h['known_admitted_directories'].append(Path(review['review_output']).name)
    actual=({x.name for x in BASE.iterdir()if x.is_dir()and(x/'admission.json').is_file()}
            if directories is None else set(directories))
    require(actual==set(h['known_admitted_directories']) and len(actual)==37,
            'No unknown intervening or omitted admitted Dory directory')
    h.update(completed_residual_review=copy.deepcopy(review_exit_ref),
             completed_residual_native=copy.deepcopy(native_ref),current_physical_release_verified=False)
    pins={}
    merge(pins,producer['source_sha256']);merge(pins,review['source_sha256'])
    return h,pins


def completed(review_exit_ref):
    reader=Reader()
    start=anchor(reader)
    h,pins=extend(start,review_exit_ref,reader)
    merge(pins,reader.pins)
    binaries={path:digest for path,digest in pins.items()if Path(path).suffix=='.bin'}
    original={path:digest for path,digest in start['spec']['source_sha256'].items()if Path(path).suffix=='.bin'}
    require(binaries==original and len(binaries)==6,'Only exact six original Dory evidence binaries')
    return dict(history=h,source_sha256=pins,historical_binary_sha256=binaries,
                metadata_files_read=len(reader.sizes),metadata_bytes_read=reader.total,
                tensor_payloads_read=False,fresh_physical_drain_verified=False,
                runtime_admitted=False,full_model_qualified=False,new_speedup_claim=False)


if __name__=='__main__':
    raise RuntimeError('No automatic admission or launch; requires actual future saved-review exit reference')
