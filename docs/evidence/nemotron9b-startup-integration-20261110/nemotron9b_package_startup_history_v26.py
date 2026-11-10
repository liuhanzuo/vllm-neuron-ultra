"""Exact completed v24, Mini saved-data r2 and startup timing history.

This is metadata authentication, not physical release. Every original service,
owner and failed outcome survives in a snapshot already authenticated by Mini.
"""
import copy
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parent
Q=ROOT/'research'
def pin(name,digest):return dict(path=str(Q/name),sha256=digest)
PARENT=pin('nemotron9b-package-runtime-v24-r10-20261110-spec.json','dfb80b098d9f25d8925b1855701b5824e7a8466e75f02b792cb0d7535d2a6a74')
PARENT_REVIEW=pin('nemotron9b-package-runtime-v24-r10-20261110-source-review.json','ce51d85c45bc39917826ee4e628d4bdec076293e71f0137aec4fdc23342575ac')
MINI_EXIT=pin('minicpm5-saved-tap-r2-actual-exit-20261110-v1.json','9c11fd1c6bd4dc7a90cb85809fe7bfd9fb130359a7b3fdd7d88569e0c6ec8abf')
TIMING_EXIT=pin('nemotron9b-preexec-timing-actual-exit-20261110-v1.json','205ebda214d116e17a2007db263134d45f98be09c892d6d10b41f9eca6c47793')
MINI_SPEC=pin('minicpm5-saved-tap-runtime-v4-20261110-r2-spec.json','c8bd856e879f618349671e270c97011ea3e3f22639404f064c8ac99b4f0cb462')
MINI_REVIEW=pin('minicpm5-saved-tap-runtime-v4-20261110-r2-source-review.json','3f205e4ef444e6390cb0319a67f254be56796e40e03d51e93e38b3d817884d70')
SNAPSHOT=dict(path=str(ROOT/'minicpm5-saved-tap-analysis-20261110-r2/predecessor-history.json'),sha256='a79ac8bbbe6798c2377a46ecd2c9133374e4d51646e662b1feb99be2ea87e0d6')
STARTUP_REVIEW=pin('nemotron9b-startup-ready-v2-source-review-20261110-v1.json','5da3de73abfff0e6140e9cc5e1b95ff7d89fca85654d5177f4a65b23e251259c')
REFERENCES=dict(parent=PARENT,parent_review=PARENT_REVIEW,mini_exit=MINI_EXIT,timing_exit=TIMING_EXIT,
                mini_spec=MINI_SPEC,mini_review=MINI_REVIEW,snapshot=SNAPSHOT,startup_review=STARTUP_REVIEW)
POLICY=dict(preparation_seconds=300.,exec_seconds=60.,independent_post_exec_ack_seconds=60.,
            empty_argv_grace_seconds=.25,completed_intent_marker='owned-0700-directory')
LAYOUT=dict(control_and_evidence_parent='/mnt/nvme',work_parent='/mnt/nvme',
            root_floor=28*2**30,nvme_floor=30*2**30,metadata_cap=2**30,work_cap=18*2**30,
            original_evidence_paths_preserved=True,retirement='none')
CHANGED=frozenset(('schema','run_id','controller','worker','units','control_root','evidence_root','work_root',
                   'source_sha256','external_observer_contract','runtime_readiness'))
ADDED=frozenset(('startup_history','startup_policy','output_layout'))
LARGE={ref['path']:ref['sha256']for ref in (MINI_SPEC,MINI_REVIEW,SNAPSHOT)}

def require(ok,message):
    if not ok:raise RuntimeError(message)

def merge(pins,additions):
    for path,digest in additions.items():
        require(Path(path).is_absolute() and re.fullmatch('[0-9a-f]{64}',digest) is not None,'Literal source pin')
        require(path not in pins or pins[path]==digest,'Historical source cannot be rebound: '+path)
        pins[path]=digest

def accepts(path):return str(path) in LARGE

def text(path,expected,stable_bytes):
    require(accepts(path) and expected==LARGE[str(path)],'Only exact completed Mini envelopes')
    raw=stable_bytes(path,32*2**20)
    require(hashlib.sha256(raw).hexdigest()==expected,'Original complete Mini envelope bytes')
    value=json.loads(raw.decode('utf8'))
    require(type(value)is dict,'Complete JSON object')
    if str(path)==SNAPSHOT['path']:
        require(len(value['rows'])==1202 and len(value['processes'])==397,'Exact saved1202/397 history')
    else:
        require(type(value['source_sha256'])is dict and 0<len(value['source_sha256'])<=100000,'Original bounded source map')
    return raw

def completed(read):
    pins={}
    def load(ref):
        # Normalize optional byte length but preserve its verification by reader.
        value=read(ref);merge(pins,{ref['path']:ref['sha256']});return value
    parent,review=load(PARENT),load(PARENT_REVIEW)
    require(parent['schema']=='nemotron9b_package_runtime_v24' and review['execution_spec_sha256']==PARENT['sha256']
            and review['passed'] is True and review['source_sha256']==parent['source_sha256'],'Original v24 spec and review')
    ended=load(MINI_EXIT);result=load(ended['result']);ad=load(result['admission'])
    mini,mreview=load(MINI_SPEC),load(MINI_REVIEW);history=load(SNAPSHOT)
    require(ended['actual_external_exit_observed'] is True and ended['exact_owner_gone'] is True
            and (ended['session_id'],ended['exit_code'],ended['tool_chunk'])==(37843,0,'1a4dcd'),
            'Actual saved-data r2 exit')
    require(ended['owner']==result['owner']==ad['owner']
            and result['complete'] is result['canonical_unchanged'] is True
            and ended['execution_spec']==result['execution_spec']==ad['execution_spec']==MINI_SPEC
            and ended['source_review']==result['source_review']==ad['source_review']==MINI_REVIEW
            and ad['predecessor_history']==SNAPSHOT,'Exact r2 admitted owner, input and history')
    require(mini['schema']=='minicpm5_saved_tap_runtime_v4' and mini['profile']=='retry-v24'
            and mini['output']==str(ROOT/'minicpm5-saved-tap-analysis-20261110-r2')
            and mini['diagnostic_only'] is True and mini['full_model_qualified'] is False
            and mreview['passed'] is True and mreview['reviewer']=='/root' and mreview['blockers']==[]
            and mreview['spec_sha256']==MINI_SPEC['sha256']
            and mreview['source_sha256']==mini['source_sha256'],'Exact completed diagnostic scope and source review')
    require(len(history['rows'])==1202 and len(history['processes'])==397
            and history['canonical_sha256']==ad['canonical_sha256']
            and history['completed_nemotron_package']['execution_spec']==PARENT
            and history['completed_nemotron_package']['source_review']==PARENT_REVIEW
            and history['completed_nemotron_package']['actual_exit']==mini['nemotron_actual_exit']
            and history['completed_nemotron_package']['failed_release_exit']==mini['nemotron_failed_release_exit']
            and history['nemotron_outcome']['producer_qualified'] is False
            and history['nemotron_outcome']['failed_release'] is True
            and history['previous_mini_failure_promoted'] is False,
            'Complete saved history retains v24 and failed Mini r1 outcomes')
    # Authenticate both terminal receipts and the released53-group failure.
    native_exit=load(mini['nemotron_actual_exit']);release_exit=load(mini['nemotron_failed_release_exit'])
    release=load(history['completed_nemotron_package']['release'])
    require(native_exit['actual_exit']['exit_code']==1
            and release_exit['actual_exit']['exit_code']==0
            and release['released'] is release['all_current_groups_recursively_drained'] is True
            and release['successful'] is False,'Actual v24 failure and separate complete release')
    timing_exit=load(TIMING_EXIT);timing=load(timing_exit['result']);tad=load(timing['admission'])
    tspec,treview=load(tad['execution_spec']),load(tad['source_review'])
    require(timing_exit['actual_external_exit_observed'] is timing_exit['exact_owner_gone'] is True
            and (timing_exit['session_id'],timing_exit['exit_code'],timing_exit['tool_chunk'])==(39153,0,'97d43f')
            and timing['complete'] is timing['canonical_unchanged'] is True
            and timing_exit['owner']==timing['owner']==tad['owner']
            and tad['canonical_sha256']==history['canonical_sha256'],'Exact completed timing and owner')
    require(tspec['target_spec']==PARENT and tspec['target_review']==PARENT_REVIEW
            and tspec['mini_actual_exit']==MINI_EXIT
            and tspec['mini_result']['path']==ended['result']['path']
            and tspec['mini_result']['sha256']==ended['result']['sha256']
            and treview['spec_sha256']==tad['execution_spec']['sha256']
            and treview['passed'] is True and treview['blockers']==[]
            and treview['bootstrap_source_sha256']==tspec['bootstrap_source_sha256'],
            'Timing binds original bundle, actual Mini completion and reviewed bootstrap')
    require(timing['model_executed'] is timing['compiler_executed'] is False,'Timing is not model qualification')
    for ref in timing['timings'].values():
        row=load(ref);require(row['original_checks_passed'] is True,'Completed original bundle checks')
    startup=load(STARTUP_REVIEW)
    require(startup['source_preparation_passed'] is True and startup['total_applicable_tests']==45
            and startup['runtime_admitted'] is False,'Reviewed startup source candidate is not runtime')
    for additions in (parent['source_sha256'],mini['source_sha256'],history['predecessor_source_sha256'],
                      tspec['bootstrap_source_sha256'],startup['source_sha256']):
        merge(pins,additions)
    extended=copy.deepcopy(history)
    for owner,role in ((ended['owner'],'minicpm5_saved_tap_r2_completed_owner'),(timing_exit['owner'],'nemotron9b_preexec_timing_completed_owner')):
        require(not any((x['pid'],x['start_ticks'])==(owner['pid'],owner['start_ticks'])for x in extended['processes']),
                'Completed owner appended exactly once')
        extended['processes'].append(dict(owner,role=role))
    require(len(extended['processes'])==399,'All399 historical owners')
    extended.update(completed_mini_saved_tap_r2=MINI_EXIT,completed_nemotron_preexec_timing=TIMING_EXIT)
    return dict(history=extended,source_sha256=pins,parent=parent)

def validate_spec(spec,base,read):
    require(spec['startup_history']==REFERENCES and spec['startup_policy']==POLICY
            and spec['output_layout']==LAYOUT,'Exact startup, completed history and storage policy')
    done=completed(read);history=done['history']
    parent=done['parent']
    require(set(spec)==set(parent)|ADDED,'Only startup and storage-location fields added')
    require(all(spec[key]==value for key,value in parent.items()if key not in CHANGED),
            'Every original model, scientific, input and resource field unchanged')
    require(len(base['rows'])==1149 and len(base['processes'])==350
            and history['rows'][:1149]==base['rows'] and history['processes'][:350]==base['processes'],
            'All prior services and owners retained in exact order')
    require(history['canonical_sha256']==base['canonical_sha256'],'Original canonical history retained')
    require(all(spec['source_sha256'].get(path)==digest for path,digest in done['source_sha256'].items()),
            'Every original and completed diagnostic source retained')
    require(not set(spec['units'].values())&{row['unit']for row in history['rows']},'Fresh services cannot reuse historical names')
    return history
