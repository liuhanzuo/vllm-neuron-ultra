"""Retain actual failed Mini saved-tap r1 before the r2 retry. Metadata only."""
import copy
from pathlib import Path
import minicpm5_after_nemotron_package_v2 as prior

Reader, merge, PROFILES, require = prior.Reader, prior.merge, prior.PROFILES, prior.require
Q, ROOT = prior.Q, prior.ROOT
FAILED = {"path":"/home/ubuntu/experiments/trn1-latest-models-20261031/research/minicpm5-saved-tap-r1-failed-actual-exit-20261110-v1.json","sha256":"376e5ec5febfba598452a81d61ad45215f2c0018809367fdbc96cac29ed038ac"}

def completed(profile, actual_exit_ref, failed_release_exit_ref=None, guard=lambda: None):
    result = prior.completed(profile, actual_exit_ref, failed_release_exit_ref, guard)
    reader = Reader(guard)
    exited = reader(FAILED)
    require(exited['reviewer']=='/root' and exited['actual_external_exit_observed'] is True
            and exited['exact_owner_gone'] is True
            and (exited['session_id'],exited['exit_code'],exited['tool_chunk'])==(22560,1,'ebd522'),
            'Exact actually failed saved-tap attempt')
    out = ROOT/'minicpm5-saved-tap-analysis-20261110-r1'
    require(exited['output']==str(out) and set(exited['files'])=={
        'admission.json','failure.json','predecessor-history.json','predecessor-drain.json','source-files.json'},
        'Every original failed output retained')
    records = {}
    for name, ref in exited['files'].items():
        require(ref['path']==str(out/name),'Original failed output namespace')
        records[name] = reader(ref)
    admission, failure = records['admission.json'], records['failure.json']
    require(admission['owner']==failure['owner']==exited['owner']
            and admission['execution_spec']==exited['execution_spec']
            and admission['source_review']==exited['source_review']
            and failure['complete'] is False
            and failure['error']=='TypeError("read_observation() missing 1 required positional argument: \'sha256\'")',
            'Preserve exact original failed owner and call-signature error')
    old_spec, old_review = reader(exited['execution_spec']), reader(exited['source_review'])
    import minicpm5_saved_tap_runtime_contract_v3 as original
    original.validate_spec(old_spec)
    original.validate_review(old_spec,exited['execution_spec'],old_review)
    require(old_spec['nemotron_actual_exit']==actual_exit_ref
            and old_spec['nemotron_failed_release_exit']==failed_release_exit_ref
            and admission['canonical_sha256']==result['history']['canonical_sha256'],
            'Same original Nemotron and canonical history')
    for name in ('numeric-report.json','result.json','query-cleanup-failure.json'):
        require(not (out/name).exists(),'No successful or unknown failure output substituted')
    from nemotron9b_package_protocol_v24 import low
    low.gone(exited['owner'])
    history=copy.deepcopy(result['history'])
    require(not any((x['pid'],x['start_ticks'])==(exited['owner']['pid'],exited['owner']['start_ticks'])
                    for x in history['processes']),'Failed attempt appended once')
    history['processes'].append(dict(exited['owner'],role='minicpm5_saved_tap_r1_failed_owner'))
    history['completed_minicpm5_saved_tap_failed_attempt']=FAILED
    history['previous_mini_failure_promoted']=False
    merge(result['source_sha256'],reader.pins)
    history['predecessor_source_sha256']=copy.deepcopy(result['source_sha256'])
    result.update(history=history,metadata_files=result['metadata_files']+len(reader.sizes),
                  metadata_bytes=result['metadata_bytes']+sum(reader.sizes.values()))
    guard()
    return result
