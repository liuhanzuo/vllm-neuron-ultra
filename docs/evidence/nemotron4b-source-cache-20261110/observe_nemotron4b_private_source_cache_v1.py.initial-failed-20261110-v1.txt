"""Observe prepared Nano4B source imports under a private empty cache prefix.

Only stdlib and already reviewed preparation modules are imported. Original
SDK/scientific functions, site hooks and model/native payload reads do not run.
This is an observer, not a worker admission or a complete source-code proof.
"""
from pathlib import Path
import datetime
import hashlib
import json
import sys

from nemotron4b_empty_bytecode_cache_v1 import EmptyBytecodeCache
from nemotron4b_process_handshake_v1 import require

ROOT=Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
Q=ROOT/'research'
REVIEW=Q/'nemotron4b-native-worker-source-preparation-review-20261110-v1.json'
REVIEW_SHA='9b105858b2725b5a3d7562d0914a3236771e13b56990c7788ab017b066ab30c0'
PLAN=Q/'nemotron4b-native-prerequisite-plan-20261106-v2.json'
PLAN_SHA='7e5f16c07ad07c56b5bd3e6f5122ac6dd13b5e09d387da18bf116e035d81a2fd'
FORBIDDEN={'torch','transformers','vllm','neuronx_distributed','libneuronxla'}


def source_only_guard():
    require(not FORBIDDEN&set(sys.modules)and 'site'not in sys.modules,
            'No original SDK/model or site execution in observer')


def observe(phase,cache_manifest):
    source_only_guard()
    raw=REVIEW.read_bytes();require(hashlib.sha256(raw).hexdigest()==REVIEW_SHA,'Prior worker source review')
    reviewed=json.loads(raw)
    raw=PLAN.read_bytes();require(hashlib.sha256(raw).hexdigest()==PLAN_SHA,'Original phase plan')
    plan=json.loads(raw);expected=plan['environments'][phase]
    require(sys.executable==expected['python']and list(sys.version_info[:2])==expected['python_version'],
            'Original actual phase-specific interpreter')
    for path,digest in reviewed['source_sha256'].items():
        require(hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,'Prior source unchanged')
    cache=EmptyBytecodeCache(cache_manifest,guard=source_only_guard)
    cache.check_runtime()
    import nemotron4b_native_worker_execution_v3
    source_only_guard()
    rows=cache.audit_preloaded_source_caches()
    return dict(schema='nemotron4b_private_source_cache_observation_v1',
        observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase=phase,
        python=sys.executable,version=list(sys.version_info[:3]),
        prior_source_review=dict(path=str(REVIEW),sha256=REVIEW_SHA),
        original_plan=dict(path=str(PLAN),sha256=PLAN_SHA),
        command_line_cache_prefix=sys._xoptions['pycache_prefix'],
        preloaded_source_modules=rows,source_module_count=len(rows),cache_policy=cache.result(),
        source_worker_imported_without_executing_science=True,original_site_hooks_executed=False,
        actual_SDK_or_model_executed=False,explicit_native_or_original_bytecode_payload_reads=0,
        complete_source_or_runtime_provenance=False,runtime_admitted=False,new_speedup_claim=False)


if __name__=='__main__':
    require(len(sys.argv)==3 and sys.argv[1]in('native-meta','nxd-meta'),'Phase and cache manifest')
    manifest=json.loads(Path(sys.argv[2]).read_text())
    print(json.dumps(observe(sys.argv[1],manifest),sort_keys=True))
