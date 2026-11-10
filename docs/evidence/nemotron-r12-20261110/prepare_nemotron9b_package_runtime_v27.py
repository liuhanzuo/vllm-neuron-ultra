"""Prepare a fresh v27 run from actual v24 failure and completed diagnostics.

No model launch or retirement. All original scientific and resource fields
remain; new control/evidence bytes share NVMe with the same combined quotas.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import sys

R=Path(__file__).resolve().parents[1];Q=R/'research'
sys.path[:0]=[str(R),str(Q)]
import nemotron9b_package_protocol_v27 as p
import nemotron9b_package_startup_history_v27 as history
import nemotron9b_r12_completion_v1 as successor
import nemotron9b_r12_storage_policy_v1 as storage_policy
from prepare_dory_residual_spec_v3 import local_imports
from prepare_nemotron9b_package_runtime_v24 import observer_policy

def merge(*parts):
    out={}
    for part in parts:history.merge(out,part)
    return out

def compose(done,run_id,pins,observer_ref):
    parent=done['parent']
    p.require(re.fullmatch('[a-z0-9-]{1,40}',run_id) is not None and run_id!=parent['run_id'],'Fresh explicit run ID')
    s=copy.deepcopy(parent)
    s.update(schema='nemotron9b_package_runtime_v27',run_id=run_id,
        controller=str(R/'nemotron9b_package_controller_v27.py'),worker=str(R/'nemotron9b_package_worker_v27.py'),
        units={stage:'trn1-latest-verify-nemotron9b-package-'+run_id+'-'+stage.replace('_','-')for stage in (*p.STAGES,'controller')},
        control_root='/mnt/nvme/nemotron9b-package-'+run_id+'-control',
        evidence_root='/mnt/nvme/nemotron9b-package-'+run_id+'-evidence',
        work_root='/mnt/nvme/nemotron9b-package-'+run_id,
        source_sha256=merge(done['source_sha256'],pins,{observer_ref['path']:observer_ref['sha256']}),
        external_observer_contract=copy.deepcopy(observer_ref),
        startup_history=copy.deepcopy(history.REFERENCES),startup_policy=copy.deepcopy(history.POLICY),
        output_layout=copy.deepcopy(history.LAYOUT),
        post_r11_completion=copy.deepcopy(successor.BINDING),root_headroom_policy=copy.deepcopy(storage_policy.POLICY),
        runtime_readiness=dict(ready=False,source_compatible=True,blockers=[
          'Final root source review of exact new execution specification',
          'Fresh complete1255-service/806-owner drain plus completed Mini and storage readers; unchanged capacity/source/payload checks',
          'All52 original stages remain unexecuted in this new run']))
    p.require(set(s)==set(parent)|history.ADDED,'Only exact successor fields added')
    p.require(all(s[k]==v for k,v in parent.items()if k not in history.CHANGED),
              'Original model/science/input/resources unchanged')
    p.require(set(p.required_sources())<=set(s['source_sha256']),'Complete executable and lifecycle closure')
    p.require(not set(s['units'].values())&{x['unit']for x in done['history']['rows']},'No historical service name reused')
    return s

def assemble(run_id):
    successor.latest_terminal()
    done=history.completed(p.metadata.load)
    latest=successor.metadata(p.metadata.load)
    done['history']=latest['previous']['history']
    r11_parent=p.metadata.load(successor.r11.SPEC)
    done['source_sha256']=merge(done['source_sha256'],r11_parent['source_sha256'],latest['source_sha256'],
        {ref['path']:ref['sha256']for ref in successor.BINDING.values()})
    entries=tuple(Path(x)for x in p.required_sources())+(Path(__file__).resolve(),
        Q/'check_nemotron9b_package_exec_v27.py',Q/'check_prepare_nemotron9b_package_runtime_v27.py',
        Q/'check_nemotron9b_completed_mini_envelopes_v27.py')
    python_entries=tuple(x for x in entries if x.is_relative_to(R) and x.suffix=='.py')
    graph=local_imports(entries=python_entries)
    pins=merge(graph['source_sha256'],{str(x):p.metadata.sha(x)for x in entries})
    parent=done['parent']
    observer=observer_policy(parent,p.metadata.load(parent['external_observer_contract']),
                             pins[str(R/'nemotron9b_package_window_v27.py')])
    prefix=Q/('nemotron9b-package-runtime-v27-'+run_id)
    opath=Path(str(prefix)+'-observer-contract.json')
    oref=dict(path=str(opath),sha256=hashlib.sha256(p.metadata.encode(opath,observer)).hexdigest())
    spec=compose(done,run_id,pins,oref)
    for key in ('control_root','evidence_root','work_root'):
        p.require(not Path(spec[key]).exists(),'Never reuse an existing output namespace')
    return spec,observer,graph

def source_checks(spec,pending):
    catalog=p.metadata.load(spec['final_predecessor_catalog'])
    parts=p.metadata.classify_sources(spec,catalog)
    for path,digest in parts['text_sha256'].items():
        if path in pending:
            raw=p.metadata.encode(path,pending[path]);actual=hashlib.sha256(raw).hexdigest()
        else:
            actual=(p.alias_metadata.source_sha(path,spec['metadata_aliases'],p.metadata.text)
                    if path in spec['metadata_aliases'] else p.metadata.sha(path,digest))
        p.require(actual==digest,'Source changed: '+path)
    for path,digest in parts['python_bytes_sha256'].items():p.metadata.python_source_bytes(path,digest)
    native=p.metadata.query_sources.authenticate_native_sources(parts['query_native_binary_sha256'],
        p.metadata.load(p.metadata.query_sources.OBSERVATION),p.quiet)
    for path,row in spec['payloads'].items():p.require(p.file_stat(path)==row['stat'],'Original payload stat unchanged')
    for path,identity in spec['runtime_binary_stat'].items():p.require(p.file_stat(path)==identity,'Runtime binary identity unchanged')
    base=p.attention_completion.validate_spec(spec,catalog)
    history.validate_spec(spec,base,p.metadata.load)
    successor.validate_spec(spec,p.metadata.load)
    from nemotron9b_package_evidence_v27 import validate_payload_closure
    validate_payload_closure(spec)
    return dict(text_files=len(parts['text_sha256']),non_utf8_files=len(parts['python_bytes_sha256']),
                query_native=native,historical_payload_pins=len(parts['historical_payload_sha256']),
                original_payload_stats=len(spec['payloads']),runtime_binary_stats=len(spec['runtime_binary_stat']))

def build(run_id):
    spec,observer,graph=assemble(run_id)
    prefix=Q/('nemotron9b-package-runtime-v27-'+run_id)
    paths={name:Path(str(prefix)+'-'+name+'.json')for name in ('spec','observer-contract','preparation-checks')}
    p.require(not any(path.exists()for path in paths.values()),'Fresh immutable preparation records')
    space=p.storage(spec,reserve=True)
    checks=source_checks(spec,{str(paths['observer-contract']):observer})
    p.once(paths['observer-contract'],observer)
    p.once(paths['spec'],spec)
    result=dict(schema='nemotron9b_v27_spec_preparation_checks_v1',reviewer='/root',
        specification=p.ref(paths['spec']),observer_contract=p.ref(paths['observer-contract']),
        parent=history.PARENT,source_checks=checks,storage=space,
        import_graph_files=len(graph['source_sha256']),original_stage_count=52,
        predecessor_services=1255,predecessor_owners=806,original_science_and_quotas_retained=True,
        completed_mini_and_maintenance_checks_required_at_launch=True,root_start_headroom_bytes=storage_policy.ROOT_START_MARGIN,
        new_metadata_outputs_on_nvme=True,retirement_performed=False,
        fresh_physical_drain_proven=False,runtime_admitted=False,native_started=False,new_speedup_claim=False)
    p.once(paths['preparation-checks'],result)
    return dict(specification=p.ref(paths['spec']),preparation_checks=p.ref(paths['preparation-checks']),runtime_admitted=False)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--run-id',required=True);args=a.parse_args()
    p.require(sys.dont_write_bytecode and __debug__,'Original assertions and no bytecode writes')
    print(json.dumps(build(args.run_id)))
