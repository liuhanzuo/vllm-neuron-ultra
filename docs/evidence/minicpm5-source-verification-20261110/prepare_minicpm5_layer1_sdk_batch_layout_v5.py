"""Prepare complete successor metadata with both actual-exit recorder closures.

No model/native hash or launch. The builder owns the original serial lock,
retains all historical inputs/sources, creates a concrete plan/spec, and leaves
final independent runtime review to a separate root review. Pending completion
rejects before lock or any output. Never use this while r12 is still live.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1];Q=ROOT/'research';sys.path[:0]=[str(ROOT),str(Q)]
from research import minicpm5_layer1_sdk_batch_layout_completion_v1 as latest
from research import minicpm5_layer1_sdk_batch_layout_sources_v2 as sources
from minicpm5_layer0_runtime_auth_v1 import identity
require=latest.require


class Reader:
    def __init__(self,guard):self.guard=guard;self.inputs={}
    def __call__(self,ref):
        self.guard();sources.reference(ref);before=identity(ref['path']);value=sources.load(ref)
        require(identity(ref['path'])==before,'Complete preparation metadata remains stable')
        row=dict(sha256=ref['sha256'],identity=before)
        require(ref['path']not in self.inputs or self.inputs[ref['path']]==row,'No input identity rebound')
        self.inputs[ref['path']]=row;self.guard();return value


def merge(target,addition,what):
    for name,value in addition.items():
        require(name not in target or target[name]==value,'Retain every original '+what+': '+name)
        target[name]=copy.deepcopy(value)


def paths(run_id):
    require(type(run_id)is str and re.fullmatch('[a-z0-9][a-z0-9-]{0,39}',run_id),'Bounded fresh run name')
    base='minicpm5-layer1-sdk-batch-layout-'+run_id
    return dict(plan=Q/(base+'-plan.json'),spec=Q/(base+'-spec.json'),review=Q/(base+'-source-review.json'),
        contract=Q/('minicpm5-layer1-sdk-batch-layout-source-contract-'+run_id+'.json'),
        output=Path('/mnt/nvme')/base,unit='trn1-mini-layer1-sdk-batch-layout-'+run_id)


def assemble(done,observed_inputs,new_sources,contract_ref,run_id,master_port,stat=identity):
    """Pure assembly except explicit descriptor stat; no source/payload opening."""
    latest.require_complete_binding(done['binding'])
    require(done['metadata_complete']is True and done['physical_checks_performed']is False,
        'Authenticated predecessor metadata, never invented physical admission')
    require(type(master_port)is int and 29000<master_port<30000,'Original finite rendezvous port')
    locations=paths(run_id);current=done['current_history'];plan=copy.deepcopy(current['plan']);spec=copy.deepcopy(current['spec'])
    for rank in (0,1):
        rows=current['records']['rank%d_result'%rank]['records'];require(len(rows)==90,'All90 original outputs per rank')
        for row in rows:
            desc=row['output'];wanted=dict(sha256=desc['sha256'],identity=desc['identity'])
            require(desc['raw_bytes']==16384 and stat(desc['path'])==desc['identity'],'Actual unchanged full previous output descriptor')
            merge(plan['inputs'],{desc['path']:wanted},'full prior output identity')
    merge(plan['inputs'],observed_inputs,'completed metadata input')
    merge(plan['source_sha256'],done['source_sha256'],'completed source pin')
    merge(plan['source_sha256'],new_sources,'new local source pin')
    merge(plan['source_sha256'],{contract_ref['path']:contract_ref['sha256']},'source classification contract')
    plan.update(output_root=str(locations['output']),latest_completion=copy.deepcopy(done['binding']),
        layer1_sdk_batch_layout=copy.deepcopy(current['binding']),source_contract=copy.deepcopy(contract_ref))
    spec.update(schema='minicpm5_layer1_sdk_batch_layout_cpu_execution_spec_v2',
        runner=str(ROOT/'minicpm5_layer1_sdk_batch_layout_cpu_runner_v2.py'),
        worker=str(ROOT/'minicpm5_layer1_sdk_batch_layout_cpu_worker_v2.py'),
        output_root=plan['output_root'],source_sha256=copy.deepcopy(plan['source_sha256']),source_contract=copy.deepcopy(contract_ref),
        unit=locations['unit'],master_port=master_port,latest_completion=copy.deepcopy(done['binding']),
        completed_current_history=copy.deepcopy(current['binding']))
    allowed={'source_sha256','inputs','output_root','latest_completion','layer1_sdk_batch_layout','source_contract'}
    require({k:v for k,v in plan.items()if k not in allowed}=={k:v for k,v in current['plan'].items()if k not in allowed},
        'Entire original scientific plan retained unchanged')
    for key in ('inputs','source_sha256'):
        require(all(plan[key].get(k)==v for k,v in current['plan'][key].items()),'All original '+key+' retained')
    return spec,plan,locations


def write_named(path,value):
    """Exact new plan/spec envelopes only; exclusive durable writes."""
    sources.envelope_shape(path,value)
    raw=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
    require(len(raw)<=sources.metadata.ENVELOPE_LIMIT,'Original32MiB named envelope cap')
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb')as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    fd=os.open(Path(path).parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:os.fsync(fd)
    finally:os.close(fd)
    ref=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest())
    require(sources.load(ref)==value,'Complete independently read-back named metadata')
    return ref


def entrypoints():
    return [ROOT/'minicpm5_layer1_sdk_batch_layout_cpu_runner_v2.py',ROOT/'minicpm5_layer1_sdk_batch_layout_cpu_worker_v2.py',
        ROOT/'minicpm5_layer1_sdk_batch_layout_runtime_context_v2.py',Path(__file__).resolve(),
        Q/'audit_minicpm5_layer1_sdk_batch_layout_result_v1.py',Q/'record_minicpm5_layer1_sdk_batch_layout_exit_v1.py',
        Q/'check_minicpm5_layer1_sdk_batch_layout_runner_v2.py',Q/'check_minicpm5_layer1_sdk_batch_layout_sources_v2.py',
        Q/'check_minicpm5_layer1_sdk_batch_layout_completion_v1.py',Q/'check_minicpm5_after_nemotron_r12_v1.py',
        Q/'check_minicpm5_layer1_sdk_batch_layout_post_exit_core_v1.py',
        Q/'check_audit_minicpm5_layer1_sdk_batch_layout_result_v1.py',
        Q/'check_record_minicpm5_layer1_sdk_batch_layout_exit_v1.py',
        Q/'check_prepare_minicpm5_layer1_sdk_batch_layout_v2.py',
        Q/'record_minicpm5_layer1_sdk_batch_layout_audit_exit_v1.py',
        Q/'check_record_minicpm5_layer1_sdk_batch_layout_audit_exit_v1.py',
        Q/'check_prepare_minicpm5_layer1_sdk_batch_layout_v3.py',
        Q/'audit_minicpm5_layer1_sdk_batch_layout_result_v2.py',
        Q/'check_audit_minicpm5_layer1_sdk_batch_layout_result_v2.py',
        Q/'record_minicpm5_layer1_sdk_batch_layout_audit_exit_v2.py',
        Q/'check_record_minicpm5_layer1_sdk_batch_layout_audit_exit_v2.py',
        Q/'check_prepare_minicpm5_layer1_sdk_batch_layout_v4.py',
        Q/'check_minicpm5_sdk_physical_evidence_v2.py',
        Q/'check_prepare_minicpm5_layer1_sdk_batch_layout_v5.py']


def build(binding,run_id,master_port):
    latest.require_complete_binding(binding);latest.require_latest_terminal(binding)
    locations=paths(run_id)
    require(all(not locations[k].exists()for k in ('plan','spec','review','contract','output')),'Every output must be fresh')
    import minicpm5_layer1_sdk_batch_layout_cpu_runner_v2 as runner
    from research.prepare_dory_residual_spec_v3 import local_imports
    require(sys.dont_write_bytecode and __debug__,'Original python -B')
    runner.p.user_bus();fd,owner,lock=runner.post.external.acquire_lock()
    def guard():runner.p.lock_proof(owner,lock);runner.p.clear_window()
    try:
        guard();runner.p.active_heavy();latest.require_latest_terminal(binding)
        reader=Reader(guard);done=latest.completed(binding,reader,guard)
        require(done['lock_identity']==lock,'Original serial lock throughout preparation')
        for old in done['owners']:guard();runner.p.gone(old)
        graph=local_imports(entries=entrypoints());new_sources=graph['source_sha256']
        for module in tuple(sys.modules.values()):
            path=getattr(module,'__file__',None)
            if path and str(path).startswith(str(ROOT)+'/'):
                guard();path=str(Path(path).resolve());raw=sources.text(path)
                merge(new_sources,{path:hashlib.sha256(raw).hexdigest()},'loaded local source')
        contract=dict(schema=sources.CONTRACT_SCHEMA,r12_spec=sources.R12_SPEC,metadata_limits=sources.metadata.LIMITS)
        runner.p.once(locations['contract'],contract);contract_ref=runner.reference(locations['contract']);reader(contract_ref)
        spec,plan,locations=assemble(done,reader.inputs,new_sources,contract_ref,run_id,master_port)
        runner.validate_current_history_extension(plan,done)
        partition=sources.classify(plan['source_sha256'],contract_ref)
        sources.verify_text_sources(partition,guard);runner.verify_loaded_sources(plan,guard)
        require({n:runner.p.sha(ROOT/n)for n in spec['canonical_sha256']}==spec['canonical_sha256'],
            'Original scientific canonical state unchanged')
        spec['plan']=write_named(locations['plan'],plan);spec_ref=write_named(locations['spec'],spec)
        guard();runner.p.active_heavy()
        return dict(spec=spec_ref,plan=spec['plan'],source_contract=contract_ref,
            source_files=len(plan['source_sha256']),inputs=len(plan['inputs']),source_closure=graph,
            final_independent_source_review_required=True,review_target=str(locations['review']),
            runtime_admitted=False,native_bytes_hashed=0,model_or_tensor_bytes_read=0,full_model_qualified=False,new_speedup_claim=False)
    finally:runner.post.external.release_lock(fd)


if __name__=='__main__':
    cli=argparse.ArgumentParser();cli.add_argument('--binding',required=True);cli.add_argument('--binding-sha256',required=True)
    cli.add_argument('--run-id',required=True);cli.add_argument('--master-port',type=int,required=True);args=cli.parse_args()
    binding=sources.load(dict(path=args.binding,sha256=args.binding_sha256))
    result=build(binding,args.run_id,args.master_port);print(json.dumps(result,indent=2))
