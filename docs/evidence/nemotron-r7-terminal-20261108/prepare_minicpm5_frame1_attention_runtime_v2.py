"""Build finite frame1 spec from real finalized history; no review or launch.

All future completed-history inputs must exist and authenticate. Source-only
preparation does not prove fresh physical drain, model correctness or speedup.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1];Q=ROOT/'research'
sys.path.insert(0,str(ROOT))
import minicpm5_frame1_attention_runner_v2 as r
r.load_history_helpers()

PREPARATION=dict(path=str(Q/'minicpm5-attention-serial-runner-source-preparation-20261108-v1.json'),
                 sha256='5587954d25fce0189556c54a54fdc8e17d57000f93ac5998fb4a78704f8c77db')
SMOL=Q/'root-post-smol-sio14-complete-predecessors-20261108-v1.json'
SELF=str(Path(__file__).resolve())
TEST=str(Q/'check_minicpm5_frame1_attention_runner_v1.py')


def merge(*groups):
    result={}
    for pins in groups:
        for name,sha in pins.items():
            r.p.require(name not in result or result[name]==sha,'Original source pin cannot change: '+name)
            result[name]=sha
    return result


def select_chain(reference,catalog,load):
    seen=set();rows=[]
    while r.smol_chain.catalog_number(reference['path']) is not None:
        r.p.require(len(rows)<16 and reference['path']not in seen,'Bounded noncyclic completed Smol chain')
        seen.add(reference['path']);audit_ref=catalog['appended_terminal_audit'];audit=load(audit_ref)
        rows.append(dict(catalog=copy.deepcopy(reference),terminal_audit=copy.deepcopy(audit_ref),
                         **{name:copy.deepcopy(audit['references'][name])for name in r.smol_chain.FIELDS}))
        reference=catalog['appended_from'];catalog=load(reference)
    r.p.require(rows and reference==r.HISTORY_ANCHOR,'Actual sio14 extension directly retains original r4 history')
    return rows


def expanded_plan(spec,base):
    plan=copy.deepcopy(base)
    plan['output_root']=spec['output_root']
    plan['source_sha256']=merge(base['source_sha256'],spec['new_sources'])
    plan['inputs']=copy.deepcopy(base['inputs'])
    for name,value in spec['added_json'].items():
        r.p.require(name not in plan['inputs'] or plan['inputs'][name]==value,'Original metadata input cannot change')
        plan['inputs'][name]=copy.deepcopy(value)
    plan['frame1_upstream']=r.worker.upstream.science();plan['frame1_attention']=copy.deepcopy(spec['science'])
    r.p.require(all(plan[key]==value for key,value in base.items()if key not in ('output_root','source_sha256','inputs')),
                'Every original scientific input and remaining gate retained')
    r.worker.validate_science(plan)
    return plan


def compose(base,catalog_ref,catalog,smol_ref,chain,aliases,new_sources,added_json,run_id,port,finalizer_exit_ref):
    """Pure construction; caller authenticates real files and finalization."""
    r.p.require(re.fullmatch('r[0-9]+-20261108-v[0-9]+',run_id),'Fresh explicit namespace')
    r.p.require(type(port)is int and 29000<port<30000,'Finite original TP2 port range')
    r.p.require(smol_ref['path']==str(SMOL)and chain and chain[0]['catalog']==smol_ref,'Exact required completed Smol anchor')
    r.p.require(catalog['reviewer']=='/root' and catalog['root_finalized']is True and catalog['all_planned_launches_accounted_for']is True,
                'Current history must already be finalized')
    r.p.require(catalog_ref['path']==str(r.history_metadata.CATALOG_PATH)and finalizer_exit_ref['path']==str(r.history_metadata.EXIT_PATH)and len(catalog['rows'])==1034,'Actual complete r7 predecessor shape and finalizer receipt required')
    r.p.require(not set(new_sources)&set(base['source_sha256']),'Strictly additive source extension')
    r.p.require(3<=len(new_sources)<=128,'Finite helper closure')
    spec=dict(schema='minicpm5_frame1_attention_spec_v1',base_plan=r.BASE_PLAN,base_spec=r.BASE_SPEC,
        runner=r.SELF,worker=str(ROOT/'minicpm5_frame1_attention_worker_v1.py'),replay=str(Q/'review_minicpm5_frame1_attention_complete_v1.py'),
        output_root=str(ROOT/('minicpm5-frame1-attention-'+run_id)),output_cap_bytes=r.CAP,root_floor_bytes=28*2**30,
        units={phase:'trn1-mini-f1-attention-'+run_id+'-'+phase for phase in r.PHASES},master_port=port,
        resources={phase:r.props(phase)for phase in r.PHASES},canonical_sha256=copy.deepcopy(r.CANONICAL),
        model_qualification_claim=False,automatic_full_gates_launch=False,new_sources=copy.deepcopy(new_sources),
        added_json=copy.deepcopy(added_json),science=r.worker.attention_observation.science(),history_anchor=r.HISTORY_ANCHOR,
        required_sio14_catalog=copy.deepcopy(smol_ref),current_catalog=copy.deepcopy(catalog_ref),
        current_finalizer_exit=copy.deepcopy(finalizer_exit_ref),
        independent_reviewer=str(Q/'review_minicpm5_frame1_attention_complete_v1.py'),
        required_smol_predecessor_chain=copy.deepcopy(chain),history_metadata_aliases=copy.deepcopy(aliases),
        source_preparation=PREPARATION,prepared_by_source=SELF,runtime_admitted=False)
    for name in (spec['runner'],spec['worker'],spec['replay'],spec['independent_reviewer'],*r.worker.ACTIVATIONS.values(),*map(str,r.worker.upstream.NORM_SOURCES.values())):
        r.p.require(name in base['source_sha256'] or name in new_sources,'Every actual entrypoint and activation source pinned')
    r.p.require(not set(spec['units'].values())&{row['unit']for row in catalog['rows']},'Never reuse historical service names')
    r.same(set(added_json),{ref['path']for ref in (*r.worker.upstream.CPU_ANCHORS,*r.worker.upstream.FULL_CPU_ANCHORS,*r.worker.upstream.R3_CAPTURES,*r.worker.attention_observation.UPSTREAM)},'Exactly ten strict, r3 and full upstream anchors')
    r.alias_metadata.validate_spec(dict(source_sha256=catalog['source_sha256'],metadata_aliases=aliases))
    expanded_plan(spec,base)
    return spec


def actual_sources(base,preparation):
    pins={}
    for row in preparation['sources']:
        r.same(r.p.sha(row['path']),row['sha256'],'Frozen worker/runner preparation source')
        pins=merge(pins,{row['path']:row['sha256']})
    required={r.SELF,str(ROOT/'minicpm5_frame1_attention_worker_v1.py'),str(Q/'review_minicpm5_frame1_attention_complete_v1.py'),
              SELF,TEST,str(Q/'review_minicpm5_frame1_attention_complete_v1.py'),*r.worker.ACTIVATIONS.values(),*map(str,r.worker.upstream.NORM_SOURCES.values())}
    required|={str(Path(module.__file__).resolve())for module in tuple(sys.modules.values())
               if getattr(module,'__file__',None)and Path(module.__file__).is_relative_to(ROOT)}
    # AST closure includes deferred local imports without importing native libraries.
    import ast
    todo=list(map(Path,required));seen=set()
    while todo:
        path=todo.pop()
        if path in seen:continue
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            names=[x.name for x in node.names]if isinstance(node,ast.Import)else[node.module]if isinstance(node,ast.ImportFrom)and node.module and not node.level else[]
            for name in names:
                for base_dir in (ROOT,Q):
                    candidate=base_dir/(name.replace('.','/')+'.py')
                    if candidate.is_file()and str(candidate)not in required:required.add(str(candidate));todo.append(candidate)
    for name in sorted(required):
        if name not in base['source_sha256']:pins=merge(pins,{name:r.p.sha(name)})
    for name,sha in base['source_sha256'].items():r.same(r.p.sha(name),sha,'Inherited original source remains exact')
    return pins


def actual_anchors():
    result={}
    for full,refs in ((False,r.worker.upstream.CPU_ANCHORS),(True,r.worker.upstream.FULL_CPU_ANCHORS)):
        for rank,ref in enumerate(refs):
            row=r.p.read_ref(ref);r.worker.validate_baseline(row,rank,full)
            result[ref['path']]=dict(identity=r.identity(ref['path']),sha256=ref['sha256'])
    for reference in r.worker.upstream.R3_CAPTURES:
        value=r.p.read_ref(reference)
        r.p.require(value['complete']is True and value['schema']=='minicpm5_frame1_mlp_capture_v2','Actual complete r3 anchor')
        result[reference['path']]=dict(identity=r.identity(reference['path']),sha256=reference['sha256'])
    for reference in r.worker.attention_observation.UPSTREAM:
        value=r.p.read_ref(reference)
        r.p.require(value['complete']is True and value['schema']=='minicpm5_frame1_upstream_capture_v1','Actual completed upstream anchor')
        result[reference['path']]=dict(identity=r.identity(reference['path']),sha256=reference['sha256'])
    return result


def build(catalog_ref,smol_ref,run_id,port,finalizer_exit_ref):
    r.p.require(smol_ref['path']==str(SMOL),'Only actual completed sio14 accepted')
    # Reading real history precedes construction; missing completion stops here.
    r.same(catalog_ref['path'],str(r.history_metadata.CATALOG_PATH),'Exact actual completed r7 catalog path')
    catalog=r.history_metadata.load(catalog_ref);smol=r.history_metadata.load(smol_ref)
    r.history_metadata.validate_binding(dict(current_catalog=catalog_ref,current_finalizer_exit=finalizer_exit_ref),catalog)
    base=r.p.read_ref(r.BASE_PLAN);prep=r.p.read_ref(PREPARATION)
    r.p.require(prep['source_only'] is True and prep['runtime_admitted'] is False,'Frozen source preparation grants no launch')
    chain=select_chain(smol_ref,smol,r.history_metadata.load)
    aliases={name:r.alias_metadata.capture(name,r.history_metadata.text)
             for name in sorted(set(catalog['source_sha256'])&set(r.alias_metadata.ALLOWED))}
    pins=actual_sources(base,prep);anchors=actual_anchors()
    spec=compose(base,catalog_ref,catalog,smol_ref,chain,aliases,pins,anchors,run_id,port,finalizer_exit_ref)
    r.same(r.current_history(spec),catalog,'Actual immutable completed history chain verified')
    history=r.authenticate_history_sources(spec,catalog)
    r.same({name:r.p.sha(ROOT/name)for name in r.CANONICAL},r.CANONICAL,'Canonical files unchanged')
    r.p.require(not Path(spec['output_root']).exists(),'Never reuse runtime output namespace')
    prefix=Q/('minicpm5-frame1-attention-'+run_id)
    spec_path=Path(str(prefix)+'-spec.json');checks_path=Path(str(prefix)+'-preparation-checks.json')
    r.p.require(not spec_path.exists()and not checks_path.exists(),'New immutable spec namespace')
    # No independent review is generated; no lock, device, service or model runs.
    spec_ref=r.worker.once(spec_path,spec)
    plan=expanded_plan(spec,base)
    checks=dict(schema='minicpm5_frame1_runtime_preparation_checks_v1',reviewer='/root',specification=spec_ref,
        source_preparation=PREPARATION,builder=dict(path=SELF,sha256=r.p.sha(SELF)),current_catalog=catalog_ref,required_sio14_catalog=smol_ref,
        inherited_scientific_fields_unchanged=True,source_count=len(plan['source_sha256']),input_count=len(plan['inputs']),
        history_source_checks=history,history_services=len(catalog['rows']),history_processes=len(catalog['processes']),
        original_canonical_unchanged=True,fresh_physical_drain_performed=False,model_framework_imported=False,
        execution_review_created=False,runtime_admitted=False,model_execution_performed=False,new_speedup_claim=False)
    r.worker.once(checks_path,checks)
    return dict(specification=spec_ref,checks=dict(path=str(checks_path),sha256=r.p.sha(checks_path)),runtime_admitted=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('catalog','catalog-sha256','smol-catalog','smol-catalog-sha256','run-id','finalizer-exit','finalizer-exit-sha256'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--port',type=int,default=29571);args=parser.parse_args()
    r.p.require(sys.dont_write_bytecode and __debug__,'No bytecode; assertions retained')
    print(json.dumps(build(dict(path=args.catalog,sha256=args.catalog_sha256),dict(path=args.smol_catalog,sha256=args.smol_catalog_sha256),args.run_id,args.port,dict(path=args.finalizer_exit,sha256=args.finalizer_exit_sha256))))
