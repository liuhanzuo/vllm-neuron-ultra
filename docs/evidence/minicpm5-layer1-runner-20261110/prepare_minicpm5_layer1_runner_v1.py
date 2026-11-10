"""Source preparation now; concrete runtime candidate only after actual r11 exit.

No native dependency authentication, model execution or real tensor reads here.
Runtime revalidates the complete original predecessor and physical release.
"""
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
Q=Path(__file__).resolve().parent;ROOT=Q.parent
sys.path[:0]=[str(ROOT),str(Q)]
import minicpm5_original32_layer1_observation_cpu_runner_v1 as runner
import minicpm5_original32_layer1_observation_cpu_worker_v1 as worker
from prepare_dory_residual_spec_v3 import local_imports
from research.prepare_minicpm5_layer1_observation_source_v1 import Reader as ScientificReader
history=runner.history
require=history.require
ENTRIES=(Path(runner.__file__),Path(worker.__file__),Q/'minicpm5_after_nemotron_r11_v1.py',
 Q/'minicpm5_layer1_drain_hold_v1.py',Path(__file__).resolve(),Q/'check_minicpm5_layer1_runner_v1.py')
SOURCE_OUTPUT=Path('/mnt/nvme/minicpm5-original32-layer0-full-bf16-cpu-layer1-runner-source-20261110')

def reference(path):
    path=Path(path)
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def source_candidate():
    reader=ScientificReader()
    original=reader.read(**dict(path=runner.RESULTS_SOURCE_PLAN['path'],digest=runner.RESULTS_SOURCE_PLAN['sha256']))
    plan=copy.deepcopy(original);plan['output_root']=str(SOURCE_OUTPUT)
    require(not SOURCE_OUTPUT.exists(),'Source placeholder is not a runtime output')
    graph=local_imports(entries=ENTRIES)
    for path,digest in graph['source_sha256'].items():
        require(path not in plan['source_sha256']or plan['source_sha256'][path]==digest,'Frozen original source retained')
        plan['source_sha256'][path]=digest
    for module in tuple(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if path and str(path).startswith(str(ROOT)+'/'):
            path=Path(path).resolve()
            require(path.suffix=='.py','Only loaded local Python source')
            digest=reference(path)['sha256']
            require(str(path)not in plan['source_sha256']or plan['source_sha256'][str(path)]==digest,
                    'Loaded source agrees with frozen closure')
            plan['source_sha256'][str(path)]=digest
    def read(path,digest):
        entry=plan['inputs'][path]
        require(entry['sha256']==digest and entry['identity']==worker.identity(path),'Original exact metadata input')
        return reader.read(path,digest)
    worker.validate_plan(plan,read)
    require(not {'torch','numpy','neuronxcc','torch_neuronx','vllm','transformers'}&set(sys.modules),
            'Source preparation does not import a model/SDK')
    require({k:v for k,v in plan.items()if k not in ('source_sha256','output_root')}==
            {k:v for k,v in original.items()if k not in ('source_sha256','output_root')},
            'All original scientific inputs and requirements unchanged')
    suffixes={};total=0
    for path,digest in plan['source_sha256'].items():
        size=Path(path).stat().st_size
        require(size<8*2**20 and Path(path).suffix not in ('.pt','.safetensors','.bin','.neff','.so','.whl','.raw','.zlib','.pb'),
                'Original complete bounded text/source policy')
        total+=size;suffix=Path(path).suffix;suffixes[suffix]=suffixes.get(suffix,0)+1
    checks=dict(schema='minicpm5_layer1_runner_source_candidate_v1',source_only=True,
        source_files=len(plan['source_sha256']),source_bytes=total,source_suffix_counts=suffixes,
        inherited_source_plan=runner.RESULTS_SOURCE_PLAN,
        all_original_scientific_fields_preserved=True,all_original_source_pins_retained=True,
        completed_original_call_metadata=84,expected_new_model_calls=84,
        source_only_metadata_files=len(reader.refs),source_only_metadata_bytes=reader.bytes,
        runtime_spec_created=False,runtime_admitted=False,
        source_entrypoints=[reference(path)for path in ENTRIES],
        current_nemotron_spec=history.SPEC,current_nemotron_review=history.REVIEW,
        actual_future_terminal_required=True,complete_previous_services_expected=1255,
        model_or_real_tensor_payloads_read=False,native_dependency_authentication_performed=False,
        full_model_qualified=False,new_speedup_claim=False)
    return plan,checks

def assemble_runtime_candidate(actual_exit_ref,failed_release_exit_ref,attempt):
    require(type(attempt)is str and re.fullmatch('[a-z0-9-]{1,40}',attempt),'Finite fresh attempt identifier')
    reader=history.Reader(lambda:None)
    receipt=reader(actual_exit_ref)
    history.terminal.validate_actual_exit(receipt,dict(owner=history.LIVE_OWNER),
        history.SPEC,history.REVIEW,history.LIVE_SESSION)
    spec=reader(history.SPEC);control=Path(spec['control_root'])
    release=reader.path(control/'root-release.json')
    require(release['audit']['path']in (str(control/'external-audit.json'),str(control/'external-recovery-audit.json')),
            'Exact original released audit')
    names=dict(intent='external-launch-intent.json',pre='external-pre-stop.json',admission='admission.json',
        release='root-release.json',audit=Path(release['audit']['path']).name)
    for key in ('terminal','failed'):
        name='terminal.json'if key=='terminal'else'failed-terminal.json'
        if (control/name).exists():names[key]=name
    if names['audit']=='external-recovery-audit.json':names['original_audit']='external-audit.json'
    refs={key:reference(control/name)for key,name in names.items()}
    records={key:reader(ref)for key,ref in refs.items()}
    failed=reader(failed_release_exit_ref)if failed_release_exit_ref is not None else None
    outcome=history.terminal_chain(spec,receipt,records,refs,failed)
    plan,checks=source_candidate()
    scientific_reader=ScientificReader()
    ending=scientific_reader.read(runner.OBSERVED_EXIT['path'],runner.OBSERVED_EXIT['sha256'])
    original=scientific_reader.read(ending['spec']['path'],ending['spec']['sha256'])
    output=Path('/mnt/nvme')/('minicpm5-original32-layer0-full-bf16-cpu-layer1-'+attempt)
    require(not output.exists(),'Fresh complete collection output')
    plan['output_root']=str(output)
    candidate=copy.deepcopy(original)
    candidate.update(schema='minicpm5_layer1_observation_cpu_execution_spec_v1',runner=runner.SELF,
        worker=str(ROOT/'minicpm5_original32_layer1_observation_cpu_worker_v1.py'),
        output_root=str(output),source_sha256=plan['source_sha256'],
        output_floor_bytes=30*2**30,unit='trn1-mini-layer1-observation-'+attempt,master_port=29617,
        prepared_source_plan=runner.RESULTS_SOURCE_PLAN,original_completed_collection=runner.OBSERVED_EXIT,
        nemotron_completion=dict(actual_exit=actual_exit_ref,failed_release_exit=failed_release_exit_ref))
    # Caller must write plan exactly and then bind its actual reference here.
    candidate['plan']=None
    checks.update(actual_predecessor_terminal_authenticated=True,nemotron_outcome=outcome,
        actual_runtime_plan_needs_durable_reference=True,final_concrete_source_review_required=True,
        actual_physical_drain_still_required=True)
    return candidate,plan,checks

def write_source():
    plan,checks=source_candidate()
    target=Q/'minicpm5-layer1-runner-source-plan-20261110-v1.json'
    proof=Q/'minicpm5-layer1-runner-source-checks-20261110-v1.json'
    require(not target.exists()and not proof.exists(),'Fresh immutable source preparation')
    with target.open('x')as f:json.dump(plan,f,indent=2);f.write('\n')
    checks['plan']=reference(target)
    with proof.open('x')as f:json.dump(checks,f,indent=2);f.write('\n')
    print(json.dumps(dict(plan=reference(target),checks=reference(proof),source_files=checks['source_files'],
        source_bytes=checks['source_bytes'],runtime_admitted=False)))
if __name__=='__main__':
    require(sys.dont_write_bytecode and __debug__,'Python -B with assertions')
    write_source()
