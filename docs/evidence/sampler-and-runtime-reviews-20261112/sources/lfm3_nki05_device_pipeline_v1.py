"""Concrete complete device pipeline connected to the retained serial owner.

Preparation/import is inert. Real launch operations are only reachable through
the root-reviewed exact serial runner and future complete r2 admission.
"""
from contextlib import contextmanager
import copy
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

import lfm3_nki05_full_cpu_metadata_v1 as cpu
import lfm3_nki05_device_protocol_v1 as protocol
import lfm3_nki05_device_budget_v1 as budget
import lfm3_nki05_device_sdk_v1 as sdk
from lfm3_nki05_device_runtime_v1 import Context,stable_hash
import run_lfm3_typed_writer_cpu_pilot_v3 as legacy

PHASES=protocol.PHASES
MINIMUM_CATALOG=dict(path=str(cpu.ROOT/'research/minicpm5-operator-post-r2-serial-workloads-root-catalog-20261112-v76.json'),sha256='d9b86ae53918266626c5ce6ee63f6909d359bcaa32012e7d5b05f5b3280bed3f')
require=protocol.require


def protocol_manifest():return protocol.manifest()


def reference_shape(ref):
    require(type(ref) is dict and set(ref)=={'path','sha256'} and type(ref['path']) is str and Path(ref['path']).is_absolute()
            and Path(ref['path']).suffix=='.json' and type(ref['sha256']) is str and len(ref['sha256'])==64
            and set(ref['sha256'])<=set('0123456789abcdef'),'Exact future JSON reference')


def validate_predecessor_bindings(spec):
    cpu.validate_binding(spec['full_cpu_prerequisite'])
    require(spec['serial_catalog']!=MINIMUM_CATALOG,'The old live-r2 catalog cannot admit devices')
    reference_shape(spec['sdk_manifest'])


def prerequisite(host,spec):
    validate_predecessor_bindings(spec)
    proof=cpu.audit(host,spec['full_cpu_prerequisite'])
    original=host.p.read(cpu.CPU_SPEC['path'],cpu.CPU_SPEC['sha256'])
    require(all(spec['source_sha256'].get(path)==digest for path,digest in original['source_sha256'].items()),'Full original CPU dependency pins retained in additive closure')
    return proof


def require_cpu_catalog(spec,proof,read):
    old=read(MINIMUM_CATALOG);current=read(spec['serial_catalog'])
    require(current['runs'][:len(old['runs'])]==old['runs'],'Every original 121 catalog row retained in order')
    rows=[row for row in current['runs'] if row['terminal']==spec['full_cpu_prerequisite']['actual_exit']]
    require(len(rows)==1 and rows[0]['tool']==proof['actual_tool'] and proof['unit'] in rows[0]['units'],'Actual r2 outer exit appears exactly once in full future catalog')
    for owner in proof['owners']:
        require(any(all(row.get(k)==owner[k] for k in ('pid','start_ticks','ppid','argv','cgroup')) for row in rows[0]['owners']),'Catalog retains actual r2 outer and service owner')
    require(current['current_handoff']==spec['terminal_handoff'],'Exact future current idle terminal')


def require_cpu_drained(proof,guard):
    import minicpm5_layer0_execution_protocol_v4 as p
    for owner in proof['owners']+proof['children']:guard();p.gone(owner)
    p.empty_group(p.GROUP_PREFIX+proof['unit']+'.service');guard()


def child_argv(host,admission_ref,task):
    require(task in protocol.tasks(),'Fixed matrix task')
    return [host.PYTHON,'-B',host.SELF,'--mode','batch','--admission',admission_ref['path'],
        '--admission-sha256',admission_ref['sha256'],'--phase',task['phase'],'--index',str(task['index'])]


def child_environment(host,root,attempt,task):
    env=host.environment(root,attempt)
    env['TMPDIR']=str(Path(root)/'tasks'/(protocol.key(task)+'-tmp'))
    return env


@contextmanager
def child_binding(host,spec,admission_ref,admission,owner,root):
    """Reuse exact original PID/start/cgroup/pidfd descendant-drain helpers."""
    parent=dict(owner,cgroup=owner['cgroup'].removeprefix('0::'))
    def read(path):
        require(Path(path)==root/'worker-owner.json' and host.p.read(path)==owner,'Exact acknowledged actual parent only')
        return parent
    def service():return host.service_identity(spec,admission_ref,admission['attempt_id'],host.properties(spec['unit']),owner)
    with patch.multiple(legacy.p,DEST=root,read=read,exact_service=service):yield


def run_one(host,spec,admission_ref,admission,owner,root,task,admit,remaining):
    label=protocol.key(task);child=None;child_owner=None;code=None;failure=None;secondary=[]
    report_path=root/'tasks'/(label+'-result.json');started=time.monotonic()
    with child_binding(host,spec,admission_ref,admission,owner,root):
        try:
            admit();argv=child_argv(host,admission_ref,task)
            with (root/'tasks'/(label+'.log')).open('xb') as stream:
                child=subprocess.Popen(argv,cwd=host.ROOT,env=child_environment(host,root,admission['attempt_id'],task),
                    stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                child_owner=legacy.child_identity(child,argv)
                host.once(root/'tasks'/(label+'-owner.json'),child_owner)
                deadline=time.monotonic()+min(remaining,budget.PHASE_SECONDS if task['index']==-1 else budget.CHILD_SECONDS)
                while True:
                    try:code=child.wait(timeout=min(2,max(.01,deadline-time.monotonic())));break
                    except subprocess.TimeoutExpired:
                        admit();require(time.monotonic()<deadline,'Finite original matrix child deadline')
            require(code==0,'Actual device child nonzero exit preserved')
            report=host.p.read(report_path)
            require(report['complete'] is report['passed'] is True and report['task']==task and report['admission']==admission_ref
                    and report['child']['pid']==child.pid and report['child']['start_ticks']==child_owner['start_ticks'],'Complete child report tied to actual exited child')
            value=host.p.read(report['science']['path'],report['science']['sha256']);protocol.validate_task(task,value,spec['source_sha256'])
        except BaseException as error:failure=error
        finally:
            if child is not None:
                try:
                    require(child_owner is not None,'Unacknowledged child requires complete service drain')
                    legacy.stop_child(child,child_owner)
                    if code is None:code=child.returncode
                except BaseException as error:secondary.append(repr(error)[:2048]);failure=failure or error
    outcome=dict(task=task,pid=child.pid if child else None,start_ticks=child_owner['start_ticks'] if child_owner else None,
        returncode=code,passed=failure is None,seconds=time.monotonic()-started,
        report=host.reference(report_path) if report_path.exists() else None,error=repr(failure)[:2048] if failure else None,secondary_errors=secondary)
    return outcome,failure


def invoke(context,task):
    phase,index=task['phase'],task['index']
    if phase=='raw_device':
        from lfm3_nki05_raw_device_leaf_v1 import run
        run(context)
    elif phase=='regression_device':
        from lfm3_nki05_regression_device_leaf_v1 import run
        run(context)
    elif phase=='regression_alias':
        from lfm3_nki05_regression_alias_leaf_v1 import run
        run(context)
    elif phase=='extended_device':
        from lfm3_nki05_extended_device_leaf_v2 import device
        device(context,index)
    elif phase=='extended_alias':
        from lfm3_nki05_extended_device_leaf_v2 import alias
        alias(context,index)
    elif phase=='fp32_alias':
        from lfm3_nki05_device_supplements_v2 import fp32_alias
        fp32_alias(context,index)
    elif phase=='wrapper_controls':
        from lfm3_nki05_device_supplements_v2 import wrapper_controls
        wrapper_controls(context,index)
    else:raise RuntimeError('Unknown scientific phase')


def run_child(host,admission_ref,task):
    require(task in protocol.tasks(),'Fixed complete scientific task')
    admission=host.p.read(admission_ref['path'],admission_ref['sha256']);spec=host.read_bundle(admission['spec'],admission['review'])
    proof=prerequisite(host,spec);host.load_history();root=Path(spec['output_root']);label=protocol.key(task)
    ack=root/'tasks'/(label+'-owner.json');deadline=time.monotonic()+60
    while not ack.exists():
        host.descriptor_proof(admission['outer_owner'],admission['retained_fd'])
        require(time.monotonic()<deadline and os.getppid()>1,'Actual parent ACK required before scientific import');time.sleep(.05)
    context=Context(host,admission_ref,task,proof)
    try:
        host.bytecode_guard(root)
        runtime=host.p.read(spec['sdk_manifest']['path'],spec['sdk_manifest']['sha256'])
        runtime_proof=sdk.verify(runtime,context)
        runtime_path=root/'tasks'/(label+'-sdk.json');host.once(runtime_path,dict(runtime_proof,manifest=spec['sdk_manifest'],admission=admission_ref,child=context.process))
        context.sdk_verified=True
        invoke(context,task)
        result_path=protocol.scientific_path(root,task);value=host.p.read(result_path)
        protocol.validate_task(task,value,spec['source_sha256']);context.authorize(task['phase'])
        host.once(root/'tasks'/(label+'-result.json'),dict(complete=True,passed=True,task=task,child=context.process,
            admission=admission_ref,science=host.reference(result_path),artifacts=context.artifacts,
            sdk=host.reference(runtime_path),warning_log=host.warning_receipt('batch'),
            full_model_qualified=False,new_speedup_claim=False,context4096_qualified=False,self_exit_claimed=False))
    except BaseException as error:
        host.report_failure(root/'tasks'/(label+'-failure.json'),error,child=context.process,task=task,admission=admission_ref)
        raise


def read_phase(host,root,phase,sources,admission_ref,*,require_summary=True):
    planned=[row for row in protocol.tasks() if row['phase']==phase]
    path=root/phase/'phase-outcomes.json';outcomes=host.p.read(path)
    require(outcomes['complete'] is outcomes['passed'] is True and outcomes['admission']==admission_ref
            and outcomes['phase']==phase and [row['task'] for row in outcomes['outcomes']]==planned,'Every actual phase child outcome in exact order')
    reports=[];refs=[]
    parent=host.p.read(root/'worker-owner.json')
    admission=host.p.read(admission_ref['path'],admission_ref['sha256'])
    admitted_spec=host.p.read(admission['spec']['path'],admission['spec']['sha256'])
    expected_sdk=admitted_spec['sdk_manifest']
    for task,row in zip(planned,outcomes['outcomes']):
        require(row['returncode']==0 and row['passed'] is True and row['error'] is None and row['secondary_errors']==[],'Every actual child exited zero without cleanup failure')
        label=protocol.key(task);expected=root/'tasks'/(label+'-result.json')
        require(row['report']==host.reference(expected),'Actual exact child result pin')
        report=host.p.read(expected,row['report']['sha256']);ack=host.p.read(root/'tasks'/(label+'-owner.json'))
        require(report['complete'] is report['passed'] is True and report['task']==task and report['admission']==admission_ref
                and row['pid']==report['child']['pid']==ack['pid'] and row['start_ticks']==report['child']['start_ticks']==ack['start_ticks']
                and ack['argv']==report['child']['argv']==child_argv(host,admission_ref,task)
                and ack['parent']==dict(parent,cgroup=parent['cgroup'].removeprefix('0::'))
                and report['child']['ppid']==parent['pid'] and report['child']['cgroup']=='0::'+ack['cgroup'],'All actual ACK/parent/PID/argv/start bindings')
        host.p.gone(report['child'])
        host.known_warnings.validate_receipt(report['warning_log'],'batch')
        require(all(report[key] is False for key in ('full_model_qualified','new_speedup_claim','context4096_qualified','self_exit_claimed')),'Primitive-only child report')
        science_path=protocol.scientific_path(root,task);require(report['science']==host.reference(science_path),'Exact actual scientific report pin')
        value=host.p.read(science_path,report['science']['sha256']);protocol.validate_task(task,value,sources)
        require(report['sdk']['path']==str(root/'tasks'/(label+'-sdk.json')),'Exact owned child SDK receipt')
        sdk_proof=host.p.read(report['sdk']['path'],report['sdk']['sha256']);require(sdk_proof['complete'] is sdk_proof['passed'] is True and sdk_proof['child']==report['child'] and sdk_proof['admission']==admission_ref and sdk_proof['manifest']==expected_sdk,'Actual same-child reviewed SDK authentication')
        expected_artifacts=value['graphs'] if task['index']==-1 else [value]
        require(len(report['artifacts'])==len(expected_artifacts),'Every compiled graph has independent artifact receipt')
        for artifact_ref,graph in zip(report['artifacts'],expected_artifacts):
            identity=host.p.read(artifact_ref['path'],artifact_ref['sha256'])
            require(identity==graph['artifact_identity'] and identity['admission']==admission_ref and identity['task']==task and identity['child']==report['child'],'Complete actual saved artifact identity')
            artifact=Path(identity['path'])
            require(artifact.name=='model.pt' and artifact.is_relative_to(root/phase) and Path(artifact_ref['path'])==artifact.parent/'artifact-identity.json','New owned compiled artifact only')
        reports.append(value);refs.append(row['report'])
    value=copy.deepcopy(reports[0]) if len(planned)==1 and planned[0]['index']==-1 else dict(complete=True,passed=True,graphs=reports)
    value.update(phase=phase,admission=admission_ref,task_reports=refs,outcomes=host.reference(path),
                 full_model_qualified=False,new_speedup_claim=False,context4096_qualified=False)
    graphs,checks=protocol.COUNTS[phase]
    require(len(value['graphs'])==graphs and (len(value['checks']) if phase=='raw_device' else sum(len(x['checks']) for x in value['graphs']))==checks,'Complete original or supplemental graph/check matrix')
    if phase=='extended_alias':
        for reference,candidate in zip(value['graphs'][::2],value['graphs'][1::2]):
            require(reference['spec']['variant']=='reference' and candidate['spec']['variant']=='candidate'
                    and reference['initial_sha256']==candidate['initial_sha256']
                    and reference['inputs_sha256']==candidate['inputs_sha256'],'Every original alias pair used identical complete fixtures')
    if require_summary:require(host.p.read(root/phase/'phase-complete.json')==value,'Complete independent phase summary matches all real outcomes')
    return value


def execute(host,admission_ref,owner,root,admit):
    admission=host.p.read(admission_ref['path'],admission_ref['sha256']);spec=host.read_bundle(admission['spec'],admission['review'])
    result=dict(schema='lfm3_nki05_device_result_v1',complete=False,passed=False,admission=admission_ref,
        phases=[],phase_counts={},protocol=protocol.manifest(),full_model_qualified=False,new_speedup_claim=False,
        context4096_target_retained=True,context4096_qualified=False,automatic_successor_launch=False)
    started=time.monotonic()
    try:
        admit();host.bytecode_guard(root);(root/'tasks').mkdir()
        for phase in PHASES:
            (root/phase).mkdir();phase_started=time.monotonic();outcomes=[];failure=None
            try:
                for task in [row for row in protocol.tasks() if row['phase']==phase]:
                    (root/'tasks'/(protocol.key(task)+'-tmp')).mkdir()
                    remaining=min(budget.PHASE_SECONDS-(time.monotonic()-phase_started),budget.WORKER_SECONDS-(time.monotonic()-started))
                    require(remaining>0,'Finite phase and whole scientific window')
                    outcome,error=run_one(host,spec,admission_ref,admission,owner,root,task,admit,remaining);outcomes.append(outcome)
                    if error is not None:raise error
            except BaseException as error:failure=error
            finally:
                host.once(root/phase/'phase-outcomes.json',dict(phase=phase,admission=admission_ref,complete=failure is None,passed=failure is None,outcomes=outcomes,seconds=time.monotonic()-phase_started))
            if failure is not None:raise failure
            summary=read_phase(host,root,phase,spec['source_sha256'],admission_ref,require_summary=False)
            host.once(root/phase/'phase-complete.json',summary);result['phases'].append(host.reference(root/phase/'phase-complete.json'))
            result['phase_counts'][phase]=list(protocol.COUNTS[phase]);admit()
        result.update(complete=True,passed=True,warning_log=host.warning_receipt('worker'))
        host.once(root/'diagnostic-result.json',result)
        audit_success(host,result)
        host.once(root/'worker-result.json',dict(complete=True,passed=True,owner=owner,admission=admission_ref,
            diagnostic=host.reference(root/'diagnostic-result.json'),self_exit_claimed=False,full_model_qualified=False,new_speedup_claim=False))
    except BaseException as error:
        if not (root/'diagnostic-result.json').exists():host.once(root/'diagnostic-result.json',dict(result,error=repr(error)[:2048],warning_log=host.warning_receipt('worker')))
        host.report_failure(root/'worker-failure.json',error,owner=owner,admission=admission_ref);raise


def audit_success(host,result):
    require(result['schema']=='lfm3_nki05_device_result_v1' and result['complete'] is result['passed'] is True
            and result['protocol']==protocol.manifest() and result['phase_counts']=={phase:list(protocol.COUNTS[phase]) for phase in PHASES},'Complete unchanged device protocol')
    require(result['context4096_target_retained'] is True and all(result[k] is False for k in ('full_model_qualified','new_speedup_claim','context4096_qualified','automatic_successor_launch')),'No model/4096/performance promotion')
    host.known_warnings.validate_receipt(result['warning_log'],'worker')
    admission=host.p.read(result['admission']['path'],result['admission']['sha256']);spec=host.read_bundle(admission['spec'],admission['review']);root=Path(spec['output_root'])
    require([ref['path'] for ref in result['phases']]==[str(root/phase/'phase-complete.json') for phase in PHASES],'Complete exact phase order')
    for phase,ref in zip(PHASES,result['phases']):
        host.descriptor_proof(admission['outer_owner'],admission['retained_fd'])
        require(ref==host.reference(root/phase/'phase-complete.json'),'Exact immutable phase summary pin')
        value=read_phase(host,root,phase,spec['source_sha256'],result['admission'])
        for graph in value['graphs']:
            record=graph['artifact_identity']
            require(stable_hash(record['path'],budget.ARTIFACT_BYTES)==record['sha256'],'Complete saved compiled artifact bytes still match')
    return True
