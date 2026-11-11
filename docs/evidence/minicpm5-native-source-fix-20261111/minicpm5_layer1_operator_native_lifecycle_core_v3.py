"""Small native runner lifecycle core with explicit reviewed callback boundaries.

No CLI, default lease adapter, process implementation or automatic IO. Imports
are stdlib/source only. Callbacks must be separately admitted by the real outer;
this core never acquires/releases a lock or observes its own external tool exit.
"""
import copy
import json
import math
from pathlib import Path
import re
import subprocess
from research import minicpm5_layer1_operator_native_output_catalog_v2 as catalog
from minicpm5_layer0_runtime_auth_v1 import digest
from research.minicpm5_native_static_binding_v1 import StaticBinding
require=catalog.require
ROOT='/home/ubuntu/experiments/trn1-latest-models-20261031'
PYTHON='/home/ubuntu/venvs/model_reference_2026/bin/python'
PROPS=dict(MemoryMax=str(16*2**30),MemoryHigh=str(14*2**30),MemorySwapMax='0',TasksMax='128',
    AllowedCPUs='6-7',CPUAffinity='6-7',Nice='19',KillMode='control-group',RemainAfterExit='yes',SendSIGKILL='yes',
    WorkingDirectory=ROOT,LimitFSIZE=str(16*2**20),RuntimeMaxSec='7200',TimeoutStopSec='15')
PROCESS_KEYS=('pid','start_ticks','ppid','argv','cgroup')


def ref(value,path=None):
    require(type(value)is dict and set(value)=={'path','sha256'}and type(value['path'])is str
        and Path(value['path']).is_absolute()and str(Path(value['path']))==value['path']and '..'not in Path(value['path']).parts
        and type(value['sha256'])is str and re.fullmatch('[a-f0-9]{64}',value['sha256'])
        and (path is None or value['path']==path),'Exact bounded receipt/source reference')
    return value


def build_contract(resolved,overlay_ref,runner_ref,unit):
    """Pure memory binding; small refs/digests, never duplicate the full plan."""
    ref(overlay_ref);ref(runner_ref);plan=resolved['plan'];c=catalog.catalog(plan['native_operator_manifest'])
    require(resolved['runtime_admitted']is False and resolved['full_runtime_catalog_pending']is False
        and resolved['runtime_integration_pending']is True and resolved['physical_checks_performed']is False,
        'Accepted complete overlay still requires actual runtime integration/admission')
    require(digest(plan)==resolved['plan_sha256']and plan['runtime_output_manifest']==c
        and resolved['native_runtime_output_catalog']==c and resolved['native_runtime_output_catalog_sha256']==digest(c),
        'Resolved full plan and complete catalog bindings')
    require(plan['interpreters']['native']==PYTHON and plan['source_sha256'].get(runner_ref['path'])==runner_ref['sha256']
        and Path(runner_ref['path']).parent==Path(ROOT)and Path(runner_ref['path']).suffix=='.py',
        'Pinned future native runner with original native interpreter')
    require(type(unit)is str and re.fullmatch('trn1-mini-layer1-operator-native-[a-z0-9-]{1,80}',unit),'Fresh exact native unit name')
    return dict(schema='minicpm5_native_lifecycle_contract_v2',overlay=copy.deepcopy(overlay_ref),
        runner=copy.deepcopy(runner_ref),resolved_plan_sha256=resolved['plan_sha256'],
        native_manifest_sha256=digest(plan['native_operator_manifest']),output_catalog_sha256=digest(c),
        output_root=plan['output_root'],unit=unit,python=PYTHON,resources=copy.deepcopy(PROPS),
        root_floor_bytes=28*2**30,output_floor_bytes=30*2**30,
        topology=['outer','coordinator','native'],native_children=1,native_threads=2,
        lease_adapter_dependency='separately_reviewed_authenticated_original_descriptor_lease_provider',
        actual_lease_adapter_bound=False,actual_process_ports_bound=False,launch_ready=False,
        external_model_exit_required=True,independent_raw_audit_required=True,
        full_model_qualified=False,new_speedup_claim=False)


def binding(contract,resolved):
    require(contract==build_contract(resolved,contract['overlay'],contract['runner'],contract['unit']),
        'Unchanged exact small contract and resolved plan/catalog/resource bindings')


def command(contract,mode,admission_ref):
    ref(admission_ref,contract['output_root']+'/admission.json')
    require(mode in ('coordinator','native'),'Original three-process native topology')
    return [contract['python'],'-B',contract['runner']['path'],'--mode',mode,
        '--admission',admission_ref['path'],'--admission-sha256',admission_ref['sha256']]


def process(owner,argv,parent=None,group=None):
    require(type(owner)is dict and all(key in owner for key in PROCESS_KEYS)
        and type(owner['pid'])is int and owner['pid']>0 and type(owner['ppid'])is int and owner['ppid']>=0
        and type(owner['start_ticks'])is str and re.fullmatch('[1-9][0-9]*',owner['start_ticks'])
        and owner['argv']==argv and type(owner['cgroup'])is str and owner['cgroup'].startswith('0::/'),
        'Complete actual PID/start/argv/parent/cgroup acknowledgment')
    if parent is not None:require(owner['ppid']==parent,'Only direct acknowledged native child')
    if group is not None:require(owner['cgroup']==group,'Original shared owned native service group')
    return {key:copy.deepcopy(owner[key])for key in PROCESS_KEYS}


def coordinator_process(contract,admission_ref,owner):
    value=process(owner,command(contract,'coordinator',admission_ref))
    require(owner['unit']==contract['unit']and type(owner['invocation_id'])is str
        and re.fullmatch('[a-f0-9]{32}',owner['invocation_id'])
        and value['cgroup'].endswith('/'+contract['unit']+'.service'),
        'Exact acknowledged coordinator unit, invocation and owned group')
    return value


def storage(contract,ports):
    available=ports.storage_available()
    require(type(available)is dict and set(available)=={'root','output'}
        and all(type(value)is int for value in available.values())
        and available['root']>=contract['root_floor_bytes']and available['output']>=contract['output_floor_bytes'],
        'Original root28GiB and output30GiB floors remain live')


def usage(contract,resolved,entries,root_entry,*,successful_driver=False,writers=None,static_binding=None):
    if static_binding is None:
        binding(contract,resolved)
        proof=catalog.inventory(resolved['plan']['native_operator_manifest'],entries,
            successful_driver=successful_driver,writers=writers)
    else:
        require(type(static_binding)is StaticBinding,'Exact authenticated static lifecycle binding')
        proof=static_binding.inventory(contract,resolved,entries,
            successful_driver=successful_driver,writers=writers)
    require(root_entry['kind']=='directory'and root_entry['uid']==1000 and root_entry['symlink']is False
        and type(root_entry['size'])is int and root_entry['size']>=0
        and type(root_entry['blocks'])is int and root_entry['blocks']>=0,'Exact owned output-root directory extent')
    allocated=max(root_entry['size'],root_entry['blocks']*512)
    require(proof['total_allocated_bytes']+allocated<=256*2**20,'Original total cap includes output-root allocation')
    return dict(proof,output_root_allocated_bytes=allocated,total_with_root_allocated_bytes=proof['total_allocated_bytes']+allocated)


def receipt(value):
    require(len((json.dumps(value,indent=2,allow_nan=False)+'\n').encode())<8*2**20,'Original strict8MiB runtime JSON cap')
    return value


def publish(ports,contract,name,value):
    require(name in catalog.RUNTIME_FILES,'Only original finite lifecycle receipt names')
    reference=ports.publish(name,receipt(value));return ref(reference,contract['output_root']+'/'+name)


def secondary(ports,primary,error,stage):
    try:ports.secondary(primary,error,stage)
    except BaseException:pass


def hold(lease,ports,primary):
    # The outer descriptor owner must retain its lease; no release method exists
    # in this module, including exceptions while reporting this requirement.
    try:lease.retain_after_failure(primary)
    except BaseException as error:secondary(ports,primary,error,'retain_lease')


def check_driver_result(value,contract,resolved):
    require(type(value)is dict and not {'process','admission'}&set(value)
        and value['schema']=='minicpm5_layer1_operator_native_driver_v4'and value['complete']is True
        and value['self_exit_claimed']is False and value['manifest_sha256']==contract['native_manifest_sha256']
        and value['frames']==24 and value['direct_attention_calls']==72 and value['raw_files']==936
        and value['raw_bytes']==resolved['plan']['native_operator_manifest']['total_raw_bytes']
        and value['native_provenance']==resolved['plan']['native_provenance']
        and value['completed_sdk_layout']==resolved['plan']['layer1_operator_boundary']
        and value['all_arm_journals_completed']is True and value['independent_post_exit_raw_audit_pending']is True
        and value['full_model_qualified']is value['new_speedup_claim']is False,
        'Original driver_v4 object without invented process/admission fields; independent raw audit remains pending')
    return value


def wrap_native_result(driver,contract,resolved,owner,admission_ref):
    """Future worker wraps the unchanged driver result exactly once."""
    binding(contract,resolved);check_driver_result(driver,contract,resolved)
    process(owner,command(contract,'native',admission_ref))
    return receipt(dict(schema='minicpm5_native_authenticated_result_v1',driver_result=copy.deepcopy(driver),
        process=copy.deepcopy(owner),admission=copy.deepcopy(admission_ref),
        resolved_plan_sha256=contract['resolved_plan_sha256'],self_exit_claimed=False))


def unwrap_native_result(value,contract,owner,admission_ref,resolved):
    require(type(value)is dict and set(value)=={'schema','driver_result','process','admission','resolved_plan_sha256','self_exit_claimed'}
        and value['schema']=='minicpm5_native_authenticated_result_v1'and value['process']==owner
        and value['admission']==admission_ref and value['resolved_plan_sha256']==contract['resolved_plan_sha256']
        and value['self_exit_claimed']is False,'Exact authenticated native wrapper binds process/admission/plan')
    return check_driver_result(value['driver_result'],contract,resolved)


def release_binding(contract,admission_ref,co,owner_ref):
    ref(owner_ref,contract['output_root']+'/native-owner.json')
    return dict(schema='minicpm5_native_child_release_v1',native_owner=copy.deepcopy(owner_ref),
        admission=copy.deepcopy(admission_ref),contract_sha256=digest(contract),
        resolved_plan_sha256=contract['resolved_plan_sha256'],coordinator_owner=copy.deepcopy(co))


def coordinator_step(contract,resolved,admission_ref,coordinator_owner,lease,ports):
    """Direct Popen-compatible child handle; all process/IO adapters injected.

    lease.check(stage) must assert the authenticated original lease and complete
    current scope. ports supplies actual observed process/clock/output operations.
    No default ports or lease provider are supplied by this source increment.
    """
    binding(contract,resolved);co=coordinator_process(contract,admission_ref,coordinator_owner)
    require(callable(getattr(lease,'check',None))and callable(getattr(lease,'retain_after_failure',None)),
        'Explicit authenticated lease-provider dependency, never a success boolean')
    child=None;owner=None;observed_exit=None
    def check(stage):
        lease.check(stage);require(ports.current_process()==co,'Coordinator actual identity remains unchanged')
        storage(contract,ports)
        entries,root,writers=ports.inventory()
        usage(contract,resolved,entries,root,successful_driver=False,writers=writers,static_binding=ports.static_binding)
    def cleanup(primary):
        if child is None:return
        try:
            if primary is not None and child.poll()is None:child.terminate()
            try:ended=child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();ended=child.wait(timeout=5)
            require(type(ended)is int and child.poll()==ended,'Actual child reaped, no boolean/missing wait status')
            ports.assert_child_drained(child,owner)
            publish(ports,contract,'children-exits.json',dict(complete=True,owners=[]if owner is None else[owner],
                exit_codes=[ended],successful=primary is None and ended==0,self_exit_claimed=False))
        except BaseException as error:
            hold(lease,ports,primary or error)
            if primary is None:raise
            secondary(ports,primary,error,'native_child_cleanup')
    primary=None;native_ref=None
    try:
        check('coordinator.before_native_spawn')
        child=ports.spawn_native(command(contract,'native',admission_ref))
        owner=process(ports.acknowledge(child,timeout=10),command(contract,'native',admission_ref),co['pid'],co['cgroup'])
        require(child.pid==owner['pid']and child.poll()is None,'Acknowledged original live child handle')
        check('coordinator.before_child_acknowledgment');owner_ref=publish(ports,contract,'native-owner.json',owner)
        check('coordinator.before_child_release')
        ports.release_native(child,release_binding(contract,admission_ref,co,owner_ref))
        started=ports.monotonic();require(type(started)in(int,float)and math.isfinite(started),'Finite actual monotonic clock')
        while child.poll()is None:
            check('coordinator.monitor_native');now=ports.monotonic()
            require(type(now)in(int,float)and math.isfinite(now)and started<=now<started+7200,'Original bounded native runtime')
            ports.idle(.5)
        observed_exit=child.wait(timeout=5)
        require(type(observed_exit)is int and observed_exit==0 and child.poll()==observed_exit,'Actual native wait exit0, never report presence')
        check('coordinator.after_native_wait')
        native_ref=ref(ports.reference('native-result.json'),contract['output_root']+'/native-result.json')
        result=ports.read_json(native_ref);unwrap_native_result(result,contract,owner,admission_ref,resolved)
    except BaseException as error:
        primary=error;hold(lease,ports,error);raise
    finally:
        # Exactly one cleanup attempt owns wait/drain/children-exits publication.
        # A later coordinator-receipt failure must not rewrite exclusive receipts.
        cleanup(primary)
    try:
        check('coordinator.after_native_cleanup')
        ports.static_binding.verify_full(contract,resolved)
        value=receipt(dict(schema='minicpm5_native_compact_coordinator_v2',complete=True,contract_sha256=digest(contract),
            admission=copy.deepcopy(admission_ref),coordinator_owner=co,native_owner=owner,native_exit=observed_exit,
            native_result=copy.deepcopy(native_ref),independent_raw_audit_pending=True,external_model_exit_pending=True,
            self_exit_claimed=False,full_model_qualified=False,new_speedup_claim=False))
        return publish(ports,contract,'coordinator-result.json',value)
    except BaseException as error:
        hold(lease,ports,error);raise


def outer_terminal_step(contract,resolved,admission_ref,coordinator_owner,lease,ports):
    """Consume actual service/drain callbacks; never record outer/tool exit here."""
    binding(contract,resolved);co=coordinator_process(contract,admission_ref,coordinator_owner)
    try:
        lease.check('outer.before_terminal_observation')
        storage(contract,ports)
        service=ports.observe_service_terminal(contract,coordinator_owner)
        require(service['Result']=='success'and service['ExecMainCode']=='1'and service['ExecMainStatus']=='0'
            and service['ExecMainPID']==str(co['pid'])and service['MainPID']==service['ControlPID']=='0'
            and (service['ActiveState'],service['SubState'])in(('inactive','dead'),('active','exited'))
            and service['InvocationID']==coordinator_owner['invocation_id'],'Actual owned service terminal exit0')
        # Callback must prove exact service/group/owner scope drained. The outer
        # adapter retains the lock on failure and records physical evidence.
        drain_ref=ref(ports.assert_service_drained(contract,coordinator_owner),contract['output_root']+'/cleanup-drained.json')
        co_ref=ref(ports.reference('coordinator-result.json'),contract['output_root']+'/coordinator-result.json')
        value=ports.read_json(co_ref)
        require(value['schema']=='minicpm5_native_compact_coordinator_v2'and value['complete']is True
            and value['contract_sha256']==digest(contract)and value['admission']==admission_ref
            and value['coordinator_owner']==co and type(value['native_exit'])is int and value['native_exit']==0
            and value['independent_raw_audit_pending']is value['external_model_exit_pending']is True
            and value['self_exit_claimed']is value['full_model_qualified']is value['new_speedup_claim']is False,
            'Compact actual coordinator result, external model/auditor exits still pending')
        owner=process(value['native_owner'],command(contract,'native',admission_ref),co['pid'],co['cgroup'])
        native_ref=ref(value['native_result'],contract['output_root']+'/native-result.json')
        unwrap_native_result(ports.read_json(native_ref),contract,owner,admission_ref,resolved)
        entries,root,writers=ports.inventory();proof=usage(contract,resolved,entries,root,successful_driver=True,writers=writers,static_binding=ports.static_binding)
        lease.check('outer.after_terminal_validation')
        return receipt(dict(schema='minicpm5_native_outer_terminal_core_v2',complete=True,contract_sha256=digest(contract),
            coordinator_result=co_ref,native_result=native_ref,service_observation=copy.deepcopy(service),
            owned_drain=drain_ref,usage=proof,self_exit_claimed=False,actual_external_tool_exit_verified=False,
            independent_raw_audit_pending=True,full_model_qualified=False,new_speedup_claim=False))
    except BaseException as primary:
        hold(lease,ports,primary);raise
