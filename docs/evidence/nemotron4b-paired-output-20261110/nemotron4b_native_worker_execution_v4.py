"""Connect unchanged native/meta science to the v3 output-aware helper successor.

Not a standalone launcher. A concrete root must provide authenticated startup
and phase admission under its serial lock; this core cannot create admission.
Tests substitute scientific functions with stdlib fixtures explicitly. Real
SDK execution requires the original phase Python and full imported origins.
"""
import copy
from datetime import datetime,timezone
import ipaddress
import os
from pathlib import Path
import subprocess
import sys

import nemotron4b_native_science_v3 as science
from nemotron4b_gated_sdk_helper_v2 import GatedSDKHelper,LIBRARY
from nemotron4b_process_handshake_v1 import require,scope,owner
from nemotron4b_worker_receipts_v1 import ReceiptWriter

MODEL='nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16'
REVISION='dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f'
FORBIDDEN_PAYLOAD=('.safetensors','.pt','.pth','.bin','.neff','.npy','.npz')


class WorkerAudit:
    def __init__(self,work,helper):self.work,self.helper=Path(work),helper

    def __call__(self,event,args):
        if event=='subprocess.Popen' or event in('os.system','os.exec','os.posix_spawn','os.fork','os.forkpty'):
            require(self.helper is not None,'Native-meta worker cannot launch external processes')
            # Before the helper first installs its own permanent hook, deny all
            # spawning. Afterwards that hook owns the exact one-event budget.
            if not self.helper.audit_installed:self.helper.audit_subprocess(event,args)
            return
        if event=='open'and args and isinstance(args[0],(str,bytes)):
            path=Path(os.fsdecode(args[0]))
            require(not path.name.endswith(FORBIDDEN_PAYLOAD),'No weights or compiled payloads in native/meta stage')
            flags=args[2]
            if flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC):
                resolved=path.resolve()
                require(resolved.is_relative_to(self.work)or str(resolved)=='/dev/null',
                        'Worker writes only under admitted work directory')
        if event in('socket.connect','socket.connect_ex','socket.sendto','socket.sendmsg'):
            address=args[-1]
            require(type(address)is tuple and address and type(address[0])is str,
                    'Only original loopback distributed traffic')
            try:loopback=ipaddress.ip_address(address[0]).is_loopback
            except ValueError:loopback=address[0]=='localhost'
            require(loopback,'No remote network traffic in prerequisite')


def original_plan():
    require(science.sha(science.ORIGINAL_WORKER)==science.ORIGINAL_WORKER_SHA256
            and science.sha(science.ORIGINAL_PLAN)==science.ORIGINAL_PLAN_SHA256,
            'Original accepted worker/plan unchanged')
    plan=science.metadata(science.ORIGINAL_PLAN)
    require(plan['model']==MODEL and plan['revision']==REVISION
            and plan['phases']==['native-meta','nxd-meta']and len(plan['source_sha256'])==119,
            'Original model/revision and full scientific plan')
    return plan


def restored_environment(plan,phase):
    expected=plan['environments'][phase];python=Path(expected['python'])
    require(list(sys.version_info[:2])==expected['python_version']
            and Path(sys.executable).resolve()==python.resolve()
            and Path(sys.prefix)==python.parent.parent,'Original phase-specific Python/venv')
    require(sys.dont_write_bytecode and sys.flags.optimize==0 and __debug__,'Original -B unoptimized runtime')
    require(Path.cwd()==science.ROOT and os.sched_getaffinity(0)
            and os.sched_getaffinity(0)<={6,7}and os.getpriority(os.PRIO_PROCESS,0)==19,
            'Original working directory, CPU allocation and priority')
    for key in('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','PYTHONNOUSERSITE','PYTHONDONTWRITEBYTECODE'):
        require(os.environ.get(key)=='1','Original offline/startup environment: '+key)
    require(os.environ.get('PYTHONOPTIMIZE')=='0'
            and os.environ.get('OMP_NUM_THREADS')==os.environ.get('MKL_NUM_THREADS')
            ==os.environ.get('OPENBLAS_NUM_THREADS')=='1','Original single-thread native environment')
    require(not any(os.environ.get(k)for k in('PYTHONPATH','PYTHONHOME','LD_PRELOAD','NEURON_CC_FLAGS','NEURON_COMPILED_ARTIFACTS')),
            'No inherited import/compile overrides')
    require(os.environ.get('NEURON_CPU_SAMPLER_FAST')=='0','Original standard CPU sampler')
    if phase=='nxd-meta':
        require(os.environ.get('GLOO_SOCKET_IFNAME')=='lo'and os.environ.get('PJRT_DEVICE')=='NEURON',
                'Original loopback TP2 and Neuron backend')


def run(*,phase,role,nonce,guard,parent_gate,startup_context,writer,helper=None):
    """One phase in its own authenticated process; no retry or service launch.

    startup_context must restore the exact phase venv after the gate and install
    full authenticated import/native policies. guard checks root/serial/resource
    ownership without requiring the scientific modules to remain unimported.
    """
    scope(nonce,role)
    require((phase=='native-meta'and role=='native-meta'and helper is None)
            or(phase=='nxd-meta'and role in('rank-0','rank-1')and type(helper)is GatedSDKHelper),
            'Exact native phase or original paired rank and helper')
    require(all(callable(x)for x in(guard,parent_gate,startup_context))and type(writer)is ReceiptWriter
            and writer.directory==science.WORK,'Actual guard/gate/startup and fixed admitted v3 output')
    if helper is not None:
        require(helper.role=='helper-'+role[-1]and helper.nonce==nonce and helper.writer is writer,
                'Actual helper bound to the same rank, attempt and output writer')
    require(os.environ.get('NEMOTRON4B_NATIVE_PREREQUISITE_ATTEMPT')==nonce,'Actual worker nonce environment')
    require(not {'torch','transformers','vllm','neuronx_distributed','libneuronxla'}&set(sys.modules),
            'No native/SDK imports before parent gate')
    guard();plan=original_plan()
    result=None;runtime=None;released=None;provenance=None;installed_audit=False
    try:
        released=parent_gate()
        require(type(released)is dict and released['ready']['role']==role
                and released['ready']['nonce']==nonce and released['ready']['owner']==owner(os.getpid()),
                'Actual matching worker release and process identity')
        writer.write(role+'-release.json',released)
        guard()
        with startup_context()as provenance:
            require(type(provenance)is dict,'Actual restored startup provenance')
            restored_environment(plan,phase)
            if phase=='nxd-meta':
                require(os.environ.get('RANK')==role[-1]and os.environ.get('WORLD_SIZE')=='2',
                        'Original exact TP2 rank environment')
            guard()
            sys.addaudithook(WorkerAudit(science.WORK,helper));installed_audit=True
            original_run=subprocess.run
            try:
                if helper is not None:subprocess.run=helper.run
                result=science.native_meta(plan)if phase=='native-meta'else science.nxd_meta(plan)
            finally:subprocess.run=original_run
            if helper is not None:
                require(helper.calls==1 and helper.proof is not None
                        and os.environ.get('NEURON_LIBRARY_PATH')==LIBRARY
                        and os.environ.get('PJRT_DEVICE')=='NEURON','Original actual SDK import helper outcome')
                result['import_helper']=copy.deepcopy(helper.proof)
            runtime=science.runtime_receipt(plan,phase)
            guard()
        guard()
        report=dict(schema='nemotron4b_native_phase_receipt_v4',passed=True,phase=phase,role=role,
            nonce=nonce,observed_at=datetime.now(timezone.utc).isoformat(),model=MODEL,revision=REVISION,
            original_plan=dict(path=str(science.ORIGINAL_PLAN),sha256=science.ORIGINAL_PLAN_SHA256),
            original_worker=dict(path=str(science.ORIGINAL_WORKER),sha256=science.ORIGINAL_WORKER_SHA256),
            producer=owner(os.getpid()),release=released,startup=provenance,result=result,runtime=runtime,
            native_execution=True,model_forward_executed=False,original_weight_payload_read=False,
            baseline_created=False,full_model_qualified=False,new_speedup_claim=False)
        reference=writer.write(role+'-result.json',report)
        return dict(passed=True,role=role,receipt=reference,full_model_qualified=False,new_speedup_claim=False)
    except BaseException as error:
        failed=dict(schema='nemotron4b_native_phase_failure_v4',passed=False,phase=phase,role=role,
            nonce=nonce,error=dict(type=type(error).__name__,message=str(error)[:2048]),
            release=released,startup=provenance,scientific_result=result,runtime=runtime,
            worker_audit_installed=installed_audit,full_model_qualified=False,new_speedup_claim=False)
        # Save any completed science/runtime even if final context/receipt fails.
        # Parent owns complete raw output and whole-group cleanup regardless.
        try:writer.write(role+'-failure.json',failed)
        except BaseException as receipt_error:
            print('Native phase failure receipt could not be saved: '+repr(receipt_error),file=sys.stderr,flush=True)
        raise


if __name__=='__main__':
    raise SystemExit('Requires concrete reviewed root admission and exact phase startup; not independently runnable')
