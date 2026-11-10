"""Bounded live-controller guard micro-measurement; no production preparation.

All source maps remain in memory. One synthetic contract is served from memory.
Audit hooks deny payload/native opens, dynamic native imports and process creation.
"""
import ast
import base64
import collections
import contextlib
import copy
import datetime
import hashlib
import fcntl
import select
import selectors
import socket
import uuid
import importlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import re
import resource
import shlex
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import time
import traceback
import types
import unittest
from unittest.mock import patch
import zlib

ROOT=Path(__file__).resolve().parents[1];Q=ROOT/'research';sys.path[:0]=[str(ROOT),str(Q)]
LIMITS=dict(address_space_bytes=1024*2**20,cpu_seconds=120,hard_cpu_seconds=125,
    wall_seconds=180,output_file_bytes=2**20,cpu=6,nice=19)
os.sched_setaffinity(0,{6});os.nice(19)
resource.setrlimit(resource.RLIMIT_AS,(LIMITS['address_space_bytes'],)*2)
resource.setrlimit(resource.RLIMIT_CPU,(LIMITS['cpu_seconds'],LIMITS['hard_cpu_seconds']))
resource.setrlimit(resource.RLIMIT_FSIZE,(LIMITS['output_file_bytes'],)*2)
signal.signal(signal.SIGALRM,lambda *a:(_ for _ in ()).throw(TimeoutError('Source-only wall bound')))
signal.signal(signal.SIGXCPU,lambda *a:(_ for _ in ()).throw(TimeoutError('Source-only CPU bound')))
signal.alarm(LIMITS['wall_seconds'])
gate_fd=int(os.environ['NEMOTRON_OBSERVER_GATE_FD']);gate_identity=os.fstat(gate_fd)
START=time.monotonic();CPU=time.process_time()
CTYPES_PY='/usr/lib/python3.10/ctypes/__init__.py'
CTYPES_NATIVE='/usr/lib/python3.10/lib-dynload/_ctypes.cpython-310-x86_64-linux-gnu.so'
DECIMAL_PY='/usr/lib/python3.10/decimal.py'
DECIMAL_NATIVE='/usr/lib/python3.10/lib-dynload/_decimal.cpython-310-x86_64-linux-gnu.so'
stdlib_bootstrap_native_modules={name:getattr(module,'__file__',None)for name,module in sys.modules.items()
    if isinstance(getattr(module,'__file__',None),str)and Path(module.__file__).suffix=='.so'}
state=dict(stdlib_decimal_exceptions=[],stdlib_ctypes_exceptions=[],stage='bootstrap',current_path=None,guard_calls=0,read_open_events=0,
    read_open_file_extent_bytes=0,forbidden_open_attempts=[],forbidden_operations=[],
    verified_text_files=0,verified_text_returned_bytes=0,verified_python_bytes_files=0,
    verified_python_bytes_returned_bytes=0,virtual_contract_reads=0,virtual_contract_bytes=0)
phases={};stage_started=START
FORBIDDEN={'.raw','.bin','.bf16','.pt','.pth.tar','.safetensors','.neff','.npy','.npz','.so','.whl','.zlib','.pb','.h5','.onnx'}
NATIVE_PREFIXES={'torch','numpy','vllm','torch_neuronx','neuronxcc','neuronx_distributed','neuronx_distributed_inference',
    'transformers','safetensors','jax','tensorflow','dbus'}
TEXT_SUFFIXES={'.py','.pyc','.json','.md','.txt','.log','.jsonl','.jinja','.patch','.yaml','.toml','.ini','.typed','.pth','.metadata'}
TEXT_BASENAMES={'.gitignore','LICENSE','METADATA','RECORD','WHEEL','PKG-INFO'}
allowed_reads=None
classified_text_paths=set()
explicitly_allowed_non_source_reads={'/proc/locks'}|{f'/proc/727516/{name}'for name in ('stat','cmdline','cgroup')}
classified_forbidden_paths=set()
def failure(message,path=None):
    if path is not None:state['forbidden_open_attempts'].append(str(path))
    raise RuntimeError(message)
def audit(event,args):
    if event=='ctypes.dlopen' and args==(None,):
        module=sys.modules.get('ctypes');initializing=getattr(getattr(module,'__spec__',None),'_initializing',False)
        existing=[x for x in state['stdlib_ctypes_exceptions']if x['operation']=='ctypes.dlopen']
        if state['stage']=='bootstrap'and initializing and getattr(module,'__file__',None)==CTYPES_PY and not existing:
            state['stdlib_ctypes_exceptions'].append(dict(operation='ctypes.dlopen',argument=None,
                purpose='Exact stdlib initialization pythonapi=PyDLL(None)',source=CTYPES_PY))
            return
    if event in {'fcntl.flock','fcntl.lockf','os.exec','subprocess.Popen','os.system','os.fork','os.forkpty','os.posix_spawn','ctypes.dlopen','ctypes.dlsym','ctypes.call_function','ctypes.call_cdeclfunction'}:
        state['forbidden_operations'].append(event);raise RuntimeError('Forbidden source-only operation: '+event)
    if event=='import':
        name=args[0];filename=args[1]if len(args)>1 else None
        if name=='_decimal'and filename==DECIMAL_NATIVE and state['stage']=='bootstrap':
            module=sys.modules.get('decimal')
            initializing=getattr(getattr(module,'__spec__',None),'_initializing',False)
            if not initializing or getattr(module,'__file__',None)!=DECIMAL_PY or state['stdlib_decimal_exceptions']:
                raise RuntimeError('Only exact one-time stdlib decimal bootstrap allowed')
            state['stdlib_decimal_exceptions'].append(dict(operation='import:_decimal',path=filename,
                source=DECIMAL_PY,purpose='Frozen protocol imports stdlib statistics -> fractions -> decimal'))
            return
        if name=='_ctypes'and filename==CTYPES_NATIVE and state['stage']=='bootstrap':
            existing=[x for x in state['stdlib_ctypes_exceptions']if x['operation']=='import:_ctypes']
            if existing:raise RuntimeError('Only one explicitly recorded stdlib _ctypes bootstrap allowed')
            state['stdlib_ctypes_exceptions'].append(dict(operation='import:_ctypes',path=filename,
                purpose='Frozen stdlib-only lifecycle module imports ctypes; no project ctypes function execution'))
            return
        if name.split('.')[0]in NATIVE_PREFIXES or (isinstance(filename,str)and('.so.'in Path(filename).name or Path(filename).suffix=='.so')):
            state['forbidden_operations'].append('import:'+name);raise RuntimeError('Forbidden native/model import: '+name)
    if event=='open':
        path=args[0]
        if isinstance(path,int):return
        path=Path(os.fsdecode(path));mode=args[1];flags=args[2]
        if isinstance(flags,int)and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):
            failure('No source-only file writes permitted',path)
        if path.suffix in FORBIDDEN or '.so.'in path.name or path.name.endswith('.pth.tar'):
            failure('Forbidden historical/model/native payload read',path)
        absolute=str(path if path.is_absolute()else ROOT/path)
        if absolute in classified_forbidden_paths:
            failure('Exact historical/business-native category read prohibited',path)
        if not(path.suffix in TEXT_SUFFIXES or path.name in TEXT_BASENAMES or absolute in classified_text_paths or absolute in explicitly_allowed_non_source_reads):
            failure('Unclassified source-only read path',path)
        if allowed_reads is not None and absolute not in allowed_reads:
            failure('Read outside complete source/metadata/guard allowlist',path)
        state['read_open_events']+=1
        try:
            info=path.stat()
            if stat.S_ISREG(info.st_mode):state['read_open_file_extent_bytes']+=info.st_size
        except OSError:pass
sys.addaudithook(audit)

def phase(name):
    global stage_started
    now=time.monotonic();phases[state['stage']]=now-stage_started
    stage_started=now;state['stage']=name
def guard():
    state['guard_calls']+=1
    current=os.fstat(gate_fd)
    if (current.st_dev,current.st_ino)!=(gate_identity.st_dev,gate_identity.st_ino):
        raise RuntimeError('Inherited observer gate identity changed')
def merge(target,addition):
    for path,wanted in addition.items():
        if path in target and target[path]!=wanted:raise RuntimeError('Inherited source pin conflict: '+path)
        target[path]=wanted


REFS={
'spec':dict(path=str(Q/'nemotron9b-package-runtime-v27-r12-20261110-spec.json'),sha256='4186c9a29921b96fc3a55fb04040cf70977ca921130df288466c8d27e65feaaa'),
'review':dict(path=str(Q/'nemotron9b-package-runtime-v27-r12-20261110-source-review.json'),sha256='99b0cdb4ab842ca97b4b6b3283aacd595f8dabb60806e0823fae89458fcdef40'),
'external_live':dict(path=str(Q/'nemotron9b-r12-actual-external-live-20261110-v1.json'),sha256='4a81f4c2eef387b6955f9da3dfcf42bc759833f460a00af4d654f80b4aa414ab'),
'controller':dict(path='/mnt/nvme/nemotron9b-package-r12-20261110-control/external-controller-owner.json',sha256='396063c49d59f8a07e9597837a0373909fd6f49e94cf2a48e1e6946fcdda9014'),
'admission':dict(path='/mnt/nvme/nemotron9b-package-r12-20261110-control/admission.json',sha256='817c074454ba5d60cec525ac791f152e124c2bed143ad17cea9417ebe7273f9f'),
'window':dict(path=str(ROOT/'nemotron9b-package-window-v3.json'),sha256='1f0e8adcff7258fe372a64eb41daa53601d2a05a4077a759d16ee2b810caa573'),
'protocol_source':dict(path=str(ROOT/'nemotron9b_package_protocol_v27.py'),sha256='402e550cb6a098ce6d85d8095ba6e5af7d6452e89717a51352eb4dd89f469ed6'),
'low_source':dict(path=str(ROOT/'minicpm5_layer0_execution_protocol_v4.py'),sha256='81e7d7dd913c6b85f0add7b946692d3270f80cc8f2fe980ecad543251657f185'),
'clear_window_source':dict(path=str(ROOT/'minicpm5_layer0_runtime_auth_v1.py'),sha256='92e6f2e7cf1009e94ce548474ea5cd455b7cc1ea00813fb44f5790ae85fb1a27')}
LOCK=dict(device=66306,inode=8294421)
EXPECTED_ID=dict(pid=727516,start_ticks='409785848',invocation_id='1eaac4fd6907434688e74b4a9cd91bd2')
TARGET_CALLS=43994
completed_calls=0;loop_start=None;loop_cpu=None;loop_end=None;loop_cpu_end=None
result=dict(schema='minicpm5_sdk_builder_live_controller_guard_micro_measurement_v1',
 source_only=True,micro_measurement_only=True,limits=LIMITS,requested_calls=TARGET_CALLS,
 held_lock_owner_is_live_controller=True,measurement_process_claims_original_lock=False,
 original_lock_acquired_or_released=False,production_builder_executed=False,
 actual_completed_or_physical_called=False,actual_native_hashing=False,
 full_source_hash_scan_repeated=False,model_or_tensor_bytes_read=0,
 runtime_ready=False,new_speedup_claim=False,refs=REFS,
 measurement_note='Only original p.low.lock_proof(controller_proc_owner,lock)+p.low.clear_window calls; audit and observer guard overhead included. Not production builder/source/worker/auditor budget.')
try:
    import nemotron9b_package_protocol_v27 as p
    if any(n.split('.')[0]in NATIVE_PREFIXES for n in sys.modules):raise RuntimeError('Native/model dependency unexpectedly imported')
    # Freeze ctypes loading/call entrypoints after its exact recorded stdlib
    # initialization. No preloading or application-native exception is used.
    ctypes_module=sys.modules.get('ctypes')
    if ctypes_module is not None:
        def denied_ctypes(*a,**kw):
            state['forbidden_operations'].append('ctypes-call')
            raise RuntimeError('No ctypes application loading or calls allowed')
        ctypes_module.CDLL.__init__=denied_ctypes
        ctypes_module.PyDLL.__init__=denied_ctypes
        for name in ('memmove','memset','string_at','wstring_at','cast'):
            if hasattr(ctypes_module,name):setattr(ctypes_module,name,denied_ctypes)

    phase('authenticate_exact_metadata')
    assert Path(p.low.__file__).resolve()==Path(REFS['low_source']['path'])
    assert p.low.clear_window.__module__=='minicpm5_layer0_runtime_auth_v1'
    assert callable(p.low.lock_proof)and callable(p.low.clear_window)
    allowed_reads={x['path']for x in REFS.values()}|explicitly_allowed_non_source_reads|{
        str(ROOT/name/'status.json')for name in ('smollm3-cache-ablation-full-v8','lfm3-package-paired-full-v2')}
    def read_ref(ref):
        state['current_path']=ref['path'];path=Path(ref['path']);before=path.lstat()
        assert stat.S_ISREG(before.st_mode)and not path.is_symlink()and before.st_size<=32*2**20
        with path.open('rb')as handle:raw=handle.read(32*2**20+1)
        after=path.lstat()
        assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(
            after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns)
        assert len(raw)==before.st_size and hashlib.sha256(raw).hexdigest()==ref['sha256']
        raw.decode('utf8')
        return json.loads(raw)if path.suffix=='.json'else raw
    values={key:read_ref(ref)for key,ref in REFS.items()}
    spec,review,live,controller_record,admission,window=[values[x]for x in ('spec','review','external_live','controller','admission','window')]
    assert spec['schema']=='nemotron9b_package_runtime_v27'and spec['run_id']=='r12-20261110'
    assert review['passed']is True and review['execution_spec_sha256']==REFS['spec']['sha256']
    assert review['source_sha256']==spec['source_sha256']
    for key in ('protocol_source','low_source','clear_window_source'):
        assert spec['source_sha256'][REFS[key]['path']]==REFS[key]['sha256']
    for value in (live,admission,window):
        assert value['execution_spec']==REFS['spec']and value['source_review']==REFS['review']
    assert live['actual_live_identity_verified']is True
    controller=controller_record['owner']
    assert controller==admission['controller']
    assert all(controller[k]==v for k,v in EXPECTED_ID.items())
    proc_owner=p.proc_owner(controller)
    assert set(proc_owner)=={'pid','start_ticks','ppid','argv','cgroup'}
    assert window['owner']==proc_owner and window['state']=='active'and window['performance_window']is False
    assert admission['lock_identity']==LOCK
    assert (window['gate']['device'],window['gate']['inode'])==(gate_identity.st_dev,gate_identity.st_ino)
    assert p.low.proc(proc_owner['pid'])==proc_owner
    assert os.getpid()!=proc_owner['pid']
    result.update(controller=controller,controller_proc_owner=proc_owner,original_lock_identity=LOCK,
        inherited_observer_gate=dict(device=gate_identity.st_dev,inode=gate_identity.st_ino),
        measurement_pid=os.getpid(),exact_refs_authenticated=True,original_guard_functions_verified=True)
    # No further source or input metadata reads are allowed during the loop.
    allowed_reads=explicitly_allowed_non_source_reads|{
        str(ROOT/name/'status.json')for name in ('smollm3-cache-ablation-full-v8','lfm3-package-paired-full-v2')}
    phase('guard_loop');state['current_path']=None
    loop_start=time.monotonic();loop_cpu=time.process_time()
    first_proof=None;last_proof=None
    for _ in range(TARGET_CALLS):
        # Fail immediately on controller PID/start/argv/cgroup or lock mismatch.
        proof=p.low.lock_proof(proc_owner,LOCK)
        p.low.clear_window()
        completed_calls+=1;state['guard_calls']=completed_calls
        if first_proof is None:first_proof=proof
        last_proof=proof
    loop_end=time.monotonic();loop_cpu_end=time.process_time()
    assert completed_calls==TARGET_CALLS and p.low.proc(proc_owner['pid'])==proc_owner
    guard()
    # guard() only checks inherited observer FD; distinguish from measured calls.
    state['guard_calls']=completed_calls
    phase('complete')
    result.update(passed=True,first_lock_proof=first_proof,last_lock_proof=last_proof,
        controller_still_exact_at_end=True,failed_path=None)
except BaseException as error:
    result.update(passed=False,error_type=type(error).__name__,error=str(error),
        failed_stage=state['stage'],failed_path=state['current_path'],
        traceback_frames=[dict(path=f.f_code.co_filename,line=n,function=f.f_code.co_name)
            for f,n in traceback.walk_tb(error.__traceback__)])
finally:
    signal.alarm(0)
    now=time.monotonic();cpu_now=time.process_time()
    if loop_start is not None:
        result['loop_elapsed_seconds']=(loop_end if loop_end is not None else now)-loop_start
        result['loop_cpu_seconds']=(loop_cpu_end if loop_cpu_end is not None else cpu_now)-loop_cpu
        if completed_calls:result['loop_seconds_per_completed_call']=result['loop_elapsed_seconds']/completed_calls
    result.update(completed_calls=completed_calls,elapsed_seconds=now-START,cpu_seconds=cpu_now-CPU,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,observations=state,
        observer_gate_retained=True,stdlib_bootstrap_native_modules=stdlib_bootstrap_native_modules,
        no_unapproved_native_imports_after_hook=not any(x.startswith('import:')for x in state['forbidden_operations']))
    print(json.dumps(result,indent=2,sort_keys=True),flush=True)
if not result['passed']:raise SystemExit(1)
