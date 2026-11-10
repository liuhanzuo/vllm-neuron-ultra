"""Bounded source-only compatibility measurement; no production preparation.

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
stdlib_bootstrap_native_modules={name:getattr(module,'__file__',None)for name,module in sys.modules.items()
    if isinstance(getattr(module,'__file__',None),str)and Path(module.__file__).suffix=='.so'}
state=dict(stdlib_ctypes_exceptions=[],stage='bootstrap',current_path=None,guard_calls=0,read_open_events=0,
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
    if event in {'subprocess.Popen','os.system','os.fork','os.forkpty','os.posix_spawn','ctypes.dlopen','ctypes.dlsym','ctypes.call_function','ctypes.call_cdeclfunction'}:
        state['forbidden_operations'].append(event);raise RuntimeError('Forbidden source-only operation: '+event)
    if event=='import':
        name=args[0];filename=args[1]if len(args)>1 else None
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
        if not(path.suffix in TEXT_SUFFIXES or path.name in TEXT_BASENAMES):
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

result=dict(schema='minicpm5_sdk_batch_layout_source_compatibility_measurement_v1',
    source_only=True,limits=LIMITS,production_files_written=False,actual_build_called=False,
    actual_r12_completion_binding_created=False,actual_completed_or_physical_called=False,
    actual_native_hashing=False,model_or_tensor_bytes_read=0)
try:
    from research import minicpm5_layer1_sdk_batch_layout_sources_v2 as sources
    from research import prepare_minicpm5_layer1_sdk_batch_layout_v2 as builder
    from research.prepare_dory_residual_spec_v3 import local_imports
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
    phase('read_frozen_maps')
    r12=sources.load(sources.R12_SPEC)
    mini_ref=dict(path=str(Q/'minicpm5-layer1-current-history-factorial-r1-20261110-plan.json'),
        sha256='88045c16e58b559561ab949671a36a3c34bd12f42ac0518df06f5825e73a6b72')
    mini=sources.load(mini_ref)
    phase('static_local_closure')
    graph=local_imports(entries=builder.entrypoints())
    pins={};merge(pins,r12['source_sha256']);merge(pins,mini['source_sha256']);merge(pins,graph['source_sha256'])
    merge(pins,{sources.R12_SPEC['path']:sources.R12_SPEC['sha256'],mini_ref['path']:mini_ref['sha256']})
    contract=dict(schema=sources.CONTRACT_SCHEMA,r12_spec=sources.R12_SPEC,metadata_limits=sources.metadata.LIMITS)
    contract_raw=(json.dumps(contract,sort_keys=True)+'\n').encode()
    contract_ref=dict(path=str(Q/'minicpm5-layer1-sdk-batch-layout-source-contract-in-memory-measurement.json'),
        sha256=hashlib.sha256(contract_raw).hexdigest())
    if Path(contract_ref['path']).exists():raise RuntimeError('Synthetic contract path must not exist')
    merge(pins,{contract_ref['path']:contract_ref['sha256']})
    status_paths={str(ROOT/name/'status.json')for name in ('smollm3-cache-ablation-full-v8','lfm3-package-paired-full-v2')}
    allowed_reads=set(pins)|status_paths|{row['target']for row in r12['metadata_aliases'].values()}
    original_text=sources.text
    original_python=sources.metadata.python_source_bytes
    def measured_text(path,expected=None):
        state['current_path']=str(path)
        if str(path)==contract_ref['path']:
            if expected!=contract_ref['sha256']:raise RuntimeError('Exact synthetic contract digest required')
            state['virtual_contract_reads']+=1;state['virtual_contract_bytes']+=len(contract_raw);return contract_raw
        raw=original_text(path,expected)
        if state['stage']=='verify_all_text':
            state['verified_text_files']+=1;state['verified_text_returned_bytes']+=len(raw)
        return raw
    def measured_python(path,expected):
        state['current_path']=str(path);raw=original_python(path,expected)
        state['verified_python_bytes_files']+=1;state['verified_python_bytes_returned_bytes']+=len(raw)
        return raw
    sources.text=measured_text;sources.metadata.python_source_bytes=measured_python
    phase('classify_complete_map')
    partition=sources.classify(pins,contract_ref)
    result.update(original_r12_source_files=len(r12['source_sha256']),original_mini_source_files=len(mini['source_sha256']),
        local_source_graph_files=graph['files'],local_source_graph_bytes=graph['bytes_read'],
        complete_source_files=len(pins),categories={k:len(v)for k,v in partition['categories'].items()},
        aliases=len(partition['metadata_aliases']),source_map_sha256=partition['source_map_sha256'],
        partition_sha256=partition['partition_sha256'],r12_source_spec=sources.R12_SPEC,mini_source_plan=mini_ref)
    if set(pins)!={p for values in partition['categories'].values()for p in values}:raise RuntimeError('Source partition is incomplete')
    phase('verify_all_text')
    proof=sources.verify_text_sources(partition,guard)
    if not proof['complete_text_verification']:raise RuntimeError('Incomplete full source verification')
    phase('complete')
    result.update(passed=True,proof=proof,failed_path=None)
except BaseException as error:
    result.update(passed=False,error_type=type(error).__name__,error=str(error),
        failed_stage=state['stage'],failed_path=state['current_path'],
        traceback=traceback.format_exc(limit=12))
finally:
    signal.alarm(0)
    now=time.monotonic();phases[state['stage']]=now-stage_started
    result.update(elapsed_seconds=now-START,cpu_seconds=time.process_time()-CPU,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        phase_seconds=phases,observations=state,
        byte_metric_note='verified_*_returned_bytes are complete authenticated source bytes returned; read_open_file_extent_bytes sums regular-file extents at read opens, not physical storage IO.',
        observer_gate_retained=True,stdlib_bootstrap_native_modules=stdlib_bootstrap_native_modules,
        no_unapproved_native_imports_after_hook=not any(x.startswith('import:')for x in state['forbidden_operations']),
        runtime_admitted=False,full_model_qualified=False,new_speedup_claim=False)
    print(json.dumps(result,indent=2,sort_keys=True),flush=True)
if not result['passed']:raise SystemExit(1)
