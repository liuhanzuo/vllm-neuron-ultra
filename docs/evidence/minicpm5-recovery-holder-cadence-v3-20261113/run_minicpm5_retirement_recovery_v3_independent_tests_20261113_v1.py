"""Independent SOURCE review: only six new holder-cadence regressions."""
import ast
import builtins
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

Q=Path(__file__).resolve().parent;sys.path[:0]=[str(Q),str(Q.parent)]
EXPECTED={
    'minicpm5_retained_retirement_recovery_v2.py':'0e4e191cae93111de77fb2570b1d76f372256e2ba7da409bc8b19beb95f7e1e2',
    'minicpm5_retained_retirement_recovery_holders_v2.py':'520b58898dee3a509eccdafa978007a5bfaad92dd13cdb3c2842b4b3a9973297',
    'minicpm5_retained_retirement_recovery_scope_v2.py':'1741a5d483cfe852d03bcd4a436cf8aa2ad052a2fef6ba03af903123fbbc9304',
    'test_minicpm5_retained_retirement_recovery_v2.py':'eb0b85f2be38f7138310fc43511915d9d3b85d6ae2b9fcfdfffeda975a06f419',
    'minicpm5_retained_retirement_recovery_v3.py':'33cb8e3afbbfb9bd09bead35f184705d0ed51d189ac56e8412c2143d419ebdda',
    'minicpm5_retained_retirement_recovery_holders_v3.py':'7435ed76e6fe524127000362cee910419850a452995360ec045a2ac3083c4c84',
    'minicpm5_retained_retirement_recovery_scope_v3.py':'3c212d9f094400c3cd7dea162df4baa26b113db12b7042ef7ccc9d00ce2a415a',
    'test_minicpm5_retained_retirement_recovery_v3.py':'2f02e7583011f5bad1cf155e6caa65281223c5e43bb4703ae4e2d7a86122158a'}
def snapshot():
    out={}
    for name,expected in EXPECTED.items():
        path=Q/name;before=path.stat();raw=path.read_bytes();after=path.stat()
        keys=('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns','st_mode','st_nlink')
        assert tuple(getattr(before,k)for k in keys)==tuple(getattr(after,k)for k in keys)
        digest=hashlib.sha256(raw).hexdigest();assert digest==expected,name
        compile(ast.parse(raw,filename=str(path)),str(path),'exec')
        out[str(path)]=dict(sha256=digest,identity={k:getattr(after,k)for k in keys})
    return out

before=snapshot();blocked=[]
real_import=builtins.__import__
def import_guard(name,*args,**kwargs):
    if name.split('.')[0]in ('torch','torch_neuronx','torch_xla','numpy','neuronx_distributed','neuronx_distributed_inference','neuronxcc','transformers','vllm','vllm_neuron'):
        blocked.append('native:'+name);raise AssertionError('No native imports in independent SOURCE audit')
    return real_import(name,*args,**kwargs)
builtins.__import__=import_guard
def audit(event,args):
    if event=='os.listdir':
        frame=sys._getframe(1)
        while frame is not None:
            if frame.f_code.co_filename=='<frozen importlib._bootstrap_external>':return
            frame=frame.f_back
    if event in ('subprocess.Popen','os.system','os.fork','os.posix_spawn','os.exec','socket.connect','socket.bind','os.listdir','os.scandir','fcntl.flock'):
        blocked.append(event);raise AssertionError('No actual runtime/namespace/network/lock operations')
    if event=='open':
        name,mode,flags=args
        if not isinstance(name,str)or not name.endswith(('.py','.pyc'))or flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):
            blocked.append('open:'+str(name));raise AssertionError('Only read-only Python sources; no runtime metadata/model/raw')
sys.addaudithook(audit)
print(json.dumps(dict(event='independent_source_review_begin',source_inputs=before,early_audit_before_local_import=True)),flush=True)
tests=importlib.import_module('test_minicpm5_retained_retirement_recovery_v3')
suite=unittest.defaultTestLoader.loadTestsFromTestCase(tests.HolderCadenceRegression)
assert suite.countTestCases()==6
result=unittest.TextTestRunner(verbosity=2).run(suite)
after=snapshot();assert before==after
print(json.dumps(dict(event='independent_source_review_end',tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
    passed=result.wasSuccessful(),old25_repeated=False,source_identities_and_hashes_unchanged=before==after,
    actual_runtime_actions_attempted=blocked,actual_history_proc_lock_or_raw_accessed=False)),flush=True)
raise SystemExit(0 if result.wasSuccessful()and not blocked else 1)
