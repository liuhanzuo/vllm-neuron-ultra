"""Exact SDK helper request routed through captured, recorded child startup.

The caller authenticates the concrete wrapper/startup/import policy and owns
the actual serial slot. This module never launches on import, loads the SDK,
hashes native payloads, or proves whole-group drain. Original helper Python
source and installed library metadata are checked before and after the call.
"""
import copy
import hashlib
import os
from pathlib import Path
import subprocess
import sys

from nemotron4b_process_handshake_v1 import require,scope
from nemotron4b_gated_child_supervisor_v2 import run as supervise
from nemotron4b_worker_receipts_v1 import ReceiptWriter

HELPER='/home/ubuntu/venvs/vllm_neuron/bin/libneuronpjrt-path'
LIBRARY='/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/libneuronxla/libneuronpjrt.so'


def source_hash(path):
    p=Path(path)
    require(p.is_absolute()and p.resolve()==p and not any(x.is_symlink()for x in(p,*p.parents)),
            'Original canonical installed helper source')
    fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        before=os.fstat(fd);require(0<before.st_size<=16384,'Bounded original helper source')
        raw=os.read(fd,16385);after=os.fstat(fd)
        fields=('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')
        require(len(raw)==before.st_size and all(getattr(before,k)==getattr(after,k)
                ==getattr(p.lstat(),k)for k in fields),'Complete unchanged helper source')
        return hashlib.sha256(raw).hexdigest()
    finally:os.close(fd)


def library_identity():
    p=Path(LIBRARY);s=p.stat()
    require(p.resolve()==p and not any(x.is_symlink()for x in(p,*p.parents)),
            'Original installed library metadata path')
    return dict(realpath=str(p),device=s.st_dev,inode=s.st_ino,size=s.st_size,
                mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)


class GatedSDKHelper:
    def __init__(self,policy,*,wrapper_prefix,environment,cwd,nonce,role,
                 expected_cgroup,guard,writer):
        scope(nonce,role)
        require(role in('helper-0','helper-1'),'Original helper for each TP2 rank')
        require(policy['argv']==[HELPER]and policy['library_path']==LIBRARY
                and policy['maximum_calls_per_worker']==1 and policy['timeout_seconds']==20,
                'Original one-call twenty-second discovery helper contract')
        require(type(wrapper_prefix)is list and len(wrapper_prefix)>=4
                and wrapper_prefix[0].startswith('/')and wrapper_prefix[1:3]==['-B','-S']
                and all(type(x)is str and x for x in wrapper_prefix),'Explicit reviewed gated wrapper argv')
        require(type(environment)is dict and all(type(k)is str and type(v)is str for k,v in environment.items())
                and type(cwd)is str and cwd.startswith('/')and callable(guard)
                and type(writer)is ReceiptWriter,'Concrete environment, guard and durable writer')
        self.policy=copy.deepcopy(policy);self.prefix=list(wrapper_prefix);self.environment=dict(environment)
        self.cwd,self.nonce,self.role,self.group=cwd,nonce,role,expected_cgroup
        self.guard,self.writer=guard,writer
        self.calls=0;self.permission=None;self.proof=None;self.supervision=None;self.audit_installed=False

    def audit_subprocess(self,event,args):
        if event=='subprocess.Popen':
            allowed=self.permission
            require(allowed is not None and not allowed['consumed']and len(args)==4
                    and args[0]==allowed['argv'][0]and args[1]==allowed['argv']
                    and args[2]==self.cwd and args[3]==self.environment,
                    'Only exact single gated helper subprocess')
            allowed['consumed']=True
        elif event in('os.system','os.exec','os.posix_spawn','os.fork','os.forkpty'):
            raise PermissionError('Worker cannot launch another external process')

    def run(self,argv,**kwargs):
        require(argv==['libneuronpjrt-path']and kwargs=={'stdout':subprocess.PIPE},
                'Exact original SDK helper request and kwargs')
        require(self.calls==0 and self.permission is None,'Helper budget exhausted or nested')
        self.calls+=1
        if not self.audit_installed:
            # This worker process is dedicated to one phase; audit hooks remain
            # installed until it exits, including when helper initialization fails.
            sys.addaudithook(self.audit_subprocess);self.audit_installed=True
        self.guard()
        require(source_hash(HELPER)==self.policy['helper_sha256'],'Original installed helper source unchanged')
        before=library_identity();require(before==self.policy['library_identity'],'Original installed library identity')
        permission=None
        def command(ready,release):
            nonlocal permission
            require(permission is None,'One exact wrapper command construction')
            actual=self.prefix+['--ready-fd',str(ready),'--release-fd',str(release)]
            permission=dict(argv=actual,consumed=False)
            self.permission=permission
            return actual
        try:
            result=supervise(command,environment=self.environment,cwd=self.cwd,
                nonce=self.nonce,role=self.role,expected_cgroup=self.group,guard=self.guard,
                record_capture=lambda capture:self.writer.write(self.role+'-capture.json',capture),
                handshake_seconds=10,execution_seconds=20,total_seconds=20,
                stdout_cap=4096,stderr_cap=16384)
        finally:self.permission=None
        self.supervision=self.writer.write(self.role+'-supervision.json',result)
        require(permission is not None and permission['consumed'],'Actual one-call spawn audit required')
        require(result['successful']and result['child_reaped']and result['actual_exit_code']==0
                and result['handshake']['capture_record']is not None,'Actual recorded successful helper lifecycle')
        require(result['output']['stdout']==(LIBRARY+'\n').encode()
                and len(result['output']['stderr'])<=16384,'Original exact helper stdout and bounded stderr')
        self.guard()
        require(library_identity()==before and source_hash(HELPER)==self.policy['helper_sha256'],
                'Original helper/library retained after execution')
        self.proof=dict(calls=1,original_helper_argv=[HELPER],actual_wrapper_argv=permission['argv'],
                        capture_record=result['handshake']['capture_record'],supervision=self.supervision,
                        original_source_sha256=self.policy['helper_sha256'],library_identity=before,
                        library_payload_digest_recomputed=False,owned_group_drained_proven=False)
        return subprocess.CompletedProcess(permission['argv'],0,stdout=result['output']['stdout'],
                                           stderr=result['output']['stderr'])
