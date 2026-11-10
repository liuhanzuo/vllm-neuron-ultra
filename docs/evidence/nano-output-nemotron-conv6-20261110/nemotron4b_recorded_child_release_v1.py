"""Actual child capture must be durably recorded before a parent grant.

Successor parent protocol for native workers and SDK helpers. Keeps the exact
child_wait wire format from handshake v1; no service admission or lock release.
"""
import copy
import hashlib
import os
from pathlib import Path
import re
import time

from nemotron4b_process_handshake_v1 import encode,owner,receive,require,scope,send


def parent_release(pid,ready_fd,release_fd,*,nonce,role,expected_argv,
                   expected_cgroup,guard,record_capture,deadline):
    scope(nonce,role)
    require(ready_fd!=release_fd and callable(guard)and callable(record_capture),
            'Distinct pipes, actual guard and durable capture writer')
    try:
        guard()
        ready=receive(ready_fd,deadline)
        require(set(ready)=={'schema','nonce','role','owner','child_ready_ns'}
                and ready['schema']=='nemotron4b_process_ready_v1'
                and ready['nonce']==nonce and ready['role']==role,'Exact ready scope')
        first=owner(pid);captured=time.monotonic_ns();parent=owner(os.getpid())
        require(first==ready['owner']and first['pid']==pid and first['ppid']==os.getpid()
                and first['argv']==expected_argv and first['cgroup']==expected_cgroup,
                'Actual exact child identity before capture record')
        require(type(ready['child_ready_ns'])is int and 0<ready['child_ready_ns']<=captured,
                'Ready precedes actual capture')
        guard();second=owner(pid)
        require(second==first and owner(os.getpid())==parent,'Stable original parent and child')
        capture=dict(schema='nemotron4b_durable_child_capture_v1',nonce=nonce,role=role,
                     parent=parent,ready=ready,first_actual_capture=first,
                     second_actual_capture=second,capture_ns=captured)
        reference=record_capture(copy.deepcopy(capture))
        require(type(reference)is dict and set(reference)=={'path','sha256'}
                and type(reference['path'])is str and Path(reference['path']).is_absolute()
                and type(reference['sha256'])is str and re.fullmatch('[0-9a-f]{64}',reference['sha256']),
                'Durable exact capture reference before child release')
        # The actual writer is supplied by the reviewed root/worker. Its file
        # durability and exact bytes require the concrete receipt writer.
        recorded=time.monotonic_ns()
        guard();require(owner(pid)==second and owner(os.getpid())==parent,
                        'Actual identities unchanged after durable capture')
        guarded=time.monotonic_ns()
        grant=dict(schema='nemotron4b_process_release_v1',nonce=nonce,role=role,
            owner=second,parent=parent,ready_sha256=hashlib.sha256(encode(ready)).hexdigest(),
            capture_ns=captured,guard_complete_ns=guarded,grant_ns=time.monotonic_ns())
        send(release_fd,grant,deadline)
        return dict(ready=ready,first_actual_capture=first,second_actual_capture=second,
                    grant=grant,capture_record=copy.deepcopy(reference),
                    capture_recorded_ns=recorded,child_release_observed=False,
                    native_or_helper_executed=False,runtime_admitted=False)
    finally:
        os.close(ready_fd);os.close(release_fd)
