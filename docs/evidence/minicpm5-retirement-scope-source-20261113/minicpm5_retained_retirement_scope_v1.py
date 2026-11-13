"""Retain the original serial owner until its foreground inspector scope drains.

No process is killed, signalled or detached on timeout. The helper authenticates
its complete helper/sudo-monitor/sudo/owner ancestry and cannot spawn children.
Missing or invalid ancestry, an interrupted spawn and signal-killed sudo remain
pending; neither a direct-child exit nor an exception is a drain certificate.
Import is inert. All process observations and waits occur only in explicit calls.
"""
import hashlib
import errno
import json
import re
import secrets
import subprocess
import time

import minicpm5_retained_retirement_contract_v2 as c
from minicpm5_layer1_drain_hold_v1 import retain_until_drained

ACK_SCHEMA='minicpm5_retained_retirement_helper_scope_start_v1'
PROOF_SCHEMA='minicpm5_retained_retirement_helper_scope_drain_v1'
DEADLINE=90
SPAWN_EVENTS=frozenset(('subprocess.Popen','os.system','os.fork','os.forkpty',
    'os.posix_spawn','os.exec','pty.spawn'))


def forbid_helper_children(event,args):
    if event in SPAWN_EVENTS:
        raise RuntimeError('Retirement inspector cannot create or replace a process: '+event)


def deadline_expired(*_):
    raise TimeoutError('Finite 90-second privileged holder inspection deadline')


def command(permit,nonce):
    c.require(type(nonce)is str and re.fullmatch('[0-9a-f]{64}',nonce),'Exact fresh helper scope nonce')
    return ['sudo','-n','/usr/bin/python3','-B',c.SOURCES[2],'--permit',permit['path'],
        '--permit-sha256',permit['sha256'],'--scope-nonce',nonce]


def identity(value):
    c.require(type(value)is dict and set(value)=={'pid','start_ticks','ppid','argv','cgroup'}
        and type(value['pid'])is int and value['pid']>0 and type(value['ppid'])is int
        and type(value['start_ticks'])is str and value['start_ticks'].isdigit()
        and type(value['argv'])is list and all(type(x)is str for x in value['argv'])
        and type(value['cgroup'])is str and len(c.canonical(value))<=65536,'Bounded complete actual process identity')
    return value


def validate_start(value,permit,nonce,owner,direct_pid):
    c.require(type(value)is dict and set(value)=={'schema','permit','nonce','chain','helper_spawning_forbidden'}
        and value['schema']==ACK_SCHEMA and value['permit']==permit and value['nonce']==nonce
        and value['helper_spawning_forbidden']is True,'Exact nonce-bound helper scope acknowledgement')
    chain=value['chain'];c.require(type(chain)is list and 2<=len(chain)<=16,'Complete bounded helper-to-owner chain')
    for row in chain:identity(row)
    c.require(chain[-1]==owner and chain[-2]['pid']==direct_pid
        and len({r['pid']for r in chain})==len(chain),'Exact direct sudo and retained original owner')
    c.require(all(a['ppid']==b['pid']for a,b in zip(chain,chain[1:])),'Unbroken observed parent chain')
    argv=command(permit,nonce)
    c.require(chain[0]['argv']==argv[2:] and all(r['argv']==argv for r in chain[1:-1]),
        'Only exact helper and foreground sudo/monitor ancestors belong to this scope')
    return chain[:-1]


def start_ack(io,permit_ref,permit,nonce,pid):
    """Called by the read-only helper before plan IO and the two holder passes."""
    chain=[];current=pid
    for _ in range(16):
        row=io.process_api().proc(current);identity(row);chain.append(row)
        if row==permit['owner']:break
        current=row['ppid']
    c.require(chain[-1]==permit['owner'],'Helper ancestry reaches exact retained owner')
    value=dict(schema=ACK_SCHEMA,permit=permit_ref,nonce=nonce,chain=chain,helper_spawning_forbidden=True)
    validate_start(value,permit_ref,nonce,permit['owner'],chain[-2]['pid'])
    # No sampled PID may have been reused or reparented during acknowledgement.
    for row in chain:c.require(io.process_api().proc(row['pid'])==row,'Stable actual inspector ancestry')
    io.descriptor(permit['owner'],permit['retained_fd'])
    return value


class ForegroundScopes:
    """One active inspector at a time, bounded rolling evidence for completed calls."""
    def __init__(self,io,lock):
        self.io=io;self.lock=lock;self.active=None;self.count=0
        self.digest='0'*64;self.last=None

    def summary(self):
        c.require(self.active is None,'Every registered inspector scope must drain first')
        return dict(schema=PROOF_SCHEMA,physically_drained=True,pending_scope=False,
            original_lock_identity=c.LOCK,owner=self.lock.owner,completed_scopes=self.count,
            scope_chain_sha256=self.digest,last_scope=self.last,unrelated_processes_signalled=False)

    def _collect(self):
        row=self.active
        if row is None or row['spawn_failed'] or row['output']is not None:return
        c.require(row['process']is not None,'Spawn was interrupted; complete owned scope is unknown')
        # communicate(timeout) never kills Popen. Repeated calls retain its pipes
        # and the complete output, unlike subprocess.run's timeout cleanup.
        row['output']=row['process'].communicate(timeout=1)

    def _verify(self):
        self.lock.check();row=self.active
        if row is None:return self.summary()
        if row['spawn_failed']:
            detail=dict(nonce=row['nonce'],permit=row['permit'],spawn_failed_before_exec=True,
                physically_drained=True,owned_processes=[])
        else:
            p=row['process'];c.require(p is not None and row['output']is not None
                and p.returncode is not None,'Foreground helper output and actual direct-child wait are incomplete')
            c.require(p.returncode>=0,'Signal-killed sudo cannot prove the complete foreground scope')
            out,err=row['output']
            c.require(len(out)<=1<<20 and len(err)<=65536,'Bounded complete helper transcript')
            lines=out.splitlines();c.require(1<=len(lines)<=2,'Helper scope start ACK is required before any result')
            ack=json.loads(lines[0]);chain=validate_start(ack,row['permit'],row['nonce'],self.lock.owner,p.pid)
            for owner in chain:self.io.process_api().gone(owner)
            self.lock.check()
            detail=dict(nonce=row['nonce'],permit=row['permit'],spawn_failed_before_exec=False,
                physically_drained=True,owned_processes=chain,actual_sudo_exit=p.returncode,
                stdout_sha256=hashlib.sha256(out).hexdigest(),stderr_sha256=hashlib.sha256(err).hexdigest())
        # Commit a constant-size summary only after every actual identity is gone.
        digest=hashlib.sha256(bytes.fromhex(self.digest)+c.canonical(detail)).hexdigest()
        self.digest=digest;self.count+=1;self.last=detail;self.active=None
        return self.summary()

    def drain(self):
        return retain_until_drained(self._collect,self._verify,
            lambda value:print(json.dumps(dict(event='mini_retirement_helper_scope_drain_pending',
                nonce=None if self.active is None else self.active['nonce'],**value)),flush=True),
            lambda:time.sleep(.25))

    def invoke(self,permit):
        c.require(self.active is None,'Previous inspector scope remains owned')
        self.lock.check();nonce=secrets.token_hex(32);argv=command(permit,nonce)
        row=dict(permit=permit,nonce=nonce,process=None,output=None,spawn_failed=False)
        # Registration happens before Popen: an interrupted constructor cannot
        # silently lose an already created process or unwind the serial owner.
        self.active=row;failure=None
        try:
            try:
                row['process']=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,close_fds=True,start_new_session=True)
            except OSError as error:
                # With shell=False, Popen reports pre-exec failures only after
                # reaping the failed child; sudo/helper never executed.
                row['spawn_failed']=(getattr(error,'filename',None)==argv[0]
                    and error.errno in (errno.ENOENT,errno.EACCES,errno.ENOEXEC,errno.ENOTDIR,errno.ETXTBSY))
                raise
            row['output']=row['process'].communicate(timeout=DEADLINE)
        except BaseException as error:failure=error
        proof=self.drain()
        if failure is not None:raise failure
        out,err=row['output'];p=row['process']
        if p.returncode:raise subprocess.CalledProcessError(p.returncode,argv,output=out,stderr=err)
        lines=out.splitlines();c.require(len(lines)==2,'Successful inspector must supply its scope ACK and inspection')
        return json.loads(lines[1]),proof
