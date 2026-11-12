"""Actual original-descriptor bootstrap implementation; inert on import.

Future reviewed caller only. Tests inject memory-only OS/resource/signal ports.
This module does not import the historical graph, start children/services,
release on uncertainty, or establish the complete historical admission itself.
"""
import copy
import fcntl
import os
from pathlib import Path
import re
import resource
import signal
import stat
import sys
import time
from research import descriptor_serial_lock_proof_v1 as descriptor
from research import minicpm5_sdk_mapping_observer_logic_v1 as logic
import minicpm5_layer0_execution_protocol_v4 as original_proc

require = logic.require
ORIGINAL_LOCK = dict(device=66306, inode=8294421)
ORIGINAL_PYTHON = '/home/ubuntu/venvs/model_reference_2026/bin/python'
ORIGINAL_EXECUTABLE = '/home/ubuntu/.local/share/uv/python/cpython-3.12.16-linux-x86_64-gnu/bin/python3.12'
LOCK_POLICY = dict(schema='minicpm5_native_descriptor_lease_policy_v1', identity=ORIGINAL_LOCK,
    parent_only=True, close_on_exec=True, child_inherits_lock_fd=False,
    no_release_before_owned_service_drained=True,
    descriptor_source=dict(path=str(Path(descriptor.__file__).resolve()),
        sha256='f3faa10356e16dce6b689298ceebf11441f3821ce391f54896cbffed1c82b003'),
    proc_source=dict(path=str(Path(original_proc.__file__).resolve()),
        sha256='81e7d7dd913c6b85f0add7b946692d3270f80cc8f2fe980ecad543251657f185'))


class SystemPorts:
    """Concrete runtime operations, never invoked by source preparation."""
    def lstat(self,path): return Path(path).lstat()
    def stat(self,path): return Path(path).stat()
    def readlink(self,path): return os.readlink(path)
    def open_binary(self,path): return Path(path).open('rb')
    def open_lock(self): return os.open(descriptor.LOCK_PATH,os.O_RDWR|os.O_NOFOLLOW|os.O_CLOEXEC)
    def fstat(self,fd): return os.fstat(fd)
    def inheritable(self,fd): return os.get_inheritable(fd)
    def flock(self,fd): return fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def close(self,fd): return os.close(fd)
    def proc(self,pid): return original_proc.proc(pid)
    def pid(self): return os.getpid()
    def uid(self): return os.getuid()
    def executable(self): return sys.executable,str(Path(sys.executable).resolve()),os.readlink('/proc/self/exe')
    def flags(self): return sys.dont_write_bytecode,__debug__,sys.flags.optimize
    def now(self): return time.monotonic()
    def affinity(self): return set(os.sched_getaffinity(0))
    def set_affinity(self,cpus): return os.sched_setaffinity(0,cpus)
    def nice(self): return os.getpriority(os.PRIO_PROCESS,0)
    def lower_priority(self,amount): return os.nice(amount)
    def get_limit(self,name): return resource.getrlimit(getattr(resource,name))
    def set_limit(self,name,values): return resource.setrlimit(getattr(resource,name),values)
    def get_alarm(self): return signal.getitimer(signal.ITIMER_REAL)
    def set_alarm(self,seconds): return signal.setitimer(signal.ITIMER_REAL,seconds,0)
    def install_alarm_handler(self,handler): return signal.signal(signal.SIGALRM,handler)
    def alarm_handler(self): return signal.getsignal(signal.SIGALRM)


class BootstrapDescriptor:
    """One nonblocking original flock, then one original resource/deadline start.

    A caller-owned strong reference exists before acquire() performs IO. There
    are no child processes in this bootstrap. After ownership is transferred to
    the full original outer, this object cannot close its fd at all.
    """
    def __init__(self, attempt_id, unit, *, ports=None):
        require(type(attempt_id) is str and re.fullmatch('[a-f0-9]{32}',attempt_id)
            and type(unit) is str and re.fullmatch('trn1-mini-[a-z0-9-]{1,150}',unit),
            'Explicit future attempt and original Mini unit label')
        self.ports = SystemPorts() if ports is None else ports
        self.attempt_id,self.unit = attempt_id,unit
        self.fd=self.binding=self.owner=self.deadline=self.started=None
        self.failed=None;self.state='new';self.alarm_callback=None
        self.adopted=False;self.limits_started=False

    def _canonical_lock(self):
        for parent in reversed(Path(descriptor.LOCK_PATH).parents):
            require(stat.S_ISDIR(self.ports.lstat(str(parent)).st_mode),
                    'Original canonical lock ancestry, no symlinks')
        row=self.ports.lstat(descriptor.LOCK_PATH)
        require(stat.S_ISREG(row.st_mode) and row.st_nlink==1
            and dict(device=row.st_dev,inode=row.st_ino)==ORIGINAL_LOCK,
            'Exact original singleton serial lock inode')
        return row

    def _prove(self):
        self._canonical_lock()
        require(not self.ports.inheritable(self.fd),'Original retained fd remains CLOEXEC')
        return descriptor.prove(self.owner,ORIGINAL_LOCK,self.fd,
            proc=self.ports.proc,open_binary=self.ports.open_binary,
            lstat=self.ports.lstat,stat=self.ports.stat,readlink=self.ports.readlink)

    def acquire(self):
        require(self.state=='new' and self.fd is None,'Once-only original bootstrap acquisition')
        self.state='acquiring'
        try:
            # No active outer timer is discarded or replaced by this bootstrap.
            require(self.ports.get_alarm()==(0.0,0.0),'Bootstrap establishes original alarm exactly once')
            before=self._canonical_lock()
            self.fd=self.ports.open_lock()
            require(not self.ports.inheritable(self.fd),'Parent descriptor actually CLOEXEC')
            after=self.ports.fstat(self.fd)
            fields=('st_dev','st_ino','st_mode','st_nlink','st_size','st_mtime_ns','st_ctime_ns')
            require(all(getattr(before,k)==getattr(after,k) for k in fields),'Same original inode across open')
            self.ports.flock(self.fd)
            self.owner=self.ports.proc(self.ports.pid())
            require(self.owner is not None and self.owner['pid']==self.ports.pid()
                and self.ports.uid()==1000,'Actual original outer identity/uid')
            first=self._prove()
            self.binding=dict(outer_owner=copy.deepcopy(self.owner),lock_identity=copy.deepcopy(ORIGINAL_LOCK),
                retained_fd=self.fd,lock_policy=copy.deepcopy(LOCK_POLICY),initial_descriptor_proof=first,
                unit=self.unit,attempt_id=self.attempt_id)
            self.state='held';return copy.deepcopy(self.binding)
        except BaseException as error:
            self.failed=error;self.state='failed'
            # No child/native history has been started. Retain fd for the
            # explicit no-child/drain release path; never hide acquisition state.
            raise

    def impose_original_limits(self):
        require(self.state=='held' and not self.limits_started,'One original limit/deadline initialization')
        self.limits_started=True;self.started=self.ports.now()
        self.deadline=self.started+logic.LIMITS['wall_seconds']
        try:
            limits=logic.LIMITS
            self.ports.set_affinity({limits['cpu']})
            current=self.ports.nice()
            if current<limits['nice']:self.ports.lower_priority(limits['nice']-current)
            for name,values in self._limit_pairs().items():self.ports.set_limit(name,values)
            def alarm(*args):
                if self.failed is None:self.failed=TimeoutError('Original finite PV worker wall bound')
                raise self.failed
            self.alarm_callback=alarm;self.ports.install_alarm_handler(alarm)
            remaining=self.deadline-self.ports.now()
            require(0<remaining<=limits['wall_seconds'],'Prerequisite time stays inside original deadline')
            self.ports.set_alarm(remaining)
            self.state='guarded';self();return self.deadline
        except BaseException as error:
            self.failed=error;self.state='failed';raise

    @staticmethod
    def _limit_pairs():
        limits=logic.LIMITS
        return dict(RLIMIT_AS=(limits['address_space_bytes'],)*2,
            RLIMIT_CPU=(limits['cpu_seconds'],limits['hard_cpu_seconds']),
            RLIMIT_FSIZE=(limits['output_file_bytes'],)*2)

    def __call__(self, **ignored):
        if self.failed is not None:raise self.failed
        try:
            require(self.state in ('guarded','adopted') and self.deadline is not None,
                    'Original descriptor and limits established before authentication')
            now=self.ports.now()
            require(now<self.deadline,'Original absolute outer deadline expired')
            remaining,period=self.ports.get_alarm()
            require(0<remaining<=self.deadline-now<=logic.LIMITS['wall_seconds'] and period==0
                and self.ports.alarm_handler() is self.alarm_callback,
                'Same finite original alarm, never restarted or replaced')
            require(self.ports.affinity()=={logic.LIMITS['cpu']} and self.ports.nice()>=logic.LIMITS['nice'],
                    'Original CPU6 and background priority retained')
            require(all(self.ports.get_limit(name)==values for name,values in self._limit_pairs().items()),
                    'Original AS/CPU/FSIZE limits retained')
            original,resolved,running=self.ports.executable()
            require(original==ORIGINAL_PYTHON and resolved==running==ORIGINAL_EXECUTABLE
                and self.ports.flags()==(True,True,0),'Original outer interpreter/-B/assertions, no substitute')
            require(self.ports.proc(self.owner['pid'])==self.owner,'Actual original owner identity retained')
            return self._prove()
        except BaseException as error:
            self.failed=error;raise

    def bootstrap_inputs(self,source_sha256):
        self()
        return dict(schema='minicpm5_sdk_mapping_history_bootstrap_v1',binding=copy.deepcopy(self.binding),
            limits=copy.deepcopy(logic.LIMITS),artifact_limits=copy.deepcopy(logic.CAPS),
            outer_deadline=self.deadline,source_sha256=copy.deepcopy(source_sha256))

    def adopt(self,original):
        """Bind the existing fd to the unmodified original Lease after gate.load."""
        self();require(not self.adopted,'Only one handoff to the complete original outer')
        require(original.r.ORIGINAL_LOCK==ORIGINAL_LOCK and original.r.LOCK_POLICY==LOCK_POLICY,
                'Exact unchanged original descriptor policy after historical import')
        original.r.validate_initial_descriptor(self.binding)
        lease=original.r.Lease(self.binding)
        bootstrap_owner=self
        class ContinuedPVGuard(original.original.PVGuard):
            def __call__(self,*,force=False):
                bootstrap_owner()
                try:
                    result=super().__call__(force=force)
                    bootstrap_owner();return result
                except BaseException as error:
                    if bootstrap_owner.failed is None:bootstrap_owner.failed=error
                    raise
        guard=ContinuedPVGuard(lease)
        metadata=original.r.Metadata(guard);lease.metadata=metadata
        self.adopted=True;self.state='adopted'
        return lease,guard,metadata

    def release_before_adoption(self,proof):
        require(not self.adopted and self.fd is not None and self.state!='closed',
                'Bootstrap cannot release descriptor adopted by original cleanup')
        require(type(proof) is dict and proof.get('physically_drained') is True
            and proof.get('child_started') is False,'Explicit actual no-child physical drain before release')
        if self.limits_started:self.ports.set_alarm(0)
        self.ports.close(self.fd);self.fd=None;self.state='closed'
