"""Actual SDK child pre-history parent lease/release guard; inert on import."""
import copy
import json
import os
import resource
import select
import signal
import stat
import sys
import time
from research import descriptor_serial_lock_proof_v1 as proof
from research import minicpm5_sdk_mapping_observer_logic_v1 as logic
from research import minicpm5_sdk_mapping_bounded_alarm_v1 as timing
from research import minicpm5_sdk_mapping_bootstrap_descriptor_v2 as parent
import minicpm5_layer0_execution_protocol_v4 as process

require=logic.require
PIPE_CAP=16*2**20


class Ports:
    def now(self):return time.monotonic()
    def pid(self):return os.getpid()
    def uid(self):return os.getuid()
    def proc(self,pid):return process.proc(pid)
    def lstat(self,path):return os.lstat(path)
    def stat(self,path):return os.stat(path)
    def readlink(self,path):return os.readlink(path)
    def open_binary(self,path):return open(path,'rb')
    def alarm(self):return signal.getitimer(signal.ITIMER_REAL)
    def handler(self):return signal.getsignal(signal.SIGALRM)
    def install(self,handler):return signal.signal(signal.SIGALRM,handler)
    def arm(self,seconds):return signal.setitimer(signal.ITIMER_REAL,seconds,0)
    def affinity(self):return set(os.sched_getaffinity(0))
    def set_affinity(self,value):return os.sched_setaffinity(0,value)
    def nice(self):return os.getpriority(os.PRIO_PROCESS,0)
    def lower(self,amount):return os.nice(amount)
    def limit(self,name):return resource.getrlimit(getattr(resource,name))
    def fstat(self,fd):return os.fstat(fd)
    def select(self,fd,seconds):return bool(select.select([fd],[],[],seconds)[0])
    def read(self,fd,size):return os.read(fd,size)


class SDKEntryGuard:
    def __init__(self,entry,admission,*,ports=None):
        entry.check()
        require(entry.role=='sdk310' and entry.admission_envelope==admission,
                'Actual SDK source entry owns exact immutable parent admission')
        self.entry,self.admission=entry,copy.deepcopy(admission)
        self.binding=copy.deepcopy(admission['lease'])
        self.ports=Ports() if ports is None else ports
        self.owner=self.release=None
        self.failure=self.alarm_callback=None
        self.bounded_alarm=None
        self.state='new'
        require(self.binding['lock_identity']==parent.ORIGINAL_LOCK
            and self.binding['lock_policy']==parent.LOCK_POLICY
            and admission['source_entry_binding']['outer_deadline']==entry.deadline,
            'Original parent-only CLOEXEC descriptor and inherited absolute outer deadline')

    def limits(self):return parent.BootstrapDescriptor._limit_pairs()

    def start(self):
        require(self.state=='new','Once-only actual SDK limits/release setup')
        self.state='starting'
        try:
            self.entry.check()
            require(self.ports.alarm()==(0.0,0.0),'No existing child timer discarded')
            # RLIMITs are inherited across exec from the original outer; SDK
            # may not reset CPU accounting or expand any limit.
            require(all(self.ports.limit(name)==values for name,values in self.limits().items()),
                    'Exact inherited original AS2GiB/CPU600+605/FSIZE2MiB')
            self.ports.set_affinity({logic.LIMITS['cpu']})
            priority=self.ports.nice()
            if priority<logic.LIMITS['nice']:self.ports.lower(logic.LIMITS['nice']-priority)
            self.owner=self.ports.proc(self.ports.pid())
            outer=self.binding['outer_owner']
            require(self.ports.uid()==1000 and self.owner is not None
                and self.owner['pid']==self.ports.pid() and self.owner['ppid']==outer['pid']
                and self.owner['cgroup']==outer['cgroup']
                and self.owner['argv']==self.entry.runtime.orig_argv,
                'Actual source-entry SDK PID/start/parent/cgroup/argv')
            def alarm(*args):
                if self.failure is None:self.failure=TimeoutError('Original inherited SDK wall deadline')
                raise self.failure
            self.alarm_callback=alarm
            self.ports.install(alarm)
            self.bounded_alarm=timing.BoundedAlarm(self.entry.deadline,now=self.ports.now,
                get_timer=self.ports.alarm,set_timer=self.ports.arm,
                get_handler=self.ports.handler,handler=self.alarm_callback)
            self.bounded_alarm.arm_once()
            self.state='guarded';self()
            return copy.deepcopy(self.owner)
        except BaseException as error:
            if self.failure is None:self.failure=error
            raise self.failure

    def __call__(self,**ignored):
        if self.failure is not None:raise self.failure
        try:
            self.entry.check()
            require(self.state in ('guarded','released','adopted'),'Actual SDK guard started')
            require(self.bounded_alarm is not None and self.bounded_alarm.deadline==self.entry.deadline,
                    'Same original inherited first-arm proof')
            self.bounded_alarm.check()
            require(self.ports.affinity()=={logic.LIMITS['cpu']} and self.ports.nice()>=logic.LIMITS['nice']
                and all(self.ports.limit(n)==v for n,v in self.limits().items()),
                'All original child limits remain exact')
            require(self.ports.proc(self.owner['pid'])==self.owner,
                    'Same actual acknowledged child owner')
            return proof.prove(self.binding['outer_owner'],parent.ORIGINAL_LOCK,self.binding['retained_fd'],
                proc=self.ports.proc,open_binary=self.ports.open_binary,lstat=self.ports.lstat,
                stat=self.ports.stat,readlink=self.ports.readlink)
        except BaseException as error:
            if self.failure is None:self.failure=error
            raise self.failure

    def read_release(self):
        require(self.state=='guarded' and self.release is None,'Single bounded original parent release')
        try:
            require(stat.S_ISFIFO(self.ports.fstat(0).st_mode),'Parent-only actual stdin FIFO')
            until=min(self.entry.deadline,self.ports.now()+60)
            parts=[];size=0
            while True:
                self();remaining=until-self.ports.now()
                require(remaining>0,'Original finite parent release timeout')
                if not self.ports.select(0,min(.1,remaining)):continue
                raw=self.ports.read(0,min(65536,PIPE_CAP-size+1))
                if not raw:break
                size+=len(raw);require(size<=PIPE_CAP,'Whole original release input cap');parts.append(raw)
            raw=b''.join(parts)
            packet=json.loads(raw)
            require((json.dumps(packet,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()==raw,
                    'Exactly one canonical original parent packet and EOF')
            require(type(packet) is dict and set(packet)=={'schema','admission','phase','owner','source_plan','own_gate'}
                and packet['schema']=='minicpm5_fixed_input_pv_sdk_import_diagnostic_release_v1'
                and packet['admission']==self.entry.plan_reference and packet['phase']==1
                and type(packet['phase']) is int and packet['owner']==self.owner and packet['own_gate'] is None,
                'Original actual acknowledged SDK release fields unchanged')
            self.release=packet;self.state='released';self()
            return copy.deepcopy(packet)
        except BaseException as error:
            if self.failure is None:self.failure=error
            raise self.failure

    def adopt(self,original):
        self();require(self.state=='released','Original child lease adopted only after actual release')
        original.r.validate_initial_descriptor(self.binding)
        lease=original.r.Lease(self.binding)
        bootstrap=self
        def guard():
            bootstrap()
            result=lease.check('pv.child')
            bootstrap()
            return result
        metadata=original.r.Metadata(guard)
        lease.metadata=metadata
        lease.processes=[self.binding['outer_owner'],self.owner]
        self.state='adopted'
        return lease,guard,metadata
