"""Original descriptor bootstrap with the actual no-cache entry's deadline.

No history/native import or operation on import. This immutable successor uses
the same acquisition/proof/adoption/cleanup implementation as v2, but never
starts a fresh 900 seconds after source-entry work has already consumed time.
"""
from research import minicpm5_sdk_mapping_bootstrap_descriptor_v2 as previous
from research import minicpm5_sdk_mapping_bounded_alarm_v1 as timing

require=previous.require


class EntryBootstrapDescriptor(previous.BootstrapDescriptor):
    def __init__(self,entry,attempt_id,unit,*,ports=None):
        entry.check()
        require(entry.role=='outer312' and entry.plan['python']==previous.ORIGINAL_PYTHON
            and entry.plan['executable']==previous.ORIGINAL_EXECUTABLE,
            'Original outer312 entry, never SDK310 descriptor ownership')
        require(type(entry.started) in (int,float) and type(entry.deadline) in (int,float)
            and 0<=entry.started<entry.deadline<float('inf')
            and entry.deadline-entry.started==previous.logic.LIMITS['wall_seconds'],
            'Only original absolute900s entry envelope')
        self.entry=entry
        self.entry_started=entry.started
        self.entry_deadline=entry.deadline
        self.bounded_alarm=None
        super().__init__(attempt_id,unit,ports=ports)

    def impose_original_limits(self):
        require(self.state=='held' and not self.limits_started,
                'Original resource limits initialized once after original descriptor')
        self.limits_started=True
        self.started,self.deadline=self.entry_started,self.entry_deadline
        try:
            self.entry.check()
            require((self.entry.started,self.entry.deadline)==(self.started,self.deadline)
                and self.ports.get_alarm()==(0.0,0.0),
                'Same entry envelope; no prior active alarm discarded')
            require(self.started<=self.ports.now()<self.deadline,
                    'All no-cache source-entry time counts before bootstrap')
            limits=previous.logic.LIMITS
            self.ports.set_affinity({limits['cpu']})
            current=self.ports.nice()
            if current<limits['nice']:
                self.ports.lower_priority(limits['nice']-current)
            for name,values in self._limit_pairs().items():
                self.ports.set_limit(name,values)
            def alarm(*args):
                if self.failed is None:
                    self.failed=TimeoutError('Original finite PV worker wall bound')
                raise self.failed
            self.alarm_callback=alarm
            self.ports.install_alarm_handler(alarm)
            self.bounded_alarm=timing.BoundedAlarm(self.deadline,now=self.ports.now,
                get_timer=self.ports.get_alarm,set_timer=self.ports.set_alarm,
                get_handler=self.ports.alarm_handler,handler=self.alarm_callback)
            self.bounded_alarm.arm_once()
            self.state='guarded'
            self()
            return self.deadline
        except BaseException as error:
            if self.failed is None:self.failed=error
            self.state='failed'
            raise self.failed

    def __call__(self,**ignored):
        if self.failed is not None:raise self.failed
        try:
            self.entry.check()
            require((self.entry.started,self.entry.deadline)==(self.entry_started,self.entry_deadline)
                and (self.started,self.deadline)==(self.entry_started,self.entry_deadline),
                'Entry source fence/deadline ownership unchanged')
            require(self.state in ('guarded','adopted') and self.bounded_alarm is not None
                and self.bounded_alarm.deadline==self.deadline,
                'Original guarded descriptor and first-arm proof retained')
            self.bounded_alarm.check()
            require(self.ports.affinity()=={previous.logic.LIMITS['cpu']}
                and self.ports.nice()>=previous.logic.LIMITS['nice'],
                'Original CPU6 and background priority retained')
            require(all(self.ports.get_limit(name)==values for name,values in self._limit_pairs().items()),
                    'Original AS/CPU/FSIZE limits retained')
            original,resolved,running=self.ports.executable()
            require(original==previous.ORIGINAL_PYTHON and resolved==running==previous.ORIGINAL_EXECUTABLE
                and self.ports.flags()==(True,True,0),'Original outer interpreter/-B/assertions')
            require(self.ports.proc(self.owner['pid'])==self.owner,'Actual original owner identity retained')
            return self._prove()
        except BaseException as error:
            if self.failed is None:self.failed=error
            raise self.failed
