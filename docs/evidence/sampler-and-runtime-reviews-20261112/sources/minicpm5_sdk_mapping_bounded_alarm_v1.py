"""Bracketed original absolute-deadline alarm; no operation on import.

Linux ITIMER_REAL reports timeval microseconds, unlike monotonic nanoseconds.
A fixed small *early* reserve pays for the initial relative-arm syscall. The
proven expiry interval is never permitted beyond the original deadline. Every
later observation intersects that original interval; it never starts a timer.
"""
import math

ARM_RESERVE_SECONDS=0.05
TIMER_QUANTUM_SECONDS=0.000001


def require(ok,message):
    if not ok:raise ValueError(message)


class BoundedAlarm:
    def __init__(self,deadline,*,now,get_timer,set_timer,get_handler,handler):
        require(type(deadline) in (int,float) and 0<deadline<float('inf')
            and all(callable(x) for x in (now,get_timer,set_timer,get_handler,handler)),
            'Exact original deadline and explicit original alarm ports')
        self.deadline=deadline
        self.now,self.get_timer,self.set_timer=now,get_timer,set_timer
        self.get_handler,self.handler=get_handler,handler
        self.state='new';self.failure=None
        self.expiry=None;self.arming=None;self.last_observation=None

    def fail(self,error):
        if self.failure is None:self.failure=error
        raise self.failure

    def arm_once(self):
        if self.failure is not None:raise self.failure
        try:
            require(self.state=='new','Only one original alarm arm, no restart')
            self.state='arming'
            require(self.get_timer()==(0.0,0.0) and self.get_handler() is self.handler,
                    'No old alarm discarded; original installed handler')
            before=self.now()
            require(0<self.deadline-before<=900,'Unchanged remaining original wall bound')
            requested=math.floor((self.deadline-before-ARM_RESERVE_SECONDS)*1_000_000)/1_000_000
            require(requested>0,'Enough original time for one bounded early alarm arm')
            previous=self.set_timer(requested)
            after=self.now()
            require(previous==(0.0,0.0),'Actual first setitimer returned no overwritten timer')
            require(before<=after<self.deadline
                and after-before+TIMER_QUANTUM_SECONDS<=ARM_RESERVE_SECONDS,
                'Actual arm syscall bracket fits early reserve; deadline never expanded')
            low=before+requested-TIMER_QUANTUM_SECONDS
            high=after+requested+TIMER_QUANTUM_SECONDS
            require(low<=high<=self.deadline,'Whole possible first expiry is within original absolute deadline')
            self.expiry=(low,high)
            self.arming=dict(before=before,after=after,requested=requested,
                expiry_interval=[low,high],absolute_deadline=self.deadline,
                early_reserve=ARM_RESERVE_SECONDS,timer_quantum=TIMER_QUANTUM_SECONDS)
            self.state='armed'
            self.check()
            return dict(self.arming)
        except BaseException as error:self.fail(error)

    def check(self):
        if self.failure is not None:raise self.failure
        try:
            require(self.state=='armed' and self.expiry is not None,'Original one-shot arm proof exists')
            before=self.now()
            remaining,period=self.get_timer()
            after=self.now()
            require(type(remaining) in (int,float) and 0<remaining<float('inf') and period==0
                and before<=after<self.deadline and self.get_handler() is self.handler,
                'Original finite timer/handler and absolute deadline remain active')
            low=before+remaining-TIMER_QUANTUM_SECONDS
            high=after+remaining+TIMER_QUANTUM_SECONDS
            intersection=(max(self.expiry[0],low),min(self.expiry[1],high))
            require(intersection[0]<=intersection[1]<=self.deadline,
                    'Bracketed timer expiry matches original arm; no timer restart or extension')
            self.expiry=intersection
            self.last_observation=dict(before=before,after=after,remaining=remaining,
                possible_expiry=[low,high],original_expiry_intersection=list(intersection))
            return dict(self.last_observation)
        except BaseException as error:self.fail(error)
