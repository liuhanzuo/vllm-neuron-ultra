"""Bracketed monotonic/timeval tests. All timer/descriptor/process ports fake."""
import math
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research import minicpm5_sdk_mapping_bounded_alarm_v1 as a
from research import minicpm5_sdk_mapping_bootstrap_descriptor_v4 as outer
from research import minicpm5_sdk_mapping_worker_guard_v2 as sdk
from research import check_minicpm5_sdk_mapping_bootstrap_descriptor_v2 as outer_f
from research import check_minicpm5_sdk_mapping_worker_guard_v1 as sdk_f


class Timer:
    def __init__(self,step=0.,arm_before=0.,arm_after=0.,get_cost=0.,quantize=False):
        self.t=100.;self.step=step;self.arm_before=arm_before;self.arm_after=arm_after
        self.get_cost=get_cost;self.quantize=quantize
        self.expiry=None;self.period=0.;self.calls=[];self.callback=lambda:None
    def now(self):
        result=self.t;self.t+=self.step;return result
    def remaining(self):
        value=max(0.,self.expiry-self.t) if self.expiry is not None else 0.
        if self.quantize:value=math.ceil(value*1e6)/1e6
        return (value,self.period)
    def get(self):
        self.t+=self.get_cost/2;result=self.remaining();self.t+=self.get_cost/2
        return result
    def set(self,seconds):
        old=self.remaining();self.t+=self.arm_before
        self.expiry=self.t+seconds if seconds else None
        self.t+=self.arm_after;self.calls.append(seconds);return old
    def handler(self):return self.callback


def timer_fixture(**kwargs):
    t=Timer(**kwargs)
    alarm=a.BoundedAlarm(750.,now=t.now,get_timer=t.get,set_timer=t.set,
        get_handler=t.handler,handler=t.callback)
    return alarm,t


def wire_ports(p,t,worker=False):
    p.now=t.now
    if worker:
        p.alarm=t.get;p.arm=t.set;p.handler=t.handler
        p.install=lambda handler:setattr(t,'callback',handler)
    else:
        p.get_alarm=t.get;p.set_alarm=t.set;p.alarm_handler=t.handler
        p.install_alarm_handler=lambda handler:setattr(t,'callback',handler)


class Checks(unittest.TestCase):
    def test_natural_microsecond_clock_progress_preserves_timer(self):
        alarm,t=timer_fixture(step=.000001,quantize=True)
        alarm.arm_once()
        for _ in range(100):alarm.check()
        self.assertEqual(len(t.calls),1)
        self.assertLessEqual(alarm.expiry[1],alarm.deadline)

    def test_getitimer_and_arming_syscall_cost_brackets(self):
        for before,after,get in [(.000003,.000004,.000009),(.008,.012,.005),(.02,.025,.01)]:
            alarm,t=timer_fixture(step=.000001,arm_before=before,arm_after=after,get_cost=get,quantize=True)
            receipt=alarm.arm_once()
            for _ in range(10):alarm.check()
            self.assertLessEqual(receipt['expiry_interval'][1],750.)
            self.assertLess(t.expiry,750.)
            self.assertLessEqual(alarm.expiry[0],t.expiry)
            self.assertGreaterEqual(alarm.expiry[1],t.expiry)

    def test_actual_arming_overrun_rejected_without_expanding_deadline(self):
        alarm,t=timer_fixture(arm_before=.06)
        with self.assertRaisesRegex(ValueError,'arm syscall'):alarm.arm_once()
        self.assertEqual(alarm.deadline,750.)
        with self.assertRaises(ValueError):alarm.check()
        self.assertEqual(len(t.calls),1)

    def test_long_getitimer_observation_keeps_initial_expiry_upper_bound(self):
        alarm,t=timer_fixture();alarm.arm_once();original=alarm.expiry
        t.get_cost=2.
        alarm.check()
        self.assertLessEqual(alarm.expiry[1],original[1])
        self.assertGreaterEqual(alarm.expiry[0],original[0])

    def test_same_handler_full900_restart_rejected(self):
        alarm,t=timer_fixture(step=.000001);alarm.arm_once();t.t+=1;t.set(900.)
        with self.assertRaisesRegex(ValueError,'restart'):alarm.check()

    def test_restart_previous_remaining_after_elapsed_rejected(self):
        alarm,t=timer_fixture();alarm.arm_once();old=t.remaining()[0];t.t+=1;t.set(old)
        with self.assertRaisesRegex(ValueError,'restart'):alarm.check()

    def test_reset_to_original_absolute_deadline_rejected_even_if_under900(self):
        alarm,t=timer_fixture();alarm.arm_once();t.t+=5;t.set(750.-t.t)
        with self.assertRaisesRegex(ValueError,'restart'):alarm.check()

    def test_failure_sticky_after_timer_restoration(self):
        alarm,t=timer_fixture();alarm.arm_once();original=t.expiry;t.set(900.)
        with self.assertRaises(ValueError) as first:alarm.check()
        t.expiry=original
        with self.assertRaises(ValueError) as second:alarm.check()
        self.assertIs(first.exception,second.exception)

    def test_period_handler_cancel_and_elapsed_deadline_rejected(self):
        for fault in ('period','handler','cancel','deadline'):
            alarm,t=timer_fixture();alarm.arm_once()
            if fault=='period':t.period=1.
            if fault=='handler':t.callback=lambda:None
            if fault=='cancel':t.expiry=None
            if fault=='deadline':t.t=751.
            with self.assertRaises(ValueError):alarm.check()

    def test_existing_timer_cannot_be_overwritten_or_rearmed(self):
        alarm,t=timer_fixture();t.expiry=t.t+10.
        with self.assertRaises(ValueError):alarm.arm_once()
        self.assertEqual(t.calls,[])
        alarm,t=timer_fixture();alarm.arm_once()
        with self.assertRaises(ValueError):alarm.arm_once()
        self.assertEqual(len(t.calls),1)

    def test_sdk_guard_with_one_microsecond_and_realistic_syscall_costs(self):
        for cost in (0.,.004):
            predecessor,p,e,_=sdk_f.fixture(False)
            t=Timer(step=.000001,arm_before=cost,arm_after=cost,get_cost=cost,quantize=True)
            wire_ports(p,t,True)
            guard=sdk.SDKEntryGuard(e,predecessor.admission,ports=p)
            guard.start()
            for _ in range(5):guard()
            self.assertLessEqual(guard.bounded_alarm.expiry[1],e.deadline)
            self.assertEqual(len(t.calls),1)

    def test_sdk_guard_restart_sticky_and_original_parent_lease_preserved(self):
        predecessor,p,e,_=sdk_f.fixture(False);t=Timer(step=.000001)
        wire_ports(p,t,True);guard=sdk.SDKEntryGuard(e,predecessor.admission,ports=p);guard.start()
        self.assertTrue(guard()['passed']);t.t+=1;t.set(900.)
        with self.assertRaises(ValueError) as first:guard()
        with self.assertRaises(ValueError) as second:guard()
        self.assertIs(first.exception,second.exception)
        self.assertEqual(p.closed,[])

    def test_outer_descriptor_first_arm_syscall_elapsed_and_progress(self):
        for cost in (0.,.004):
            p=outer_f.Ports();t=Timer(step=.000001,arm_before=cost,arm_after=cost,get_cost=cost,quantize=True)
            wire_ports(p,t)
            class Entry:
                role='outer312';started=40.;deadline=940.
                plan=dict(python=outer.previous.ORIGINAL_PYTHON,executable=outer.previous.ORIGINAL_EXECUTABLE)
                def check(self):
                    if t.t>=self.deadline:raise TimeoutError('fake deadline')
            e=Entry();guard=outer.EntryBootstrapDescriptor(e,'a'*32,'trn1-mini-bracket-test',ports=p)
            guard.acquire();guard.impose_original_limits()
            for _ in range(5):guard()
            self.assertEqual(guard.deadline,940.)
            self.assertLessEqual(guard.bounded_alarm.expiry[1],940.)
            self.assertEqual(len(t.calls),1)
            self.assertEqual(p.closed,[])

    def test_no_native_imports(self):
        for name in ('ctypes','_ctypes','torch','numpy'):
            self.assertNotIn(name,sys.modules)


if __name__=='__main__':unittest.main()
