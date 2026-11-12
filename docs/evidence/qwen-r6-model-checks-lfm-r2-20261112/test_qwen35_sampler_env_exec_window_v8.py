"""Exact env->Python interleavings and virtual command costs; no real runtime."""
import copy
from contextlib import contextmanager
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

Q=Path(__file__).resolve().parent
sys.path[:0]=[str(Q),str(Q.parent)]
import qwen35_sampler_runtime_adapter_v5 as a5
import qwen35_sampler_runtime_service_v5 as s5
import qwen35_sampler_runtime_adapter_v8 as a6
import qwen35_sampler_runtime_service_v8 as s6
import qwen35_sampler_runtime_contract_v8 as c
from test_qwen35_sampler_active_scope_v8 import World


class VirtualClock:
    def __init__(self):self.now=0.;self.commands=0
    def command_elapsed(self):self.commands+=1;self.now+=.05


class ExecWindowTests(unittest.TestCase):
    def world(self, adapter=a6, service=s6):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        return World(tmp.name,adapter,service)

    def interleaving(self, w, alter=None):
        cold=w.phase('cold');pid=cold.owner['process']['pid'];original=copy.deepcopy(w.procs[pid])
        command=w.s.command(cold.unit,cold.argv,cold.env,cold.props,cold.log,cold.parent)
        wrapper=command[command.index('/usr/bin/env'):]
        w.procs[pid]['argv']=wrapper;calls=[]
        def proc(current_pid):
            if current_pid==pid:
                calls.append(current_pid)
                if len(calls)==3:w.procs[pid]['argv']=cold.argv
                if alter is not None:alter(w,cold,len(calls),wrapper)
            return copy.deepcopy(w.procs.get(current_pid))
        w.p.proc=proc
        return cold,original,calls

    def test01_frozen_v5_rejects_exact_env_to_Python_between_real_calls(self):
        w=self.world(a5,s5);cold,original,calls=self.interleaving(w)
        monitor=a5.ActiveScopeMonitor(w.spec,w.admission,w.co,lambda:None)
        with w.patched(),self.assertRaisesRegex(ValueError,'active process identity changed'):monitor.observe()
        self.assertEqual(len(calls),3)
        self.assertEqual(w.procs[original['pid']],original)
        self.assertFalse((cold.log.parent/'supervisor-owner.json').exists())

    def test02_v8_reauthenticates_exact_one_way_exec_with_same_identity(self):
        w=self.world();cold,original,calls=self.interleaving(w);monitor=w.monitor()
        with w.patched():monitor.observe()
        self.assertEqual(monitor.retained[cold.unit].owner['process'],original)
        self.assertEqual(monitor.observation['attempts'],1)
        self.assertEqual(len(calls),6)

    def test03_unreviewed_argv_is_not_an_exec_transition(self):
        def alter(w,cold,n,wrapper):
            if n==3:w.procs[cold.owner['process']['pid']]['argv']=['/unreviewed']
        w=self.world();_,_,calls=self.interleaving(w,alter)
        with w.patched(),self.assertRaisesRegex(ValueError,'active process identity changed'):w.monitor().observe()
        self.assertEqual(len(calls),3)

    def test04_same_pid_different_start_ticks_is_not_an_exec_transition(self):
        def alter(w,cold,n,wrapper):
            if n==3:w.procs[cold.owner['process']['pid']]['start_ticks']='999999'
        w=self.world();_,_,calls=self.interleaving(w,alter)
        with w.patched(),self.assertRaisesRegex(ValueError,'active process identity changed'):w.monitor().observe()
        self.assertEqual(len(calls),3)

    def test05_invocation_change_during_exec_reauthentication_is_rejected(self):
        def alter(w,cold,n,wrapper):
            if n==3:w.props[cold.unit]['InvocationID']='f'*32
        w=self.world();_,_,calls=self.interleaving(w,alter)
        with w.patched(),self.assertRaisesRegex(ValueError,'invocation changed'):w.monitor().observe()
        self.assertEqual(len(calls),3)

    def test06_reverting_to_env_does_not_trigger_unbounded_retry(self):
        def alter(w,cold,n,wrapper):
            if n==4:w.procs[cold.owner['process']['pid']]['argv']=wrapper
        w=self.world();_,_,calls=self.interleaving(w,alter)
        with w.patched(),self.assertRaisesRegex(ValueError,'did not converge'):w.monitor().observe()
        self.assertEqual(len(calls),4)

    def test07_full_properties_rechecked_during_exec_transition(self):
        def alter(w,cold,n,wrapper):
            if n==3:w.props[cold.unit]['MemoryMax']='changed'
        w=self.world();_,_,calls=self.interleaving(w,alter)
        with w.patched(),self.assertRaisesRegex(ValueError,'resource changed: MemoryMax'):w.monitor().observe()
        self.assertEqual(len(calls),3)

    def test08_process_swap_after_reauthentication_is_rejected(self):
        def alter(w,cold,n,wrapper):
            if n==5:w.procs[cold.owner['process']['pid']]['start_ticks']='999999'
        w=self.world();_,_,calls=self.interleaving(w,alter)
        with w.patched(),self.assertRaisesRegex(ValueError,'did not converge'):w.monitor().observe()
        self.assertEqual(len(calls),5)

    def slow_world(self, adapter=a6, service=s6):
        w=self.world(adapter,service)
        for arm,name in c.PHASES:w.stop(w.phase(name,arm=arm,ack=True))
        clock=VirtualClock();properties=w.properties;command=w.command
        def slow_properties(unit):clock.command_elapsed();return properties(unit)
        def slow_command(argv,**kwargs):clock.command_elapsed();return command(argv,**kwargs)
        w.properties=slow_properties;w.command=slow_command;w.p.command=slow_command
        return w,clock

    def test09_frozen_v5_five_seconds_rejects_stable_virtual_slow_scan(self):
        w,clock=self.slow_world(a5,s5)
        monitor=a5.ActiveScopeMonitor(w.spec,w.admission,w.co,lambda:None)
        with w.patched(),patch.object(a5.time,'monotonic',side_effect=lambda:clock.now):
            with self.assertRaisesRegex(ValueError,'transition window expired'):monitor.observe()
        self.assertEqual(clock.commands,124)
        self.assertAlmostEqual(clock.now,6.2)

    def test10_v8_accepts_same_stable_slow_scan_with_original_guard_live(self):
        w,clock=self.slow_world();monitor=w.monitor()
        with w.patched(),patch.object(a6.time,'monotonic',side_effect=lambda:clock.now):monitor.observe()
        self.assertEqual(clock.commands,124);self.assertAlmostEqual(clock.now,6.2)
        self.assertEqual(monitor.observation['attempts'],1)

    def test11_original_outer_deadline_still_stops_slow_scan(self):
        w,clock=self.slow_world();clock.now=c.LIMITS['outer_wall_seconds']-3
        def original_guard():
            c.require(clock.now<c.LIMITS['outer_wall_seconds'],'original outer deadline expired')
        monitor=a6.ActiveScopeMonitor(w.spec,w.admission,w.co,original_guard)
        with w.patched(),patch.object(a6.time,'monotonic',side_effect=lambda:clock.now):
            with self.assertRaisesRegex(ValueError,'original outer deadline expired'):monitor.observe()
        self.assertLess(clock.commands,124)

    def test12_guard_failure_before_exec_reauthentication_stops_additional_IO(self):
        w=self.world();_,_,calls=self.interleaving(w)
        def guard():
            if len(calls)>=3:raise RuntimeError('original guard rejected lease')
        monitor=a6.ActiveScopeMonitor(w.spec,w.admission,w.co,guard)
        with w.patched(),self.assertRaisesRegex(RuntimeError,'original guard rejected lease'):monitor.observe()
        self.assertEqual(len(calls),3)


if __name__=='__main__':unittest.main(verbosity=2)
