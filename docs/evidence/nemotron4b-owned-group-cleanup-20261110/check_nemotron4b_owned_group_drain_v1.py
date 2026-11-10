"""Generated cgroup metadata/fake systemd lifecycle, plus actual stdlib IO.

No systemd service is launched/stopped/killed, and no SDK/native model runs.
The bounded command reader is exercised only with owned Python fixtures.
"""
import copy
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import nemotron4b_owned_group_drain_v1 as drain
import nemotron4b_gated_rank_pair_v1 as pair
import nemotron4b_gated_child_supervisor_v1 as child
import nemotron4b_process_handshake_v1 as handshake

NONCE='b'*32
ARGV=['/usr/bin/env','-i','NEMOTRON4B_NATIVE_PREREQUISITE_ATTEMPT='+NONCE,
      '/usr/bin/python3','-B','/tmp/generated-worker.py']
RESOURCES={'MemoryMax':str(12*2**30),'KillMode':'control-group'}
GROUP=drain.GROUP_PREFIX+drain.UNIT_PREFIX+'nxd-meta.service'


def fixture():
    ready,release=int(sys.argv[2]),int(sys.argv[3]);role=sys.argv[4]
    handshake.child_wait(ready,release,nonce=NONCE,role=role,deadline=time.monotonic()+5)
    print('generated direct-child output',flush=True)


class FakeProbe:
    def __init__(self):
        self.time=0.;self.calls=[];self.on_stop=lambda:self.end();self.on_kill=lambda:self.end()
        self.on_wait=lambda:None
        self.value=dict(LoadState='loaded',ActiveState='active',SubState='running',
            MainPID='101',ControlPID='0',Job='',InvocationID='c'*32,ControlGroup=GROUP,
            ExecStart='{ path=/usr/bin/env ; argv[]='+' '.join(ARGV)+' ; ignore_errors=no ; }',
            ExecMainPID='101',ExecMainCode='0',ExecMainStatus='0',Result='success',**RESOURCES)
        self.processes={101:dict(pid=101,start_ticks='123',cgroup='0::'+GROUP,state='S')}
        self.tree=dict(group=GROUP,absent=False,empty=False,rows=[dict(path='.',pids=[101],populated='1')])

    def end(self):
        self.value.update(ActiveState='active',SubState='exited',MainPID='0',ControlPID='0',
                          Job='',ExecMainCode='1',ExecMainStatus='0')
        self.tree.update(empty=True,rows=[dict(path='.',pids=[],populated='0')])
        self.processes={}

    def absent(self):
        self.end();self.value.update(LoadState='not-found',ActiveState='inactive',SubState='dead',
             InvocationID='',ControlGroup='',ExecStart='',ExecMainPID='0',ExecMainCode='0',Result='')

    def properties(self,unit,resource_keys):return copy.deepcopy(self.value)
    def group(self,name):return copy.deepcopy(self.tree)
    def process(self,pid):return copy.deepcopy(self.processes.get(pid))
    def clock(self):return self.time
    def wait(self,seconds):self.time+=seconds;self.on_wait()
    def stop(self,unit):self.calls.append(('stop',unit));self.on_stop()
    def kill(self,unit):self.calls.append(('kill',unit));self.on_kill()


class Lifecycle(unittest.TestCase):
    def setUp(self):
        self.probe=FakeProbe();self.records=[]
        self.phase=drain.OwnedPhase(phase='nxd-meta',nonce=NONCE,exec_argv=ARGV,
            resources=RESOURCES,probe=self.probe,record=self.records.append)

    def capture(self):self.phase.register_owner(self.probe.processes[101])

    def test_natural_exit_requires_exact_owner_and_two_whole_group_samples(self):
        self.capture();self.probe.end()
        result=self.phase.finish({'successful':True,'child_reaped':True})
        self.assertTrue(result['successful']);self.assertTrue(result['owned_group_drained'])
        self.assertEqual(len(result['drain']['samples']),2);self.assertEqual(self.probe.calls,[])
        self.assertFalse(result['serial_lock_release_authorized'])

    def test_empty_group_without_actual_owner_capture_is_not_success(self):
        self.probe.end();result=self.phase.finish({'successful':True})
        self.assertTrue(result['owned_group_drained']);self.assertFalse(result['successful'])

    def test_failed_child_remains_failed_after_clean_exit(self):
        self.capture();self.probe.end()
        result=self.phase.finish({'successful':False,'actual_exit_code':7})
        self.assertTrue(result['owned_group_drained']);self.assertFalse(result['successful'])

    def test_terminal_main_pid_must_match_captured_service_main(self):
        self.capture();self.probe.end();self.probe.value['ExecMainPID']='202'
        result=self.phase.finish({'successful':True})
        self.assertTrue(result['owned_group_drained']);self.assertFalse(result['successful'])

    def test_captured_descendant_cannot_substitute_for_main(self):
        owner=dict(pid=202,start_ticks='124',cgroup='0::'+GROUP,state='S')
        self.probe.processes[202]=owner;self.phase.register_owner(owner)
        self.probe.end()
        self.assertFalse(self.phase.finish({'successful':True})['successful'])

    def test_failed_service_cannot_use_successful_child_as_denominator(self):
        self.capture();self.probe.end();self.probe.value.update(Result='exit-code',ExecMainStatus='1')
        self.assertFalse(self.phase.finish({'successful':True})['successful'])

    def test_remaining_descendant_triggers_owned_stop_and_prevents_success(self):
        self.capture();self.probe.end()
        self.probe.tree.update(empty=False,rows=[dict(path='child',pids=[202],populated='1')])
        result=self.phase.finish({'successful':True})
        self.assertTrue(result['owned_group_drained']);self.assertFalse(result['successful'])
        self.assertEqual(self.probe.calls,[('stop',self.phase.unit)])
        self.assertTrue(result['cleanup_actions_required'])

    def test_wrong_resources_still_allow_cleanup_of_authenticated_owner(self):
        self.capture();self.probe.value['MemoryMax']='1'
        with self.assertRaisesRegex(ValueError,'resource'):self.phase.running_guard()
        result=self.phase.finish({'successful':False})
        self.assertTrue(result['owned_group_drained']);self.assertFalse(result['successful'])
        self.assertEqual(self.probe.calls,[('stop',self.phase.unit)])

    def test_stop_then_kill_all_when_descendants_remain(self):
        self.capture();self.probe.on_stop=lambda:None
        result=self.phase.cleanup(seconds=1)
        self.assertTrue(result['owned_group_drained'])
        self.assertEqual([x[0]for x in self.probe.calls],['stop','kill'])

    def test_timeout_retains_failure_receipt_and_does_not_authorize_release(self):
        self.capture();self.probe.on_stop=self.probe.on_kill=lambda:None
        with self.assertRaises(drain.PendingDrain):self.phase.finish({'successful':True})
        result=self.records[-1]
        self.assertFalse(result['successful']);self.assertFalse(result['owned_group_drained'])
        self.assertFalse(result['serial_lock_release_authorized']);self.assertIn('cleanup_failure',result)

    def test_changed_invocation_never_stops_replacement(self):
        self.capture();self.probe.value['InvocationID']='d'*32
        with self.assertRaisesRegex(ValueError,'replaced'):self.phase.cleanup()
        self.assertEqual(self.probe.calls,[])

    def test_changed_command_or_group_never_signalled(self):
        for field,value in [('ExecStart','wrong'),('ControlGroup','/unowned')]:
            with self.subTest(field=field):
                old=self.probe.value[field];self.probe.value[field]=value
                with self.assertRaises(ValueError):self.phase.cleanup()
                self.assertEqual(self.probe.calls,[]);self.probe.value[field]=old

    def test_invocation_change_after_stop_prevents_kill(self):
        self.capture()
        self.probe.on_stop=lambda:self.probe.value.update(InvocationID='d'*32)
        with self.assertRaisesRegex(ValueError,'replaced'):self.phase.cleanup()
        self.assertEqual([x[0]for x in self.probe.calls],['stop'])

    def test_no_process_identity_invented_for_early_service_exit(self):
        self.probe.end();self.probe.value.update(Result='exit-code',ExecMainStatus='1')
        result=self.phase.finish({'successful':False})
        self.assertTrue(result['owned_group_drained']);self.assertEqual(self.phase.owners,[])
        self.assertEqual(self.phase.invocation,'c'*32)

    def test_unlaunched_or_collected_service_requires_fresh_empty_group(self):
        self.probe.absent()
        result=self.phase.finish({'successful':False})
        self.assertTrue(result['owned_group_drained']);self.assertFalse(result['successful'])
        self.assertEqual(self.probe.calls,[]);self.assertIsNone(self.phase.invocation)

    def test_record_failure_prevents_any_cleanup_operation(self):
        def fail(record):raise OSError('durable receipt failed')
        self.phase.record=fail
        with self.assertRaises(OSError):self.phase.cleanup()
        self.assertEqual(self.probe.calls,[])

    def test_operation_intent_persisted_before_action(self):
        self.capture()
        def check_then_end():
            self.assertEqual(self.records[-1]['kind'],'cleanup_intent')
            self.assertEqual(self.records[-1]['invocation_id'],'c'*32)
            self.probe.end()
        self.probe.on_stop=check_then_end
        self.phase.cleanup()

    def test_stop_error_after_normal_exit_requires_new_drain(self):
        self.capture()
        def stop():self.probe.end();raise ValueError('systemctl raced with exit')
        self.probe.on_stop=stop
        result=self.phase.finish({'successful':True})
        self.assertTrue(result['owned_group_drained']);self.assertFalse(result['successful'])
        self.assertFalse(result['drain']['actions'][0]['completed'])
        self.assertEqual(len(result['drain']['operation_errors']),1)

    def test_reused_pid_is_not_signalled_or_treated_as_original_owner(self):
        self.capture();self.probe.end()
        self.probe.processes[101]=dict(pid=101,start_ticks='999',cgroup='0::/other',state='S')
        self.assertTrue(self.phase.finish({'successful':True})['successful'])
        self.assertEqual(self.probe.calls,[])

    def test_same_start_process_outside_group_blocks_drain(self):
        self.capture();original=self.probe.processes[101];original['cgroup']='0::/other'
        def keep_escape():self.probe.end();self.probe.processes[101]=original
        keep_escape();self.probe.on_stop=self.probe.on_kill=keep_escape
        with self.assertRaises(drain.PendingDrain):self.phase.cleanup(seconds=1)
        self.assertTrue(all(unit==self.phase.unit for _,unit in self.probe.calls))

    def test_zombie_is_not_counted_as_gone(self):
        self.capture();self.probe.processes[101]['state']='Z'
        with self.assertRaises(drain.PendingDrain):self.phase.prove_drained()

    def test_second_sample_repopulation_rejected(self):
        self.capture();self.probe.end()
        self.probe.on_wait=lambda:self.probe.tree.update(empty=False)
        with self.assertRaises(drain.PendingDrain):self.phase.prove_drained()


class Tree(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='nano-group-tree-fixture-');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.base=self.root/GROUP.lstrip('/')

    def group(self,relative='',pids='',populated='0'):
        path=self.base/relative;path.mkdir(parents=True,exist_ok=True)
        (path/'cgroup.procs').write_text(pids);(path/'cgroup.events').write_text('populated '+populated+'\nfrozen 0\n')
        return path

    def test_absent_group_and_recursive_empty_group(self):
        self.assertTrue(drain.inspect_group(GROUP,root=self.root)['absent'])
        self.group();self.group('child');self.group('child/grandchild')
        result=drain.inspect_group(GROUP,root=self.root)
        self.assertTrue(result['empty']);self.assertEqual(len(result['rows']),3)

    def test_nested_process_and_kernel_population_detected(self):
        self.group();child=self.group('child',pids='123\n',populated='1')
        self.assertFalse(drain.inspect_group(GROUP,root=self.root)['empty'])
        (child/'cgroup.procs').write_text('')
        self.assertFalse(drain.inspect_group(GROUP,root=self.root)['empty'])

    def test_symlink_subtree_cannot_redirect_scan(self):
        self.group();(self.base/'link').symlink_to(self.root,target_is_directory=True)
        with self.assertRaises(ValueError):drain.inspect_group(GROUP,root=self.root)

    def test_missing_or_duplicate_kernel_population_is_not_empty_proof(self):
        self.group()
        for content in ['frozen 0\n','populated 0\npopulated 0\n']:
            (self.base/'cgroup.events').write_text(content)
            with self.assertRaises(ValueError):drain.inspect_group(GROUP,root=self.root)

    def test_incomplete_metadata_read_and_scope_rejected(self):
        self.group(pids='1\n'*10000)
        with self.assertRaises(ValueError):drain.inspect_group(GROUP,root=self.root)
        with self.assertRaises(ValueError):drain.inspect_group('/unrelated.service',root=self.root)

    def test_subtree_budget_is_not_silent_truncation(self):
        self.group()
        for i in range(128):self.group('child'+str(i))
        with self.assertRaises(ValueError):drain.inspect_group(GROUP,root=self.root)

    def test_current_kernel_process_metadata(self):
        value=drain.process(os.getpid())
        self.assertEqual(value['pid'],os.getpid());self.assertTrue(value['start_ticks'].isdigit())
        self.assertTrue(value['cgroup'].startswith('0::/'))


class CommandIO(unittest.TestCase):
    def test_actual_fixture_output_and_nonzero_exit(self):
        probe=drain.LiveProbe(lambda:None)
        self.assertEqual(probe.command([sys.executable,'-B','-c','print("fixture")']),'fixture\n')
        with self.assertRaisesRegex(ValueError,'operation failed'):
            probe.command([sys.executable,'-B','-c','raise SystemExit(9)'])

    def test_actual_fixture_output_cap_reaps_child(self):
        with tempfile.TemporaryDirectory(prefix='nano-command-fixture-')as directory:
            marker=Path(directory)/'pid'
            code='import os,time;from pathlib import Path;Path('+repr(str(marker))+').write_text(str(os.getpid()));os.write(1,b"x"*131072);time.sleep(10)'
            with self.assertRaisesRegex(ValueError,'Bounded systemd response'):
                drain.LiveProbe(lambda:None).command([sys.executable,'-B','-c',code])
            pid=int(marker.read_text());self.assertFalse(Path('/proc',str(pid)).exists())

    def test_guard_denial_prevents_fixture_spawn(self):
        def deny():raise RuntimeError('no serial ownership')
        with tempfile.TemporaryDirectory(prefix='nano-command-denied-')as directory:
            marker=Path(directory)/'ran'
            code='from pathlib import Path;Path('+repr(str(marker))+').touch()'
            with self.assertRaisesRegex(RuntimeError,'serial ownership'):
                drain.LiveProbe(deny).command([sys.executable,'-B','-c',code])
            self.assertFalse(marker.exists())


class ExistingSupervisorIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='nano-group-integration-');self.addCleanup(self.tmp.cleanup)
        self.directory=Path(self.tmp.name)
        self.env=dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
        self.cgroup=handshake.owner(os.getpid())['cgroup']

    def command(self,role):
        return lambda ready,release:[str(Path(sys.executable).resolve()),'-B','-S',
            str(Path(__file__).resolve()),'--fixture',str(ready),str(release),role]

    def test_actual_gated_child_result_adapts_with_original_binary_output(self):
        result=child.run(self.command('native-meta'),environment=self.env,cwd=self.tmp.name,
            nonce=NONCE,role='native-meta',expected_cgroup=self.cgroup,guard=lambda:None,
            handshake_seconds=2,execution_seconds=3)
        summary=drain.supervisor_summary(result,NONCE)
        self.assertTrue(summary['successful']);self.assertEqual(summary['direct_children'],1)
        self.assertEqual(result['output']['stdout'],b'generated direct-child output\n')
        self.assertTrue(result['child_reaped']);self.assertFalse(result['owned_group_drained_proven'])

    def actual_pair(self):
        def record(value):
            target=self.directory/'capture.json'
            import json
            with target.open('x')as f:json.dump(value,f);f.flush();os.fsync(f.fileno())
            return {'path':str(target)}
        return pair.run({role:self.command(role)for role in pair.ROLES},
            environments={role:dict(self.env)for role in pair.ROLES},cwd=self.tmp.name,
            nonce=NONCE,expected_cgroup=self.cgroup,guard=lambda:None,record_pair=record,
            handshake_seconds=2,execution_seconds=3)

    def test_actual_pair_result_persists_before_synthetic_group_terminal(self):
        result=self.actual_pair();summary=drain.supervisor_summary(result,NONCE)
        self.assertTrue(summary['successful']);self.assertEqual(summary['direct_children'],2)
        self.assertTrue(all(r['child_reaped']for r in result['ranks'].values()))
        probe=FakeProbe();probe.end();records=[];saved=[]
        phase=drain.OwnedPhase(phase='nxd-meta',nonce=NONCE,exec_argv=ARGV,
            resources=RESOURCES,probe=probe,record=records.append)
        def persist(original):
            self.assertIs(original,result);saved.append(original)
            self.assertTrue(all(r['output']['stdout']==b'generated direct-child output\n'for r in original['ranks'].values()))
            return {'path':'/synthetic/caller/receipt','sha256':'e'*64}
        terminal=phase.finish_supervisor(result,persist_result=persist)
        self.assertEqual(len(saved),1)
        self.assertTrue(terminal['owned_group_drained']) # Generated service metadata only.
        # Real fixture children belong to the observer group; never manufacture
        # actual Nano service ownership from these fixture process identities.
        self.assertFalse(terminal['successful']);self.assertEqual(phase.owners,[])

    def test_inconsistent_positive_pair_result_rejected_and_owned_cleanup_runs(self):
        result=self.actual_pair();result['ranks']['rank-1']['actual_exit_code']=19
        probe=FakeProbe();records=[];phase=drain.OwnedPhase(phase='nxd-meta',nonce=NONCE,
            exec_argv=ARGV,resources=RESOURCES,probe=probe,record=records.append)
        with self.assertRaisesRegex(ValueError,'exited zero'):
            phase.finish_supervisor(result,persist_result=lambda x:None)
        self.assertEqual([x[0]for x in probe.calls],['stop'])
        self.assertFalse(records[-1]['successful']);self.assertTrue(records[-1]['owned_group_drained'])

    def test_durable_output_failure_still_cleans_owned_group(self):
        result=self.actual_pair();probe=FakeProbe();records=[]
        phase=drain.OwnedPhase(phase='nxd-meta',nonce=NONCE,exec_argv=ARGV,
            resources=RESOURCES,probe=probe,record=records.append)
        def fail(original):raise OSError('output reservation exhausted')
        with self.assertRaisesRegex(OSError,'reservation'):
            phase.finish_supervisor(result,persist_result=fail)
        self.assertEqual([x[0]for x in probe.calls],['stop'])
        self.assertFalse(records[-1]['successful'])


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--fixture':fixture()
    else:unittest.main(verbosity=2)
