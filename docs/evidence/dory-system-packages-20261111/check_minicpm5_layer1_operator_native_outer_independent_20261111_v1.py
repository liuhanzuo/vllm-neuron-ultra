"""Independent source review controls: memory only; no service/lock/native IO."""
import copy
import io
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'research'),str(ROOT)]
import minicpm5_layer1_operator_native_outer_runner_v2 as outer
import minicpm5_python_trace_lifecycle_v4 as atomic
import nemotron9b_package_startup_drain_v27 as startup
import nemotron9b_r12_completion_v1 as r12
from research import minicpm5_layer1_sdk_batch_layout_completion_v3 as historical
from research import minicpm5_layer1_operator_native_output_catalog_v2 as catalog
from research import check_minicpm5_layer1_operator_native_outer_runner_v2 as fixtures

UNIT='trn1-mini-layer1-operator-native-independent-synthetic'
RUNNING=UNIT+'.service loaded active running synthetic\n'

class Independent(unittest.TestCase):
    def test_real_property_representation_accepts_systemd_two_hours(self):
        f=fixtures.Outer('test_service_exact_original_resources_command_invocation');f.setUp()
        value=copy.deepcopy(f.service)
        value.update(RuntimeMaxUSec='2h',TimeoutStopUSec='15s',Nice='19',AllowedCPUs='6-7',CPUAffinity='6-7')
        # These scalar representations occur in the previously saved actual SDK r2 metadata.
        raw='\n'.join(k+'='+v for k,v in value.items())
        with patch.object(outer.p,'command',return_value=raw) as command:
            parsed=outer.r.native_service_properties(f.contract['unit'])
        outer.service_identity(f.contract,f.ref,f.a,parsed,f.co)
        keys=set(command.call_args.args[0][-1].removeprefix('--property=').split(','))
        self.assertTrue({'Nice','RuntimeMaxUSec','TimeoutStopUSec'}<=keys)

    def test_real_atomic_once_has_no_final_visibility_before_full_fsync(self):
        events=[];written=[];visible={};destination='/synthetic/native-result.json'
        class Stream:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def write(self,raw):written.append(raw);events.append('write');return len(raw)
            def flush(self):events.append('flush')
            def fileno(self):return 10
        def rename(stage,path):
            self.assertEqual(events,['write','flush','fsync:10'])
            self.assertEqual(visible,{})
            visible[str(path)]=b''.join(written);events.append('rename')
        with patch.object(atomic.low,'clear_window'),patch.object(atomic.os,'getpid',return_value=102), \
             patch.object(atomic.os,'open',side_effect=[10,11]) as opened,patch.object(atomic.os,'fdopen',return_value=Stream()), \
             patch.object(atomic.os,'fsync',side_effect=lambda fd:events.append('fsync:'+str(fd))), \
             patch.object(atomic,'rename_noreplace',side_effect=rename),patch.object(atomic.os,'close'):
            atomic.once(destination,{'complete':True})
        stage=str(opened.call_args_list[0].args[0]);self.assertEqual(stage,'/synthetic/.pending-native-result-102.json')
        self.assertEqual(catalog.staging_destination(Path(stage).name,{'native':102}),'native-result.json')
        self.assertEqual(events,['write','flush','fsync:10','rename','fsync:11'])
        self.assertEqual(visible[destination],b'{\n  "complete": true\n}\n')
        for name,role in catalog.STAGING_WRITERS.items():
            self.assertEqual(catalog.staging_destination('.pending-'+Path(name).stem+'-102.json',{role:102}),name)

    def test_nested_startup_scan_rejects_current_service_after_double_history_sample(self):
        name='trn1-historical-synthetic'
        candidate=dict(processes=[],missing_child_start_identities=[],rows=[dict(unit=name,owner=None,never_launched=True,absent_records=[],recorded_cgroups=[])])
        value=dict(Id=name+'.service',LoadState='not-found',InvocationID='',ExecMainPID='0')
        life=SimpleNamespace(gone=Mock(),props=Mock(return_value=value),terminal=Mock(return_value=True),
            GROUP_PREFIX=outer.p.GROUP_PREFIX,cgroup_inventory=Mock(return_value={'empty':True}),command=Mock(return_value=RUNNING))
        with patch.object(startup.attention,'validate_spec',return_value=candidate), \
             patch.object(startup.startup_history,'validate_spec',return_value=candidate), \
             patch.dict(sys.modules,{'research.finalize_smollm3_sio14_root_20261108_v1':SimpleNamespace(life=life)}), \
             patch.object(startup.time,'sleep'),patch.object(startup.successor,'physical') as next_layer:
            with self.assertRaisesRegex(RuntimeError,'Only explicitly acknowledged current controller may be live'):
                startup.physical_drain(candidate,spec={})
            self.assertEqual(life.props.call_count,2);self.assertEqual(life.cgroup_inventory.call_count,2)
            next_layer.assert_not_called()
            with self.assertRaisesRegex(RuntimeError,'Only one exact new controller may be live'):
                startup.physical_drain(candidate,allowed_units=(UNIT,),spec={})

    def test_next_layer_uses_child_as_lock_owner_instead_of_admitted_outer(self):
        parent=dict(pid=100,start_ticks='10',ppid=1,argv=['outer'],cgroup='0::/outer')
        child=dict(pid=101,start_ticks='11',ppid=1,argv=['coordinator'],cgroup='0::/native')
        lock=outer.r.ORIGINAL_LOCK
        info=SimpleNamespace(st_mode=stat.S_IFREG|0o600,st_nlink=1,st_dev=lock['device'],st_ino=lock['inode'])
        row='20: FLOCK ADVISORY WRITE 100 %02x:%02x:%d 0 EOF\n'%(os.major(lock['device']),os.minor(lock['device']),lock['inode'])
        with patch.object(r12,'validate_spec'),patch.object(r12,'metadata') as metadata, \
             patch.object(r12.os,'getpid',return_value=child['pid']),patch.object(outer.p,'proc',return_value=child), \
             patch.object(outer.p.Path,'lstat',return_value=info),patch.object(outer.p.Path,'is_symlink',return_value=False), \
             patch.object(outer.p.Path,'open',return_value=io.StringIO(row)):
            with self.assertRaisesRegex(RuntimeError,'Controller does not hold whole serial flock'):
                r12.physical({'units':{'controller':'old-controller'}},lambda ref:{'lock_identity':lock},outer.p)
        metadata.assert_not_called()

    def test_upper_historical_physical_still_demands_global_idle(self):
        from research import minicpm5_after_nemotron_r12_recovered_failure_v1 as nemotron
        done=dict(binding={},metadata_complete=True,nemotron={},owners=[],lock_identity=outer.r.ORIGINAL_LOCK)
        with patch.object(historical,'require_complete_binding'),patch.object(nemotron,'physical',return_value={}) as inherited, \
             patch.object(outer.p,'command',return_value=RUNNING):
            # The correctly scoped current guard passes; the nested no-argument check fails.
            self.assertEqual(outer.p.active_heavy((UNIT,)),[UNIT])
            with self.assertRaisesRegex(RuntimeError,'Competing active heavy service'):
                historical.physical(done,lambda:None)
        inherited.assert_called_once()

if __name__=='__main__':unittest.main(verbosity=2)
