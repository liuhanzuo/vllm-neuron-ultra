"""Actual gated stdlib worker fixtures; original science checked, not run.

Inside each temporary child only, scientific functions, startup restoration
and environment inspection are explicitly replaced by synthetic callbacks.
This tests orchestration and failure preservation, not either native model.
"""
import ast
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import nemotron4b_native_science_v3 as science
import nemotron4b_native_worker_execution_v3 as worker
import nemotron4b_process_handshake_v1 as handshake
import nemotron4b_gated_child_supervisor_v2 as supervisor
import nemotron4b_gated_rank_pair_v1 as pair
import nemotron4b_worker_receipts_v1 as receipts
import nemotron4b_gated_sdk_helper_v1 as sdk
import nemotron4b_owned_group_drain_v2 as group

NONCE='8'*32
FUNCTIONS=('inventory_from_header','verify_native_shapes','verify_rank_shapes',
           'native_meta','runtime_receipt','nxd_meta')


def fds():
    return (int(sys.argv[sys.argv.index('--ready-fd')+1]),
            int(sys.argv[sys.argv.index('--release-fd')+1]))


def helper_fixture():
    ready,release=fds();root=Path(sys.argv[2]);mode=sys.argv[3];role=sys.argv[4]
    handshake.child_wait(ready,release,nonce=NONCE,role=role,deadline=time.monotonic()+5)
    receipts.unpack((root/(role+'-capture.json')).read_bytes())
    if mode=='helper-fail':raise SystemExit(29)
    print(sdk.LIBRARY,flush=True)


def worker_fixture():
    root=Path(sys.argv[2]);mode=sys.argv[3];phase=sys.argv[4]
    role='native-meta'if phase=='native-meta'else 'rank-'+os.environ['RANK']
    helper_role='helper-'+os.environ.get('RANK','0')
    science.WORK=root
    names=[role+'-release.json',role+'-result.json',role+'-failure.json',
           helper_role+'-capture.json',helper_role+'-supervision.json']
    writer=receipts.ReceiptWriter(root,allowed_names=names,guard=lambda:None)
    timeline=[]
    def mark(stage):
        timeline.append(stage)
        (root/(role+'-fixture-timeline.json')).write_text(json.dumps(timeline))
    plan=worker.original_plan();helper=None
    if phase=='nxd-meta':
        helper=sdk.GatedSDKHelper(plan['import_helper'],wrapper_prefix=[str(Path(sys.executable).resolve()),
            '-B','-S',str(Path(__file__).resolve()),'--helper-fixture',str(root),mode,helper_role],
            environment=dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1'),
            cwd=str(root),nonce=NONCE,role=helper_role,expected_cgroup=handshake.owner(os.getpid())['cgroup'],
            guard=lambda:None,writer=writer)
    def synthetic_science(original):
        mark('science')
        assert original['model']==worker.MODEL and original['scope']['ssm_cache_dtype']=='FP32'
        if mode=='science-fail':raise RuntimeError('synthetic science failure')
        if mode=='science-exit':raise SystemExit(23)
        if mode=='remote-network':
            sys.audit('socket.connect',object(),('192.0.2.1',9))
        if mode=='weight-read':
            with(root/'forbidden.safetensors').open('rb')as f:f.read(1)
        if mode=='outside-write':
            with(root.parent/'nano-worker-outside-should-not-exist').open('w')as f:f.write('bad')
        if mode=='extra-spawn':subprocess.Popen(['/bin/true'])
        if phase=='nxd-meta':
            response=subprocess.run(['libneuronpjrt-path'],stdout=subprocess.PIPE)
            assert response.stdout==(sdk.LIBRARY+'\n').encode()
            os.environ['NEURON_LIBRARY_PATH']=sdk.LIBRARY
        if mode=='pair-rendezvous':
            (root/(role+'-rendezvous.json')).write_text(json.dumps({'synthetic':True}))
            peer=root/('rank-'+str(1-int(role[-1]))+'-rendezvous.json')
            deadline=time.monotonic()+3
            while not peer.exists():
                if time.monotonic()>deadline:raise RuntimeError('Synthetic peer missing')
                time.sleep(.01)
        return {'synthetic_science':True,'meta_only':True}
    def synthetic_runtime(original,which):
        mark('runtime')
        if mode=='runtime-fail':raise RuntimeError('synthetic runtime failure')
        return {'synthetic_runtime':True,'phase':which}
    science.native_meta=synthetic_science;science.nxd_meta=synthetic_science;science.runtime_receipt=synthetic_runtime
    worker.restored_environment=lambda original,which:mark('environment')
    if mode=='result-write-fail':
        original_write=writer.write
        def write(name,value):
            if name==role+'-result.json':raise OSError('synthetic result reservation failure')
            return original_write(name,value)
        writer.write=write
    @contextmanager
    def startup():
        mark('startup-enter')
        if mode=='startup-fail':raise RuntimeError('synthetic startup failure')
        value={'synthetic_startup':True,'complete':False}
        try:yield value
        finally:mark('startup-exit')
        if mode=='startup-exit-fail':raise RuntimeError('synthetic startup exit failure')
        value['complete']=True
    ready,release=fds()
    def gate():
        proof=handshake.child_wait(ready,release,nonce=NONCE,role=role,deadline=time.monotonic()+5)
        mark('released');return proof
    original_run=subprocess.run
    try:
        result=worker.run(phase=phase,role=role,nonce=NONCE,guard=lambda:None,
                          parent_gate=gate,startup_context=startup,writer=writer,helper=helper)
        observation={'fixture_success':True,'result':result}
    except BaseException as error:
        observation={'fixture_success':False,'error':{'type':type(error).__name__,'message':str(error)}}
        if isinstance(error,SystemExit):observation['original_exit_value']=error.code
    observation.update(actual_original_science_executed=False,actual_SDK_helper_executed=False,
                       subprocess_run_restored=subprocess.run is original_run,timeline=timeline)
    (root/(role+'-fixture-observation.json')).write_text(json.dumps(observation))


class OriginalScience(unittest.TestCase):
    def test_six_scientific_functions_are_ast_identical_to_pinned_original(self):
        original=science.read(science.ORIGINAL_WORKER)
        self.assertEqual(hashlib.sha256(original).hexdigest(),science.ORIGINAL_WORKER_SHA256)
        def definitions(raw):return {x.name:ast.dump(x,include_attributes=False)for x in ast.parse(raw).body if isinstance(x,ast.FunctionDef)}
        old=definitions(original);new=definitions(Path(science.__file__).read_bytes())
        self.assertEqual({k:old[k]for k in FUNCTIONS},{k:new[k]for k in FUNCTIONS})
        self.assertNotEqual(science.WORK,science.ROOT/'nemotron4b-native-prerequisite-v2')

    def test_original_scope_environments_and119pins_retained(self):
        plan=worker.original_plan();self.assertEqual(len(plan['source_sha256']),119)
        self.assertEqual(plan['scope'],dict(model=worker.MODEL,revision=worker.REVISION,dtype='BF16',
            tp=2,batch=4,context=4096,standard_cpu_sampler=True,cache='original',ssm_cache_dtype='FP32'))
        self.assertEqual(plan['environments']['native-meta']['python_version'],[3,12])
        self.assertEqual(plan['environments']['nxd-meta']['python_version'],[3,10])


class WorkerFlow(unittest.TestCase):
    def invoke(self,mode='normal',phase='native-meta',capture=None):
        temporary=tempfile.TemporaryDirectory(prefix='nano-science-worker-fixture-');self.addCleanup(temporary.cleanup)
        root=Path(temporary.name);role='native-meta'if phase=='native-meta'else 'rank-0'
        writer=receipts.ReceiptWriter(root,allowed_names=['parent-capture.json'],guard=lambda:None)
        def command(ready,release):
            return [str(Path(sys.executable).resolve()),'-B','-S',str(Path(__file__).resolve()),
                    '--worker-fixture',str(root),mode,phase,'--ready-fd',str(ready),'--release-fd',str(release)]
        env=dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
                 NEMOTRON4B_NATIVE_PREREQUISITE_ATTEMPT=NONCE,RANK='0',WORLD_SIZE='2',PJRT_DEVICE='NEURON')
        result=supervisor.run(command,environment=env,cwd=str(root),nonce=NONCE,role=role,
            expected_cgroup=handshake.owner(os.getpid())['cgroup'],guard=lambda:None,
            record_capture=capture or(lambda value:writer.write('parent-capture.json',value)),
            handshake_seconds=3,execution_seconds=5,stdout_cap=4096,stderr_cap=16384)
        self.assertTrue(result['child_reaped']);self.assertEqual(result['cleanup_errors'],[])
        observation_path=root/(role+'-fixture-observation.json')
        observation=json.loads(observation_path.read_text())if observation_path.exists()else None
        return root,role,result,observation

    def test_native_phase_gate_startup_science_runtime_and_durable_receipt(self):
        root,role,child,observation=self.invoke()
        self.assertTrue(child['successful'],child['output']['stderr'])
        self.assertTrue(observation['fixture_success'],observation)
        self.assertEqual(observation['timeline'],['released','startup-enter','environment','science','runtime','startup-exit'])
        report=receipts.unpack((root/(role+'-result.json')).read_bytes())
        self.assertTrue(report['passed']);self.assertTrue(report['result']['synthetic_science'])
        self.assertTrue(report['startup']['complete']);self.assertFalse(report['baseline_created'])
        self.assertFalse(observation['actual_original_science_executed'])
        self.assertTrue(observation['subprocess_run_restored'])

    def test_nxd_phase_real_helper_interface_is_composed_once(self):
        root,role,child,observation=self.invoke(phase='nxd-meta')
        self.assertTrue(child['successful'],child['output']['stderr'])
        self.assertTrue(observation['fixture_success'],observation)
        report=receipts.unpack((root/(role+'-result.json')).read_bytes())
        self.assertEqual(report['result']['import_helper']['calls'],1)
        helper=receipts.unpack((root/'helper-0-supervision.json').read_bytes())
        self.assertTrue(helper['successful']);self.assertTrue(group.supervisor_summary(helper,NONCE)['successful'])
        self.assertFalse(observation['actual_SDK_helper_executed'])

    def test_startup_failure_precedes_science(self):
        root,role,child,observation=self.invoke('startup-fail')
        self.assertFalse(observation['fixture_success']);self.assertNotIn('science',observation['timeline'])
        failed=receipts.unpack((root/(role+'-failure.json')).read_bytes())
        self.assertFalse(failed['worker_audit_installed']);self.assertIsNotNone(failed['release'])
        self.assertFalse((root/(role+'-result.json')).exists())

    def test_science_exception_and_exit_preserve_failure_without_success(self):
        for mode in ['science-fail','science-exit']:
            with self.subTest(mode=mode):
                root,role,child,observation=self.invoke(mode)
                self.assertFalse(observation['fixture_success']);self.assertTrue(observation['subprocess_run_restored'])
                failed=receipts.unpack((root/(role+'-failure.json')).read_bytes())
                self.assertIsNone(failed['scientific_result']);self.assertFalse(failed['passed'])
                self.assertFalse((root/(role+'-result.json')).exists())
                if mode=='science-exit':self.assertEqual(observation['original_exit_value'],23)

    def test_post_science_failures_keep_completed_result_and_runtime(self):
        for mode in ['runtime-fail','startup-exit-fail','result-write-fail']:
            with self.subTest(mode=mode):
                root,role,child,observation=self.invoke(mode)
                self.assertFalse(observation['fixture_success']);self.assertTrue(observation['subprocess_run_restored'])
                failed=receipts.unpack((root/(role+'-failure.json')).read_bytes())
                self.assertTrue(failed['scientific_result']['synthetic_science'])
                if mode!='runtime-fail':self.assertTrue(failed['runtime']['synthetic_runtime'])
                self.assertFalse((root/(role+'-result.json')).exists())

    def test_helper_failure_records_real_helper_exit_before_worker_failure(self):
        root,role,child,observation=self.invoke('helper-fail',phase='nxd-meta')
        self.assertFalse(observation['fixture_success']);self.assertTrue(observation['subprocess_run_restored'])
        helper=receipts.unpack((root/'helper-0-supervision.json').read_bytes())
        self.assertEqual(helper['actual_exit_code'],29);self.assertFalse(helper['successful'])
        self.assertFalse(receipts.unpack((root/(role+'-failure.json')).read_bytes())['passed'])

    def test_no_payload_external_write_remote_network_or_extra_spawn(self):
        for mode in ['weight-read','outside-write','remote-network','extra-spawn']:
            with self.subTest(mode=mode):
                root,role,child,observation=self.invoke(mode)
                self.assertFalse(observation['fixture_success']);self.assertEqual(observation['error']['type'],'ValueError')
                self.assertFalse((root/(role+'-result.json')).exists())
                self.assertFalse((root.parent/'nano-worker-outside-should-not-exist').exists())

    def test_parent_capture_failure_prevents_worker_startup_and_science(self):
        def fail(value):raise OSError('parent capture unavailable')
        root,role,child,observation=self.invoke(capture=fail)
        self.assertFalse(child['successful'])
        if observation is not None:self.assertEqual(observation['timeline'],[])
        self.assertFalse((root/(role+'-release.json')).exists())
        self.assertFalse((root/(role+'-result.json')).exists())

    def test_two_actual_rank_workers_and_helpers_share_receipt_directory(self):
        temporary=tempfile.TemporaryDirectory(prefix='nano-science-pair-fixture-');self.addCleanup(temporary.cleanup)
        root=Path(temporary.name);writer=receipts.ReceiptWriter(root,allowed_names=['pair-capture.json'],guard=lambda:None)
        commands={};environments={}
        for role in pair.ROLES:
            commands[role]=lambda ready,release,role=role:[str(Path(sys.executable).resolve()),'-B','-S',
                str(Path(__file__).resolve()),'--worker-fixture',str(root),'pair-rendezvous','nxd-meta',
                '--ready-fd',str(ready),'--release-fd',str(release)]
            environments[role]=dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
                NEMOTRON4B_NATIVE_PREREQUISITE_ATTEMPT=NONCE,RANK=role[-1],WORLD_SIZE='2',PJRT_DEVICE='NEURON')
        result=pair.run(commands,environments=environments,cwd=str(root),nonce=NONCE,
            expected_cgroup=handshake.owner(os.getpid())['cgroup'],guard=lambda:None,
            record_pair=lambda value:writer.write('pair-capture.json',value),
            handshake_seconds=3,execution_seconds=5,stdout_cap=4096,stderr_cap=16384)
        self.assertTrue(result['direct_children_successful'],result)
        self.assertTrue(group.supervisor_summary(result,NONCE)['successful'])
        pids=[]
        for role in pair.ROLES:
            observation=json.loads((root/(role+'-fixture-observation.json')).read_text())
            self.assertTrue(observation['fixture_success'],observation)
            report=receipts.unpack((root/(role+'-result.json')).read_bytes())
            self.assertEqual(report['role'],role);self.assertEqual(report['result']['import_helper']['calls'],1)
            helper=receipts.unpack((root/('helper-'+role[-1]+'-supervision.json')).read_bytes())
            self.assertTrue(helper['successful']);self.assertTrue(helper['child_reaped'])
            pids.append(report['producer']['pid']);pids.append(helper['handshake']['first_actual_capture']['pid'])
        self.assertEqual(len(set(pids)),4)
        self.assertTrue(all(not Path('/proc',str(pid)).exists()for pid in pids))


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--worker-fixture':worker_fixture()
    elif len(sys.argv)>1 and sys.argv[1]=='--helper-fixture':helper_fixture()
    else:unittest.main(verbosity=2)
