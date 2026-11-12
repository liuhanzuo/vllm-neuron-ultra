"""Stdlib/mocks only: block application libraries, ctypes and full history.

No real process/proc/cgroup, lock, service, native-payload or weight operations.
New temporary text/JSON metadata fixtures are the only non-source file IO.
"""
import sys

class NativeImportBlocker:
    roots={'ctypes','_ctypes','torch','torch_xla','torch_neuronx','numpy','transformers','safetensors',
           'neuronx_distributed','neuronx_distributed_inference','neuronxcc','libneuronxla','vllm','vllm_neuron',
           'lfm3_nki05_serial_history_v1'}
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in self.roots:raise RuntimeError('Source-test import blocker: '+fullname)
        return None

BLOCKER=NativeImportBlocker()
if BLOCKER.roots.intersection(sys.modules):raise RuntimeError('Tests require fresh unloaded native/history modules')
sys.meta_path.insert(0,BLOCKER)

import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import run_spark26_elf_preflight_outer_v1 as outer
from research.minicpm5_layer1_drain_hold_v1 import retain_until_drained


def build_fixture_spec():
    candidate=outer.worker.core.CANDIDATE
    sources=json.loads((candidate/'source-map.json').read_text())
    for name in ('spark26_elf_preflight_core_v1.py','spark26_elf_preflight_worker_v1.py','run_spark26_elf_preflight_outer_v1.py'):
        path=outer.ROOT/name;sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    manifest=outer.read(outer.worker.core.MANIFEST_REF);preparation=outer.read(outer.worker.core.PREPARATION_REF)
    payloads=outer.worker.core.provider.merge_manifest(manifest,sources,lambda ref:outer.worker.metadata_bytes(ref).decode())
    plan=outer.worker.core.plan_from_payloads(manifest,preparation,payloads)
    return dict(schema='spark26_elf_preflight_worker_spec_v1',preflight_plan=plan,
        output_root='/mnt/nvme/spark26-elf-preflight-fixture',unit='trn1-spark26-elf-preflight-fixture',
        properties=outer.worker.PROPS,budget=outer.worker.BUDGET,source_sha256=sources,
        native_manifest=outer.worker.core.MANIFEST_REF,metadata_preparation=outer.worker.core.PREPARATION_REF,
        serial_catalog=dict(path='/fixture/catalog.json',sha256='c'*64),
        terminal_handoff=dict(path='/fixture/handoff.json',sha256='d'*64),
        lock_identity=outer.LOCK,native_python=outer.PYTHON,automatic_successor=False)


_FIXTURE=None
def fixture_spec():
    global _FIXTURE
    if _FIXTURE is None:_FIXTURE=build_fixture_spec()
    return copy.deepcopy(_FIXTURE)


def refs(spec):return dict(path=spec['output_root']+'/admission.json',sha256='1'*64)


def process(spec,ref,wrapper=False):
    command=outer.worker.service_command(spec,ref)
    argv=command[command.index('/usr/bin/env'):] if wrapper else outer.worker.argv(ref)
    return dict(pid=301,start_ticks='12345',ppid=201,argv=argv,
                cgroup='0::/user.slice/user-1000.slice/user@1000.service/app.slice/'+spec['unit']+'.service')


def props(spec,ref,phase='running',invocation='a'*32,status=0):
    command=outer.worker.service_command(spec,ref)
    result=dict(Id=spec['unit']+'.service',LoadState='loaded',ActiveState='active',SubState='running',
        MainPID='301',ControlPID='0',ExecMainPID='301',InvocationID=invocation,Job='',
        ControlGroup='/user.slice/user-1000.slice/user@1000.service/app.slice/'+spec['unit']+'.service',
        ExecStart='{ path=/usr/bin/env ; argv[]='+' '.join(command[command.index('/usr/bin/env'):])+' ; ignore_errors=no ; }',
        Environment='',ExecStopPost='',StandardOutput='append',StandardError='inherit',
        ExecMainCode='0',ExecMainStatus='0',Result='success')
    for key,value in outer.worker.PROPS.items():result[key.replace('Sec','USec') if key.endswith('Sec') else key]=value+'s' if key.endswith('Sec') else value
    if phase=='activating':result.update(ActiveState='activating',SubState='start')
    if phase=='terminal':result.update(MainPID='0',ActiveState='active' if status==0 else 'failed',
        SubState='exited' if status==0 else 'failed',ExecMainCode='1',ExecMainStatus=str(status),Result='success' if status==0 else 'exit-code')
    if phase=='not-found':result.update(LoadState='not-found',MainPID='0',ExecMainPID='0',ActiveState='inactive',SubState='dead',InvocationID='',ControlGroup='')
    return result


class FakeHost:
    def __init__(self,spec,ref,steps=None):
        self.spec,self.ref=spec,ref;self.now=0.;self.commands=[];self.publications={};self.groups=[]
        self.steps=list(steps or [(props(spec,ref),process(spec,ref))]);self.last=None;self.actual=None
        self.hold=retain_until_drained;self.empty_error=None
    def properties(self,unit):
        if self.steps:self.last,self.actual=self.steps.pop(0)
        return copy.deepcopy(self.last)
    def proc(self,pid):return copy.deepcopy(self.actual) if self.actual is not None and self.actual['pid']==pid else None
    def gone(self,owner):
        actual=self.proc(owner['pid'])
        outer.require(actual is None or actual['start_ticks']!=owner['start_ticks'],'mock process still live')
    def empty_group(self,group):
        self.groups.append(group)
        if self.empty_error is not None:raise self.empty_error
        return dict(group=group,absent=True)
    def terminal(self,value):return value['MainPID']==value['ControlPID']=='0' and not value['Job'] and (
        value['ActiveState'] in ('inactive','failed') or (value['ActiveState'],value['SubState'])==('active','exited'))
    def clock(self):return self.now
    def wait(self,seconds):self.now+=seconds
    def command(self,argv,timeout=30):self.commands.append(list(argv));return ''
    def publish(self,path,value):
        outer.require(str(path) not in self.publications,'mock immutable publication')
        self.publications[str(path)]=copy.deepcopy(value)
        return dict(path=str(path),sha256=hashlib.sha256(outer.encode(value)).hexdigest())
    def read(self,ref):return copy.deepcopy(self.publications[ref['path']])


def service_fixture(steps=None):
    spec=fixture_spec();ref=refs(spec)
    host=FakeHost(spec,ref,steps)
    admission=dict(outer_owner=dict(pid=201,start_ticks='111',ppid=1,argv=['outer'],cgroup='0::/outer'))
    service=outer.OwnedService(host,spec,ref,admission)
    return service,host,spec,ref


class ControllerChecks(unittest.TestCase):
    def test_blocker_still_active_and_history_not_imported(self):
        self.assertIn(BLOCKER,sys.meta_path)
        self.assertFalse(BLOCKER.roots.intersection(sys.modules))

    def test_exact_outer_binding_requires_both_root_reviews(self):
        spec=fixture_spec();spec_ref=dict(path='/fixture/spec.json',sha256='e'*64)
        binding_ref=dict(path='/fixture/binding.json',sha256='f'*64)
        binding=dict(schema='spark26_elf_preflight_outer_binding_v1',worker_spec=spec_ref,
            worker_review=dict(path='/fixture/review.json',sha256='e'*64),source_sha256=spec['source_sha256'],
            outer_limits=outer.OUTER_LIMITS,host_boundary=outer.HOST_BOUNDARY)
        review=dict(schema='spark26_elf_preflight_outer_root_review_v1',reviewer='/root',binding=binding_ref,
            source_sha256=spec['source_sha256'],passed=True,source_only=True,blockers=[],runtime_admitted=False)
        wr=dict(schema='spark26_elf_preflight_worker_root_review_v1',reviewer='/root',spec=spec_ref,
            source_sha256=spec['source_sha256'],passed=True,source_only=True,blockers=[],runtime_admitted=False)
        outer.validate_binding(binding,review,binding_ref,spec,wr)
        wr['schema']='spark26_native_provider_worker_root_source_review_v1'
        with self.assertRaises(ValueError):outer.validate_binding(binding,review,binding_ref,spec,wr)

    def test_host_finder_blocks_native_and_gates_ctypes(self):
        finder=outer.PinnedHostFinder({},[])
        with self.assertRaises(ValueError):finder.find_spec('torch')
        with self.assertRaises(ValueError):finder.find_spec('ctypes')
        finder.history_gate_open=True
        self.assertIsNone(finder.find_spec('ctypes'))  # fake empty finder list: no import
        with self.assertRaises(ValueError):finder.find_spec('torch_neuronx')

    def test_history_entry_is_only_in_explicit_locked_loader(self):
        tree=ast.parse(Path(outer.__file__).read_text())
        self.assertFalse(any(isinstance(node,ast.Import) and any(a.name=='lfm3_nki05_serial_history_v1' for a in node.names)
                             for node in tree.body))
        host=object.__new__(outer.RealHost);host.history=None;host.finder=SimpleNamespace(history_gate_open=False)
        calls=[];host.unchanged=lambda:calls.append('sources')
        lease=SimpleNamespace(check=lambda:calls.append('lease'))
        def fake_import(name):
            self.assertEqual(name,outer.HOST_BOUNDARY['history_entrypoint'])
            self.assertTrue(host.finder.history_gate_open);self.assertIn('lease',calls)
            calls.append('history');return object()
        with mock.patch.object(outer.importlib,'import_module',side_effect=fake_import):host.load_history(lease)
        self.assertEqual(calls,['lease','sources','history','lease','sources'])

    def test_service_install_ack_is_not_worker_ack(self):
        service,host,spec,ref=service_fixture();service.launch()
        self.assertTrue(service.attempted);self.assertIsNone(service.owner)
        self.assertIn(str(Path(spec['output_root'])/'launch-intent.json'),host.publications)
        self.assertNotIn(str(Path(spec['output_root'])/'worker-owner.json'),host.publications)

    def test_env_to_python_same_identity_is_independently_acknowledged(self):
        service,host,spec,ref=service_fixture()
        host.steps=[(props(spec,ref,'activating'),process(spec,ref,True)),(props(spec,ref),process(spec,ref))]
        service.launch();owner=service.acknowledge(lambda:None)
        self.assertEqual(owner['process'],process(spec,ref));self.assertEqual(owner['admission'],ref)
        self.assertEqual(host.publications[str(Path(spec['output_root'])/'worker-owner.json')],owner)

    def test_env_to_python_pid_or_start_replacement_fails(self):
        service,host,spec,ref=service_fixture();wrong=process(spec,ref);wrong['start_ticks']='999'
        host.steps=[(props(spec,ref,'activating'),process(spec,ref,True)),(props(spec,ref),wrong)]
        service.launch()
        with self.assertRaises(ValueError):service.acknowledge(lambda:None)
        self.assertIsNone(service.owner)

    def test_foreign_startup_argv_or_group_fails(self):
        for key,value in [('argv',['foreign']),('cgroup','0::/foreign')]:
            service,host,spec,ref=service_fixture();wrong=process(spec,ref);wrong[key]=value
            host.steps=[(props(spec,ref),wrong)];service.launch()
            with self.assertRaises(ValueError):service.acknowledge(lambda:None)

    def test_changed_invocation_cannot_be_adopted(self):
        service,host,spec,ref=service_fixture();service.launch();service.acknowledge(lambda:None)
        host.steps=[(props(spec,ref,invocation='b'*32),process(spec,ref))]
        with self.assertRaises(ValueError):service.observe()

    def test_failed_exit_before_ack_is_retained_without_fabricated_owner(self):
        service,host,spec,ref=service_fixture()
        host.steps=[(props(spec,ref,'terminal',status=7),None)];service.launch()
        with self.assertRaises(ValueError):service.acknowledge(lambda:None)
        proof=service.drained()
        self.assertTrue(proof['exit_observed']);self.assertIsNone(proof['owner'])
        self.assertFalse(proof['unacknowledged']['actual_start_ticks_observed'])
        self.assertEqual(proof['samples'][0]['status']['ExecMainStatus'],7)

    def test_never_created_service_has_no_exit_claim(self):
        service,host,spec,ref=service_fixture();host.steps=[(props(spec,ref,'not-found'),None)]
        service.launch();proof=service.drained()
        self.assertTrue(proof['physically_drained']);self.assertFalse(proof['exit_observed'])
        self.assertEqual(len(host.groups),2)

    def test_no_process_startup_failure_is_not_an_actual_process_exit(self):
        service,host,spec,ref=service_fixture();value=props(spec,ref,'terminal',status=1)
        value.update(ExecMainPID='0',ExecMainCode='0',ExecMainStatus='0',Result='resources')
        host.steps=[(value,None)];service.launch();proof=service.drained()
        self.assertTrue(proof['physically_drained']);self.assertFalse(proof['exit_observed'])
        self.assertEqual(proof['samples'][0]['status']['Result'],'resources')

    def test_captured_service_disappearance_does_not_prove_exit(self):
        service,host,spec,ref=service_fixture();service.launch();service.acknowledge(lambda:None)
        host.steps=[(props(spec,ref,'not-found'),None)]
        with self.assertRaises(ValueError):service.drained()

    def test_actual_success_requires_process_gone_and_two_empty_samples(self):
        service,host,spec,ref=service_fixture();service.launch();service.acknowledge(lambda:None)
        host.steps=[(props(spec,ref,'terminal'),None)];proof=service.drained()
        self.assertTrue(proof['physically_drained']);self.assertTrue(proof['exit_observed'])
        self.assertEqual(len(host.groups),2)
        self.assertEqual(proof['samples'][0]['status'],dict(ExecMainCode=1,ExecMainStatus=0,Result='success'))

    def test_nonempty_recursive_group_blocks_drain(self):
        service,host,spec,ref=service_fixture();service.launch();service.acknowledge(lambda:None)
        host.steps=[(props(spec,ref,'terminal'),None)];host.empty_error=RuntimeError('descendant still alive')
        with self.assertRaises(RuntimeError):service.drained()

    def test_unstable_second_observation_blocks_release_proof(self):
        service,host,spec,ref=service_fixture();service.launch();service.acknowledge(lambda:None)
        first=props(spec,ref,'terminal');second=dict(first,Result='timeout')
        host.steps=[(first,None),(second,None)]
        with self.assertRaises(ValueError):service.drained()

    def test_cleanup_signals_only_bound_group_and_retains_terminal_metadata(self):
        service,host,spec,ref=service_fixture();service.launch();service.acknowledge(lambda:None)
        original=host.command
        def signal_command(argv,timeout=30):
            original(argv,timeout)
            if '--signal=SIGTERM' in argv:
                value=props(spec,ref,'terminal');value.update(ExecMainCode='2',ExecMainStatus='15',Result='signal')
                host.steps=[(value,None)]
        host.command=signal_command
        with mock.patch.object(outer,'read',side_effect=host.read):proof=service.cleanup()
        self.assertTrue(proof['physically_drained']);self.assertTrue(proof['stop_requested'])
        self.assertEqual(proof['samples'][0]['status']['ExecMainStatus'],15)
        self.assertTrue(any('--kill-who=all' in command for command in host.commands))
        self.assertFalse(any('stop' in command or 'pkill' in command for command in host.commands))

    def test_cleanup_never_signals_replaced_invocation(self):
        service,host,spec,ref=service_fixture();service.launch();service.acknowledge(lambda:None)
        host.steps=[(props(spec,ref,invocation='b'*32),process(spec,ref))]
        with mock.patch.object(outer,'read',side_effect=host.read):
            with self.assertRaises(ValueError):service.cleanup()
        self.assertEqual(len(host.commands),1)

    def test_pending_cleanup_and_reporting_interruptions_do_not_unlock(self):
        order=[];attempt=[0]
        def verify():
            attempt[0]+=1;order.append('verify')
            if attempt[0]<3:raise RuntimeError('still live')
            return dict(physically_drained=True)
        def wait(seconds):
            order.append('wait')
            if attempt[0]==1:raise KeyboardInterrupt()
        def publish(*args):order.append('write');raise OSError('full disk')
        host=SimpleNamespace(hold=retain_until_drained,wait=wait,publish=publish)
        lease=SimpleNamespace(release=lambda proof:order.append('unlock'))
        service=SimpleNamespace(cleanup=lambda:order.append('cleanup'),drained=verify)
        proof=outer.release_after_drain(host,lease,service,Path('/unused/mock'))
        self.assertTrue(proof['physically_drained']);self.assertEqual(order[-1],'unlock')
        self.assertEqual(order.count('verify'),3);self.assertEqual(order.count('unlock'),1)

    def test_no_launch_failure_can_release_without_claiming_process_exit(self):
        seen=[];host=SimpleNamespace(hold=retain_until_drained,wait=lambda _:None,publish=lambda *args:None)
        lease=SimpleNamespace(release=lambda proof:seen.append(proof))
        outer.release_after_drain(host,lease,None,None)
        self.assertFalse(seen[0]['exit_observed']);self.assertTrue(seen[0]['no_service_launch_attempted'])

    def test_invalid_drain_does_not_reach_real_unlock(self):
        lease=object.__new__(outer.RetainedLease);lease.fd=123
        lease.check=lambda:self.fail('invalid proof must fail before actual descriptor observation')
        with self.assertRaises(ValueError):lease.release(dict(physically_drained=False))

    def test_atomic_metadata_publication_is_complete_and_never_reused(self):
        with tempfile.TemporaryDirectory(prefix='spark-outer-metadata-') as directory:
            root=Path(directory);(root/'tmp').mkdir()
            path=root/'worker-owner.json';value=dict(process={'pid':1},fixture=True)
            ref=outer.immutable_publish(path,value)
            self.assertEqual(outer.read(ref),value);self.assertEqual(path.stat().st_nlink,1)
            with self.assertRaises(ValueError):outer.immutable_publish(path,dict(replacement=True))

    def test_original_history_transport_is_used_without_new_small_cap(self):
        value={'original':'history'};ref=dict(path='/virtual/original-large-history.json',sha256='f'*64,bytes=32*2**20)
        seen=[]
        serial=SimpleNamespace(identity=lambda path:dict(path=path,fixture=True),
            sources=SimpleNamespace(load=lambda r:seen.append(r) or copy.deepcopy(value)))
        reader=outer.HistoricalReader(SimpleNamespace(serial=serial),lambda:None)
        self.assertEqual(reader.read(ref),value);reader.unchanged();self.assertEqual(seen,[ref])

    def test_original_worker_budget_and_proof_limits_remain_explicit(self):
        self.assertEqual(outer.worker.BUDGET['output_total_bytes'],48*2**20)
        self.assertEqual(outer.worker.BUDGET['receipt_total_bytes'],16*2**20)
        self.assertEqual(outer.worker.BUDGET['single_file_bytes'],8*2**20)
        self.assertEqual(outer.OUTER_LIMITS['address_space_bytes'],2*2**30)
        self.assertNotIn('cpu_seconds',outer.OUTER_LIMITS)


if __name__=='__main__':
    unittest.main(verbosity=2)
