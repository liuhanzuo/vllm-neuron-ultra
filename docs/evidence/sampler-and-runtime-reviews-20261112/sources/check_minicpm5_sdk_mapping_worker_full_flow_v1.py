"""Execute the complete worker source_entry body against synthetic original ports.

Every model/native/descriptor/IO boundary is fake. These tests execute actual
integration control flow, input binding and failure propagation, not AST slices.
"""
import copy
from contextlib import ExitStack
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


class RejectNative:
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in ('ctypes','_ctypes','torch','numpy','neuronx_distributed','neuronx_distributed_inference'):
            raise AssertionError('No actual native import in full source_entry mock')


sys.meta_path.insert(0,RejectNative())
from research import minicpm5_sdk_mapping_worker_source_entry_v3 as worker


def ref(name):return dict(path='/synthetic/'+name+'.json',sha256='a'*64)


class Flow:
    def __init__(self):
        self.calls=[];self.failure_at=None;self.result_code=0;self.summary=None;self.gate=None
        site='/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages'
        self.sdk_plan=dict(role='sdk310',bootstrap_sources={str(ROOT/'research/worker_fixture.py'):'b'*64},
            native_authority=dict(path='/synthetic/sdk-native.json',sha256='c'*64,role='sdk310',
                python='/home/ubuntu/venvs/vllm_neuron/bin/python'))
        pins={**self.sdk_plan['bootstrap_sources'],site+'/torch/__init__.py':'d'*64,
            site+'/neuronx_distributed_inference/__init__.py':'e'*64}
        self.pins=pins
        self.native_nine={f'/synthetic/native-{i}.so':str(i)*64 for i in range(9)}
        self.owner=dict(pid=321,start_ticks='child',ppid=123,cgroup='0::/fake',argv=['python','-c','fake'])
        self.outer=dict(pid=123,start_ticks='parent',ppid=1,cgroup='0::/fake',argv=['outer','-c','fake'])
        self.refs={name:ref(name) for name in ('admission','spec','review','original-native','outer-spec',
            'outer-review','outer-entry','worker-template','sdk-source','pv-spec','pv-review')}
        self.witness=dict(complete_source_map_sha256=worker.native.history.digest(pins),complete_source_files=len(pins),
            spec=self.refs['pv-spec'],review=self.refs['pv-review'])
        self.spec=dict(worker='/synthetic/original-worker.py',pv_spec=self.refs['pv-spec'],pv_review=self.refs['pv-review'],
            source_extension=dict(pins))
        self.supplemental=dict(outer_spec=self.refs['outer-spec'],outer_review=self.refs['outer-review'],
            outer_entry_plan=self.refs['outer-entry'],sdk_entry_source=self.refs['sdk-source'],
            original_worker=self.spec['worker'])
        self.admission=dict(witness=self.witness,lease=dict(outer_owner=self.outer,attempt_id='a'*32),
            output_root='/synthetic/output',original_native_source_receipt=self.refs['original-native'],
            diagnostic_spec=self.refs['spec'],diagnostic_review=self.refs['review'],
            source_entry_binding=self.supplemental)
        self.outer_spec=dict(schema='minicpm5_mapping_source_outer_spec_v1',source_extension=pins,
            diagnostic_spec=self.refs['spec'],diagnostic_review=self.refs['review'],limits={'original':True},
            artifact_limits={'original_caps':True},sdk_entry_template=self.refs['worker-template'],
            sdk_entry_source=self.refs['sdk-source'],sdk_import_paths=[site])
        self.outer_review=dict(schema='minicpm5_mapping_source_outer_review_v1',reviewer='/root',passed=True,
            blockers=[],spec=self.refs['outer-spec'],source_extension=pins)
        self.data={self.refs['admission']['path']:self.admission,self.refs['spec']['path']:self.spec,
            self.refs['review']['path']:{'original':'review'},
            self.refs['original-native']['path']:dict(native_sha256=self.native_nine),
            self.refs['outer-spec']['path']:self.outer_spec,self.refs['outer-review']['path']:self.outer_review,
            self.refs['outer-entry']['path']:{'role':'outer312'},self.refs['worker-template']['path']:self.sdk_plan}
        self.packet=dict(source_plan=dict(source_sha256=pins,source_contract=ref('contract')))
        self.descriptor_plan=dict(all_original1284='retained')
        self.entry=NS(role='sdk310',admission_envelope=self.admission,plan=self.sdk_plan,deadline=750.,
            plan_reference=self.refs['admission'],runtime=NS(orig_argv=self.owner['argv'],path=[]),
            bind_sdk_history_owner=lambda *a:self.hit('entry.bind_owner'))
        self.io=NS(_read=False)
        flow=self

        class Bootstrap:
            def __init__(self,entry,admission):
                flow.hit('bootstrap.construct');self.owner=flow.owner
                assert entry is flow.entry and admission is flow.admission
            def start(self):flow.hit('bootstrap.start')
            def read_release(self):flow.hit('bootstrap.release');return flow.packet
            def adopt(self,original):
                flow.hit('bootstrap.adopt');assert original is flow.original
                return object(),flow.guard,flow.metadata

        class Auth:
            def __init__(self,bootstrap,pins,authority):
                flow.hit('native.construct');assert pins is flow.pins
                self.reader=NS(read=lambda reference,cap:(flow.source_read(reference,cap),{}))

        class Gate:
            def __init__(self,entry,auth,bootstrap):flow.hit('gate.construct');flow.gate=self;self.state='new'
            def load(self):flow.hit('gate.load');self.state='history_loaded';return flow.original

        class Finder:
            def __init__(self,auth):pass
            def __enter__(self):flow.hit('sourcefinder.enter');return self
            def __exit__(self,*args):flow.hit('sourcefinder.exit');return False

        class Store:
            def __init__(self,path,guard):
                flow.hit('store.construct');assert path=='/synthetic/output'
            def inventory(self):flow.hit('store.inventory')
            def json(self,name,value):flow.hit('store.summary');flow.summary=(name,value)

        self.Bootstrap,self.Auth,self.Gate,self.Finder=Bootstrap,Auth,Gate,Finder
        self.metadata=NS(read=self.read,unchanged=lambda:self.hit('metadata.unchanged'))
        self.original=NS(r=NS(sources=NS(classify=self.classify,verify_text_sources=lambda *a:self.hit('sources.text'),
            check_native_identities=lambda *a:self.hit('native.original9')),verify_loaded=lambda *a:self.hit('sources.loaded'),
            digest=worker.native.history.digest,life=NS(process=self.process),
            p=NS(environment=lambda *a:{},proc=self.proc)),serial=NS(CAPS={'original_caps':True},PHASES={1:('sdk','own_before')}),
            prior=NS(LIMITS={'original':True}),restore_descriptor_plan=self.restore,
            core=NS(saved=NS(validate_plan=self.validate_plan,StableCaptureIO=self.capture_io)),
            DiagnosticStore=Store,validate_spec=lambda *a:self.hit('spec.original'),execute_import=self.execute)

    def hit(self,name):
        self.calls.append(name)
        if self.failure_at==name:raise RuntimeError('injected:'+name)
    def guard(self):self.hit('guard')
    def read(self,reference):self.hit('metadata.read:'+reference['path']);return self.data[reference['path']]
    def classify(self,pins,contract,read):
        self.hit('sources.classify');assert pins==self.pins
        return dict(categories=dict(query_native_binary_sha256=self.native_nine))
    def restore(self,witness,read):self.hit('plan.restore1284');return self.descriptor_plan
    def validate_plan(self,plan):self.hit('plan.validate1284');assert plan==self.descriptor_plan;return {'complete':1284}
    def capture_io(self,allowed,guard,filesystem):self.hit('io.construct');assert allowed=={'complete':1284};return self.io
    def source_read(self,reference,cap):
        self.hit('entry.source_read');assert reference==self.refs['sdk-source'] and cap==65535;return b'synthetic reviewed entry'
    def argv(self,*args):self.hit('entry.actual_argv');return self.owner['argv']
    def process(self,owner,argv,ppid,cgroup):
        self.hit('owner.original');assert (owner,argv,ppid,cgroup)==(self.owner,self.owner['argv'],123,'0::/fake')
    def proc(self,pid):self.hit('parent.live');assert pid==123;return self.outer
    def execute(self,pins_plan,guard):
        self.hit('execute.original_import');assert pins_plan==self.packet['source_plan']
        assert self.gate.state=='complete_original_authentication'
        return {'diagnostic_passed':self.result_code==0,'complete':'synthetic original summary'},self.result_code
    def run(self):
        class Directory:
            def __init__(self,path):self.path=path
            def resolve(self):return self
            @property
            def parents(self):return ()
            def lstat(self):return NS(st_mode=0o040755)
        with ExitStack() as stack:
            stack.enter_context(patch.object(worker.startup,'SDKEntryGuard',self.Bootstrap))
            stack.enter_context(patch.object(worker.native,'SDKNativeAuthenticator',self.Auth))
            stack.enter_context(patch.object(worker.native,'SDKHistoryGate',self.Gate))
            stack.enter_context(patch.object(worker.native,'SourceFinder',self.Finder))
            stack.enter_context(patch.object(worker.launch,'admitted_sdk',self.argv))
            stack.enter_context(patch.object(worker.launch,'separate_authorities',lambda *a:self.hit('authority.separate')))
            stack.enter_context(patch.object(worker,'Path',Directory))
            stack.enter_context(patch.object(worker,'os',NS(environ={})))
            return worker.source_entry(self.entry,'admission:self')


class Checks(unittest.TestCase):
    def test_complete_worker_happy_flow_preserves_all_original_gates(self):
        f=Flow();self.assertEqual(f.run(),0)
        required=['bootstrap.start','bootstrap.release','entry.bind_owner','gate.load','bootstrap.adopt',
            'sources.classify','sources.text','native.original9','sources.loaded','plan.restore1284',
            'plan.validate1284','io.construct','spec.original','authority.separate','entry.actual_argv',
            'owner.original','parent.live','execute.original_import','metadata.unchanged','store.summary']
        positions=[f.calls.index(name) for name in required]
        self.assertEqual(positions,sorted(positions))
        self.assertEqual(f.summary[0],'phase-1-summary.json')
        self.assertEqual(f.entry.runtime.path,['/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages'])
        self.assertEqual(f.calls[-1],'sourcefinder.exit')

    def test_import_failure_records_original_summary_and_returns_actual_code(self):
        f=Flow();f.result_code=1
        self.assertEqual(f.run(),1)
        self.assertFalse(f.summary[1]['diagnostic_passed'])
        self.assertEqual(f.calls.count('execute.original_import'),1)

    def test_every_preimport_failure_prevents_sdk_import_and_summary(self):
        for boundary in ('bootstrap.start','bootstrap.release','gate.load','bootstrap.adopt',
                         'sources.classify','sources.text','native.original9','plan.restore1284',
                         'plan.validate1284','spec.original','authority.separate','entry.actual_argv','parent.live'):
            f=Flow();f.failure_at=boundary
            with self.subTest(boundary=boundary),self.assertRaisesRegex(RuntimeError,'injected'):
                f.run()
            self.assertNotIn('execute.original_import',f.calls)
            self.assertIsNone(f.summary)

    def test_missing_original_native_member_fails_before_original_import(self):
        f=Flow();f.native_nine.pop(next(iter(f.native_nine)))
        with self.assertRaises(ValueError):f.run()
        self.assertNotIn('execute.original_import',f.calls)

    def test_full_source_map_digest_failure_precedes_native_gate(self):
        f=Flow();f.witness['complete_source_map_sha256']='0'*64
        with self.assertRaises(ValueError):f.run()
        self.assertNotIn('native.construct',f.calls)

    def test_supplemental_unaccepted_review_blocks_import(self):
        f=Flow();f.outer_review['passed']=False
        with self.assertRaises(ValueError):f.run()
        self.assertNotIn('execute.original_import',f.calls)

    def test_wrong_actual_new_argv_is_not_accepted_as_old_worker(self):
        f=Flow();f.entry.runtime.orig_argv=['python','old-worker.py']
        with self.assertRaises(ValueError):f.run()
        self.assertNotIn('execute.original_import',f.calls)

    def test_original_capture_io_remains_zero_read(self):
        f=Flow();f.io._read=True
        with self.assertRaises(ValueError):f.run()
        self.assertIn('execute.original_import',f.calls)
        self.assertIsNone(f.summary)

    def test_post_import_auth_or_persistence_failure_propagates(self):
        for boundary in ('metadata.unchanged','store.summary'):
            f=Flow();f.failure_at=boundary
            with self.assertRaisesRegex(RuntimeError,'injected'):f.run()
            self.assertEqual(f.calls.count('execute.original_import'),1)
            self.assertIsNone(f.summary)
            self.assertEqual(f.calls[-1],'sourcefinder.exit')

    def test_no_native_modules_loaded(self):
        for name in ('ctypes','_ctypes','torch','numpy'):
            self.assertNotIn(name,sys.modules)


if __name__=='__main__':unittest.main()
