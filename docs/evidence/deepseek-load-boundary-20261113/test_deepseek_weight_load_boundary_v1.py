"""SOURCE ONLY: real boundary/observer/descriptor bodies, fake SDK and archive IO."""
import ast
import builtins
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import stat
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

R=Path(__file__).resolve().parent.parent;Q=R/'research';sys.path[:0]=[str(R),str(Q)]
import deepseek_weight_load_boundary_v1 as boundary
import deepseek_weight_load_observer_v2 as observer
import deepseek_weight_serialized_descriptor_v1 as descriptor
import deepseek_cold_control_device_entry_v2 as off
import deepseek_cold_boundary_device_entry_v2 as on
import deepseek_capture_graph_load_boundary_v1 as loader
import deepseek_capture_graph_entry_v1 as graph
import deepseek_cold_kv_protocol_v2 as protocol
import test_deepseek_weight_load_observer_v2 as tensor_fixture
import test_deepseek_weight_serialized_descriptor_v1 as archive_fixture
import test_deepseek_cold_control_device_entry_v1 as off_fixture


class Fixture:
    def __init__(self,arm):
        self.arm=arm;self.off=off_fixture.Fixture();self.events=[];self.reports=[];self.reads=[];self.calls=[]
        self.error=None;self.missing_key=False;self.archive_changed=False;self.guard_revoked=False
        self.sink_return=None;self.sink_error=None;self.fresh_marker=False
        self.artifact=graph.ORIGINAL_ARTIFACT if arm=='off'else Path('/synthetic/deepseek-capture-artifact')
        self.cache=Path('/synthetic/deepseek-capture-cache');self.work=Path('/synthetic/deepseek-capture-work')
        self.dest=Path('/synthetic/meta');self.reference=self.dest/'cold-reference';self.data=Path('/synthetic/raw')
        self.references={str(self.reference/name):'b'*64 for name in ('cold-reference.pt','cold-reference.json')}
        self.payload=archive_fixture.archive_bytes(archive_fixture.fixture(large_neff=False))
        self.st=NS(st_dev=1,st_ino=2,st_size=len(self.payload),st_mtime_ns=3,st_ctime_ns=4,st_mode=stat.S_IFREG|0o600,st_nlink=1)
        self.binding=dict(schema='deepseek_weight_load_boundary_binding_v1',arm=arm,
            archive=dict(path=str(self.artifact/'model.pt'),sha256=hashlib.sha256(self.payload).hexdigest(),identity=descriptor.file_identity(self.st)),
            descriptor_limits=dict(archive_bytes=32<<20,pickle_bytes=24<<20,members=16,code_bytes=1<<20),
            checkpoint_provenance=dict(original_manifest=dict(path='/synthetic/official-weight-manifest.json',sha256='d'*64),
                original_loader_manifest=dict(path='/synthetic/loader-view-manifest.json',sha256='e'*64)))
        self.expected_sha=self.binding['archive']['sha256'];self.config=deepcopy(self.off.config)
        self.config['_name_or_path']=str(graph.MODEL_PATH)
        self.environment={'synthetic':'exact-original-environment'}
        self.files={str(p):dict(bytes=10,sha256='f'*64)for p in (self.artifact/'model.pt',self.artifact/'neuron_config.json',self.work/'neuron_config.json')}
        self.files.update({str(self.work/tag/('_tp0_bk'+str(i))/name):dict(bytes=10,sha256='f'*64)
            for tag in graph.TAGS for i in range(6)for name in ('neuron_config.json','metaneff.pb','graph.neff')})
        self.files[str(self.artifact/'model.pt')]=dict(bytes=len(self.payload),sha256=self.expected_sha)
        self.report=dict(schema=graph.SCHEMA,complete=True,artifact=str(self.artifact),cache=str(self.cache),workdir=str(self.work),
            environment=self.environment,model_path=str(graph.MODEL_PATH),original_config_sha256=graph.ORIGINAL_CONFIG_SHA256,
            config_sha256=hashlib.sha256(graph.json_bytes(self.config)).hexdigest(),files=self.files,graphs={'all12':'original-inspection'},
            full_model_qualified=False,new_speedup_claim=False,output_semantics_neutrality_proven=False,baseline_weight_layout_equality_proven=False)
        self.report_bytes=json.dumps(self.report).encode();self.report_sha=hashlib.sha256(self.report_bytes).hexdigest()
        owner=self
        class Builder:
            tp_degree=2;start_rank_id=0;local_ranks_size=2
            def shard_checkpoint(self):
                owner.events.append('actual-shard')
                if owner.error=='shard':raise RuntimeError('synthetic original shard error')
                keys=('a.weight','missing.weight'if owner.missing_key else'b.weight')
                self.returned=[{key:tensor_fixture.Tensor()for key in keys}for rank in (0,1)]
                return self.returned
        class Base:
            def load_weights(self,path):
                owner.events.append('actual-load-weights');values=self.builder.shard_checkpoint()
                self.traced_model=NS(nxd_model=NS(weights=values,weight_loader=object()))
                if owner.error=='loader':raise RuntimeError('synthetic original loader error')
                if owner.archive_changed:owner.st.st_ino+=1
                if owner.error=='guard-after-initialize':owner.guard_revoked=True
                owner.events.append('initialize-returned');return 'original-load-weights-return'
        class Config:
            @classmethod
            def from_json_string(cls,text):return NS(to_json_string=lambda:json.dumps(json.loads(text)))
        class Model(self.off.cls,Base):
            def __init__(self,path,config):
                owner.events.append('construct')
                if owner.arm=='off':super().__init__(path,config)
                else:self.config=config
                self.builder=Builder();owner.model=self
            def load(self,path,**kwargs):
                owner.calls.append((path,kwargs));owner.events.append('model-load')
                self.load_weights(path)
                owner.events.append('model-load-returned')
        self.Builder,self.Base,self.Model,self.Config=Builder,Base,Model,Config
        self.old_loader=Base.load_weights;self.old_sharder=Builder.shard_checkpoint
        self.off.cls=Model;self.model=None
        self.off.artifact_hashes[str(self.off.artifact/'model.pt')]=self.expected_sha
    def guard(self):
        self.events.append('guard')
        if self.guard_revoked:raise RuntimeError('synthetic guard revoked')
    def sink(self,value):
        self.events.append('retain-partial');self.reports.append(deepcopy(value))
        if self.sink_error:raise self.sink_error
        value['inference_admitted']=True
        return self.sink_return
    def native(self):
        self.events.append('native');self.fresh_marker=True
        return self.Model if self.arm=='off'else(self.Config,self.Model,None)
    def dependencies(self):
        if not self.fresh_marker:raise AssertionError('Observer installed before original native dependency entry')
        self.events.append('observer-dependencies');return tensor_fixture.TORCH,self.Builder,self.Base
    def record(self,path):
        name=str(path);self.events.append('authenticated-file:'+name)
        if name in self.references:return dict(bytes=10,sha256=self.references[name])
        if name in self.off.reference_hashes:return dict(bytes=10,sha256=self.off.reference_hashes[name])
        if name==str(self.off.artifact/'neuron_config.json'):return dict(bytes=8378,sha256=graph.ORIGINAL_CONFIG_SHA256)
        return deepcopy(self.files[name])
    def stat(self,path):
        if str(path).endswith('model.pt'):return self.st
        return NS(st_size=len(self.report_bytes))
    def patches(self,stack):
        owner=self
        class ArchivePath:
            parents=()
            def __init__(self,value):self.value=value
            def is_absolute(self):return True
            def resolve(self):return self
            def is_symlink(self):return False
            def stat(self):return owner.st
        def archive_open(path,flags):
            owner.reads.append(path.value);owner.events.append('archive-open')
            if path.value!=owner.binding['archive']['path']:raise AssertionError('Unbound actual archive')
            return 321
        for obj,name,value in ((observer,'_dependencies',self.dependencies),(descriptor,'Path',ArchivePath),
            (descriptor.os,'open',archive_open),(descriptor.os,'close',lambda fd:None),(descriptor.os,'fstat',lambda fd:self.st),
            (descriptor.os,'fdopen',lambda *a,**k:io.BytesIO(self.payload)),(Path,'resolve',lambda p:p),
            (Path,'is_dir',lambda p:True),(Path,'is_symlink',lambda p:False),(Path,'exists',lambda p:False),
            (Path,'stat',lambda p:self.stat(p)),(Path,'read_bytes',lambda p:self.report_bytes),
            (off,'_config',lambda *a:deepcopy(self.off.config)),(off,'_native_model_class',self.native),
            (graph,'file_record',self.record),(loader,'file_record',self.record),
            (graph,'check_paths',lambda artifact,cache,workdir,**k:(Path(artifact),Path(cache),Path(workdir))),
            (graph,'check_environment',lambda *a:self.environment),(graph,'original_config',lambda:deepcopy(self.config)),
            (loader,'_native_dependencies',self.native),(loader,'_inspect',lambda *a:(deepcopy(self.files),deepcopy(self.report['graphs']))),
            (loader,'check_environment',lambda *a:self.environment),(protocol,'expected_case',lambda:self.off.case)):
            stack.enter_context(patch.object(obj,name,value))
    def run(self):
        with ExitStack()as stack:
            self.patches(stack)
            if self.arm=='off':
                return off.run(self.off.artifact,self.off.artifact_hashes,self.off.case,self.off.dest,self.off.reference,
                    self.off.data,self.off.original,self.off.reference_hashes,self.guard,
                    weight_load_binding=self.binding,retain_weight_partial=self.sink)
            return on.run(self.artifact,self.cache,self.work,self.report_sha,self.off.case,self.dest,self.reference,self.data,
                self.off.original,self.references,self.guard,weight_load_binding=self.binding,retain_weight_partial=self.sink)


class Integration(unittest.TestCase):
    def test_bound_off_on_comparison_retains_every_partial_flag_and_exact_missing_inputs(self):
        values=[]
        for arm in ('off','on'):
            f=Fixture(arm)
            with self.assertRaises(boundary.LayoutSemanticsUnresolved):f.run()
            values.append(f.reports[0])
        result=boundary.compare_bound_evidence(*values)
        self.assertTrue(result['actual_load_comparison']['pre_layout_full_rank_key_values_equal'])
        self.assertTrue(result['serialized_comparison']['strict_saved_components_equal'])
        self.assertFalse(result['inference_admitted']);self.assertEqual(len(result['next_required_observation']),5)
        values[1]['serialized_descriptor_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'unchanged partial'):boundary.compare_bound_evidence(*values)
    def test_off_on_original_fresh_gates_then_single_real_load_and_bound_descriptor(self):
        for arm in ('off','on'):
            with self.subTest(arm=arm):
                f=Fixture(arm)
                with self.assertRaises(boundary.LayoutSemanticsUnresolved):f.run()
                self.assertEqual(f.calls,[(str(f.artifact),dict(skip_warmup=True))]);self.assertEqual(f.reads,[str(f.artifact/'model.pt')])
                self.assertEqual(f.events.count('actual-shard'),1);self.assertEqual(f.events.count('actual-load-weights'),1)
                self.assertLess(f.events.index('native'),f.events.index('observer-dependencies'))
                self.assertLess(f.events.index('archive-open'),f.events.index('model-load'))
                self.assertLess(f.events.index('initialize-returned'),f.events.index('retain-partial'))
                self.assertNotIn('model-load-returned',f.events);self.assertEqual(len(f.reports),1)
                report=f.reports[0];self.assertEqual(report['load_binding'],f.binding)
                self.assertTrue(report['serialized_descriptor']['whole_archive_sha256_verified'])
                self.assertEqual(len(report['serialized_descriptor']['graphs']),12)
                self.assertTrue(report['all12_declared_weight_keys_observed_in_both_ranks'])
                self.assertFalse(report['inference_admitted']);self.assertFalse(report['layout_semantics_proven'])
                self.assertIs(f.Base.load_weights,f.old_loader);self.assertIs(f.Builder.shard_checkpoint,f.old_sharder)
    def test_preexisting_native_modules_rejected_before_archive_or_new_observer(self):
        for arm in ('off','on'):
            f=Fixture(arm)
            with self.subTest(arm=arm),patch.dict(sys.modules,{'torch':NS()}),self.assertRaisesRegex(ValueError,'[Ff]resh'):
                f.run()
            self.assertFalse(f.calls);self.assertFalse(f.reads);self.assertNotIn('native',f.events)
    def test_truthy_sink_result_cannot_authorize_inference(self):
        for arm in ('off','on'):
            f=Fixture(arm);f.sink_return=dict(inference_admitted=True,layout_semantics_proven=True)
            with self.subTest(arm=arm),self.assertRaises(boundary.LayoutSemanticsUnresolved):f.run()
            self.assertNotIn('model-load-returned',f.events);self.assertFalse(f.reports[0]['inference_admitted'])
    def test_wrong_arm_path_or_archive_digest_rejected_before_sdk_load(self):
        for arm in ('off','on'):
            for change in ('arm','path','sha256'):
                f=Fixture(arm)
                if change=='arm':f.binding['arm']='on'if arm=='off'else'off'
                else:f.binding['archive'][change]='/synthetic/unbound.pt'if change=='path'else'0'*64
                with self.subTest(arm=arm,change=change),self.assertRaises(ValueError):f.run()
                self.assertFalse(f.calls);self.assertFalse(f.reads)
    def test_actual_archive_digest_and_caps_checked_before_original_loader(self):
        for change in ('bytes','cap'):
            f=Fixture('on')
            if change=='bytes':f.payload=f.payload[:-1]+b'X'
            else:f.binding['descriptor_limits']['members']=1
            with self.subTest(change=change),self.assertRaises(ValueError):f.run()
            self.assertFalse(f.calls);self.assertNotIn('observer-dependencies',f.events)
    def test_archive_mutation_during_initialize_rejected_before_sink(self):
        f=Fixture('off');f.archive_changed=True
        with self.assertRaisesRegex(ValueError,'identity changed during'):f.run()
        self.assertFalse(f.reports);self.assertIs(f.Base.load_weights,f.old_loader)
    def test_every_declared_weight_key_must_be_observed_for_both_ranks(self):
        f=Fixture('on');f.missing_key=True
        with self.assertRaisesRegex(ValueError,'Every declared graph weight key'):f.run()
        self.assertFalse(f.reports);self.assertIs(f.Builder.shard_checkpoint,f.old_sharder)
    def test_original_shard_loader_guard_and_sink_exceptions_are_preserved(self):
        for cause in ('shard','loader','guard-after-initialize','sink'):
            f=Fixture('on');f.error=cause
            error=RuntimeError('synthetic sink error')
            if cause=='sink':f.sink_error=error
            with self.subTest(cause=cause),self.assertRaises(RuntimeError)as caught:f.run()
            if cause=='sink':self.assertIs(caught.exception,error)
            self.assertIs(f.Base.load_weights,f.old_loader);self.assertIs(f.Builder.shard_checkpoint,f.old_sharder)
            self.assertNotIn('model-load-returned',f.events)
    def test_partial_evidence_rejects_forged_semantics_or_wrong_wlt_presence(self):
        f=Fixture('off')
        with self.assertRaises(boundary.LayoutSemanticsUnresolved):f.run()
        saved=f.reports[0]
        for change in ('descriptor','observation','wlt'):
            d=deepcopy(saved['serialized_descriptor']);o=deepcopy(saved['actual_load_observation'])
            if change=='descriptor':d['layout_semantics_proven']=True
            elif change=='observation':o['graph_weight_ABI_proven']=True
            else:o['weight_loader_present']=False
            with self.subTest(change=change),self.assertRaises(ValueError):boundary.partial_evidence(f.binding,d,o)


def function(path,name):return next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef)and n.name==name)


class MechanicalPreservation(unittest.TestCase):
    def normalized(self,node,arm):
        node=deepcopy(node);node.args.kwonlyargs=[];node.args.kw_defaults=[]
        result=[]
        for part in node.body:
            if isinstance(part,ast.ImportFrom)and part.module in ('deepseek_weight_load_boundary_v1','deepseek_capture_graph_load_boundary_v1'):continue
            if isinstance(part,ast.With)and ast.unparse(part.items[0].context_expr).startswith('observe_bound_load('):
                result.extend(part.body);continue
            if arm=='on'and isinstance(part,ast.Assign)and isinstance(part.value,ast.Call)and isinstance(part.value.func,ast.Name)and part.value.func.id=='observed_load':
                part.value.func=ast.Attribute(value=ast.Name(id='graph',ctx=ast.Load()),attr='load',ctx=ast.Load());part.value.keywords=[]
            result.append(part)
        node.body=result;return node
    def test_off_complete_scientific_driver_and_all_original_checks_ast_preserved(self):
        old=function(R/'deepseek_cold_control_device_entry_v1.py','run');new=function(R/'deepseek_cold_control_device_entry_v2.py','run')
        self.assertEqual(ast.dump(old),ast.dump(self.normalized(new,'off')))
    def test_on_complete_scientific_driver_and_all_original_checks_ast_preserved(self):
        old=function(R/'deepseek_cold_boundary_device_entry_v1.py','run');new=function(R/'deepseek_cold_boundary_device_entry_v2.py','run')
        self.assertEqual(ast.dump(old),ast.dump(self.normalized(new,'on')))
    def test_capture_loader_fresh_checks_config39_files_and_all_routes_ast_preserved(self):
        old=function(R/'deepseek_capture_graph_entry_v1.py','load');new=function(R/'deepseek_capture_graph_load_boundary_v1.py','load')
        self.assertEqual(ast.dump(old),ast.dump(self.normalized(new,'loader')))
    def test_new_shared_boundary_has_no_model_forward_compile_or_launch(self):
        tree=ast.parse((R/'deepseek_weight_load_boundary_v1.py').read_text())
        names=[ast.unparse(n.func)for n in ast.walk(tree)if isinstance(n,ast.Call)]
        self.assertFalse(any(n.split('.')[-1]in ('forward','compile','Popen','system','flock','unlink','rmtree')for n in names))


if __name__=='__main__':
    real_import=builtins.__import__
    def guarded_import(name,*args,**kwargs):
        if name.split('.')[0]in ('torch','numpy','torch_neuronx','neuronx_distributed','neuronx_distributed_inference','neuronxcc','transformers','vllm','vllm_neuron'):
            raise AssertionError('No actual native imports in source tests')
        return real_import(name,*args,**kwargs)
    builtins.__import__=guarded_import
    def audit(event,args):
        if event in ('subprocess.Popen','os.system','socket.connect','socket.bind','os.listdir','os.scandir','fcntl.flock'):
            raise AssertionError('No actual process/network/namespace/lock operation')
        if event=='open':
            name,mode,flags=args
            if not isinstance(name,str)or not name.endswith(('.py','.pyc')):raise AssertionError('No actual archive/weights/metadata/raw opens')
            if flags&(1|2|64|512):raise AssertionError('No actual file mutations')
    sys.addaudithook(audit);unittest.main(verbosity=2)
