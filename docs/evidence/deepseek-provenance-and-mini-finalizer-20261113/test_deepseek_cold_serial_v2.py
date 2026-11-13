"""New lifecycle/provenance interfaces only; no rerun of frozen v1 test suite."""
import copy
from io import BytesIO
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace,ModuleType
import sys
import unittest
from unittest.mock import patch,Mock
from deepseek_cold_serial_v2 import contract as c,bootstrap as b,checkpoint as cp,lifecycle as life,runtime as rt,service
from render_deepseek_cold_serial_v2 import PINS,render
# Fixture builder only. The 25 frozen v1 test methods are not included/run.
from test_deepseek_cold_serial_v1 import fixture as old_fixture,policy
Q=Path(__file__).resolve().parent
SHA='a'*64
def ref(path):return dict(path=str(path),sha256=SHA)
def blob(value):return json.dumps(value,sort_keys=True).encode()
def hash_bytes(data):return hashlib.sha256(data).hexdigest()

def fixture():
    spec,binding,review=old_fixture();spec['schema']='deepseek_cold_serial_spec_v2'
    spec['original_case']['model_path']=str(cp.VIEW);spec['original_checkpoint_manifest']=ref('/synthetic/checkpoint-manifest.json')
    spec['source_sha256'].update({spec['original_checkpoint_manifest']['path']:SHA,'/synthetic/original-authority.json':SHA})
    review.update(schema='deepseek_cold_serial_root_review_v2',original_checkpoint_manifest=spec['original_checkpoint_manifest'])
    return spec,binding,review

def lifecycle_fixture():
    spec,spec_ref,review=fixture();admission_ref=ref('/synthetic/control/admission.json');coordinator_ref=ref('/synthetic/control/coordinator-owner.json')
    admission=dict(schema='deepseek_cold_serial_admission_v2',spec=spec_ref,review=ref('/synthetic/review.json'),
        outer_owner=dict(pid=101,start_ticks='50',ppid=1,argv=['parent'],cgroup='root'),lock_identity=dict(c.LOCK_IDENTITY),
        units=c.units(spec['run_name']),current_serial_catalog=spec['current_serial_catalog'],
        checkpoint_provenance=ref('/synthetic/control/original-checkpoint-provenance.json'),freshness=dict(latest_handoff=ref('/synthetic/idle.json')))
    unit=admission['units']['coordinator']
    coordinator=dict(process=dict(pid=102,start_ticks='51',ppid=100,argv=service.argv('coordinator',admission_ref),
        cgroup='0::/synthetic/'+unit+'.service'),service_owner=dict(unit=unit,pid=102,start_ticks=51,invocation_id='1'*32))
    previous=dict(schema='actual_idle_fixture',updated_at='2026-11-13T09:00:00+00:00',serial_execution=True,running=False,
        active_serial=None,live_tool_sessions=[],active_heavy_units=[],complete_history={'failed_original_exit':1,'all_rows':list(range(50))},
        other_models={'qwen':{'qualified':False},'minicpm':{'unchanged':'fixture'}})
    tool=dict(kind='tool_session',session_id=12345)
    return previous,tool,admission_ref,admission,coordinator_ref,coordinator,spec

class Lifecycle(unittest.TestCase):
    def running(self):
        previous,tool,binding,admission,owner_ref,owner,spec=lifecycle_fixture()
        return life.expected_running(previous,'2026-11-13T10:00:00+00:00',tool,binding,admission,owner_ref,owner)
    def test_real_idle_fields_become_complete_deepseek_running(self):
        previous,tool,binding,admission,owner_ref,owner,spec=lifecycle_fixture();saved=copy.deepcopy(previous)
        value=life.expected_running(previous,'2026-11-13T10:00:00+00:00',tool,binding,admission,owner_ref,owner)
        self.assertTrue(value['running']);self.assertEqual(previous,saved)
        execution=value['deepseek_cold_serial_current_execution']
        self.assertEqual(value['active_serial'],life.active_descriptor(execution))
        self.assertEqual(value['active_serial']['kind'],'deepseek_cold_serial')
        self.assertEqual(execution['tool'],tool);self.assertEqual(execution['coordinator'],owner)
        self.assertEqual(execution['current_serial_catalog'],admission['current_serial_catalog'])
        self.assertEqual(execution['checkpoint_provenance'],admission['checkpoint_provenance'])
        self.assertEqual(value['live_tool_sessions'],[12345]);self.assertEqual(value['active_heavy_units'],[owner['service_owner']['unit']])
        self.assertNotIn('qwen35_sampler_current_execution',value)
        changed={'schema','updated_at','running','active_serial','previous','publication_only_since_bound_terminal',
            'live_tool_sessions','active_heavy_units','deepseek_cold_serial_current_execution'}
        self.assertEqual({k:v for k,v in value.items()if k not in changed},{k:v for k,v in previous.items()if k not in changed})
        value['complete_history']['all_rows'].append('new');self.assertEqual(previous,saved)
    def test_nonidle_or_replaced_owner_cannot_start_new_running(self):
        for bad in('running','active_serial','live_tools','owner'):
            previous,tool,binding,admission,owner_ref,owner,spec=lifecycle_fixture()
            if bad=='running':previous['running']=True
            if bad=='active_serial':previous['active_serial']={'other':'job'}
            if bad=='live_tools':previous['live_tool_sessions']=[9]
            if bad=='owner':owner['process']['start_ticks']='99'
            with self.assertRaises(ValueError):life.expected_running(previous,'2026-11-13T10:00:00+00:00',tool,binding,admission,owner_ref,owner)
    def test_actual_exit_required_and_never_self_publish_idle(self):
        value=self.running();saved=copy.deepcopy(value);execution=value['deepseek_cold_serial_current_execution']
        terminal=dict(admission=execution['admission'],outer_owner=execution['outer_owner'],self_exit_claimed=False,
            actual_external_tool_exit_pending=True,whole_owned_scope_drained=True)
        actual=dict(kind='tool_session',session_id=12345,exit_code=0,terminal_chunk='abcd12')
        for owner_gone,drained in((False,False),(True,False),(False,True)):
            with self.assertRaises(ValueError):life.require_external_exit(value,terminal,actual,owner_gone=owner_gone,all_owned_drained=drained)
        with self.assertRaises(ValueError):life.require_external_exit(value,terminal,{},owner_gone=True,all_owned_drained=True)
        proof=life.require_external_exit(value,terminal,actual,owner_gone=True,all_owned_drained=True)
        self.assertFalse(proof['idle_publication_authorized']);self.assertEqual(value,saved);self.assertTrue(value['running'])
    def test_bootstrap_ack_rejects_old_false_idle_shape(self):
        previous,tool,binding,admission,owner_ref,owner,spec=lifecycle_fixture()
        running=life.expected_running(previous,'2026-11-13T10:00:00+00:00',tool,binding,admission,owner_ref,owner)
        running_ref=ref('/synthetic/running.json');observation={'synthetic':'current'}
        ack=dict(admission=binding,coordinator_owner=owner_ref,coordinator=owner,
            running=dict(reference=running_ref,value=running,observation=observation))
        latest=ModuleType('research.minicpm5_layer1_operator_sdk_runtime_completion_v4')
        latest.latest_handoff=lambda:observation
        package=ModuleType('research');package.minicpm5_layer1_operator_sdk_runtime_completion_v4=latest
        def read(value):
            if value==running_ref:return running
            if value==admission['freshness']['latest_handoff']:return previous
            return ack
        with (patch.object(b,'ref',side_effect=lambda path:owner_ref if path.name=='coordinator-owner.json'else ref(path)),
             patch.object(b,'read',side_effect=read),patch.dict(sys.modules,{'research':package,latest.__name__:latest})):
            self.assertEqual(b.authenticate_running(binding,admission,spec,owner)['value'],running)
            running['running']=False;running['active_serial']=None
            with self.assertRaisesRegex(ValueError,'Full inherited'):b.authenticate_running(binding,admission,spec,owner)

def identity(inode,size,*,mode=0o100600,nlink=1):
    return dict(device=1,inode=inode,mode=mode,nlink=nlink,size=size,mtime_ns=2,ctime_ns=3,uid=1000,gid=1000)

def checkpoint_fixture():
    spec,spec_ref,review=fixture()
    original=dict(model_type='qwen3',architectures=['Qwen3ForCausalLM'],torch_dtype='bfloat16',rope_scaling={'factor':4})
    index=dict(metadata=dict(total_size=cp.PARAMETERS*2),weight_map={'embed':'model-00001-of-00002.safetensors',
        'head':'model-00002-of-00002.safetensors'})
    original_files={'config.json':blob(original),'model.safetensors.index.json':blob(index),'tokenizer.json':b'{"tokens":[1,2]}',
        'tokenizer_config.json':b'{"template":"original"}','generation_config.json':b'{"temperature":1}',
        'model-00001-of-00002.safetensors':b'fake-complete-shard-one','model-00002-of-00002.safetensors':b'fake-complete-shard-two'}
    data={str(cp.OFFICIAL/name):value for name,value in original_files.items()}
    data[str(cp.VIEW/'config.json')]=blob(dict(original,model_type='deepseek_qwen3_neuron_yarn',architectures=['DeepSeekQwen3NeuronYarnForCausalLM']))
    physical={};logical={}
    for number,(path,value)in enumerate(data.items(),10):
        shared=path.endswith('model-00001-of-00002.safetensors')
        expected=identity(number,len(value),nlink=2 if shared else 1)
        physical[path]=dict(sha256=hash_bytes(value),bytes=len(value),identity=expected,hardlink_paths={path:copy.deepcopy(expected)})
        logical[path]=dict(kind='regular',physical=path,identity=copy.deepcopy(expected),link_target=None)
    for number,name in enumerate(['tokenizer.json','tokenizer_config.json','model.safetensors.index.json',
        'model-00001-of-00002.safetensors','model-00002-of-00002.safetensors'],100):
        path=str(cp.VIEW/name);target=str(cp.OFFICIAL/name)
        if '00001' in name:
            expected=copy.deepcopy(physical[target]['identity']);physical[target]['hardlink_paths'][path]=expected
            logical[path]=dict(kind='hardlink',physical=target,identity=copy.deepcopy(expected),link_target=None)
        else:logical[path]=dict(kind='symlink',physical=target,identity=identity(number,len(target),mode=0o120777),link_target=target)
    authority=dict(schema='deepseek_original_source_file_hashes_v1',repo=cp.REPO,revision=cp.REVISION,
        files={name:dict(bytes=len(value),sha256=hash_bytes(value))for name,value in original_files.items()})
    manifest=dict(schema='deepseek_original_checkpoint_manifest_v1',repo=cp.REPO,revision=cp.REVISION,official_root=str(cp.OFFICIAL),
        view_root=str(cp.VIEW),authority=ref('/synthetic/original-authority.json'),logical_files=logical,physical_files=physical,
        directory_identities={str(cp.OFFICIAL):identity(1,0,mode=0o40700,nlink=2),str(cp.VIEW):identity(2,0,mode=0o40700,nlink=2)})
    return spec,spec_ref,manifest,authority,data

class FakeFiles:
    def __init__(self,manifest,data):
        self.manifest=copy.deepcopy(manifest);self.data=copy.deepcopy(data);self.hash_calls=[]
        self.identities=copy.deepcopy(manifest['directory_identities'])
        for row in manifest['physical_files'].values():self.identities.update(copy.deepcopy(row['hardlink_paths']))
        self.identities.update({path:copy.deepcopy(row['identity'])for path,row in manifest['logical_files'].items()})
        self.links={path:row['link_target']for path,row in manifest['logical_files'].items()if row['kind']=='symlink'}
    def identity(self,path):return copy.deepcopy(self.identities[path])
    def inventory(self,root):return sorted(path for path in self.manifest['logical_files']if str(Path(path).parent)==root)
    def link(self,path):return self.links[path]
    def hash(self,path,expected,guard):self.hash_calls.append(path);return cp.stream_hash(BytesIO(self.data[path]),expected['bytes'],guard)
    def small(self,path,expected):return self.data[path]

class Checkpoint(unittest.TestCase):
    def authenticate(self,mutate=None):
        spec,binding,manifest,authority,data=checkpoint_fixture();files=FakeFiles(manifest,data)
        if mutate:mutate(spec,manifest,authority,files)
        def read(value):return manifest if value==spec['original_checkpoint_manifest']else authority
        with patch.object(cp,'read',side_effect=read):
            checked=cp.authenticate_checkpoint(spec,binding,{'pid':101},lambda:None,files=files)
        return checked,files,spec,binding,manifest,authority
    def test_all_bytes_hashed_once_per_inode_with_symlinks_and_hardlinks(self):
        checked,files,*_=self.authenticate()
        self.assertEqual(len(files.hash_calls),len(checked.manifest['physical_files']))
        self.assertEqual(len(files.hash_calls),len(set(files.hash_calls)))
        self.assertLess(len(files.hash_calls),len(checked.manifest['logical_files']))
        self.assertTrue(checked.proof['all_file_bytes_authenticated']);self.assertFalse(checked.proof['loaded_weight_layout_proven'])
        self.assertFalse(checked.proof['graph_weight_ABI_proven']);self.assertTrue(checked.check())
    def test_full_byte_mismatch_rejected_despite_unchanged_identity(self):
        def mutate(spec,manifest,authority,files):
            key=str(cp.OFFICIAL/'model-00001-of-00002.safetensors');files.data[key]=b'X'+files.data[key][1:]
        with self.assertRaisesRegex(ValueError,'physical bytes disagree'):self.authenticate(mutate)
    def test_truncated_or_growing_physical_payload_rejected(self):
        for suffix in(-1,1):
            def mutate(spec,manifest,authority,files):
                key=str(cp.OFFICIAL/'model-00001-of-00002.safetensors');files.data[key]=files.data[key][:-1]if suffix==-1 else files.data[key]+b'X'
            with self.assertRaisesRegex(ValueError,'extent|truncation'):self.authenticate(mutate)
    def test_unaccounted_hardlink_rejected_without_disallowing_declared_sharing(self):
        def mutate(spec,manifest,authority,files):
            manifest['physical_files'][str(cp.OFFICIAL/'model-00001-of-00002.safetensors')]['identity']['nlink']=3
        with self.assertRaisesRegex(ValueError,'hardlink is explicitly'):self.authenticate(mutate)
    def test_retargeted_symlink_or_changed_shared_inode_revokes_future_guard(self):
        for mode in('symlink','hardlink'):
            checked,files,*_=self.authenticate()
            if mode=='symlink':files.links[str(cp.VIEW/'tokenizer.json')]='/unreviewed/arbitrary'
            else:files.identities[str(cp.VIEW/'model-00001-of-00002.safetensors')]['inode']=999
            with self.assertRaisesRegex(ValueError,'changed'):checked.check()
    def test_missing_original_metadata_or_shard_cannot_be_hidden(self):
        for name in('tokenizer.json','generation_config.json','model-00002-of-00002.safetensors'):
            def mutate(spec,manifest,authority,files):del manifest['logical_files'][str(cp.OFFICIAL/name)]
            with self.assertRaisesRegex(ValueError,'Complete official'):self.authenticate(mutate)
    def test_unauthenticated_manifest_or_wrong_original_revision_rejected(self):
        for mode in('authority','revision'):
            def mutate(spec,manifest,authority,files):
                if mode=='authority':del spec['source_sha256'][manifest['authority']['path']]
                else:authority['revision']='0'*40
            with self.assertRaises(ValueError):self.authenticate(mutate)
    def test_config_math_change_rejected_even_if_view_bytes_have_matching_hash(self):
        spec,binding,manifest,authority,data=checkpoint_fixture();path=str(cp.VIEW/'config.json')
        value=json.loads(data[path]);value['rope_scaling']['factor']=8;data[path]=blob(value)
        row=manifest['physical_files'][path];row.update(bytes=len(data[path]),sha256=hash_bytes(data[path]))
        row['identity']['size']=len(data[path]);row['hardlink_paths'][path]['size']=len(data[path]);manifest['logical_files'][path]['identity']['size']=len(data[path])
        files=FakeFiles(manifest,data)
        with patch.object(cp,'read',side_effect=lambda value:manifest if value==spec['original_checkpoint_manifest']else authority):
            with self.assertRaisesRegex(ValueError,'registration identity'):cp.authenticate_checkpoint(spec,binding,{'pid':101},lambda:None,files=files)
    def test_worker_consumes_complete_parent_proof_not_review_boolean(self):
        checked,files,spec,binding,manifest,authority=self.authenticate();proof_ref=ref('/synthetic/control/original-checkpoint-provenance.json')
        def read(value):return checked.proof if value==proof_ref else manifest if value==spec['original_checkpoint_manifest']else authority
        with patch.object(cp,'read',side_effect=read),patch.object(cp,'Files',return_value=files):
            self.assertTrue(cp.load_bound_proof(spec,binding,{'pid':101},proof_ref,lambda:None).check())
            self.assertEqual(len(files.hash_calls),len(manifest['physical_files']),'Worker never rehashes all physical weights')
            checked.proof['unique_physical_files'].pop(next(iter(checked.proof['unique_physical_files'])))
            with self.assertRaisesRegex(ValueError,'partial weight proof'):cp.load_bound_proof(spec,binding,{'pid':101},proof_ref,lambda:None)
    def test_initial_guard_revocation_precedes_manifest_or_payload_read(self):
        spec,binding,manifest,authority,data=checkpoint_fixture()
        with patch.object(cp,'read',side_effect=AssertionError('No read after lost lease')):
            with self.assertRaisesRegex(ValueError,'lost lease'):
                cp.authenticate_checkpoint(spec,binding,{},lambda:(_ for _ in()).throw(ValueError('lost lease')),files=FakeFiles(manifest,data))
    def test_real_nofollow_stream_reader_accepts_explicit_hardlinked_inode(self):
        data=b'complete original bytes';expected=dict(bytes=len(data),identity=identity(10,len(data),nlink=2))
        info=SimpleNamespace(**{field:expected['identity'][key]for key,field in zip(cp.ATTRS,cp.STAT_ATTRS)})
        files=cp.Files()
        with (patch.object(files,'identity',return_value=expected['identity']),patch.object(cp.os,'open',return_value=91)as opened,
            patch.object(cp.os,'fstat',return_value=info),patch.object(cp.os,'fdopen',return_value=BytesIO(data)),patch.object(cp.os,'close')as closed):
            actual=files.hash('/synthetic/original-shard',expected,lambda:None)
        self.assertEqual(actual,dict(bytes=len(data),sha256=hash_bytes(data)))
        self.assertTrue(opened.call_args.args[1]&cp.os.O_NOFOLLOW);closed.assert_called_once_with(91)
    def test_real_stream_reader_refuses_identity_change_during_full_hash(self):
        data=b'whole bytes';expected=dict(bytes=len(data),identity=identity(10,len(data),nlink=2));changed=dict(expected['identity'],ctime_ns=99)
        info=SimpleNamespace(**{field:expected['identity'][key]for key,field in zip(cp.ATTRS,cp.STAT_ATTRS)})
        files=cp.Files()
        with (patch.object(files,'identity',side_effect=[expected['identity'],changed]),patch.object(cp.os,'open',return_value=91),
            patch.object(cp.os,'fstat',return_value=info),patch.object(cp.os,'fdopen',return_value=BytesIO(data)),patch.object(cp.os,'close')as closed):
            with self.assertRaisesRegex(ValueError,'unchanged after'):files.hash('/synthetic/original-shard',expected,lambda:None)
        closed.assert_called_once_with(91)
    def test_identity_fields_come_from_one_atomic_stat_snapshot(self):
        before=identity(10,100,nlink=2);after=dict(before,inode=99,ctime_ns=999)
        snapshots=[SimpleNamespace(**{field:value[key]for key,field in zip(cp.ATTRS,cp.STAT_ATTRS)})for value in(before,after)]
        with (patch.object(Path,'resolve',lambda path:path),patch.object(Path,'is_symlink',return_value=False),
            patch.object(Path,'lstat',side_effect=snapshots*5)):
            self.assertEqual(cp.Files().identity('/synthetic/shared-shard'),before)

class Integration(unittest.TestCase):
    def test_mechanical_successor_and_frozen_service_phase_bodies(self):
        inputs={name:(Q/'deepseek_cold_serial_v1'/name).read_text()for name in PINS}
        values=render(inputs,(Q/'minicpm5_retained_replay_contract_v2.py').read_text())
        for name,text in values.items():self.assertEqual((Q/'deepseek_cold_serial_v2'/name).read_text(),text)
        self.assertEqual(values['service.py'],inputs['service.py']);self.assertEqual(values['phase.py'],inputs['phase.py'])
    def test_full_manifest_binding_mandatory_despite_root_review_true(self):
        spec,binding,review=fixture();c.validate_spec(spec,review,binding)
        del spec['original_checkpoint_manifest']
        with self.assertRaises(ValueError):c.validate_spec(spec,review,binding)
    def test_original_pre_cost_layout_blocker_retained(self):
        with patch.object(rt,'read',side_effect=AssertionError('No inputs before missing-layout refusal')),patch.object(rt,'RetainedLock',side_effect=AssertionError('No lock')):
            with self.assertRaisesRegex(ValueError,'proof producer'):rt.outer(ref('/synthetic/spec.json'),ref('/synthetic/review.json'))
    def test_fast_guard_requires_provenance_and_checks_identity_without_payload_rehash(self):
        spec,_,_=fixture();observer=b.Observer(dict(outer_owner={},retained_fd=9,deadline_monotonic=100),spec,policy(spec))
        checkpoint=SimpleNamespace(check=Mock());observer.checkpoint=checkpoint
        with (patch.object(b,'descriptor'),patch.object(b.time,'monotonic',return_value=1),patch.object(b,'resources'),
            patch.object(b,'process_api',return_value=SimpleNamespace(active_heavy=Mock())),patch.object(b,'loaded_local_sources')):
            for _ in range(10):observer.check()
            self.assertEqual(checkpoint.check.call_count,1)
            observer.check(full=True);self.assertEqual(checkpoint.check.call_count,2)
            observer.checkpoint=None
            with self.assertRaisesRegex(ValueError,'checkpoint byte proof'):observer.check(full=True)

if __name__=='__main__':unittest.main(verbosity=2)
