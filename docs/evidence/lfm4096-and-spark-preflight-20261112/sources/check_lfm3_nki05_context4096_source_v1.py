"""Targeted source/mock checks for additive LFM4096; no native or runtime launch."""
import ast
import copy
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import resource
import sys
import types
import unittest
from unittest import mock

class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('torch','torch_neuronx','torch_xla','numpy','ml_dtypes','nki','neuronxcc','ctypes','vllm','vllm_neuron'):
            raise AssertionError('Native import forbidden: '+fullname)
sys.meta_path.insert(0,NoNative())

import lfm3_nki05_device_protocol_v1 as old
import lfm3_nki05_device_protocol_v2 as protocol
import lfm3_nki05_context4096_matrix_v1 as matrix
import lfm3_nki05_context4096_device_leaf_v1 as leaf
import lfm3_nki05_device_pipeline_v2 as pipeline
import lfm3_nki05_device_runtime_v2 as runtime
import run_lfm3_nki05_device_serial_v2 as runner
import check_lfm3_nki05_concrete_device_source_v2 as accepted_test

ROOT=Path(__file__).resolve().parent


def sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def science(task):
    spec=protocol.graph_spec(task);phase=task['phase'];b=spec['capacity'];sources={'/synthetic.py':'a'*64}
    identity=dict(schema=old.ARTIFACT_SCHEMA,variant=old.VARIANT,software=old.DEVICE_SOFTWARE,
        source_sha256=sources,phase=phase,sha256='b'*64)
    value=dict(complete=True,passed=True,spec=spec,artifact_sha256='b'*64)
    if phase=='context4096_device':
        identity['case']=dict(spec,variant='candidate')
        value['checks']=[dict(slots=s,bitwise_equal=True,cache_input_unchanged=True,updates_input_unchanged=True,
            slots_input_unchanged=True,actual_sha256='c'*64,expected_sha256='c'*64)
            for s in matrix.mappings(b,spec['rows'])]
    else:
        sequence=matrix.alias_mappings(b)
        identity['case']=dict(spec,dtype='bfloat16',rows=b,shapes=[[b,2,4096,128]]*2)
        value.update(attention_rounding_is_diagnostic_only=True,initial_sha256=['c'*64]*2,
            inputs_sha256=[['c'*64]*5 for _ in sequence],positions=[matrix.alias_positions(b,i) for i in range(len(sequence))],
            checks=[dict(slots=s,step=i,output_finite=True,cache_outputs_bitwise=True,persistent_states_bitwise=True,
                bindings_unchanged=True,inputs_unchanged=True,output_sha256='c'*64,state_sha256=['c'*64]*2)
                for i,s in enumerate(sequence)])
    value['artifact_identity']=identity
    return value,sources


class PhaseWorld:
    """All metadata in memory; actual read_phase verifier, mocked process/IO boundary."""
    def __init__(self,fault=None):
        self.values={};self.root=Path('/synthetic/device-run');self.phase='context4096_alias'
        self.parent=dict(pid=10,start_ticks='20',ppid=1,argv=['mock-parent'],cgroup='0::/mock-parent')
        self.sdk=dict(path='/synthetic/sdk.json',sha256='d'*64)
        specref=self.put('/synthetic/spec.json',dict(sdk_manifest=self.sdk))
        self.admission=self.put(self.root/'admission.json',dict(spec=specref))
        self.put(self.root/'worker-owner.json',self.parent)
        self.host=types.SimpleNamespace(PYTHON='/synthetic/python',SELF='/synthetic/runner',
            p=types.SimpleNamespace(read=self.read,gone=mock.Mock()),reference=self.ref,
            known_warnings=types.SimpleNamespace(validate_receipt=mock.Mock()))
        outcomes=[]
        for task in [x for x in protocol.tasks() if x['phase']==self.phase]:
            label=protocol.key(task);value,self.sources=science(task);i=task['index']
            if fault=='pair_inputs' and i==1:value['inputs_sha256'][0][0]='f'*64
            if fault=='pair_initial' and i==1:value['initial_sha256'][0]='f'*64
            child=dict(pid=100+i,start_ticks=str(200+i),ppid=10,argv=pipeline.child_argv(self.host,self.admission,task),cgroup='0::/mock-child')
            identity=value['artifact_identity'];identity.update(admission=self.admission,task=task,child=child,
                path=str(self.root/self.phase/value['spec']['name']/'model.pt'))
            artifact=self.put(Path(identity['path']).parent/'artifact-identity.json',identity)
            sci=self.put(protocol.scientific_path(self.root,task),value)
            sdk=self.put(self.root/'tasks'/(label+'-sdk.json'),dict(complete=True,passed=True,child=child,admission=self.admission,manifest=self.sdk))
            report=self.put(self.root/'tasks'/(label+'-result.json'),dict(complete=True,passed=True,task=task,
                admission=self.admission,child=child,warning_log={},science=sci,sdk=sdk,artifacts=[artifact],
                full_model_qualified=False,new_speedup_claim=False,context4096_qualified=False,self_exit_claimed=False))
            self.put(self.root/'tasks'/(label+'-owner.json'),dict(pid=child['pid'],start_ticks=child['start_ticks'],argv=child['argv'],
                parent=dict(self.parent,cgroup='/mock-parent'),cgroup='/mock-child'))
            outcomes.append(dict(task=task,returncode=0,passed=True,error=None,secondary_errors=[],pid=child['pid'],start_ticks=child['start_ticks'],report=report))
        if fault=='missing_child':outcomes.pop()
        if fault=='failed_exit':outcomes[-1]['returncode']=1
        self.put(self.root/self.phase/'phase-outcomes.json',dict(complete=True,passed=True,phase=self.phase,admission=self.admission,outcomes=outcomes))
    def put(self,path,value):self.values[str(path)]=copy.deepcopy(value);return self.ref(path)
    def ref(self,path):return dict(path=str(path),sha256=sha(self.values[str(path)]))
    def read(self,path,wanted=None):
        if wanted is not None:assert wanted==self.ref(path)['sha256']
        return copy.deepcopy(self.values[str(path)])
    def run(self):return pipeline.read_phase(self.host,self.root,self.phase,self.sources,self.admission,require_summary=False)


class AdditiveTests(unittest.TestCase):
    def test_exact_original_prefix_and_gates(self):
        current=protocol.manifest();prior=old.manifest()
        self.assertEqual(current['tasks'][:91],prior['tasks'])
        self.assertEqual(current['retained_protocol'],prior)
        self.assertEqual(current['original_model_gates'],prior['original_model_gates'])
        self.assertEqual(current['first_actual_graph'],dict(dtype='bfloat16',shape=[4,7168]))
        self.assertEqual(len(current['tasks']),105);self.assertEqual(current['graphs'],127)
        self.assertEqual(current['checks'],3491+sum(v[1] for k,v in protocol.COUNTS.items() if k.startswith('context4096')))
        for k in ('context4096_qualified','full_model_qualified','new_speedup_claim','automatic_successor_launch'):self.assertIs(current[k],False)
    def test_all_91_original_validators_are_retained(self):
        for task in old.tasks():
            value,sources=accepted_test.science_value(task)
            self.assertIs(protocol.validate_task(task,value,sources),value)
    def test_exact_real_lfm_shapes_rows_and_capacities(self):
        self.assertEqual([(s['capacity'],s['rows']) for s in matrix.device_specs()],[(4,1),(4,4),(8,1),(8,4),(8,8),(16,1),(16,4),(16,16)])
        for s in matrix.device_specs():self.assertEqual(s['shape'],[s['capacity'],2,4096,128])
        self.assertEqual([(s['capacity'],s['variant']) for s in matrix.alias_specs()],[(b,v) for b in (4,8,16) for v in ('reference','candidate')])
    def test_original_mapping_prefix_and_all_signed_controls(self):
        from lfm3_nki05_device_matrix_v1 import mappings as prior
        for s in matrix.device_specs():
            b,rows=s['capacity'],s['rows'];actual=matrix.mappings(b,rows)
            if b in (8,16):self.assertEqual(actual[:len(prior(b,rows))],prior(b,rows))
            for bad in (-(2**63),2**63-1,-(2**32),2**32,2**32+b-1):self.assertIn([bad]*rows,actual)
        for b in matrix.CAPACITIES:
            rows=matrix.alias_mappings(b);self.assertEqual(rows[:8],old.alias_mappings(b))
            self.assertEqual(rows[-3:],[list(range(b)),[-1]*b,list(reversed(range(b)))])
    def test_every_slot_crosses_2048_and_last_valid_index(self):
        for b in matrix.CAPACITIES:
            for slot in range(b):
                seen={matrix.alias_positions(b,i)[slot][0] for i in range(len(matrix.alias_mappings(b)))}
                self.assertTrue({0,2047,2048,4094,4095}<=seen);self.assertTrue(all(0<=x<4096 for x in seen))
    def test_all_new_task_validators_accept_complete_synthetic_reports(self):
        for task in protocol.tasks()[91:]:
            value,sources=science(task);self.assertIs(protocol.validate_task(task,value,sources),value)
    def test_omitted_last_check_and_wrong_shape_rejected(self):
        for task in protocol.tasks()[91:]:
            for fault in ('last_check','shape'):
                value,sources=science(task)
                if fault=='last_check':value['checks'].pop()
                elif task['phase']=='context4096_device':value['artifact_identity']['case']['shape'][1]=4
                else:value['artifact_identity']['case']['shapes'][0]=[value['spec']['capacity'],4,4096,128]
                with self.subTest(task=task,fault=fault),self.assertRaises(RuntimeError):protocol.validate_task(task,value,sources)
    def test_alias_position_state_binding_and_hash_failures_rejected(self):
        task=protocol.tasks()[-1]
        for fault in ('position','state','bindings','input','digest'):
            value,sources=science(task)
            if fault=='position':value['positions'][-1][0][0]=4096
            elif fault=='digest':value['checks'][-1]['state_sha256'][0]=''
            else:value['checks'][-1][{'state':'persistent_states_bitwise','bindings':'bindings_unchanged','input':'inputs_unchanged'}[fault]]=False
            with self.subTest(fault=fault),self.assertRaises(RuntimeError):protocol.validate_task(task,value,sources)
    def test_invalid_task_and_capacity_rejected(self):
        for task in (dict(phase='context4096_alias',index=True),dict(phase='context4096_device',index=-1),dict(phase='context4096_alias',index=6)):
            with self.assertRaises(RuntimeError):protocol.key(task)
        for capacity,rows in ((True,1),(2,1),(4,8),(16,0)):
            with self.assertRaises(RuntimeError):matrix.mappings(capacity,rows)
    def test_same_scientific_leaf_is_connected_and_admission_precedes_native(self):
        for phase,fn in (('context4096_device','device'),('context4096_alias','alias')):
            ctx=object()
            with mock.patch.object(leaf,fn) as invoked:pipeline.invoke(ctx,dict(phase=phase,index=0));invoked.assert_called_once_with(ctx,0)
        for fn in (leaf.device,leaf.alias):
            with self.assertRaises(RuntimeError):fn(object(),0)
    def test_complete_alias_phase_cross_checks_pairs_and_real_exit_metadata(self):
        world=PhaseWorld();value=world.run();self.assertEqual(len(value['graphs']),6)
        self.assertEqual(world.host.p.gone.call_count,6);self.assertIs(value['context4096_qualified'],False)
    def test_alias_phase_rejects_pair_drift_omission_and_nonzero_exit(self):
        for fault in ('pair_inputs','pair_initial','missing_child','failed_exit'):
            with self.subTest(fault=fault),self.assertRaises(RuntimeError):PhaseWorld(fault).run()
    def test_current_v1_manifest_cannot_admit_v2_owner(self):
        value=accepted_test.spec();value['runner']=runner.SELF;value['diagnostic']=pipeline.__file__
        with self.assertRaises(RuntimeError):runner.validate_spec(value)
        value['device_protocol']=protocol.manifest();runner.validate_spec(value)
    def test_sdk_gate_and_immediate_predecessor_remain_required(self):
        ctx=runtime.Context.__new__(runtime.Context);ctx.sdk_verified=False
        with self.assertRaises(RuntimeError):ctx.writer('context4096_device')
        ctx.phase='context4096_alias'
        with self.assertRaises(RuntimeError):ctx.prerequisite('wrapper_controls')
    def test_all_616_prior_pins_required_by_actual_prerequisite_function(self):
        closure={'source_sha256':{f'/synthetic/{i}.py':'a'*64 for i in range(616)}}
        cpu={'source_sha256':{'/synthetic/0.py':'a'*64}}
        value={'full_cpu_prerequisite':{},'source_sha256':dict(closure['source_sha256'])}
        host=types.SimpleNamespace(p=types.SimpleNamespace(read=mock.Mock(side_effect=[cpu,closure])))
        with mock.patch.object(pipeline,'validate_predecessor_bindings'),mock.patch.object(pipeline.cpu,'audit',return_value={'full':True}):
            self.assertEqual(pipeline.prerequisite(host,value),{'full':True})
            value['source_sha256'].pop('/synthetic/615.py');host.p.read.side_effect=[cpu,closure]
            with self.assertRaises(RuntimeError):pipeline.prerequisite(host,value)
    def test_serial_owner_identical_except_pipeline_import(self):
        old_source=(ROOT/'run_lfm3_nki05_device_serial_v1.py').read_text()
        new_source=(ROOT/'run_lfm3_nki05_device_serial_v2.py').read_text()
        self.assertEqual(new_source.replace('lfm3_nki05_device_pipeline_v2','lfm3_nki05_device_pipeline_v1'),old_source)
        old_source=(ROOT/'lfm3_nki05_device_runtime_v1.py').read_text()
        new_source=(ROOT/'lfm3_nki05_device_runtime_v2.py').read_text()
        for name in ('device_pipeline','device_protocol'):
            new_source=new_source.replace('lfm3_nki05_'+name+'_v2','lfm3_nki05_'+name+'_v1')
        new_source=new_source.replace('run_lfm3_nki05_device_serial_v2.py','run_lfm3_nki05_device_serial_v1.py')
        self.assertEqual(new_source,old_source)


if __name__=='__main__':
    resource.setrlimit(resource.RLIMIT_AS,(256*2**20,256*2**20));resource.setrlimit(resource.RLIMIT_CPU,(60,60));resource.setrlimit(resource.RLIMIT_FSIZE,(2*2**20,2*2**20))
    os.sched_setaffinity(0,{7});os.nice(19)
    unittest.main(verbosity=2)
