"""Synthetic full-plan assembly and pending rejection; never production prep."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch,Mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research import prepare_minicpm5_layer1_sdk_batch_layout_v3 as b


def ref(name):return dict(path='/synthetic/'+name+'.json',sha256='a'*64)


class Preparation(unittest.TestCase):
    def setUp(self):
        self.binding=dict(schema='minicpm5_layer1_sdk_batch_layout_latest_completion_v1',status='complete',
            current_history=copy.deepcopy(b.latest.CURRENT),terminal_handoff=ref('future-terminal'),nemotron_r12=dict(
                schema='minicpm5_after_nemotron_r12_binding_v1',actual_exit=ref('future-r12-exit'),failed_release_exit=None))
        self.contract=dict(path=str(b.Q/'minicpm5-layer1-sdk-batch-layout-source-contract-synthetic.json'),sha256='c'*64)
        self.identities={};records={}
        for rank in (0,1):
            rows=[]
            for n in range(90):
                path='/synthetic/rank%d/output-%d.bf16'%(rank,n);i=dict(realpath=path,device=1,inode=1000+rank*100+n,
                    size=16384,mtime_ns=1,ctime_ns=1);self.identities[path]=i
                rows.append(dict(output=dict(path=path,sha256='b'*64,identity=i,raw_bytes=16384)))
            records['rank%d_result'%rank]=dict(records=rows)
        plan=dict(schema='minicpm5_layer0_runtime_inputs_v1',inputs={'/synthetic/old':{'unchanged':True}},
            source_sha256={'/synthetic/old.py':'d'*64},output_root='/synthetic/old-output',
            fixtures=[{'fixture':'entire'}],calls=[{'call':'all'}],full_model_qualified=False,new_speedup_claim=False,
            source_only=True,runtime_admitted=False,original_precision='BF16',cold4096=True,longdecode=True,slotreuse=True)
        spec=dict(source_sha256=copy.deepcopy(plan['source_sha256']),schema='old',output_root=plan['output_root'],
            resources={'unchanged':'16GiB/TP2'},root_floor_bytes=28*2**30,output_floor_bytes=30*2**30,
            completed_storage={'retained':'yes'},canonical_sha256={'unchanged':'sha'})
        self.done=dict(binding=self.binding,metadata_complete=True,physical_checks_performed=False,source_sha256={'/synthetic/new-history.py':'e'*64},
            current_history=dict(plan=plan,spec=spec,binding=b.latest.CURRENT,records=records))
        self.inputs={'/synthetic/completed.json':{'sha256':'a'*64,'identity':{'synthetic':True}}}
        self.sources={'/synthetic/new.py':'f'*64}
    def assemble(self):return b.assemble(self.done,self.inputs,self.sources,self.contract,'synthetic',29631,self.identities.__getitem__)
    def test_all180_outputs_and_complete_original_science_retained(self):
        original=copy.deepcopy(self.done);spec,plan,paths=self.assemble()
        self.assertEqual(self.done,original);self.assertEqual(len(plan['inputs']),182)
        for k in ('fixtures','calls','original_precision','cold4096','longdecode','slotreuse'):
            self.assertEqual(plan[k],self.done['current_history']['plan'][k])
        for k in ('resources','root_floor_bytes','output_floor_bytes','completed_storage','canonical_sha256'):
            self.assertEqual(spec[k],self.done['current_history']['spec'][k])
        self.assertEqual(spec['source_sha256'],plan['source_sha256']);self.assertEqual(spec['source_contract'],plan['source_contract'])
        self.assertFalse(plan['runtime_admitted']);self.assertNotIn('passed',spec)
    def test_pending_rejects_before_any_lock_or_path(self):
        pending=copy.deepcopy(self.binding);pending['status']='pending'
        with patch.object(b,'paths',side_effect=AssertionError('must reject before output paths')):
            with self.assertRaises(RuntimeError):b.build(pending,'synthetic',29631)
    def test_conflicting_old_source_rejects(self):
        self.sources['/synthetic/old.py']='f'*64
        with self.assertRaises(RuntimeError):self.assemble()
    def test_conflicting_old_input_rejects(self):
        self.inputs['/synthetic/old']={'replacement':True}
        with self.assertRaises(RuntimeError):self.assemble()
    def test_changed_previous_output_identity_rejects(self):
        path=next(iter(self.identities));self.identities[path]=dict(self.identities[path],inode=99999)
        with self.assertRaises(RuntimeError):self.assemble()
    def test_incomplete_prior_rank_cannot_shrink_scope(self):
        self.done['current_history']['records']['rank1_result']['records'].pop()
        with self.assertRaises(RuntimeError):self.assemble()
    def test_wrong_full_output_bytes_rejects(self):
        self.done['current_history']['records']['rank0_result']['records'][0]['output']['raw_bytes']=4096
        with self.assertRaises(RuntimeError):self.assemble()
    def test_invalid_run_namespace_or_port_rejects(self):
        for name in ('../bad','bad/part','UPPER','a'*41):
            with self.assertRaises(RuntimeError):b.paths(name)
        with self.assertRaises(RuntimeError):b.assemble(self.done,self.inputs,self.sources,self.contract,'synthetic',30000,self.identities.__getitem__)
    def test_reader_authenticates_before_after_identity(self):
        guard=Mock();r=b.Reader(guard);ident={'stable':True};reference=ref('metadata')
        with patch.object(b,'identity',return_value=ident),patch.object(b.sources,'load',return_value={'value':1}):
            self.assertEqual(r(reference),{'value':1});self.assertEqual(r.inputs[reference['path']]['identity'],ident)
        self.assertEqual(guard.call_count,2)
    def test_reader_changed_identity_rejects(self):
        r=b.Reader(lambda:None)
        with patch.object(b,'identity',side_effect=[{'inode':1},{'inode':2}]),patch.object(b.sources,'load',return_value={}):
            with self.assertRaises(RuntimeError):r(ref('metadata'))
        self.assertEqual(r.inputs,{})
    def test_no_raw_reads_or_native_imports(self):
        with patch.object(Path,'read_bytes',side_effect=AssertionError('no reads during assembly')):self.assemble()
        self.assertFalse({'torch','numpy','torch_neuronx','neuronxcc'}&set(sys.modules))
    def test_auditor_and_exit_recorder_in_source_entrypoints(self):
        names={p.name for p in b.entrypoints()}
        self.assertIn('audit_minicpm5_layer1_sdk_batch_layout_result_v1.py',names)
        self.assertIn('record_minicpm5_layer1_sdk_batch_layout_exit_v1.py',names)
        self.assertIn('record_minicpm5_layer1_sdk_batch_layout_audit_exit_v1.py',names)
        self.assertIn('check_record_minicpm5_layer1_sdk_batch_layout_audit_exit_v1.py',names)
        self.assertIn('check_prepare_minicpm5_layer1_sdk_batch_layout_v3.py',names)


if __name__=='__main__':unittest.main(verbosity=2)
