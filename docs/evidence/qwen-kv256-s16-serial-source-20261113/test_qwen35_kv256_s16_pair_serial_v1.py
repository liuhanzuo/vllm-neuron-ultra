"""Focused stdlib/mock checks for the narrow source-only pair serial adapter."""
import ast
import copy
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

Q=Path(__file__).resolve().parent
sys.path[:0]=[str(Q),str(Q.parent)]
BLOCKED={'torch','torch_neuronx','torch_xla','neuronxcc','numpy','ctypes','nki','vllm','vllm_neuron'}
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0]in BLOCKED:raise AssertionError('Forbidden native import: '+fullname)
assert not BLOCKED.intersection(sys.modules)
sys.meta_path.insert(0,NoNative())
from test_qwen35_kv256_s16_pair_v1 import Fixture, metadata_fixture
from qwen35_kv256_s16_pair_serial_v1 import protocol,contract,storage,run

PINS={'/synthetic/source.py':'a'*64}
WORK=Path('/synthetic-pair/kv256_s16_pair')
NAME='kv256_s16_pair'

def fixture():
    report=metadata_fixture()
    old=report['output_root'];new=str(WORK/'artifacts')
    def replace(value):
        if isinstance(value,str):return value.replace(old,new)
        if isinstance(value,list):return [replace(x)for x in value]
        if isinstance(value,dict):return {k:replace(v)for k,v in value.items()}
        return value
    report=replace(report);report['source_sha256']=PINS
    return report

def receipts(report):return [protocol.receipt(g)for g in report['graphs']]

def prefix(calls=0,returned=0,started=0,graphs=1,proof=True):
    report=fixture();report.update(complete=False,diagnostic_completed=False,
        started_calls=started,returned_calls=returned,calls=report['calls'][:calls],graphs=report['graphs'][:graphs])
    for g in report['graphs']:g['calls']=sum(r['graph']==g['name']for r in report['calls'][:returned])
    for i,row in enumerate(report['calls']):
        row['call_returned']=i<returned
        if i>=returned:row.pop('all_expected_bytes_equal')
    if graphs and not proof:report['graphs'][-1].pop('hlo_proof')
    return report

class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.pin=patch.object(protocol,'verify_sources',return_value=PINS);self.pin.start();self.addCleanup(self.pin.stop)
    def test_exact_two_graph_six_call_contract(self):
        self.assertEqual(protocol.counts()['total'],dict(graphs=2,correctness_calls=6,total_runtime_calls=6))
        self.assertEqual(len(list(protocol.expected_calls())),6)
        self.assertEqual(protocol.workload(),protocol.leaf.workload())
    def test_full_success_retains_all_checks(self):
        v=fixture();result=protocol.validate_report(v,NAME,receipts(v))
        self.assertEqual(result['calls'],6);self.assertFalse(result['full_geometry_qualified'])
        self.assertEqual(sum(len(x['proofs'])for x in v['calls']),18)
        self.assertEqual(sum(len(x['host_inputs_after'])for x in v['calls']),12)
    def test_old_five_field_receipt_rejected(self):
        v=receipts(fixture())[0];v.pop('hlo');v.pop('hlo_proof')
        with self.assertRaises(ValueError):protocol.validate_receipt(v,NAME,WORK)
    def test_exact_full_receipt_accepted(self):
        for v in receipts(fixture()):self.assertEqual(protocol.validate_receipt(v,NAME,WORK),v)
    def test_missing_extra_short_or_invalid_file_refs_rejected(self):
        for key in ('artifact','hlo'):
            for change in ('missing','extra','size','path','hash','oversize'):
                v=receipts(fixture())[0]
                if change=='missing':v[key].pop('size_bytes')
                elif change=='extra':v[key]['bytes']=v[key]['size_bytes']
                elif change=='size':v[key]['size_bytes']=0
                elif change=='path':v[key]['path']+='.wrong'
                elif change=='hash':v[key]['sha256']='G'*64
                else:v[key]['size_bytes']=contract.GIB
                with self.subTest(key=key,change=change),self.assertRaises(ValueError):protocol.validate_receipt(v,NAME,WORK)
    def test_wrong_hlo_proof_binding_rejected(self):
        for key,value in [('arm','wrong'),('hlo_sha256','f'*64),('hlo_bytes',0),('state_alias_output',0),
                          ('concat_dtype','f32'),('numeric_converts_in_assembly',1)]:
            v=receipts(fixture())[0];v['hlo_proof'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):protocol.validate_receipt(v,NAME,WORK)
    def test_one_graph_zero_receipts_preserves_hlo_failure(self):
        v=prefix(proof=False);v['error']="ValueError('actual compiler HLO gate')"
        p=protocol.validate_prefix(v,NAME,[])
        self.assertEqual((p['graphs'],p['graphs_without_receipt'],p['calls']),(1,1,0))
        self.assertTrue(p['pending_hlo_proof']);self.assertEqual(p['saved_failure'],v['error'])
        self.assertEqual(p['last_saved_stage'],'graph_hlo_captured_before_receipt')
    def test_pending_receipt_with_finished_hlo_proof(self):
        v=prefix();p=protocol.validate_prefix(v,NAME,[])
        self.assertFalse(p['pending_hlo_proof']);self.assertEqual(p['graphs_without_receipt'],1)
    def test_pending_second_graph_preserves_first_three_calls(self):
        v=prefix(calls=3,returned=3,started=3,graphs=2,proof=False)
        p=protocol.validate_prefix(v,NAME,[protocol.receipt(v['graphs'][0])])
        self.assertEqual((p['calls'],p['graphs'],p['returned_calls']),(3,2,3))
    def test_invalid_unreceipted_graph_refs_and_identity_rejected(self):
        for key,value in [('alias',0),('operation','wrong'),('physical_output_order',['new','old'])]:
            v=prefix(proof=False);v['graphs'][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):protocol.validate_prefix(v,NAME,[])
        v=prefix(proof=False);v['graphs'][0]['hlo']['sha256']='broken'
        with self.assertRaises(ValueError):protocol.validate_prefix(v,NAME,[])
    def test_unreceipted_graph_cannot_have_calls(self):
        v=prefix(calls=1,returned=1,started=1)
        with self.assertRaises(ValueError):protocol.validate_prefix(v,NAME,[])
    def test_two_unreceipted_graphs_rejected(self):
        with self.assertRaises(ValueError):protocol.validate_prefix(prefix(graphs=2),NAME,[])
    def test_complete_missing_receipt_rejected(self):
        v=fixture()
        with self.assertRaises(ValueError):protocol.validate_report(v,NAME,receipts(v)[:1])
    def test_complete_missing_old_new_state_or_host_capture_rejected(self):
        for missing in ('old','new','state','host'):
            v=fixture()
            if missing=='host':v['calls'][0]['host_inputs_after'].pop()
            else:v['calls'][0]['proofs'].pop(missing)
            with self.subTest(missing=missing),self.assertRaises(ValueError):protocol.validate_report(v,NAME,receipts(v))
    def test_interrupted_before_invocation_keeps_zero_counters(self):
        v=prefix(calls=1);p=protocol.validate_prefix(v,NAME,receipts(v))
        self.assertEqual((p['calls'],p['returned_calls'],p['incomplete_proofs']),(0,0,1))
    def test_interrupted_invocation_keeps_started_only(self):
        v=prefix(calls=1,started=1);p=protocol.validate_prefix(v,NAME,receipts(v))
        self.assertEqual((v['started_calls'],p['returned_calls']),(1,0))
    def test_returned_call_with_partial_proofs_preserved(self):
        v=prefix(calls=1,started=1,returned=1);v['calls'][0].pop('all_expected_bytes_equal')
        p=protocol.validate_prefix(v,NAME,receipts(v));self.assertEqual((p['returned_calls'],p['incomplete_proofs']),(1,1))
    def test_forged_return_flag_rejected(self):
        v=prefix(calls=1,started=1);v['calls'][0]['call_returned']=True
        with self.assertRaises(ValueError):protocol.validate_prefix(v,NAME,receipts(v))
    def test_calls_without_reported_graph_rejected(self):
        v=prefix(calls=1,graphs=0)
        with self.assertRaises(ValueError):protocol.validate_prefix(v,NAME,[])
    def test_complete_prefix_requires_full_validator(self):
        v=fixture();v['calls'][0]['reset_exact']=False
        with self.assertRaises(ValueError):protocol.validate_prefix(v,NAME,receipts(v))
    def test_partial_cannot_promote_model_or_geometry(self):
        for key in ('full_geometry_qualified','full_model_qualified','new_speedup_claim'):
            v=prefix();v[key]=True
            with self.subTest(key=key),self.assertRaises(ValueError):protocol.validate_prefix(v,NAME,receipts(v))

class CallbackTests(unittest.TestCase):
    def callbacks(self,info,payload,**overrides):
        paths={info[k]['path']:info[k]for k in ('artifact','hlo')}
        def identity(p):return dict(size=paths[str(p)]['size_bytes'],inode=1)
        args=dict(identity=identity,raw=lambda p,cap:payload)
        args.update(overrides)
        return patch.multiple(run,**args)
    def test_callback_recomputes_complete_actual_hlo_proof(self):
        for arm in range(2):
            info=receipts(fixture())[arm];payload=Fixture(info['name']).encode()
            with self.callbacks(info,payload),patch.object(protocol,'digest_file',side_effect=lambda p:info['hlo'if str(p).endswith('graph.hlo')else'artifact']['sha256']):
                self.assertEqual(run.artifact_record(info,NAME,WORK),info)
    def test_callback_rejects_changed_hlo_bytes(self):
        info=receipts(fixture())[1]
        with self.callbacks(info,b'wrong'),patch.object(protocol,'digest_file',side_effect=lambda p:info['hlo'if str(p).endswith('graph.hlo')else'artifact']['sha256']):
            with self.assertRaises(ValueError):run.artifact_record(info,NAME,WORK)
    def test_callback_rejects_forged_full_proof(self):
        info=receipts(fixture())[1];payload=Fixture(info['name']).encode();info['hlo_proof']['writer_func_literal_sha256']='a'*64
        with self.callbacks(info,payload),patch.object(protocol,'digest_file',side_effect=lambda p:info['hlo'if str(p).endswith('graph.hlo')else'artifact']['sha256']):
            with self.assertRaises(ValueError):run.artifact_record(info,NAME,WORK)
    def test_callback_rejects_identity_change(self):
        info=receipts(fixture())[0];payload=Fixture(info['name']).encode()
        seq=[dict(size=123,inode=1),dict(size=123,inode=2)]
        with patch.object(run,'identity',side_effect=seq),patch.object(protocol,'digest_file',return_value=info['artifact']['sha256']):
            with self.assertRaises(ValueError):run.artifact_record(info,NAME,WORK)
    def test_callback_rejects_truncated_artifact(self):
        info=receipts(fixture())[0]
        with patch.object(run,'identity',return_value=dict(size=122)):
            with self.assertRaises(ValueError):run.artifact_record(info,NAME,WORK)

class StorageAndInheritanceTests(unittest.TestCase):
    def test_paths_use_exact_pair_layout_and_graph_names(self):
        for graph in protocol.GRAPHS:
            base=Path(NAME)/'artifacts'/'graphs'/graph['name']
            self.assertEqual(storage.output_class(base/'model.pt',False),'fresh_artifacts')
            self.assertEqual(storage.output_class(base/'model'/'graph.hlo',False),'all_other_outputs')
        for directory,ext in [('inputs','.raw'),('calls','.raw'),('calls','.diffbits')]:
            self.assertEqual(storage.output_class(Path(NAME)/'artifacts'/directory/('x'+ext),False),'raw_and_bitmaps')
        self.assertEqual(storage.output_class(Path('kv256_alias_diag/artifacts/graphs/bf16_cat/model.pt'),False),'all_other_outputs')
    def test_all_output_classes_and_per_file_caps_retained(self):
        self.assertEqual(sum(storage.OUTPUT_CLASS_CAPS.values()),contract.GIB)
        self.assertEqual(storage.file_cap(Path('model.pt')),32*contract.MIB)
        self.assertEqual(storage.file_cap(Path('graph.hlo')),16*contract.MIB)
        self.assertEqual(storage.file_cap(Path('native.log')),8*contract.MIB)
        self.assertEqual(contract.LIMITS['cpus'],[6,7]);self.assertEqual(contract.LIMITS['nice'],19)
        self.assertEqual(contract.PROPERTIES['MemoryMax'],str(6*contract.GIB))
        self.assertEqual(contract.PROPERTIES['MemorySwapMax'],'0')
        self.assertEqual(protocol.workload()['per_leaf_seconds'],900)
    def fake_resources(self,nvme,allocated=0):
        def fst(path):return types.SimpleNamespace(st_dev=1 if path==storage.ROOT else 2)
        def fs(path):return types.SimpleNamespace(f_bavail=(28*contract.GIB if path==storage.ROOT else nvme),f_frsize=1)
        with patch.object(storage,'scan_output',return_value={'allocated_bytes':allocated}),\
             patch.object(Path,'is_mount',return_value=True),patch.object(Path,'stat',fst),\
             patch.object(storage.os,'statvfs',side_effect=fs),patch.object(Path,'read_text',return_value='MemAvailable: 4194304 kB'),\
             patch.object(storage,'current_layout',return_value=None):
            return storage.resources(dict(output_root='/mnt/nvme/neuron-work/synthetic'))
    def test_original_nvme_reservation_not_silently_lowered(self):
        self.fake_resources(40*contract.GIB)
        with self.assertRaises(ValueError):self.fake_resources(40*contract.GIB-1)
        self.fake_resources(39*contract.GIB,allocated=contract.GIB)
        with self.assertRaises(ValueError):self.fake_resources(39*contract.GIB-1,allocated=contract.GIB)
    def test_lifecycle_environment_and_scanner_functions_unchanged(self):
        def functions(package,name):
            text=(Q/package/(name+'.py')).read_text().replace('qwen35_kv256_alias_diagnostic','qwen35_kv256_s16_pair')
            return {x.name:ast.dump(x,include_attributes=False)for x in ast.parse(text).body if isinstance(x,ast.FunctionDef)}
        selections={
            'contract':['environment','bytecode_prefix','check_environment','check_process_settings','update_json','current_layout'],
            'storage':['scan_stat','scan_once','scan_output','resources','publication_role'],
            'run':['entry_argv','supervisor_argv','loaded_sources','group_members','check_group','only_supervisor','capture_layout',
                   'run_layouts','read_history_reference','supervisor','layout','terminal_storage','outer','reconcile','main']}
        for module,names in selections.items():
            old=functions('qwen35_kv256_alias_diagnostic_serial_v1',module);new=functions('qwen35_kv256_s16_pair_serial_v1',module)
            for name in names:
                with self.subTest(module=module,name=name):self.assertEqual(old[name],new[name])
    def test_manifest_only_pins_sources_and_matched_pair(self):
        p=Path(contract.MANIFEST['path']);data=p.read_bytes();v=json.loads(data)
        self.assertEqual(len(data),contract.MANIFEST['bytes']);self.assertEqual(hashlib.sha256(data).hexdigest(),contract.MANIFEST['sha256'])
        self.assertEqual(v['workload'],protocol.workload());self.assertEqual(len(v['source_files']),7)
        self.assertTrue(all(Path(x['path']).suffix=='.py'for x in v['source_files']))
    def test_no_native_module_was_imported(self):self.assertFalse(BLOCKED.intersection(sys.modules))

if __name__=='__main__':unittest.main(verbosity=2)
