"""Source-only renderer tests; all fake completion bytes and outputs stay in memory."""
import ast
import copy
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

class BlockNative(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in {'torch','numpy','ctypes','_ctypes','torch_neuronx','torch_xla','neuronxcc',
                'neuronx_distributed','neuronx_distributed_inference','nki','vllm'}:
            raise AssertionError('No native imports: '+name)
sys.meta_path.insert(0, BlockNative())
Q=Path(__file__).resolve().parent
sys.path[:0]=[str(Q),str(Q.parent)]
import render_minicpm5_completed_writer_domain_sources_v1 as renderer
from research import minicpm5_completed_qwen_package_text_v3 as previous

PREDECESSORS={n:(Q/n).read_bytes() for n in renderer.PREDECESSORS}
BASE=Path(renderer.BASE_CATALOG['path']).read_bytes()
INVENTORY=json.loads((Q/'minicpm5-geometry-r2-pending-source-domain-inventory-20261112-v1.json').read_bytes())

def envelope(name, value):
    raw=(json.dumps(value,sort_keys=True,ensure_ascii=False,indent=2)+'\n').encode()
    return dict(reference=dict(path=str(Q/name),sha256=hashlib.sha256(raw).hexdigest()),payload=raw)

def fixture():
    catalog=json.loads(BASE)
    delta=dict(INVENTORY['all_extra_pins'],**renderer.ROOT_HELPERS)
    assert len(delta)==75
    ended=dict(schema='qwen35_writer_geometry_root_actual_exit_v1',reviewer='/root',
        actual_tool=renderer.ACTUAL_OUTER_TOOL,actual_tool_exit_observed=True,actual_outer_exit_observed=True,
        owners=[],units=['synthetic-unit'],owners_gone=True,physically_drained=True,
        original_lock_held_during_audit=True,lock_identity=renderer.LOCK_IDENTITY,
        completed_layouts=1,completed_calls=7865,compiled_graphs=7,geometry_qualified=False,
        full_model_qualified=False,new_speedup_claim=False,reconciliation={'passed':False})
    end=envelope('synthetic-writer-ended-test-only.json',ended)
    catalog['runs'].append(dict(terminal=end['reference'],tool=renderer.ACTUAL_OUTER_TOOL,
        owners=ended['owners'],units=ended['units'],source_sha256=delta))
    catalog['current_handoff']=dict(path=str(Q/'synthetic-idle-test-only.json'),sha256='b'*64)
    cat=envelope('synthetic-writer-catalog-test-only.json',catalog)
    final=dict(schema='qwen35_writer_geometry_root_finalization_v1',reviewer='/root',actual_exit=end['reference'],
        actual_outer_tool=renderer.ACTUAL_OUTER_TOOL,complete_old_catalog=renderer.BASE_CATALOG,catalog=cat['reference'],
        idle_handoff=catalog['current_handoff'],source_composition=renderer.COMPOSITION,
        all_prior163_rows_unchanged=True,original_8MiB_cap_preserved=True,total_rows=164,
        original_lock_identity=renderer.LOCK_IDENTITY,geometry_qualified=False,full_model_qualified=False,new_speedup_claim=False,
        actual_independent_auditor_tool=dict(kind='tool_session',session_id=9791,terminal_chunk='abc123',exit_code=0),
        external_sources=[dict(path=p,sha256=s) for p,s in renderer.ROOT_HELPERS.items()],
        two_current_drain_observations=[{'synthetic-unit':{'empty':True}}]*2)
    return dict(base=dict(reference=renderer.BASE_CATALOG,payload=BASE),catalog=cat,ended=end,
        finalization=envelope('synthetic-writer-final-test-only.json',final))

def render(inputs=None, predecessors=None):
    with patch('builtins.open',side_effect=AssertionError('No IO')),patch('os.open',side_effect=AssertionError('No FD IO')),\
         patch('fcntl.flock',side_effect=AssertionError('No lock')),patch('subprocess.Popen',side_effect=AssertionError('No process')):
        return renderer.render(PREDECESSORS if predecessors is None else predecessors,fixture() if inputs is None else inputs)

def domain(result):
    module=types.ModuleType('synthetic_domain');exec(compile(result['files'][str(Q/'minicpm5_completed_qwen_package_text_v4.py')],'<in-memory-domain>','exec'),module.__dict__)
    return module

def mutate(inputs,key,change):
    value=json.loads(inputs[key]['payload']);change(value)
    inputs[key]=envelope(Path(inputs[key]['reference']['path']).name,value)
    return inputs

def reader(inputs):
    byref={(v['reference']['path'],v['reference']['sha256']):json.loads(v['payload']) for v in inputs.values()}
    return lambda ref:copy.deepcopy(byref[(ref['path'],ref['sha256'])])

class WriterDomainTests(unittest.TestCase):
    def test_no_io_full_reversal_and_no_admission(self):
        result=render();self.assertEqual(len(result['files']),7)
        for flag in ('source_accepted','launch_authorized','actual_inputs_independently_verified','history_executed','native_executed','later_unknown_root_helper_paths_accepted'):
            self.assertIs(result[flag],False)
        for path,row in result['exact_predecessor_reversal'].items():
            text=result['files'][path].decode()
            for edit in reversed(row['edits']):text=text.replace(edit['after'],edit['before'])
            self.assertEqual(text.encode(),PREDECESSORS[row['predecessor']])
        self.assertEqual(result['sdk_worker_unchanged'],'minicpm5_sdk_mapping_worker_source_entry_v8.py')

    def test_mutated_frozen_source_and_metadata_bytes_rejected(self):
        for key in PREDECESSORS:
            copied=dict(PREDECESSORS);copied[key]+=b'\n'
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'Exact frozen predecessor'):render(predecessors=copied)
        for key in fixture():
            copied=fixture();copied[key]['payload']+=b'\n'
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'Full actual metadata digest'):render(copied)

    def test_original_cap_and_malformed_reference_rejected(self):
        row=dict(reference=dict(renderer.BASE_CATALOG),payload=b' '*((8*2**20)+1))
        row['reference']['sha256']=hashlib.sha256(row['payload']).hexdigest()
        with self.assertRaisesRegex(ValueError,'original 8 MiB'):renderer.metadata(row)
        for path in ['/tmp/x.json',str(Q/'../research/x.json'),str(Q/'x.py')]:
            with self.assertRaises(ValueError):renderer.reference(dict(path=path,sha256='a'*64))

    def test_incomplete_ended_or_fabricated_final_status_rejected(self):
        cases=[('ended',lambda v:v.update(owners_gone=False)),('ended',lambda v:v.update(completed_calls=7864)),
            ('ended',lambda v:v.update(geometry_qualified=True)),('ended',lambda v:v['actual_tool'].update(exit_code=0)),
            ('finalization',lambda v:v['actual_independent_auditor_tool'].update(exit_code=1)),
            ('finalization',lambda v:v['actual_independent_auditor_tool'].update(terminal_chunk='pending')),
            ('finalization',lambda v:v.update(two_current_drain_observations=[{}])),
            ('finalization',lambda v:v['source_composition'].update(new_row_sources=74))]
        for key,change in cases:
            with self.subTest(key=key),self.assertRaises(ValueError):render(mutate(fixture(),key,change))

    def test_missing_changed_source_or_changed_historical_prefix_rejected(self):
        for change in [lambda v:v['runs'].pop(),lambda v:v['runs'].append(v['runs'][-1]),
            lambda v:v['runs'][0]['tool'].update(exit_code=99),
            lambda v:v['runs'][-1]['source_sha256'].pop(next(iter(renderer.EXTERNAL_TEXT))),
            lambda v:v['runs'][-1]['source_sha256'].update({next(iter(renderer.EXTERNAL_TEXT)):'f'*64})]:
            with self.assertRaises(ValueError):render(mutate(fixture(),'catalog',change))

    def test_generated_domain_authenticates_all38_without_changing_ordinary_rule(self):
        inputs=fixture();result=render(inputs);module=domain(result);read=reader(inputs)
        extension=dict(renderer.EXTERNAL_TEXT);extension[str(Q/'ordinary.py')]='e'*64
        self.assertEqual(module.extension_shape(extension,completed_lfm_text={},catalog_ref=inputs['catalog']['reference'],read=read),extension)
        self.assertEqual(module.authenticate_completed_writer(inputs['catalog']['reference'],read,Mock()),renderer.EXTERNAL_TEXT)

    def test_domain_hash_subset_and_neighbor_paths_rejected(self):
        inputs=fixture();module=domain(render(inputs));ref=inputs['catalog']['reference'];read=reader(inputs)
        for mode in ('subset','hash','neighbor'):
            extension=dict(renderer.EXTERNAL_TEXT);key=next(iter(extension))
            if mode=='subset':del extension[key]
            elif mode=='hash':extension[key]='f'*64
            else:extension[key+'.extra']=extension[key]
            with self.subTest(mode=mode),self.assertRaises(ValueError):
                module.extension_shape(extension,completed_lfm_text={},catalog_ref=ref,read=read)

    def test_future_catalog_preserves_whole164_but_unknown_r3_domain_is_rejected(self):
        inputs=fixture();module=domain(render(inputs));saved=reader(inputs);later=json.loads(inputs['catalog']['payload'])
        later['runs'].append({'synthetic_future':True});ref=dict(path=str(Q/'synthetic-later.json'),sha256='f'*64)
        read=lambda r:copy.deepcopy(later) if r==ref else saved(r)
        self.assertEqual(module.authenticate_completed_writer(ref,read,Mock()),renderer.EXTERNAL_TEXT)
        extension=dict(renderer.EXTERNAL_TEXT)
        extension['/home/ubuntu/finalize_qwen35_writer_geometry_r3_20261113_v1.py']='f'*64
        with self.assertRaisesRegex(ValueError,'no new external permission'):
            module.extension_shape(extension,completed_lfm_text={},catalog_ref=ref,read=read)
        later['runs'][163]['tool']['exit_code']=0
        with self.assertRaisesRegex(ValueError,'Entire exact164'):module.authenticate_completed_writer(ref,read,Mock())

    def test_actual_reader_and_guard_still_own_all_saved_metadata(self):
        inputs=fixture();module=domain(render(inputs));guard=Mock(side_effect=RuntimeError('deadline'))
        with self.assertRaisesRegex(RuntimeError,'deadline'):module.authenticate_completed_writer(inputs['catalog']['reference'],Mock(),guard)
        ordinary={str(Q/'ordinary.py'):'e'*64}
        self.assertIs(module.extension_shape(ordinary,completed_lfm_text={},catalog_ref=inputs['catalog']['reference'],read=Mock(side_effect=AssertionError('No ended lookup'))),ordinary)

    def test_generated_wiring_keeps_full_prior_sources_and_original_worker(self):
        result=render();ordered=['minicpm5_completed_qwen_package_text_v4','prepare_minicpm5_pv_sdk_import_catalog_v5',
            'minicpm5_fixed_input_pv_catalog_auth_v5','minicpm5_sdk_mapping_outer_source_entry_v14',
            'prepare_minicpm5_catalog_formal_binding_v7','prepare_minicpm5_formal_v7_metadata_root_v4',
            'accept_minicpm5_formal_v7_metadata_root_v4']
        import research
        modules={}
        with patch.dict(sys.modules,{}):
            for name in ordered:
                full='research.'+name;module=types.ModuleType(full);module.__file__=str(Q/(name+'.py'))
                sys.modules[full]=module;modules[name]=module
                exec(compile(result['files'][module.__file__],module.__file__,'exec'),module.__dict__)
            binder=modules['prepare_minicpm5_catalog_formal_binding_v7'];prepare=modules['prepare_minicpm5_formal_v7_metadata_root_v4']
            self.assertEqual(len(binder.SOURCES),31);self.assertEqual(len(binder.RUNTIME),21);self.assertEqual(len(prepare.HELPERS),9)
            self.assertIs(binder.catalog_prepare,modules['prepare_minicpm5_pv_sdk_import_catalog_v5'])
            self.assertIs(binder.catalog_prepare.completed,modules[ordered[0]])
            self.assertIs(modules['minicpm5_fixed_input_pv_catalog_auth_v5'].completed,modules[ordered[0]])
            self.assertIs(modules['minicpm5_sdk_mapping_outer_source_entry_v14'].catalog_auth,modules['minicpm5_fixed_input_pv_catalog_auth_v5'])
            self.assertIn('minicpm5_sdk_mapping_worker_source_entry_v8.py',binder.RUNTIME)
            self.assertIn('minicpm5_saved_operator_reconstruction_v2.py',binder.RUNTIME)

if __name__=='__main__':unittest.main(verbosity=2)
