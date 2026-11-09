"""v20 integration: exact old science, completed-history gating and no native run."""
import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

R=Path(__file__).resolve().parents[1];Q=R/'research'
sys.path[:0]=[str(R),str(Q)]
import nemotron9b_package_protocol_v19 as old
import nemotron9b_package_protocol_v20 as p
import nemotron9b_package_evidence_v20 as evidence
import nemotron9b_residual_completion_v1 as completion
import nemotron9b_completed_history_drain_v6 as drain
import check_nemotron9b_package_exec_v19 as previous
from research import nemotron9b_after_dory_residual_review_v1 as residual


class Source(unittest.TestCase):
    def test_frozen_v19_source_pins_preserved(self):
        receipt=json.loads((Q/'nemotron9b-v19-output-integration-tests-20261109-v1.json').read_text())
        paths=list(R.glob('nemotron9b_package_*v19.py'))
        self.assertEqual(len(paths),16)
        for path in paths:self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),receipt['source_sha256'][str(path)])

    def test_execution_bodies_only_rename_and_pass_exact_spec_to_drain(self):
        count=0
        def normalized(path,prior=False):
            text=path.read_text().replace('_v19','_v20')if prior else path.read_text()
            tree=ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node,ast.Call)and ast.unparse(node.func)=='p.predecessor_drain':
                    if prior:node.keywords.append(ast.keyword(arg='spec',value=ast.Name(id='spec',ctx=ast.Load())))
                    else:self.assertEqual([ast.unparse(k.value)for k in node.keywords if k.arg=='spec'],['spec'])
            return ast.dump(tree,include_attributes=False)
        for path in R.glob('nemotron9b_package_*v19.py'):
            if path.stem in ('nemotron9b_package_protocol_v19','nemotron9b_package_metadata_v19'):continue
            self.assertEqual(normalized(path,True),normalized(path.with_name(path.name.replace('_v19','_v20'))))
            count+=1
        self.assertEqual(count,14)

    def test_protocol_function_bodies_only_rename_and_source_list(self):
        def functions(text):
            return{n.name:ast.dump(n)for n in ast.parse(text).body
                   if isinstance(n,ast.FunctionDef)and n.name!='required_sources'}
        self.assertEqual(functions(Path(old.__file__).read_text().replace('_v19','_v20')),
                         functions(Path(p.__file__).read_text()))

    def test_full_original52_science_stages_and_limits_retained(self):
        for key in('STAGES','SCIENTIFIC_STAGES','ROOT_FLOOR','NVME_FLOOR','ROOT_CAP','NVME_CAP','CAPS',
                   'BUCKETS','SPLIT_FLAGS','ALIAS_FLAGS','PACKAGE','COMMIT'):
            self.assertEqual(getattr(p,key),getattr(old,key))
        for stage in(*p.STAGES,'controller'):self.assertEqual(p.resources(stage),old.resources(stage))
        self.assertEqual(len(p.STAGES),52)

    def test_new_metadata_completion_and_drain_are_connected(self):
        self.assertIs(p.attention_completion,completion)
        self.assertIs(p.predecessor_drain,drain.physical_drain)
        self.assertEqual(p.metadata.__name__,'nemotron9b_package_metadata_v20')
        self.assertIs(evidence.p,p)
        self.assertIs(p.compiler_links,p.output_inventory.compiler_links)
        self.assertEqual(p.compiler_links.__name__,'nemotron9b_compiler_tiled_links_v1')
        self.assertIs(drain.attention,completion)

    def test_all_old_new_sources_in_required_closure(self):
        paths=p.required_sources()
        self.assertTrue(set(old.required_sources())<set(paths))
        self.assertTrue({str(x)for x in R.glob('nemotron9b_package_*v20.py')}<=set(paths))
        self.assertTrue(all(Path(x).is_file()for x in paths))
        for name in('nemotron9b_residual_completion_v1.py','nemotron9b_completed_history_drain_v6.py',
                     'research/nemotron9b_residual_source_partition_v1.py',
                     'research/nemotron9b_after_dory_residual_review_v1.py'):
            self.assertIn(str(R/name),paths)

    def test_drain_cannot_be_called_without_current_spec(self):
        with self.assertRaises(TypeError):drain.physical_drain({})
        with patch.object(completion,'validate_spec',side_effect=RuntimeError('exact completion required'))as verify:
            with self.assertRaisesRegex(RuntimeError,'exact completion'):
                drain.physical_drain({'base':1},spec={'current':2})
            verify.assert_called_once_with({'current':2},{'base':1})

    def test_no_historical_controller_can_be_exempted(self):
        name='trn1-latest-verify-nemotron9b-package-r9-20261109-controller'
        with patch.object(completion,'validate_spec',return_value=dict(rows=[dict(unit=name)])):
            with self.assertRaisesRegex(RuntimeError,'No historical'):
                drain.physical_drain({},allowed_units=[name],spec={})

    def test_complete_original_drain_body_preserved_after_authentication(self):
        oldtree=ast.parse((R/'nemotron9b_completed_history_drain_v5.py').read_text())
        newtree=ast.parse((R/'nemotron9b_completed_history_drain_v6.py').read_text())
        left=next(n for n in oldtree.body if isinstance(n,ast.FunctionDef)and n.name=='physical_drain')
        right=next(n for n in newtree.body if isinstance(n,ast.FunctionDef)and n.name=='physical_drain')
        self.assertEqual([ast.dump(n)for n in left.body[1:]],[ast.dump(n)for n in right.body[1:]])

    def test_no_runtime_spec_or_native_libraries(self):
        self.assertFalse(list(Q.glob('nemotron9b-package-runtime-v20-*-spec.json')))
        self.assertFalse(any(x.split('.')[0]in{'torch','numpy','vllm','neuronxcc'}for x in sys.modules))


class Completion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base=completion.base_history.load(completion.base_history.CATALOG)
        cls.anchor=residual.anchor(residual.Reader())

    def setUp(self):
        self.h=copy.deepcopy(self.anchor['history'])
        self.exit=dict(path=str(Q/'synthetic-never-written-residual-exit.json'),sha256='a'*64)
        self.h['processes'].extend([dict(pid=990001+i,start_ticks=str(99999999+i))for i in range(3)])
        self.h['known_admitted_directories']+=['residual-replay-r2','residual-replay-review-r2']
        self.h['completed_residual_review']=self.exit
        self.proof=dict(history=self.h,source_sha256={'/synthetic/pinned.json':'b'*64},
            historical_binary_sha256={f'/synthetic/{i}.bin':'c'*64 for i in range(6)},
            runtime_admitted=False,fresh_physical_drain_verified=False,full_model_qualified=False,new_speedup_claim=False)
        self.spec=dict(required_residual_source_anchor=residual.PRODUCER,required_residual_completion=self.exit,
            required_attention_completion=completion.maintenance.attention.EXIT,
            required_journal_compaction=completion.maintenance.EXIT,source_sha256=dict(self.proof['source_sha256']),
            dory_historical_payload_sha256=dict(self.proof['historical_binary_sha256']))

    def check(self):
        with patch.object(residual,'completed',return_value=self.proof)as completed:
            result=completion.validate_spec(self.spec,self.base)
            completed.assert_called_once_with(self.exit)
            return result

    def test_synthetic_valid_completion_keeps_full1149_348_history(self):
        self.assertIs(self.check(),self.h)
        self.assertFalse(self.h['r9_outcome']['producer_qualified'])

    def test_actual_missing_completion_ref_is_rejected_without_future_read(self):
        self.spec.pop('required_residual_completion')
        with patch.object(residual,'completed')as completed:
            with self.assertRaisesRegex(RuntimeError,'mandatory'):completion.validate_spec(self.spec,self.base)
            completed.assert_not_called()

    def test_omitted_old_attention_or_maintenance_rejected(self):
        for key in('required_attention_completion','required_journal_compaction'):
            with self.subTest(key=key):
                self.setUp();self.spec.pop(key)
                with self.assertRaises(RuntimeError):self.check()

    def test_omitted_service_or_owner_or_directory_rejected(self):
        for key in('rows','processes','known_admitted_directories'):
            with self.subTest(key=key):
                self.setUp();self.h[key].pop()
                with self.assertRaises(RuntimeError):self.check()

    def test_changed_original_prefix_rejected(self):
        self.h['processes'][0]['pid']+=1
        with self.assertRaises(RuntimeError):self.check()

    def test_r9_failed_outcome_cannot_be_promoted(self):
        self.h['r9_outcome']['producer_qualified']=True
        with self.assertRaises(RuntimeError):self.check()

    def test_completed_source_pin_cannot_be_missing_or_rebound(self):
        self.spec['source_sha256']['/synthetic/pinned.json']='0'*64
        with self.assertRaises(RuntimeError):self.check()

    def test_six_dory_binary_pins_cannot_be_removed(self):
        self.spec['dory_historical_payload_sha256'].popitem()
        with self.assertRaises(RuntimeError):self.check()

    def test_metadata_cannot_claim_fresh_drain_or_qualification(self):
        for key in('runtime_admitted','fresh_physical_drain_verified','full_model_qualified','new_speedup_claim'):
            with self.subTest(key=key):
                self.setUp();self.proof[key]=True
                with self.assertRaises(RuntimeError):self.check()


class FullPaths(previous.FullPaths):
    def setUp(self):
        for name,value in(('p',p),('evidence',evidence)):
            change=patch.object(previous,name,value);change.start();self.addCleanup(change.stop)
        super().setUp()


if __name__=='__main__':unittest.main(verbosity=2)
