"""Post-storage source integration; preserves all original model/science paths."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

R=Path(__file__).resolve().parents[1];Q=R/'research'
sys.path[:0]=[str(R),str(Q)]
import nemotron9b_package_protocol_v20 as old
import nemotron9b_package_protocol_v21 as p
import nemotron9b_package_evidence_v21 as evidence
import nemotron9b_completed_storage_completion_v1 as completion
import nemotron9b_completed_history_drain_v7 as drain
import check_nemotron9b_package_exec_v20 as previous
from research import nemotron9b_completed_storage_metadata_v1 as extra_metadata


class Source(unittest.TestCase):
    def test_published_v20_source_pins_preserved(self):
        receipt=json.loads((Q/'nemotron9b-v20-source-integration-review-20261109-v1.json').read_text())
        paths=list(R.glob('nemotron9b_package_*v20.py'))
        self.assertEqual(len(paths),16)
        for path in paths:
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),receipt['source_sha256'][str(path)])

    def test_fourteen_execution_modules_only_rename_version(self):
        paths=[path for path in R.glob('nemotron9b_package_*v20.py')
               if path.stem not in ('nemotron9b_package_protocol_v20','nemotron9b_package_metadata_v20')]
        self.assertEqual(len(paths),14)
        for path in paths:
            original=ast.parse(path.read_text().replace('_v20','_v21'))
            successor=ast.parse(path.with_name(path.name.replace('_v20','_v21')).read_text())
            self.assertEqual(ast.dump(original),ast.dump(successor))

    def test_protocol_functions_only_rename_and_extend_source_closure(self):
        def functions(text):
            return {n.name:ast.dump(n) for n in ast.parse(text).body
                    if isinstance(n,ast.FunctionDef) and n.name!='required_sources'}
        self.assertEqual(functions(Path(old.__file__).read_text().replace('_v20','_v21')),
                         functions(Path(p.__file__).read_text()))

    def test_all52_science_stages_and_resources_unchanged(self):
        for name in ('STAGES','SCIENTIFIC_STAGES','ROOT_FLOOR','NVME_FLOOR','ROOT_CAP','NVME_CAP','CAPS',
                     'BUCKETS','SPLIT_FLAGS','ALIAS_FLAGS','PACKAGE','COMMIT'):
            self.assertEqual(getattr(old,name),getattr(p,name))
        self.assertEqual(len(p.STAGES),52)
        for stage in (*p.STAGES,'controller'):self.assertEqual(old.resources(stage),p.resources(stage))
        self.assertEqual(p.BUCKETS, (128,256,512,1024,2048,4096))

    def test_successor_completion_and_drain_connected(self):
        self.assertIs(p.attention_completion,completion)
        self.assertIs(p.predecessor_drain,drain.physical_drain)
        self.assertIs(drain.attention,completion)
        self.assertIs(evidence.p,p)
        self.assertIs(p.metadata.storage_metadata,extra_metadata)
        self.assertEqual(p.metadata.__name__,'nemotron9b_package_metadata_v21')

    def test_full_original_source_closure_retained(self):
        paths=set(p.required_sources())
        self.assertTrue(set(old.required_sources()) < paths)
        self.assertTrue(all(Path(path).is_file() for path in paths))
        for name in ('nemotron9b_completed_storage_completion_v1.py','nemotron9b_completed_history_drain_v7.py',
                     'research/nemotron9b_after_completed_storage_v1.py',
                     'research/nemotron9b_completed_storage_metadata_v1.py',
                     'research/check_nemotron9b_package_exec_v21.py'):
            self.assertIn(str(R/name),paths)

    def test_fresh_physical_drain_requires_exact_completed_spec(self):
        with self.assertRaises(TypeError):drain.physical_drain({})
        with patch.object(completion,'validate_spec',side_effect=RuntimeError('actual completion required')) as check:
            with self.assertRaisesRegex(RuntimeError,'actual completion'):
                drain.physical_drain({'base':1},spec={'spec':2})
        check.assert_called_once_with({'spec':2},{'base':1})

    def test_entire_physical_drain_body_preserved(self):
        def body(path):
            return ast.dump(next(n for n in ast.parse(path.read_text()).body
                                if isinstance(n,ast.FunctionDef) and n.name=='physical_drain'))
        self.assertEqual(body(R/'nemotron9b_completed_history_drain_v6.py'),
                         body(R/'nemotron9b_completed_history_drain_v7.py'))

    def test_old_historical_controller_never_exempted(self):
        name='trn1-latest-verify-nemotron9b-package-r9-20261109-controller'
        with patch.object(completion,'validate_spec',return_value=dict(rows=[dict(unit=name)])):
            with self.assertRaisesRegex(RuntimeError,'No historical'):
                drain.physical_drain({},allowed_units=[name],spec={})

    def test_actual_six_new_text_paths_use_bounded_exact_reader(self):
        for path,digest in extra_metadata.EXACT_TEXT.items():
            self.assertEqual(p.metadata.sha(path,digest),digest)
        with self.assertRaises(RuntimeError):p.metadata.sha('/tmp/unapproved-new-download.metadata','0'*64)
        self.assertEqual(p.metadata.LIMITS,old.metadata.LIMITS)

    def test_no_runtime_spec_or_native_framework_import(self):
        self.assertFalse(list(Q.glob('nemotron9b-package-runtime-v21-*-spec.json')))
        self.assertFalse(any(name.split('.')[0] in {'torch','numpy','vllm','neuronxcc'} for name in sys.modules))


class Completion(unittest.TestCase):
    def setUp(self):
        self.prior=dict(rows=[dict(unit=f'synthetic-service-{i}')for i in range(1149)],
            processes=[dict(pid=990000+i,start_ticks=str(i))for i in range(348)],
            known_admitted_directories=[f'synthetic-directory-{i}'for i in range(37)])
        self.exit=dict(path='/synthetic/storage-exit-not-written.json',sha256='a'*64)
        self.residual_exit=dict(path='/synthetic/residual-exit-not-written.json',sha256='b'*64)
        self.history=copy.deepcopy(self.prior)
        self.history['processes'].append(dict(pid=999000,start_ticks='999'))
        self.history.update(completed_storage_maintenance=self.exit,completed_residual_review=self.residual_exit)
        self.proof=dict(history=self.history,source_sha256={'/synthetic/pin.py':'c'*64},
            historical_binary_sha256={f'/synthetic/{i}.bin':'d'*64 for i in range(6)},
            runtime_admitted=False,fresh_physical_drain_verified=False,full_model_qualified=False,new_speedup_claim=False)
        self.spec=dict(required_completed_storage=self.exit,required_residual_completion=self.residual_exit,
                       source_sha256=copy.deepcopy(self.proof['source_sha256']),
                       dory_historical_payload_sha256=copy.deepcopy(self.proof['historical_binary_sha256']))

    def check(self):
        with patch.object(completion.previous,'validate_spec',return_value=self.prior) as old_check, \
             patch.object(completion.maintenance,'completed',return_value=self.proof) as done:
            result=completion.validate_spec(self.spec,{'original':'catalog'})
        old_check.assert_called_once_with(self.spec,{'original':'catalog'})
        done.assert_called_once_with(self.exit)
        return result

    def test_complete_history_adds_only_actual_maintenance_owner(self):
        self.assertIs(self.check(),self.history)

    def test_original_residual_validation_cannot_be_skipped(self):
        with patch.object(completion.previous,'validate_spec',side_effect=RuntimeError('residual not complete')), \
             patch.object(completion.maintenance,'completed') as done:
            with self.assertRaisesRegex(RuntimeError,'residual not complete'):completion.validate_spec(self.spec,{})
        done.assert_not_called()

    def test_missing_actual_storage_exit_rejected(self):
        self.spec.pop('required_completed_storage')
        with patch.object(completion.previous,'validate_spec',return_value=self.prior), \
             patch.object(completion.maintenance,'completed') as done:
            with self.assertRaisesRegex(RuntimeError,'mandatory'):completion.validate_spec(self.spec,{})
        done.assert_not_called()

    def test_omitted_service_owner_or_directory_rejected(self):
        for name in ('rows','processes','known_admitted_directories'):
            with self.subTest(name=name):
                self.setUp();self.history[name].pop()
                with self.assertRaises(RuntimeError):self.check()

    def test_changed_predecessor_owner_rejected(self):
        self.history['processes'][0]['pid']+=1
        with self.assertRaises(RuntimeError):self.check()

    def test_mismatched_residual_completion_rejected(self):
        self.spec['required_residual_completion']={'path':'/other','sha256':'0'*64}
        with self.assertRaises(RuntimeError):self.check()

    def test_incomplete_maintenance_source_closure_rejected(self):
        self.spec['source_sha256'].clear()
        with self.assertRaises(RuntimeError):self.check()

    def test_original_six_binary_pins_required(self):
        self.spec['dory_historical_payload_sha256'].popitem()
        with self.assertRaises(RuntimeError):self.check()

    def test_metadata_does_not_qualify_runtime_or_speed(self):
        for name in ('runtime_admitted','fresh_physical_drain_verified','full_model_qualified','new_speedup_claim'):
            with self.subTest(name=name):
                self.setUp();self.proof[name]=True
                with self.assertRaises(RuntimeError):self.check()


class FullPaths(previous.FullPaths):
    def setUp(self):
        for name,value in (('p',p),('evidence',evidence)):
            change=patch.object(previous,name,value);change.start();self.addCleanup(change.stop)
        super().setUp()


if __name__=='__main__':unittest.main(verbosity=2)
