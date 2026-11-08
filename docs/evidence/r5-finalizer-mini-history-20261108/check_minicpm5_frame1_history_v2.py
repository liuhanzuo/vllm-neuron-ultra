"""Actual large-history reader regression and preservation of original Mini science."""
import ast,copy,hashlib,json,sys,unittest
from pathlib import Path
from unittest.mock import patch
R=Path(__file__).resolve().parents[1];Q=R/'research';sys.path.insert(0,str(R));sys.path.insert(0,str(Q))
import minicpm5_frame1_mlp_runner_v1 as old
import minicpm5_frame1_mlp_runner_v2 as new
import prepare_minicpm5_frame1_runtime_v2 as builder
CAT=dict(path=str(Q/'nemotron9b-package-post-smol-maintenance-r4-20261108-catalog-v2.json'),sha256='1bc9c18fb1c809018be394928e8b52bdce8c49a6b854a1b66dcada9ce386edd6')
class Checks(unittest.TestCase):
 def test_runner_only_history_reader_import_changed(self):
  source=Path(old.__file__).read_text().replace('import nemotron9b_package_metadata_v12 as history_metadata','import nemotron9b_package_metadata_v13 as history_metadata')
  self.assertEqual(source,Path(new.__file__).read_text())
 def test_builder_only_runner_and_test_entrypoint_changed(self):
  source=(Q/'prepare_minicpm5_frame1_runtime_v1.py').read_text().replace('import minicpm5_frame1_mlp_runner_v1 as r','import minicpm5_frame1_mlp_runner_v2 as r').replace("TEST=str(Q/'check_prepare_minicpm5_frame1_runtime_v1.py')","TEST=str(Q/'check_prepare_minicpm5_frame1_runtime_v2.py')")
  self.assertEqual(source,Path(builder.__file__).read_text())
 def test_actual_old_reader_rejects_new_large_history(self):
  with self.assertRaisesRegex(RuntimeError,'Explicit named research envelope schema'):old.history_metadata.load(CAT)
 def test_actual_new_reader_accepts_exact_history_and_rejects_hash_change(self):
  value=new.history_metadata.load(CAT);self.assertEqual(len(value['rows']),863);self.assertEqual(len(value['processes']),168)
  with self.assertRaisesRegex(RuntimeError,'fingerprint'):new.history_metadata.load(dict(CAT,sha256='0'*64))
 def test_complete_actual_smol_anchor_and_maintenance_history(self):
  current=new.history_metadata.load(CAT);smol_ref=current['maintenance_extension']['prior_catalog'];smol=new.history_metadata.load(smol_ref)
  chain=builder.select_chain(smol_ref,smol,new.history_metadata.load)
  spec=dict(history_anchor=new.HISTORY_ANCHOR,required_sio14_catalog=smol_ref,current_catalog=CAT,required_smol_predecessor_chain=chain)
  self.assertEqual(new.current_history(spec),current)
 def test_exact_raw_partition_and_alias_rules_remain(self):
  self.assertIs(new.history_partition,old.history_partition);self.assertIs(new.alias_metadata,old.alias_metadata)
  self.assertEqual(new.history_metadata.NON_UTF8,old.history_metadata.NON_UTF8);self.assertEqual(new.history_metadata.LIMITS,old.history_metadata.LIMITS)
 def test_science_and_all_resource_limits_identical(self):
  for name in ['PHASES','CAP','BASE_PLAN','BASE_SPEC','MODEL_EXIT','NUMERIC_EXIT','DESIGN_REVIEW','FAILED_ATTEMPT','HISTORY_ANCHOR','CANONICAL']:self.assertEqual(getattr(new,name),getattr(old,name),name)
  self.assertIs(new.worker,old.worker);self.assertEqual(new.worker.science(),old.worker.science())
  for phase in new.PHASES:self.assertEqual(new.props(phase),old.props(phase));self.assertEqual(new.environment(phase,'fixture',29571),old.environment(phase,'fixture',29571))
 def test_generic_unrelated_raw_reader_not_enabled(self):
  with patch.object(new.history_metadata,'stable_bytes')as read:
   with self.assertRaisesRegex(RuntimeError,'Reviewed UTF8'):new.history_metadata.text(R/'unrelated.raw','a'*64)
   read.assert_not_called()
 def test_no_model_or_native_framework_import(self):self.assertFalse({'torch','torch_neuronx','numpy','vllm','neuronxcc'}&set(sys.modules))
if __name__=='__main__':
 import check_minicpm5_frame1_mlp_runner_v2 as runner_tests
 import check_prepare_minicpm5_frame1_runtime_v2 as builder_tests
 suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(Checks),unittest.defaultTestLoader.loadTestsFromModule(runner_tests),unittest.defaultTestLoader.loadTestsFromModule(builder_tests)])
 result=unittest.TextTestRunner(verbosity=2).run(suite);raise SystemExit(0 if result.wasSuccessful()else 1)
