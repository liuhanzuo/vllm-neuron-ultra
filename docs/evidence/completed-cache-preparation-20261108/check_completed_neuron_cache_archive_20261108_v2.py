"""Verify destructive-path rejection and source-only behavior; no archive runs."""
from pathlib import Path
import argparse,copy,contextlib,io,json,sys,unittest
from unittest.mock import patch
Q=Path(__file__).resolve().parent;sys.path.insert(0,str(Q))
import archive_completed_neuron_cache_20261108_v2 as a
VALUE=json.loads(a.CANDIDATE.read_text())
class Checks(unittest.TestCase):
 def setUp(self):self.value=copy.deepcopy(VALUE)
 def reject(self):
  with self.assertRaises((AssertionError,KeyError,ValueError)):a.validate_candidate(self.value)
 def test_actual_selected_metadata_and_live_file_identities(self):
  value,root=a.check_candidate();self.assertEqual(value['cache_entries'],253);self.assertEqual(value['files'],1039);self.assertEqual(value['allocated_bytes'],4607754240)
 def test_unrelated_cache_root_rejected(self):self.value['cache_root']='/mnt/nvme/models';self.reject()
 def test_parent_escape_entry_rejected(self):self.value['candidates'][0]['path']=str(a.DEST/'other');self.reject()
 def test_file_outside_entry_rejected(self):self.value['candidates'][0]['files'][0]['path']='/mnt/nvme/models/model.safetensors';self.reject()
 def test_incomplete_entry_rejected(self):
  item=self.value['candidates'][0];item['files']=[x for x in item['files']if not x['path'].endswith('/model.done')];self.reject()
 def test_lock_bearing_entry_rejected(self):
  item=self.value['candidates'][0];row=copy.deepcopy(item['files'][0]);row['path']=item['path']+'/model.hlo_module.pb.lock';item['files'].append(row);self.reject()
 def test_file_hardlink_rejected(self):self.value['candidates'][0]['files'][0]['nlink']=2;self.reject()
 def test_file_symlink_rejected(self):self.value['candidates'][0]['files'][0]['mode']=0o120777;self.reject()
 def test_duplicate_entry_rejected(self):self.value['candidates'].append(self.value['candidates'][0]);self.reject()
 def test_total_size_tampering_rejected(self):self.value['allocated_bytes']+=1;self.reject()
 def test_completed_status_cannot_be_invented(self):self.value['candidates'][0]['done_marker']=False;self.reject()
 def test_check_never_acquires_lock_or_launches_native_tools(self):
  with patch.object(a.lifecycle,'acquire',side_effect=AssertionError('must not acquire')),patch.object(a.p,'accelerator_idle',side_effect=AssertionError('must not inspect native')),patch.object(a,'completed_catalog',side_effect=AssertionError('must not accept terminal')),contextlib.redirect_stdout(io.StringIO())as out:
   a.main(argparse.Namespace(check=True,apply=False,catalog=None,catalog_sha256=None))
  result=json.loads(out.getvalue());self.assertFalse(result['archive_started']);self.assertFalse(result['deletion_performed']);self.assertEqual(result['payload_bytes_read'],0)
 def test_apply_without_actual_final_catalog_rejected_before_lock(self):
  with patch.object(a.lifecycle,'acquire',side_effect=AssertionError('must not acquire'))as acquire:
   with self.assertRaisesRegex(AssertionError,'finalized sio14 catalog'):a.main(argparse.Namespace(check=False,apply=True,catalog=None,catalog_sha256=None))
   acquire.assert_not_called()
 def test_old_actual_nemotron_catalog_cannot_substitute_for_smol_exit(self):
  with self.assertRaises(AssertionError):a.completed_catalog(argparse.Namespace(catalog=str(a.Q/'nemotron9b-package-r4-20261108-complete-catalog-v3.json'),catalog_sha256='17623034921ec7a5e48d68bd9a22e6377bf2ce0846931d280aa57ca20f612207'))
if __name__=='__main__':unittest.main(verbosity=2)
