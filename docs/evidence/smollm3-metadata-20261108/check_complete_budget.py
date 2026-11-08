"""Budget refusal and actual durable-writer equivalence, no native execution."""
import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
Q=Path(__file__).resolve().parent;sys.path[:0]=[str(Q),str(Q.parent)]
import smollm3_sio12_prospective_budget_v2 as b
import smollm3_observer_metadata_store_v1 as store
class Budget(unittest.TestCase):
 def ledger(self,**change):
  kw=dict(known_control=40*b.MIB,ordinary_parent_peak=1*b.MIB,export_peak=12*b.MIB,observer_peak=7*b.MIB,raw_peak=400000,source_count=36603,history_current={'path':'/synthetic-only.json','sha256':'a'*64});kw.update(change);return b.ledger(**kw)
 def test_full_scientific_matrix_and_all_categories(self):
  v=self.ledger();self.assertTrue(v['forecast_fits_original_caps']);self.assertEqual(len(v['rounds']),12);self.assertEqual(v['full_scientific_request_count'],264);self.assertEqual(v['full_scientific_output_tokens'],33696)
  self.assertEqual([x['mode']for x in v['rounds']],['off','on','on','off']*3)
  for x in v['rounds']:
   self.assertEqual(x['components']['two_service_logs'],32*b.MIB);self.assertEqual(x['components']['concurrent_pending_and_failure_bytes'],16*b.MIB)
   self.assertEqual('active_engine_observer'in x['components'],x['mode']=='on');self.assertEqual(x['headroom_bytes'],x['original_run_cap_bytes']-x['forecast_allocated_bytes'])
  self.assertEqual(v['control_components']['original_terminal_reserve'],8*b.MIB);self.assertFalse(v['runtime_admitted'])
 def test_control_overflow_rejected(self):self.assertFalse(self.ledger(known_control=100*b.MIB)['forecast_fits_original_caps'])
 def test_off_run_growth_rejected(self):self.assertFalse(self.ledger(ordinary_parent_peak=10*b.MIB)['forecast_fits_original_caps'])
 def test_response_growth_rejected(self):self.assertFalse(self.ledger(raw_peak=10*b.MIB)['forecast_fits_original_caps'])
 def test_export_growth_rejected(self):self.assertFalse(self.ledger(export_peak=64*b.MIB)['forecast_fits_original_caps'])
 def test_total_cap_includes_control_and_all12rounds(self):
  with patch.object(b.p,'OUTPUT_CAP',b.GIB):self.assertFalse(self.ledger()['forecast_fits_original_caps'])
 def test_invalid_allocation_inputs_rejected(self):
  for changes in [dict(known_control=-1),dict(raw_peak=True),dict(source_count=0)]:
   with self.assertRaises(AssertionError):self.ledger(**changes)
 def test_actual_writer_matches_both_wire_formats(self):
  values=[{'a':list(range(30)),'nested':{'b':True}}, {'source_stats':{},'none':None},{'runtime_source_sha256':{'/source.py':'a'*64},'owner':{'pid':7,'exit':False}}]
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp);(root/'control').mkdir(mode=0o700);store.provision(root)
   with patch.object(b.life,'DEST',root):
    for i,value in enumerate(values):
     path=root/'control'/('sample'+str(i)+'.json');b.life.record(path,value);wire,blobs,kind=b.encoded_record(value);self.assertEqual(path.read_bytes(),wire);self.assertEqual(store.decode(root,path,wire),value)
     for digest,raw in blobs.items():self.assertEqual((root/'control/source-maps'/(digest+'.json')).read_bytes(),raw)
     self.assertEqual(kind,'compact_source_record'if i==2 else'original_indented_json')
 def test_projection_preserves_failure_and_owner_fields(self):
  original={'source_stats':{'/old.py':{}},'runtime_source_sha256':{'/old.py':'a'*64},'failure':True,'exit_code':1,'owner':{'pid':7,'start_ticks':'8'},'rows':[{'unit':'old','passed':False}]}
  v=b.transform(original,{'/new.py':'b'*64},{'/new.py':{'size':9}}, {'/new.py':'b'*64})
  for key in ('failure','exit_code','owner','rows'):self.assertEqual(v[key],original[key])
  self.assertEqual(v['runtime_source_sha256'],{'/new.py':'b'*64});self.assertIn('/old.py',original['runtime_source_sha256'])
if __name__=='__main__':unittest.main(verbosity=2)
