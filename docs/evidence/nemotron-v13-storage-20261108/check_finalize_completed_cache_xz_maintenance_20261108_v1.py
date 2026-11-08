"""Exercise the finalizer transformation on explicitly synthetic future outcomes."""
import copy,sys,unittest
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from research.check_nemotron9b_post_smol_maintenance_v2 import fixture
import nemotron9b_post_smol_maintenance_v2 as bridge
class Checks(unittest.TestCase):
 def test_exact_append_and_all_original_fields(self):
  spec,catalog,docs=fixture();prior=docs[bridge.PRIOR['path']]
  self.assertEqual(catalog['processes'][:-3],prior['processes']);self.assertEqual(len(catalog['processes']),168)
  self.assertEqual(catalog['rows'],prior['rows']);self.assertEqual(catalog['root_release_refs'],prior['root_release_refs'])
  self.assertTrue(all(catalog['source_sha256'].get(k)==v for k,v in prior['source_sha256'].items()))
  result=bridge.validate(spec,catalog,lambda ref:docs[ref['path']]);self.assertFalse(result['fresh_physical_drain_proven'])
  self.assertEqual([docs[row['actual_exit']['path']]['exit_code'] for row in result['maintenance']],[1,0,0])
if __name__=='__main__':unittest.main(verbosity=2)
