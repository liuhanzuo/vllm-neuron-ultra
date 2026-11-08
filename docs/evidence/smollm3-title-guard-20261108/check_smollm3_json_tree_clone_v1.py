"""Copy contract tests for retained permits; no runtime admission/native work."""
from pathlib import Path
import copy,json,math,random,struct,sys,unittest
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import smollm3_json_tree_clone_v1 as candidate
import smollm3_observer_metadata_store_v3 as store
import smollm3_observer_invariance_lifecycle_v22 as life

def strict_equal(a,b):
 assert type(a)is type(b),(type(a),type(b))
 if type(a)is dict:
  assert a.keys()==b.keys()and a is not b
  for key in a:strict_equal(a[key],b[key])
 elif type(a)is list:
  assert len(a)==len(b)and a is not b
  for x,y in zip(a,b):strict_equal(x,y)
 elif type(a)is float:assert struct.pack('!d',a)==struct.pack('!d',b)
 else:assert a==b

class Clone(unittest.TestCase):
 def test_scalar_types_unicode_and_float_bits(self):
  v={'nested':[None,False,True,0,-1,2**200,'中\n',0.0,-0.0,1e-300,1e308],'empty':{},'array':[]}
  strict_equal(v,candidate.clone(v))
 def test_mutable_containers_detached_in_both_directions(self):
  a={'x':[{'y':[1]}]};b=candidate.clone(a);b['x'][0]['y'].append(2);self.assertEqual(a,{'x':[{'y':[1]}]});a['x'].append(3);self.assertEqual(len(b['x']),1)
 def test_mutable_aliases_and_cycles_rejected(self):
  child={};alias={'a':child,'b':child};cycle={};cycle['a']=cycle
  for v in [alias,cycle]:
   with self.assertRaises(ValueError):candidate.clone(v)
 def test_non_json_objects_and_keys_rejected(self):
  class D(dict):pass
  class S(str):pass
  for v in [{1:'x'},{'v':(1,)},{'v':set()},{'v':b'x'},{'v':D()},{'v':S('x')},D(),[],None]:
   with self.assertRaises(ValueError):candidate.clone(v)
 def test_nonfinite_numbers_rejected(self):
  for x in [math.inf,-math.inf,math.nan]:
   with self.assertRaises(ValueError):candidate.clone({'v':x})
 def test_depth_bound(self):
  v={};current=v
  for _ in range(48):current['v']={};current=current['v']
  candidate.clone(v);current['v']={}
  with self.assertRaises(ValueError):candidate.clone(v)
 def test_byte_bound_uses_utf8(self):
  original=candidate.MAX_BYTES;candidate.MAX_BYTES=12
  try:
   candidate.clone({'v':'中'})
   with self.assertRaises(ValueError):candidate.clone({'v':'中中'})
  finally:candidate.MAX_BYTES=original
 def test_random_plain_json_trees_match_deepcopy(self):
  rng=random.Random(634288)
  def value(depth=0):
   choices=[None,False,True,rng.randrange(-1000,1000),'中'+str(rng.randrange(50)),rng.uniform(-1e10,1e10)]
   if depth<4:choices.extend([{str(i):value(depth+1)for i in range(rng.randrange(3))},[value(depth+1)for _ in range(rng.randrange(3))]])
   return rng.choice(choices)
  for _ in range(1000):
   v={'v':value()};actual=candidate.clone(v);self.assertEqual(actual,copy.deepcopy(v));strict_equal(v,actual)
 def test_actual_retained_parent_and_child_permits(self):
  for pid in [2622787,2626813]:
   path=R/'sio13/r00'/('permit-server-'+str(pid)+'.json');v=store.decode(R/'sio13',path,life.read_bytes(path))['permit'];actual=candidate.clone(v);strict_equal(v,actual)
   self.assertEqual(life.p.digest(v),life.p.digest(actual));self.assertEqual(len(v['root_runtime_source_sha256']),len(actual['root_runtime_source_sha256']))
if __name__=='__main__':unittest.main(verbosity=2)
