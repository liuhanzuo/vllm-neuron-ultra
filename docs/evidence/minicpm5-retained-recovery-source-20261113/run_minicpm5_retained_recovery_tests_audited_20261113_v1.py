import importlib.abc,os,sys,unittest
from pathlib import Path
q=Path('/home/ubuntu/experiments/trn1-latest-models-20261031/research');sys.path[:0]=[str(q),str(q.parent)]
class RejectNative(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0]in {'torch','numpy','transformers','torch_neuronx','neuronxcc','vllm','neuronx_distributed','neuronx_distributed_inference'}:
   raise AssertionError('No native import in synthetic preparer tests: '+fullname)
sys.meta_path.insert(0,RejectNative())
def audit(event,args):
 if event in {'subprocess.Popen','os.system','os.fork','os.posix_spawn','fcntl.flock','socket.connect','socket.bind','socket.getaddrinfo'}:
  raise AssertionError('No runtime operation in preparer tests: '+event)
 if event=='open'and isinstance(args[0],(str,bytes)):
  name=os.fsdecode(args[0]);p=Path(name);mode=args[1];flags=args[2]
  if name.startswith(('/proc/','/sys/','/mnt/nvme/'))or p.suffix in ('.json','.raw','.pt','.bin','.safetensors','.neff','.npy'):
   raise AssertionError('No actual metadata/plan/payload access in preparer tests: '+name)
  writing=isinstance(mode,str)and any(x in mode for x in 'wax+')or isinstance(flags,int)and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
  if writing:raise AssertionError('All synthetic fixture values are memory only: '+name)
sys.addaudithook(audit)
suite=unittest.TestLoader().loadTestsFromNames(['test_minicpm5_retained_replay_recovery_v1'])
result=unittest.TextTestRunner(verbosity=2).run(suite)
print('AUDIT_GUARD: no actual JSON/plan/raw/model/SDK input; no lock/process/native; all metadata in-memory',flush=True)
raise SystemExit(0 if result.wasSuccessful()else 1)
