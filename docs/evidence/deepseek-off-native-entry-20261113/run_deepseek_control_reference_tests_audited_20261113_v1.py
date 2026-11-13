"""Run source/in-memory native-loader tests; prohibit real inputs and execution."""
import importlib.abc
import os
from pathlib import Path
import sys
import unittest

Q = Path(__file__).resolve().parent
sys.path[:0] = [str(Q),str(Q.parent)]
class RejectNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','numpy','transformers','torch_neuronx','neuronxcc','vllm',
                'vllm_neuron','neuronx_distributed','neuronx_distributed_inference'}:
            raise AssertionError('No native imports in synthetic native-loader tests: '+fullname)
sys.meta_path.insert(0,RejectNative())
def audit(event,args):
    if event in {'subprocess.Popen','os.system','os.fork','os.forkpty','os.posix_spawn','os.exec','os.kill','os.killpg',
            'fcntl.flock','socket.connect','socket.bind','socket.getaddrinfo'}:
        raise AssertionError('No process/lock/network in native-loader source tests: '+event)
    if event == 'open' and isinstance(args[0],(str,bytes)):
        name=os.fsdecode(args[0]); p=Path(name); mode=args[1]; flags=args[2]
        if name.startswith(('/proc/','/sys/','/mnt/nvme/')) or p.suffix in {'.json','.raw','.pt','.bin','.safetensors','.neff','.npy','.pth','.pb'}:
            raise AssertionError('No actual runtime/closure/model/payload metadata: '+name)
        writing=isinstance(mode,str) and any(x in mode for x in 'wax+') or isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
        if writing: raise AssertionError('Fixture state must remain in memory: '+name)
sys.addaudithook(audit)
suite=unittest.TestLoader().loadTestsFromName('test_deepseek_cold_control_reference_entry_v1')
result=unittest.TextTestRunner(verbosity=2).run(suite)
print('AUDIT_GUARD: source/in-memory fixtures only; full42-loop fixture uses fake tensors; no native/model/payload/proc/lock',flush=True)
raise SystemExit(0 if result.wasSuccessful() else 1)
