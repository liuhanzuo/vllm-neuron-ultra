"""Independent audit guard around already delivered SOURCE/MOCK tests only."""
import os,sys,unittest
from pathlib import Path
q=Path('/home/ubuntu/experiments/trn1-latest-models-20261031/research')
sys.path[:0]=[str(q),str(q.parent),str(q/'qwen35_package_serial_runtime_v9')]
def audit(event,args):
    if event in {'subprocess.Popen','os.system','os.fork','os.posix_spawn','fcntl.flock','socket.connect','socket.bind','socket.getaddrinfo'}:
        raise AssertionError('Independent source-only guard forbids '+event)
    if event=='open'and isinstance(args[0],(str,bytes)):
        p=os.fsdecode(args[0]);path=Path(p);mode=args[1];flags=args[2]
        if p.startswith(('/proc/','/sys/','/mnt/nvme/'))or path.name in ('serial-enrollment.lock','continuation-state.json')or path.name.startswith('continuation-handoff-'):
            raise AssertionError('Independent source-only guard forbids runtime path '+p)
        if p.startswith('/home/ubuntu/venvs/')and path.suffix in ('.bin','.pt','.raw','.npy','.safetensors','.neff'):
            raise AssertionError('Independent source-only guard forbids SDK/model binary '+p)
        writing=isinstance(mode,str)and any(c in mode for c in 'wax+')or isinstance(flags,int)and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
        if writing and not p.startswith('/tmp/'):
            raise AssertionError('Only temporary fixtures may be written: '+p)
sys.addaudithook(audit)
loader=unittest.TestLoader()
group=sys.argv[1]
if group=='runtime':suite=loader.discover(str(q/'qwen35_package_serial_runtime_v9'),pattern='test*.py')
elif group=='helpers':suite=loader.loadTestsFromNames(['test_prepare_qwen35_package_serial_candidate_v8','test_render_qwen35_package_r7_root_helpers_source_v1','test_qwen35_package_r7_preserved_observer_v1'])
elif group=='independent':
    sys.path.insert(0,'/tmp');suite=loader.loadTestsFromName('test_qwen35_r7_mini_independent_20261113_v1')
elif group=='pure-stage25':
    import runpy
    runpy.run_path(str(q/'check_qwen35_package_stage25_import_v2.py'),run_name='__main__')
    print('INDEPENDENT_SOURCE_ONLY_AUDIT_GUARD_ACTIVE pure-stage25 checks 5',flush=True)
    raise SystemExit(0)
else:raise AssertionError('Exact finite group')
result=unittest.TextTestRunner(verbosity=2).run(suite)
print('INDEPENDENT_SOURCE_ONLY_AUDIT_GUARD_ACTIVE',group,'tests',result.testsRun,flush=True)
raise SystemExit(0 if result.wasSuccessful()else 1)
