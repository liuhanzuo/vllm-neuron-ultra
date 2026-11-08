"""Import and unchanged upstream Dory tests in an isolated official CPU build.

No original-checkpoint forward, serving benchmark, or Neuron execution.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

SOURCE = Path('/mnt/nvme/dory-official-cpu-20261108-v1/vllm')
ENV = Path('/mnt/nvme/venvs/dory-official-cpu-20261108-v1')
PINS = {
    'vllm/model_executor/models/dory.py': '483a3adbbef4d2b66e275579e1e505a3d812f3c1c09b10c349e6cbf286df517b',
    'vllm/transformers_utils/configs/dory.py': 'cf0d79e7cbdbf44d34e385f7aa66ff4cfc531dad280d001b8b850e15143817be',
    'tests/models/test_dory.py': '54b9d821d7509cb2bbceb2886ac581255b26c33f3170d2288bea928821ee021f',
    'tests/transformers_utils/test_dory_config.py': 'ecfb396c1b84d02cfc351aa4786aca4d275d701d2d9ba41032a147cd78722f0b',
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path, row):
    encoded=json.dumps(row,indent=2)+'\n'
    with path.open('x') as f:f.write(encoded)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    out = Path(a.output)
    out.mkdir()
    started = time.monotonic()
    try:
        assert Path(sys.executable) == ENV/'bin/python'
        assert os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] == '1'
        assert os.environ['VLLM_PLUGINS'] == ''
        for name, digest in PINS.items():
            assert sha(SOURCE/name) == digest, name
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        assert str(torch.__version__) == '2.14.0+cpu'
        assert torch.version.cuda is None
        import vllm
        import vllm._custom_ops as ops
        from vllm.platforms import current_platform
        from vllm.model_executor.models import dory
        from vllm.transformers_utils.configs import dory as config
        assert current_platform.is_cpu()
        assert Path(vllm.__file__).is_relative_to(ENV)
        assert sha(Path(dory.__file__)) == PINS['vllm/model_executor/models/dory.py']
        assert sha(Path(config.__file__)) == PINS['vllm/transformers_utils/configs/dory.py']
        versions = {n:importlib.metadata.version(n) for n in
            ['torch','vllm','transformers','huggingface_hub','triton','numpy','pytest']}
        save(out/'imports.json', dict(versions=versions,
            vllm_file=vllm.__file__, torch_file=torch.__file__,
            platform=type(current_platform).__name__,
            onednn_available=bool(ops._supports_onednn),
            cpu_capabilities=dict(torch.cpu.get_capabilities()),
            float32_matmul_precision=torch.get_float32_matmul_precision(),
            source_sha256=PINS, ld_preload=os.environ.get('LD_PRELOAD'),
            full_model_forward=False))
        tests = out/'unchanged-official-tests'
        tests.mkdir()
        test_paths=[]
        for name,digest in PINS.items():
            if not name.startswith('tests/'):
                continue
            dest=tests/Path(name).name
            dest.write_bytes((SOURCE/name).read_bytes())
            assert sha(dest)==digest
            test_paths.append(str(dest))
        ini=tests/'pytest.ini'
        ini.write_text('[pytest]\n')
        import pytest
        code=int(pytest.main(['-q','-c',str(ini),'--confcutdir',str(tests),
            '--rootdir',str(tests),'-p','no:cacheprovider',
            '--junitxml',str(out/'upstream-tests.xml'),*test_paths]))
        save(out/'result.json',dict(complete=True,official_test_exit=code,
            passed=code==0,versions=versions,elapsed_seconds=time.monotonic()-started,
            unmodified_official_tests=True,real_official_cpu_linear_modules=True,
            attention_scope='Upstream ReferenceAttention fixture; not serving paged attention',
            original_weight_forward=False,original_two_loop_fixture_added=False,
            full_model_qualified=False,new_speedup_claim=False))
        return code
    except BaseException as e:
        save(out/'failure.json',dict(error_type=type(e).__name__,error=str(e),
            elapsed_seconds=time.monotonic()-started,full_model_qualified=False))
        raise

if __name__=='__main__':
    sys.exit(main())
