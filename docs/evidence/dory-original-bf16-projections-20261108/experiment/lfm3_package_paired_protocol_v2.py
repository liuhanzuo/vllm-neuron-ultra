"""Frozen LFM3 full-model paired successor: stdlib protocol, no work on import."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
DEST = ROOT / 'lfm3-package-paired-full-v2'
WORK = Path('/mnt/nvme/lfm3-package-paired-full-v2')
CORE = ROOT / 'lfm3-package-native-core-v1'
TOKENIZER = ROOT / 'lfm3-package-tokenizer-v1'
TARGET = CORE / 'work/target'
VIEW = TOKENIZER / 'view'
PLAN = ROOT / 'research/lfm3-package-paired-full-plan-20261107-v2.json'
COMMIT = '822e1bd826321dc078226c2da3f199e1d385146f'
PYTHON = '/home/ubuntu/venvs/vllm_neuron/bin/python'
CLI = '/home/ubuntu/venvs/vllm_neuron/bin/vllm'
INPUTS = ROOT / 'research/lfm3-package-native-preflight-inputs-20261106-v1.json'
ROLES = ('reference', 'candidate')
CRITERIA = dict(cosine_min=.995, kl_max_nats=.02, max_abs_error_max=2.,
                argmax_match_fraction_min=.9, all_reference_margins_gt_0_5_must_match=True)
LENGTH = 2048
STEPS = 1529
MARKS = sorted(set([0,22,30,31,32,33,63,64,65,126,127,128,129,255,256,257,
                    510,511,512,513,1023,1024,1025,1530,1531,1656,1657,2039,2040,2041]))
SPECS = [('short_decode',128,512,{}), ('medium_prompt',512,128,{}),
         ('long_prompt',1536,128,{}),
         ('short_stochastic',128,512,dict(temperature=.7,top_p=.9,top_k=40)),
         ('mixed_lengths',None,None,dict(temperature=.7,top_p=.9,top_k=40))]
# Alternate complete role visits by round. Each visit uses a fresh service,
# identical warmup and all five original workloads. Raw measurements retained.
ROUND_ORDER = [('reference','candidate'),('candidate','reference'),('reference','candidate')]

def read(path):
    path = Path(path)
    assert path.is_file() and not path.is_symlink() and path.stat().st_size < 4 * 2**20, path
    return json.loads(path.read_text())

def sha(path):
    path = Path(path)
    assert path.stat().st_size < 4 * 2**20 and path.suffix not in ('.pt','.safetensors','.neff','.whl','.so'), path
    return hashlib.sha256(path.read_bytes()).hexdigest()

def stream_sha(path):
    # Only admitted native phase uses this on frozen oracle payloads.
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(2**20): result.update(block)
    return result.hexdigest()

def once(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())

def quiet():
    value = read(ROOT / 'smollm3-cache-ablation-full-v8/status.json')
    assert value['performance_window'] is False, 'Smol timed window: stop at tiny status read'
    return value

def case(role):
    assert role in ROLES
    return dict(name='lfm3-package-paired-full-v2-'+role, role=role,
        model_path=str(VIEW), served_name='lfm3-1.7b', artifact=str(WORK / ('artifact-'+role)),
        batch=4, tp=2, max_model_len=2048, cpu=True, cpu_fast_sampler=True,
        adapter='packaged_lfm3', extra_overrides=dict(apply_seq_ids_mask=True),
        implementation_commit=COMMIT)

def passed(rows):
    return bool(rows) and all(r['cosine']>=.995 and r['kl_reference_to_neuron']<=.02
        and r['max_abs_error']<=2 and (r['reference_margin']<=.5 or r['argmax_match']) for r in rows) \
        and sum(r['argmax_match'] for r in rows)/len(rows)>=.9

def environment(role, compile_allowed=False):
    assert role in ROLES
    return dict(HOME='/home/ubuntu', USER='ubuntu', LANG='C.UTF-8',
        PATH='/home/ubuntu/venvs/vllm_neuron/bin:/opt/aws/neuron/bin:/usr/bin:/bin',
        PYTHONOPTIMIZE='0', PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
        PYTHONPYCACHEPREFIX=str(DEST/'fresh-pycache'), PYTHONPATH=str(TARGET)+':'+str(ROOT),
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HOME=str(DEST/'hf'),
        XDG_CACHE_HOME=str(WORK/'xdg'), TMPDIR=str(WORK/'tmp'),
        NEURON_COMPILE_CACHE_URL=str(WORK/('compile-cache-'+role)),
        BASE_COMPILE_WORK_DIR=str(WORK/('compile-work-'+role)),
        NEURON_COMPILED_ARTIFACTS=case(role)['artifact'],
        VLLM_PLUGINS='neuron,lfm3', VLLM_WORKER_MULTIPROC_METHOD='spawn',
        NEURON_LFM3='1', NEURON_LFM3_NKI_WRITER='1' if role=='candidate' else '0',
        NEURON_LFM3_ALLOW_COMPILE='1' if compile_allowed else '0', NEURON_SMOLLM3='0',
        NEURON_CPU_SAMPLER_FAST='1', NEURON_RT_VISIBLE_CORES='0-1',
        TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        TORCHINDUCTOR_COMPILE_THREADS='1', TORCHINDUCTOR_CACHE_DIR=str(WORK/'torch-cache'),
        PIP_NO_INDEX='1', PIP_DISABLE_PIP_VERSION_CHECK='1')

def server_argv():
    return [PYTHON,'-B',CLI,'serve',str(VIEW),'--served-model-name','lfm3-1.7b',
        '--tensor-parallel-size','2','--dtype','bfloat16','--max-model-len','2048',
        '--max-num-seqs','4','--no-enable-prefix-caching','--host','127.0.0.1','--port','8010',
        '--additional-config',json.dumps(dict(override_neuron_config=dict(
            on_device_sampling_config=None,apply_seq_ids_mask=True)),separators=(',',':'))]


def primitive_admission(phase):
    quiet()
    assert os.environ.get('LFM_PACKAGE_PRIMITIVE_ADMITTED')=='1'
    assert os.environ['LFM_PACKAGE_PHASE']==phase
    plan=read(PLAN);assert sha(PLAN)==os.environ['LFM_PACKAGE_PLAN_SHA256']
    for name,digest in plan['source_sha256'].items(): assert sha(name)==digest,name
    gate=read(DEST/'admission.json');assert gate['native_authorized'] and gate['fresh_compilation_authorized']
    owner=read(DEST/(os.environ['LFM_PACKAGE_LABEL']+'-owner.json'))
    assert Path('/proc',str(owner['pid'])).exists()
    assert Path('/proc/self/cgroup').read_text().strip()=='0::'+owner['cgroup']
