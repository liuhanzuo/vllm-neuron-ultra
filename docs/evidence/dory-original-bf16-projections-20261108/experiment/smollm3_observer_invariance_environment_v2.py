"""Exact source-pinned dependency environment transitions; stdlib only.

Never assigns environment variables. Unknown keys fail closed without values.
Off/on use the same policy; native import and scientific behavior stay intact.
"""
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parent
SITE=Path('/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages')
CONTRACT=ROOT/'research/smollm3-observer-environment-source-contract-20261107-v2.json'
STAGES=('clean','startup','import','serve')
SYSTEM_KEYS={'INVOCATION_ID','JOURNAL_STREAM','SYSTEMD_EXEC_PID','MEMORY_PRESSURE_WATCH','MEMORY_PRESSURE_WRITE'}


def constants():
    return dict(PYTORCH_NVML_BASED_CUDA_CHECK='1',TF_CPP_MIN_LOG_LEVEL='1',GRPC_VERBOSITY='ERROR',
        XLA_FLAGS=' --xla_cpu_enable_fast_math=false --xla_gpu_force_compilation_parallelism=8',
        NEURON_LIBRARY_PATH=str(SITE/'libneuronxla/libneuronpjrt.so'),PJRT_DEVICE='NEURON',
        XLA_DISABLE_FUNCTIONALIZATION='1',XLA_IR_SHAPE_CACHE_SIZE='20480',
        NEURON_RT_ROOT_COMM_ID='localhost:62182',ALLREDUCE_GRADIENTS_BUCKET_SIZE_MB='50',JAX_LOGGING_LEVEL='NOTSET')


def permitted(launch,kind,stage,spawned):
    if kind=='client':return {}
    if stage in ('clean','startup')and not spawned:return {}
    values=constants()
    # Both are conditional in pinned source. Only the exact installed-source
    # value is accepted; absence is permitted, arbitrary cache paths are not.
    values.update(PJRT_DYNAMIC_PLUGINS='1',TORCHINDUCTOR_CACHE_DIR=str(Path(launch['TMPDIR'])/'torchinductor_ubuntu'))
    # Exact setdefault values from pinned sklearn/threadpoolctl/joblib sources.
    # Conditional only: absence is allowed, but alternate spelling is rejected.
    values.update(KMP_DUPLICATE_LIB_OK='True',KMP_INIT_AT_FORK='FALSE')
    if stage=='serve'or spawned:values['VLLM_USE_V1']='1'
    return values


def check(actual,launch,kind,stage,spawned,system):
    if kind not in ('server','client')or stage not in STAGES or type(spawned)is not bool or(kind=='client'and spawned):
        raise RuntimeError('Unknown environment phase identity')
    if set(system)-SYSTEM_KEYS:raise RuntimeError('Unreviewed system environment keys')
    wanted=dict(launch);wanted.update(system)
    changed=sorted(k for k,v in wanted.items()if actual.get(k)!=v)
    if changed:raise RuntimeError('Environment launch/system value changed; keys='+','.join(changed))
    extras=permitted(launch,kind,stage,spawned)
    unknown=sorted(set(actual)-set(wanted)-set(extras))
    if unknown:raise RuntimeError('Unreviewed environment keys='+','.join(unknown))
    invalid=sorted(k for k,v in extras.items()if k in actual and actual[k]!=v)
    if invalid:raise RuntimeError('Dependency environment value mismatch; keys='+','.join(invalid))
    required=set(constants())if kind=='server'and(stage=='serve'or spawned)else set()
    if spawned:required.add('VLLM_USE_V1')
    missing=sorted(required-set(actual))
    if missing:raise RuntimeError('Required dependency environment missing; keys='+','.join(missing))
    return dict(stage=stage,kind=kind,spawned=spawned,dependency_keys=sorted(set(actual)&set(extras)),
                exact_values_checked=True,unknown_keys=[],values_recorded=False)


def capture_system(actual,launch,permit):
    """Systemd slots retain exact initial values; no dependency key is trusted here."""
    system={k:actual[k]for k in SYSTEM_KEYS if k in actual and k not in launch}
    if 'INVOCATION_ID'in system and system['INVOCATION_ID']!=permit['invocation_id']:raise RuntimeError('System invocation environment mismatch')
    if 'SYSTEMD_EXEC_PID'in system and system['SYSTEMD_EXEC_PID']!=str(permit['pid']):raise RuntimeError('System PID environment mismatch')
    if 'JOURNAL_STREAM'in system and not re.fullmatch(r'[0-9]+:[0-9]+',system['JOURNAL_STREAM']):raise RuntimeError('System journal environment shape')
    # env -i intentionally strips systemd memory-pressure fields in this harness.
    if {'MEMORY_PRESSURE_WATCH','MEMORY_PRESSURE_WRITE'}&set(system):raise RuntimeError('Unexpected system memory-pressure environment')
    return system
