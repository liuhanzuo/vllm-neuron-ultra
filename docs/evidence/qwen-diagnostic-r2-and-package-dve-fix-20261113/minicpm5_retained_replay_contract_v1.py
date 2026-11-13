"""Pure finite pair/replay binding. No current-file discovery or IO on import."""
import copy
import hashlib
import json
from pathlib import PurePosixPath as Path
import re

ROOT=Path('/home/ubuntu/experiments/trn1-latest-models-20261031');Q=ROOT/'research'
PYTHON='/home/ubuntu/venvs/vllm_neuron/bin/python'
LOCK=dict(device=66306,inode=8294421)  # Exact inherited qwen35_sampler_runtime_contract_v12 literal.
PLAN=dict(path=str(Q/'minicpm5-retained-restore-control-transport-plan-20261113-v1.json'),
    sha256='996215f14e08e8b1a2a1e8f0e53e776a0dfe919102a60411a7e316706c5d61da')
ORIGINAL_PLAN=dict(path=str(ROOT/'minicpm5-history-analysis-recovery-v2/plan.json'),
    sha256='ea3954682dfd8ba14e1559ed02fa77d938bc987a839cd77c89238520c32c3252')
ACCEPTED=dict(path=str(Q/'minicpm5-retained-raw-finite-successor-root-source-review-20261113-v1.json'),
    sha256='d730c77b604c45d3439b9914afded76658edbe06d87dd621525f8355968c1bfd')
FROZEN={str(Q/('minicpm5_retained_raw_'+name+'_v1.py')):sha for name,sha in [
 ('transport','9ac6a3642de1d46e1517ca85a0b0dd94abcf6779369d508b306279d2a53616a9'),
 ('legacy','3ca6b7f7eed43a21955862f8caf99c757ee7a09a2a9c1a21c34961410ceb9d85'),
 ('compare','32423f081dce3fb168db206c262a93fc8f169bb272ee0684d6d1064e005e8dbc'),
 ('transaction','254cc33487f0f96e0df7df5b685d4b89fae3775159fe27c14f3f7175e5edb690')]}
FILES=tuple(str(Q/('minicpm5_retained_replay_'+name+'_v1.py'))for name in ('contract','history','caller','observer'))
LIMITS=dict(pair_seconds=7200,replay_seconds=43200,outer_seconds=51000,ack_seconds=180,
    ordinary_metadata_bytes=8<<20,spec_bytes=64<<20,control_bytes=128<<20,replay_bytes=256<<20,
    file_bytes=32<<20,log_bytes=8<<20,max_entries=512,root_floor_bytes=28<<30,nvme_floor_bytes=5<<30)
PROPERTIES=dict(MemoryMax=str(6<<30),MemoryHigh=str(5<<30),MemorySwapMax='0',TasksMax='32',
    AllowedCPUs='7',CPUAffinity='7',Nice='19',KillMode='control-group',RemainAfterExit='yes',
    WorkingDirectory=str(ROOT),TimeoutStopSec='60',LimitFSIZE=str(32<<20))


def require(ok,message):
    if not ok:raise ValueError(message)


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(data):return hashlib.sha256(data).hexdigest()


def reference(value):
    require(type(value)is dict and set(value)=={'path','sha256'} and type(value['path'])is str
        and Path(value['path']).is_absolute() and '..'not in Path(value['path']).parts
        and str(Path(value['path']))==value['path'] and type(value['sha256'])is str
        and re.fullmatch('[0-9a-f]{64}',value['sha256']),'Exact absolute reference')


def names(run):
    require(type(run)is str and re.fullmatch('[a-z0-9][a-z0-9-]{0,47}',run),'Finite root-supplied run name')
    return dict(pair='trn1-latest-verify-minicpm5-retained-pairs-'+run,
        replay='trn1-latest-verify-minicpm5-history-retained-replay-'+run+'-compare')


def environment(destination):
    return dict(PYTHONOPTIMIZE='0',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
        PYTHONPYCACHEPREFIX=str(Path(destination)/'empty-pycache'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        TMPDIR=str(Path(destination)/'tmp'),CUDA_VISIBLE_DEVICES='',NEURON_RT_VISIBLE_CORES='',PATH='/usr/bin:/bin')


def phase_argv(phase, admission_ref):
    require(phase in ('pair','replay'),'Only paired verification then complete replay');reference(admission_ref)
    return [PYTHON,'-B','-u',FILES[2],'--mode',phase,'--admission',admission_ref['path'],
        '--admission-sha256',admission_ref['sha256']]


def render_binding(context, source_bytes, original_plan_bytes):
    """Return canonical future spec bytes; never read or publish any supplied path.

    Context consists of externally authenticated complete History bindings and
    source/input maps. No actual catalog, owner, exit or tool session is invented.
    The future runtime independently resolves all History and freshness again.
    """
    require(type(context)is dict and set(context)=={'run_name','history_overlay','current_serial_catalog',
        'lfm_r2_scope','inherited_source_sha256','input_extension','exact_envelopes'},'Complete explicit future context')
    require(type(source_bytes)is dict and set(source_bytes)==set(FILES),'All four concrete caller sources')
    require(type(original_plan_bytes)is bytes and sha(original_plan_bytes)==ORIGINAL_PLAN['sha256'],'Exact unchanged old logical plan supplied')
    original=json.loads(original_plan_bytes);pins=copy.deepcopy(context['inherited_source_sha256'])
    require(all(pins.get(p)==h for p,h in original['source_sha256'].items()),'Full original1869 sources retained')
    require(all(pins.get(p)==h for p,h in FROZEN.items()),'All accepted v1 sources retained without replacement')
    require(pins.get(ACCEPTED['path'])==ACCEPTED['sha256'],'Root source acceptance retained')
    for path,data in source_bytes.items():
        require(type(data)is bytes and 0<len(data)<1<<20,'Finite concrete caller source bytes')
        require(path not in pins or pins[path]==sha(data),'No existing source replacement')
        pins[path]=sha(data)
    for key in ('history_overlay','current_serial_catalog','lfm_r2_scope'):reference(context[key])
    run=context['run_name'];units=names(run)
    value=dict(schema='minicpm5_retained_replay_caller_spec_v1',run_name=run,units=units,
        output_root=str(ROOT/('minicpm5-retained-verification-'+run)),
        replay_root=str(ROOT/('minicpm5-history-retained-replay-'+run)),
        observer_root=str(ROOT/('minicpm5-retained-verification-'+run)/'independent-exit'),
        terminal_handoff_path=str(Q/('continuation-handoff-mini-retained-'+run+'-complete.json')),
        running_paths={p:str(Q/('continuation-handoff-mini-retained-'+run+'-'+p+'.json'))for p in ('pair','replay')},
        history_overlay=copy.deepcopy(context['history_overlay']),current_serial_catalog=copy.deepcopy(context['current_serial_catalog']),
        lfm_r2_scope=copy.deepcopy(context['lfm_r2_scope']),input_extension=copy.deepcopy(context['input_extension']),
        exact_envelopes=copy.deepcopy(context['exact_envelopes']),source_sha256=pins,
        storage_plan=copy.deepcopy(PLAN),original_plan=copy.deepcopy(ORIGINAL_PLAN),source_acceptance=copy.deepcopy(ACCEPTED),
        limits=copy.deepcopy(LIMITS),lock_identity=copy.deepcopy(LOCK),automatic_successor=False,deletion_authorized=False)
    validate_spec(value);return dict(spec=value,spec_bytes=canonical(value),spec_sha256=sha(canonical(value)),
        mandatory_source_sha256={p:pins[p]for p in FILES},runtime_enrolled=False,current_references_selected_here=False)


def validate_spec(spec):
    require(type(spec)is dict and set(spec)=={'schema','run_name','units','output_root','replay_root','observer_root',
        'terminal_handoff_path','running_paths','history_overlay','current_serial_catalog','lfm_r2_scope',
        'input_extension','exact_envelopes','source_sha256','storage_plan','original_plan','source_acceptance',
        'limits','lock_identity','automatic_successor','deletion_authorized'},'Exact finite caller binding fields')
    require(spec['schema']=='minicpm5_retained_replay_caller_spec_v1'and spec['units']==names(spec['run_name']), 'Finite two-phase caller')
    run=spec['run_name'];require(spec['output_root']==str(ROOT/('minicpm5-retained-verification-'+run))
        and spec['replay_root']==str(ROOT/('minicpm5-history-retained-replay-'+run))
        and spec['observer_root']==str(Path(spec['output_root'])/'independent-exit')
        and spec['terminal_handoff_path']==str(Q/('continuation-handoff-mini-retained-'+run+'-complete.json')), 'Disjoint fresh namespaces')
    require(spec['limits']==LIMITS and spec['lock_identity']==LOCK and spec['storage_plan']==PLAN
        and spec['original_plan']==ORIGINAL_PLAN and spec['source_acceptance']==ACCEPTED
        and spec['automatic_successor']is spec['deletion_authorized']is False,'No deletion or automatic continuation admission')
    for key in ('history_overlay','current_serial_catalog','lfm_r2_scope'):reference(spec[key])
    for p,h in spec['source_sha256'].items():reference(dict(path=p,sha256=h))
    require(all(spec['source_sha256'].get(p)==h for p,h in FROZEN.items())and set(FILES)<=set(spec['source_sha256'])
        and spec['source_sha256'].get(ACCEPTED['path'])==ACCEPTED['sha256'], 'Complete fixed source closure')
    for path,row in spec['input_extension'].items():
        require(set(row)=={'sha256','identity'}and row['identity']['realpath']==path,'Complete exact metadata input identities')
        reference(dict(path=path,sha256=row['sha256']))
    for path,row in spec['exact_envelopes'].items():
        require(set(row)=={'sha256','bytes','role'} and row['role']=='complete_json_metadata_envelope'
            and Path(path).parent==Q and Path(path).suffix=='.json'
            and type(row['bytes'])is int and LIMITS['ordinary_metadata_bytes']<row['bytes']<=LIMITS['spec_bytes']
            and spec['source_sha256'].get(path)==row['sha256'],'Only exact root-reviewed complete metadata envelopes')
    require(spec['running_paths']=={p:str(Q/('continuation-handoff-mini-retained-'+run+'-'+p+'.json'))for p in ('pair','replay')}, 'Two explicit future running publications')
    return spec


def running_value(previous,phase,tool,admission_ref,admission,worker,worker_ref,updated_at):
    require(phase in ('pair','replay')and set(tool)=={'kind','session_id'}and tool['kind']=='tool_session'
        and type(tool['session_id'])is int and tool['session_id']>0,'Externally observed actual outer tool session')
    require(previous['serial_execution']is True,'Complete original serial handoff')
    reference(admission_ref);reference(worker_ref)
    require(admission['schema']=='minicpm5_retained_replay_outer_admission_v1'
        and admission['lock_identity']==LOCK and worker['unit']==names(admission['run_name'])[phase]
        and worker['nonce']==admission['attempts'][phase]
        and worker['process']['argv']==phase_argv(phase,admission_ref)
        and worker['owner']['pid']==worker['process']['pid']
        and str(worker['owner']['start_ticks'])==str(worker['process']['start_ticks']), 'Exact actual admitted worker')
    if phase=='pair':require(previous['live_tool_sessions']==[]and previous['active_heavy_units']==[],'Actual idle predecessor required')
    else:
        prior=previous['minicpm5_retained_replay_current_execution']
        require(prior['phase']=='pair'and prior['tool']==tool and prior['admission']==admission_ref,'Same actual pair-to-replay outer')
    result=copy.deepcopy(previous)
    result.update(schema='minicpm5_retained_replay_running_v1',updated_at=updated_at,
        live_tool_sessions=[tool['session_id']],active_heavy_units=[worker['unit']],
        minicpm5_retained_replay_current_execution=dict(phase=phase,tool=copy.deepcopy(tool),admission=admission_ref,
            outer_owner=admission['outer_owner'],worker=worker,worker_ref=worker_ref,lock_identity=LOCK,
            current_serial_catalog=admission['current_serial_catalog'],automatic_successor=False,deletion_authorized=False))
    return result
