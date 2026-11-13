"""Explicit future phase contract; no selected run, resource budget or admission."""
import copy
import hashlib
import json
from pathlib import Path
import re

ROOT=Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
Q=ROOT/'research'
PYTHON='/home/ubuntu/venvs/vllm_neuron/bin/python'
NATIVE='/home/ubuntu/venvs/model_reference_2026/bin/python'
LOCK_IDENTITY={'device':66306,'inode':8294421}
PHASES=(('off','native'),('off','device'),('off','reproduce'),('capture','build'),
        ('on','native'),('on','device'),('shared','neutrality'))
RAW_PAIR_BYTES=47150344192
LIMITS=dict(startup_ack_seconds=60,owned_cleanup_seconds=45,server_log_bytes=64<<20,
            leaf_log_bytes=8<<20,metadata_bytes=8<<20,spec_envelope_bytes=64<<20)
PATH_KEYS={'control','off_metadata','off_native_payload','off_device_payload','on_metadata',
           'on_native_payload','on_device_payload','capture_artifact','capture_cache','capture_workdir'}
INITIAL_TAILS=('result.json','cold-reference/cold-reference.pt','cold-reference/cold-reference.json',
              'cold-buckets.pt','cold-buckets.json','cache-captures/native/manifest.json',
              'cache-captures/device/manifest.json')
OFF_TAILS=('cold-reference/cold-reference.pt','cold-reference/cold-reference.json','cold-buckets.pt',
           'cold-buckets.json','state-copy.json','state-lifecycle.json','cache-captures/native/manifest.json',
           'cache-captures/device/manifest.json')
ON_TAILS=('cold-reference/cold-reference.pt','cold-reference/cold-reference.json','cold-buckets.pt',
          'cold-buckets.json','state-copy.json','state-lifecycle.json','boundary-captures/native/manifest.json',
          'boundary-captures/device/manifest.json')

def require(value,message):
    if not value:raise ValueError(message)

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def reference(value):
    require(type(value)is dict and set(value)=={'path','sha256'} and type(value['path'])is str
        and Path(value['path']).is_absolute() and str(Path(value['path']))==value['path']
        and '..'not in Path(value['path']).parts and type(value['sha256'])is str
        and re.fullmatch('[a-f0-9]{64}',value['sha256']),'Exact external hash-authenticated reference required')
    return value

def phase_python(arm,phase):
    require((arm,phase)in PHASES,'Exactly one declared full scientific phase')
    return NATIVE if phase=='native' else PYTHON

def units(run_name):
    require(type(run_name)is str and re.fullmatch('[a-z0-9][a-z0-9-]{0,39}',run_name),'Fresh bounded run name')
    return {'coordinator':'trn1-deepseek-cold-'+run_name+'-coordinator',
        **{arm+':'+phase:'trn1-deepseek-cold-'+run_name+'-'+arm+'-'+phase for arm,phase in PHASES}}

def phase_directory(spec,arm,phase):
    require((arm,phase)in PHASES,'Declared complete phase required')
    return Path(spec['paths']['control'])/(arm+'-'+phase)

def validate_spec(spec,review,spec_ref):
    reference(spec_ref)
    required={'schema','run_name','paths','original_root','original_case','original_cold_metadata','original_input_sha256',
        'original_artifact_sha256','source_sha256','input_extension','history_overlay','current_serial_catalog',
        'lfm_r2_scope','exact_envelopes','resource_review','phase_resources','phase_environments','lock_identity',
        'automatic_successor','full_model_qualified','new_speedup_claim','running_handoff_path','outer_seconds','original_checkpoint_manifest'}
    require(type(spec)is dict and set(spec)==required and spec['schema']=='deepseek_cold_serial_spec_v2','Exact future full cold spec')
    require(spec['lock_identity']==LOCK_IDENTITY and spec['automatic_successor']is False
        and spec['full_model_qualified']is False and spec['new_speedup_claim']is False,'Original lease and diagnostic scope')
    names=units(spec['run_name'])
    require(type(spec['paths'])is dict and set(spec['paths'])==PATH_KEYS,'All separate control/output/compiler roots required')
    paths=list(map(Path,spec['paths'].values()))
    require(all(p.is_absolute()and str(p)==str(p.absolute())and '..'not in p.parts
        and 'deepseek-cold-kv-diagnostic-v2'not in p.parts for p in paths),'Canonical independent fresh namespace')
    require(len(set(paths))==len(paths)and all(a not in b.parents and b not in a.parents
        for i,a in enumerate(paths)for b in paths[i+1:]),'All output roots must be disjoint')
    require(spec['original_root']==str(ROOT/'deepseek-cold-kv-diagnostic-v2'),'Exact original completed cold diagnostic')
    for key in ('original_checkpoint_manifest','original_cold_metadata','resource_review','history_overlay','current_serial_catalog','lfm_r2_scope'):reference(spec[key])
    require(type(spec['source_sha256'])is dict and spec['source_sha256'],'Full untrimmed root source map required')
    for path,sha in spec['source_sha256'].items():reference(dict(path=path,sha256=sha))
    expected={str(Path(spec['original_root'])/tail)for tail in INITIAL_TAILS}
    require(type(spec['original_input_sha256'])is dict and set(spec['original_input_sha256'])==expected,'All original raw logits and outer reports/manifests')
    require(spec['original_input_sha256'][spec['original_cold_metadata']['path']]==spec['original_cold_metadata']['sha256'],
        'Original complete84 metadata bound by the same external original input map')
    for path,sha in spec['original_input_sha256'].items():reference(dict(path=path,sha256=sha))
    require(type(spec['original_artifact_sha256'])is dict and set(spec['original_artifact_sha256'])=={
        str(Path(spec['original_case']['artifact'])/name)for name in ('model.pt','neuron_config.json')},'Complete original artifact map')
    for path,sha in spec['original_artifact_sha256'].items():reference(dict(path=path,sha256=sha))
    require(set(spec['phase_resources'])==set(spec['phase_environments'])==set(names),'All seven phase envelopes required')
    require(type(spec['outer_seconds'])is int and spec['outer_seconds']>60,'Explicit finite entire attempt budget')
    handoff=Path(spec['running_handoff_path'])
    require(handoff.parent==Q and re.fullmatch('continuation-handoff-[a-z0-9-]+\\.json',handoff.name),
        'Existing complete running-handoff namespace')
    for key in names:
        props=spec['phase_resources'][key];env=spec['phase_environments'][key]
        require(type(props)is dict and props.get('KillMode')=='control-group'and props.get('RemainAfterExit')=='yes'
            and props.get('SendSIGKILL')=='yes'and props.get('WorkingDirectory')==str(ROOT)
            and props.get('TimeoutStopSec')=='15'and props.get('TasksMax')=='256','Retained bounded owned service contract')
        for field in ('MemoryMax','MemoryHigh','MemorySwapMax','LimitFSIZE','RuntimeMaxSec'):
            require(type(props.get(field))is str and props[field].isdigit()and int(props[field])>=0,'Explicit reviewed finite phase resource')
        require(int(props['MemoryMax'])>=int(props['MemoryHigh'])>0 and int(props['LimitFSIZE'])>0
            and int(props['RuntimeMaxSec'])>60,'No implicit/unbounded resource defaults')
        require(type(env)is dict and env.get('PYTHONDONTWRITEBYTECODE')=='1'and env.get('PYTHONOPTIMIZE')=='0'
            and env.get('PYTHONNOUSERSITE')=='1'
            and all(type(k)is str and type(v)is str for k,v in env.items()),'Exact explicit phase environment')
        temporary=(Path(spec['paths']['control'])/'tmp'if key=='coordinator'
            else phase_directory(spec,*key.split(':'))/'tmp')
        require(env.get('TMPDIR')==str(temporary),'Native/compiler temporary files stay in the exact monitored phase root')
    require(type(review)is dict and review==dict(schema='deepseek_cold_serial_root_review_v2',reviewer='/root',
        passed=True,blockers=[],spec=spec_ref,source_sha256=spec['source_sha256'],resource_review=spec['resource_review'],
        original_checkpoint_manifest=spec['original_checkpoint_manifest'],launch_authorized=True),'Independent exact source/resource/checkpoint binding before invocation')
    return names

def require_phase(spec,arm,phase,directory):
    require(Path(directory)==phase_directory(spec,arm,phase),'Exact complete cold phase directory')

def expected_phase(spec,admission_ref,outer,run_id,arm,phase,directory,host=None):
    require_phase(spec,arm,phase,directory)
    reference(host)
    stage=phase_directory(spec,arm,phase)
    return dict(schema='deepseek_cold_phase_spec_v2',admission=copy.deepcopy(admission_ref),coordinator=copy.deepcopy(outer),
        run_id=run_id,arm=arm,phase=phase,directory=str(stage),control=str(stage),unit=units(spec['run_name'])[arm+':'+phase],
        host_identity=copy.deepcopy(host),properties=copy.deepcopy(spec['phase_resources'][arm+':'+phase]),log=str(stage/'phase.log'),
        self_exit_claimed=False)

def required_output_paths(spec,arm,phase):
    p={k:Path(v)for k,v in spec['paths'].items()}
    if phase=='native':return [p[arm+'_metadata']/tail for tail in ('cold-reference/cold-reference.pt','cold-reference/cold-reference.json',
        ('cache-captures'if arm=='off'else'boundary-captures')+'/native/manifest.json')]
    if phase=='device':return [p[arm+'_metadata']/tail for tail in ('cold-buckets.pt','cold-buckets.json','state-copy.json',
        'state-lifecycle.json',('cache-captures'if arm=='off'else'boundary-captures')+'/device/manifest.json')]
    if phase=='reproduce':return [p['off_metadata']/tail for tail in ('result.json','native42-reproduction.json','device84-reproduction.json')]
    if phase=='build':return [p['capture_artifact']/'capture-build.json']
    require((arm,phase)==('shared','neutrality'),'Exact final diagnostic phase')
    return [p['on_metadata']/'capture-neutrality.json']
