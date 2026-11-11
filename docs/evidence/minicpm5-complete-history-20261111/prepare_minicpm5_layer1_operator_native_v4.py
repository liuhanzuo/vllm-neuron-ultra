"""Successor preparation retaining exact old and catalog256 post-exit producer witnesses; source/mock review is not admission.

prepare: one original descriptor acquisition, full original metadata/source and
physical checks, immutable manifest/package/result. bind: root metadata-only
binding after recording the actual preparation exit in the latest serial catalog.
Original 44,503 source pins and 8,791 inputs are never rewritten or projected.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import signal
import sys
ROOT=Path(__file__).resolve().parents[1];Q=ROOT/'research';sys.path[:0]=[str(Q),str(ROOT)]
from research import minicpm5_layer1_operator_native_process_adapter_v7 as r
from research import minicpm5_layer1_operator_native_outer_support_v3 as support
from research import minicpm5_layer1_operator_native_manifest_v1 as manifest
from research.prepare_dory_residual_spec_v3 import local_imports
from minicpm5_layer0_runtime_auth_v1 import identity,digest
require=r.require;SELF=str(Path(__file__).resolve())
LIMITS=copy.deepcopy(r.completion.audit.LIMITS)
BASE_SOURCE_COUNT=44503;BASE_INPUT_COUNT=8791
ACCEPTED_CLOSURE=dict(path=str(Q/'minicpm5-native-outer-static-source-closure-20261111-v3.json'),sha256='fea2673e9971ee66187a710a40d5871e855282ad847c019f0066ba013ce82701')
SCHEMA='minicpm5_native_preparation_package_v1'
RESULT_SCHEMA='minicpm5_native_preparation_result_v1'
BIND_SCHEMA='minicpm5_native_preparation_binding_result_v1'
OUTER=ROOT/'minicpm5_layer1_operator_native_outer_runner_v4.py'
RUNNER=ROOT/'minicpm5_layer1_operator_native_runner_v10.py'
AUDITOR_NAMES=re.compile(r'(?:audit_minicpm5_layer1_operator_native_result_v[0-9]+|minicpm5_layer1_operator_native_post_exit_core_v[0-9]+)\.py')


def audit_policy(pins,history):
    """An auditor name is allowed only at its authenticated completed hash."""
    for path,sha in pins.items():
        require(not AUDITOR_NAMES.fullmatch(Path(path).name)or history.get(path)==sha,
            'No future or changed auditor source without an actual completed catalog witness: '+path)


def executed_extension(terminal,row,read):
    """Only the two concrete root exit contracts; no invented failure schema."""
    tool=row['tool']
    require(type(tool)is dict and (
        tool.get('kind')=='tool_session'and set(tool)=={'kind','session_id','terminal_chunk','exit_code'}
        and type(tool['session_id'])is int and tool['session_id']>0
        or tool.get('kind')=='tool_call'and set(tool)=={'kind','terminal_chunk','exit_code'})
        and type(tool['exit_code'])is int and 0<=tool['exit_code']<=255
        and type(tool['terminal_chunk'])is str and re.fullmatch('[a-f0-9]{6,64}',tool['terminal_chunk']),
        'Actual completed recorder/auditor tool, preserving original nonzero code')
    require(terminal['reviewer']=='/root'and terminal['actual_tool_exit_observed']is True
        and terminal['actual_tool']==row['tool']and terminal['owner']in row['owners']
        and all(terminal[key]is False for key in ('self_exit_claimed','full_model_qualified','new_speedup_claim')),
        'Root external actual tool/owner/outcome, never producer self exit')
    recorder=terminal['schema']=='minicpm5_native_model_exit_recorder_actual_exit_v1'
    if recorder:
        require(terminal['owner_gone']is True,'Original recorder owner externally gone')
        r.life.ref(terminal['model_exit']);result=read(terminal['model_exit'])
        require(result['schema']in ('minicpm5_native_model_actual_exit_v1','minicpm5_native_model_failed_actual_exit_v1')
            and result['reviewer']=='/root'and result['recorder']==terminal['owner']
            and result['recorder_spec']==terminal['recorder_spec']and result['recorder_review']==terminal['recorder_review']
            and result['self_exit_claimed']is False and result['actual_recorder_tool_exit_verified']is False,
            'Existing model terminal binds the actual recorder identity and exact executed references')
        spec_ref=terminal['recorder_spec'];review_ref=terminal['recorder_review']
        spec_schema='minicpm5_native_model_exit_recorder_spec_v1'
        review_schema='minicpm5_native_model_exit_recorder_source_review_v1'
        scripts={str(Q/'record_minicpm5_layer1_operator_native_model_exit_v1.py'),
            str(Q/'record_minicpm5_layer1_operator_native_model_exit_v2.py')}
        output=terminal['model_exit'];source=result['recorder_source']
    else:
        require(terminal['schema']=='minicpm5_native_independent_audit_actual_exit_v1'
            and set(terminal)=={'schema','reviewer','audit_spec','audit_review','result','owner','actual_tool',
                'actual_tool_exit_observed','exact_owner_gone','lock_identity','self_exit_claimed','full_model_qualified','new_speedup_claim'}
            and terminal['exact_owner_gone']is True and terminal['lock_identity']==r.ORIGINAL_LOCK
            and row['tool']['kind']=='tool_session'and row['tool']['exit_code']==0,
            'Exact completed successful native auditor external exit contract')
        r.life.ref(terminal['result']);result=read(terminal['result'])
        require(result['schema']=='minicpm5_native_post_exit_independent_audit_v1'and result['complete']is True
            and result['reviewer']=='/root'and result['auditor_owner']==terminal['owner']
            and result['audit_spec']==terminal['audit_spec']and result['audit_review']==terminal['audit_review']
            and result['lock_identity']==r.ORIGINAL_LOCK and result['actual_auditor_tool_exit_verified']is False
            and all(result[key]is False for key in ('self_exit_claimed','full_model_qualified','new_speedup_claim')),
            'Actual audit output binds owner/spec/review while preserving diagnostic status')
        spec_ref=terminal['audit_spec'];review_ref=terminal['audit_review']
        spec_schema='minicpm5_native_post_exit_audit_spec_v1'
        review_schema='minicpm5_native_post_exit_audit_source_review_v1'
        scripts={str(Q/'audit_minicpm5_layer1_operator_native_result_v2.py'),
            str(Q/'audit_minicpm5_layer1_operator_native_result_v3.py')}
        output=terminal['result'];source=result['source']
    r.life.ref(source);script=source['path']
    require(script in scripts,'Only exact original or catalog256 producer, never an arbitrary future version')
    r.life.ref(spec_ref);r.life.ref(review_ref);spec=read(spec_ref);review=read(review_ref)
    expected={'schema','live','model_spec','model_review','actual_tool','source_extension','output','limits'}if recorder else {
        'schema','model_exit','recorder_exit','current_serial_catalog','source_extension','output','limits'}
    require(set(spec)==expected and spec['schema']==spec_schema and spec['limits']==LIMITS
        and spec['output']==output['path'],'Exact executed producer spec and original resource limits')
    require(set(review)=={'schema','reviewer','passed','blockers','spec','source_extension'}
        and review['schema']==review_schema and review['reviewer']=='/root'and review['passed']is True
        and review['blockers']==[]and review['spec']==spec_ref and review['source_extension']==spec['source_extension'],
        'Hash-authenticated exact executed source review, not a standalone source assertion')
    pins=spec['source_extension'];r.sources.metadata.pin_shape(pins)
    require(0<len(pins)<=2048 and source==dict(path=script,sha256=pins[script])
        and all(row['source_sha256'].get(path)==sha for path,sha in pins.items()),
        'Every executed extension source retained exactly in its catalog row')
    current=script in {str(Q/'record_minicpm5_layer1_operator_native_model_exit_v2.py'),
        str(Q/'audit_minicpm5_layer1_operator_native_result_v3.py')}
    if current:
        # Auditor v3 imports core v5, which exports the unchanged full validator
        # from v4; recorder v2 imports that same exact audit/core chain. These
        # are new paths, so the original disjoint-extension rule stays intact.
        executed={str(Q/'audit_minicpm5_layer1_operator_native_result_v3.py'),
            str(Q/'minicpm5_layer1_operator_native_post_exit_core_v4.py'),
            str(Q/'minicpm5_layer1_operator_native_post_exit_core_v5.py')}
        require(executed<=set(pins),'Complete actually executed catalog256 auditor/core source chain')
    else:
        executed={str(Q/'audit_minicpm5_layer1_operator_native_result_v2.py'),
            str(Q/'minicpm5_layer1_operator_native_post_exit_core_v2.py')}
        if recorder:executed.add(str(Q/'minicpm5_layer1_operator_native_post_exit_core_v3.py'))
    require(all(path in executed for path in pins if AUDITOR_NAMES.fullmatch(Path(path).name)),
        'Only these frozen producers actual auditor/core imports; no future version enrolled by a historical spec')
    if recorder:
        require(terminal['source_extension']==pins and spec['live']==result['live']
            and spec['model_spec']==result['spec']and spec['model_review']==result['review']
            and spec['actual_tool']==result['actual_tool'],'Original model result/spec and root recorder extension agree')
    else:
        require(spec['model_exit']==result['model_exit']and spec['recorder_exit']==result['recorder_exit'],
            'Executed audit retains its actual model and recorder terminals')
    argv=terminal['owner']['argv']
    require(type(argv)is list and len(argv)==11 and type(argv[0])is str and bool(argv[0])
        and argv[1:]==['-B',script,'--spec',spec_ref['path'],'--spec-sha256',spec_ref['sha256'],
            '--review',review_ref['path'],'--review-sha256',review_ref['sha256']],
        'Actual owner command binds the executed script and hash-authenticated spec/review')
    if not recorder:require(argv[0]=='python3','Exact original external auditor interpreter command')
    r.life.process(terminal['owner'],argv)
    return pins


def historical_auditors(serial_ref,serial,read,guard):
    """Complete original catalog validation, then exact executed-source witnesses."""
    pins={}
    for row in serial['runs']:merge(pins,row['source_sha256'],'complete serial source')
    view=dict(serial_predecessors=serial_ref,source_sha256=pins)
    validated=r.completion.latest.serial_catalog(view,read,guard)
    require(read(serial_ref)==serial and validated['runs']==serial['runs'],'Unchanged complete authenticated current catalog')
    history={}
    schemas={'minicpm5_native_model_exit_recorder_actual_exit_v1','minicpm5_native_independent_audit_actual_exit_v1'}
    for row in validated['runs']:
        if not any(AUDITOR_NAMES.fullmatch(Path(path).name)for path in row['source_sha256']):continue
        guard();terminal=read(row['terminal'])
        if terminal.get('schema')not in schemas:continue
        extension=executed_extension(terminal,row,read)
        merge(history,{path:sha for path,sha in extension.items()if AUDITOR_NAMES.fullmatch(Path(path).name)},
            'actually completed historical auditor source')
    audit_policy(pins,history)
    return history


def merge(target,addition,label):
    for path,value in addition.items():
        require(path not in target or target[path]==value,'Never replace original '+label+': '+path)
        if path not in target:target[path]=copy.deepcopy(value)


def extension(base,addition,label):
    joined=copy.deepcopy(base);merge(joined,addition,label)
    return {path:value for path,value in joined.items()if path not in base}


def encoded(value):return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
def object_ref(path,value):return dict(path=str(path),sha256=hashlib.sha256(encoded(value)).hexdigest())


def paths(run_id):
    require(type(run_id)is str and re.fullmatch('[a-z0-9][a-z0-9-]{0,39}',run_id),'Bounded root-selected fresh run id')
    base='minicpm5-layer1-operator-native-'+run_id
    return {key:Q/(base+'-'+suffix+'.json')for key,suffix in
        [('manifest','manifest'),('package','preparation-package'),('result','preparation-result'),
         ('overlay','overlay'),('spec','spec'),('binding','binding-result'),('review','source-review')]}|dict(
        output=Path('/mnt/nvme')/base,unit='trn1-mini-layer1-operator-native-'+run_id)


class Reader:
    """Stable original metadata transport, with only newly observed input rows."""
    def __init__(self,guard):self.guard=guard;self.inputs={};self.metadata=r.Metadata(guard)
    def __call__(self,ref):
        value=self.metadata.read(ref);row=dict(sha256=ref['sha256'],identity=copy.deepcopy(self.metadata.identities[ref['path']]))
        merge(self.inputs,{ref['path']:row},'authenticated metadata input');return value
    def unchanged(self):self.metadata.unchanged()


def floors():
    for path,floor in [(ROOT,28*2**30),(Path('/mnt/nvme'),30*2**30)]:
        fs=os.statvfs(path);require(fs.f_bavail*fs.f_frsize>=floor+8*2**20,'Original storage floor plus small preparation reserve')


def loaded_sources(guard,history):
    result={}
    for module in tuple(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if path and str(path).startswith(str(ROOT)+'/'):
            guard();path=str(Path(path).resolve());raw=r.sources.text(path)
            result[path]=hashlib.sha256(raw).hexdigest()
            audit_policy({path:result[path]},history)
    return result


def source_additions(closure,serial,loaded,closure_ref,serial_ref,history):
    require(closure_ref==ACCEPTED_CLOSURE,'Exact accepted outer578-file source closure')
    require(closure['source_only']is True and closure['original_complete_source_map_must_remain']is True
        and closure['dynamic_import_resolution_verified']is False and closure['runtime_import_identity_verified']is False,
        'Reviewed static closure supplements rather than replaces original scientific sources')
    graph=closure['runtime_graph'];pins=dict(graph['source_sha256'])
    require(graph['files']==len(pins)and 578<=len(pins)<=2048 and graph['native_executed']is False,
        'Complete accepted578-file outer closure retained')
    require(serial['schema']=='minicpm5_operator_later_serial_metadata_catalog_v1'and serial['reviewer']=='/root'
        and serial['passed']is True and serial['blockers']==[]and serial['all_later_serial_runs_included']is True
        and type(serial['runs'])is list and len(serial['runs'])<=r.completion.latest.MAX_SERIAL_RUNS,'Root complete current serial catalog')
    for row in serial['runs']:merge(pins,row['source_sha256'],'executed later-serial source')
    merge(pins,loaded,'actually loaded preparation source')
    merge(pins,{closure_ref['path']:closure_ref['sha256'],serial_ref['path']:serial_ref['sha256']},'closure/catalog metadata source')
    audit_policy(pins,history)
    r.sources.metadata.pin_shape(pins);return pins


def native_donors(recovered,stat=identity):
    result={};records=recovered['native']['native']['records']
    require(len(records)==32,'All original native donor frames')
    for record in records:
        for name in ('key','value','taps'):
            desc=record['outputs'][name];path=desc['path'];wanted=desc['identity']
            require(path not in result and stat(path)==wanted and wanted['realpath']==path
                and wanted['size']==desc['raw_bytes'],'Exact complete native donor identity without byte reads')
            result[path]=dict(sha256=desc['sha256'],identity=copy.deepcopy(wanted))
    require(len(result)==96,'All96 native donors retained');return result


def preserved(recovered):
    base=recovered['base_plan']
    require(recovered['binding']==r.completion.BINDING and recovered['metadata_complete']is True
        and recovered['runtime_admitted']is False and recovered['original_model_exit_code']==1,
        'Original completed recovery, never promote failed SDK model')
    require(len(base['source_sha256'])==BASE_SOURCE_COUNT and len(base['inputs'])==BASE_INPUT_COUNT,
        'Complete original44503 sources and8791 inputs, never a reduced mock plan')
    require(all(recovered['source_sha256'].get(path)==sha for path,sha in base['source_sha256'].items()),'Every original source retained')
    return base


def overlay_value(recovered,run_id,serial_ref,manifest_ref,new_sources,observed_inputs):
    base=preserved(recovered)
    return dict(schema=r.overlays.SCHEMA,base_plan=copy.deepcopy(recovered['base_plan_ref']),
        sdk_operator_recovery=copy.deepcopy(r.completion.BINDING),current_serial_catalog=copy.deepcopy(serial_ref),
        native_manifest=copy.deepcopy(manifest_ref),
        source_extension=extension(recovered['source_sha256'],new_sources,'source pin'),
        input_extension=extension(base['inputs'],observed_inputs,'input identity'),output_root=str(paths(run_id)['output']))


def resolve_binding(recovered,overlay,read,guard):
    resolved=r.overlays.resolve(overlay,recovered,read,guard)
    runner=dict(path=str(RUNNER),sha256=resolved['plan']['source_sha256'][str(RUNNER)])
    overlay_ref=object_ref(paths(Path(overlay['output_root']).name.removeprefix('minicpm5-layer1-operator-native-'))['overlay'],overlay)
    contract=r.life.build_contract(resolved,overlay_ref,runner,'trn1-mini-layer1-operator-native-'+Path(overlay['output_root']).name.removeprefix('minicpm5-layer1-operator-native-'))
    spec=dict(schema='minicpm5_native_process_spec_v1',overlay=overlay_ref,runner=runner,unit=contract['unit'],lock_policy=copy.deepcopy(r.LOCK_POLICY))
    for key in ('source_sha256','inputs'):
        require(all(resolved['plan'][key].get(path)==value for path,value in recovered['base_plan'][key].items()),'Unchanged complete base '+key)
    return resolved,contract,spec


def write_small(path,value,guard):
    guard();raw=encoded(value);require(len(raw)<LIMITS['output_file_bytes'],'Original2MiB individual preparation metadata bound')
    r.atomic_once(path,value);ref=object_ref(path,value)
    require(r.sources.load(ref)==value,'Complete authenticated durable small metadata readback');guard();return ref


def actual_preparation_tool(value):
    require(type(value)is dict and (
        value.get('kind')=='tool_call'and set(value)=={'kind','terminal_chunk','exit_code'}
        or value.get('kind')=='tool_session'and set(value)=={'kind','session_id','terminal_chunk','exit_code'}
        and type(value['session_id'])is int and value['session_id']>0)
        and type(value['exit_code'])is int and value['exit_code']==0
        and type(value['terminal_chunk'])is str and re.fullmatch('[a-f0-9]{3,64}',value['terminal_chunk']),
        'Real successful external preparation tool_call or tool_session, never invented handle')
    return value


def complete(reader,guard):
    return r.completion.completed(r.completion.BINDING,reader,guard,lambda path,sha:r.sources.text(path,sha))


def fresh(view,reader,guard,expected=None):
    value=r.completion.latest.require_latest_terminal(view,reader,guard)
    require(expected is None or value==expected,'Original complete latest terminal unchanged during preparation/binding')
    return value


def prepare(serial_ref,closure_ref,run_id,lease):
    loc=paths(run_id);require(all(not loc[k].exists()for k in ('manifest','package','result','overlay','spec','binding','review','output')),'All new preparation and model paths fresh')
    authority=r.current_authority.Authority(r,lease)
    def guard():authority.check();floors()
    reader=Reader(guard);lease.metadata=reader.metadata
    guard();authority.active_heavy();closure=reader(closure_ref);serial=reader(serial_ref)
    history=historical_auditors(serial_ref,serial,reader,guard)
    graph=local_imports(entries=[Path(SELF),OUTER,RUNNER]);pins=source_additions(closure,serial,graph['source_sha256'],closure_ref,serial_ref,history)
    merge(pins,loaded_sources(guard,history),'loaded prep source')
    # The latest-idle proof precedes full historical recovery traversal and source IO.
    ended=reader(r.completion.BINDING['actual_exit']);recovery_spec=reader(ended['spec']);base=reader(recovery_spec['original_plan'])
    seed=dict(base['source_sha256']);merge(seed,recovery_spec['source_extension'],'recovery source');merge(seed,pins,'preparation source')
    view=dict(latest_completion=base['latest_completion'],serial_predecessors=serial_ref,source_sha256=seed)
    initial=fresh(view,reader,guard)
    recovered=complete(reader,guard);preserved(recovered)
    require(recovered['lock_identity']==r.ORIGINAL_LOCK,'Original recovery lock')
    donors=native_donors(recovered);merge(reader.inputs,donors,'all native donor identities')
    native_manifest=manifest.build(recovered['base_plan'],recovered['native']['native'])
    manifest_ref=write_small(loc['manifest'],native_manifest,guard);reader(manifest_ref)
    merge(pins,{manifest_ref['path']:manifest_ref['sha256']},'manifest source');merge(pins,loaded_sources(guard,history),'loaded recovery source')
    overlay=overlay_value(recovered,run_id,serial_ref,manifest_ref,pins,reader.inputs)
    resolved,contract,_=resolve_binding(recovered,overlay,reader,guard)
    partition=r.sources.classify(resolved['plan']['source_sha256'],resolved['plan']['source_contract'],reader)
    r.sources.verify_text_sources(partition,guard);r.verify_loaded(resolved['plan'],guard)
    lease.scope=resolved['current_scope']
    pair=r.current_physical.collect_pair(lease.scope,guard,authority)
    required={row['unit']for row in lease.scope['nemotron']['history']['rows']};known=set(lease.scope['complete_known_units'])
    require(r.completion.physical.validate_pair(pair,required,lease.scope['owners'],r.ORIGINAL_LOCK,known_units=known)==known,'Complete unchanged historical two-pass physical scope')
    # The entire44503+ partition was fully text-verified above. Each late local
    # import is itself completely read/hash-authenticated by loaded_sources; all
    # metadata identities remain latched. Do not repeat the entire44k IO loop.
    # Any local helper imported by physical traversal is authenticated and added.
    merge(pins,loaded_sources(guard,history),'loaded physical source')
    audit_policy(pins,history)
    overlay=overlay_value(recovered,run_id,serial_ref,manifest_ref,pins,reader.inputs)
    resolved,contract,_=resolve_binding(recovered,overlay,reader,guard)
    r.sources.check_partition(r.sources.classify(resolved['plan']['source_sha256'],resolved['plan']['source_contract'],reader))
    r.verify_loaded(resolved['plan'],guard);reader.unchanged();authority.active_heavy()
    fresh(resolved['current_serial_view'],reader,guard,initial)
    package=dict(schema=SCHEMA,run_id=run_id,base_plan=recovered['base_plan_ref'],sdk_operator_recovery=r.completion.BINDING,
        preparation_serial_catalog=serial_ref,closure=closure_ref,native_manifest=manifest_ref,
        source_extension=overlay['source_extension'],input_extension=overlay['input_extension'],
        base_source_files=BASE_SOURCE_COUNT,base_inputs=BASE_INPUT_COUNT,original_model_exit_code=1,
        final_serial_rebind_required=True,runtime_admitted=False,launch_ready=False)
    package_ref=write_small(loc['package'],package,guard)
    result=dict(schema=RESULT_SCHEMA,complete=True,package=package_ref,manifest=manifest_ref,owner=lease.admission['outer_owner'],
        lock_identity=r.ORIGINAL_LOCK,source=dict(path=SELF,sha256=resolved['plan']['source_sha256'][SELF]),
        preparation_serial_catalog=serial_ref,serial_freshness=initial,base_source_files=BASE_SOURCE_COUNT,base_inputs=BASE_INPUT_COUNT,
        source_files=len(resolved['plan']['source_sha256']),inputs=len(resolved['plan']['inputs']),
        full_physical_pair_sha256=digest(pair),historical_units=len(known),historical_owners=len(lease.scope['owners']),
        descriptor_owner_no_exclusions=True,full_physical_passes=2,original_sdk_model_exit_code=1,
        source_extension=overlay['source_extension'],native_bytes_hashed=0,raw_payload_bytes_read=0,
        self_exit_claimed=False,actual_external_tool_exit_verified=False,final_serial_rebind_required=True,
        runtime_admitted=False,launch_ready=False,full_model_qualified=False,new_speedup_claim=False)
    result_ref=write_small(loc['result'],result,guard)
    fresh(resolved['current_serial_view'],reader,guard,initial);reader.unchanged();authority.active_heavy();return result_ref


def bind(package_ref,preparation_exit_ref,serial_ref,*,reader=None,guard=None):
    """Root metadata-only binding, after separately observed prep exit; no lock/physical/model IO.

    This publication does not introduce a new serial worker into its own catalog.
    Original full base metadata is reauthenticated in memory; no raw/native bytes.
    """
    require(sys.dont_write_bytecode and __debug__,'Original root binding -B/assertions')
    guard=guard or r.p.clear_window;reader=reader or Reader(guard)
    package=reader(package_ref);require(package['schema']==SCHEMA and package['final_serial_rebind_required']is True
        and package['runtime_admitted']is False and package['original_model_exit_code']==1,'Exact immutable completed preparation package')
    run_id=package['run_id'];loc=paths(run_id)
    require(package_ref['path']==str(loc['package'])and all(not loc[k].exists()for k in ('overlay','spec','binding','review','output')),'Fresh final native binding paths')
    ended=reader(preparation_exit_ref);result_ref=ended['result'];r.life.ref(result_ref,str(loc['result']));result=reader(result_ref)
    require(ended['schema']=='minicpm5_native_preparation_actual_exit_v1'and ended['reviewer']=='/root'
        and ended['actual_tool_exit_observed']is True and ended['exact_owner_gone']is True
        and ended['result']==result_ref and ended['owner']==result['owner']and ended['lock_identity']==r.ORIGINAL_LOCK,
        'Root independently observed actual preparation exit and original owner/lock/result')
    actual_preparation_tool(ended['actual_tool'])
    require(result['schema']==RESULT_SCHEMA and result['complete']is True and result['package']==package_ref
        and result['source_extension']==package['source_extension']and result['self_exit_claimed']is False,
        'Full exact successful preparation result, not artifact presence')
    serial=reader(serial_ref);rows=[row for row in serial['runs']if row['terminal']==preparation_exit_ref]
    require(len(rows)==1 and rows[0]['tool']==ended['actual_tool']and ended['owner']in rows[0]['owners'],
        'Final latest catalog includes actual preparation exit, tool and owner')
    history=historical_auditors(serial_ref,serial,reader,guard)
    root_pins=loaded_sources(guard,history)
    initial_seed=reader(package['base_plan'])
    initial_pins=dict(initial_seed['source_sha256']);merge(initial_pins,package['source_extension'],'prepared source')
    for row in serial['runs']:merge(initial_pins,row['source_sha256'],'latest catalog source')
    merge(initial_pins,root_pins,'root binding source')
    initial_view=dict(latest_completion=initial_seed['latest_completion'],serial_predecessors=serial_ref,source_sha256=initial_pins)
    initial=fresh(initial_view,reader,guard)
    recovered=complete(reader,guard);base=preserved(recovered)
    require(package['base_plan']==recovered['base_plan_ref']and package['sdk_operator_recovery']==r.completion.BINDING,'Original package recovery binding')
    additions=copy.deepcopy(package['source_extension'])
    for row in serial['runs']:merge(additions,row['source_sha256'],'final root serial sources')
    merge(additions,loaded_sources(guard,history),'actual root binding source')
    audit_policy(additions,history)
    for ref in (package_ref,result_ref,preparation_exit_ref,serial_ref):merge(additions,{ref['path']:ref['sha256']},'acyclic final metadata source')
    observed=dict(package['input_extension']);merge(observed,reader.inputs,'final authenticated metadata')
    overlay=overlay_value(recovered,run_id,serial_ref,package['native_manifest'],additions,observed)
    resolved,contract,spec=resolve_binding(recovered,overlay,reader,guard)
    fresh(resolved['current_serial_view'],reader,guard,initial)
    merge(observed,reader.inputs,'final binding observed metadata')
    overlay=overlay_value(recovered,run_id,serial_ref,package['native_manifest'],additions,observed)
    resolved,contract,spec=resolve_binding(recovered,overlay,reader,guard)
    # Full source hashing was performed by serial prepare. New root binding only
    # authenticates exact added text and metadata, with the base map immutable.
    prepared=set(recovered['source_sha256'])|set(package['source_extension'])
    for path,sha in resolved['plan']['source_sha256'].items():
        if path not in prepared:guard();require(hashlib.sha256(r.sources.text(path,sha)).hexdigest()==sha,'Exact newly bound source')
    partition=r.sources.classify(resolved['plan']['source_sha256'],resolved['plan']['source_contract'],reader);r.sources.check_partition(partition)
    r.verify_loaded(resolved['plan'],guard)
    for path,row in overlay['input_extension'].items():require(identity(path)==row['identity'],'Final metadata/donor input identity unchanged')
    fresh(resolved['current_serial_view'],reader,guard,initial);reader.unchanged()
    overlay_ref=write_small(loc['overlay'],overlay,guard);require(overlay_ref==spec['overlay'],'No circular overlay digest')
    spec_ref=write_small(loc['spec'],spec,guard)
    value=dict(schema=BIND_SCHEMA,complete=True,preparation=package_ref,preparation_exit=preparation_exit_ref,
        serial_predecessors=serial_ref,serial_freshness=initial,manifest=package['native_manifest'],overlay=overlay_ref,spec=spec_ref,
        resolved_plan_sha256=resolved['plan_sha256'],contract=contract,source_files=len(resolved['plan']['source_sha256']),inputs=len(resolved['plan']['inputs']),
        base_source_files=BASE_SOURCE_COUNT,base_inputs=BASE_INPUT_COUNT,review_target=str(loc['review']),
        required_review_schema='minicpm5_native_process_source_review_v1',source_review_accepted=False,
        root_metadata_binding_only=True,independent_concrete_review_required=True,no_new_serial_worker=True,
        runtime_admitted=False,launch_ready=False,native_bytes_hashed=0,raw_payload_bytes_read=0,
        full_model_qualified=False,new_speedup_claim=False)
    ref=write_small(loc['binding'],value,guard);fresh(resolved['current_serial_view'],reader,guard,initial);reader.unchanged();return ref


def run(serial_ref,closure_ref,run_id):
    require(sys.dont_write_bytecode and __debug__,'Original preparation -B/assertions')
    os.sched_setaffinity(0,{LIMITS['cpu']});os.nice(LIMITS['nice'])
    resource.setrlimit(resource.RLIMIT_AS,(LIMITS['address_space_bytes'],)*2)
    resource.setrlimit(resource.RLIMIT_CPU,(LIMITS['cpu_seconds'],LIMITS['hard_cpu_seconds']))
    resource.setrlimit(resource.RLIMIT_FSIZE,(LIMITS['output_file_bytes'],)*2)
    previous=signal.signal(signal.SIGALRM,lambda *args:(_ for _ in ()).throw(TimeoutError('Original preparation wall bound')))
    signal.alarm(LIMITS['wall_seconds']);fd=None
    try:
        r.p.user_bus();fd,owner,first=support.acquire()
        lease=r.Lease(dict(outer_owner=owner,lock_identity=r.ORIGINAL_LOCK,retained_fd=fd,lock_policy=r.LOCK_POLICY,
            initial_descriptor_proof=first,unit='trn1-mini-layer1-operator-native-preparation'))
        return prepare(serial_ref,closure_ref,run_id,lease)
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,previous)
        if fd is not None:support.release_after_physical_drain(fd,dict(physically_drained=True,no_preparation_children_or_services_started=True))


def main():
    cli=argparse.ArgumentParser();sub=cli.add_subparsers(dest='mode',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--run-id',required=True)
    final=sub.add_parser('bind')
    for parser,keys in [(prep,('serial-catalog','closure')),(final,('package','preparation-exit','serial-catalog'))]:
        for key in keys:parser.add_argument('--'+key,required=True);parser.add_argument('--'+key+'-sha256',required=True)
    a=cli.parse_args();ref=lambda key:dict(path=getattr(a,key.replace('-','_')),sha256=getattr(a,key.replace('-','_')+'_sha256'))
    result=run(ref('serial-catalog'),ref('closure'),a.run_id)if a.mode=='prepare'else bind(ref('package'),ref('preparation-exit'),ref('serial-catalog'))
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
