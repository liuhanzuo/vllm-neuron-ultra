"""Metadata-only Mini pair/full-replay candidate preparation; inert on import.

All current refs are explicit. No History, lock, process, payload/native read or
runtime operation occurs here. Stable readers are injected into prepare(); the
future CLI only authenticates JSON/source inputs and publishes fresh candidates.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys

Q=Path(__file__).resolve().parent;ROOT=Q.parent
for directory in (Q,ROOT):
    if str(directory)not in sys.path:sys.path.insert(0,str(directory))
import minicpm5_retained_replay_contract_v1 as c
import minicpm5_retained_replay_finalizer_v2 as finalizer
import minicpm5_retained_replay_contract_v2 as successor
from prepare_qwen35_package_serial_candidate_v8 import (
    StableReaders,file_identity,identity_shape,merge_sources,merge_inputs,catalog_shape,
    short_ref,finite_json_numbers,retained_reference_paths,_unique)

SELF=str(Q/'prepare_minicpm5_retained_replay_binding_v2.py')
MINI_REVIEW=dict(path=str(Q/'minicpm5-retained-caller-finalizer-v2-root-source-review-20261113-v1.json'),
    sha256='7e4133b08095c667b749e1240096379c6dae69ffde2406c0b6edc8a906890509')
ORDINARY=8<<20;ENVELOPE=64<<20
REQUEST_KEYS={'schema','seed_spec','seed_review','seed_binding','catalog','idle_handoff','prerequisite_exit',
    'prerequisite_tool','mini_source_review','preparer_source_review','preparer_delivery','future',
    'successor_source_review','successor_delivery'}
FUTURE_KEYS={'run_name','overlay_path','overlay_output_root','spec_path','review_draft_path','review_path','candidate_path'}
REQUIRED_SOURCES=tuple(str(Q/name)for name in (
    'prepare_minicpm5_retained_replay_binding_v2.py','prepare_qwen35_package_serial_candidate_v8.py',
    'qwen35_package_serial_runtime_v2/__init__.py','qwen35_package_serial_runtime_v2/contract.py',
    'qwen35_sampler_pair_protocol_v1.py','qwen35_sampler_runtime_io_v12.py','qwen35_sampler_runtime_contract_v12.py',
    'qwen35_sampler_runtime_service_v12.py','qwen35_sampler_runtime_history_v12.py',
    'qwen35_package_preflight_history_v3.py','qwen35_package_preflight_serial_v2.py',
    'qwen35_package_serial_runtime_v7/__init__.py','qwen35_package_serial_runtime_v7/source_text.py',
    'qwen35_package_serial_runtime_v7/io.py','qwen35_sampler_history_supplement_v1.py',
    'minicpm5_layer1_drain_hold_v1.py'))+successor.SUCCESSOR_FILES


def encode(value,cap=ENVELOPE):
    value=c.canonical(value)+b'\n';c.require(len(value)<=cap,'Original bounded metadata publication');return value


def object_ref(path,payload):return dict(path=str(path),sha256=hashlib.sha256(payload).hexdigest())


def require_tool(tool):
    c.require(type(tool)is dict and set(tool)=={'kind','session_id','terminal_chunk','exit_code'}
        and tool['kind']=='tool_session'and type(tool['session_id'])is int and tool['session_id']>0
        and type(tool['exit_code'])is int and -255<=tool['exit_code']<=255
        and type(tool['terminal_chunk'])is str and re.fullmatch('[A-Za-z0-9_-]{1,128}',tool['terminal_chunk']),
        'Actual completed prerequisite tool including its original failure')


def prepare(request,read_source,identity_only,fresh_path,guard):
    """Return complete candidate objects/bytes; never publish or admit runtime.

    read_source(ref,cap) returns {payload: bytes,identity: exact immutable stat}.
    Only JSON and explicit Python/source files are requested; metadata payload
    declarations are preserved verbatim and never statted or opened. All refs
    and freshness authority come from the future external root.
    """
    c.require(type(request)is dict and set(request)==REQUEST_KEYS
        and request['schema']=='minicpm5_retained_replay_future_binding_v2','Exact explicit future request')
    c.require(all(callable(x)for x in (read_source,identity_only,fresh_path,guard)),'Explicit bounded source preparation callbacks')
    request=copy.deepcopy(request);observed={};payloads={};retained=set();large={};new_sources={}
    for key in REQUEST_KEYS-{'schema','future','prerequisite_tool'}:c.reference(request[key])
    c.require(request['mini_source_review']==MINI_REVIEW,'Accepted caller plus corrected finalizer v2 review required')
    future=request['future'];c.require(type(future)is dict and set(future)==FUTURE_KEYS,'Complete future candidate outputs')
    c.names(future['run_name'])
    for key in ('overlay_path','spec_path','review_draft_path','review_path','candidate_path'):
        p=Path(future[key]);c.require(p.parent==Q and str(p)==future[key]
            and re.fullmatch('[a-z0-9][a-z0-9-]*\.json',p.name),'Exact future research JSON path')
    virtual=Path(future['overlay_output_root'])
    c.require(virtual.parent==Path('/mnt/nvme')and str(virtual)==future['overlay_output_root']
        and re.fullmatch('minicpm5-layer1-operator-native-[a-z0-9-]{1,64}',virtual.name),'Exact original virtual overlay namespace')

    def source(reference,cap=ORDINARY):
        c.reference(reference);path=reference['path']
        c.require(Path(path).suffix in ('.json','.py','.md','.txt','.patch'),'Explicit metadata/source only; no binary payload input')
        c.require(type(cap)is int and 0<cap<=ENVELOPE,'Original finite source envelope')
        guard();saved=read_source(copy.deepcopy(reference),cap)
        c.require(type(saved)is dict and set(saved)=={'payload','identity'}and type(saved['payload'])is bytes,'Complete stable source receipt')
        identity_shape(path,saved['identity']);raw=saved['payload']
        c.require(len(raw)==saved['identity']['size']<=cap and hashlib.sha256(raw).hexdigest()==reference['sha256'],'Complete authenticated bytes')
        c.require(b'\x00'not in raw,'Text source only');raw.decode('utf-8')
        row=dict(sha256=reference['sha256'],identity=copy.deepcopy(saved['identity']))
        c.require(path not in observed or observed[path]==row,'No source/input identity rebound')
        observed[path]=row;payloads[path]=raw;guard();return raw

    def read(reference,cap=ORDINARY):
        c.require(Path(reference['path']).suffix=='.json','Only JSON metadata')
        raw=source(reference,cap);value=json.loads(raw,object_pairs_hook=_unique,
            parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Finite JSON only')))
        finite_json_numbers(value);retained.update(retained_reference_paths(value))
        if len(raw)>ORDINARY:
            c.require(Path(reference['path']).parent==Q,'Only exact research metadata envelope extension')
            large[reference['path']]=dict(sha256=reference['sha256'],bytes=len(raw),role='complete_json_metadata_envelope')
        return value

    seed=read(request['seed_spec'],ENVELOPE);review=read(request['seed_review'],ENVELOPE);binding=read(request['seed_binding'])
    c.require(binding['reviewer']=='/root'and binding['passed']is True and binding['full_source_history_resolution_passed']is True
        and binding['original_failed_statuses_preserved']is True and binding['lfm_children']==874
        and binding['spec']==request['seed_spec']and binding['review']==request['seed_review'],
        'Actual accepted complete seed History and all original874 children')
    c.require(review['reviewer']=='/root'and review['passed']is True and review['launch_authorized']is True
        and review['blockers']==[]and review['spec']==request['seed_spec']and review['source_sha256']==seed['source_sha256'],
        'Exact independently accepted seed source map')
    c.require(seed['lock_identity']==c.LOCK and seed['automatic_successor']is False,'Original serial lock and no automatic successor')
    overlay=read(seed['history_overlay']);base=read(short_ref(overlay['base_plan']),ENVELOPE)
    c.require(set(overlay)=={'schema','base_plan','sdk_operator_recovery','current_serial_catalog','native_manifest',
        'source_extension','input_extension','output_root'}and overlay['schema']=='minicpm5_layer1_operator_native_small_overlay_v2'
        and overlay['current_serial_catalog']==seed['current_serial_catalog']==binding['catalog']
        and binding['history_overlay']==seed['history_overlay'],'Exact accepted full seed overlay lineage')
    old_catalog=read(seed['current_serial_catalog']);catalog=read(request['catalog']);catalog_shape(old_catalog);rows=catalog_shape(catalog)
    c.require(catalog['runs'][:len(old_catalog['runs'])]==old_catalog['runs']
        and len(catalog['runs'])>=len(old_catalog['runs'])and catalog['completed_sdk']==old_catalog['completed_sdk']
        and catalog['current_handoff']==request['idle_handoff'],'Complete ordered catalog and exact root-supplied idle')
    idle=read(request['idle_handoff'])
    c.require(idle['serial_execution']is True and idle['running']is False and idle['active_serial']is None
        and idle['live_tool_sessions']==idle['active_heavy_units']==[],'Saved idle evidence only; actual physical freshness remains pending')
    actual=read(request['prerequisite_exit']);last=catalog['runs'][-1];require_tool(request['prerequisite_tool'])
    c.require(last['terminal']==request['prerequisite_exit']and last['tool']==request['prerequisite_tool']==actual['actual_tool']
        and actual['reviewer']=='/root'and actual['actual_tool_exit_observed']is True
        and actual['owners_gone']is True and actual['physically_drained']is True
        and last['owners']==actual['owners']and last['units']==actual.get('units',actual.get('all_declared_units'))
        and actual.get('lock_identity',actual.get('audit_lock_identity'))==c.LOCK,'Actual complete last predecessor exit/drain, without changing failure')
    scope=read(seed['lfm_r2_scope'])
    c.require(scope['schema']=='lfm3_full_cpu_successor_scope_v1'and len(scope['children'])==874,'All exact original874 child records retained')

    accepted=read(request['mini_source_review'])
    c.require(accepted['schema']=='minicpm5_retained_caller_finalizer_v2_root_source_review_v1'
        and accepted['reviewer']=='/root'and accepted['passed']is accepted['source_accepted']is True
        and accepted['blockers']==[]and accepted['launch_authorized']is accepted['runtime_admitted']is accepted['production_deletion_admitted']is False
        and accepted['required_finalization_entry']=='minicpm5_retained_replay_finalizer_v2.py'
        and accepted['original_finalizer_v1_admitted']is False,'Only accepted corrected Mini source; no runtime or deletion admission')
    caller_delivery=read(accepted['caller_delivery']);finalizer_delivery=read(accepted['finalizer_delivery'])
    c.require(caller_delivery['schema']=='minicpm5_retained_replay_caller_source_delivery_v1'
        and finalizer_delivery['schema']=='minicpm5_retained_replay_finalizer_v2_source_delivery_v1'
        and caller_delivery['launch_authorized']is finalizer_delivery['launch_authorized']is False,'Exact source-only caller deliveries')
    caller_pins=caller_delivery['mandatory_new_source_sha256'];v2_ref=short_ref(finalizer_delivery['mandatory_new_source_pin'])
    c.require(set(caller_pins)==set(c.FILES)and v2_ref['path']==finalizer.SELF,'All four old modules plus exact v2 entry')
    caller_bytes={p:source(dict(path=p,sha256=h),1<<20)for p,h in caller_pins.items()};v2_bytes=source(v2_ref,1<<20)
    successor_review=read(request['successor_source_review']);successor_delivery=read(request['successor_delivery'])
    c.require(successor_review['schema']=='minicpm5_retained_replay_lifecycle_root_source_review_v1'
        and successor_review['reviewer']=='/root'and successor_review['passed']is successor_review['source_accepted']is True
        and successor_review['blockers']==[]and successor_review['runtime_admitted']is False
        and successor_review['launch_authorized']is successor_review['production_deletion_admitted']is False
        and successor_review['source_delivery']==request['successor_delivery'],
        'Separate root source acceptance of the complete lifecycle successor; not runtime admission')
    c.require(successor_delivery['schema']=='minicpm5_retained_replay_lifecycle_source_delivery_v1'
        and successor_delivery['source_only']is True and successor_delivery['launch_authorized']is False
        and set(successor_delivery['mandatory_new_source_sha256'])==set(successor.SUCCESSOR_FILES),
        'Exact complete lifecycle successor source delivery')
    successor_bytes={p:source(dict(path=p,sha256=h),1<<20)
        for p,h in successor_delivery['mandatory_new_source_sha256'].items()}
    old_review=read(c.ACCEPTED)
    c.require(old_review['passed']is True and old_review['source_ready_for_caller_integration']is True
        and old_review['launch_authorized']is old_review['production_deletion_admitted']is False,'Original raw source acceptance retained')
    for path,sha in c.FROZEN.items():source(dict(path=path,sha256=sha),1<<20)
    original_bytes=source(c.ORIGINAL_PLAN,16<<20);original=json.loads(original_bytes,object_pairs_hook=_unique)
    finite_json_numbers(original)
    c.require(len(original['source_sha256'])==1869 and len(original['analysis_payloads'])==11859,'Complete original1869 sources and11859 logical inputs')
    # Authentication above binds the exact immutable transport plan; no pair or payload is read.
    storage_plan=read(c.PLAN)
    c.require(len(storage_plan['entries'])==2352,'All original2352 pair declarations')

    prep_review=read(request['preparer_source_review']);prep_delivery=read(request['preparer_delivery'])
    c.require(prep_review['schema']=='minicpm5_retained_replay_preparer_root_source_review_v2'
        and prep_review['reviewer']=='/root'and prep_review['source_accepted']is prep_review['passed']is True
        and prep_review['blockers']==[]and prep_review['runtime_admitted']is False
        and prep_review['source_delivery']==request['preparer_delivery'],'Explicit root acceptance of exact preparer')
    c.require(prep_delivery['schema']=='minicpm5_retained_replay_preparer_source_delivery_v2'
        and prep_delivery['source_only']is True and prep_delivery['launch_authorized']is False
        and set(REQUIRED_SOURCES)<=set(prep_delivery['source_sha256']),'Complete accepted preparer/runtime import closure')
    for path,sha in prep_delivery['source_sha256'].items():
        source(dict(path=path,sha256=sha),ORDINARY);new_sources[path]=sha

    all_inputs={};merge_inputs(all_inputs,base['inputs'],json_only=False)
    for inputs in (overlay['input_extension'],seed['input_extension']):merge_inputs(all_inputs,inputs)
    c.require(not set(base['inputs'])&set(overlay['input_extension']),'No original base input replaced')
    seed_pins=merge_sources(seed['source_sha256'],base['source_sha256'],overlay['source_extension'])
    c.require(seed_pins==seed['source_sha256'],'Seed contains complete recovered/base/overlay source closure')
    recovered={p:h for p,h in seed_pins.items()if p not in overlay['source_extension']}
    c.require(all(recovered.get(p)==h for p,h in base['source_sha256'].items()),'Original recovered source partition remains complete')
    pins=merge_sources(seed_pins,*[row['source_sha256']for row in catalog['runs']],original['source_sha256'],
        c.FROZEN,caller_pins,{v2_ref['path']:v2_ref['sha256']},successor_delivery['mandatory_new_source_sha256'],new_sources)
    c.require(all(path in pins for path in REQUIRED_SOURCES),'Every explicitly imported preparer/runtime helper retained')
    # Retain exact inherited large JSON inputs, with no refreshed identity. Source
    # JSON envelopes are derived from stat and fully hashed, never wildcarded.
    inherited_metadata={};merge_inputs(inherited_metadata,overlay['input_extension']);merge_inputs(inherited_metadata,seed['input_extension'])
    for path,row in inherited_metadata.items():
        read(dict(path=path,sha256=row['sha256']),ENVELOPE if row['identity']['size']>ORDINARY else ORDINARY)
        c.require(observed[path]==row,'Every inherited metadata identity retained exactly')
    for path,sha in list(pins.items()):
        if Path(path).parent!=Q or Path(path).suffix!='.json':continue
        guard();ident=identity_only(path);identity_shape(path,ident)
        if ident['size']>ORDINARY:
            read(dict(path=path,sha256=sha),ENVELOPE)
    # Original replay-plan bytes are preparation evidence. The frozen runtime
    # already authenticates that exact constant through its 16 MiB reader;
    # never migrate it into a new ordinary History JSON input/source role.
    runtime_observed={p:r for p,r in observed.items()if p!=c.ORIGINAL_PLAN['path']}
    pins=merge_sources(pins,{p:r['sha256']for p,r in runtime_observed.items()})
    merge_inputs(all_inputs,{p:r for p,r in runtime_observed.items()if Path(p).suffix=='.json'})
    source_extension={p:h for p,h in pins.items()if p not in recovered}
    overlay_inputs=copy.deepcopy(overlay['input_extension'])
    merge_inputs(overlay_inputs,{p:r for p,r in seed['input_extension'].items()if p not in base['inputs']})
    merge_inputs(overlay_inputs,{p:r for p,r in runtime_observed.items()if Path(p).suffix=='.json'and p not in base['inputs']})
    spec_inputs=copy.deepcopy(seed['input_extension']);merge_inputs(spec_inputs,overlay_inputs)
    new_overlay=dict(overlay,current_serial_catalog=request['catalog'],output_root=future['overlay_output_root'],
        source_extension=source_extension,input_extension=overlay_inputs)
    overlay_bytes=encode(new_overlay,ORDINARY);overlay_ref=object_ref(future['overlay_path'],overlay_bytes)
    context=dict(run_name=future['run_name'],history_overlay=overlay_ref,current_serial_catalog=request['catalog'],
        lfm_r2_scope=seed['lfm_r2_scope'],inherited_source_sha256=pins,input_extension=spec_inputs,exact_envelopes=large)
    rendered=successor.render_binding(context,caller_bytes,original_bytes,successor_bytes)
    spec=rendered['spec'];spec_bytes=rendered['spec_bytes'];c.require(len(spec_bytes)<=ENVELOPE,'Original spec64MiB cap')
    spec_ref=object_ref(future['spec_path'],spec_bytes)
    draft=dict(schema='minicpm5_retained_replay_caller_source_review_v2',reviewer='/root',passed=False,
        blockers=['Independent concrete binding review and fresh complete History/physical/resource admission remain pending'],
        spec=spec_ref,source_sha256=copy.deepcopy(spec['source_sha256']),launch_authorized=False,deletion_authorized=False)
    draft_bytes=encode(draft);draft_ref=object_ref(future['review_draft_path'],draft_bytes)
    future_paths=[future[k]for k in ('overlay_path','spec_path','review_draft_path','review_path','candidate_path','overlay_output_root')]
    future_paths += [spec['output_root'],spec['replay_root'],spec['terminal_handoff_path'],*spec['running_paths'].values(),*spec['publication_paths'].values()]
    forbidden=set(observed)|set(pins)|set(all_inputs)|retained|{seed['output_root'],overlay['output_root']}
    if 'running_handoff_path'in seed:forbidden.add(seed['running_handoff_path'])
    c.require(len(set(future_paths))==len(future_paths)and not set(future_paths)&forbidden
        and future['run_name']!=seed['run_name'],'All fresh finite outputs disjoint from complete historical references')
    c.require(all(fresh_path(p)is True for p in future_paths),'Every exact prospective path absent')
    for path,row in observed.items():
        guard();c.require(identity_only(path)==row['identity'],'Metadata/source changed before candidate return')
    guard()
    candidate=dict(schema='minicpm5_retained_replay_binding_candidate_v2',passed=False,launch_authorized=False,deletion_authorized=False,
        paths=copy.deepcopy(future),spec_ref=spec_ref,overlay_ref=overlay_ref,review_draft_ref=draft_ref,
        seed_references={k:request[k]for k in ('seed_spec','seed_review','seed_binding')},catalog=request['catalog'],idle_handoff=request['idle_handoff'],
        prerequisite_exit=request['prerequisite_exit'],actual_prerequisite_tool=request['prerequisite_tool'],
        mini_source_review=request['mini_source_review'],preparer_source_review=request['preparer_source_review'],
        successor_source_review=request['successor_source_review'],successor_delivery=request['successor_delivery'],
        observed_metadata={p:r for p,r in observed.items()if Path(p).suffix=='.json'},
        observed_sources={p:r for p,r in observed.items()if Path(p).suffix!='.json'},
        preparation_only_references={c.ORIGINAL_PLAN['path']:observed[c.ORIGINAL_PLAN['path']]},
        exact_envelopes=large,final_publisher=rendered['final_publisher'],
        preservation=dict(original_sources=1869,original_logical_inputs=11859,original_pairs=2352,lfm_children=874,
            seed_sources=len(seed_pins),current_sources=len(pins),old_catalog_rows=len(old_catalog['runs']),current_catalog_rows=len(catalog['runs']),
            base_inputs=len(base['inputs']),seed_inputs=len(seed['input_extension']),complete_input_sources_retained=True,
            full32_14reports_98layouts_science_unchanged=True,old_failures_preserved=True),
        source_only=True,full_history_precheck_pending=True,physical_freshness_pending=True,actual_resource_check_pending=True,
        history_resolved=False,lock_acquired=False,physical_pair_performed=False,native_executed=False,runtime_launched=False,
        full_model_qualified=False,new_speedup_claim=False)
    candidate_bytes=encode(candidate)
    return dict(values=dict(overlay=new_overlay,spec=spec,review_draft=draft,candidate=candidate),
        files={future['overlay_path']:overlay_bytes,future['spec_path']:spec_bytes,future['review_draft_path']:draft_bytes,
            future['candidate_path']:candidate_bytes},observed=observed,future_paths=future_paths,
        candidate_ref=object_ref(future['candidate_path'],candidate_bytes))


def fresh_path(path):
    p=Path(path)
    # Two exact receipts live beneath the prebound, still absent control root.
    parent_ready=p.parent.is_dir()or(p.name in ('outer-actual-exit.json','observer-actual-exit.json')
        and p.parent.parent==ROOT and p.parent.name.startswith('minicpm5-retained-verification-')
        and not os.path.lexists(p.parent)and ROOT.is_dir())
    return (p.is_absolute()and str(p)==path and '..'not in p.parts and not os.path.lexists(p)
        and parent_ready and not any(parent.is_symlink()for parent in p.parents))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',required=True);parser.add_argument('--request-sha256',required=True)
    args=parser.parse_args(argv);c.require(sys.dont_write_bytecode and __debug__,'Explicit -B/assertions')
    from qwen35_sampler_runtime_io_v12 import loaded_local_sources
    bootstrap=StableReaders(lambda:None);request_ref=dict(path=args.request,sha256=args.request_sha256)
    request=bootstrap.metadata(request_ref,1<<20)['value']
    idle_path=request['idle_handoff']['path'];idle_identity=file_identity(idle_path)
    def guard():
        c.require(file_identity(idle_path)==idle_identity,'Explicit idle metadata identity changed during preparation')
        c.require(not {'torch','numpy','transformers','torch_neuronx','neuronxcc','vllm'}&set(sys.modules),'Metadata/source preparation only')
    reader=StableReaders(guard);result=prepare(request,reader.source,file_identity,fresh_path,guard)
    loaded_local_sources(result['values']['spec']['source_sha256'])
    c.require(file_identity(request_ref['path'])==bootstrap.observed[request_ref['path']]['identity'],'Explicit root request unchanged')
    for path,row in result['observed'].items():c.require(file_identity(path)==row['identity'],'Complete input/source identity unchanged before publication')
    c.require(all(fresh_path(p)for p in result['future_paths']),'Every prospective path still absent')
    for path,payload in result['files'].items():
        guard()
        with Path(path).open('xb')as stream:stream.write(payload);stream.flush();os.fsync(stream.fileno())
    directory=os.open(Q,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC)
    try:os.fsync(directory)
    finally:os.close(directory)
    print(json.dumps(dict(candidate=result['candidate_ref'],spec=result['values']['candidate']['spec_ref'],
        launch_authorized=False,full_history_precheck_pending=True)),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
