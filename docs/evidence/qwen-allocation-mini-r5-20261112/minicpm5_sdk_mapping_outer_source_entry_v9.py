"""Concrete future outer source_entry: original auth, one SDK child, real drain.

Inert on import. The no-cache -c entry supplies the actual source-execution
object. No historical module is imported until separate current outer312
native/source authentication succeeds. No present spec/authority is generated.
"""
import copy
import json
import signal
import sys
import time
from research import minicpm5_sdk_mapping_bootstrap_descriptor_v4 as descriptor
from research import minicpm5_sdk_mapping_history_gate_v1 as history
from research import minicpm5_sdk_mapping_native_authenticator_v3 as native
from research import minicpm5_sdk_mapping_source_adapter_v3 as adapter
from research import minicpm5_sdk_mapping_nocache_launch_v3 as launch
from research import minicpm5_formal_literal_result_v1 as literal_result
from research import minicpm5_formal_literal_codec_v1 as literal_codec
from research import minicpm5_fixed_input_pv_catalog_auth_v2 as catalog_auth

require=descriptor.require


def _reference(value):
    launch.reference(value)
    return value


def admission_types_observation(observation):
    """Project only the two frozen v6 stat tuples at the JSON boundary.

    The early source fence retains its original observation and tuple receipts.
    Every field/value is preserved; original admission readback equality stays
    exact. No recursive coercion, unknown identity shape or missing receipt is
    accepted here.
    """
    require(type(observation) is dict
        and observation.get('schema') == 'minicpm5_outer_types_source_observation_v1'
        and observation.get('executed_through_pinned_loader') is True
        and type(observation.get('execution_receipt')) is dict
        and set(observation['execution_receipt']) == {'sha256', 'bytes', 'identity'},
        'Completed exact frozen types observation and execution receipt')
    current = observation.get('current_source_identity')
    executed = observation['execution_receipt']['identity']
    for identity in (current, executed):
        require(type(identity) is tuple and len(identity) == 7
            and all(type(field) is int and field >= 0 for field in identity),
            'Exact seven original stat identity fields at admission boundary')
    result = copy.deepcopy(observation)
    result['current_source_identity'] = list(current)
    result['execution_receipt']['identity'] = list(executed)
    require(literal_codec.encoded(result) == literal_codec.encoded(observation),
            'Every original observation byte/value preserved in JSON domain')
    return result


def _entry_json(entry,ref):
    _reference(ref)
    raw,_=entry.read_source(ref['path'],ref['sha256'])
    return native.decode(raw)


def _complete_source_map(reader,reference,original,metadata):
    """Only this full map uses the original pipe cap; retain metadata identity."""
    metadata.guard();original.r.sources.reference(reference)
    require(original.PIPE_CAP==16*2**20,'Unchanged original complete source pipe ceiling')
    path=reference['path'];before=original.r.identity(path)
    value=reader.json(reference,original.PIPE_CAP)
    require(original.r.identity(path)==before,'Stable complete authenticated source-map metadata')
    require(path not in metadata.identities or metadata.identities[path]==before,
            'No complete source-map metadata identity rebound')
    metadata.identities[path]=before;metadata.guard()
    return value


def validate_inputs(entry,text):
    """Concrete reviewed binding, additive to all original diagnostic inputs."""
    require(entry.role=='outer312' and entry.history_owner is None,
            'Actual untouched outer312 literal entry')
    refs=native.decode(text.encode())
    require(type(refs) is dict and set(refs)=={'spec','review'},'Explicit future outer spec/review references')
    spec,review=_entry_json(entry,refs['spec']),_entry_json(entry,refs['review'])
    require(type(spec) is dict and set(spec)=={'schema','attempt_id','unit','diagnostic_spec',
        'diagnostic_review','complete_source_map','sdk_entry_template','sdk_entry_source',
        'limits','artifact_limits','source_extension','sdk_import_paths'}
        and spec['schema']=='minicpm5_mapping_source_outer_spec_v1'
        and spec['limits']==descriptor.previous.logic.LIMITS
        and spec['artifact_limits']==descriptor.previous.logic.CAPS,
        'Exact original resource/output/scientific domain; additive outer binding only')
    require(type(review) is dict and set(review)=={'schema','reviewer','passed','blockers','spec','source_extension'}
        and review['schema']=='minicpm5_mapping_source_outer_review_v1'
        and review['reviewer']=='/root' and review['passed'] is True and review['blockers']==[]
        and review['spec']==refs['spec'] and review['source_extension']==spec['source_extension'],
        'Concrete independently accepted root review, never fabricated here')
    require(spec['source_extension']==entry.plan['bootstrap_sources'],
            'Every actual outer bootstrap source explicitly reviewed')
    require(spec['sdk_import_paths']==['/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages'],
            'Only original SDK310 site directory; -S never executes .pth/sitecustomize')
    for key in ('diagnostic_spec','diagnostic_review','complete_source_map','sdk_entry_template','sdk_entry_source'):
        _reference(spec[key])
    return refs,spec,review


def load_history(entry,bootstrap,pins):
    """Implement the concrete order, preserving original full-nine later."""
    inputs=bootstrap.bootstrap_inputs(pins)
    authority={key:entry.plan['native_authority'][key] for key in ('path','sha256')}
    authenticator=native.NativeAuthenticator(bootstrap)
    def authenticate_before_import(current,review,guard):
        receipt=authenticator.authenticate_bootstrap(current,review,guard)
        # This runs inside original PreImportFence, before ctypes/history.
        entry.confirm_bootstrap_sources(authenticator)
        return receipt
    gate=history.HistoricalImportGate(inputs,authority,guard=bootstrap,
        authenticate_bootstrap=authenticate_before_import,
        verify_loaded_native=authenticator.verify_loaded_native,
        verify_loaded_history=authenticator.verify_loaded_history,
        ctypes_loader=authenticator.load_ctypes,history_loader=authenticator.load_history)
    entry.bind_history_owner(gate,authenticator,bootstrap)
    original=gate.load()
    return original,gate,authenticator


def source_entry(entry,successor_input):
    """Actual future -c outer call, not an injected authorization callback."""
    refs,spec,review=validate_inputs(entry,successor_input)
    bootstrap=descriptor.EntryBootstrapDescriptor(entry,spec['attempt_id'],spec['unit'])
    ports=original=authenticator=gate=None
    try:
        bootstrap.acquire()
        bootstrap.impose_original_limits()
        reader=native.StableReader(bootstrap)
        # The original complete source release already approaches8MiB. This
        # one input uses the original16MiB pipe ceiling, with the unchanged
        # full packet plus65536-byte reservation checked again before spawn.
        pins=reader.json(spec['complete_source_map'],16*2**20)
        require(type(pins) is dict and all(pins.get(path)==sha
            for path,sha in spec['source_extension'].items()),'Complete original map includes every new executed source')
        original,gate,authenticator=load_history(entry,bootstrap,pins)
        lease,guard,metadata=bootstrap.adopt(original)
        # All history/native imports are now under the independently verified
        # source loader; the entry audit fence remains installed permanently.
        with native.SourceFinder(authenticator):
            original.r.p.user_bus()
            authenticated=catalog_auth.authenticate_original(gate,spec['diagnostic_spec'],spec['diagnostic_review'],
                lease=lease,metadata=metadata,guard=guard)
            context=authenticated['context']
            require(metadata.read(refs['spec'])==spec and metadata.read(refs['review'])==review,
                    'Supplemental entry review remains bound after full original authentication')
            require(_complete_source_map(reader,spec['complete_source_map'],original,metadata)==context['pins_plan']['source_sha256']==pins,
                    'No bootstrap subset replaces whole frozen historical reconstruction')
            worker_plan=metadata.read(spec['sdk_entry_template'])
            launch.separate_authorities(entry.plan,worker_plan)
            require(worker_plan['successor_input']=='admission:self'
                and worker_plan['entry_module']=='research.minicpm5_sdk_mapping_worker_source_entry_v6'
                and worker_plan['entry_path']==str(original.ROOT/'research/minicpm5_sdk_mapping_worker_source_entry_v6.py')
                and all(pins.get(path)==sha for path,sha in worker_plan['bootstrap_sources'].items()),
                'Exact separately reviewed SDK310 worker source closure')
            entry_raw,_=authenticator.reader.read(spec['sdk_entry_source'],65535)
            require(worker_plan['entry_sha256']==spec['sdk_entry_source']['sha256']
                and pins.get(spec['sdk_entry_source']['path'])==spec['sdk_entry_source']['sha256'],
                'Actual SDK literal entry bytes also in complete original source map')
            source_subset={key:context['pins_plan'][key] for key in ('source_contract','source_sha256')}
            require(len(original.serial.encoded(source_subset))+65536<=original.PIPE_CAP,
                    'Original complete source release packet fits before actual child spawn')
            classes=adapter.make_adapter_class(original)
            store=classes.ObserverStore(authenticated['output_root'],guard,literal=entry_raw.decode('utf-8'))
            store.create()
            model=metadata.read(context['original_spec']['model_exit'])
            admission=dict(schema=original.ADMISSION_SCHEMA,witness=authenticated['witness'],
                lease=bootstrap.binding,output_root=authenticated['output_root'],
                original_native_source_receipt=model['outputs']['source_authentication'],
                diagnostic_spec=spec['diagnostic_spec'],diagnostic_review=spec['diagnostic_review'],
                source_entry_plan=copy.deepcopy(worker_plan),source_entry_binding=dict(
                    outer_spec=refs['spec'],outer_review=refs['review'],
                    outer_entry_plan=copy.deepcopy(entry.plan_reference),
                    outer_started=entry.started,outer_deadline=entry.deadline,
                    outer_types_source_observation=admission_types_observation(entry.types_observation),
                    original_worker=authenticated['spec']['worker'],
                    sdk_entry_source=copy.deepcopy(spec['sdk_entry_source'])))
            _,admission_ref=store.prospective_admission(admission)
            sdk_argv=launch.admitted_sdk(entry_raw,spec['sdk_entry_source'],admission,admission_ref,entry.deadline)
            store.reserve_runtime(admission,admission_ref,sdk_argv)
            actual_admission_ref=original.original.short(store.json('admission.json',admission))
            require(actual_admission_ref==admission_ref,'Actual durable admission equals reserved complete wire')
            admission_metadata=literal_codec.AdmissionMetadata(metadata,admission_ref,entry_raw.decode('utf-8'),guard)
            ports=classes.SourceEntrySDKAdapter(authenticated,admission_ref,admission,lease,guard,admission_metadata,store,
                outer_deadline=entry.deadline,authority=gate.authority,
                authenticate=authenticator.authenticate,verify_loaded_native=authenticator.verify_loaded_native,
                sdk_entry_bytes=entry_raw,sdk_entry_ref=spec['sdk_entry_source'])
            ports.authenticate_complete_prerequisites()
            child_owner,argv=ports.start()
            ports.finish(child_owner,argv)
            result=ports.result()
            reader.unchanged({spec['complete_source_map']['path']})
            raw=literal_result.render(result,store,entry_raw.decode('utf-8'))
            require(len(raw)<=original.SUMMARY_CAP,'Whole explicit observed result fits original summary budget')
            print(raw.decode(),end='',flush=True)
            return result['exit_code']
    finally:
        if bootstrap.fd is not None:
            if not bootstrap.adopted:
                require(ports is None,'No child exists before original descriptor adoption')
                bootstrap.release_before_adoption(dict(physically_drained=True,child_started=False))
            else:
                # Original failure cleanup: no timer restart and no weaker
                # writer. Observer retains its one real wait/Popen owner.
                signal.alarm(0)
                warned=False;persistence_failed=False
                while True:
                    try:
                        proof=ports.drain_owned() if ports is not None else dict(
                            physically_drained=True,no_children_started=True)
                        break
                    except BaseException as error:
                        if ports is not None and ports.state=='drained':persistence_failed=True
                        if not warned:
                            print(json.dumps(dict(event='sdk_mapping_source_cleanup_pending_original_lock_retained',
                                error_type=type(error).__name__)),file=sys.stderr,flush=True)
                            warned=True
                        time.sleep(.5)
                original.prior.audited.support.release_after_physical_drain(bootstrap.fd,proof)
                require(not persistence_failed,'Actual child drained but original durable exit receipt failed')
