"""Concrete future SDK310 source entry, preserving original import diagnostic.

Actual callable body only. Source tests replace OS/native boundaries. No real
authority is created and no original model/scientific gate is removed. The
worker never reports its own exit or releases the original parent's descriptor.
"""
import copy
import json
import os
from pathlib import Path
import signal
import stat
from research import minicpm5_sdk_mapping_worker_guard_v2 as startup
from research import minicpm5_sdk_mapping_worker_native_v4 as native
from research import minicpm5_sdk_mapping_nocache_launch_v2 as launch

require=startup.require


def source_entry(entry,successor_input):
    require(entry.role=='sdk310' and successor_input=='admission:self'
        and entry.admission_envelope is not None,'Exact actual parent-admitted SDK source entry')
    admission=entry.admission_envelope
    bootstrap=startup.SDKEntryGuard(entry,admission)
    bootstrap.start()
    packet=bootstrap.read_release()
    pins_plan=packet['source_plan']
    pins=pins_plan['source_sha256'];witness=admission['witness']
    require(native.history.digest(pins)==witness['complete_source_map_sha256']
        and len(pins)==witness['complete_source_files'],
        'Complete inherited original source map received via acknowledged parent release')
    authority={key:entry.plan['native_authority'][key] for key in ('path','sha256')}
    authenticator=native.SDKNativeAuthenticator(bootstrap,pins,authority)
    gate=native.SDKHistoryGate(entry,authenticator,bootstrap)
    entry.bind_sdk_history_owner(gate,authenticator,bootstrap)
    original=gate.load()
    lease,guard,metadata=bootstrap.adopt(original)
    with native.SourceFinder(authenticator):
        require(metadata.read(entry.plan_reference)==admission,
                'Actual same immutable admission after original child metadata authentication')
        r,serial=original.r,original.serial
        partition=r.sources.classify(pins,pins_plan['source_contract'],metadata.read)
        r.sources.verify_text_sources(partition,guard)
        source_receipt=metadata.read(admission['original_native_source_receipt'])
        require(partition['categories']['query_native_binary_sha256']==source_receipt['native_sha256']
            and len(source_receipt['native_sha256'])==9,
            'Exact original complete nine native sources, no replacement by SDK closure')
        r.sources.check_native_identities(source_receipt,partition,guard)
        r.verify_loaded(pins_plan,guard)
        plan=original.restore_descriptor_plan(witness,metadata.read)
        plan_sha=r.digest(plan)
        allowed=original.core.saved.validate_plan(plan)
        io=original.core.saved.StableCaptureIO(allowed,guard,filesystem=os)
        store=original.DiagnosticStore(admission['output_root'],guard)
        store.inventory()
        spec_ref,review_ref=admission['diagnostic_spec'],admission['diagnostic_review']
        spec,review=metadata.read(spec_ref),metadata.read(review_ref)
        original.validate_spec(spec_ref,review_ref,spec,review)
        supplemental=admission['source_entry_binding']
        require(spec['worker']==supplemental['original_worker']
            and spec['pv_spec']==witness['spec'] and spec['pv_review']==witness['review']
            and all(pins.get(path)==sha for path,sha in spec['source_extension'].items()),
            'All frozen original worker/spec/source semantics retained')
        outer_spec=metadata.read(supplemental['outer_spec'])
        outer_review=metadata.read(supplemental['outer_review'])
        require(outer_spec['schema']=='minicpm5_mapping_source_outer_spec_v1'
            and outer_review['schema']=='minicpm5_mapping_source_outer_review_v1'
            and outer_review['reviewer']=='/root' and outer_review['passed'] is True
            and outer_review['blockers']==[] and outer_review['spec']==supplemental['outer_spec']
            and outer_review['source_extension']==outer_spec['source_extension']
            and outer_spec['diagnostic_spec']==spec_ref and outer_spec['diagnostic_review']==review_ref
            and outer_spec['limits']==original.prior.LIMITS and outer_spec['artifact_limits']==serial.CAPS,
            'Actual separate accepted entry binding preserves original resource/artifact limits')
        require(metadata.read(outer_spec['sdk_entry_template'])==entry.plan
            and outer_spec['sdk_entry_source']==supplemental['sdk_entry_source']
            and all(pins.get(path)==sha for path,sha in entry.plan['bootstrap_sources'].items()),
            'Worker entry template and executed source closure independently bound by root review')
        outer_plan=metadata.read(supplemental['outer_entry_plan'])
        launch.separate_authorities(outer_plan,entry.plan)
        entry_ref=supplemental['sdk_entry_source']
        entry_bytes,_=authenticator.reader.read(entry_ref,65535)
        expected_argv=launch.admitted_sdk(entry_bytes,entry_ref,admission,entry.plan_reference,entry.deadline)
        require(expected_argv==entry.runtime.orig_argv==bootstrap.owner['argv'],
                'Actual new literal argv, never fake old worker argv')
        r.life.process(bootstrap.owner,expected_argv,admission['lease']['outer_owner']['pid'],
            admission['lease']['outer_owner']['cgroup'])
        expected_env=r.p.environment({},'cpu_control',admission['lease']['attempt_id'])
        require(dict(os.environ)==expected_env,'Original exact SDK child environment before import')
        require(r.p.proc(admission['lease']['outer_owner']['pid'])==admission['lease']['outer_owner'],
                'Same real original outer still owns parent descriptor')
        require(serial.PHASES[1]==('sdk','own_before') and r.digest(plan)==plan_sha,
                'Original initial SDK control and complete scientific descriptor plan unchanged')
        paths=outer_spec['sdk_import_paths']
        require(paths==['/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages'],
                'Exact independently reviewed SDK310 import location; no outer312 site reuse')
        for path in paths:
            directory=Path(path)
            require(directory.resolve()==directory and all(stat.S_ISDIR(item.lstat().st_mode)
                for item in (directory,*directory.parents)),'Actual canonical SDK source directory')
            require(path+'/torch/__init__.py' in pins
                and path+'/neuronx_distributed_inference/__init__.py' in pins,
                'Actual SDK package entries are already in whole source closure')
            require(path not in entry.runtime.path,'Site initialization has not already occurred')
            entry.runtime.path.append(path)
        gate.state='complete_original_authentication'
        result,code=original.execute_import(pins_plan,guard)
        require(not io._read,'Original import-only diagnostic reads zero model/capture tensors')
        r.verify_loaded(pins_plan,guard);metadata.unchanged();guard()
        store.json('phase-1-summary.json',result);store.inventory();guard()
        # Do not cancel/reset alarm inside the scientific body. CPython process
        # exit retains deadline enforcement through entry.check/main exit. Only
        # the real parent owns wait/drain and original descriptor release.
        return code
