"""Explicit legacy replay successor; import is stdlib/source only and inert.

Frozen modules are never patched. Same-code function copies receive their own
globals only where evidence has to record historical capture identity separately
from live storage identity. Full11859 admission and original32 scientific work
remain mandatory. Run only in a separately admitted disposable CPU worker.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import struct
import sys
import tempfile
import types

import minicpm5_retained_raw_transport_v1 as transport


def isolated_functions(module, replacements):
    """Explicit dependency injection, preserving original function code objects.

    No module object or sys.modules entry changes. Mutable evidence accumulators
    must be supplied as fresh replacements by the one caller below.
    """
    namespace=dict(vars(module));namespace.update(replacements)
    for name,value in vars(module).items():
        if isinstance(value,types.FunctionType) and value.__globals__ is vars(module) and name not in replacements:
            function=types.FunctionType(value.__code__,namespace,value.__name__,value.__defaults__,value.__closure__)
            function.__kwdefaults__=value.__kwdefaults__;namespace[name]=function
    return types.SimpleNamespace(**namespace)


class Payloads:
    def __init__(self, original_plan_bytes, storage_plan, verification_ref, guard):
        transport.require(transport.digest(original_plan_bytes)==transport.REFERENCES['recovery_plan']['sha256'],
            'Unmodified original recovery plan bytes')
        self.original_bytes=original_plan_bytes;self.plan=json.loads(original_plan_bytes)
        self.replay=transport.Replay(storage_plan,verification_ref,guard)
        self.guard=guard;self.resolutions={}
        transport.require(len(self.plan['analysis_payloads'])==11859,'Complete original11859 input map')
        for path,row in self.replay.entries.items():
            transport.require(self.plan['analysis_payloads'][path]==dict(identity=row['original'],sha256=row['sha256']),
                'All mapped captures belong to exact original plan')

    def check_capture(self, path, declared):
        """Return TWO identities; never masquerade recorded identity as live stat."""
        path=str(path);self.guard()
        transport.require(self.plan['analysis_payloads'].get(path)==declared,'Exact original logical declaration')
        pair=self.replay.entries.get(path)
        physical=pair['keeper'] if pair is not None else declared['identity']
        transport.require(transport.identity(physical['realpath'])==physical,'Actual storage identity changed')
        proof=dict(logical_path=path,capture_identity=copy.deepcopy(declared['identity']),
            storage_identity=copy.deepcopy(physical),sha256=declared['sha256'],
            relocated=pair is not None,original_identity_is_current_stat=pair is None)
        if pair is not None:self.resolutions[path]=proof
        return proof

    def verify_payload_identities(self, plan):
        transport.require(plan==self.plan,'No projection or rewritten original logical plan')
        for path,row in plan['analysis_payloads'].items():self.check_capture(path,row)
        return True

    def admit_all(self, plan):
        self.verify_payload_identities(plan);hashes={}
        for path,row in plan['analysis_payloads'].items():
            proof=self.check_capture(path,row);ident=proof['storage_identity'];fd=transport._open(ident)
            try:
                h=hashlib.sha256();count=0
                while True:
                    self.guard();block=os.read(fd,transport.CHUNK)
                    if not block:break
                    count+=len(block);transport.require(count<=ident['size'],'Payload grew');h.update(block)
                transport.require(count==row['identity']['size'] and h.hexdigest()==row['sha256'],'Full logical payload SHA and bytes')
                transport._unchanged(fd,ident)
            finally:os.close(fd)
            hashes[path]=h.hexdigest()
        self.verify_payload_identities(plan)
        return dict(passed=True,plan_sha256=transport.REFERENCES['recovery_plan']['sha256'],sha256=hashes,
            payloads=len(plan['analysis_payloads']),model_payloads_loaded=False,producer_layout_preserved=True,
            stored_layout_separately_checked=False,stored_layout_validation_required=True,
            original_capture_identities_preserved=True,physical_resolutions=copy.deepcopy(self.resolutions))


def evidence_view(original, payloads):
    """All original numeric/evidence functions unchanged; only capture IO injected.

    STATS is explicitly the original capture provenance used by the frozen audit,
    while LIVE_STORAGE records current physical resolution separately. The
    compatibility return value is a recorded capture identity, never live stat.
    """
    stats={};live={}
    def recorded_capture(path, identity, digest):
        original.checks.check_hash(digest)
        if str(path)in payloads.plan['analysis_payloads']:
            proof=payloads.check_capture(path,dict(identity=identity,sha256=digest))
        else:
            # The acquisition evidence also stats earlier matched32 control
            # inputs outside the reanalysis11859. Preserve their original live
            # identity check; they are NEVER eligible for alias transport.
            payloads.guard();actual=original.p.identity(path)
            transport.require(actual==identity,'Unmapped historical producer input changed')
            proof=dict(capture_identity=actual,storage_identity=actual,relocated=False)
        entry=dict(identity=copy.deepcopy(proof['capture_identity']),worker_sha256=digest,auditor_payload_read=False)
        transport.require(str(path)not in stats or stats[str(path)]==entry,'Stable original capture provenance')
        stats[str(path)]=entry
        if proof['relocated']:live[str(path)]=proof
        return copy.deepcopy(proof['capture_identity'])
    return isolated_functions(original,dict(payload_stat=recorded_capture,TEXT={},STATS=stats,MANIFESTS={},LIVE_STORAGE=live))


class Tensors:
    def __init__(self, original, payloads):self.original,self.payloads=original,payloads
    def __getattr__(self,name):return getattr(self.original,name)
    def load_state(self,path,row):
        transport.require(str(path)==row['file'],'Exact producer logical path')
        if str(path)not in self.payloads.replay.entries:return self.original.load_state(path,row)
        import torch
        data,proof=self.payloads.replay.read_state(row)
        assert proof['capture_identity']==row['identity'] and proof['sha256']==row['raw_sha256']
        assert len(data)==row['raw_bytes']==self.original.p.CACHE_BYTES
        value=torch.frombuffer(bytearray(data),dtype=torch.uint8).view(torch.bfloat16).reshape(row['shape'])
        assert int((~torch.isfinite(value)).sum())==row['nonfinite_elements']==0
        return value


class Protocol:
    """New output namespace plus unchanged original input/science contract.

    guard() must be supplied by the root-reviewed external worker admission and
    must continuously verify the original serial lock, complete current History,
    all owners/cgroups and this exact worker/source/runtime binding. This class
    does not replace those physical admission responsibilities with booleans.
    """
    def __init__(self, original, payloads, destination, guard):
        self.original,self.payloads,self.external_guard=original,payloads,guard
        self.DEST=Path(destination);self.RUNTIME_READY=False
        transport.require(self.DEST.parent==transport.ROOT and self.DEST.name.startswith('minicpm5-history-retained-replay-'),
            'Fresh independent replay namespace')
        self.io=isolated_functions(original,{'DEST':self.DEST})
    def __getattr__(self,name):return getattr(self.io,name)
    def verify(self):
        self.external_guard();return verify_protocol_metadata(self.original,self.payloads)
    def verify_payload_identities(self,plan):return self.payloads.verify_payload_identities(plan)
    def guard(self,phase):
        transport.require(phase=='compare','Retained analysis phase only');lease=self.external_guard()
        transport.require(type(lease)is dict and lease['schema']=='minicpm5_retained_raw_replay_worker_lease_v1'
            and lease['output_root']==str(self.DEST)
            and lease['original_plan_sha256']==transport.REFERENCES['recovery_plan']['sha256']
            and lease['storage_plan_sha256']==transport.PRODUCTION_PLAN_SHA256,
            'Explicit root-reviewed standalone worker and complete History/source lease')
        assert __debug__ and sys.flags.optimize==0 and sys.dont_write_bytecode
        assert Path(sys.executable).resolve()==Path(self.PYTHON).resolve() and Path.cwd()==self.ROOT
        assert all(os.environ.get(k)==v for k,v in self.io.environment().items())
        assert sys.pycache_prefix==str(self.DEST/'empty-pycache')
        assert not any(os.environ.get(k)for k in ('PYTHONPATH','PYTHONHOME','PYTHONSTARTUP','LD_PRELOAD','LD_LIBRARY_PATH','VLLM_PLUGINS'))
        assert not {'torch','numpy','transformers','torch_neuronx','vllm'}&set(sys.modules)
        assert os.sched_getaffinity(0)=={7} and os.getpriority(os.PRIO_PROCESS,0)==19
        unit='trn1-latest-verify-'+self.DEST.name+'-compare'
        assert lease['worker_unit']==unit
        own=dict(pid=os.getpid(),start_ticks=self.pid_start(os.getpid()))
        assert lease['worker_owner']==own
        props=self.original.exact_owner(own,lease['worker_argv'],unit)
        assert [int(props[k])for k in ('MemoryMax','MemoryHigh','MemorySwapMax','TasksMax')]==[6*2**30,5*2**30,0,32]
        assert self.old.heavy_units()==[unit+'.service']
        plan=self.verify();self.io.storage(start=True)
        transport.require(not os.path.lexists(self.DEST),'Never overwrite an old or interrupted replay')
        self.DEST.mkdir();(self.DEST/'tmp').mkdir(mode=0o700);(self.DEST/'empty-pycache').mkdir()
        with (self.DEST/'plan.json').open('xb')as f:
            f.write(self.payloads.original_bytes);f.flush();os.fsync(f.fileno())
        return plan
    def admit_payloads(self,plan):
        self.external_guard();self.io.once(self.DEST/'payload-admission.json',self.payloads.admit_all(plan))
    def activate(self,plan):
        self.external_guard();assert plan==self.payloads.plan and not self.RUNTIME_READY
        admission=self.read(self.DEST/'payload-admission.json')
        assert admission['passed'] and admission['sha256']=={p:r['sha256']for p,r in plan['analysis_payloads'].items()}
        # Preserve original devnull policy and fresh-empty temporary directory
        # behavior, recording only this explicit execution-namespace translation.
        actual=self.io.runtime_io_contract();original=self.original.runtime_io_contract()
        assert {k:v for k,v in actual.items()if k!='temporary_directory'}=={k:v for k,v in original.items()if k!='temporary_directory'}
        assert plan['runtime_io']==original and not list((self.DEST/'tmp').iterdir())
        assert os.environ.get('TMPDIR')==str(self.DEST/'tmp')
        tempfile.tempdir=str(self.DEST/'tmp');assert tempfile.gettempdir()==str(self.DEST/'tmp')
        self.io.once(self.DEST/'retained-runtime-io.json',dict(original=original,execution=actual,
            historical_plan_unchanged=True,only_temporary_directory_namespace_translated=True))
        allowed=set(plan['analysis_payloads'])
        assert all(r['keeper']['realpath']in allowed for r in self.payloads.replay.entries.values())
        def runtime_audit(event,args):
            if event=='import':assert args[0].split('.')[0]not in self.FORBIDDEN_IMPORTS,args[0]
            assert event not in {'subprocess.Popen','os.system','os.posix_spawn','os.exec','os.fork','os.forkpty','pty.spawn','socket.connect','socket.connect_ex','socket.getaddrinfo','socket.bind','socket.sendto','socket.sendmsg'},event
            if event=='open'and isinstance(args[0],(str,bytes)):
                path=Path(os.fsdecode(args[0]));mode=args[1];flags=args[2]
                if path==Path('/dev/null'):
                    self.original.devnull_open(args[0],mode,flags,original['devnull']);return
                writing=isinstance(mode,str)and any(k in mode for k in 'wax+')or isinstance(flags,int)and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
                if writing:
                    assert path.is_relative_to(self.DEST)and path.suffix in ('.json','.log')and not path.is_relative_to(self.DEST/'tmp')
                    assert path.resolve()==path and all(not item.is_symlink()for item in(path,*path.parents))
                if path.suffix in ('.pt','.raw','.bin','.safetensors','.neff','.zlib'):assert str(path)in allowed,path
        sys.addaudithook(runtime_audit);self.RUNTIME_READY=True


def check_saved_terminal(original_protocol, original_queue, payloads):
    """Successor to the old terminal auditor's q.verify()/check_receipt pair.

    Full original queue receipt and check_comparison are reused. Supplying the
    independently verified original plan prevents queue.check_receipt from
    internally calling its frozen inode-only verify(). No old result is edited.
    """
    plan=verify_protocol_metadata(original_protocol,payloads)
    result=original_queue.check_receipt(plan)
    payloads.verify_payload_identities(plan)
    return dict(original_result=result,original_plan_unchanged=True,
        storage_resolution=copy.deepcopy(payloads.resolutions),original_model_qualification_unchanged=True)


def run_replay(payloads, destination, external_guard):
    """Explicit future admitted-worker entry, never called by source tests.

    Caller sets TMPDIR to <destination>/tmp before entry and provides complete
    original/new source authentication, environment/resource controls and actual
    serial ownership through external_guard. It must remain continuously live.
    """
    transport.require(payloads.guard is external_guard,'One identical admitted worker lease for all payload and runtime checks')
    external_guard()
    import minicpm5_history_recovery_protocol_v3 as old_protocol
    import minicpm5_history_recovery_evidence_v2 as old_evidence
    import minicpm5_history_tensors_v1 as old_tensors
    import minicpm5_history_recovery_tensors_v3 as old_saved
    import run_minicpm5_history_analysis_recovery_v3 as old_analysis
    p=Protocol(old_protocol,payloads,destination,external_guard)
    e=evidence_view(old_evidence,payloads);t=Tensors(old_tensors,payloads)
    saved=isolated_functions(old_saved,{'p':p,'e':e})
    science=isolated_functions(old_analysis,{'p':p,'e':e,'saved':saved})
    replay_analysis(p,e,saved,t,science.check_comparison,'compare')
    external_guard()
    return dict(output_root=str(p.DEST),comparison=str(p.DEST/'comparison.json'),
        physical_resolution=copy.deepcopy(payloads.resolutions),original_plan_unchanged=True,
        full_model_qualified=False,new_speedup_claim=False)


# The following two concrete functions are mechanically derived from exact
# frozen source bodies. Source tests retain all old metadata/scientific checks;
# only the explicitly documented IO plumbing differs.


def verify_protocol_metadata(protocol, payloads):
    plan = protocol.read(protocol.DEST / 'plan.json')
    assert plan['schema'] == protocol.SCHEMA and plan['scope'] == protocol.scope() and (plan['criteria'] == protocol.old.CRITERIA)
    assert plan['phases'] == list(protocol.PHASES) and plan['queue_script'] == str(protocol.QUEUE_SCRIPT) and (plan['queue_unit'] == protocol.QUEUE)
    assert plan['source_author'] == protocol.SOURCE_AUTHOR and plan['original_plan_sha256'] == protocol.evidence.PLAN_SHA256
    assert plan['storage'] == dict(owned_cap_bytes=protocol.CAP, root_floor_bytes=protocol.ROOT_FLOOR, nvme_floor_bytes=protocol.NVME_FLOOR, nvme_growth_bytes=0)
    assert plan['runtime_io'] == protocol.runtime_io_contract()
    assert plan['failed_recovery'] == dict(path=str(protocol.FAILED_RECOVERY), sha256=protocol.FAILED_RECOVERY_SHA256)
    assert protocol.sha(protocol.FAILED_RECOVERY) == protocol.FAILED_RECOVERY_SHA256
    assert all((plan['source_sha256'][str(path)] == protocol.sha(path) for path in protocol.import_sources()))
    assert plan['remaining_gates'] == protocol.read(protocol.evidence.c.DESIGN)['preserved_followup_gates']
    assert plan['failed_audit'] == dict(path=str(protocol.evidence.FAILED_AUDIT), sha256=protocol.evidence.FAILED_SHA256, kind='large_immutable_json_metadata')
    assert protocol.sha(protocol.evidence.FAILED_AUDIT) == protocol.evidence.FAILED_SHA256
    for (name, digest) in plan['source_sha256'].items():
        assert protocol.sha(name) == digest, name
    for (name, digest) in plan['acquisition_metadata_sha256'].items():
        assert protocol.sha(name) == digest, name
    for (name, row) in plan['analysis_payloads'].items():
        payloads.check_capture(name, row)
    assert len(plan['analysis_payloads']) == 11859
    frozen = protocol.read(protocol.evidence.FAILED_AUDIT)['salvage_design']['exact_retained_stat_and_digest_provenance']
    expected = {name: dict(sha256=row['worker_sha256'], identity=row['identity']) for (name, row) in frozen.items() if protocol.Path(name).is_relative_to(protocol.old.DATA) and protocol.Path(name).suffix in ('.pt', '.raw')}
    expected[str(protocol.old.HF)] = protocol.read(protocol.old.DEST / 'plan.json')['payloads'][str(protocol.old.HF)]
    assert plan['analysis_payloads'] == expected
    assert all((str(protocol.ROOT / name) in plan['source_sha256'] for name in protocol.source_names())) and str(protocol.QUEUE_SCRIPT) in plan['source_sha256']
    assert plan['independent_review'] in plan['source_sha256']
    tail = plan['previous_tail']
    lineage = protocol.read(protocol.ROOT / tail['plan'])['predecessors'] + [tail['queue']]
    assert plan['predecessors'] == lineage and protocol.read(protocol.ROOT / tail['directory'] / 'queue-identity.json') == tail['queue']
    assert any((row['unit'] == 'trn1-smollm3-cache-ablation-full-v1-queue' for row in lineage))
    review = protocol.read(plan['independent_review'])
    assert all((review.get(name) is True for name in ('passed', 'source_only', 'independent_review', 'suitable_for_enrollment_as_independent_review', 'original_gates_preserved')))
    assert review['blockers'] == [] and review['source_author'] == protocol.SOURCE_AUTHOR
    assert isinstance(review['reviewer'], str) and review['reviewer'].strip() and (review['reviewer'] != protocol.SOURCE_AUTHOR)
    assert all((review['source_sha256'][str(protocol.ROOT / name)] == plan['source_sha256'][str(protocol.ROOT / name)] for name in protocol.source_names()))
    assert all((plan[name] is False for name in ('acquisition_replay', 'new_model_execution', 'new_compilation', 'device_execution', 'full_model_qualified', 'new_speedup_claim')))
    return plan


def replay_analysis(p, e, saved, t, check_comparison, phase):
    plan = p.guard(phase)
    context = e.verify()
    assert context['analysis_payloads'] == plan['analysis_payloads']
    p.admit_payloads(plan)
    p.activate(plan)
    import torch
    torch.set_num_threads(1)
    resource.setrlimit(resource.RLIMIT_FSIZE, (p.FILE_CAP, p.FILE_CAP))
    allrows = context['calls']
    layout_proofs = []
    for index in range(14):
        for (arm, rank) in allrows:
            row = allrows[arm, rank][index]
            (data, proof) = saved.load(row, plan)
            layout_proofs.append(dict(call=index, arm=arm, rank=rank, payload=row['payload'], **proof))
            del data
    assert len(layout_proofs) == 98
    p.once(p.DEST / 'saved-layout-admission.json', dict(passed=True, records=layout_proofs, outputs=98, all_values_match_producer_raw_digest=True, producer_metadata_unmodified=True, contiguous_saved_layout_checked_independently=True, scoring_not_started=True))
    native = saved.load_hf(plan)
    directory = p.DEST / 'analysis'
    directory.mkdir()
    hf = {key: [] for key in allrows}
    reports = []
    cache_count = stage_count = 0
    partials = []
    for spec in e.c.calls():
        index = spec['index']
        source = context['source_calls'][index]
        ids = [spec['frames'][0]['slot']] if spec['phase'] == 'prefill' else [0, 1, 2, 3]
        assert source['input_metadata'][3]['sha256'] == p.hashlib.sha256(struct.pack('<' + 'i' * len(ids), *ids)).hexdigest()
        rows = {key: values[index] for (key, values) in allrows.items()}
        payloads = {}
        for (key, row) in rows.items():
            (payloads[key], _) = saved.load(row, plan)
        frame_results = []
        for frame in spec['frames']:
            batch = ids.index(frame['slot'])
            reference = native[frame['fixture']]['logits'][frame['step']].float().reshape(-1)
            assert list(reference.shape) == [p.old.VOCAB] and bool(torch.isfinite(reference).all())
            arm_rows = {}
            for (key, data) in payloads.items():
                value = data['output'][batch, 0]
                assert value.shape == reference.shape
                result = dict(**frame, **saved.metrics(value, reference))
                hf[key].append(result)
                arm_rows[str(key)] = result
            effects = []
            for (left, right) in e.PAIRINGS:
                for rank in (0, 1):
                    a = payloads[e.armkey(left, rank)]['output'][batch, 0]
                    b = payloads[e.armkey(right, rank)]['output'][batch, 0]
                    am = saved.metrics(a, reference)
                    bm = saved.metrics(b, reference)
                    effects.append(dict(left=left, right=right, rank=rank, raw_equal=t.raw(a) == t.raw(b), metrics=saved.metrics(a, b), original_hf_before=bm, original_hf_after=am, original_hf_delta={name: am[name] - bm[name] for name in e.DELTAS}, before_failed=bool(e.failed(bm)), after_failed=bool(e.failed(am))))
            frame_results.append(dict(frame=frame, against_original_hf=arm_rows, effects=effects))
        stage_rows = []
        for arm in e.c.CPU_ARMS:
            for name in e.c.stages():
                (a, b) = (payloads[arm, rank]['taps'][name] for rank in (0, 1))
                for (rank, value) in enumerate((a, b)):
                    assert t.digest(t.raw(value)) == rows[arm, rank]['stage_raw_sha256'][name]
                stage_rows.append(dict(arm=arm, stage=name, rank_raw_equal=t.raw(a) == t.raw(b), prediction_vectors=len(spec['frames'])))
                stage_count += len(spec['frames'])
        cache_rows = []
        for (left, right) in e.PAIRINGS:
            for rank in (0, 1):
                aa = e.manifest(rows[e.armkey(left, rank)]['after'], rank)
                bb = e.manifest(rows[e.armkey(right, rank)]['after'], rank)
                for (a, b) in zip(aa, bb):
                    assert a['name'] == b['name'] and a['rank'] == b['rank'] == rank
                    assert a['file'] in plan['analysis_payloads'] and b['file'] in plan['analysis_payloads']
                    av = t.load_state(a['file'], a)
                    bv = t.load_state(b['file'], b)
                    stats = t.region_record(bv, av, spec, enforce_transition=False)
                    cache_rows.append(dict(left=left, right=right, rank=rank, name=a['name'], raw_equal=a['raw_sha256'] == b['raw_sha256'], **stats))
                    cache_count += 1
                    del av, bv
        value = dict(complete=True, call=spec, frames=frame_results, caches=cache_rows, cpu_rank_hooks=stage_rows, all14_baseline_restore_controls_previously_passed=True)
        if index < 4:
            assert value == p.read(p.old.DEST / 'analysis' / f'call-{index:02d}.json')
            partials.append(index)
        path = directory / f'call-{index:02d}.json'
        p.once(path, value)
        reports.append(dict(path=str(path), sha256=p.sha(path)))
        del payloads
        p.storage()
    summaries = []
    for ((arm, rank), rows) in hf.items():
        assert len(rows) == 32
        summaries.append(dict(arm=arm, rank=rank, passed=e.gate(rows), failed_frames=[row for row in rows if e.failed(row)], rows=rows, prefill_frame_indices=[row['index'] for row in rows if row['step'] == 0], decode_frame_indices=[row['index'] for row in rows if row['step'] > 0]))
    assert cache_count == 9408 and stage_count == 13632 and (partials == [0, 1, 2, 3])
    report = dict(complete=True, diagnostic_valid=True, logical_calls=70, logical_frames=160, physical_cpu_rank_frames=128, device_frames=96, all14_baseline_and_restore_passed=True, failed_triplet_invalidates_all_intervention_conclusions=True, criteria=p.old.CRITERIA, hf_summaries=summaries, reports=reports, cache_pairs=cache_count, cpu_rank_stage_prediction_vectors=stage_count, separate_attention_frames=[1, 4, 12, 16, 5, 31], original_failure_preserved=True, full_model_qualified=False, new_speedup_claim=False, first_internal_device_cause_proven=False, history_is_known_correct_hf=False, remaining_gates=p.read(e.c.DESIGN)['preserved_followup_gates'], partial_prefill_reports_reproduced=partials, all98_serialized_outputs_validated_before_metrics=True, acquisition_replayed=False, original_plan_sha256=e.PLAN_SHA256, plan_sha256=p.sha(p.DEST / 'plan.json'))
    check_comparison(report)
    p.once(p.DEST / 'comparison.json', report)
    p.verify_payload_identities(plan)
    p.verify()
    p.storage()
    outputs = [p.DEST / name for name in ('payload-admission.json', 'saved-layout-admission.json', 'comparison.json')]
    p.once(p.DEST / 'compare-result.json', dict(passed=True, phase=phase, plan_sha256=p.sha(p.DEST / 'plan.json'), outputs={str(path): p.sha(path) for path in outputs}, full_model_qualified=False, new_speedup_claim=False))
