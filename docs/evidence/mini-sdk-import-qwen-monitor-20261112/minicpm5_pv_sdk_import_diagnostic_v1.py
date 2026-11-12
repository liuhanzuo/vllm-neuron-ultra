"""Independent, inert SDK-import diagnostic. SOURCE CANDIDATE, not admission.

Original full PV prerequisites and original limits are retained. Exactly one
real SDK child is observed; no PV predecessor, scientific output or historical
exit is manufactured. Native imports occur only inside the released worker.
"""
import copy
import json
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import sys
import time
import uuid

from research import minicpm5_layer1_fixed_input_pv_adapter_v3 as original

auth, serial, prior, r = original.auth, original.serial, original.prior, original.r
core, require, ROOT = original.core, original.require, original.ROOT
PIPE_CAP = original.PIPE_CAP
SUMMARY_CAP = 16 * 1024
STATUS_CAP = 64 * 1024
ERROR_CAP = 8 * 1024
DIAGNOSTIC_FILES = frozenset(('admission.json', 'phase-1-owner.json', 'phase-1-summary.json', 'phase-1-exit.json'))
SPEC_SCHEMA = 'minicpm5_fixed_input_pv_sdk_import_diagnostic_spec_v1'
REVIEW_SCHEMA = 'minicpm5_fixed_input_pv_sdk_import_diagnostic_review_v1'
ADMISSION_SCHEMA = 'minicpm5_fixed_input_pv_sdk_import_diagnostic_admission_v1'
SUMMARY_SCHEMA = 'minicpm5_fixed_input_pv_sdk_import_diagnostic_summary_v1'
RELEASE_SCHEMA = 'minicpm5_fixed_input_pv_sdk_import_diagnostic_release_v1'
RUNTIME_SOURCES = tuple(str(ROOT/'research'/name) for name in (
    'minicpm5_pv_sdk_import_diagnostic_v1.py',
    'run_minicpm5_pv_sdk_import_diagnostic_v1.py',
    'minicpm5_pv_sdk_import_diagnostic_worker_v1.py'))
impose_original_limits = original.impose_original_limits
restore_descriptor_plan = original.restore_descriptor_plan
read_packet = original.read_packet
authenticate_framework = original.authenticate_framework


class DiagnosticStore(original.Store):
    """Original complete byte/allocation accounting, narrower artifact names."""
    @staticmethod
    def legal(name):
        return name in DIAGNOSTIC_FILES

    def json(self, name, value):
        raw = serial.encoded(value)
        if name == 'phase-1-summary.json':
            require(len(raw) <= SUMMARY_CAP, 'Complete diagnostic summary fits original16KiB')
        return self.write(name, raw)


def validate_spec(spec_ref, review_ref, spec, review):
    for reference in (spec_ref, review_ref): r.life.ref(reference)
    require(type(spec) is dict and set(spec) == {'schema', 'pv_spec', 'pv_review',
        'worker', 'limits', 'artifact_limits', 'source_extension'} and spec['schema'] == SPEC_SCHEMA
        and serial.exact(spec['limits'], prior.LIMITS) and serial.exact(spec['artifact_limits'], serial.CAPS),
        'Independent diagnostic spec, unchanged original limits and caps')
    for key in ('pv_spec', 'pv_review'): r.life.ref(spec[key])
    require(type(review) is dict and set(review) == {'schema', 'reviewer', 'passed', 'blockers', 'spec', 'source_extension'}
        and review['schema'] == REVIEW_SCHEMA and review['reviewer'] == '/root'
        and review['passed'] is True and review['blockers'] == [] and review['spec'] == spec_ref
        and serial.exact(review['source_extension'], spec['source_extension']), 'Independent concrete root diagnostic review')
    pins = spec['source_extension']
    require(type(pins) is dict and 0 < len(pins) <= 2048 and spec['worker'] == RUNTIME_SOURCES[2],
        'Exact independent SDK diagnostic worker')
    for path in (*RUNTIME_SOURCES, str(Path(original.__file__).resolve())):
        require(path in pins and type(pins[path]) is str and re.fullmatch('[a-f0-9]{64}', pins[path]),
            'All diagnostic executed sources and original adapter explicitly pinned')


def bind_authenticated(spec, authenticated):
    pv = authenticated['spec']; witness = authenticated['witness']
    require(spec['pv_spec'] == witness['spec'] and spec['pv_review'] == witness['review']
        and pv['worker'] == spec['worker'] and serial.exact(pv['source_extension'], spec['source_extension']),
        'Complete original PV authentication binds the same new reviewed source extension')
    out = Path(authenticated['output_root'])
    require(out.parent == ROOT/'research' and out == Path(pv['output_root'])
        and re.fullmatch('minicpm5-fixed-input-pv-sdk-import-[a-z0-9][a-z0-9-]{0,28}', out.name),
        'Fresh explicitly diagnostic output namespace within original PV namespace')
    pins = authenticated['context']['pins_plan']['source_sha256']
    require(all(pins.get(path) == sha for path, sha in spec['source_extension'].items()),
        'Whole authenticated source map includes every reviewed diagnostic extension pin')


def parse_status(raw):
    require(type(raw) is bytes and len(raw) <= STATUS_CAP, 'Bounded whole proc status')
    text = raw.decode('ascii'); result = {key: {'present': False} for key in ('VmSize', 'VmPeak', 'VmRSS', 'Threads')}
    for line in text.splitlines():
        key = line.split(':', 1)[0]
        if key not in result: continue
        require(not result[key]['present'], 'No duplicate proc status field')
        pattern = key + (r':\s+([0-9]+)\s+kB\s*' if key != 'Threads' else r':\s+([0-9]+)\s*')
        match = re.fullmatch(pattern, line)
        require(match is not None, 'Exact proc status field units and nonnegative integer')
        result[key] = dict(present=True, value=int(match[1]), unit='kB' if key != 'Threads' else 'count', raw=line)
    return result


def read_status():
    fd = os.open('/proc/self/status', os.O_RDONLY | os.O_CLOEXEC)
    try:
        parts = []; size = 0
        while True:
            raw = os.read(fd, min(65536, STATUS_CAP-size+1))
            if not raw: break
            size += len(raw); require(size <= STATUS_CAP, 'Proc status exceeds64KiB/EOF bound'); parts.append(raw)
        return parse_status(b''.join(parts))
    finally: os.close(fd)


def observe(label):
    """Actual worker only. Mock tests replace every OS/environment accessor."""
    started = time.monotonic_ns()
    result = dict(label=label, monotonic_ns=started, unix_time_ns=time.time_ns(),
        interpreter=sys.executable, resolved_interpreter=str(Path(sys.executable).resolve()),
        dont_write_bytecode=sys.dont_write_bytecode, assertions=__debug__, environment=dict(os.environ),
        limits={name: list(resource.getrlimit(kind)) for name, kind in (
            ('AS', resource.RLIMIT_AS), ('CPU', resource.RLIMIT_CPU), ('FSIZE', resource.RLIMIT_FSIZE))},
        affinity=sorted(os.sched_getaffinity(0)), nice=os.getpriority(os.PRIO_PROCESS, 0), status=read_status())
    result['observation_finished_monotonic_ns'] = time.monotonic_ns()
    return result


def exception_record(error):
    """Whole bounded exception graph and every traceback frame, no linecache IO.

Source excerpts are deliberately absent. All actual frames, messages, errno,
cause/context edges and suppression flags are retained. Overflow or a cycle
fails; no truncated result is ever labelled complete.
"""
    nodes = []; active = set(); seen = {}; frame_count = 0
    def visit(exc):
        nonlocal frame_count
        if exc is None: return None
        identity = id(exc)
        require(identity not in active, 'Exception cause/context cycle rejected')
        if identity in seen: return seen[identity]
        require(len(nodes) < 16, 'Bounded complete exception graph')
        active.add(identity); index = len(nodes); seen[identity] = index; nodes.append(None)
        message = str(exc); require(len(message.encode('utf-8')) <= ERROR_CAP, 'Whole exception message budget')
        kind = type(exc).__module__ + '.' + type(exc).__qualname__
        frames = []; tb = exc.__traceback__; tb_seen = set()
        while tb is not None:
            require(id(tb) not in tb_seen and frame_count < 128, 'Acyclic bounded complete traceback')
            tb_seen.add(id(tb)); frame_count += 1
            frames.append(dict(file=tb.tb_frame.f_code.co_filename, line=tb.tb_lineno, function=tb.tb_frame.f_code.co_name))
            tb = tb.tb_next
        traceback_text = ('Traceback (most recent call last):\n' if frames else '')
        traceback_text += ''.join('  File '+json.dumps(f['file'])+', line '+str(f['line'])+', in '+f['function']+'\n' for f in frames)
        traceback_text += kind + ': ' + message + '\n'
        errno_present = hasattr(exc, 'errno'); errno = getattr(exc, 'errno', None)
        require(errno is None or type(errno) is int or type(errno) is str, 'Actual serializable errno only')
        row = dict(type=kind, message=message, traceback_frames=frames, traceback=traceback_text,
            errno_present=errno_present, errno=errno, errno_obtained=errno_present and errno is not None,
            suppress_context=bool(exc.__suppress_context__), cause=visit(exc.__cause__), context=visit(exc.__context__))
        nodes[index] = row; active.remove(identity)
        return index
    root = visit(error)
    result = dict(root=root, nodes=nodes, complete=True, traceback_source_lines_included=False,
                  errno_inferred=False, root_cause_determined=False)
    require(len(serial.encoded(result)) <= ERROR_CAP, 'Complete exception graph exceeds reserved8KiB')
    return result


def execute_import(pins_plan, guard, *, observer=observe):
    """Original SDK import/auth/thread order; no tensor operation or model."""
    observations = []; stage = 'observe_before_import'; error = None; recording_error = None
    import_complete = False
    try:
        observations.append(observer('before_import'))
        stage = 'torch_import'
        import torch
        stage = 'sdk_repeat_kv_import'
        from neuronx_distributed_inference.modules.attention.utils import repeat_kv
        stage = 'framework_authentication'
        authenticate_framework(torch, repeat_kv, 'sdk', pins_plan['source_sha256'], guard)
        stage = 'torch_set_num_threads'
        torch.set_num_threads(serial.design.ENVIRONMENTS['sdk']['torch_threads'])
        stage = 'loaded_source_authentication'
        r.verify_loaded(pins_plan, guard)
        import_complete = True; stage = 'observe_after_import'
        observations.append(observer('after_import'))
        stage = 'complete'
    except BaseException as failure:
        try: error = exception_record(failure)
        except BaseException as rejected: recording_error = type(rejected).__name__
        try: observations.append(observer('after_failure'))
        except BaseException as rejected:
            recording_error = type(rejected).__name__
    result = dict(schema=SUMMARY_SCHEMA, phase=1, diagnostic_only=True, stage=stage,
        import_sequence_completed=import_complete, diagnostic_passed=stage == 'complete',
        observations=observations, exception=error, complete_error_record_saved=error is not None,
        error_recording_failure_type=recording_error, observation_and_error_payload_complete=recording_error is None,
        raw_capture_bytes_read=0, model_executed=False, full_model_qualified=False, new_speedup_claim=False,
        self_exit_claimed=False, root_cause_determined=False)
    if len(serial.encoded(result)) > SUMMARY_CAP:
        result = dict(schema=SUMMARY_SCHEMA, phase=1, diagnostic_only=True, stage=stage,
            diagnostic_passed=False, import_sequence_completed=import_complete,
            complete_error_record_saved=False, observation_and_error_payload_complete=False,
            error_recording_failure_type='CompleteSummaryExceeds16KiB', raw_capture_bytes_read=0,
            model_executed=False, full_model_qualified=False, new_speedup_claim=False,
            self_exit_claimed=False, root_cause_determined=False)
    return result, 0 if result['diagnostic_passed'] and result['observation_and_error_payload_complete'] else 1


def worker(admission_ref, index, worker_path):
    """Original authenticated pre-import path, specialized to SDK index1."""
    require(type(index) is int and index == 1, 'Only independent SDK diagnostic child')
    require(sys.dont_write_bytecode and __debug__ and Path(sys.executable).resolve() == Path(serial.python_for(index)).resolve(), 'Exact original interpreter/-B/assertions')
    impose_original_limits(); admission = r.sources.load(admission_ref)
    require(admission['schema'] == ADMISSION_SCHEMA, 'Exact independent diagnostic admission')
    lease = r.Lease(admission['lease']); guard = lambda: lease.check('pv.child')
    metadata = r.Metadata(guard); lease.metadata = metadata
    require(metadata.read(admission_ref) == admission, 'Stable exact admitted outer gate')
    owner = r.p.proc(os.getpid()); argv = serial.phase_argv(worker_path, admission_ref, index)
    r.life.process(owner, argv, admission['lease']['outer_owner']['pid'], admission['lease']['outer_owner']['cgroup'])
    lease.processes = [admission['lease']['outer_owner'], owner]
    gate = read_packet(guard)
    require(type(gate) is dict and set(gate) == {'schema', 'admission', 'phase', 'owner', 'source_plan', 'own_gate'}
        and gate['schema'] == RELEASE_SCHEMA and gate['admission'] == admission_ref
        and type(gate['phase']) is int and gate['phase'] == index and serial.exact(gate['owner'], owner), 'Explicit acknowledged exact child release')
    pins_plan = gate['source_plan']; witness = admission['witness']
    require(r.digest(pins_plan['source_sha256']) == witness['complete_source_map_sha256']
        and len(pins_plan['source_sha256']) == witness['complete_source_files'], 'Whole inherited source map received over direct bounded parent pipe')
    partition = r.sources.classify(pins_plan['source_sha256'], pins_plan['source_contract'], metadata.read)
    r.sources.verify_text_sources(partition, guard)
    source_receipt = metadata.read(admission['original_native_source_receipt'])
    require(partition['categories']['query_native_binary_sha256'] == source_receipt['native_sha256'],
        'Exactly the inherited authenticated nine native binary sources; no replacement or omission')
    r.sources.check_native_identities(source_receipt, partition, guard)
    r.verify_loaded(pins_plan, guard)
    plan = restore_descriptor_plan(witness, metadata.read); plan_sha = r.digest(plan)
    allowed = core.saved.validate_plan(plan); io = core.saved.StableCaptureIO(allowed, guard, filesystem=os)
    store = DiagnosticStore(admission['output_root'], guard); store.inventory()
    own_gate = gate['own_gate']
    require(own_gate is None, 'No forged earlier gate in initial controls')
    def authenticate(environment, stage, sha):
        guard(); require((environment, stage) == serial.PHASES[index] and sha == plan_sha, 'Only exact released phase/plan')
        require(r.p.proc(admission['lease']['outer_owner']['pid']) == admission['lease']['outer_owner'], 'Same real authenticated outer remains alive')
        return True
    authenticate('sdk', 'own_before', plan_sha)
    spec_ref, review_ref = admission['diagnostic_spec'], admission['diagnostic_review']
    spec, review = metadata.read(spec_ref), metadata.read(review_ref)
    validate_spec(spec_ref, review_ref, spec, review)
    require(spec['worker'] == worker_path and spec['pv_spec'] == witness['spec'] and spec['pv_review'] == witness['review']
        and all(pins_plan['source_sha256'].get(path) == sha for path, sha in spec['source_extension'].items()),
        'Original witness and entire diagnostic source extension remain bound in child')
    expected_env = r.p.environment({}, 'cpu_control', admission['lease']['attempt_id'])
    require(dict(os.environ) == expected_env, 'Exact original SDK child environment before import')
    result, code = execute_import(pins_plan, guard)
    # As originally, IO is constructed after descriptor authentication. It is
    # never used to read any tensor bytes in this diagnostic.
    require(not io._read, 'Diagnostic performs zero original capture reads')
    r.verify_loaded(pins_plan, guard); metadata.unchanged(); guard()
    store.json('phase-1-summary.json', result); store.inventory(); guard()
    signal.alarm(0)
    return code


class SingleSDKAdapter:
    """Once-only SDK lifecycle with actual parent wait/drain evidence."""
    def __init__(self, authenticated, admission_ref, admission, lease, guard, metadata, store):
        self.authenticated, self.admission_ref, self.admission = authenticated, admission_ref, admission
        self.lease, self.guard, self.metadata, self.store = lease, guard, metadata, store
        self.child = self.owner = self.argv = None
        self.state = 'new'; self.child_started = False
        self.terminal = None; self.receipt_saved = False; self.receipt_attempted = False

    def authenticate_complete_prerequisites(self):
        a = self.authenticated
        require(self.metadata.read(self.admission_ref) == self.admission and self.admission['witness'] == a['witness']
            and r.digest(a['context']['pins_plan']['source_sha256']) == a['witness']['complete_source_map_sha256']
            and r.digest(a['context']['plan']) == a['witness']['descriptor_plan_sha256']
            and self.admission['lease'] == self.lease.admission
            and r.p.proc(os.getpid()) == self.lease.admission['outer_owner'], 'Actual original authenticated outer and exclusive admission')
        self.guard(force=True)

    def start(self):
        require(self.state == 'new' and self.child is None, 'Only one actual SDK diagnostic child; no retry or resume')
        self.state = 'starting'; index = 1
        a = self.authenticated; worker_path = a['spec']['worker']; argv = serial.phase_argv(worker_path, self.admission_ref, index)
        self.argv = argv
        self.guard(force=True); r.sources.text(worker_path, a['context']['pins_plan']['source_sha256'][worker_path])
        env = r.p.environment({}, 'cpu_control', self.lease.admission['attempt_id'])
        self.child = subprocess.Popen(argv, env=env, cwd=ROOT, stdin=subprocess.PIPE, close_fds=True, start_new_session=True)
        self.child_started = True
        deadline = time.monotonic()+10
        while True:
            self.guard(); current = r.p.proc(self.child.pid)
            if current is not None and current['argv'] == argv:
                r.life.process(current, argv, self.lease.admission['outer_owner']['pid'], self.lease.admission['outer_owner']['cgroup'])
                require(str(Path('/proc/%d/exe' % current['pid']).resolve()) == str(Path(serial.python_for(index)).resolve()), 'Actual child executable matches original interpreter')
                self.owner = current; break
            require(self.child.poll() is None and time.monotonic() < deadline, 'Bounded original child exec/acknowledgement'); time.sleep(.01)
        self.store.json('phase-1-owner.json', current)
        payload = dict(schema=RELEASE_SCHEMA, admission=self.admission_ref, phase=index,
            owner=current, source_plan={key: a['context']['pins_plan'][key] for key in ('source_contract', 'source_sha256')}, own_gate=None)
        raw = serial.encoded(payload); require(len(raw) <= PIPE_CAP, 'Complete inherited source plan fits original16MiB input pipe cap')
        require(r.p.proc(current['pid']) == current, 'Exact acknowledged child before explicit release')
        try: require(self.child.stdin.write(raw) == len(raw), 'Whole bounded source/phase release'); self.child.stdin.flush()
        finally: self.child.stdin.close()
        self.state = 'released'
        return copy.deepcopy(current), argv

    def _record_drained(self, returncode, *, cleanup):
        require(type(returncode) is int and self.child is not None, 'Actual child wait returncode required')
        if self.owner is not None: r.p.gone(self.owner)
        require(not r.group_members(self.child.pid), 'Entire original owned process group empty')
        self.terminal = dict(schema='minicpm5_pv_sdk_import_diagnostic_actual_child_exit_v1', phase=1,
            owner=copy.deepcopy(self.owner), argv=copy.deepcopy(self.argv), child_pid=self.child.pid,
            returncode=returncode, parent_observed_wait=True, process_group_empty=True,
            exact_owner_gone=True if self.owner is not None else None,
            owner_acknowledged=self.owner is not None, self_exit_claimed=False, cleanup_after_failure=cleanup)
        self.child = None; self.state = 'drained'
        self._persist_terminal()
        return self.terminal

    def _persist_terminal(self):
        require(self.terminal is not None and self.state == 'drained', 'Physical drain before terminal persistence')
        if self.receipt_attempted: return
        self.receipt_attempted = True
        try:
            self.store.json('phase-1-exit.json', self.terminal); self.receipt_saved = True
        except BaseException as error:
            # Sticky guard or IO failure never authorizes a weaker writer.
            # The true wait/drain facts remain visible, explicitly NOT saved.
            message = dict(event='sdk_import_diagnostic_exit_receipt_write_failed',
                actual_parent_observed_terminal=self.terminal, exit_receipt_saved=False,
                reporting_error_type=type(error).__name__, diagnostic_passed=False)
            raw = serial.encoded(message)
            require(len(raw) <= SUMMARY_CAP, 'Bounded actual terminal fallback')
            print(raw.decode(), file=sys.stderr, flush=True)
            raise

    def finish(self, owner, argv):
        require(self.state == 'released' and owner == self.owner and argv == self.argv and self.child is not None, 'Exact owned active SDK child')
        deadline = time.monotonic()+prior.LIMITS['wall_seconds']
        while self.child.poll() is None:
            self.guard()
            current = r.p.proc(self.child.pid)
            if current != owner:
                require(current is None or current == dict(owner, argv=[]),
                    'Unexpected child identity is never downgraded to terminal')
                require(self.child.poll() is not None, 'Exact child identity while live')
                break
            require(time.monotonic() < deadline, 'Original whole-phase wall bound'); time.sleep(.1)
        returncode = self.child.wait(timeout=10)
        r.drain_group(self.child, owner, r.p)
        return self._record_drained(returncode, cleanup=False)

    def drain_owned(self):
        """Always physical cleanup first; original lock remains held on doubt."""
        if self.child is not None:
            if self.child.stdin is not None and not self.child.stdin.closed: self.child.stdin.close()
            r.drain_group(self.child, self.owner, r.p)
            returncode = self.child.wait(timeout=10)
            self._record_drained(returncode, cleanup=True)
        return dict(physically_drained=True, no_services_started=True,
            exit_receipt_saved=self.receipt_saved, child_started=self.child_started)


def run(spec_ref, review_ref):
    """Future explicit root-reviewed execution only; no implicit admission."""
    require(sys.dont_write_bytecode and __debug__ and Path(sys.executable).resolve() == Path(serial.python_for(0)).resolve(),
        'Original native outer interpreter/-B/assertions for source reconstruction')
    r.p.user_bus(); fd, owner, first = prior.audited.support.acquire()
    binding = dict(outer_owner=owner, lock_identity=r.ORIGINAL_LOCK, retained_fd=fd, lock_policy=r.LOCK_POLICY,
        initial_descriptor_proof=first, unit='trn1-mini-sdk-import-diagnostic-outer', attempt_id=uuid.uuid4().hex)
    lease = r.Lease(binding); ports = None
    try:
        impose_original_limits(); guard = original.PVGuard(lease); metadata = r.Metadata(guard); lease.metadata = metadata
        spec, review = metadata.read(spec_ref), metadata.read(review_ref)
        validate_spec(spec_ref, review_ref, spec, review)
        authenticated = auth.authenticate(spec['pv_spec'], spec['pv_review'], lease=lease, metadata=metadata, guard=guard)
        bind_authenticated(spec, authenticated)
        context = authenticated['context']
        source_subset = {key: context['pins_plan'][key] for key in ('source_contract', 'source_sha256')}
        require(len(serial.encoded(source_subset))+65536 <= PIPE_CAP, 'Whole source gate fits before any child spawn')
        store = DiagnosticStore(authenticated['output_root'], guard); store.create()
        model = metadata.read(context['original_spec']['model_exit'])
        admission = dict(schema=ADMISSION_SCHEMA, witness=authenticated['witness'],
            lease=binding, output_root=authenticated['output_root'], original_native_source_receipt=model['outputs']['source_authentication'],
            diagnostic_spec=spec_ref, diagnostic_review=review_ref)
        admission_ref = original.short(store.json('admission.json', admission))
        ports = SingleSDKAdapter(authenticated, admission_ref, admission, lease, guard, metadata, store)
        ports.authenticate_complete_prerequisites()
        child_owner, argv = ports.start(); terminal = ports.finish(child_owner, argv)
        if terminal['returncode'] != 0:
            return dict(terminal=terminal, exit_receipt=original.short(store.reference('phase-1-exit.json')),
                diagnostic_passed=False, full_model_qualified=False, new_speedup_claim=False)
        summary, summary_ref = store.load('phase-1-summary.json')
        require(summary['schema'] == SUMMARY_SCHEMA and summary['diagnostic_passed'] is True
            and summary['stage'] == 'complete' and summary['observation_and_error_payload_complete'] is True,
            'Actual successful child plus complete independent diagnostic observation')
        require(set(store.inventory()['names']) == DIAGNOSTIC_FILES, 'Exactly four diagnostic artifacts, no scientific result')
        metadata.unchanged(); r.verify_loaded(context['pins_plan'], guard); guard(force=True)
        return dict(terminal=terminal, summary=original.short(summary_ref), diagnostic_passed=True,
            full_model_qualified=False, new_speedup_claim=False)
    finally:
        signal.alarm(0); warned = False; persistence_failed = False
        while True:
            try:
                proof = ports.drain_owned() if ports is not None else dict(physically_drained=True, no_children_started=True)
                break
            except BaseException as error:
                if ports is not None and ports.state == 'drained':
                    persistence_failed = True
                if not warned:
                    print(json.dumps(dict(event='sdk_import_diagnostic_cleanup_pending_original_lock_retained',
                        error_type=type(error).__name__)), file=sys.stderr, flush=True)
                    warned = True
                time.sleep(.5)
        prior.audited.support.release_after_physical_drain(fd, proof)
        require(not persistence_failed, 'True child exit observed but diagnostic receipt could not be saved')
