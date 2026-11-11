"""Narrow actual IO/process adapter for the fixed five-stage PV core.

Inert on import. No services or replacement locks. The already admitted outer
owns its original CLOEXEC descriptor throughout. This adapter only creates
five finite direct children, each in its original interpreter, and reuses the
existing exact process/group drain and complete source/metadata validators.
No CLI or current spec/catalog/terminal generation happens at import time.
"""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import select
import signal
import stat
import subprocess
import sys
import time
import uuid

from research import minicpm5_layer1_fixed_input_pv_prerequisites_v2 as auth

serial, prior, r = auth.serial, auth.prior, auth.r
core, require = serial.core, serial.require
ROOT = auth.ROOT
REPEAT_SOURCES = {
    'native': (str(ROOT/'research'/'transformers-reference'/'src'/'transformers'/'models'/'llama'/'modeling_llama.py'),
        '363df58c077a8fcfd3593b2817538e53f90eeab32294d3c6bbb7241f094669de', 'transformers.models.llama.modeling_llama'),
    'sdk': ('/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/neuronx_distributed_inference/modules/attention/utils.py',
        '282142ea4a2a761e4287d8d6962f7e11d8ca8e57261feb74c038535d13319016', 'neuronx_distributed_inference.modules.attention.utils'),
}
PIPE_CAP = core.saved.MAX_METADATA_BYTES  # Input only; no second on-disk source map.
RUNTIME_METADATA_CAP = 79008  # 8MiB minus frozen conservative event envelopes and five16KiB summaries.
RUNTIME = {'admission.json', 'own-gate.json', 'final.json'} | {
    'phase-%d-%s.json' % (index, suffix) for index in range(5) for suffix in ('owner', 'exit', 'audit', 'summary')}


def ref(raw, path): return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
def short(value): return {key: value[key] for key in ('path', 'sha256')}


class Store:
    """One flat exclusive tree; all allocation/JSON/raw bytes counted globally.

    No per-phase reset of aggregate budgets. No temp files or directory-per-
    tensor expansion. A failed partial output remains immutable evidence.
    """
    def __init__(self, root, guard):
        self.root, self.guard = Path(root), guard; self.events = {}; self.rows = {}; self.directory_identity = None
        self.logical_metadata = self.logical_raw = self.allocated = 0; self.last = None

    def _directory(self):
        for parent in reversed((self.root, *self.root.parents)):
            require(stat.S_ISDIR(parent.lstat().st_mode), 'Every output parent is a real non-symlink directory')
        st = self.root.lstat(); ident = (st.st_dev, st.st_ino, st.st_uid)
        require(st.st_uid == 1000 and (self.directory_identity is None or ident == self.directory_identity), 'Exact owned output directory inode')
        self.directory_identity = ident
        return st

    def create(self):
        self.guard(); require(not self.root.exists(), 'Never replace or reuse an output tree')
        self.root.mkdir(mode=0o700); self._directory(); self.inventory(); self.guard()

    @staticmethod
    def legal(name):
        return name in RUNTIME or re.fullmatch(r'[0-4]-r-[a-f0-9]{64}\.bin', name) or re.fullmatch(r'[0-4]-e-[0-9]{4}\.json', name)

    @staticmethod
    def identity(st):
        return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_blocks)

    def inventory(self):
        self.guard(); directory = self._directory(); rows = {}; metadata = raw = 0; allocated = directory.st_blocks*512
        for entry in self.root.iterdir():
            require(self.legal(entry.name), 'No extra artifact, temp, log, symlink or nested output directory')
            st = entry.lstat()
            require(stat.S_ISREG(st.st_mode) and st.st_uid == 1000 and st.st_nlink == 1
                and st.st_size <= serial.CAPS['one_file_bytes'], 'Singleton regular bounded whole output file')
            ident = self.identity(st)
            require(entry.name not in self.rows or ident == self.rows[entry.name], 'Every previously admitted output inode/content identity remains unchanged')
            rows[entry.name] = ident; allocated += st.st_blocks*512
            if entry.suffix == '.bin': raw += st.st_size
            else: metadata += st.st_size
        require(set(self.rows) <= set(rows), 'No removal of any already observed artifact')
        runtime_bytes = sum(row[2] for name, row in rows.items() if name in RUNTIME and not name.endswith('-summary.json'))
        require(runtime_bytes <= RUNTIME_METADATA_CAP, 'Finite runtime metadata shares original8MiB cap without duplicating whole source maps or fingerprints')
        require(metadata <= serial.CAPS['metadata_bytes'] and raw <= serial.CAPS['raw_bytes']
            and allocated <= serial.CAPS['allocated_bytes'], 'Global actual bytes/allocation stay within8MiB/32MiB caps')
        self.rows, self.logical_metadata, self.logical_raw, self.allocated = rows, metadata, raw, allocated
        self.last = time.monotonic(); self.guard()
        return dict(names=sorted(rows), identities=copy.deepcopy(rows), metadata_bytes=metadata, raw_bytes=raw,
                    allocated_bytes=allocated, directory_blocks_bytes=directory.st_blocks*512)

    def path_for(self, index, key):
        require(type(index) is int and 0 <= index < 5 and type(key) is str and len(key) <= 240, 'Bounded exact scientific key')
        return str(self.root/('%d-r-%s.bin' % (index, core.sha(key.encode()))))

    def write(self, name, raw):
        require(self.legal(name) and type(raw) is bytes and len(raw) <= serial.CAPS['one_file_bytes'], 'One finite complete exclusive output')
        self.guard()
        if self.last is None or time.monotonic()-self.last >= 1: self.inventory()
        self._directory(); path = self.root/name
        require(name not in self.rows and not path.exists(), 'No output overwrite, resume or retry')
        require((self.logical_raw if name.endswith('.bin') else self.logical_metadata)+len(raw)
            <= serial.CAPS['raw_bytes' if name.endswith('.bin') else 'metadata_bytes'], 'Reserve complete payload before writing')
        if name in RUNTIME and not name.endswith('-summary.json'):
            require(sum(row[2] for existing, row in self.rows.items() if existing in RUNTIME and not existing.endswith('-summary.json'))+len(raw)
                <= RUNTIME_METADATA_CAP, 'Whole runtime metadata reserved within frozen conservative aggregate envelope')
        # Conservative one-file allocation reservation, then actual st_blocks
        # and directory allocation are checked again before this write returns.
        require(self.allocated+((len(raw)+4095)//4096)*4096+65536 <= serial.CAPS['allocated_bytes'], 'Conservative actual-allocation room before a new artifact')
        fd = os.open(path, os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW|os.O_CLOEXEC, 0o600)
        try:
            position = 0
            while position < len(raw):
                self.guard(); count = os.write(fd, raw[position:position+65536]); require(count > 0, 'Complete output write progress'); position += count
            os.fsync(fd); st = os.fstat(fd)
            require(stat.S_ISREG(st.st_mode) and st.st_nlink == 1 and st.st_size == len(raw), 'Entire exclusive output written')
        finally: os.close(fd)
        directory = os.open(self.root, os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC)
        try: os.fsync(directory)
        finally: os.close(directory)
        result = ref(raw, path); require(self.read(result) == raw, 'Whole exclusive raw readback before returning reference')
        self.rows[name] = self.identity(st)
        if name.endswith('.bin'): self.logical_raw += len(raw)
        else: self.logical_metadata += len(raw)
        now_directory = self._directory()
        # A full inventory also accounts for this directory's changed extent.
        self.allocated = sum(row[5]*512 for row in self.rows.values())+now_directory.st_blocks*512
        require(self.allocated <= serial.CAPS['allocated_bytes'], 'Measured total allocated blocks including directory')
        self.guard(); return result

    def read(self, reference):
        require(type(reference) is dict and set(reference) == {'path', 'sha256', 'bytes'}
            and Path(reference['path']).parent == self.root and self.legal(Path(reference['path']).name)
            and type(reference['bytes']) is int and 0 <= reference['bytes'] <= serial.CAPS['one_file_bytes'], 'Whole exact local output reference')
        self.guard(); self._directory(); path = Path(reference['path']); before = path.lstat()
        require(stat.S_ISREG(before.st_mode) and before.st_uid == 1000 and before.st_nlink == 1
            and before.st_size == reference['bytes'], 'Stable original singleton whole output')
        fd = os.open(path, os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK|os.O_CLOEXEC)
        try:
            require(self.identity(os.fstat(fd)) == self.identity(before), 'Exact opened output inode')
            blocks = []; size = 0
            while True:
                self.guard(); block = os.read(fd, min(65536, reference['bytes']-size+1))
                if not block: break
                blocks.append(block); size += len(block); require(size <= reference['bytes'], 'Whole output EOF bound')
            raw = b''.join(blocks)
            require(size == reference['bytes'] and core.sha(raw) == reference['sha256']
                and self.identity(os.fstat(fd)) == self.identity(before) == self.identity(path.lstat()), 'Every raw byte/EOF/SHA and stable output identity')
        finally: os.close(fd)
        self._directory(); self.guard(); return raw

    def json(self, name, value): return self.write(name, serial.encoded(value))
    def raw_sink(self, index):
        return lambda key, raw, metadata: self.write(Path(self.path_for(index, key)).name, raw)
    def event_sink(self, index):
        require(index not in self.events, 'One event stream per child phase'); self.events[index] = 0
        def emit(value):
            number = self.events[index]; require(number < serial.EVENT_COUNTS[index], 'Finite exact event number')
            result = self.json('%d-e-%04d.json' % (index, number), value); self.events[index] += 1; return result
        return emit
    def reference(self, name):
        self.guard(); self._directory(); path = self.root/name; st = path.lstat()
        require(stat.S_ISREG(st.st_mode) and st.st_uid == 1000 and st.st_nlink == 1
            and 0 <= st.st_size <= serial.CAPS['one_file_bytes'], 'Whole bounded artifact for first reference')
        fd = os.open(path, os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK|os.O_CLOEXEC)
        try:
            require(self.identity(os.fstat(fd)) == self.identity(st), 'Exact first-reference inode')
            parts = []; size = 0
            while True:
                self.guard(); raw = os.read(fd, min(65536, st.st_size-size+1))
                if not raw: break
                size += len(raw); require(size <= st.st_size, 'Bounded first-reference whole read'); parts.append(raw)
            raw = b''.join(parts)
            require(size == st.st_size and self.identity(os.fstat(fd)) == self.identity(st) == self.identity(path.lstat()), 'Complete stable first reference')
        finally: os.close(fd)
        return ref(raw, path)
    def load(self, name):
        reference = self.reference(name); return json.loads(self.read(reference)), reference


def restore_descriptor_plan(witness, read):
    """Reassemble original six existing reports without making a new plan file."""
    summary = read(witness['comparison_summary']); plan = copy.deepcopy(summary['descriptor_plan_header']); calls = []
    require(len(witness['comparison_reports']) == 6, 'Every original descriptor report')
    for expected_call, reference in zip(serial.design.CALLS, witness['comparison_reports']):
        value = read(reference)
        require(value['call'] == expected_call and value['descriptor_plan_sha256'] == witness['descriptor_plan_sha256'], 'Exact original call descriptor binding')
        calls.append(value['complete_call_capture_descriptors'])
    plan['calls'] = calls
    require(r.digest(plan) == witness['descriptor_plan_sha256'], 'Whole original1284 descriptor plan reconstructed')
    core.saved.validate_plan(plan); return plan


def read_packet(guard, *, fd=0):
    require(stat.S_ISFIFO(os.fstat(fd).st_mode), 'Explicit parent-only stdin release/input pipe')
    deadline = time.monotonic()+60; parts = []; size = 0
    while True:
        guard(); remaining = deadline-time.monotonic(); require(remaining > 0, 'Bounded source/phase gate deadline')
        ready, _, _ = select.select([fd], [], [], min(.1, remaining))
        if not ready: continue
        raw = os.read(fd, min(65536, PIPE_CAP-size+1))
        if not raw: break
        size += len(raw); require(size <= PIPE_CAP, 'Entire source/input gate remains within original16MiB input metadata cap'); parts.append(raw)
    raw = b''.join(parts); value = json.loads(raw)
    require(serial.encoded(value) == raw, 'Exactly one canonical complete parent release packet followed by EOF')
    return value


def impose_original_limits():
    limits = prior.LIMITS
    os.sched_setaffinity(0, {limits['cpu']})
    if os.getpriority(os.PRIO_PROCESS, 0) < limits['nice']: os.nice(limits['nice']-os.getpriority(os.PRIO_PROCESS, 0))
    resource.setrlimit(resource.RLIMIT_AS, (limits['address_space_bytes'],)*2)
    resource.setrlimit(resource.RLIMIT_CPU, (limits['cpu_seconds'], limits['hard_cpu_seconds']))
    resource.setrlimit(resource.RLIMIT_FSIZE, (limits['output_file_bytes'],)*2)
    signal.signal(signal.SIGALRM, lambda *args: (_ for _ in ()).throw(TimeoutError('Original finite PV worker wall bound')))
    signal.alarm(limits['wall_seconds'])


def authenticate_framework(torch, repeat_kv, environment, pins, guard):
    path, sha, module = REPEAT_SOURCES[environment]
    require(pins.get(path) == sha and repeat_kv.__module__ == module and repeat_kv.__name__ == 'repeat_kv'
        and str(Path(repeat_kv.__code__.co_filename).resolve()) == path, 'Actual original repeat_kv callable, file, module and complete source hash')
    source = r.sources.text(path, sha); nodes = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == 'repeat_kv']
    require(len(nodes) == 1 and repeat_kv.__code__.co_firstlineno == nodes[0].lineno, 'Original callable starts at its authenticated source definition')
    torch_path = str(Path(torch.__file__).resolve())
    require(torch_path in pins, 'Actually imported original framework entry is in complete authenticated closure')
    r.sources.text(torch_path, pins[torch_path]); guard()


def worker(admission_ref, index, worker_path):
    """Actual original-interpreter child entry; never used in source tests."""
    require(sys.dont_write_bytecode and __debug__ and Path(sys.executable).resolve() == Path(serial.python_for(index)).resolve(), 'Exact original interpreter/-B/assertions')
    impose_original_limits(); admission = r.sources.load(admission_ref)
    require(admission['schema'] == 'minicpm5_fixed_input_pv_serial_admission_v1', 'Exact PV admission')
    lease = r.Lease(admission['lease']); guard = lambda: lease.check('pv.child')
    metadata = r.Metadata(guard); lease.metadata = metadata
    require(metadata.read(admission_ref) == admission, 'Stable exact admitted outer gate')
    owner = r.p.proc(os.getpid()); argv = serial.phase_argv(worker_path, admission_ref, index)
    r.life.process(owner, argv, admission['lease']['outer_owner']['pid'], admission['lease']['outer_owner']['cgroup'])
    lease.processes = [admission['lease']['outer_owner'], owner]
    gate = read_packet(guard)
    require(type(gate) is dict and set(gate) == {'schema', 'admission', 'phase', 'owner', 'source_plan', 'own_gate'}
        and gate['schema'] == 'minicpm5_fixed_input_pv_phase_release_v1' and gate['admission'] == admission_ref
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
    store = Store(admission['output_root'], guard); store.inventory()
    own_gate = gate['own_gate']
    if index < 2: require(own_gate is None, 'No forged earlier gate in initial controls')
    else:
        core.saved.reference(own_gate); proof = metadata.read(own_gate)
        require(proof['schema'] == 'minicpm5_fixed_input_pv_own_gate_v1' and proof['admission'] == admission_ref
            and proof['descriptor_plan_sha256'] == plan_sha and proof['equalities'] == 60
            and proof['parent_owner'] == admission['lease']['outer_owner'], 'Parent-bound exact60 independently verified own-output gate')
        for phase, row in enumerate(proof['completed']):
            require(phase < 2, 'Exactly two own completions')
            serial.actual_exit(row['terminal'], row['terminal']['owner'], serial.phase_argv(worker_path, admission_ref, phase), phase)
            r.p.gone(row['terminal']['owner'])
        require(len(proof['completed']) == 2, 'Both true completed own workers')
    def authenticate(environment, stage, sha):
        guard(); require((environment, stage) == serial.PHASES[index] and sha == plan_sha, 'Only exact released phase/plan')
        require(r.p.proc(admission['lease']['outer_owner']['pid']) == admission['lease']['outer_owner'], 'Same real authenticated outer remains alive')
        return True
    def variants(sha):
        guard(); require(index >= 2 and sha == plan_sha and own_gate is not None
            and metadata.read(own_gate) == proof, 'Exact parent-verified gate stable throughout variants'); return True
    callbacks = dict(read_bytes=io.read_bytes, write_raw=store.raw_sink(index), write_event=store.event_sink(index),
        guard=guard, authenticate_stage=authenticate, authorize_variants=variants)
    if index == 4: result = serial.leaf.run_exact_reference(plan, plan_sha, **callbacks)
    else:
        # Import only after real outer release and complete source/descriptor
        # authentication; each process executes exactly one original bundle.
        if index in (0, 2):
            sys.path.insert(0, str(ROOT/'research'/'transformers-reference'/'src'))
            import torch
            from transformers.models.llama.modeling_llama import repeat_kv
        else:
            import torch
            from neuronx_distributed_inference.modules.attention.utils import repeat_kv
        environment, stage = serial.PHASES[index]
        authenticate_framework(torch, repeat_kv, environment, pins_plan['source_sha256'], guard)
        torch.set_num_threads(serial.design.ENVIRONMENTS[environment]['torch_threads'])
        r.verify_loaded(pins_plan, guard)
        with torch.inference_mode(): result = serial.leaf.PVLeaf(torch, environment, repeat_kv=repeat_kv, **callbacks).run(plan, plan_sha, stage)
    r.verify_loaded(pins_plan, guard); metadata.unchanged(); guard()
    store.json('phase-%d-summary.json' % index, result); store.inventory(); guard()
    # Parent alone records actual wait/reap/exit/drain. No child exit claims.
    signal.alarm(0)


class Adapter:
    """Five direct children under the already admitted original outer."""
    def __init__(self, authenticated, admission_ref, admission, lease, guard, metadata, store):
        self.authenticated, self.admission_ref, self.admission = authenticated, admission_ref, admission
        self.lease, self.guard, self.metadata, self.store = lease, guard, metadata, store
        self.child = self.owner = self.index = None; self.audits = {}; self.readback = {}; self.last_exit = None

    def authenticate_complete_prerequisites(self):
        # This Adapter cannot manufacture authentication by setting a flag:
        # actual outer must construct it from authenticate() and bind the full
        # reconstructed context/plan to the exclusive admitted witness bytes.
        a = self.authenticated
        require(self.metadata.read(self.admission_ref) == self.admission and self.admission['witness'] == a['witness']
            and r.digest(a['context']['pins_plan']['source_sha256']) == a['witness']['complete_source_map_sha256']
            and r.digest(a['context']['plan']) == a['witness']['descriptor_plan_sha256']
            and self.admission['lease'] == self.lease.admission
            and r.p.proc(os.getpid()) == self.lease.admission['outer_owner'], 'Actual original authenticated outer and exclusive admission')
        self.guard(force=True)

    def start(self, index, own_gate):
        require(self.child is None and (index == 0 or self.last_exit == index-1), 'Only exact next direct child after previous real drain')
        a = self.authenticated; worker_path = a['spec']['worker']; argv = serial.phase_argv(worker_path, self.admission_ref, index)
        self.guard(force=True); r.sources.text(worker_path, a['context']['pins_plan']['source_sha256'][worker_path])
        env = r.p.environment({}, 'cpu_control' if index in (1, 3) else 'native', self.lease.admission['attempt_id'])
        self.child = subprocess.Popen(argv, env=env, cwd=ROOT, stdin=subprocess.PIPE, close_fds=True, start_new_session=True)
        self.index = index; deadline = time.monotonic()+10
        while True:
            self.guard(); current = r.p.proc(self.child.pid)
            if current is not None and current['argv'] == argv:
                r.life.process(current, argv, self.lease.admission['outer_owner']['pid'], self.lease.admission['outer_owner']['cgroup'])
                require(str(Path('/proc/%d/exe' % current['pid']).resolve()) == str(Path(serial.python_for(index)).resolve()), 'Actual child executable matches original interpreter')
                self.owner = current; break
            require(self.child.poll() is None and time.monotonic() < deadline, 'Bounded original child exec/acknowledgement'); time.sleep(.01)
        self.store.json('phase-%d-owner.json' % index, current)
        payload = dict(schema='minicpm5_fixed_input_pv_phase_release_v1', admission=self.admission_ref, phase=index,
            owner=current, source_plan={key: a['context']['pins_plan'][key] for key in ('source_contract', 'source_sha256')}, own_gate=own_gate)
        raw = serial.encoded(payload); require(len(raw) <= PIPE_CAP, 'Complete inherited source plan fits original16MiB input pipe cap')
        require(r.p.proc(current['pid']) == current, 'Exact acknowledged child before explicit release')
        # Closing this pipe supplies the explicit EOF gate. Original lock fd
        # remains parent-only/CLOEXEC and is never passed to the child.
        try: require(self.child.stdin.write(raw) == len(raw), 'Whole bounded source/phase release'); self.child.stdin.flush()
        finally: self.child.stdin.close()
        return copy.deepcopy(current), argv

    def finish(self, index, owner, argv):
        require(index == self.index and owner == self.owner and self.child is not None, 'Exact owned active phase')
        deadline = time.monotonic()+prior.LIMITS['wall_seconds']
        while self.child.poll() is None:
            self.guard(); require(r.p.proc(self.child.pid) == owner, 'Exact child identity while live')
            require(time.monotonic() < deadline, 'Original whole-phase wall bound'); time.sleep(.1)
        returncode = self.child.wait(timeout=10)
        r.drain_group(self.child, owner, r.p); r.p.gone(owner)
        require(not r.group_members(self.child.pid), 'Entire original owned process group empty')
        terminal = dict(phase=index, owner=owner, argv=argv, returncode=returncode, parent_observed_wait=True,
            process_group_empty=True, exact_owner_gone=True, self_exit_claimed=False)
        self.child = self.owner = self.index = None; self.last_exit = index
        self.store.json('phase-%d-exit.json' % index, terminal); return terminal

    def drain_owned(self):
        """Call on ANY exception. Failure propagates so outer retains its lease."""
        if self.child is not None:
            r.drain_group(self.child, self.owner, r.p)
            if self.owner is not None: r.p.gone(self.owner)
            require(not r.group_members(self.child.pid), 'Entire failed owned process group really drained')
            self.child = self.owner = self.index = None
        return dict(physically_drained=True, no_services_started=True)

    def audit_phase(self, index, terminal):
        self.guard(force=True); self.store.inventory(); plan = self.authenticated['context']['plan']
        allowed = core.saved.validate_plan(plan); io = core.saved.StableCaptureIO(allowed, self.guard, filesystem=os)
        def event_at(position): return self.store.load('%d-e-%04d.json' % (index, position))
        summary, summary_ref = self.store.load('phase-%d-summary.json' % index)
        verifier = serial.IndependentOutputs(plan, index, read_input=io.read_bytes, read_output=self.store.read,
            event_at=event_at, guard=self.guard, path_for=lambda key: self.store.path_for(index, key))
        result = verifier.verify(summary); result.update(summary=summary_ref, terminal_sha256=r.digest(terminal),
            independent_original_files_read=len(io._read), independent_original_bytes_read=sum(allowed[path]['raw_bytes'] for path in io._read))
        wanted = {Path(row['path']).name for row in [*verifier.raw_refs.values(), *verifier.event_refs]}
        actual = {name for name in self.store.inventory()['names'] if name.startswith(str(index)+'-')}
        require(actual == wanted, 'No omitted/extra/duplicate raw or event output in the complete phase')
        self.readback[index] = [*verifier.raw_refs.values(), *verifier.event_refs, summary_ref]
        self.audits[index] = copy.deepcopy(result); self.store.json('phase-%d-audit.json' % index, result)
        return result

    def publish_own_gate(self, completed):
        require(len(completed) == 2 and self.last_exit == 1 and self.child is None, 'Both genuine own workers already exited/drained')
        value = dict(schema='minicpm5_fixed_input_pv_own_gate_v1', admission=self.admission_ref,
            descriptor_plan_sha256=self.admission['witness']['descriptor_plan_sha256'], parent_owner=self.lease.admission['outer_owner'],
            equalities=60, completed=completed, full_model_qualified=False, new_speedup_claim=False)
        return short(self.store.json('own-gate.json', value))

    def revalidate_own_gate(self, reference):
        value = self.metadata.read(reference)
        require(value['admission'] == self.admission_ref and value['equalities'] == 60
            and [row['audit'] for row in value['completed']] == [self.audits[0], self.audits[1]], 'Exact immutable independently checked60 gate')
        for index, row in enumerate(value['completed']):
            serial.actual_exit(row['terminal'], row['terminal']['owner'], serial.phase_argv(self.authenticated['spec']['worker'], self.admission_ref, index), index)
            r.p.gone(row['terminal']['owner'])
            for artifact in self.readback[index]: self.store.read(artifact)
        self.guard(force=True)

    def final_audit(self, completed, own_gate):
        require(self.child is None and self.last_exit == 4 and set(self.audits) == set(range(5)), 'All five actual children complete')
        for index, row in enumerate(completed):
            serial.actual_exit(row['terminal'], row['terminal']['owner'], serial.phase_argv(self.authenticated['spec']['worker'], self.admission_ref, index), index)
            require(row['audit'] == self.audits[index], 'Every actual child bound to its independent all-word audit')
            r.p.gone(row['terminal']['owner'])
            for artifact in self.readback[index]: self.store.read(artifact)
        require(all(serial.exact(self.audits[before]['runtime_fingerprint_sha256'], self.audits[after]['runtime_fingerprint_sha256'])
            for before, after in ((0, 2), (1, 3))), 'Original environment bundle unchanged across own-before and variants/own-after workers')
        expected_names = {Path(ref['path']).name for refs in self.readback.values() for ref in refs} | {
            'admission.json', 'own-gate.json'} | {'phase-%d-%s.json' % (i, suffix) for i in range(5) for suffix in ('owner', 'exit', 'audit')}
        usage = self.store.inventory(); require(set(usage['names']) == expected_names, 'Every artifact accounted, no ignored extra files')
        context = self.authenticated['context']; self.metadata.unchanged()
        r.sources.verify_text_sources(context['partition'], self.guard)
        r.sources.check_native_identities(context['records']['source_authentication'], context['model_partition'], self.guard)
        r.verify_loaded(context['pins_plan'], self.guard)
        authority = r.current_authority.Authority(r, self.lease)
        after = prior.audited.pair(context['scope'], self.guard, authority)
        require(after == self.authenticated['before_physical']
            and context['owned'].verify_drained() == context['model_drain']
            and r.completion.latest.require_latest_terminal(context['current_view'], self.metadata.read, self.guard) == context['freshness'],
            'Original full physical scope/current catalog remains unchanged through complete five-phase replay')
        result = dict(schema='minicpm5_fixed_input_pv_serial_complete_v1', complete=True, admission=self.admission_ref,
            own_gate=own_gate, completed=completed, descriptor_plan_sha256=self.admission['witness']['descriptor_plan_sha256'],
            raw_files=sum(row['raw_files'] for row in self.audits.values()), raw_bytes=sum(row['raw_bytes'] for row in self.audits.values()),
            initial_own_equalities=sum(self.audits[i]['own_equalities'] for i in (0, 1)), final_own_equalities=sum(self.audits[i]['own_equalities'] for i in (2, 3)),
            all_five_actual_children_exited_and_drained=True, every_output_reread_after_all_children=True,
            original_sdk_model_exit_code=1, actual_outer_tool_exit_verified=False, external_post_exit_audit_still_required=True,
            self_exit_claimed=False, causal_operator_proven=False, full_model_qualified=False, new_speedup_claim=False,
            usage_before_final_summary={k: v for k, v in usage.items() if k not in ('names', 'identities')})
        self.store.json('final.json', result); self.store.inventory(); return result


class PVGuard(prior.previous.ComparisonGuard):
    def __call__(self, *, force=False):
        super().__call__(force=force)
        require(self.storage['available_bytes']['root'] >= 28*2**30+serial.CAPS['allocated_bytes'],
            'Original28GiB root floor plus entire32MiB PV output reservation')


def run(spec_ref, review_ref):
    """Actual outer entry, only after separate concrete root review/admission.

    No service creation. One retained original descriptor and five original-
    interpreter children. All original process/resource limits apply to this
    outer as a whole; each child separately applies those same finite limits.
    """
    require(sys.dont_write_bytecode and __debug__ and Path(sys.executable).resolve() == Path(serial.python_for(0)).resolve(),
        'Original native outer interpreter/-B/assertions for source reconstruction')
    r.p.user_bus(); fd, owner, first = prior.audited.support.acquire()
    binding = dict(outer_owner=owner, lock_identity=r.ORIGINAL_LOCK, retained_fd=fd, lock_policy=r.LOCK_POLICY,
        initial_descriptor_proof=first, unit='trn1-mini-fixed-input-pv-outer', attempt_id=uuid.uuid4().hex)
    lease = r.Lease(binding); ports = None
    try:
        impose_original_limits(); guard = PVGuard(lease); metadata = r.Metadata(guard); lease.metadata = metadata
        authenticated = auth.authenticate(spec_ref, review_ref, lease=lease, metadata=metadata, guard=guard)
        context = authenticated['context']
        source_subset = {key: context['pins_plan'][key] for key in ('source_contract', 'source_sha256')}
        require(len(serial.encoded(source_subset))+65536 <= PIPE_CAP, 'Whole source gate fits before any child spawn')
        store = Store(authenticated['output_root'], guard); store.create()
        model = metadata.read(context['original_spec']['model_exit'])
        admission = dict(schema='minicpm5_fixed_input_pv_serial_admission_v1', witness=authenticated['witness'],
            lease=binding, output_root=authenticated['output_root'], original_native_source_receipt=model['outputs']['source_authentication'])
        admission_ref = short(store.json('admission.json', admission))
        ports = Adapter(authenticated, admission_ref, admission, lease, guard, metadata, store)
        result = serial.FivePhases(ports).run()
        return short(store.reference('final.json'))
    finally:
        # Alarm/ordinary work cannot accidentally release the retained lock
        # while an owned child or descendant survives. Original drain retries
        # are the same finite TERM/KILL/group checks as the existing runner.
        signal.alarm(0); warned = False
        while True:
            try:
                proof = ports.drain_owned() if ports is not None else dict(physically_drained=True, no_children_started=True)
                break
            except BaseException as error:
                if not warned:
                    print(json.dumps(dict(event='pv_cleanup_pending_original_lock_retained', error=repr(error))), file=sys.stderr, flush=True)
                    warned = True
                time.sleep(.5)
        prior.audited.support.release_after_physical_drain(fd, proof)
