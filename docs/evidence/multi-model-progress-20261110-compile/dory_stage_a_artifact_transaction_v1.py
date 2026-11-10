"""Stage A artifact transaction source; no native imports or executable runner.

The admitted caller supplies authenticated bind_cases output, the installed
StageAObserver, its exact NativeTensorAdapter and a guard enforcing the current
serial/resource lease. No source/weight/exit authentication is inferred here.
Actual filesystem operations occur only when the caller constructs/uses a store.
No files are removed or replaced, including partial files after any failure.
"""
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import stat

from dory_stage_a_archive_core_v1 import ArchiveWriter, TensorEmitter, require
from dory_stage_a_capture_core_v1 import layouts, WIDE, sha
from dory_stage_a_saved_core_v1 import stage_a_requirements
from dory_stage_a_saved_files_v1 import verify_file, identity, ordinary, STAT_FIELDS

SCOPES = ('target', 'selected', 'original')
SCOPE_BYTES = dict(target=221736960, selected=80281600, original=1165094912)
STAGE_A_BYTES = 1467113472
ALL_THREE_STAGE_BYTES = 2046031872
PLANNED_RESERVATION_BYTES = 3 * 1024**3
UNPROVEN = ('independent_saved_byte_review_complete',
    'original_source_weight_runtime_identity_verified_here',
    'production_slot_reuse_qualified', 'original_FP32_acceptance_passed',
    'all87_marked_calls_qualified', 'full_model_qualified', 'new_speedup_claim')
LIMITS = dict(runtime_admitted_here=False,
    source_weight_and_actual_exit_provenance_verified_here=False,
    production_slot_reuse_qualified=False, original_FP32_acceptance_passed=False,
    all87_marked_calls_qualified=False, full_model_qualified=False, new_speedup_claim=False)


def extent(layout):
    return sum(math.prod(x['shape']) * (2 if x['dtype'] == 'BF16' else 4)
               for x in layout.values())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def name_for(case, scope):
    require(scope in SCOPES, 'Known archive scope')
    return f"event-{case['identity']['event']:06d}-{scope}.bin"


def _plan(cases):
    require(isinstance(cases, (tuple, list)) and len(cases) == 6, 'Exactly six authenticated cases')
    plans, totals = [], {scope: 0 for scope in SCOPES}
    for case, expected in zip(cases, WIDE):
        owner = case['identity']
        require(tuple(owner[k] for k in ('event', 'fixture', 'slot', 'epoch', 'end', 'phase')) == expected
                and owner['start'] == 0 and owner['kind'] == 'call'
                and isinstance(owner['request'], str) and bool(owner['request']),
                'Exact six original owners in original order')
        require(len(case['input_ids']) == owner['end']
                and all(type(x) is int and 0 <= x < 131072 for x in case['input_ids'])
                and len(case['original_top1']) == len(case['selected_rows']), 'Original complete token/row controls')
        independent = stage_a_requirements(case)
        plan = {scope: independent[scope]['layout'] for scope in SCOPES}
        require(plan == layouts(case), 'Independent reviewer and producer geometry agree')
        for scope in ('selected', 'original'):
            controls = case[scope+'_archive']['tensors']
            for label, item in plan[scope].items():
                old = controls[label]
                require(old['shape'] == item['shape'] and old['dtype'] == item['dtype']
                        and old['finite'] is True and old['bytes'] == extent({label: item}),
                        'Exact original control geometry')
                sha(old['sha256'])
        for scope in SCOPES:
            totals[scope] += extent(plan[scope])
        plans.append(plan)
    require(totals == SCOPE_BYTES and sum(totals.values()) == STAGE_A_BYTES,
            'Unchanged complete Stage A byte budget before any output')
    return plans


class ArchiveFileStore:
    """Owned canonical existing directory; exclusive ordinary files, no mkdir.

    guard is mandatory and must implement current admission/resource checks.
    seal returns identities only after every flush/fsync/close and directory
    fsync. Identities detect ordinary later changes; they are not a snapshot
    or protection against a malicious concurrent writer. The caller owns the
    directory exclusively through producer exit and independent saved review.
    """
    def __init__(self, root, *, guard):
        require(callable(guard), 'Explicit admitted producer guard')
        self.guard, self.root, self.directory = guard, Path(root), None
        self.handles = {}
        guard()
        require(self.root.is_absolute() and self.root.resolve() == self.root
                and self.root.is_dir(), 'Canonical existing output directory')
        self.directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            self.parent = os.fstat(self.directory)
            require(stat.S_ISDIR(self.parent.st_mode) and self.parent.st_uid == os.getuid(),
                    'Owned actual output directory')
            self._stable()
        except BaseException:
            self.abort()
            raise

    def _stable(self):
        self.guard()
        require(self.directory is not None and self.root.resolve() == self.root,
                'Live directory and unchanged canonical ancestors')
        now, held = self.root.stat(follow_symlinks=False), os.fstat(self.directory)
        require(all(stat.S_ISDIR(x.st_mode) and x.st_uid == self.parent.st_uid
                    and (x.st_dev, x.st_ino) == (self.parent.st_dev, self.parent.st_ino)
                    for x in (now, held)), 'Same original owned directory')

    def open_exclusive(self, name):
        self._stable()
        require(isinstance(name, str) and name not in ('', '.', '..')
                and Path(name).name == name and name not in self.handles,
                'Fresh single archive basename')
        fd = os.open(name, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                     0o600, dir_fd=self.directory)
        try:
            stream = os.fdopen(fd, 'wb')
        except BaseException:
            try:
                os.close(fd)
            except OSError:
                pass
            raise
        self.handles[name] = stream
        initial = os.fstat(stream.fileno())
        ordinary(initial, identity(initial), 0)
        named = os.stat(name, dir_fd=self.directory, follow_symlinks=False)
        ordinary(named, identity(initial), 0)
        return stream

    def seal(self, names_and_sizes):
        require(len(names_and_sizes) == 3, 'One complete three-archive transaction')
        before = {}
        for name, size in names_and_sizes.items():
            self._stable()
            require(name in self.handles and type(size) is int and size > 0, 'Known nonempty archive')
            stream = self.handles[name]
            require(not stream.closed, 'Unsealed stream')
            stream.flush()
            os.fsync(stream.fileno())
            info = os.fstat(stream.fileno())
            before[name] = identity(info)
            ordinary(info, before[name], size)
            ordinary(os.stat(name, dir_fd=self.directory, follow_symlinks=False), before[name], size)
        require(len({(x['st_dev'], x['st_ino']) for x in before.values()}) == 3,
                'Three independent archive files')
        for name in names_and_sizes:
            self.handles[name].close()
            require(self.handles[name].closed, 'All archives actually closed')
        os.fsync(self.directory)
        self._stable()
        for name, size in names_and_sizes.items():
            ordinary(os.stat(name, dir_fd=self.directory, follow_symlinks=False), before[name], size)
        return before

    def close(self):
        self._stable()
        require(all(stream.closed for stream in self.handles.values()), 'All producer archive handles closed')
        fd, self.directory = self.directory, None
        os.close(fd)

    def abort(self):
        # Preserve every file and the first failure; best-effort close is not
        # a success receipt or a claim that failed close operations succeeded.
        for stream in self.handles.values():
            try:
                if not stream.closed:
                    stream.close()
            except BaseException:
                pass
        if self.directory is not None:
            fd, self.directory = self.directory, None
            try:
                os.close(fd)
            except BaseException:
                pass


def _write_capture(case, planned, observer, adapter, store, guard, tokens, positions):
    """The single real consumer chain; also exercised with tiny BytesIO fixtures."""
    try:
        writers = {}
        for scope in SCOPES:
            guard()
            stream = store.open_exclusive(name_for(case, scope))
            writers[scope] = ArchiveWriter(stream, planned[scope], byte_limit=extent(planned[scope]), guard=guard)
        emitter = TensorEmitter(writers, adapter.iter_bytes)
        observed = observer.collect(case, tokens, positions, emitter)
        archives = emitter.finish()
        require(set(observed['records']) == set(SCOPES)
                and all(observed['records'][s] == archives[s]['tensors'] for s in SCOPES),
                'Observer descriptors are exactly those emitted to all three archives')
        identities = store.seal({name_for(case, s): archives[s]['bytes'] for s in SCOPES})
        return dict(observation=observed, files={s: dict(name=name_for(case, s),
            producer_identity=identities[name_for(case, s)], archive=archives[s]) for s in SCOPES})
    except BaseException:
        store.abort()
        raise


def _validate_receipt(case, receipt, requirements):
    require(set(receipt) == {'case_sha256', 'identity', 'original_event', 'residual_event',
                            'observation', 'files'} and receipt['case_sha256'] == digest(case)
            and receipt['identity'] == case['identity']
            and receipt['original_event'] == case['original_event']
            and receipt['residual_event'] == case['residual_event'], 'Exact original case and evidence binding')
    observed = receipt['observation']
    required = {'identity', 'selected_rows', 'records', 'source_layouts', 'original_top1',
                'physical_visits', 'onednn_calls', 'original_byte_controls_matched', *UNPROVEN}
    require(set(observed) == required and observed['identity'] == case['identity']
            and observed['selected_rows'] == case['selected_rows']
            and observed['original_top1'] == case['original_top1']
            and type(observed['physical_visits']) is int and observed['physical_visits'] == 112
            and type(observed['onednn_calls']) is int and observed['onednn_calls'] == 281
            and observed['original_byte_controls_matched'] is True
            and all(observed[k] is False for k in UNPROVEN), 'Exact passive observer result without promoted claims')
    require(set(receipt['files']) == set(observed['records']) == set(observed['source_layouts']) == set(SCOPES),
            'Exactly three complete archive scopes')
    for scope in SCOPES:
        spec, file = requirements[scope], receipt['files'][scope]
        layout = spec['layout']
        require(set(file) == {'name', 'producer_identity', 'archive'}
                and file['name'] == name_for(case, scope), 'Exact archive owner and scope filename')
        archive, file_id = file['archive'], file['producer_identity']
        require(set(file_id) == set(STAT_FIELDS) and all(type(v) is int and v >= 0 for v in file_id.values())
                and file_id['st_size'] == extent(layout), 'Exact finite producer file identity and size')
        require(set(archive) == {'complete', 'bytes', 'sha256', 'tensors'}
                and archive['complete'] is True and type(archive['bytes']) is int
                and archive['bytes'] == extent(layout) and list(archive['tensors']) == list(layout)
                and observed['records'][scope] == archive['tensors']
                and list(observed['source_layouts'][scope]) == list(layout), 'Complete bound archive metadata')
        sha(archive['sha256'])
        offset = 0
        for label, item in layout.items():
            desc, source = archive['tensors'][label], observed['source_layouts'][scope][label]
            size = extent({label: item})
            require(set(desc) == {'shape', 'dtype', 'finite', 'offset', 'bytes', 'sha256'}
                    and desc['shape'] == item['shape'] and desc['dtype'] == item['dtype']
                    and desc['finite'] is True and type(desc['bytes']) is int and desc['bytes'] == size
                    and type(desc['offset']) is int and desc['offset'] == offset, 'Exact dense finite tensor descriptor')
            sha(desc['sha256'])
            offset += size
            if 'expected_tensors' in spec:
                old = spec['expected_tensors'][label]
                require(all(desc[k] == old[k] for k in ('shape', 'dtype', 'finite', 'bytes', 'sha256')),
                        'Every original control bound before opening payload')
            shape = [case['identity']['end'], 2560] if scope == 'selected' else item['shape']
            require(set(source) == {'shape', 'dtype', 'stride', 'storage_offset'}
                    and source['shape'] == shape and source['dtype'] == item['dtype']
                    and isinstance(source['stride'], list) and len(source['stride']) == len(shape)
                    and all(type(v) is int and v >= 0 for v in source['stride'])
                    and type(source['storage_offset']) is int and source['storage_offset'] >= 0,
                    'Full original source geometry with actual nonnegative layout')


class CaptureArtifactBatch:
    """Six ordered captures; failure poisons the batch and leaves partial files.

    A returned per-case receipt proves only this producer-side transaction.
    finish returns an in-memory manifest only after all 18 files and observer
    finish. Caller must durably persist/authenticate that metadata, remove hooks,
    actually exit the producer, then admit an independent reviewer separately.
    """
    def __init__(self, cases, observer, adapter, store, *, guard):
        require(callable(guard) and observer.ops is adapter, 'Exact installed observer adapter and explicit guard')
        self.cases = deepcopy(tuple(cases))
        self.plans = _plan(self.cases)
        self.observer, self.adapter, self.store, self.guard = observer, adapter, store, guard
        self.receipts, self.committed_bytes = [], 0
        self.poisoned = self.finished = False

    def collect(self, tokens, positions):
        require(not self.poisoned and not self.finished and len(self.receipts) < 6, 'Healthy unfinished six-case batch')
        try:
            self.guard()
            case, planned = self.cases[len(self.receipts)], self.plans[len(self.receipts)]
            size = sum(extent(planned[s]) for s in SCOPES)
            require(self.committed_bytes + size <= STAGE_A_BYTES, 'Cumulative exact Stage A budget')
            result = _write_capture(case, planned, self.observer, self.adapter, self.store,
                                    self.guard, tokens, positions)
            receipt = dict(case_sha256=digest(case), identity=deepcopy(case['identity']),
                           original_event=deepcopy(case['original_event']),
                           residual_event=deepcopy(case['residual_event']), **result)
            _validate_receipt(case, receipt, stage_a_requirements(case))
            self.receipts.append(deepcopy(receipt))
            self.committed_bytes += size
            return deepcopy(receipt)
        except BaseException:
            self.poisoned = True
            self.store.abort()
            raise

    def finish(self):
        require(not self.poisoned and not self.finished and len(self.receipts) == 6
                and self.committed_bytes == STAGE_A_BYTES, 'All six complete transactions only')
        try:
            self.guard()
            summary = self.observer.finish()
            require(summary == dict(stage_a_collector_complete=True, events=[x[0] for x in WIDE],
                independent_saved_byte_review_complete=False, all87_marked_calls_qualified=False,
                full_model_qualified=False, new_speedup_claim=False), 'Original six-case observer finish')
            self.store.close()
            result = dict(schema='dory_stage_a_artifact_batch_v1', producer_artifacts_complete=True,
                cases=deepcopy(self.receipts), scope_bytes=dict(SCOPE_BYTES), bytes=STAGE_A_BYTES,
                observer_summary=summary, independent_saved_byte_review_complete=False, **LIMITS)
            self.finished = True
            return result
        except BaseException:
            self.poisoned = True
            self.store.abort()
            raise


def review_saved_batch(root, cases, manifest, *, guard):
    """All18 independently reopened complete files, after actual producer exit.

    Caller must authenticate originals, this manifest and actual producer exit
    before calling. The explicit guard owns independent serial review admission;
    this function does not generate admission or certify exit/source/weights.
    Every metadata check occurs before the first payload file is opened.
    """
    require(callable(guard), 'Explicit independently admitted review guard')
    guard()
    cases, manifest = deepcopy(tuple(cases)), deepcopy(manifest)
    _plan(cases)
    require(set(manifest) == {'schema', 'producer_artifacts_complete', 'cases', 'scope_bytes', 'bytes',
                             'observer_summary', 'independent_saved_byte_review_complete', *LIMITS}
            and manifest['schema'] == 'dory_stage_a_artifact_batch_v1'
            and manifest['producer_artifacts_complete'] is True
            and manifest['independent_saved_byte_review_complete'] is False
            and all(manifest[k] is False for k in LIMITS)
            and manifest['scope_bytes'] == SCOPE_BYTES and type(manifest['bytes']) is int
            and manifest['bytes'] == STAGE_A_BYTES and isinstance(manifest['cases'], list)
            and len(manifest['cases']) == 6, 'Exactly complete original Stage A manifest without promoted claims')
    require(manifest['observer_summary'] == dict(stage_a_collector_complete=True,
        events=[x[0] for x in WIDE], independent_saved_byte_review_complete=False,
        all87_marked_calls_qualified=False, full_model_qualified=False, new_speedup_claim=False),
        'Original complete collector summary')
    requirements, inodes = [], set()
    for case, receipt in zip(cases, manifest['cases']):
        spec = stage_a_requirements(case)
        _validate_receipt(case, receipt, spec)
        requirements.append(spec)
        for file in receipt['files'].values():
            file_id = file['producer_identity']
            inode = file_id['st_dev'], file_id['st_ino']
            require(inode not in inodes, 'All18 distinct producer files')
            inodes.add(inode)
    reviews = []
    for case, receipt, specs in zip(cases, manifest['cases'], requirements):
        for scope in SCOPES:
            guard()
            spec, file = specs[scope], receipt['files'][scope]
            result = verify_file(root, file['name'], file['producer_identity'], file['archive'], spec['layout'],
                max_bytes=extent(spec['layout']), guard=guard,
                expected_tensors=spec.get('expected_tensors'), selected_row_checks=spec.get('selected_row_checks'))
            require(result['complete'] is True and result['file'] == file['name']
                    and result['file_identity'] == file['producer_identity']
                    and result['bytes_verified'] == extent(spec['layout'])
                    and result['whole_file_from_offset_zero'] is True
                    and result['stable_producer_identity_verified'] is True
                    and result['byte_verification']['complete'] is True,
                    'Independent stable whole-file bytes actually verified')
            reviews.append(dict(event=case['identity']['event'], scope=scope, review=result))
    guard()
    return dict(schema='dory_stage_a_saved_batch_review_v1', complete=True,
        manifest_sha256=digest(manifest), files=reviews, file_count=18,
        bytes_verified=STAGE_A_BYTES, independent_saved_byte_review_complete=True, **LIMITS)
