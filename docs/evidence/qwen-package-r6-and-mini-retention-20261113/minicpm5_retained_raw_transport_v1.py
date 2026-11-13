"""Finite lossless restore-control transport; no import-time IO or entrypoint.

The production plan is built only from the exact authenticated metadata below.
Original capture identities are historical facts, never returned as live stat.
This module deliberately has NO unlink/rename/hardlink/symlink implementation.
It prepares complete byte-verification receipts and a durable retirement intent;
an independently reviewed external remover is required before accepting commit.
Legacy frozen recovery scripts are not monkeypatched or made directly runnable.
"""
from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import stat

ROOT = Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
Q = ROOT / 'research'
DATA = Path('/mnt/nvme/minicpm5-history-crossover-v1/device_crossover')
PAIR_BYTES = 4325376
CHUNK = 1 << 20
PLAN_SCHEMA = 'minicpm5_restore_raw_transport_plan_v1'
PRODUCTION_SCOPE = 'minicpm5_history_restore_after_all14x168'
PRODUCTION_PLAN_SHA256 = '996215f14e08e8b1a2a1e8f0e53e776a0dfe919102a60411a7e316706c5d61da'
REFERENCES = {
    'candidate': dict(path=str(Q/'minicpm5-retained-restore-control-duplicate-feasibility-20261113-v1.json'),
        sha256='831da717d9501ee5164c459fb1a4f988c75c7a82b8a1bd086f016024441fea46'),
    'recovery_plan': dict(path=str(ROOT/'minicpm5-history-analysis-recovery-v2/plan.json'),
        sha256='ea3954682dfd8ba14e1559ed02fa77d938bc987a839cd77c89238520c32c3252'),
    'admission': dict(path=str(ROOT/'minicpm5-history-analysis-recovery-v2/payload-admission.json'),
        sha256='cc91a1414c9e29bafe408812d4a0612499f6f688532c1005513f2bb5a13758d2'),
}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def identity(path):
    path = Path(path)
    require(path.is_absolute() and str(path) == str(path.resolve()), 'Canonical absolute path required')
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'Symlink forbidden')
    value = path.lstat()
    require(stat.S_ISREG(value.st_mode) and value.st_nlink == 1, 'Singleton regular file required')
    return _stat_identity(path, value)


def _stat_identity(path, value):
    return dict(realpath=str(path), device=value.st_dev, inode=value.st_ino,
        size=value.st_size, mtime_ns=value.st_mtime_ns, ctime_ns=value.st_ctime_ns)


def _id_shape(row):
    require(type(row) is dict and set(row) == {'realpath','device','inode','size','mtime_ns','ctime_ns'},
        'Complete physical identity required')
    require(type(row['realpath']) is str and Path(row['realpath']).is_absolute()
        and str(Path(row['realpath'])) == row['realpath'] and '..' not in Path(row['realpath']).parts,
        'Exact absolute identity path')
    require(all(type(row[k]) is int and row[k] >= 0 for k in row if k != 'realpath'), 'Identity integer fields')


def validate_plan(plan, *, production=True):
    require(type(plan) is dict and set(plan) == {'schema','scope','provenance','entries'}, 'Exact plan fields')
    require(plan['schema'] == PLAN_SCHEMA and type(plan['entries']) is list, 'Transport schema')
    entries = plan['entries']
    require(0 < len(entries) <= 2352, 'Finite map cardinality')
    originals, keepers = set(), set()
    for index, row in enumerate(entries):
        require(type(row) is dict and set(row) == {'index','original','keeper','sha256','bytes','original_row','keeper_row'},
            'Complete exact pair record')
        require(row['index'] == index and type(row['index']) is int, 'Stable complete pair order')
        for key in ('original','keeper'):
            _id_shape(row[key])
        require(type(row['sha256']) is str and len(row['sha256']) == 64
            and set(row['sha256']) <= set('0123456789abcdef'), 'Full SHA256')
        require(type(row['bytes']) is int and 0 < row['bytes'] <= PAIR_BYTES
            and row['original']['size'] == row['keeper']['size'] == row['bytes'], 'Exact full-file size')
        a, b = row['original']['realpath'], row['keeper']['realpath']
        require(a not in originals and b not in keepers and a != b, 'Distinct one-to-one pair paths')
        require((row['original']['device'],row['original']['inode']) !=
            (row['keeper']['device'],row['keeper']['inode']), 'Original and keeper are distinct physical files')
        originals.add(a); keepers.add(b)
        for key, ident in (('original_row','original'), ('keeper_row','keeper')):
            source = row[key]
            require(type(source) is dict and source['file'] == row[ident]['realpath']
                and source['identity'] == row[ident] and source['encoding'] == 'raw_bf16_le'
                and source['raw_bytes'] == row['bytes']
                and source['payload_sha256'] == source['raw_sha256'] == row['sha256'], 'Exact producer row linkage')
    require(not originals & keepers, 'No mapping chains or cycles')
    if production:
        require(digest(canonical(plan)) == PRODUCTION_PLAN_SHA256, 'Exact metadata-derived complete production map')
        require(plan['scope'] == PRODUCTION_SCOPE and len(entries) == 2352, 'Only complete production selection')
        require(plan['provenance']['references'] == REFERENCES, 'Original immutable metadata anchors')
        for row in entries:
            call, state = divmod(row['index'], 168)
            parent = DATA / ('call-%02d' % call)
            require(row['bytes'] == PAIR_BYTES
                and row['original']['realpath'] == str(parent/'D_Hd_restore_control/after'/('%03d.raw' % state))
                and row['keeper']['realpath'] == str(parent/'D_Hd_control/after'/('%03d.raw' % state)),
                'Only same-call/index restore-after to control-after mapping')
    return plan


def synthetic_plan(plan):
    """Private test guard: all physical paths must be in one /tmp fixture."""
    validate_plan(plan, production=False)
    require(plan['scope'] == 'synthetic-tests-only', 'Synthetic scope only')
    paths = [Path(r[k]['realpath']) for r in plan['entries'] for k in ('original','keeper')]
    roots = {p.parts[:3] for p in paths}
    require(len(roots) == 1 and all(p.parts[1] == 'tmp' and len(p.parts) >= 4 for p in paths),
        'Synthetic transport cannot target production paths')
    return plan


def build_production_plan(read_metadata):
    """Metadata-only callback(ref)->bytes; no payload stat/read or History.

    Every callback result is independently hashed here. All producer row fields
    are preserved; the original recovery plan and admission remain unchanged.
    """
    observed = {}
    def read(ref):
        require(Path(ref['path']).suffix == '.json', 'JSON metadata only')
        raw = read_metadata(copy.deepcopy(ref))
        require(type(raw) is bytes and len(raw) <= 16 << 20 and digest(raw) == ref['sha256'], 'Authenticated bounded metadata')
        observed[ref['path']] = dict(ref, bytes=len(raw))
        return json.loads(raw)
    candidate, old, admission = (read(REFERENCES[k]) for k in ('candidate','recovery_plan','admission'))
    require(admission['sha256'] == {p:r['sha256'] for p,r in old['analysis_payloads'].items()}, 'Full11859 admission retained')
    require(len(old['analysis_payloads']) == 11859, 'Full original logical payload map')
    manifests = {}
    for ref in candidate['producer_manifest_refs']:
        require(old['acquisition_metadata_sha256'][ref['path']] == ref['sha256'], 'Producer manifest in original frozen plan')
        manifests[ref['path']] = read({k:ref[k] for k in ('path','sha256')})
    require(len(manifests) == 28, 'All28 complete producer manifests')
    entries = []
    for index, pair in enumerate(candidate['narrow_selection']['complete_pair_manifest']):
        item = dict(index=index, original=pair['duplicate']['identity'], keeper=pair['keep']['identity'],
            sha256=pair['sha256'], bytes=pair['bytes'])
        for side, arm in (('original','D_Hd_restore_control'),('keeper','D_Hd_control')):
            ident = item[side]
            require(old['analysis_payloads'][ident['realpath']] == dict(identity=ident,sha256=item['sha256']),
                'Each original physical identity and digest matches immutable recovery plan')
            path = ROOT/'minicpm5-history-crossover-v1/device_crossover'/('call-%02d'%pair['call_index'])/arm/'after/manifest.json'
            source = manifests[str(path)]['rows'][pair['state_index']]
            require(source['identity'] == ident, 'Exact producer identity')
            item[side+'_row'] = copy.deepcopy(source)
        entries.append(copy.deepcopy(item))
    plan = dict(schema=PLAN_SCHEMA,scope=PRODUCTION_SCOPE,
        provenance=dict(references=copy.deepcopy(REFERENCES),authenticated_metadata=observed),entries=entries)
    return validate_plan(plan)


def _open(ident):
    path = Path(ident['realpath'])
    require(identity(path) == ident, 'Physical file identity changed before open')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        st = os.fstat(fd)
        require(stat.S_ISREG(st.st_mode) and st.st_nlink == 1 and _stat_identity(path,st) == ident,
            'Opened descriptor identity changed')
    except BaseException:
        os.close(fd)
        raise
    return fd


def _unchanged(fd, ident):
    st = os.fstat(fd)
    require(st.st_nlink == 1 and _stat_identity(ident['realpath'],st) == ident and identity(ident['realpath']) == ident,
        'Physical file changed across read')


def _pair(row, guard):
    guard(); a = _open(row['original'])
    try:
        b = _open(row['keeper'])
        try:
            ah, bh, count = hashlib.sha256(), hashlib.sha256(), 0
            while True:
                guard(); x, y = os.read(a,CHUNK), os.read(b,CHUNK)
                require(x == y, 'Complete paired bytes differ')
                if not x: break
                count += len(x); require(count <= row['bytes'], 'Unbounded pair growth')
                ah.update(x); bh.update(y)
            require(count == row['bytes'] and ah.hexdigest() == bh.hexdigest() == row['sha256'], 'Full size/SHA mismatch')
            guard(); _unchanged(a,row['original']); _unchanged(b,row['keeper'])
            return dict(index=row['index'],bytes=count,sha256=row['sha256'],
                original_identity=copy.deepcopy(row['original']),keeper_identity=copy.deepcopy(row['keeper']),
                original_allocated_bytes=os.fstat(a).st_blocks*512)
        finally: os.close(b)
    finally: os.close(a)


def _write_once(path, value):
    """Durable exclusive JSON. Partial interrupted files are never accepted."""
    path = Path(path)
    require(path.is_absolute() and path.suffix=='.json' and path.parent.is_dir()
        and not any(p.is_symlink() for p in (path,*path.parents)), 'Existing canonical receipt parent')
    raw = canonical(value) + b'\n'
    with path.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    directory = os.open(path.parent,os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try: os.fsync(directory)
    finally: os.close(directory)
    return dict(path=str(path),sha256=digest(raw),bytes=len(raw))


def verify_pairs(plan, receipt_path, guard):
    """Full paired stream comparison; production guard must hold external lease.

    Bounded synthetic plans may be tested through _verify_pairs only. The public
    entrypoint accepts only the fixed production path/cardinality contract.
    """
    validate_plan(plan)
    return _verify_pairs(plan,receipt_path,guard)


def _verify_pairs(plan, receipt_path, guard):
    (synthetic_plan(plan) if plan['scope']=='synthetic-tests-only' else validate_plan(plan))
    require(callable(guard),'Explicit external admission guard')
    records = [_pair(row,guard) for row in plan['entries']]
    # No early partial receipt can authorize retirement. Recheck all identities
    # after the last pair, to detect changes to an earlier pair during the pass.
    for row in plan['entries']:
        guard()
        require(identity(row['original']['realpath']) == row['original']
            and identity(row['keeper']['realpath']) == row['keeper'], 'Earlier pair changed during full pass')
    receipt = dict(schema='minicpm5_restore_raw_full_pair_verification_v1',plan_sha256=digest(canonical(plan)),
        records=records,complete=True,original_and_keeper_bytes_read=sum(r['bytes']*2 for r in records),
        removals_performed=False)
    guard(); return _write_once(receipt_path,receipt)


def read_receipt(ref):
    require(type(ref) is dict and set(ref) == {'path','sha256','bytes'} and 0 < ref['bytes'] <= 16 << 20,
        'Exact bounded receipt reference')
    ident = identity(ref['path']); require(ident['size'] == ref['bytes'],'Receipt size changed')
    fd = _open(ident)
    try:
        chunks, count = [], 0
        while True:
            block = os.read(fd,min(CHUNK,ref['bytes']+1-count))
            if not block: break
            chunks.append(block); count += len(block)
            require(count <= ref['bytes'],'Receipt grew')
        _unchanged(fd,ident)
    finally: os.close(fd)
    raw = b''.join(chunks); require(len(raw)==ref['bytes'] and digest(raw)==ref['sha256'],'Receipt digest changed')
    return json.loads(raw)


def _verification(plan, ref):
    value = read_receipt(ref)
    require(value['schema']=='minicpm5_restore_raw_full_pair_verification_v1' and value['complete'] is True
        and value['removals_performed'] is False and value['plan_sha256']==digest(canonical(plan)), 'Exact complete verification receipt')
    require(len(value['records'])==len(plan['entries']), 'No partial verification may authorize retirement')
    for row, record in zip(plan['entries'],value['records']):
        require(record==dict(index=row['index'],bytes=row['bytes'],sha256=row['sha256'],
            original_identity=row['original'],keeper_identity=row['keeper'],
            original_allocated_bytes=record['original_allocated_bytes'])
            and type(record['original_allocated_bytes'])is int and record['original_allocated_bytes']>=0,
            'Every exact full pair retained')
    require(value['original_and_keeper_bytes_read']==sum(r['bytes']*2 for r in plan['entries']), 'Complete byte accounting')
    return value


def prepare_retirement_intent(plan, verification_ref, path, admission, guard):
    """Metadata protocol only: durable intent, never permission to unlink here.

    Caller supplies a root-reviewed removal/replay implementation reference and
    independently verified complete-pair receipt. All original physical files
    must still exist unchanged. A resume must reuse this exact immutable intent;
    attempting to create another intent after deletion fails closed.
    """
    validate_plan(plan); return _prepare_intent(plan,verification_ref,path,admission,guard)


def _prepare_intent(plan, verification_ref, path, admission, guard):
    _verification(plan,verification_ref)
    require(type(admission)is dict and set(admission)=={'plan_sha256','independent_verification','remover_source','replay_source','serial_lease','full_history_compatibility'},
        'Explicit independent review, replay, remover and serial admission required')
    require(admission['plan_sha256']==digest(canonical(plan)), 'Admission binds entire plan')
    for key in ('independent_verification','remover_source','replay_source','serial_lease','full_history_compatibility'):
        ref=admission[key]
        require(type(ref)is dict and set(ref)=={'path','sha256'} and Path(ref['path']).is_absolute()
            and type(ref['sha256'])is str and len(ref['sha256'])==64
            and set(ref['sha256'])<=set('0123456789abcdef'),'Exact independently accepted prerequisite reference')
    for row in plan['entries']:
        guard(); require(identity(row['original']['realpath'])==row['original']
            and identity(row['keeper']['realpath'])==row['keeper'],'Intent requires all original files present unchanged')
    guard()
    return _write_once(path,dict(schema='minicpm5_restore_raw_retirement_intent_v1',
        plan_sha256=digest(canonical(plan)),verification=verification_ref,admission=copy.deepcopy(admission),
        complete_original_selection=[r['original']['realpath']for r in plan['entries']],
        prerequisite_semantics_authenticated_here=False,unlink_authority_implemented_here=False))


class Replay:
    """Explicit finite raw reader. No implicit fallback for other legacy inputs.

    read_state accepts the full original producer row and returns original bytes
    plus separate capture/storage identities. It can also validate and re-export
    bytes before retirement. Original frozen validators need an explicit reviewed
    successor that consumes this result; this object never spoofs their identity.
    """
    def __init__(self, plan, verification_ref, guard, *, _synthetic=False):
        validate_plan(plan,production=not _synthetic)
        if _synthetic: synthetic_plan(plan)
        _verification(plan,verification_ref)
        self.plan=copy.deepcopy(plan); self.guard=guard
        self.entries={r['original']['realpath']:r for r in self.plan['entries']}

    def read_state(self, original_row):
        require(type(original_row)is dict and original_row.get('file')in self.entries,'Unknown logical input; no fallback')
        row=self.entries[original_row['file']]
        require(original_row==row['original_row'],'Exact entire original producer row required')
        self.guard(); fd=_open(row['keeper'])
        try:
            chunks,count=[],0
            while True:
                self.guard(); block=os.read(fd,CHUNK)
                if not block: break
                count+=len(block); require(count<=row['bytes'],'Keeper grew'); chunks.append(block)
            data=b''.join(chunks)
            require(count==row['bytes'] and digest(data)==row['sha256'],'Keeper complete bytes/hash mismatch')
            self.guard(); _unchanged(fd,row['keeper'])
        finally: os.close(fd)
        return data,dict(logical_path=row['original']['realpath'],capture_identity=copy.deepcopy(row['original']),
            storage_identity=copy.deepcopy(row['keeper']),sha256=row['sha256'],bytes=len(data),
            original_identity_is_current_stat=False)

    def export_state(self, original_row, destination, receipt_path):
        """Byte recovery into a fresh independent file, with honest new identity.

        Existing paths are never replaced. On interruption a partial export may
        remain, but no completion receipt exists; use a fresh path to retry.
        Neither an export nor its new inode masquerades as the original capture.
        """
        destination=Path(destination)
        require(str(destination)not in self.entries and all(str(destination)!=r['keeper']['realpath']for r in self.entries.values()),
            'Export must not overwrite any original or keeper path')
        require(destination.is_absolute() and destination.parent.is_dir()
            and not any(p.is_symlink()for p in (destination,*destination.parents)), 'Fresh canonical export path')
        data,proof=self.read_state(original_row);self.guard()
        with destination.open('xb')as stream:
            stream.write(data);stream.flush();os.fsync(stream.fileno())
        exported=identity(destination)
        parent=os.open(destination.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try:os.fsync(parent)
        finally:os.close(parent)
        fd=_open(exported)
        try:
            h=hashlib.sha256();count=0
            while True:
                self.guard();block=os.read(fd,CHUNK)
                if not block:break
                count+=len(block);require(count<=len(data),'Export grew');h.update(block)
            require(count==len(data) and h.hexdigest()==proof['sha256'],'Exported actual bytes differ')
            _unchanged(fd,exported)
        finally:os.close(fd)
        self.guard()
        return _write_once(receipt_path,dict(schema='minicpm5_restore_raw_export_v1',
            original=proof,export_identity=exported,bytes=len(data),sha256=digest(data),
            capture_inode_recreated=False))


def audit_retired_state(plan, verification_ref, intent_ref, result_path, guard):
    """After an external removal: fully reread keepers, assert all originals absent.

    This is a postcondition receipt, not an invented observation of who unlinked
    files. The external remover must separately preserve per-file durable progress
    and actual exit/failure evidence. Incomplete removal is rejected; the same
    intent and verification remain usable for explicit resumed review.
    """
    validate_plan(plan); return _audit_retired(plan,verification_ref,intent_ref,result_path,guard)


def _audit_retired(plan, verification_ref, intent_ref, result_path, guard):
    intent=read_receipt(intent_ref)
    require(intent['schema']=='minicpm5_restore_raw_retirement_intent_v1'
        and intent['plan_sha256']==digest(canonical(plan)) and intent['verification']==verification_ref
        and intent['complete_original_selection']==[r['original']['realpath']for r in plan['entries']],
        'Full original selection and pre-removal intent retained')
    replay=Replay(plan,verification_ref,guard,_synthetic=plan['scope']=='synthetic-tests-only')
    proofs=[]
    for row in plan['entries']:
        guard(); require(not os.path.lexists(row['original']['realpath']),'Original still present or replaced')
        data,proof=replay.read_state(row['original_row']); proofs.append(proof); del data
    for row in plan['entries']:
        guard();require(not os.path.lexists(row['original']['realpath'])
            and identity(row['keeper']['realpath'])==row['keeper'],'Retired state changed during complete audit')
    guard();return _write_once(result_path,dict(schema='minicpm5_restore_raw_retired_postcondition_v1',
        plan_sha256=digest(canonical(plan)),verification=verification_ref,intent=intent_ref,complete=True,
        retained_full_byte_proofs=proofs,original_paths_absent=len(proofs),
        unlink_execution_observed_here=False,full_model_qualified=False,new_speedup_claim=False))
