"""Inventory-only descriptor policy binding; never imported by old runtimes.

Runtime IO below is reachable only from the new separately reviewed launcher.
The original resource/history/drain policy remains in contract_v2/launcher_v5.
"""
import copy
import fcntl
import json
import os
from pathlib import Path
import stat
import sys
import dory_serial_stage_contract_v2 as c
import descriptor_serial_lock_proof_v1 as core
import run_dory_serial_stage_root_v5 as lifecycle

CAP = 8*c.MIB
ORIGINAL_LOCK = dict(device=66306, inode=8294421)
POLICY = dict(schema='dory_inventory_descriptor_policy_v2', original_lock_identity=copy.deepcopy(ORIGINAL_LOCK),
    core=dict(path=str(c.Q/'descriptor_serial_lock_proof_v1.py'),
              sha256='f3faa10356e16dce6b689298ceebf11441f3821ce391f54896cbffed1c82b003'),
    original_proc=dict(path=str(c.ROOT/'minicpm5_layer0_execution_protocol_v4.py'),
                      sha256='81e7d7dd913c6b85f0add7b946692d3270f80cc8f2fe980ecad543251657f185'),
    retained_fd_binding='allocated_once_by_parent_then_bound_in_initial_proof_and_admission',
    parent_only_fd=True, close_on_exec_required=True, inherited_by_worker=False,
    no_unlock_close_until_owned_group_drained=True,
    same_inode_global_conflicts_deny=True, ordinal_only_target_repetitions_allowed=True,
    original_unique_row_guard_called=False, retry_after_failure=False)


def validate(plan, review, plan_ref):
    c.require(plan.get('stage') == 'inventory', 'Descriptor successor is inventory-only')
    c.validate(plan, review, plan_ref)
    c.require(plan.get('lock_policy') == review.get('lock_policy') == POLICY,
              'Exact new policy explicitly reviewed with frozen plan')
    for row in (POLICY['core'], POLICY['original_proc']):
        c.require(plan['source_sha256'].get(row['path']) == row['sha256'], 'Exact descriptor/original-proc source pins')
    return plan


def canonical_lock():
    path = Path(core.LOCK_PATH)
    for parent in reversed(path.parents):
        info = parent.lstat()
        c.require(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode), 'Canonical lock ancestry without symlinks')
    info = path.lstat()
    c.require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, 'Original canonical singleton lock')
    return info


def observe(p, owner, lock, fd):
    c.require(str(p.LOCK) == core.LOCK_PATH, 'Original lock path unchanged')
    c.require(lock==ORIGINAL_LOCK,'Pinned original lock binding for every observation')
    before=canonical_lock();c.require(dict(device=before.st_dev,inode=before.st_ino)==ORIGINAL_LOCK,'Pinned original path inode')
    result = core.prove(owner, lock, fd, proc=p.proc,
        open_binary=lambda path: Path(path).open('rb'), lstat=lambda path: Path(path).lstat(),
        stat=lambda path: Path(path).stat(), readlink=os.readlink)
    canonical_lock()
    return result


def encoded(report):
    raw = (json.dumps(report, separators=(',', ':'), allow_nan=False)+'\n').encode()
    c.require(len(raw) <= CAP, 'Complete descriptor failure evidence within original8MiB cap')
    return raw


def error_report(error, role, owner, lock, fd, plan_ref, review_ref, admission_ref=None):
    return dict(schema='dory_inventory_descriptor_failure_v1', role=role,
        error_type=type(error).__name__, error=str(error), observations=getattr(error, 'observations', None),
        owner=copy.deepcopy(owner), lock=copy.deepcopy(lock), retained_fd=fd,
        plan=copy.deepcopy(plan_ref), review=copy.deepcopy(review_ref), admission=copy.deepcopy(admission_ref),
        lock_policy=copy.deepcopy(POLICY), original_exception_preserved=True, proof_retried=False,
        actual_process_exit_claimed=False, runtime_success_claimed=False, full_model_qualified=False, new_speedup_claim=False)


def preoutput_failure(error, owner, lock, fd, plan_ref, review_ref, pending=None):
    report = error_report(error, 'parent-pre-output', owner, lock, fd, plan_ref, review_ref)
    report.update(in_tree_receipt=False, output_directory_created=False, unpersisted_first_duplicate=pending)
    try:
        raw = encoded(report); written = 0
        while written < len(raw):
            size = os.write(2, raw[written:]); c.require(size > 0, 'Complete stderr evidence write'); written += size
    except BaseException as secondary:
        try: os.write(2, ('Dory descriptor pre-output evidence persistence failed: '+repr(secondary)[:512]+'\n').encode())
        except BaseException: pass


def acquire(p, plan_ref, review_ref):
    """Exactly one nonblocking original-inode acquisition, never old proof/fallback."""
    fd = None; owner = None; lock = None
    try:
        c.require(str(p.LOCK) == core.LOCK_PATH, 'Original lock path unchanged')
        before = canonical_lock()
        c.require(dict(device=before.st_dev,inode=before.st_ino)==ORIGINAL_LOCK,'Pinned original lock inode before open')
        fd = os.open(core.LOCK_PATH, os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC)
        c.require(not os.get_inheritable(fd), 'Actual parent descriptor noninheritable across exec')
        info = os.fstat(fd)
        c.require(dict(device=info.st_dev,inode=info.st_ino)==ORIGINAL_LOCK
                  and c.identity(before) == c.identity(info) and stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
                  'Original inode unchanged across open')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        owner = p.proc(os.getpid()); lock = dict(device=info.st_dev, inode=info.st_ino)
        first = observe(p, owner, lock, fd)
        return fd, owner, lock, first
    except BaseException as original:
        preoutput_failure(original, owner, lock, fd, plan_ref, review_ref)
        if fd is not None:
            try: os.close(fd)  # No child exists. Last close releases this acquisition.
            except BaseException as secondary:
                try: os.write(2, ('Dory pre-output descriptor close failed: '+repr(secondary)[:512]+'\n').encode())
                except BaseException: pass
        raise


def initial_record(first, owner, lock, fd, plan_ref, review_ref):
    c.require(first['schema']=='descriptor_serial_lock_observation_v1' and first['stage']=='complete'
              and lock==ORIGINAL_LOCK and first['passed'] is True and first['owner'] == owner and first['lock'] == lock
              and first['retained_fd'] == fd and first['runtime_admitted'] is False,
              'Actual first descriptor proof binds retained fd/owner/inode')
    return dict(schema='dory_inventory_initial_descriptor_proof_v1', owner=owner, lock=lock,
                retained_fd=fd, plan=plan_ref, review=review_ref, lock_policy=copy.deepcopy(POLICY),
                first_observation=first, actual_process_exit_claimed=False)


def admission(plan, plan_ref, review_ref, owner, lock, fd, command, started, first_ref, duplicate_ref=None):
    c.require(type(fd) is int and fd >= 0 and first_ref['path'] == str(Path(plan['output_root'])/'initial-lock-proof.json'),
              'One actual retained fd and exact first proof path')
    return dict(schema='dory_inventory_descriptor_admission_v2', owner=owner, lock=lock,
        retained_fd=fd, lock_policy=copy.deepcopy(POLICY), first_lock_proof=first_ref,
        parent_first_duplicate=copy.deepcopy(duplicate_ref),
        plan=plan_ref, review=review_ref, stage='inventory', command=command,
        started_monotonic=started, original_lock_fd_inherited=False)


def validate_admission(value, plan, plan_ref, first, parent_duplicate=None):
    c.require(value['schema'] == 'dory_inventory_descriptor_admission_v2' and value['stage'] == 'inventory'
        and value['plan'] == plan_ref and value['lock_policy'] == plan['lock_policy'] == POLICY
        and value['lock']==ORIGINAL_LOCK and value['original_lock_fd_inherited'] is False and type(value['retained_fd']) is int and value['retained_fd'] >= 0,
        'Exact new inventory admission and retained parent fd policy')
    c.require(value['first_lock_proof']['path'] == str(Path(plan['output_root'])/'initial-lock-proof.json'),
              'Exact first proof receipt inside admitted tree')
    expected = initial_record(first['first_observation'], value['owner'], value['lock'], value['retained_fd'], plan_ref, value['review'])
    c.require(first == expected, 'Complete first proof, fd, owner, policy and plan/review bindings')
    duplicate_ref=value['parent_first_duplicate']
    if duplicate_ref==value['first_lock_proof']:
        c.require(first['first_observation']['global']['target_row_count']>1 and parent_duplicate is None,
                  'Initial duplicate already retained in exact first proof')
    elif duplicate_ref is not None:
        c.require(duplicate_ref['path']==str(Path(plan['output_root'])/'first-descriptor-duplicate-parent.json')
            and type(parent_duplicate)is dict and parent_duplicate['schema']=='dory_inventory_first_duplicate_observation_v1'
            and parent_duplicate['role']=='parent' and parent_duplicate['owner']==value['owner']
            and parent_duplicate['lock']==value['lock'] and parent_duplicate['retained_fd']==value['retained_fd']
            and parent_duplicate['plan']==plan_ref and parent_duplicate['review']==value['review']
            and parent_duplicate['lock_policy']==POLICY and parent_duplicate['admission']is None,
            'Pre-admission duplicate receipt cryptographically bound by new admission')
        initial_record(parent_duplicate['observations'],value['owner'],value['lock'],value['retained_fd'],plan_ref,value['review'])
        c.require(parent_duplicate['observations']['global']['target_row_count']>1,'Actual duplicate rows retained')
    else:c.require(parent_duplicate is None,'No invented duplicate receipt')
    return value


class Guard:
    """One failure record, latched denial, no per-observation growing logs."""
    def __init__(self, p, owner, lock, fd, out, plan_ref, review_ref, budget, *, role, admission_ref=None, created=lambda:True, first_observation=None):
        c.require(role in ('parent', 'worker'), 'Fixed descriptor guard role')
        self.p, self.owner, self.lock, self.fd = p, owner, lock, fd
        self.out, self.plan_ref, self.review_ref = Path(out), plan_ref, review_ref
        self.budget, self.role, self.admission_ref, self.created = budget, role, admission_ref, created
        self.failed = None; self.successful_observations = 0; self.last_target_rows = None
        c.require(lock==ORIGINAL_LOCK,'Guard binds pinned original lock inode')
        self.duplicate_observations=0;self.first_duplicate=None;self.duplicate_reference=None
        self.first_duplicate_is_acquire=False
        if first_observation is not None:
            initial_record(first_observation,owner,lock,fd,plan_ref,review_ref)
            if first_observation['global']['target_row_count']>1:
                self.first_duplicate=copy.deepcopy(first_observation);self.first_duplicate_is_acquire=True
    def pending_evidence(self):
        return self.first_duplicate if self.duplicate_reference is None else None
    def initial_persisted(self,reference):
        c.require(self.role=='parent' and reference['path']==str(self.out/'initial-lock-proof.json'),
                  'Exact persisted first parent proof reference')
        if self.first_duplicate_is_acquire:self.duplicate_reference=copy.deepcopy(reference)
        self.persist_duplicate()
    def persist_duplicate(self):
        if self.first_duplicate is None or self.duplicate_reference is not None:return
        c.require(self.created(),'Pending duplicate evidence requires created output before workload')
        report=dict(schema='dory_inventory_first_duplicate_observation_v1',role=self.role,
            owner=self.owner,lock=self.lock,retained_fd=self.fd,lock_policy=copy.deepcopy(POLICY),
            plan=self.plan_ref,review=self.review_ref,admission=self.admission_ref,
            observations=self.first_duplicate,source_policy_accepted=True,
            actual_process_exit_claimed=False,full_model_qualified=False,new_speedup_claim=False)
        encoded(report)
        c.require(lifecycle.storage_usage(self.out)+CAP<=self.budget['output_bytes'],'First duplicate receipt within original aggregate budget')
        fs=os.statvfs(self.out)
        c.require(fs.f_bavail*fs.f_frsize>=30*c.GIB+CAP,'Original NVMe floor plus first duplicate reserve')
        self.duplicate_reference=c.save(self.out/('first-descriptor-duplicate-'+self.role+'.json'),report,CAP)
    def summary(self):
        return dict(successful_observations=self.successful_observations, last_target_row_count=self.last_target_rows,
                    failed=self.failed is not None,duplicate_observations=self.duplicate_observations,
                    first_duplicate_reference=copy.deepcopy(self.duplicate_reference),
                    first_duplicate_was_acquire=self.first_duplicate_is_acquire,
                    pending_duplicate=self.pending_evidence() is not None,repeated_full_success_logs=False)
    def __call__(self):
        if self.failed is not None: raise self.failed
        try:
            result = observe(self.p, self.owner, self.lock, self.fd)
            if result['global']['target_row_count']>1:
                self.duplicate_observations+=1
                if self.first_duplicate is None:self.first_duplicate=copy.deepcopy(result)
                if self.created():self.persist_duplicate()
            self.successful_observations += 1; self.last_target_rows = result['global']['target_row_count']
            return result
        except BaseException as original:
            self.failed = original
            if not self.created():
                preoutput_failure(original, self.owner, self.lock, self.fd, self.plan_ref, self.review_ref,self.pending_evidence())
            else:
                try:
                    report = error_report(original, self.role, self.owner, self.lock, self.fd, self.plan_ref, self.review_ref, self.admission_ref)
                    report['guard_summary'] = self.summary();report['unpersisted_first_duplicate']=self.pending_evidence(); encoded(report)
                    c.require(lifecycle.storage_usage(self.out)+CAP <= self.budget['output_bytes'], 'Failure reserve within original aggregate budget')
                    fs = os.statvfs(self.out)
                    c.require(fs.f_bavail*fs.f_frsize >= 30*c.GIB+CAP, 'Original NVMe floor plus bounded evidence reserve')
                    c.save(self.out/('descriptor-lock-failure-'+self.role+'.json'), report, CAP)
                except BaseException as secondary:
                    try: os.write(2, ('Dory descriptor evidence persistence failed: '+repr(secondary)[:512]+'\n').encode())
                    except BaseException: pass
            raise
