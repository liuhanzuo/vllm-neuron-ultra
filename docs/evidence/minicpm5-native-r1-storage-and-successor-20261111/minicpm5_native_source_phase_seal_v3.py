"""Bounded source-phase snapshots; complete alias digests remain unchanged.

Only objects consumed in the authentication loop are copied. Complete resolved
history is validated and hashed as a stream at entry and exit, never sealed.
The original dynamic guards, file hashing, failure latch and model gate remain.
"""
import hashlib
import json
import math
from types import MappingProxyType
from research.minicpm5_native_source_phase_seal_v1 import SourcePhase as OriginalPhase, life
require = life.require


def validate_json(value, active=None, done=None):
    """Exact finite JSON types, cycles rejected; visit shared containers once."""
    if type(value) in (str, int, bool, type(None)):
        return
    if type(value) is float:
        require(math.isfinite(value), 'Finite sealed JSON number')
        return
    require(type(value) in (dict, list), 'Only exact JSON types; no custom containers')
    active = set() if active is None else active
    done = set() if done is None else done
    key = id(value)
    require(key not in active, 'No cyclic static JSON')
    if key in done:
        return
    active.add(key)
    try:
        if type(value) is dict:
            require(all(type(k) is str for k in value), 'Exact string JSON keys')
            children = value.values()
        else:
            children = value
        for item in children:
            validate_json(item, active, done)
    finally:
        active.remove(key)
    done.add(key)


def stream_digest(value):
    """Byte-identical to original compact sorted JSON SHA256, without full str/bytes."""
    validate_json(value)
    result = hashlib.sha256()
    encoder = json.JSONEncoder(sort_keys=True, separators=(',', ':'), allow_nan=False)
    for chunk in encoder.iterencode(value):
        result.update(chunk.encode())
    return result.hexdigest()


def seal_json(value, active=None, memo=None):
    """One private immutable snapshot per distinct JSON container, no cycles."""
    if type(value) in (str, int, bool, type(None)):
        return value
    if type(value) is float:
        require(math.isfinite(value), 'Finite sealed JSON number')
        return value
    require(type(value) in (dict, list), 'Only exact JSON types; no custom containers')
    active = set() if active is None else active
    memo = {} if memo is None else memo
    key = id(value)
    require(key not in active, 'No cyclic static JSON')
    if key in memo:
        return memo[key]
    active.add(key)
    try:
        if type(value) is list:
            result = tuple(seal_json(item, active, memo) for item in value)
        else:
            require(all(type(k) is str for k in value), 'Exact string JSON keys')
            result = MappingProxyType({k: seal_json(v, active, memo) for k, v in value.items()})
        memo[key] = result
        return result
    finally:
        active.remove(key)


class SourcePhase(OriginalPhase):
    def _enter(self):
        require(not self.started, 'One source phase entry')
        self.started = True
        context = self.context
        sources = self.sources
        require(context._sources_checked is False, 'No model permission inside source phase')
        context.guard('native')
        self.original_plan = context.plan
        self.original_external_guard = context.external_guard
        self.original_context_guard = context.guard
        self.original_permit = context.permit
        self.lease_callback = self.lease.check
        self.storage_callback = self.ports.storage_available
        require(self.ports.resolved is self.resolved_alias and self.ports.contract is self.contract_alias,
                'Exact original ports bindings')
        values = dict(contract=self.contract_alias, resolved=self.resolved_alias,
                      context_plan=self.original_plan, partition=self.partition_alias, permit=self.original_permit)
        # Reject unsupported values before invoking ordinary serialization.
        self.snapshots = seal_json(dict(contract=self.contract_alias, context_plan=self.original_plan,
            partition=self.partition_alias, catalog=self.resolved_alias['native_runtime_output_catalog']))
        self.initial_digests = MappingProxyType({key: stream_digest(value) for key, value in values.items()})
        life.binding(self.contract_alias, self.resolved_alias)
        require(stream_digest(self.original_plan) == self.resolved_alias['plan_sha256']
                and self.original_plan == self.resolved_alias['plan'], 'Context copy and resolved plan agree completely')
        combined = sources.check_partition(self.partition_alias)
        require(combined == self.original_plan['source_sha256'], 'Lossless complete phase source map')
        self.plan = self.snapshots['context_plan']
        self.static_contract = self.snapshots['contract']
        self.static_catalog = self.snapshots['catalog']
        self.partition = self.snapshots['partition']
        self.writers = seal_json(self._writers())
        self.phase_guard = self.guard
        def context_guard(role=None):
            if self.failure is not None:
                raise self.failure
            try:
                return self.original_context_guard(role)
            except BaseException as error:
                self._latch(error)
                raise
        self.phase_context_guard = context_guard
        context.plan = self.plan
        context.external_guard = self.phase_guard
        context.guard = self.phase_context_guard
        return self

    def __exit__(self, kind, error, traceback):
        try:
            if kind is not None:
                self._latch(error)
                return False
            if self.failure is not None:
                raise self.failure
            require(self.authenticated, 'No successful phase exit before all original source hashes')
            self.context.guard()
            values = dict(contract=self.contract_alias, resolved=self.resolved_alias,
                          context_plan=self.original_plan, partition=self.partition_alias, permit=self.original_permit)
            for name, value in values.items():
                require(stream_digest(value) == self.initial_digests[name], 'Mutable alias changed before source phase exit: ' + name)
            life.binding(self.contract_alias, self.resolved_alias)
            self.sources.check_partition(self.partition_alias)
        except BaseException as failure:
            self._latch(failure)
            raise
        finally:
            self.closed = True
            self.context._sources_checked = False
            if self.failure is None:
                self.context.plan = self.original_plan
                self.context.external_guard = self.original_external_guard
                self.context.guard = self.original_context_guard
        # Native authentication receipt validation remains the original caller's
        # next step. This prototype never enables model/weight operations.
        self.__dict__.clear()
        self.started = self.closed = self.authenticated = True
        self.failure = None
        return False
