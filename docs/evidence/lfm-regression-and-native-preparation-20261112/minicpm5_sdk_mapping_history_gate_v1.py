"""Pure entry gate before the frozen historical diagnostic's first import.

This is source wiring, not a native authenticator or actual outer/worker.
The caller must already retain the original descriptor and run the original
limits/deadline through independently reviewed bootstrap code which imports no
historical module. No lock, timer, subprocess, publication or current binding
is created here. All original modules and nine-native checks remain unchanged.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys
from research import minicpm5_sdk_mapping_observer_backend_v2 as backend

require = backend.require
ROOT = Path(__file__).resolve().parents[1]
HISTORY_NAME = 'research.minicpm5_pv_sdk_import_diagnostic_v1'
HISTORY_PATH = str(ROOT/'research/minicpm5_pv_sdk_import_diagnostic_v1.py')
LIFECYCLE_NAME = 'minicpm5_python_trace_lifecycle_v4'
ORIGINAL_LOCK = dict(device=66306, inode=8294421)
BLOCKED_BEFORE_AUTH = ('ctypes', '_ctypes', 'torch', 'numpy',
    'neuronx_distributed', 'neuronx_distributed_inference', HISTORY_NAME, LIFECYCLE_NAME)


def _load_ctypes():
    import ctypes  # Only after independent bootstrap authority below.
    return ctypes


def _load_history():
    from importlib import import_module
    # Canonical full historical import, with no fake module or AST rewriting.
    return import_module(HISTORY_NAME)


def _loaded_names():
    return tuple(sys.modules)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def validate_bootstrap(bootstrap):
    require(type(bootstrap) is dict and set(bootstrap) == {
        'schema', 'binding', 'limits', 'artifact_limits', 'outer_deadline', 'source_sha256'},
        'Exact explicitly reviewed pre-import bootstrap inputs')
    require(bootstrap['schema'] == 'minicpm5_sdk_mapping_history_bootstrap_v1'
        and bootstrap['limits'] == backend.logic.LIMITS
        and bootstrap['artifact_limits'] == backend.logic.CAPS,
        'Original complete limits and artifact caps, no bootstrap replacement')
    binding = bootstrap['binding']
    require(type(binding) is dict and binding.get('lock_identity') == ORIGINAL_LOCK
        and type(binding.get('retained_fd')) is int and binding['retained_fd'] >= 0
        and type(binding.get('outer_owner')) is dict,
        'Original descriptor and actual outer binding required, never acquired here')
    deadline = bootstrap['outer_deadline']
    require(type(deadline) in (int, float) and 0 < deadline < float('inf'),
        'Same absolute outer deadline established before bootstrap authentication')
    pins = bootstrap['source_sha256']
    require(type(pins) is dict and pins and all(type(path) is str and type(sha) is str
        and len(sha) == 64 and all(c in '0123456789abcdef' for c in sha)
        for path, sha in pins.items()), 'Whole authoritative source map, no source subset')
    for path in (HISTORY_PATH, str(Path(__file__).resolve()),
                 str(Path(backend.__file__).resolve()), str(Path(backend.logic.__file__).resolve())):
        require(path in pins, 'Original history and new gate/backend/logic source pins retained')


def validate_bootstrap_receipt(receipt, bootstrap, authority):
    """Bind an external authenticator result; do not authenticate from flags.

    The concrete authenticator and its independent bootstrap dependencies must
    themselves be source-reviewed before this gate can ever be admitted.
    """
    require(type(receipt) is dict and set(receipt) == {'schema', 'binding',
        'outer_deadline', 'source_map_sha256', 'importable_sources_authenticated',
        'original_descriptor_verified', 'original_limits_verified', 'native_authority'},
        'Exact external pre-import authentication receipt')
    require(receipt['schema'] == 'minicpm5_sdk_mapping_history_bootstrap_receipt_v1'
        and receipt['binding'] == bootstrap['binding']
        and receipt['outer_deadline'] == bootstrap['outer_deadline']
        and receipt['source_map_sha256'] == digest(bootstrap['source_sha256'])
        and receipt['importable_sources_authenticated'] is True
        and receipt['original_descriptor_verified'] is True
        and receipt['original_limits_verified'] is True,
        'Independent current bootstrap authentication bound to exact original inputs')
    backend.validate_authority_receipt(receipt['native_authority'], authority)


class PreImportFence:
    """Reject premature model/ctypes imports without inserting fake modules."""
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == prefix or fullname.startswith(prefix+'.') for prefix in BLOCKED_BEFORE_AUTH):
            raise backend.logic.ObservationError('Pre-authority native/history import forbidden: '+fullname)
        return None

    def __enter__(self):
        sys.meta_path.insert(0, self)
        return self

    def __exit__(self, kind, error, traceback):
        require(sum(item is self for item in sys.meta_path) == 1,
                'Exact temporary import fence retained until authentication returns')
        sys.meta_path.remove(self)
        return False


class HistoricalImportGate:
    """Retained caller-owned bootstrap object; every irreversible step explicit.

    guard() must continuously verify the SAME original descriptor, owner,
    resource limits and absolute deadline. authenticate_bootstrap() must verify
    real native origin/current bytes/dependencies and every importable source
    before ctypes (including its import-time PyDLL(None)) can execute. Neither
    callback may import the original history to do that work.
    """
    def __init__(self, bootstrap, authority, *, guard, authenticate_bootstrap,
                 verify_loaded_native, verify_loaded_history,
                 ctypes_loader=_load_ctypes, history_loader=_load_history,
                 loaded_names=_loaded_names):
        validate_bootstrap(bootstrap)
        require(authority is not None and all(callable(fn) for fn in
            (guard, authenticate_bootstrap, verify_loaded_native, verify_loaded_history,
             ctypes_loader, history_loader, loaded_names)), 'No implicit bootstrap/native authority')
        self.bootstrap = copy.deepcopy(bootstrap)
        self.authority = copy.deepcopy(authority)
        self.guard = guard
        self.authenticate_bootstrap = authenticate_bootstrap
        self.verify_loaded_native = verify_loaded_native
        self.verify_loaded_history = verify_loaded_history
        self.ctypes_loader, self.history_loader, self.loaded_names = ctypes_loader, history_loader, loaded_names
        self.state = 'new'; self.receipt = None
        self.ctypes = self.libc = self.original = self.authenticated = None

    def _pre_import_clear(self):
        names = self.loaded_names()
        require(type(names) in (tuple, list) and all(type(name) is str for name in names),
                'Explicit loaded-module-name observation')
        require(not any(name == prefix or name.startswith(prefix+'.')
                        for name in names for prefix in BLOCKED_BEFORE_AUTH),
                'No historical/native preimport before independent authority')

    def load(self):
        require(self.state == 'new', 'Once-only history entry; no retry after partial native import')
        self.state = 'authenticating'
        try:
            with PreImportFence():
                self._pre_import_clear(); self.guard()
                receipt = self.authenticate_bootstrap(copy.deepcopy(self.bootstrap),
                    copy.deepcopy(self.authority), self.guard)
                validate_bootstrap_receipt(receipt, self.bootstrap, self.authority)
                self._pre_import_clear(); self.guard()
                self.receipt = copy.deepcopy(receipt)
            self.ctypes = self.ctypes_loader()
            require(self.ctypes.__file__ == backend.CTYPES_ROOT+'/__init__.py',
                    'Exact original authenticated ctypes source')
            require(self.ctypes.sizeof(self.ctypes.c_int) == 4
                and self.ctypes.sizeof(self.ctypes.c_long) == 8
                and self.ctypes.sizeof(self.ctypes.c_ulong) == 8
                and self.ctypes.sizeof(self.ctypes.c_void_p) == 8,
                'Exact original LP64 observer ABI')
            self.libc = self.ctypes.CDLL(receipt['native_authority']['libc_path'], use_errno=True)
            require(self.libc._name == backend.LIBC, 'Exact authenticated original libc handle')
            require(self.verify_loaded_native(self.ctypes, self.libc,
                copy.deepcopy(receipt['native_authority']), self.guard) is True,
                'Actual imported sources and loaded providers verified before history')
            self.guard()
            self.original = self.history_loader()
            require(self.original.__name__ == HISTORY_NAME and self.original.__file__ == HISTORY_PATH,
                    'Canonical unchanged original diagnostic, no alias/substitute')
            require(self.verify_loaded_history(self.original, copy.deepcopy(self.bootstrap['source_sha256']),
                self.ctypes, self.guard) is True, 'Full actual history source identities verified')
            self.guard(); self.state = 'history_loaded'
            return self.original
        except BaseException:
            self.state = 'failed'
            # Caller retains this object, its descriptor and all partial native
            # handles. This gate never closes or reacquires that descriptor.
            raise

    def authenticate_original(self, spec_ref, review_ref, *, lease, metadata, guard):
        """Run frozen complete original authentication; never call old run().

        The caller adopts its existing bootstrap fd into the original Lease;
        it must not acquire another lock or call impose_original_limits again.
        The original nine-native and all historical reconstruction execute in
        the unmodified original.auth.authenticate path below.
        """
        require(self.state == 'history_loaded', 'Actual authenticated history required first')
        self.state = 'full_authenticating'
        try:
            require(lease.admission == self.bootstrap['binding'] and guard.lease is lease
                and lease.metadata is metadata, 'Adopt exact original descriptor/owner and shared guard')
            self.guard(); guard(force=True)
            spec, review = metadata.read(spec_ref), metadata.read(review_ref)
            self.original.validate_spec(spec_ref, review_ref, spec, review)
            authenticated = self.original.auth.authenticate(spec['pv_spec'], spec['pv_review'],
                lease=lease, metadata=metadata, guard=guard)
            self.original.bind_authenticated(spec, authenticated)
            pins = authenticated['context']['pins_plan']['source_sha256']
            require(pins == self.bootstrap['source_sha256'],
                    'Original full historical source map exactly equals pre-import map; no omitted/replaced pins')
            require(authenticated['witness']['complete_source_map_sha256'] == self.original.r.digest(pins),
                    'Original witness binds the same complete map')
            self.guard(); guard(force=True)
            self.authenticated = authenticated; self.state = 'complete_original_authentication'
            return authenticated
        except BaseException:
            self.state = 'failed'
            raise

    def adapter_inputs(self):
        require(self.state == 'complete_original_authentication' and self.authenticated is not None,
                'No observed adapter input before complete frozen original authentication')
        return dict(original=self.original, authenticated=self.authenticated,
            outer_deadline=self.bootstrap['outer_deadline'], authority=copy.deepcopy(self.authority))
