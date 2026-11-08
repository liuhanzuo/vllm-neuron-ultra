"""Exact EngineCore entry/alias witness for the existing Ray trace owner.

No model or native import, process observation, or trace installation on import.
The future protocol must implement the two explicit title transaction methods.
This module never owns sys.settrace and never bypasses context authentication.
"""
import inspect
import struct
from pathlib import Path
import sys
import threading
import types

import smollm3_enginecore_title_transition_v1 as title

require = title.require
SITE = Path('/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages')
CORE = str(SITE / 'vllm/v1/engine/core.py')
UTILS = str(SITE / 'vllm/utils/__init__.py')
ENVS = str(SITE / 'vllm/envs.py')
PINS = {
    CORE: '4573563d926cafe0f1e11ebb25e862ce3bc722af9a24425dec791b9d43cfbedd',
    UTILS: '15644926d0877d9b11694fc03629717a7a47d6f995afdef141746ed385cd4ce4',
    ENVS: 'a70016521a542d7bfa4884c0f9c3efc74df8cd030f5971a58774a87ab35d3803',
}
MODULES = {CORE: 'vllm.v1.engine.core', UTILS: 'vllm.utils', ENVS: 'vllm.envs'}
EXTENSION = str(SITE / 'setproctitle/_setproctitle.cpython-310-x86_64-linux-gnu.so')
PACKAGE = str(SITE / 'setproctitle/__init__.py')
EXTENSION_IDENTITY = dict(dev=66306, ino=8268093, mode=33277, size=68784,
                          mtime_ns=1790148771481374161, ctime_ns=1790148771481374161)
NATIVE_PROBE = dict(path='/home/ubuntu/experiments/trn1-latest-models-20261031/research/smollm3-setproctitle-representation-probe-actual-r1-20261107-v1.json',
                    sha256='c9d00687ccad943e0d2f324866c43148d42664ddad9c911b36d43dfdcf06d84b')


def select(code, names):
    if not names:
        return code
    matches = [value for value in code.co_consts if type(value) is types.CodeType and value.co_name == names[0]]
    require(len(matches) == 1, 'unique original complete-module code')
    return select(matches[0], names[1:])


# CPython3.10 exposes these17 data fields on code objects.  Serialized
# marshal reference flags are deliberately not part of executable structure:
# holding a nested expected code object changes FLAG_REF without changing code.
CODE_FIELDS_310 = ('co_argcount', 'co_cellvars', 'co_code', 'co_consts',
    'co_filename', 'co_firstlineno', 'co_flags', 'co_freevars',
    'co_kwonlyargcount', 'co_linetable', 'co_lnotab', 'co_name', 'co_names',
    'co_nlocals', 'co_posonlyargcount', 'co_stacksize', 'co_varnames')


def _same_code_value(actual, expected):
    if type(actual) is not type(expected):
        return False
    kind = type(actual)
    if kind is types.CodeType:
        return _same_code_structure(actual, expected)
    if kind in (type(None), type(Ellipsis), bool, int, str, bytes):
        return actual == expected
    if kind is float:
        return struct.pack('!d', actual) == struct.pack('!d', expected)
    if kind is complex:
        return struct.pack('!dd', actual.real, actual.imag) == struct.pack('!dd', expected.real, expected.imag)
    if kind is tuple:
        return len(actual) == len(expected) and all(_same_code_value(a, b) for a, b in zip(actual, expected))
    if kind is frozenset:
        remaining = list(expected)
        if len(actual) != len(remaining):
            return False
        for value in actual:
            matches = [i for i, other in enumerate(remaining) if _same_code_value(value, other)]
            if not matches:
                return False
            remaining.pop(matches[0])
        return not remaining
    # Unknown mutable/implementation-specific constant types are never accepted.
    return False


def _constant_order_key(value):
    """Unique value ordering only; ambiguous unordered constants fail closed.

    This is the pinned-source contract, not a general graph-isomorphism solver.
    In particular, two distinct NaNs with identical bits in one frozenset are
    ambiguous and rejected even if some global matching might exist.
    """
    kind = type(value)
    if kind in (type(None), type(Ellipsis)):
        return (kind.__name__,)
    if kind in (bool, int, str, bytes):
        return (kind.__name__, value)
    if kind is float:
        return ('float', struct.pack('!d', value))
    if kind is complex:
        return ('complex', struct.pack('!dd', value.real, value.imag))
    if kind is tuple:
        return ('tuple', tuple(_constant_order_key(item) for item in value))
    if kind is frozenset:
        keys = tuple(_constant_order_key(item) for item in value)
        if len(set(keys)) != len(keys):
            raise ValueError('ambiguous unordered constant values')
        return ('frozenset', tuple(sorted(keys, key=repr)))
    # Code objects are valid ordered co_consts members, but are outside this
    # deliberately narrow unordered-member ordering contract.
    raise ValueError('unsupported unordered constant type')


def _same_code_structure(actual, expected):
    left_to_right = {}
    right_to_left = {}

    def bind(left, right):
        """Return whether seen, preserving a bijection of internal objects."""
        li, ri = id(left), id(right)
        if li in left_to_right or ri in right_to_left:
            if left_to_right.get(li) != ri or right_to_left.get(ri) != li:
                raise ValueError('internal constant sharing changed')
            return True
        left_to_right[li] = ri
        right_to_left[ri] = li
        return False

    def constant(left, right):
        if type(left) is not type(right):
            return False
        if type(left) is types.CodeType:
            return code(left, right)
        if bind(left, right):
            return True
        if type(left) is tuple:
            return len(left) == len(right) and all(constant(a, b) for a, b in zip(left, right))
        if type(left) is frozenset:
            lk = [_constant_order_key(item) for item in left]
            rk = [_constant_order_key(item) for item in right]
            if len(set(lk)) != len(lk) or len(set(rk)) != len(rk):
                raise ValueError('ambiguous unordered constant values')
            la, ra = dict(zip(lk, left)), dict(zip(rk, right))
            return la.keys() == ra.keys() and all(constant(la[key], ra[key]) for key in sorted(la, key=repr))
        return _same_code_value(left, right)

    def code(left, right):
        if type(left) is not types.CodeType or type(right) is not types.CodeType:
            return False
        if bind(left, right):
            return True
        wanted = set(CODE_FIELDS_310)
        for value in (left, right):
            fields = {name for name in dir(value) if name.startswith('co_') and not callable(getattr(value, name))}
            if fields != wanted:
                return False
        # Ordinary fields are compared by strict value. Their computed getter
        # products (notably co_lnotab) are not internal constant graph nodes.
        if not all(_same_code_value(getattr(left, name), getattr(right, name))
                   for name in CODE_FIELDS_310 if name != 'co_consts'):
            return False
        return constant(left.co_consts, right.co_consts)

    try:
        return code(actual, expected)
    except ValueError:
        return False


def same_code(actual, expected, path):
    require(sys.version_info[:2] == (3, 10), 'exact CPython3.10 code-field contract')
    require(type(actual) is types.CodeType and type(expected) is types.CodeType
            and actual.co_filename == expected.co_filename == path and _same_code_structure(actual, expected),
            'exact original compiled source code')


def module_binding(modules, name, path):
    module = modules.get(name)
    require(type(module) is types.ModuleType and module.__name__ == name and module.__file__ == path,
            'exact imported module and file')
    spec = module.__spec__
    require(spec is not None and spec.name == name and spec.origin == path and not getattr(spec, '_initializing', True),
            'actual original module completed naturally')
    return module


class EngineCoreTitleHook:
    """One source-bound core call and one original setter invocation per child."""
    def __init__(self, context, modules=None):
        require(context.kind == 'server' and context.spawned is True, 'only actual admitted spawned server')
        title.permit_binding(context.permit)
        require(not hasattr(context, 'title_witness') or context.title_witness is None, 'one title witness')
        self.context = context
        self.modules = sys.modules if modules is None else modules
        self.thread = threading.get_ident()
        self.codes = {}
        for path, wanted in PINS.items():
            require(context.plan['source_sha256'].get(path) == wanted, 'title source in complete original closure')
            raw = context.source_bytes(Path(path), wanted)
            self.codes[path] = compile(raw, path, 'exec', dont_inherit=True, optimize=0)
        self.core_code = select(self.codes[CORE], ('EngineCoreProc', 'run_engine_core'))
        self.setter_code = select(self.codes[UTILS], ('set_process_title',))
        self.module_started = self.module_returned = self.entered = self.called = self.committed = self.failed = False
        self.original = self.wrapper = self.engine = self.core = self.utils = None
        self.native = self.package = self.native_setter = None
        self.trace_owner = None

    def needs_event(self, frame):
        return frame.f_code.co_filename == CORE and (frame.f_code.co_name == '<module>' or frame.f_code.co_name == 'run_engine_core')

    def set_trace_owner(self, owner):
        require(self.trace_owner is None and callable(owner), 'one original Ray trace owner binding')
        self.trace_owner = owner

    def live(self):
        require(not self.failed and threading.get_ident() == self.thread, 'same-thread live title witness')
        require(self.trace_owner is not None and sys.gettrace() is self.trace_owner, 'existing Ray trace remains sole owner')
        require(self.context.title_witness is self, 'same context-owned title witness')
        self.context.poll()

    def event(self, frame, event, argument):
        """Called by Ray's callback while tracing is suspended by CPython."""
        if self.failed:
            return
        require(self.needs_event(frame), 'finite EngineCore event only')
        require(threading.get_ident() == self.thread and not self.failed, 'same-thread original entry event')
        try:
            if frame.f_code.co_name == '<module>':
                same_code(frame.f_code, self.codes[CORE], CORE)
                if event == 'call':
                    require(not self.module_started and not self.entered, 'no EngineCore reload or second module execution')
                    self.module_started = True
                elif event == 'return':
                    require(self.module_started and not self.module_returned and argument is None, 'one ordinary EngineCore module return')
                    self.module_returned = True
                elif event == 'exception':
                    # An import may catch an exception; a completed module and
                    # full callable binding are still required at actual entry.
                    pass
                return
            same_code(frame.f_code, self.core_code, CORE)
            if event == 'call':
                require(not self.entered and not self.called, 'one actual original EngineCore entry')
                self.authenticate_entry(frame)
                self.entered = True
                self.install_alias()
            elif event == 'line':
                require(self.entered and self.core is not None and frame.f_globals is vars(self.core), 'original active EngineCore frame')
                require(self.core.set_process_title is self.wrapper, 'installed exact alias unchanged')
                if frame.f_lineno == 698:
                    require(self.committed, 'engine initialization cannot proceed before committed title')
            elif event == 'return':
                require(self.entered, 'return of previously observed entry')
        except BaseException:
            self.failed = True
            raise

    def authenticate_entry(self, frame):
        self.live()
        self.core = module_binding(self.modules, MODULES[CORE], CORE)
        self.utils = module_binding(self.modules, MODULES[UTILS], UTILS)
        envs = module_binding(self.modules, MODULES[ENVS], ENVS)
        require(frame.f_globals is vars(self.core) and frame.f_builtins is self.core.__builtins__, 'actual original engine globals/builtins')
        declaration = inspect.getattr_static(self.core.EngineCoreProc, 'run_engine_core')
        require(type(declaration) is staticmethod, 'original EngineCore staticmethod binding')
        self.engine = declaration.__func__
        require(type(self.engine) is types.FunctionType and self.engine.__globals__ is vars(self.core)
                and self.engine.__closure__ is None and self.engine.__defaults__ is None
                and self.engine.__kwdefaults__ == {'dp_rank': 0, 'local_dp_rank': 0}, 'original EngineCore function closure/defaults')
        same_code(self.engine.__code__, self.core_code, CORE)
        require(frame.f_code is self.engine.__code__ and frame.f_locals['dp_rank'] == 0 and frame.f_locals['local_dp_rank'] == 0,
                'actual original single-DP entry')
        require(type(frame.f_locals['dp_rank']) is int and type(frame.f_locals['local_dp_rank']) is int, 'exact integer ranks')
        self.original = self.core.set_process_title
        require(self.original is self.utils.set_process_title and type(self.original) is types.FunctionType
                and self.original.__globals__ is vars(self.utils) and self.original.__closure__ is None
                and self.original.__defaults__ == ('', 'VLLM') and self.original.__kwdefaults__ is None,
                'original core alias and unchanged Python title function')
        same_code(self.original.__code__, self.setter_code, UTILS)
        require(self.utils.envs is envs and envs.VLLM_PROCESS_NAME_PREFIX == 'VLLM'
                and 'VLLM_PROCESS_NAME_PREFIX' not in self.context.actual_environment()
                and 'SPT_NOENV' not in self.context.actual_environment(), 'exact original prefix and native environment policy')
        self.bind_native()

    def bind_native(self):
        self.package = module_binding(self.modules, 'setproctitle', PACKAGE)
        self.native = module_binding(self.modules, 'setproctitle._setproctitle', EXTENSION)
        self.native_setter = self.native.setproctitle
        require(self.utils.setproctitle is self.package and self.package.setproctitle is self.native_setter
                and type(self.native_setter) is types.BuiltinFunctionType and self.native_setter.__name__ == 'setproctitle'
                and self.package.__version__ == '1.3.7', 'actual original extension builtin alias')
        # The old probe authenticated this binary. This checks its unchanged
        # file identity, not a claim that source fixtures authenticate a binary.
        file = Path(EXTENSION).stat()
        actual = dict(dev=file.st_dev, ino=file.st_ino, mode=file.st_mode, size=file.st_size,
                      mtime_ns=file.st_mtime_ns, ctime_ns=file.st_ctime_ns)
        require(actual == EXTENSION_IDENTITY, 'independently hashed extension unchanged')
        self.context.source_bytes(Path(NATIVE_PROBE['path']), NATIVE_PROBE['sha256'])

    def check_binding(self):
        require(self.entered and self.core.set_process_title is self.wrapper and self.utils.set_process_title is self.original,
                'single installed core alias and unchanged original utils alias')
        require(self.original.__code__ == self.setter_code and self.original.__defaults__ == ('', 'VLLM')
                and self.original.__globals__ is vars(self.utils) and self.original.__closure__ is None,
                'original setter remains unchanged')
        require(self.utils.setproctitle is self.package and self.package.setproctitle is self.native_setter
                and self.native.setproctitle is self.native_setter, 'unchanged original native setter binding')

    def install_alias(self):
        require(self.entered and self.wrapper is None, 'single alias installation at original entry')
        def wrapped(*args, **kwargs):
            frame = sys._getframe(1)
            try:
                self.live(); self.check_binding()
                require(not self.called and frame.f_code is self.engine.__code__ and frame.f_globals is vars(self.core)
                        and frame.f_lineno == 697, 'only one exact original EngineCore caller line')
                require(args == ('EngineCore',) and kwargs == {}, 'original exact setter arguments')
                local = frame.f_locals
                config = local['kwargs']['vllm_config'].parallel_config
                require(local['parallel_config'] is config and type(config.data_parallel_size) is int
                        and config.data_parallel_size == 1 and type(local['dp_rank']) is int and local['dp_rank'] == 0
                        and type(local['local_dp_rank']) is int and local['local_dp_rank'] == 0, 'original single-DP config branch')
                self.called = True
                witness = dict(core_source_sha256=PINS[CORE], utils_source_sha256=PINS[UTILS], envs_source_sha256=PINS[ENVS],
                    native_binding=dict(package='setproctitle', version='1.3.7', extension=EXTENSION,
                        file_identity=EXTENSION_IDENTITY, setter_is_original_builtin=True),
                    call=dict(function='EngineCoreProc.run_engine_core', line=697, data_parallel_size=1, dp_rank=0,
                        local_dp_rank=0, args=['EngineCore'], kwargs={}, defaults=['', 'VLLM']),
                    original_callable_verified=True, single_original_call=True)
                title.source_witness(witness)
                # These are dedicated protocol methods, not a permissive writer
                # callback. They must retain all ordinary guards/resource checks.
                token = self.context.begin_engine_title(self, witness)
                self.check_binding()
                try:
                    result = self.original(*args, **kwargs)
                except BaseException as primary:
                    try: self.context.fail_engine_title(self, token, type(primary).__name__)
                    except BaseException as secondary: primary.smollm3_title_secondary_error = type(secondary).__name__
                    raise
                self.check_binding()
                require(result is None, 'exact original title setter return')
                self.context.complete_engine_title(self, token, result)
                self.committed = True
                self.live()
                return result
            except BaseException:
                self.failed = True
                raise
            finally:
                del frame
        self.wrapper = wrapped
        self.core.set_process_title = wrapped


def run_admitted(*args, **kwargs):
    raise RuntimeError('source component only; dedicated title protocol and full lifecycle integration still required')
