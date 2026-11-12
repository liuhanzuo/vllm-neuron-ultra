"""Exact SDK initialization delta and the pinned SDK context restoration fix.

No native import or launch. The original leaf, trace options and clean process
environment stay unchanged. Only the SDK context's incorrect restore statement
is corrected in memory for its two original trace call sites.
"""
import contextlib
import hashlib
from pathlib import Path
import sys
import threading

SITE = Path('/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages')
CONTEXTS = str(SITE / 'torch_neuronx/contexts.py')
# Full source pins are filled from the reviewed installed Python source files.
SOURCE_SHA256 = {'/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/libneuronxla/__init__.py': 'ce74235a966860162ad075118461d96cc79d91cc88dc61cff3793290a51d7f54', '/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/libneuronxla/hook.py': 'ddb8b401223aba91b57b284be8e61c113cc0fe2a7d43c10ea44dafda642ae582', '/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/libneuronxla/libneuronpjrt_path.py': 'd2eb6ccf4796e396927243c57843d46f60f0457f87cc5da34a56076bd781c62c', '/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/libneuronxla/version.py': '0e1ec41dac19b19e245c63704d5f20e1cfed265e03b9dab853008f02cc739ee8', '/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/torch/version.py': 'ba23e699ea8931779f14ccfcb3ed6dc67a14452d5e49ebd91875d418dba2a7f1', '/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/torch_neuronx/__init__.py': '1842b2414de6a85a259d04a6786385f67e78513345bd6935be4351171f42d8ca', '/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/torch_neuronx/contexts.py': '67a256026207cfd206f1303ec2f084e22ac7be8740ac98717c96c7b14291f81e', '/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/torch_neuronx/initialization/__init__.py': '9318cd15d847544734b2d4732156860a5248b78cfd3f2492e2cf93ac5ad243d3', '/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/torch_neuronx/initialization/initializer.py': '34b4c73980c4513696569f4cf2ef51fe9de5952a9957a9af706e2457fcb90116', '/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/torch_neuronx/utils/utils.py': '834f4ace222bd6c6079432b2c438bf040f5cda2843e8b87e7d727360c5d06f2a', '/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/torch_neuronx/xla.py': 'cc5f040784603de5817817874bff18e7f79d469ed45d41836d72979e82d97a01', '/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/torch_neuronx/xla_impl/trace.py': '5bab452aaae0a73883c5ae6c3e57ed86fbe121095c3a622ad6c3675d2d390786', '/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/torch_xla/__init__.py': 'cbde3e68a24209523c66db254ef3060fa77af3b8d50702fe45bf78c2dbd4e279', '/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/torch_xla/_internal/neuron.py': 'ed0f6d4dcf972e7c13b90ca8bfc28cf2ef1c3aa83672567fd38af17604f03d13', '/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/torch_xla/_internal/pjrt.py': '6b3b29a771252ed6bc10ff0a28b5d6d129b1deef4f6ffbd4244788b5895ef073', '/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/torch_xla/experimental/plugins.py': '10f15e65ce6a2e5e5ea5e075aff076038bfc139faddcc324385a8fc39f14e7ab', '/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/torch_xla/runtime.py': 'e9cca412711ed12310033280afda70387316ad336ea5618a8d61d39db4b9d3bc', '/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/torch_xla/version.py': 'b32ec3638fa05c17246fe549b6c95db6c5b58238fff663836f1646997a42b4e7'}
INITIALIZED = {
    'TF_CPP_MIN_LOG_LEVEL': '1',
    'GRPC_VERBOSITY': 'ERROR',
    'XLA_FLAGS': ' --xla_cpu_enable_fast_math=false --xla_gpu_force_compilation_parallelism=8',
    'XLA_DISABLE_FUNCTIONALIZATION': '1',
    'XLA_IR_SHAPE_CACHE_SIZE': '20480',
    'PJRT_DEVICE': 'NEURON',
    'NEURON_LIBRARY_PATH': str(SITE / 'libneuronxla/libneuronpjrt.so'),
    'NEURON_RT_ROOT_COMM_ID': 'localhost:62182',
    'ALLREDUCE_GRADIENTS_BUCKET_SIZE_MB': '50',
    'JAX_LOGGING_LEVEL': 'NOTSET',
}
# torch_xla.__init__ selects this one finite branch using its native build flag.
# Its actual presence is latched once; subsequent changes are rejected.
OPTIONAL_INITIALIZED = {'PJRT_DYNAMIC_PLUGINS': '1'}
MOCK_TRACE = {'NEURON_INTERNAL_MOCK_TPU': '1', 'TPU_NUM_DEVICES': '1'}
CPU_TRACE = {'PJRT_DEVICE': 'CPU'}


def require(value, message):
    if not value:
        raise ValueError(message)


def verify_sources(pins):
    require(SOURCE_SHA256 and all(pins.get(path) == sha for path, sha in SOURCE_SHA256.items()),
            'Every SDK environment source is in the complete reviewed source map')
    for path, sha in SOURCE_SHA256.items():
        source = Path(path)
        require(source.resolve() == source and source.suffix == '.py'
                and hashlib.sha256(source.read_bytes()).hexdigest() == sha,
                'Unchanged complete SDK environment Python source: ' + path)


class SDKEnvironment:
    """A single fresh leaf: clean -> SDK import -> exact initialized state."""
    def __init__(self, clean, environ, pins):
        verify_sources(pins)
        self.clean, self.environ = dict(clean), environ
        self.phase = 'clean'
        self.initialized = None
        self.module = self.original = self.replacement = None
        self.stack = []
        self.failed = None
        self.thread = threading.get_ident()
        self._equal(self.clean)

    def _equal(self, expected):
        require(dict(self.environ) == expected, 'Exact slot SDK environment for ' + self.phase)

    def check(self, event):
        require(threading.get_ident() == self.thread, 'Original leaf thread owns SDK environment')
        if self.failed is not None:
            raise self.failed
        require(not self.stack, 'Leaf guard cannot cross an active SDK trace context')
        if self.phase == 'clean':
            self._equal(self.clean)
            if event == 'before_native_imports':
                self.phase = 'importing'
            else:
                require(not event.startswith('before_compile:') and event != 'after_leaf',
                        'SDK import transition precedes compilation and completion')
        elif self.phase == 'importing':
            require(event.startswith('before_compile:'), 'First post-import boundary is the original compile guard')
            expected = dict(self.clean, **INITIALIZED)
            require(not set(self.clean) & (set(INITIALIZED) | set(OPTIONAL_INITIALIZED)),
                    'SDK defaults cannot replace fixed clean input variables')
            for key, value in OPTIONAL_INITIALIZED.items():
                if key in self.environ:
                    require(self.environ[key] == value, 'Exact SDK optional initialization branch')
                    expected[key] = value
            self._equal(expected)
            self.initialized = expected
            self.phase = 'initialized'
        else:
            require(self.phase == 'initialized', 'One exact initialized SDK state')
            self._equal(self.initialized)
        if self.module is not None:
            require(self.module.environment is self.replacement, 'Pinned SDK context fix remains installed')

    def install_context_fix(self, module):
        require(self.phase == 'initialized' and self.module is None and not self.stack,
                'Install context restoration once after authenticated SDK initialization')
        require(module is not None and module.__file__ == CONTEXTS, 'Exact imported SDK contexts source')
        original = module.environment
        unwrapped = getattr(original, '__wrapped__', None)
        require(unwrapped is not None and unwrapped.__code__.co_filename == CONTEXTS
                and unwrapped.__code__.co_name == 'environment', 'Original SDK environment context object')
        callers = {}
        for name in ('mock_neuron_cores', 'set_pjrt_device'):
            function = getattr(module, name)
            inner = getattr(function, '__wrapped__', None)
            require(inner is not None and inner.__code__.co_filename == CONTEXTS
                    and inner.__code__.co_name == name, 'Original exact SDK trace context caller')
            callers[name] = inner.__code__

        def replacement(values):
            caller = sys._getframe(1).f_code
            if caller is callers['mock_neuron_cores']:
                require(type(values) is dict and values == MOCK_TRACE and not self.stack,
                        'Only the original outer mock-Neuron trace context')
            else:
                require(caller is callers['set_pjrt_device'] and type(values) is dict
                        and values == CPU_TRACE and len(self.stack) == 1
                        and self.stack[0] == MOCK_TRACE, 'Only the original nested CPU lowering context')
            return self._context(dict(values))

        self.module, self.original, self.replacement = module, original, replacement
        module.environment = replacement

    def _current(self):
        expected = dict(self.initialized)
        for values in self.stack:
            expected.update(values)
        return expected

    @contextlib.contextmanager
    def _context(self, values):
        require(threading.get_ident() == self.thread and self.module.environment is self.replacement,
                'Same original leaf thread and installed trace context')
        self._equal(self._current())
        previous = {key: self.environ.get(key) for key in values}
        self.environ.update(values)
        self.stack.append(values)
        original_error = None
        try:
            yield
        except BaseException as error:
            original_error = error
            raise
        finally:
            mismatch = None
            try:
                self._equal(self._current())
            except BaseException as error:
                mismatch = error
            # Same SDK enter/exit semantics, with the one incorrect update
            # replaced by assignment to the actual saved environment key.
            for key, value in previous.items():
                if value is None:
                    self.environ.pop(key, None)
                else:
                    self.environ[key] = value
            self.stack.pop()
            try:
                self._equal(self._current())
            except BaseException as error:
                mismatch = mismatch or error
            if mismatch is not None:
                self.failed = mismatch
                if original_error is None:
                    raise mismatch

    def close(self):
        """Restore only the exact function installed by this leaf, even on error."""
        require(not self.stack, 'All original SDK trace contexts have unwound')
        if self.module is not None:
            require(self.module.environment is self.replacement, 'Do not replace another context owner')
            self.module.environment = self.original
            self.module = self.original = self.replacement = None
        if self.failed is not None:
            raise self.failed
