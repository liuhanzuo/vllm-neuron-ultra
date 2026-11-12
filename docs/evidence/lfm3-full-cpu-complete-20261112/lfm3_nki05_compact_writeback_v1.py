"""Inert simulator-only writeback adapter; no native import or lifecycle.

An independently admitted caller supplies the already loaded simulator module.
Only BF16 input extraction at the pinned omitted-precision call is changed.
Four metadata records per case replace diagnostic per-tile payload captures.
Never attach this adapter to JIT, tracing, device execution, or model serving.
"""
import ast
import hashlib
from pathlib import Path
import sys
from types import CodeType, FunctionType

SDK_SOURCE = Path('/home/ubuntu/venvs/vllm_neuron_053_py312/lib/python3.12/site-packages/nki/_simulator.py')
SDK_SHA256 = '3c28f10492ef8c6c5413acbd811c059586f7ff16d3ef795430e794c57ed8f119'
MAX_PAYLOAD_BYTES = 128 * 2**20
MISSING = object()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _shape(value):
    result = tuple(value.shape)
    require(1 <= len(result) <= 2 and all(type(n) is int and n > 0 for n in result),
            'Positive bounded operand dimensions')
    return result


def _metadata(value):
    dtype, shape = value.dtype.name, _shape(value)
    require(dtype in ('bfloat16', 'float32', 'int32'), 'Only original typed operands')
    size = 2 if dtype == 'bfloat16' else 4
    count = 1
    for n in shape:
        count *= n
    require(value.dtype.itemsize == size and value.nbytes == count * size <= MAX_PAYLOAD_BYTES,
            'Exact original byte size, at most 128 MiB per operand')
    return dict(dtype=dtype, shape=list(shape), bytes=count * size)


def authenticate_simulator(module):
    """Hash and inspect Python source without importing the SDK or executing it."""
    require(Path(module.__file__).resolve() == SDK_SOURCE and not SDK_SOURCE.is_symlink()
            and SDK_SOURCE.stat().st_size < 2**20, 'Exact bounded pinned SDK source')
    raw = SDK_SOURCE.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == SDK_SHA256, 'Unchanged simulator source')
    tree = ast.parse(raw)
    compiled = compile(raw, str(SDK_SOURCE), 'exec', dont_inherit=True, optimize=0)
    expected = {c.co_name: c for c in compiled.co_consts if isinstance(c, CodeType)}
    for name, defaults in (('_tensor_to_numpy', (False,)), ('simulate_kernel', (1, None))):
        function = getattr(module, name)
        require(type(function) is FunctionType and function.__code__ == expected[name]
                and function.__globals__ is module.__dict__ and function.__defaults__ == defaults
                and function.__kwdefaults__ is None and function.__closure__ is None,
                'Actual unwrapped callable, globals, defaults and code match pinned source')
    simulate_code = module.simulate_kernel.__code__
    return_codes = [c for c in simulate_code.co_consts
                    if isinstance(c, CodeType) and c.co_name == '_validate_and_convert']
    require(len(return_codes) == 1, 'Unique actual output conversion code')
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'simulate_kernel')
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == '_tensor_to_numpy']
    omitted = [n for n in calls if len(n.args) == 1 and not n.keywords]
    explicit = [n for n in calls if len(n.args) == 1 and len(n.keywords) == 1
                and n.keywords[0].arg == 'precise'
                and isinstance(n.keywords[0].value, ast.Constant)
                and n.keywords[0].value.value is True]
    require(len(calls) == 2 and len(omitted) == len(explicit) == 1,
            'Exactly one input and one precise output callsite')
    return simulate_code, return_codes[0], omitted[0].lineno, explicit[0].lineno


class CompactWriteback:
    """One fresh instance per admitted arm or full-suite batch; no file IO."""
    def __init__(self, mode):
        require(mode in ('original', 'precise_bf16_writeback'), 'Explicit independent arm')
        self.mode = mode
        self.module = None
        self.installed = self.restored = self.failed = False
        self.case = None
        self.incomplete_case = None
        self.completed_cases = self.total_calls = self.total_substitutions = 0

    def install(self, module, admit):
        require(callable(admit) and admit() is True, 'Caller owns fresh serial runtime admission')
        require(self.module is None and not self.installed, 'Fresh single installation')
        self._install_bound(module, authenticate_simulator(module))

    def _install_bound(self, module, binding):
        """Private pure-test seam; production must use install and its source gate."""
        require(self.module is None and not self.installed, 'Fresh single installation')
        self.module, self.original, self.simulate_function = module, module._tensor_to_numpy, module.simulate_kernel
        self.original_code = self.original.__code__
        self.simulate_code, self.return_code, self.writeback_line, self.return_line = binding
        self.original_defaults = self.original.__defaults__
        self.simulate_defaults = self.simulate_function.__defaults__

        def wrapped(tensor, precise=MISSING):
            try:
                self._intact()
                require(not self.failed and self.case is not None, 'One active, nonfailed case')
                events, operands = self.case['events'], self.case['operands']
                index = len(events)
                require(index < 4, 'Exactly four extractions per selector case')
                frame = sys._getframe(1)
                try:
                    if index < 3:
                        require(frame.f_code is self.simulate_code and frame.f_lineno == self.writeback_line
                                and precise is MISSING and frame.f_locals.get('tensor') is tensor
                                and frame.f_locals.get('orig_arr') is operands[index],
                                'Exact ordered input writeback callsite, tensor and host identity')
                        reference = self.case['inputs'][index]
                        require(operands[index].flags.writeable is True and operands[index].flags.c_contiguous is True,
                                'Original writable contiguous input; never disable input writeback')
                        require(_metadata(operands[index]) == reference, 'Host metadata unchanged before writeback')
                    else:
                        require(frame.f_code is self.return_code and frame.f_lineno == self.return_line
                                and precise is True and frame.f_locals.get('tensor') is tensor,
                                'Exact pinned precise output callsite')
                        reference = self.case['inputs'][0]
                finally:
                    del frame
                require(tensor.dtype == reference['dtype'] and list(_shape(tensor)) == reference['shape'],
                        'Original tensor dtype and shape')
                require(_metadata(tensor._storage.data) == reference, 'Exact typed complete storage')
                change = index < 3 and self.mode == 'precise_bf16_writeback' and tensor.dtype == 'bfloat16'
                result = self.original(tensor, precise=True) if change else (
                    self.original(tensor) if precise is MISSING else self.original(tensor, precise=precise))
                metadata = _metadata(result)
                expected_dtype = ('float32' if index < 3 and tensor.dtype == 'bfloat16'
                                  and not change else reference['dtype'])
                require(metadata['dtype'] == expected_dtype and metadata['shape'] == reference['shape'],
                        'Extractor preserves pinned original/precise dtype and complete shape')
                events.append(dict(operand=('cache', 'updates', 'slots', 'output')[index],
                                   tensor_dtype=reference['dtype'], extracted_dtype=metadata['dtype'],
                                   shape=metadata['shape'], bytes=metadata['bytes'],
                                   requested_precise='omitted' if precise is MISSING else precise,
                                   effective_precise=True if change else False if precise is MISSING else precise,
                                   substitution=change))
                return result
            except BaseException:
                self.failed = True
                raise

        self.wrapper = wrapped
        module._tensor_to_numpy = wrapped
        self.installed = True

    def _intact(self):
        intact = (self.installed and not self.restored and self.module._tensor_to_numpy is self.wrapper
                and self.module.simulate_kernel is self.simulate_function
                and self.simulate_function.__code__ is self.simulate_code
                and self.simulate_function.__defaults__ == self.simulate_defaults
                and self.simulate_function.__kwdefaults__ is None
                and self.original.__code__ is self.original_code
                and self.original.__defaults__ == self.original_defaults
                and self.original.__kwdefaults__ is None)
        if not intact:
            self.failed = True
        require(intact, 'Original callables/defaults and unique installed wrapper remain intact')

    def begin_case(self, phase, case_id, operands):
        self._intact()
        require(not self.failed and self.case is None and self.completed_cases < 16,
                'One active case, fail-stop, at most original sixteen cases per batch')
        require(phase in ('regression_cpu', 'extended_cpu') and type(case_id) is int and case_id >= 0,
                'Original phase and case identifier')
        require(type(operands) is tuple and len(operands) == 3
                and len({id(value) for value in operands}) == 3, 'Three distinct original operands')
        metadata = [_metadata(value) for value in operands]
        cache, updates, slots = metadata
        require(cache['dtype'] in ('bfloat16', 'float32') and updates['dtype'] == cache['dtype']
                and slots['dtype'] == 'int32' and len(cache['shape']) == len(updates['shape']) == 2
                and cache['shape'][0] in (4, 8, 16) and cache['shape'][1] % 128 == 0
                and 1 <= updates['shape'][0] <= cache['shape'][0]
                and updates['shape'][1] == cache['shape'][1]
                and slots['shape'] == [updates['shape'][0]], 'Full original selector shape contract')
        require(all(x.flags.writeable is True and x.flags.c_contiguous is True for x in operands),
                'Original contiguous writable inputs only')
        self.case = dict(phase=phase, id=case_id, operands=operands, inputs=metadata, events=[])

    def end_case(self):
        self._intact()
        require(not self.failed and self.case is not None, 'Active successful extraction case')
        row = self.case
        require(len(row['events']) == 4, 'All three writebacks and returned output actually observed')
        substitutions = sum(event['substitution'] for event in row['events'])
        expected = 2 if self.mode == 'precise_bf16_writeback' and row['inputs'][0]['dtype'] == 'bfloat16' else 0
        require(substitutions == expected, 'Only both BF16 inputs use precise substitution')
        result = dict(phase=row['phase'], id=row['id'], mode=self.mode,
                      extraction_calls=4, substitutions=substitutions, events=row['events'])
        self.completed_cases += 1
        self.total_calls += 4
        self.total_substitutions += substitutions
        self.case = None
        return result

    def restore(self):
        require(self.installed and not self.restored, 'Restore each installed scope once')
        try:
            self._intact()
        finally:
            # Restoration is attempted even after a callable replacement or a
            # failed native invocation; a pending case is never marked complete.
            self.failed = self.failed or self.case is not None
            if self.case is not None:
                self.incomplete_case = dict(phase=self.case['phase'], id=self.case['id'],
                                            extraction_calls=len(self.case['events']),
                                            events=self.case['events'])
            self.module._tensor_to_numpy = self.original
            self.restored = self.module._tensor_to_numpy is self.original
            self.case = None

    def report(self):
        result = dict(mode=self.mode, installed=self.installed, restored=self.restored, failed=self.failed,
                      completed_cases=self.completed_cases, extraction_calls=self.total_calls,
                      substitutions=self.total_substitutions, installed_sdk_source_modified=False)
        if self.incomplete_case is not None:
            result['incomplete_case'] = self.incomplete_case
        return result
