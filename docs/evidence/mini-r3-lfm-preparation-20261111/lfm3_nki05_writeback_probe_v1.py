"""Inert source-only proposal for a separately admitted simulator diagnostic.

No SDK/framework import, launch or IO on import. A future caller supplies the
already authenticated simulator/NumPy modules after original admission. Only
one exact omitted-precision BF16 input-writeback call site can be substituted.
"""
import ast
import hashlib
from pathlib import Path
import sys
from types import CodeType

MISSING = object()
MAX_BYTES = 256 * 2**10
MAX_EVENTS = 160


def require(value, message):
    if not value:
        raise RuntimeError(message)


def bit_difference(before, after, dtype):
    """Inspect every word and differing bit; never normalize floating values."""
    require(type(before) is type(after) is bytes and len(before) == len(after) <= MAX_BYTES,
            'Complete equally sized bounded byte operands')
    require(dtype in ('bfloat16', 'float32', 'int32'), 'Exact diagnostic operand dtype')
    width = 2 if dtype == 'bfloat16' else 4
    require(len(before) % width == 0, 'Whole typed words')
    result = dict(bytes=len(before), before_sha256=hashlib.sha256(before).hexdigest(),
                  after_sha256=hashlib.sha256(after).hexdigest(), words=len(before)//width,
                  changed_words=0, changed_bits=0, bit_position_counts=[0]*(width*8),
                  both_nan=0, nan_to_non_nan=0, non_nan_to_nan=0,
                  other_changed_words=0, sign_bit_changes=0, signed_zero_changes=0, first8=[])
    if before == after:
        return result
    exp, frac = (0x7f80, 0x7f) if width == 2 else (0x7f800000, 0x7fffff)
    sign = 1 << (width*8-1)
    for offset in range(0, len(before), width):
        a = int.from_bytes(before[offset:offset+width], 'little')
        b = int.from_bytes(after[offset:offset+width], 'little')
        changed = a ^ b
        if not changed:
            continue
        result['changed_words'] += 1
        result['changed_bits'] += changed.bit_count()
        for bit in range(width*8):
            result['bit_position_counts'][bit] += bool(changed & (1 << bit))
        if dtype != 'int32':
            an = a & exp == exp and bool(a & frac)
            bn = b & exp == exp and bool(b & frac)
            key = 'both_nan' if an and bn else 'nan_to_non_nan' if an else 'non_nan_to_nan' if bn else 'other_changed_words'
            result[key] += 1
            result['sign_bit_changes'] += bool(changed & sign)
            result['signed_zero_changes'] += (a & ~sign == 0 and b & ~sign == 0)
        else:
            result['other_changed_words'] += 1
        if len(result['first8']) < 8:
            result['first8'].append(dict(word=offset//width,
                                         before=before[offset:offset+width].hex(),
                                         after=after[offset:offset+width].hex(), xor=changed))
    require(sum(result['bit_position_counts']) == result['changed_bits'], 'Complete all-bit count')
    return result


def host_snapshot(operands):
    rows = []
    for name, value in zip(('cache', 'updates', 'slots'), operands):
        raw = value.tobytes()
        require(len(raw) <= MAX_BYTES, 'Original bounded diagnostic input')
        rows.append((name, raw, value.dtype.name, tuple(value.shape)))
    require(len(rows) == 3, 'All three original operands')
    return rows


def host_differences(before, operands):
    rows = []
    for (name, raw, dtype, shape), value in zip(before, operands):
        require(value.dtype.name == dtype and tuple(value.shape) == shape,
                'Input dtype and shape cannot change during diagnostic')
        rows.append(dict(operand=name, dtype=dtype, shape=list(shape),
                         difference=bit_difference(raw, value.tobytes(), dtype)))
    require(len(rows) == 3, 'All three original operand differences')
    return rows


class WritebackProbe:
    """One serial process-local observation/shim scope, restored on every exit."""
    def __init__(self, mode):
        require(mode in ('original', 'precise_bf16_writeback'), 'Explicit separate experimental arm')
        self.mode = mode
        self.module = self.original = self.wrapper = None
        self.events = []
        self.installed = self.restored = False
        self.label = None
        self.substitutions = 0
        self.case_errors = []

    def install(self, module, numpy, path, wanted):
        require(not self.installed and self.module is None, 'One probe installation per arm')
        path = Path(path)
        require(Path(module.__file__).resolve() == path and path.suffix == '.py', 'Exact pinned simulator source path')
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == wanted, 'Unchanged installed simulator source')
        tree = ast.parse(raw)
        code = compile(raw, str(path), 'exec', dont_inherit=True, optimize=0)
        expected = {v.co_name: v for v in code.co_consts if isinstance(v, CodeType)}
        require(module._tensor_to_numpy.__code__ == expected['_tensor_to_numpy']
                and module.simulate_kernel.__code__ == expected['simulate_kernel'],
                'Actual original callable bytecode matches pinned source; no preexisting wrapper')
        require(module._tensor_to_numpy.__defaults__ == (False,)
                and module._tensor_to_numpy.__kwdefaults__ is None
                and module.simulate_kernel.__defaults__ == (1, None)
                and module.simulate_kernel.__kwdefaults__ is None,
                'Original extraction and simulation default arguments retained')
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'simulate_kernel')
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == '_tensor_to_numpy'
                 and len(n.args) == 1 and not n.keywords]
        require(len(calls) == 1, 'One exact omitted-precision writeback call in pinned simulator')
        self._install_bound(module, numpy, calls[0].lineno)

    def _install_bound(self, module, numpy, line):
        """Already authenticated binding; isolated mock tests never import NKI."""
        self.module, self.numpy = module, numpy
        self.original = module._tensor_to_numpy
        self.simulate_code = module.simulate_kernel.__code__
        self.writeback_line = line
        original = self.original

        def wrapped(tensor, precise=MISSING):
            require(self.installed and not self.restored and module._tensor_to_numpy is wrapped,
                    'Active original single probe binding')
            frame = sys._getframe(1)
            try:
                omitted = precise is MISSING
                eligible = (omitted and frame.f_code is self.simulate_code
                            and frame.f_lineno == self.writeback_line
                            and frame.f_locals.get('tensor') is tensor
                            and 'orig_arr' in frame.f_locals)
                host = frame.f_locals.get('orig_arr') if eligible else None
                caller = frame.f_code.co_name
            finally:
                del frame
            change = self.mode == 'precise_bf16_writeback' and eligible and tensor.dtype == 'bfloat16'
            result = original(tensor, precise=True) if change else (
                original(tensor) if omitted else original(tensor, precise=precise))
            self.substitutions += change
            require(len(self.events) < MAX_EVENTS, 'Finite exact extraction call inventory')
            event = dict(case=self.label, caller=caller, writeback_call=eligible,
                         requested_precise='omitted' if omitted else precise,
                         effective_precise=True if change else False if omitted else precise,
                         substitution=change, tensor_dtype=tensor.dtype,
                         storage_dtype=tensor._storage.data.dtype.name,
                         extracted_dtype=result.dtype.name, shape=list(result.shape))
            if eligible and tensor.dtype == 'bfloat16':
                # Read-only counterfactual: same pinned extractor and exact
                # np.copyto primitive, but destinations are detached clones.
                # Neither clone supplies data to the real kernel or writeback.
                stored_before = tensor._storage.data.tobytes()
                host_before = host.tobytes()
                control = {}
                for setting in (False, True):
                    extracted = original(tensor, precise=setting)
                    clone = host.copy()
                    numpy.copyto(clone, extracted)
                    control[str(setting).lower()] = dict(extracted_dtype=extracted.dtype.name,
                        difference=bit_difference(host_before, clone.tobytes(), host.dtype.name))
                require(host.tobytes() == host_before and tensor._storage.data.tobytes() == stored_before,
                        'Counterfactual extraction cannot mutate real host or storage')
                event['precision_controls'] = control
                event['storage_matches_host_before'] = stored_before == host_before
            self.events.append(event)
            return result

        self.wrapper = wrapped
        module._tensor_to_numpy = wrapped
        self.installed = True

    def restore(self):
        if not self.installed:
            return
        intact = (self.module._tensor_to_numpy is self.wrapper
                  and self.module.simulate_kernel.__code__ is self.simulate_code)
        try:
            require(intact, 'No concurrent wrapper or simulator replacement')
        finally:
            self.module._tensor_to_numpy = self.original
            self.restored = self.module._tensor_to_numpy is self.original
        require(self.restored, 'Exact original extractor object restored')

    def report(self):
        return dict(mode=self.mode, installed=self.installed, restored=self.restored,
                    substitutions=self.substitutions, extraction_calls=self.events,
                    case_errors=self.case_errors, installed_sdk_source_modified=False)
