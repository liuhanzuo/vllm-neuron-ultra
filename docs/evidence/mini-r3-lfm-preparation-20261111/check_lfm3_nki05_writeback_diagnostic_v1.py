"""Source-only/mocked controls; no native import or simulator execution."""
import ast
import copy
import hashlib
from pathlib import Path
import struct
import sys
from types import CodeType, FunctionType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent


def no_native(event, args):
    if event == 'import' and args[0].split('.')[0] in ('numpy', 'ml_dtypes', 'nki', 'torch', 'neuronxcc'):
        raise AssertionError('No native/framework import')
    if event.startswith('socket.') or event in ('subprocess.Popen', 'os.system', 'fcntl.flock', 'fcntl.lockf'):
        raise AssertionError('No runtime operations')


sys.addaudithook(no_native)
import lfm3_nki05_writeback_probe_v1 as probe
import lfm3_nki05_transport_diagnostic_v2 as diagnostic


class Array:
    """Byte-only synthetic conversion fixture, not a NumPy behavior claim."""
    def __init__(self, raw, dtype='bfloat16'):
        self.raw = raw
        self.dtype = SimpleNamespace(name=dtype)
        self.shape = (1, len(raw)//(2 if dtype == 'bfloat16' else 4))

    def tobytes(self): return self.raw
    def copy(self): return Array(self.raw, self.dtype.name)


def copyto(dst, source):
    if dst.dtype.name == source.dtype.name:
        dst.raw = source.raw
    else:
        assert dst.dtype.name == 'bfloat16' and source.dtype.name == 'float32'
        words = []
        for offset in range(0, len(source.raw), 4):
            word = int.from_bytes(source.raw[offset:offset+4], 'little') >> 16
            # Deliberately lossy synthetic conversion, solely for causal controls.
            if word & 0x7f80 == 0x7f80 and word & 0x7f:
                word = (word & 0x8000) | 0x7fc0
            words.append(word)
        dst.raw = struct.pack('<'+'H'*len(words), *words)


def extractor(tensor, precise=False):
    raw = tensor._storage.data.tobytes()
    if precise:
        return Array(raw)
    values = [int.from_bytes(raw[i:i+2], 'little') << 16 for i in range(0, len(raw), 2)]
    return Array(struct.pack('<'+'I'*len(values), *values), 'float32')


def simulate(module, tensor, orig_arr):
    result = module._tensor_to_numpy(tensor)
    copyto(orig_arr, result)
    return module._tensor_to_numpy(tensor, precise=True)


def fixture(mode):
    module = SimpleNamespace(_tensor_to_numpy=extractor, simulate_kernel=simulate)
    value = Array(struct.pack('<4H', 0x3f80, 0x7fc1, 0x8000, 0xffc3))
    tensor = SimpleNamespace(dtype='bfloat16', _storage=SimpleNamespace(data=value.copy()))
    controller = probe.WritebackProbe(mode)
    controller._install_bound(module, SimpleNamespace(copyto=copyto), simulate.__code__.co_firstlineno+1)
    return controller, module, tensor, value


class Tests(unittest.TestCase):
    def test_all_65536_words_and_every_bit_counted(self):
        before = b''.join(struct.pack('<H', word) for word in range(65536))
        after = b''.join(struct.pack('<H', word ^ 0xffff) for word in range(65536))
        result = probe.bit_difference(before, after, 'bfloat16')
        self.assertEqual(result['changed_words'], 65536)
        self.assertEqual(result['changed_bits'], 65536*16)
        self.assertEqual(result['bit_position_counts'], [65536]*16)
        self.assertEqual(len(result['first8']), 8)
        self.assertEqual(sum(result[key] for key in ('both_nan', 'nan_to_non_nan', 'non_nan_to_nan', 'other_changed_words')), 65536)

    def test_exact_nan_signed_zero_and_finite_differences(self):
        before = struct.pack('<4H', 0x7fc1, 0x0000, 0x3f80, 0xff80)
        after = struct.pack('<4H', 0x7fc0, 0x8000, 0x3f81, 0xffc3)
        result = probe.bit_difference(before, after, 'bfloat16')
        self.assertEqual(result['both_nan'], 1)
        self.assertEqual(result['signed_zero_changes'], 1)
        self.assertEqual(result['non_nan_to_nan'], 1)
        self.assertEqual(result['other_changed_words'], 2)
        with self.assertRaises(RuntimeError): probe.bit_difference(b'1', b'2', 'bfloat16')
        with self.assertRaises(RuntimeError): probe.bit_difference(b'00', b'00', 'uint16')

    def test_original_instrumentation_never_repairs_and_probes_are_detached(self):
        p, module, tensor, host = fixture('original')
        before = host.tobytes()
        try:
            result = simulate(module, tensor, host)
            self.assertEqual(result.tobytes(), before)
            self.assertNotEqual(host.tobytes(), before)
            self.assertEqual(tensor._storage.data.tobytes(), before)
            event = p.events[0]
            self.assertEqual((event['requested_precise'], event['effective_precise'], event['extracted_dtype']),
                             ('omitted', False, 'float32'))
            self.assertEqual(event['precision_controls']['false']['difference']['changed_words'], 2)
            self.assertEqual(event['precision_controls']['true']['difference']['changed_words'], 0)
            self.assertEqual(p.substitutions, 0)
        finally:
            p.restore()
        self.assertIs(module._tensor_to_numpy, extractor)

    def test_candidate_only_exact_omitted_writeback_and_restores(self):
        p, module, tensor, host = fixture('precise_bf16_writeback')
        before = host.tobytes()
        try:
            result = simulate(module, tensor, host)
            self.assertEqual(host.tobytes(), before)
            self.assertEqual(result.tobytes(), before)
            self.assertEqual(p.substitutions, 1)
            self.assertTrue(p.events[0]['writeback_call'])
            self.assertEqual(p.events[0]['effective_precise'], True)
            self.assertEqual(p.events[1]['requested_precise'], True)
        finally:
            p.restore()
        self.assertIs(module._tensor_to_numpy, extractor)
        simulate(module, tensor, host)
        self.assertNotEqual(host.tobytes(), before)

    def test_explicit_false_and_other_default_callers_unchanged(self):
        p, module, tensor, host = fixture('precise_bf16_writeback')
        try:
            for value in (module._tensor_to_numpy(tensor, precise=False), module._tensor_to_numpy(tensor)):
                self.assertEqual(value.dtype.name, 'float32')
            self.assertEqual(p.substitutions, 0)
            self.assertTrue(all(not event['writeback_call'] for event in p.events))
        finally:
            p.restore()

    def test_tampering_still_restores_original_before_raising(self):
        p, module, tensor, host = fixture('precise_bf16_writeback')
        module._tensor_to_numpy = lambda *args, **kwargs: None
        with self.assertRaises(RuntimeError): p.restore()
        self.assertIs(module._tensor_to_numpy, extractor)
        self.assertTrue(p.restored)

    def test_source_authentication_compiles_but_never_executes_sdk(self):
        path = diagnostic.SDK/'_simulator.py'
        raw = path.read_bytes()
        nodes = compile(raw, str(path), 'exec', dont_inherit=True, optimize=0).co_consts
        codes = {n.co_name:n for n in nodes if isinstance(n, CodeType)}
        original_fn = FunctionType(codes['_tensor_to_numpy'], {}, argdefs=(False,))
        module = SimpleNamespace(__file__=str(path), _tensor_to_numpy=original_fn,
                                 simulate_kernel=FunctionType(codes['simulate_kernel'], {}, argdefs=(1, None)))
        p = probe.WritebackProbe('original')
        p.install(module, None, path, diagnostic.PINS[path])
        self.assertEqual(p.writeback_line, 215)
        p.restore()
        self.assertIs(module._tensor_to_numpy, original_fn)
        module._tensor_to_numpy = extractor
        with self.assertRaises(RuntimeError):
            probe.WritebackProbe('original').install(module, None, path, diagnostic.PINS[path])

    def test_original_scientific_suite_ast_and_hash_unchanged(self):
        old = ROOT/'lfm3_nki05_transport_diagnostic_v1.py'
        self.assertEqual(hashlib.sha256(old.read_bytes()).hexdigest(),
                         '62fcdb4faba6c26420f2a5d29123b704b297216f7f4ae6d3c916cca4d547fb55')
        a = next(n for n in ast.parse(old.read_bytes()).body if isinstance(n, ast.FunctionDef) and n.name == 'diagnose')
        b = next(n for n in ast.parse((ROOT/'lfm3_nki05_transport_diagnostic_v2.py').read_bytes()).body
                 if isinstance(n, ast.FunctionDef) and n.name == '_suite')
        a.name = b.name
        self.assertEqual(ast.dump(a), ast.dump(b))

    def test_keyboard_interrupt_restores_wrapper_and_clears_active_arm(self):
        holder = {}
        def fail(admit):
            active = diagnostic._ACTIVE_PROBE
            module = SimpleNamespace(_tensor_to_numpy=extractor, simulate_kernel=simulate)
            active._install_bound(module, SimpleNamespace(copyto=copyto), simulate.__code__.co_firstlineno+1)
            holder['module'] = module
            raise KeyboardInterrupt('synthetic cancellation')
        with patch.object(diagnostic, '_suite', side_effect=fail):
            with self.assertRaises(KeyboardInterrupt): diagnostic.diagnose(lambda: True)
        self.assertIs(holder['module']._tensor_to_numpy, extractor)
        self.assertIsNone(diagnostic._ACTIVE_PROBE)


if __name__ == '__main__': unittest.main()
