"""Synthetic source tests: no native imports, old payload reads or disk payloads.

Real JSON binds six original case identities and expected controls. The fake
model tests orchestration only, never numerical equivalence or native behavior.
Independent stream tests use tiny freshly constructed in-memory byte fixtures.
"""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import struct
from types import SimpleNamespace
import unittest

from dory_stage_a_capture_core_v1 import (
    bind_cases, StageAObserver, layouts, TARGET_LABELS, SELECTED_LABELS,
    ORIGINAL_LABELS, CACHE_IDS, VISITS, WIDE)
from dory_stage_a_archive_core_v1 import ArchiveWriter, TensorEmitter
from dory_stage_a_saved_core_v1 import verify_archive, stage_a_requirements

ROOT = Path(__file__).resolve().parent


def load_json(pin):
    path = Path(pin['path'])
    assert path.suffix == '.json'
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == pin['sha256']
    return json.loads(data)


def metadata_cases():
    design = load_json(dict(path=str(ROOT/'dory-targeted-mixer-residual-design-20261110-v1.json'),
                           sha256='cb1a5b9be6733e8bff370bbb70db17b3d148bc64c188fd6cb640e863aeee55fe'))
    plan = load_json(design['prior_evidence']['selection_plan'])
    ids = {row[0] for row in WIDE}
    items = [item for item in design['selected_call_manifest'] if item['event'] in ids]
    originals = {item['event']: load_json(item['original_event']) for item in items}
    replays = {item['event']: load_json(item['residual_event']) for item in items}
    return design, plan, originals, replays


class FakeTensor:
    def __init__(self, shape, dtype='BF16', tag='', values=None):
        self.shape, self.dtype, self.tag, self.values = list(shape), dtype, tag, values


class FakeOps:
    def __init__(self, case):
        self.case = case

    def describe(self, tensor):
        stride, n = [], 1
        for size in reversed(tensor.shape):
            stride.insert(0, n)
            n *= size
        return dict(shape=list(tensor.shape), dtype=tensor.dtype, stride=stride, storage_offset=0)

    def integers(self, tensor):
        return list(tensor.values)

    def select(self, tensor, rows):
        assert all(0 <= r < tensor.shape[0] for r in rows)
        return FakeTensor([len(rows), tensor.shape[1]], tensor.dtype, 'head')

    def top1(self, logits):
        return list(self.case['original_top1'])


class FakeHook:
    def __init__(self, hooks, fn):
        self.hooks, self.fn = hooks, fn
        hooks.append(fn)

    def remove(self):
        self.hooks.remove(self.fn)


class FakeModule:
    def __init__(self, forward):
        self.forward, self.pre, self.post = forward, [], []

    def register_forward_pre_hook(self, fn, with_kwargs):
        assert with_kwargs
        return FakeHook(self.pre, fn)

    def register_forward_hook(self, fn, with_kwargs):
        assert with_kwargs
        return FakeHook(self.post, fn)

    def __call__(self, *args, **kwargs):
        for fn in tuple(self.pre):
            assert fn(self, args, kwargs) is None
        result = self.forward(*args, **kwargs)
        for fn in tuple(self.post):
            assert fn(self, args, kwargs, result) is None
        return result


class FakeModel:
    def __init__(self):
        self.count, self.forward_calls, self.seen_shapes = 0, 0, []
        self.caches = {i: SimpleNamespace(keys=None, values=None, start_pos=0) for i in CACHE_IDS}
        cfg = SimpleNamespace(num_hidden_layers=72, n_input_layers=16, n_recurrent_layers=40,
            n_output_layers=16, n_recurrent_loops=2, hidden_size=2560, num_attention_heads=16,
            num_key_value_heads=4, head_dim=256, sliding_window=1024, recurrent_kv_cache_mode='per_loop',
            ngpt_alpha_init=.05, initializer_range=.0177, layer_types=['sliding_attention']*72)
        layers = []
        for p in range(72):
            def mix(positions, hidden, loop_idx=0, physical=p):
                return FakeTensor(hidden.shape, tag=f'mixer-{physical}-{loop_idx}')
            mixer = FakeModule(mix)
            mixer.per_loop_kv = True
            mixer.attn = ([self.caches[p+i*72] for i in range(2 if 16 <= p < 56 else 1)]
                          if p % 2 == 0 else [])
            def layer(positions, hidden, loop_idx=0, physical=p, mixing=mixer):
                self.seen_shapes.append((physical, loop_idx, list(hidden.shape)))
                mixing(positions, hidden, loop_idx=loop_idx)
                return FakeTensor(hidden.shape, tag=f'residual-{physical}-{loop_idx}')
            block = FakeModule(layer)
            block.mixer, block.is_attention, block._alpha_scale = mixer, True, .05/.0177
            layers.append(block)
        self.backbone = SimpleNamespace(config=cfg, num_loops=2, layers=layers)
        self.order = VISITS

    def __call__(self, tokens, positions):
        self.forward_calls += 1
        self.last_tokens, self.last_positions = tokens, positions
        rows = len(tokens.values)
        h = FakeTensor([rows, 2560], tag='embedding')
        for physical, loop in self.order:
            h = self.backbone.layers[physical](positions, h, loop_idx=loop)
        for i, cache in self.caches.items():
            cache.keys = FakeTensor([4, rows, 256], tag=f'key-{i}')
            cache.values = FakeTensor([4, rows, 256], tag=f'value-{i}')
        self.count += 280
        return h

    def compute_logits(self, head):
        self.count += 1
        return FakeTensor([head.shape[0], 131072], 'F32', 'logits')

    def empty_caches(self):
        for c in self.caches.values():
            c.keys = c.values = None
            c.start_pos = 0


class FakeEmitter:
    """Metadata spy; deliberately not a payload writer or numerical oracle."""
    def __init__(self, case, failure=None):
        self.case, self.failure, self.calls = case, failure, []

    def __call__(self, scope, label, tensor, rows):
        self.calls.append((scope, label, list(tensor.shape), None if rows is None else list(rows)))
        shape_dtype = layouts(self.case)[scope][label]
        size = 2 if shape_dtype['dtype'] == 'BF16' else 4
        for n in shape_dtype['shape']:
            size *= n
        digest = (self.case[scope+'_archive']['tensors'][label]['sha256']
                  if scope in ('selected', 'original') else hashlib.sha256(tensor.tag.encode()).hexdigest())
        result = dict(**shape_dtype, finite=True, offset=0, bytes=size, sha256=digest)
        if self.failure and (scope, label) == self.failure[:2]:
            result[self.failure[2]] = self.failure[3]
        return result


class ObserverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata = metadata_cases()
        cls.cases = bind_cases(*cls.metadata)

    def setUp(self):
        self.model = FakeModel()
        self.case = self.cases[0]
        self.ops = FakeOps(self.case)
        self.observer = StageAObserver(self.model, self.model.caches, self.ops,
                                       lambda: self.model.count, lambda: None)
        self.observer.install()
        self.addCleanup(self.observer.close)
        self.tokens = FakeTensor([2060], 'I64', values=self.case['input_ids'])
        self.positions = FakeTensor([2060], 'I64', values=list(range(2060)))

    def collect(self, case=None, emitter=None):
        case = self.case if case is None else case
        return self.observer.collect(case, self.tokens, self.positions,
                                     FakeEmitter(case) if emitter is None else emitter)

    def test_all_six_full_shapes_original_objects_and_limits(self):
        for case in self.cases:
            self.model.empty_caches()
            self.ops.case = case
            rows = case['identity']['end']
            tokens = FakeTensor([rows], 'I64', values=case['input_ids'])
            positions = FakeTensor([rows], 'I64', values=list(range(rows)))
            sink = FakeEmitter(case)
            before = len(self.model.seen_shapes)
            result = self.observer.collect(case, tokens, positions, sink)
            self.assertIs(self.model.last_tokens, tokens)
            self.assertIs(self.model.last_positions, positions)
            self.assertEqual([r[2] for r in self.model.seen_shapes[before:]], [[rows, 2560]]*112)
            self.assertEqual(len(sink.calls), 347)
            self.assertEqual([r[1] for r in sink.calls if r[0] == 'target'], list(TARGET_LABELS))
            self.assertTrue(all(r[2] == [rows, 2560] and r[3] is None for r in sink.calls if r[0] == 'target'))
            self.assertEqual(len(result['records']['selected']), 224)
            self.assertEqual(len(result['records']['original']), 114)
            self.assertFalse(result['independent_saved_byte_review_complete'])
            self.assertFalse(result['original_source_weight_runtime_identity_verified_here'])
            self.assertFalse(result['production_slot_reuse_qualified'])
            self.assertFalse(result['all87_marked_calls_qualified'])
        self.assertEqual(self.model.forward_calls, 6)
        self.assertEqual(self.observer.finish()['events'], [x[0] for x in WIDE])

    def test_unfinished_capture_cannot_finish(self):
        with self.assertRaises(RuntimeError):
            self.observer.finish()

    def test_actual_original_case_binding_and_independent_layout_agreement(self):
        for case in self.cases:
            independent = stage_a_requirements(case)
            self.assertEqual({k: v['layout'] for k, v in independent.items()}, layouts(case))
        self.assertEqual(sum(c['identity']['end'] for c in self.cases), 4812)
        self.assertEqual(sum(len(c['selected_rows']) for c in self.cases), 70)

    def test_binding_rejects_relabelled_slot1_reuse(self):
        args = deepcopy(self.metadata)
        row = next(x for x in args[0]['selected_call_manifest'] if x['event'] == 6181)
        row['slot'] = 3
        with self.assertRaises(RuntimeError):
            bind_cases(*args)

    def test_binding_rejects_missing_original_mark(self):
        args = deepcopy(self.metadata)
        args[0]['selected_call_manifest'].pop()
        with self.assertRaises(RuntimeError):
            bind_cases(*args)

    def test_binding_rejects_nonempty_old_prefix(self):
        args = deepcopy(self.metadata)
        args[2][1]['official']['tensors']['cache-32-key']['expected_previous'] = dict(length=1)
        with self.assertRaises(RuntimeError):
            bind_cases(*args)

    def test_changed_tokens_rejected_before_forward(self):
        self.tokens.values = list(self.tokens.values)
        self.tokens.values[0] ^= 1
        with self.assertRaises(RuntimeError):
            self.collect()
        self.assertEqual(self.model.forward_calls, 0)

    def test_changed_positions_rejected_before_forward(self):
        self.positions.values[-1] = 0
        with self.assertRaises(RuntimeError):
            self.collect()
        self.assertEqual(self.model.forward_calls, 0)

    def test_shortened_native_call_rejected_before_forward(self):
        case = deepcopy(self.case)
        case['identity']['end'] = 33
        with self.assertRaises(RuntimeError):
            self.collect(case)
        self.assertEqual(self.model.forward_calls, 0)

    def test_stale_cache_rejected_before_forward(self):
        self.model.caches[104].keys = FakeTensor([4, 1, 256])
        with self.assertRaises(RuntimeError):
            self.collect()
        self.assertEqual(self.model.forward_calls, 0)

    def test_loop_cache_alias_rejected_before_forward(self):
        self.model.backbone.layers[32].mixer.attn[1] = self.model.caches[32]
        with self.assertRaises(RuntimeError):
            self.collect()
        self.assertEqual(self.model.forward_calls, 0)

    def test_unrelated_physical_layer_replaced_rejected(self):
        self.model.backbone.layers[3] = self.model.backbone.layers[5]
        with self.assertRaises(RuntimeError):
            self.collect()
        self.assertEqual(self.model.forward_calls, 0)

    def test_wrong_visit_order_poisoned_and_cannot_retry(self):
        self.model.order = (VISITS[1], VISITS[0]) + VISITS[2:]
        with self.assertRaises(RuntimeError):
            self.collect()
        self.assertTrue(self.observer.poisoned)
        self.model.order = VISITS
        with self.assertRaises(RuntimeError):
            self.collect()
        self.assertEqual(self.model.forward_calls, 1)

    def test_missing_mixer_rejected(self):
        self.model.backbone.layers[32].forward = lambda p, h, loop_idx: h
        with self.assertRaises(RuntimeError):
            self.collect()
        self.assertTrue(self.observer.poisoned)

    def test_duplicate_mixer_rejected(self):
        mixer = self.model.backbone.layers[32].mixer
        def duplicate(p, h, loop_idx):
            mixer(p, h, loop_idx=loop_idx)
            return mixer(p, h, loop_idx=loop_idx)
        self.model.backbone.layers[32].forward = duplicate
        with self.assertRaises(RuntimeError):
            self.collect()

    def test_changed_mixer_input_rejected(self):
        mixer = self.model.backbone.layers[32].mixer
        self.model.backbone.layers[32].forward = lambda p, h, loop_idx: mixer(p, FakeTensor(h.shape), loop_idx=loop_idx)
        with self.assertRaises(RuntimeError):
            self.collect()

    def test_mixer_dtype_conversion_rejected(self):
        self.model.backbone.layers[32].mixer.forward = lambda p, h, loop_idx: FakeTensor(h.shape, 'F32')
        with self.assertRaises(RuntimeError):
            self.collect()

    def test_model_exception_not_swallowed(self):
        def fail(*args, **kwargs):
            raise ValueError('synthetic original error')
        self.model.backbone.layers[17].forward = fail
        with self.assertRaisesRegex(ValueError, 'original error'):
            self.collect()
        self.assertTrue(self.observer.poisoned)
        self.assertFalse(self.observer.active)

    def test_selected_boundary_mismatch_stops_forward(self):
        sink = FakeEmitter(self.case, ('selected', SELECTED_LABELS[0], 'sha256', 'f'*64))
        with self.assertRaises(RuntimeError):
            self.collect(emitter=sink)
        self.assertEqual(self.observer.completed, 0)

    def test_cache_mismatch_cannot_finish(self):
        sink = FakeEmitter(self.case, ('original', 'cache-104-key', 'sha256', 'f'*64))
        with self.assertRaises(RuntimeError):
            self.collect(emitter=sink)
        with self.assertRaises(RuntimeError):
            self.observer.finish()

    def test_emitted_nonfinite_or_short_or_wrong_dtype_fails(self):
        for key, value in [('finite', False), ('bytes', 2), ('dtype', 'F32')]:
            with self.subTest(field=key):
                model = FakeModel()
                obs = StageAObserver(model, model.caches, self.ops, lambda: model.count, lambda: None)
                obs.install()
                try:
                    with self.assertRaises(RuntimeError):
                        obs.collect(self.case, self.tokens, self.positions,
                                    FakeEmitter(self.case, ('target', TARGET_LABELS[0], key, value)))
                finally:
                    obs.close()

    def test_wrong_top1_fails(self):
        self.ops.top1 = lambda t: [0]*33
        with self.assertRaises(RuntimeError):
            self.collect()

    def test_projection_count_must_include_head(self):
        original = self.model.compute_logits
        def no_count(head):
            out = original(head)
            self.model.count -= 1
            return out
        self.model.compute_logits = no_count
        with self.assertRaises(RuntimeError):
            self.collect()

    def test_no_hooks_left_after_close(self):
        self.observer.close()
        self.assertTrue(all(not x.pre and not x.post for x in self.model.backbone.layers))
        self.assertTrue(all(not self.model.backbone.layers[p].mixer.pre
                            and not self.model.backbone.layers[p].mixer.post for p in (32, 56)))

    def test_partial_install_rollback(self):
        self.observer.close()
        def fail(*a, **k):
            raise ValueError('registration failed')
        self.model.backbone.layers[5].register_forward_hook = fail
        with self.assertRaises(ValueError):
            self.observer.install()
        self.assertEqual(self.observer.handles, [])
        self.assertTrue(all(not x.pre and not x.post for x in self.model.backbone.layers))

    def test_duplicate_event_rejected(self):
        self.collect()
        self.model.empty_caches()
        with self.assertRaises(RuntimeError):
            self.collect()
        self.assertEqual(self.model.forward_calls, 1)


class ByteTests(unittest.TestCase):
    def fixture(self):
        layout = dict(input=dict(shape=[3, 4], dtype='BF16'), logits=dict(shape=[1, 2], dtype='F32'))
        data = dict(input=struct.pack('<12H', *range(0x3f80, 0x3f8c)),
                    logits=struct.pack('<2f', 1.25, -2.5))
        stream = io.BytesIO()
        writer = ArchiveWriter(stream, layout, byte_limit=32)
        for label, raw in data.items():
            writer.write(label, [raw[:4], raw[4:]])
        return layout, data, stream.getvalue(), writer.finish()

    def check(self, raw, archive, layout, **kwargs):
        return verify_archive(io.BytesIO(raw), archive, layout, max_bytes=len(raw), **kwargs)

    def test_all_bytes_original_hashes_and_selected_rows(self):
        layout, data, raw, archive = self.fixture()
        result = self.check(raw, archive, layout, expected_tensors=archive['tensors'],
                            selected_row_checks={'input': dict(rows=[0, 2],
                                sha256=hashlib.sha256(data['input'][:8]+data['input'][16:]).hexdigest())})
        self.assertEqual(result['bytes_verified'], 32)
        self.assertEqual(result['full_original_controls'], 2)
        self.assertEqual(result['selected_target_rows_verified'], 2)
        self.assertFalse(result['full_model_qualified'])

    def test_partial_reads_cross_row_boundaries(self):
        layout, data, raw, archive = self.fixture()
        class SmallRead(io.BytesIO):
            def read(self, size=-1):
                return super().read(min(size, 4))
        result = verify_archive(SmallRead(raw), archive, layout, max_bytes=32,
            selected_row_checks={'input': dict(rows=[1], sha256=hashlib.sha256(data['input'][8:16]).hexdigest())})
        self.assertEqual(result['selected_target_rows_verified'], 1)

    def test_mutated_unselected_row_is_still_checked(self):
        layout, data, raw, archive = self.fixture()
        mutated = bytearray(raw)
        mutated[8] ^= 1
        with self.assertRaises(RuntimeError):
            self.check(bytes(mutated), archive, layout, selected_row_checks={
                'input': dict(rows=[0], sha256=hashlib.sha256(data['input'][:8]).hexdigest())})

    def test_truncation_trailing_bytes_and_wrong_whole_hash(self):
        layout, data, raw, archive = self.fixture()
        for changed in [raw[:-2], raw+b'\0\0']:
            with self.subTest(length=len(changed)), self.assertRaises(RuntimeError):
                self.check(changed, archive, layout)
        bad = deepcopy(archive)
        bad['sha256'] = '0'*64
        with self.assertRaises(RuntimeError):
            self.check(raw, bad, layout)

    def test_wrong_rows_or_selected_row_hash_fail(self):
        layout, data, raw, archive = self.fixture()
        for rows in [[3], [-1], [1, 1], [2, 0], []]:
            with self.subTest(rows=rows), self.assertRaises(RuntimeError):
                self.check(raw, archive, layout, selected_row_checks={
                    'input': dict(rows=rows, sha256='0'*64)})
        with self.assertRaises(RuntimeError):
            self.check(raw, archive, layout, selected_row_checks={'input': dict(rows=[1], sha256='0'*64)})

    def test_omitted_extra_or_reordered_tensor_fails(self):
        layout, data, raw, archive = self.fixture()
        for operation in ['missing', 'extra', 'reorder']:
            bad = deepcopy(archive)
            if operation == 'missing':
                bad['tensors'].pop('input')
            elif operation == 'extra':
                bad['tensors']['extra'] = bad['tensors']['input']
            else:
                bad['tensors'] = dict(reversed(list(bad['tensors'].items())))
            with self.subTest(operation=operation), self.assertRaises(RuntimeError):
                self.check(raw, bad, layout)

    def test_non_dense_offset_and_false_shape_fail(self):
        layout, data, raw, archive = self.fixture()
        for key, value in [('offset', 2), ('bytes', 22), ('shape', [2, 6]), ('finite', False)]:
            bad = deepcopy(archive)
            bad['tensors']['input'][key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                self.check(raw, bad, layout)

    def test_corrupt_full_original_control_binding_fails(self):
        layout, data, raw, archive = self.fixture()
        original = deepcopy(archive['tensors'])
        original['input']['sha256'] = 'a'*64
        with self.assertRaises(RuntimeError):
            self.check(raw, archive, layout, expected_tensors=original)

    def test_budget_checked_before_payload_read(self):
        layout, data, raw, archive = self.fixture()
        class ForbiddenRead:
            def read(self, n):
                raise AssertionError('must not read')
        with self.assertRaises(RuntimeError):
            verify_archive(ForbiddenRead(), archive, layout, max_bytes=31)

    def test_writer_budget_checked_before_any_write(self):
        stream = io.BytesIO()
        with self.assertRaises(RuntimeError):
            ArchiveWriter(stream, dict(x=dict(shape=[2], dtype='BF16')), byte_limit=3)
        self.assertEqual(stream.getvalue(), b'')

    def test_writer_rejects_nonfinite_bf16_and_float32(self):
        for dtype, values, fmt in [('BF16', [0x7f80, 0xff80, 0x7fc1], '<H'),
                                   ('F32', [0x7f800000, 0xff800000, 0x7fc00001], '<I')]:
            for value in values:
                raw = struct.pack(fmt, value)
                writer = ArchiveWriter(io.BytesIO(), dict(x=dict(shape=[1], dtype=dtype)), byte_limit=4)
                with self.subTest(dtype=dtype, value=value), self.assertRaises(RuntimeError):
                    writer.write('x', [raw])
                self.assertTrue(writer.poisoned)

    def test_independent_verifier_rejects_nonfinite_even_with_correct_hash(self):
        for dtype, raw in [('BF16', b'\x80\x7f'), ('BF16', b'\xc1\xff'),
                           ('F32', struct.pack('<I', 0x7f800000)), ('F32', struct.pack('<I', 0xffc00001))]:
            h = hashlib.sha256(raw).hexdigest()
            layout = dict(x=dict(shape=[1], dtype=dtype))
            archive = dict(complete=True, bytes=len(raw), sha256=h, tensors={
                'x': dict(shape=[1], dtype=dtype, finite=True, offset=0, bytes=len(raw), sha256=h)})
            with self.subTest(dtype=dtype, raw=raw), self.assertRaises(RuntimeError):
                self.check(raw, archive, layout)

    def test_writer_short_write_poisoned_and_not_counted_as_complete(self):
        class Short:
            def write(self, raw):
                return len(raw)-1
        writer = ArchiveWriter(Short(), dict(x=dict(shape=[2], dtype='BF16')), byte_limit=4)
        with self.assertRaises(RuntimeError):
            writer.write('x', [b'\0'*4])
        self.assertTrue(writer.poisoned)
        self.assertEqual(writer.records, {})
        with self.assertRaises(RuntimeError):
            writer.finish()

    def test_writer_iterator_failure_cannot_be_retried(self):
        writer = ArchiveWriter(io.BytesIO(), dict(x=dict(shape=[2], dtype='BF16')), byte_limit=4)
        def blocks():
            yield b'\0\0'
            raise ValueError('synthetic read failure')
        with self.assertRaises(ValueError):
            writer.write('x', blocks())
        self.assertEqual(writer.written, 2)
        self.assertTrue(writer.poisoned)
        with self.assertRaises(RuntimeError):
            writer.write('x', [b'\0'*4])

    def test_writer_truncated_overlong_empty_unaligned_or_huge_chunk(self):
        for chunks in [[b'\0\0'], [b'\0'*6], [b''], [b'\0'], [b'\0'*65538]]:
            writer = ArchiveWriter(io.BytesIO(), dict(x=dict(shape=[2], dtype='BF16')), byte_limit=4)
            with self.subTest(lengths=[len(x) for x in chunks]), self.assertRaises(RuntimeError):
                writer.write('x', chunks)
            self.assertTrue(writer.poisoned)

    def test_duplicate_order_and_partial_finish(self):
        writer = ArchiveWriter(io.BytesIO(), dict(a=dict(shape=[1], dtype='BF16'),
                              b=dict(shape=[1], dtype='BF16')), byte_limit=4)
        with self.assertRaises(RuntimeError):
            writer.write('b', [b'\0\0'])
        writer.write('a', [b'\0\0'])
        with self.assertRaises(RuntimeError):
            writer.write('a', [b'\0\0'])
        with self.assertRaises(RuntimeError):
            writer.finish()
        writer.write('b', [b'\0\0'])
        writer.finish()
        with self.assertRaises(RuntimeError):
            writer.finish()

    def test_emitter_preserves_full_vs_selected_adapter_requests(self):
        seen = []
        writers = {}
        for scope in ('target', 'selected', 'original'):
            writers[scope] = ArchiveWriter(io.BytesIO(), dict(x=dict(shape=[1], dtype='BF16')), byte_limit=2)
        token = object()
        def blocks(t, rows):
            seen.append((t, rows))
            yield b'\0\0'
        emitter = TensorEmitter(writers, blocks)
        emitter('target', 'x', token, None)
        emitter('selected', 'x', token, [6])
        emitter('original', 'x', token, None)
        self.assertEqual(seen, [(token, None), (token, [6]), (token, None)])
        self.assertEqual(set(emitter.finish()), {'target', 'selected', 'original'})


if __name__ == '__main__':
    unittest.main(verbosity=2)
