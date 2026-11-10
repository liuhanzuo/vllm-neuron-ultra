"""Source-only Stage A observation core, with no native or filesystem imports.

The caller must supply the unchanged full official model, authenticated original
metadata, native tensor adapters and synchronous bounded archive emitters. This
module neither constructs a model nor admits or launches a native job. Hooks
return None and never replace tensors, mixer math, residual math or precision.
"""
from copy import deepcopy

from dory_residual_visit_capture_v1 import visits, require


TARGETS = ((32, 0), (32, 1), (56, 0))
WIDE = ((1, 0, 10, 1, 2060, 'full_causal'),
        (4, 1, 11, 1, 2060, 'full_causal'),
        (7, 0, 3, 1, 23, 'prefill'), (9, 1, 1, 1, 127, 'prefill'),
        (11, 0, 0, 1, 511, 'prefill'), (6179, 1, 3, 2, 31, 'slot_reuse_prefill'))
CACHE_IDS = tuple(sorted((*range(0, 16, 2), *range(16, 56, 2),
                          *range(88, 128, 2), *range(56, 72, 2))))
VISITS = (tuple((i, 0) for i in range(16))
          + tuple((i, loop) for loop in range(2) for i in range(16, 56))
          + tuple((i, 0) for i in range(56, 72)))
SELECTED_LABELS = tuple(f'layer-{p}-loop-{l}-{k}' for p, l in VISITS
                        for k in ('input', 'residual_output'))
TARGET_LABELS = tuple(f'layer-{p}-loop-{l}-{k}' for p, l in TARGETS
                      for k in ('input', 'mixer_output', 'residual_output'))
ORIGINAL_LABELS = ('hidden', 'logits') + tuple(
    f'cache-{i}-{k}' for i in CACHE_IDS for k in ('key', 'value'))


def sha(value):
    require(isinstance(value, str) and len(value) == 64
            and all(c in '0123456789abcdef' for c in value), 'Exact SHA256')
    return value


def bind_cases(design, selection_plan, originals, replays):
    """Validate already-authenticated JSON; caller owns file/hash provenance.

    Keep all89/87/7 original marks in the parent design while selecting the six
    true full-input gaps for Stage A. An empty-cache replay of event6179 is not
    proof of production slot reuse; event6181 belongs to continuing slot1.
    """
    require(design['schema'] == 'dory_targeted_mixer_residual_design_v1'
            and design['scope']['selected_calls'] == 89
            and design['scope']['marked_calls'] == 87
            and design['original_acceptance'] == dict(cosine_min=0.995, max_abs_max=2.0,
                KL_max=0.02, relaxations_allowed=False,
                metric_key_kl_reference_to_neuron_is_legacy_name=True), 'Original design and gates')
    manifest = {item['event']: item for item in design['selected_call_manifest']}
    require(len(manifest) == 89 and sum(x['original_pointwise_failed'] is True for x in manifest.values()) == 7
            and (manifest[6181]['slot'], manifest[6181]['epoch'], manifest[6181]['phase'])
                == (1, 1, 'decode_after_slot_reuse'), 'All marks and actual reuse semantics')
    require(set(originals) == set(replays) == {row[0] for row in WIDE}, 'Exactly six original wide calls')
    positions = selection_plan['original_prefix_plan']['reference_positions']
    require(len(positions) == 33 and positions == sorted(set(positions)), 'Original full head selection')
    cases = []
    for event, fixture, slot, epoch, end, phase in WIDE:
        original, replay, item = originals[event], replays[event], manifest[event]
        identity = dict(event=event, kind='call', fixture=fixture, slot=slot, epoch=epoch,
                        start=0, end=end, phase=phase, request=original['request'])
        require(isinstance(identity['request'], str) and bool(identity['request'])
                and all(original[k] == replay[k] == value for k, value in identity.items())
                and all(item[k] == value for k, value in identity.items() if k != 'kind'), 'Unchanged original owner and shape')
        selected = positions if phase == 'full_causal' else [end-1]
        require(item['selected_local_rows'] == replay['selected_local_rows'] == selected
                and replay['actual_archive'] == original['official']
                and replay['archive_comparison']['exact'] is True,
                'Exact original replay and selected rows')
        old_selected = replay['selected_archive']
        require(list(old_selected['tensors']) == list(SELECTED_LABELS)
                and old_selected['complete'] is True, 'Every old selected boundary')
        archive = original['official']
        require(list(archive['tensors']) == list(ORIGINAL_LABELS)
                and archive['start'] == 0 and archive['end'] == end
                and archive['slot'] == slot and archive['epoch'] == epoch,
                'Every original hidden/logit/history tensor')
        for label, desc in old_selected['tensors'].items():
            require(desc['shape'] == [len(selected), 2560] and desc['dtype'] == 'BF16'
                    and desc['bytes'] == len(selected)*5120 and desc['finite'] is True,
                    'Original selected residual geometry')
            sha(desc['sha256'])
        for label, desc in archive['tensors'].items():
            shape, dtype = (([4, end, 256], 'BF16') if label.startswith('cache-') else
                            ([end, 2560], 'BF16') if label == 'hidden' else
                            ([len(selected), 131072], 'F32'))
            require(desc['shape'] == shape and desc['dtype'] == dtype and desc['finite'] is True
                    and desc['sha256'] == desc['full_sha256'], 'Original complete zero-prefix tensors')
            sha(desc['sha256'])
            if label.startswith('cache-'):
                require(desc['expected_previous'] is None and desc['prefix_valid'] is True
                        and desc['prefix_sha256'] == 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
                        'Original empty cache prefix')
        ids = selection_plan['original_prefix_plan']['token_fixtures'][fixture]['input_ids'][:end]
        require(len(ids) == end and all(type(x) is int and 0 <= x < 131072 for x in ids)
                and original['official_onednn_calls'] == 281
                and original['physical_layer_invocations'] == {str(i): 2 if 16 <= i < 56 else 1 for i in range(72)},
                'Original tokens and complete model visit/projection counts')
        cases.append(dict(identity=identity, input_ids=ids, selected_rows=list(selected),
            original_top1=list(original['official_top1']), original_archive=deepcopy(archive),
            selected_archive=deepcopy(old_selected), original_event=deepcopy(item['original_event']),
            residual_event=deepcopy(item['residual_event'])))
    return tuple(cases)


def layouts(case):
    rows, selected = case['identity']['end'], len(case['selected_rows'])
    return dict(
        target={label: dict(shape=[rows, 2560], dtype='BF16') for label in TARGET_LABELS},
        selected={label: dict(shape=[selected, 2560], dtype='BF16') for label in SELECTED_LABELS},
        original={label: dict(shape=list(desc['shape']), dtype=desc['dtype'])
                  for label, desc in case['original_archive']['tensors'].items()})


class StageAObserver:
    """Synchronous passive observation of one unmodified complete forward.

    ops.describe(t): shape/dtype/stride/storage_offset, without value conversion.
    ops.integers(t): exact int64 token/position vector as Python ints.
    ops.select(t, rows): original dtype, full-width row selection for output head.
    ops.top1(logits): original output-row token IDs.
    emit(scope, label, t, rows): synchronous archive emission, returns descriptor.
    rows=None means every real row. No projected, zero-filled or padded inputs.

    Native implementations of these callbacks must be source-pinned separately.
    Neither callbacks nor synthetic tests prove weight/source/runtime provenance.
    """
    def __init__(self, official, cache_by_id, ops, projection_count, guard):
        self.official, self.ops = official, ops
        self.caches = dict(cache_by_id)
        self.projection_count, self.guard = projection_count, guard
        self.layers = tuple(official.backbone.layers)
        self.mixers = {p: self.layers[p].mixer for p in (32, 56)}
        self.handles, self.active, self.poisoned = [], False, False
        self.completed_cases = []
        self.pending = None
        self._structure()

    def _structure(self):
        require(visits(self.official.backbone.config) == VISITS
                and self.official.backbone.num_loops == 2
                and tuple(self.official.backbone.layers) == self.layers
                and len(self.layers) == len({id(x) for x in self.layers}) == 72,
                'Same complete original 72-layer, 112-visit model')
        require(set(self.caches) == set(CACHE_IDS)
                and len({id(x) for x in self.caches.values()}) == 56,
                'All56 independent per-loop caches')
        cfg = self.official.backbone.config
        require((cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim,
                 cfg.sliding_window, cfg.recurrent_kv_cache_mode) == (2560, 16, 4, 256, 1024, 'per_loop')
                and (cfg.ngpt_alpha_init, cfg.initializer_range) == (.05, .0177), 'Original geometry and alpha scale')
        for p, mixer in self.mixers.items():
            layer = self.layers[p]
            require(layer.mixer is mixer and layer.is_attention is True
                    and layer._alpha_scale == cfg.ngpt_alpha_init/cfg.initializer_range
                    and cfg.layer_types[p] == 'sliding_attention'
                    and mixer.per_loop_kv is True
                    and len(mixer.attn) == (2 if p == 32 else 1)
                    and all(mixer.attn[loop] is self.caches[p+loop*72] for loop in range(len(mixer.attn))),
                    'Unchanged target mixers with distinct original loop caches')

    def _tensor(self, tensor, shape, dtype):
        desc = self.ops.describe(tensor)
        require(set(desc) == {'shape', 'dtype', 'stride', 'storage_offset'}
                and desc['shape'] == shape and desc['dtype'] == dtype
                and isinstance(desc['stride'], list) and len(desc['stride']) == len(shape)
                and all(type(x) is int and x >= 0 for x in desc['stride'])
                and type(desc['storage_offset']) is int and desc['storage_offset'] >= 0,
                'Full original tensor geometry/dtype and recorded actual layout')
        return desc

    def _identity(self, p, args, kwargs):
        require(len(args) in (2, 3) and set(kwargs) <= {'loop_idx'}
                and not (len(args) == 3 and 'loop_idx' in kwargs), 'Original layer/mixer arguments')
        loop = args[2] if len(args) == 3 else kwargs.get('loop_idx', 0)
        require(type(loop) is int and args[0] is self.positions, 'Unchanged positions and loop identity')
        return p, loop

    def _emit(self, scope, label, tensor, rows=None):
        self.guard()
        layout = self.expected_layouts[scope][label]
        source = self._tensor(tensor, layout['shape'] if rows is None else [self.rows, 2560], layout['dtype'])
        require(label not in self.records[scope], 'Each boundary/output emitted once')
        result = self.emit(scope, label, tensor, rows)
        require(result['shape'] == layout['shape'] and result['dtype'] == layout['dtype']
                and result['finite'] is True and type(result['bytes']) is int
                and result['bytes'] == self._extent(layout), 'Complete finite emitted tensor')
        sha(result['sha256'])
        if scope in ('selected', 'original'):
            old = self.case[scope+'_archive']['tensors'][label]
            require(result['sha256'] == old['sha256'], 'Original control bytes must match exactly')
        self.records[scope][label] = dict(result)
        self.source_layouts[scope][label] = source

    @staticmethod
    def _extent(layout):
        size = 2 if layout['dtype'] == 'BF16' else 4
        for n in layout['shape']:
            size *= n
        return size

    def install(self):
        require(not self.handles and not self.active and not self.poisoned, 'Fresh inactive observer')
        self._structure()

        def before(p):
            def hook(module, args, kwargs):
                require(self.active and self.pending is None and self.completed < 112,
                        'No nested, missing or extra physical visit')
                visit = self._identity(p, args, kwargs)
                require(module is self.layers[p] and visit == VISITS[self.completed], 'Exact original visit order')
                self.pending, self.layer_input, self.mixer_started, self.mixer_done = visit, args[1], False, False
                key = f'layer-{p}-loop-{visit[1]}-input'
                self._emit('selected', key, args[1], self.case['selected_rows'])
                if visit in TARGETS:
                    self._emit('target', key, args[1])
            return hook

        def after(p):
            def hook(module, args, kwargs, output):
                visit = self._identity(p, args, kwargs)
                require(self.active and module is self.layers[p] and visit == self.pending
                        and (visit not in TARGETS or self.mixer_started and self.mixer_done),
                        'Original target mixer completed exactly once before residual output')
                key = f'layer-{p}-loop-{visit[1]}-residual_output'
                self._emit('selected', key, output, self.case['selected_rows'])
                if visit in TARGETS:
                    self._emit('target', key, output)
                self.completed += 1
                self.pending, self.layer_input = None, None
            return hook

        def mixer_before(p):
            def hook(module, args, kwargs):
                visit = self._identity(p, args, kwargs)
                require(self.active and module is self.mixers[p] and visit == self.pending
                        and visit in TARGETS and args[1] is self.layer_input
                        and not self.mixer_started and not self.mixer_done, 'Original single mixer with unchanged input')
                self.mixer_started = True
            return hook

        def mixer_after(p):
            def hook(module, args, kwargs, output):
                visit = self._identity(p, args, kwargs)
                require(self.active and module is self.mixers[p] and visit == self.pending
                        and self.mixer_started and not self.mixer_done, 'Exactly one original mixer output')
                self._emit('target', f'layer-{p}-loop-{visit[1]}-mixer_output', output)
                self.mixer_done = True
            return hook

        try:
            for p, layer in enumerate(self.layers):
                self.handles.append(layer.register_forward_pre_hook(before(p), with_kwargs=True))
                self.handles.append(layer.register_forward_hook(after(p), with_kwargs=True))
            for p, mixer in self.mixers.items():
                self.handles.append(mixer.register_forward_pre_hook(mixer_before(p), with_kwargs=True))
                self.handles.append(mixer.register_forward_hook(mixer_after(p), with_kwargs=True))
        except BaseException:
            self.poisoned = True
            self.close()
            raise

    def collect(self, case, tokens, positions, emit):
        """Future admitted caller only. Empty native caches are a precondition.

        Never clears/rebinds caches, catches model errors as successes, changes
        precision or computes a replacement residual. No native entrypoint.
        """
        require(not self.active and not self.poisoned and len(self.handles) == 148, 'Installed healthy observer')
        expected_event = WIDE[len(self.completed_cases)][0] if len(self.completed_cases) < 6 else None
        require(case['identity']['event'] == expected_event, 'Six original wide calls once in original order')
        require(case['identity']['start'] == 0
                and tuple(case['identity'][k] for k in ('event', 'fixture', 'slot', 'epoch', 'end', 'phase'))
                    == WIDE[len(self.completed_cases)], 'Full original identity and unshrunk call shape')
        self._structure()
        self.case, self.emit, self.positions = deepcopy(case), emit, positions
        self.rows = case['identity']['end']
        self._tensor(tokens, [self.rows], 'I64')
        self._tensor(positions, [self.rows], 'I64')
        require(self.ops.integers(tokens) == case['input_ids']
                and self.ops.integers(positions) == list(range(self.rows)), 'Every original token and position')
        require(all(c.keys is None and c.values is None and c.start_pos == 0 for c in self.caches.values()),
                'All56 original native histories empty at position0')
        self.expected_layouts = layouts(case)
        self.records = {k: {} for k in self.expected_layouts}
        self.source_layouts = {k: {} for k in self.expected_layouts}
        self.completed, self.pending = 0, None
        initial = self.projection_count()
        require(type(initial) is int, 'Actual oneDNN counter')
        self.active = True
        try:
            hidden = self.official(tokens, positions)
            require(self.completed == 112 and self.pending is None, 'Complete unchanged original backbone')
            self._emit('original', 'hidden', hidden)
            head = self.ops.select(hidden, case['selected_rows'])
            self._tensor(head, [len(case['selected_rows']), 2560], 'BF16')
            logits = self.official.compute_logits(head)
            self._emit('original', 'logits', logits)
            require(self.ops.top1(logits) == case['original_top1'], 'Original top1 output rows')
            for key in CACHE_IDS:
                cache = self.caches[key]
                for label, value in (('key', cache.keys), ('value', cache.values)):
                    self._emit('original', f'cache-{key}-{label}', value)
            require(self.projection_count()-initial == 281, 'Original complete 281 oneDNN calls')
            self._structure()
            require(all(list(self.records[k]) == list(self.expected_layouts[k]) for k in self.records),
                    'All9 full targets,224 selected controls,114 original outputs in order')
            result = dict(identity=deepcopy(case['identity']), selected_rows=list(case['selected_rows']),
                records=deepcopy(self.records), source_layouts=deepcopy(self.source_layouts),
                original_top1=list(case['original_top1']), physical_visits=112, onednn_calls=281,
                original_byte_controls_matched=True, independent_saved_byte_review_complete=False,
                original_source_weight_runtime_identity_verified_here=False,
                production_slot_reuse_qualified=False, original_FP32_acceptance_passed=False,
                all87_marked_calls_qualified=False, full_model_qualified=False, new_speedup_claim=False)
            self.completed_cases.append(result['identity']['event'])
            return result
        except BaseException:
            self.poisoned = True
            raise
        finally:
            self.active = False
            self.emit = None

    def finish(self):
        require(not self.active and not self.poisoned and self.completed_cases == [x[0] for x in WIDE],
                'Only all six successful complete captures can finish')
        return dict(stage_a_collector_complete=True, events=list(self.completed_cases),
                    independent_saved_byte_review_complete=False, all87_marked_calls_qualified=False,
                    full_model_qualified=False, new_speedup_claim=False)

    def close(self):
        require(not self.active, 'Cannot remove hooks during a forward')
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
