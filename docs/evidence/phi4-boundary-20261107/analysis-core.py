"""Unexecuted saved-tensor analysis of all six Phi gateup v4 arms.

No model, weights, device, compiler, or launch interface. A separately reviewed
serial supervisor must admit run_native. Source checks use stdlib only.
"""
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
VARIANTS = ('instruct', 'reasoning')
ARMS = ('unchanged', 'wrapper', 'candidate')
ARM_NAMES = dict(unchanged='unchanged', wrapper='wrapper_batch4', candidate='decode_row1_view')
ANCHORS = ('native_single', 'native_batch', 'port')
GATE_NAMES = dict(native_single='original_hf', native_batch='native_batch', port='original_port')
FIELDS = dict(native_single='saved_single', native_batch='native_batch', port='port')
BOUNDARIES = {
    'gateup_output': ('layers.0.mlp.gate_up_proj.output', 16384),
    'down_input': ('layers.0.mlp.down_proj.input', 8192),
    'down_output': ('layers.0.mlp.down_proj.output', 3072),
    'layer0_output': ('layers.0.output', 3072),
    'layer1_norm': ('layers.1.input_layernorm.output', 3072),
    'layer1_qkv': ('layers.1.self_attn.qkv_proj.output', 5120),
    'layer1_oproj_input': ('layers.1.self_attn.o_proj.input', 3072),
    'final_norm': ('norm.output', 3072),
    'logits': ('lm_head.output', 200064),
}
CRITERIA = {'cosine_min': .995, 'kl_max_nats': .02, 'max_abs_error_max': 2.,
            'argmax_match_fraction_min': .9, 'all_reference_margins_gt_0.5_must_match': True}
EXPECTED = dict(logical_frames=64, arms=6, arm_frame_occurrences=192, boundaries=9,
                anchors=3, comparisons=5184, head_comparisons=576,
                activation_payload_inventory=160, selected_activation_payloads=104,
                excluded_TP2_activation_payloads=56, manifests_inventory=10, selected_manifests=6)


def require(value, reason):
    if not value:
        raise ValueError(reason)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def keys():
    result = []
    for order in ([0, 5, 2, 4], [1, 3, 0, 2]):
        result.extend(dict(fixture=f, phase='prefill', slot=s) for f, s in zip(order, [3, 1, 0, 2]))
        result.extend(dict(fixture=f, phase='decode', step=step, slot=s)
                      for step in (1, 2, 3) for f, s in zip(order, [3, 1, 0, 2]))
    return result


def frame_key(value, single=False):
    return (value['fixture'], value['phase'], value.get('step', 0)) + (() if single else (value['slot'],))


def all_stages():
    suffixes = ('input_layernorm.input', 'input_layernorm.output', 'self_attn.qkv_proj.output',
        'self_attn.o_proj.input', 'self_attn.o_proj.output', 'post_attention_layernorm.input',
        'post_attention_layernorm.output', 'mlp.gate_up_proj.output', 'mlp.down_proj.input',
        'mlp.down_proj.output', 'output')
    return ['embed_tokens.output'] + [f'layers.{i}.{s}' for i in range(32) for s in suffixes] + [
        'norm.input', 'norm.output', 'lm_head.input', 'lm_head.output']


def width(stage):
    if stage == 'lm_head.output': return 200064
    if stage.endswith('qkv_proj.output'): return 5120
    if stage.endswith('gate_up_proj.output'): return 16384
    if stage.endswith('down_proj.input'): return 8192
    return 3072


def occurrences(manifest, backend, variant, payload_inventory):
    """Return 32 ordered occurrences. Native24 reuse is explicit, never dropped."""
    require(backend in ANCHORS and variant in VARIANTS, 'reference backend/variant')
    require(manifest['complete'] is True and manifest['backend'] == backend and manifest['variant'] == variant,
            'manifest producer identity')
    require(manifest['prediction_rows_only'] is True and manifest['all_original32_frames_mapped'] is True,
            'selected row scope')
    require(len(manifest['calls']) == (24 if backend == 'native_single' else 14), 'all producer calls')
    observed, mapping, paths = [], {}, set()
    for index, call in enumerate(manifest['calls']):
        require(call['index'] == index and call['stage_order'] == all_stages(), 'producer call/stage order')
        require(list(call['stages']) == all_stages(), 'complete 357 stage metadata')
        expected_path = ROOT / 'phi4-intermediate-capture-v1/activations' / variant / backend / f'call-{index:02d}.pt'
        require(call['payload'] == str(expected_path) and call['payload'] in payload_inventory, 'producer payload path')
        require(call['payload_bytes'] == payload_inventory[call['payload']][2], 'producer payload bytes')
        require(call['payload'] not in paths, 'repeated producer payload')
        paths.add(call['payload'])
        n = len(call['keys'])
        require(n in (1, 4), 'selected batch')
        for stage, meta in call['stages'].items():
            require(meta['selected_shape'] == [n, width(stage)] and meta['dtype'] in ('torch.bfloat16', 'torch.float32'),
                    'full feature geometry')
            require(meta['finite'] is True and len(meta['raw_sha256']) == 64, 'finite/raw stage receipt')
            require(len(meta['source_shape']) == len(meta['source_stride']) == 3 and
                    meta['source_shape'][0] == n and meta['source_shape'][2] == width(stage), 'source geometry')
            require(len(meta['source_token_indices']) == n and all(0 <= x < meta['source_shape'][1]
                    for x in meta['source_token_indices']), 'selected prediction positions')
        for row, key in enumerate(call['keys']):
            lookup = frame_key(key, backend == 'native_single')
            require(lookup not in mapping, 'ambiguous producer frame')
            mapping[lookup] = dict(call_index=index, physical_row=row, payload=call['payload'])
            observed.append(key)
    wanted = [dict(fixture=f, phase='prefill' if s == 0 else 'decode', **({'step': s} if s else {}))
              for f in range(6) for s in range(4)] if backend == 'native_single' else keys()
    require(observed == wanted, 'producer full frame schedule')
    return [dict(ordinal=i, key=key, reference= mapping[frame_key(key, backend == 'native_single')])
            for i, key in enumerate(keys())]


def metric_gate(rows):
    return len(rows) == 32 and all(
        all(math.isfinite(r[k]) for k in ('cosine', 'kl_reference_to_neuron', 'max_abs_error', 'reference_margin'))
        and r['cosine'] >= .995 and r['kl_reference_to_neuron'] <= .02 and r['max_abs_error'] <= 2
        and (r['reference_margin'] <= .5 or r['argmax_match']) for r in rows
    ) and sum(r['argmax_match'] for r in rows) / 32 >= .9


def identity(path):
    value = Path(path).stat()
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


class Budget:
    REQUIRED = {'input_read_bytes', 'output_bytes', 'single_payload_bytes', 'wall_seconds'}
    def __init__(self, limits, guard):
        require(set(limits) == self.REQUIRED and all(type(x) is int and x > 0 for x in limits.values()), 'explicit budgets')
        require(callable(guard), 'external serial resource guard')
        self.limits, self.guard, self.started = limits, guard, time.monotonic()
        self.read_bytes = self.output_bytes = 0
    def check(self):
        self.guard()
        require(time.monotonic() - self.started <= self.limits['wall_seconds'], 'analysis wall budget')
    def charge(self, kind, size):
        self.check()
        field = 'read_bytes' if kind == 'input_read_bytes' else 'output_bytes'
        require(type(size) is int and size >= 0 and getattr(self, field) + size <= self.limits[kind], 'analysis byte budget')
        setattr(self, field, getattr(self, field) + size)


class Inputs:
    """Runtime-only exact admitted files; no glob, symlink, model or weight load."""
    def __init__(self, budget):
        self.budget = budget
        self.loaded = set()
    def verified(self, spec):
        path = Path(spec['path'])
        require(path.is_absolute() and path.resolve() == path and not path.is_symlink() and path.is_file(), 'regular fixed input')
        before = identity(path)
        if spec.get('identity') is not None: require(before == spec['identity'], 'producer file identity changed')
        require(before[2] == spec['bytes'] <= self.budget.limits['single_payload_bytes'], 'input file cap')
        h = hashlib.sha256()
        with path.open('rb') as handle:
            while True:
                data = handle.read(1024 * 1024)
                if not data: break
                self.budget.charge('input_read_bytes', len(data)); h.update(data)
        require(h.hexdigest() == spec['sha256'] and identity(path) == before, 'input bytes changed')
        return path, before
    def json(self, spec):
        path, before = self.verified(spec)
        require(before[2] <= 8 * 1024**2, 'metadata cap')
        self.budget.charge('input_read_bytes', before[2])
        value = json.loads(path.read_text())
        require(identity(path) == before, 'metadata mutated during read')
        return value
    def tensor(self, spec):
        import torch
        path, before = self.verified(spec)
        self.budget.charge('input_read_bytes', before[2])
        value = torch.load(path, map_location='cpu', weights_only=True)
        require(identity(path) == before, 'payload mutated during load')
        self.loaded.add(str(path))
        return value


def raw_sha(value):
    import torch
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def raw_equal(a, b):
    import torch
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def difference(reference, actual):
    import torch
    require(reference.ndim == actual.ndim == 1 and reference.shape == actual.shape, 'entire matching feature vectors')
    require(bool(torch.isfinite(reference).all() and torch.isfinite(actual).all()), 'nonfinite selected vector')
    a, b = reference.double(), actual.double(); delta = b - a
    norm_a, norm_b = torch.linalg.vector_norm(a), torch.linalg.vector_norm(b)
    cosine = float(torch.dot(a, b) / (norm_a * norm_b)) if norm_a > 0 and norm_b > 0 else (1. if norm_a == norm_b == 0 else None)
    return dict(raw_equal=raw_equal(reference, actual), width=actual.numel(),
        reference_dtype=str(reference.dtype), actual_dtype=str(actual.dtype),
        reference_raw_sha256=raw_sha(reference), actual_raw_sha256=raw_sha(actual),
        max_abs=float(delta.abs().max()), mean_abs=float(delta.abs().mean()), rms=float(delta.square().mean().sqrt()),
        cosine=cosine, intermediate_tolerance_applied=False)


def original_metrics(source):
    """Exact original metrics AST, validated by inventory SHA before execution."""
    import torch
    tree = ast.parse(source)
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'metrics']
    require(len(nodes) == 1, 'original metrics function')
    namespace = dict(torch=torch)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'pinned_validate_real_logits_metrics', 'exec'), namespace)
    return namespace['metrics']


def reference_vectors(store, inventory, variant, anchor):
    """Load each producer payload once, verify all357 stages, retain only9."""
    source = inventory['references'][variant][anchor]
    manifest = store.json(source['manifest'])
    schedule = occurrences(manifest, anchor, variant, inventory['all160_payload_identities'])
    selected = {}
    for call in manifest['calls']:
        store.budget.check()
        spec = dict(path=call['payload'], bytes=call['payload_bytes'], sha256=call['payload_sha256'],
                    identity=inventory['all160_payload_identities'][call['payload']])
        tensors = store.tensor(spec)
        require(list(tensors) == all_stages(), 'all producer tensors present')
        for name, value in tensors.items():
            meta = call['stages'][name]
            require(list(value.shape) == meta['selected_shape'] and str(value.dtype) == meta['dtype'] and
                    value.device.type == 'cpu' and raw_sha(value) == meta['raw_sha256'], 'raw producer tensor mismatch')
        for row in range(len(call['keys'])):
            selected[(call['index'], row)] = {name: tensors[stage][row].clone() for name, (stage, _) in BOUNDARIES.items()}
        del tensors
    return [(entry, selected[(entry['reference']['call_index'], entry['reference']['physical_row'])]) for entry in schedule]


def arm_vectors(store, inventory, variant, arm):
    import torch
    source = inventory['arms'][variant][arm]
    report = store.json(source['report'])
    require(report['complete'] is report['passed'] is True and report['variant'] == variant and
            report['arm'] == ARM_NAMES[arm] and report['frames'] == 32 and report['model_calls'] == 15, 'six-arm metadata')
    require(report['payload'] == source['payload']['path'] and report['payload_sha256'] == source['payload']['sha256']
            and report['payload_bytes'] == source['payload']['bytes'], 'arm producer binding')
    payload = store.tensor(source['payload'])
    require(set(payload) == {'frames'} and len(payload['frames']) == len(report['frame_records']) == 32, 'all arm frames')
    for key, row, meta in zip(keys(), payload['frames'], report['frame_records']):
        require(all(row[k] == meta[k] == v for k, v in key.items()), 'arm frame ordinal/key')
        for field, dtype in (('logits_bf16', torch.bfloat16), ('logits_float32', torch.float32)):
            value = row[field]
            require(value.shape == (200064,) and value.dtype == dtype and value.device.type == 'cpu'
                    and raw_sha(value) == meta[field + '_sha256'], 'arm head raw receipt')
        require(raw_equal(row['logits_bf16'].float(), row['logits_float32']), 'arm FP32 head conversion')
        require(list(row['propagation']) == list(BOUNDARIES)[:-1], 'all eight propagation vectors')
        for name, value in row['propagation'].items():
            require(value.shape == (BOUNDARIES[name][1],) and value.dtype == torch.bfloat16 and value.device.type == 'cpu'
                    and raw_sha(value) == meta['propagation'][name]['sha256'], 'arm propagation raw receipt')
        if key['phase'] == 'decode':
            for field in ('target_input', 'target_output'):
                require(raw_sha(row[field]) == meta[field + '_sha256'], 'preserve selected projection receipts')
    return report, payload['frames']


def run_native(inventory, admission, guard, sink):
    """Future admitted callable. Never called by source checker; no CLI launch."""
    require(admission['independent_source_review_passed'] is True and admission['serial_supervisor_admitted'] is True,
            'separate independent/runtime admission missing')
    require(admission['inventory_sha256'] == digest(inventory) and callable(sink), 'exact inventory/sink admission')
    budget = Budget(admission['limits'], guard); budget.check()
    require(inventory['schema'] == 'phi4_saved_boundary_alignment_source_v1' and inventory['counts'] == EXPECTED,
            'complete analysis scope')
    require(inventory['runtime_admission'] is False, 'frozen source inventory claim')
    store = Inputs(budget)
    for spec in inventory['source_closure']:
        store.verified(spec)
    metric_spec = inventory['metric_source']; path, before = store.verified(metric_spec)
    budget.charge('input_read_bytes', before[2])
    metric_text = path.read_text(); require(identity(path) == before, 'metric source changed')
    import torch
    require(torch.__version__ == '2.9.1+cpu', 'original CPU metrics environment')
    torch.set_num_threads(2)
    metrics = original_metrics(metric_text)
    def emit(value):
        data = (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
        budget.charge('output_bytes', len(data)); sink(value)
    comparisons = heads = 0; reproduced = {}; divergences = {}
    for variant in VARIANTS:
        budget.check()
        old_report = store.json(inventory['old_reports'][variant])
        require(old_report['passed'] is False and old_report['criteria'] == CRITERIA and len(old_report['rows']) == 32,
                'original failed HF gate retained')
        hf = store.tensor(inventory['logit_sources'][variant]['hf'])
        paired = store.tensor(inventory['logit_sources'][variant]['paired'])
        require(len(hf) == 6 and [row['id'] for row in hf] == list(range(6)) and len(paired) == 32, 'all original reference logits')
        references = {anchor: reference_vectors(store, inventory, variant, anchor) for anchor in ANCHORS}
        arms = {arm: arm_vectors(store, inventory, variant, arm) for arm in ARMS}
        for arm in ARMS:
            require(arms[arm][0]['criteria'] == CRITERIA, 'original three-anchor criteria')
        for ordinal in range(32):
            left, right = arms['unchanged'][1][ordinal], arms['wrapper'][1][ordinal]
            require(raw_equal(left['logits_float32'], right['logits_float32']) and all(
                raw_equal(left['propagation'][name], right['propagation'][name]) for name in list(BOUNDARIES)[:-1]),
                'unchanged/wrapper control altered')
        for ordinal, key in enumerate(keys()):
            for field in ('saved_single', 'native_batch', 'port'):
                require(paired[ordinal][field].shape == (200064,) and paired[ordinal][field].dtype == torch.float32,
                        'original full head geometry')
            require(all(paired[ordinal][k] == v for k, v in key.items()), 'original paired ordinal')
            require(raw_equal(paired[ordinal]['saved_single'], hf[key['fixture']]['logits'][key.get('step', 0)]),
                    'original official HF logits changed')
            for anchor in ANCHORS:
                require(raw_equal(references[anchor][ordinal][1]['logits'].float(), paired[ordinal][FIELDS[anchor]]),
                        'captured head differs from original reference')
        for arm in ARMS:
            report, frames = arms[arm]
            reproduced[variant + '/' + arm] = {}
            for anchor in ANCHORS:
                label = GATE_NAMES[anchor]; head_rows = []; divergent = {name: [] for name in BOUNDARIES}
                for ordinal, key in enumerate(keys()):
                    budget.check()
                    entry, refs = references[anchor][ordinal]; actual = frames[ordinal]
                    metric = metrics(actual['logits_float32'], paired[ordinal][FIELDS[anchor]])
                    row = dict(**key, **metric, raw_equal=raw_equal(actual['logits_float32'], paired[ordinal][FIELDS[anchor]]))
                    require(row == report['comparisons'][label][ordinal], 'original three-anchor metrics did not reproduce')
                    if arm != 'candidate' and anchor == 'native_single':
                        require(dict(**key, **metric) == old_report['rows'][ordinal], 'original HF metric regression')
                    if arm != 'candidate':
                        require(raw_equal(actual['logits_float32'], paired[ordinal]['port']), 'original control port logits changed')
                    head_rows.append(row)
                    for boundary in BOUNDARIES:
                        ref = refs[boundary].float() if boundary == 'logits' else refs[boundary]
                        value = actual['logits_float32'] if boundary == 'logits' else actual['propagation'][boundary]
                        baseline = arms['unchanged'][1][ordinal]
                        unchanged = baseline['logits_float32'] if boundary == 'logits' else baseline['propagation'][boundary]
                        stats = difference(ref, value)
                        if not stats['raw_equal']: divergent[boundary].append(ordinal)
                        emit(dict(event='comparison', variant=variant, arm=arm, anchor=anchor, ordinal=ordinal,
                            key=key, boundary=boundary, source_stage=BOUNDARIES[boundary][0],
                            reference_location=entry['reference'], actual_payload=inventory['arms'][variant][arm]['payload']['path'],
                            difference=stats, changed_from_unchanged=not raw_equal(value, unchanged),
                            original_logit_metrics=row if boundary == 'logits' else None,
                            selected_prediction_row_only=True, global_first_cause_proven=False))
                        comparisons += 1; heads += int(boundary == 'logits')
                gate = metric_gate(head_rows)
                require(gate == report['gates'][label], 'original gate changed')
                reproduced[variant + '/' + arm][label] = gate
                divergences[variant + '/' + arm + '/' + anchor] = divergent
        del references, arms, hf, paired
    require(comparisons == 5184 and heads == 576, 'complete comparison matrix')
    expected_paths = set(inventory['selected104_payload_paths'])
    require(expected_paths <= store.loaded and len(expected_paths) == 104, 'activation payload omissions')
    summary = dict(event='complete', comparisons=comparisons, head_comparisons=heads,
        preserved_original_gates=reproduced, raw_divergence_ordinals=divergences,
        activation_payloads_verified=104, full_payload_files_loaded=len(store.loaded),
        all_feature_vectors_compared=True, duplicate_logical_occurrences_preserved=True,
        model_qualified=False, first_global_cause_proven=False, gateup_cause_proven=False,
        KV_history_restored=False, new_speedup_claim=False, original_gates_replaced=False)
    emit(summary)
    return summary


if __name__ == '__main__':
    raise SystemExit('Source-only saved analysis. Separate serial supervisor, independent review and runtime admission required.')
