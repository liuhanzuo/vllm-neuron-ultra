"""Fresh diagnostic payload writer and native hooks for the cold42/84 successor.

All Torch/model access is deferred until an admitted caller constructs Capture.
No raw files are read or written during source preparation. Full feature vectors
at every original prediction position are saved; KV retains the full history.
"""
import hashlib
import inspect
import json
from pathlib import Path
import sys

import deepseek_cpu_capture_bridge_v1 as b


class Capture:
    def __init__(self, backend, destination, data_root, model=None):
        b.require(backend in ('native', 'device'), 'backend')
        self.backend, self.model = backend, model
        self.directory = Path(destination) / 'boundary-captures' / backend
        self.raw_directory = Path(data_root) / backend
        b.require(not self.directory.exists() and not self.raw_directory.exists(), 'fresh capture namespace required')
        b.require('deepseek-cold-kv-diagnostic-v2' not in self.directory.parts
                  and 'deepseek-cold-kv-diagnostic-v2' not in self.raw_directory.parts, 'preserve original payloads')
        self.directory.mkdir(parents=True)
        self.raw_directory.mkdir(parents=True)
        self.frames = []; self.calls = []; self.previous = {}; self.prefixes = {}; self.bindings = None
        self.values = {}; self.handles = []; self.active = False
        if backend == 'native':
            b.require(model is not None, 'already admitted native model required')
            self._native_hooks()

    def _save_json(self, name, value):
        path = self.directory / name
        b.require(not path.exists(), 'refuse overwrite')
        path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

    def _payload(self, name, tensors):
        import torch
        path = self.raw_directory / name
        b.require(not path.exists(), 'refuse overwrite')
        torch.save(tensors, path)
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b''):
                digest.update(chunk)
        return dict(path=str(path), sha256=digest.hexdigest(), bytes=path.stat().st_size)

    def _boundaries(self, values, inputs):
        import torch
        b.require(set(values) == set(b.NAMES), 'all declared boundaries required')
        payload = {}; geometry = []
        for name in b.NAMES:
            tensor = values[name]
            b.require(str(tensor.dtype) == 'torch.bfloat16', 'boundary precision')
            if self.backend == 'device':
                b.require(tensor.shape[-1] == 2, 'both TP ranks required')
            b.require(bool(torch.isfinite(tensor).all()), 'nonfinite captured boundary')
            geometry.append(dict(name=name, shape=list(tensor.shape), stride=list(tensor.stride()), dtype=str(tensor.dtype)))
            payload[name] = tensor.detach().cpu().clone().contiguous()
        record = dict(index=len(self.calls), backend=self.backend, inputs=inputs, boundaries=geometry,
                      rank_axis=-1 if self.backend == 'device' else None,
                      prediction_positions_only=True,
                      payload=self._payload(f'boundaries-{len(self.calls):02d}.pt', payload))
        self._save_json(f'call-{len(self.calls):02d}.json', record)
        self.calls.append(record)
        return record['index']

    def _cache(self, frame, tensors, call_index):
        import torch
        from deepseek_cold_kv_codec_v2 import encode
        import deepseek_cold_kv_contract_v2 as original
        order = original.frames(self.backend)
        b.require(frame == order[len(self.frames)], 'original frame order required')
        lineage = (frame['prefix'], frame.get('slot', frame['fixture']))
        prior = self.previous.get(lineage)
        b.require((prior is None) == (frame['step'] == 0), 'cache history order')
        rows = []; payload = {}; hashes = {}
        for name in original.stages():
            tensor = tensors[name]
            b.require(tuple(tensor.shape) == (8, frame['prefix'] + frame['step'], 128), 'full used KV')
            b.require(tensor.dtype == torch.bfloat16, 'KV BF16')
            row, saved = encode(tensor, None if prior is None else prior['hashes'][name])
            row.update(stage=name, previous_frame=None if prior is None else prior['key'])
            original.check_record(row, frame, self.backend, name)
            rows.append(row); payload[name] = saved; hashes[name] = row['full_sha256']
        key = original.key(frame, self.backend)
        record = dict(frame=frame, key=key, boundary_call=call_index, rows=rows,
                      payload=self._payload(key + '.pt', payload))
        self._save_json(key + '.json', record)
        self.frames.append(record)
        self.previous[lineage] = dict(key=key, hashes=hashes)

    def native(self, cache, prefix, fixture, step, input_ids):
        import torch
        b.require(not self.active and set(self.values) == set(b.NAMES), 'native call incomplete')
        call = self._boundaries(self.values, dict(prefix=prefix, fixture=fixture, step=step, input_ids=input_ids))
        self.values = {}
        tensors = {}
        b.require(len(cache.layers) == b.LAYERS, 'native 36 layers')
        for index, layer in enumerate(cache.layers):
            for side, tensor in (('key', layer.keys), ('value', layer.values)):
                b.require(tuple(tensor.shape) == (1, 8, prefix + step, 128) and tensor.dtype == torch.bfloat16,
                          'original native cache shape and dtype')
                tensors[f'layers.{index}.{side}'] = tensor[0].detach().cpu().clone().contiguous()
        self._cache(dict(prefix=prefix, fixture=fixture, step=step), tensors, call)

    def device(self, model, ids, positions, slots, prefill, route, result):
        import torch
        b.require(tuple(result.capture_names) == b.NAMES and result.capture_rank_axis == -1, 'capture ABI metadata')
        active = slots.tolist()
        # The source successor preserves the original sorted B4 schedule, so
        # captures and restored logits have the same row-to-slot mapping.
        b.require((prefill and len(active) == 1) or active == [0, 1, 2, 3], 'capture row/slot mapping')
        inputs = dict(input_ids=ids.tolist(), position_ids=positions.tolist(), slots=active,
                      physical_prediction_indices=[ids.shape[1]-1] if prefill else [0]*4,
                      capture_rows_in_sorted_slot_order=True, prefill=prefill, route=dict(route))
        call = self._boundaries(dict(zip(b.NAMES, result.captured_tensors)), inputs)
        from deepseek_prior_mask_state_v1 import bindings
        observed = bindings(model)
        if self.bindings is None:
            self.bindings = observed
        b.require(observed == self.bindings, 'original state bindings must remain stable')
        if prefill:
            b.require(len(active) == 1 and ids.shape[0] == 1, 'original CTE B1')
            prefix = ids.shape[1]
            b.require(prefix in b.PREFIXES and positions.tolist() == [list(range(prefix))], 'prefill positions')
            self.prefixes[active[0]] = prefix
            frames = [dict(prefix=prefix, slot=active[0], fixture=active[0] % 2, step=0)]
        else:
            b.require(active == [0, 1, 2, 3] and tuple(ids.shape) == tuple(positions.shape) == (4, 1), 'original TKG B4')
            b.require(len(set(self.prefixes.values())) == 1, 'same cold prefix')
            frames = [dict(prefix=self.prefixes[s], slot=s, fixture=s % 2, step=int(positions[s, 0]) - self.prefixes[s] + 1) for s in active]
            b.require(all(f['step'] in (1, 2) for f in frames), 'original decode steps')
        states = model.traced_model.nxd_model.state
        tensors = [{} for _ in frames]
        for layer in range(b.LAYERS):
            for offset, side in enumerate(('key', 'value')):
                name = f'kv_mgr.past_key_values.{2 * layer + offset}'
                ranks = [states[r][name].detach().cpu() for r in (0, 1)]
                for target, frame in zip(tensors, frames):
                    length = frame['prefix'] + frame['step']; slot = frame['slot']
                    target[f'layers.{layer}.{side}'] = torch.cat([v[slot, :, :length, :] for v in ranks], dim=0).clone().contiguous()
        b.require(bindings(model) == self.bindings, 'capture must not replace state tensors')
        for frame, values in zip(frames, tensors):
            self._cache(frame, values, call)

    def _native_hooks(self):
        # Source pin is checked by the owning native loader; these are hooks on
        # its actual model, not reconstructed attention calculations.
        model = self.model
        b.require(len(model.model.layers) == b.LAYERS, 'original native model')
        source = Path(inspect.getsourcefile(model.model.layers[0].self_attn.forward))
        b.require(hashlib.sha256(source.read_bytes()).hexdigest() ==
                  '64fec98db8cd0edaa740baf026be6adb1d5011e38bfa050ccdf6693b1da9f95d',
                  'original pinned native attention source required')
        self.trace_targets = {}
        for index, layer in enumerate(model.model.layers):
            prefix = f'layers.{index}.'

            def pre(module, name):
                def hook(mod, args):
                    self._native_value(name, args[0])
                self.handles.append(module.register_forward_pre_hook(hook))

            def post(module, name):
                def hook(mod, args, value):
                    # Native decoder and MLP return tensors in the pinned HF.
                    self._native_value(name, value)
                self.handles.append(module.register_forward_hook(hook))

            pre(layer.input_layernorm, prefix + 'layer_input')
            post(layer.input_layernorm, prefix + 'normalized_input')
            post(layer.self_attn.q_norm, prefix + 'q_normalized')
            post(layer.self_attn.k_norm, prefix + 'k_normalized')
            pre(layer.self_attn.o_proj, prefix + 'attention_heads')
            post(layer.self_attn.o_proj, prefix + 'attention_projected')
            pre(layer.post_attention_layernorm, prefix + 'attention_residual')
            post(layer.post_attention_layernorm, prefix + 'mlp_normalized')
            post(layer.mlp, prefix + 'mlp_output')
            post(layer, prefix + 'layer_output')
            function = layer.self_attn.forward.__func__
            lines, start = inspect.getsourcelines(function)
            markers = [start + i for i, line in enumerate(lines) if line.strip() == 'if past_key_values is not None:']
            b.require(len(markers) == 1, 'pinned native post-RoPE trace point')
            self.trace_targets[id(layer.self_attn)] = (function.__code__, markers[0], prefix)

        def begin(mod, args, kwargs):
            b.require(not self.active and not self.values and sys.gettrace() is None, 'nested/stale native capture')
            self.native_indices = kwargs['position_ids'].argmax(dim=1)
            self.active = True
            sys.settrace(self._trace)

        def end(mod, args, output):
            sys.settrace(None)
            self.active = False
            if output is not None:
                b.require(set(self.values) == set(b.NAMES), 'native full boundary capture')
        self.handles.append(model.register_forward_pre_hook(begin, with_kwargs=True))
        self.handles.append(model.register_forward_hook(end, always_call=True))

    def _native_value(self, name, value):
        b.require(self.active and name not in self.values, 'native capture order/duplicates')
        self.values[name] = b.select_prediction(value, name, self.native_indices)

    def _trace(self, frame, event, arg):
        target = self.trace_targets.get(id(frame.f_locals.get('self')))
        if target is None or frame.f_code is not target[0]:
            return None
        if event == 'line' and frame.f_lineno == target[1]:
            for name, variable in (('query', 'query_states'), ('key', 'key_states'), ('value', 'value_states')):
                self._native_value(target[2] + name, frame.f_locals[variable])
        return self._trace

    def finish(self):
        import deepseek_cold_kv_contract_v2 as original
        b.require([r['frame'] for r in self.frames] == original.frames(self.backend), 'full42/84 coverage')
        b.require(len(self.calls) == 42, 'original 42 forward calls per backend')
        b.require(not self.active and not self.values, 'no pending native capture')
        self.close()
        result = dict(complete=True, backend=self.backend, frames=self.frames, calls=self.calls,
                      graph_instrumented=self.backend == 'device', original_model_precision='BF16',
                      full_model_qualified=False, neutrality_proven=False, new_speedup_claim=False,
                      prediction_positions_only=True, boundary_byte_budget=b.boundary_byte_budget(),
                      unobserved=['other prefill token intermediates', 'attention probabilities', 'MLP gate/up/product intermediates'])
        self._save_json('manifest.json', result)
        return result

    def close(self):
        if self.active:
            sys.settrace(None)
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.active = False
