"""Transparent future graph/request observer; no native imports on import."""
import base64
import hashlib
import json
import os
from pathlib import Path
import time
import zlib


def tensor_record(value, torch):
    """Pack logical row-major values into new CPU storage; never write source.

    contiguous() may return a singleton view with last stride zero. Allocating
    a one-dimensional destination forces stride1 even for scalar/empty inputs;
    copy_ acts only on the new destination and preserves the source dtype bits.
    """
    if not isinstance(value, torch.Tensor):
        raise TypeError('Graph input is not a tensor')
    count=value.numel();expected=count*value.element_size()
    if expected>2**20:
        raise RuntimeError('Unexpected graph input size')
    shape=list(value.shape)
    layout=dict(stride=list(value.stride()),storage_offset=value.storage_offset(),
                storage_nbytes=value.untyped_storage().nbytes(),contiguous=value.is_contiguous(),device=str(value.device))
    cpu=value.detach().cpu()
    packed=torch.empty((count,),dtype=cpu.dtype,device='cpu')
    if packed.stride()!=(1,) or packed.storage_offset()!=0 or packed.numel()!=count:
        raise RuntimeError('Packed destination layout differs')
    packed.view(shape).copy_(cpu)
    raw=packed.view(torch.uint8).numpy().tobytes()
    if len(raw)!=expected or list(cpu.shape)!=shape or packed.dtype!=value.dtype:
        raise RuntimeError('Packed serialization size/shape/dtype differs')
    after=dict(stride=list(value.stride()),storage_offset=value.storage_offset(),
               storage_nbytes=value.untyped_storage().nbytes(),contiguous=value.is_contiguous(),device=str(value.device))
    if layout!=after or list(value.shape)!=shape:
        raise RuntimeError('Serialization changed original layout')
    return dict(dtype=str(value.dtype),shape=shape,raw_bytes=len(raw),original_layout=layout,
                raw_sha256=hashlib.sha256(raw).hexdigest(),codec='zlib+base64',
                data=base64.b64encode(zlib.compress(raw,1)).decode())


def install(role, directory, model_class, runner_class, torch):
    """Called only after admitted native startup. Never edits tensor/model inputs."""
    path = Path(directory) / ('observer-' + str(os.getpid()) + '.jsonl')
    if path.exists():
        raise RuntimeError('Observer path already exists')
    context = {'step': None, 'steps': 0, 'event': 0, 'bytes': 0, 'ever_engine':False}

    def emit(kind, **fields):
        context['event'] += 1
        row = dict(kind=kind, role=role, pid=os.getpid(), event=context['event'],
                   engine_step=context['step'], **fields)
        raw = (json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n').encode()
        if len(raw) > 256 * 2**10 or context['bytes'] + len(raw) > 64 * 2**20:
            raise RuntimeError('Observer metadata cap exceeded')
        with path.open('ab') as f:
            f.write(raw); f.flush()
        context['bytes'] += len(raw)

    def tensor(value):
        return tensor_record(value,torch)

    original_load = model_class.load
    original_execute = runner_class.execute_model
    original_sample = runner_class._sample

    def load(self, *args, **kwargs):
        emit('model_load_begin',internal_load_graphs_not_observed=True)
        result = original_load(self, *args, **kwargs)
        if getattr(self, '_token_capture_observer_installed', False):
            raise RuntimeError('Repeated model load/observer installation')
        self._token_capture_observer_installed = True
        for tag in ('context_encoding_model', 'token_generation_model'):
            wrapper = getattr(self, tag)
            forward = wrapper._forward

            def observe(*inputs, _tag=tag, _forward=forward):
                if context['step'] is None and context['ever_engine']:
                    raise RuntimeError('Graph outside engine after request execution began')
                if len(inputs) != 7:
                    raise RuntimeError('Expected all seven original graph inputs')
                shapes = [list(value.shape) for value in inputs]
                route = self.traced_model.nxd_model.input_shape_map[str(shapes)]
                if route[0] != _tag or (_tag == 'token_generation_model' and shapes[1][1] != 2048):
                    raise RuntimeError('Graph routing changed')
                emit('graph_inputs',tag=_tag,route=list(route),inputs=[tensor(x) for x in inputs],
                     scope='request' if context['step'] is not None else 'startup')
                return _forward(*inputs)

            wrapper._forward = observe
        emit('model_loaded', decode_floor=self.token_generation_model._smollm3_minimum_decode_bucket,
             additional_reset_calls=0, compile_calls=0)
        return result

    def execute(self, scheduler_output, *args, **kwargs):
        if context['step'] is not None:
            raise RuntimeError('Concurrent/reentrant engine execution not admitted')
        context['steps'] += 1; context['step'] = context['steps'];context['ever_engine']=True
        emit('engine_begin', finished_req_ids=sorted(scheduler_output.finished_req_ids),
             num_scheduled_tokens=dict(scheduler_output.num_scheduled_tokens),
             total_num_scheduled_tokens=scheduler_output.total_num_scheduled_tokens,
             slot_mapping_before=dict(self.vllm_req_to_neuron_seq_id_mapping),
             free_slots_before=sorted(self.free_seq_ids))
        try:
            result = original_execute(self, scheduler_output, *args, **kwargs)
            emit('engine_end', slot_mapping_after=dict(self.vllm_req_to_neuron_seq_id_mapping),
                 free_slots_after=sorted(self.free_seq_ids))
            return result
        except BaseException as error:
            emit('engine_failure', error=repr(error))
            raise
        finally:
            context['step'] = None

    def sample(self, hidden_states, model_input):
        # The original _sample reorders model_input.request_ids to input_batch
        # order before calling _cpu_sample. Keep both orders, never enumerate one
        # to label the other. Read no full logits here and do not replace them.
        before_order = list(model_input.request_ids)
        after_order = list(self.input_batch.req_ids)
        mapping = dict(self.vllm_req_to_neuron_seq_id_mapping)
        metadata = {key: tensor(getattr(model_input, key)) for key in
                    ('input_tokens', 'position_ids', 'slot_mapping', 'sampling_params')
                    if getattr(model_input, key, None) is not None}
        scope='request'if context['step']is not None else'startup'
        if scope=='startup'and context['ever_engine']:
            raise RuntimeError('Sample outside engine after request execution began')
        try:result = original_sample(self, hidden_states, model_input)
        except BaseException as error:
            emit('sample_failure',scope=scope,model_request_order=before_order,sampled_request_order=after_order,
                 request_to_slot=mapping,model_inputs=metadata,error=repr(error))
            raise
        emit('sample', model_request_order=before_order, sampled_request_order=after_order,
             request_to_slot=mapping, model_inputs=metadata,scope=scope,
             sampled_token_ids=result.sampled_token_ids.detach().cpu().tolist(),
             on_device_sampling=self.model.neuron_config.on_device_sampling_config is not None)
        return result

    model_class.load = load
    runner_class.execute_model = execute
    runner_class._sample = sample
    emit('installed', time=time.time(), model_class=model_class.__name__, runner_class=runner_class.__name__,
         tensor_mutations=False, cache_readback=False, logits_capture=False,
         observation_timing_not_benchmark=True)
