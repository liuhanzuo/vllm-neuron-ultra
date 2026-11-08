"""Lossless bounded tensor snapshots and observational CPU layer hooks."""
from pathlib import Path
import hashlib
import os
import shutil
import zlib
import minicpm5_history_protocol_v1 as p

_written_bytes = 0


def raw(value):
    import torch
    return value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def meta(value):
    return dict(shape=list(value.shape), stride=list(value.stride()), offset=value.storage_offset(), dtype=str(value.dtype), sha256=digest(raw(value)))


def cpu_indices(inputs):
    import torch
    assert len(inputs) == 7
    result = []
    records = []
    for index, value in enumerate(inputs):
        if index < 4:
            assert value.dtype in (torch.int32, torch.int64)
            actual = value.to(torch.int64)
            assert raw(actual.to(value.dtype)) == raw(value)
            records.append(dict(argument=index, source_dtype=str(value.dtype), execution_dtype=str(actual.dtype),
                values_roundtrip_exact=True, reason='CPU scatter/gather index tensors use int64'))
        else:
            actual = value
        result.append(actual)
    return result, records


def write_state(directory,tensors,layouts):
    global _written_bytes
    import torch
    assert len(tensors)==len(layouts) and len(layouts) in (84,168) and not directory.exists()
    p.storage_check();p.no_symlink(directory);directory.mkdir()
    data_directory=p.DATA/directory.relative_to(p.DEST);data_directory.mkdir(parents=True,exist_ok=False)
    rows=[]
    for index,(value,spec) in enumerate(zip(tensors,layouts)):
        assert list(value.shape)==spec['shape'] and str(value.dtype)==spec['dtype']
        cpu=value.detach().cpu().contiguous();data=raw(cpu)
        assert len(data)==p.CACHE_BYTES and bool(torch.isfinite(cpu).all())
        phase=os.environ['TRN1_MINICPM5_HISTORY_PHASE']
        assert phase in (*p.CPU_PHASES,'device_crossover')
        limit=(14*84 if phase in p.CPU_PHASES else 42*168)*p.CACHE_BYTES
        assert _written_bytes+len(data)<=limit<=p.STATE_BUDGET
        assert shutil.disk_usage(p.DATA).free>=len(data)+p.NVME_FLOOR
        path=data_directory/f'{index:03d}.raw'
        with path.open('xb') as stream:stream.write(data)
        _written_bytes+=len(data)
        assert path.stat().st_size==len(data) and p.sha(path)==digest(data)
        rows.append(dict(**spec,file=str(path),encoding='raw_bf16_le',raw_bytes=len(data),raw_sha256=digest(data),
          payload_sha256=digest(data),identity=p.identity(path),nonfinite_elements=0))
    manifest=dict(complete=True,rows=rows,raw_bytes=sum(x['raw_bytes'] for x in rows))
    p.once(directory/'manifest.json',manifest)
    return reference(directory/'manifest.json',tensors=len(rows))


def load_state(path,row):
    import torch
    path=Path(path);assert p.identity(path)==row['identity']
    if row.get('encoding')=='raw_bf16_le':
        assert path.stat().st_size==row['raw_bytes']==p.CACHE_BYTES
        data=path.read_bytes();assert digest(data)==row['payload_sha256']
    else:
        assert 'encoding' not in row and path.suffix=='.zlib' and path.stat().st_size<=p.CACHE_BYTES+2048
        packed=path.read_bytes();assert digest(packed)==row['payload_sha256']
        decoder=zlib.decompressobj();data=decoder.decompress(packed,row['raw_bytes']+1)
        assert decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail
    assert len(data)==row['raw_bytes']==p.CACHE_BYTES and digest(data)==row['raw_sha256'] and p.identity(path)==row['identity']
    value=torch.frombuffer(bytearray(data),dtype=torch.uint8).view(torch.bfloat16).reshape(row['shape'])
    assert int((~torch.isfinite(value)).sum())==row['nonfinite_elements']==0
    return value


def reference(path,**extra):return dict(path=str(path),sha256=p.text_sha(path),**extra)


def manifest(ref,rank=None):
    import minicpm5_history_contract_v1 as c
    assert p.text_sha(ref['path'])==ref['sha256'];value=p.read(ref['path'])
    c.check_manifest(value,None if len(value['rows'])==168 else value['rows'][0]['rank'])
    if rank is not None:
        rows=[x for x in value['rows'] if x['rank']==rank]
        assert len(rows)==84;return rows
    assert len(value['rows'])==168;return value['rows']


def source(index):
    control=p.read(p.DEST/'control-result.json');p.check_control(control);ref=control['reports'][index]
    assert p.text_sha(ref['path'])==ref['sha256'];row=p.read(ref['path'])
    assert row['call']==p.calls()[index];return row


def payload(row):
    import torch
    path=Path(row['payload']);assert p.identity(path)==row['payload_identity'] and p.sha(path)==row['payload_sha256']
    return torch.load(path,map_location='cpu',weights_only=True)


def graph_inputs(row):
    values=payload(row)['inputs'];assert len(values)==7
    assert [meta(x) for x in values]==row['input_metadata']
    p.check_input_metadata(row['input_metadata'],row['call']['phase']);verify_graph_inputs(row['compressed_inputs'],values)
    return values


def post_hashes(ref):return [x['raw_sha256'] for x in manifest(ref)]


def import_device(model,ref,expected):
    import torch
    assert binding(model)==expected
    rows=manifest(ref)
    with torch.inference_mode():
        for row in rows:
            value=model.traced_model.nxd_model.state[row['rank']][row['name']]
            cpu=load_state(row['file'],row);value.copy_(cpu)
            # Exact native roundtrip is required before any compiled forward.
            assert digest(raw(value))==row['raw_sha256']
    assert binding(model)==expected
    return dict(all168_imports_reexport_exact=True,manifest=ref,raw_sha256=[x['raw_sha256'] for x in rows])


def region_record(before,after,spec,enforce_transition=False):
    """All slots/words plus active writes and physical padding, without resets."""
    import torch
    assert before.shape==after.shape and tuple(before.shape)==(4,1,4224,128)
    changed=before.contiguous().view(torch.int16)!=after.contiguous().view(torch.int16)
    slots=[]
    for slot in range(4):
        matching=[(frame,pos) for frame,pos in zip(spec['frames'],spec['logical_positions']) if frame['slot']==slot]
        active=bool(matching);mask=torch.zeros((4224,),dtype=torch.bool)
        if active:
            _,pos=matching[0]
            if spec['phase']=='prefill':mask[:4096]=True
            else:mask[pos]=True
        selected=changed[slot,0]
        slots.append(dict(slot=slot,active=active,changed_words=int(selected.sum()),active_write_changed_words=int(selected[mask].sum()),
          outside_active_write_changed_words=int(selected[~mask].sum()),physical_padding_changed_words=int(selected[4096:].sum())))
    if enforce_transition:
        assert all(x['physical_padding_changed_words']==0 for x in slots)
        assert all(x['changed_words']==0 for x in slots if not x['active'])
    return dict(changed_words=int(changed.sum()),per_slot=slots,before_nonfinite=int((~torch.isfinite(before)).sum()),after_nonfinite=int((~torch.isfinite(after)).sum()))


def binding(model):
    import torch
    state = model.traced_model.nxd_model.state
    assert len(state) == 2 and all(set(x) == {f'kv_mgr.past_key_values.{i}' for i in range(84)} for x in state)
    result = []
    for spec in p.layout():
        value = state[spec['rank']][spec['name']]
        assert list(value.shape) == spec['shape'] and value.dtype == torch.bfloat16
        assert str(value.device) == f'privateuseone:{spec["rank"]}'
        result.append(dict(**spec, tensor_impl=int(value._cdata)))
    assert len({x['tensor_impl'] for x in result}) == 168
    return result


def device_state(model, directory, expected):
    assert binding(model) == expected
    tensors = [model.traced_model.nxd_model.state[x['rank']][x['name']] for x in p.layout()]
    result = write_state(directory, tensors, p.layout())
    assert binding(model) == expected
    return result


class PredictionHooks:
    def __init__(self, model, names=None):
        self.names = p.stages() if names is None else names
        modules = dict(model.named_modules())
        assert len(self.names) == len(set(self.names)) and all(x in modules for x in self.names)
        self.handles = [modules[name].register_forward_hook(self.hook(name)) for name in self.names]
        self.values = {}; self.source_metadata = {}; self.lengths = None; self.prefill = None

    def begin(self, lengths, prefill):
        assert self.lengths is None
        self.values = {}; self.source_metadata = {}; self.lengths = lengths; self.prefill = prefill

    def hook(self, name):
        def observe(module, args, output):
            import torch
            assert self.lengths is not None and name not in self.values
            value = output.hidden_states if hasattr(output, 'hidden_states') else (output[0] if isinstance(output, tuple) else output)
            assert isinstance(value, torch.Tensor) and value.ndim >= 3 and value.shape[0] == len(self.lengths)
            index = ((self.lengths - 1) if self.prefill and value.shape[1] != 1 else torch.zeros_like(self.lengths)).long()
            assert bool((index >= 0).all() and (index < value.shape[1]).all())
            shape = [value.shape[0], 1] + list(value.shape[2:])
            gathered = torch.gather(value, 1, index.reshape([-1] + [1] * (value.ndim - 1)).expand(shape))
            self.values[name] = gathered.detach().cpu().clone().contiguous()
            self.source_metadata[name] = dict(shape=list(value.shape), stride=list(value.stride()),
                offset=value.storage_offset(), dtype=str(value.dtype), prediction_positions=index.tolist())
            # No return value: never replace a module's original output.
        return observe

    def finish(self):
        assert list(self.values) == self.names and list(self.source_metadata) == self.names
        result = self.values, self.source_metadata
        self.values = {}; self.source_metadata = {}; self.lengths = None; self.prefill = None
        return result

    def close(self):
        for handle in self.handles: handle.remove()


def check_tensor_controls():
    import torch
    class Pair(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.a = torch.nn.Identity(); self.b = torch.nn.Identity()
        def forward(self, value):
            return self.b(self.a(value))
    model = Pair(); observer = PredictionHooks(model, ['a', 'b'])
    original = torch.arange(2 * 7 * 4, dtype=torch.float32).reshape(2, 7, 4).to(torch.bfloat16)
    observer.begin(torch.tensor([3, 7]), True)
    result = model(original); values, metadata = observer.finish()
    assert result is original and raw(original) == raw(result)
    for value in values.values():
        assert raw(value) == raw(torch.stack([original[0, 2], original[1, 6]])[:, None])
        assert value.data_ptr() != original.data_ptr()
    observer.begin(torch.tensor([99, 100], dtype=torch.int32), False)
    single = original[:, :1]; assert model(single) is single
    values, _ = observer.finish(); assert all(raw(x) == raw(single) for x in values.values())
    observer.close()
    integer_inputs = [torch.tensor([[0, 1], [2, 130559]], dtype=torch.int32) for _ in range(4)] + [torch.ones(2, 3)]
    integer_inputs += [torch.zeros(2, dtype=torch.int32), torch.zeros(2, dtype=torch.int32)]
    widened, conversions = cpu_indices(integer_inputs)
    assert all(x.dtype == torch.int64 for x in widened[:4]) and widened[4] is integer_inputs[4]
    assert all(a is b and b.dtype == torch.int32 for a, b in zip(integer_inputs[5:], widened[5:]))
    assert all(raw(a) == raw(b.to(torch.int32)) for a, b in zip(integer_inputs[:4], widened[:4]))
    words = torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(torch.bfloat16)
    payload = raw(words)
    assert zlib.decompress(zlib.compress(payload, level=1)) == payload
    assert len(payload) == 131072
    bad = 0
    observer = PredictionHooks(model, ['a', 'b'])
    for lengths in [torch.tensor([0, 1]), torch.tensor([1, 8])]:
        observer.begin(lengths, True)
        try: model(original)
        except AssertionError: bad += 1
        else: raise AssertionError('Invalid prediction index accepted')
        observer.values = {}; observer.source_metadata = {}; observer.lengths = None
    observer.close()
    return dict(passed=True, unchanged_output_identity=True, full_prediction_rows_checked=True,
        decode_absolute_positions_not_used_as_tensor_indices=True, bf16_word_patterns_lossless=65536,
        invalid_prediction_indices_rejected=bad, integer_input_widening_roundtrips=4,
        sdk_placeholders_preserved=2)


def write_graph_inputs(directory, values):
    """All seven actual padded graph inputs, reversible raw bytes and exact ABI metadata."""
    assert len(values)==7 and not directory.exists(); p.no_symlink(directory); directory.mkdir()
    rows=[]
    for name,value in zip(p.INPUT_NAMES,values):
        data=raw(value); packed=zlib.compress(data,1); path=directory/(name+'.zlib')
        assert len(data)<16*1024**2
        with path.open('xb') as stream: stream.write(packed)
        assert zlib.decompress(path.read_bytes())==data
        rows.append(dict(name=name,metadata=meta(value),raw_bytes=len(data),raw_sha256=digest(data),
            payload_sha256=digest(packed),path=str(path),identity=p.identity(path)))
    return rows


def verify_graph_inputs(rows, values):
    assert len(rows)==len(values)==7
    for name,row,value in zip(p.INPUT_NAMES,rows,values):
        assert row['name']==name and row['metadata']==meta(value)
        path=Path(row['path']); assert p.identity(path)==row['identity'] and path.stat().st_size <16*1024**2
        packed=path.read_bytes(); assert digest(packed)==row['payload_sha256']
        decoder=zlib.decompressobj(); data=decoder.decompress(packed,row['raw_bytes']+1)
        assert decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail
        assert len(data)==row['raw_bytes'] and data==raw(value) and digest(data)==row['raw_sha256']
