"""Read saved Neuron runtime descriptors without importing torch or constructing models."""
import collections
import hashlib
import importlib.util
import io
import json
import pickle
import struct
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'lfm3-slot-state-crossover'


class Descriptor:
    def __setstate__(self, state):
        self.state = state


class DescriptorReader(pickle._Unpickler):
    """Only inert descriptors and explicitly enumerated data constructors."""
    dispatch = pickle._Unpickler.dispatch.copy()

    def find_class(self, module, name):
        if module.startswith('__torch__.'):
            return type(name, (Descriptor,), {'origin': module + '.' + name})
        key = (module, name)
        if key == ('torch.jit._pickle', 'restore_type_tag'):
            return lambda value, tag: value
        if key == ('torch.jit._pickle', 'build_intlist'):
            return lambda value: list(value)
        if key == ('collections', 'OrderedDict'):
            return collections.OrderedDict
        if key == ('torch._utils', '_rebuild_tensor_v2'):
            return lambda *args: ('tensor_descriptor', args)
        if key == ('torch', 'LongStorage'):
            return 'LongStorage'
        raise ValueError('Unsupported pickle global: ' + repr(key))

    def persistent_load(self, storage):
        return ('storage_descriptor', storage)

    def load_binary_unicode(self):
        # Torch's C++ archive encodes arbitrary bytes as BINUNICODE. Retain them
        # losslessly; standard pickle's strict UTF-8 reader rejects NEFF payloads.
        length = struct.unpack('<I', self.read(4))[0]
        self.append(self.read(length).decode('utf8', 'surrogateescape'))


DescriptorReader.dispatch[pickle.BINUNICODE[0]] = DescriptorReader.load_binary_unicode


def raw(value):
    return value.encode('utf8', 'surrogateescape') if isinstance(value, str) else value


def sha(value):
    return hashlib.sha256(value).hexdigest()


def main():
    proto = Path('/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/torch_neuronx/proto/metaneff_pb2.py')
    spec = importlib.util.spec_from_file_location('local_metaneff_pb2', proto)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = {'models': {}, 'scope': 'Actual saved per-bucket runtime MetaNeff interfaces, '
              'state initializer values and NEFF identity. Only serialized_graph_def is '
              'removed before interface comparison. No model loading, device-state '
              'values, executable scheduling or performance verification.'}
    for label, filename in [('reference', 'lfm3-17b-baseline.json'),
                            ('candidate', 'lfm3-slot-update-full/case.json')]:
        case = json.loads((ROOT / filename).read_text())
        artifact = Path(case['artifact']) / 'model.pt'
        with zipfile.ZipFile(artifact) as archive:
            data = archive.read('model/data.pkl')
        obj = DescriptorReader(io.BytesIO(data)).load().state['nxd_model']
        assert obj.origin == '__torch__.neuronx_distributed.trace.spmd.NxDModel'
        state = obj.state
        initializer = state['state_initializer'].state
        info = {'artifact': str(artifact), 'pickle_sha256': sha(data),
                'input_shape_map': state['input_shape_map'], 'start_rank_id': state['start_rank_id'],
                'state_initializer': initializer, 'graphs': []}
        for kind in ['context_encoding_model', 'token_generation_model']:
            models = state['models'].state[kind].state['models']
            assert len(models) == 5
            for index, model in enumerate(models):
                assert model.origin == '__torch__.torch.classes.neuron.SPMDModel'
                neff, encoded, serialization_tag, local_ranks, world_size = model.state
                meta = module.MetaNeff()
                meta.ParseFromString(raw(encoded))
                graph_def = meta.serialized_graph_def
                meta.ClearField('serialized_graph_def')
                type_counts = collections.Counter(tensor.type for tensor in meta.input_tensors)
                row = {'kind': kind, 'index': index, 'serialization_tag': serialization_tag, 'local_ranks': local_ranks,
                       'world_size': world_size, 'neff_sha256': sha(raw(neff)),
                       'metaneff_sha256': sha(raw(encoded)), 'serialized_graph_def_sha256': sha(graph_def),
                       'runtime_interface_sha256': sha(meta.SerializeToString(deterministic=True)),
                       'inputs': len(meta.input_tensors), 'outputs': len(meta.output_tensors),
                       'alias_count': len(meta.output_aliases_to),
                       'aliases': dict(meta.output_aliases_to), 'input_type_counts': dict(type_counts),
                       'summary_count_fields': {'num_states': meta.num_states,
                         'num_weights': meta.num_weights, 'num_user_inputs': meta.num_user_inputs}}
                # This SDK leaves summary counts at zero; use actual tensor types.
                assert row['alias_count'] == type_counts[module.MetaTensor.INPUT_STATE] == 35
                assert all(meta.input_tensors[index].type == module.MetaTensor.INPUT_STATE
                           for index in meta.output_aliases_to.values())
                assert local_ranks == world_size == 2
                info['graphs'].append(row)
        report['models'][label] = info
        del data, obj, state
    left, right = (report['models'][label] for label in ['reference', 'candidate'])
    report['input_routing_equal'] = left['input_shape_map'] == right['input_shape_map']
    report['state_initializer_equal'] = left['state_initializer'] == right['state_initializer']
    report['all_ten_runtime_interfaces_equal'] = all(
        a['kind'] == b['kind'] and a['index'] == b['index'] and
        a['runtime_interface_sha256'] == b['runtime_interface_sha256']
        for a, b in zip(left['graphs'], right['graphs']))
    # Independently recorded full NEFF byte identities validate the binary reader.
    assert left['graphs'][8]['neff_sha256'] == '427665f1467a9fdc00f6d34421bb44220dc21bc71fdb016ddb0dda181906d6e0'
    assert left['graphs'][9]['neff_sha256'] == 'b200643d02455ff28c69f4fadeca09645f86db61ccfa30fa2b6ab5044d84ca0a'
    report['known_original_neff_hashes_match'] = True
    assert 'torch' not in sys.modules
    report['complete'] = True
    report['source_sha256'] = {Path(__file__).name: sha(Path(__file__).read_bytes()), str(proto): sha(proto.read_bytes())}
    (DEST / 'serialized-cache-interface-audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key not in ('models', 'source_sha256')}))


if __name__ == '__main__':
    main()
