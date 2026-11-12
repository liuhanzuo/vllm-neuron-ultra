"""Execute v2 host-wrapper AST on stdlib list mocks; never import model code."""
import ast
import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from spark26_port_contract_v2 import validate_host_request

ROOT = Path(__file__).resolve().parent


class HostTensor:
    def __init__(self, data, dtype='int32'):
        self.data = copy.deepcopy(data)
        self.dtype = dtype
        self.device = SimpleNamespace(type='cpu')

    def tolist(self):
        return copy.deepcopy(self.data)

    def int(self):
        return HostTensor(self.data, 'int32')

    def __getitem__(self, index):
        return HostTensor(self.data[index], self.dtype)


class HostTorch:
    int32 = 'int32'
    int64 = 'int64'
    bool = 'bool'

    @staticmethod
    def tensor(data, dtype):
        return HostTensor(data, dtype)


class HostF:
    @staticmethod
    def pad(input_tensor, padding, value_to_pad=0, **kwargs):
        fill = kwargs.get('value', value_to_pad)
        if padding != (0, 1):
            raise AssertionError('Only the proposed one-token host padding is mocked')
        return HostTensor([row + [fill] for row in input_tensor.data], input_tensor.dtype)


class HostBase:
    def __init__(self):
        self.config = SimpleNamespace(pad_token_id=2)
        self.default_sampling_params = HostTensor([[1, 1, 1]] * 4, 'float32')
        self.calls = []
        self.reset_calls = 0

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        return 'opaque-base-output'

    def reset(self):
        self.reset_calls += 1


def main():
    path = ROOT / 'spark26_neuron_adapter_v2.py'
    tree = ast.parse(path.read_text())
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'NeuronSparkForCausalLM')
    methods = [copy.deepcopy(n) for n in original.body
               if isinstance(n, ast.FunctionDef) and n.name in ('__init__', 'forward', 'reset')]
    cls = ast.ClassDef(name='CandidateHost', bases=[ast.Name(id='HostBase', ctx=ast.Load())],
                      keywords=[], body=methods, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    scope = {'HostBase': HostBase, 'torch': HostTorch, 'F': HostF, 'validate_host_request': validate_host_request}
    exec(compile(module, str(path) + ':selected-host-methods', 'exec'), scope)
    model = scope['CandidateHost']()
    checks = []
    def check(ok, name):
        if not ok:
            raise AssertionError(name)
        checks.append(name)
    def rejects(fn, name):
        before = len(model.calls)
        try:
            fn()
        except ValueError:
            check(len(model.calls) == before, name)
        else:
            raise AssertionError(name)
    def request(tokens, positions, slots, mask=None):
        return model.forward(input_ids=HostTensor(tokens), position_ids=HostTensor(positions),
                             seq_ids=HostTensor(slots), attention_mask=None if mask is None else HostTensor(mask))

    check(request([[7]], [[0]], [0]) == 'opaque-base-output', 'single-token-prefill-forwards')
    call = model.calls[-1]
    check(call['input_ids'].data == [[7, 2]], 'single-token-prompt-pad-token')
    check(call['position_ids'].data == [[0, 0]], 'single-token-keeps-prefill-position')
    check(call['attention_mask'].data == [[1, 0]], 'single-token-mask-excludes-pad')
    check(len(call['sampling_params'].data) == 1, 'ctx-B1-sampling-sliced-from-B4')
    check(model._spark_seen == {0: 1}, 'single-token-seen-count')
    request([[8]], [[1]], [0])
    check(model._spark_seen == {0: 2}, 'decode-advances-history')
    rejects(lambda: request([[8]], [[3]], [0]), 'reject-skipped-history-before-base-call')
    rejects(lambda: request([[8]], [[1]], [2]), 'reject-decode-without-prefill')
    rejects(lambda: request([[8]], [[1]], [-1]), 'host-rejects-sdk-only-dummy-slot')
    lengths = [511, 513]
    positions = [list(range(n)) + [0] * (513 - n) for n in lengths]
    request([[7] * 513, [8] * 513], positions, [3, 2])
    check([sum(row) for row in model.calls[-1]['attention_mask'].data] == lengths,
          'right-padding-actual-host-mask-lengths')
    check(model._spark_seen == {0: 2, 3: 511, 2: 513}, 'prefill-records-only-requested-slots')
    request([[8], [9], [10]], [[513], [2], [511]], [2, 0, 3])
    check(model._spark_seen == {0: 3, 3: 512, 2: 514}, 'permuted-mixed-length-decode-crosses511')
    check(len(model.calls[-1]['sampling_params'].data) == 3, 'partial-B3-sampling-rows')
    request([[11]], [[0]], [0])
    check(model._spark_seen == {0: 1, 3: 512, 2: 514}, 'slot0-reuse-leaves-other-host-histories')
    request([[12]], [[1]], [0])
    check(model._spark_seen[0] == 2, 'reused-slot-decodes-from-new-history')
    rejects(lambda: request([[7, 8]], [[0, 0]], [1], [[0, 1]]), 'reject-leading-pad-before-model-call')
    rejects(lambda: request([[7], [8]], [[0], [2]], [1, 2]), 'reject-mixed-phase-before-model-call')
    model.reset()
    check(model.reset_calls == 1 and model._spark_seen == {}, 'reset-clears-host-bookkeeping')
    rejects(lambda: request([[9]], [[2]], [0]), 'reset-requires-new-prefill')
    check(not any(name in sys.modules for name in ('torch', 'numpy', 'transformers', 'neuronx_distributed_inference')),
          'no-native-imports')
    return dict(passed=True, checks=len(checks), scope='selected host-wrapper AST on stdlib list mocks only',
                source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (path, Path(__file__), ROOT / 'spark26_port_contract_v2.py')},
                native_tensor_execution=False, model_qualified=False, runtime_admitted=False)


if __name__ == '__main__':
    print(json.dumps(main(), ensure_ascii=False, indent=2))
