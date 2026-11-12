"""Stdlib/list-mock checks only; no real host/native/tensor execution here."""
import ast
import copy
from contextlib import contextmanager
import itertools
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import spark26_native_port_diagnostic_contract_v1 as old_contract
import spark26_native_port_diagnostic_contract_v2 as contract
import spark26_native_port_diagnostic_v2 as old_core
import spark26_native_port_diagnostic_v3 as core


class Rows:
    """Opaque rows for orchestration tests; deliberately not a numeric tensor."""
    def __init__(self, rows, dtype='mock_bf16'):
        self.rows, self.dtype = copy.deepcopy(rows), dtype
        self.shape = (len(rows), len(rows[0])) if rows and isinstance(rows[0], list) else (len(rows),)
    def clone(self): return Rows(self.rows, self.dtype)
    def contiguous(self): return self
    def view(self, dtype): return Rows(self.rows, dtype)
    def tolist(self): return copy.deepcopy(self.rows)
    def __getitem__(self, index):
        if isinstance(index, slice): return Rows(self.rows[index], self.dtype)
        return copy.deepcopy(self.rows[index])


class ListTorch:
    int32, long, uint8 = 'mock_int32', 'mock_long', 'mock_u8'
    @staticmethod
    def tensor(rows, dtype): return Rows(rows, dtype)
    @staticmethod
    def cat(tensors, dim=0):
        assert dim == 0
        return Rows([row for tensor in tensors for row in tensor.rows], tensors[0].dtype)
    @staticmethod
    def index_select(tensor, dim, index):
        assert dim == 0
        return Rows([tensor.rows[i] for i in index.rows], tensor.dtype)
    @staticmethod
    def equal(left, right): return left.rows == right.rows and left.dtype == right.dtype


def mocked_host(events, *, bad_sort=False, bad_restore=False):
    """Mock provider cannot pass real runtime provider provenance validation."""
    class Host:
        @staticmethod
        def _sort_inputs(inputs, index):
            events.append('sort_inputs')
            return {key: ListTorch.index_select(value, 0, index) for key, value in inputs.items()}
        @contextmanager
        def _reordered(self, ids, **inputs):
            events.append('reordered')
            assert self.is_reorder_needed is True
            order = sorted(range(len(ids.rows)), key=lambda i: ids.rows[i])
            if bad_sort: order = list(reversed(order))
            index = Rows(order, ListTorch.long)
            sorted_ids = ListTorch.index_select(ids, 0, index)
            reordered = self._sort_inputs(inputs, index)
            inverse = Rows([order.index(i) for i in range(len(order))], ListTorch.long)
            def restore(tensor):
                events.append('restore')
                return tensor if bad_restore else ListTorch.index_select(tensor, 0, inverse)
            yield sorted_ids, reordered, restore
    return Host


class MockCache:
    def __init__(self, *, bad_slot_write=False, mutate=False, bad_return_order=False):
        self.calls = []
        self.bad_slot_write, self.mutate, self.bad_return_order = bad_slot_write, mutate, bad_return_order
    def decode_cache(self, key, value, old_k, old_v, ids, positions, sliding):
        self.calls.append(ids.rows[:])
        next_k, next_v = old_k.clone(), old_v.clone()
        for row, slot in enumerate(ids.rows):
            if slot == -1: continue
            target = row if self.bad_slot_write else slot
            next_k.rows[target], next_v.rows[target] = key.rows[row][:], value.rows[row][:]
        if self.mutate: old_k.rows[0] = ['changed']
        out_k = key.clone()
        if self.bad_return_order: out_k.rows.reverse()
        return (out_k, value.clone(), positions.clone(), Rows([[True] for _ in ids.rows], 'mock_bool'), next_k, next_v)


class SuccessorChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.plan = contract.plan()
    def authority(self):
        return dict(schema='spark26_native_port_diagnostic_authority_v1', model=contract.original.MODEL_ID,
                    revision=contract.original.REVISION, plan_sha256=contract.digest(self.plan),
                    checkpoint_root='/future/spark', native_python=contract.PYTHON,
                    lock_identity=dict(contract.LOCK_IDENTITY), native_sources_authenticated=True,
                    native_libraries_authenticated=True, actual_owned_serial_worker=True,
                    source_sha256=dict(contract.NATIVE_SOURCES))

    def test_complete_original_plan_is_preserved(self):
        new = copy.deepcopy(self.plan)
        for key in ('original_plan_sha256', 'host_reorder_check', 'physical_row_order_check'): new.pop(key)
        new['schema'] = 'spark26_native_port_diagnostic_plan_v1'
        self.assertEqual(new, old_contract.plan())
        self.assertEqual(len(new['cache_trace']), 534)
        self.assertEqual(len(new['qkv_weights']) * len(new['projection_shapes']) * len(new['tp_ranks']), 216)
        self.assertIs(core.qkv_diagnostic, old_core.qkv_diagnostic)
        self.assertIs(core.OriginalWeights, old_core.OriginalWeights)

    def test_snapshots_are_all_four_original_batch_sizes(self):
        rows = self.plan['physical_row_order_check']['snapshots']
        self.assertEqual([r['step'] for r in rows], [4, 5, 6, 7])
        self.assertEqual({r['active_rows'] for r in rows}, {1, 2, 3, 4})
        for row in rows: self.assertEqual(row['event'], self.plan['cache_trace'][row['step']])
        permutations = self.plan['physical_row_order_check']['permutations']
        self.assertEqual(permutations, [list(p) for p in itertools.permutations(range(4))])
        self.assertEqual(self.plan['physical_row_order_check']['total_additional_calls'], 384)

    def test_contract_refuses_removed_or_altered_additions(self):
        for mutate in (lambda p:p['cache_trace'].pop(), lambda p:p['qkv_weights'].pop(),
                       lambda p:p['physical_row_order_check']['snapshots'].pop(),
                       lambda p:p['physical_row_order_check']['permutations'].pop(),
                       lambda p:p['host_reorder_check'].update(methods=[]),
                       lambda p:p['original_logit_criteria'].update(kl_max_nats=.03)):
            value = copy.deepcopy(self.plan); mutate(value)
            with self.assertRaises(ValueError): contract.validate_plan(value)

    def test_host_source_pin_required_before_factory(self):
        value = self.authority(); value['source_sha256'].pop(contract.HOST_LOADER); calls=[]
        with self.assertRaisesRegex(ValueError, 'host loader'):
            core.run(self.plan, '/future/spark', lambda:value, lambda:calls.append('native'), lambda *a:None)
        self.assertEqual(calls, [])

    def test_host_interface_and_restore_on_all64_request_orders(self):
        count = 0
        for k in range(1,5):
            for slots in itertools.permutations(range(4),k):
                events=[]; guard=[]
                event=dict(slots=list(slots),positions=[100+s for s in slots],epochs=[2]*k)
                keys={s:Rows([[s, 10+s]]) for s in slots}; values={s:Rows([[s, -10-s]]) for s in slots}
                result=core.real_host_reorder(ListTorch,mocked_host(events),event,keys,values,lambda:guard.append('g'))
                self.assertEqual(result['ids'].rows,sorted(slots))
                self.assertEqual(events,['reordered','sort_inputs']+['restore']*4)
                self.assertEqual(len(guard),2);count+=1
        self.assertEqual(count,64)

    def test_wrong_host_sort_or_restore_fails(self):
        event=dict(slots=[3,1],positions=[10,20],epochs=[0,1])
        keys={s:Rows([[s]]) for s in event['slots']}
        for kwargs in ({'bad_sort':True},{'bad_restore':True}):
            with self.assertRaises(ValueError):
                core.real_host_reorder(ListTorch,mocked_host([],**kwargs),event,keys,keys,lambda:None)

    def run_order_mock(self, candidate=None, k=2):
        slots=[3,1,0,2][:k];active=set(slots)
        ids=Rows([s if s in active else -1 for s in range(4)],ListTorch.int32)
        positions=Rows([[s+10] for s in range(4)],ListTorch.int32)
        key=Rows([[s,100+s] for s in range(4)]);value=Rows([[s,-100-s] for s in range(4)])
        old_k=Rows([[s,20+s] for s in range(4)]);old_v=Rows([[s,-20-s] for s in range(4)])
        reference=MockCache().decode_cache(key,value,old_k,old_v,ids,positions,True)
        output=[];guard=[];candidate=candidate or MockCache()
        count=core.physical_order_diagnostic(ListTorch,candidate,old_k,old_v,key,value,ids,positions,True,
            reference,self.plan['physical_row_order_check']['permutations'],lambda:guard.append(1),
            lambda *args:output.append(args),{'layer':0,'rank':1,'step':4,'active_rows':k})
        return count,candidate.calls,output,guard

    def test_all24_row_permutations_are_exercised_for_each_batch_size(self):
        for k in range(1,5):
            count,calls,output,guard=self.run_order_mock(k=k)
            self.assertEqual((count,len(calls),len(output),len(guard)),(24,24,24,24))
            self.assertEqual([row[1]['permutation'] for row in output],self.plan['physical_row_order_check']['permutations'])
            self.assertTrue(any([s for s in row if s!=-1]!=sorted(s for s in row if s!=-1) for row in calls) if k>1 else True)

    def test_wrong_persistent_slot_write_fails(self):
        with self.assertRaisesRegex(ValueError,'Persistent cache'): self.run_order_mock(MockCache(bad_slot_write=True),k=4)

    def test_wrong_return_row_order_fails(self):
        with self.assertRaisesRegex(ValueError,'Inverse physical'): self.run_order_mock(MockCache(bad_return_order=True))

    def test_input_cache_mutation_fails(self):
        with self.assertRaisesRegex(ValueError,'prior cache'): self.run_order_mock(MockCache(mutate=True))

    def test_runtime_host_provider_cannot_be_replaced_by_mock(self):
        provider={key:None for key in ['torch','safe_open','configuration','modeling','candidate','create_local_weight_qkv']}
        provider['NeuronModelBase']=mocked_host([])
        with patch.object(core.previous,'validate_native_objects',lambda value:None):
            with self.assertRaisesRegex(ValueError,'Actual imported'):core.validate_native_objects(provider)

    def test_runtime_source_does_not_extract_or_execute_AST(self):
        source=Path(core.__file__).read_text();tree=ast.parse(source)
        calls=[n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
        self.assertFalse({'exec','eval','compile'}&set(calls))
        self.assertNotIn('import ast',source)

    def test_no_native_imported(self):
        self.assertFalse({'torch','numpy','transformers','safetensors','ctypes','neuronx_distributed'}&set(sys.modules))


if __name__=='__main__': unittest.main(verbosity=2)
