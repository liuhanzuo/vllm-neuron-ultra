"""Exercise the exact production call with the original scope assembler.

Only synthetic metadata; no native libraries, history, processes or captures.
"""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'research'), str(ROOT)]
from research import compare_minicpm5_layer1_saved_operators_v1 as previous


def scope_call(version, context):
    path = ROOT / 'research' / ('minicpm5_layer1_fixed_input_pv_prerequisites_' + version + '.py')
    tree = ast.parse(path.read_text())
    fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'authenticate')
    call = next(x.value for x in fn.body if isinstance(x, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'old_scope' for t in x.targets))
    historical = dict(owners=[dict(pid=2)], units=['historical-unit'], runs=[])
    model = dict(owner=dict(pid=3), coordinator=dict(pid=4), native=dict(pid=5))
    recorder = dict(owner=dict(pid=6))
    return eval(compile(ast.Expression(call), str(path), 'eval'),
                dict(previous=previous, context=context, historical=historical, model=model, recorder=recorder))


class HistoricalScope(unittest.TestCase):
    def context(self):
        return dict(resolved=dict(current_scope=dict(owners=[dict(pid=1)],
                    complete_known_units=['original-unit'], preserved='entire-scope')),
                    owned=SimpleNamespace(contract=dict(unit='authenticated-model-unit')))

    def test_original_call_reproduces_actual_missing_contract(self):
        with self.assertRaisesRegex(KeyError, 'contract'):
            scope_call('v1', self.context())

    def test_new_call_retains_all_owners_and_authenticated_contract_unit(self):
        result = scope_call('v2', self.context())
        self.assertEqual(result['owners'], [dict(pid=i) for i in range(1, 7)])
        self.assertEqual(result['complete_known_units'],
                         ['authenticated-model-unit', 'historical-unit', 'original-unit'])
        self.assertEqual(result['preserved'], 'entire-scope')
        self.assertEqual(result['later_serial']['owners'], [dict(pid=2)])

    def test_missing_authenticated_contract_still_fails(self):
        context = self.context(); context['owned'] = SimpleNamespace()
        with self.assertRaises(AttributeError):
            scope_call('v2', context)

    def test_full_cgroup_derived_units_still_retained(self):
        context = self.context()
        context['resolved']['current_scope']['owners'][0]['cgroup'] = (
            '0::' + previous.r.p.GROUP_PREFIX + 'trn1-history-extra.service')
        self.assertIn('trn1-history-extra', scope_call('v2', context)['complete_known_units'])


if __name__ == '__main__':
    unittest.main()
