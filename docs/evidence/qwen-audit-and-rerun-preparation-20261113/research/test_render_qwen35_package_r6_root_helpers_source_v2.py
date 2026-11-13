"""Source/mock only; rendered fake references stay exclusively in memory."""
import ast
import copy
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock,patch

class NativeBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'torch','numpy','transformers','vllm','vllm_neuron','neuronxcc',
                'torch_neuronx','neuronx_distributed','neuronx_distributed_inference'}:
            raise AssertionError('native import forbidden: '+fullname)

sys.meta_path.insert(0,NativeBlocker())
import render_qwen35_package_r6_root_helpers_source_v2 as p

PAYLOADS={}
for key,(path,sha) in p.PINS.items():
    path=Path(path) if Path(path).is_absolute() else Path(p.Q)/path
    assert path.stat().st_size<4*2**20
    PAYLOADS[key]=path.read_bytes()
    assert hashlib.sha256(PAYLOADS[key]).hexdigest()==sha

def fixture():
    return dict(catalog=dict(path=str(p.Q/'minicpm5-operator-post-r2-serial-workloads-root-catalog-20261112-v117.json'),sha256='a'*64),
        idle=dict(path=str(p.Q/'continuation-handoff-20261112-v594-synthetic-test-only.json'),sha256='b'*64),
        running_handoff=str(p.Q/'continuation-handoff-20261112-v595-qwen35-package-full-r6-running.json'),catalog_rows=164)

def render(bindings=None,payloads=None):
    with patch('builtins.open',side_effect=AssertionError('no file IO')),patch('os.open',side_effect=AssertionError('no fd IO')),\
         patch('os.system',side_effect=AssertionError('no shell')),patch('subprocess.Popen',side_effect=AssertionError('no process')),\
         patch('fcntl.flock',side_effect=AssertionError('no lock')):
        return p.render(PAYLOADS if payloads is None else payloads,fixture() if bindings is None else bindings)

def source(result,key):
    old=p.PINS[key][0];path=old if old.startswith('/') else str(p.Q/old)
    return result['files'][path.replace('r5','r6')].decode()

def parser_only(text):
    tree=ast.parse(text)
    tree.body=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.Assign)) or
        isinstance(n,ast.FunctionDef) and n.name in {'parser','selected_refs'}]
    namespace={};exec(compile(tree,'<mock-cli>','exec'),namespace)
    return namespace

class SourceSuccessor(unittest.TestCase):
    def test_entire_original_sources_recover_exactly(self):
        result=render()
        for key in ('prepare','accept','ack','observer'):
            text=source(result,key)
            path=next(path for path in result['exact_predecessor_reversal'] if path.endswith(p.PINS[key][0].split('/')[-1].replace('r5','r6')))
            for row in reversed(result['exact_predecessor_reversal'][path]):text=text.replace(row['after'],row['before'])
            self.assertEqual(text.encode(),PAYLOADS[key])
        self.assertEqual(len(result['files']),6)
        for key in ('passed','launch_authorized','actual_bindings_checked','actual_prepare_executed','history_resolved','lock_acquired','native_executed'):
            self.assertIs(result[key],False)

    def test_prior_helper_or_budget_mutation_rejected(self):
        for key in PAYLOADS:
            payloads=dict(PAYLOADS);payloads[key]+=b'\n'
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'frozen predecessor changed'):render(payloads=payloads)

    def test_budget_science_caps_and_all_binary_identities_unchanged(self):
        result=render();budget=json.loads(result['files'][str(p.Q/'qwen35-package-full-r6-20261112-resource-budget.json')])
        inputs=json.loads(result['files'][str(p.Q/'qwen35-package-full-r6-20261112-stat-inputs.json')])
        old=json.loads(PAYLOADS['budget']);oldinputs=json.loads(PAYLOADS['stat-inputs'])
        for key in old:
            if key not in {'case','output_root','control_root'}:self.assertEqual(budget[key],old[key])
        for key in old['case']:
            if key not in {'model_path','artifact','compiler_cache','compiler_workdir'}:self.assertEqual(budget['case'][key],old['case'][key])
            else:self.assertTrue(budget['case'][key].endswith('-r6'));self.assertNotEqual(budget['case'][key],old['case'][key])
        self.assertEqual(inputs['binary_inputs'],oldinputs['binary_inputs']);self.assertEqual(len(inputs['binary_inputs']),6)
        self.assertEqual(inputs['case'],budget['case'])
        self.assertEqual(inputs['resource_budget']['sha256'],hashlib.sha256(result['files'][inputs['resource_budget']['path']]).hexdigest())

    def test_bindings_are_explicit_canonical_complete_and_future(self):
        for mode in ('missing','digest','old-catalog','old-idle','wrong-running','wrong-suffix','old-running','bool-rows','old-rows'):
            b=fixture()
            if mode=='missing':del b['catalog']
            elif mode=='digest':b['catalog']['sha256']='pending'
            elif mode=='old-catalog':b['catalog']['path']=b['catalog']['path'].replace('v117','v116')
            elif mode=='old-idle':b['idle']['path']=b['idle']['path'].replace('v594','v591')
            elif mode=='wrong-running':b['running_handoff']=b['running_handoff'].replace('r6','r5')
            elif mode=='wrong-suffix':b['running_handoff']=b['running_handoff'].replace('.json','xjson')
            elif mode=='old-running':b['running_handoff']=b['running_handoff'].replace('v595','v594')
            elif mode=='bool-rows':b['catalog_rows']=True
            else:b['catalog_rows']=163
            with self.subTest(mode=mode),self.assertRaises(ValueError):render(b)

    def test_prepare_accept_cli_require_selected_exact_references(self):
        result=render()
        for key in ('prepare','accept'):
            subject=parser_only(source(result,key));refs=subject['EXACT_REFS'];argv=[]
            for name,(path,sha) in refs.items():argv.extend(['--'+name,path,'--'+name+'-sha256',sha or 'c'*64])
            selected=subject['selected_refs'](subject['parser']().parse_args(argv))
            self.assertEqual(selected['runtime-review'],p.RUNTIME_REVIEW)
            self.assertEqual(selected['preparer-review'],p.PREPARER_REVIEW)
            self.assertEqual(selected['catalog'],fixture()['catalog'])
            self.assertEqual(selected['idle'],fixture()['idle'])
            for bad in ('sha','path'):
                changed=list(argv);index=changed.index('--catalog-sha256' if bad=='sha' else '--catalog')+1
                changed[index]='d'*64 if bad=='sha' else changed[index].replace('v117','v116')
                with self.subTest(key=key,bad=bad),self.assertRaises(ValueError):subject['selected_refs'](subject['parser']().parse_args(changed))

    def test_future_candidate_requires_actual_sha_and_no_invented_digest(self):
        result=render();subject=parser_only(source(result,'accept'))
        self.assertIsNone(subject['EXACT_REFS']['candidate'][1])
        self.assertIsNone(result['required_actual_refs']['candidate']['sha256'])
        with self.assertRaises(SystemExit),patch('sys.stderr'):subject['parser']().parse_args([])

    def test_observer_complete_catalog_count_failure_and_source_mapping(self):
        result=render();namespace={'__name__':'mock_observer'}
        exec(compile(source(result,'observer'),'<mock-observer>','exec'),namespace)
        rows=[dict(terminal={'path':'/mock/terminal'+str(i),'sha256':'a'*64},source_sha256={}) for i in range(164)]
        old={'runs':rows};shape=Mock()
        got=namespace['append_catalog'](old,{'mock':'idle'},{'mock':'terminal'},{'exit_code':1},[],[],{},shape)
        self.assertEqual(got['runs'][:-1],rows);self.assertEqual(len(got['runs']),165);self.assertEqual(len(old['runs']),164)
        with self.assertRaisesRegex(ValueError,'164 rows'):namespace['append_catalog']({'runs':rows[:-1]}, {},{},{},[],[],{},shape)
        audit=Mock();out=namespace['audit_outcome']({'exit_code':1},None,{}, {}, {},Mock(),audit)
        audit.assert_not_called();self.assertFalse(out['full_model_qualified']);self.assertFalse(out['all25_stages_audited'])

    def test_generation_changes_no_caller_inputs(self):
        b=fixture();before=copy.deepcopy(b);payloads=dict(PAYLOADS);render(b,payloads)
        self.assertEqual(b,before);self.assertEqual(payloads,PAYLOADS)

    def test_stage_count_and_resource_lock_history_gates_retained_in_sources(self):
        result=render();accept=source(result,'accept');prepare=source(result,'prepare');ack=source(result,'ack')
        for fragment in ('Bundle(spec_ref, review_ref, guard).resolve()',"len(bound.supplement['scope']['children']) == 874",'lock.acquire()',
                         'lock.release(True)','(2 * 2**30, 2 * 2**30)','(600, 605)','signal.alarm(600)'):
            self.assertIn(fragment,accept)
        self.assertIn('prep.prepare(request, readers.metadata, readers.source, prep.file_identity, guard)',prepare)
        self.assertIn('history.expected_running(',ack)
        observer=source(result,'observer');self.assertIn('final_audit',observer)
        self.assertIn('all25_stages_audited',observer);self.assertIn('source_delta(',observer)

class DatedBindings(unittest.TestCase):
    def dated(self,catalog='20261113',idle='20261113',running='20261113'):
        value=fixture()
        value['catalog']['path']=value['catalog']['path'].replace('20261112',catalog)
        value['idle']['path']=value['idle']['path'].replace('20261112',idle)
        value['running_handoff']=value['running_handoff'].replace('20261112',running)
        return value

    def test_nov12_output_identical_to_frozen_v1(self):
        import render_qwen35_package_r6_root_helpers_source_v1 as old
        with patch('builtins.open',side_effect=AssertionError('no file IO')),patch('os.open',side_effect=AssertionError('no fd IO')):
            expected=old.render(PAYLOADS,fixture())
        self.assertEqual(render(),expected)

    def test_nov13_exact_refs_running_path_and_fixed_r6_namespace(self):
        b=self.dated();result=render(b)
        self.assertEqual(result['required_actual_refs']['catalog'],b['catalog'])
        self.assertEqual(result['required_actual_refs']['idle'],b['idle'])
        for path in result['files']:
            self.assertIn('20261112',path);self.assertNotIn('20261113',path)
        for key in ('prepare','accept'):
            text=source(result,key);self.assertIn(b['running_handoff'].split('/')[-1],text)
            selected=parser_only(text)['EXACT_REFS']
            self.assertEqual(selected['catalog'],(b['catalog']['path'],b['catalog']['sha256']))
            self.assertEqual(selected['idle'],(b['idle']['path'],b['idle']['sha256']))
        for key in ('budget','stat-inputs'):
            path=str(p.Q/('qwen35-package-full-r6-20261112-'+('resource-budget' if key=='budget' else key)+'.json'))
            self.assertEqual(result['files'][path],render()['files'][path])

    def test_cross_midnight_and_later_year_valid_dates(self):
        for dates in [('20261112','20261112','20261113'),('20261112','20261113','20261113'),
                      ('20261113','20261112','20261113'),('20261231','20261231','20270101'),
                      ('20280229','20280229','20280301')]:
            with self.subTest(dates=dates):self.assertFalse(render(self.dated(*dates))['launch_authorized'])

    def test_invalid_calendar_dates_and_noncanonical_spellings_fail(self):
        for field in ('catalog','idle','running'):
            for date in ('20261301','20261131','20270229','20260012','00001113','2026111','2026-11-13','２０２６１１１３'):
                dates=dict(catalog='20261113',idle='20261113',running='20261113');dates[field]=date
                with self.subTest(field=field,date=date),self.assertRaises(ValueError):render(self.dated(**dates))

    def test_date_cannot_move_backwards_despite_larger_versions(self):
        for dates in [('20261113','20261113','20261112'),('20261114','20261113','20261113'),
                      ('20261112','20261114','20261113'),('20261111','20261113','20261113'),
                      ('20261113','20261111','20261113')]:
            with self.subTest(dates=dates),self.assertRaises(ValueError):render(self.dated(*dates))

    def test_cross_date_does_not_reset_version_or_row_thresholds(self):
        for mode in ('catalog','idle','running-equal','running-lower','rows'):
            b=self.dated('20261113','20261113','20261114')
            if mode=='catalog':b['catalog']['path']=b['catalog']['path'].replace('v117','v116')
            elif mode=='idle':b['idle']['path']=b['idle']['path'].replace('v594','v593')
            elif mode=='running-equal':b['running_handoff']=b['running_handoff'].replace('v595','v594')
            elif mode=='running-lower':b['running_handoff']=b['running_handoff'].replace('v595','v593')
            else:b['catalog_rows']=163
            with self.subTest(mode=mode),self.assertRaises(ValueError):render(b)

    def test_dated_refs_still_require_exact_canonical_research_parent(self):
        for field in ('catalog','idle','running_handoff'):
            b=self.dated()
            if field=='running_handoff':b[field]=b[field].replace('/research/','/research/../research/')
            else:b[field]['path']=b[field]['path'].replace('/research/','/research/../research/')
            with self.subTest(field=field),self.assertRaises(ValueError):render(b)

    def test_later_row_count_is_kept_exact_without_state_lookup(self):
        b=self.dated();b['catalog_rows']=165;result=render(b)
        namespace={'__name__':'mock_observer'};exec(compile(source(result,'observer'),'<mock>','exec'),namespace)
        rows=[{'source_sha256':{}} for _ in range(165)]
        self.assertEqual(len(namespace['append_catalog']({'runs':rows},{},{},{},[],[],{},Mock())['runs']),166)
        with self.assertRaisesRegex(ValueError,'165 rows'):
            namespace['append_catalog']({'runs':rows[:-1]},{},{},{},[],[],{},Mock())

    def test_dated_render_all_sources_remain_exactly_reversible(self):
        result=render(self.dated())
        for key in ('prepare','accept','ack','observer'):
            text=source(result,key);name=p.PINS[key][0].split('/')[-1].replace('r5','r6')
            path=next(path for path in result['exact_predecessor_reversal'] if path.endswith(name))
            for row in reversed(result['exact_predecessor_reversal'][path]):text=text.replace(row['after'],row['before'])
            self.assertEqual(text.encode(),PAYLOADS[key])

if __name__=='__main__':unittest.main(verbosity=2)
