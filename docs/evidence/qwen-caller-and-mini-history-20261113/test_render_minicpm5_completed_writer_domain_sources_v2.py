"""Pure source tests; synthetic completion objects and generated modules are in memory."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

Q = Path(__file__).resolve().parent
sys.path[:0] = [str(Q), str(Q.parent)]
import test_render_minicpm5_completed_writer_domain_sources_v1 as oldtest
import render_minicpm5_completed_writer_domain_sources_v2 as renderer
r2 = renderer.r2


def pinned_json(name, sha):
    raw = (Q/name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == sha
    return json.loads(raw)

R2_SPEC = pinned_json('qwen35-writer-geometry-r2-20261112-spec.json',
    '9208483c46d292a1f1fa7224e9033bf177c86dd231e3e64e0592bc823fed5a46')
R2_OVERLAY = pinned_json('qwen35-writer-geometry-r2-20261112-history-overlay.json',
    'a2a59f46e34c7dd9f8f385f56335184044a17bf51c0bb05c372e8fd6b8132154')
envelope = oldtest.envelope


def _fixture(success=False):
    inputs = oldtest.fixture()
    old = json.loads(inputs['catalog']['payload'])
    runtime = dict(R2_SPEC['source_sha256'], **r2.ROOT_HELPERS, **renderer.R3_RUNTIME_PINS)
    overlay = copy.deepcopy(R2_OVERLAY)
    overlay['current_serial_catalog'] = inputs['catalog']['reference']
    overlay['source_extension'] = dict(overlay['source_extension'], **old['runs'][163]['source_sha256'],
                                      **renderer.R3_RUNTIME_PINS)
    inputs['r3_overlay'] = envelope('synthetic-r3-overlay-test-only.json', overlay)
    spec = dict(schema='qwen35_writer_geometry_serial_spec_v1', run_name='writer-r3-20261113', current_serial_catalog=inputs['catalog']['reference'],
        history_overlay=inputs['r3_overlay']['reference'], source_sha256=runtime)
    inputs['r3_spec'] = envelope('synthetic-r3-spec-test-only.json', spec)
    actual = dict(kind='tool_session', session_id=123456, terminal_chunk='abc123', exit_code=0 if success else 1)
    ended = dict(schema='qwen35_writer_geometry_root_actual_exit_v1',reviewer='/root',actual_tool=actual,
        actual_tool_exit_observed=True,actual_outer_exit_observed=True,owners=[],units=['synthetic-r3-unit'],
        owners_gone=True,physically_drained=True,original_lock_held_during_audit=True,lock_identity=r2.LOCK_IDENTITY,
        geometry_qualified=success,full_model_qualified=False,new_speedup_claim=False,reconciliation={'passed':success},
        completed_layouts=8 if success else 1,completed_calls=40000 if success else 5000,
        compiled_graphs=32 if success else 4,auditor_source=renderer.AUDITOR,spec=inputs['r3_spec']['reference'],
        reports={},source_sha256={})
    inputs['r3_ended'] = envelope('synthetic-r3-ended-test-only.json', ended)
    recovered = {p: s for p, s in runtime.items() if p not in overlay['source_extension']}
    prior = renderer.merge(recovered, *(row['source_sha256'] for row in old['runs']))
    executed = renderer.merge(runtime, renderer.R3_HELPERS)
    delta = {p: s for p, s in executed.items() if p not in prior}
    composition = dict(recovered_sources=len(recovered),retained_prior_sources=len(prior),runtime_sources=len(runtime),
        post_audit_sources=len(executed),new_row_sources=len(delta),all_previous_rows_retained=True,
        exact_full_mapping_reconstructed=True)
    catalog = copy.deepcopy(old)
    catalog['runs'].append(dict(terminal=inputs['r3_ended']['reference'],tool=actual,owners=ended['owners'],
        units=ended['units'],source_sha256=delta))
    catalog['current_handoff'] = dict(path=str(Q/'synthetic-r3-idle-test-only.json'),sha256='d'*64)
    inputs['r3_catalog'] = envelope('synthetic-r3-catalog-test-only.json', catalog)
    final = dict(schema='qwen35_writer_geometry_root_finalization_v1',reviewer='/root',
        actual_exit=inputs['r3_ended']['reference'],actual_outer_tool=actual,
        actual_independent_auditor_tool=dict(kind='tool_session',session_id=123457,terminal_chunk='def456',exit_code=0),
        source_composition=composition,external_sources=[dict(path=p,sha256=s) for p,s in renderer.R3_HELPERS.items()],
        complete_old_catalog=inputs['catalog']['reference'],catalog=inputs['r3_catalog']['reference'],
        idle_handoff=catalog['current_handoff'],two_current_drain_observations=[{'synthetic-r3-unit':{'empty':True}}]*2,
        all_prior164_rows_unchanged=True,total_rows=165,original_8MiB_cap_preserved=True,
        original_lock_identity=r2.LOCK_IDENTITY,geometry_qualified=success,qualification=None,
        full_model_qualified=False,new_speedup_claim=False)
    inputs['r3_qualification'] = None
    if success:
        proof = dict(schema='qwen35_writer_geometry_independent_qualification_v1',reviewer='/root',passed=True,
            blockers=[],actual_exit=inputs['r3_ended']['reference'],reports=ended['reports'],
            source_sha256=ended['source_sha256'],full_model_qualified=False)
        inputs['r3_qualification'] = envelope('synthetic-r3-qualified-test-only.json', proof)
        final['qualification'] = inputs['r3_qualification']['reference']
    inputs['r3_finalization'] = envelope('synthetic-r3-final-test-only.json', final)
    return inputs


_FIXTURES = {}

def fixture(success=False):
    if success not in _FIXTURES:
        _FIXTURES[success] = _fixture(success)
    return copy.deepcopy(_FIXTURES[success])



def mutate(inputs, key, change):
    """Keep dependent hashes valid so negatives reach the intended semantic gate."""
    oldtest.mutate(inputs, key, change)
    if key == 'r3_finalization':
        return inputs
    def update(field, fn):
        if inputs[field] is None:
            return
        value=json.loads(inputs[field]['payload']); fn(value)
        inputs[field]=envelope(Path(inputs[field]['reference']['path']).name,value)
    if key == 'r3_overlay':
        update('r3_spec', lambda v:v.update(history_overlay=inputs['r3_overlay']['reference']))
    if key in ('r3_overlay', 'r3_spec'):
        update('r3_ended', lambda v:v.update(spec=inputs['r3_spec']['reference']))
    if key in ('r3_overlay', 'r3_spec', 'r3_ended'):
        update('r3_catalog', lambda v:v['runs'][164].update(terminal=inputs['r3_ended']['reference']))
        update('r3_qualification', lambda v:v.update(actual_exit=inputs['r3_ended']['reference']))
    update('r3_finalization', lambda v:v.update(actual_exit=inputs['r3_ended']['reference'],
        catalog=inputs['r3_catalog']['reference'], qualification=inputs['r3_qualification']['reference']
        if inputs['r3_qualification'] is not None else None))
    return inputs


def render(inputs=None):
    with patch('builtins.open', side_effect=AssertionError('No IO')), \
         patch('os.open', side_effect=AssertionError('No FD IO')), \
         patch('fcntl.flock', side_effect=AssertionError('No lock')), \
         patch('subprocess.Popen', side_effect=AssertionError('No process')):
        return renderer.render(oldtest.PREDECESSORS, fixture() if inputs is None else inputs)


def domain(result):
    module = types.ModuleType('synthetic_r3_domain')
    exec(compile(result['files'][str(Q/'minicpm5_completed_qwen_package_text_v4.py')], '<memory>', 'exec'), module.__dict__)
    return module


def reader(inputs):
    return oldtest.reader({k: v for k, v in inputs.items() if v is not None})


class R3WriterTests(unittest.TestCase):
    def test_failed_and_successful_r3_allowed_with_exact_full_history(self):
        for success in (False, True):
            inputs=fixture(success); result=render(inputs); module=domain(result)
            self.assertEqual(len(result['files']),7)
            self.assertGreater(len(inputs['r3_spec']['payload']),8*2**20)
            self.assertEqual(result['completed_prefix_rows'],165)
            self.assertEqual(len(result['accepted_new_text_domain']),42)
            self.assertEqual(module.authenticate_completed_writer(inputs['r3_catalog']['reference'],reader(inputs),Mock()),
                dict(r2.EXTERNAL_TEXT,**renderer.R3_HELPERS))
            for key in ('source_accepted','launch_authorized','actual_inputs_independently_verified',
                'history_executed','native_executed','failed_qualification_absence_independently_verified'):
                self.assertIs(result[key],False)

    def test_exact_mechanical_sources_and_budgets_unchanged(self):
        inputs=fixture(); result=render(inputs)
        original=r2.render(oldtest.PREDECESSORS,{k:inputs[k] for k in renderer.R2_KEYS})
        target=str(Q/'minicpm5_completed_qwen_package_text_v4.py')
        for path,raw in result['files'].items():
            if path!=target:self.assertEqual(raw,original['files'][path])
        self.assertTrue(result['files'][target].startswith(original['files'][target]))
        for key in ('exact_predecessor_reversal','sdk_worker_unchanged','reconstruction_unchanged',
                'expected_bootstrap_sources','expected_catalog_source_refs','expected_helper_source_refs'):
            self.assertEqual(result[key],original[key])

    def test_all_nine_runtime_additions_pass_original_ordinary_domain(self):
        inputs=fixture(); module=domain(render(inputs))
        extension=dict(r2.EXTERNAL_TEXT,**renderer.R3_HELPERS,**renderer.R3_RUNTIME_PINS)
        self.assertEqual(module.extension_shape(extension,completed_lfm_text={},
            catalog_ref=inputs['r3_catalog']['reference'],read=reader(inputs)),extension)
        for path in renderer.R3_RUNTIME_PINS:
            with self.subTest(path=path),self.assertRaisesRegex(ValueError, 'nine accepted'):
                render(mutate(fixture(),'r3_spec',lambda v,p=path:v['source_sha256'].pop(p)))

    def test_changed_and_missing_historical_prefix_rejected(self):
        changes=[lambda v:v['runs'].pop(),lambda v:v['runs'].append(v['runs'][-1]),
            lambda v:v['runs'][0]['tool'].update(exit_code=99),
            lambda v:v['runs'][163]['source_sha256'].pop(next(iter(r2.ROOT_HELPERS))),
            lambda v:v.update(completed_sdk={})]
        for change in changes:
            with self.assertRaises(ValueError):render(mutate(fixture(),'r3_catalog',change))

    def test_exact_four_helpers_no_draft_neighbor_subset_or_changed_hash(self):
        changes=[lambda v:v['external_sources'].pop(),lambda v:v['external_sources'].append(v['external_sources'][0]),
            lambda v:v['external_sources'][0].update(sha256='f'*64),
            lambda v:v['external_sources'][0].update(path='/home/ubuntu/finalize_qwen35_writer_geometry_r3_20261113_v1.py')]
        for change in changes:
            with self.assertRaises(ValueError):render(mutate(fixture(),'r3_finalization',change))
        for path in renderer.R3_HELPERS:
            with self.assertRaises(ValueError):
                render(mutate(fixture(),'r3_catalog',lambda v,p=path:v['runs'][164]['source_sha256'].pop(p)))

    def test_full_delta_composition_and_collision_checked(self):
        key=next(iter(R2_SPEC['source_sha256']))
        changes=[('r3_spec',lambda v:v['source_sha256'].update({key:'f'*64})),
            ('r3_spec',lambda v:v['source_sha256'].update({'/home/ubuntu/unknown-root.py':'e'*64})),
            ('r3_overlay',lambda v:v['source_extension'].update({key:'f'*64})),
            ('r3_catalog',lambda v:v['runs'][164]['source_sha256'].update({str(Q/'surprise.py'):'e'*64})),
            ('r3_finalization',lambda v:v['source_composition'].update(new_row_sources=1)),
            ('r3_spec',lambda v:v['source_sha256'].update({next(iter(renderer.R3_HELPERS)):'e'*64}))]
        for field,change in changes:
            with self.subTest(field=field),self.assertRaises(ValueError):render(mutate(fixture(),field,change))

    def test_status_counts_original_lock_and_drains_checked(self):
        changes=[('r3_ended',lambda v:v.update(completed_calls=40001)),
            ('r3_ended',lambda v:v.update(completed_calls=True)),
            ('r3_ended',lambda v:v.update(owners_gone=False)),
            ('r3_ended',lambda v:v['lock_identity'].update(inode=1)),
            ('r3_ended',lambda v:v['actual_tool'].update(exit_code=0)),
            ('r3_ended',lambda v:v['auditor_source'].update(sha256='f'*64)),
            ('r3_finalization',lambda v:v.update(two_current_drain_observations=[{}])),
            ('r3_finalization',lambda v:v['actual_independent_auditor_tool'].update(exit_code=1)),
            ('r3_finalization',lambda v:v['actual_independent_auditor_tool'].update(session_id=123456)),
            ('r3_finalization',lambda v:v['actual_independent_auditor_tool'].update(terminal_chunk='pending'))]
        for field,change in changes:
            with self.subTest(field=field),self.assertRaises(ValueError):render(mutate(fixture(),field,change))

    def test_success_requires_separate_exact_qualification(self):
        inputs=fixture(True); inputs['r3_qualification']=None
        with self.assertRaises(ValueError):render(inputs)
        for change in (lambda v:v.update(passed=False),lambda v:v.update(blockers=['bad']),
            lambda v:v.update(reports={'invented':1}),lambda v:v.update(source_sha256={'invented':'f'*64})):
            with self.assertRaises(ValueError):render(mutate(fixture(True),'r3_qualification',change))
        inputs=fixture(False);inputs['r3_qualification']=fixture(True)['r3_qualification']
        with self.assertRaises(ValueError):render(inputs)

    def test_digest_original_metadata_cap_and_unexpected_envelopes_rejected(self):
        for key in renderer.R3_KEYS-{'r3_qualification'}:
            inputs=fixture();inputs[key]['payload']+=b'\n'
            with self.subTest(key=key),self.assertRaises(ValueError):render(inputs)
        inputs=fixture();inputs['unrecognized']=None
        with self.assertRaises(ValueError):render(inputs)
        raw=b' '*((8*2**20)+1)
        inputs=fixture();inputs['r3_ended']['payload']=raw
        inputs['r3_ended']['reference']['sha256']=hashlib.sha256(raw).hexdigest()
        with self.assertRaisesRegex(ValueError,'original 8 MiB'):render(inputs)

    def test_generated_auth_rechecks_selected_status_and_complete_metadata_refs(self):
        inputs=fixture(True); module=domain(render(inputs));saved=reader(inputs);ref=inputs['r3_catalog']['reference']
        for field in ('r3_ended','r3_finalization','r3_qualification'):
            def read(r,field=field):
                value=saved(r)
                if r==inputs[field]['reference']:
                    value[{'r3_ended':'owners_gone','r3_finalization':'all_prior164_rows_unchanged','r3_qualification':'passed'}[field]]=False
                return value
            with self.subTest(field=field),self.assertRaises(ValueError):module.authenticate_completed_writer(ref,read,Mock())
        with self.assertRaisesRegex(RuntimeError,'deadline'):
            module.authenticate_completed_writer(ref,saved,Mock(side_effect=RuntimeError('deadline')))

    def test_future_domain_denied_and_whole165_prefix_preserved(self):
        inputs=fixture();module=domain(render(inputs));saved=reader(inputs)
        later=json.loads(inputs['r3_catalog']['payload']);later['runs'].append({'synthetic_future':True})
        ref=dict(path=str(Q/'synthetic-later-test-only.json'),sha256='f'*64)
        read=lambda r:copy.deepcopy(later) if r==ref else saved(r)
        extension=dict(r2.EXTERNAL_TEXT,**renderer.R3_HELPERS,**renderer.R3_RUNTIME_PINS)
        self.assertEqual(module.extension_shape(extension,completed_lfm_text={},catalog_ref=ref,read=read),extension)
        extension['/home/ubuntu/future-root-helper.py']='f'*64
        with self.assertRaisesRegex(ValueError,'no new external permission'):
            module.extension_shape(extension,completed_lfm_text={},catalog_ref=ref,read=read)
        later['runs'][164]['source_sha256'].pop(next(iter(renderer.R3_HELPERS)))
        with self.assertRaisesRegex(ValueError,'exact165'):module.authenticate_completed_writer(ref,read,Mock())

    def test_large_saved_history_not_duplicated_into_generated_source(self):
        inputs=mutate(fixture(),'r3_ended',lambda v:v.update(full_history_pair={'synthetic_large_history':'x'*(5*2**20)}))
        result=render(inputs); raw=result['files'][str(Q/'minicpm5_completed_qwen_package_text_v4.py')]
        self.assertGreater(len(inputs['r3_ended']['payload']),5*2**20)
        self.assertLess(len(raw),256*2**10)
        self.assertNotIn(b'synthetic_large_history',raw)
        module=domain(result)
        self.assertEqual(module.R3_ACTUAL_EXIT,inputs['r3_ended']['reference'])
        self.assertEqual(module.authenticate_completed_writer(inputs['r3_catalog']['reference'],reader(inputs),Mock()),
            dict(r2.EXTERNAL_TEXT,**renderer.R3_HELPERS))

    def test_schema_catalog_links_and_original_recovered_pairs_checked(self):
        cases=[('r3_overlay',lambda v:v.update(schema='foreign')),
            ('r3_overlay',lambda v:v.update(current_serial_catalog=r2.BASE_CATALOG)),
            ('r3_spec',lambda v:v.update(current_serial_catalog=r2.BASE_CATALOG)),
            ('r3_spec',lambda v:v.update(schema='foreign'))]
        for key,change in cases:
            with self.subTest(key=key),self.assertRaises(ValueError):render(mutate(fixture(),key,change))
        recovered=next(p for p in R2_SPEC['source_sha256'] if p not in R2_OVERLAY['source_extension'])
        with self.assertRaisesRegex(ValueError,'Every exact original recovered'):
            render(mutate(fixture(),'r3_spec',lambda v:v['source_sha256'].update({recovered:'e'*64})))

    def test_generated_exact42_subset_hash_and_neighbor_rejected(self):
        inputs=fixture();module=domain(render(inputs));ref=inputs['r3_catalog']['reference'];read=reader(inputs)
        for mode in ('subset','hash','neighbor'):
            extension=dict(r2.EXTERNAL_TEXT,**renderer.R3_HELPERS,**renderer.R3_RUNTIME_PINS)
            path=next(iter(renderer.R3_HELPERS))
            if mode=='subset':del extension[path]
            elif mode=='hash':extension[path]='e'*64
            else:extension[path+'.extra']=extension[path]
            with self.subTest(mode=mode),self.assertRaises(ValueError):
                module.extension_shape(extension,completed_lfm_text={},catalog_ref=ref,read=read)


if __name__=='__main__':unittest.main(verbosity=2)
