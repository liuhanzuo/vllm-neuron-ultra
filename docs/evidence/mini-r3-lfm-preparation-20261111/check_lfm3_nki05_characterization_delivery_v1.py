"""Complete synthetic serialization envelope and source-only runner controls."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import lfm3_nki05_transport_diagnostic_v1 as original


def require(ok, message):
    if not ok: raise RuntimeError(message)


def definitions(path, names, scope):
    nodes = [n for n in ast.parse(path.read_bytes()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    require(len(nodes) == len(names), 'Exact source definitions')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)


def writer_scope():
    scope = dict(Path=Path, os=os, json=json, FILE_CAP=256*2**10, p=SimpleNamespace(require=require))
    definitions(ROOT/'run_lfm3_nki05_transport_serial_v1.py', {'encode', 'once'}, scope)
    return scope


def difference(dtype, changed=False):
    bits = 16 if dtype == 'bfloat16' else 32
    # Conservative serialized lengths: full 256 KiB operand, every word/bit
    # changed, all eight examples, 64-byte hash strings. This envelope bounds
    # representation size; it is explicitly not scientific/runtime evidence.
    words = 262144//(bits//8)
    return dict(bytes=262144, before_sha256='a'*64, after_sha256=('b' if changed else 'a')*64,
                words=words, changed_words=words if changed else 0,
                changed_bits=words*bits if changed else 0,
                bit_position_counts=[words if changed else 0]*bits,
                both_nan=words if changed and dtype != 'int32' else 0, nan_to_non_nan=0,
                non_nan_to_nan=0, other_changed_words=words if changed and dtype == 'int32' else 0,
                sign_bit_changes=words if changed and dtype != 'int32' else 0, signed_zero_changes=0,
                first8=[dict(word=words-1-i, before='ffff' if bits==16 else 'ffffffff',
                             after='0000' if bits==16 else '00000000', xor=2**bits-1)
                        for i in range(8)] if changed else [])


def descriptors():
    rows = []
    for label, width, count in (('exact_v5_first_case', 256, 1), ('all65536_bf16_encodings', 8192, 8)):
        rows.extend((label, op, 'bfloat16', 8, width, count) for op in ('identity', 'copy', 'permutation', 'selector'))
    for dtype in ('bfloat16', 'float32'):
        for capacity in (4, 8, 16):
            for name in ('inactive', 'duplicate_last_writer', 'last_winner_before_inactive'):
                rows.append((f'{dtype}_B{capacity}_{name}', 'selector', dtype, capacity, 256, 4))
            if capacity == 8:
                rows.extend((dtype+'_movement_controls', op, dtype, capacity, 256, 4)
                            for op in ('identity', 'copy', 'permutation'))
    assert len(rows) == 32 and sum(row[2]=='bfloat16' for row in rows) == 20
    return rows


def case(row, passed=True):
    label, op, dtype, capacity, width, count = row
    stages = {key:dict(observations=value, view_failures=0, backing_failures=0)
              for key, value in original.expected_observations(op, capacity, width).items()}
    byte_count = capacity*width*(2 if dtype=='bfloat16' else 4)
    inputs = list(original.OLD_HASHES[:3]) if label=='exact_v5_first_case' else ['a'*64]*3
    expected = original.OLD_HASHES[0] if label=='exact_v5_first_case' and op in ('identity','copy') else 'c'*64
    return dict(name=label, operation=op, passed=passed, shape=[capacity,width], dtype=dtype,
                inputs_unchanged=passed, observer_coverage=True, input_sha256=inputs,
                expected_sha256=expected, actual_sha256=expected, actual_dtype=dtype,
                actual_shape=[capacity,width], returned_difference=dict(actual_bytes=byte_count,
                    expected_bytes=byte_count, mismatched_words=0, nan_payload_mismatches=0,
                    other_mismatches=0, first8=[]), stages=stages, first_boundary_divergence=None,
                host_input_differences=[dict(operand=name, dtype=kind, shape=shape,
                    difference=difference(kind, changed=not passed and name!='slots'))
                    for name,kind,shape in (('cache',dtype,[capacity,width]),('updates',dtype,[count,width]),
                                            ('slots','int32',[count]))])


def extraction(row, candidate):
    label, op, dtype, capacity, width, count = row
    events = []
    for kind, shape in ((dtype,[capacity,width]),(dtype,[count,width]),('int32',[count])):
        changed = candidate and kind=='bfloat16'
        event = dict(case=[label,op], caller='simulate_kernel', writeback_call=True,
                     requested_precise='omitted', effective_precise=changed, substitution=changed,
                     tensor_dtype=kind, storage_dtype=kind,
                     extracted_dtype='float32' if kind=='bfloat16' and not candidate else kind, shape=shape)
        if kind=='bfloat16':
            event.update(storage_matches_host_before=True, precision_controls={
                'false':dict(extracted_dtype='float32', difference=difference(kind, changed=True)),
                'true':dict(extracted_dtype='bfloat16', difference=difference(kind))})
        events.append(event)
    events.append(dict(case=[label,op], caller='_validate_and_convert', writeback_call=False,
                       requested_precise=True, effective_precise=True, substitution=False,
                       tensor_dtype=dtype, storage_dtype=dtype, extracted_dtype=dtype, shape=[capacity,width]))
    return events


def suite(rows, passed):
    result = dict(schema='lfm3_nki05_transport_diagnostic_v1', complete=passed, passed=passed,
                  source_sha256={str(path):sha for path,sha in original.PINS.items()},
                  diagnostic_source_sha256='d'*64, legacy_candidate_changed=False,
                  legacy_full3716_passed=False, device_behavior_proven=False, full_model_qualified=False,
                  new_speedup_claim=False, target='trn1', precise_fp=True,
                  checks=[case(row,passed) for row in rows], negative_controls=original.negative_controls(),
                  versions=dict(nki='0.5.0+28631259367.ga768afa6', **{'neuronx-cc':'2.26.6360.0+6f180f47'},
                                numpy='2.5.4',ml_dtypes='0.6.0'),
                  exact_fixture_hashes_verified=True, historical_failed_actual_sha256=original.OLD_ACTUAL,
                  seconds=1799.999999999999, max_rss_kib=10485760)
    if passed:
        result.update(bf16_encoding_count=65536,bf16_encoding_source_sha256='e'*64,
                      arithmetic_control=True,round_to_nearest_even_cast_control=True)
    else:
        result['error']='E'*4096
    return result


def representative():
    rows = descriptors()
    result = dict(schema='lfm3_nki05_writeback_diagnostic_v2', complete=True, passed=False,
                  original_suite_passed=False, candidate_suite_passed=True,
                  original_failure_preserved=True, restored_original_reproduced=True,
                  writeback_precision_explanation_supported=True, legacy_candidate_changed=False,
                  legacy_full3716_passed=False, device_behavior_proven=False, full_model_qualified=False,
                  new_speedup_claim=False, installed_sdk_source_modified=False,
                  source_sha256={str(ROOT/name):'f'*64 for name in (
                      'lfm3_nki05_writeback_probe_v1.py','lfm3_nki05_transport_diagnostic_v2.py',
                      'lfm3_nki05_transport_diagnostic_v1.py')}, seconds=1799.999999999999,max_rss_kib=10485760)
    for name in ('original','candidate','restored_original'):
        candidate = name=='candidate'
        selected = rows if candidate else rows[:1]
        events = [event for row in selected for event in extraction(row,candidate)]
        if candidate:
            for output in ('float32','bfloat16'):
                events.extend([dict(case=None,caller='simulate_kernel',writeback_call=True,
                    requested_precise='omitted',effective_precise=False,substitution=False,
                    tensor_dtype='float32',storage_dtype='float32',extracted_dtype='float32',shape=[1,4]),
                    dict(case=None,caller='_validate_and_convert',writeback_call=False,requested_precise=True,
                         effective_precise=True,substitution=False,tensor_dtype=output,storage_dtype=output,
                         extracted_dtype=output,shape=[1,4])])
        result[name] = dict(suite=suite(selected,candidate),probe=dict(mode='precise_bf16_writeback' if candidate else 'original',
            installed=True,restored=True,substitutions=40 if candidate else 0,extraction_calls=events,
            case_errors=[],installed_sdk_source_modified=False))
    return result


class EnvelopeTests(unittest.TestCase):
    def test_complete_evidence_and_actual_writer_bytes(self):
        report = representative()
        self.assertEqual(len(report['candidate']['suite']['checks']),32)
        events = report['candidate']['probe']['extraction_calls']
        self.assertEqual(len(events),132)
        self.assertEqual(sum('precision_controls'in e for e in events),40)
        self.assertEqual(len(report['original']['suite']['checks']),1)
        self.assertEqual(len(report['restored_original']['suite']['checks']),1)
        data = writer_scope()['encode'](report)
        self.assertEqual(json.loads(data),report)
        self.assertLessEqual(len(data),256*2**10)
        self.assertLessEqual(len(json.dumps(report,allow_nan=False).encode()),256*2**10)

    def test_actual_writer_has_no_pretty_print_path(self):
        writer = writer_scope()
        value = dict(z=[1,2],a=True)
        self.assertEqual(writer['encode'](value),b'{"a":true,"z":[1,2]}\n')
        text=(ROOT/'run_lfm3_nki05_transport_serial_v2.py').read_bytes()
        new={n.name:n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
        old={n.name:n for n in ast.parse((ROOT/'run_lfm3_nki05_transport_serial_v1.py').read_bytes()).body
             if isinstance(n,ast.FunctionDef)}
        for name in ('encode','once'):
            self.assertEqual(ast.dump(old[name]),ast.dump(new[name]))


def runner_scope():
    scope=dict(p=SimpleNamespace(require=require),diagnostic=original,Path=Path)
    definitions(ROOT/'run_lfm3_nki05_transport_serial_v2.py',
                {'require_diagnostic_success','require_characterization','execute_diagnostic','execution_source_paths'},scope)
    return scope


class RunnerTests(unittest.TestCase):
    def test_all_original_lifecycle_functions_preserved(self):
        def nodes(version):
            tree=ast.parse((ROOT/f'run_lfm3_nki05_transport_serial_v{version}.py').read_bytes())
            return {n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        before,after=nodes(1),nodes(2)
        self.assertEqual(set(after)-set(before),{'require_characterization'})
        for name in before:
            if name not in ('execution_source_paths','execute_diagnostic'):
                self.assertEqual(ast.dump(before[name]),ast.dump(after[name]),name)

    def test_complete_characterization_still_has_original_failure(self):
        status=runner_scope()['require_characterization'](representative())
        self.assertTrue(status['characterization_complete'])
        self.assertFalse(status['original_passed'])
        self.assertTrue(status['candidate_passed'])
        self.assertTrue(status['nonzero_worker_exit_required'])
        self.assertFalse(status['actual_tool_exit_verified'])

    def test_incomplete_or_promoted_evidence_rejected(self):
        changes=(lambda r:r.__setitem__('passed',True),
                 lambda r:r.__setitem__('original_suite_passed',True),
                 lambda r:r['candidate']['suite']['checks'].pop(),
                 lambda r:r['candidate']['probe']['extraction_calls'].pop(),
                 lambda r:r['candidate']['probe'].__setitem__('restored',False),
                 lambda r:r['candidate']['probe'].__setitem__('substitutions',39),
                 lambda r:r['restored_original']['suite']['checks'][0].__setitem__('inputs_unchanged',True))
        for change in changes:
            report=representative();change(report)
            with self.subTest(change=change),self.assertRaises(RuntimeError):
                runner_scope()['require_characterization'](report)

    def test_complete_evidence_published_then_real_exception_retained(self):
        scope=runner_scope();writes={};admissions=[];root=Path('/synthetic/lfm-characterization')
        report=representative()
        scope.update(diagnostic=SimpleNamespace(diagnose=lambda admit:report,negative_controls=original.negative_controls,
                     OLD_HASHES=original.OLD_HASHES),
                     once=lambda path,value:writes.setdefault(path.name,copy.deepcopy(value)),
                     reference=lambda path:dict(path=str(path),sha256='d'*64),
                     report_failure=lambda path,error,**context:writes.setdefault(path.name,dict(error=str(error),**context)))
        owner={'pid':123,'start_ticks':'456'};admission={'path':'/synthetic/admission.json','sha256':'a'*64}
        with self.assertRaisesRegex(RuntimeError,'Characterization complete; original implementation failed'):
            scope['execute_diagnostic'](admission,owner,root,lambda:admissions.append('live'))
        self.assertEqual(set(writes),{'diagnostic-result.json','characterization-status.json','worker-failure.json'})
        self.assertEqual(writes['diagnostic-result.json'],report)
        self.assertEqual(admissions,['live','live'])
        self.assertTrue(writes['characterization-status.json']['characterization_complete'])
        self.assertFalse(writes['characterization-status.json']['original_passed'])
        self.assertTrue(writes['worker-failure.json']['characterization_complete'])
        self.assertEqual(writes['characterization-status.json']['owner'],owner)
        self.assertEqual(writes['characterization-status.json']['admission'],admission)

    def test_failed_characterization_retains_full_report_without_complete_status(self):
        scope=runner_scope();writes={};report=representative();report['complete']=False
        scope.update(diagnostic=SimpleNamespace(diagnose=lambda admit:report),
                     once=lambda path,value:writes.setdefault(path.name,value),
                     report_failure=lambda path,error,**context:writes.setdefault(path.name,context))
        with self.assertRaises(RuntimeError):
            scope['execute_diagnostic']({}, {},Path('/synthetic/output'),lambda:None)
        self.assertEqual(set(writes),{'diagnostic-result.json','worker-failure.json'})
        self.assertFalse(writes['worker-failure.json']['characterization_complete'])
        self.assertIsNone(writes['worker-failure.json']['original_passed'])

    def test_source_preparer_requires_complete_supplied_map(self):
        scope=runner_scope();seen=[];scope['verify_sources']=lambda spec:seen.append(spec)
        with self.assertRaises(RuntimeError):scope['execution_source_paths']({})
        with self.assertRaises(TypeError):scope['execution_source_paths']()
        pins={f'/synthetic/source-{i}.py':'a'*64 for i in range(561)}
        self.assertEqual(set(map(str,scope['execution_source_paths'](pins))),set(pins))
        self.assertEqual(seen,[dict(source_sha256=pins)])


if __name__=='__main__':
    if sys.argv[1:]==['--sizes']:
        report=representative()
        print(json.dumps(dict(actual_once_encode_bytes=len(writer_scope()['encode'](report)),
                              diagnostic_default_json_bytes=len(json.dumps(report,allow_nan=False).encode()),
                              comparison_indent2_bytes=len((json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()),
                              file_cap_bytes=256*2**10)))
    else:
        unittest.main()
