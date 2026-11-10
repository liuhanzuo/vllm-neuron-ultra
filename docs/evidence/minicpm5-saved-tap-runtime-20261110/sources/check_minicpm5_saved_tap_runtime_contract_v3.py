"""Generated numerical report and temporary output fixtures, no real raw IO."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import minicpm5_saved_tap_runtime_contract_v3 as c
import minicpm5_saved_tap_report_v1 as report
from check_minicpm5_saved_tap_report_v1_r2 import trajectory


def spec_fixture():
    refs=[c.inputs.DIAGNOSIS_REF,c.MANIFEST,c.IO_REVIEW,c.QUERY_OBSERVATION,
          dict(path=str(c.Q/'synthetic-nemotron-exit.json'),sha256='1'*64)]
    names=['minicpm5_after_nemotron_package_v2.py','minicpm5_saved_tap_runtime_contract_v3.py',
        'run_minicpm5_saved_tap_serial_v3.py','minicpm5_saved_tap_inputs_v1.py','minicpm5_saved_tap_io_v1.py',
        'minicpm5_saved_tap_report_v1.py','minicpm5_saved_tap_analysis_v1.py',
        'dory_saved_review_guard_v1.py','dory_service_query_auth_v1.py']
    pins={str(c.Q/n):'2'*64 for n in names};pins.update({r['path']:r['sha256']for r in refs})
    return dict(schema=c.SCHEMA,output=str(c.OUT),executor=str(c.EXECUTOR),budget=copy.deepcopy(c.BUDGET),
        profile='retry-v24',nemotron_actual_exit=refs[-1],nemotron_failed_release_exit=None,diagnosis=refs[0],input_manifest=refs[1],
        io_preparation_review=refs[2],query_dependency_observation=refs[3],source_sha256=pins,
        essential_sources=sorted(pins),source_read_cap_bytes=2**20,diagnostic_only=True,
        runtime_admitted=False,automatic_launch=False,full_model_qualified=False,new_speedup_claim=False)


def generated_report():
    docs,raw=trajectory()
    def read(ref):
        expected,value=docs[ref['path']]
        assert ref==expected
        return copy.deepcopy(value)
    value=report.analyze(read,lambda ref,size:raw[ref['path']],lambda:None)
    value['input_io']=dict(metadata=dict(files=95,bytes_read=27711882,all_pinned_inputs_read_once=True,
                                       all_input_hashes_verified=True),
        raw=dict(files=84,bytes_read=3145728,all_pinned_inputs_read_once=True,all_input_hashes_verified=True),
        exact_manifest_validated=True,runtime_admission_proven_by_this_module=False)
    return value


class Contract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.report=generated_report()

    def test_cumulative_streamed_reads_above_old_cap_and_at_finite_limit(self):
        for size in (6721948351+4*2**20, c.SOURCE_READ_LIMIT):
            s=spec_fixture();s['source_read_cap_bytes']=size
            c.validate_spec(s)
            self.assertEqual(s['budget']['address_space_bytes'],2*2**30)

    def test_invalid_or_unbounded_cumulative_reads_rejected(self):
        for size in (0,-1,True,1.5,c.SOURCE_READ_LIMIT+1):
            s=spec_fixture();s['source_read_cap_bytes']=size
            with self.subTest(size=size),self.assertRaises(ValueError):c.validate_spec(s)

    def test_exact_proposed_scope_is_diagnostic_not_admission(self):
        value=c.validate_spec(spec_fixture())
        self.assertFalse(value['runtime_admitted']);self.assertFalse(value['new_speedup_claim'])

    def test_each_original_input_reference_mandatory(self):
        for field in ('diagnosis','input_manifest','io_preparation_review','query_dependency_observation'):
            s=spec_fixture();s[field]=dict(s[field],sha256='f'*64)
            with self.subTest(field=field),self.assertRaises(ValueError):c.validate_spec(s)

    def test_resources_and_output_namespace_not_relaxed(self):
        for change in ('budget','output','executor'):
            s=spec_fixture()
            if change=='budget':s['budget']['root_floor']=1
            else:s[change]='/tmp/not-approved'
            with self.subTest(change=change),self.assertRaises(ValueError):c.validate_spec(s)

    def test_source_omission_or_rebinding_rejected(self):
        for mode in ('missing','rebind','not-essential'):
            s=spec_fixture()
            if mode=='missing':s['source_sha256'].pop(str(c.EXECUTOR))
            elif mode=='rebind':s['source_sha256'][c.MANIFEST['path']]='f'*64
            else:s['essential_sources'].pop()
            with self.subTest(mode=mode),self.assertRaises((ValueError,RuntimeError)):c.validate_spec(s)

    def test_no_automatic_launch_or_qualification(self):
        for field in ('automatic_launch','runtime_admitted','full_model_qualified','new_speedup_claim'):
            s=spec_fixture();s[field]=True
            with self.subTest(field=field),self.assertRaises(ValueError):c.validate_spec(s)

    def test_optional_failed_release_requires_its_exact_pin(self):
        s=spec_fixture();s['nemotron_failed_release_exit']=dict(path='/synthetic/release.json',sha256='f'*64)
        with self.assertRaises(ValueError):c.validate_spec(s)
        s['source_sha256']['/synthetic/release.json']='f'*64;s['essential_sources']=sorted(s['source_sha256'])
        c.validate_spec(s)

    def test_final_review_binds_concrete_spec(self):
        s=spec_fixture();ref=dict(path='/synthetic/spec.json',sha256='a'*64)
        review=dict(schema='minicpm5_saved_tap_runtime_source_review_v3',reviewer='/root',passed=True,
                    blockers=[],spec_sha256=ref['sha256'],source_sha256=s['source_sha256'])
        c.validate_review(s,ref,review)
        review['spec_sha256']='b'*64
        with self.assertRaises(ValueError):c.validate_review(s,ref,review)

    def test_generated_complete_report_passes_all_scope_counts(self):
        c.validate_report(self.report)
        self.assertEqual(len(self.report['comparisons']),3)
        self.assertFalse(self.report['absolute_HF_tap_error_measured'])

    def test_missing_frame_pair_bundle_or_metadata_rejected(self):
        for change in ('frame','pair','bundle','metadata'):
            value=copy.deepcopy(self.report)
            if change=='frame':value['comparisons'][0]['frames'].pop()
            elif change=='pair':value['comparisons'].pop()
            elif change=='bundle':value['bundle_inputs'].pop()
            else:value['metadata_inputs'].pop()
            with self.subTest(change=change),self.assertRaises(ValueError):c.validate_report(value)

    def test_replicas_original_gates_and_no_HF_error_promotion(self):
        for flag in ('replicated_ranks_byte_equal','ranks_counted_once','original_FP32_metrics_recomputed',
                     'scientific_thresholds_changed','absolute_HF_tap_error_measured','full_model_qualified'):
            value=copy.deepcopy(self.report);value[flag]=not value[flag]
            with self.subTest(flag=flag),self.assertRaises(ValueError):c.validate_report(value)

    def test_exact_input_io_bytes_and_hash_accounting(self):
        for kind in ('metadata','raw'):
            for field in ('bytes_read','files','all_input_hashes_verified'):
                value=copy.deepcopy(self.report);value['input_io'][kind][field]=0
                with self.subTest(kind=kind,field=field),self.assertRaises(ValueError):c.validate_report(value)


class Outputs(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='mini-output-fixture-');self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'out'

    def writer(self,cap=1024):
        value=c.OutputWriter(self.path,file_cap=cap,total_cap=2*cap);self.addCleanup(value.close);return value

    def test_fresh_durable_output_digest_and_no_overwrite(self):
        w=self.writer();ref=w.save('result.json',dict(complete=True))
        self.assertEqual(hashlib.sha256(Path(ref['path']).read_bytes()).hexdigest(),ref['sha256'])
        with self.assertRaises(FileExistsError):w.save('result.json',{})
        with self.assertRaises(FileExistsError):c.OutputWriter(self.path,file_cap=1,total_cap=2)

    def test_foreign_names_nonfinite_and_size_rejected(self):
        w=self.writer(64)
        for name,value in (('../elsewhere.json',{}),('result.json',{'n':float('nan')}),('result.json',{'x':'a'*100})):
            with self.subTest(name=name,value=str(value)[:20]),self.assertRaises(ValueError):w.save(name,value)
        self.assertEqual(list(self.path.iterdir()),[])

    def test_output_directory_replacement_rejected(self):
        w=self.writer();self.path.rename(self.path.with_name('old'));self.path.mkdir()
        with self.assertRaisesRegex(ValueError,'retained'):w.save('result.json',{})

    def test_output_directory_symlink_rejected(self):
        target=self.path.with_name('target');target.mkdir();self.path.symlink_to(target,target_is_directory=True)
        with self.assertRaises(ValueError):self.writer()

    def test_closed_writer_cannot_reopen(self):
        w=self.writer();w.close()
        with self.assertRaises(ValueError):w.save('result.json',{})

    def test_failed_fsync_still_reserves_output_budget(self):
        w=self.writer()
        with patch.object(c.os,'fsync',side_effect=OSError('synthetic sync failure')):
            with self.assertRaises(OSError):w.save('result.json',{'fixture':'retained'})
        self.assertEqual(w.bytes,(self.path/'result.json').stat().st_size)
        with self.assertRaises(FileExistsError):w.save('result.json',{})


if __name__=='__main__':unittest.main(verbosity=2)
