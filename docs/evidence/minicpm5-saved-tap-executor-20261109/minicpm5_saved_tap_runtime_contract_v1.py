"""Exact future saved-tap job; no runtime admission or IO on import."""
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys

import minicpm5_after_nemotron_package_v1 as history
import minicpm5_saved_tap_inputs_v1 as inputs
from minicpm5_saved_tap_analysis_v1 import ARMS, TAPS, WIDTH, require
from dory_saved_review_retry_contract_v1 import QUERY_OBSERVATION

Q, ROOT = history.Q, history.ROOT
OUT = ROOT/'minicpm5-saved-tap-analysis-20261109-r1'
EXECUTOR = Q/'run_minicpm5_saved_tap_serial_v1.py'
MANIFEST = dict(path=str(Q/'minicpm5-saved-tap-input-manifest-20261109-v1.json'),
    sha256='b143a3f303280cbec08704e181bafea2bda2121075309c95429b1e279fd0e7e6')
IO_REVIEW = dict(path=str(Q/'minicpm5-saved-tap-io-source-preparation-review-20261109-v1.json'),
    sha256='f5d6f3f16a56f7e4471520eec2cd6ad7ab2cc37399b4dac6c50ed8832410cb10')
SCHEMA = 'minicpm5_saved_tap_runtime_v1'
BUDGET = dict(cpus=[4], nice=19, wall_seconds=1800, cpu_seconds=1500,
    address_space_bytes=2*2**30, single_file_bytes=32*2**20, output_bytes=64*2**20,
    root_floor=28*2**30, nvme_floor=8*2**30, nvme_new_bytes=512*2**20,
    initial_memory_available_bytes=4*2**30, memory_available_floor_bytes=2**30)
OUTPUT_NAMES = {'predecessor-history.json','predecessor-drain.json','source-files.json','admission.json',
                'numeric-report.json','service-query-authentication.json','result.json','failure.json',
                'query-cleanup-failure.json'}


def reference(value):
    require(type(value) is dict and set(value) == {'path','sha256'}
            and type(value['path']) is str and Path(value['path']).is_absolute()
            and Path(value['path']).suffix == '.json' and type(value['sha256']) is str
            and re.fullmatch('[0-9a-f]{64}',value['sha256']), 'Exact JSON metadata reference')


def validate_spec(spec):
    require(set(spec) == {'schema','output','executor','budget','profile','nemotron_actual_exit','nemotron_failed_release_exit',
        'diagnosis','input_manifest','io_preparation_review','query_dependency_observation','source_sha256',
        'essential_sources','source_read_cap_bytes','diagnostic_only','runtime_admitted',
        'automatic_launch','full_model_qualified','new_speedup_claim'}, 'Exact saved-tap job fields')
    require(spec['schema'] == SCHEMA and spec['output'] == str(OUT) and spec['executor'] == str(EXECUTOR)
            and spec['budget'] == BUDGET and spec['profile'] in history.PROFILES, 'Exact scope and resources')
    reference(spec['nemotron_actual_exit'])
    if spec['nemotron_failed_release_exit'] is not None:reference(spec['nemotron_failed_release_exit'])
    require(spec['diagnosis'] == inputs.DIAGNOSIS_REF and spec['input_manifest'] == MANIFEST
            and spec['io_preparation_review'] == IO_REVIEW
            and spec['query_dependency_observation'] == QUERY_OBSERVATION, 'Original frozen analysis inputs')
    require(spec['diagnostic_only'] is True and all(spec[k] is False for k in
        ('runtime_admitted','automatic_launch','full_model_qualified','new_speedup_claim')),
        'A specification is neither admission nor scientific qualification')
    pins = spec['source_sha256']
    require(type(pins) is dict and pins and spec['essential_sources'] == sorted(pins)
            and type(spec['source_read_cap_bytes']) is int and 0 < spec['source_read_cap_bytes'] <= 2**31,
            'Every bounded execution source is required')
    history.merge({},pins)
    for ref in (spec['nemotron_actual_exit'], inputs.DIAGNOSIS_REF, MANIFEST, IO_REVIEW, QUERY_OBSERVATION):
        require(pins.get(ref['path']) == ref['sha256'], 'Every exact input is source-bound')
    ref=spec['nemotron_failed_release_exit']
    if ref is not None:require(pins.get(ref['path'])==ref['sha256'],'Actual failed release is source-bound')
    for name in ('minicpm5_after_nemotron_package_v1.py','minicpm5_saved_tap_runtime_contract_v1.py',
        'run_minicpm5_saved_tap_serial_v1.py','minicpm5_saved_tap_inputs_v1.py','minicpm5_saved_tap_io_v1.py',
        'minicpm5_saved_tap_report_v1.py','minicpm5_saved_tap_analysis_v1.py',
        'dory_saved_review_guard_v1.py','dory_service_query_auth_v1.py'):
        require(str(Q/name) in pins, 'Required original execution source: '+name)
    return spec


def validate_review(spec, spec_ref, review):
    require(review['schema'] == 'minicpm5_saved_tap_runtime_source_review_v1'
            and review['reviewer'] == '/root' and review['passed'] is True
            and review['blockers'] == [] and review['spec_sha256'] == spec_ref['sha256']
            and review['source_sha256'] == spec['source_sha256'], 'Final concrete source review')


def input_metadata(spec, read, uid):
    diagnosis, manifest = read(spec['diagnosis']), read(spec['input_manifest'])
    io_review = read(spec['io_preparation_review'])
    require(io_review['source_preparation_passed'] is True and io_review['input_manifest'] == MANIFEST
            and all(spec['source_sha256'].get(p) == digest for p,digest in io_review['source_sha256'].items()),
            'Every original IO/report source and rejection retained')
    inputs.validate_manifest(diagnosis,manifest,uid)
    return diagnosis,manifest


def validate_report(value):
    require(value['schema'] == 'minicpm5_saved_tap_numeric_report_v1' and value['complete'] is True
            and value['pairwise_vector_comparisons'] == 384 and value['raw_bytes_validated'] == 3145728
            and value['frames_per_pair'] == 32 and value['taps'] == list(TAPS) and value['hidden_width'] == WIDTH,
            'All original vector comparisons and raw bytes')
    require(value['replicated_ranks_byte_equal'] is value['ranks_counted_once'] is True
            and len(value['metadata_inputs']) == 95 and len(value['bundle_inputs']) == 84,
            'Complete metadata and both ranks, counted once')
    for key in ('original_FP32_metrics_recomputed','scientific_thresholds_changed','absolute_HF_tap_error_measured',
                'runtime_admission_proven_by_this_module','full_model_qualified','new_speedup_claim'):
        require(value[key] is False, 'No numerical or qualification claim promotion: '+key)
    pairs=[(ARMS[0],ARMS[1]),(ARMS[0],ARMS[2]),(ARMS[1],ARMS[2])]
    require([(x['left_arm'],x['right_arm']) for x in value['comparisons']] == pairs,
            'All three comparison roles preserved')
    for comparison in value['comparisons']:
        require(len(comparison['frames']) == 32
                and sorted(x['frame_index'] for x in comparison['frames']) == list(range(32)),
                'No dropped original frames')
    for kind,count,size in (('metadata',95,27711882),('raw',84,3145728)):
        row=value['input_io'][kind]
        require(row == dict(files=count,bytes_read=size,all_pinned_inputs_read_once=True,
                            all_input_hashes_verified=True), 'Complete exact input IO accounting')
    require(value['input_io']['exact_manifest_validated'] is True
            and value['input_io']['runtime_admission_proven_by_this_module'] is False,
            'IO completion is distinct from lifecycle admission')
    return value


def loaded_local_sources(pins):
    found = {}
    for name,module in list(sys.modules.items()):
        origin = getattr(module,'__file__',None)
        if not origin: continue
        path=Path(origin)
        if path.is_absolute() and path.is_relative_to(ROOT):
            require(path.suffix == '.py' and str(path) in pins, 'Loaded local module outside source closure: '+name)
            found[name] = dict(path=str(path),sha256=pins[str(path)])
    require(found and not {'torch','numpy','transformers','vllm'} & set(sys.modules),
            'Only metadata and stdlib numerical analysis, no model runtime')
    return found


class OutputWriter:
    """Fresh fixed directory, finite JSON outputs and durable non-overwrite writes."""
    def __init__(self,path,*,file_cap,total_cap):
        self.path=Path(path)
        require(self.path.resolve() == self.path and not any(p.is_symlink() for p in (self.path,*self.path.parents)),
                'Ordinary output namespace')
        self.path.mkdir(mode=0o700)
        self.fd=os.open(self.path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC)
        self.identity=(os.fstat(self.fd).st_dev,os.fstat(self.fd).st_ino)
        self.file_cap,self.total_cap,self.bytes,self.closed=file_cap,total_cap,0,False

    def save(self,name,value):
        require(not self.closed and name in OUTPUT_NAMES,'Exact open output writer/name')
        now=self.path.stat(follow_symlinks=False)
        require(stat.S_ISDIR(now.st_mode) and (now.st_dev,now.st_ino)==self.identity,'Original output directory retained')
        raw=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
        require(len(raw)<=self.file_cap and self.bytes+len(raw)<=self.total_cap,'Bounded finite result metadata')
        fd=os.open(name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW|os.O_CLOEXEC,0o600,dir_fd=self.fd)
        # Reserve the full attempted output even if a write/fsync fails.
        self.bytes+=len(raw)
        with os.fdopen(fd,'wb')as stream:
            stream.write(raw);stream.flush();os.fsync(stream.fileno());written=os.fstat(stream.fileno())
        os.fsync(self.fd)
        current=self.path.stat(follow_symlinks=False)
        require(stat.S_ISDIR(current.st_mode) and (current.st_dev,current.st_ino)==self.identity,
                'Output directory unchanged after durable write')
        actual=os.stat(name,dir_fd=self.fd,follow_symlinks=False)
        require(stat.S_ISREG(actual.st_mode) and actual.st_nlink==1 and actual.st_size==len(raw)
                and all(getattr(actual,k)==getattr(written,k) for k in
                    ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')), 'Exact saved output retained')
        return dict(path=str(self.path/name),sha256=hashlib.sha256(raw).hexdigest())

    def close(self):
        if not self.closed:os.close(self.fd);self.closed=True


if __name__=='__main__':raise SystemExit('Source preparation only; requires actual predecessor completion and final review')
