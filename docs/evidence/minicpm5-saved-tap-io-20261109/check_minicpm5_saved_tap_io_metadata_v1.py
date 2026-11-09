"""Real pinned metadata IO compatibility; explicitly refuses all raw reads."""
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import analyze_minicpm5_saved_tap_metadata_v1 as metadata
import minicpm5_saved_tap_inputs_v1 as inputs
import minicpm5_saved_tap_io_v1 as io
import minicpm5_saved_tap_report_v1 as report
from minicpm5_saved_tap_analysis_v1 import require

MANIFEST_REF = dict(path=str(metadata.Q/'minicpm5-saved-tap-input-manifest-20261109-v1.json'),
    sha256='b143a3f303280cbec08704e181bafea2bda2121075309c95429b1e279fd0e7e6')


class BeforeRawRead(Exception):
    pass


def main():
    resource.setrlimit(resource.RLIMIT_AS, (512*2**20, 512*2**20))
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    started = time.monotonic()
    preparation = metadata.Reader()
    diagnosis = preparation(inputs.DIAGNOSIS_REF)
    manifest = preparation(MANIFEST_REF)
    meta, raw = inputs.validate_manifest(diagnosis, manifest, os.getuid())
    calls = []

    def guard():
        # Observer gate and outer timeout cover this metadata-only check;
        # there is no serial/model/raw-analysis admission here.
        require(time.monotonic()-started < 60, 'Metadata-only wall budget')

    reader = io.PinnedReader(meta, guard, uid=os.getuid(), file_cap=8*2**20, total_cap=27711882)

    def forbidden_raw(ref, size):
        first = diagnosis['bundle_inventory'][0]['bundle']
        require(ref == dict(path=first['path'], sha256=first['sha256']) and size == first['raw_bytes'],
                'Exact first original planned raw read')
        calls.append(ref)
        raise BeforeRawRead()

    try:
        report.analyze(lambda ref:inputs.strict_json(reader(ref)), forbidden_raw, guard)
    except BeforeRawRead:
        require(len(calls) == 1, 'One stop before any raw IO')
    else:
        raise RuntimeError('Unexpected numeric report during metadata-only check')
    counts = reader.finish()
    require(counts['files'] == 95 and counts['bytes_read'] == 27711882,
            'Exact complete actual metadata read accounting')
    print(json.dumps(dict(schema='minicpm5_saved_tap_pinned_metadata_io_check_v1',
        manifest=MANIFEST_REF, original_diagnosis=inputs.DIAGNOSIS_REF,
        metadata_io=counts, original_metadata_compatible=True,
        first_raw_callback_refused=calls[0], raw_tensor_bytes_read=0,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__).resolve(),Path(inputs.__file__).resolve(),Path(io.__file__).resolve())},
        runtime_admitted=False, actual_tensor_analysis_executed=False,
        numeric_report_computed=False, full_model_qualified=False, new_speedup_claim=False)))


if __name__ == '__main__':
    main()
