"""Authenticate real JSON compatibility; stop before the first raw-byte read.

Run under the established observer wrapper. This is not a tensor analysis or
serial runtime admission. The supplied raw callback only raises a sentinel.
"""
import hashlib
import json
from pathlib import Path

import analyze_minicpm5_saved_tap_metadata_v1 as metadata
import minicpm5_saved_tap_report_v1 as report
from minicpm5_saved_tap_analysis_v1 import require

DIAGNOSIS = dict(path=str(metadata.Q/'minicpm5-saved-tap-metadata-diagnosis-20261109-v1.json'),
    sha256='d14e24cbc371039b709ea9acfe5bda7e6b71e1208c050f8e87d9bc26b384e4be')


class BeforeRawRead(Exception):
    pass


def main():
    read = metadata.Reader()
    original = metadata.Reader()(DIAGNOSIS)
    stopped = []

    def guard():
        # The outer observer wrapper covers this whole metadata-only process.
        # This is intentionally not a raw-analysis ownership/resource guard.
        pass

    def forbidden_raw(ref, size):
        require(len(read.pins) == 95 and read.bytes == 27711882,
                'All original95 JSON records checked before any raw callback')
        require(read.pins == original['metadata_source_sha256'], 'Same original metadata fingerprints')
        first = original['bundle_inventory'][0]
        require(first['rank'] == 0 and first['arm'] == report.ARMS[0] and first['call'] == 0,
                'Original first bundle identity')
        bundle = first['bundle']
        require(ref == dict(path=bundle['path'], sha256=bundle['sha256']) and size == bundle['raw_bytes'],
                'Exact original first planned raw read')
        stopped.append(dict(reference=ref, declared_bytes=size))
        raise BeforeRawRead()

    try:
        report.analyze(read, forbidden_raw, guard)
    except BeforeRawRead:
        require(len(stopped) == 1, 'Exactly one stop before raw IO')
    else:
        raise RuntimeError('Metadata-only check unexpectedly returned a numeric report')
    print(json.dumps(dict(schema='minicpm5_saved_tap_report_metadata_compatibility_v1',
        metadata_compatible=True, authenticated_original_metadata_files=len(read.pins),
        original_metadata_bytes=read.bytes, original_metadata_diagnosis=DIAGNOSIS,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__).resolve(), Path(report.__file__).resolve())},
        stopped_before_first_raw_read=stopped[0], actual_tensor_bytes_read=0,
        runtime_admitted=False, numeric_report_computed=False,
        full_model_qualified=False, new_speedup_claim=False)))


if __name__ == '__main__':
    main()
