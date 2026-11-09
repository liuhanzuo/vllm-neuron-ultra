"""Exact original input binding and IO integration for the MiniCPM tap report.

No executable entrypoint or serial admission. The eventual executor must
authenticate the diagnosis/manifest/source pins and supply a live guard.
"""
import json
import math
from pathlib import Path

import analyze_minicpm5_saved_tap_metadata_v1 as metadata
import minicpm5_saved_tap_report_v1 as report
from minicpm5_saved_tap_analysis_v1 import ARMS, require
from minicpm5_saved_tap_io_v1 import PinnedReader, validate_entry

DIAGNOSIS_REF = dict(path=str(metadata.Q/'minicpm5-saved-tap-metadata-diagnosis-20261109-v1.json'),
    sha256='d14e24cbc371039b709ea9acfe5bda7e6b71e1208c050f8e87d9bc26b384e4be')


def validate_manifest(diagnosis, manifest, uid):
    require(diagnosis['schema'] == 'minicpm5_saved_tap_metadata_diagnosis_v1'
            and diagnosis['complete'] is True and diagnosis['original_actual_exit'] == metadata.EXIT
            and diagnosis['rank_count'] == 2 and diagnosis['arms'] == list(ARMS)
            and diagnosis['calls_per_arm'] == 14 and diagnosis['frames_per_arm'] == 32,
            'Original complete two-rank three-arm diagnosis')
    require(set(manifest) == {'schema', 'reviewer', 'diagnosis', 'uid', 'metadata', 'bundles',
                             'raw_payloads_opened_during_snapshot', 'runtime_admitted',
                             'full_model_qualified', 'new_speedup_claim'}
            and manifest['schema'] == 'minicpm5_saved_tap_input_manifest_v1'
            and manifest['reviewer'] == '/root' and manifest['diagnosis'] == DIAGNOSIS_REF
            and type(manifest['uid']) is int and manifest['uid'] == uid,
            'Exact root-reviewed source-preparation manifest')
    require(all(manifest[k] is False for k in ('raw_payloads_opened_during_snapshot',
                'runtime_admitted', 'full_model_qualified', 'new_speedup_claim')),
            'Preparation does not claim runtime or scientific qualification')
    meta, raw = manifest['metadata'], manifest['bundles']
    require(type(meta) is list and len(meta) == 95 and type(raw) is list and len(raw) == 84,
            'All95 JSON and84 raw descriptors')
    for entry in meta:
        validate_entry(entry, uid, 8*2**20)
        p = Path(entry['path'])
        require(p.suffix == '.json' and (p.is_relative_to(metadata.Q) or p.is_relative_to(metadata.BASE)),
                'Original approved metadata roots')
    require(len({x['path'] for x in meta}) == 95
            and {x['path']: x['sha256'] for x in meta} == diagnosis['metadata_source_sha256']
            and sum(x['bytes'] for x in meta) == diagnosis['metadata_bytes_read'] == 27711882,
            'All original exact metadata pins and bytes')
    expected = {}
    seen_keys = set()
    for item in diagnosis['bundle_inventory']:
        rank, arm, ci = item['rank'], item['arm'], item['call']
        require(type(rank) is int and rank in (0, 1) and arm in ARMS
                and type(ci) is int and 0 <= ci < 14 and (rank, arm, ci) not in seen_keys,
                'All original unique rank-arm-call keys')
        seen_keys.add((rank, arm, ci))
        path = str(metadata.BASE/f'cpu-rank{rank}'/arm/f'call-{ci:02d}'/'selected-taps.bf16')
        bundle = item['bundle']
        require(bundle['path'] == path and bundle['encoding'] == 'raw_bf16_le'
                and bundle['raw_bytes'] in (16384, 65536), 'Original exact raw namespace and size')
        expected[path] = (bundle['sha256'], bundle['raw_bytes'])
    require(len(expected) == len(seen_keys) == 84, 'Original complete84 bundle inventory')
    for entry in raw:
        validate_entry(entry, uid, 65536)
    require(len({x['path'] for x in raw}) == 84
            and {x['path']: (x['sha256'], x['bytes']) for x in raw} == expected
            and sum(x['bytes'] for x in raw) == 3145728,
            'All84 exact bundle paths, hashes and bytes')
    require(len({(x['identity']['st_dev'], x['identity']['st_ino']) for x in (*meta, *raw)}) == 179,
            'All179 input files have distinct pinned physical identities')
    return meta, raw


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON keys rejected')
            result[key] = value
        return result

    def constant(value):
        raise ValueError('Nonfinite JSON constant rejected: '+value)

    def number(value):
        parsed = float(value)
        require(math.isfinite(parsed), 'Overflowing JSON numeric literal rejected')
        return parsed

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant, parse_float=number)


def analyze(diagnosis, manifest, guard, *, uid):
    """Actual bytes may be read only by a separately admitted serial caller."""
    guard()
    meta, raw = validate_manifest(diagnosis, manifest, uid)
    meta_reader = PinnedReader(meta, guard, uid=uid, file_cap=8*2**20, total_cap=27711882)
    raw_reader = PinnedReader(raw, guard, uid=uid, file_cap=65536, total_cap=3145728)

    def read_json(ref):
        return strict_json(meta_reader(ref))

    result = report.analyze(read_json, raw_reader, guard)
    result['input_io'] = dict(metadata=meta_reader.finish(), raw=raw_reader.finish(),
        exact_manifest_validated=True, runtime_admission_proven_by_this_module=False)
    guard()
    return result


if __name__ == '__main__':
    raise SystemExit('Source preparation only; requires a separately reviewed serial executor')
