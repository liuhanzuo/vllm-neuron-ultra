"""Observer-only preparation: authenticate JSON and stat all original inputs.

Never opens raw tensor payloads, acquires serial ownership, or admits a model
job. Run under the shared observer wrapper. The resulting manifest still needs
source/runtime review and fresh guards before a later serial worker uses it.
"""
import hashlib
import json
import os
from pathlib import Path
import resource
import sys

import analyze_minicpm5_saved_tap_metadata_v1 as metadata
import minicpm5_saved_tap_inputs_v1 as inputs
import minicpm5_saved_tap_io_v1 as io
from minicpm5_saved_tap_analysis_v1 import require

OUT = metadata.Q/'minicpm5-saved-tap-input-manifest-20261109-v1.json'
REVIEW = metadata.Q/'minicpm5-saved-tap-input-snapshot-20261109-v1.json'


def once(path, value):
    raw = (json.dumps(value, indent=2, allow_nan=False)+'\n').encode()
    require(len(raw) <= 2**20, 'Bounded source-preparation output')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())


def main():
    require(sys.dont_write_bytecode and __debug__, 'No bytecode writes and original assertions')
    resource.setrlimit(resource.RLIMIT_AS, (512*2**20, 512*2**20))
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    diagnosis = metadata.Reader()(inputs.DIAGNOSIS_REF)
    read = metadata.Reader()
    meta = []
    for path, digest in diagnosis['metadata_source_sha256'].items():
        ref = dict(path=path, sha256=digest)
        read(ref)
        size = Path(path).stat(follow_symlinks=False).st_size
        meta.append(io.snapshot(ref, size, os.getuid()))
    require(len(read.pins) == 95 and read.bytes == 27711882
            and read.pins == diagnosis['metadata_source_sha256'], 'All95 original JSON inputs authenticated')
    raw = []
    for item in diagnosis['bundle_inventory']:
        bundle = item['bundle']
        raw.append(io.snapshot(dict(path=bundle['path'], sha256=bundle['sha256']),
                               bundle['raw_bytes'], os.getuid()))
    manifest = dict(schema='minicpm5_saved_tap_input_manifest_v1', reviewer='/root',
        diagnosis=inputs.DIAGNOSIS_REF, uid=os.getuid(), metadata=meta, bundles=raw,
        raw_payloads_opened_during_snapshot=False, runtime_admitted=False,
        full_model_qualified=False, new_speedup_claim=False)
    inputs.validate_manifest(diagnosis, manifest, os.getuid())
    manifest_ref = once(OUT, manifest)
    result = dict(schema='minicpm5_saved_tap_input_snapshot_v1', reviewer='/root',
        manifest=manifest_ref, original_diagnosis=inputs.DIAGNOSIS_REF,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__).resolve(), Path(inputs.__file__).resolve(), Path(io.__file__).resolve())},
        authenticated_metadata_files=95, authenticated_metadata_bytes=27711882,
        raw_file_identities_snapshotted=84, raw_declared_bytes=3145728,
        raw_payloads_opened=False, raw_hashes_recomputed=False,
        all179_file_identities_distinct=True, all_ancestor_chains_ordinary=True,
        runtime_admitted=False, actual_tensor_analysis_executed=False,
        full_model_qualified=False, new_speedup_claim=False)
    result_ref = once(REVIEW, result)
    print(json.dumps(dict(manifest=manifest_ref, snapshot=result_ref,
        authenticated_metadata_files=95, raw_file_identities_snapshotted=84,
        raw_payloads_opened=False, runtime_admitted=False)))


if __name__ == '__main__':
    main()
