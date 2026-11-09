"""Observer-only actual source graph and stat-only input compatibility check.

Does not reconstruct a future completed Nemotron run, acquire the serial lock,
build/write a runtime specification, import query native libraries or open raw
tensor files. Source tests and source preparation never qualify the model.
"""
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

import prepare_minicpm5_saved_tap_runtime_v1 as builder
import minicpm5_saved_tap_io_v1 as io
from prepare_dory_residual_spec_v3 import local_imports,python_bytes

c=builder.c
OUT=c.Q/'minicpm5-saved-tap-runtime-source-inventory-20261109-v1.json'
NEMOTRON_REVIEW=dict(path=str(c.Q/'nemotron9b-v23-after-saved-retry-source-preparation-review-20261109-v1.json'),
    sha256='785667dccbafb116f94351598ef2ce4e548c0497056116a63180a4b45ff83a40')


def main():
    c.require(sys.dont_write_bytecode and sys.flags.optimize==0,'No bytecode or native execution')
    resource.setrlimit(resource.RLIMIT_AS,(512*2**20,)*2)
    resource.setrlimit(resource.RLIMIT_CPU,(60,60))
    started=time.monotonic()
    def guard():c.require(time.monotonic()-started<60,'Source/metadata-only time budget')
    reader=c.history.Reader(guard)
    io_review=reader(c.IO_REVIEW);nemotron_review=reader(NEMOTRON_REVIEW)
    c.require(io_review['source_preparation_passed'] is nemotron_review['source_preparation_passed'] is True,
              'Existing original source preparation records')
    original={}
    for record in (io_review,nemotron_review):c.history.merge(original,record['source_sha256'])
    for path,digest in original.items():
        guard();raw=python_bytes(path)
        c.require(hashlib.sha256(raw).hexdigest()==digest,'Original source unchanged: '+path)
    graph=local_imports(entries=builder.ENTRIES)
    for path,digest in original.items():
        c.require(path not in graph['source_sha256'] or graph['source_sha256'][path]==digest,
                  'No prior scientific/guard source changed in new import graph')
    loaded=c.loaded_local_sources(graph['source_sha256'])
    diagnosis=reader(c.inputs.DIAGNOSIS_REF);manifest=reader(c.MANIFEST)
    meta,raw=c.inputs.validate_manifest(diagnosis,manifest,os.getuid())
    for row in (*meta,*raw):
        guard()
        current=io.snapshot(dict(path=row['path'],sha256=row['sha256']),row['bytes'],os.getuid())
        c.require(current==row,'Original input/ancestor identity changed; immutable successor required')
    c.require(not c.OUT.exists(),'No actual saved-tap run has started')
    value=dict(schema='minicpm5_saved_tap_runtime_source_inventory_v1',reviewer='/root',
        prior_io_review=c.IO_REVIEW,prior_nemotron_review=NEMOTRON_REVIEW,
        original_source_pins_revalidated=original,local_import_graph=graph,
        loaded_local_metadata_module_origins=loaded,input_manifest=c.MANIFEST,
        original_input_identities_stat_verified=179,metadata_input_identities=95,raw_input_identities=84,
        all_input_identities_and_ancestors_unchanged=True,
        prepared_profiles=list(c.history.PROFILES),standard_failed_release_supported=True,
        original52stages_and_final_raw_check_retained=True,
        metadata_references_authenticated=reader.pins,
        raw_tensor_bytes_read=0,native_query_executed=False,model_executed=False,
        actual_future_predecessor_completion_observed=False,runtime_specification_written=False,
        serial_lock_acquired=False,runtime_admitted=False,full_model_qualified=False,new_speedup_claim=False)
    data=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
    c.require(len(data)<=8*2**20,'Bounded source preparation inventory')
    with OUT.open('xb')as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
    print(json.dumps(dict(result=dict(path=str(OUT),sha256=hashlib.sha256(data).hexdigest()),
        local_python_sources=graph['files'],local_source_bytes=graph['bytes_read'],
        original_pins_unchanged=len(original),stat_only_input_files=179,
        loaded_local_metadata_modules=len(loaded),raw_tensor_bytes_read=0,native_query_executed=False,
        runtime_specification_written=False)))


if __name__=='__main__':main()
