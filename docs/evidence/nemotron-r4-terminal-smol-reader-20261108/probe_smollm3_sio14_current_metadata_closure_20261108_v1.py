"""Full current source-byte closure, no model/native payload bodies or admission."""
from pathlib import Path
import sys,json,os,hashlib,time
R=Path(__file__).resolve().parents[1];Q=R/'research';sys.path.insert(0,str(R))
import smollm3_observer_invariance_lifecycle_v24 as life
import historical_b3_payload_partition_v1 as history
import nemotron9b_package_protocol_v10 as p
cat_ref=dict(path=str(Q/'nemotron9b-package-r4-20261108-complete-catalog-v1.json'),sha256='d13b07ca6f7afbc0571806705b3f7b5a145cc780fcea58acd0cf737c4aac1f40');cat=life.reference(cat_ref);parts=history.partition(cat)
handoff=json.loads((Q/'smollm3-sio13-metadata-source-handoff-20261108-v1.json').read_text());aliases=handoff['original_metadata_symlink_resolutions']
manifest=json.loads((Q/'smollm3-sio14-runtime-source-candidate-20261108-v2.json').read_text());pins=dict(parts['source_sha256']);pins.update(parts['provenance_sha256']);pins[cat_ref['path']]=cat_ref['sha256']
for row in manifest['files']+[manifest['reader']]:pins[row['path']]=row['sha256']
owner=p.low.proc(os.getpid());started=time.monotonic();stats=life.source_closure(dict(source_sha256=pins),{},aliases,hash_bytes=True);elapsed=time.monotonic()-started
result=dict(schema='smollm3_sio14_full_current_metadata_closure_v1',reviewer='/root',owner=owner,source_only=True,native_imported=life.p.native_loaded(),runtime_admitted=False,final_catalog=cat_ref,source_entries_rehashed=len(pins),stat_entries=len(stats),metadata_aliases=aliases,historical_payload_count=len(parts['historical_payload_sha256']),historical_payload_bodies_read=False,reader_sha256=hashlib.sha256((R/'smollm3_observer_metadata_source_v4.py').read_bytes()).hexdigest(),candidate_manifest_sha256=hashlib.sha256((Q/'smollm3-sio14-runtime-source-candidate-20261108-v2.json').read_bytes()).hexdigest(),complete=True,elapsed_seconds=elapsed,not_model_speedup=True)
path=Q/'smollm3-sio14-full-current-metadata-closure-20261108-v1.json'
with path.open('x')as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
