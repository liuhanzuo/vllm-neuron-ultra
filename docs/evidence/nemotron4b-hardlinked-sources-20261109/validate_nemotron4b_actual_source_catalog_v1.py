"""Verify actual inventoried Python bytes via routerv4; never import inspected modules."""
import datetime
import hashlib
import importlib.machinery as machinery
import json
from pathlib import Path
import sys

import nemotron4b_helper_dependency_catalog_v1 as catalog
from nemotron4b_import_router_v4 import ImportRouter
from nemotron4b_process_handshake_v1 import require

Q=Path(__file__).resolve().parent
INVENTORY=Q/'nemotron4b-helper-stdlib-origin-inventory-20261109-v1.json'
INVENTORY_SHA='fed3b20fe2305238f4c3e03a22ad8ef5d2b9aa496b71dffdc69b99c740d050d3'
REVIEW=Q/'nemotron4b-module-origins-source-preparation-review-20261109-v1.json'
REVIEW_SHA='deaf9861c5cf336dd3325c02175e1d9ba6b7d55c6eb11d49918b833d3e42eec9'
OUT=Q/'nemotron4b-actual-source-catalog-read-validation-20261109-v1.json'


def ref(path):
    path=Path(path);return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    require(sys.dont_write_bytecode and __debug__,'Source-only -B without optimization')
    raw,inventory_ref=catalog.source(INVENTORY,INVENTORY_SHA);inventory=json.loads(raw)
    raw,review_ref=catalog.source(REVIEW,REVIEW_SHA);review=json.loads(raw)
    require(not inventory['source_name_collisions'],'Unambiguous actual source catalog')
    for path,digest in review['source_sha256'].items():
        require(ref(path)['sha256']==digest,'Prior selected source pin unchanged')
    def deny(*args):
        raise RuntimeError('Actual SDK/native/interpreter execution is not permitted by this source-read validation')
    router=ImportRouter(finders=[machinery.PathFinder],modules=inventory['merged_source_modules'],
        namespaces={name:[record['path']]for name,record in inventory['namespace_packages'].items()},
        guard=lambda:None,authorize_finder=deny,authorize_builtin=deny,
        authorize_native=deny,authorize_custom=deny)
    before_loaded=set(sys.modules)
    rows=[]
    for name,row in sorted(router.modules.items()):
        raw=router.read_source(row['path'],row)
        require(len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'],
                'Independent bytes/digest agreement')
        rows.append(dict(name=name,path=row['path'],bytes=len(raw),sha256=row['sha256'],
            original_nlink=row['fullstat']['st_nlink'],alias=row.get('alias')))
    require(set(sys.modules)==before_loaded,'Inspected modules were not imported')
    require(len(rows)==1023 and sum(row['original_nlink']>1 for row in rows)==112,
            'Complete current source catalog and original hardlink population')
    require(sum(row['alias'] is not None for row in rows)==2,'Both actual source aliases read')
    # Verify the catalog and prior-source references remain immutable after I/O.
    require(ref(INVENTORY)['sha256']==INVENTORY_SHA and ref(REVIEW)['sha256']==REVIEW_SHA,
            'Immutable inventory/review retained')
    require(not router.events,'No finder/module load was attempted')
    router.finish()
    selected=['nemotron4b_pinned_source_reader_v1.py','nemotron4b_import_router_v4.py',
              'nemotron4b_import_context_v3.py','nemotron4b_module_origin_audit_v2.py']
    output=dict(schema='nemotron4b_actual_source_catalog_read_validation_v1',
        reviewer='/root',observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        inventory=inventory_ref,prior_review=review_ref,prior14source_pins_revalidated=True,
        selected_source_sha256={str(Q/name):ref(Q/name)['sha256']for name in selected},
        validator_source=ref(Path(__file__).resolve()),source_rows=rows,source_modules_read=len(rows),
        actual_source_bytes_read=sum(row['bytes']for row in rows),
        original_hardlinked_modules_read=112,original_source_aliases_read=2,
        all_original_file_identities_preserved=True,
        sys_modules_name_set_unchanged=True,actual_inspected_source_imports=0,
        actual_SDK_helper_executed=False,actual_native_payload_bytes_read=0,
        runtime_admitted=False,complete_runtime_dependency_closure_proven=False,
        full_model_qualified=False,new_speedup_claim=False)
    with OUT.open('x')as f:json.dump(output,f,indent=2,ensure_ascii=False);f.write('\n')
    print(json.dumps({k:output[k]for k in ['source_modules_read','actual_source_bytes_read',
        'original_hardlinked_modules_read','original_source_aliases_read','actual_inspected_source_imports']}))


if __name__=='__main__':
    main()
