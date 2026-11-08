"""Root-side catalog transport review. No launch, success or release inference.

Run only after independent actual external exit and physical drain. This helper
returns an expanded catalog and all physical dependency hashes for a root
finalizer to append, without rewriting or authorizing the catalog itself.
"""
import hashlib,json,os,re,sys
from pathlib import Path
R=Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
sys.path.insert(0,str(R))
import smollm3_observer_metadata_store_v1 as store
import smollm3_observer_metadata_codec_v2 as codec
require=codec.require

def snapshot(path,cap):
    path=Path(path);before=store.ordinary(path);require(before.st_size<=cap,'Bounded finalizer metadata')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        require(store.identity(os.fstat(fd))==store.identity(before),'Finalizer path changed on open')
        with os.fdopen(fd,'rb',closefd=False)as f:raw=f.read(cap+1)
        require(len(raw)==before.st_size and store.identity(os.fstat(fd))==store.identity(before)==store.identity(store.ordinary(path)),'Finalizer metadata changed while reading')
    finally:os.close(fd)
    return raw,dict(path=str(path),sha256=codec.sha(raw),bytes=len(raw)),store.identity(before)

def collect(root,expected_catalog):
    root=Path(root);path=root/'control/successor-catalog.json'
    require(type(expected_catalog)is dict and set(expected_catalog)=={'path','sha256'}and expected_catalog['path']==str(path)and re.fullmatch('[0-9a-f]{64}',expected_catalog['sha256']),'Exact root-observed catalog reference')
    original,reference,original_stat=snapshot(path,codec.RECORD_CAP)
    require(reference['sha256']==expected_catalog['sha256'],'Catalog changed since root observation')
    catalog=store.decode(root,path,original)
    require(type(catalog)is dict and type(catalog.get('source_sha256'))is dict and type(catalog.get('rows'))is list and type(catalog.get('processes'))is list,'Complete expanded catalog fields')
    require(catalog.get('root_finalized')is False and catalog.get('requires_root_terminal_review')is True,'Unfinalized catalog never self-authorizes')
    dependencies=store.authenticated_chunks(root,path,original)
    expected={item['path']:item for item in dependencies};all_files=[];pending=[];unreferenced=[];non_source=[];identities={}
    for leaf in ('source-maps','catalog-maps'):
        directory=root/'control'/leaf;before=store.ordinary(directory,True);names=sorted(x.name for x in directory.iterdir())
        require(len(names)<=4096,'Bounded metadata chunk directory')
        for name in names:
            child=directory/name;raw,item,identity=snapshot(child,codec.CHUNK_CAP);identities[str(child)]=identity
            is_pending=name.startswith('.pending-')
            if is_pending:
                require(re.fullmatch(r'\.pending-[0-9a-f]{64}\.json-[1-9][0-9]*-[0-9a-f]{32}',name),'Known retained pending chunk name')
                pending.append(item)
            else:
                require(re.fullmatch('[0-9a-f]{64}\.json',name)and child.stem==item['sha256'],'Complete chunk filename/hash')
                if leaf=='catalog-maps':
                    if str(child)in expected:require(expected[str(child)]==item,'All declared catalog dependencies exact')
                    else:
                        unreferenced.append(item)
                        try:codec.decode_json(raw)
                        except (ValueError,UnicodeError,RecursionError):non_source.append(item)
                else:require(catalog['source_sha256'].get(str(child))==item['sha256'],'Every ordinary source chunk retained in catalog')
            all_files.append(item)
        require(names==sorted(x.name for x in directory.iterdir())and store.directory_identity(store.ordinary(directory,True))==store.directory_identity(before),'Finalizer directory changed')
    require(set(expected)<={x['path']for x in all_files},'Complete referenced catalog dependency set')
    for name,identity in identities.items():require(store.identity(store.ordinary(name))==identity,'Finalizer dependency changed after collection')
    require(store.identity(store.ordinary(path))==original_stat,'Catalog changed after dependency expansion')
    result=dict(schema='smollm3_sio12_catalog_transport_root_review_v2',catalog_reference=reference,catalog_semantic_sha256=codec.sha(codec.encode_json(catalog)),
                catalog_dependencies=dependencies,all_physical_chunk_files=all_files,retained_pending_files=pending,unreferenced_catalog_chunks=unreferenced,
                physical_sha256={item['path']:item['sha256']for item in [reference,*all_files]},
                authenticated_source_sha256_additions={item['path']:item['sha256']for item in [reference,*all_files]if not Path(item['path']).name.startswith('.pending-')and item not in non_source},
                pending_bytes_are_not_source_text=True,non_source_extra_files=non_source,semantic_rows=len(catalog['rows']),semantic_processes=len(catalog['processes']),
                closure_authenticated=True,extra_evidence_requires_review=bool(pending or unreferenced),actual_exit_observed=False,physical_drain_proven=False,root_finalization_granted=False,scientific_success_proven=False,new_speedup_claim=False)
    return catalog,result
