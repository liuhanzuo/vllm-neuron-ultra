"""Explicit root metadata preparation only; never launch or approve a run.

Run later in root's serial metadata context. Import is inert. Catalog/handoff/
last terminal are mandatory caller references, with no latest-file discovery.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

Q=Path(__file__).resolve().parent
R4_RECEIPT=dict(path=str(Q/'minicpm5-formal-import-r4-20261112-preparation-receipt.json'),
    sha256='30221d646fabf49bd2cbd7498b349447690ff7a5eb66df78b9a93ff2007352f7')
NEW_SOURCES=('minicpm5_completed_qwen_package_text_v2.py','minicpm5_fixed_input_pv_catalog_auth_v2.py',
    'prepare_minicpm5_pv_sdk_import_catalog_v3.py','minicpm5_sdk_mapping_outer_source_entry_v9.py',
    'prepare_minicpm5_catalog_formal_binding_v3.py','prepare_minicpm5_formal_r5_metadata_root_v1.py')
BLOCKED={'torch','numpy','torch_neuronx','torch_xla','neuronxcc','neuronx_distributed','neuronx_distributed_inference','nki'}


class NoFramework:
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in BLOCKED:raise RuntimeError('No framework in metadata preparation: '+name)


def assemble_candidates(inputs,*,read,text,guard,original):
    """Actual original reconstruction runs only when root explicitly calls this."""
    from research import prepare_minicpm5_catalog_formal_binding_v3 as binder
    require=binder.require
    require(set(inputs)=={'catalog','handoff','predecessor','handoff_catalog_field','artifact_prefix',
        'output_root','attempt_id','unit','source_delivery'},'Every future metadata input explicit')
    for key in ('catalog','handoff','predecessor','source_delivery'):binder.launch.reference(inputs[key])
    delivery=read(inputs['source_delivery'])
    require(delivery['schema']=='minicpm5_completed_slot_catalog_source_delivery_v1'
        and delivery['source_only'] is True and delivery['runtime_binding_created'] is False,
        'Exact source-only successor delivery')
    rows=delivery['implementation_sources']
    require(len(rows)==len(NEW_SOURCES) and {r['path'] for r in rows}=={str(Q/n) for n in NEW_SOURCES},
            'Exactly six reviewed successor/preparer files')
    for row in rows:
        payload=text(row['path'],row['sha256'])
        require(len(payload)==row['bytes'] and hashlib.sha256(payload).hexdigest()==row['sha256'],
                'Full frozen successor source unchanged')
    receipt=read(R4_RECEIPT);old_formal=read(receipt['request']);old=read(old_formal['original_request'])
    prefix=inputs['artifact_prefix'];virtual={};artifacts={}
    def add(name,value,cap=16*2**20):
        payload=binder.launch.encoded(value)
        require(0<len(payload)<=cap,'Complete candidate metadata cap')
        reference=dict(path=prefix+'-'+name+'.json',sha256=hashlib.sha256(payload).hexdigest())
        artifacts[name]=dict(reference=reference,bytes=len(payload),value=copy.deepcopy(value))
        virtual[reference['path']]=(reference,copy.deepcopy(value));return reference
    old=copy.deepcopy(old);old.update(current_serial_catalog=copy.deepcopy(inputs['catalog']),output_root=inputs['output_root'])
    own=next(row for row in rows if row['path']==str(Path(__file__).resolve()))
    old['additional_source_refs']=old['additional_source_refs']+[dict(path=own['path'],sha256=own['sha256'])]
    original_ref=add('original-request',old)
    request=copy.deepcopy(old_formal)
    request.update(original_request=original_ref,current_handoff=copy.deepcopy(inputs['handoff']),
        expected_predecessor_exit=copy.deepcopy(inputs['predecessor']),
        handoff_catalog_field=inputs['handoff_catalog_field'],artifact_prefix=prefix,
        attempt_id=inputs['attempt_id'],unit=inputs['unit'])
    request_ref=add('request',request)
    def virtual_read(reference):
        if reference['path'] in virtual:
            expected,value=virtual[reference['path']]
            require(reference==expected,'Exact in-memory future request reference')
            return copy.deepcopy(value)
        return read(reference)
    refs=[]
    for name in binder.SOURCES:
        path=str(Q/name);payload=text(path,None)
        refs.append(dict(path=path,sha256=hashlib.sha256(payload).hexdigest()))
    result=binder.prepare(request,literal_source_manifest=receipt['literal_source_manifest'],
        preparation_source=receipt['literal_preparation_source'],catalog_source_refs=refs,
        read=virtual_read,text=text,guard=guard,original=original)
    require(result['original_nine_native_checks_pending']==9 and result['original_scientific_rows_still_pending']==1284
        and result['runtime_admitted'] is False and result['launch_authorized'] is False,
        'Original native/scientific gates remain pending')
    require(not(set(artifacts)&set(result['artifacts'])),'Unique fresh artifact names')
    artifacts.update(result['artifacts'])
    add('original-preparation',result['original_preparation'])
    out_receipt=dict(schema='minicpm5_formal_import_root_preparation_receipt_v2',source_metadata_only=True,
        request=request_ref,catalog=inputs['catalog'],handoff=inputs['handoff'],predecessor=inputs['predecessor'],
        source_delivery=inputs['source_delivery'],catalog_rows=result['catalog_rows'],
        complete_source_files=result['original_preparation']['complete_source_files'],
        complete_source_map_sha256=result['original_preparation']['complete_source_map_sha256'],
        entry_source=result['entry_source'],outer_entry_plan=result['outer_entry_plan'],
        artifacts={name:{key:row[key] for key in ('reference','bytes')} for name,row in artifacts.items()},
        original_nine_native_checks_pending=9,original_scientific_rows_still_pending=1284,
        reviews_still_pending=True,runtime_admitted=False,launch_authorized=False)
    add('preparation-receipt',out_receipt)
    return artifacts


def persist_candidates(artifacts):
    """Write exact complete candidates exclusively; all reviews stay drafts."""
    from research import minicpm5_sdk_mapping_nocache_launch_v3 as launch
    paths=[Path(row['reference']['path']) for row in artifacts.values()]
    if len(set(paths))!=len(paths) or not all(p.parent==Q and not p.exists() and not p.is_symlink() for p in paths):
        raise ValueError('All metadata candidate destinations must be fresh direct research files')
    for row in artifacts.values():
        payload=launch.encoded(row['value'])
        if len(payload)!=row['bytes'] or hashlib.sha256(payload).hexdigest()!=row['reference']['sha256']:
            raise ValueError('Exact complete candidate bytes before publication')
    for row in artifacts.values():
        fd=os.open(row['reference']['path'],os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW|os.O_CLOEXEC,0o600)
        with os.fdopen(fd,'wb') as stream:
            stream.write(launch.encoded(row['value']));stream.flush();os.fsync(stream.fileno())
    return artifacts['preparation-receipt']['reference']


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('catalog','handoff','predecessor','source-delivery'):
        parser.add_argument('--'+name,required=True);parser.add_argument('--'+name+'-sha256',required=True)
    for name in ('artifact-prefix','output-root','attempt-id','unit'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--handoff-catalog-field',choices=('complete_current_serial_catalog','complete_current_serial_catalog_next'),required=True)
    args=parser.parse_args(argv)
    if not(sys.dont_write_bytecode and sys.flags.isolated and __debug__):raise ValueError('Use isolated -I -B metadata interpreter')
    if BLOCKED.intersection(sys.modules):raise ValueError('Framework already imported')
    sys.meta_path.insert(0,NoFramework());sys.path[:0]=[str(Q),str(Q.parent)]
    from research import prepare_minicpm5_pv_sdk_import_diagnostic_v1 as original
    reader=original.MetadataOnlyReader()
    inputs={key:dict(path=getattr(args,key),sha256=getattr(args,key+'_sha256'))
            for key in ('catalog','handoff','predecessor','source_delivery')}
    inputs.update({key:getattr(args,key) for key in ('artifact_prefix','output_root','attempt_id','unit','handoff_catalog_field')})
    # Current runtime output is never created, and prior outputs are never reused.
    if Path(args.output_root).exists() or Path(args.output_root).is_symlink():raise ValueError('Fresh future runtime output required')
    artifacts=assemble_candidates(inputs,read=reader.read,text=reader.text,guard=reader.guard,original=original)
    reader.unchanged()
    receipt=persist_candidates(artifacts)
    print(json.dumps(dict(preparation_receipt=receipt,metadata_artifacts=len(artifacts),
        runtime_admitted=False,launch_authorized=False,original_native_checks_pending=9,scientific_rows_pending=1284),sort_keys=True))
    return 0


if __name__=='__main__':raise SystemExit(main())
