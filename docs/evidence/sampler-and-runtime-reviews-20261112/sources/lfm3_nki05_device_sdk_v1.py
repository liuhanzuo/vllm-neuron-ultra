"""Concrete finite RECORD verification for the entire isolated Python3.12 prefix.

Manifest preparation reads only dist-info text. Actual payload verification is
called only by an owned future native child; this module imports no native API.
"""
import base64
import csv
import email
import hashlib
import os
from pathlib import Path
import stat
import sys
import time
from slot_cache_nki05_typed_jit_v1 import DEVICE_SOFTWARE
import lfm3_nki05_device_budget_v1 as budget

PREFIX=Path('/home/ubuntu/venvs/vllm_neuron_053_py312')
SITE=PREFIX/'lib/python3.12/site-packages'
REQUIRED=dict(DEVICE_SOFTWARE,**{'torch-xla':'2.9.0','numpy':'2.5.4','ml-dtypes':'0.6.0'})


def require(value,message):
    if not value:raise RuntimeError(message)


def canonical(name):return name.lower().replace('_','-').replace('.','-')


def metadata_bytes(path):
    path=Path(path)
    require(path.name in ('METADATA','RECORD') and path.parent.parent==SITE
            and path.resolve()==path and not any(p.is_symlink() for p in (path,*path.parents)),'Exact installed text metadata only')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC)
    try:
        before=os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_size<=budget.METADATA_BYTES,'Bounded metadata before reading')
        chunks=[];size=0
        while True:
            block=os.read(fd,min(2**20,budget.METADATA_BYTES+1-size))
            if not block:break
            size+=len(block);require(size<=budget.METADATA_BYTES,'Finite complete metadata');chunks.append(block)
        after=os.fstat(fd)
        require(size==before.st_size and all(getattr(before,k)==getattr(after,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')),'Stable complete installed metadata')
        raw=b''.join(chunks);raw.decode('utf8');return raw
    finally:os.close(fd)


def metadata_manifest():
    """Concrete metadata-only preparation, no output write or artifact hash."""
    rows=[]
    directories=sorted(SITE.glob('*.dist-info'))
    require(0<len(directories)<=512,'Finite complete installed prefix distribution census')
    for directory in directories:
        require(directory.resolve()==directory and not directory.is_symlink(),'Exact dist-info directory')
        files={}
        for name in ('METADATA','RECORD'):
            path=directory/name;raw=metadata_bytes(path)
            files[name]=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest())
            if name=='METADATA':message=email.message_from_bytes(raw)
        rows.append(dict(name=canonical(message['Name']),version=message['Version'],directory=str(directory),files=files))
    require(len({row['name'] for row in rows})==len(rows),'No ambiguous installed distribution names')
    versions={row['name']:row['version'] for row in rows}
    require(all(versions.get(k)==v for k,v in REQUIRED.items()),'Exact scientific device distributions')
    return dict(schema='lfm3_nki05_device_sdk_manifest_v1',prefix=str(PREFIX),site=str(SITE),
        distributions=rows,complete_prefix_census=True,source_overrides={},native_payload_verified=False)


def validate_manifest(value):
    require(type(value) is dict and set(value)=={'schema','prefix','site','distributions','complete_prefix_census','source_overrides','native_payload_verified'},'Exact SDK manifest fields')
    require(value['schema']=='lfm3_nki05_device_sdk_manifest_v1' and value['prefix']==str(PREFIX) and value['site']==str(SITE)
            and value['complete_prefix_census'] is True and value['native_payload_verified'] is False,'Separate reviewed installed-prefix provenance')
    rows=value['distributions'];require(type(rows) is list and 0<len(rows)<=512,'Complete bounded distribution rows')
    require([row['directory'] for row in rows]==sorted(row['directory'] for row in rows),'Stable complete distribution order')
    require(len({row['directory'] for row in rows})==len(rows),'Every metadata directory exactly once')
    for row in rows:
        require(type(row) is dict and set(row)=={'name','version','directory','files'} and type(row['name']) is str and row['name']==canonical(row['name']),'Exact canonical distribution record')
        directory=Path(row['directory'])
        require(directory.parent==SITE and directory.name.endswith('.dist-info') and set(row['files'])=={'METADATA','RECORD'},'Exact isolated installed metadata directory')
        for name,ref in row['files'].items():
            require(set(ref)=={'path','sha256'} and ref['path']==str(directory/name) and type(ref['sha256']) is str and len(ref['sha256'])==64 and set(ref['sha256'])<=set('0123456789abcdef'),'Exact bounded metadata references')
    versions={row['name']:row['version'] for row in rows}
    require(len(versions)==len(rows) and all(versions.get(k)==v for k,v in REQUIRED.items()),'Exact scientific package versions')
    require(type(value['source_overrides']) is dict and len(value['source_overrides'])<=32,'Explicit bounded previously reviewed source corrections only')
    for path,row in value['source_overrides'].items():
        require(Path(path).is_absolute() and str(Path(path))==path and '..' not in Path(path).parts
                and Path(path).is_relative_to(SITE) and Path(path).suffix=='.py'
                and set(row)=={'record_sha256','actual_sha256','review'} and set(row['review'])=={'path','sha256'},'No unauthenticated native binary override')
        require(Path(row['review']['path']).is_absolute() and Path(row['review']['path']).suffix=='.json'
                and all(type(digest) is str and len(digest)==64 and set(digest)<=set('0123456789abcdef')
                        for digest in (row['record_sha256'],row['actual_sha256'],row['review']['sha256'])),'Complete exact reviewed source override digests')
    return value


def verify(value,context):
    """Concrete file hashing follows actual owned Context authorization."""
    validate_manifest(value);context.authorize(context.phase)
    require(sys.prefix==str(PREFIX) and sys.version_info[:2]==(3,12) and sys.dont_write_bytecode,'Exact isolated native child interpreter')
    import smollm3_record_bytecode_v1 as bytecode
    bytecode.isolated(context.run_root/'tmp'/'empty-pycache')
    require([str(p) for p in sorted(SITE.glob('*.dist-info'))]==[row['directory'] for row in value['distributions']], 'No missing or extra installed distribution')
    began=time.monotonic();total=count=0;seen={};checked=[];overrides=set()
    for distribution in value['distributions']:
        context.authorize(context.phase)
        records={}
        for ref in distribution['files'].values():
            raw=metadata_bytes(ref['path'])
            require(hashlib.sha256(raw).hexdigest()==ref['sha256'],'Pinned installed metadata unchanged')
            records[Path(ref['path']).name]=raw
        record=Path(distribution['files']['RECORD']['path'])
        raw=records['RECORD']
        files=csv.reader(raw.decode().splitlines());local=0
        for relative,encoded,size in files:
            context.fast_authorize()
            require(time.monotonic()-began<budget.SDK_VERIFY_SECONDS,'Finite entire-prefix payload verification')
            path=(SITE/relative).resolve()
            require(path.is_relative_to(PREFIX),'RECORD target stays inside the exact prefix')
            require(path.suffix!='.pyc','Current reviewed prefix has zero pyc RECORD rows; no bytecode exemption')
            if not encoded:
                require(path==record,'Only RECORD itself may be unrecorded')
                continue
            algorithm,value64=encoded.split('=',1);require(algorithm=='sha256','Strong RECORD hash')
            expected=base64.urlsafe_b64decode(value64+'='*(-len(value64)%4)).hex()
            override=value['source_overrides'].get(str(path))
            if override:
                require(expected==override['record_sha256'],'Override tied to original RECORD digest')
                review=context.host.p.read(override['review']['path'],override['review']['sha256'])
                require(review['reviewer']=='/root' and review['passed'] is True and review['source_sha256'].get(str(path))==override['actual_sha256'],'Independent exact accepted source override review')
                expected=override['actual_sha256'];overrides.add(str(path))
            if str(path) in seen:
                require(seen[str(path)]==expected,'Shared package file has one exact identity');continue
            before=path.stat();require(stat.S_ISREG(before.st_mode) and before.st_size<=budget.SDK_FILE_BYTES,'Bounded regular runtime payload')
            digest=hashlib.sha256();read_bytes=0
            with path.open('rb') as stream:
                require(os.fstat(stream.fileno()).st_ino==before.st_ino,'Stable runtime file before hashing')
                while True:
                    block=stream.read(2**20)
                    if not block:break
                    read_bytes+=len(block);digest.update(block)
                    require(read_bytes<=budget.SDK_FILE_BYTES and time.monotonic()-began<budget.SDK_VERIFY_SECONDS,'Bounded complete runtime hash')
                after=os.fstat(stream.fileno())
            require(all(getattr(before,k)==getattr(after,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns'))
                    and digest.hexdigest()==expected and read_bytes==before.st_size,'Complete stable SDK file matches reviewed RECORD/source override')
            if not override:require(read_bytes==int(size),'Complete original RECORD size')
            count+=1;total+=read_bytes;local+=1;seen[str(path)]=expected
            require(count<=budget.SDK_FILE_COUNT and total<=budget.SDK_TOTAL_BYTES,'Finite whole-prefix SDK payload bound')
        checked.append(dict(name=distribution['name'],version=distribution['version'],files=local))
    require(overrides==set(value['source_overrides']),'Every explicit source override actually verified')
    context.authorize(context.phase)
    return dict(complete=True,passed=True,files=count,bytes=total,distributions=checked,source_overrides=sorted(overrides),
                seconds=time.monotonic()-began,model_weights_read=False)
