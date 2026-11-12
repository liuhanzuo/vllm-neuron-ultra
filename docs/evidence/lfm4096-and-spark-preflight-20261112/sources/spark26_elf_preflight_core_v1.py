"""Future owned metadata-only preflight; import performs no file/native IO.

Hash the frozen declared application payloads and inspect ELF metadata through
the same still-open descriptor. Never execute a payload, resolve a dependency
by loading it, discover/drop candidate files, or read the model checkpoint.
Only the concrete successor WorkerContext may invoke collect().
"""
import hashlib
import os
from pathlib import Path
import stat
import struct

import spark26_native_provider_v2 as provider

require=provider.require
GIB,MIB=2**30,2**20
CAPS=dict(payload_hash_bytes=16*GIB,total_payload_and_ELF_bytes=16*GIB,
          ELF_metadata_bytes=128*MIB,ELF_per_object_bytes=MIB,
          program_headers=1024,dynamic_bytes=256*1024,string_bytes=4096,
          dynamic_strings=256,output_row_bytes=3072)
MANIFEST_SHA='ee81ffa1a4465b972ebf6cf73dc472789946ff244818dd0a4375276e4d603524'
PREPARATION_SHA='25b9b086e504f828e099ba685630dc9d9d2cc6912cac1bd9332ccdb99d9a949e'
CANDIDATE=provider.contract.Q/'spark26-native-manifest-candidate-20261112-v2'
MANIFEST_REF=dict(path=str(CANDIDATE/'manifest.json'),sha256=MANIFEST_SHA)
PREPARATION_REF=dict(path=str(CANDIDATE/'preparation.json'),sha256=PREPARATION_SHA)
COUNTS=dict(payload_authentication=41173,ELF_metadata=1271,conflicting_owner=15)


class ObjectFailure(ValueError):pass


def object_require(ok,message):
    if not ok:raise ObjectFailure(message)


def parse_elf(read_at,file_bytes):
    """Pure bounded ELF64/LE/x86-64 parser; read_at is a byte-range callback.

    Production callback is an owned, precharged os.pread on the descriptor
    already fully hashed. Tests use in-memory synthetic bytes only.
    Unsupported ELF and malformed metadata produce explicit object failures.
    """
    def read(offset,size):
        object_require(type(offset)is int and type(size)is int and 0<=offset<=file_bytes
                       and 0<=size<=file_bytes-offset,'ELF range outside complete file')
        raw=read_at(offset,size);object_require(len(raw)==size,'Truncated ELF range');return raw
    header=read(0,64)
    object_require(header[:7]==b'\x7fELF\x02\x01\x01','Only ELF64 little-endian version1')
    values=struct.unpack('<HHIQQQIHHHHHH',header[16:])
    kind,machine,version,entry,phoff,shoff,flags,ehsize,phsize,phnum,shsize,shnum,shstr=values
    object_require(kind in (2,3) and machine==62 and version==1 and ehsize==64
        and phsize==56 and 0<phnum<=CAPS['program_headers'],'Bounded x86_64 executable/shared ELF headers')
    raw=read(phoff,phnum*phsize)
    segments=[]
    for i in range(phnum):
        ptype,pflags,offset,vaddr,paddr,filesz,memsz,align=struct.unpack_from('<IIQQQQQQ',raw,i*phsize)
        object_require(offset<=file_bytes and filesz<=file_bytes-offset and filesz<=memsz,
                       'Complete ELF program segment in file/memory bounds')
        segments.append(dict(type=ptype,flags=pflags,offset=offset,vaddr=vaddr,filesz=filesz,memsz=memsz))
    dynamic=[p for p in segments if p['type']==2]
    object_require(len(dynamic)==1,'Exactly one dynamic segment required for metadata collection')
    dyn=dynamic[0];object_require(0<dyn['filesz']<=CAPS['dynamic_bytes']and dyn['filesz']%16==0,'Bounded aligned dynamic entries')
    raw=read(dyn['offset'],dyn['filesz']);entries=[];terminated=False
    for offset in range(0,len(raw),16):
        tag,value=struct.unpack_from('<qQ',raw,offset)
        if tag==0:terminated=True;break
        entries.append((tag,value))
    object_require(terminated,'Dynamic table must terminate within its declared segment')
    def singleton(tag,required=False):
        values=[value for key,value in entries if key==tag]
        object_require(len(values)<=1 and (values or not required),'Ambiguous/missing dynamic singleton '+str(tag))
        return values[0]if values else None
    string_tags={1:'needed',14:'soname',15:'rpath',29:'runpath',0x7ffffffd:'auxiliary',
                 0x7fffffff:'filter',0x6ffffefb:'depaudit',0x6ffffefc:'audit'}
    strings=[(tag,value)for tag,value in entries if tag in string_tags]
    object_require(len(strings)<=CAPS['dynamic_strings'],'Bounded dynamic string count')
    straddr=singleton(5,bool(strings));strsize=singleton(10,bool(strings))
    table_offset=None
    if strings:
        object_require(strsize>0,'Nonempty dynamic string table')
        maps=[p['offset']+(straddr-p['vaddr'])for p in segments if p['type']==1
              and p['vaddr']<=straddr and straddr-p['vaddr']<=p['filesz']
              and strsize<=p['filesz']-(straddr-p['vaddr'])]
        object_require(len(maps)==1,'String table must have one complete file-backed LOAD mapping')
        table_offset=maps[0]
    result=dict(elf_class=64,endianness='little',machine=machine,elf_type=kind,needed=[],
                search_tags=[],extra_loader_directives=[],numeric_dynamic_entries=[list(x)for x in entries],
                actual_loader_selection_proven=False,dependency_closure_proven=False)
    for tag,index in strings:
        object_require(index<strsize,'String offset in dynamic string table')
        raw=read(table_offset+index,min(CAPS['string_bytes'],strsize-index))
        object_require(b'\0'in raw,'Terminated bounded dynamic string')
        try:value=raw.split(b'\0',1)[0].decode('utf8')
        except UnicodeError as error:raise ObjectFailure('Non-UTF8 dynamic loader string')from error
        object_require(value and '\n'not in value and '\r'not in value,'Nonempty single-line loader string')
        name=string_tags[tag]
        if name=='needed':result['needed'].append(value)
        elif name=='soname':
            object_require('soname'not in result,'Unique SONAME');result['soname']=value
        elif name in ('rpath','runpath'):result['search_tags'].append(dict(tag=name,value=value))
        else:result['extra_loader_directives'].append(dict(tag=name,value=value))
    interpreters=[p for p in segments if p['type']==3]
    object_require(len(interpreters)<=1,'Unique ELF interpreter')
    result['interpreter']=None
    if interpreters:
        p=interpreters[0];object_require(0<p['filesz']<=CAPS['string_bytes'],'Bounded interpreter path')
        raw=read(p['offset'],p['filesz']);object_require(raw.endswith(b'\0')and b'\0'not in raw[:-1],'Exact terminated interpreter')
        try:result['interpreter']=raw[:-1].decode('utf8')
        except UnicodeError as error:raise ObjectFailure('Non-UTF8 interpreter path')from error
    return result


def plan_from_payloads(manifest,preparation,payloads):
    require(manifest['schema']=='spark26_native_provider_manifest_v1'
            and len(payloads)==COUNTS['payload_authentication']
            and sum(row['bytes']for row in payloads.values())==13247146338
            and provider.contract.digest(payloads)==manifest['payload_sha256'],
            'Exact existing entire manifest retained')
    native=sorted(path for path in payloads if '.so'in Path(path).name)
    conflicts=[row for row in preparation['issues']if row['kind']=='ambient_conflicting_RECORD_owner']
    require(len(native)==COUNTS['ELF_metadata']and len(conflicts)==COUNTS['conflicting_owner']
        and len({row['path']for row in conflicts})==len(conflicts)
        and all(row['path']in payloads and row['selected_record']==payloads[row['path']]for row in conflicts),
        'Preserve every original1271 native candidate and15 conflicting owner')
    require(preparation['runtime_admitted']is False and preparation['complete_runtime_closure']is False,
            'Unadmitted source candidate only')
    return dict(schema='spark26_elf_preflight_plan_v1',manifest=MANIFEST_REF,preparation=PREPARATION_REF,
        payload_count=len(payloads),payload_bytes=manifest['payload_bytes'],payload_sha256=manifest['payload_sha256'],
        payload_order='lexicographically sorted canonical absolute path',ELF_paths=native,
        conflicting_owners=conflicts,counts=COUNTS,caps=CAPS,original_required=provider.REQUIRED,
        preserve_candidate_extras_and_all_blockers=True,source_candidate_runtime_admitted=False,
        metadata_collection_only=True,native_imports=False,model_checkpoint_reads=False,
        package_mutations=False,model_qualification=False,new_speedup_claim=False)


class ReadBudget:
    def __init__(self):self.hash_bytes=0;self.elf_bytes=0
    def reserve(self,kind,amount):
        require(type(amount)is int and amount>=0,'Finite nonnegative actual read request')
        h=self.hash_bytes+(amount if kind=='hash'else 0)
        e=self.elf_bytes+(amount if kind=='elf'else 0)
        require(kind in ('hash','elf')and h<=CAPS['payload_hash_bytes']and e<=CAPS['ELF_metadata_bytes']
            and h+e<=CAPS['total_payload_and_ELF_bytes'],'Actual payload hash plus ELF reads stay within original16GiB ceiling')
        self.hash_bytes,self.elf_bytes=h,e


def same_stat(value):return [value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns]


def payload_identity(path):
    try:return provider.identity(path)
    except (OSError,ValueError)as error:raise ObjectFailure('Payload path identity: '+str(error))from error


class PayloadObserver:
    """Real future fd/hash/pread operations; not used in source-only tests."""
    def __init__(self,context,budget):self.context,self.budget=context,budget
    def observe(self,path,expected,wants_elf):
        self.context.check();path=Path(path)
        before=payload_identity(path)
        object_require(before[2]==expected['bytes'],'Actual file size differs from selected RECORD')
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC)
        metadata=None;metadata_error=None
        try:
            object_require(same_stat(os.fstat(fd))==before,'Opened identity differs from selected path')
            digest=hashlib.sha256();count=0
            while count<expected['bytes']:
                self.context.check();request=min(MIB,expected['bytes']-count);self.budget.reserve('hash',request)
                block=os.read(fd,request);object_require(bool(block),'Truncated complete payload authentication')
                digest.update(block);count+=len(block)
            self.budget.reserve('hash',1);object_require(os.read(fd,1)==b'','Payload grew while hashing')
            actual_sha=digest.hexdigest();matched=actual_sha==expected['sha256']
            if wants_elf:
                local_bytes=0
                def read_at(offset,size):
                    nonlocal local_bytes
                    self.context.check();object_require(local_bytes+size<=CAPS['ELF_per_object_bytes'],'Per-object metadata read cap')
                    local_bytes+=size;self.budget.reserve('elf',size)
                    return os.pread(fd,size,offset)
                try:metadata=parse_elf(read_at,expected['bytes'])
                except ObjectFailure as error:metadata_error=str(error)
            object_require(same_stat(os.fstat(fd))==before and payload_identity(path)==before,
                           'Changed path or descriptor during full hash/ELF inspection')
            return dict(identity=before,actual_sha256=actual_sha,selected_RECORD_match=matched,
                        ELF=metadata,ELF_error=metadata_error)
        finally:os.close(fd)


def collect(context):
    from spark26_elf_preflight_worker_v1 import WorkerContext
    require(type(context)is WorkerContext,'Concrete independently bound metadata worker only')
    # No injected manifest, synthetic auth callback, or generic native runtime.
    context.check(full=True)
    manifest=context.read_metadata(MANIFEST_REF);preparation=context.read_metadata(PREPARATION_REF)
    payloads=provider.merge_manifest(manifest,context.spec['source_sha256'],
                                    lambda ref:context.metadata_bytes(ref).decode())
    plan=plan_from_payloads(manifest,preparation,payloads)
    require(plan==context.spec['preflight_plan'],'Same independently reviewed complete preflight plan')
    budget=ReadBudget();observer=PayloadObserver(context,budget)
    indexes={path:index for index,path in enumerate(sorted(payloads))}
    conflicts={row['path']:row for row in plan['conflicting_owners']};observations={}
    failures=0;ELF_failures=0
    for path,index in indexes.items():
        context.check();wants_elf=path in plan['ELF_paths'];observation=None
        try:observation=observer.observe(path,payloads[path],wants_elf)
        except (ObjectFailure,OSError)as error:
            context.emit('payload_authentication',dict(index=index,status='failed',reason=str(error)[:512]))
            failures+=1
            if wants_elf:
                context.emit('ELF_metadata',dict(index=index,status='unresolved',reason='Payload identity/read failed'))
                ELF_failures+=1
        else:
            row={k:observation[k]for k in ('identity','actual_sha256','selected_RECORD_match')}
            row.update(index=index,status='observed')
            context.emit('payload_authentication',row)
            failures+=not observation['selected_RECORD_match']
            if wants_elf:
                row=dict(index=index,status='observed',selected_RECORD_match=observation['selected_RECORD_match'],ELF=observation['ELF'])
                if observation['ELF_error']is not None or len(context.encode_event('ELF_metadata',row))>CAPS['output_row_bytes']:
                    row=dict(index=index,status='unresolved',reason=observation['ELF_error']or'ELF metadata exceeds original3072-byte event cap')
                    ELF_failures+=1
                context.emit('ELF_metadata',row)
        if path in conflicts:
            owner=conflicts[path]
            row=dict(index=index,selected=owner['selected_record'],ambient=owner['ambient_record'],
                     unselected_distribution=owner['unselected_distribution'],resolved_ownership=False)
            if observation is not None:
                row.update(actual_sha256=observation['actual_sha256'],identity=observation['identity'],
                    matches_selected=observation['selected_RECORD_match'],
                    matches_ambient=observation['actual_sha256']==owner['ambient_record']['sha256']
                    and observation['identity'][2]==owner['ambient_record']['bytes'])
            else:row['actual_observation_failed']=True
            context.emit('conflicting_owner',row)
        # Keep only identities/hashes, not any payload bytes, to recheck after scan.
        if observation is not None:observations[path]=observation['identity']
    for path,before in observations.items():
        context.check();require(provider.identity(path)==before,'Observed payload changed before complete receipt')
    context.check(full=True)
    return dict(schema='spark26_elf_preflight_result_v1',metadata_collection_complete=True,
        plan_sha256=provider.contract.digest(plan),payload_count=len(payloads),payload_bytes=manifest['payload_bytes'],
        payload_sha256=manifest['payload_sha256'],counts=COUNTS,authentication_failures=failures,
        ELF_metadata_failures=ELF_failures,actual_hash_read_requests_bytes=budget.hash_bytes,
        actual_ELF_read_requests_bytes=budget.elf_bytes,all_manifest_payloads_authenticated=failures==0,
        all_candidate_ELF_metadata_collected=ELF_failures==0,conflicting_owner_records_preserved=15,
        conflicts_resolved=False,dependency_closure_proven=False,source_candidate_runtime_admitted=False,
        native_imports_executed=False,model_checkpoint_reads=False,package_mutations=False,
        full_model_qualified=False,new_speedup_claim=False,worker_exit_or_outer_drain_proven=False)
