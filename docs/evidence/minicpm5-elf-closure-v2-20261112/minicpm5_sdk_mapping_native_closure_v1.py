"""Pure observer ELF/maps closure checks; bytes supplied, never files loaded.

ELF parser preserves the accepted Dory parser's semantics with an explicit
64 MiB per-object cap for the original 31 MiB interpreter. Maps parsing follows
the accepted Nemotron source. Positive FFI function definitions are mandatory;
absence of a libffi dependency or mapping is never treated as proof.
"""
import re
import struct
from pathlib import Path
MAX_FILE=64*2**20
MAX_NEEDED=64
MAP_CAP=8*2**20
STRING_TAGS={1:'NEEDED',14:'SONAME',15:'RPATH',29:'RUNPATH',0x6FFFFEFA:'CONFIG',
             0x6FFFFEFB:'DEPAUDIT',0x6FFFFEFC:'AUDIT',0x7FFFFFFD:'AUXILIARY',0x7FFFFFFF:'FILTER'}
FFI_FUNCTIONS=frozenset(('ffi_call','ffi_prep_cif','ffi_prep_cif_var','ffi_closure_alloc',
                       'ffi_closure_free','ffi_prep_closure_loc'))
def require(ok,message):
    if not ok:raise ValueError(message)

def normalized(strings, interpreter, flags):
    result = dict(needed=[], soname=None, search_tags=[], interpreter=interpreter,
                  extra_loader_directives=[], flags=flags)
    for tag, value in strings:
        require(type(value) is str and value.isascii() and "\0" not in value
                and len(value) <= 4096, "Bounded ASCII loader string")
        if tag == "NEEDED":
            require(value and "/" not in value and len(result["needed"]) < MAX_NEEDED,
                    "Simple finite NEEDED name; path-valued needs require a new review")
            require(value not in result["needed"], "Unique NEEDED entries")
            result["needed"].append(value)
        elif tag == "SONAME":
            require(result["soname"] is None and value and "/" not in value, "One simple SONAME")
            result["soname"] = value
        elif tag in ("RPATH", "RUNPATH"):
            require(not any(row["tag"] == tag for row in result["search_tags"]), "Unique path tag")
            result["search_tags"].append(dict(tag=tag, value=value))
        else:
            result["extra_loader_directives"].append(dict(tag=tag, value=value))
    require(interpreter is None or interpreter.startswith("/") and len(interpreter) <= 4096,
            "Absolute finite interpreter path")
    return result

def parse_elf64(raw):
    """Read an in-memory ELF64 little-endian x86-64 object, never execute it.

    Program headers map virtual DT_STRTAB to file bytes. Section headers, symbol
    lookup, relocation processing and ELF constructors are deliberately unused.
    Unsupported/malformed forms fail closed instead of guessing dependencies.
    """
    require(type(raw) is bytes and 64 <= len(raw) <= MAX_FILE, "Finite ELF payload")
    require(raw[:7] == b"\x7fELF\x02\x01\x01", "ELF64 little-endian current version")
    header = struct.unpack_from("<HHIQQQIHHHHHH", raw, 16)
    kind, machine, version, _, phoff, _, _, ehsize, phsize, phnum, _, _, _ = header
    require(kind in (2, 3) and machine == 62 and version == 1 and ehsize == 64,
            "Only ordinary x86-64 EXEC/DYN ELF")
    require(phsize == 56 and 1 <= phnum <= 128 and phoff >= 64
            and phoff + phsize * phnum <= len(raw), "Bounded complete program headers")
    loads = []
    dynamic = []
    interpreter = []
    for i in range(phnum):
        typ, _, offset, vaddr, _, filesz, memsz, _ = struct.unpack_from("<IIQQQQQQ", raw, phoff + i * 56)
        require(offset <= len(raw) and filesz <= len(raw) - offset, "Segment file range in bounds")
        if typ == 1:
            require(filesz <= memsz, "LOAD file bytes fit memory range")
            loads.append((vaddr, offset, filesz))
        elif typ == 2:
            dynamic.append((offset, filesz))
        elif typ == 3:
            require(1 < filesz <= 4096, "Bounded INTERP")
            value = raw[offset:offset + filesz]
            require(value.endswith(b"\0") and b"\0" not in value[:-1], "Complete INTERP string")
            interpreter.append(value[:-1].decode("ascii"))
    require(loads and len(dynamic) == 1 and len(interpreter) <= 1, "One dynamic table and optional interpreter")
    offset, size = dynamic[0]
    require(16 <= size <= 65536 and size % 16 == 0, "Bounded aligned dynamic table")
    values = {}
    string_offsets = []
    ended = False
    for at in range(offset, offset + size, 16):
        tag, value = struct.unpack_from("<qQ", raw, at)
        if tag == 0:
            ended = True
            break
        if tag in STRING_TAGS:
            string_offsets.append((STRING_TAGS[tag], value))
        elif tag in (5, 10, 30, 0x6FFFFFFB):
            require(tag not in values, "Unique string-table/flags tag")
            values[tag] = value
    require(ended and 5 in values and 10 in values, "Terminated table with explicit STRTAB/STRSZ")
    address, length = values[5], values[10]
    require(1 <= length <= 2**20, "Bounded dynamic string table")
    matches = [base + address - start for start, base, count in loads
               if start <= address and address + length <= start + count]
    require(len(matches) == 1, "STRTAB maps uniquely to file-backed LOAD bytes")
    strings = raw[matches[0]:matches[0] + length]
    decoded = []
    for name, pos in string_offsets:
        require(pos < length, "String offset in STRSZ")
        end = strings.find(b"\0", pos, min(length, pos + 4097))
        require(end >= 0, "Bounded terminated dynamic string")
        decoded.append((name, strings[pos:end].decode("ascii")))
    return normalized(decoded, interpreter[0] if interpreter else None,
                      {"DT_FLAGS": values.get(30), "DT_FLAGS_1": values.get(0x6FFFFFFB)})

def parse_maps(text):
    require(type(text)is str and len(text.encode('utf-8'))<=MAP_CAP, 'Bounded complete process maps text')
    rows=[];previous_end=0
    lines=text.split('\n')
    if lines[-1]=='':lines.pop()
    for line in lines:
        fields=line.split(None,5)
        require(len(fields)>=5,'Complete process map fields')
        address,permissions,offset,device,inode=fields[:5]
        require(re.fullmatch('[0-9a-f]+-[0-9a-f]+',address)and
                re.fullmatch('[r-][w-][x-][ps]',permissions)and
                re.fullmatch('[0-9a-f]+',offset)and re.fullmatch('[0-9a-f]+:[0-9a-f]+',device)
                and inode.isdecimal(),'Original Linux process map syntax')
        begin,end=(int(value,16)for value in address.split('-'))
        require(0<=previous_end<=begin<end,'Ordered nonoverlapping mappings')
        previous_end=end
        path=fields[5]if len(fields)==6 else ''
        # Linux renders both a newline and a literal backslash-012 filename
        # identically. Never guess which disk path a mapping came from.
        require('\\012'not in path and '\x00'not in path,
                'Unambiguous process map pathname')
        row=dict(begin=begin,end=end,permissions=permissions,offset=int(offset,16),
                 device=[int(value,16)for value in device.split(':')],inode=int(inode),path=path)
        require(len(rows)<32768,'Bounded mapping row count')
        rows.append(row)
    require(rows,'Nonempty actual process maps')
    return rows

def ffi_definitions(raw):
    """Positive complete function-symbol definitions inside authenticated ELF.

    Stripped/hidden symbols that cannot supply this proof fail closed. This
    routine does not invent a static-link conclusion from missing dependencies.
    """
    parse_elf64(raw)
    header=struct.unpack_from('<HHIQQQIHHHHHH',raw,16)
    phoff,shoff,phnum,shsize,shnum=header[4],header[5],header[9],header[10],header[11]
    require(shsize==64 and 1<=shnum<=8192 and shoff>=64 and shoff+64*shnum<=len(raw),
            'Complete bounded ELF section table required for positive FFI proof')
    sections=[struct.unpack_from('<IIQQQQIIQQ',raw,shoff+64*i) for i in range(shnum)]
    executable=[]
    for i in range(phnum):
        typ,flags,offset,vaddr,_,filesz,memsz,_=struct.unpack_from('<IIQQQQQQ',raw,phoff+56*i)
        if typ==1 and flags&1:executable.append((vaddr,vaddr+memsz))
    found={};undefined=set();tables=0
    for section in sections:
        _,kind,_,_,offset,size,linked,_,_,entry_size=section
        if kind not in (2,11):continue
        require(entry_size==24 and size%24==0 and size//24<=262144
            and offset+size<=len(raw) and linked<shnum,'Complete finite symbol table')
        strings=sections[linked]
        require(strings[1]==3 and strings[4]+strings[5]<=len(raw) and strings[5]<=32*2**20,
                'Complete linked symbol string table')
        string_bytes=raw[strings[4]:strings[4]+strings[5]];tables+=1
        for at in range(offset,offset+size,24):
            name_at,info,other,index,value,length=struct.unpack_from('<IBBHQQ',raw,at)
            require(name_at<len(string_bytes),'Symbol name offset in complete string table')
            end=string_bytes.find(b'\0',name_at,min(len(string_bytes),name_at+4097))
            require(end>=0,'Bounded complete symbol name')
            name=string_bytes[name_at:end].decode('ascii')
            if not name.startswith('ffi_'):continue
            if index==0:undefined.add(name);continue
            if name not in FFI_FUNCTIONS:continue
            require(index<shnum and info&15 in (2,10) and value>0 and length>0
                and any(low<=value and value+length<=high for low,high in executable),
                'FFI definition is a real function in executable LOAD memory')
            row=dict(value=value,bytes=length,type=info&15,section=index)
            require(name not in found or found[name]==row,'Unambiguous FFI function definition')
            found[name]=row
    require(tables>0 and not undefined and set(found)==FFI_FUNCTIONS,
            'Positive definitions of every required FFI function; no unresolved external FFI symbol')
    return dict(defined_functions=found,undefined_ffi_symbols=sorted(undefined),symbol_tables=tables,
                positive_embedded_ffi_definitions=True)


def mapped_closure(metadata,identities,maps_text,executable,loader,resolve_path):
    """Bind needed SONAMEs to unique actual mapped authenticated files.

    Only the original interpreter and explicit libc6 objects are admitted.
    The current mapping selection matters; copied-path RPATH candidates do not.
    """
    require(set(metadata)==set(identities) and executable in metadata and loader in metadata,
            'Same complete authenticated ELF and identity catalog')
    rows=parse_maps(maps_text);selected={};kernel=[]
    for row in rows:
        path=row['path']
        if path.startswith('/') and ('x' in row['permissions'] or re.search(r'\.so(?:\.|$)',Path(path.removesuffix(' (deleted)')).name)):
            require(not path.endswith(' (deleted)') and path in identities,'Every mapped native provider authenticated')
            identity=identities[path]
            require(row['device']==identity['device'] and row['inode']==identity['inode'],
                    'Actual mapped provider device/inode equals full-byte-authenticated file')
            selected.setdefault(path,[]).append(row)
        elif 'x' in row['permissions']:
            require(path in ('[vdso]','[vsyscall]') and row['device']==[0,0] and row['inode']==0,
                    'No foreign anonymous executable mapping')
            kernel.append(row)
    require(set(selected)==set(metadata),'Every expected native provider actually mapped, no inferred candidates')
    sonames={}
    for path,obj in metadata.items():
        require(not obj['extra_loader_directives'],'No unsupported AUDIT/FILTER/AUXILIARY directive')
        if obj['soname'] is not None:
            require(obj['soname'] not in sonames,'Unique actual loaded SONAME')
            sonames[obj['soname']]=path
    edges=[]
    for path,obj in metadata.items():
        for name in obj['needed']:
            require(name in sonames,'Every ELF dependency has a unique authenticated actual provider')
            edges.append(dict(requester=path,needed=name,provider=sonames[name]))
        if obj['interpreter'] is not None:
            require(path==executable and obj['interpreter']=='/lib64/ld-linux-x86-64.so.2'
                and resolve_path(obj['interpreter'])==loader,'Original-path actual ELF interpreter selection')
    require(metadata[executable]['interpreter'] is not None,'Original executable PT_INTERP is explicit')
    return dict(paths=sorted(selected),edges=edges,map_rows=len(rows),kernel_executable_rows=kernel,
                selected_mappings=selected,actual_original_path_selection=True)
