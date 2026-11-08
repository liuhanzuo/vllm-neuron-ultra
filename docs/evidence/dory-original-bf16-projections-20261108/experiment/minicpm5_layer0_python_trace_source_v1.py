"""Bounded authenticated source text for original88; no payload reader reuse."""
import hashlib
from pathlib import Path
from minicpm5_layer0_runtime_auth_v1 import require,identity


def source_text(context,module,kind):
    require(kind in('native','cpu'),'Exact original context kind')
    context.guard(kind);require(context._sources_checked,'Original complete source closure required')
    expected=context.plan['native_provenance']['source']if kind=='native'else context.plan['sdk_attention_source']
    path=Path(expected)
    require(path.is_absolute()and path.resolve()==path and path.suffix=='.py'and module.__file__==str(path),
            'Exact canonical original module source')
    require(not any(item.is_symlink()for item in(path,*path.parents)),'No source alias')
    wanted=context.plan['source_sha256'].get(str(path))
    require(type(wanted)is str and len(wanted)==64,'Exact admitted source digest')
    before=identity(path);require(0<before['size']<=8*2**20,'Bounded source text')
    raw=path.read_bytes()
    require(len(raw)==before['size']and identity(path)==before and hashlib.sha256(raw).hexdigest()==wanted,
            'Source identity and bytes authenticated before/after')
    context.guard(kind)
    return dict(path=str(path),sha256=wanted,text=raw.decode('utf-8'))


def metadata_source(plan,kind,guard):
    """Independent postrun reader; exact source pins, no runtime permit required."""
    require(kind in('native','cpu'),'Exact source kind');guard()
    name=plan['native_provenance']['source']if kind=='native'else plan['sdk_attention_source']
    path=Path(name);require(path.is_absolute()and path.resolve()==path and path.suffix=='.py'and
        not any(item.is_symlink()for item in(path,*path.parents)),'Exact independent source path')
    wanted=plan['source_sha256'][name];before=identity(path)
    require(0<before['size']<=8*2**20,'Bounded independent source')
    raw=path.read_bytes()
    require(len(raw)==before['size']and identity(path)==before and hashlib.sha256(raw).hexdigest()==wanted,'Independent source identity/hash')
    guard();return dict(path=name,sha256=wanted,text=raw.decode('utf-8'))
