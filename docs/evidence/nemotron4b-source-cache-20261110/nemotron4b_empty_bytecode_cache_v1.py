"""Keep source-loader startup away from existing bytecode caches.

The parent supplies a precreated empty private directory and its exact identity.
-B plus a per-participant -X pycache_prefix makes ordinary source loaders miss
existing caches and prevents new cache writes. This module does not create that
directory, authenticate source/native code, admit work or launch a process.
"""
import copy
import importlib.machinery as machinery
import importlib.util as util
import os
from pathlib import Path
import stat
import sys
import types

from nemotron4b_process_handshake_v1 import require

FIELDS=('st_dev','st_ino','st_mode','st_uid','st_gid','st_nlink','st_size','st_mtime_ns','st_ctime_ns')


def directory(path):
    p=Path(path)
    require(p.is_absolute()and p.resolve()==p and not any(x.is_symlink()for x in(p,*p.parents)),
            'Canonical private bytecode-cache directory')
    s=p.lstat()
    require(stat.S_ISDIR(s.st_mode)and stat.S_IMODE(s.st_mode)==0o700 and s.st_uid==os.geteuid(),
            'Existing empty directory owned by current user with mode0700')
    with os.scandir(p)as entries:
        require(next(entries,None)is None,'Bytecode-cache directory must remain empty')
    return {k:getattr(s,k)for k in FIELDS}


def record_empty_directory(path,*,guard):
    require(callable(guard),'Actual caller resource/serial guard')
    guard()
    return dict(path=str(path),identity=directory(path))


def typing_registry_aliases():
    """Identify two observed stdlib namespace objects without granting provenance.

    Their owning typing module/source must still be authenticated by the caller's
    full origin policy. Other nonmodule entries are not allowed by this scan.
    """
    module=sys.modules.get('typing')
    if module is None:return {}
    require(type(module)is types.ModuleType,'Original typing module object')
    values=vars(module);version=sys.version_info[:2]
    require(version in((3,10),(3,12)),'Reviewed typing namespace versions only')
    metaclass=type if version==(3,10)else values.get('_DeprecatedType')
    require(type(metaclass)is type,'Original typing namespace metaclass')
    result={}
    for short,exports in(('io',('IO','TextIO','BinaryIO')),('re',('Pattern','Match'))):
        name='typing.'+short;value=values.get(short)
        require(type(value)is metaclass and sys.modules.get(name)is value,
                'Original typing namespace object and registry alias')
        namespace=vars(value)
        require(set(namespace)=={'__module__','__dict__','__weakref__','__doc__','__all__',*exports}
                and namespace['__module__']=='typing'
                and type.__getattribute__(value,'__name__')==name,
                'Original typing namespace class metadata')
        require(type(namespace['__all__'])is list and set(namespace['__all__'])==set(exports)
                and len(namespace['__all__'])==len(exports)
                and all(namespace[x]is values[x]for x in exports),
                'Original typing namespace exports')
        result[name]=value
    return result


class EmptyBytecodeCache:
    def __init__(self,manifest,*,guard):
        require(callable(guard)and type(manifest)is dict and set(manifest)=={'path','identity'},
                'Exact empty-cache identity and caller guard')
        require(type(manifest['path'])is str and type(manifest['identity'])is dict
                and set(manifest['identity'])==set(FIELDS),'Complete cache directory identity')
        self.manifest=copy.deepcopy(manifest);self.guard=guard;self.checked_sources=None;self.typing_aliases=[]
        self.check_directory()

    def check_directory(self):
        self.guard()
        require(directory(self.manifest['path'])==self.manifest['identity'],
                'Original empty cache directory identity retained')

    def prefix_argv(self,python,script):
        self.check_directory()
        require(all(type(x)is str and x and '\0'not in x and Path(x).is_absolute()
                    for x in(python,script)),'Original explicit Python and script paths')
        return [python,'-B','-S','-X','pycache_prefix='+self.manifest['path'],script]

    def check_runtime(self):
        self.check_directory()
        require(sys.pycache_prefix==self.manifest['path']
                and sys._xoptions.get('pycache_prefix')==self.manifest['path'],
                'Actual command-line private pycache prefix')
        require(sys.dont_write_bytecode and sys.flags.dont_write_bytecode==1
                and sys.flags.no_site==1 and sys.flags.no_user_site==1
                and sys.flags.optimize==0 and __debug__,'Original gated -B -S no-user-site unoptimized runtime')

    def audit_preloaded_source_caches(self):
        """Before the pinned-source dispatch context replaces source loaders.

        Checks source-loader cache paths only. Frozen/builtin/native modules and
        __main__ require their separate interpreter/native/script provenance.
        """
        self.check_runtime();rows=[];aliases=typing_registry_aliases()
        for name,module in tuple(sys.modules.items()):
            if module is None:continue
            if not isinstance(module,types.ModuleType):
                require(name in aliases and module is aliases[name],
                        'No unreviewed nonmodule registry entries')
                continue
            values=vars(module);spec=values.get('__spec__')
            if spec is None:continue
            origin=spec.origin
            if type(origin)is not str or not origin.endswith('.py'):continue
            require(type(spec)is machinery.ModuleSpec and spec.has_location
                    and type(spec.loader)is machinery.SourceFileLoader
                    and values.get('__loader__')is spec.loader
                    and values.get('__file__')==origin and spec.loader.path==origin,
                    'Original pre-context source loader and file')
            expected=util.cache_from_source(origin)
            require(values.get('__cached__')==spec.cached==expected
                    and Path(expected).is_relative_to(self.manifest['path'])
                    and not Path(expected).exists(),'Source cache points only into empty private prefix')
            rows.append(dict(name=name,source=origin,cached=expected))
        self.check_runtime();self.typing_aliases=sorted(aliases)
        self.checked_sources=sorted(rows,key=lambda x:x['name'])
        return copy.deepcopy(self.checked_sources)

    def result(self):
        self.check_runtime()
        return dict(schema='nemotron4b_empty_bytecode_cache_v1',
            directory=copy.deepcopy(self.manifest),
            preloaded_source_cache_paths_checked=self.checked_sources is not None,
            checked_source_modules=0 if self.checked_sources is None else len(self.checked_sources),
            typing_namespace_aliases=list(self.typing_aliases),typing_source_provenance_authenticated=False,
            original_source_or_native_code_authenticated=False,
            full_runtime_dependency_closure_proven=False,runtime_admitted=False,new_speedup_claim=False)
