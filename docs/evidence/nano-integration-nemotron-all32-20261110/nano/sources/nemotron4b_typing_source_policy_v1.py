"""Bind observed typing namespace classes to pinned source/code definitions.

Requires an already checked private empty bytecode cache and an explicit policy
for native cache wrappers. This is code-definition and object-identity checking,
not arbitrary module-global/closure-content or loaded-memory authentication.
No SDK/native payload reads, site restoration, worker launch or admission.
"""
import copy
import importlib.machinery as machinery
import importlib.util as util
from pathlib import Path
import sys
import types

from nemotron4b_empty_bytecode_cache_v1 import EmptyBytecodeCache,typing_registry_aliases
from nemotron4b_interpreter_origin_policy_v1 import code_sha256
from nemotron4b_pinned_source_reader_v1 import source as read_source,identity,canonical
from nemotron4b_process_handshake_v1 import require


def source_definitions(raw,path):
    root=compile(raw,path,'exec',dont_inherit=True,optimize=0)
    definitions=set();stack=[root]
    while stack:
        code=stack.pop();definitions.add(code_sha256(code))
        stack.extend(x for x in code.co_consts if type(x)is types.CodeType)
    return definitions


def scan(owner,path,*,describe):
    functions={};classes={};external={};bindings={};held=[];visited=set()
    todo=[(key,value,0)for key,value in vars(owner).items()]
    while todo:
        name,value,depth=todo.pop()
        require(depth<=16 and len(visited)<=8192,'Bounded static typing code scan')
        if id(value)in visited:continue
        visited.add(id(value))
        if type(value)is types.FunctionType:
            if value.__module__!='typing'and value.__code__.co_filename!=path:continue
            held.extend([value,value.__code__,value.__defaults__,value.__kwdefaults__])
            closure=[]
            if value.__closure__:
                for index,cell in enumerate(value.__closure__):
                    try:child=cell.cell_contents
                    except ValueError:child=None
                    held.extend([cell,child]);closure.append((id(cell),id(child)))
                    if type(child)is types.FunctionType:todo.append((name+'.<closure'+str(index)+'>',child,depth+1))
            bindings['function:'+name]=(id(value),id(value.__code__),id(value.__globals__),
                id(value.__defaults__),id(value.__kwdefaults__),tuple(closure))
            if describe:
                functions[name]=dict(name=value.__name__,qualname=value.__qualname__,
                    module=value.__module__,filename=value.__code__.co_filename,
                    code_sha256=code_sha256(value.__code__),globals_are_typing=value.__globals__ is vars(owner))
            wrapped=vars(value).get('__wrapped__')
            if wrapped is not None:todo.append((name+'.__wrapped__',wrapped,depth+1))
        elif isinstance(value,type)and type.__getattribute__(value,'__module__')=='typing':
            namespace=type.__getattribute__(value,'__dict__')
            held.extend([value,type(value),*namespace.values()])
            bindings['class:'+name]=(id(value),id(type(value)),tuple(sorted((k,id(v))for k,v in namespace.items())))
            if describe:
                classes[name]=dict(name=type.__getattribute__(value,'__name__'),
                    qualname=type.__getattribute__(value,'__qualname__'),metaclass=type(value).__name__)
            for key,child in namespace.items():
                if type(child)in(classmethod,staticmethod):child=child.__func__
                if type(child)is property:
                    for part in('fget','fset','fdel'):
                        f=getattr(child,part)
                        if f is not None:todo.append((name+'.'+key+'.'+part,f,depth+1))
                elif type(child)is types.FunctionType or isinstance(child,type):
                    todo.append((name+'.'+key,child,depth+1))
        elif type(value).__module__=='functools'and type(value).__name__=='_lru_cache_wrapper':
            wrapped=vars(value).get('__wrapped__')
            if type(wrapped)is types.FunctionType and wrapped.__module__=='typing':
                held.extend([value,type(value),*vars(value).values()])
                bindings['external:'+name]=(id(value),id(type(value)),
                    tuple(sorted((k,id(v))for k,v in vars(value).items())))
                external[name]=value
                todo.append((name+'.__wrapped__',wrapped,depth+1))
    return dict(functions=functions,classes=classes,external=external,bindings=bindings,held=held)


class TypingSourcePolicy:
    def __init__(self,manifest,*,cache,guard,authenticate_external):
        require(type(cache)is EmptyBytecodeCache and all(callable(x)for x in(guard,authenticate_external)),
                'Actual cache guard and explicit external wrapper authentication')
        require(type(manifest)is dict and set(manifest)=={'version','source','functions','classes','external_wrappers'},
                'Reviewed complete typing source/code manifest')
        self.manifest=copy.deepcopy(manifest);self.cache=cache;self.guard=guard
        self.authenticate_external=authenticate_external;self.owner=sys.modules.get('typing')
        require(type(self.owner)is types.ModuleType,'Original loaded typing module')
        self.path=self.manifest['source']['path'];self.aliases=None;self.live=None;self.checks=0
        self.guard();self.cache.check_runtime()
        require(list(sys.version_info[:3])==self.manifest['version'],'Exact reviewed typing interpreter version')
        self.loader_metadata()
        require(self.cache.checked_sources is not None and any(
            x['name']=='typing'and x['source']==self.path for x in self.cache.checked_sources),
            'Actual typing source cache exclusion must already be checked')
        raw,_=read_source(self.path,self.manifest['source'])
        definitions=source_definitions(raw,self.path)
        observed=scan(self.owner,self.path,describe=True)
        require(observed['functions']==self.manifest['functions']
                and observed['classes']==self.manifest['classes']
                and sorted(observed['external'])==self.manifest['external_wrappers'],
                'Exact reviewed typing functions/classes/external boundaries')
        require(observed['functions']and all(
            x['filename']==self.path and x['globals_are_typing']and x['code_sha256']in definitions
            for x in observed['functions'].values()),'Live typing code definitions match pinned original source')
        self.aliases=typing_registry_aliases()
        self.live=observed
        self.check()

    def loader_metadata(self):
        values=vars(self.owner);spec=values.get('__spec__')
        require(sys.modules.get('typing')is self.owner and type(spec)is machinery.ModuleSpec
                and spec.name=='typing'and spec.origin==self.path and spec.has_location
                and spec.submodule_search_locations is None
                and type(spec.loader)is machinery.SourceFileLoader
                and spec.loader.name=='typing'and spec.loader.path==self.path
                and values.get('__loader__')is spec.loader and values.get('__file__')==self.path
                and values.get('__name__')=='typing'and values.get('__package__')=='',
                'Original preloaded typing source loader and metadata')
        expected=util.cache_from_source(self.path)
        require(values.get('__cached__')==spec.cached==expected
                and Path(expected).is_relative_to(self.cache.manifest['path'])
                and not Path(expected).exists(),'Typing cache excluded by actual private prefix')
        current=(id(spec),id(spec.loader),expected)
        if hasattr(self,'loader_binding'):
            require(current==self.loader_binding,'Original typing loader/spec object identity retained')
        else:self.loader_binding=current

    def check(self):
        self.guard();self.cache.check_runtime();self.loader_metadata()
        canonical(Path(self.path))
        require(identity(Path(self.path).lstat())==self.manifest['source']['fullstat'],
                'Original source identity retained after full initial hash')
        require(self.live is not None,'Typing source/code must be bound first')
        aliases=typing_registry_aliases()
        require(set(aliases)==set(self.aliases)and all(aliases[n]is self.aliases[n]for n in aliases),
                'Original typing class namespace objects retained')
        current=scan(self.owner,self.path,describe=False)
        require(current['bindings']==self.live['bindings'],'Original typing code and object bindings retained')
        require(set(current['external'])==set(self.live['external']),'Complete original external wrapper set')
        for name,value in current['external'].items():
            require(value is self.live['external'][name]and self.authenticate_external(name,value) is True,
                    'Actual external typing cache-wrapper origin authentication')
        self.checks+=1
        return True

    def authenticate_owner(self,name,module):
        require(name=='typing'and module is self.owner,'Exact typing owner callback')
        return self.check()

    def authenticate_namespace(self,name,value,owner):
        require(owner is self.owner and name in self.aliases and value is self.aliases[name],
                'Exact namespace and owning typing object')
        return self.check()

    def result(self):
        self.check()
        return dict(schema='nemotron4b_typing_source_policy_v1',checks=self.checks,
            source=copy.deepcopy(self.manifest['source']),functions_checked=len(self.manifest['functions']),
            classes_checked=len(self.manifest['classes']),external_wrappers=list(self.manifest['external_wrappers']),
            original_source_hash_and_code_definitions_checked=True,
            arbitrary_globals_or_mutable_closure_contents_authenticated=False,
            complete_runtime_dependency_closure_proven=False,runtime_admitted=False,new_speedup_claim=False)
