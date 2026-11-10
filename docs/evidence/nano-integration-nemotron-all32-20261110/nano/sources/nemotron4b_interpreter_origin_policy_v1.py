"""Bind interpreter import providers to an already authenticated native catalog.

This policy does not hash native files or admit work. The serial caller must
first authenticate NativeFiles, supply the exact process mapping audit and a
reviewed runtime/module manifest. Disk site.py never authenticates frozen code.
These disk/mapping/code-definition checks are not arbitrary live-memory or
Python-global attestation, nor a complete preloaded or SDK dependency policy.
"""
import copy
import hashlib
import importlib.machinery as machinery
import _imp
import json
import os
from pathlib import Path
import struct
import sys
import types

from nemotron4b_native_origin_audit_v1 import NativeMappingAudit,fullstat
from nemotron4b_process_handshake_v1 import require

PROVIDERS=('BuiltinImporter','FrozenImporter','PathFinder')
PRIMITIVES=('is_builtin','is_frozen','is_frozen_package','get_frozen_object')


def code_sha256(code):
    """Hash complete public code fields, independent of marshal reference flags.

    Nested frozensets sort by canonical encoded values, so process hash seeds
    cannot change a frozen-code pin. CPython minor versions have distinct fields.
    """
    nodes=0;size=0
    def encode(value):
        nonlocal nodes,size
        nodes+=1;require(nodes<=100000,'Bounded frozen code structure')
        t=type(value)
        if t is types.CodeType:
            fields=('co_argcount','co_posonlyargcount','co_kwonlyargcount','co_nlocals',
                'co_stacksize','co_flags','co_code','co_consts','co_names','co_varnames',
                'co_filename','co_name','co_firstlineno','co_freevars','co_cellvars')
            if not hasattr(value,'co_linetable'):fields+=('co_lnotab',)
            fields+=tuple(k for k in('co_qualname','co_linetable','co_exceptiontable')if hasattr(value,k))
            return ['code',[[k,encode(getattr(value,k))]for k in fields]]
        if value is None:return ['none']
        if value is Ellipsis:return ['ellipsis']
        if t in(bool,int):return [t.__name__,str(value)]
        if t is float:return ['float',struct.pack('!d',value).hex()]
        if t is complex:return ['complex',struct.pack('!dd',value.real,value.imag).hex()]
        if t in(str,bytes):
            size+=len(value);require(size<=4*2**20,'Bounded frozen code strings')
            return [t.__name__,value if t is str else value.hex()]
        if t in(tuple,frozenset):
            values=[encode(x)for x in value]
            if t is frozenset:values.sort(key=lambda x:json.dumps(x,separators=(',',':'),ensure_ascii=True))
            return [t.__name__,values]
        raise ValueError('Unexpected frozen code constant: '+t.__name__)
    require(type(code)is types.CodeType,'Actual code object required')
    raw=json.dumps(encode(code),separators=(',',':'),ensure_ascii=True).encode()
    require(len(raw)<=16*2**20,'Bounded canonical frozen code record')
    return hashlib.sha256(raw).hexdigest()


def function_state(value,*,describe=False,depth=0):
    """Include decorator closures/defaults; arbitrary function globals excluded."""
    require(depth<=16,'Bounded provider function closure')
    def child(x):return function_state(x,describe=describe,depth=depth+1)
    t=type(value)
    if value is None or t in(str,int,bool):return value
    if t in(tuple,list):return [t.__name__,[child(x)for x in value]]
    if t is dict:
        require(all(type(k)is str for k in value),'Provider default string keys')
        return ['dict',[[k,child(value[k])]for k in sorted(value)]]
    if t is types.FunctionType:
        return ['function',value.__module__,value.__qualname__,
            code_sha256(value.__code__)if describe else(value,id(value),value.__code__,id(value.__code__)),
            child(value.__defaults__),child(value.__kwdefaults__),
            None if value.__closure__ is None else[child(c.cell_contents)for c in value.__closure__]]
    if t is type and value in tuple(getattr(machinery,k)for k in PROVIDERS):
        return ['provider',value.__module__,value.__qualname__]if describe else['provider',value]
    if t is types.BuiltinFunctionType:
        return ['builtin',value.__module__,value.__name__]if describe else['builtin',value]
    raise ValueError('Unexpected provider closure/default value: '+t.__name__)


def provider_bindings():
    """Exact original descriptor/code objects, not bound-method temporaries."""
    bound={};described={}
    for name in PROVIDERS:
        provider=getattr(machinery,name);bound[name]=provider;methods={}
        for key,value in vars(provider).items():
            function=value.__func__ if type(value)in(classmethod,staticmethod)else value
            if type(function)is types.FunctionType:
                bound[name+'.'+key]=(value,function,function_state(function))
                methods[key]=dict(descriptor=type(value).__name__,function=function_state(function,describe=True))
        described[name]=methods
    for name in PRIMITIVES:
        value=getattr(_imp,name)
        require(type(value)is types.BuiltinFunctionType and value.__module__=='_imp',
                'Original interpreter primitive type')
        bound['_imp.'+name]=value
    return bound,described


def running_executable():
    """Only proc metadata, never opens interpreter payload bytes."""
    return dict(path=os.readlink('/proc/self/exe'),fullstat=fullstat(os.stat('/proc/self/exe')))


def runtime_description():
    return dict(executable=sys.executable,resolved_executable=str(Path(sys.executable).resolve()),
        version=sys.version,version_info=list(sys.version_info),implementation=sys.implementation.name,
        cache_tag=sys.implementation.cache_tag,builtin_module_names=list(sys.builtin_module_names),
        optimize=sys.flags.optimize,dont_write_bytecode=sys.dont_write_bytecode)


def spec_description(spec):
    require(type(spec)is machinery.ModuleSpec,'Exact interpreter spec object')
    state=spec.loader_state
    require(state is None or type(state)is types.SimpleNamespace,'Original frozen loader state type')
    if state is not None:
        require(set(vars(state))=={'filename','origname'}and
                all(x is None or type(x)is str for x in vars(state).values()),'Original frozen loader state fields')
    return dict(name=spec.name,origin=spec.origin,has_location=spec.has_location,cached=spec.cached,
        locations=None if spec.submodule_search_locations is None else list(spec.submodule_search_locations),
        loader_state=None if state is None else dict(vars(state)))


class InterpreterPolicy:
    def __init__(self,manifest,*,mapping_audit,guard):
        require(type(mapping_audit)is NativeMappingAudit and callable(guard),
                'Actual native mapping policy and serial/resource guard required')
        require(type(manifest)is dict and set(manifest)=={'runtime','executable','providers','modules'},
                'Exact reviewed interpreter manifest')
        self.manifest=copy.deepcopy(manifest);self.native=mapping_audit;self.guard=guard
        self.selected={};self.module_objects={};self.events=[];self.providers=None
        self.check_runtime()
        bound,described=provider_bindings()
        require(described==self.manifest['providers'],'Original interpreter provider code definitions')
        self.providers=bound
        require(type(self.manifest['modules'])is dict and 0<len(self.manifest['modules'])<=1024,
                'Explicit bounded interpreter module policy')
        for name,row in self.manifest['modules'].items():
            require(set(row)=={'spec','module_name','module_package','module_file','code_sha256'}
                    and row['spec']['name']==name and row['spec']['origin']in('built-in','frozen'),
                    'Exact interpreter module metadata and code pin')
            digest=row['code_sha256']
            require((row['spec']['origin']=='built-in'and digest is None)or
                    (row['spec']['origin']=='frozen'and type(digest)is str and len(digest)==64
                     and all(c in'0123456789abcdef'for c in digest)), 'Frozen code requires its own pin')

    def check_runtime(self):
        self.guard()
        require(runtime_description()==self.manifest['runtime'],'Exact original interpreter runtime')
        actual=running_executable();expected=self.manifest['executable']
        require(actual==expected and actual['path']==self.manifest['runtime']['resolved_executable'],
                'Actual executing interpreter, not just a path spelling')
        require(self.native.files.identity(actual['path'])==actual['fullstat'],
                'Interpreter belongs to original authenticated native catalog')
        snapshot=self.native.audit()
        require(actual['path']in snapshot['paths'],'Actual interpreter executable mapping required')

    def check_providers(self):
        require(self.providers is not None,'Interpreter providers already bound')
        for name in PROVIDERS:
            provider=getattr(machinery,name)
            require(provider is self.providers[name],'Original interpreter finder object retained')
            for key in self.manifest['providers'][name]:
                old,function,state=self.providers[name+'.'+key]
                actual=vars(provider).get(key)
                require(actual is old and function_state(function)==state,
                        'Original interpreter provider implementation retained')
        for name in PRIMITIVES:
            require(getattr(_imp,name)is self.providers['_imp.'+name],'Original interpreter primitive retained')

    def check(self):self.check_runtime();self.check_providers()

    def authorize_finder(self,finder):
        self.check()
        require(any(finder is self.providers[name]for name in PROVIDERS),'Exact original interpreter finder')

    def _spec(self,name,spec):
        require(name in self.manifest['modules'],'Interpreter module outside reviewed manifest')
        row=self.manifest['modules'][name];kind=row['spec']['origin']
        loader=self.providers['BuiltinImporter'if kind=='built-in'else'FrozenImporter']
        require(spec.loader is loader and spec_description(spec)==row['spec'],
                'Original interpreter module spec and loader state')
        if kind=='built-in':
            require(name in sys.builtin_module_names and _imp.is_builtin(name),'Actual interpreter builtin name')
        else:
            require(_imp.is_frozen(name)and _imp.is_frozen_package(name)==(row['spec']['locations']is not None),
                    'Actual frozen module and package kind')
            require(code_sha256(_imp.get_frozen_object(name))==row['code_sha256'],
                    'Original frozen code definition unchanged')
        return row

    def authorize_builtin(self,name,spec,finder):
        self.check();row=self._spec(name,spec)
        require(finder is spec.loader,'Interpreter module selected by original provider')
        require(len(self.selected)<4096 and len(self.events)<4096,'Bounded selected interpreter specs')
        self.selected[(name,id(spec))]=spec
        self.events.append(dict(name=name,phase='selected',origin=row['spec']['origin']))

    def authenticate_module(self,name,module,policy=None):
        self.check();require(type(module)is types.ModuleType,'Actual interpreter module object')
        values=vars(module);spec=values.get('__spec__');row=self._spec(name,spec)
        require(self.selected.get((name,id(spec)))is spec,'Interpreter module requires selected or preloaded binding')
        if policy is not None:require(policy==dict(origin=row['spec']['origin']),'Exact ledger interpreter policy')
        require(values.get('__name__')==row['module_name']and values.get('__package__')==row['module_package']
                and values.get('__file__')==row['module_file']and values.get('__loader__')is spec.loader
                and (None if '__path__'not in values else list(values['__path__']))==row['spec']['locations'],
                'Original interpreter module identity and locations')
        prior=self.module_objects.get((name,id(spec)))
        require(prior is None or prior is module,'Selected interpreter module cannot be replaced')
        self.module_objects[(name,id(spec))]=module

    def bind_preloaded(self,name,module):
        self.check();require(type(module)is types.ModuleType,'Actual preloaded interpreter module')
        spec=vars(module).get('__spec__');self._spec(name,spec)
        require(name not in self.module_objects and len(self.selected)<4096,'One bounded preloaded binding')
        self.selected[(name,id(spec))]=spec
        self.authenticate_module(name,module)
        self.module_objects[name]=module

    def authenticate_frozen_site(self,spec,module):
        require(spec.name=='site'and spec.origin=='frozen','Only original frozen site callback')
        if module is None:self.authorize_builtin('site',spec,self.providers['FrozenImporter'])
        else:self.authenticate_module('site',module)

    def ledger_modules(self):
        return {name:dict(origin=row['spec']['origin'])for name,row in self.manifest['modules'].items()}

    def result(self):
        self.check()
        return dict(schema='nemotron4b_interpreter_origin_policy_v1',selected_specs=len(self.selected),
            bound_modules=len([key for key in self.module_objects if type(key)is tuple]),
            events=copy.deepcopy(self.events),complete_runtime_dependency_closure_proven=False,
            arbitrary_live_memory_authenticated=False,runtime_admitted=False,new_speedup_claim=False)


class InterpreterAliases:
    """Bind explicitly reviewed existing aliases to authenticated canonical objects.

    Actual candidates record importlib._bootstrap / _external and Python3.12
    os.path aliases. This does not authorize six, distutils or arbitrary aliases.
    """
    def __init__(self,interpreter,rows):
        require(type(interpreter)is InterpreterPolicy and type(rows)is list and len(rows)<=16,
                'Exact interpreter policy and bounded reviewed alias rows')
        self.interpreter=interpreter;self.aliases={};interpreter.check()
        for row in rows:
            require(type(row)is dict and set(row)=={'registry_name','spec_name','module_name'},
                    'Exact reviewed interpreter alias fields')
            name,target=row['registry_name'],row['spec_name']
            require(type(name)is str and type(target)is str and name!=target and name not in self.aliases,
                    'Distinct unique explicit interpreter alias')
            module=interpreter.module_objects.get(target)
            require(type(module)is types.ModuleType and sys.modules.get(target)is module
                    and sys.modules.get(name)is module and vars(module).get('__name__')==row['module_name'],
                    'Original alias must reference already bound canonical interpreter module')
            interpreter.authenticate_module(target,module)
            self.aliases[name]=(target,module)

    def authenticate_alias(self,name,module):
        if name not in self.aliases:return False
        target,original=self.aliases[name]
        require(module is original and sys.modules.get(name)is original and sys.modules.get(target)is original,
                'Original interpreter alias and canonical object retained')
        self.interpreter.authenticate_module(target,original)
        return True
