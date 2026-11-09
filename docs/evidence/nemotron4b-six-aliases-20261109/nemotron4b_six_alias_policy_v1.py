"""Concrete six.moves origin policy for the reviewed Python3/Linux alias table.

Source identity, exact known-module objects and CPython loader spec transitions
are checked. Real code/interpreter/target authentication remains mandatory
caller input; this module never imports six or a target on its own.
"""
import copy
import importlib.machinery as machinery
import sys
import types

from nemotron4b_import_router_v4 import PinnedSourceLoader, CheckedDelegatingLoader
from nemotron4b_module_origin_audit_v2 import metadata
from nemotron4b_process_handshake_v1 import require


def without_spec(meta):
    return None if meta is None else meta[:4]+meta[5:]


class SixAliasPolicy:
    def __init__(self, router, aliases, *, authenticate_six_code, authenticate_target):
        require(type(aliases)is dict and 0<len(aliases)<=128, 'Bounded exact six alias table')
        require(all(name.startswith('six.moves') for name in aliases), 'Original six.moves namespace')
        require(callable(authenticate_six_code)and callable(authenticate_target),
                'Actual source-code and target authentication callbacks required')
        self.router=router
        self.aliases=copy.deepcopy(aliases)
        self.authenticate_six_code=authenticate_six_code
        self.authenticate_target=authenticate_target
        self.module=None
        self.provider=None
        self.known=None
        self.objects=None
        self.selected_specs=[]
        self.target_origins={}
        self.events=[]

    def six_origin(self, module):
        row=self.router.modules.get('six')
        require(row is not None and row['kind']=='source','Pinned original six source record')
        v=vars(module);spec=v.get('__spec__');loader=v.get('__loader__')
        require(type(spec)is machinery.ModuleSpec and spec.name=='six'
                and spec.origin==row['path']and spec.loader is loader
                and type(loader)is PinnedSourceLoader and loader.router is self.router
                and loader.reference==row and loader.path==row['path'],
                'Exact authenticated six source loader')
        require(v.get('__name__')=='six'and v.get('__file__')==row['path']
                and v.get('__package__')=='', 'Original six source metadata')
        # six1.17 makes this source module a virtual package near end-of-body.
        require(('__path__'not in v or v['__path__']==[])
                and(spec.submodule_search_locations is None or spec.submodule_search_locations==[]),
                'Only original six empty virtual-package path transition')
        self.authenticate_six_code(module,row)

    def binding(self):
        module=sys.modules.get('six')
        require(isinstance(module,types.ModuleType),'Original loaded six module')
        self.six_origin(module)
        values=vars(module);provider=values.get('_importer')
        require(provider is not None and type(provider)is values.get('_SixMetaPathImporter')
                and provider.name=='six','Exact original six provider instance')
        known=provider.known_modules
        require(type(known)is dict and set(known)==set(self.aliases),'Complete original six alias registry')
        if self.module is None:
            self.module,self.provider,self.known=module,provider,known
            self.objects=dict(known)
        require(module is self.module and provider is self.provider and known is self.known,
                'Bound six/provider/registry identity unchanged')
        for name,row in self.aliases.items():
            obj=known[name]
            require(obj is self.objects[name],'Bound original alias entry unchanged')
            if row['kind']=='moved':
                require(type(obj)is values.get('MovedModule') and obj.name==row['name']
                        and obj.mod==row['target'],'Exact moved-module declaration')
            else:
                require(row['kind']=='synthetic'and type(obj)is values.get(row['class_name'])
                        and isinstance(obj,types.ModuleType)
                        and vars(obj).get('__name__')==row['module_name'],
                        'Exact synthetic six module class/object/name')
                require(all(known[n]is obj for n in row['aliases']),
                        'Shared synthetic aliases preserve object identity')
        return provider

    def selected_spec(self,spec,names):
        for alias,original,fields in self.selected_specs:
            if alias not in names or spec is not original:continue
            actual=(spec.name,spec.origin,spec.has_location,
                    None if spec.submodule_search_locations is None else tuple(spec.submodule_search_locations))
            require(actual==fields,'Selected alias spec fields unchanged')
            loader=spec.loader
            require(type(loader)is CheckedDelegatingLoader and loader.router is self.router
                    and loader.original is self.provider and loader.fullname==alias
                    and loader.kind=='custom','Selected alias retains checked original loader')
            return True
        return False

    def check_target(self,name,module):
        require(sys.modules.get(name)is module,'Actual canonical moved target object')
        saved=self.target_origins.get(name)
        if saved is None:
            # Bind target before CPython overwrites its spec for an alias.
            self.authenticate_target(name,module)
            self.target_origins[name]=(module,metadata(module))
            return
        require(saved[0]is module,'Canonical moved target identity unchanged')
        current=metadata(module)
        require(without_spec(current)==without_spec(saved[1]),
                'Moved target non-spec import metadata unchanged')
        if current==saved[1]:
            self.authenticate_target(name,module)
        else:
            spec=vars(module).get('__spec__')
            names=[alias for alias,row in self.aliases.items()
                   if row['kind']=='moved'and row['target']==name]
            require(self.selected_spec(spec,names),
                    'Target spec changed only to exact selected six alias spec')
            # Authenticate original code/native identity with saved original
            # metadata; callback must not trust the alias spec as a new origin.
            self.authenticate_target(name,module,copy.deepcopy(saved[1]))

    def alias(self,name,module):
        if name=='six':
            self.six_origin(module)
            return True
        if name in self.target_origins:
            self.binding()
            self.check_target(name,module)
            return True
        row=self.aliases.get(name)
        if row is None:return False
        provider=self.binding()
        require(sys.modules.get(name)is module,'Alias name maps to exact returned object')
        if row['kind']=='moved':
            self.check_target(row['target'],module)
        else:
            require(module is self.objects[name],'Synthetic alias is original provider object')
            spec=vars(module).get('__spec__')
            require(spec is None or self.selected_spec(spec,row['aliases']),
                    'Synthetic alias spec is absent or exact selected alias spec')
            require(vars(module).get('__loader__') in (None,provider)
                    or any(vars(module).get('__loader__')is original.loader
                           for n,original,fields in self.selected_specs if n in row['aliases']),
                    'Synthetic alias original loader')
        return True

    def preloaded_transition(self,name,module,original_metadata):
        saved=self.target_origins.get(name)
        if saved is None:return False
        require(saved[0]is module and saved[1]==original_metadata,
                'Preloaded transition agrees with authenticated original target')
        return self.alias(name,module)

    def authorize_loader(self,fullname,spec,finder):
        provider=self.binding()
        require(fullname in self.aliases and finder is provider and spec.loader is provider
                and spec.name==fullname, 'Exact original six finder/loader and requested alias')
        # Retain the actual spec object before router wraps its loader.
        require(len(self.selected_specs)<4096,'Bounded exact selected alias specs')
        fields=(spec.name,spec.origin,spec.has_location,
                None if spec.submodule_search_locations is None else tuple(spec.submodule_search_locations))
        self.selected_specs.append((fullname,spec,fields))
        def before(name,original,effective_spec):
            require(name==fullname and original is self.binding(),'Exact bound six lifecycle')
            require(effective_spec is spec,'Only selected six alias spec for lifecycle')
        def after(name,original,module,phase):
            require(name==fullname and original is self.binding(),'Exact bound six after lifecycle')
            require(phase in ('create','exec','create_failed','exec_failed'),'Known six lifecycle phase')
            if phase.endswith('_failed'):
                # A provider can fail after partially registering an alias.
                partial=sys.modules.get(name)
                if partial is not None:self.alias(name,partial)
            else:
                require(isinstance(module,types.ModuleType),'Original six returns a module object')
                self.alias(name,module)
            require(len(self.events)<4096,'Bounded six lifecycle events')
            self.events.append(dict(name=name,phase=phase))
        return before,after
