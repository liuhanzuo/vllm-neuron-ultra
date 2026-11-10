"""Compose Nano4B startup origin checks without creating runtime admission.

Caller owns serial/resource admission, complete source/native manifests and
the remaining preloaded/custom-target authentication. NativeFiles must already
be fully authenticated. This connects the actual selected policy classes; it
does not execute site, start a worker, authenticate omitted dependencies or
claim arbitrary Python-state/mapped-memory attestation.
"""
from contextlib import contextmanager
import copy
import sys
import types

from nemotron4b_empty_bytecode_cache_v1 import EmptyBytecodeCache
from nemotron4b_interpreter_origin_policy_v1 import InterpreterPolicy,InterpreterAliases
from nemotron4b_native_origin_audit_v1 import NativeMappingAudit
from nemotron4b_typing_source_policy_v1 import TypingSourcePolicy
from nemotron4b_bootstrap_dispatch_origin_v1 import BootstrapDispatchPolicy
from nemotron4b_import_router_v4 import ImportRouter,PinnedSourceLoader
from nemotron4b_import_context_v3 import DispatchContext,SdkFinderBindings
from nemotron4b_module_origin_audit_v4 import ModuleOriginAudit
from nemotron4b_six_alias_policy_v1 import SixAliasPolicy
from nemotron4b_process_handshake_v1 import require
from nemotron4b_phase_site_restore_v2 import restore as restore_phase


class PhaseOriginContext:
    def __init__(self,*,cache,interpreter,mapping,typing_policy,
                 dispatch_manifest,interpreter_aliases,modules,namespaces,
                 six_aliases,guard,authenticate_other_preloaded,audit_preloaded,
                 authenticate_source_code,authenticate_alias_target,audit_environment):
        require(type(cache)is EmptyBytecodeCache and type(interpreter)is InterpreterPolicy
                and type(mapping)is NativeMappingAudit and type(typing_policy)is TypingSourcePolicy,
                'Exact selected cache/interpreter/native/typing policies')
        require(interpreter.native is mapping and typing_policy.cache is cache,
                'One shared native catalog and cache context')
        callbacks=(guard,authenticate_other_preloaded,audit_preloaded,
                   authenticate_source_code,authenticate_alias_target,audit_environment)
        require(all(callable(x)for x in callbacks),'Explicit complete remaining origin/resource callbacks')
        self.cache,self.interpreter,self.mapping,self.typing=cache,interpreter,mapping,typing_policy
        self.guard=guard;self.other_preloaded=authenticate_other_preloaded
        self.preloaded_audit=audit_preloaded;self.source_code=authenticate_source_code
        self.alias_target=authenticate_alias_target;self.environment=audit_environment
        self.used=False;self.ready=False;self.imports=None;self.ledger=None
        self.preloaded=dict(sys.modules);self.finders=sys.meta_path
        require(type(self.finders)is list and self.finders,'Original live finder list')
        guard();cache.check_runtime();mapping.files.require_complete()
        interpreter.check();typing_policy.check()
        for name in interpreter.manifest['modules']:
            module=self.preloaded.get(name)
            if module is None:continue
            require(type(module)is types.ModuleType,'Original canonical interpreter module')
            if name in interpreter.module_objects:
                require(interpreter.module_objects[name]is module,'Previously bound original interpreter object')
                interpreter.authenticate_module(name,module)
            else:interpreter.bind_preloaded(name,module)
        self.interpreter_aliases=InterpreterAliases(interpreter,interpreter_aliases)
        self.bootstrap=BootstrapDispatchPolicy(interpreter,dispatch_manifest)
        self.finder_policy=SdkFinderBindings(authenticate_module=self.authenticate_registered,
            authenticate_interpreter_finder=interpreter.authorize_finder)
        self.router=ImportRouter(finders=self.finders,modules=modules,namespaces=namespaces,
            guard=guard,authorize_finder=self.finder_policy.authorize,
            authorize_builtin=interpreter.authorize_builtin,
            authorize_native=mapping.authorize_native,authorize_custom=self.authorize_custom)
        require(type(six_aliases)is dict,'Explicit reviewed six alias catalog, empty when unused')
        self.six=(SixAliasPolicy(self.router,six_aliases,
            authenticate_six_code=self.authenticate_six_code,
            authenticate_target=self.authenticate_alias_target)if six_aliases else None)
        self.ledger=ModuleOriginAudit(self.router,preloaded=self.preloaded,
            authenticate_preloaded=self.authenticate_preloaded,
            interpreter_modules=interpreter.ledger_modules(),
            authenticate_interpreter=interpreter.authenticate_module,
            authenticate_native=self.authenticate_native,authenticate_alias=self.authenticate_alias,
            authenticate_preloaded_transition=self.preloaded_transition,
            authenticate_typing_namespace=typing_policy.authenticate_namespace)
        self.router.authenticate(self.ledger.bind_preloaded)
        self.scope=DispatchContext(self.router,
            authenticate_bootstrap=self.bootstrap.authenticate_bootstrap,
            audit_modules=self.ledger.audit,audit_environment=self.audit_environment,
            approve_existing_spec=self.approve_existing_spec)
        self.ready=True

    def authenticate_preloaded(self,name,module):
        require(name in self.preloaded and module is self.preloaded[name],
                'Exact composed preloaded object')
        if name=='typing':return self.typing.authenticate_owner(name,module)
        if name in self.typing.aliases:
            return self.typing.authenticate_namespace(name,module,self.typing.owner)
        if self.interpreter_aliases.authenticate_alias(name,module):return True
        if name in self.interpreter.module_objects:
            self.interpreter.authenticate_module(name,module);return True
        require(self.other_preloaded(name,module)is True,
                'Remaining preloaded source/native/None origin must explicitly authenticate')
        return True

    def authenticate_source(self,name,module,row):
        require(self.source_code(name,module,copy.deepcopy(row))is True,
                'Actual source-code callback must explicitly authenticate')

    def authenticate_six_code(self,module,row):
        self.authenticate_source('six',module,row)

    def authenticate_alias_target(self,name,module,original_metadata=None):
        require(self.alias_target(name,module,original_metadata)is True,
                'Canonical alias target code/native origin explicitly authenticated')

    def authenticate_native(self,name,module,row):
        require(row['path']in self.mapping.files.verified,'Actually authenticated native module file')
        snapshot=self.mapping.audit()
        require(row['path']in snapshot['paths'],'Actual module payload is mapped from original file')

    def authenticate_alias(self,name,module):
        if self.interpreter_aliases.authenticate_alias(name,module):return True
        return self.six.alias(name,module)if self.six is not None else False

    def preloaded_transition(self,name,module,original_metadata):
        return self.six.preloaded_transition(name,module,original_metadata)if self.six is not None else False

    def authorize_custom(self,name,spec,finder):
        require(self.six is not None,'No reviewed custom alias provider for this phase')
        return self.six.authorize_loader(name,spec,finder)

    def authenticate_registered(self,name,module):
        require(self.ledger is not None and sys.modules.get(name)is module,
                'Actual registered provider module after ledger construction')
        if name in self.preloaded:
            self.authenticate_preloaded(name,module);return
        if self.authenticate_alias(name,module):return
        if name in self.interpreter.manifest['modules']:
            self.interpreter.authenticate_module(name,module);return
        row=self.router.modules.get(name)
        require(row is not None,'Provider outside reviewed module catalog')
        if row['kind']=='source':
            self.ledger.source_module(name,module,row)
            self.authenticate_source(name,module,row)
        elif row['kind']=='native':self.ledger.native_module(name,module,row)
        else:raise ValueError('Unreviewed provider origin')

    def approve_existing_spec(self,name,module,spec,finder):
        require(type(module)is types.ModuleType and vars(module).get('__spec__')is spec,
                'Existing-spec side effect must use actual module spec object')
        self.finder_policy.authorize(finder)
        self.authenticate_registered(name,module)


    def authenticate_site(self,spec,module):
        require(spec.name=='site','Only the exact phase site module')
        if spec.origin=='frozen':
            self.interpreter.authenticate_frozen_site(spec,module);return
        row=self.router.modules.get('site')
        require(row is not None and row['kind']=='source'and spec.origin==row['path'],
                'Original phase source site in reviewed router catalog')
        loader=spec.loader
        require(type(loader)is PinnedSourceLoader and loader.router is self.router
                and loader.reference==row and loader.path==row['path'],
                'Phase source site selected by authenticated source loader')
        if module is not None:
            require(sys.modules.get('site')is module and vars(module).get('__spec__')is spec,
                    'Actual original phase source site object')
            self.ledger.source_module('site',module,row)
            self.authenticate_source('site',module,row)

    @contextmanager
    def restored_startup(self,policy):
        """Concrete startup_context for WorkerExecution, still requiring admission."""
        provenance={}
        with restore_phase(policy,guard=self.guard,origin_context=self.installed,
                           authenticate_site=self.authenticate_site,provenance=provenance)as result:
            yield result

    def audit_environment(self):
        self.guard()
        require(sys.meta_path is self.finders,'Original finder list object retained')
        self.cache.check_runtime();self.interpreter.check();self.typing.check()
        require(self.preloaded_audit(self.preloaded)is True,
                'Remaining preloaded code/runtime state revalidated')
        require(self.environment()is True,'Actual phase environment remains authenticated')

    @contextmanager
    def installed(self):
        require(self.ready and not self.used,'Prepared one-use composed origin context')
        self.used=True
        report=dict(schema='nemotron4b_phase_origin_context_v1',complete=False,
            complete_runtime_dependency_closure_proven=False,actual_SDK_runtime_qualified=False,
            runtime_admitted=False,full_model_qualified=False,new_speedup_claim=False)
        try:
            with self.scope.installed()as imports:
                report['dispatch']=imports
                yield report
                self.router.authenticate(self.audit_environment)
                report['module_origins']=self.router.authenticate(self.ledger.result)
                report['typing_source']=self.typing.result()
                report['interpreter']=self.interpreter.result()
                report['bootstrap']=self.bootstrap.result()
                report['native_mappings']=self.mapping.audit()
            self.guard();self.bootstrap.check_original();self.cache.check_runtime()
            report['complete']=True
        except BaseException as error:
            report['failure']=dict(type=type(error).__name__,message=str(error)[:2048])
            raise
        finally:self.imports=report
