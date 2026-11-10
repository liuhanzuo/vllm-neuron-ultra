"""Authenticate the original frozen import dispatcher before installing its hook.

This supplies DispatchContext.authenticate_bootstrap using the already bound
InterpreterPolicy. It binds the original function to its reviewed frozen module
definition, globals, defaults and exact code identity. It does not authenticate
all importlib.util globals, preloaded source modules or the complete SDK closure.
No native payload is opened or loaded here.
"""
import copy
import importlib._bootstrap as bootstrap
import importlib.util as util
import _imp
import sys
import types

from nemotron4b_interpreter_origin_policy_v1 import InterpreterPolicy,code_sha256,function_state,spec_description
from nemotron4b_process_handshake_v1 import require

CANONICAL='_frozen_importlib'
ALIAS='importlib._bootstrap'
FUNCTION='_find_spec'


def frozen_dispatch_code():
    root=_imp.get_frozen_object(CANONICAL)
    require(type(root)is types.CodeType,'Original frozen bootstrap module code')
    matches=[x for x in root.co_consts if type(x)is types.CodeType and x.co_name==FUNCTION]
    require(len(matches)==1,'One original top-level frozen import dispatcher')
    return matches[0]


def description():
    """Candidate observation only; production requires InterpreterPolicy first."""
    module=sys.modules.get(CANONICAL)
    require(type(module)is types.ModuleType and module is bootstrap
            and sys.modules.get(ALIAS)is module,'Actual original bootstrap aliases')
    original=vars(module).get(FUNCTION)
    require(type(original)is types.FunctionType and original.__globals__ is vars(module),
            'Actual original dispatcher function and module globals')
    require(sys.modules.get('importlib.util')is util and vars(util).get(FUNCTION)is original,
            'Original importlib.util dispatch alias')
    definition=code_sha256(frozen_dispatch_code())
    require(code_sha256(original.__code__)==definition,'Dispatcher matches frozen module definition')
    return dict(canonical_name=CANONICAL,registry_alias=ALIAS,module_name=vars(module)['__name__'],
        spec=spec_description(vars(module)['__spec__']),function_name=FUNCTION,
        definition_sha256=definition,function_state=function_state(original,describe=True))


class BootstrapDispatchPolicy:
    def __init__(self,interpreter,manifest):
        require(type(interpreter)is InterpreterPolicy and type(manifest)is dict,
                'Actual bound interpreter policy and reviewed dispatch manifest')
        self.interpreter=interpreter
        self.manifest=copy.deepcopy(manifest)
        self.authenticated=False
        interpreter.check()
        module=interpreter.module_objects.get(CANONICAL)
        require(type(module)is types.ModuleType and module is bootstrap
                and sys.modules.get(CANONICAL)is module and sys.modules.get(ALIAS)is module,
                'Bootstrap canonical module must already be authenticated and bound')
        interpreter.authenticate_module(CANONICAL,module)
        require(description()==self.manifest,'Exact reviewed frozen dispatcher description')
        self.module,self.util=module,util
        self.original=vars(module)[FUNCTION]
        self.state=function_state(self.original)
        self.globals=vars(module)

    def check_original(self):
        self.interpreter.authenticate_module(CANONICAL,self.module)
        require(sys.modules.get(CANONICAL)is self.module and sys.modules.get(ALIAS)is self.module
                and bootstrap is self.module and sys.modules.get('importlib.util')is self.util,
                'Original dispatcher module objects and aliases retained')
        require(self.original.__globals__ is self.globals
                and function_state(self.original)==self.state,
                'Original dispatcher globals, code identity and defaults retained')
        require(code_sha256(frozen_dispatch_code())==self.manifest['definition_sha256'],
                'Original frozen dispatcher definition retained')

    def authenticate_bootstrap(self,original):
        """One installation callback; the context then owns its deliberate hook."""
        require(not self.authenticated,'One dispatcher authentication per context')
        self.check_original()
        require(original is self.original
                and vars(self.module).get(FUNCTION)is self.original
                and vars(self.util).get(FUNCTION)is self.original,
                'Both original dispatch aliases must match before hook installation')
        self.authenticated=True

    def result(self):
        self.check_original()
        return dict(schema='nemotron4b_bootstrap_dispatch_origin_v1',
            original_dispatcher_authenticated=self.authenticated,
            definition_sha256=self.manifest['definition_sha256'],
            complete_preloaded_source_provenance=False,
            complete_runtime_dependency_closure_proven=False,
            arbitrary_live_memory_authenticated=False,runtime_admitted=False,new_speedup_claim=False)
