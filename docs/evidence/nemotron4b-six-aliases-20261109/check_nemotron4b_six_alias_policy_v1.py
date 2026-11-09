"""Actual synthetic six-style alias imports; installed six/SDK code is not executed."""
from contextlib import contextmanager
import importlib
import importlib.machinery as machinery
import json
from pathlib import Path
import sys
import types
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import check_nemotron4b_import_router_v4_r2 as fixtures
import check_nemotron4b_import_context_v3 as contexts
import nemotron4b_import_router_v4 as routing
import nemotron4b_module_origin_audit_v3 as origins
import nemotron4b_six_alias_policy_v1 as aliases
import analyze_nemotron4b_six_alias_policy_v1 as analyzer

SIX_FIXTURE = '''import sys,types,importlib,importlib.util
class MovedModule:
 def __init__(self,name,target):self.name=name;self.mod=target
 def _resolve(self):return importlib.import_module(self.mod)
 def __getattr__(self,name):return getattr(self._resolve(),name)
class _MovedItems(types.ModuleType):
 __path__=[]
class Module_six_moves_virtual(types.ModuleType):pass
class _SixMetaPathImporter:
 def __init__(self):self.name="six";self.known_modules={}
 def find_spec(self,name,path=None,target=None):
  if name in self.known_modules:return importlib.util.spec_from_loader(name,self)
 def is_package(self,name):return hasattr(self.known_modules[name],"__path__")
 def create_module(self,spec):
  name=spec.name
  if name in sys.modules:return sys.modules[name]
  entry=self.known_modules[name]
  if isinstance(entry,MovedModule):module=entry._resolve()
  else:
   module=entry;module.__loader__=self
  sys.modules[name]=module
  return module
 def exec_module(self,module):pass
_importer=_SixMetaPathImporter()
moves=_MovedItems("six.moves")
virtual=Module_six_moves_virtual("six.moves.virtual")
_importer.known_modules={"six.moves":moves,
 "six.moves.demo":MovedModule("demo","_nano_import_alias_target"),
 "six.moves.virtual":virtual,"six.moves.virtual_alt":virtual}
__path__=[]
__spec__.submodule_search_locations=[]
sys.meta_path.append(_importer)
'''

MAPPING={
 'six.moves':dict(kind='synthetic',class_name='_MovedItems',module_name='six.moves',aliases=['six.moves']),
 'six.moves.demo':dict(kind='moved',name='demo',target='_nano_import_alias_target'),
 'six.moves.virtual':dict(kind='synthetic',class_name='Module_six_moves_virtual',module_name='six.moves.virtual',
                         aliases=['six.moves.virtual','six.moves.virtual_alt']),
 'six.moves.virtual_alt':dict(kind='synthetic',class_name='Module_six_moves_virtual',module_name='six.moves.virtual',
                         aliases=['six.moves.virtual','six.moves.virtual_alt']),
}


class SixImports(unittest.TestCase):
    source=fixtures.Imports.source
    router=fixtures.Imports.router
    runtime=contexts.ContextImports.runtime
    make=contexts.ContextImports.make

    def setUp(self):
        fixtures.Imports.setUp(self)
        saved={n:m for n,m in sys.modules.items()if n=='six'or n.startswith('six.moves')}
        for n in saved:sys.modules.pop(n)
        def restore():
            for n in list(sys.modules):
                if n=='six'or n.startswith('six.moves'):sys.modules.pop(n)
            sys.modules.update(saved)
        self.addCleanup(restore)
        self.mapping=MAPPING
        self.source('six',SIX_FIXTURE)
        self.source('_nano_import_alias_target','VALUE=227\n')

    @contextmanager
    def integrated(self,preloaded=False):
        with self.runtime()as finders:
            originals=tuple(finders)
            if preloaded:
                # Actual target source import under ordinary fixture path before
                # installing the context; callback below explicitly binds it.
                importlib.import_module('_nano_import_alias_target')
            router=self.router(finders=finders)
            before=dict(sys.modules);snapshots={n:origins.metadata(m)for n,m in before.items()}
            target_audits=[]
            def target(name,module,original=None):
                self.assertEqual(name,'_nano_import_alias_target')
                self.assertEqual(module.VALUE,227)
                self.assertEqual(module.__file__,self.modules[name]['path'])
                if original is None:
                    self.assertEqual(module.__spec__.name,name)
                    self.assertEqual(module.__spec__.origin,self.modules[name]['path'])
                else:
                    self.assertEqual(original[4][1],name)
                    self.assertEqual(original[4][2],self.modules[name]['path'])
                target_audits.append((name,original is not None))
            def six_code(module,row):
                self.assertEqual(row,router.modules['six'])
                self.assertEqual(module.__file__,self.modules['six']['path'])
            policy=aliases.SixAliasPolicy(router,self.mapping,authenticate_six_code=six_code,authenticate_target=target)
            def authorize_finder(f):
                if any(f is x for x in originals):return
                self.assertIs(f,policy.binding())
            router.authorize_finder=authorize_finder
            router.authorize_custom=policy.authorize_loader
            def deny(*a):raise RuntimeError('No native/builtin fixture permission')
            def preloaded_auth(n,m):
                self.assertIs(m,before[n]);self.assertEqual(origins.metadata(m),snapshots[n])
            ledger=origins.ModuleOriginAudit(router,preloaded=before,
                authenticate_preloaded=preloaded_auth,interpreter_modules={},
                authenticate_interpreter=deny,authenticate_native=deny,
                authenticate_alias=policy.alias,authenticate_preloaded_transition=policy.preloaded_transition)
            router.authenticate(ledger.bind_preloaded)
            with self.make(router,audit_modules=ledger.audit).installed()as proof:
                yield router,policy,ledger,target_audits
                proof['ledger']=router.authenticate(ledger.result)

    def test_six_empty_package_transition_is_accepted(self):
        with self.integrated()as(router,policy,ledger,audits):
            six=importlib.import_module('six')
            self.assertEqual(six.__path__,[])
            self.assertEqual(six.__spec__.submodule_search_locations,[])
            self.assertTrue(policy.alias('six',six))
            ledger.audit()

    def test_moved_source_alias_preserves_identity_and_actual_spec_transition(self):
        with self.integrated()as(router,policy,ledger,audits):
            module=importlib.import_module('six.moves.demo')
            self.assertIs(module,sys.modules['_nano_import_alias_target'])
            self.assertEqual(module.__spec__.name,'six.moves.demo')
            self.assertEqual(module.__name__,'_nano_import_alias_target')
            self.assertTrue(policy.alias('_nano_import_alias_target',module))
            ledger.audit()
            self.assertTrue(any(changed for name,changed in audits))
            self.assertEqual([e['phase']for e in policy.events if e['name']=='six.moves.demo'],['create','exec'])

    def test_preloaded_target_spec_transition_is_explicitly_authorized(self):
        with self.integrated(preloaded=True)as(router,policy,ledger,audits):
            old=sys.modules['_nano_import_alias_target']
            module=importlib.import_module('six.moves.demo')
            self.assertIs(module,old)
            self.assertEqual(module.__spec__.name,'six.moves.demo')
            ledger.audit()

    def test_two_synthetic_aliases_reuse_one_module_with_latest_selected_spec(self):
        with self.integrated()as(router,policy,ledger,audits):
            first=importlib.import_module('six.moves.virtual')
            second=importlib.import_module('six.moves.virtual_alt')
            self.assertIs(first,second)
            self.assertEqual(first.__spec__.name,'six.moves.virtual_alt')
            self.assertTrue(policy.alias('six.moves.virtual',first))
            ledger.audit()

    def test_changed_alias_registry_entry_is_sticky_failure(self):
        with self.assertRaises(routing.ProvenanceError):
            with self.integrated()as(router,policy,ledger,audits):
                importlib.import_module('six.moves')
                provider=sys.modules['six']._importer
                provider.known_modules['six.moves.demo']=sys.modules['six'].MovedModule('demo','_nano_import_alias_target')
                router.check()

    def test_moved_target_change_rejected_before_import(self):
        with self.assertRaises(routing.ProvenanceError):
            with self.integrated()as(router,policy,ledger,audits):
                importlib.import_module('six.moves')
                sys.modules['six']._importer.known_modules['six.moves.demo'].mod='os'
                importlib.import_module('six.moves.demo')

    def test_same_looking_spec_replacement_is_rejected(self):
        with self.assertRaises(routing.ProvenanceError):
            with self.integrated()as(router,policy,ledger,audits):
                module=importlib.import_module('six.moves.demo')
                spec=module.__spec__
                module.__spec__=machinery.ModuleSpec(spec.name,spec.loader,origin=spec.origin)
                router.check()

    def test_target_file_mutation_is_not_covered_by_alias_spec_permission(self):
        with self.assertRaises(routing.ProvenanceError):
            with self.integrated(preloaded=True)as(router,policy,ledger,audits):
                module=importlib.import_module('six.moves.demo')
                module.__file__='/unexpected.py'
                router.check()

    def test_extra_virtual_package_path_is_rejected(self):
        with self.assertRaises(routing.ProvenanceError):
            with self.integrated()as(router,policy,ledger,audits):
                six=importlib.import_module('six')
                six.__path__.append('/unexpected')
                router.check()

    def test_unlisted_alias_cannot_be_inserted_into_registry(self):
        with self.assertRaises(routing.ProvenanceError):
            with self.integrated()as(router,policy,ledger,audits):
                importlib.import_module('six.moves')
                sys.modules['six']._importer.known_modules['six.moves.extra']=types.ModuleType('six.moves.extra')
                router.check()

    def test_missing_optional_alias_target_retains_normal_importerror(self):
        self.modules.pop('_nano_import_alias_target')
        (self.root/'_nano_import_alias_target.py').unlink()
        with self.integrated()as(router,policy,ledger,audits):
            with self.assertRaises(ImportError):
                importlib.import_module('six.moves.demo')
            self.assertFalse(router.poisoned)

    def test_source_code_authorization_failure_cannot_become_fallback(self):
        with self.assertRaises(routing.ProvenanceError):
            with self.integrated()as(router,policy,ledger,audits):
                def deny(*a):raise ImportError('Six code authorization failed')
                policy.authenticate_six_code=deny
                importlib.import_module('six')

    def test_optional_alias_reimport_accepts_new_exact_spec_and_original_target(self):
        with self.integrated(preloaded=True)as(router,policy,ledger,audits):
            first=importlib.import_module('six.moves.demo')
            old_spec=first.__spec__
            del sys.modules['six.moves.demo']
            second=importlib.import_module('six.moves.demo')
            self.assertIs(first,second)
            self.assertIsNot(old_spec,second.__spec__)
            ledger.audit()

    def test_inplace_selected_spec_origin_change_is_rejected(self):
        with self.assertRaises(routing.ProvenanceError):
            with self.integrated()as(router,policy,ledger,audits):
                module=importlib.import_module('six.moves.demo')
                module.__spec__.origin='/unexpected'
                router.check()

    def test_inplace_selected_spec_loader_change_is_rejected(self):
        with self.assertRaises(routing.ProvenanceError):
            with self.integrated()as(router,policy,ledger,audits):
                module=importlib.import_module('six.moves.demo')
                module.__spec__.loader=sys.modules['six']._importer
                router.check()

    def test_all54_source_declared_bindings_validate_without_loading_targets(self):
        q=Path(__file__).resolve().parent
        self.mapping=json.loads((q/'nemotron4b-six-alias-source-policy-20261109-v1.json').read_text())['aliases']
        build=SIX_FIXTURE.replace('sys.meta_path.append(_importer)','')
        build+='definitions='+repr(self.mapping)+'\n'
        build+='''
_importer.known_modules={}
created={}
for fullname,row in definitions.items():
 if row["kind"]=="moved":obj=MovedModule(row["name"],row["target"])
 else:
  group=tuple(row["aliases"])
  if group not in created:
   if row["class_name"] not in globals():
    globals()[row["class_name"]]=type(row["class_name"],(types.ModuleType,),{})
   created[group]=globals()[row["class_name"]](row["module_name"])
  obj=created[group]
 _importer.known_modules[fullname]=obj
sys.meta_path.append(_importer)
'''
        self.source('six',build)
        before=set(sys.modules)
        with self.integrated()as(router,policy,ledger,audits):
            six=importlib.import_module('six')
            self.assertIs(policy.binding(),six._importer)
            self.assertEqual(len(policy.objects),54)
            self.assertEqual(audits,[])
            self.assertFalse(any(name not in before and name in sys.modules
                                 for name in ('tkinter','http.server','xmlrpc.client')))


class StaticTable(unittest.TestCase):
    def test_generated_policy_keeps_linux_targets_and_synthetic_overrides(self):
        q=Path(__file__).resolve().parent
        policy=json.loads((q/'nemotron4b-six-alias-source-policy-20261109-v1.json').read_text())
        self.assertEqual(policy['alias_count'],54)
        self.assertEqual(policy['moved_alias_count'],42)
        self.assertEqual(policy['synthetic_alias_count'],12)
        self.assertEqual(policy['aliases']['six.moves.collections_abc']['target'],'collections.abc')
        self.assertEqual(policy['aliases']['six.moves._dummy_thread']['target'],'_thread')
        self.assertEqual(policy['aliases']['six.moves.urllib_parse']['kind'],'synthetic')
        self.assertNotIn('six.moves.winreg',policy['aliases'])
        self.assertFalse(policy['installed_package_code_executed'])

    def test_static_expression_rejects_execution(self):
        import ast
        with self.assertRaises(ValueError):
            analyzer.value(ast.parse("__import__('six')",mode='eval').body)


if __name__=='__main__':
    unittest.main(verbosity=2)
