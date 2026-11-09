"""Generated source-only tests for implicit initializer/import boundaries."""
import unittest

import nemotron4b_helper_import_graph_v1 as source


def module(code='', package=False):
    return dict(source=code, is_package=package)


class ImportGraph(unittest.TestCase):
    def graph(self, entry, modules, namespaces=None):
        return source.graph(entry,modules,namespaces or set(),package='example')

    def test_submodule_import_includes_initializer_and_its_side_imports(self):
        result=self.graph('from example.path import main',{
            'example':module('from . import compiler',True),
            'example.path':module('def main(): pass'),
            'example.compiler':module('import external_cache')})
        self.assertEqual(result['reachable_python_modules'],['example','example.compiler','example.path'])
        self.assertTrue(any(r['target']=='external_cache'for r in result['external_import_boundaries']))
        self.assertFalse(result['complete_runtime_dependency_closure_proven'])
        self.assertFalse(result['native_or_package_code_executed'])

    def test_relative_import_and_namespace_parent(self):
        result=self.graph('from example.path import main',{
            'example':module('from .proto import data',True),
            'example.path':module(),
            'example.proto.data':module('from .. import version'),
            'example.version':module()}, {'example.proto'})
        self.assertEqual(result['reachable_namespace_packages'],['example.proto'])
        self.assertIn('example.version',result['reachable_python_modules'])

    def test_functions_deferred_but_module_and_class_conditionals_included(self):
        result=self.graph('import example',{
            'example':module('''
if condition:
    import example.left
else:
    import example.right
class A:
    import example.body
    def method(self):
        import example.lazy
async def work():
    import example.lazy
''',True),
            **{f'example.{n}':module()for n in ('left','right','body','lazy')}})
        self.assertEqual(result['reachable_python_modules'],['example','example.body','example.left','example.right'])
        self.assertEqual(len(result['module_scans']['example']['function_body_imports']),2)

    def test_attribute_import_does_not_invent_submodule(self):
        result=self.graph('from example import function',{'example':module('function=1',True)})
        self.assertEqual(result['reachable_python_modules'],['example'])

    def test_external_members_remain_unresolved_boundaries(self):
        result=self.graph('import example',{'example':module('from other import member',True)})
        self.assertEqual([r['target']for r in result['external_import_boundaries']],['other','other.member'])
        self.assertEqual(result['external_import_boundaries'][1]['reason'],'external_member_or_submodule_unresolved')

    def test_star_import_not_claimed_complete(self):
        result=self.graph('from example import *',{'example':module('',True)})
        self.assertTrue(any(x['reason']=='star_export_resolution_unproven'for x in result['external_import_boundaries']))

    def test_dynamic_imports_report_context_without_execution(self):
        scanned=source.inspect_imports('''
value = __import__('never_execute')
@decorate(importlib.import_module('decorator_module'))
def f(default=__import__('default_module')):
    exec('import lazy_dynamic')
''')
        calls=scanned['dynamic_import_or_execution_calls']
        self.assertEqual([x['in_function_body']for x in calls],[False,False,False,True])

    def test_cycles_terminate(self):
        result=self.graph('import example',{'example':module('from . import one',True),
            'example.one':module('from . import two'),'example.two':module('from . import one')})
        self.assertEqual(result['reachable_python_modules'],['example','example.one','example.two'])

    def test_missing_explicit_local_module_and_parent_rejected(self):
        for modules in ({'example':module('import example.missing',True)},
                        {'example':module('',True),'example.deep.child':module()}):
            entry='import example' if len(modules)==1 else 'import example.deep.child'
            with self.assertRaisesRegex(ValueError,'Unresolved|Missing implicit'):self.graph(entry,modules)

    def test_relative_import_beyond_root_rejected(self):
        with self.assertRaisesRegex(ValueError,'Relative import'):
            self.graph('import example',{'example':module('from .. import invalid',True)})

    def test_sources_never_executed(self):
        result=self.graph('raise RuntimeError("entry executed")\nimport example',{
            'example':module('raise RuntimeError("module executed")',True)})
        self.assertEqual(result['reachable_python_modules'],['example'])


if __name__=='__main__':
    unittest.main(verbosity=2)
