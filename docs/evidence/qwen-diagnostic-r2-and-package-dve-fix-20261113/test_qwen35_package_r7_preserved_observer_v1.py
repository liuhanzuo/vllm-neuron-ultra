"""Preserved independent observer behavior with r7 renderer / v2 adapters."""
import ast
from pathlib import Path
import types
import unittest
from unittest.mock import patch
import test_render_qwen35_package_r7_root_helpers_source_v1 as current
import render_qwen35_package_r7_root_helpers_source_v1 as p
import test_prepare_qwen35_writer_pair_candidate_v1 as writer
import qwen35_package_audit_physical_v2 as physical_adapter
import qwen35_package_post_audit_v2 as final_audit
OBSERVER=types.ModuleType('synthetic_r7_observer')
OBSERVER.__file__='/home/ubuntu/qwen35_package_full_r7_post_exit_20261113_v1.py'
exec(compile(current.source(current.render(),'observer'),'<synthetic-r7-observer>','exec'),OBSERVER.__dict__)

class Freshness(writer.FreshnessTests):
    def setUp(self):
        ctx=patch.object(writer,'observer',OBSERVER);ctx.start();self.addCleanup(ctx.stop)


class PhysicalPasses(writer.PhysicalPassTests):
    def setUp(self):
        for obj,name,value in ((writer,'observer',OBSERVER),(writer,'physical_adapter',physical_adapter)):
            ctx=patch.object(obj,name,value);ctx.start();self.addCleanup(ctx.stop)


class OwnedScope(writer.ScopeChecks):
    def setUp(self):
        ctx=patch.object(writer.old_scope,'p',OBSERVER);ctx.start();self.addCleanup(ctx.stop)


class ScientificPreservation(unittest.TestCase):
    def test_complete_pair_collector_is_exact_accepted_v5_body(self):
        def fn(path,name):
            return next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
        self.assertEqual(ast.dump(fn(physical_adapter.__file__,'_collect_pair'),include_attributes=False),
                         ast.dump(fn(Path(p.Q)/'qwen35_geometry_audit_physical_v5.py','_collect_pair'),include_attributes=False))

    def test_full25_scientific_exit_and_source_checks_preserved(self):
        source=Path(final_audit.__file__).read_text().replace('qwen35_package_serial_runtime_v9.','.')
        source=source.replace('from .history import Authority\nfrom qwen35_package_audit_physical_v2 import Bundle',
                              'from .history import Bundle, Authority')
        source=source.replace('import importlib\n','').replace('import sys\n','').replace('PHASES, LIMITS, PACKAGE','PHASES, LIMITS')
        source=source.replace('def guard(force=False):','def guard():').replace('execution_guard(force=force)','execution_guard()')
        class Normalize(ast.NodeTransformer):
            def visit_Expr(self,node):
                if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id in ('guard','bind_pure_package_sources'):
                    return None
                return self.generic_visit(node)
            def visit_FunctionDef(self,node):
                if node.name=='bind_pure_package_sources':return None
                return self.generic_visit(node)
        expected=Path(p.Q)/'qwen35_package_serial_runtime_v9/final_audit.py'
        self.assertEqual(ast.dump(Normalize().visit(ast.parse(source)),include_attributes=False),
                         ast.dump(Normalize().visit(ast.parse(expected.read_text())),include_attributes=False))

    def test_inherited_history_methods_and_original25_stages_unchanged(self):
        from qwen35_package_serial_runtime_v9.history import Bundle
        from qwen35_package_serial_runtime_v9.wiring import STAGES
        self.assertEqual(len(STAGES),25);self.assertEqual(final_audit.STAGES,STAGES)
        for name in ('__init__','resolve','unchanged','freshness','scope'):
            self.assertIs(getattr(physical_adapter.Bundle,name),getattr(Bundle,name))
        self.assertIs(final_audit.Bundle,physical_adapter.Bundle)

    def test_each_audit_publication_has_own_forced_discovery(self):
        tree=ast.parse(Path(final_audit.__file__).read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='audit')
        outputs=[]
        for i,node in enumerate(fn.body):
            if isinstance(node,ast.Assign) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='once':
                before=fn.body[i-1];self.assertIsInstance(before,ast.Expr);self.assertEqual(before.value.func.id,'guard')
                self.assertEqual([(k.arg,getattr(k.value,'value',None)) for k in before.value.keywords],[('force',True)])
                outputs.append(node.targets[0].id)
        self.assertEqual(outputs,['result_ref','receipt'])



if __name__=='__main__':unittest.main(verbosity=2)
