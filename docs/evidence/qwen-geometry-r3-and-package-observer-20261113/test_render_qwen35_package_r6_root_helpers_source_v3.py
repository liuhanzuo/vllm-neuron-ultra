"""Pure r6 source/mock successor tests; no current refs or runtime operations."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

import test_render_qwen35_package_r6_root_helpers_source_v2 as old
import render_qwen35_package_r6_root_helpers_source_v3 as p
import test_prepare_qwen35_writer_pair_candidate_v1 as writer
import qwen35_package_audit_physical_v1 as physical_adapter
import qwen35_package_post_audit_v1 as final_audit

PAYLOADS = old.PAYLOADS
AUDIT_SOURCES = {key:Path(path).read_bytes() for key,(path,_) in p.AUDIT_PINS.items()}
for key,(_,sha) in p.AUDIT_PINS.items():assert hashlib.sha256(AUDIT_SOURCES[key]).hexdigest()==sha


def render(bindings=None,payloads=None,audit_sources=None):
    with patch('builtins.open',side_effect=AssertionError('no file IO')),patch('os.open',side_effect=AssertionError('no fd IO')),\
         patch('os.system',side_effect=AssertionError('no shell')),patch('subprocess.Popen',side_effect=AssertionError('no process')),\
         patch('fcntl.flock',side_effect=AssertionError('no lock')):
        return p.render(PAYLOADS if payloads is None else payloads,old.fixture() if bindings is None else bindings,
                        AUDIT_SOURCES if audit_sources is None else audit_sources)


def observer_module():
    module=types.ModuleType('mock_r6_observer_source')
    module.__file__='/home/ubuntu/qwen35_package_full_r6_post_exit_20261112_v1.py'
    source=old.source(render(),'observer')
    exec(compile(source,'<rendered-source-only-observer>','exec'),module.__dict__)
    return module


OBSERVER=observer_module()


class Renderer(old.SourceSuccessor):
    def setUp(self):
        for obj,name,value in ((old,'render',render),(old,'p',p)):
            ctx=patch.object(obj,name,value);ctx.start();self.addCleanup(ctx.stop)

    def test_exact_static_support_payloads_and_all_adapter_pins_required(self):
        for key in p.AUDIT_PINS:
            value=dict(AUDIT_SOURCES);value[key]+=b'\n'
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'frozen audit support'):render(audit_sources=value)
        value=dict(AUDIT_SOURCES);value.pop('physical')
        with self.assertRaisesRegex(ValueError,'complete exact'):render(audit_sources=value)

    def test_four_other_outputs_are_byte_identical_to_v2(self):
        result=render();before=p.original.render(PAYLOADS,old.fixture())
        for key in ('accept','ack','budget','stat-inputs'):
            if key in ('accept','ack'):
                path=next(path for path in result['files'] if path.endswith(p.PINS[key][0].replace('r5','r6')))
            else:path=str(p.Q/('qwen35-package-full-r6-20261112-'+('resource-budget' if key=='budget' else key)+'.json'))
            self.assertEqual(result['files'][path],before['files'][path])
        self.assertEqual(result['required_actual_refs'],before['required_actual_refs'])

    def test_all_helpers_prebound_before_prepare_and_observer_checks(self):
        result=render();source=old.source(result,'prepare');tree=ast.parse(source)
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        expected=next(n for n in main.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)
                      and n.targets[0].id=='expected_audit_helper_pins')
        pins=ast.literal_eval(expected.value)
        self.assertEqual(pins,result['audit_source_prebinding']['generated_and_static_helpers'])
        self.assertEqual(len(pins),4)
        for path,sha in pins.items():
            payload=result['files'][path] if path in result['files'] else AUDIT_SOURCES[
                next(key for key,(target,_) in p.AUDIT_PINS.items() if target==path)]
            self.assertEqual(hashlib.sha256(payload).hexdigest(),sha)
            self.assertIn(Path(path).name,source)
        self.assertLess(source.index('assert all(sources.get(path)'),source.index('candidate = prep.prepare('))
        self.assertIn('preparation_sources=sources, required_helpers=[str(p) for p in helpers]',source)
        observer=old.source(result,'observer')
        self.assertIn('observer, ACK and audit adapters must be pinned before actual runtime admission',observer)
        for path,sha in result['audit_source_prebinding']['static_sources'].items():
            self.assertIn(path,observer);self.assertIn(sha,observer)

    def test_metadata_identity_checks_stay_per_guard(self):
        text=old.source(render(),'observer')
        self.assertIn("require(all(io.identity(path) == identity for path, identity in observed_metadata.items()),",text)
        self.assertIn("require(not terminal_missing or not os.path.lexists(terminal_path)",text)
        self.assertIn('freshness(force=force)',text)

    def test_every_generated_publication_has_forced_freshness_boundary(self):
        text=old.source(render(),'observer')
        for fragment in ('guard(force=True); sources_unchanged(); external_ref',
                         "io.loaded_local_sources(spec['source_sha256'])\n        guard(force=True)",
                         "guard(force=True)\n        require(io.once(Path(outputs['idle_handoff'])"):
            self.assertIn(fragment,text)
        # Catalog follows root's expected new idle; checking the old tip there
        # would reject the observer's own valid publication.
        self.assertNotIn("idle publication hash')\n        guard(force=True)",text)


class DatedBindings(old.DatedBindings):
    def setUp(self):
        for obj,name,value in ((old,'render',render),(old,'p',p)):
            ctx=patch.object(obj,name,value);ctx.start();self.addCleanup(ctx.stop)

    def test_nov12_output_identical_to_frozen_v1(self):
        result=render();before=p.original.render(PAYLOADS,old.fixture())
        self.assertEqual(set(result['files']),set(before['files']))
        self.assertEqual(result['required_actual_refs'],before['required_actual_refs'])
        self.assertEqual(result['inherited_stat_identities_pending_actual_recheck'],True)


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
        source=Path(final_audit.__file__).read_text().replace('qwen35_package_serial_runtime_v8.','.')
        source=source.replace('from .history import Authority\nfrom qwen35_package_audit_physical_v1 import Bundle',
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
        expected=Path(p.Q)/'qwen35_package_serial_runtime_v8/final_audit.py'
        self.assertEqual(ast.dump(Normalize().visit(ast.parse(source)),include_attributes=False),
                         ast.dump(Normalize().visit(ast.parse(expected.read_text())),include_attributes=False))

    def test_inherited_history_methods_and_original25_stages_unchanged(self):
        from qwen35_package_serial_runtime_v8.history import Bundle
        from qwen35_package_serial_runtime_v8.wiring import STAGES
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
