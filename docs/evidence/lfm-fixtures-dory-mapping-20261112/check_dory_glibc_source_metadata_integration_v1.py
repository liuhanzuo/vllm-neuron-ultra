"""Binding/dispatch/lifecycle mock checks; no live runtime or fixed data reads."""
import ast
import copy
from contextlib import ExitStack, contextmanager, redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

Q=Path(__file__).resolve().parent
sys.path[:0]=[str(Q),str(Q.parent)]
import dory_glibc_source_metadata_binding_v1 as b
import dory_glibc_source_metadata_runtime_v1 as m
import run_dory_glibc_source_metadata_root_v1 as parent
import dory_glibc_source_metadata_worker_v1 as worker
import run_dory_ubuntu_metadata_root_v5 as original_parent
import dory_ubuntu_metadata_worker_v5 as original_worker
from check_dory_glibc_source_metadata_runtime_v1 import fixture, MemoryOutput


@contextmanager
def completed(options=None):
    with fixture(options)as(inputs,raw,contents,opener):
        out=Path('/mnt/nvme/mock-stage');output=MemoryOutput(out/'glibc-source-metadata',Mock())
        paths={r['path']:raw[k]for k,r in inputs.items()};reads=[]
        def stable(path,cap,digest,guard,expected_size=None):
            guard();path=str(path);reads.append(path)
            data=paths[path]if path in paths else output.files[Path(path).name]
            if len(data)>cap or len(data)!=expected_size or m.sha(data)!=digest:raise ValueError('synthetic stable input mismatch')
            return data
        with patch.object(m,'stable_read',side_effect=stable),patch.object(m,'Output',return_value=output), \
             patch.object(b.c,'load',side_effect=lambda ref,cap:json.loads(output.files[Path(ref['path']).name])):
            authenticated=b.authenticate(Mock());value=m.run(Mock(),Mock(),output.root)
            yield SimpleNamespace(out=out,output=output,authenticated=authenticated,value=value,reads=reads,opener=opener)


class IntegrationTests(unittest.TestCase):
    def test01_exact_plan_fields_seven_inputs_policy_and_source_closure(self):
        sources={**b.SOURCE_PINS,**{str(Q/name):'a'*64 for name in b.RUNTIME_FILES}}
        plan=dict(work_kind=b.WORK_KIND,metadata_inputs=copy.deepcopy(b.INPUTS),network_policy=copy.deepcopy(b.NETWORK_POLICY),
                  source_sha256=sources,external_authorities=None)
        review={k:copy.deepcopy(plan[k])for k in ('work_kind','metadata_inputs','network_policy')}
        with patch.object(b.runtime,'validate')as original:
            self.assertIs(b.validate(plan,review,{}),plan);original.assert_called_once()
            for key in ('work_kind','metadata_inputs','network_policy','external_authorities','native_authority','ubuntu_trust'):
                changed=copy.deepcopy(plan);changed[key]={}
                with self.subTest(key=key),self.assertRaises(RuntimeError):b.validate(changed,review,{})
            changed=copy.deepcopy(plan);changed['source_sha256'].pop(str(Q/b.RUNTIME_FILES[0]))
            with self.assertRaises(RuntimeError):b.validate(changed,review,{})

    def test02_authentication_reads_exact_seven_pinned_inputs_and_binds_admission(self):
        with fixture()as(inputs,raw,_,_):
            bypath={r['path']:raw[k]for k,r in inputs.items()}
            with patch.object(m,'stable_read',side_effect=lambda path,*args:bypath[str(path)])as read:
                result=b.authenticate(Mock())
            self.assertEqual([call.args[0]for call in read.call_args_list],[r['path']for r in inputs.values()])
            self.assertEqual(read.call_count,7)
            self.assertEqual(result['source'],{'package':'glibc','version':'2.35-0ubuntu3.15'})
            b.validate_admission(dict(work_kind=b.WORK_KIND,metadata_input_binding=result),result)
            with self.assertRaises(RuntimeError):b.validate_admission(dict(work_kind='other',metadata_input_binding=result),result)

    def test03_complete_saved_result_is_replayed_without_network(self):
        with completed()as f:
            calls=f.opener.open.call_count
            result=b.verify_collected(f.out,f.value,f.authenticated,Mock())
            self.assertEqual(f.opener.open.call_count,calls)
            self.assertEqual(result['status'],'exact_source_metadata_ready')
            self.assertEqual(f.reads[-3:],[str(f.output.root/name)for name in ('jammy-updates-main-Sources.xz','jammy-security-main-Sources.xz','glibc-exact.dsc')])
            self.assertFalse(result['exact_loader_source_semantics_established'])

    def test04_absent_version_replay_preserves_missing_result(self):
        with completed(dict(source_version='2.35-0ubuntu3.16'))as f:
            result=b.verify_collected(f.out,f.value,f.authenticated,Mock())
            self.assertEqual(result['status'],'exact_source_version_missing')
            self.assertEqual(f.opener.open.call_count,2)

    def test05_saved_bytes_or_authentication_claim_tampering_rejected(self):
        for kind in ('bytes','speed','transport','preparation','path'):
            with completed()as f:
                result=json.loads(f.output.files['source-result.json'])
                if kind=='bytes':f.output.files['glibc-exact.dsc']+=b'bad'
                if kind=='speed':result['new_speedup_claim']=True
                if kind=='transport':result['runtime_transport']['requests'][0]['complete_body_EOF_observed']=False
                if kind=='preparation':result['preparation']['source']['version']='wrong'
                if kind=='path':result['saved_metadata'][0]['path']='/unreviewed/source.tar.xz'
                f.output.files['source-result.json']=json.dumps(result).encode()
                with self.subTest(kind=kind),self.assertRaises((RuntimeError,ValueError)):
                    b.verify_collected(f.out,f.value,f.authenticated,Mock())

    def test06_worker_receipt_includes_original_budget_and_no_promotion(self):
        with completed()as f,patch.object(b.c,'save',return_value={'path':'fake-receipt'})as save, \
             patch.object(b,'storage_usage',return_value=4096):
            result=b.save_collected(f.out,f.value,f.authenticated,{'plan':'mock'},{'admission':'mock'},Mock())
            self.assertEqual(result,{'path':'fake-receipt'})
            receipt=save.call_args.args[1]
            self.assertTrue(receipt['complete_saved_metadata_replay']);self.assertEqual(receipt['unresolved_loader_edges'],15)
            self.assertFalse(receipt['full_model_qualified']);self.assertFalse(receipt['actual_source_archive_download'])
            self.assertEqual(save.call_args.args[2],8*b.c.MIB)
        self.assertEqual(b.c.INVENTORY['address_space_bytes'],2*b.c.GIB)
        self.assertEqual(b.c.INVENTORY['cpu_seconds'],600);self.assertEqual(b.c.INVENTORY['wall_seconds'],900)
        self.assertEqual(b.c.INVENTORY['output_bytes'],128*b.c.MIB)

    def test07_actual_worker_dispatch_uses_authenticated_API_after_ACK(self):
        tree=ast.parse(Path(worker.__file__).read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main')
        body=next(n for n in main.body if isinstance(n,ast.Try)).body
        start=next(i for i,n in enumerate(body)if isinstance(n,ast.ImportFrom)and n.module=='dory_glibc_source_metadata_runtime_v1')
        stop=next(i for i,n in enumerate(body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='result_ref'for t in n.targets))
        block=ast.fix_missing_locations(ast.Module(body=body[start:stop+1],type_ignores=[]))
        order=[];bound={'mock':'binding'};journal=Mock()
        inputs=SimpleNamespace(authenticate=Mock(side_effect=lambda _:order.append('authenticate')or bound),
            validate_admission=Mock(side_effect=lambda *_:order.append('admission')),
            save_collected=Mock(side_effect=lambda *_:order.append('save')or {'path':'mock'}))
        module=ModuleType('dory_glibc_source_metadata_runtime_v1');module.run=Mock(side_effect=lambda *_:order.append('run')or {})
        jmodule=ModuleType('dory_serial_stage_journal_v2');jmodule.Journal=Mock(return_value=journal)
        ns=dict(c=SimpleNamespace(verify_loaded=Mock()),plan={'source_sha256':{}},inputs=inputs,guard=Mock(),admission={},
                out=Path('/mock'),plan_ref={},a=SimpleNamespace(admission='/mock/admission',admission_sha256='a'*64))
        with patch.dict(sys.modules,{module.__name__:module,jmodule.__name__:jmodule}):exec(compile(block,'<actual worker dispatch>','exec'),ns)
        self.assertEqual(order,['authenticate','admission','run','save'])
        module.run.assert_called_once_with(ns['guard'],journal.emit,Path('/mock/glibc-source-metadata'))
        text=Path(worker.__file__).read_text()
        self.assertLess(text.index("os.read(a.ready_fd,1)==b'R'"),text.index('metadata_binding=inputs.authenticate(guard)'))

    def test08_original_resource_launch_guard_and_cleanup_AST_preserved(self):
        for previous,current in ((original_parent,parent),(original_worker,worker)):
            before=ast.parse(Path(previous.__file__).read_text());after=ast.parse(Path(current.__file__).read_text())
            def calls(tree):
                names={'setrlimit','sched_setaffinity','nice','Popen','acquire','validate_admission','release_after_drain','drain'}
                return [ast.dump(n)for n in ast.walk(tree)if isinstance(n,ast.Call)and
                        ((isinstance(n.func,ast.Attribute)and n.func.attr in names)or(isinstance(n.func,ast.Name)and n.func.id in names))]
            self.assertEqual(calls(before),calls(after))
            one=next(n for n in ast.walk(before)if isinstance(n,ast.FunctionDef)and n.name=='guard')
            two=next(n for n in ast.walk(after)if isinstance(n,ast.FunctionDef)and n.name=='guard')
            self.assertEqual(ast.dump(one),ast.dump(two))
        for name in ('free','memory_free','drain','release_after_drain','preflight_history'):
            self.assertIs(getattr(parent,name),getattr(b.runtime.lifecycle,name))
        self.assertIs(parent.storage_usage,b.storage_usage)
        self.assertIn('pass_fds=(readfd,)',Path(parent.__file__).read_text())

    def test09_original_cleanup_releases_only_after_retry_drains(self):
        lifecycle=b.runtime.lifecycle;events=[]
        def drain(*args):
            events.append('drain')
            if events.count('drain')==1:raise RuntimeError('synthetic still-owned group')
        external=SimpleNamespace(release_lock=lambda fd:events.append('release'))
        with patch.object(lifecycle,'drain',side_effect=drain),patch.object(lifecycle.time,'sleep'):
            parent.release_after_drain(SimpleNamespace(returncode=7),{},object(),external,999,None)
        self.assertEqual(events,['drain','drain','release'])

    def test10_original_drain_never_signals_reused_group_leader(self):
        lifecycle=b.runtime.lifecycle;p=SimpleNamespace(proc=lambda pid:{'start_ticks':'new'})
        with patch.object(lifecycle,'group_members',return_value=[{}]),patch.object(lifecycle.os,'killpg')as signal:
            with self.assertRaisesRegex(RuntimeError,'reused process-group leader'):
                parent.drain(SimpleNamespace(pid=123),{'start_ticks':'old'},p)
            signal.assert_not_called()

    def test11_imports_and_help_do_not_open_transport_spawn_or_acquire(self):
        for module in (parent,worker):
            with patch.object(module.sys,'argv',['source-only','--help']),redirect_stdout(io.StringIO()), \
                 patch.object(module.runtime,'acquire',side_effect=AssertionError('no acquire')), \
                 patch.object(m.urllib.request,'build_opener',side_effect=AssertionError('no transport')):
                with self.assertRaises(SystemExit)as stopped:module.main()
                self.assertEqual(stopped.exception.code,0)
        for index,name in enumerate(b.RUNTIME_FILES):
            spec=importlib.util.spec_from_file_location('inert_dory_'+str(index),Q/name);module=importlib.util.module_from_spec(spec)
            with patch.object(m.os,'open',side_effect=AssertionError('no explicit file IO on import')), \
                 patch.object(parent.subprocess,'Popen',side_effect=AssertionError('no child on import')), \
                 patch.object(m.ssl,'create_default_context',side_effect=AssertionError('no TLS context on import')):
                spec.loader.exec_module(module)

    def test12_all_frozen_API_and_lifecycle_source_hashes_match(self):
        for path,digest in b.SOURCE_PINS.items():self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(),digest)


if __name__=='__main__':unittest.main(verbosity=2)
