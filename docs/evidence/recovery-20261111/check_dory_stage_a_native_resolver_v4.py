"""Source/JSON and fake-import checks; no native import/hash/model execution.

The tokenizer stub replays original saved output only to test the new checker;
it is not independent tokenization or proof of native readiness. The resolver
entry fixture stubs real inventory/import operations and never returns a usable
native API. Original-context tests read only pinned source and small metadata.
"""
import ast
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import stat
import sys
from types import SimpleNamespace,ModuleType
import unittest
from unittest.mock import patch

import dory_stage_a_native_resolver_v4 as resolver
from prepare_dory_stage_a_source_inputs_v1 import prepare_source_inputs


def message_key(messages,kwargs):return json.dumps([messages,kwargs],sort_keys=True,ensure_ascii=False)


class ReplayTokenizer:
    def __init__(self,context,template):
        self.chat_template=template;self.bos_token_id=1;self.eos_token_id=self.pad_token_id=11
        self.prompts={};self.encode_map={};self.decode_map={};self.fail=None
        for fixture,old in zip(context['fixtures'],context['captures']['cases']):
            self.prompts[message_key(fixture['messages'],fixture['kwargs'])]=(old['rendered'],old['input_ids'])
            self.encode_map[(old['rendered'],True)]=old['input_ids'];self.encode_map[(old['rendered'],False)]=old['input_ids']
            self.decode_map[tuple(old['input_ids'])]=old['rendered']
        for token in context['captures']['specials']:
            self.encode_map[(token['content'],False)]=[token['id']];self.decode_map[(token['id'],)]=token['content']
        full=context['full_fixtures']
        for prompt,paragraph,old in zip(full['protocol']['prefixes'],full['protocol']['paragraphs'],full['fixtures']):
            self.prompts[message_key([dict(role='user',content=prompt)],dict(add_generation_prompt=True))]=('',old['prefix_ids'])
            self.encode_map[(paragraph,False)]=old['body_ids']
    def __len__(self):return 131072
    def apply_chat_template(self,messages,*,tokenize,**kwargs):
        rendered,ids=self.prompts[message_key(messages,kwargs)]
        if self.fail=='rendered' and not tokenize:return rendered+'bad'
        if self.fail=='ids' and tokenize:return [999]
        return deepcopy(ids) if tokenize else rendered
    def encode(self,text,*,add_special_tokens):
        values=self.encode_map[(text,add_special_tokens)]
        if self.fail=='special' and len(values)==1:return [0]
        return deepcopy(values)
    def decode(self,ids,*,skip_special_tokens):
        assert skip_special_tokens is False
        return self.decode_map[tuple(ids)]
    def convert_ids_to_tokens(self,x):return {2:'</s>',11:'<|im_end|>'}[x]


class ContextAndTokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context=resolver.original_context(guard=lambda:None)
        cls.prepared=prepare_source_inputs(guard=lambda:None)
        ref=cls.context['token_refs']['chat_template.jinja']
        _,raw=resolver.stable_file(ref,cap=65536,guard=lambda:None,retain=True)
        cls.template=raw.decode()

    def setUp(self):
        self.tokenizer=ReplayTokenizer(self.context,self.template)
        self.raw=SimpleNamespace(encode=lambda text,add_special_tokens:SimpleNamespace(ids=self.tokenizer.encode(text,add_special_tokens=add_special_tokens)))

    def check(self,context=None,prepared=None,policy=None):
        return resolver.check_tokenizer(self.tokenizer,self.raw,context or self.context,prepared or self.prepared,
            self.template,self.context['captures']['generation_policy'] if policy is None else policy,guard=lambda:None)

    def test_fixed_original_sources_versions_templates_no_native_ready_claim(self):
        c=self.context
        self.assertEqual(c['versions'],dict(torch='2.14.0+cpu',vllm='0.1.dev1+g75fc05a53.cpu',transformers='5.20.2',pytest='9.0.2',tokenizers='0.23.3'))
        self.assertEqual(len(c['source_pins']),7)
        self.assertEqual(len(c['fixtures']),13);self.assertEqual(len(c['captures']['specials']),1000)
        self.assertEqual(c['token_refs']['tokenizer.json']['bytes'],17077484)
        self.assertFalse(c['native_inventory_anchor_present_in_prior_runtime'])
        self.assertFalse(c['runtime_admitted']);self.assertFalse(c['singleton_native_adapter_ABI_verified'])
        self.assertNotIn('run_dory_tokenizer_template_v1',sys.modules)

    def test_all13_saved_templates1000_specials_two_full_fixtures_routed(self):
        r=self.check();self.assertTrue(r['complete']);self.assertEqual(len(r['template_cases']),13)
        self.assertEqual((r['special_symbols'],r['full_token_fixtures'],r['stage_a_cases_checked']),(1000,2,6))
        self.assertFalse(r['independent_tokenizer_implementation_parity']);self.assertFalse(r['model_forward_executed'])

    def test_template_change_rejected(self):
        self.tokenizer.chat_template+='bad'
        with self.assertRaisesRegex(RuntimeError,'template/vocabulary'):self.check()

    def test_wrong_bos_rejected(self):
        self.tokenizer.bos_token_id=0
        with self.assertRaises(RuntimeError):self.check()

    def test_changed_rendered_prompt_rejected(self):
        self.tokenizer.fail='rendered'
        with self.assertRaisesRegex(RuntimeError,'template byte'):self.check()

    def test_changed_template_token_ids_rejected(self):
        self.tokenizer.fail='ids'
        with self.assertRaisesRegex(RuntimeError,'template IDs'):self.check()

    def test_changed_special_mapping_rejected(self):
        self.tokenizer.fail='special'
        with self.assertRaisesRegex(RuntimeError,'1000 special'):self.check()

    def test_generation_policy_and_stops_not_relaxed(self):
        p=deepcopy(self.context['captures']['generation_policy']);p['eos_token_id']=[11]
        with self.assertRaisesRegex(RuntimeError,'generation policy'):self.check(policy=p)

    def test_original_fixture_input_change_rejected(self):
        c=deepcopy(self.context);c['full_fixtures']['fixtures'][0]['input_ids'][0]=0
        with self.assertRaisesRegex(RuntimeError,'prompt/body composition'):self.check(context=c)

    def test_stage_a_case_token_mismatch_rejected(self):
        p=deepcopy(self.prepared);p['cases'][-1]['input_ids'][-1]=0
        with self.assertRaisesRegex(RuntimeError,'Every Stage A original token'):self.check(prepared=p)

    def test_raw_shared_backend_mismatch_rejected(self):
        self.raw.encode=lambda *a,**kw:SimpleNamespace(ids=[0])
        with self.assertRaisesRegex(RuntimeError,'template IDs'):self.check()

    def test_scalar_nested_multiple_conversation_ids_rejected(self):
        for ids in ([],[[1],[2]],[-1],[True],[131072],list(range(8193))):
            with self.subTest(ids=str(ids)[:50]),self.assertRaises(RuntimeError):resolver.normalize_ids(ids)


class MemoryFile:
    def __init__(self,raw):self.raw=raw;self.closed=False;self.offset=0;self.mode=stat.S_IFREG|0o600
    def info(self):return SimpleNamespace(st_mode=self.mode,st_uid=1000,st_nlink=1,st_dev=1,st_ino=2,
        st_size=len(self.raw),st_mtime_ns=3,st_ctime_ns=4)


class BoundedFileTests(unittest.TestCase):
    def setUp(self):
        self.file=MemoryFile(b'actual source bytes')
        self.events=[];outer=self
        class FakePath:
            def __init__(self,text):self.text=str(text)
            def __str__(self):return self.text
            def __eq__(self,other):return isinstance(other,FakePath) and self.text==other.text
            def is_absolute(self):return self.text.startswith('/')
            def resolve(self):return self
            def stat(self,follow_symlinks=False):return outer.file.info()
        class OS:
            O_RDONLY=0;O_NOFOLLOW=1;O_NONBLOCK=2
            def getuid(self):return 1000
            def open(self,path,flags):outer.events.append(('open',str(path),flags));return 7
            def fstat(self,fd):return outer.file.info()
            def read(self,fd,n):
                outer.events.append(('read',n));part=outer.file.raw[outer.file.offset:outer.file.offset+n];outer.file.offset+=len(part);return part
            def close(self,fd):outer.file.closed=True
        self.patches=[patch.object(resolver,'Path',FakePath),patch.object(resolver,'os',OS())]
        for p in self.patches:p.start()
        self.addCleanup(lambda:[p.stop() for p in reversed(self.patches)])
        self.ref=dict(path='/synthetic/source.py',bytes=len(self.file.raw),sha256=hashlib.sha256(self.file.raw).hexdigest())

    def test_exact_streamed_hash_identity_and_eof(self):
        result,raw=resolver.stable_file(self.ref,cap=65536,guard=lambda:None,retain=True)
        self.assertEqual(raw,self.file.raw);self.assertEqual(result['bytes'],len(raw));self.assertTrue(self.file.closed)
        self.assertTrue(all(event[1]<=65536 for event in self.events if event[0]=='read'))

    def test_no_binary_retention_with_streaming_hash(self):
        result,raw=resolver.stable_file(self.ref,cap=65536,guard=lambda:None)
        self.assertIsNone(raw);self.assertEqual(result['sha256'],self.ref['sha256'])

    def test_wrong_hash_closes_without_result(self):
        self.ref['sha256']='0'*64
        with self.assertRaisesRegex(RuntimeError,'SHA256'):resolver.stable_file(self.ref,cap=65536,guard=lambda:None)
        self.assertTrue(self.file.closed)

    def test_wrong_size_rejected(self):
        self.ref['bytes']+=1
        with self.assertRaisesRegex(RuntimeError,'Exact owned'):resolver.stable_file(self.ref,cap=65536,guard=lambda:None)
        self.assertTrue(self.file.closed)

    def test_fifo_rejected_without_blocking_read(self):
        self.file.mode=stat.S_IFIFO|0o600
        with self.assertRaisesRegex(RuntimeError,'owned'):resolver.stable_file(self.ref,cap=65536,guard=lambda:None)
        self.assertFalse(any(x[0]=='read' for x in self.events));self.assertTrue(self.file.closed)

    def test_bound_rejects_before_open(self):
        with self.assertRaises(RuntimeError):resolver.stable_file(self.ref,cap=1,guard=lambda:None)
        self.assertEqual(self.events,[])


    def test_explicit_empty_source_hash_and_eof_are_verified(self):
        self.file.raw=b''
        self.ref.update(bytes=0,sha256=hashlib.sha256(b'').hexdigest())
        result,raw=resolver.stable_file(self.ref,cap=65536,guard=lambda:None,retain=True)
        self.assertEqual(raw,b'');self.assertEqual(result['bytes'],0)
        self.assertEqual(result['sha256'],hashlib.sha256(b'').hexdigest())
        self.assertEqual([x for x in self.events if x[0]=='read'],[('read',1)])
        self.assertTrue(self.file.closed)

    def test_empty_source_wrong_hash_rejected_after_eof(self):
        self.file.raw=b'';self.ref.update(bytes=0,sha256='0'*64)
        with self.assertRaisesRegex(RuntimeError,'SHA256'):
            resolver.stable_file(self.ref,cap=65536,guard=lambda:None)
        self.assertTrue(self.file.closed)
        self.assertEqual([x for x in self.events if x[0]=='read'],[('read',1)])

    def test_empty_pin_rejects_nonempty_actual_file(self):
        self.ref.update(bytes=0,sha256=hashlib.sha256(b'').hexdigest())
        with self.assertRaisesRegex(RuntimeError,'Exact owned'):
            resolver.stable_file(self.ref,cap=65536,guard=lambda:None)
        self.assertTrue(self.file.closed)
        self.assertFalse(any(x[0]=='read' for x in self.events))

    def test_nonempty_pin_rejects_empty_actual_file(self):
        self.file.raw=b''
        with self.assertRaisesRegex(RuntimeError,'Exact owned'):
            resolver.stable_file(self.ref,cap=65536,guard=lambda:None)
        self.assertTrue(self.file.closed)

    def test_empty_fifo_still_rejected_before_read(self):
        self.file.raw=b'';self.file.mode=stat.S_IFIFO|0o600
        self.ref.update(bytes=0,sha256=hashlib.sha256(b'').hexdigest())
        with self.assertRaisesRegex(RuntimeError,'Exact owned'):
            resolver.stable_file(self.ref,cap=65536,guard=lambda:None)
        self.assertTrue(self.file.closed)
        self.assertFalse(any(x[0]=='read' for x in self.events))

    def test_negative_or_boolean_empty_size_rejected_before_open(self):
        for size in (-1,False,True):
            with self.subTest(size=size):
                self.ref['bytes']=size
                with self.assertRaisesRegex(RuntimeError,'bounded file pin'):
                    resolver.stable_file(self.ref,cap=65536,guard=lambda:None)
        self.assertEqual(self.events,[])


class AuthorityTests(unittest.TestCase):
    def setUp(self):
        self.ref=dict(path=str(resolver.SITE/'synthetic.py'),sha256='1'*64,bytes=4)
        self.authority=resolver.NativeFileAuthority([self.ref],byte_limit=4,guard=lambda:None)

    def test_explicit_zero_byte_inventory_entry_is_retained(self):
        ref=dict(self.ref,bytes=0,sha256=hashlib.sha256(b'').hexdigest())
        authority=resolver.NativeFileAuthority([ref],byte_limit=1,guard=lambda:None)
        self.assertEqual(authority.files,{ref['path']:ref});self.assertEqual(authority.bytes,0)
        with (patch.object(resolver,'stable_file',return_value=({'path':ref['path'],'identity':{}},None)) as reader,
              patch.object(authority,'verify_stable')):
            authority.verify_all({'source_pins':{ref['path']:ref['sha256']}})
            self.assertEqual(reader.call_count,1)
            self.assertEqual(reader.call_args.args[0],ref)
        authority.active=True
        authority.audit('open',(ref['path'],'rb',0))

    def test_negative_boolean_or_unpinned_empty_inventory_still_rejected(self):
        for size in (-1,False,True):
            with self.subTest(size=size),self.assertRaises(RuntimeError):
                resolver.NativeFileAuthority([dict(self.ref,bytes=size)],byte_limit=4,guard=lambda:None)
        self.authority.active=True
        with self.assertRaisesRegex(RuntimeError,'outside authenticated inventory'):
            self.authority.audit('open',(str(resolver.SITE/'empty_unpinned.py'),'rb',0))

    def test_inventory_empty_duplicate_or_overbudget_rejected(self):
        for files,budget in [([],4),([self.ref,self.ref],8),([self.ref],3)]:
            with self.assertRaises(RuntimeError):resolver.NativeFileAuthority(files,byte_limit=budget,guard=lambda:None)

    def test_original_pin_cannot_be_rebound_before_hash(self):
        with patch.object(resolver,'stable_file') as reader:
            with self.assertRaisesRegex(RuntimeError,'cannot be rebound'):
                self.authority.verify_all({'source_pins':{self.ref['path']:'0'*64}})
            reader.assert_not_called()

    def test_all_file_hashes_are_required_before_import_state(self):
        with (patch.object(resolver,'stable_file',return_value=({'path':self.ref['path'],'identity':{}},None)) as reader,
              patch.object(self.authority,'verify_stable')):
            self.authority.verify_all({'source_pins':{self.ref['path']:'1'*64}})
            self.assertEqual(reader.call_count,1);self.assertEqual(set(self.authority.verified),{self.ref['path']})
            with self.assertRaisesRegex(RuntimeError,'Fresh single-use'):self.authority.verify_all({'source_pins':{}})

    def test_system_library_or_interpreter_inventory_is_explicitly_external(self):
        for path in ('/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4',str(resolver.ENV/'bin/python'),str(resolver.ENV/'lib/libiomp5.so')):
            ref=dict(self.ref,path=path)
            with self.subTest(path=path),self.assertRaises(RuntimeError):
                resolver.NativeFileAuthority([ref],byte_limit=4,guard=lambda:None)

    def test_cached_bytecode_not_accepted(self):
        self.authority.active=True
        with self.assertRaises(OSError):self.authority.audit('open',('/synthetic/x.pyc','rb',0))

    def test_unverified_site_code_or_extension_rejected(self):
        self.authority.active=True
        for name in (str(resolver.SITE/'bad.py'),str(resolver.SITE/'bad.so')):
            with self.assertRaises(RuntimeError):self.authority.audit('open',(name,'rb',0))
        with self.assertRaises(RuntimeError):self.authority.audit('import',('bad',str(resolver.SITE/'bad.so')))

    def test_unknown_callable_origin_rejected(self):
        with self.assertRaises(RuntimeError):self.authority.origin('/synthetic/else.py')

    def test_audit_is_inert_after_exit(self):
        self.authority.audit('open',('/synthetic/x.pyc','rb',0))


class ContextManagerOriginTests(unittest.TestCase):
    def setUp(self):
        self.path=str(resolver.SITE/'vllm/config/vllm.py')
        self.seen=[]
        self.authority=SimpleNamespace(origin=lambda p:self.seen.append(p) or dict(path=p,sha256='1'*64))
        namespace={}
        exec(compile('def set_current_vllm_config(value):\n    yield value\n',self.path,'exec'),namespace)
        self.original=namespace['set_current_vllm_config'];self.wrapped=contextmanager(self.original)

    def test_real_stdlib_contextmanager_binds_authenticated_original_generator(self):
        self.assertNotEqual(resolver.inspect.getsourcefile(self.wrapped),self.path)
        result=resolver.callable_origin('set_current_vllm_config',self.wrapped,self.authority)
        self.assertEqual(self.seen,[self.path]);self.assertEqual(result['path'],self.path)
        self.assertFalse(result['stdlib_identity_authenticated_here'])

    def test_arbitrary_wrapper_with_wrapped_attribute_rejected(self):
        def forged(*a,**kw):return self.original(*a,**kw)
        forged.__wrapped__=self.original
        with self.assertRaisesRegex(RuntimeError,'contextmanager code'):resolver.callable_origin('set_current_vllm_config',forged,self.authority)
        self.assertEqual(self.seen,[])

    def test_changed_wrapped_attribute_cannot_hide_real_closure(self):
        self.wrapped.__wrapped__=lambda:None
        with self.assertRaisesRegex(RuntimeError,'contextmanager code'):resolver.callable_origin('set_current_vllm_config',self.wrapped,self.authority)
        self.assertEqual(self.seen,[])

    def test_non_generator_contextmanager_rejected(self):
        def set_current_vllm_config(value):return value
        with self.assertRaises(RuntimeError):resolver.callable_origin('set_current_vllm_config',contextmanager(set_current_vllm_config),self.authority)

    def test_foreign_generator_source_rejected(self):
        self.authority.origin=lambda p:dict(path=str(resolver.SITE/'foreign.py'),sha256='1'*64)
        with self.assertRaisesRegex(RuntimeError,'Original vLLM context generator'):resolver.callable_origin('set_current_vllm_config',self.wrapped,self.authority)


class EntryBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prepared=prepare_source_inputs(guard=lambda:None);cls.context=resolver.original_context(guard=lambda:None)

    def test_missing_or_wrong_native_environment_blocks_before_import(self):
        with patch.object(resolver.importlib,'import_module') as importing:
            with self.assertRaisesRegex(RuntimeError,'original CPU interpreter'):
                resolver.resolve_native_api(self.prepared,[],byte_limit=1,guard=lambda:None).__enter__()
            importing.assert_not_called()

    def test_missing_inventory_blocks_even_when_process_is_stubbed(self):
        with patch.object(resolver,'_check_process'),patch.object(resolver.importlib,'import_module') as importing:
            with self.assertRaisesRegex(RuntimeError,'Explicit reviewed inventory'):
                resolver.resolve_native_api(self.prepared,[],byte_limit=1,guard=lambda:None).__enter__()
            importing.assert_not_called()

    def test_rehashed_prepared_input_cannot_change_original(self):
        p=deepcopy(self.prepared);p['cases'][0]['input_ids'][0]=0
        p['case_set_sha256']=resolver.digest(p['cases']);p['input_sha256']=resolver.digest({k:v for k,v in p.items() if k!='input_sha256'})
        with patch.object(resolver.importlib,'import_module') as importing:
            with self.assertRaisesRegex(RuntimeError,'Reauthenticate original'):resolver.resolve_native_api(p,[],byte_limit=1,guard=lambda:None).__enter__()
            importing.assert_not_called()

    def test_version_environment_check_preserves_no_admission(self):
        env=dict(self.context['environment']);env['VLLM_TARGET_DEVICE']='cuda'
        with (patch.object(resolver.sys,'executable',str(resolver.ENV/'bin/python')),
              patch.object(resolver.sys,'prefix',str(resolver.ENV)),patch.object(resolver.sys,'dont_write_bytecode',True),
              patch.object(resolver.sys,'flags',SimpleNamespace(optimize=0,no_user_site=1)),patch.dict(resolver.os.environ,env,clear=True)):
            with self.assertRaisesRegex(RuntimeError,'Exact original CPU'):resolver._check_process(self.context)

    def fake_resolution(self,*,fail_import=None,fail_guard_at=None,fail_body=False,fail_late_origin=False):
        # Fake imported modules exercise concrete resolver wiring only. They
        # contain no native API implementation and never escape this test.
        context=deepcopy(self.context);context['token_refs']={
            'chat_template.jinja':dict(path=str(resolver.META/'chat_template.jinja'),sha256='1'*64,bytes=1),
            'generation_config.json':dict(path=str(resolver.META/'generation_config.json'),sha256='2'*64,bytes=2)}
        upstream_path=str(resolver.OFFICIAL/'tests/models/test_dory.py')
        upstream_bytes=b'class ReferenceAttention: pass\ndef _prepare_for_inference(model): pass\n'
        events=[];guards=[0];authorities=[]
        def guard():
            guards[0]+=1
            if guards[0]==fail_guard_at:raise ValueError('real guard failure')
        class FakeAuthority:
            def __init__(self,*a,**kw):self.files={upstream_path:dict(path=upstream_path,sha256='3'*64,bytes=len(upstream_bytes))};self.bytes=100;self.active=False;authorities.append(self)
            def verify_all(self,c):events.append('verify-all')
            def verify_stable(self):events.append('verify-stable')
            def audit(self,*a):pass
            def origin(self,p):return dict(path=p,sha256='4'*64)
        modules={}
        for name in resolver.MODULES:
            m=ModuleType(name);m.__file__=str(resolver.SITE/name.replace('.','/')/'__init__.py')
            if name in context['versions']:m.__version__=context['versions'][name]
            modules[name]=m
        modules['vllm.model_executor.models.dory'].__file__=str(resolver.SITE/'vllm/model_executor/models/dory.py')
        def api_callable(*a,**kw):pass
        modules['pytest'].MonkeyPatch=api_callable
        for key in ('VllmConfig','DeviceConfig','CompilationConfig'):setattr(modules['vllm.config'],key,api_callable)
        def set_current_vllm_config(*args):yield
        modules['vllm.config'].set_current_vllm_config=contextmanager(set_current_vllm_config)
        modules['vllm.model_executor.models.dory'].DoryForCausalLM=api_callable
        modules['vllm._custom_ops'].onednn_mm=api_callable
        modules['vllm.transformers_utils.configs.dory'].DoryConfig=api_callable
        modules['vllm.model_executor.models.utils'].extract_layer_index=api_callable
        class Tokenizer:
            @staticmethod
            def from_pretrained(root,*,local_files_only):
                self.assertTrue(local_files_only);events.append('tokenizer-create');return object()
        modules['transformers'].PreTrainedTokenizerFast=Tokenizer
        modules['tokenizers'].Tokenizer=SimpleNamespace(from_file=lambda path:object())
        def sourcefile(value):
            if getattr(value,'__name__','')=='set_current_vllm_config':return str(resolver.SITE/'vllm/config/vllm.py')
            return upstream_path if getattr(value,'__module__','')==resolver.UPSTREAM_NAME else str(resolver.SITE/'synthetic.py')
        def importer(name):
            events.append('import:'+name)
            if name==fail_import:raise ValueError('actual import failure')
            return modules[name]
        def stable(ref,**kw):
            data=upstream_bytes if ref['path']==upstream_path else b'{}' if ref['path'].endswith('.json') else b'T'
            return dict(path=ref['path'],identity={}),data
        origin_calls=[0]
        def module_origins(authority):
            origin_calls[0]+=1
            if fail_late_origin and origin_calls[0]==2:raise ValueError('late imported origin failure')
            return []
        patches=[patch.object(resolver,'prepare_source_inputs',return_value=self.prepared),
            patch.object(resolver,'original_context',return_value=context),patch.object(resolver,'_check_process'),
            patch.object(resolver,'NativeFileAuthority',FakeAuthority),patch.object(resolver,'stable_file',side_effect=stable),
            patch.object(resolver.importlib,'import_module',side_effect=importer),patch.object(resolver.sys,'addaudithook'),
            patch.object(resolver.inspect,'getsourcefile',side_effect=sourcefile),patch.object(resolver,'_module_origins',side_effect=module_origins),
            patch.object(resolver,'check_tokenizer',side_effect=lambda *a,**kw:events.append('tokenizer-check') or {'synthetic_check':True}),
            patch.object(resolver.Path,'stat',return_value=SimpleNamespace()),patch.object(resolver,'identity',return_value={})]
        for p in patches:p.start()
        try:
            with resolver.resolve_native_api(self.prepared,[dict(synthetic_fixture=True)],byte_limit=100,guard=guard) as (api,report):
                self.assertTrue(authorities[0].active)
                self.assertFalse(report['worker_context_source_stability_verified'])
                if fail_body:raise ValueError('actual worker exception')
                self.assertIs(api.torch,modules['torch']);self.assertIs(api.monkeypatch_factory,modules['pytest'].MonkeyPatch)
                self.assertIs(api.official_source,modules['vllm.model_executor.models.dory'])
                self.assertIs(api.ops,modules['vllm._custom_ops']);self.assertTrue(callable(api.upstream.ReferenceAttention))
                self.assertEqual(events[0],'verify-all');self.assertIn('tokenizer-check',events)
                self.assertFalse(report['inventory_trust_anchor_authenticated_here'])
                self.assertFalse(report['interpreter_system_libraries_authenticated_here'])
                self.assertFalse(report['native_adapter_ABI_verified']);self.assertFalse(report['singleton_native_layout_verified'])
                self.assertFalse(report['model_forward_executed_by_resolver']);self.assertFalse(report['admission_created_here'])
            self.assertFalse(authorities[0].active)
            self.assertTrue(report['worker_context_source_stability_verified'])
            self.assertEqual(origin_calls[0],2)
            return events
        finally:
            if authorities:self.assertFalse(authorities[0].active)
            for p in reversed(patches):p.stop()
            sys.modules.pop(resolver.UPSTREAM_NAME,None)

    def test_concrete_import_api_binding_and_tokenizer_sequence_with_fake_modules(self):
        self.fake_resolution()

    def test_worker_exception_propagates_and_deactivates_scope(self):
        with self.assertRaisesRegex(ValueError,'actual worker exception'):self.fake_resolution(fail_body=True)

    def test_newly_imported_origin_failure_blocks_normal_scope_completion(self):
        with self.assertRaisesRegex(ValueError,'late imported origin failure'):self.fake_resolution(fail_late_origin=True)

    def test_actual_import_error_is_not_converted_to_success(self):
        with self.assertRaisesRegex(ValueError,'actual import failure'):self.fake_resolution(fail_import='pytest')

    def test_guard_exception_is_not_swallowed(self):
        with self.assertRaisesRegex(ValueError,'real guard failure'):self.fake_resolution(fail_guard_at=2)

    def test_direct_source_resolver_does_not_invoke_model_worker_or_ABI_probe(self):
        tree=ast.parse(Path(resolver.__file__).read_text());calls=[];imports=[]
        for n in ast.walk(tree):
            if isinstance(n,ast.Call):calls.append(ast.unparse(n.func))
            if isinstance(n,ast.Import):imports += [x.name for x in n.names]
            if isinstance(n,ast.ImportFrom):imports.append(n.module)
        self.assertFalse(any(x in calls for x in ('run_worker','load_checkpoint','subprocess.Popen','api.official_source.DoryForCausalLM')))
        self.assertFalse(any(x.split('.')[0] in ('torch','numpy','vllm','transformers','tokenizers','pytest') for x in imports))
        self.assertNotIn('torch',sys.modules);self.assertNotIn('numpy',sys.modules)
        self.assertFalse(any('ABI' in x or 'singleton' in x for x in calls))


if __name__=='__main__':unittest.main(verbosity=2)
