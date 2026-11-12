"""Synthetic HTTPS and saved metadata fixtures; no real network or processes."""
import copy
from contextlib import ExitStack, contextmanager
import email.message
import io
import json
from pathlib import Path
import ssl
import tempfile
import unittest
from unittest.mock import Mock, patch
import urllib.error
import urllib.request

import dory_glibc_source_metadata_runtime_v1 as m
import dory_glibc_source_metadata_binding_v1 as binding
import check_dory_glibc_source_metadata_v1 as fixtures


class Response(io.BytesIO):
    def __init__(self, body, url, *, status=200, headers=None):
        super().__init__(body)
        self.url, self.status = url, status
        self.headers = email.message.Message()
        for key, value in (headers if headers is not None else [('Content-Length',str(len(body)))]):
            self.headers[key] = value
    def geturl(self):return self.url


class MemoryOutput:
    def __init__(self, root, guard):self.root=Path(root);self.guard=guard;self.files={};self.refs=[]
    def save(self, name, body):
        self.guard()
        if name in self.files:raise AssertionError('duplicate fake output')
        self.files[name]=body
        ref=dict(path=str(self.root/name),bytes=len(body),sha256=m.sha(body));self.refs.append(ref)
        return ref


@contextmanager
def fixture(options=None):
    inputs,indexes,raw,contents=fixtures.fixture(**(options or {}))
    opener=Mock();opener.open.side_effect=lambda request,**kwargs:Response(contents[request.full_url],request.full_url)
    with patch.object(m.core,'INPUTS',inputs),patch.object(m.core,'SOURCE_INDEXES',indexes), \
         patch.object(binding,'INPUTS',inputs),patch.object(m.ssl,'create_default_context',return_value=object()), \
         patch.object(m.urllib.request,'build_opener',return_value=opener):
        yield inputs,raw,contents,opener


class TransportTests(unittest.TestCase):
    def test01_complete_ordered_two_indexes_and_DSC_with_real_API(self):
        with fixture()as(inputs,raw,contents,opener):
            output=MemoryOutput('/mnt/nvme/mock/glibc-source-metadata',Mock())
            bypath={r['path']:raw[k]for k,r in inputs.items()}
            with patch.object(m,'Output',return_value=output),patch.object(m,'stable_read',side_effect=lambda path,*args:bypath[str(path)]):
                value=m.run(Mock(),Mock(),output.root)
            result=json.loads(output.files['source-result.json'])
            self.assertEqual(value['metadata_requests'],3);self.assertEqual(opener.open.call_count,3)
            self.assertEqual(result['metadata_requests'],[call.args[0].full_url for call in opener.open.call_args_list])
            self.assertTrue(all(r['complete_body_EOF_observed']for r in result['runtime_transport']['requests']))
            self.assertTrue(all(call.kwargs=={'timeout':5}for call in opener.open.call_args_list))
            self.assertFalse(result['actual_source_archive_download'])

    def test02_exact_version_missing_produces_two_requests_only(self):
        with fixture(dict(source_version='2.35-0ubuntu3.16'))as(inputs,raw,contents,opener):
            transport=m.Transport(Mock(),Mock())
            result=m.collector.run(lambda ref,cap:raw[next(k for k,r in inputs.items()if r==ref)],transport,
                lambda name,body:fixtures.ref(body,'/mock/'+name),Mock())
            self.assertEqual(result['status'],'exact_source_version_missing');self.assertEqual(opener.open.call_count,2)

    def test03_unknown_or_reordered_URL_denied_before_open(self):
        for kind in ('second','archive','wronghash'):
            with fixture()as(_,_,_,opener):
                transport=m.Transport(Mock(),Mock());request=transport.requests[0]
                url=transport.requests[1]['url']if kind=='second'else request['url']+'extra'if kind=='wronghash'else 'https://archive.ubuntu.com/ubuntu/pool/main/g/glibc/source.tar.xz'
                with self.assertRaisesRegex(ValueError,'ordered by-hash'):transport(url,request['compressed']['bytes'])
                opener.open.assert_not_called()

    def test04_archive_cannot_replace_DSC_after_two_real_index_parses(self):
        with fixture()as(_,_,_,opener):
            transport=m.Transport(Mock(),Mock())
            for request in transport.requests:transport(request['url'],request['compressed']['bytes'])
            with self.assertRaisesRegex(ValueError,'single DSC'):
                transport('https://archive.ubuntu.com/ubuntu/pool/main/g/glibc/glibc_2.35.orig.tar.xz',1234)
            self.assertEqual(opener.open.call_count,2)

    def test05_cross_pocket_conflict_prevents_DSC(self):
        with fixture(dict(conflict=True))as(_,_,_,opener):
            transport=m.Transport(Mock(),Mock())
            for request in transport.requests:transport(request['url'],request['compressed']['bytes'])
            with self.assertRaisesRegex(ValueError,'Consistent complete'):
                transport('https://archive.ubuntu.com/ubuntu/pool/main/g/glibc/glibc_2.35-0ubuntu3.15.dsc',1)
            self.assertEqual(opener.open.call_count,2)

    def test06_404_failure_is_latched_without_retry_or_fallback(self):
        with fixture()as(_,_,_,opener):
            opener.open.side_effect=urllib.error.HTTPError('synthetic',404,'missing',{},None)
            transport=m.Transport(Mock(),Mock());one,two=transport.requests
            for request in (one,one,two):
                with self.assertRaises(urllib.error.HTTPError):transport(request['url'],request['compressed']['bytes'])
            self.assertEqual(opener.open.call_count,1)

    def test07_redirect_handler_rejects_before_followup(self):
        with self.assertRaises(urllib.error.HTTPError):
            m.NoRedirect().redirect_request(urllib.request.Request('https://archive.ubuntu.com/ubuntu/x'),None,302,'found',{},'https://elsewhere.invalid/')

    def test08_HTTP_status_and_final_URL_are_exact(self):
        for wrong in ('status','url'):
            with fixture()as(_,_,contents,opener):
                transport=m.Transport(Mock(),Mock());r=transport.requests[0];url=r['url']
                opener.open.side_effect=None;opener.open.return_value=Response(contents[url],url if wrong=='status'else url+'other',status=302 if wrong=='status'else 200)
                with self.assertRaisesRegex(ValueError,'status and URL'):transport(url,r['compressed']['bytes'])

    def test09_missing_duplicate_or_wrong_Content_Length_rejected(self):
        for mode in ('missing','duplicate','short','long','combined'):
            with fixture()as(_,_,contents,opener):
                transport=m.Transport(Mock(),Mock());r=transport.requests[0];url=r['url'];n=r['compressed']['bytes']
                headers=[]if mode=='missing'else [('Content-Length',str(n)),('Content-Length',str(n))]if mode=='duplicate'else [('Content-Length',str(n-1)if mode=='short'else str(n+1)if mode=='long'else f'{n}, {n}')]
                opener.open.side_effect=None;opener.open.return_value=Response(contents[url],url,headers=headers)
                with self.assertRaises(ValueError):transport(url,n)

    def test10_truncated_or_extra_body_rejected_at_EOF(self):
        for ending in ('short','extra'):
            with fixture()as(_,_,contents,opener):
                transport=m.Transport(Mock(),Mock());r=transport.requests[0];url=r['url'];n=r['compressed']['bytes']
                body=contents[url][:-1]if ending=='short'else contents[url]+b'x'
                opener.open.side_effect=None;opener.open.return_value=Response(body,url,headers=[('Content-Length',str(n))])
                with self.assertRaises(ValueError):transport(url,n)

    def test11_encoding_chunking_or_partial_response_rejected(self):
        for key,value in [('Content-Encoding','gzip'),('Transfer-Encoding','chunked'),('Content-Range','bytes 0-2/4')]:
            with fixture()as(_,_,contents,opener):
                transport=m.Transport(Mock(),Mock());r=transport.requests[0];url=r['url'];n=r['compressed']['bytes']
                opener.open.side_effect=None;opener.open.return_value=Response(contents[url],url,headers=[('Content-Length',str(n)),(key,value)])
                with self.assertRaisesRegex(ValueError,'identity HTTP bodies'):transport(url,n)

    def test12_full_digest_rejected_even_when_length_matches(self):
        with fixture()as(_,_,contents,opener):
            transport=m.Transport(Mock(),Mock());r=transport.requests[0];url=r['url']
            opener.open.side_effect=None;opener.open.return_value=Response(b'x'*len(contents[url]),url)
            with self.assertRaisesRegex(ValueError,'metadata bytes'):transport(url,r['compressed']['bytes'])

    def test13_request_deadline_checked_after_delayed_EOF(self):
        with fixture()as(_,_,contents,opener):
            transport=m.Transport(Mock(),Mock());r=transport.requests[0];url=r['url'];clock=[0.]
            response=Response(contents[url],url);read=response.read1
            def delayed(size):
                data=read(size)
                if not data:clock[0]=61.
                return data
            response.read1=delayed;opener.open.side_effect=None;opener.open.return_value=response
            with patch.object(m.time,'monotonic',side_effect=lambda:clock[0]),self.assertRaisesRegex(ValueError,'wall deadline'):
                transport(url,r['compressed']['bytes'])

    def test14_socket_error_propagates_with_no_second_attempt(self):
        with fixture()as(_,_,_,opener):
            opener.open.side_effect=TimeoutError('synthetic socket timeout');transport=m.Transport(Mock(),Mock());r=transport.requests[0]
            for _ in range(2):
                with self.assertRaises(TimeoutError):transport(r['url'],r['compressed']['bytes'])
            self.assertEqual(opener.open.call_count,1)

    def test15_guard_denial_prevents_network_call(self):
        with fixture()as(_,_,_,opener):
            guard=Mock(side_effect=RuntimeError('synthetic lease lost'));transport=m.Transport(guard,Mock());r=transport.requests[0]
            with self.assertRaisesRegex(RuntimeError,'lease lost'):transport(r['url'],r['compressed']['bytes'])
            opener.open.assert_not_called()

    def test16_fixed_saved_input_rejects_extra_file_or_changed_cap(self):
        reference=next(iter(m.core.INPUTS.values()))
        with patch.object(m,'stable_read')as read:
            for row,cap in [(dict(reference,path='/unreviewed/file'),reference['bytes']),(reference,reference['bytes']+1)]:
                with self.assertRaises(ValueError):m.read_saved(row,cap,Mock())
            read.assert_not_called()

    def test17_atomic_output_preserves_complete_bytes_and_exclusive_final_name(self):
        with tempfile.TemporaryDirectory()as directory:
            output=object.__new__(m.Output);output.root=Path(directory);output.guard=Mock();output.used=4096;output.refs=[]
            result=output.save('glibc-exact.dsc',b'synthetic metadata')
            self.assertEqual((output.root/'glibc-exact.dsc').read_bytes(),b'synthetic metadata')
            self.assertEqual((output.root/'glibc-exact.dsc').stat().st_nlink,1)
            self.assertFalse((output.root/'publishing-glibc-exact.dsc').exists())
            self.assertEqual(output.refs,[result])
            with self.assertRaises(ValueError):output.save('glibc-exact.dsc',b'replace')
            with self.assertRaises(ValueError):output.save('source.tar.xz',b'archive')

    def test18_existing_final_file_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory()as directory:
            output=object.__new__(m.Output);output.root=Path(directory);output.guard=Mock();output.used=4096;output.refs=[]
            (output.root/'glibc-exact.dsc').write_bytes(b'original')
            with self.assertRaises(FileExistsError):output.save('glibc-exact.dsc',b'new')
            self.assertEqual((output.root/'glibc-exact.dsc').read_bytes(),b'original')


if __name__=='__main__':unittest.main(verbosity=2)
