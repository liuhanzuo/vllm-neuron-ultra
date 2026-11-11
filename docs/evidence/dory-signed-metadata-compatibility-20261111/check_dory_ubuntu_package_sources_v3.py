"""One exact empty second XZ stream; synthetic inputs and source reads only."""
import ast
import copy
import lzma
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import check_dory_ubuntu_metadata_post_exit_independent_20261111_v1 as isolated
import dory_ubuntu_package_sources_v3 as candidate
fixture=isolated.fixture_module
Q=Path(__file__).resolve().parent
TAIL=bytes.fromhex('fd377a585a000004e6d6b446000000001cdf44211fb6f37d010000000004595a')


class Tests(unittest.TestCase):
    def test_only_exact_tail_helper_and_one_validation_call_change(self):
        old=ast.parse((Q/'dory_ubuntu_package_sources_v2.py').read_bytes())
        new=ast.parse((Q/'dory_ubuntu_package_sources_v3.py').read_bytes())
        new.body=[n for n in new.body if not(isinstance(n,ast.FunctionDef)and n.name=='empty_xz_tail')
            and not(isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='EMPTY_XZ_TAIL'for t in n.targets))]
        class Normalize(ast.NodeTransformer):
            changed=0
            def visit_Call(self,node):
                if isinstance(node.func,ast.Name)and node.func.id=='empty_xz_tail':
                    self.changed+=1;return ast.UnaryOp(op=ast.Not(),operand=node.args[0])
                return self.generic_visit(node)
            def visit_Constant(self,node):
                if node.value=='Signed exact raw index; only the reviewed empty second stream':
                    node.value='Signed exact raw index; no trailing/concatenated stream'
                return node
        normalize=Normalize();new=normalize.visit(new)
        self.assertEqual(normalize.changed,1);self.assertEqual(ast.dump(old),ast.dump(new))

    def test_no_tail_or_exact_empty_tail_preserves_complete_twelve_request_discovery(self):
        baseline=fixture.Fixture().run()
        for tail in (b'',TAIL):
            f=fixture.Fixture(compressed_suffix=tail);result=f.run()
            self.assertEqual(len(f.calls),12);self.assertEqual(result['status'],'exact_addresses_ready')
            self.assertEqual([p['package']for p in result['packages']],[p['package']for p in baseline['packages']])
            self.assertFalse(result['actual_archive_download']);self.assertFalse(result['package_content_authenticated'])
            self.assertEqual(result['unresolved_closed'],0)
        self.assertEqual(len(TAIL),32);self.assertEqual(candidate.EMPTY_XZ_TAIL,TAIL)
        self.assertTrue(candidate.empty_xz_tail(b''));self.assertTrue(candidate.empty_xz_tail(TAIL))

    def test_padding_other_empty_stream_nonempty_third_truncated_and_garbage_rejected(self):
        variants=[b'\0'*4,b'\0'*32,b'\0',b'garbage',TAIL[:-1],TAIL+b'\0'*4,
            b'\0'*4+TAIL,TAIL+TAIL,TAIL+lzma.compress(b'third'),lzma.compress(b'nonempty'),
            lzma.compress(b'',check=lzma.CHECK_CRC32),lzma.compress(b'',check=lzma.CHECK_NONE),
            TAIL[:-1]+bytes([TAIL[-1]^1])]
        for tail in variants:
            with self.subTest(tail=tail.hex()),self.assertRaises(ValueError):fixture.Fixture(compressed_suffix=tail).run()
        for bad in (None,'',bytearray(),bytearray(TAIL)):
            with self.assertRaises(ValueError):candidate.empty_xz_tail(bad)

    def test_complete_signed_compressed_hash_and_size_include_tail(self):
        for kind in ('hash','size'):
            f=fixture.Fixture(compressed_suffix=TAIL)
            url=next(u for u in f.urls if u.endswith('main/binary-amd64/Packages.xz'))
            if kind=='hash':f.urls[url]=f.urls[url][:-1]+bytes([f.urls[url][-1]^1])
            else:f.urls[url]=f.urls[url][:-1]
            with self.subTest(kind=kind),self.assertRaisesRegex(ValueError,'Signed compressed index bytes'):f.run()

    def test_wrong_signed_raw_hash_or_size_still_rejected_with_exact_tail(self):
        for kind in ('hash','short','long'):
            def mutated(raw):
                lines=raw.splitlines(keepends=True)
                for n,line in enumerate(lines):
                    if line.endswith(b' main/binary-amd64/Packages\n'):
                        sha,size,path=line.split()
                        if kind=='hash':sha=b'0'*64
                        else:size=str(int(size)+(-1 if kind=='short'else 1)).encode()
                        lines[n]=b' '+sha+b' '+size+b' '+path+b'\n'
                return b''.join(lines)
            with self.subTest(kind=kind),self.assertRaisesRegex(ValueError,'Signed exact raw index'):
                fixture.Fixture(compressed_suffix=TAIL,mutate_release=mutated).run()

    def test_both_decoder_limits_and_empty_second_stream_are_checked(self):
        actual=candidate.lzma.LZMADecompressor;calls=[]
        class Observed:
            def __init__(self,**kwargs):
                self.decoder=actual(**kwargs);self.kwargs=kwargs
            def decompress(self,raw,**kwargs):
                calls.append((self.kwargs,kwargs,raw==TAIL))
                return self.decoder.decompress(raw,**kwargs)
            def __getattr__(self,name):return getattr(self.decoder,name)
        with patch.object(candidate.lzma,'LZMADecompressor',Observed):fixture.Fixture(compressed_suffix=TAIL).run()
        self.assertEqual(len(calls),12);self.assertEqual(sum(tail for _,_,tail in calls),6)
        for kwargs,limits,tail in calls:
            self.assertEqual(kwargs,dict(format=lzma.FORMAT_XZ,memlimit=64*candidate.MIB))
            if tail:self.assertEqual(limits,{'max_length':1})
            else:self.assertGreater(limits['max_length'],1)
        for output,eof,unused in ((b'x',True,b''),(b'',False,b''),(b'',True,b'extra')):
            fake=SimpleNamespace(decompress=lambda raw,**kwargs:output,eof=eof,unused_data=unused)
            with patch.object(candidate.lzma,'LZMADecompressor',return_value=fake),self.assertRaises(ValueError):
                candidate.empty_xz_tail(TAIL)

    def test_release_at_path_fix_and_original_package_filename_restriction_remain(self):
        def extra(raw):return raw+b' '+b'd'*64+b' 123 main/dep11/icons-128x128@2.tar.gz\n'
        f=fixture.Fixture(compressed_suffix=TAIL,mutate_release=extra)
        self.assertEqual(f.run()['status'],'exact_addresses_ready');self.assertEqual(len(f.calls),12)
        with self.assertRaises(ValueError):candidate.relative('pool/main/a@2.deb')


if __name__=='__main__':
    with patch.object(candidate,'INPUT_PINS',copy.deepcopy(isolated.discovery.INPUT_PINS)),patch.object(fixture,'m',candidate):
        suite=unittest.TestSuite((unittest.defaultTestLoader.loadTestsFromTestCase(fixture.Tests),
                                 unittest.defaultTestLoader.loadTestsFromTestCase(Tests)))
        result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful()else 1)
