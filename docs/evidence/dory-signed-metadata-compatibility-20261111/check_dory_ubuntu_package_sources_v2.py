"""Release-only @ path tests with source-isolated synthetic metadata."""
import ast
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

# Source-only audit hook and synthetic replacement of the old fixture's four
# original JSON reads. No actual Release, package, keyring or process is used.
import check_dory_ubuntu_metadata_post_exit_independent_20261111_v1 as isolated
import dory_ubuntu_package_sources_v2 as candidate

fixture=isolated.fixture_module
Q=Path(__file__).resolve().parent
ICONS=[component+'/dep11/icons-'+size+'@2.tar'+suffix
       for component in ('main','multiverse','universe')
       for size in ('128x128','48x48','64x64')for suffix in ('','.gz')]


def add_paths(raw,paths):
    return raw+b''.join((' '+'d'*64+' 123 '+path+'\n').encode()for path in paths)


class Tests(unittest.TestCase):
    def test_only_new_release_validator_and_single_call_site_change(self):
        old=ast.parse((Q/'dory_ubuntu_package_sources_v1.py').read_bytes())
        new=ast.parse((Q/'dory_ubuntu_package_sources_v2.py').read_bytes())
        helper=next(n for n in new.body if isinstance(n,ast.FunctionDef)and n.name=='release_relative')
        self.assertEqual(sum(isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='release_relative'
                             for n in ast.walk(new)),1)
        new.body.remove(helper)
        calls=0
        for node in ast.walk(new):
            if isinstance(node,ast.Call)and isinstance(node.func,ast.Name)and node.func.id=='release_relative':
                node.func.id='relative';calls+=1
        self.assertEqual(calls,1);self.assertEqual(ast.dump(old),ast.dump(new))

    def test_all_eighteen_observed_dep11_name_shapes_allowed_only_in_release(self):
        self.assertEqual(len(ICONS),18)
        for path in ICONS:
            with self.subTest(path=path):
                self.assertEqual(candidate.release_relative(path),path)
                with self.assertRaises(ValueError):candidate.relative(path)
        f=fixture.Fixture();url=candidate.SOURCES[0][0]+'dists/jammy/Release';raw=f.urls[url]
        selected,dates=candidate.release_indexes(add_paths(raw,ICONS),'jammy',fixture.NOW)
        baseline=candidate.release_indexes(raw,'jammy',fixture.NOW)
        self.assertEqual((selected,dates),baseline)
        self.assertEqual(set(selected),{c+'/binary-amd64/Packages'+suffix for c in ('main','universe')for suffix in ('','.xz')})

    def test_malicious_release_paths_and_segments_remain_rejected(self):
        f=fixture.Fixture();raw=f.urls[candidate.SOURCES[0][0]+'dists/jammy/Release']
        for path in ('','/main/dep11/icons@2.tar','//evil.example/icons@2.tar',
                     'main//dep11/icons@2.tar','main/./icons@2.tar','main/../icons@2.tar',
                     './main/icons@2.tar','../main/icons@2.tar','main/dep11/',
                     'main/dep11/.','main/dep11/..','https://evil.example/icons@2.tar',
                     'main/dep11/icons@2.tar?x=1','main/dep11/icons@2.tar#fragment',
                     'main/dep11/icons@2.tar:extra','main\\dep11\\icons@2.tar',
                     'main/%2e%2e/icons@2.tar','main/dep11/icons%402.tar',
                     'main/dep11/icons @2.tar','main/dep11/icons\t@2.tar',
                     'main/dep11/icons\n@2.tar','main/dep11/icons\r@2.tar',
                     'main/dep11/icons\x00@2.tar','main/dep11/雪@2.tar','@main/icons.tar'):
            with self.subTest(path=repr(path)):
                with self.assertRaises(ValueError):candidate.release_relative(path)
                with self.assertRaises(ValueError):candidate.release_indexes(add_paths(raw,[path]),'jammy',fixture.NOW)

    def test_complete_discovery_keeps_fixed_twelve_urls_and_never_fetches_icons(self):
        f=fixture.Fixture(mutate_release=lambda raw:add_paths(raw,ICONS))
        result=f.run()
        expected=[base+'dists/'+suite+'/'+name for base,suite in candidate.SOURCES
            for name in ('Release','Release.gpg','main/binary-amd64/Packages.xz','universe/binary-amd64/Packages.xz')]
        self.assertEqual(f.calls,expected);self.assertEqual(len(f.calls),12)
        self.assertEqual(result['status'],'exact_addresses_ready');self.assertEqual(len(result['packages']),6)
        self.assertTrue(all('@'not in url for url in f.calls))
        self.assertFalse(result['actual_archive_download']);self.assertFalse(result['package_content_authenticated'])
        self.assertEqual(result['unresolved_closed'],0)

    def test_package_filename_at_sign_is_still_rejected_in_actual_discovery(self):
        f=fixture.Fixture(mutate_release=lambda raw:add_paths(raw,ICONS),
            mutate_index=lambda raw,suite,component:fixture.stanza('libc6',filename='pool/main/m/mock/libc6@2.deb')if component=='main'else raw)
        with self.assertRaisesRegex(ValueError,'Safe relative metadata path'):f.run()
        self.assertEqual(len(f.calls),3)

    def test_fixed_index_missing_duplicate_and_digest_guards_remain(self):
        f=fixture.Fixture();raw=f.urls[candidate.SOURCES[0][0]+'dists/jammy/Release']
        target=next(line for line in raw.splitlines(keepends=True)if line.endswith(b' main/binary-amd64/Packages.xz\n'))
        for bad in (raw+target,raw.replace(target,b''),raw.replace(b'main/binary-amd64/Packages.xz',b'main/binary-amd64/Packages@2.xz'),
                    add_paths(raw,ICONS).replace(b' '+b'd'*64+b' 123 main/dep11/',b' '+b'z'*64+b' 123 main/dep11/',1)):
            with self.assertRaises(ValueError):candidate.release_indexes(bad,'jammy',fixture.NOW)


if __name__=='__main__':
    # Run all ten unchanged original source-discovery tests on v2 plus six
    # focused additions; synthetic INPUT_PINS substitute only the fixture IO.
    with patch.object(candidate,'INPUT_PINS',copy.deepcopy(isolated.discovery.INPUT_PINS)),patch.object(fixture,'m',candidate):
        suite=unittest.TestSuite((unittest.defaultTestLoader.loadTestsFromTestCase(fixture.Tests),
                                 unittest.defaultTestLoader.loadTestsFromTestCase(Tests)))
        result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful()else 1)
