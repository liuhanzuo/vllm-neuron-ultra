"""Independent Release-@ / compliance-23 wiring checks, source and memory only."""
import ast
import copy
import datetime as dt
import hashlib
from pathlib import Path
import unittest
from unittest.mock import patch
import check_dory_ubuntu_metadata_post_exit_independent_20261111_v4 as isolated
import dory_ubuntu_package_sources_v2 as discovery
import dory_ubuntu_gpgv_status_compliance_v1 as helper
import dory_ubuntu_metadata_adapter_v4 as adapter
import dory_ubuntu_metadata_input_binding_v2 as old_binding
import dory_ubuntu_metadata_input_binding_v3 as binding
import dory_ubuntu_metadata_worker_v3 as worker
import run_dory_ubuntu_metadata_root_v3 as parent
import dory_ubuntu_metadata_saved_audit_v3 as audit
import record_dory_ubuntu_metadata_exit_v3 as recorder
Q=Path(__file__).resolve().parent


class Independent(unittest.TestCase):
    def test_complete_modules_only_have_the_reviewed_wiring_and_two_pins(self):
        old=(Q/'dory_ubuntu_metadata_adapter_v3.py').read_text()
        expected=old.replace('dory_ubuntu_package_sources_v1','dory_ubuntu_package_sources_v2').replace(
            '7a9f1fe5754fb7b29dedad6d2532779f7af2252b2da0053683652b7569c608bd',
            'eb3bdcd4a532bec4905c12195813a95514a27ede39145e7e944be349a152688d')
        self.assertEqual(ast.dump(ast.parse(expected)),ast.dump(ast.parse((Q/'dory_ubuntu_metadata_adapter_v4.py').read_bytes())))
        old=ast.parse((Q/'dory_ubuntu_metadata_input_binding_v2.py').read_text().replace(
            'import dory_ubuntu_package_sources_v1 as discovery','import dory_ubuntu_package_sources_v2 as discovery'))
        new=ast.parse((Q/'dory_ubuntu_metadata_input_binding_v3.py').read_bytes())
        def pin_node(tree):return next(n for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='SOURCE_PINS'for t in n.targets))
        self.assertTrue(all(binding.SOURCE_PINS.get(p)==v for p,v in old_binding.SOURCE_PINS.items()))
        self.assertEqual(set(binding.SOURCE_PINS)-set(old_binding.SOURCE_PINS),{
            str(Q/'dory_ubuntu_package_sources_v2.py'),str(Q/'dory_ubuntu_metadata_adapter_v4.py')})
        self.assertEqual(len(pin_node(new).value.keys)-len(pin_node(old).value.keys),2)
        pin_node(new).value=pin_node(old).value;self.assertEqual(ast.dump(old),ast.dump(new))
        replacements={'dory_ubuntu_metadata_input_binding_v2':'dory_ubuntu_metadata_input_binding_v3',
            'dory_ubuntu_package_sources_v1':'dory_ubuntu_package_sources_v2',
            'dory_ubuntu_metadata_adapter_v3':'dory_ubuntu_metadata_adapter_v4',
            'dory_ubuntu_metadata_worker_v2':'dory_ubuntu_metadata_worker_v3',
            'dory_ubuntu_metadata_saved_audit_v2':'dory_ubuntu_metadata_saved_audit_v3'}
        for stem in ('dory_ubuntu_metadata_worker','run_dory_ubuntu_metadata_root',
                     'dory_ubuntu_metadata_saved_audit','record_dory_ubuntu_metadata_exit'):
            text=(Q/(stem+'_v2.py')).read_text()
            for before,after in replacements.items():text=text.replace(before,after)
            self.assertEqual(ast.dump(ast.parse(text)),ast.dump(ast.parse((Q/(stem+'_v3.py')).read_bytes())),stem)

    def test_actual_imports_source_authentication_and_old_pin_retention(self):
        self.assertIs(adapter.discovery,discovery);self.assertIs(binding.discovery,discovery)
        self.assertIs(audit.discovery,discovery);self.assertIs(audit.adapter,adapter)
        self.assertIs(worker.inputs,binding);self.assertIs(parent.inputs,binding)
        self.assertIs(recorder.binding,binding);self.assertIs(recorder.audit,audit)
        self.assertIs(adapter.parse_status,helper.parse_status)
        self.assertEqual(adapter.DISCOVERY_SHA,hashlib.sha256(Path(discovery.__file__).read_bytes()).hexdigest())
        for path,wanted in binding.SOURCE_PINS.items():
            self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(),wanted)
        self.assertEqual(adapter.URL_FILES,helper.original.URL_FILES)
        for name in ('free','memory_free','drain','release_after_drain','preflight_history'):
            self.assertIs(getattr(parent,name),getattr(binding.runtime.lifecycle,name))

    def test_complete_replay_combines_eighteen_release_at_paths_and_three_compliance_statuses(self):
        icons=[component+'/dep11/icons-'+size+'@2.tar'+suffix
            for component in ('main','multiverse','universe')for size in ('128x128','48x48','64x64')for suffix in ('','.gz')]
        addition=b''.join((' '+'d'*64+' 123 '+path+'\n').encode()for path in icons)
        annotation=helper.COMPLIANCE+b'\n'
        status=isolated.authored.good_status()+annotation
        original_fixture=isolated.fixture_module.Fixture
        with patch.object(isolated.authored,'Fixture',side_effect=lambda:original_fixture(mutate_release=lambda raw:raw+addition)),\
             patch.object(isolated.authored,'good_status',return_value=status):
            f=isolated.authored.fixture()
        before=copy.deepcopy(f[4]);calls=[]
        original_indexes=discovery.release_indexes
        def indexes(raw,suite,now):
            self.assertTrue(raw.endswith(addition));result=original_indexes(raw,suite,now)
            self.assertEqual(set(result[0]),{c+'/binary-amd64/Packages'+s for c in ('main','universe')for s in ('','.xz')})
            calls.append(suite);return result
        with patch.object(adapter,'parse_status',wraps=helper.parse_status)as parsed,\
             patch.object(discovery,'release_indexes',side_effect=indexes):
            result=isolated.authored.run_replay(f)
        self.assertEqual(calls,[suite for base,suite in discovery.SOURCES]);self.assertEqual(parsed.call_count,3)
        self.assertTrue(all(call.args[0]==status for call in parsed.call_args_list))
        self.assertEqual(f[4],before);self.assertEqual(len(f[3]),19);self.assertEqual(len(audit.NESTED),27)
        downloads=[e['url']for e in f[3]if e['event']=='ubuntu_metadata_download']
        self.assertEqual(downloads,[base+'dists/'+suite+'/'+name for base,suite in discovery.SOURCES
            for name in ('Release','Release.gpg','main/binary-amd64/Packages.xz','universe/binary-amd64/Packages.xz')])
        self.assertTrue(result['entire_discovery_reproduced']);self.assertFalse(result['signature_verification_rerun'])
        self.assertFalse(result['package_content_authenticated']);self.assertEqual(result['unresolved_closed'],0)
        with self.assertRaises(ValueError):discovery.relative('pool/main/m/mock/libc6@2.deb')


if __name__=='__main__':
    with patch.object(dt,'datetime',isolated.FixedDateTime):unittest.main(verbosity=2)
