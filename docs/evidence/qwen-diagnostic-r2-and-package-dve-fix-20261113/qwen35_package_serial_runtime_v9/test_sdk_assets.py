"""Stdlib/tempfile controls only; never read real SDK binary assets."""
import ast
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_source import HERE, Q, c, io
from test_compiler_storage import CompilerStorage
from qwen35_package_serial_runtime_v9 import compiler_assets as ca, budget


class SDKAssets(unittest.TestCase):
    @contextmanager
    def fixture(self):
        with CompilerStorage().fixture() as (spec, reviewed, control, sleep):
            sdk = control.parent/'sdk-fixture'; sdk.mkdir()
            dve = control.parent/'dve-fixture'; dve.mkdir()
            assets = {}
            for name in ca.NAMES:
                path = (sdk if name in ca.PWP_NAMES else dve)/name; raw = (name+'-synthetic-only').encode(); path.write_bytes(raw)
                assets[str(path)] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw),
                    identity=dict(zip(ca.ATTRIBUTES, ca.attrs(path.lstat()))))
            with patch.object(ca, 'ASSET_ROOT', sdk), patch.object(ca, 'DVE_ROOT', dve), patch.object(ca, 'manifest', return_value={'assets':assets}):
                yield spec, reviewed, control, sleep, sdk, assets

    def link(self, spec, sdk, name='act_info.json', bucket=0, sg='sg00'):
        path = Path(spec['case']['compiler_workdir'])/f'token_generation_model/_tp0_bk{bucket}/neuronxcc-fixture/{sg}'/name
        path.parent.mkdir(parents=True, exist_ok=True); path.symlink_to((sdk if name in ca.PWP_NAMES else ca.DVE_ROOT)/name)
        return path

    def test_all20_and_repeated_targets_fully_counted_once(self):
        with self.fixture() as (spec, reviewed, control, sleep, sdk, assets):
            for name in ca.NAMES:
                self.link(spec,sdk,name); self.link(spec,sdk,name,bucket=5)
            result, inventory = io.storage_scan(spec)
            self.assertEqual(len(result['compiler_links']),40)
            self.assertEqual(result['existing_sdk_targets'],sorted(assets))
            charged = sum(ca.target_path(name).stat().st_blocks*512 for name in ca.NAMES)
            self.assertEqual(result['existing_sdk_allocated_bytes'],charged)
            self.assertEqual(result['allocated_bytes'],sum(row[6]*512 for row in inventory))
            self.assertEqual(result['output_allocated_bytes'],result['allocated_bytes']-charged)
            for target in assets:self.assertEqual(sum(row[0]==target for row in inventory),1)
            self.assertEqual(io.storage(spec,terminal=True),result);sleep.assert_not_called()

    def test_constant_branch_remains_metadata_only(self):
        from test_compiler_links_terminal import CompilerLinks
        with self.fixture() as (spec, reviewed, control, sleep, sdk, assets):
            CompilerLinks().link(spec)
            with patch.object(ca,'asset',side_effect=AssertionError('constant must not read SDK')),\
                 patch.object(io.os,'read',side_effect=AssertionError('no constant payload read')):
                result,_=io.storage_scan(spec)
            self.assertEqual(result['existing_sdk_allocated_bytes'],0)
            self.assertEqual(result['compiler_links'][0]['payload_bytes_read'],0)

    def test_exact_target_only_rejects_relative_traversal_other_asset_and_unlisted_name(self):
        with self.fixture() as (spec, reviewed, control, sleep, sdk, assets):
            path=self.link(spec,sdk)
            for target in ('../act_info.json',str(sdk/'silu_and_others.json'),str(sdk/'../sdk-fixture/act_info.json'),'/tmp/act_info.json'):
                path.unlink();path.symlink_to(target)
                with self.subTest(target=target),self.assertRaisesRegex(ValueError,'exact installed'):io.storage_scan(spec)
            path.unlink();unknown=self.link(spec,sdk,'unexpected.json')
            with self.assertRaisesRegex(ValueError,'observed bucket'):io.storage_scan(spec)

    def test_wrong_scope_and_external_links_in_other_trees_rejected(self):
        with self.fixture() as (spec, reviewed, control, sleep, sdk, assets):
            path=self.link(spec,sdk,bucket=6)
            with self.assertRaisesRegex(ValueError,'observed bucket'):io.storage_scan(spec)
            path.unlink()
            for kind in ('artifact','compiler_cache'):
                parent=Path(spec['case'][kind]);parent.mkdir();bad=parent/'act_info.json';bad.symlink_to(sdk/'act_info.json')
                with self.assertRaisesRegex(ValueError,'symlinks forbidden'):io.storage_scan(spec)
                bad.unlink()

    def test_target_symlink_hardlink_directory_fifo_missing_rejected(self):
        for kind in ('symlink','hardlink','directory','fifo','missing'):
            with self.subTest(kind=kind),self.fixture() as (spec,reviewed,control,sleep,sdk,assets):
                path=self.link(spec,sdk);target=sdk/path.name
                if kind=='hardlink':os.link(target,sdk/'another')
                else:
                    target.unlink()
                    if kind=='symlink':target.symlink_to(sdk/'silu_and_others.json')
                    elif kind=='directory':target.mkdir()
                    elif kind=='fifo':os.mkfifo(target)
                with self.assertRaises(ValueError):io.storage_scan(spec)
                sleep.assert_not_called()

    def test_any_target_ancestry_symlink_refused(self):
        with self.fixture() as (spec,reviewed,control,sleep,sdk,assets):
            self.link(spec,sdk);renamed=sdk.with_name('moved-sdk');sdk.rename(renamed);sdk.symlink_to(renamed)
            with self.assertRaises(OSError):io.storage_scan(spec)

    def test_link_ancestry_symlink_refused(self):
        with self.fixture() as (spec,reviewed,control,sleep,sdk,assets):
            path=self.link(spec,sdk);original=path.parent;new=original.with_name('sg01');original.rename(new);original.symlink_to(new)
            with self.assertRaises(OSError):ca.compiler_sdk_link(path,spec,2**20,assets)

    def test_every_compiler_ancestor_requires_owner_and_same_device(self):
        for kind in ('owner', 'device'):
            with self.subTest(kind=kind), self.fixture() as (spec,reviewed,control,sleep,sdk,assets):
                path=self.link(spec,sdk); ancestor=path.parents[2]; selected=ancestor.stat().st_ino
                original=ca.os.fstat
                def changed(fd):
                    value=original(fd)
                    if value.st_ino != selected:return value
                    fields={key:getattr(value,key) for key in ca.ATTRIBUTES}
                    fields['st_uid' if kind=='owner' else 'st_dev'] += 1
                    return SimpleNamespace(**fields)
                with patch.object(ca.os,'fstat',side_effect=changed),self.assertRaisesRegex(ValueError,'owned compiler ancestry'):
                    ca.compiler_sdk_link(path,spec,2**20,assets)

    def test_directory_replaced_during_asset_read_rejected_before_success(self):
        with self.fixture() as (spec,reviewed,control,sleep,sdk,assets):
            path=self.link(spec,sdk); original=ca.asset; done=[]
            def racing(*args,**kwargs):
                result=original(*args,**kwargs)
                if not done:
                    done.append(True);path.parent.rename(path.parent.with_name('moved'));path.parent.mkdir()
                return result
            with patch.object(ca,'asset',side_effect=racing),self.assertRaisesRegex(ValueError,'directory edge changed'):
                ca.compiler_sdk_link(path,spec,2**20,assets)

    def test_wrong_hash_despite_matching_current_stat_rejected(self):
        with self.fixture() as (spec,reviewed,control,sleep,sdk,assets):
            self.link(spec,sdk);target=sdk/'act_info.json';target.write_bytes(b'y'*target.stat().st_size)
            assets[str(target)]['identity']=dict(zip(ca.ATTRIBUTES,ca.attrs(target.stat())))
            with self.assertRaisesRegex(ValueError,'content/hash'):io.storage_scan(spec)

    def test_same_content_replacement_identity_rejected(self):
        with self.fixture() as (spec,reviewed,control,sleep,sdk,assets):
            self.link(spec,sdk);target=sdk/'act_info.json';raw=target.read_bytes();other=sdk/'replacement';other.write_bytes(raw);other.replace(target)
            with self.assertRaisesRegex(ValueError,'identity changed'):io.storage_scan(spec)

    def test_target_changed_during_read_never_returns_usage(self):
        with self.fixture() as (spec,reviewed,control,sleep,sdk,assets):
            self.link(spec,sdk);target=sdk/'act_info.json';read=ca.os.read;done=[]
            def racing(*args):
                result=read(*args)
                if result and not done:done.append(True);target.write_bytes(b'z'*target.stat().st_size)
                return result
            with patch.object(ca.os,'read',side_effect=racing),self.assertRaisesRegex(ValueError,'content/hash'):io.storage_scan(spec)

    def test_link_retarget_during_content_read_never_returns_usage(self):
        with self.fixture() as (spec,reviewed,control,sleep,sdk,assets):
            path=self.link(spec,sdk);original=ca.asset;done=[]
            def racing(*args,**kwargs):
                result=original(*args,**kwargs)
                if not done:done.append(True);path.unlink();path.symlink_to(sdk/'silu_and_others.json')
                return result
            with patch.object(ca,'asset',side_effect=racing),self.assertRaises(io.CompilerTreeRace):io.storage_scan(spec)

    def test_last_target_is_rechecked_after_complete_inventory(self):
        with self.fixture() as (spec,reviewed,control,sleep,sdk,assets):
            target=ca.target_path(ca.NAMES[-1]);self.link(spec,sdk,target.name);original=ca.compiler_sdk_link;calls=[]
            def racing(*args,**kwargs):
                result=original(*args,**kwargs);calls.append(True)
                if len(calls)==2:target.write_bytes(b'z'*target.stat().st_size)
                return result
            with patch.object(ca,'compiler_sdk_link',side_effect=racing),self.assertRaisesRegex(ValueError,'identity changed'):io.storage_scan(spec)

    def test_original_class_entry_and_single_file_caps_apply(self):
        for kind in ('class','entries','single'):
            with self.subTest(kind=kind),self.fixture() as (spec,reviewed,control,sleep,sdk,assets):
                self.link(spec,sdk)
                key={'class':'allocated_bytes','entries':'entries','single':'single_file_bytes'}[kind]
                reviewed['classes']['compiler_workdir'][key]=1
                with self.assertRaises(ValueError):io.storage_scan(spec)

    def test_existing_root_assets_do_not_reduce_nvme_reservation(self):
        with self.fixture() as (spec,reviewed,control,sleep,sdk,assets):
            self.link(spec,sdk);usage,_=io.storage_scan(spec)
            review=copy.deepcopy(reviewed);review['headroom_bytes']={'root':17,'nvme':19}
            result=budget.remaining_reserve(spec,usage,review)
            self.assertEqual(result['root'],17)
            self.assertEqual(result['nvme'],max(0,c.LIMITS['total_allocated_output_bytes']+sum(v['allocated_bytes'] for v in review['classes'].values())-usage['output_allocated_bytes'])+19)


    def test_all20_verify_all_reads_exact_two_roots(self):
        with self.fixture() as (spec, reviewed, control, sleep, sdk, assets):
            records=ca.verify_all(spec)
            self.assertEqual({r['path'] for r in records},set(assets))
            self.assertEqual({Path(r['path']).parent for r in records},{sdk,ca.DVE_ROOT})
            self.assertEqual(len(records),20)

    def test_dve_cannot_be_relocated_under_pwp_or_other_root(self):
        with self.fixture() as (spec, reviewed, control, sleep, sdk, assets):
            for name in ca.DVE_NAMES:
                link=self.link(spec,sdk,name)
                for target in (sdk/name,ca.DVE_ROOT/'..'/'dve-fixture'/name,Path('/tmp')/name):
                    link.unlink();link.symlink_to(target)
                    with self.subTest(name=name,target=str(target)),self.assertRaisesRegex(ValueError,'exact installed'):
                        io.storage_scan(spec)
                link.unlink()
            link=self.link(spec,sdk,'act_info.json');link.unlink();link.symlink_to(ca.DVE_ROOT/'act_info.json')
            with self.assertRaisesRegex(ValueError,'exact installed'):io.storage_scan(spec)

    def test_every_dve_content_pin_and_identity_enforced(self):
        for name in ca.DVE_NAMES:
            for field in ('hash','identity','missing'):
                with self.subTest(name=name,field=field),self.fixture() as (spec,reviewed,control,sleep,sdk,assets):
                    self.link(spec,sdk,name);target=ca.target_path(name)
                    if field=='hash':assets[str(target)]['sha256']='0'*64
                    elif field=='identity':assets[str(target)]['identity']['st_ino']+=1
                    else:target.unlink()
                    with self.assertRaises(ValueError):io.storage_scan(spec)

    def test_dve_symlink_ancestry_and_target_refused(self):
        for mode in ('ancestor','target'):
            with self.subTest(mode=mode),self.fixture() as (spec,reviewed,control,sleep,sdk,assets):
                name=ca.DVE_NAMES[0];self.link(spec,sdk,name);target=ca.target_path(name)
                if mode=='ancestor':
                    original=target.parent;renamed=original.with_name('moved-dve');original.rename(renamed);original.symlink_to(renamed)
                    with self.assertRaises(OSError):io.storage_scan(spec)
                else:
                    target.unlink();target.symlink_to(sdk/'act_info.json')
                    with self.assertRaises(ValueError):io.storage_scan(spec)

    def test_all20_assets_never_reduce_nvme_reserve(self):
        with self.fixture() as (spec,reviewed,control,sleep,sdk,assets):
            for name in ca.NAMES:self.link(spec,sdk,name)
            usage,_=io.storage_scan(spec)
            reserve=budget.remaining_reserve(spec,usage,reviewed)
            expected=c.LIMITS['total_allocated_output_bytes']+sum(row['allocated_bytes'] for row in reviewed['classes'].values())-usage['output_allocated_bytes']+reviewed['headroom_bytes']['nvme']
            self.assertEqual(reserve['nvme'],expected)
            self.assertGreater(usage['existing_sdk_allocated_bytes'],0)


class ManifestAndPreservation(unittest.TestCase):
    def test_real_manifest_is_metadata_only_fixed20(self):
        raw=(HERE/'compiler_assets.json').read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),ca.MANIFEST_SHA256)
        spec={'source_sha256':{str(HERE/'compiler_assets.json'):ca.MANIFEST_SHA256}}
        with patch.object(io,'raw',return_value=raw):value=ca.manifest(spec)
        self.assertEqual(len(value['assets']),20)
        self.assertEqual(sum(v['bytes'] for v in value['assets'].values()),387197)
        self.assertEqual(sum(v['identity']['st_blocks']*512 for v in value['assets'].values()),417792)
        for bad in ({'source_sha256':{}},{'source_sha256':{str(HERE/'compiler_assets.json'):'0'*64}}):
            with self.assertRaises(ValueError),patch.object(io,'raw',side_effect=AssertionError('must fail before read')):ca.manifest(bad)

    def test_modified_manifest_refused(self):
        spec={'source_sha256':{str(HERE/'compiler_assets.json'):ca.MANIFEST_SHA256}}
        with patch.object(io,'raw',return_value=b'{}'),self.assertRaisesRegex(ValueError,'manifest changed'):ca.manifest(spec)

    def test_all20_checked_prelock_without_resources_or_lock_on_failure(self):
        spec={}
        with patch.object(budget,'validate_spec'),patch.object(io,'read',return_value=spec),\
             patch.object(budget,'reviewed_budget'),patch.object(budget,'unchanged_inputs'),patch.object(budget,'fresh_paths'),\
             patch.object(ca,'verify_all',side_effect=ValueError('asset mismatch')) as check,\
             patch.object(io,'resources',side_effect=AssertionError('must not reach resources')):
            with self.assertRaisesRegex(ValueError,'asset mismatch'):budget.prelock({}, {})
            check.assert_called_once_with(spec)

    def test_frozen_v8_members_and_preparer_unchanged(self):
        for path,row in json.loads((HERE/'predecessor_pins.json').read_text())['files'].items():
            self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(),row['sha256'],path)

    def test_all_science_stop_terminal_lock_and_other_functions_unchanged(self):
        old=Q/'qwen35_package_serial_runtime_v7'
        for name in c.RUNTIME_FILES:
            if name in ('contract.py','io.py','budget.py','run.py','compiler_assets.py','compiler_assets.json'):continue
            self.assertEqual((HERE/name).read_bytes(),(old/name).read_bytes(),name)
        def nodes(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        for name,changed in [('io.py',{'storage_scan'}),('budget.py',{'prelock','remaining_reserve'}),('contract.py',set())]:
            before,after=nodes(old/name),nodes(HERE/name)
            for key in before:
                if key not in changed:self.assertEqual(before[key],after[key],name+':'+key)


if __name__=='__main__':unittest.main(verbosity=2)
