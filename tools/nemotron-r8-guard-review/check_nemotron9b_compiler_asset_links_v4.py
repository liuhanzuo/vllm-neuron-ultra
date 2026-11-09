"""Filesystem/source regressions only; no native SDK or model execution."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

R = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(R), str(R / 'research')]
import nemotron9b_compiler_asset_links_v3 as old
import nemotron9b_compiler_asset_links_v4 as new
import nemotron9b_package_storage_inventory_candidate_v5 as inventory
import check_nemotron9b_compiler_dynamic_links_v1 as previous


class GeneratedRegression(previous.GeneratedChecks):
    def setUp(self):
        for name, value in [('combined', new), ('inventory', inventory)]:
            change = patch.object(previous, name, value)
            change.start()
            self.addCleanup(change.stop)
        super().setUp()


class InstalledRegression(previous.InstalledRegression):
    def setUp(self):
        for name, value in [('combined', new), ('inventory', inventory)]:
            change = patch.object(previous, name, value)
            change.start()
            self.addCleanup(change.stop)
        super().setUp()


class SgLnkChecks(GeneratedRegression):
    def installed_link(self, target=None, subgraph='sg00'):
        target = target or next(iter(self.policy['assets']))
        path = self.link_dir / subgraph / Path(target).name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(target)
        return path

    def test_reproduces_r8_rejection_then_accepts_exact_policy(self):
        path = self.installed_link()
        with self.assertRaisesRegex(RuntimeError, 'Reviewed generated numeric basename'):
            old.link_record(path, self.compiler, self.policy, hash_content=False)
        row = new.link_record(path, self.compiler, self.policy, hash_content=False)
        self.assertEqual(row['target'], self.policy['assets'][row['symlink']])
        self.assertEqual(row['compiler_asset_kind'], 'installed_sglnk')
        self.assertGreater(self.scan(), 0)

    def test_all83_original_targets_in_new_container(self):
        for index, target in enumerate(self.policy['assets']):
            path = self.installed_link(target, f'sg{index:02d}')
            row = new.link_record(path, self.compiler, self.policy)
            self.assertEqual(row['target'], self.policy['assets'][target])

    def test_same_basename_unpinned_target_rejected(self):
        path = self.installed_link()
        target = self.unit / path.name
        target.write_bytes(b'synthetic untrusted resource')
        path.unlink()
        path.symlink_to(target)
        self.reject(lambda: new.link_record(path, self.compiler, self.policy))

    def test_relative_installed_target_rejected(self):
        path = self.installed_link()
        path.unlink()
        path.symlink_to('../' + path.name)
        self.reject(lambda: new.link_record(path, self.compiler, self.policy))

    def test_installed_scope_not_broadened(self):
        for subgraph in ['sg100', 'other', 'sg00/nested']:
            path = self.installed_link(subgraph=subgraph)
            self.reject(lambda: new.link_record(path, self.compiler, self.policy))

    def test_installed_parent_symlink_rejected(self):
        path = self.installed_link()
        path.parent.rename(self.link_dir / 'moved')
        path.parent.symlink_to(self.link_dir / 'moved', target_is_directory=True)
        self.reject(lambda: new.link_record(path, self.compiler, self.policy))

    def test_installed_link_replacement_rejected(self):
        path = self.installed_link()
        def mutate():
            target = path.readlink()
            path.unlink()
            path.symlink_to(target)
        self.reject(lambda: new.link_record(path, self.compiler, self.policy,
                                            guard=self.mutation_guard(mutate)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
