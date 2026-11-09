"""Successor for the r8 sgLnk resource-routing failure; frozen v16 unchanged.

The compiler places both installed PWP/DVE resources and generated outputs
inside sgLnk. Installed basenames still require the exact original RECORD
policy target and identity. Generated outputs retain their separate policy.
No model, compiler, or numerical behavior is changed here.
"""
import os
from pathlib import Path
import stat

import nemotron9b_compiler_asset_links_v2 as installed
import nemotron9b_compiler_dynamic_links_v1 as generated

ASSET_REF = installed.ASSET_REF
CANONICAL_SHA = installed.CANONICAL_SHA
KEYS = installed.KEYS
attributes = installed.attributes
policy_shape = installed.policy_shape
load_policy = installed.load_policy
verify_assets = installed.verify_assets
scope = installed.scope
require = installed.require


def installed_sglnk_record(path, root, policy, *, guard=lambda: None):
    guard()
    parts = path.relative_to(root).parts
    require(root.is_absolute() and root.name == 'compile-work'
            and len(parts) == 6 and '..' not in parts
            and generated.MODEL.fullmatch(parts[0])
            and generated.BUCKET.fullmatch(parts[1])
            and generated.UNIT.fullmatch(parts[2])
            and parts[3] == 'sgLnk'
            and generated.SUBGRAPH.fullmatch(parts[4]),
            'Exact installed sgLnk model/bucket/unit/subgraph scope')
    handles = generated.ParentHandles.__new__(generated.ParentHandles)
    handles.fds = []
    try:
        handles.__init__(root)
        parent, name = handles.parent(path)
        before = os.stat(name, dir_fd=parent, follow_symlinks=False)
        require(stat.S_ISLNK(before.st_mode) and before.st_uid == os.getuid()
                and before.st_nlink == 1, 'Owned single installed sgLnk symlink')
        target = os.readlink(name, dir_fd=parent)
        require(target in policy['assets'], 'Exact RECORD-pinned installed sgLnk target')
        row = policy['assets'][target]
        require(name == row['name'] and target == row['path'],
                'Original installed sgLnk basename and absolute target')
        target_path = Path(target)
        require(target_path.resolve() == target_path
                and attributes(target_path.lstat()) == row['identity'],
                'Original installed sgLnk target identity')
        guard()
        require(attributes(os.stat(name, dir_fd=parent, follow_symlinks=False))
                == attributes(before) and os.readlink(name, dir_fd=parent) == target,
                'Installed sgLnk changed during observation')
        require(target_path.resolve() == target_path
                and attributes(target_path.lstat()) == row['identity'],
                'Installed sgLnk target changed during observation')
        handles.verify()
        return dict(path=str(path), compiler_asset=True,
                    compiler_asset_kind='installed_sglnk', symlink=target,
                    stat=attributes(before), target=dict(row))
    finally:
        handles.close()


def link_record(path, compiler_root, policy, *, hash_content=True,
                guard=lambda: None):
    installed.policy_shape(policy)
    path, root = Path(path), Path(compiler_root)
    require(path.is_relative_to(root), 'Link outside compile-work')
    parts = path.relative_to(root).parts
    if 'sgLnk' not in parts:
        return installed.link_record(path, root, policy)
    if path.name in {row['name'] for row in policy['assets'].values()}:
        return installed_sglnk_record(path, root, policy, guard=guard)
    return generated.link_record(path, root, hash_content=hash_content, guard=guard)
