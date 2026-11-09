"""Route exact r9 tiled transpose outputs through the extended strict reader."""
from pathlib import Path
import nemotron9b_compiler_asset_links_v4 as original
from research import nemotron9b_compiler_dynamic_links_v2 as generated

load_policy = original.load_policy
ASSET_REF = original.ASSET_REF
CANONICAL_SHA = original.CANONICAL_SHA
verify_assets = original.verify_assets
scope = original.scope
require = original.require


def link_record(path, compiler_root, policy, *, hash_content=True, guard=lambda: None):
    original.policy_shape(policy)
    path, root = Path(path), Path(compiler_root)
    require(path.is_relative_to(root), 'Link outside compile-work')
    if 'sgLnk' not in path.relative_to(root).parts:
        return original.installed.link_record(path, root, policy)
    if path.name in {row['name'] for row in policy['assets'].values()}:
        return original.installed_sglnk_record(path, root, policy, guard=guard)
    return generated.link_record(path, root, hash_content=hash_content, guard=guard)
