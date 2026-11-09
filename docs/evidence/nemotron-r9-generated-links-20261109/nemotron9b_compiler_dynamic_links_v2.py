"""Recognize the exact tiled transpose numeric filename observed in r9.

Frozen v18 and its original failure remain unchanged. The original no-follow
directory traversal, same-unit target, bounded file and stability checks are
reused with only the pure filename-to-target parser extended.
"""
from pathlib import Path
import re
from types import FunctionType
import nemotron9b_compiler_dynamic_links_v1 as original

TILED_VALUE = re.compile(
    r'value_sg(?P<sg>[0-9]{4})_tiled_pf_transposetiled_pf_transpose_sg(?P=sg)'
    r'_[0-9]{1,10}_identity_[0-9]{1,10}_CRSM\.npy')
require = original.require


def target_for(path, compiler_root):
    path, root = Path(path), Path(compiler_root)
    match = TILED_VALUE.fullmatch(path.name)
    if match is None:
        return original.target_for(path, root)
    require(root.is_absolute() and root.name == 'compile-work'
            and path.is_absolute() and path.is_relative_to(root), 'Exact compile-work scope')
    parts = path.relative_to(root).parts
    require(len(parts) == 6 and '..' not in parts
            and original.MODEL.fullmatch(parts[0]) and original.BUCKET.fullmatch(parts[1])
            and original.UNIT.fullmatch(parts[2]) and parts[3] == 'sgLnk'
            and original.SUBGRAPH.fullmatch(parts[4]), 'Exact tiled numeric link scope')
    index = int(match.group('sg'))
    require(index < 100, 'Original two-digit target subgraph scope')
    return root.joinpath(*parts[:3])/f'sg{index:02d}'/parts[5], 'generated_tiled_transpose_numeric'


# Bind an unchanged reader body to private globals; never mutate original
# module globals or its pinned source. This keeps every original I/O check.
_read = FunctionType(original.link_record.__code__, dict(vars(original), target_for=target_for),
                     'read_generated_link', original.link_record.__defaults__)
_read.__kwdefaults__ = original.link_record.__kwdefaults__


def link_record(path, compiler_root, *, hash_content=True, guard=lambda: None):
    return _read(path, compiler_root, hash_content=hash_content, guard=guard)
