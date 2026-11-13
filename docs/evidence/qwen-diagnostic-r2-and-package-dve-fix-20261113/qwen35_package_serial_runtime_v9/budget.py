"""Explicit external reviewed capacity interface; no guessed compile allowance.

This module reads metadata/stat and, prelock, verifies only the fixed16 small
installed SDK PWP/DVE assets. It never opens model, checkpoint, graph, tensor or
packet contents. Prepare/build retain the original actual child ACK.
"""
from pathlib import Path
import json
import os
import stat

from .contract import (HERE, LIMITS, ROOT, budget, control_root, require,
                       validate_spec, phase_properties)

CASE_KEYS = ('model_path', 'artifact', 'compiler_cache', 'compiler_workdir')
CLASSES = ('view', 'artifact', 'compiler_cache', 'compiler_workdir', 'build_auxiliary')


def roots(spec):
    return (Path(spec['output_root']), control_root(spec), *(Path(spec['case'][k]) for k in CASE_KEYS))


def parent_identity(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and not any(p.is_symlink() for p in (path, *path.parents)),
            'ordinary exact directory ancestry required')
    s = path.lstat()
    require(stat.S_ISDIR(s.st_mode), 'existing ordinary parent directory required')
    return dict(realpath=str(path), device=s.st_dev, inode=s.st_ino)


def validate_budget(spec, value):
    keys = {'schema', 'reviewer', 'passed', 'blockers', 'case', 'output_root', 'control_root',
            'classes', 'headroom_bytes', 'parent_identities', 'source_files', 'build_file_limit_bytes'}
    require(type(value) is dict and set(value) == keys and value['schema'] == 'qwen35_package_reviewed_budget_v1'
            and value['reviewer'] == '/root' and value['passed'] is True and value['blockers'] == [],
            'explicit independently reviewed build/storage budget required; unbound refuses execution')
    require(value['case'] == spec['case'] and value['output_root'] == spec['output_root']
            and value['control_root'] == str(control_root(spec)), 'budget bound to all six exact fresh trees')
    require(type(value['classes']) is dict and set(value['classes']) == set(CLASSES), 'all finite package budget classes required')
    for row in value['classes'].values():
        require(type(row) is dict and set(row) == {'allocated_bytes', 'single_file_bytes', 'entries'}
                and all(type(n) is int and 0 < n < 2**63 for n in row.values())
                and row['single_file_bytes'] <= row['allocated_bytes'], 'finite positive per-class/single/entry caps required')
    require(value['classes']['build_auxiliary']['allocated_bytes'] >= 2 * LIMITS['leaf_log_bytes'],
            'prepare/build logs need their own reviewed allowance outside original 2 GiB science budget')
    require(value['build_file_limit_bytes'] == spec['build_file_limit_bytes']
            == max(value['classes'][k]['single_file_bytes'] for k in CLASSES), 'build LimitFSIZE matches reviewed file envelope')
    require(value['headroom_bytes'].keys() == {'root', 'nvme'}
            and all(type(x) is int and 0 < x < 2**63 for x in value['headroom_bytes'].values()), 'explicit positive root/NVMe headroom required')
    paths = roots(spec)
    require(len(set(paths)) == 6 and not any(a in b.parents for a in paths for b in paths if a != b), 'six disjoint nonnested trees required')
    parents = {str(p.parent) for p in paths}
    require(set(value['parent_identities']) == parents, 'all existing parent identities required')
    for path, row in value['parent_identities'].items():
        require(type(row) is dict and set(row) == {'realpath', 'device', 'inode'} and row['realpath'] == path,
                'exact parent identity schema')
    # This bounded text was derived from the pinned package provenance literal.
    from .io import raw
    originals = json.loads(raw(HERE / 'source_files.json'))
    require(set(value['source_files']) == set(originals), 'all original SOURCE_FILES metadata identities required')
    for name, row in value['source_files'].items():
        require(type(row) is dict and set(row) == {'pin', 'identity'} and row['pin'] == originals[name]
                and row['identity'].get('realpath') == str(Path('/mnt/nvme/models/Qwen3.5-2B') / name)
                and row['identity'].get('size') == originals[name]['size'], 'exact original source pin/identity required')
    return value


def reviewed_budget(spec):
    from .io import read
    require('resource_budget' in spec and spec['resource_budget'] is not None, 'unbound reviewed resource budget; no default capacity')
    return validate_budget(spec, read(spec['resource_budget']))


def unchanged_inputs(spec, value=None):
    from .io import identity
    value = reviewed_budget(spec) if value is None else value
    nvme_device = Path('/mnt/nvme').stat().st_dev
    for path, expected in value['parent_identities'].items():
        require(parent_identity(path) == expected and expected['device'] == nvme_device,
                'reviewed output parent changed or is outside reserved NVMe filesystem')
    for name, row in value['source_files'].items():
        require(identity(row['identity']['realpath']) == row['identity'], 'original model/view source stat identity changed')
    for path, row in spec['binary_inputs'].items():
        require(identity(path) == row['identity'], 'original scientific binary stat identity changed')
    if (Path(spec['output_root']) / 'build-immutable-identities.json').exists():
        from .wiring import check_build_seal
        check_build_seal(spec)
    return value


def fresh_paths(spec):
    for path in roots(spec):
        require(not path.exists() and not path.is_symlink(), 'all six run trees must be absent: ' + str(path))
    return True


def prelock(spec_ref, review_ref):
    """Fail missing/invalid budgets and inputs before acquiring original lock."""
    from .io import read, resources
    spec = read(spec_ref, LIMITS['spec_envelope_bytes'])
    review = read(review_ref, LIMITS['spec_envelope_bytes'])
    validate_spec(spec, review, spec_ref)
    value = reviewed_budget(spec)
    unchanged_inputs(spec, value); fresh_paths(spec)
    from .compiler_assets import verify_all
    verify_all(spec)
    resources(spec, reserve=True)
    return spec


def check_view_link(path, spec, value):
    """Only prepared SOURCE_FILES links to exact pinned original targets."""
    from .io import identity
    root = Path(spec['case']['model_path'])
    require(path.parent == root and path.name != 'config.json' and path.name in value['source_files'],
            'only exact prepared view SOURCE_FILES symlinks allowed')
    row = value['source_files'][path.name]
    target = row['identity']['realpath']
    require(os.readlink(path) == target and identity(target) == row['identity'], 'view link target differs from pinned original file')
    s = path.lstat()
    return dict(path=str(path), device=s.st_dev, inode=s.st_ino, size=s.st_size,
                mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns, target=target, target_identity=row['identity'])


def remaining_reserve(spec, usage, value):
    # All six roots are independently bound to the NVMe parent identities.
    additional = sum(row['allocated_bytes'] for row in value['classes'].values())
    # Fixed SDK assets live on the root disk already. Although conservatively
    # charged to the existing compiler class, they do not consume this run's
    # NVMe output reservation. Never subtract them as produced NVMe bytes.
    remaining = max(0, LIMITS['total_allocated_output_bytes'] + additional -
                    usage.get('output_allocated_bytes', usage['allocated_bytes']))
    return {'root': value['headroom_bytes']['root'], 'nvme': remaining + value['headroom_bytes']['nvme']}
