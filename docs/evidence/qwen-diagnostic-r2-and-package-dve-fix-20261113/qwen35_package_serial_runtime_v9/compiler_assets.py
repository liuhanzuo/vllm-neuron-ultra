"""Exactly the 20 installed PWP/DVE assets observed in r5/r6, never executable input.

These typed fixed binary/text assets are authenticated by a frozen JSON source
pin, not misclassified as Python/text in the historical source map. No reads
or native imports occur at module import. No arbitrary absolute link allowance.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import stat

from .contract import HERE, SDK, require

MANIFEST_SHA256 = 'd46ae4f1e70e5b2f735704b0572429e3a6c9c44394a31cf50e59da0d2f97567d'
ASSET_ROOT = SDK / 'neuronxcc/pwp/pwp_bin_trainium'
DVE_ROOT = SDK / 'neuronxcc/dve/dve_bin_gen2'
GROUPS = ('derivative_silu_and_others', 'natural_log_exp_and_others',
          'sigmoid_and_others', 'silu_and_others', 'softplus_and_others')
PWP_NAMES = ('act_info.json',) + tuple(group + suffix for group in GROUPS
                                for suffix in ('.json', '_bkt.bin', '_ctrl.bin'))
DVE_NAMES = ('dve_info.json', 'transformer_control_table.bin',
             'transformer_datapath_table.bin', 'transformer_opcode_table.bin')
NAMES = PWP_NAMES + DVE_NAMES
ATTRIBUTES = ('st_dev', 'st_ino', 'st_mode', 'st_nlink', 'st_size', 'st_blocks',
              'st_mtime_ns', 'st_ctime_ns', 'st_uid', 'st_gid')


def target_path(name):
    require(name in NAMES, 'fixed SDK basename required')
    return (ASSET_ROOT if name in PWP_NAMES else DVE_ROOT) / name


def attrs(value):
    return tuple(getattr(value, key) for key in ATTRIBUTES)


def directory(value):
    require(stat.S_ISDIR(value.st_mode), 'ordinary nofollow directory required')
    return value.st_dev, value.st_ino, value.st_mode, value.st_uid, value.st_gid


@contextmanager
def parent_fd(path, *, owned_from=None):
    """Open every edge from / without following links; recheck before success."""
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts, 'absolute exact path required')
    if owned_from is not None:
        owned_from = Path(owned_from)
        require(owned_from.is_absolute() and owned_from in path.parents and '..' not in owned_from.parts,
                'exact owned compiler subtree required')
    fds, edges = [], []
    try:
        fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        fds.append(fd); current = Path('/'); owned_device = None
        for component in path.parts[1:-1]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
            fds.append(child); expected = directory(os.fstat(child))
            current /= component
            if owned_from is not None and (current == owned_from or owned_from in current.parents):
                if current == owned_from: owned_device = expected[0]
                require(expected[3] == os.getuid() and expected[0] == owned_device,
                        'owned compiler ancestry remains on one filesystem')
            edges.append((fd, component, child, expected)); fd = child
        yield fd
        for parent, name, child, expected in edges:
            require(directory(os.stat(name, dir_fd=parent, follow_symlinks=False)) == expected == directory(os.fstat(child)),
                    'nofollow directory edge changed')
    finally:
        for fd in reversed(fds): os.close(fd)


def manifest(spec):
    from .io import raw
    path = HERE / 'compiler_assets.json'
    require(spec['source_sha256'].get(str(path)) == MANIFEST_SHA256,
            'independently pinned exact compiler asset manifest required')
    payload = raw(path, 64 * 1024)
    require(hashlib.sha256(payload).hexdigest() == MANIFEST_SHA256, 'fixed compiler asset manifest changed')
    value = json.loads(payload)
    require(value['schema'] == 'qwen35_exact_sdk_pwp_dve_assets_v1' and
            value['sdk_version'] == '2.23.6484.0+3b612583' and
            set(value['assets']) == {str(target_path(name)) for name in NAMES},
            'only twenty actually observed installed SDK assets allowed')
    require(value['logical_bytes'] == sum(row['bytes'] for row in value['assets'].values()) == 387197 and
            value['allocated_bytes'] == sum(row['identity']['st_blocks'] * 512 for row in value['assets'].values()) == 417792,
            'fixed full SDK asset size/allocated inventory required')
    for path, row in value['assets'].items():
        require(set(row['identity']) == set(ATTRIBUTES) and row['bytes'] == row['identity']['st_size'] and
                0 < row['bytes'] < 45000 and row['identity']['st_nlink'] == 1 and
                row['record_path'] == str(Path(path).relative_to(SDK)) and row['record_bytes'] == row['bytes'],
                'fixed singleton SDK/RECORD identity required')
    return value


def asset(path, expected, *, content=True):
    """Stable bounded nofollow read (or stat); never loads/interprets asset bytes."""
    path = Path(path)
    require(path.name in NAMES and path == target_path(path.name), 'fixed SDK path required')
    expected_identity = tuple(expected['identity'][key] for key in ATTRIBUTES)
    require(0 < expected['bytes'] < 45000 and expected['bytes'] == expected['identity']['st_size'], 'bounded exact SDK asset')
    with parent_fd(path) as parent:
        before = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and attrs(before) == expected_identity,
                'installed SDK asset identity changed')
        if content:
            fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent)
            try:
                require(attrs(os.fstat(fd)) == expected_identity, 'SDK asset changed on open')
                digest = hashlib.sha256(); count = 0
                while count <= expected['bytes']:
                    part = os.read(fd, min(65536, expected['bytes'] + 1 - count))
                    if not part: break
                    digest.update(part); count += len(part)
                require(count == expected['bytes'] and digest.hexdigest() == expected['sha256'] and
                        attrs(os.fstat(fd)) == expected_identity, 'SDK asset content/hash changed')
            finally:
                os.close(fd)
        require(attrs(os.stat(path.name, dir_fd=parent, follow_symlinks=False)) == expected_identity,
                'SDK asset changed after observation')
    return dict(path=str(path), sha256=expected['sha256'], identity=expected_identity,
                inventory=(str(path), *expected_identity[:8]), allocated_bytes=before.st_blocks * 512,
                logical_bytes=before.st_size)


def verify_all(spec):
    """All 20 content pins checked prelock; repeated linked targets checked in scans."""
    value = manifest(spec)
    return [asset(path, row) for path, row in sorted(value['assets'].items())]


def compiler_sdk_link(path, spec, cap, expected_assets, observed=None, *, content=True):
    from .io import CompilerTreeRace, compiler_mutation
    path = Path(path); root = Path(spec['case']['compiler_workdir'])
    require(path.is_absolute() and root.is_absolute() and root in path.parents and '..' not in path.parts,
            'exact compiler SDK link scope')
    parts = path.relative_to(root).parts
    require(len(parts) == 5 and parts[0] in ('context_encoding_model', 'token_generation_model') and
            re.fullmatch(r'_tp0_bk[0-5]', parts[1]) and
            re.fullmatch(r'neuronxcc-[A-Za-z0-9_-]{1,64}', parts[2]) and
            re.fullmatch(r'sg[0-9]{2}', parts[3]) and parts[4] in NAMES,
            'only observed bucket/unit/subgraph SDK asset link form')
    target = target_path(parts[4])
    require(str(target) in expected_assets, 'fixed asset absent from authenticated manifest')
    try:
        with parent_fd(path, owned_from=root) as parent:
            before = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
            require(stat.S_ISLNK(before.st_mode) and before.st_uid == os.getuid() and before.st_nlink == 1 and
                    max(before.st_size, before.st_blocks * 512) <= cap, 'bounded owned singleton compiler SDK link')
            require(before.st_dev == os.fstat(parent).st_dev,
                    'SDK link remains in owned compiler filesystem')
            if observed is not None and attrs(before) != attrs(observed):
                raise CompilerTreeRace('SDK compiler link changed after inventory stat')
            text = os.readlink(path.name, dir_fd=parent)
            require(text == str(target), 'only exact installed SDK absolute link target allowed')
            # Installed SDK assets are immutable. Missing/changed targets are
            # permanent failures; only the fresh compiler link may churn.
            try:
                target_record = asset(target, expected_assets[str(target)], content=content)
            except FileNotFoundError as error:
                raise ValueError('fixed installed SDK asset disappeared') from error
            require(max(target_record['logical_bytes'], target_record['allocated_bytes']) <= cap,
                    'SDK target exceeds unchanged compiler single-file cap')
            if attrs(os.stat(path.name, dir_fd=parent, follow_symlinks=False)) != attrs(before) or os.readlink(path.name, dir_fd=parent) != text:
                raise CompilerTreeRace('SDK compiler link changed during observation')
            return dict(path=str(path), symlink=text, identity=attrs(before), target=str(target),
                        target_identity=target_record['identity'], target_inventory=target_record['inventory'],
                        sdk_asset_sha256=target_record['sha256'], existing_sdk_target=True)
    except FileNotFoundError:
        if compiler_mutation(path, spec): raise CompilerTreeRace('fresh compiler SDK link disappeared')
        raise
