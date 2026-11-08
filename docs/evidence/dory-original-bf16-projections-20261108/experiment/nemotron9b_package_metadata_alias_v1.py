"""Exact historical *metadata* aliases only; no filesystem action on import.

The four allowable links come from the accepted sio10 source handoff. The
current complete catalog contains all four, including chat_template.jinja.
Real link/target identities are first
captured only in the explicit serial preparation, then frozen in its spec.
"""
import hashlib
import os
from pathlib import Path
import stat

BASE = '/mnt/nvme/models/SmolLM3-3B-neuron-text-exp20261031/'
TARGET = '/mnt/nvme/models/SmolLM3-3B/'
STAT_FIELDS = ('dev', 'ino', 'mode', 'nlink', 'uid', 'size', 'mtime_ns', 'ctime_ns')
ALLOWED = {
    BASE+'chat_template.jinja': dict(target=TARGET+'chat_template.jinja', sha256='3e0e275a527f249670056e0078e474eadbdd007c134b4fdfae5a7b3089702814'),
    BASE+'generation_config.json': dict(target=TARGET+'generation_config.json', sha256='3458d62b3f9654f4bbe285223b0f4e5a1e67cae61c92bb6aeaadf04468bf9e01'),
    BASE+'model.safetensors.index.json': dict(target=TARGET+'model.safetensors.index.json', sha256='e3a7254c086c4a78f95ad5864f435d165ffed1ed8998326ce5107b4423c15b76'),
    BASE+'special_tokens_map.json': dict(target=TARGET+'special_tokens_map.json', sha256='12403efb970f069ba62ddf3b66166a27254be4a198166b61ad1ee604bd3a4515'),
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def stat_row(value):
    return {key: getattr(value, 'st_'+key) for key in STAT_FIELDS}


def validate_spec(spec):
    aliases = spec['metadata_aliases']
    require(type(aliases) is dict and set(aliases) == set(spec['source_sha256']) & set(ALLOWED),
            'Exact used subset of four reviewed historical aliases')
    for name, row in aliases.items():
        require(set(row) == {'target', 'sha256', 'link_stat', 'target_stat'}, 'Exact alias fields')
        require({key: row[key] for key in ('target', 'sha256')} == ALLOWED[name], 'Frozen alias mapping/hash')
        require(spec['source_sha256'][name] == row['sha256'], 'Original source pin retained')
        for key in ('link_stat', 'target_stat'):
            value = row[key]
            require(set(value) == set(STAT_FIELDS), 'Exact stat fields')
            require(all(type(v) is int and v >= 0 for v in value.values()), 'Real integer alias identity')
            require(value['ino'] > 0 and value['nlink'] == 1 and value['uid'] == 1000, 'Owned unshared alias identity')
        require(stat.S_ISLNK(row['link_stat']['mode']) and stat.S_ISREG(row['target_stat']['mode']), 'Link and regular target')
    return True


def capture(name, ordinary_text):
    """Explicit runtime operation. Never follows an arbitrary link."""
    require(name in ALLOWED, 'Only four reviewed metadata aliases')
    expected = ALLOWED[name]
    link, target = Path(name), Path(expected['target'])
    require(not any(parent.is_symlink() for parent in link.parents), 'Ordinary alias parents')
    before = stat_row(link.lstat())
    require(stat.S_ISLNK(before['mode']) and before['nlink'] == 1 and before['uid'] == 1000, 'Owned original metadata link')
    require(os.readlink(link) == str(target), 'Exact absolute link text; no retargeting')
    target_before = stat_row(target.lstat())
    require(stat.S_ISREG(target_before['mode']) and target_before['nlink'] == 1 and target_before['uid'] == 1000,
            'Owned ordinary metadata target')
    raw = ordinary_text(target)
    require(hashlib.sha256(raw).hexdigest() == expected['sha256'], 'Original metadata target hash')
    require(before == stat_row(link.lstat()) and target_before == stat_row(target.lstat())
            and os.readlink(link) == str(target), 'Alias/target unchanged during read')
    return dict(**expected, link_stat=before, target_stat=target_before)


def source_sha(name, admitted, ordinary_text):
    if name not in admitted:
        return hashlib.sha256(ordinary_text(name)).hexdigest()
    actual = capture(name, ordinary_text)
    require(actual == admitted[name], 'Exact captured alias link/target identities retained')
    return actual['sha256']
