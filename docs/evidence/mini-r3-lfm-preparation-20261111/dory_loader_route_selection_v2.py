"""Pure, bounded selection over authenticated event values; no filesystem I/O.

Only the future admitted collector reads the original event stream or selected
installed sources. Missing matches never establish whole-package absence.
"""
import ast
from copy import deepcopy
from pathlib import PurePosixPath
import re

SITE = '/mnt/nvme/venvs/dory-official-cpu-20261108-v1/lib/python3.12/site-packages'
REQUIRED = ('torch/_utils_internal.py', 'vllm/env_override.py', 'vllm/envs.py',
            'vllm/distributed/__init__.py', 'vllm/distributed/parallel_state.py')
PREFIXES = ('vllm/distributed/device_communicators/', 'vllm/v1/kv_offload/',
            'vllm/distributed/kv_transfer/kv_connector/v1/offloading/')
NAME_TOKENS = ('fs_io', 'spinloop', 'filesystem', 'file_system', 'offload')
NEEDLES = ('fs_io_C', 'spinloop', 'shm_broadcast', 'USE_GLOBAL_DEPS',
           'USE_RTLD_GLOBAL_WITH_LIBTORCH', 'TORCH_USE_RTLD_GLOBAL',
           'VLLM_TARGET_DEVICE', 'VLLM_PLUGINS', 'LD_PRELOAD')
IDENTITY_FIELDS = ('st_dev', 'st_ino', 'st_mode', 'st_nlink', 'st_size',
                   'st_mtime_ns', 'st_ctime_ns')
POLICY = dict(schema='dory_loader_route_source_selection_v2', site=SITE,
              required_sources=list(REQUIRED), candidate_prefixes=list(PREFIXES),
              candidate_path_substrings=list(NAME_TOKENS),
              source_file_cap=192, per_source_bytes=512 * 2**10,
              total_source_bytes=8 * 2**20, per_source_finding_cap=512,
              total_findings_cap=2048, total_findings_bytes=2 * 2**20,
              finding_text_char_cap=2048, source_only=True,
              filesystem_enumeration=False, whole_package_search=False,
              missing_match_proves_absence=False,
              automatic_scope_expansion=False)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def selection_reason(path):
    """Select only names already present in the finite authenticated stream."""
    if type(path) is not str or not path.startswith(SITE + '/'):
        return None
    relative = path[len(SITE) + 1:]
    if relative in REQUIRED:
        return 'required'
    if not relative.startswith('vllm/') or not relative.endswith('.py'):
        return None
    if any(relative.startswith(prefix) for prefix in PREFIXES):
        return 'candidate_prefix'
    if any(token in relative for token in NAME_TOKENS):
        return 'candidate_path_substring'
    return None


def select_targets(events, *, expected_events):
    """Consume and authenticate the full stream before returning any sources.

The caller owns gzip/EOF/full-hash authentication. This selector exhausts that
generator, validates each selected original identity/provenance and enforces
all aggregate limits before the caller can read any installed source bytes.
"""
    require(type(expected_events) is int and 0 < expected_events <= 100000,
            'Finite exact event count')
    selected = {}; count = 0; total = 0
    for event in events:
        count += 1
        require(count <= expected_events and type(event) is dict,
                'Bounded complete event stream')
        if event.get('event') != 'SITE_inventory':
            continue
        ref = event.get('reference')
        if type(ref) is not dict or selection_reason(ref.get('path')) is None:
            continue
        path = ref['path']; relative = path[len(SITE) + 1:]
        require(str(PurePosixPath(path)) == path and '\\' not in path
                and '\0' not in path and all(p not in ('', '.', '..')
                                            for p in path.split('/')[1:]),
                'Canonical selected source path')
        require(path not in selected, 'Duplicate selected source event')
        require(set(ref) == {'path', 'bytes', 'sha256'}
                and type(ref['bytes']) is int and 0 <= ref['bytes'] <= POLICY['per_source_bytes']
                and type(ref['sha256']) is str
                and re.fullmatch('[0-9a-f]{64}', ref['sha256']),
                'Exact complete bounded source reference')
        provenance = event.get('provenance', {})
        evidence = provenance.get('evidence', {})
        require(provenance.get('kind') == 'upstream_wheel_member'
                and evidence.get('package') == relative.split('/')[0]
                and evidence.get('member') == relative
                and evidence.get('archive_sha256') == ref['sha256'],
                'Exact authenticated wheel member provenance')
        identity = event.get('identity')
        require(type(identity) is dict and set(identity) == set(IDENTITY_FIELDS)
                and all(type(v) is int for v in identity.values())
                and identity['st_size'] == ref['bytes'] and identity['st_nlink'] == 1,
                'Complete saved source identity')
        selected[path] = deepcopy(event); total += ref['bytes']
        require(len(selected) <= POLICY['source_file_cap'], 'Finite selected source count cap')
        require(total <= POLICY['total_source_bytes'], 'Aggregate selected source byte cap')
    require(count == expected_events, 'Exact event count through authenticated EOF')
    require(all(SITE + '/' + name in selected for name in REQUIRED),
            'All five required source references present')
    return [selected[path] for path in sorted(selected)]


def inspect_source(relative, raw):
    """Parse authenticated bytes; report references, never execute target code."""
    require(type(raw) is bytes and len(raw) <= POLICY['per_source_bytes']
            and selection_reason(SITE + '/' + relative) is not None,
            'One selected bounded Python source')
    text = raw.decode('utf-8'); tree = ast.parse(text, filename=relative)
    lines = text.splitlines(); imports = []; mentions = []
    scopes = [(n.lineno, n.end_lineno, type(n).__name__ + ':' + n.name)
              for n in ast.walk(tree)
              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    for number, line in enumerate(lines, 1):
        found = [needle for needle in NEEDLES if needle in line]
        if found:
            enclosing = [name for start, end, name in sorted(scopes)
                         if start <= number <= end]
            mentions.append(dict(line=number, tokens=found,
                                 text=line[:POLICY['finding_text_char_cap']],
                                 text_truncated=len(line) > POLICY['finding_text_char_cap'],
                                 enclosing_scopes=enclosing))
    for node in ast.walk(tree):
        modules = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            root = ('.' * node.level) + (node.module or '')
            modules = [root] + [root + '.' + alias.name for alias in node.names]
        targets = [name for name in modules if any(t in name for t in ('fs_io_C', 'spinloop'))]
        if targets:
            imports.append(dict(line=node.lineno, end_line=node.end_lineno,
                                kind=type(node).__name__, imported_names=targets))
    require(len(mentions) + len(imports) <= POLICY['per_source_finding_cap'],
            'Finite per-source findings cap')
    return dict(relative=relative, ast_parsed_without_execution=True,
                direct_import_statements=imports, textual_mentions=mentions,
                actual_import_or_call_observed=False,
                scope_complete_only_for_this_source=True,
                whole_package_absence_established=False)
