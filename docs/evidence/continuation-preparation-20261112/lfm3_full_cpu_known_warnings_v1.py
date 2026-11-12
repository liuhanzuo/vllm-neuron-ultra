"""Exact reviewed SyntaxWarning deduplication for the full CPU v3 runner.

Import is inert. No warnings filters, process operations, or mutable shared file.
Only main worker/batch startup installs the hook before local imports.
Counts cover calls delivered to showwarning, not warnings hidden by Python's
unchanged default filters or by a library's own catch_warnings context.
"""
import hashlib
import os
import stat
import warnings

MAX_SOURCE_BYTES = 256 * 2**10
MAX_COUNT = 2**31 - 1
KNOWN = ({'filename': '/home/ubuntu/experiments/trn1-latest-models-20261031/smollm3_observer_invariance_protocol_v20.py',
  'lineno': 265,
  'category': 'SyntaxWarning',
  'message': "invalid escape sequence '\\.'",
  'source_sha256': '9b1dea96d2734fe6c86bd081de0eb8837bef8c100f44cf9feea5e608a302acfa',
  'source_line': "            self.poll();require(re.fullmatch('[A-Za-z0-9_-]+\\.(json|raw)',name) and "
                 "type(raw)is bytes and len(raw)<=4*MIB,'Bounded exclusive control/request receipt')"},
 {'filename': '/home/ubuntu/experiments/trn1-latest-models-20261031/smollm3_observer_invariance_protocol_v20.py',
  'lineno': 283,
  'category': 'SyntaxWarning',
  'message': "invalid escape sequence '\\.'",
  'source_sha256': '9b1dea96d2734fe6c86bd081de0eb8837bef8c100f44cf9feea5e608a302acfa',
  'source_line': "            require(re.fullmatch('[A-Za-z0-9_-]+\\.(json|raw)',name) and type(raw)is bytes "
                 "and len(raw)<=4*MIB,'Bounded exclusive control/request receipt')"},
 {'filename': '/home/ubuntu/experiments/trn1-latest-models-20261031/research/minicpm5_layer1_metric_diagnostic_runner_v1.py',
  'lineno': 167,
  'category': 'SyntaxWarning',
  'message': "invalid escape sequence '\\.'",
  'source_sha256': 'fdff9d786b9088966164a223fb46da96e38622bde8f3486da1ca0fd663cb5d29',
  'source_line': '        require(target.parent==Q and '
                 "re.fullmatch('minicpm5-layer1-saved-audit-[a-z0-9-]{1,80}\\.json',target.name)"},
 {'filename': '/home/ubuntu/experiments/trn1-latest-models-20261031/research/smollm3_sio14_catalog_transport_review_v1.py',
  'lineno': 45,
  'category': 'SyntaxWarning',
  'message': "invalid escape sequence '\\.'",
  'source_sha256': '1579154a2072abb1b7bbc52661541a4c9c17a91dd1bcffe03b8f7de4c448088f',
  'source_line': "                require(re.fullmatch('[0-9a-f]{64}\\.json',name)and "
                 "child.stem==item['sha256'],'Complete chunk filename/hash')"})


def require(value, message):
    if not value:
        raise RuntimeError(message)


def source_matches(signature):
    """Each match rechecks bounded source bytes; a changed source is forwarded."""
    try:
        # Use a named open: the retained child audit forbids wrapping arbitrary
        # numeric descriptors, even when they refer to an allowed source file.
        def opener(path, flags):
            return os.open(path, flags | os.O_NOFOLLOW | os.O_CLOEXEC)
        with open(signature['filename'], 'rb', opener=opener) as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_SOURCE_BYTES:
                return False
            raw = stream.read(MAX_SOURCE_BYTES + 1)
            after = os.fstat(stream.fileno())
        fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
        return (len(raw) == before.st_size
                and all(getattr(before, key) == getattr(after, key) for key in fields)
                and hashlib.sha256(raw).hexdigest() == signature['source_sha256']
                and raw.decode('utf-8').splitlines()[signature['lineno'] - 1] == signature['source_line'])
    except (OSError, UnicodeError, IndexError):
        return False


def validate_receipt(value, mode):
    require(type(value) is dict and set(value) == {'schema', 'mode', 'scope', 'known', 'other_forwarded'}
            and value['schema'] == 'lfm3_full_cpu_known_warning_receipt_v1'
            and value['mode'] == mode and mode in ('outer', 'worker', 'batch')
            and value['scope'] == 'calls_delivered_to_showwarning'
            and type(value['other_forwarded']) is int and 0 <= value['other_forwarded'] <= MAX_COUNT,
            'Exact bounded known-warning receipt')
    require(type(value['known']) is list and len(value['known']) == len(KNOWN),
            'All four complete reviewed warning signatures retained')
    for row, signature in zip(value['known'], KNOWN):
        require(type(row) is dict and set(row) == set(signature) | {'observed', 'forwarded', 'deduplicated'}
                and all(row[key] == expected for key, expected in signature.items()),
                'Exact category/message/path/line/source-hash/source-line warning identity')
        require(all(type(row[key]) is int and 0 <= row[key] <= MAX_COUNT
                    for key in ('observed', 'forwarded', 'deduplicated')),
                'Finite exact warning counts')
        require(row['observed'] == row['forwarded'] + row['deduplicated']
                and row['forwarded'] == (0 if mode == 'batch' else min(row['observed'], 1)),
                'Complete warning accounting; worker/outer retain first, batch records all')
    return value


class KnownWarningLog:
    def __init__(self, mode, forward, matcher=source_matches):
        require(mode in ('outer', 'worker', 'batch') and callable(forward), 'Exact warning-log mode')
        self.mode, self.forward, self.matcher = mode, forward, matcher
        self.rows = [dict(row, observed=0, forwarded=0, deduplicated=0) for row in KNOWN]
        self.other_forwarded = 0
        self.installed = False
        self.callback = self.showwarning

    def showwarning(self, message, category, filename, lineno, file=None, line=None):
        selected = None
        if category is SyntaxWarning and file is None:
            for i, signature in enumerate(KNOWN):
                if (filename == signature['filename'] and type(lineno) is int and lineno == signature['lineno']
                        and str(message) == signature['message']
                        and (line is None or line == signature['source_line'])
                        and self.matcher(signature)):
                    selected = self.rows[i]
                    break
        if selected is None:
            self.other_forwarded += 1
            return self.forward(message, category, filename, lineno, file=file, line=line)
        selected['observed'] += 1
        # Counts fail closed at receipt validation instead of silently overflowing.
        if self.mode != 'batch' and selected['forwarded'] == 0:
            selected['forwarded'] += 1
            return self.forward(message, category, filename, lineno, file=file, line=line)
        selected['deduplicated'] += 1

    def snapshot(self):
        require(not self.installed or warnings.showwarning is self.callback,
                'Installed exact warning hook must remain active at receipt time')
        value = dict(schema='lfm3_full_cpu_known_warning_receipt_v1', mode=self.mode,
                     scope='calls_delivered_to_showwarning', known=[dict(row) for row in self.rows],
                     other_forwarded=self.other_forwarded)
        return validate_receipt(value, self.mode)


def bootstrap(name, argv):
    # This only chooses logging behavior; the unchanged runner still owns every
    # admission/argv/identity check. Imported modules never change warning state.
    if name != '__main__' or len(argv) < 3 or argv[1] != '--mode' or argv[2] not in ('worker', 'batch'):
        return None
    journal = KnownWarningLog(argv[2], warnings.showwarning)
    warnings.showwarning = journal.callback
    journal.installed = True
    return journal
