"""Pure parser bridge for one reviewed GnuPG status line; no cryptographic IO.

Keep the complete original stdout as the saved evidence. Only this in-memory
parse view removes an optional exact de-vs compliance annotation. The frozen
v2 parser remains responsible for every signature, key, date and exit check.
"""
import dory_ubuntu_metadata_adapter_v2 as original

COMPLIANCE = b'[GNUPG:] VERIFICATION_COMPLIANCE_MODE 23'


def parse_status(stdout, returncode, trust, now):
    original.need(returncode == 0 and type(stdout) is bytes and 0 < len(stdout) <= original.MIB,
                  'Successful bounded original gpgv status')
    original.need(stdout.endswith(b'\n') and b'\r' not in stdout and b'\x00' not in stdout,
                  'Complete original LF gpgv status')
    kept, group = [], None
    for line in stdout.split(b'\n')[:-1]:
        tag = line[9:].partition(b' ')[0] if line.startswith(b'[GNUPG:] ') else None
        if tag == b'NEWSIG':
            group = dict(good=False, valid=False, compliance=False)
        elif tag == b'GOODSIG' and group is not None:
            group['good'] = True
        elif tag == b'VALIDSIG' and group is not None:
            group['valid'] = True
        elif tag == b'VERIFICATION_COMPLIANCE_MODE':
            original.need(line == COMPLIANCE and group is not None
                          and group['good'] and group['valid'] and not group['compliance'],
                          'One exact de-vs annotation after GOODSIG and VALIDSIG in its NEWSIG group')
            group['compliance'] = True
            continue
        # Every other byte is passed to the frozen parser, including all errors,
        # unknown tags, malformed signature fields and duplicate signatures.
        kept.append(line + b'\n')
    return original.parse_status(b''.join(kept), returncode, trust, now)
