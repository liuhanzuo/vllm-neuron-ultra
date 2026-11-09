"""Bounded privileged process inspection and exact verified journal retirement."""
import json
import os
from pathlib import Path
import sys

import completed_storage_contract_after_retry_v1 as c
import completed_storage_files_v1 as f


def scan(rows):
    targets = {row['path'] for row in rows}
    inodes = {(row['stat']['st_dev'],row['stat']['st_ino']):row['path'] for row in rows}
    matches, errors, processes = [], [], 0
    for proc in sorted(Path('/proc').iterdir()):
        if not proc.name.isdigit(): continue
        try:
            before = (proc/'stat').read_text().rsplit(')',1)[1].split()[19]
            processes += 1
            with (proc/'cmdline').open('rb') as stream: argv = stream.read(2**20+1)
            f.require(len(argv) <= 2**20, 'Bounded process argv')
            for path in targets | {str(c.MODEL_ROOT)}:
                if path.encode() in argv:
                    matches.append(dict(pid=int(proc.name),kind='argv',path=path))
            for fd in (proc/'fd').iterdir():
                try:
                    info = fd.stat()
                    key = (info.st_dev,info.st_ino)
                    if key in inodes: matches.append(dict(pid=int(proc.name),kind='fd',path=inodes[key]))
                except (FileNotFoundError,ProcessLookupError): pass
            with (proc/'maps').open('r') as stream: maps = stream.read(16*2**20+1)
            f.require(len(maps) <= 16*2**20, 'Bounded process maps')
            for line in maps.splitlines():
                fields = line.split(None,5)
                if len(fields) < 5: continue
                major,minor = fields[3].split(':')
                key = (os.makedev(int(major,16),int(minor,16)),int(fields[4]))
                if key in inodes: matches.append(dict(pid=int(proc.name),kind='map',path=inodes[key]))
            after = (proc/'stat').read_text().rsplit(')',1)[1].split()[19]
            f.require(before == after, 'PID changed during privileged inspection')
        except (FileNotFoundError,ProcessLookupError): continue
        except Exception as error: errors.append(dict(pid=int(proc.name),error_type=type(error).__name__))
    return dict(processes=processes,matches=matches,errors=errors,complete=not errors)


def unused(rows):
    result = scan(rows)
    f.require(result['complete'] and not result['matches'], 'All target argv/fd/maps unused: '+json.dumps(result))
    return result


def retire(request):
    plan = c.load(request['plan'],plan=True)
    c.validation(plan)
    ad = c.load(request['admission'])
    f.require(request['admission']['path'] == str(c.OUT/'admission.json')
              and ad['plan'] == request['plan'] and ad['fully_drained'] is True,
              'Exact admitted serial maintenance')
    expected = ['/usr/bin/python3','-B',str(c.EXECUTOR),'--plan',request['plan']['path'],
                '--plan-sha256',request['plan']['sha256'],'--review',ad['source_review']['path'],
                '--review-sha256',ad['source_review']['sha256']]
    f.require(ad['owner']['argv'] == expected, 'Exact lock owner invocation')
    c.p.low.lock_proof(ad['owner'],ad['lock'])
    checked = c.load(request['verification'])
    f.require(request['verification']['path'] == str(c.OUT/'verified-before-retirement.json')
              and checked['all_five_weights_authenticated'] is True
              and checked['all_eight_journals_decompressed'] is True
              and checked['plan'] == request['plan'], 'All sources verified before any deletion')
    rows = [row for row in plan['journals'] if row['path'] == request['path']]
    f.require(len(rows) == 1, 'One exact admitted closed journal')
    row = rows[0]
    archives = [r for r in checked['journals'] if r['source'] == row['path']]
    f.require(len(archives) == 1, 'Exact verified archive')
    archive = archives[0]
    f.require(archive['archive'] == str(c.OUT/(Path(row['path']).name+'.gz'))
              and archive['source_stat'] == row['stat'], 'Archive belongs to exact original')
    f.require(f.file_hash(archive['archive'],archive['archive_stat'],lambda: None)['sha256']
              == archive['gzip_sha256'], 'Retained compressed bytes unchanged')
    f.verify_gzip(archive['archive'],archive['original_bytes'],archive['original_sha256'],lambda: None)
    observed = unused(rows)
    c.p.low.lock_proof(ad['owner'],ad['lock'])
    f.unlink_exact(row['path'],row['stat'])
    return dict(retired=row['path'],archive=archive['archive'],unused=observed)


def main():
    f.require(os.geteuid() == 0 and sys.dont_write_bytecode and __debug__, 'Privileged bounded helper')
    raw = sys.stdin.buffer.read(2**20+1)
    f.require(len(raw) <= 2**20, 'Bounded helper request')
    request = json.loads(raw)
    if request['operation'] == 'check-unused':
        _,weights,journals = c.fixed_targets()
        f.require(request['rows'] == weights+journals, 'Exact admitted candidate set for inspection')
        result = unused(request['rows'])
    elif request['operation'] == 'unlink-journal': result = retire(request)
    else: raise RuntimeError('No other privileged operation')
    print(json.dumps(result),flush=True)


if __name__ == '__main__': main()
