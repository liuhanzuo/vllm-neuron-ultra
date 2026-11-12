"""Finite public metadata refresh; no model weights or framework imports."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import urllib.parse
import urllib.request

Q = Path(__file__).resolve().parent
PREVIOUS = Q / 'official-small-delta-20261111-v1'
OUTPUT = Q / 'official-small-delta-20261112-v1'
CAP = 512 * 1024


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def reference(path):
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def save(name, value):
    with (OUTPUT / name).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError('Metadata redirect is outside this finite refresh')


class WholeRefreshTimeout(BaseException):
    """Escape individual request handling while preserving final receipts."""


def main():
    os.sched_setaffinity(0, {4})
    os.nice(19)
    resource.setrlimit(resource.RLIMIT_AS, (512 * 2**20,) * 2)
    resource.setrlimit(resource.RLIMIT_CPU, (30, 35))
    resource.setrlimit(resource.RLIMIT_FSIZE, (2**20,) * 2)
    def timed_out(*_):
        raise WholeRefreshTimeout('Whole metadata refresh exceeded 120 seconds')
    signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(120)
    prior = json.loads((PREVIOUS / 'summary.json').read_text())
    authors = prior['authors']
    assert len(authors) == len(set(authors)) == 19
    OUTPUT.mkdir(exist_ok=False)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    started, receipts, changes, counts, errors = stamp(), [], [], {}, []
    try:
        for author in authors:
            url = 'https://huggingface.co/api/models?' + urllib.parse.urlencode(
                dict(author=author, sort='createdAt', direction=-1, limit=16, full='true'))
            receipt = dict(author=author, url=url, started_at=stamp())
            try:
                req = urllib.request.Request(url, headers={
                    'User-Agent': 'trn1-public-model-metadata/1.0', 'Accept': 'application/json'})
                with opener.open(req, timeout=8) as response:
                    assert response.status == 200 and response.geturl() == url
                    raw = response.read(CAP + 1)
                    assert len(raw) <= CAP
                    receipt.update(status=response.status, final_url=response.geturl(),
                        headers={k: response.headers.get(k) for k in ('Date', 'ETag', 'Age', 'X-Cache')})
                rows = json.loads(raw)
                assert isinstance(rows, list) and len(rows) <= 16
                assert all(isinstance(row, dict) and row['id'].split('/')[0].casefold() == author.casefold()
                           for row in rows)
                name = 'list--' + author + '--createdAt.json'
                with (OUTPUT / name).open('xb') as stream:
                    stream.write(raw)
                receipt.update(reference(OUTPUT / name))
                old = {row['id']: row for row in json.loads((PREVIOUS / name).read_text())}
                counts[author] = len(rows)
                for row in rows:
                    before = old.get(row['id'])
                    if before is None or before.get('sha') != row.get('sha'):
                        change = {k: row.get(k) for k in
                            ('id', 'sha', 'createdAt', 'lastModified', 'downloads', 'likes', 'pipeline_tag')}
                        change.update(previous_sha=before.get('sha') if before else None,
                                      absent_from_previous_latest16=before is None)
                        changes.append(change)
            except Exception as error:
                receipt.update(error_type=type(error).__name__, error=str(error))
                errors.append(dict(receipt))
            except WholeRefreshTimeout as error:
                receipt.update(error_type=type(error).__name__, error=str(error))
                errors.append(dict(receipt))
                raise
            finally:
                receipt['finished_at'] = stamp()
                receipts.append(receipt)
    finally:
        save('list-receipts.json', receipts)
        save('discovery.json', dict(schema='official_small_delta_discovery_20261112_v1',
            retrieval_start_utc=started, retrieval_end_utc=stamp(), authors=authors,
            previous_summary=reference(PREVIOUS / 'summary.json'), counts=counts,
            entries=sum(counts.values()), requests=len(receipts), errors=errors, changes=changes,
            scope='Newest16 created repositories per same19 publishers; not exhaustive and not a release-date proof.',
            all15existing_families_preserved=True, experimental_revisions_changed=False,
            weights_downloaded=False, remote_model_code_executed=False,
            native_framework_imported=False, device_or_compiler_launched=False,
            new_model_qualification=False, new_speedup_claim=False))
        signal.alarm(0)
    print(json.dumps(dict(entries=sum(counts.values()), requests=len(receipts), errors=len(errors),
                          changes=changes), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
