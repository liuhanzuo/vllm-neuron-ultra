"""Bounded public metadata retrieval. No model/native/weight execution."""
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request

OUT = Path(__file__).resolve().parent
PREVIOUS = OUT.parent / 'official-small-delta-20261108-v1'
CAP = 2 * 2**20


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def fetch(name, url):
    assert url.startswith('https://huggingface.co/')
    assert '/resolve/' not in url or url.rsplit('/', 1)[-1] in ('README.md', 'config.json', 'tokenizer_config.json')
    assert '/' not in name and '\\' not in name
    record = dict(url=url, at=now(), path=str(OUT / name))
    request = urllib.request.Request(url, headers={'User-Agent': 'trn1-model-metadata-research/1.0'})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read(CAP + 1)
            assert len(body) <= CAP, 'Public text response cap'
            record.update(status=response.status, final_url=response.url,
                          headers={key: response.headers.get(key) for key in ('Date', 'ETag', 'Age', 'X-Cache', 'Content-Type')})
    except urllib.error.HTTPError as error:
        body = error.read(CAP + 1)
        assert len(body) <= CAP
        record.update(status=error.code, error=str(error))
    except Exception as error:
        body = b''
        record.update(status=None, error=repr(error))
    if body:
        with (OUT / name).open('xb') as stream:
            stream.write(body)
        record.update(bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
    record['finished_at'] = now()
    return record


def main():
    summary = json.loads((PREVIOUS / 'summary.json').read_text())
    requests = [(f'list--{author}--createdAt.json', 'https://huggingface.co/api/models?' +
                 urllib.parse.urlencode(dict(author=author, sort='createdAt', direction=-1, limit=16, full='true')))
                for author in summary['authors']]
    started = now()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(lambda item: fetch(*item), requests))
    (OUT / 'list-receipts.json').write_text(json.dumps(receipts, indent=2) + '\n')
    changes = []
    total = 0
    for author, receipt in zip(summary['authors'], receipts):
        if receipt['status'] != 200:
            continue
        current = json.loads(Path(receipt['path']).read_text())
        assert isinstance(current, list) and len(current) <= 16
        previous = {row['id']: row for row in json.loads((PREVIOUS / f'list--{author}--createdAt.json').read_text())}
        total += len(current)
        for row in current:
            prior = previous.get(row['id'])
            if prior is None or row.get('sha') != prior.get('sha'):
                changes.append({key: row.get(key) for key in ('id', 'sha', 'createdAt', 'lastModified', 'downloads', 'likes', 'pipeline_tag')} |
                               dict(not_in_previous_latest16=prior is None,
                                    created_since_previous_retrieval=(row.get('createdAt') or '') > summary['retrieval_end_utc'],
                                    previous_sha=prior.get('sha') if prior else None))
    result = dict(schema='official_small_list_delta_20261111_v1', retrieval_start_utc=started, retrieval_end_utc=now(),
                  authors=summary['authors'], list_requests=len(receipts), successful_list_requests=sum(x['status'] == 200 for x in receipts),
                  repositories_scanned=total, changed_revisions_or_new_repositories=changes,
                  previous_summary=dict(path=str(PREVIOUS / 'summary.json'), sha256=hashlib.sha256((PREVIOUS / 'summary.json').read_bytes()).hexdigest()),
                  scope='Newest16 repositories per same19 official publishers, finite comparison against Nov8; not exhaustive.',
                  weights_downloaded=False, remote_model_code_executed=False, native_framework_imported=False)
    (OUT / 'list-delta.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
