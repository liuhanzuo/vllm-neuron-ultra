"""Exact original14 + Dory6 source partition, no runtime admission.

Read only authenticated metadata. Historical payload bytes are never opened;
all original pins remain in the source map. The fixed Dory tokenizer JSON has
an explicit32MiB exception; every other metadata rule remains with v19.
"""
import hashlib
import json
from pathlib import Path

import nemotron9b_package_metadata_v19 as original
from research import nemotron9b_after_dory_residual_review_v1 as residual

Q=Path(__file__).resolve().parent
PARENT=dict(path=str(Q/'nemotron9b-package-runtime-v18-r9-20261109-spec.json'),
    sha256='9f55222ec65715e72502ebbc954d012454b38e1b17b16d568d4d90a91c25f3ec')
TOKENIZER=dict(path=str(Q/'dory-checkpoint-metadata-20261108-v1/tokenizer.json'),
    sha256='623c34567aebb18582765289fbe23d901c62704d6518d71866e0e58db892b5b7')
TOKENIZER_BYTES=17077484
require=original.require


def exact_dory_payloads(start):
    pins=start['spec']['source_sha256']
    original.pin_shape(pins)
    binaries={path:digest for path,digest in pins.items()if Path(path).suffix in original.history.FORBIDDEN}
    root=residual.BASE/'long-fresh-B-r2/diagnostic'
    expected={str(root/f'{frame:06d}-{kind}.bin')for frame in(1,2,3)for kind in('official','control')}
    require(set(binaries)==expected,'Only exact six original Dory archived binaries')
    rows=start['sources']['files']
    require(all(rows[path]['sha256']==digest for path,digest in binaries.items())
            and sum(rows[path]['bytes']for path in binaries)==18622464,
            'Six original binary pins match actual admitted source authentication ledger')
    return binaries


def classify_maps(spec,catalog,parent,start,original_parts):
    """Pure map checks over already authenticated inputs; does not read files."""
    pins=spec['source_sha256'];original.pin_shape(pins)
    require(spec['metadata_limits']==original.LIMITS,'Original8/32MiB metadata policy')
    require(spec.get('required_residual_source_anchor')==residual.PRODUCER,
            'Exact admitted Dory source anchor required, not completion')
    require(all(pins.get(k)==v for k,v in parent['source_sha256'].items()),
            'Every original r9 source and historical payload pin retained')
    require(all(pins.get(k)==v for k,v in start['spec']['source_sha256'].items()),
            'Every admitted Dory source and historical payload pin retained')
    for ref in(PARENT,residual.PRODUCER,residual.ADMISSION,residual.HISTORY,residual.CHILD,residual.SOURCE_FILES):
        require(pins.get(ref['path'])==ref['sha256'],'All actual anchor metadata pins retained')
    require(all(pins.get(k)==v for k,v in catalog['source_sha256'].items()),
            'Original final catalog source closure retained')
    require(all(pins.get(k)==v for k,v in original_parts['provenance_sha256'].items()),
            'Original14 historical provenance retained')
    old=original_parts['historical_payload_sha256']
    require(len(old)==14 and old==parent['historical_payload_sha256']==spec['historical_payload_sha256'],
            'Original14 historical raw map unchanged')
    dory=exact_dory_payloads(start)
    require(spec.get('dory_historical_payload_sha256')==dory and not set(old)&set(dory),
            'Explicit disjoint six Dory binary additions')
    merged=dict(old,**dory)
    require({k:v for k,v in pins.items()if Path(k).suffix in original.history.FORBIDDEN}==merged,
            'All20 exact historical payloads, no unknown or omitted binaries')
    require(spec['runtime_python_non_utf8_sha256']==original.NON_UTF8 and all(
            pins.get(k)==spec['runtime_python_source_sha256'].get(k)==v for k,v in original.NON_UTF8.items()),
            'All three original non-UTF8 Python sources retained')
    ordinary={k:v for k,v in pins.items()if k not in merged and k not in original.NON_UTF8}
    require(not(set(ordinary)&set(merged)or set(ordinary)&set(original.NON_UTF8)or set(merged)&set(original.NON_UTF8))
            and dict(ordinary,**merged,**original.NON_UTF8)==pins,'Lossless disjoint complete source partition')
    require(ordinary.get(TOKENIZER['path'])==TOKENIZER['sha256'],'Exact original Dory tokenizer text pin')
    return dict(text_sha256=ordinary,historical_payload_sha256=merged,
                python_bytes_sha256=dict(original.NON_UTF8))


def classify_sources(spec,catalog):
    """Authenticate fixed inputs, then classify; completion/drain are separate."""
    parent=original.load(PARENT)
    original.classify_sources(parent,catalog)
    parts=original.history.partition(catalog)
    reader=residual.Reader()
    start=residual.anchor(reader)
    return classify_maps(spec,catalog,parent,start,parts)


def accepts(path):
    return str(path)==TOKENIZER['path']


def text(path,expected):
    """One exact SHA/path/size tokenizer JSON exception, no generic cap change."""
    require(accepts(path) and expected==TOKENIZER['sha256'],
            'Only exact fixed Dory tokenizer JSON exception')
    original.low.clear_window()
    raw=original.stable_bytes(path,original.ENVELOPE_LIMIT)
    require(len(raw)==TOKENIZER_BYTES and hashlib.sha256(raw).hexdigest()==expected,
            'Exact original Dory tokenizer bytes')
    value=json.loads(raw.decode('utf8'))
    require(isinstance(value,dict) and value.get('version')=='1.0'
            and isinstance(value.get('model'),dict) and value['model'].get('type')=='BPE'
            and isinstance(value['model'].get('vocab'),dict) and isinstance(value.get('added_tokens'),list),
            'Original tokenizer JSON structure')
    return raw


if __name__=='__main__':
    raise RuntimeError('No launch or spec creation; source classification only')
