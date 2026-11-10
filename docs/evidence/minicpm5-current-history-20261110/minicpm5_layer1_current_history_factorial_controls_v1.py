"""Read authenticated old four-row controls in full under current runtime admission."""
from pathlib import Path
from research import minicpm5_layer1_payload_io_v1 as payload
from research import minicpm5_layer1_decode_stream_v2 as stream
require=stream.require
OLD_MODES=('own','official_key','official_value','official_both','restore')
MAPPING={'own':'own','official_history':'official_both','restore':'restore'}


def for_call(plan,bundle,case,guard):
    guard(); rank=case['rank']; result=bundle['records']['rank%d_result'%rank]
    expected=[(arm,ci,mode) for arm in stream.raw.ARMS for ci in stream.boundary.DECODE_CALLS for mode in OLD_MODES]
    require(result['rank']==rank and result['complete'] is True and len(result['records'])==90,
        'All actual previous official history rows')
    actual=[(row['result']['arm'],row['result']['call']['index'],row['result']['mode']) for row in result['records']]
    require(actual==expected,'Exact complete previous call order')
    output={}
    for old_mode in MAPPING.values():
        index=expected.index((case['arm'],case['call']['index'],old_mode)); row=result['records'][index]
        previous=row['result']; desc=row['output']
        require(previous['rank']==rank and previous['call']==case['call']
            and previous['schema']=='minicpm5_layer1_official_history_callable_result_v1'
            and previous['complete'] is True,'Exact actual previous four-row call')
        path=Path(bundle['spec']['output_root'])/('cpu-rank%d'%rank)/case['arm']/('call-%02d'%case['call']['index'])/old_mode/'attention.bf16'
        require(desc['path']==str(path) and desc['raw_bytes']==16384
            and desc['sha256']==previous['attention_output_sha256']
            and plan['inputs'][str(path)]==dict(sha256=desc['sha256'],identity=desc['identity']),
            'Frozen exact previous output descriptor and identity')
        raw=payload.read_owned_payload(dict(desc,payload_bytes=16384,payload_sha256=desc['sha256']),path,guard)
        require(type(raw) is bytes and len(raw)==16384,'Full previous control output bytes')
        output[old_mode]=dict(raw=raw,reference=desc)
    require(output['own']['raw']==output['restore']['raw'],'Actual previous own and restore remain byte exact')
    guard();return output


def verify(previous,mode,raw):
    if mode not in MAPPING:return None
    old_mode=MAPPING[mode]; old=previous[old_mode]
    require(type(raw) is bytes and len(raw)==16384 and raw==old['raw'],
        'Full raw bytes reproduce previous real '+old_mode+' output')
    if mode=='restore':require(raw==previous['own']['raw'],'Restore also equals previous own bytes')
    return dict(previous_mode=old_mode,output=old['reference'],full_raw_bytes_equal=True)
