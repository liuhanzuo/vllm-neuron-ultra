"""Pinned public Dory metadata/header fetch only. Never reads weight payloads."""
from pathlib import Path
import datetime
import hashlib
import json
import struct
import urllib.request

ROOT=Path('/home/ubuntu/experiments/trn1-latest-models-20261031/research')
OUT=ROOT/'dory-checkpoint-metadata-20261108-v1'
REPO='nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16'
REV='f6c832e6963ddb74f62c75813a3e8c74b1af67b1'
BASE='https://huggingface.co/'+REPO+'/resolve/'+REV+'/'
records=[]
def once(name,raw):
    with (OUT/name).open('xb') as f:f.write(raw)
    return dict(path=str(OUT/name),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
def fetch(name,url,cap,byte_range=None):
    headers={'User-Agent':'trn1-dory-metadata-audit/1.0','Accept-Encoding':'identity'}
    if byte_range:headers['Range']='bytes='+str(byte_range[0])+'-'+str(byte_range[1])
    req=urllib.request.Request(url,headers=headers)
    with urllib.request.urlopen(req,timeout=25) as response:
        status=response.status
        content_range=response.headers.get('Content-Range')
        length=response.headers.get('Content-Length')
        if byte_range:
            assert status==206,'Server ignored bounded header range'
            prefix='bytes '+str(byte_range[0])+'-'+str(byte_range[1])+'/'
            assert content_range and content_range.startswith(prefix),'Wrong header range'
            assert length is None or int(length)==byte_range[1]-byte_range[0]+1
        else:assert status==200
        raw=response.read(cap+1)
        assert len(raw)<=cap,'Metadata exceeds explicit cap'
        if byte_range:assert len(raw)==byte_range[1]-byte_range[0]+1
        record=dict(file=once(name,raw),requested_url=url,status=status,content_range=content_range,
                    content_length=length,etag=response.headers.get('ETag'),revision=REV)
        records.append(record)
        once('request-'+str(len(records))+'.json',(json.dumps(record,indent=2)+'\n').encode())
        print(json.dumps(dict(file=name,bytes=len(raw),status=status)),flush=True)
        return raw,record
def main():
    OUT.mkdir()
    api='https://huggingface.co/api/models/'+REPO+'/revision/'+REV+'?blobs=true'
    raw,_=fetch('repository.json',api,1024*1024)
    repo=json.loads(raw);assert repo['sha']==REV
    inventory={x['rfilename']:x for x in repo['siblings']}
    for name in ('config.json','tokenizer_config.json','special_tokens_map.json','generation_config.json','chat_template.jinja'):
        assert name in inventory
        fetch(name,BASE+name,2*1024*1024)
    raw,first=fetch('safetensors-prefix.bin',BASE+'model.safetensors?dory_header_range=prefix_20261108_v1',8,(0,7))
    length=struct.unpack('<Q',raw)[0];assert 0<length<=1024*1024
    raw,second=fetch('safetensors-header.json',BASE+'model.safetensors?dory_header_range=header_20261108_v1',length,(8,7+length))
    assert first['content_range'].split('/')[1]==second['content_range'].split('/')[1]
    header=json.loads(raw);assert isinstance(header,dict)
    summary=dict(schema='dory_pinned_metadata_fetch_v1',repository=REPO,revision=REV,
        observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),requests=records,
        safetensors_header_bytes=length,remote_file_bytes=int(second['content_range'].split('/')[1]),
        model_weight_payload_bytes_read=0,model_framework_imported=False,model_executed=False)
    once('fetch-summary.json',(json.dumps(summary,indent=2)+'\n').encode())
    print(json.dumps(dict(completed=True,requests=len(records),header_bytes=length,total_bytes=sum(r['file']['bytes']for r in records),weight_payload_bytes=0)))
if __name__=='__main__':main()
