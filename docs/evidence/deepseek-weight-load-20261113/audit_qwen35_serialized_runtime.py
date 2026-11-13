"""Bounded structural read of the actual failed Qwen model archive, without Torch.

NEFF bytes and embedded graph definitions are streamed to hashes. Only small
runtime protobuf interface fields become Python values; no protobuf package or
compiler/model import is needed.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import resource
import struct
import sys
import zipfile
from audit_lfm3_serialized_cache_interfaces import DescriptorReader

ROOT=Path(__file__).resolve().parent
DEST=ROOT/'qwen35-serving-failure'


def varint(read, first=None):
    value=shift=0
    raw=bytearray()
    for index in range(10):
        part=first if index==0 and first is not None else read(1)
        assert len(part)==1
        raw.extend(part);value|=(part[0]&127)<<shift
        if part[0]<128:return value,bytes(raw)
        shift+=7
    raise AssertionError('Oversized protobuf varint')


def fields(payload):
    stream=io.BytesIO(payload);result=[]
    while stream.tell()<len(payload):
        key,_=varint(stream.read);number,wire=key>>3,key&7
        assert number>0
        if wire==0:value,_=varint(stream.read)
        elif wire==2:
            length,_=varint(stream.read);value=stream.read(length);assert len(value)==length
        else:raise AssertionError(('Unsupported interface wire type',wire))
        result.append((number,wire,value))
    assert stream.tell()==len(payload)
    return result


def tensor(payload):
    result=dict(name='',shape=[],data_type=0,type=0)
    for number,wire,value in fields(payload):
        if number in [1,6,8]:
            assert wire==2
            result[{1:'name',6:'checkpoint_key',8:'user_input_key'}[number]]=value.decode('utf8')
        elif number==2:
            if wire==0:result['shape'].append(value)
            else:
                stream=io.BytesIO(value)
                while stream.tell()<len(value):result['shape'].append(varint(stream.read)[0])
        elif number in [3,5,7]:
            assert wire==0
            result[{3:'data_type',5:'allow_dynamic_batch_size',7:'type'}[number]]=value
        elif number==4:
            result['constant_content_bytes']=len(value)
            result['constant_content_sha256']=hashlib.sha256(value).hexdigest()
        else:raise AssertionError(('Unknown tensor field',number))
    return result


class Payload:
    def __init__(self,size,digest,interface=None):
        self.size,self.digest,self.interface=size,digest,interface


class RuntimeReader(DescriptorReader):
    dispatch=DescriptorReader.dispatch.copy()
    def load_binary_unicode(self):
        length=struct.unpack('<I',self.read(4))[0]
        if length<=1<<20:
            self.append(self.read(length).decode('utf8','surrogateescape'));return
        remaining=length;digest=hashlib.sha256()
        def take(size):
            nonlocal remaining
            assert 0<=size<=remaining
            raw=self.read(size);assert len(raw)==size
            remaining-=size;digest.update(raw);return raw
        first=take(1)
        if first==b'\x02':
            assert take(7)==b'\0'*7, 'Expected known serialized NEFF version2 header'
            while remaining:take(min(65536,remaining))
            interface=None
        else:
            assert first==b'\x0a', 'Expected MetaNeff input_tensors as first field'
            interface=dict(inputs=[],outputs=[],aliases={},graph_definition=None,scalars={})
            initial=first
            while remaining or initial:
                key,_=varint(take,initial);initial=None
                number,wire=key>>3,key&7
                if wire==0:
                    value,_=varint(take);interface['scalars'][str(number)]=value
                else:
                    assert wire==2
                    size,_=varint(take);assert size<=remaining
                    if number==4:
                        graph_hash=hashlib.sha256();left=size
                        while left:
                            raw=take(min(65536,left));graph_hash.update(raw);left-=len(raw)
                        interface['graph_definition']=dict(bytes=size,sha256=graph_hash.hexdigest())
                        continue
                    assert size<1<<20, ('Only graph_def may be a large MetaNeff field',number,size)
                    value=take(size)
                    if number in [1,2]:interface['inputs' if number==1 else 'outputs'].append(tensor(value))
                    elif number==6:
                        pair={n:v for n,w,v in fields(value)}
                        assert set(pair)<={1,2}
                        out,inp=pair.get(1,0),pair.get(2,0)
                        assert out not in interface['aliases'];interface['aliases'][out]=inp
                    elif number==3:interface['model_config']={str(n):v for n,w,v in fields(value)}
                    elif number==5:interface['name']=value.decode('utf8')
                    else:raise AssertionError(('Unknown MetaNeff bytes field',number))
            assert remaining==0 and interface['graph_definition'] is not None
        self.append(Payload(length,digest.hexdigest(),interface))


RuntimeReader.dispatch[pickle.BINUNICODE[0]]=RuntimeReader.load_binary_unicode


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(65536),b''):h.update(data)
    return h.hexdigest()


def main():
    resource.setrlimit(resource.RLIMIT_AS,(256<<20,256<<20))
    resource.setrlimit(resource.RLIMIT_CPU,(30,30))
    output=DEST/'serialized-runtime-interface-audit.json'
    assert not output.exists()
    case_path=ROOT/'qwen35-2b-baseline-skip256.json';case=json.loads(case_path.read_text())
    artifact=Path(case['artifact'])/'model.pt'
    manifest_path=ROOT/'qwen35-skip256-baseline/compiled-graphs-audit.json'
    manifest=json.loads(manifest_path.read_text())
    config_path=artifact.with_name('neuron_config.json');config=json.loads(config_path.read_text())
    plan=json.loads((DEST/'plan.json').read_text())
    stat=artifact.stat();actual_identity=dict(realpath=str(artifact.resolve()),device=stat.st_dev,inode=stat.st_ino,
        size=stat.st_size,mtime_ns=stat.st_mtime_ns,ctime_ns=stat.st_ctime_ns)
    assert actual_identity==plan['large_artifact_identities'][str(artifact)]
    with zipfile.ZipFile(artifact) as archive:
        names=[n for n in archive.namelist() if n.endswith('/data.pkl')];assert len(names)==1
        with archive.open(names[0]) as stream:obj=RuntimeReader(stream).load()
    state=obj.state['nxd_model'].state
    assert len(state['input_shape_map'])==11
    init=state['state_initializer'].state
    assert len(init['shapes'])==48
    routes={}
    for key,(kind,index) in state['input_shape_map'].items():
        shapes=json.loads(key);routes[(kind,index)]=shapes
    expected={('context_encoding_model',i):b for i,b in enumerate([128,512,1024,2048,4096])}
    expected.update({('token_generation_model',i):b for i,b in enumerate([128,256,512,1024,2048,4096])})
    cache_graphs={(r['kind'],r['bucket_index']):r for r in manifest['graphs']}
    assert set(routes)==set(expected)==set(cache_graphs)
    graphs=[]
    for kind in ['context_encoding_model','token_generation_model']:
        for index,model in enumerate(state['models'].state[kind].state['models']):
            neff,meta,tag,local,world=model.state
            assert isinstance(neff,Payload) and isinstance(meta,Payload)
            assert neff.interface is None and meta.interface is not None
            assert local==world==2
            old=cache_graphs[(kind,index)]
            assert neff.digest==old['neff_sha256'] and neff.size==old['neff_bytes']
            interface=meta.interface;inputs=interface['inputs']
            assert all(t['name']=='input'+str(i) for i,t in enumerate(inputs))
            counts=Counter(t['type'] for t in inputs)
            assert counts[2]==48 and len(interface['aliases'])==48
            assert set(interface['aliases'].values())=={i for i,t in enumerate(inputs) if t['type']==2}
            assert set(interface['aliases'])==set(range(1,49))
            for out,inp in interface['aliases'].items():
                assert interface['outputs'][out]['shape']==inputs[inp]['shape']
                assert interface['outputs'][out]['data_type']==inputs[inp]['data_type']
                assert inputs[inp]['checkpoint_key'] in init['shapes']
                assert inputs[inp]['shape']==init['shapes'][inputs[inp]['checkpoint_key']]
            shapes=routes[(kind,index)];bucket=expected[(kind,index)]
            assert shapes[0]==([1,bucket] if kind=='context_encoding_model' else [4,1])
            assert shapes[1][1]==bucket
            graphs.append(dict(kind=kind,index=index,bucket=bucket,route_shapes=shapes,
                serialization_tag=tag,local_ranks=local,world_size=world,
                neff_bytes=neff.size,neff_sha256=neff.digest,metaneff_bytes=meta.size,
                metaneff_sha256=meta.digest,input_type_counts=dict(counts),interface=interface,
                compiled_cache_hlo_sha256=old['hlo_sha256']))
    def semantic(t):return {k:v for k,v in t.items() if k!='name'}
    common_weights=[semantic(t) for t in graphs[0]['interface']['inputs'] if t['type']==3]
    common_states=[semantic(t) for t in graphs[0]['interface']['inputs'] if t['type']==2]
    def named(rows):
        assert len({t['checkpoint_key'] for t in rows})==len(rows)
        return {t['checkpoint_key']:t for t in rows}
    for g in graphs:
        weights=[semantic(t) for t in g['interface']['inputs'] if t['type']==3]
        assert named(weights)==named(common_weights), (g['kind'],g['index'])
    assert all(named([semantic(t) for t in g['interface']['inputs'] if t['type']==2])==named(common_states) for g in graphs)
    source_paths=[Path(__file__),ROOT/'audit_lfm3_serialized_cache_interfaces.py',case_path,config_path,
                  manifest_path,ROOT/'qwen35_neuron_adapter.py',ROOT/'qwen35_text_port.py',DEST/'plan.json']
    assert 'torch' not in sys.modules
    result=dict(complete=True,observed_at=datetime.now(timezone.utc).isoformat(),artifact=str(artifact),
        artifact_identity=actual_identity,previously_verified_archive_sha256=plan['known_large_sha256'][str(artifact)],
        archive_hash_not_recomputed=True,graphs=graphs,state_initializer=init,
        nxd_state_keys=list(state),weight_name_to_idx=state.get('weight_name_to_idx'),
        common_weight_count=len(common_weights),common_state_count=len(common_states),
        all_eleven_neff_hashes_match_compiled_cache_audit=True,all_eleven_named_weight_shapes_types_equal=True,
        all_eleven_named_state_shapes_types_equal=True,all_528_state_aliases_have_matching_shapes_and_types=True,
        raw_weight_order_equal_across_graphs=all([semantic(t) for t in g['interface']['inputs'] if t['type']==3]==common_weights for g in graphs),
        raw_state_order_equal_across_graphs=all([semantic(t) for t in g['interface']['inputs'] if t['type']==2]==common_states for g in graphs),
        source_sha256={str(p):digest(p) for p in source_paths},
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        scope='Actual failedmodel.pt runtime interfaces andstreamedpayload hashes only. ElevenNEFFs match archivedcache hashes. '
              'All48state aliases pergraph match declaredinitializers; weight/state keys,shapes,types identicalwhenmatchedbycheckpointname. '
              'Original positionalorders retained and equalityreported separately; graph-specific runtimeordering stillneedsverification. '
              'Positionalinput names differ betweencontext4 userinputs anddecode3; eachname matches itsactual inputindex. '
              'No actualloadedweightvalues,rawdevicestate,HLOarithmetic,numericalcorrectness orperformance proof. '
              'Previous wholearchivehash bound by unchangedrecordedfilesystemidentity,notreread here.')
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['complete','common_weight_count','common_state_count',
        'all_eleven_neff_hashes_match_compiled_cache_audit','all_528_state_aliases_have_matching_shapes_and_types','peak_rss_bytes']}))


if __name__=='__main__':main()
