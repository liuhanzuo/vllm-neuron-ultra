"""Validate pinned Dory checkpoint and tokenizer metadata without native imports."""
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

Q=Path(__file__).resolve().parent
BUNDLE=Q/'dory-native-reference-source-20261107-v1'
sys.path.insert(0,str(BUNDLE))
from dory_contract import DoryContract
REV='f6c832e6963ddb74f62c75813a3e8c74b1af67b1'
REPO='nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16'
CONTRACT_SHA='4f0f9cdd3cb17ae23330944088dfc0334077b96463bb84e91b6194b8979fe95f'


def require(value,message):
    if not value:raise ValueError(message)


def unique_pairs(pairs):
    result={}
    for key,value in pairs:
        require(key not in result,'Duplicate JSON key: '+key)
        result[key]=value
    return result


def parse(raw):
    return json.loads(raw,object_pairs_hook=unique_pairs)


def checkpoint(header,expected,header_bytes,file_bytes):
    require(type(header_bytes)is int and 0<header_bytes<=2**20,'Bounded actual header')
    require(type(file_bytes)is int and file_bytes>header_bytes+8,'Positive data extent')
    metadata=header.get('__metadata__',{})
    require(isinstance(metadata,dict)and all(isinstance(k,str)and isinstance(v,str)for k,v in metadata.items()),'Text-only safetensors metadata')
    tensors={k:v for k,v in header.items()if k!='__metadata__'}
    require(set(tensors)==set(expected),'Every physical weight name, no extra tensors')
    intervals=[];elements=0
    for name,row in tensors.items():
        require(set(row)=={'dtype','shape','data_offsets'}and row['dtype']=='BF16','Exact BF16 tensor record')
        shape=row['shape'];require(isinstance(shape,list)and all(type(n)is int and n>0 for n in shape),'Positive integral shape')
        require(tuple(shape)==tuple(expected[name]),'Original checkpoint shape: '+name)
        offsets=row['data_offsets'];require(type(offsets)is list and len(offsets)==2 and all(type(x)is int for x in offsets),'Two byte offsets')
        begin,end=offsets;n=math.prod(shape)
        require(0<=begin<end and begin%2==0 and end-begin==2*n,'Exact complete BF16 interval')
        intervals.append((begin,end,name));elements+=n
    cursor=0
    for begin,end,name in sorted(intervals):
        require(begin==cursor,'No gap or overlap in checkpoint data')
        cursor=end
    require(cursor+8+header_bytes==file_bytes,'Exact file extent without trailing data')
    return dict(tensors=len(tensors),parameters=elements,data_bytes=cursor,file_bytes=file_bytes,
                header_bytes=header_bytes,all_names_shapes_match=True,dtypes=['BF16'],
                contiguous_complete_data_intervals=True,weight_values_read=False)


def tokenizer(tok,cfg,special,generation,vocab_size):
    model=tok['model'];require(model['type']=='BPE','Original BPE tokenizer')
    vocab=model['vocab'];ids=list(vocab.values())
    require(len(vocab)==vocab_size and all(type(x)is int for x in ids)and set(ids)==set(range(vocab_size)),
            'Unique complete vocabulary IDs matching model embeddings')
    added=tok['added_tokens'];require(len({x['id']for x in added})==len(added),'Unique added token IDs')
    require(len({x['content']for x in added})==len(added),'Unique added token strings')
    for row in added:
        require(type(row['id'])is int and vocab.get(row['content'])==row['id'],'Added token belongs to original vocabulary')
        require(cfg['added_tokens_decoder'][str(row['id'])]=={k:v for k,v in row.items()if k!='id'},'Tokenizer and config agree on complete added token record')
    require(set(cfg['added_tokens_decoder'])=={str(x['id'])for x in added},'Complete added token config')
    for name in ('bos_token','eos_token','unk_token','pad_token'):
        require(cfg[name]==special[name]['content']and cfg[name]in vocab,'Special token text matches')
    require(generation['bos_token_id']==vocab[cfg['bos_token']],'BOS ID')
    require(generation['pad_token_id']==vocab[cfg['pad_token']],'Pad ID')
    eos=generation['eos_token_id']
    require(type(eos)is list and len(set(eos))==len(eos)and all(type(x)is int and 0<=x<vocab_size for x in eos),'Complete generation stop IDs')
    require(vocab[cfg['eos_token']]in eos,'Tokenizer EOS included in generation stops')
    inverse={index:token for token,index in vocab.items()}
    merges=model['merges']
    require(all(type(pair)is list and len(pair)==2 and all(x in vocab for x in pair)and ''.join(pair)in vocab for pair in merges),
            'Complete BPE merge operands and results in vocabulary')
    return dict(vocabulary=vocab_size,added_tokens=len(added),merges=len(merges),
                stop_tokens=[dict(id=x,token=inverse[x])for x in eos],
                tokenizer_eos=dict(id=vocab[cfg['eos_token']],token=cfg['eos_token']),
                bos_id=generation['bos_token_id'],pad_id=generation['pad_token_id'],
                add_bos_token=cfg['add_bos_token'],add_eos_token=cfg['add_eos_token'],
                ignore_merges=model['ignore_merges'],tokenizer_executed=False,chat_template_rendered=False)


def main():
    root=Q/'dory-checkpoint-metadata-20261108-v1'
    require(hashlib.sha256((BUNDLE/'dory_contract.py').read_bytes()).hexdigest()==CONTRACT_SHA,'Frozen contract source')
    fetch=parse((root/'fetch-summary.json').read_bytes());token_fetch=parse((root/'tokenizer-fetch.json').read_bytes())
    pins={}
    for row in [x['file']for x in fetch['requests']]+[token_fetch['file']]:
        path=Path(row['path']);require(path.parent==root,'Exact metadata directory')
        raw=path.read_bytes();require(len(raw)==row['bytes']and hashlib.sha256(raw).hexdigest()==row['sha256'],'Actual metadata hash')
        pins[path.name]=row
    repo=parse((root/'repository.json').read_bytes());require(repo['id']==REPO and repo['sha']==REV,'Original pinned revision')
    config=parse((root/'config.json').read_bytes())
    require((root/'config.json').read_bytes()==(BUNDLE/'official-config.json').read_bytes(),'Original config byte identity')
    c=DoryContract.from_mapping(config);expected=c.weight_shapes()
    header_raw=(root/'safetensors-header.json').read_bytes()
    require(struct.unpack('<Q',(root/'safetensors-prefix.bin').read_bytes())[0]==len(header_raw),'Original header length prefix')
    file=next(x for x in repo['siblings']if x['rfilename']=='model.safetensors')
    require(file['size']==file['lfs']['size']==fetch['remote_file_bytes'],'Original remote file size agreement')
    report=checkpoint(parse(header_raw),expected,len(header_raw),file['size'])
    require(report['parameters']==repo['safetensors']['total']==repo['safetensors']['parameters']['BF16']<10**10,'Strict total parameter limit')
    tok=parse((root/'tokenizer.json').read_bytes());cfg=parse((root/'tokenizer_config.json').read_bytes())
    tok_report=tokenizer(tok,cfg,parse((root/'special_tokens_map.json').read_bytes()),
                         parse((root/'generation_config.json').read_bytes()),c.vocab_size)
    require(cfg['model_max_length']==c.max_position_embeddings,'Tokenizer and model context limit')
    template=(root/'chat_template.jinja').read_text()
    require(template==repo['config']['chat_template_jinja'],'Saved official model metadata template matches')
    require('chat_template'not in cfg,'Template must be loaded from separate pinned jinja file')
    file_tok=next(x for x in repo['siblings']if x['rfilename']=='tokenizer.json')
    require(file_tok['lfs']['sha256']==pins['tokenizer.json']['sha256'],'Full tokenizer LFS checksum')
    require(file_tok['size']==pins['tokenizer.json']['bytes'],'Full tokenizer bytes')
    histories=c.cache_plan();bpt=len(histories)*2*c.num_key_value_heads*c.head_dim*2
    out=dict(schema='dory_checkpoint_tokenizer_metadata_review_v1',reviewer='/root',revision=REV,repository=REPO,
        input_files=pins,contract_source_sha256=CONTRACT_SHA,checkpoint=report,tokenizer=tok_report,
        template=dict(file='chat_template.jinja',bytes=pins['chat_template.jinja']['bytes'],external_file_required=True,
                      default_enable_thinking=True,default_medium_effort=False,default_truncate_history_thinking=True,rendered=False),
        checkpoint_lfs=file['lfs'],checkpoint_full_lfs_hash_verified=False,
        architecture=dict(hidden_size=c.hidden_size,attention_projection_width=c.num_attention_heads*c.head_dim,
                          kv_projection_width=c.num_key_value_heads*c.head_dim,head_dim=c.head_dim,
                          physical_layers=c.num_hidden_layers,layer_visits=len(c.visits()),recurrent_loops=c.n_recurrent_loops,
                          attention_histories=len(histories),sliding_histories=sum(x['window']is not None for x in histories)),
        reference_cache_storage=dict(bytes_per_token=bpt,
            bytes_at_tokens={str(n):bpt*n for n in (1024,2048,4096,c.max_position_embeddings)},
            formula='56 histories * 2(K,V) * 4 KV heads * 256 head dim * 2 BF16 bytes * tokens',
            scope='Retained K/V logical storage only, full-history CPU reference. Excludes weights, activations, temporary copies and framework memory.',
            peak_memory_fit_proven=False),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        model_framework_imported=False,weights_loaded=False,model_executed=False,numerical_parity_proven=False,
        native_reference_qualified=False,neuron_baseline_qualified=False,new_speedup_claim=False)
    dest=root/'metadata-review.json'
    with dest.open('x')as f:f.write(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:out[k]for k in ('checkpoint','tokenizer','architecture','reference_cache_storage')},indent=2))
if __name__=='__main__':main()
