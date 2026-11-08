"""Pure metadata decode/closure helpers for future post-drain replay export.

Consumes caller-authenticated JSON rows only. No model/payload imports or I/O.
The compressed graph input arrays were recorded by the new observer, not inferred
from decoded text. This does not run graph replay or claim physical equivalence.
"""
import base64
import hashlib
import struct
import zlib


def require(value, message):
    if not value:
        raise ValueError(message)


def decode_tensor(row):
    require(row['codec'] == 'zlib+base64', 'Input codec')
    dtypes = {'torch.int32': ('i',4), 'torch.int64': ('q',8),
              'torch.float32': ('f',4), 'torch.bfloat16': ('H',2),
              'torch.bool': ('?',1)}
    require(row['dtype'] in dtypes, 'Unexpected graph input dtype')
    fmt, size = dtypes[row['dtype']]
    count = 1
    for dimension in row['shape']:
        require(type(dimension) is int and dimension >= 0, 'Invalid shape')
        count *= dimension
    require(count * size == row['raw_bytes'] and row['raw_bytes'] <= 2**20, 'Input byte shape')
    layout=row['original_layout']
    require(len(layout['stride'])==len(row['shape']) and all(type(x)is int and x>=0 for x in layout['stride']),
            'Original strides missing')
    require(type(layout['storage_offset'])is int and layout['storage_offset']>=0 and
            type(layout['storage_nbytes'])is int and layout['storage_nbytes']>=0,'Original storage geometry')
    if count:
        max_element=layout['storage_offset']+sum((n-1)*s for n,s in zip(row['shape'],layout['stride']))
        require((max_element+1)*size<=layout['storage_nbytes'],'View exceeds original backing storage')
    packed = base64.b64decode(row['data'], validate=True)
    decompress = zlib.decompressobj()
    raw = decompress.decompress(packed, row['raw_bytes'] + 1)
    require(decompress.eof and not decompress.unconsumed_tail and not decompress.unused_data,
            'Truncated/trailing compressed input')
    require(len(raw) == row['raw_bytes'] and hashlib.sha256(raw).hexdigest() == row['raw_sha256'],
            'Raw input identity')
    values = list(struct.unpack('<' + fmt * count, raw))
    return dict(dtype=row['dtype'], shape=row['shape'], raw_sha256=row['raw_sha256'],
                values=values, original_layout=layout,bfloat16_values_are_raw_u16_bits=row['dtype']=='torch.bfloat16')


def export(rows, capture,retain_decoded=True):
    """Bind each API request to exact engine IDs and recorded scheduler steps."""
    require(capture['passed'] is True and capture['request_scope']==22 and len(capture['requests'])==22,
            'Requires successful complete fresh capture')
    require(rows and len(rows)<=50000, 'Bounded observer rows')
    require([x['event'] for x in rows]==list(range(1,len(rows)+1)), 'Missing/reordered observer event')
    pid=rows[0]['pid'];role=capture['role']
    require(all(x['pid']==pid and x['role']==role for x in rows), 'Mixed observer process or role')
    frames=[];samples={};steps={};finished=[];startup_frames=[];startup_samples=[];ever_engine=False
    current=None
    for row in rows:
        kind=row['kind'];step=row['engine_step']
        if kind=='engine_begin':
            require(current is None and step not in steps and type(step)is int,'Engine step nesting')
            current=step;steps[step]=dict(begin=row,frames=[],samples=[])
            ever_engine=True
            finished.extend(row['finished_req_ids'])
        elif kind=='engine_end':
            require(step==current and step in steps,'Engine end mismatch')
            steps[step]['end']=row;current=None
        elif kind=='graph_inputs':
            require(len(row['inputs'])==7,'Incomplete graph invocation')
            startup=row['scope']=='startup'
            if startup:
                require(step is None and current is None and not ever_engine,'Startup graph after request execution')
            else:
                require(row['scope']=='request' and step==current and step is not None,'Unbound graph invocation')
            decoded=[decode_tensor(x) for x in row['inputs']]
            require(row['tag'] in ('context_encoding_model','token_generation_model'),'Graph tag')
            if row['tag']=='token_generation_model':
                require(decoded[1]['shape'][1]==2048,'Decode floor changed')
            frame=dict(index=len(frames),engine_step=step,tag=row['tag'],route=row['route'],scope=row['scope'],
                       inputs=decoded,raw_inputs=row['inputs'])
            if not retain_decoded:
                frame['observer_event']=row['event']
                frame['inputs']=[{k:v for k,v in item.items()if k!='values'}for item in decoded]
                del frame['raw_inputs']
            frames.append(frame)
            if startup:startup_frames.append(frame['index'])
            else:steps[step]['frames'].append(frame['index'])
        elif kind in('sample','sample_failure')and row.get('scope')=='startup':
            require(step is None and current is None and not ever_engine,'Startup sample after request execution')
            # The original _dummy_run catches its own sampling exceptions. Keep
            # their evidence and do not reinterpret them as user requests.
            startup_samples.append(row)
        elif kind=='sample':
            require(row.get('scope')=='request','Unclassified sample')
            require(step==current and step is not None and row['on_device_sampling'] is False,'Sample outside CPU step')
            require(not steps[step]['samples'],'Multiple sample events in one engine step')
            order=row['sampled_request_order'];tokens=row['sampled_token_ids']
            require(len(order)==len(tokens) and len(set(order))==len(order),'Sample row order')
            require(set(order)==set(row['model_request_order']),'Model/request sample mapping')
            for request,tok in zip(order,tokens):
                require(request in row['request_to_slot'],'Missing physical slot')
                require(type(tok)is list and len(tok)==1 and type(tok[0])is int,'Unexpected speculative/partial sample')
                samples.setdefault(request,[]).append(dict(engine_step=step,token=tok[0],
                    physical_slot=row['request_to_slot'][request],frames=list(steps[step]['frames'])))
            steps[step]['samples'].append(row)
        else:
            require(kind in ('installed','model_load_begin','model_loaded'),'Observer failure/unknown event')
    require(current is None and frames and all('end'in x for x in steps.values()),'Incomplete engine recording')
    api=[]
    for request in capture['requests']:
        engine_id=request['request_id']+'-0'  # exact pinned completion.py request_id_item
        require(engine_id in samples,'API request not present in engine trace')
        stream=samples[engine_id]
        ids=[x['token']for x in stream]
        # No -1 filtering permitted here: admitted simple B1/B4 calls have no
        # chunked-prefill partial sample. A -1 means this export contract failed.
        require(ids==request['token_ids'],'Engine/API token sequence mismatch')
        require(all(0<=x['physical_slot']<4 for x in stream),'Invalid physical slot')
        api.append(dict(group=request['group'],index=request['index'],api_request_id=request['request_id'],
            engine_request_id=engine_id,prompt_token_ids=request['prompt_token_ids'],token_ids=ids,
            samples=stream))
    require(set(samples)=={x['engine_request_id']for x in api},'Unidentified engine request')
    return dict(passed=True,role=role,pid=pid,requests=api,frames=frames,
        engine_steps=[steps[x]for x in sorted(steps)],finished_req_ids=finished,
        startup_frame_indices=startup_frames,startup_frames_preserved=True,
        startup_sample_events=startup_samples,
        graph_request_logical_alignment_proven=False,replay_admitted=False,
        replay_pending=['Full percall seq_ids/token/position/mask correspondence to request prompts and prior sampled history',
            'Original storage alias relations and model.load internal effects are not reconstructed'],
        export_contains_original_graph_bytes=retain_decoded,
        original_graph_bytes_retained_in_observer_rows=True,teacher_forced_replay_executed=False,
        no_cache_snapshot_included=True,full_model_qualified=False,new_speedup_claim=False)
