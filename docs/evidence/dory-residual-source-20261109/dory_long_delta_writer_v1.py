"""Lossless KV append capture; checks actual prefixes against saved SHA, never aliases.

Torch is injected by an admitted worker. All output rows are recorded before any
alignment rejection; changed prefixes fall back to full payloads for diagnosis.
"""
import hashlib


class DeltaWriter:
    def __init__(self, out, torch, guard):
        self.out,self.torch,self.guard=out,torch,guard
        self.previous={}

    def reset(self, slot):
        self.previous={key:value for key,value in self.previous.items() if key[1]!=slot}

    def raw(self,t):
        return t.detach().contiguous().view(self.torch.uint8).numpy().tobytes()

    def write(self,event,path,slot,epoch,start,end,tensors):
        name=f'{event:06d}-{path}.bin'
        result=dict(file=name,slot=slot,epoch=epoch,path=path,start=start,end=end,tensors={})
        offset,h=0,hashlib.sha256()
        with (self.out/name).open('xb') as f:
            for label,t in tensors.items():
                self.guard()
                is_cache=label.startswith('cache-')
                assert t.dtype in (self.torch.bfloat16,self.torch.float32)
                key=(path,slot,epoch,label)
                prefix_ok=True
                whole=self.raw(t)
                full_sha=hashlib.sha256(whole).hexdigest()
                row=dict(shape=list(t.shape),dtype='BF16' if t.dtype==self.torch.bfloat16 else 'F32',
                         full_sha256=full_sha,mode='full')
                if is_cache:
                    assert t.dtype==self.torch.bfloat16 and list(t.shape)==[4,end,256]
                    prior=self.previous.get(key)
                    prefix=self.raw(t[:,:start,:])
                    actual_prefix=hashlib.sha256(prefix).hexdigest()
                    prefix_ok=(prior is None and start==0) or (prior is not None and prior['length']==start and prior['sha256']==actual_prefix and prior['finite'])
                    row.update(prefix_sha256=actual_prefix,expected_previous=prior,prefix_valid=prefix_ok)
                    if prefix_ok:
                        payload=self.raw(t[:,start:,:])
                        row['mode']='append'
                        finite=bool(self.torch.isfinite(t[:,start:,:]).all())
                    else:
                        payload=whole
                        finite=bool(self.torch.isfinite(t).all())
                    self.previous[key]=dict(length=end,sha256=full_sha,finite=finite and prefix_ok)
                else:
                    payload=whole
                    finite=bool(self.torch.isfinite(t).all())
                row.update(offset=offset,bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest(),finite=finite)
                f.write(payload);h.update(payload);offset+=len(payload)
                result['tensors'][label]=row
        result.update(bytes=offset,sha256=h.hexdigest())
        return result
