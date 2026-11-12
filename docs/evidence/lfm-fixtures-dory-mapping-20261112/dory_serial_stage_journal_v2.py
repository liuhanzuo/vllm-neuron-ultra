"""Bounded complete event journal; all writes use the supplied serial guard."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import dory_serial_stage_contract_v2 as c


class Journal:
    """Complete JSONL events,256MiB raw/64MiB gzip; each event durable."""
    def __init__(self,path,guard):
        self.path=Path(path);self.guard=guard;self.raw=0;self.compressed=0;self.events=0
        self.digest=hashlib.sha256();self.durable=dict(events=0,raw=0,compressed=0);self.closed=False
        self.file=self.path.open('xb')
        try:self.stream=gzip.GzipFile(fileobj=self,mode='wb',mtime=0)
        except BaseException:self.file.close();self.closed=True;raise
    def write(self,data):
        self.guard();c.require(self.compressed+len(data)<=64*c.MIB,'Complete gzip journal cap')
        written=self.file.write(data);c.require(written==len(data),'Complete gzip write');self.compressed+=written;return written
    def flush(self):self.file.flush()
    def tell(self):return self.file.tell()
    def emit(self,value):
        pending=bytearray()
        def write_block(block):
            self.guard();c.require(self.raw+len(block)<=256*c.MIB,'Complete raw journal cap')
            self.raw+=len(block);self.digest.update(block);self.stream.write(block)
        for part in json.JSONEncoder(separators=(',',':'),allow_nan=False).iterencode(value):
            raw=part.encode()
            start=0
            while start<len(raw):
                count=min(65536-len(pending),len(raw)-start)
                pending.extend(raw[start:start+count]);start+=count
                if len(pending)==65536:write_block(pending);pending.clear()
        pending.extend(b'\n');write_block(pending)
        self.stream.flush();self.file.flush();os.fsync(self.file.fileno());self.events+=1
        self.durable=dict(events=self.events,raw=self.raw,compressed=self.file.tell())
    def close(self):
        if not self.closed:
            try:self.stream.close();self.file.flush();os.fsync(self.file.fileno())
            finally:self.file.close();self.closed=True
    def verify(self):
        c.require(self.closed,'Closed journal before roundtrip');h=hashlib.sha256();n=0
        with gzip.open(self.path,'rb')as stream:
            while True:
                self.guard();part=stream.read(65536)
                if not part:break
                n+=len(part);c.require(n<=self.raw,'Bounded journal roundtrip');h.update(part)
        c.require(n==self.raw and h.digest()==self.digest.digest(),'Complete lossless journal')
        h=hashlib.sha256();n=0
        with self.path.open('rb')as stream:
            while True:
                self.guard();part=stream.read(65536)
                if not part:break
                n+=len(part);h.update(part)
        c.require(n==self.compressed,'Complete compressed bytes')
        return dict(path=str(self.path),bytes=n,sha256=h.hexdigest(),raw_bytes=self.raw,raw_sha256=self.digest.hexdigest(),events=self.events)

