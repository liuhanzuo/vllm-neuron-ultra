"""Detached copy for admitted plain JSON trees; no cache or import-time I/O.

Unlike deepcopy, this deliberately rejects shared mutable containers and non-JSON
Python objects. Runtime permits have already been decoded into plain JSON trees.
Immutable scalar sharing is unobservable; every mutable output container is new.
"""
import json
import math

MAX_BYTES=32*1024*1024
MAX_DEPTH=48

def clone(value):
    if type(value)is not dict:raise ValueError('JSON permit root must be a plain dict')
    pending=[(value,0)];seen=set()
    while pending:
        current,depth=pending.pop()
        if depth>MAX_DEPTH:raise ValueError('JSON tree nesting bound')
        kind=type(current)
        if kind in (dict,list):
            identity=id(current)
            if identity in seen:raise ValueError('JSON tree cannot contain cycles or mutable aliases')
            seen.add(identity)
            if kind is dict:
                for key,child in current.items():
                    if type(key)is not str:raise ValueError('JSON tree keys must be plain strings')
                    pending.append((child,depth+1))
            else:
                for child in current:pending.append((child,depth+1))
        elif current is None or kind in (str,bool,int):pass
        elif kind is float:
            if not math.isfinite(current):raise ValueError('JSON tree numbers must be finite')
        else:raise ValueError('Plain JSON tree values only')
    raw=json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False)
    if len(raw.encode('utf8'))>MAX_BYTES:raise ValueError('JSON tree byte bound')
    result=json.loads(raw)
    if result!=value:raise ValueError('JSON tree changed during detached copy')
    return result
