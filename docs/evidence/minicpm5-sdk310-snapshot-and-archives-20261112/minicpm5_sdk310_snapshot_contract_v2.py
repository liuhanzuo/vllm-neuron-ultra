"""Narrow metadata source-count successor; all original resource gates retained.

The old 98 sources, two snapshot-stage sources and three read integration sources plus three explicit status-observation
sources need 106 entries. This changes only that count, not byte/file limits.
"""
from dory_serial_stage_contract_v2 import *

SOURCE_COUNT_LIMIT = 106


def verify_sources(pins,guard=lambda:None):
    require(type(pins)is dict and 1<=len(pins)<=106,'Bounded 106-source snapshot reader closure')
    total=0
    for name,digest in pins.items():
        guard();require(Path(name).is_relative_to(ROOT)and Path(name).suffix=='.py','Local source closure only')
        ref,_=reference(name,2*MIB);total+=ref['bytes']
        require(total<=8*MIB and ref['sha256']==digest,'Complete exact local source SHA')
    return total
