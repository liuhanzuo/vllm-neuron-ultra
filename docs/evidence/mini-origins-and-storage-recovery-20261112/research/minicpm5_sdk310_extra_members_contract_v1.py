"""Retain original bytes/resources; bind 106 old sources plus six exact successors."""
from dory_serial_stage_contract_v2 import *

SOURCE_COUNT_LIMIT = 112


def verify_sources(pins,guard=lambda:None):
    require(type(pins)is dict and 1<=len(pins)<=112,'Bounded 112-source supplemental member source closure')
    total=0
    for name,digest in pins.items():
        guard();require(Path(name).is_relative_to(ROOT)and Path(name).suffix=='.py','Local source closure only')
        ref,_=reference(name,2*MIB);total+=ref['bytes']
        require(total<=8*MIB and ref['sha256']==digest,'Complete exact local source SHA')
    return total
