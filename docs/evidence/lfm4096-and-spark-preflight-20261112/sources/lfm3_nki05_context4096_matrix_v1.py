"""Additive actual LFM KV4096 matrices; every prior matrix remains separate.

These are fixed source-level plans. No native imports, files, or execution on
import. Capacity/row coverage follows the original B4/B8/B16 model contract.
"""
from lfm3_nki05_device_matrix_v1 import BoundMatrix as OriginalBoundMatrix
from lfm3_nki05_device_matrix_v1 import alias_mappings as original_alias_mappings
from lfm3_nki05_device_supplements_v2 import control_mappings

CAPACITIES=(4,8,16)
LENGTH=4096
POSITION_BOUNDARIES=(0,127,128,255,256,511,512,1023,1024,2047,2048,4094,4095)


def require(value,message):
    if not value:raise RuntimeError(message)


def mappings(capacity,rows,exhaustive_winners=False):
    """Original mapping algorithm extended to B4, plus signed-int64 controls."""
    require(type(capacity) is int and capacity in CAPACITIES and type(rows) is int
            and 1<=rows<=capacity,'Exact supported capacity and rows')
    result=[]
    def add(value):
        require(len(value)==rows,'Exact mapping row count')
        if value not in result:result.append(value)
    for shift in range(capacity):add([(i+shift)%capacity for i in range(rows)])
    add([(capacity-1-i)%capacity for i in range(rows)])
    for missing in (-1,-9,capacity,capacity+9,-(2**31),2**31-1):add([missing]*rows)
    for destination in range(capacity):
        winners=range(rows) if exhaustive_winners else sorted({0,rows//2,rows-1})
        for winner in winners:
            single=[-1]*rows;single[winner]=destination;add(single)
            add([destination if i<=winner else -1 for i in range(rows)])
    add([i%capacity if i%2 else -1 for i in range(rows)])
    add([i//2 for i in range(rows)])
    for value in control_mappings(capacity,rows):add(value)
    return result


def device_specs():
    return [dict(name=f'b{b}-lfm-kv4096-r{rows}',capacity=b,rows=rows,
                 shape=[b,2,LENGTH,128],dtype='bfloat16')
            for b in CAPACITIES for rows in sorted({1,4,b})]


def alias_specs():
    return [dict(name=f'b{b}-lfm-kv4096-{variant}',capacity=b,kind='kv4096',
                 length=LENGTH,variant=variant)
            for b in CAPACITIES for variant in ('reference','candidate')]


def alias_mappings(capacity):
    require(type(capacity) is int and capacity in CAPACITIES,'Exact alias capacity')
    # Preserve all original eight ordered steps, then signed-int64 controls.
    # The final three steps repopulate, leave inactive, and reuse all slots.
    return original_alias_mappings(capacity)+control_mappings(capacity,capacity)+[
        list(range(capacity)),[-1]*capacity,list(reversed(range(capacity)))]


def alias_positions(capacity,step):
    require(type(capacity) is int and capacity in CAPACITIES and type(step) is int
            and 0<=step<len(alias_mappings(capacity)),'Exact persistent alias step')
    # Every slot visits both sides of 2048 and the final valid index 4095.
    return [[POSITION_BOUNDARIES[(step+i)%len(POSITION_BOUNDARIES)]]
            for i in range(capacity)]


class BoundMatrix(OriginalBoundMatrix):
    mappings=staticmethod(mappings)
    device_specs=staticmethod(device_specs)
    alias_specs=staticmethod(alias_specs)
    alias_mappings=staticmethod(alias_mappings)
    alias_positions=staticmethod(alias_positions)
