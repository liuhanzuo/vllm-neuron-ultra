# SPDX-License-Identifier: Apache-2.0
"""Explicit diagnostic buffer inspection/clearing; never called by serving.

Normal serving retains the original request lifecycle. This API is only for
independently admitted cold-prefix validation of an already loaded model.
"""
import math
from .identity import EXPECTED_MODEL, require
STATE_BYTES=845283328


def state_layout():
    c=EXPECTED_MODEL;pattern=c["hybrid_override_pattern"]
    require(len(pattern)==56 and {k:pattern.count(k)for k in "M*-"}=={"M":27,"*":4,"-":25},"Original56-layer pattern")
    shapes=[]
    for kind in pattern:
        if kind=="M":
            shapes += [([4,6144,4],"torch.bfloat16"),([4,64,80,128],"torch.float32")]
        elif kind=="*":shapes += [([4,4,4096,128],"torch.bfloat16")]*2
    rows=sorted([dict(rank=rank,name="kv_mgr.past_key_values."+str(i),shape=shape,dtype=dtype)
                 for rank in (0,1)for i,(shape,dtype)in enumerate(shapes)],key=lambda x:(x["rank"],x["name"]))
    require(len(rows)==124 and sum(math.prod(x["shape"])*({"torch.bfloat16":2,"torch.float32":4}[x["dtype"]])for x in rows)==STATE_BYTES,"Original124-buffer state layout")
    return rows


def bindings(nxd_model):
    require(len(nxd_model.state)==2,"Two device state ranks required")
    rows=[]
    for rank,tensors in enumerate(nxd_model.state):
        for name,value in sorted(tensors.items()):
            require(value.device.type=="privateuseone" and value.device.index==rank,"Original state device")
            rows.append(dict(rank=rank,name=name,shape=list(value.shape),dtype=str(value.dtype),device=str(value.device),tensor_impl=int(value._cdata)))
    require([{k:x[k]for k in ("rank","name","shape","dtype")}for x in rows]==state_layout(),"State shape/dtype/name changed")
    require(len({x["tensor_impl"]for x in rows})==124 and all(x["tensor_impl"]>0 for x in rows),"Original distinct state bindings")
    return rows


def clear_for_cold_validation(nxd_model,expected_bindings):
    """Copy zeros in place, verify all bytes, and leave graph ownership intact."""
    import torch
    before=bindings(nxd_model);require(before==expected_bindings,"State bindings changed before explicit cold clear")
    for rank,tensors in enumerate(nxd_model.state):
        for name,value in sorted(tensors.items()):
            value.copy_(torch.zeros(tuple(value.shape),dtype=value.dtype,device="cpu"))
            require(int(torch.count_nonzero(value.detach().cpu().contiguous().view(torch.uint8)))==0,"Cold buffer not raw zero")
    require(bindings(nxd_model)==before,"Cold clear replaced tensor bindings")
    return dict(states_written=124,bytes_written=STATE_BYTES,bindings_unchanged=True,graph_initialization_called=False)
