"""Separate logical K equality from the actual consumer layout; stdlib only.

No relaxed bytes, dtype or shape check. A layout change is legal only at the
native post-RoPE/cache-to-attention edge, and is explicitly retained for replay.
"""
import copy,re
from minicpm5_frame1_attention_observer_v1 import require,same,geometry,layout
KEYS={'shape','stride','offset','dtype','sha256'}

def compare(producer,consumer,native):
 require(type(native)is bool,'Explicit native/CPU role')
 require(set(producer)==set(consumer)==KEYS,'Complete exact tensor metadata')
 expected=geometry(native)['k_post_rope']
 for meta in (producer,consumer):
  same(meta['shape'],expected['shape'],'Complete original K shape')
  same(meta['dtype'],expected['dtype'],'Original K precision')
  require(type(meta['sha256'])is str and re.fullmatch('[0-9a-f]{64}',meta['sha256']),'Complete logical K digest')
  layout(meta['shape'],meta['stride'],meta['offset'])
 equal=producer['sha256']==consumer['sha256'];same_layout=all(producer[k]==consumer[k]for k in ('stride','offset'))
 return dict(schema='minicpm5_attention_k_consumer_boundary_v1',native=native,producer=copy.deepcopy(producer),consumer=copy.deepcopy(consumer),logical_bytes_exact=equal,layout_exact=same_layout,allowed_layout_transition=native,passed=equal and(native or same_layout),consumer_layout_required_for_attention_replay=True,model_arithmetic_changed=False,new_speedup_claim=False)

def validate(row,native):
 expected=compare(row['producer'],row['consumer'],native)
 same(row,expected,'Every consumer-boundary conclusion independently rederived')
 require(row['passed']is True,'K boundary may change only native layout, never logical values')
 return row

def replay_row(saved,boundary,native):
 validate(boundary,native);same(saved['metadata'],boundary['producer'],'Saved RoPE K is original producer')
 same(saved['sha256'],boundary['consumer']['sha256'],'Exact full saved bytes are consumer values')
 row=copy.deepcopy(saved);row['metadata']=copy.deepcopy(boundary['consumer'])
 return row
