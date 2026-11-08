"""Exact post-Dory history adapter for future Mini upstream runs; no launch."""
import hashlib
from pathlib import Path
import dory_rotary_completed_history_v1 as dory
import minicpm5_frame1_completed_history_v3 as legacy
R,Q=dory.R,dory.Q
NON_UTF8=legacy.NON_UTF8
python_source_bytes=legacy.python_source_bytes
CATALOG=dict(path=str(Q/'dory-rotary-r1-20261108-v1-complete-catalog-v1.json'),sha256='5709b6f85d994ffddd25461b4664cdadb86b594595a81b61d166ed211790cf83')
EXIT=dict(path=str(Q/'dory-rotary-r1-20261108-v1-finalizer-external-exit.json'),sha256='e80951e681d3a64e6edde9add37e6eeb13d1996d5b6c1645660ebab4a2d1d89e')

def text(path,expected=None):
    path=Path(path)
    # The installed v14 spec is 13.3 MB. The legacy v13 transport only
    # recognizes its own large schemas; retain the exact frozen v14 reader.
    if path==Path(dory.prior.SPEC['path']):
        dory.same(expected,dory.prior.SPEC['sha256'],'Exact historical v14 runtime spec')
        return dory.p.metadata.text(path,expected)
    return dory.text(path,expected)
def sha(path,expected=None):return hashlib.sha256(text(path,expected)).hexdigest()
def load(reference):
    if Path(reference['path'])==Path(dory.prior.SPEC['path']):
        import json
        return json.loads(text(reference['path'],reference['sha256']))
    return dory.load(reference)

def check_exit(catalog,receipt):
 dory.require(receipt['schema']=='dory_rotary_finalizer_external_exit_v1'and receipt['reviewer']=='/root','Actual Dory finalizer receipt')
 dory.same(receipt['catalog'],CATALOG,'Exact actual post-Dory catalog')
 dory.same(receipt['owner'],catalog['dory_rotary_append']['finalizer_owner'],'Actual Dory finalizer owner')
 dory.same(receipt['actual_exit'],dict(session_id=56360,exit_code=0,tool_chunk='277023'),'Actually observed Dory finalizer exit')
 dory.require(all(receipt[k]is True for k in ('actual_tool_exit_observed','exact_owner_gone','catalog_validated','physical_drain_passed')),'Dory completion independently observed')
 dory.require(receipt['services']==len(catalog['rows'])==978 and receipt['processes']==len(catalog['processes'])==190,'Complete actual Dory history')
 dory.require(receipt['full_model_qualified']is receipt['new_speedup_claim']is False,'No promotion of Dory arithmetic')
 return receipt

def validate_binding(spec,catalog):
 dory.same(spec['current_catalog'],CATALOG,'Only actual released Dory predecessor')
 dory.same(spec['current_finalizer_exit'],EXIT,'Exact actual finalizer exit receipt')
 return check_exit(catalog,dory.load_small(EXIT))

def physical_drain(value):
 check_exit(value,dory.load_small(EXIT))
 dory.physical_drain(value)
 return dict(all_services=978,all_known_processes=190,dory_finalizer_gone=True)
