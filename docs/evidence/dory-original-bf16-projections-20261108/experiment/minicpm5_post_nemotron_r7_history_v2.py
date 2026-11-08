"""Future Mini admission requires actual finalized r7 history and external exit.

No expected successful outcome, future digest or process completion is invented.
The admitted spec must bind the files produced after the real r7 terminal event.
"""
import hashlib,json,re
from pathlib import Path
import nemotron9b_package_r7_completed_history_v2 as history
import minicpm5_post_dory_history_v1 as legacy
import nemotron9b_package_metadata_v15 as current_metadata
R,Q=history.R,history.Q
NON_UTF8=legacy.NON_UTF8
python_source_bytes=legacy.python_source_bytes
CATALOG_PATH=history.CURRENT
EXIT_PATH=Q/'nemotron9b-r7-finalizer-external-exit-20261108-v1.json'
require=history.require


def text(path,expected=None):
    path=Path(path)
    if path==Path(history.SPEC['path']):
        require(expected==history.SPEC['sha256'],'Exact original v15 runtime specification')
        return current_metadata.text(path,expected)
    return history.text(path,expected)


def sha(path,expected=None):return hashlib.sha256(text(path,expected)).hexdigest()
def load(reference):return json.loads(text(reference['path'],reference['sha256']))


def check_exit(catalog,receipt,catalog_ref):
    require(catalog_ref['path']==str(CATALOG_PATH)and re.fullmatch('[0-9a-f]{64}',catalog_ref['sha256']),'Exact actual completed r7 catalog reference')
    require(receipt['schema']=='nemotron9b_r7_finalizer_actual_external_exit_v1'and receipt['reviewer']=='/root','Actual root-observed finalizer exit')
    require(receipt['catalog']==catalog_ref and receipt['owner']==catalog['nemotron9b_package_r7_append']['finalizer_owner'],'Actual completed catalog and finalizer owner')
    actual=receipt['actual_exit']
    require(type(actual['session_id'])is int and actual['session_id']>0 and type(actual['exit_code'])is int and actual['exit_code']==0 and re.fullmatch('[0-9a-f]{6}',actual['tool_chunk']),'Actual successful finalizer tool return')
    require(all(receipt[k]is True for k in ('actual_tool_exit_observed','exact_owner_gone','catalog_validated','physical_drain_passed')),'Finalizer exit and release externally verified')
    require(receipt['services']==len(catalog['rows'])==1034 and receipt['processes']==len(catalog['processes']),'All actual preceding model/service/process history')
    require(receipt['new_speedup_claim']is False,'History completion adds no speedup claim')
    return receipt


def validate_binding(spec,catalog):
    require(spec['current_catalog']['path']==str(CATALOG_PATH)and spec['current_finalizer_exit']['path']==str(EXIT_PATH),'Only real complete r7 predecessor may admit successor')
    require(history.terminal.ref(CATALOG_PATH)==spec['current_catalog'],'Exact current completed catalog bytes')
    return check_exit(catalog,history.terminal.load(spec['current_finalizer_exit']),spec['current_catalog'])


def physical_drain(value,spec):
    receipt=validate_binding(spec,value)
    history.physical_drain(value)
    return dict(all_services=1034,all_known_processes=len(value['processes']),r7_finalizer_gone=True,maintenance_retained=True)
