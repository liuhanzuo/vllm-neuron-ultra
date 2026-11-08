"""Exact latest completed Mini/Dory/r6 history and actual finalizer exit."""
from pathlib import Path
import minicpm5_upstream_completed_history_v1 as h
import nemotron9b_post_mini_history_v1 as old
R,Q=h.R,h.Q
CURRENT=h.CURRENT
SCHEMA=h.SCHEMA
CATALOG=dict(path=str(CURRENT),sha256='c10efe16810816c6dc2107dc07f8c67a28a5fe91c2a532dbf38b44df1fd71a41')
EXIT=dict(path=str(Q/'minicpm5-frame1-upstream-r1-20261108-v1-finalizer-external-exit.json'),sha256='cf10b2ffb9a50dd3136cdb301e6d68b595c79e65a95d9103740b13254af6cb38')
MISSING_ACK_UNIT=old.prior.previous.UNIT
HISTORY_PINS={r['path']:r['sha256'] for r in (CATALOG,h.PRIOR,dict(path=str(h.previous.dory.prior.CURRENT),sha256='235218915d505824659fb029b75fd8c4046da2dfc9c5b124e12c862fd5d2a865'))}
require=h.require
def accepts(path):return str(path) in HISTORY_PINS
def text(path,expected=None):
 require(HISTORY_PINS.get(str(path))==expected,'Exact completed-history path/hash')
 return h.text(path,expected)
def validate(value):
 h.validate(value)
 require(len(value['rows'])==981 and len(value['processes'])==196,'All actual completed predecessors')
 return value
def finalizer(value,receipt):
 h.same(receipt['catalog'],CATALOG,'Exact latest finalizer catalog')
 h.same(receipt['owner'],h.small(value['minicpm5_upstream_append']['audit'])['finalizer_owner'],'Actual latest finalizer owner')
 h.same(receipt['actual_exit'],dict(session_id=77243,exit_code=0,tool_chunk='03b05b'),'Actual observed finalizer exit')
 require(receipt['schema']=='minicpm5_upstream_finalizer_actual_external_exit_v1'and receipt['reviewer']=='/root','Actual external receipt')
 require(all(receipt[k]is True for k in ('actual_tool_exit_observed','exact_owner_gone','catalog_validated','physical_drain_passed')),'Actual finalizer completion')
 require(receipt['services']==981 and receipt['processes']==196 and receipt['full_model_qualified']is receipt['new_speedup_claim']is False,'Diagnostic scope retained')
 return receipt

MAINTENANCE={'path': '/home/ubuntu/experiments/trn1-latest-models-20261031/research/nemotron9b-r7-maintenance-external-exit-20261108-v1.json', 'sha256': 'cac6e34df99e551c3e47b981aac56fa36bb5381da07a89b5688e545698cc99b9'}
MAINTENANCE_PINS={'/home/ubuntu/experiments/trn1-latest-models-20261031/research/nemotron9b-r7-space-retirement-20261108-v1-result.json': '85a0166fd13cd87f82a34ca31033664986367cf9caf0b9580b52b4fc11811631', '/home/ubuntu/experiments/trn1-latest-models-20261031/research/nemotron9b-r7-space-retirement-plan-20261108-v1.json': 'b482eae9246668dbbe137ebf175e8e9031751588be09e09f76486b30c966abd6', '/home/ubuntu/experiments/trn1-latest-models-20261031/research/retire_nemotron9b_r7_space_v1.py': '100ebd84705dc5ee258a02d55e25d4f78e17d486e04668b2a877e9f99e5ade0e', '/home/ubuntu/experiments/trn1-latest-models-20261031/research/nemotron9b-r7-space-retirement-20261108-v1-intent.json': '6b5edf4766ea55d4262ced02f6921d94c06994dc7659bfbdaba8e10cff2e5f69', '/home/ubuntu/experiments/trn1-latest-models-20261031/research/nemotron9b-r7-space-retirement-20261108-v1-hashes.json': 'a77e6d45901a5f47708d84b36cdd3ad73f78c6cb306381ee60478e63875b80af', '/home/ubuntu/experiments/trn1-latest-models-20261031/research/nemotron9b-r7-space-retirement-20261108-v1-actions.jsonl': 'f8efbe8967e9468b7e18ca9bd2d1afe75146e2111929a05cc21e2a9225ebe1e8', '/home/ubuntu/experiments/trn1-latest-models-20261031/research/inspect_nemotron9b_r7_retirement_processes_v1.py': '88a6cdfb44984686fa7038fa13ea0de29cf6b5b7e215853b467b46b3a17a2471', '/home/ubuntu/experiments/trn1-latest-models-20261031/research/nemotron9b-r7-maintenance-external-exit-20261108-v1.json': 'cac6e34df99e551c3e47b981aac56fa36bb5381da07a89b5688e545698cc99b9'}

def validate_maintenance(spec,load):
 h.same(spec['required_prelaunch_maintenance'],MAINTENANCE,'Exact completed r7 maintenance')
 receipt=load(MAINTENANCE);result=load(receipt['result']);plan=load(result['plan'])
 h.same(receipt['actual_exit'],dict(session_id=63644,exit_code=0,tool_chunk='59fd2a'),'Actual maintenance external exit')
 require(receipt['schema']=='nemotron9b_r7_maintenance_actual_external_exit_v1'and receipt['reviewer']=='/root','Exact actual maintenance receipt')
 require(receipt['actual_tool_exit_observed']is receipt['exact_owner_gone']is receipt['all9_removed_paths_absent']is True,'Completed maintenance externally verified')
 h.same(receipt['owner'],result['owner'],'Exact maintenance owner');h.m.p.gone(receipt['owner'])
 h.same(receipt['catalog'],CATALOG,'Same completed predecessor')
 require(result['complete']and all(result[k]is True for k in ('protected_unchanged','canonical_unchanged','original_payload_and_runtime_stats_unchanged','compiler_assets_unchanged')),'Original inputs and artifacts retained')
 require(result['actual_self_exit_claimed']is result['model_or_compiler_started']is result['new_speedup_claim']is False,'Maintenance scope only')
 require(len(plan['files'])==len(result['removed'])==9,'Exact bounded completed compiler intermediates')
 h.same(result['removed'],[x['path']for x in plan['files']],'Exact removed paths')
 require(all(not Path(x).exists()for x in result['removed']),'Retired completed intermediates remain absent')
 require(all(spec['source_sha256'].get(k)==v for k,v in MAINTENANCE_PINS.items()),'Complete maintenance evidence closure')
 return receipt
