"""Bind actual completed Dory residual review to a future Nemotron successor.

No exit reference exists by default, and no launch, payload read, or cleanup is
performed. Runtime callers must provide the exact root-observed saved-review
exit and retain the full original metadata/source closure.
"""
from research import nemotron9b_after_dory_saved_retry_v2 as residual
import nemotron9b_post_dory_history_v1 as base_history
import nemotron9b_after_journal_compaction_v1 as maintenance

require=base_history.require


def validate_spec(spec,base):
    require(base==base_history.load(base_history.CATALOG),'Exact original immutable base catalog')
    base_history.finalizer(base,base_history.load(base_history.EXIT))
    require(spec.get('required_residual_source_anchor')==residual.PRODUCER,
            'Exact admitted residual producer anchor required')
    require(spec.get('required_saved_retry_query_observation')==residual.retry.QUERY_OBSERVATION
            and spec['source_sha256'].get(residual.retry.QUERY_OBSERVATION['path'])==
                residual.retry.QUERY_OBSERVATION['sha256'],
            'Exact source-pinned retry query observation required')
    exit_ref=spec.get('required_residual_completion')
    require(isinstance(exit_ref,dict)and set(exit_ref)=={'path','sha256'},
            'Actual completed residual saved-review exit is mandatory')
    require(spec.get('required_attention_completion')==maintenance.attention.EXIT
            and spec.get('required_journal_compaction')==maintenance.EXIT,
            'All original attention and maintenance requirements retained')
    completed=residual.completed(exit_ref)
    h=completed['history']
    require(h['base_catalog']==base_history.CATALOG
            and h['rows'][:len(base['rows'])]==base['rows']
            and h['processes'][:len(base['processes'])]==base['processes']
            and h['complete_count']==len(h['rows'])==1149 and len(h['processes'])==349
            and len(h['known_admitted_directories'])==38,
            'Complete exact original catalog plus r9 and terminal Dory extension')
    require(isinstance(h.get('failed_residual_saved_review'),dict)
            and set(h['failed_residual_saved_review'])=={'path','sha256'}
            and h['failed_residual_saved_review']!=exit_ref,
            'Original failed review and successful retry remain distinct')
    require(h['completed_attention_review']==spec['required_attention_completion']
            and h['completed_journal_compaction']==spec['required_journal_compaction']
            and h['completed_residual_review']==exit_ref
            and h['r9_outcome']['producer_qualified'] is False and h['r9_outcome']['failed_release'] is True,
            'All actual old completions and r9 failure retained')
    require(all(spec['source_sha256'].get(path)==digest for path,digest in completed['source_sha256'].items()),
            'Complete actual residual exit, metadata and original source closure')
    require(spec.get('dory_historical_payload_sha256')==completed['historical_binary_sha256']
            and len(completed['historical_binary_sha256'])==6,
            'Exact six original Dory binary pins retained separately')
    require(completed['runtime_admitted'] is completed['fresh_physical_drain_verified'] is False
            and completed['full_model_qualified'] is completed['new_speedup_claim'] is False,
            'Metadata completion never grants physical release or model qualification')
    return h
