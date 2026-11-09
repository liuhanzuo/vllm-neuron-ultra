"""Require actual completed storage maintenance after the full residual review."""
import nemotron9b_residual_completion_after_retry_v1 as previous
from research import nemotron9b_after_completed_storage_retry_v1 as maintenance

base_history = previous.base_history
residual = previous.residual
require = previous.require


def validate_spec(spec,base):
    prior = previous.validate_spec(spec,base)
    exit_ref = spec.get('required_completed_storage')
    require(isinstance(exit_ref,dict) and set(exit_ref)=={'path','sha256'},
            'Actual completed maintenance external exit is mandatory')
    completed = maintenance.completed(exit_ref)
    history = completed['history']
    require(history['rows']==prior['rows'] and history['processes'][:-1]==prior['processes']
            and len(history['rows'])==1149 and len(history['processes'])==350
            and history['known_admitted_directories']==prior['known_admitted_directories']
            and len(history['known_admitted_directories'])==38,
            'All original services, owners and Dory directories plus one actual maintenance owner')
    require(history['completed_storage_maintenance']==exit_ref
            and history['completed_residual_review']==spec['required_residual_completion']
            and history['failed_residual_saved_review']==prior['failed_residual_saved_review'],
            'Exact ordered residual then maintenance completion')
    require(all(spec['source_sha256'].get(k)==v for k,v in completed['source_sha256'].items()),
            'Full original and actual maintenance source/evidence closure')
    require(completed['historical_binary_sha256']==spec['dory_historical_payload_sha256']
            and len(completed['historical_binary_sha256'])==6,
            'All six original Dory binary pins retained unchanged')
    require(completed['fresh_physical_drain_verified'] is completed['runtime_admitted'] is False
            and completed['full_model_qualified'] is completed['new_speedup_claim'] is False,
            'Metadata completion never grants runtime admission or model qualification')
    return history
