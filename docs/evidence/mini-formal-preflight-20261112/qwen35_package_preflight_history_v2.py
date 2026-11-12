"""One-stage use of the existing Qwen complete-history/physical collectors."""
import copy
import hashlib
from pathlib import Path

from qwen35_sampler_runtime_history_v12 import Bundle as OriginalHistory, Authority
from qwen35_sampler_runtime_io_v12 import Metadata, identity, loaded_local_sources, raw
from qwen35_sampler_runtime_contract_v12 import LOCK_IDENTITY, require


class History(OriginalHistory):
    """Reuse physical_pair/freshness/unchanged; do not bind the sampler phases."""
    def __init__(self, spec, spec_ref, review_ref, names, guard):
        self.spec, self.spec_ref, self.review_ref = copy.deepcopy(spec), spec_ref, review_ref
        self.names, self.guard, self.meta = names, guard, Metadata(guard)
        self.binding_identities = {r['path']: identity(r['path']) for r in (spec_ref, review_ref)}
        self.resolved = None

    def resolve(self):
        from research import minicpm5_layer1_operator_recovered_completion_v2 as completed
        from research import minicpm5_layer1_operator_native_overlay_v3 as overlays
        from research import minicpm5_layer1_operator_sdk_sources_v1 as sources
        from research import prepare_minicpm5_layer1_operator_native_v7 as witnesses
        from qwen35_sampler_history_supplement_v1 import build, extend_scope
        self.guard()
        recovered = completed.completed(completed.BINDING, self.meta.read, self.guard,
                                         lambda path, sha: sources.text(path, sha))
        overlay = self.meta.read(self.spec['history_overlay'])
        require(overlay['current_serial_catalog'] == self.spec['current_serial_catalog'], 'Complete then-current catalog binding')
        resolved = overlays.resolve(overlay, recovered, self.meta.read, self.guard)
        self.supplement = build(self.meta.read, self.spec['lfm_r2_scope'], self.meta.read(self.spec['current_serial_catalog']))
        resolved['current_scope'] = extend_scope(resolved['current_scope'], self.supplement)
        cache, witnessed = {}, {}
        for row in resolved['current_scope']['later_serial']['runs']:
            terminal = self.meta.read(row['terminal'])
            if terminal.get('schema') in ('minicpm5_native_model_exit_recorder_actual_exit_v1',
                                           'minicpm5_native_independent_audit_actual_exit_v1'):
                additions = witnesses.executed_extension(terminal, row, self.meta.read, self.guard, cache)
                for path, sha in additions.items():
                    require(path not in witnessed or witnessed[path] == sha, 'Completed source witness conflict')
                    witnessed[path] = sha
        pins = self.spec['source_sha256']
        for historical in (resolved['plan']['source_sha256'], witnessed):
            require(all(pins.get(path) == sha for path, sha in historical.items()), 'Every original/current source retained without replacement')
        require(recovered['original_model_exit_code'] == 1 and resolved['runtime_admitted'] is False,
                'Historical failed model stays failed; metadata is not native admission')
        require(not set(self.names.values()) & set(resolved['current_scope']['complete_known_units']), 'New unit disjoint from all history')
        for path, row in self.spec['input_extension'].items():
            require(path not in resolved['plan']['inputs'] or resolved['plan']['inputs'][path] == row, 'No historical input identity replaced')
            require(identity(path) == row['identity'], 'Exact current metadata input identity changed')
            self.meta.read({'path': path, 'sha256': row['sha256']})
        partition = sources.classify(pins, resolved['plan']['source_contract'], self.meta.read)
        sources.verify_text_sources(partition, self.guard)
        self.resolved = resolved
        self.guard(); loaded_local_sources(pins)
        return self
