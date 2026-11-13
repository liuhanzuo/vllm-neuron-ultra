"""Full inherited History; only explicit exact metadata-envelope dispatch differs."""
import copy
import hashlib
import json
from pathlib import Path
from qwen35_package_preflight_history_v3 import History as OriginalHistory, Authority
from qwen35_sampler_runtime_io_v12 import Metadata as OriginalMetadata, identity, raw, loaded_local_sources
from qwen35_package_serial_runtime_v7 import source_text
from minicpm5_retained_replay_contract_v1 import LIMITS,require


class Metadata(OriginalMetadata):
    def __init__(self,guard,envelopes):super().__init__(guard);self.envelopes=copy.deepcopy(envelopes)
    def envelope(self,path,sha):
        row=self.envelopes[path];self.guard()
        require(sha==row['sha256'],'Exact bound envelope hash')
        before=identity(path);require(before['size']==row['bytes'],'Exact bound envelope size before read')
        data=raw(path,LIMITS['spec_bytes'])
        require(len(data)==row['bytes']and hashlib.sha256(data).hexdigest()==sha and identity(path)==before,'Complete envelope digest/identity')
        require(path not in self.identities or self.identities[path]==before,'No metadata rebinding')
        self.identities[path]=before;self.guard();return data
    def read(self,value):
        if value['path']not in self.envelopes:return super().read(value)
        require(set(value)in ({'path','sha256'},{'path','sha256','bytes'}),'Exact complete metadata ref')
        data=self.envelope(value['path'],value['sha256'])
        require('bytes'not in value or len(data)==value['bytes'],'Full reference bytes')
        return json.loads(data)


class History(OriginalHistory):
    def __init__(self,spec,spec_ref,review_ref,names,guard):
        super().__init__(spec,spec_ref,review_ref,names,guard)
        self.meta=Metadata(guard,spec['exact_envelopes'])

    def text(self,sources,path,sha):
        if path in self.meta.envelopes:return self.meta.envelope(path,sha)
        return source_text.text_bytes(sources,path,sha)

    def verify_text(self,sources,partition):
        self.guard();sources.check_partition(partition)
        for path,sha in partition['categories']['text_sha256'].items():
            self.guard()
            actual=(sources.aliases.source_sha(path,partition['metadata_aliases'],sources.text)
                if path in partition['metadata_aliases']else hashlib.sha256(self.text(sources,path,sha)).hexdigest())
            require(actual==sha,'Every unchanged full text pin')
        for path,sha in partition['categories']['python_bytes_sha256'].items():
            self.guard();sources.metadata.python_source_bytes(path,sha)
        self.guard()

    def resolve(self):
        from research import minicpm5_layer1_operator_recovered_completion_v2 as completed
        from research import minicpm5_layer1_operator_native_overlay_v3 as overlays
        from research import minicpm5_layer1_operator_sdk_sources_v1 as sources
        from research import prepare_minicpm5_layer1_operator_native_v7 as witnesses
        from qwen35_sampler_history_supplement_v1 import build,extend_scope
        self.guard()
        recovered=completed.completed(completed.BINDING,self.meta.read,self.guard,lambda path,sha:self.text(sources,path,sha))
        overlay=self.meta.read(self.spec['history_overlay'])
        require(overlay['current_serial_catalog']==self.spec['current_serial_catalog'],'Complete then-current catalog binding')
        resolved=overlays.resolve(overlay,recovered,self.meta.read,self.guard)
        self.supplement=build(self.meta.read,self.spec['lfm_r2_scope'],self.meta.read(self.spec['current_serial_catalog']))
        resolved['current_scope']=extend_scope(resolved['current_scope'],self.supplement)
        cache,witnessed={},{}
        for row in resolved['current_scope']['later_serial']['runs']:
            terminal=self.meta.read(row['terminal'])
            if terminal.get('schema')in ('minicpm5_native_model_exit_recorder_actual_exit_v1','minicpm5_native_independent_audit_actual_exit_v1'):
                additions=witnesses.executed_extension(terminal,row,self.meta.read,self.guard,cache)
                for path,sha in additions.items():
                    require(path not in witnessed or witnessed[path]==sha,'Completed source witness conflict');witnessed[path]=sha
        pins=self.spec['source_sha256']
        for historical in (resolved['plan']['source_sha256'],witnessed):
            require(all(pins.get(path)==sha for path,sha in historical.items()),'Every original/current source retained without replacement')
        require(recovered['original_model_exit_code']==1 and resolved['runtime_admitted']is False,'Historical failed model stays failed; metadata is not native admission')
        require(not set(self.names.values())&set(resolved['current_scope']['complete_known_units']),'New unit disjoint from all history')
        for path,row in self.spec['input_extension'].items():
            require(path not in resolved['plan']['inputs']or resolved['plan']['inputs'][path]==row,'No historical input identity replaced')
            require(identity(path)==row['identity'],'Exact current metadata input identity changed')
            self.meta.read({'path':path,'sha256':row['sha256']})
        partition=sources.classify(pins,resolved['plan']['source_contract'],self.meta.read)
        self.verify_text(sources,partition)
        self.resolved=resolved;self.guard();loaded_local_sources(pins)
        return self
