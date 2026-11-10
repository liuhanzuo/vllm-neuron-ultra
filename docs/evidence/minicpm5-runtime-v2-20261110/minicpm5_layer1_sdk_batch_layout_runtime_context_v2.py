"""Partition-aware successor runtime context; original model admission unchanged.

Only source/JSON transport extends frozen RuntimeContext. Model payload
verify_input, environment checks, permit authentication, activation and CPU
transition tracking remain inherited. Native bytes are never read here.
"""
import copy
from pathlib import Path
from minicpm5_layer0_runtime_auth_v1 import RuntimeContext as OriginalContext,identity,digest
from research import minicpm5_layer1_sdk_batch_layout_sources_v2 as sources
require=sources.require


class RuntimeContext(OriginalContext):
    def guard(self,role=None):
        permit=super().guard(role)
        for path,wanted in getattr(self,'_source_runtime_identities',{}).items():
            require(identity(path)==wanted,'Source authentication admission/receipt identity changed')
        if hasattr(self,'_inherited_native_receipt'):
            require(sources.native_identities(self._source_partition)==self._inherited_native_receipt['native_identities'],
                'Inherited native authentication identity changed')
        return permit

    def verify_sources(self):
        self.guard();require(not self._sources_checked,'One complete source authentication before model work')
        def read(ref):
            self.guard()
            require(self.plan['source_sha256'].get(ref['path'])==ref['sha256'],
                'Every classifier metadata reference source-pinned')
            return sources.load(ref)
        partition=sources.classify(self.plan['source_sha256'],self.plan['source_contract'],read)
        sources.verify_text_sources(partition,self.guard)
        root=Path(self.plan['output_root']);runtime={}
        def receipt(ref,name):
            self.guard();sources.reference(ref)
            require(ref['path']==str(root/name),'Exact admitted runtime metadata file')
            before=identity(ref['path']);require(before['size']<sources.metadata.LIMIT,'Bounded original runtime receipt')
            value=sources.load(ref);require(identity(ref['path'])==before,'Stable complete runtime metadata read')
            runtime[ref['path']]=before;return value
        permit=self.permit
        a=receipt(permit['runtime_admission'],'admission.json')
        require(a['schema']=='minicpm5_layer1_sdk_batch_layout_cpu_admission_v2'
            and a['source_contract']==self.plan['source_contract']
            and a['plan']==permit['runtime_plan'] and a['outer_owner']==permit['outer_owner']
            and a['lock_identity']==permit['lock_identity'] and a['attempt_id']==permit['attempt_id']
            and a['latest_completion']==self.plan['latest_completion']
            and a['source_authentication']==permit['source_authentication']
            and a['runtime_admitted']is a['predecessors_checked']is True,
            'Actual source authentication bound by externally guarded outer admission')
        authenticated=receipt(a['source_authentication'],'source-authentication.json')
        sources.validate_native_receipt(authenticated,partition,a['outer_owner'],a['lock_identity'],a['attempt_id'])
        sources.check_native_identities(authenticated,partition,self.guard)
        self._source_partition=partition;self._inherited_native_receipt=copy.deepcopy(authenticated)
        self._source_runtime_identities=runtime
        self.guard();self._sources_checked=True

    def read_json(self,path,wanted=None):
        # Do not call inherited verify_input before determining the metadata
        # class: its generic payload hash/read path has a different policy.
        self.guard();require(self._sources_checked,'Complete source authentication before metadata read')
        path=str(Path(path));require(path in self.plan['inputs'] and Path(path).suffix=='.json',
            'Only original admitted JSON input')
        row=self.plan['inputs'][path]
        require(wanted is None or row['sha256']==wanted,'Original exact metadata digest')
        before=identity(path);require(before==row['identity'],'Original admitted metadata identity')
        value=sources.load(dict(path=path,sha256=row['sha256'],bytes=before['size']))
        require(identity(path)==before,'Metadata identity unchanged across authenticated transport')
        self._verified[path]=before;self.guard();return value
