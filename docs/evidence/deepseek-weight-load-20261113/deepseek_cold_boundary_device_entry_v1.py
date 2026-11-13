"""Fresh-process capture-on DEVICE leaf; no CLI, launch or import-time IO.

The external serial owner authenticates all sources, original weights/case,
native-reference hashes and original cold metadata; independently supplies the
accepted build-report hash; reserves full resources; and provides a live guard.
This leaf neither issues admission nor qualifies model correctness or speed.
"""
from pathlib import Path
import re
import sys

STRICT_FAILURE = 'Strict cold-bucket numerical/state gate failed'
NATIVE_MODULES = ('torch','torch_neuronx','neuronxcc','neuronx_distributed',
                  'neuronx_distributed_inference','numpy','transformers','vllm','vllm_neuron')


def require(value,message):
    if not value:raise ValueError(message)


def sha256_shape(value):
    require(type(value) is str and re.fullmatch('[a-f0-9]{64}',value),
            'externally authenticated exact SHA256 required')


def paths(destination,reference_directory,data_root,artifact,cache,workdir):
    destination,reference_directory,data_root,artifact,cache,workdir=map(
        Path,(destination,reference_directory,data_root,artifact,cache,workdir))
    selected=(destination,reference_directory,data_root,artifact,cache,workdir)
    require(all(p.is_absolute() and p.resolve()==p and not p.is_symlink() and p.is_dir() and
                'deepseek-cold-kv-diagnostic-v2' not in p.parts for p in selected),
            'ordinary independent existing roots required')
    require(reference_directory==destination/'cold-reference','exact completed native reference directory required')
    independent=(destination,data_root,artifact,cache,workdir)
    require(len(set(independent))==len(independent) and all(a not in b.parents and b not in a.parents
        for i,a in enumerate(independent) for b in independent[i+1:]),'disjoint capture and graph roots required')
    targets=[destination/'boundary-captures/device',data_root/'device']
    targets += [destination/name for name in ('state-copy.json','state-lifecycle.json','lifecycle-states',
                                            'cold-buckets.json','cold-buckets.json.tmp','cold-buckets.pt')]
    require(all(not p.exists() and not p.is_symlink() and p.resolve()==p for p in targets),
            'fresh complete DEVICE outputs required')
    return selected


class GuardedSink:
    """Keep the accepted Capture implementation and each complete callback."""
    backend='device'

    def __init__(self,capture,guard):
        self.capture,self.guard=capture,guard
        self.completed=None

    def device(self,*args,**kwargs):
        self.guard();result=self.capture.device(*args,**kwargs);self.guard()
        return result

    def finish(self):
        require(self.completed is None,'capture finish occurs exactly once')
        self.guard();result=self.capture.finish();self.guard()
        self.completed=result
        return result


def capture_complete(value,frames):
    require(type(value) is dict and value['complete'] is True and value['backend']=='device' and
            value['graph_instrumented'] is True and value['original_model_precision']=='BF16' and
            value['prediction_positions_only'] is True,'complete original DEVICE capture required')
    require([row['frame'] for row in value['frames']]==frames and len(value['calls'])==42,
            'all84 DEVICE frames and42 original forward calls required')
    require(all(value[key] is False for key in ('full_model_qualified','neutrality_proven','new_speedup_claim')),
            'capture cannot grant qualification')


def run(artifact,cache,workdir,expected_report_sha256,case,destination,reference_directory,
        data_root,original_cold_metadata,expected_reference_sha256,execution_guard):
    """Load accepted capture graph and call the entire original cold84 driver.

    expected_report_sha256 comes from the root's independent actual build exit
    audit. expected_reference_sha256 binds exactly native cold-reference.pt/json
    before the original driver deserializes either. original_cold_metadata is
    an already authenticated root input, not a replacement set of failing rows.
    Resource, physical ownership, full closure and freshness are outer duties.
    """
    require(callable(execution_guard),'owning serial execution guard required')
    guard_revoked=False
    def guard():
        nonlocal guard_revoked
        try:execution_guard()
        except BaseException:
            guard_revoked=True
            raise
    guard()
    require(__debug__ and not any(name.split('.')[0] in NATIVE_MODULES for name in sys.modules),
            'fresh DEVICE process before native imports required')
    require(sys.gettrace() is None,'pre-existing Python trace forbidden')
    sha256_shape(expected_report_sha256)
    selected=paths(destination,reference_directory,data_root,artifact,cache,workdir)
    destination,reference_directory,data_root,artifact,cache,workdir=selected
    expected_paths={str(reference_directory/name) for name in ('cold-reference.pt','cold-reference.json')}
    require(type(expected_reference_sha256) is dict and set(expected_reference_sha256)==expected_paths,
            'both exact caller-authenticated native reference hashes required')
    for digest in expected_reference_sha256.values():sha256_shape(digest)
    guard()
    # All these frozen modules are stdlib-only at import. graph.load performs
    # native imports only after its own fresh-process and external-guard checks.
    import deepseek_capture_graph_entry_v1 as graph
    import deepseek_cpu_capture_bridge_v1 as bridge
    from deepseek_cold_kv_protocol_v2 import expected_case,check_device_metadata,check_copy,check_lifecycle,read
    from deepseek_cold_kv_contract_v2 import frames
    require(case==expected_case(),'original complete BF16 TP2 B4 context4096 case required')
    # Validate the complete authenticated historical input before loading.
    check_device_metadata(original_cold_metadata,original_cold_metadata)
    bridge.check_neutrality_metadata(original_cold_metadata,original_cold_metadata)
    def reference_unchanged():
        for path,digest in expected_reference_sha256.items():
            guard();require(graph.file_record(path)['sha256']==digest,'native reference changed before complete driver')
    reference_unchanged();guard()
    model=graph.load(artifact,cache,workdir,expected_report_sha256,guard)
    guard()
    paths(*selected[:3],*selected[3:])
    reference_unchanged()
    from deepseek_cold_boundary_capture_v1 import Capture
    from validate_deepseek_cold_boundary_v2 import main as original_loop

    # Snapshot both attributes before installing anything. The original driver
    # replaces both _forward callables for route observation; restore even a
    # partial installation. Restore inherited descriptors by deleting our own
    # value, not by leaving a new bound-method instance attribute behind.
    saved=[]
    for wrapper in (model.context_encoding_model,model.token_generation_model):
        own='_forward' in vars(wrapper)
        saved.append((wrapper,own,vars(wrapper).get('_forward'),wrapper._forward))
    capture=Capture.__new__(Capture);sink=None
    try:
        Capture.__init__(capture,'device',destination,data_root,model=model)
        sink=GuardedSink(capture,guard)
        for wrapper,own,stored,original in saved:
            def guarded(*args,_original=original,**kwargs):
                guard();result=_original(*args,**kwargs);guard()
                return result
            wrapper._forward=guarded
        guard()
        try:
            original_loop(model,case,destination,reference_directory,sink)
        except SystemExit as error:
            if guard_revoked:raise
            require(type(error) is SystemExit and type(error.code) is str and error.code==STRICT_FAILURE,
                    'unexpected DEVICE termination')
        else:
            raise ValueError('original failed cold gate unexpectedly changed')
        guard()
        capture_complete(sink.completed,frames('device'))
        report=read(destination/'cold-buckets.json')
        require('error' not in report,'partial/error cold report cannot classify original gate')
        metadata=check_device_metadata(report,original_cold_metadata)
        classification=bridge.check_neutrality_metadata(original_cold_metadata,report)
        check_copy(read(destination/'state-copy.json'))
        lifecycle=read(destination/'state-lifecycle.json');check_lifecycle(lifecycle)
        require(lifecycle['state_tensors_changed_by_warmup']==144 and lifecycle['same144_bindings_preserved'],
                'complete original144 state lifecycle required')
        guard()
    finally:
        # Sink cleanup and route restoration are independent: one failing must
        # not prevent the other. No model files, outputs or state buffers are
        # deleted/reset here; partial evidence belongs to the external owner.
        try:
            if hasattr(capture,'handles') and hasattr(capture,'active'):capture.close()
        finally:
            errors=[]
            for wrapper,own,stored,original in reversed(saved):
                try:
                    if own:wrapper._forward=stored
                    elif '_forward' in vars(wrapper):delattr(wrapper,'_forward')
                except BaseException as error:errors.append(error)
            if errors:raise errors[0]
    require(not capture.handles and capture.active is False and sys.gettrace() is None,'sink hooks/trace not closed')
    for wrapper,own,stored,original in saved:
        require(('_forward' in vars(wrapper))==own and (not own or vars(wrapper)['_forward'] is stored),
                'original wrapper attribute ownership restored')
    guard()
    return dict(device_frames=84,original_forward_calls=42,full_device_KV_tensors=6048,
        state_objects=144,original_failed_gate_preserved=True,metadata_reproduction=metadata,
        full_frame_classification=classification,build_report_sha256=expected_report_sha256,
        native_reference_sha256=dict(expected_reference_sha256),hooks_closed=True,route_hooks_restored=True,
        original_precision='BF16',tp=2,batch=4,context_length=4096,
        original_raw_reproduction_pending=True,neutrality_proven=False,
        full_model_qualified=False,new_speedup_claim=False)
